// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <aggregation/selectors/mis2_aggregates.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

namespace amgx
{
namespace aggregation
{

namespace
{

const int MIS2_BLOCK_SIZE = 256;

enum { MIS2_OUT = 0, MIS2_UNDECIDED = 1, MIS2_IN = 2 };
const int MIS2_PENDING = -2;

inline int mis2NumBlocks(int n)
{
    return std::min<int>(4096, (n + MIS2_BLOCK_SIZE - 1) / MIS2_BLOCK_SIZE);
}

__device__ __forceinline__ unsigned int mis2Hash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Priority key ordered lexicographically by (state, priority, index), non-negative as int64.
// The priority is a hash of the index, or the reversed index (index_priority): lower indices
// first, so the rounds produce the lexicographically first MIS(2), i.e. the seeds of a greedy pass
// in index order as AMGCL's plain_aggregates (exact below 2^29 rows).
__device__ __forceinline__ int64_t mis2Key(int state, int i, bool index_priority)
{
    const unsigned int p = index_priority ? (~(unsigned int) i) : mis2Hash(i);
    return ((int64_t) state << 61) | ((int64_t)(p & 0x1FFFFFFF) << 32) | (int64_t) i;
}

template <typename IndexType>
__global__ void initStateKernel(const IndexType *row_offsets, const bool *strong, int num_rows,
                                int *state, int *aggregates)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        bool has_strong = false;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            has_strong = has_strong || strong[k];
        }

        state[i] = has_strong ? MIS2_UNDECIDED : MIS2_OUT;
        aggregates[i] = has_strong ? MIS2_PENDING : -1;
    }
}

// k1[i] = max key over i and its strong neighbours
template <typename IndexType>
__global__ void maxKeyRing1Kernel(const IndexType *row_offsets, const IndexType *col_indices,
                                  const bool *strong, const int *state, int num_rows, bool index_priority, int64_t *k1)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        int64_t best = mis2Key(state[i], i, index_priority);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (strong[k])
            {
                const IndexType c = col_indices[k];
                const int64_t key = mis2Key(state[c], c, index_priority);
                best = key > best ? key : best;
            }
        }

        k1[i] = best;
    }
}

// k2[i] = max k1 over i and its strong neighbours, then decide undecided nodes
template <typename IndexType>
__global__ void maxKeyRing2AndUpdateKernel(const IndexType *row_offsets, const IndexType *col_indices,
        const bool *strong, const int64_t *k1, int num_rows, int *state)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        if (state[i] != MIS2_UNDECIDED) { continue; }

        int64_t best = k1[i];

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (strong[k] && k1[col_indices[k]] > best) { best = k1[col_indices[k]]; }
        }

        if ((int)(best & 0xFFFFFFFF) == i)
        {
            state[i] = MIS2_IN;
        }
        else if ((int)(best >> 61) == MIS2_IN)
        {
            state[i] = MIS2_OUT;
        }
    }
}

// Roots take their own aggregate; other nodes join the highest priority adjacent root.
template <typename IndexType>
__global__ void assignToRootsKernel(const IndexType *row_offsets, const IndexType *col_indices,
                                    const bool *strong, const int *state, const int *root_id,
                                    int num_rows, bool index_priority, int *aggregates)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        if (aggregates[i] != MIS2_PENDING) { continue; }

        if (state[i] == MIS2_IN)
        {
            aggregates[i] = root_id[i];
            continue;
        }

        int64_t best = -1;
        int best_col = -1;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType c = col_indices[k];

            if (strong[k] && state[c] == MIS2_IN)
            {
                const int64_t key = mis2Key(MIS2_IN, c, index_priority);

                if (key > best) { best = key; best_col = c; }
            }
        }

        if (best_col >= 0) { aggregates[i] = root_id[best_col]; }
    }
}

// Remaining nodes join the aggregate of the highest priority neighbour assigned in the previous pass.
template <typename IndexType>
__global__ void assignToNeighboursKernel(const IndexType *row_offsets, const IndexType *col_indices,
        const bool *strong, const int *aggregates_in, int num_rows, bool index_priority, int *aggregates_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        int agg = aggregates_in[i];

        if (agg == MIS2_PENDING)
        {
            int64_t best = -1;
            agg = -1;

            for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
            {
                const IndexType c = col_indices[k];

                if (strong[k] && aggregates_in[c] >= 0)
                {
                    const int64_t key = mis2Key(0, c, index_priority);

                    if (key > best) { best = key; agg = aggregates_in[c]; }
                }
            }
        }

        aggregates_out[i] = agg;
    }
}

struct is_mis2_state
{
    int state;
    is_mis2_state(int s) : state(s) {}
    __host__ __device__ bool operator()(int s) const { return s == state; }
};

struct mis2_in_flag
{
    __host__ __device__ int operator()(int s) const { return s == MIS2_IN ? 1 : 0; }
};

} // anonymous namespace

template <class T_Config>
void MIS2Aggregates<T_Config>::compute(const Matrix<TConfig> &A, const BVector &strong,
                                       IVector &aggregates, int &num_aggregates, int &num_removed, bool index_priority)
{
    const int num_rows = A.get_num_rows();
    aggregates.resize(num_rows);
    num_aggregates = 0;
    num_removed = 0;

    if (num_rows == 0) { return; }

    const IndexType *row_offsets = A.row_offsets.raw();
    const IndexType *col_indices = A.col_indices.raw();
    const int num_blocks = mis2NumBlocks(num_rows);
    IVector state(num_rows);
    I64Vector k1(num_rows);
    initStateKernel <<< num_blocks, MIS2_BLOCK_SIZE>>>(row_offsets, strong.raw(), num_rows,
            state.raw(), aggregates.raw());
    cudaCheckError();

    // Each round fixes at least the undecided node with the globally largest key.
    while (thrust_wrapper::count_if<TConfig::memSpace>(state.begin(), state.end(), is_mis2_state(MIS2_UNDECIDED)) > 0)
    {
        maxKeyRing1Kernel <<< num_blocks, MIS2_BLOCK_SIZE>>>(row_offsets, col_indices, strong.raw(),
                state.raw(), num_rows, index_priority, k1.raw());
        cudaCheckError();
        maxKeyRing2AndUpdateKernel <<< num_blocks, MIS2_BLOCK_SIZE>>>(row_offsets, col_indices,
                strong.raw(), k1.raw(), num_rows, state.raw());
        cudaCheckError();
    }

    // Number the roots
    IVector root_id(num_rows + 1, 0);
    thrust_wrapper::transform<TConfig::memSpace>(state.begin(), state.end(), root_id.begin(), mis2_in_flag());
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(root_id.begin(), root_id.end(), root_id.begin());
    cudaCheckError();
    num_aggregates = root_id[num_rows];

    if (num_aggregates == 0)
    {
        thrust_wrapper::fill<TConfig::memSpace>(aggregates.begin(), aggregates.end(), -1);
        cudaCheckError();
        num_removed = num_rows;
        return;
    }

    assignToRootsKernel <<< num_blocks, MIS2_BLOCK_SIZE>>>(row_offsets, col_indices, strong.raw(),
            state.raw(), root_id.raw(), num_rows, index_priority, aggregates.raw());
    cudaCheckError();
    // state is no longer needed, reuse it as output of the last pass
    assignToNeighboursKernel <<< num_blocks, MIS2_BLOCK_SIZE>>>(row_offsets, col_indices, strong.raw(),
            aggregates.raw(), num_rows, index_priority, state.raw());
    cudaCheckError();
    thrust_wrapper::copy<TConfig::memSpace>(state.begin(), state.end(), aggregates.begin());
    cudaCheckError();
    num_removed = (int) thrust_wrapper::count_if<TConfig::memSpace>(aggregates.begin(), aggregates.end(), is_mis2_state(-1));
    cudaCheckError();
}

#define AMGX_CASE_LINE(CASE) template struct MIS2Aggregates<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace aggregation
} // namespace amgx
