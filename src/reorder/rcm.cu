// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Reverse Cuthill-McKee reordering, after AMGCL (amgcl/reorder/cuthill_mckee.hpp,
// amgcl/adapter/reorder.hpp), Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>,
// MIT License (see LICENSES/MIT.txt).

#include <reorder/rcm.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <climits>

namespace amgx
{
namespace reorder
{

namespace
{

const int RCM_BLOCK_SIZE = 256;
const int RCM_PERIPHERAL_SEARCHES = 8;
const int64_t RCM_NO_KEY = LLONG_MAX;

inline int rcmNumBlocks(int64_t n)
{
    return (int) std::max<int64_t>(1, std::min<int64_t>(4096, (n + RCM_BLOCK_SIZE - 1) / RCM_BLOCK_SIZE));
}

// Keys (i, j) and (j, i) of the off-diagonal block pattern; others are marked RCM_NO_KEY
template <typename IndexType>
__global__ void graphKeysKernel(const IndexType *row_offsets, const IndexType *col_indices, int n, int64_t *keys)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType j = col_indices[k];
            const bool edge = j != i && j >= 0 && j < n;
            keys[2 * k] = edge ? (int64_t) i * n + j : RCM_NO_KEY;
            keys[2 * k + 1] = edge ? (int64_t) j * n + i : RCM_NO_KEY;
        }
    }
}

__global__ void splitKeysKernel(const int64_t *keys, int64_t nnz, int64_t n, int *rows, int *cols)
{
    for (int64_t k = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; k < nnz; k += (int64_t) gridDim.x * blockDim.x)
    {
        rows[k] = (int)(keys[k] / n);
        cols[k] = (int)(keys[k] % n);
    }
}

// Persistent single-block sweep over consecutive levels, ordered by (first parent position,
// degree, index) with a shared-memory bitonic sort when SORT. Levels are inherently sequential;
// each one is processed by the whole block, so the sweep needs no host round trip per level.
// It stops when the component is exhausted (status DONE) or when a level has more than
// RCM_SWEEP_CAPACITY nodes (status OVERFLOW: the level is left unsorted in queue[0..count) for the
// multi-block path). state = {lo, hi, filled, level, count, status}: order[lo..hi) is the frontier.
const int RCM_SWEEP_CAPACITY = 1024;
const int RCM_SWEEP_DONE = 1, RCM_SWEEP_OVERFLOW = 2;

__device__ __forceinline__ bool levelLess(int pa, int da, int va, int pb, int db, int vb)
{
    return pa != pb ? pa < pb : (da != db ? da < db : va < vb);
}

template <bool SORT>
__global__ void __launch_bounds__(RCM_SWEEP_CAPACITY) levelSweepKernel(const int *g_ro, const int *g_ci, const int *degree,
        int *mark, int *parent_min, int *order, int *queue, int *state)
{
    __shared__ int s_count;
    __shared__ int s_p[RCM_SWEEP_CAPACITY], s_d[RCM_SWEEP_CAPACITY], s_v[RCM_SWEEP_CAPACITY];
    int lo = state[0], hi = state[1], filled = state[2], level = state[3];

    for (;;)
    {
        level++;

        if (threadIdx.x == 0) { s_count = 0; }

        __syncthreads();

        for (int t = threadIdx.x; t < hi - lo; t += blockDim.x)
        {
            const int u = order[lo + t];

            for (int e = g_ro[u]; e < g_ro[u + 1]; e++)
            {
                const int v = g_ci[e];
                const int old = atomicCAS(&mark[v], -1, level);

                if (old == -1) { queue[atomicAdd(&s_count, 1)] = v; }

                if (SORT && (old == -1 || old == level)) { atomicMin(&parent_min[v], lo + t); }
            }
        }

        __syncthreads();
        const int count = s_count;

        if (count == 0 || count > RCM_SWEEP_CAPACITY)
        {
            if (threadIdx.x == 0)
            {
                state[0] = lo; state[1] = hi; state[2] = filled; state[3] = level; state[4] = count;
                state[5] = count == 0 ? RCM_SWEEP_DONE : RCM_SWEEP_OVERFLOW;
            }

            return;
        }

        if (SORT)
        {
            int P = 1;

            while (P < count) { P <<= 1; }

            for (int t = threadIdx.x; t < P; t += blockDim.x)
            {
                const int v = t < count ? queue[t] : INT_MAX;
                s_v[t] = v;
                s_p[t] = t < count ? parent_min[v] : INT_MAX;
                s_d[t] = t < count ? degree[v] : INT_MAX;
            }

            __syncthreads();

            for (int k = 2; k <= P; k <<= 1)
                for (int j = k >> 1; j > 0; j >>= 1)
                {
                    for (int i = threadIdx.x; i < P; i += blockDim.x)
                    {
                        const int ixj = i ^ j;

                        if (ixj > i)
                        {
                            const bool up = (i & k) == 0;
                            const bool greater = levelLess(s_p[ixj], s_d[ixj], s_v[ixj], s_p[i], s_d[i], s_v[i]);

                            if (greater == up)
                            {
                                int a = s_p[i]; s_p[i] = s_p[ixj]; s_p[ixj] = a;
                                a = s_d[i]; s_d[i] = s_d[ixj]; s_d[ixj] = a;
                                a = s_v[i]; s_v[i] = s_v[ixj]; s_v[ixj] = a;
                            }
                        }
                    }

                    __syncthreads();
                }

            for (int t = threadIdx.x; t < count; t += blockDim.x) { order[filled + t] = s_v[t]; }
        }
        else
        {
            for (int t = threadIdx.x; t < count; t += blockDim.x) { order[filled + t] = queue[t]; }
        }

        __syncthreads();
        lo = filled;
        hi = filled + count;
        filled += count;
    }
}

// Sort keys of a level: (degree, index), then the first parent's position
__global__ void levelKeysKernel(const int *nodes, int count, const int *degree, const int *parent_min,
                                int64_t *degree_keys, int *parent_keys)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < count; t += gridDim.x * blockDim.x)
    {
        const int v = nodes[t];

        if (degree_keys != NULL) { degree_keys[t] = ((int64_t) degree[v] << 32) | (int64_t) v; }

        if (parent_keys != NULL) { parent_keys[t] = parent_min[v]; }
    }
}

__global__ void setMarksKernel(const int *nodes, int count, int level, int *mark)
{
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < count; t += gridDim.x * blockDim.x)
    {
        mark[nodes[t]] = level;
    }
}

__global__ void reverseKernel(const int *order, int n, int *perm)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        perm[i] = order[n - 1 - i];
    }
}

__global__ void inverseKernel(const int *perm, int n, int *iperm)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        iperm[perm[i]] = i;
    }
}

template <typename IndexType>
__global__ void permutedKeysKernel(const IndexType *row_offsets, const IndexType *col_indices, int n, const int *iperm,
                                   int64_t *keys, int *source)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int64_t r = iperm[i];

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            keys[k] = r * n + iperm[col_indices[k]];
            source[k] = k;
        }
    }
}

template <typename IndexType>
__global__ void diagonalSourceKernel(const IndexType *diag, const int *perm, int n, int *source)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        source[i] = diag[perm[i]];
    }
}

template <typename ValueType>
__global__ void gatherBlocksKernel(const ValueType *src, const int *map, int64_t count, int bs, ValueType *dst)
{
    for (int64_t t = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; t < count * bs; t += (int64_t) gridDim.x * blockDim.x)
    {
        dst[t] = src[(int64_t) map[t / bs] * bs + t % bs];
    }
}

template <typename ValueType>
__global__ void scatterBlocksKernel(const ValueType *src, const int *map, int64_t count, int bs, ValueType *dst)
{
    for (int64_t t = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; t < count * bs; t += (int64_t) gridDim.x * blockDim.x)
    {
        dst[(int64_t) map[t / bs] * bs + t % bs] = src[t];
    }
}

template <typename IndexType>
__global__ void bandwidthKernel(const IndexType *row_offsets, const IndexType *col_indices, int n, int *bw)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int b = 0;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType j = col_indices[k];

            if (j < n) { b = max(b, (int) (j > i ? j - i : i - j)); }
        }

        atomicMax(bw, b);
    }
}

struct is_no_key
{
    __host__ __device__ bool operator()(int64_t k) const { return k == RCM_NO_KEY; }
};

// (degree << 32 | v) of the unvisited nodes, RCM_NO_KEY otherwise
struct unvisited_key
{
    const int *mark, *degree;
    unvisited_key(const int *mark, const int *degree) : mark(mark), degree(degree) {}
    __host__ __device__ int64_t operator()(int v) const { return mark[v] == -1 ? (((int64_t) degree[v] << 32) | (int64_t) v) : RCM_NO_KEY; }
};

struct degree_key
{
    const int *degree;
    degree_key(const int *degree) : degree(degree) {}
    __host__ __device__ int64_t operator()(int v) const { return ((int64_t) degree[v] << 32) | (int64_t) v; }
};

struct is_isolated
{
    const int *degree;
    is_isolated(const int *degree) : degree(degree) {}
    __host__ __device__ bool operator()(int v) const { return degree[v] == 0; }
};

// Symmetric adjacency (CSR) of the off-diagonal pattern of A and the node degrees
template <class TConfig>
void buildGraph(const Matrix<TConfig> &A, typename RCM<TConfig>::IVector &g_ro, typename RCM<TConfig>::IVector &g_ci,
                typename RCM<TConfig>::IVector &degree)
{
    typedef typename RCM<TConfig>::I64Vector I64Vector;
    typedef typename RCM<TConfig>::IVector IVector;
    const int n = A.get_num_rows();
    const int64_t nnz = A.get_num_nz();
    I64Vector keys(2 * nnz);

    if (n > 0 && nnz > 0)
    {
        graphKeysKernel <<< rcmNumBlocks(n), RCM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), n, keys.raw());
        cudaCheckError();
    }

    thrust_wrapper::sort<TConfig::memSpace>(keys.begin(), keys.end());
    cudaCheckError();
    int64_t m = amgx::thrust::unique(keys.begin(), keys.end()) - keys.begin();
    m = amgx::thrust::lower_bound(keys.begin(), keys.begin() + m, RCM_NO_KEY) - keys.begin();
    cudaCheckError();
    IVector rows(m);
    g_ci.resize(m);
    g_ro.resize(n + 1);

    if (m > 0)
    {
        splitKeysKernel <<< rcmNumBlocks(m), RCM_BLOCK_SIZE>>>(keys.raw(), m, n, rows.raw(), g_ci.raw());
        cudaCheckError();
    }

    amgx::thrust::lower_bound(rows.begin(), rows.end(), amgx::thrust::counting_iterator<int>(0),
                              amgx::thrust::counting_iterator<int>(n + 1), g_ro.begin());
    degree.resize(n);
    amgx::thrust::transform(g_ro.begin() + 1, g_ro.end(), g_ro.begin(), degree.begin(), amgx::thrust::minus<int>());
    cudaCheckError();
}

// Level-ordered sweep of the component of root into order[filled..]: Cuthill-McKee order when
// sort (parent_min required), plain breadth-first order otherwise. mark must hold -1 for the
// unvisited nodes. Returns the number of levels after the root's; the last level is
// order[last_lo..last_hi) and filled is advanced past the component.
template <class TConfig>
int sweepComponent(int root, bool sort, const typename RCM<TConfig>::IVector &g_ro,
                   const typename RCM<TConfig>::IVector &g_ci, const typename RCM<TConfig>::IVector &degree,
                   typename RCM<TConfig>::IVector &mark, typename RCM<TConfig>::IVector &parent_min,
                   typename RCM<TConfig>::IVector &order, int &filled, int &level,
                   typename RCM<TConfig>::IVector &queue, typename RCM<TConfig>::I64Vector &degree_keys,
                   typename RCM<TConfig>::IVector &parent_keys, typename RCM<TConfig>::IVector &state,
                   int &last_lo, int &last_hi)
{
    mark[root] = level;
    order[filled] = root;
    const int root_level = level;
    std::vector<int> h_state(6);
    h_state[0] = filled; h_state[1] = filled + 1; h_state[2] = filled + 1; h_state[3] = level; h_state[4] = 0; h_state[5] = 0;

    for (;;)
    {
        amgx::thrust::copy(h_state.begin(), h_state.end(), state.begin());

        if (sort)
        {
            levelSweepKernel<true> <<< 1, RCM_SWEEP_CAPACITY>>>(g_ro.raw(), g_ci.raw(), degree.raw(), mark.raw(),
                    parent_min.raw(), order.raw(), queue.raw(), state.raw());
        }
        else
        {
            levelSweepKernel<false> <<< 1, RCM_SWEEP_CAPACITY>>>(g_ro.raw(), g_ci.raw(), degree.raw(), mark.raw(),
                    NULL, order.raw(), queue.raw(), state.raw());
        }

        cudaCheckError();
        amgx::thrust::copy(state.begin(), state.end(), h_state.begin());

        if (h_state[5] == RCM_SWEEP_DONE) { break; }

        // A level too large for one block: order it with device sorts and continue
        const int count = h_state[4];

        if (sort)
        {
            levelKeysKernel <<< rcmNumBlocks(count), RCM_BLOCK_SIZE>>>(queue.raw(), count, degree.raw(),
                    parent_min.raw(), degree_keys.raw(), NULL);
            cudaCheckError();
            thrust_wrapper::sort_by_key<TConfig::memSpace>(degree_keys.begin(), degree_keys.begin() + count, queue.begin());
            levelKeysKernel <<< rcmNumBlocks(count), RCM_BLOCK_SIZE>>>(queue.raw(), count, degree.raw(),
                    parent_min.raw(), NULL, parent_keys.raw());
            cudaCheckError();
            thrust_wrapper::stable_sort_by_key<TConfig::memSpace>(parent_keys.begin(), parent_keys.begin() + count, queue.begin());
        }

        thrust_wrapper::copy<TConfig::memSpace>(queue.begin(), queue.begin() + count, order.begin() + h_state[2]);
        cudaCheckError();
        h_state[0] = h_state[2];
        h_state[1] = h_state[2] + count;
        h_state[2] += count;
    }

    last_lo = h_state[0];
    last_hi = h_state[1];
    filled = h_state[2];
    level = h_state[3] + 1;
    return h_state[3] - root_level - 1;
}

} // anonymous namespace

template <class T_Config>
void RCM<T_Config>::permutation(const Matrix<TConfig> &A, IVector &perm)
{
    if (TConfig::memSpace != AMGX_device)
    {
        FatalError("RCM reordering is only implemented on device", AMGX_ERR_NOT_IMPLEMENTED);
    }

    const int n = A.get_num_rows();
    perm.resize(n);

    if (n == 0) { return; }

    IVector g_ro, g_ci, degree;
    buildGraph<TConfig>(A, g_ro, g_ci, degree);
    // Cuthill-McKee order, filled level by level
    IVector order(n), mark(n, -1), parent_min(n, INT_MAX);
    IVector search_mark(n), queue_a(n), parent_keys(n);
    I64Vector degree_keys(n);
    // Isolated nodes first (they end up last after the reversal)
    int filled = amgx::thrust::copy_if(amgx::thrust::counting_iterator<int>(0), amgx::thrust::counting_iterator<int>(n),
                                       order.begin(), is_isolated(degree.raw())) - order.begin();
    cudaCheckError();

    if (filled > 0)
    {
        setMarksKernel <<< rcmNumBlocks(filled), RCM_BLOCK_SIZE>>>(order.raw(), filled, 0, mark.raw());
        cudaCheckError();
    }

    int level = 1;
    IVector state(6), scratch(n);

    while (filled < n)
    {
        // Next component: from its minimum-degree node to a pseudo-peripheral node (George-Liu),
        // with breadth-first sweeps into a scratch order on a scratch mark
        const int64_t smin = amgx::thrust::transform_reduce(amgx::thrust::counting_iterator<int>(0),
                             amgx::thrust::counting_iterator<int>(n), unvisited_key(mark.raw(), degree.raw()),
                             RCM_NO_KEY, amgx::thrust::minimum<int64_t>());
        cudaCheckError();
        // George-Liu: move to a minimum-degree node of the last level while the eccentricity grows
        int candidate = (int)(smin & 0xffffffffLL), root = candidate, ecc = -1;

        for (int s = 0; s < RCM_PERIPHERAL_SEARCHES; s++)
        {
            amgx::thrust::copy(mark.begin(), mark.end(), search_mark.begin());
            int scratch_filled = 0, scratch_level = level, last_lo = 0, last_hi = 0;
            const int e = sweepComponent<TConfig>(candidate, false, g_ro, g_ci, degree, search_mark, parent_min, scratch,
                                                  scratch_filled, scratch_level, queue_a, degree_keys, parent_keys,
                                                  state, last_lo, last_hi);

            if (e <= ecc) { break; }

            root = candidate;
            ecc = e;
            const int64_t cmin = amgx::thrust::transform_reduce(scratch.begin() + last_lo, scratch.begin() + last_hi,
                                 degree_key(degree.raw()), RCM_NO_KEY, amgx::thrust::minimum<int64_t>());
            candidate = (int)(cmin & 0xffffffffLL);

            if (candidate == root) { break; }
        }

        // Cuthill-McKee levels from the root
        int last_lo = 0, last_hi = 0;
        sweepComponent<TConfig>(root, true, g_ro, g_ci, degree, mark, parent_min, order, filled, level, queue_a,
                                degree_keys, parent_keys, state, last_lo, last_hi);
    }

    reverseKernel <<< rcmNumBlocks(n), RCM_BLOCK_SIZE>>>(order.raw(), n, perm.raw());
    cudaCheckError();
}

template <class T_Config>
void RCM<T_Config>::inverse(const IVector &perm, IVector &iperm)
{
    const int n = (int) perm.size();
    iperm.resize(n);

    if (n > 0)
    {
        inverseKernel <<< rcmNumBlocks(n), RCM_BLOCK_SIZE>>>(perm.raw(), n, iperm.raw());
        cudaCheckError();
    }
}

template <class T_Config>
void RCM<T_Config>::permuteStructure(const Matrix<TConfig> &A, const IVector &perm, const IVector &iperm,
                                     Matrix<TConfig> &B, IVector &nnz_map)
{
    if (TConfig::memSpace != AMGX_device)
    {
        FatalError("RCM reordering is only implemented on device", AMGX_ERR_NOT_IMPLEMENTED);
    }

    const int n = A.get_num_rows(), nnz = A.get_num_nz();
    const bool has_diag = A.hasProps(DIAG);
    I64Vector keys(nnz);
    nnz_map.resize(nnz + (has_diag ? n : 1));

    if (n > 0 && nnz > 0)
    {
        permutedKeysKernel <<< rcmNumBlocks(n), RCM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), n,
                iperm.raw(), keys.raw(), nnz_map.raw());
        cudaCheckError();
        thrust_wrapper::sort_by_key<TConfig::memSpace>(keys.begin(), keys.end(), nnz_map.begin());
        cudaCheckError();
    }

    if (has_diag)
    {
        if (n > 0)
        {
            diagonalSourceKernel <<< rcmNumBlocks(n), RCM_BLOCK_SIZE>>>(A.diag.raw(), perm.raw(), n, nnz_map.raw() + nnz);
            cudaCheckError();
        }
    }
    else
    {
        nnz_map[nnz] = nnz;   // zero sentinel block
    }

    B.set_initialized(0);
    B.addProps(CSR);

    if (has_diag) { B.addProps(DIAG); }

    B.resize(n, n, nnz, A.get_block_dimy(), A.get_block_dimx(), 1);
    BlockFormat format = A.getBlockFormat();
    B.setBlockFormat(format);
    IVector rows(nnz);

    if (nnz > 0)
    {
        splitKeysKernel <<< rcmNumBlocks(nnz), RCM_BLOCK_SIZE>>>(keys.raw(), nnz, n, rows.raw(), B.col_indices.raw());
        cudaCheckError();
    }

    amgx::thrust::lower_bound(rows.begin(), rows.end(), amgx::thrust::counting_iterator<int>(0),
                              amgx::thrust::counting_iterator<int>(n + 1), B.row_offsets.begin());
    cudaCheckError();
    permuteValues(A, nnz_map, B);
    B.computeDiagonal();
    B.set_initialized(1);
}

template <class T_Config>
void RCM<T_Config>::permuteValues(const Matrix<TConfig> &A, const IVector &nnz_map, Matrix<TConfig> &B)
{
    const int64_t count = nnz_map.size();

    if (count > 0)
    {
        gatherBlocksKernel <<< rcmNumBlocks(count * A.get_block_size()), RCM_BLOCK_SIZE, 0,
                           amgx::thrust::global_thread_handle::get_stream()>>>(A.values.raw(), nnz_map.raw(), count,
                                   A.get_block_size(), B.values.raw());
        cudaCheckError();
    }
}

template <class T_Config>
int RCM<T_Config>::bandwidth(const Matrix<TConfig> &A)
{
    const int n = A.get_num_rows();
    IVector bw(1, 0);

    if (n > 0)
    {
        bandwidthKernel <<< rcmNumBlocks(n), RCM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), n, bw.raw());
        cudaCheckError();
    }

    return bw[0];
}

template <class T_Config>
void RCM<T_Config>::gather(const VVector &x, const IVector &perm, int bsize, VVector &y)
{
    const int64_t n = perm.size();

    if (n > 0)
    {
        gatherBlocksKernel <<< rcmNumBlocks(n * bsize), RCM_BLOCK_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            x.raw(), perm.raw(), n, bsize, y.raw());
        cudaCheckError();
    }

    y.dirtybit = 1;
}

template <class T_Config>
void RCM<T_Config>::scatter(const VVector &y, const IVector &perm, int bsize, VVector &x)
{
    const int64_t n = perm.size();

    if (n > 0)
    {
        scatterBlocksKernel <<< rcmNumBlocks(n * bsize), RCM_BLOCK_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            y.raw(), perm.raw(), n, bsize, x.raw());
        cudaCheckError();
    }

    x.dirtybit = 1;
}

#define AMGX_CASE_LINE(CASE) template struct RCM<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace reorder
} // namespace amgx
