// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Pointwise aggregation adapted from AMGCL (amgcl/coarsening/pointwise_aggregates.hpp,
// amgcl/backend/builtin.hpp pointwise_matrix), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <aggregation/pointwise.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

namespace amgx
{
namespace aggregation
{

namespace
{

const int PW_BLOCK_SIZE = 256;

inline int pwNumBlocks(int n)
{
    return std::max(1, std::min<int>(4096, (n + PW_BLOCK_SIZE - 1) / PW_BLOCK_SIZE));
}

template <typename IndexType>
__global__ void blockToScalarOffsetsKernel(const IndexType *row_offsets, int num_block_rows, int b, bool has_diag,
        IndexType *scalar_row_offsets)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_block_rows * b; i += gridDim.x * blockDim.x)
    {
        const int I = i / b, r = i % b;
        const IndexType blocks = row_offsets[I + 1] - row_offsets[I] + (has_diag ? 1 : 0);
        const IndexType base = (row_offsets[I] + (has_diag ? I : 0)) * b * b;
        scalar_row_offsets[i] = base + r * blocks * b;

        if (i == num_block_rows * b - 1)
        {
            scalar_row_offsets[i + 1] = (row_offsets[num_block_rows] + (has_diag ? num_block_rows : 0)) * b * b;
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void blockToScalarFillKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                        const IndexType *diag, int num_block_rows, int b, bool has_diag, bool col_major,
                                        const IndexType *scalar_row_offsets, IndexType *scalar_cols, ValueType *scalar_values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_block_rows * b; i += gridDim.x * blockDim.x)
    {
        const int I = i / b, r = i % b, bs = b * b;
        IndexType pos = scalar_row_offsets[i];

        for (IndexType k = row_offsets[I]; k < row_offsets[I + 1] + (has_diag ? 1 : 0); k++)
        {
            const bool is_diag = (k == row_offsets[I + 1]);
            const IndexType J = is_diag ? I : col_indices[k];
            const IndexType block = is_diag ? diag[I] : k;

            for (int c = 0; c < b; c++)
            {
                scalar_cols[pos] = J * b + c;
                scalar_values[pos++] = values[block * bs + (col_major ? c * b + r : r * b + c)];
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void pointKeysKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                int num_rows, int b, int64_t num_points, int64_t *keys, ValueType *abs_values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            keys[k] = (int64_t)(i / b) * num_points + col_indices[k] / b;
            abs_values[k] = fabs(values[k]);
        }
    }
}

template <typename IndexType>
__global__ void splitPointKeysKernel(const int64_t *keys, int nnz, int64_t num_points, IndexType *rows, IndexType *cols)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        rows[k] = (IndexType)(keys[k] / num_points);
        cols[k] = (IndexType)(keys[k] % num_points);
    }
}

template <typename IndexType, typename ValueType>
__global__ void expandStrengthKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                     int num_rows, int b, const IndexType *Ap_offsets, const IndexType *Ap_cols,
                                     const bool *strong_points, bool *strong, ValueType *filtered_diag)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const int I = i / b;
        ValueType d_f = ValueType(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType j = col_indices[k];
            const IndexType J = j / b;
            bool s = false;

            if (j != i)
            {
                if (J == I)
                {
                    s = true;
                }
                else
                {
                    // binary search of J in the sorted row I of Ap
                    IndexType lo = Ap_offsets[I], hi = Ap_offsets[I + 1];

                    while (lo < hi)
                    {
                        const IndexType mid = (lo + hi) / 2;

                        if (Ap_cols[mid] < J) { lo = mid + 1; }
                        else { hi = mid; }
                    }

                    s = (lo < Ap_offsets[I + 1] && Ap_cols[lo] == J) ? strong_points[lo] : false;
                }
            }

            strong[k] = s;

            if (!s) { d_f += values[k]; }
        }

        filtered_diag[i] = d_f;
    }
}

__global__ void countAggregateSizesKernel(const int *aggregates, int n, int *sizes)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        if (aggregates[i] >= 0) { atomicAdd(&sizes[aggregates[i]], 1); }
    }
}

__global__ void renumberAggregatesKernel(const int *new_ids, const int *sizes, int b, int min_unknowns, int n,
        int *aggregates)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int a = aggregates[i];

        if (a >= 0) { aggregates[i] = (sizes[a] * b >= min_unknowns) ? new_ids[a] : -1; }
    }
}

struct keep_aggregate
{
    int b, min_unknowns;
    keep_aggregate(int b, int min_unknowns) : b(b), min_unknowns(min_unknowns) {}
    __host__ __device__ int operator()(int size) const { return size * b >= min_unknowns ? 1 : 0; }
};

struct is_removed
{
    __host__ __device__ bool operator()(int a) const { return a < 0; }
};

} // anonymous namespace

template <class T_Config>
void PointwiseAggregation<T_Config>::blockToScalar(const Matrix<TConfig> &A, Matrix<TConfig> &A_scalar)
{
    const int n = A.get_num_rows(), b = A.get_block_dimy();

    if (A.get_block_dimx() != b)
    {
        FatalError("Pointwise aggregation requires square blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    const bool has_diag = A.hasProps(DIAG);
    const int nnz = (A.get_num_nz() + (has_diag ? n : 0)) * b * b;
    A_scalar.set_initialized(0);
    A_scalar.addProps(CSR);
    A_scalar.resize(n * b, A.get_num_cols() * b, nnz, 1, 1, 1);

    if (n > 0)
    {
        blockToScalarOffsetsKernel <<< pwNumBlocks(n * b), PW_BLOCK_SIZE>>>(A.row_offsets.raw(), n, b, has_diag,
                A_scalar.row_offsets.raw());
        cudaCheckError();
        blockToScalarFillKernel <<< pwNumBlocks(n * b), PW_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(),
                A.values.raw(), A.diag.raw(), n, b, has_diag, A.getBlockFormat() == COL_MAJOR,
                A_scalar.row_offsets.raw(), A_scalar.col_indices.raw(), A_scalar.values.raw());
        cudaCheckError();
    }

    A_scalar.computeDiagonal();
    A_scalar.set_initialized(1);
}

template <class T_Config>
void PointwiseAggregation<T_Config>::buildPointMatrix(const Matrix<TConfig> &A, int block_size, Matrix<TConfig> &Ap)
{
    const int n = A.get_num_rows(), nnz = A.get_num_nz();

    if (n % block_size != 0)
    {
        FatalError("Matrix size must be divisible by aggregation_block_size", AMGX_ERR_BAD_PARAMETERS);
    }

    const int64_t np = n / block_size;
    I64Vector keys(nnz), point_keys(nnz);
    MVector abs_values(nnz), point_values(nnz);

    if (n > 0)
    {
        pointKeysKernel <<< pwNumBlocks(n), PW_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                n, block_size, np, keys.raw(), abs_values.raw());
        cudaCheckError();
    }

    thrust_wrapper::sort_by_key<TConfig::memSpace>(keys.begin(), keys.end(), abs_values.begin());
    cudaCheckError();
    const int nnz_p = amgx::thrust::reduce_by_key(keys.begin(), keys.end(), abs_values.begin(), point_keys.begin(),
                      point_values.begin(), amgx::thrust::equal_to<int64_t>(), amgx::thrust::maximum<ValueTypeA>()).first - point_keys.begin();
    cudaCheckError();
    Ap.set_initialized(0);
    Ap.addProps(CSR);
    Ap.resize((int) np, (int) np, nnz_p, 1, 1, 1);
    IVector rows(nnz_p);

    if (nnz_p > 0)
    {
        splitPointKeysKernel <<< pwNumBlocks(nnz_p), PW_BLOCK_SIZE>>>(point_keys.raw(), nnz_p, np, rows.raw(),
                Ap.col_indices.raw());
        cudaCheckError();
    }

    thrust_wrapper::copy<TConfig::memSpace>(point_values.begin(), point_values.begin() + nnz_p, Ap.values.begin());
    amgx::thrust::lower_bound(rows.begin(), rows.end(), amgx::thrust::counting_iterator<int>(0),
                              amgx::thrust::counting_iterator<int>((int) np + 1), Ap.row_offsets.begin());
    cudaCheckError();
    Ap.computeDiagonal();
    Ap.set_initialized(1);
}

template <class T_Config>
void PointwiseAggregation<T_Config>::expandStrongConnections(const Matrix<TConfig> &A, int block_size,
        const Matrix<TConfig> &Ap, const BVector &strong_points, BVector &strong, MVector &filtered_diag)
{
    const int n = A.get_num_rows();
    strong.resize(A.get_num_nz());
    filtered_diag.resize(n);

    if (n == 0) { return; }

    expandStrengthKernel <<< pwNumBlocks(n), PW_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
            n, block_size, Ap.row_offsets.raw(), Ap.col_indices.raw(), strong_points.raw(), strong.raw(),
            filtered_diag.raw());
    cudaCheckError();
}

template <class T_Config>
int PointwiseAggregation<T_Config>::removeSmallAggregates(IVector &aggregates, int &num_aggregates, int block_size,
        int min_unknowns)
{
    const int n = aggregates.size();

    if (min_unknowns <= block_size || num_aggregates == 0 || n == 0) { return 0; }

    IVector sizes(num_aggregates, 0), new_ids(num_aggregates + 1, 0);
    countAggregateSizesKernel <<< pwNumBlocks(n), PW_BLOCK_SIZE>>>(aggregates.raw(), n, sizes.raw());
    cudaCheckError();
    const int removed_before = (int) thrust_wrapper::count_if<TConfig::memSpace>(aggregates.begin(), aggregates.end(), is_removed());
    thrust_wrapper::transform<TConfig::memSpace>(sizes.begin(), sizes.end(), new_ids.begin(), keep_aggregate(block_size, min_unknowns));
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(new_ids.begin(), new_ids.end(), new_ids.begin());
    cudaCheckError();
    num_aggregates = new_ids[num_aggregates];
    renumberAggregatesKernel <<< pwNumBlocks(n), PW_BLOCK_SIZE>>>(new_ids.raw(), sizes.raw(), block_size, min_unknowns,
            n, aggregates.raw());
    cudaCheckError();
    return (int) thrust_wrapper::count_if<TConfig::memSpace>(aggregates.begin(), aggregates.end(), is_removed()) - removed_before;
}

#define AMGX_CASE_LINE(CASE) template struct PointwiseAggregation<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace aggregation
} // namespace amgx
