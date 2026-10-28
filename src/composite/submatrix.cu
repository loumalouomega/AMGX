// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Sub-matrix utilities for the composite preconditioners (CPR, Schur pressure correction),
// following AMGCL (amgcl/preconditioner/cpr.hpp, schur_pressure_correction.hpp),
// Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <composite/submatrix.h>
#include <aggregation/pointwise.h>
#include <csr_multiply.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace amgx
{
namespace composite
{

namespace
{

const int SM_BLOCK_SIZE = 256;

inline int smNumBlocks(int n)
{
    return std::max(1, std::min<int>(4096, (n + SM_BLOCK_SIZE - 1) / SM_BLOCK_SIZE));
}

template <typename IndexType>
__global__ void splitKeysKernel(const int64_t *keys, int nnz, int64_t m, IndexType *rows, IndexType *cols)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        rows[k] = (IndexType)(keys[k] / m);
        cols[k] = (IndexType)(keys[k] % m);
    }
}

template <typename IndexType>
__global__ void countExtractKernel(const IndexType *row_offsets, const IndexType *col_indices, int n,
                                   const int *row_map, const int *col_map, IndexType *counts)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int r = row_map[i];

        if (r < 0) { continue; }

        IndexType c = 0;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (col_map[col_indices[k]] >= 0) { c++; }
        }

        counts[r] = c;
    }
}

template <typename IndexType, typename ValueType>
__global__ void fillExtractKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                  int n, const int *row_map, const int *col_map, const IndexType *out_offsets,
                                  IndexType *out_cols, ValueType *out_values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int r = row_map[i];

        if (r < 0) { continue; }

        IndexType pos = out_offsets[r];

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const int c = col_map[col_indices[k]];

            if (c >= 0)
            {
                out_cols[pos] = c;
                out_values[pos] = values[k];
                pos++;
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void cooKeysKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                              int n, int64_t m, ValueType alpha, int64_t *keys, ValueType *out_values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            keys[k] = (int64_t) i * m + col_indices[k];
            out_values[k] = alpha * values[k];
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void scaleRowsKernel(const IndexType *row_offsets, int n, const ValueType *d, ValueType *values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            values[k] = d[i] * values[k];
        }
    }
}

template <typename ValueType>
__global__ void gatherKernel(const ValueType *x, const int *idx, int n, ValueType *y)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
    {
        y[k] = x[idx[k]];
    }
}

template <typename ValueType>
__global__ void scatterKernel(const ValueType *y, const int *idx, int n, ValueType *x)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
    {
        x[idx[k]] = y[k];
    }
}

__global__ void maskMapsKernel(const int *mask, const int *set_rank, int n, int *map_set, int *map_unset,
                               int *idx_set, int *idx_unset)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int s = set_rank[i];      // set entries before i
        const int u = i - s;

        if (mask[i])
        {
            map_set[i] = s;
            map_unset[i] = -1;
            idx_set[s] = i;
        }
        else
        {
            map_set[i] = -1;
            map_unset[i] = u;
            idx_unset[u] = i;
        }
    }
}

// blockToScalar is instantiated for device builds only
template <class TConfig, AMGX_MemorySpace M = TConfig::memSpace>
struct BlockToScalar
{
    static void run(const Matrix<TConfig> &A, Matrix<TConfig> &out) { aggregation::PointwiseAggregation<TConfig>::blockToScalar(A, out); }
};

template <class TConfig>
struct BlockToScalar<TConfig, AMGX_host>
{
    static void run(const Matrix<TConfig> &A, Matrix<TConfig> &out) { FatalError("blockToScalar is only implemented on device", AMGX_ERR_NOT_IMPLEMENTED); }
};

// The SpGEMM workspace exists on device only
template <class TConfig, AMGX_MemorySpace M = TConfig::memSpace>
struct SparseProduct
{
    static void run(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C)
    {
        void *wk = CSR_Multiply<TConfig>::csr_workspace_create();
        CSR_Multiply<TConfig>::csr_multiply(A, B, C, wk);
        CSR_Multiply<TConfig>::csr_workspace_delete(wk);
    }
};

template <class TConfig>
struct SparseProduct<TConfig, AMGX_host>
{
    static void run(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C) { FatalError("Sparse products are only implemented on device", AMGX_ERR_NOT_IMPLEMENTED); }
};

template <typename IndexType, typename ValueType>
__global__ void zeroDiagonalKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                   int n, int *count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        ValueType d = ValueType(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (col_indices[k] == i) { d = values[k]; }
        }

        if (d == ValueType(0)) { atomicAdd(count, 1); }
    }
}

struct nonzero_flag
{
    __host__ __device__ int operator()(int v) const { return v != 0 ? 1 : 0; }
};

} // anonymous namespace

template <class T_Config>
void SubmatrixOps<T_Config>::requireDevice(const char *who)
{
    if (TConfig::memSpace != AMGX_device)
    {
        std::string msg = std::string(who) + " is only implemented on device";
        FatalError(msg.c_str(), AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <class T_Config>
const Matrix<T_Config> &SubmatrixOps<T_Config>::scalarView(const Matrix<TConfig> &A, Matrix<TConfig> &storage)
{
    if (A.get_block_size() == 1 && !A.hasProps(DIAG))
    {
        return A;
    }

    BlockToScalar<TConfig>::run(A, storage);
    return storage;
}

template <class T_Config>
void SubmatrixOps<T_Config>::fromCOO(I64Vector &keys, MVector &values, int n, int m, Matrix<TConfig> &out)
{
    requireDevice("SubmatrixOps::fromCOO");
    const int nnz_in = (int) keys.size();
    I64Vector out_keys(nnz_in);
    MVector out_values(nnz_in);
    int nnz = 0;

    if (nnz_in > 0)
    {
        thrust_wrapper::sort_by_key<TConfig::memSpace>(keys.begin(), keys.end(), values.begin());
        cudaCheckError();
        nnz = amgx::thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), out_keys.begin(),
                                          out_values.begin()).first - out_keys.begin();
        cudaCheckError();
    }

    out.set_initialized(0);
    out.addProps(CSR);
    out.resize(n, m, nnz, 1, 1, 1);
    IVector rows(nnz);

    if (nnz > 0)
    {
        splitKeysKernel <<< smNumBlocks(nnz), SM_BLOCK_SIZE>>>(out_keys.raw(), nnz, (int64_t) m, rows.raw(),
                out.col_indices.raw());
        cudaCheckError();
        thrust_wrapper::copy<TConfig::memSpace>(out_values.begin(), out_values.begin() + nnz, out.values.begin());
    }

    amgx::thrust::lower_bound(rows.begin(), rows.end(), amgx::thrust::counting_iterator<int>(0),
                              amgx::thrust::counting_iterator<int>(n + 1), out.row_offsets.begin());
    cudaCheckError();
    out.computeDiagonal();
    out.set_initialized(1);
}

template <class T_Config>
void SubmatrixOps<T_Config>::extract(const Matrix<TConfig> &A, const IVector &row_map, const IVector &col_map,
                                     int n_rows, int n_cols, Matrix<TConfig> &out)
{
    requireDevice("SubmatrixOps::extract");
    const int n = A.get_num_rows();
    IVector offsets(n_rows + 1, 0);

    if (n > 0)
    {
        countExtractKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), n,
                row_map.raw(), col_map.raw(), offsets.raw());
        cudaCheckError();
    }

    thrust_wrapper::exclusive_scan<TConfig::memSpace>(offsets.begin(), offsets.end(), offsets.begin());
    cudaCheckError();
    const int nnz = offsets[n_rows];
    out.set_initialized(0);
    out.addProps(CSR);
    out.resize(n_rows, n_cols, nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(offsets.begin(), offsets.end(), out.row_offsets.begin());

    if (n > 0 && nnz > 0)
    {
        fillExtractKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(),
                A.values.raw(), n, row_map.raw(), col_map.raw(), offsets.raw(), out.col_indices.raw(),
                out.values.raw());
        cudaCheckError();
    }

    out.computeDiagonal();
    out.set_initialized(1);
}

template <class T_Config>
void SubmatrixOps<T_Config>::add(ValueTypeA alpha, const Matrix<TConfig> &A, ValueTypeA beta,
                                 const Matrix<TConfig> &B, Matrix<TConfig> &out)
{
    requireDevice("SubmatrixOps::add");
    const int n = A.get_num_rows(), m = A.get_num_cols();
    const int nnz_a = A.get_num_nz(), nnz_b = B.get_num_nz();
    I64Vector keys(nnz_a + nnz_b);
    MVector values(nnz_a + nnz_b);

    if (n > 0 && nnz_a > 0)
    {
        cooKeysKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(),
                A.values.raw(), n, (int64_t) m, alpha, keys.raw(), values.raw());
        cudaCheckError();
    }

    if (n > 0 && nnz_b > 0)
    {
        cooKeysKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(B.row_offsets.raw(), B.col_indices.raw(),
                B.values.raw(), n, (int64_t) m, beta, keys.raw() + nnz_a, values.raw() + nnz_a);
        cudaCheckError();
    }

    fromCOO(keys, values, n, m, out);
}

template <class T_Config>
void SubmatrixOps<T_Config>::product(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C)
{
    requireDevice("SubmatrixOps::product");
    C.set_initialized(0);
    C.addProps(CSR);
    SparseProduct<TConfig>::run(A, B, C);
    C.computeDiagonal();
    C.set_initialized(1);
}

template <class T_Config>
void SubmatrixOps<T_Config>::scaleRows(Matrix<TConfig> &A, const MVector &d)
{
    requireDevice("SubmatrixOps::scaleRows");
    const int n = A.get_num_rows();

    if (n > 0)
    {
        scaleRowsKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(A.row_offsets.raw(), n, d.raw(), A.values.raw());
        cudaCheckError();
    }
}

template <class T_Config>
int SubmatrixOps<T_Config>::countZeroDiagonal(const Matrix<TConfig> &A)
{
    requireDevice("SubmatrixOps::countZeroDiagonal");
    const int n = A.get_num_rows();
    IVector count(1, 0);

    if (n > 0)
    {
        zeroDiagonalKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(),
                n, count.raw());
        cudaCheckError();
    }

    return count[0];
}

template <class T_Config>
void SubmatrixOps<T_Config>::gather(const VVector &x, const IVector &idx, VVector &y)
{
    const int n = (int) idx.size();

    if (n > 0)
    {
        gatherKernel <<< smNumBlocks(n), SM_BLOCK_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            x.raw(), idx.raw(), n, y.raw());
        cudaCheckError();
    }

    y.dirtybit = 1;
}

template <class T_Config>
void SubmatrixOps<T_Config>::scatter(const VVector &y, const IVector &idx, VVector &x)
{
    const int n = (int) idx.size();

    if (n > 0)
    {
        scatterKernel <<< smNumBlocks(n), SM_BLOCK_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            y.raw(), idx.raw(), n, x.raw());
        cudaCheckError();
    }

    x.dirtybit = 1;
}

template <class T_Config>
int SubmatrixOps<T_Config>::splitMask(const IVector &mask, IVector &map_set, IVector &map_unset, IVector &idx_set,
                                      IVector &idx_unset)
{
    requireDevice("SubmatrixOps::splitMask");
    const int n = (int) mask.size();
    IVector rank(n + 1, 0);
    amgx::thrust::transform(mask.begin(), mask.end(), rank.begin(), nonzero_flag());
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(rank.begin(), rank.end(), rank.begin());
    cudaCheckError();
    const int n_set = rank[n];
    map_set.resize(n);
    map_unset.resize(n);
    idx_set.resize(n_set);
    idx_unset.resize(n - n_set);

    if (n > 0)
    {
        maskMapsKernel <<< smNumBlocks(n), SM_BLOCK_SIZE>>>(mask.raw(), rank.raw(), n, map_set.raw(),
                map_unset.raw(), idx_set.raw(), idx_unset.raw());
        cudaCheckError();
    }

    return n_set;
}

#define AMGX_CASE_LINE(CASE) template struct SubmatrixOps<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace composite
} // namespace amgx
