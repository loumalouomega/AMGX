// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Strength-of-connection criterion adapted from AMGCL
// (amgcl/coarsening/plain_aggregates.hpp, amgcl/coarsening/smoothed_aggregation.hpp),
// Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License
// (see LICENSES/MIT.txt).

#include <aggregation/strength/symmetric_scaled.h>
#include <distributed/distributed_manager.h>
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

const int SA_BLOCK_SIZE = 256;

inline int saNumBlocks(int n)
{
    return std::min<int>(4096, (n + SA_BLOCK_SIZE - 1) / SA_BLOCK_SIZE);
}

template <typename IndexType, typename ValueType>
__global__ void extractDiagonalKernel(const IndexType *row_offsets, const IndexType *col_indices,
                                      const ValueType *values, int num_rows, ValueType *diag)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        ValueType d = ValueType(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (col_indices[k] == i) { d = values[k]; }
        }

        diag[i] = d;
    }
}

template <typename IndexType, typename ValueType>
__global__ void strongConnectionsKernel(const IndexType *row_offsets, const IndexType *col_indices,
                                        const ValueType *values, const ValueType *diag, int num_rows,
                                        double eps_squared, bool *strong, ValueType *filtered_diag)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const double eps_dia_i = eps_squared * (double) diag[i];
        ValueType d_f = ValueType(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType c = col_indices[k];
            const ValueType v = values[k];
            const bool s = (c != i) && (eps_dia_i * (double) diag[c] < (double) v * (double) v);
            strong[k] = s;

            if (!s) { d_f += v; }
        }

        filtered_diag[i] = d_f;
    }
}

template <typename IndexType>
__global__ void countFilteredKernel(const IndexType *row_offsets, const IndexType *col_indices,
                                    const bool *strong, int num_rows, IndexType *row_counts)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        IndexType count = 0;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (strong[k] || col_indices[k] == i) { count++; }
        }

        row_counts[i] = count;
    }
}

template <typename IndexType, typename ValueType>
__global__ void fillFilteredScaledKernel(const IndexType *row_offsets, const IndexType *col_indices,
        const ValueType *values, const bool *strong, const ValueType *filtered_diag, int num_rows,
        const IndexType *S_row_offsets, IndexType *S_col_indices, ValueType *S_values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const ValueType d_f = filtered_diag[i];
        const ValueType d_inv = (d_f != ValueType(0)) ? ValueType(1) / d_f : ValueType(0);
        IndexType pos = S_row_offsets[i];

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType c = col_indices[k];

            if (c == i)
            {
                S_col_indices[pos] = c;
                S_values[pos++] = ValueType(1);
            }
            else if (strong[k])
            {
                S_col_indices[pos] = c;
                S_values[pos++] = d_inv * values[k];
            }
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void haloStrengthKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                   const ValueType *diag, int num_rows, double eps_squared, bool *strong, ValueType *filtered_diag)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const double eps_dia_i = eps_squared * (double) diag[i];
        ValueType d_f = ValueType(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const IndexType c = col_indices[k];
            const ValueType v = values[k];

            if (c >= num_rows) { strong[k] = eps_dia_i * (double) diag[c] < (double) v * (double) v; }

            if (!strong[k]) { d_f += v; }
        }

        filtered_diag[i] = d_f;
    }
}

// Owned rows plus the halo of the columns (distributed)
template <class Mat>
int fullSize(const Mat &A)
{
    int size = std::max((int) A.get_num_rows(), (int) A.get_num_cols());

    if (A.is_matrix_distributed())
    {
        int offset, full;
        A.getOffsetAndSizeForView(FULL, &offset, &full);
        size = std::max(size, full);
    }

    return size;
}

} // anonymous namespace

template <class T_Config>
void SymmetricScaledStrength<T_Config>::computeStrongConnections(const Matrix<TConfig> &A,
        double eps_strong, BVector &strong, MVector &filtered_diag)
{
    if (A.get_block_size() != 1 || A.hasProps(DIAG))
    {
        FatalError("SYMMETRIC_SCALED strength requires a scalar CSR matrix without external diagonal",
                   AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    const int num_rows = A.get_num_rows();
    const bool distributed = !A.is_matrix_singleGPU();
    // Distributed: columns reach the halo, whose diagonal comes from the neighbours (the halo rows of
    // A are identity rows with one ring)
    MVector diag(distributed ? fullSize(A) : num_rows);
    strong.resize(A.get_num_nz());
    filtered_diag.resize(num_rows);

    if (num_rows > 0)
    {
        extractDiagonalKernel <<< saNumBlocks(num_rows), SA_BLOCK_SIZE>>>(
            A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), num_rows, diag.raw());
        cudaCheckError();
    }

    if (distributed)
    {
        diag.dirtybit = 1;
        A.manager->exchange_halo(diag, diag.tag);
    }

    if (num_rows == 0) { return; }

    strongConnectionsKernel <<< saNumBlocks(num_rows), SA_BLOCK_SIZE>>>(
        A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), diag.raw(), num_rows,
        eps_strong * eps_strong, strong.raw(), filtered_diag.raw());
    cudaCheckError();
}

template <class T_Config>
void SymmetricScaledStrength<T_Config>::patchHaloStrength(const Matrix<TConfig> &A, double eps_strong, BVector &strong,
        MVector &filtered_diag)
{
    const int num_rows = A.get_num_rows();
    MVector diag(fullSize(A));

    if (num_rows > 0)
    {
        extractDiagonalKernel <<< saNumBlocks(num_rows), SA_BLOCK_SIZE>>>(
            A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), num_rows, diag.raw());
        cudaCheckError();
    }

    diag.dirtybit = 1;
    A.manager->exchange_halo(diag, diag.tag);

    if (num_rows > 0)
    {
        haloStrengthKernel <<< saNumBlocks(num_rows), SA_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(),
                A.values.raw(), diag.raw(), num_rows, eps_strong * eps_strong, strong.raw(), filtered_diag.raw());
        cudaCheckError();
    }
}

template <class T_Config>
void SymmetricScaledStrength<T_Config>::buildFilteredScaledOperator(const Matrix<TConfig> &A,
        const BVector &strong, const MVector &filtered_diag, Matrix<TConfig> &S)
{
    const int num_rows = A.get_num_rows();
    IVector row_counts(num_rows + 1, 0);

    if (num_rows > 0)
    {
        countFilteredKernel <<< saNumBlocks(num_rows), SA_BLOCK_SIZE>>>(
            A.row_offsets.raw(), A.col_indices.raw(), strong.raw(), num_rows, row_counts.raw());
        cudaCheckError();
    }

    thrust_wrapper::exclusive_scan<TConfig::memSpace>(row_counts.begin(), row_counts.end(), row_counts.begin());
    cudaCheckError();
    const int nnz = row_counts[num_rows];
    S.set_initialized(0);
    S.addProps(CSR);
    S.resize(num_rows, A.is_matrix_distributed() ? fullSize(A) : num_rows, nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(row_counts.begin(), row_counts.end(), S.row_offsets.begin());
    cudaCheckError();

    if (num_rows > 0)
    {
        fillFilteredScaledKernel <<< saNumBlocks(num_rows), SA_BLOCK_SIZE>>>(
            A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), strong.raw(), filtered_diag.raw(),
            num_rows, S.row_offsets.raw(), S.col_indices.raw(), S.values.raw());
        cudaCheckError();
    }

    S.computeDiagonal();
    S.set_initialized(1);
}

#define AMGX_CASE_LINE(CASE) template struct SymmetricScaledStrength<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace aggregation
} // namespace amgx
