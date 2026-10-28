// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoothed aggregation prolongator adapted from AMGCL
// (amgcl/coarsening/smoothed_aggregation.hpp, amgcl/coarsening/tentative_prolongation.hpp,
// amgcl/backend/builtin.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <aggregation/sa_prolongation.h>
#include <csr_multiply.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>

namespace amgx
{
namespace aggregation
{

namespace
{

const int SAP_BLOCK_SIZE = 256;

inline int sapNumBlocks(int n)
{
    return std::min<int>(4096, (n + SAP_BLOCK_SIZE - 1) / SAP_BLOCK_SIZE);
}

template <typename IndexType, typename ValueType>
__global__ void absRowSumKernel(const IndexType *row_offsets, const ValueType *values, int num_rows,
                                double *row_sums)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        double s = 0.0;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            s += fabs((double) values[k]);
        }

        row_sums[i] = s;
    }
}

template <typename IndexType, typename ValueType>
__global__ void spmvKernel(const IndexType *row_offsets, const IndexType *col_indices,
                           const ValueType *values, int num_rows, const double *x, double *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        double s = 0.0;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            s += (double) values[k] * x[col_indices[k]];
        }

        y[i] = s;
    }
}

// Deterministic pseudo-random start vector with entries in [-1, 1]
__global__ void hashVectorKernel(int n, double *x)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        unsigned int a = (unsigned int) i * 2654435761u + 0x9e3779b9u;
        a ^= a >> 16;
        a *= 0x85ebca6bu;
        a ^= a >> 13;
        a *= 0xc2b2ae35u;
        a ^= a >> 16;
        x[i] = 2.0 * ((double) a / 4294967295.0) - 1.0;
    }
}

__global__ void scaleVectorKernel(int n, double alpha, const double *x, double *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = alpha * x[i];
    }
}

template <typename IndexType, typename ValueType>
__global__ void tentativeProlongatorKernel(const int *aggregates, int num_rows,
        const IndexType *row_offsets, IndexType *col_indices, ValueType *values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        if (aggregates[i] >= 0)
        {
            col_indices[row_offsets[i]] = aggregates[i];
            values[row_offsets[i]] = ValueType(1);
        }
    }
}

template <typename IndexType, typename ValueType>
__global__ void jacobiOperatorKernel(const IndexType *row_offsets, const IndexType *col_indices,
                                     int num_rows, ValueType omega, ValueType *values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            const ValueType v = -omega * values[k];
            values[k] = (col_indices[k] == i) ? ValueType(1) + v : v;
        }
    }
}

struct dot_functor
{
    const double *x, *y;
    dot_functor(const double *x, const double *y) : x(x), y(y) {}
    __host__ __device__ double operator()(int i) const { return x[i] * y[i]; }
};

struct is_aggregated
{
    __host__ __device__ int operator()(int a) const { return a >= 0 ? 1 : 0; }
};

template <class DVector>
double dotProduct(const DVector &x, const DVector &y)
{
    return thrust_wrapper::transform_reduce<AMGX_device>(
               amgx::thrust::counting_iterator<int>(0), amgx::thrust::counting_iterator<int>((int) x.size()),
               dot_functor(x.raw(), y.raw()), 0.0, amgx::thrust::plus<double>());
}

} // anonymous namespace

template <class T_Config>
double SAProlongation<T_Config>::estimateSpectralRadius(const Matrix<TConfig> &S, int power_iters)
{
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecDouble>::Type> DVector;
    const int n = S.get_num_rows();

    if (n == 0) { return 0.0; }

    const int num_blocks = sapNumBlocks(n);
    DVector b0(n), b1(n);

    if (power_iters <= 0)
    {
        // Gershgorin disk theorem
        absRowSumKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(S.row_offsets.raw(), S.values.raw(), n, b0.raw());
        cudaCheckError();
        double radius = thrust_wrapper::reduce<AMGX_device>(b0.begin(), b0.end(), 0.0, amgx::thrust::maximum<double>());
        cudaCheckError();
        return radius;
    }

    // Power iteration, radius = <S b0, b0> with ||b0|| = 1
    hashVectorKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(n, b0.raw());
    cudaCheckError();
    double b_norm = std::sqrt(dotProduct(b0, b0));
    scaleVectorKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(n, 1.0 / b_norm, b0.raw(), b0.raw());
    cudaCheckError();
    double radius = 0.0;

    for (int iter = 0; iter < power_iters;)
    {
        spmvKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(S.row_offsets.raw(), S.col_indices.raw(), S.values.raw(),
                n, b0.raw(), b1.raw());
        cudaCheckError();
        radius = dotProduct(b1, b0);

        if (++iter < power_iters)
        {
            b_norm = std::sqrt(dotProduct(b1, b1));

            if (b_norm == 0.0) { break; }

            scaleVectorKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(n, 1.0 / b_norm, b1.raw(), b0.raw());
            cudaCheckError();
        }
    }

    return radius < 0.0 ? 2.0 : radius;
}

template <class T_Config>
void SAProlongation<T_Config>::buildTentativeProlongator(const IVector &aggregates, int num_aggregates,
        Matrix<TConfig> &P_tent)
{
    const int num_rows = aggregates.size();
    IVector row_offsets(num_rows + 1, 0);
    thrust_wrapper::transform<TConfig::memSpace>(aggregates.begin(), aggregates.end(), row_offsets.begin(), is_aggregated());
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(row_offsets.begin(), row_offsets.end(), row_offsets.begin());
    cudaCheckError();
    const int nnz = row_offsets[num_rows];
    P_tent.set_initialized(0);
    P_tent.addProps(CSR);
    P_tent.resize(num_rows, num_aggregates, nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(row_offsets.begin(), row_offsets.end(), P_tent.row_offsets.begin());
    cudaCheckError();

    if (num_rows > 0)
    {
        tentativeProlongatorKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(aggregates.raw(), num_rows,
                P_tent.row_offsets.raw(), P_tent.col_indices.raw(), P_tent.values.raw());
        cudaCheckError();
    }

    P_tent.computeDiagonal();
    P_tent.set_initialized(1);
}

template <class T_Config>
void SAProlongation<T_Config>::smoothProlongator(Matrix<TConfig> &S, double omega,
        const Matrix<TConfig> &P_tent, Matrix<TConfig> &P, void *csr_workspace)
{
    const int num_rows = S.get_num_rows();

    if (num_rows > 0)
    {
        jacobiOperatorKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(S.row_offsets.raw(),
                S.col_indices.raw(), num_rows, ValueTypeA(omega), S.values.raw());
        cudaCheckError();
    }

    P.set_initialized(0);
    P.addProps(CSR);
    CSR_Multiply<TConfig>::csr_multiply(S, P_tent, P, csr_workspace);
}

#define AMGX_CASE_LINE(CASE) template struct SAProlongation<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace aggregation
} // namespace amgx
