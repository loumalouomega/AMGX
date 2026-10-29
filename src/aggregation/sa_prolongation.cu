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
#include <distributed/distributed_manager.h>
#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>

#include <climits>
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
__global__ void hashVectorKernel(int n, int64_t offset, double *x)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        unsigned int a = (unsigned int)(i + offset) * 2654435761u + 0x9e3779b9u;
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


// Row i of P = J P_tent when every row of P_tent has at most one entry: the entries J_ij p_j of the
// row, merged by coarse column (sorted, at most SAP_MAX_ROW per row). Count pass (vals == NULL) and fill.
const int SAP_MAX_ROW = 64;

template <typename IndexType, typename ValueType>
__global__ void singleEntryProductKernel(const IndexType *J_ro, const IndexType *J_ci, const ValueType *J_v, int num_rows,
        const IndexType *T_ro, const IndexType *T_ci, const ValueType *T_v, const IndexType *P_ro, int *counts,
        int *overflow, IndexType *P_ci, ValueType *P_v)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        IndexType cols[SAP_MAX_ROW];
        ValueType vals[SAP_MAX_ROW];
        int m = 0;
        bool full = false;

        for (IndexType k = J_ro[i]; k < J_ro[i + 1]; k++)
        {
            const IndexType j = J_ci[k];

            if (T_ro[j] == T_ro[j + 1]) { continue; }

            const IndexType c = T_ci[T_ro[j]];
            const ValueType v = J_v[k] * T_v[T_ro[j]];
            // insert into the sorted list, merging equal columns
            int q = m;

            while (q > 0 && cols[q - 1] > c) { q--; }

            if (q > 0 && cols[q - 1] == c) { vals[q - 1] = vals[q - 1] + v; continue; }

            if (m == SAP_MAX_ROW) { full = true; break; }

            for (int r = m; r > q; r--) { cols[r] = cols[r - 1]; vals[r] = vals[r - 1]; }

            cols[q] = c;
            vals[q] = v;
            m++;
        }

        if (P_v == NULL)
        {
            counts[i] = m;

            if (full) { atomicExch(overflow, 1); }
        }
        else
        {
            for (int q = 0; q < m; q++) { P_ci[P_ro[i] + q] = cols[q]; P_v[P_ro[i] + q] = vals[q]; }
        }
    }
}

// Truncation: kept count and rescaling factor per row (count pass), then the kept entries scaled (fill)
template <typename IndexType, typename ValueType>
__global__ void truncateRowSumKernel(const IndexType *ro, const IndexType *ci, const ValueType *v, int num_rows,
                                     double drop_tol, int *counts, const IndexType *new_ro, IndexType *new_ci, ValueType *new_v)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        double max_abs = 0.0;
        ValueType sum = ValueType(0), kept_sum = ValueType(0);

        for (IndexType k = ro[i]; k < ro[i + 1]; k++)
        {
            max_abs = fmax(max_abs, fabs((double) v[k]));
            sum = sum + v[k];
        }

        const double threshold = drop_tol * max_abs;
        int count = 0;

        for (IndexType k = ro[i]; k < ro[i + 1]; k++)
            if (fabs((double) v[k]) >= threshold) { kept_sum = kept_sum + v[k]; count++; }

        if (new_v == NULL)
        {
            counts[i] = count;
            continue;
        }

        const ValueType scale = (kept_sum == ValueType(0)) ? ValueType(1) : sum / kept_sum;
        IndexType w = new_ro[i];

        for (IndexType k = ro[i]; k < ro[i + 1]; k++)
            if (fabs((double) v[k]) >= threshold) { new_ci[w] = ci[k]; new_v[w] = v[k] * scale; w++; }
    }
}

template <typename IndexType, typename ValueType>
__global__ void blockTentativeKernel(const int *aggregates, int num_rows, int b,
                                     const IndexType *row_offsets, IndexType *col_indices, ValueType *values)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const int a = aggregates[i / b];

        if (a >= 0)
        {
            col_indices[row_offsets[i]] = a * b + i % b;
            values[row_offsets[i]] = ValueType(1);
        }
    }
}

struct row_entries
{
    int b, per_row;
    const int *aggregates;
    row_entries(const int *aggregates, int b, int per_row) : b(b), per_row(per_row), aggregates(aggregates) {}
    __host__ __device__ int operator()(int i) const { return aggregates[i / b] >= 0 ? per_row : 0; }
};

// Unknown-level aggregate ids, removed unknowns are sorted last
__global__ void unknownAggregatesKernel(const int *aggregates, int num_rows, int b, int *keys, int *order)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const int a = aggregates[i / b];
        keys[i] = a >= 0 ? a : INT_MAX;
        order[i] = i;
    }
}

__global__ void aggregateUnknownCountKernel(const int *aggregates, int num_rows, int b, int *sizes)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const int a = aggregates[i / b];

        if (a >= 0) { atomicAdd(&sizes[a], 1); }
    }
}

__device__ __forceinline__ double warpSum(double v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }

    return v;
}

// One warp per aggregate: Householder QR (LAPACK xGEQR2/xORG2R conventions, as in AMGCL) of the
// m x K block of B, stored column-major in W. Q goes to the rows of P_tent, R to the coarse nullspace.
template <typename IndexType, typename ValueType>
__global__ void nullspaceQRKernel(const int *order, const int *agg_offsets, int num_aggregates, int K,
                                  const double *B, double *W, double *Q, double *tau_all,
                                  const IndexType *P_offsets, IndexType *P_cols, ValueType *P_vals, double *Bc)
{
    const int lane = threadIdx.x & 31;
    const int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int a = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; a < num_aggregates; a += num_warps)
    {
        const int beg = agg_offsets[a];
        const int m = agg_offsets[a + 1] - beg;
        double *Wa = W + (size_t) beg * K;
        double *Qa = Q + (size_t) beg * K;
        double *tau = tau_all + (size_t) a * K;

        for (int r = lane; r < m; r += 32)
            for (int c = 0; c < K; c++) { Wa[c * m + r] = B[(size_t) order[beg + r] * K + c]; }

        __syncwarp();

        for (int j = 0; j < K; j++)
        {
            const double alpha = Wa[j * m + j];
            double xnorm2 = 0.0;

            for (int r = j + 1 + lane; r < m; r += 32) { xnorm2 += Wa[j * m + r] * Wa[j * m + r]; }

            xnorm2 = warpSum(xnorm2);
            double t = 0.0;

            if (xnorm2 > 0.0)
            {
                const double beta = -copysign(sqrt(alpha * alpha + xnorm2), alpha);
                t = (beta - alpha) / beta;
                const double scale = 1.0 / (alpha - beta);

                for (int r = j + 1 + lane; r < m; r += 32) { Wa[j * m + r] *= scale; }

                __syncwarp();

                if (lane == 0) { Wa[j * m + j] = beta; }
            }

            if (lane == 0) { tau[j] = t; }

            __syncwarp();

            // Apply H_j = I - t [1; v] [1; v]^T to the remaining columns
            for (int c = j + 1; c < K; c++)
            {
                double w = 0.0;

                for (int r = j + 1 + lane; r < m; r += 32) { w += Wa[j * m + r] * Wa[c * m + r]; }

                w = warpSum(w) + Wa[c * m + j];
                __syncwarp();

                if (lane == 0) { Wa[c * m + j] -= t * w; }

                for (int r = j + 1 + lane; r < m; r += 32) { Wa[c * m + r] -= t * w * Wa[j * m + r]; }

                __syncwarp();
            }
        }

        // Q = H_0 ... H_{K-1} [I; 0]
        for (int r = lane; r < m; r += 32)
            for (int c = 0; c < K; c++) { Qa[c * m + r] = (r == c) ? 1.0 : 0.0; }

        __syncwarp();

        for (int j = K - 1; j >= 0; j--)
        {
            const double t = tau[j];

            for (int c = j; c < K; c++)
            {
                double w = 0.0;

                for (int r = j + 1 + lane; r < m; r += 32) { w += Wa[j * m + r] * Qa[c * m + r]; }

                w = warpSum(w) + Qa[c * m + j];
                __syncwarp();

                if (lane == 0) { Qa[c * m + j] -= t * w; }

                for (int r = j + 1 + lane; r < m; r += 32) { Qa[c * m + r] -= t * w * Wa[j * m + r]; }

                __syncwarp();
            }
        }

        for (int r = lane; r < m; r += 32)
        {
            const IndexType pos = P_offsets[order[beg + r]];

            for (int c = 0; c < K; c++)
            {
                P_cols[pos + c] = a * K + c;
                P_vals[pos + c] = ValueType(Qa[c * m + r]);
            }
        }

        for (int idx = lane; idx < K * K; idx += 32)
        {
            const int i = idx / K, j = idx % K;
            Bc[((size_t) a * K + i) * K + j] = (j >= i) ? Wa[j * m + i] : 0.0;
        }

        __syncwarp();
    }
}

__global__ void blockIdentityKernel(int n, int b, double *Bc)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n * b; i += gridDim.x * blockDim.x)
    {
        Bc[i] = ((i / b) % b == i % b) ? 1.0 : 0.0;
    }
}

// Minimum-norm correction of the kept entries of a prolongator row so that sum_k p_k Bc(c_k, :)
// is unchanged. Returns false when the kept columns cannot reproduce it.
template <typename IndexType, typename ValueType, int MAXK>
__device__ bool truncateRow(const IndexType *cols, const ValueType *vals, int len, int K, const double *Bc,
                            double threshold, double *y)
{
    double r[MAXK], G[MAXK * MAXK];

    for (int i = 0; i < K; i++)
    {
        r[i] = 0.0;

        for (int j = 0; j < K; j++) { G[i * MAXK + j] = 0.0; }
    }

    for (int k = 0; k < len; k++)
    {
        const double *bk = Bc + (size_t) cols[k] * K;
        const double p = (double) vals[k];

        if (fabs(p) >= threshold)
        {
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) { G[i * MAXK + j] += bk[i] * bk[j]; }
        }
        else
        {
            for (int i = 0; i < K; i++) { r[i] += p * bk[i]; }
        }
    }

    // Cholesky of the semi-definite G, skipping null directions (y_i = 0)
    double trace = 0.0, rnorm = 0.0;

    for (int i = 0; i < K; i++) { trace += G[i * MAXK + i]; rnorm += r[i] * r[i]; }

    const double tol = 1e-12 * (trace > 0.0 ? trace : 1.0);
    bool null_dir[MAXK];

    for (int j = 0; j < K; j++)
    {
        double d = G[j * MAXK + j];

        for (int k = 0; k < j; k++) { d -= G[j * MAXK + k] * G[j * MAXK + k]; }

        null_dir[j] = !(d > tol);
        d = null_dir[j] ? 0.0 : sqrt(d);
        G[j * MAXK + j] = d;

        for (int i = j + 1; i < K; i++)
        {
            double s = G[i * MAXK + j];

            for (int k = 0; k < j; k++) { s -= G[i * MAXK + k] * G[j * MAXK + k]; }

            G[i * MAXK + j] = null_dir[j] ? 0.0 : s / d;
        }
    }

    double z[MAXK];

    for (int i = 0; i < K; i++)
    {
        double s = r[i];

        for (int k = 0; k < i; k++) { s -= G[i * MAXK + k] * z[k]; }

        z[i] = null_dir[i] ? 0.0 : s / G[i * MAXK + i];
    }

    for (int i = K - 1; i >= 0; i--)
    {
        double s = z[i];

        for (int k = i + 1; k < K; k++) { s -= G[k * MAXK + i] * y[k]; }

        y[i] = null_dir[i] ? 0.0 : s / G[i * MAXK + i];
    }

    // The dropped part must be reproducible by the kept columns: check G y = r
    double err = 0.0;

    for (int i = 0; i < K; i++)
    {
        double s = -r[i];

        for (int k = 0; k < len; k++)
        {
            const double *bk = Bc + (size_t) cols[k] * K;

            if (fabs((double) vals[k]) >= threshold)
            {
                double bky = 0.0;

                for (int j = 0; j < K; j++) { bky += bk[j] * y[j]; }

                s += bk[i] * bky;
            }
        }

        err += s * s;
    }

    return err <= 1e-16 * (rnorm > 1e-300 ? rnorm : 1e-300) || rnorm == 0.0;
}

template <typename IndexType, typename ValueType, int MAXK>
__global__ void truncateNullspaceKernel(const IndexType *row_offsets, const IndexType *cols, const ValueType *vals,
                                        int num_rows, int K, const double *Bc, double drop_tol, bool fill,
                                        IndexType *new_counts, const IndexType *new_offsets, IndexType *new_cols,
                                        ValueType *new_vals)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const IndexType beg = row_offsets[i];
        const int len = row_offsets[i + 1] - beg;
        double row_max = 0.0;

        for (int k = 0; k < len; k++) { row_max = fmax(row_max, fabs((double) vals[beg + k])); }

        double threshold = drop_tol * row_max;
        double y[MAXK];

        if (!truncateRow<IndexType, ValueType, MAXK>(cols + beg, vals + beg, len, K, Bc, threshold, y))
        {
            threshold = 0.0; // keep the row untouched

            for (int j = 0; j < K; j++) { y[j] = 0.0; }
        }

        IndexType count = 0;

        for (int k = 0; k < len; k++)
        {
            const double p = (double) vals[beg + k];

            if (fabs(p) >= threshold)
            {
                if (fill)
                {
                    const double *bk = Bc + (size_t) cols[beg + k] * K;
                    double delta = 0.0;

                    for (int j = 0; j < K; j++) { delta += bk[j] * y[j]; }

                    new_cols[new_offsets[i] + count] = cols[beg + k];
                    new_vals[new_offsets[i] + count] = ValueType(p + delta);
                }

                count++;
            }
        }

        if (!fill) { new_counts[i] = count; }
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

// <x, y> over the first n entries, summed over the partitions of comm when distributed
template <class DVector, class Mat>
double dotProduct(const DVector &x, const DVector &y, int n, const Mat *comm)
{
    double d = thrust_wrapper::transform_reduce<AMGX_device>(
                   amgx::thrust::counting_iterator<int>(0), amgx::thrust::counting_iterator<int>(n),
                   dot_functor(x.raw(), y.raw()), 0.0, amgx::thrust::plus<double>());

    if (comm != NULL) { comm->manager->global_reduce_sum(&d); }

    return d;
}

} // anonymous namespace

template <class T_Config>
double SAProlongation<T_Config>::estimateSpectralRadius(const Matrix<TConfig> &S, int power_iters, const Matrix<TConfig> *comm)
{
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecDouble>::Type> DVector;
    // every partition takes part in the reductions, also one without neighbours
    const Matrix<TConfig> *dist = (comm != NULL && comm->is_matrix_distributed() && comm->manager->get_num_partitions() > 1) ? comm : NULL;
    const int n = S.get_num_rows();
    // the power-iteration vector carries the halo of S's columns
    const int n_full = dist != NULL ? std::max(n, (int) S.get_num_cols()) : n;

    if (dist == NULL && n == 0) { return 0.0; }

    const int num_blocks = sapNumBlocks(std::max(n, 1));
    DVector b0(n_full, 0.0), b1(std::max(n, 1));

    if (power_iters <= 0)
    {
        // Gershgorin disk theorem
        double radius = 0.0;

        if (n > 0)
        {
            absRowSumKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(S.row_offsets.raw(), S.values.raw(), n, b0.raw());
            cudaCheckError();
            radius = thrust_wrapper::reduce<AMGX_device>(b0.begin(), b0.begin() + n, 0.0, amgx::thrust::maximum<double>());
            cudaCheckError();
        }

#ifdef AMGX_WITH_MPI

        if (dist != NULL)
        {
            double global = radius;
            MPI_Allreduce(&radius, &global, 1, MPI_DOUBLE, MPI_MAX, dist->manager->getComms()->get_mpi_comm());
            radius = global;
        }

#endif
        return radius;
    }

    // Power iteration, radius = <S b0, b0> with ||b0|| = 1
    if (n > 0)
    {
        hashVectorKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(n, dist != NULL ? dist->manager->base_index() : 0, b0.raw());
        cudaCheckError();
    }

    double b_norm = std::sqrt(dotProduct(b0, b0, n, dist));

    if (n > 0)
    {
        scaleVectorKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(n, 1.0 / b_norm, b0.raw(), b0.raw());
        cudaCheckError();
    }

    double radius = 0.0;

    for (int iter = 0; iter < power_iters;)
    {
        if (dist != NULL)
        {
            b0.dirtybit = 1;
            dist->manager->exchange_halo(b0, b0.tag);
        }

        if (n > 0)
        {
            spmvKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(S.row_offsets.raw(), S.col_indices.raw(), S.values.raw(),
                    n, b0.raw(), b1.raw());
            cudaCheckError();
        }

        radius = dotProduct(b1, b0, n, dist);

        if (++iter < power_iters)
        {
            b_norm = std::sqrt(dotProduct(b1, b1, n, dist));

            if (b_norm == 0.0) { break; }

            if (n > 0)
            {
                scaleVectorKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(n, 1.0 / b_norm, b1.raw(), b0.raw());
                cudaCheckError();
            }
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
    // P_tent with at most one entry per row: P row by row (merging J's entries by coarse column)
    const int tent_rows = P_tent.get_num_rows();
    const bool single = tent_rows > 0 && P_tent.get_num_nz() <= tent_rows &&
                        amgx::thrust::inner_product(P_tent.row_offsets.begin() + 1, P_tent.row_offsets.begin() + tent_rows + 1,
                                P_tent.row_offsets.begin(), 0, amgx::thrust::maximum<int>(), amgx::thrust::minus<int>()) <= 1;

    if (single && num_rows > 0)
    {
        IVector counts(num_rows + 1, 0), overflow(1, 0);
        singleEntryProductKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(S.row_offsets.raw(), S.col_indices.raw(),
                S.values.raw(), num_rows, P_tent.row_offsets.raw(), P_tent.col_indices.raw(), P_tent.values.raw(),
                (const IndexType *) NULL, counts.raw(), overflow.raw(), (IndexType *) NULL, (ValueTypeA *) NULL);
        cudaCheckError();

        if (overflow[0] == 0)
        {
            thrust_wrapper::exclusive_scan<TConfig::memSpace>(counts.begin(), counts.end(), counts.begin());
            cudaCheckError();
            const int nnz = counts[num_rows];
            P.resize(num_rows, P_tent.get_num_cols(), nnz, 1, 1, 1);
            thrust_wrapper::copy<TConfig::memSpace>(counts.begin(), counts.end(), P.row_offsets.begin());
            singleEntryProductKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(S.row_offsets.raw(), S.col_indices.raw(),
                    S.values.raw(), num_rows, P_tent.row_offsets.raw(), P_tent.col_indices.raw(), P_tent.values.raw(),
                    P.row_offsets.raw(), counts.raw(), overflow.raw(), P.col_indices.raw(), P.values.raw());
            cudaCheckError();
            P.computeDiagonal();
            P.set_initialized(1);
            return;
        }
    }

    CSR_Multiply<TConfig>::csr_multiply(S, P_tent, P, csr_workspace);
}

template <class T_Config>
void SAProlongation<T_Config>::truncateRowSum(Matrix<TConfig> &P, double drop_tol)
{
    const int num_rows = P.get_num_rows();

    if (num_rows == 0) { return; }

    IVector counts(num_rows + 1, 0);
    truncateRowSumKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(P.row_offsets.raw(), P.col_indices.raw(),
            P.values.raw(), num_rows, drop_tol, counts.raw(), (const IndexType *) NULL, (IndexType *) NULL, (ValueTypeA *) NULL);
    cudaCheckError();
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(counts.begin(), counts.end(), counts.begin());
    cudaCheckError();
    const int nnz = counts[num_rows];

    if (nnz == P.get_num_nz()) { return; }  // nothing dropped (as truncateByFactor, no rescaling)

    Matrix<TConfig> T;
    T.set_initialized(0);
    T.addProps(CSR);
    T.resize(num_rows, P.get_num_cols(), nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(counts.begin(), counts.end(), T.row_offsets.begin());
    truncateRowSumKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(P.row_offsets.raw(), P.col_indices.raw(),
            P.values.raw(), num_rows, drop_tol, counts.raw(), T.row_offsets.raw(), T.col_indices.raw(), T.values.raw());
    cudaCheckError();
    P.set_initialized(0);
    P.row_offsets.swap(T.row_offsets);
    P.col_indices.swap(T.col_indices);
    P.values.swap(T.values);
    P.set_num_nz(nnz);
    P.computeDiagonal();
    P.set_initialized(1);
}

template <class T_Config>
void SAProlongation<T_Config>::buildTentativeProlongatorBlock(const IVector &aggregates, int num_aggregates,
        int block_size, Matrix<TConfig> &P_tent)
{
    const int num_rows = aggregates.size() * block_size;
    IVector row_offsets(num_rows + 1, 0);
    thrust_wrapper::transform<TConfig::memSpace>(amgx::thrust::counting_iterator<int>(0), amgx::thrust::counting_iterator<int>(num_rows),
            row_offsets.begin(), row_entries(aggregates.raw(), block_size, 1));
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(row_offsets.begin(), row_offsets.end(), row_offsets.begin());
    cudaCheckError();
    const int nnz = row_offsets[num_rows];
    P_tent.set_initialized(0);
    P_tent.addProps(CSR);
    P_tent.resize(num_rows, num_aggregates * block_size, nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(row_offsets.begin(), row_offsets.end(), P_tent.row_offsets.begin());

    if (num_rows > 0)
    {
        blockTentativeKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(aggregates.raw(), num_rows, block_size,
                P_tent.row_offsets.raw(), P_tent.col_indices.raw(), P_tent.values.raw());
        cudaCheckError();
    }

    P_tent.computeDiagonal();
    P_tent.set_initialized(1);
}

template <class T_Config>
void SAProlongation<T_Config>::buildTentativeProlongatorNullspace(const IVector &aggregates, int num_aggregates,
        int block_size, int K, const DVector &B, Matrix<TConfig> &P_tent, DVector &Bc)
{
    const int num_rows = aggregates.size() * block_size;

    if ((size_t) B.size() != (size_t) num_rows * K)
    {
        FatalError("Near-nullspace size does not match the matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    // Unknowns sorted by aggregate and aggregate offsets in that order
    IVector keys(num_rows), order(num_rows), agg_offsets(num_aggregates + 1, 0);

    if (num_rows > 0)
    {
        unknownAggregatesKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(aggregates.raw(), num_rows, block_size,
                keys.raw(), order.raw());
        cudaCheckError();
        aggregateUnknownCountKernel <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(aggregates.raw(), num_rows, block_size,
                agg_offsets.raw());
        cudaCheckError();
    }

    thrust_wrapper::stable_sort_by_key<TConfig::memSpace>(keys.begin(), keys.end(), order.begin());
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(agg_offsets.begin(), agg_offsets.end(), agg_offsets.begin());
    cudaCheckError();
    // Each aggregated unknown gets K entries
    IVector row_offsets(num_rows + 1, 0);
    thrust_wrapper::transform<TConfig::memSpace>(amgx::thrust::counting_iterator<int>(0), amgx::thrust::counting_iterator<int>(num_rows),
            row_offsets.begin(), row_entries(aggregates.raw(), block_size, K));
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(row_offsets.begin(), row_offsets.end(), row_offsets.begin());
    cudaCheckError();
    const int nnz = row_offsets[num_rows];
    P_tent.set_initialized(0);
    P_tent.addProps(CSR);
    P_tent.resize(num_rows, num_aggregates * K, nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(row_offsets.begin(), row_offsets.end(), P_tent.row_offsets.begin());
    Bc.resize((size_t) num_aggregates * K * K);

    if (num_aggregates > 0)
    {
        DVector W((size_t) num_rows * K), Q((size_t) num_rows * K), tau((size_t) num_aggregates * K);
        const int warps_per_block = SAP_BLOCK_SIZE / 32;
        const int num_blocks = std::min(4096, (num_aggregates + warps_per_block - 1) / warps_per_block);
        nullspaceQRKernel <<< num_blocks, SAP_BLOCK_SIZE>>>(order.raw(), agg_offsets.raw(), num_aggregates, K,
                B.raw(), W.raw(), Q.raw(), tau.raw(), P_tent.row_offsets.raw(), P_tent.col_indices.raw(),
                P_tent.values.raw(), Bc.raw());
        cudaCheckError();
    }

    P_tent.computeDiagonal();
    P_tent.set_initialized(1);
}

template <class T_Config>
void SAProlongation<T_Config>::blockIdentityNullspace(int num_aggregates, int block_size, DVector &Bc)
{
    const int n = num_aggregates * block_size;
    Bc.resize((size_t) n * block_size);

    if (n > 0)
    {
        blockIdentityKernel <<< sapNumBlocks(n * block_size), SAP_BLOCK_SIZE>>>(n, block_size, Bc.raw());
        cudaCheckError();
    }
}

template <class T_Config>
void SAProlongation<T_Config>::truncateNullspace(Matrix<TConfig> &P, double drop_tol, int K, const DVector &Bc)
{
    if (K < 1 || K > MAX_TRUNCATION_K)
    {
        FatalError("Prolongator truncation supports 1 to 8 near-nullspace vectors", AMGX_ERR_NOT_IMPLEMENTED);
    }

    const int num_rows = P.get_num_rows();

    if (num_rows == 0) { return; }

    IVector counts(num_rows + 1, 0);
    truncateNullspaceKernel<IndexType, ValueTypeA, MAX_TRUNCATION_K> <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(
        P.row_offsets.raw(), P.col_indices.raw(), P.values.raw(), num_rows, K, Bc.raw(), drop_tol, false,
        counts.raw(), NULL, NULL, NULL);
    cudaCheckError();
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(counts.begin(), counts.end(), counts.begin());
    cudaCheckError();
    const int nnz = counts[num_rows];

    if (nnz == P.get_num_nz()) { return; }

    Matrix<TConfig> P_trunc;
    P_trunc.addProps(CSR);
    P_trunc.resize(num_rows, P.get_num_cols(), nnz, 1, 1, 1);
    thrust_wrapper::copy<TConfig::memSpace>(counts.begin(), counts.end(), P_trunc.row_offsets.begin());
    truncateNullspaceKernel<IndexType, ValueTypeA, MAX_TRUNCATION_K> <<< sapNumBlocks(num_rows), SAP_BLOCK_SIZE>>>(
        P.row_offsets.raw(), P.col_indices.raw(), P.values.raw(), num_rows, K, Bc.raw(), drop_tol, true,
        NULL, P_trunc.row_offsets.raw(), P_trunc.col_indices.raw(), P_trunc.values.raw());
    cudaCheckError();
    P_trunc.computeDiagonal();
    P_trunc.set_initialized(1);
    P.set_initialized(0);
    P.copy(P_trunc);
}

#define AMGX_CASE_LINE(CASE) template struct SAProlongation<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace aggregation
} // namespace amgx
