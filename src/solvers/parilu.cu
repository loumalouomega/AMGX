// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Incomplete LU factorizations computed with parallel fixed-point sweeps:
//  - ParILU on a fixed pattern (E. Chow, A. Patel, "Fine-grained parallel incomplete LU
//    factorization", SIAM J. Sci. Comput. 37(2), 2015), used for ILUP(k) on the pattern of A^(k+1)
//    as in AMGCL (amgcl/relaxation/ilup.hpp);
//  - ParILUT (H. Anzt, E. Chow, J. Dongarra, "ParILUT - A new parallel threshold ILU
//    factorization", SIAM J. Sci. Comput. 40(4), 2018) with the dropping rule of AMGCL
//    (amgcl/relaxation/ilut.hpp).
// The Jacobi triangular solve follows AMGCL (amgcl/relaxation/detail/ilu_solve.hpp).
// AMGCL: Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License
// (see LICENSES/MIT.txt).

#include <solvers/parilu.h>
#include <cub/device/device_radix_sort.cuh>
#include <amgx_cusparse.h>
#include <csr_multiply.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>
#include <util.h>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#include <cmath>

namespace amgx
{

namespace
{

const int PILU_BLOCK_SIZE = 256;

inline int piluNumBlocks(int n)
{
    return std::max(1, std::min<int>(4096, (n + PILU_BLOCK_SIZE - 1) / PILU_BLOCK_SIZE));
}

// ---------------- Patterns as sorted keys row * n + col ----------------

template <typename IndexType, typename ValueType>
__global__ void csrToKeysKernel(const IndexType *row_offsets, const IndexType *cols, const ValueType *vals,
                                int num_rows, int64_t n, bool unit_values, int64_t *keys, ValueType *out_vals)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            keys[k] = (int64_t) i * n + cols[k];
            out_vals[k] = unit_values ? ValueType(1) : vals[k];
        }
    }
}

template <typename IndexType>
__global__ void splitKeysKernel(const int64_t *keys, int nnz, int64_t n, IndexType *rows, IndexType *cols)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        rows[k] = (IndexType)(keys[k] / n);
        cols[k] = (IndexType)(keys[k] % n);
    }
}

__global__ void diagonalKeysKernel(int n, int64_t *keys)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        keys[i] = (int64_t) i * n + i;
    }
}

template <typename ValueType>
__device__ __forceinline__ int keyIndex(const int64_t *keys, int nnz, int64_t key)
{
    int lo = 0, hi = nnz;

    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;

        if (keys[mid] < key) { lo = mid + 1; }
        else { hi = mid; }
    }

    return (lo < nnz && keys[lo] == key) ? lo : -1;
}

template <typename ValueType>
__device__ __forceinline__ ValueType keyValue(const int64_t *keys, const ValueType *vals, int nnz, int64_t key)
{
    const int k = keyIndex<ValueType>(keys, nnz, key);
    return k >= 0 ? vals[k] : ValueType(0);
}

// ---------------- Factor initialization and sweeps ----------------

// Per row: position of the diagonal in the factor pattern
template <typename ValueType>
__global__ void diagonalPositionKernel(const int64_t *keys, int nnz, int n, int *diag_pos)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        diag_pos[i] = keyIndex<ValueType>(keys, nnz, (int64_t) i * n + i);
    }
}

// F = A on the pattern: l_ij = a_ij / a_jj, u_ij = a_ij
template <typename ValueType>
__global__ void initFactorKernel(const int64_t *keys, int nnz, int64_t n, const int64_t *A_keys,
                                 const ValueType *A_vals, int A_nnz, const ValueType *A_diag, ValueType *F)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        const int64_t i = keys[k] / n, j = keys[k] % n;
        const ValueType a = keyValue(A_keys, A_vals, A_nnz, keys[k]);
        F[k] = (j < i) ? (A_diag[j] != ValueType(0) ? a / A_diag[j] : ValueType(0)) : a;
    }
}

// Jacobi-style ParILU sweep: l_ij += (a_ij - (LU)_ij) / u_jj, u_ij += a_ij - (LU)_ij
template <typename ValueType>
__global__ void parILUSweepKernel(const int64_t *keys, const ValueType *F_old, int nnz, int64_t n,
                                  const int64_t *A_keys, const ValueType *A_vals, int A_nnz,
                                  const int64_t *T_keys, const ValueType *T_vals, int T_nnz,
                                  const int *diag_pos, ValueType *F_new)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        const int64_t i = keys[k] / n, j = keys[k] % n;
        const ValueType r = keyValue(A_keys, A_vals, A_nnz, keys[k]) - keyValue(T_keys, T_vals, T_nnz, keys[k]);

        if (j < i)
        {
            const ValueType u = F_old[diag_pos[j]];
            F_new[k] = F_old[k] + (u != ValueType(0) ? r / u : ValueType(0));
        }
        else
        {
            F_new[k] = F_old[k] + r;
        }
    }
}

// r_k = a_k - T_k on the pattern
template <typename ValueType>
__global__ void patternResidualKernel(const int64_t *keys, int nnz, const int64_t *A_keys, const ValueType *A_vals, int A_nnz,
                                      const int64_t *T_keys, const ValueType *T_vals, int T_nnz, ValueType *R)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        R[k] = keyValue(A_keys, A_vals, A_nnz, keys[k]) - keyValue(T_keys, T_vals, T_nnz, keys[k]);
    }
}

// Tiny pivots are replaced by +-1e-12 ||a_i||
template <typename ValueType>
__global__ void pivotGuardKernel(int n, const int *diag_pos, const double *row_norms, ValueType *F)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const double tiny = 1e-12 * (row_norms[i] > 0.0 ? row_norms[i] : 1.0);
        const ValueType u = F[diag_pos[i]];

        if (fabs((double) u) < tiny) { F[diag_pos[i]] = ValueType(u < ValueType(0) ? -tiny : tiny); }
    }
}

template <typename IndexType, typename ValueType>
__global__ void rowStatsKernel(const IndexType *row_offsets, const IndexType *cols, const ValueType *vals, int n,
                               double *row_norms, double *row_abs_sums, ValueType *diag, int *len_lower, int *len_upper)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        double s = 0.0, abs_sum = 0.0;
        ValueType d = ValueType(0);
        int lower = 0, upper = 0;

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            s += (double) vals[k] * (double) vals[k];
            abs_sum += fabs((double) vals[k]);

            if (cols[k] == i) { d += vals[k]; }
            else if (cols[k] < i) { lower++; }
            else { upper++; }
        }

        row_norms[i] = sqrt(s);
        row_abs_sums[i] = abs_sum;
        diag[i] = d;
        len_lower[i] = lower;
        len_upper[i] = upper;
    }
}

// Old factor values on a larger pattern (0 for the new entries)
template <typename ValueType>
__global__ void gatherFactorKernel(const int64_t *keys, int nnz, const int64_t *old_keys, const ValueType *old_vals,
                                   int old_nnz, ValueType *vals)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        vals[k] = keyValue(old_keys, old_vals, old_nnz, keys[k]);
    }
}

// ---------------- ParILUT selection ----------------

// AMGCL tolerance: tau * sum_j |a_ij| / (number of off-diagonal entries)
__global__ void dropToleranceKernel(int n, double drop_tol, const double *row_abs_sums, const int *len_lower,
                                    const int *len_upper, double *thresholds)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int off = len_lower[i] + len_upper[i];
        thresholds[i] = drop_tol * row_abs_sums[i] / (off > 0 ? off : 1);
    }
}

// Segment of each entry: 2 i (lower) or 2 i + 1 (upper); diagonals get their own segment past 2 n
template <typename ValueType>
__global__ void selectionKeysKernel(const int64_t *keys, const ValueType *vals, int nnz, int64_t n,
                                    int64_t *segment, double *neg_abs)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        const int64_t i = keys[k] / n, j = keys[k] % n;
        segment[k] = (j == i) ? 2 * n + i : 2 * i + (j > i ? 1 : 0);
        neg_abs[k] = -fabs((double) vals[k]);
    }
}

// Keep the diagonal and the largest entries above the threshold, at most `limit` per row part
__global__ void selectionKeepKernel(const int64_t *segment_sorted, const double *neg_abs_sorted, const int *rank,
                                    const int *order, int nnz, int64_t n, const double *thresholds,
                                    const int *len_lower, const int *len_upper, double fill_factor, int *keep)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        const int64_t seg = segment_sorted[k];
        bool kept;

        if (seg >= 2 * n)
        {
            kept = true;
        }
        else
        {
            const int64_t i = seg / 2;
            const int len = (seg % 2 == 0) ? len_lower[i] : len_upper[i];
            const int limit = (int)(len * fill_factor);
            kept = rank[k] < limit && -neg_abs_sorted[k] > thresholds[i];
        }

        keep[order[k]] = kept ? 1 : 0;
    }
}

// ---------------- Triangular solves ----------------

template <typename ValueTypeB>
__global__ void scaleKernel(int n, ValueTypeB d, const ValueTypeB *in, ValueTypeB *out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { out[i] = d * in[i]; }
}

template <typename ValueTypeA, typename ValueTypeB>
__global__ void scaleDiagKernel(int n, ValueTypeB d, const ValueTypeA *dinv, const ValueTypeB *in, ValueTypeB *out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { out[i] = d * ValueTypeB(dinv[i]) * in[i]; }
}

// y_new = d (rhs - L y_old) + (1 - d) y_old, L strictly lower
template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void lowerJacobiKernel(int n, const IndexType *row_offsets, const IndexType *cols, const ValueTypeA *vals,
                                  const ValueTypeB *rhs, const ValueTypeB *y_old, ValueTypeB d, ValueTypeB *y_new)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        ValueTypeB s = rhs[i];

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++) { s -= ValueTypeB(vals[k]) * y_old[cols[k]]; }

        y_new[i] = d * s + (ValueTypeB(1) - d) * y_old[i];
    }
}

// x_new = d D^{-1} (rhs - U_strict x_old) + (1 - d) x_old
template <typename IndexType, typename ValueTypeA, typename ValueTypeB>
__global__ void upperJacobiKernel(int n, const IndexType *row_offsets, const IndexType *cols, const ValueTypeA *vals,
                                  const ValueTypeA *dinv, const ValueTypeB *rhs, const ValueTypeB *x_old, ValueTypeB d,
                                  ValueTypeB *x_new)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        ValueTypeB s = rhs[i];

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
            if (cols[k] != i) { s -= ValueTypeB(vals[k]) * x_old[cols[k]]; }

        x_new[i] = d * ValueTypeB(dinv[i]) * s + (ValueTypeB(1) - d) * x_old[i];
    }
}

template <typename IndexType, typename ValueTypeA>
__global__ void inverseDiagonalKernel(int n, const IndexType *row_offsets, const IndexType *cols, const ValueTypeA *vals,
                                      ValueTypeA *dinv)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        ValueTypeA d = ValueTypeA(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
            if (cols[k] == i) { d = vals[k]; }

        dinv[i] = d != ValueTypeA(0) ? ValueTypeA(1) / d : ValueTypeA(0);
    }
}

template <typename In, typename Out>
__global__ void convertKernel(int n, const In *in, Out *out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { out[i] = Out(in[i]); }
}

struct is_strictly_lower
{
    int64_t n;
    is_strictly_lower(int64_t n) : n(n) {}
    __host__ __device__ bool operator()(int64_t key) const { return key % n < key / n; }
};

struct is_upper
{
    int64_t n;
    is_upper(int64_t n) : n(n) {}
    __host__ __device__ bool operator()(int64_t key) const { return key % n >= key / n; }
};

struct is_kept
{
    __host__ __device__ bool operator()(int k) const { return k != 0; }
};

template <typename ValueType>
struct square_of
{
    __host__ __device__ double operator()(ValueType v) const { return (double) v * (double) v; }
};

// ---------------- Persistent factor state (ParILUFactor) ----------------

// keys i * n + j of A with the position of each entry
template <typename IndexType>
__global__ void csrKeysPositionsKernel(const IndexType *row_offsets, const IndexType *cols, int n, int64_t m,
                                       int64_t *keys, int *pos)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            keys[k] = (int64_t) i * m + cols[k];
            pos[k] = k;
        }
    }
}

// position of key in A (through the sorted keys), -1 if absent
__device__ __forceinline__ int aPosition(const int64_t *A_keys, const int *A_pos, int A_nnz, int64_t key)
{
    int lo = 0, hi = A_nnz;

    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;

        if (A_keys[mid] < key) { lo = mid + 1; }
        else { hi = mid; }
    }

    return (lo < A_nnz && A_keys[lo] == key) ? A_pos[lo] : -1;
}

// position of column j in the sorted columns [lo, hi), -1 if absent
__device__ __forceinline__ int findInRow(const int *ci, int lo, int hi, int j)
{
    const int end = hi;

    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;

        if (ci[mid] < j) { lo = mid + 1; }
        else { hi = mid; }
    }

    return (lo < end && ci[lo] == j) ? lo : -1;
}

// sum_{k < min(i, j), (i,k), (k,j) in S} F(i,k) F(k,j), k ascending
template <typename ValueType>
__device__ __forceinline__ ValueType lowerUpperDot(int i, int j, const int *ro, const int *ci, const ValueType *F)
{
    ValueType s = ValueType(0);
    const int lim = i < j ? i : j;

    for (int q = ro[i]; q < ro[i + 1] && ci[q] < lim; q++)
    {
        const int k = ci[q];
        const int p = findInRow(ci, ro[k], ro[k + 1], j);

        if (p >= 0) { s = s + F[q] * F[p]; }
    }

    return s;
}

template <typename ValueType>
__device__ __forceinline__ ValueType guardPivot(ValueType u, double row_norm)
{
    const double tiny = 1e-12 * (row_norm > 0.0 ? row_norm : 1.0);
    return fabs((double) u) < tiny ? ValueType(u < ValueType(0) ? -tiny : tiny) : u;
}

__global__ void patternRowsKernel(const int64_t *keys, int nnz, int64_t n, int *rows, int *cols)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < nnz; e += gridDim.x * blockDim.x)
    {
        rows[e] = (int)(keys[e] / n);
        cols[e] = (int)(keys[e] % n);
    }
}

__global__ void diagonalOfRowsKernel(const int *ro, const int *ci, int n, int *diag)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        diag[i] = findInRow(ci, ro[i], ro[i + 1], i);
    }
}

__global__ void aPositionsKernel(const int *rows, const int *cols, int nnz, int64_t n, const int64_t *A_keys,
                                 const int *A_pos, int A_nnz, int *apos)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < nnz; e += gridDim.x * blockDim.x)
    {
        apos[e] = aPosition(A_keys, A_pos, A_nnz, (int64_t) rows[e] * n + cols[e]);
    }
}

template <typename ValueType>
__global__ void gatherAValuesKernel(const int *apos, int nnz, const ValueType *A_vals, ValueType *aval)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < nnz; e += gridDim.x * blockDim.x)
    {
        aval[e] = apos[e] >= 0 ? A_vals[apos[e]] : ValueType(0);
    }
}

// F = A on the pattern: l_ij = a_ij / a_jj, u_ij = a_ij
template <typename ValueType>
__global__ void initPatternKernel(const int *rows, const int *cols, const ValueType *aval, int nnz,
                                  const ValueType *A_diag, ValueType *F)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < nnz; e += gridDim.x * blockDim.x)
    {
        const int i = rows[e], j = cols[e];
        const ValueType a = aval[e];
        F[e] = (j < i) ? (A_diag[j] != ValueType(0) ? a / A_diag[j] : ValueType(0)) : a;
    }
}

// One Jacobi-style ParILU sweep on S, pivot guard fused in
template <typename ValueType>
__global__ void patternSweepKernel(const int *rows, const int *ro, const int *ci, const int *diag, const ValueType *aval,
                                   int nnz, const double *row_norms, const ValueType *F_old, ValueType *F_new)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < nnz; e += gridDim.x * blockDim.x)
    {
        const int i = rows[e], j = ci[e];
        const ValueType s = aval[e] - lowerUpperDot(i, j, ro, ci, F_old);

        if (j < i)
        {
            const ValueType u = F_old[diag[j]];
            F_new[e] = u != ValueType(0) ? s / u : F_old[e];
        }
        else
        {
            F_new[e] = (j == i) ? guardPivot(s, row_norms[i]) : s;
        }
    }
}

// L (strictly lower) and U (upper with diagonal) from S, rows split at the diagonal
template <typename ValueType>
__global__ void splitRowsKernel(const int *ro, const int *ci, const int *diag, const ValueType *F, int n,
                                const int *L_ro, const int *U_ro, int *L_ci, ValueType *L_v, int *U_ci, ValueType *U_v)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int l = L_ro[i], u = U_ro[i];

        for (int q = ro[i]; q < ro[i + 1]; q++)
        {
            if (q < diag[i]) { L_ci[l] = ci[q]; L_v[l] = F[q]; l++; }
            else { U_ci[u] = ci[q]; U_v[u] = F[q]; u++; }
        }
    }
}

__global__ void splitCountsKernel(const int *ro, const int *diag, int n, int *L_count, int *U_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        L_count[i] = diag[i] - ro[i];
        U_count[i] = ro[i + 1] - diag[i];
    }
}

// ParILUT candidates of row i: A_i, S_i and the upper parts U_k of the rows k in L_i
template <typename IndexType>
__global__ void candidateCountKernel(const IndexType *A_ro, const int *ro, const int *ci, const int *diag, int n,
                                     int *counts)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int c = (A_ro[i + 1] - A_ro[i]) + (ro[i + 1] - ro[i]);

        for (int q = ro[i]; q < diag[i]; q++)
        {
            const int k = ci[q];
            c += ro[k + 1] - diag[k];
        }

        counts[i] = c;
    }
}

template <typename IndexType>
__global__ void candidateFillKernel(const IndexType *A_ro, const IndexType *A_ci, const int *ro, const int *ci,
                                    const int *diag, int n, const int *offsets, int64_t *keys)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int w = offsets[i];
        const int64_t base = (int64_t) i * n;

        for (IndexType q = A_ro[i]; q < A_ro[i + 1]; q++) { keys[w++] = base + A_ci[q]; }

        for (int q = ro[i]; q < ro[i + 1]; q++) { keys[w++] = base + ci[q]; }

        for (int q = ro[i]; q < diag[i]; q++)
        {
            const int k = ci[q];

            for (int p = diag[k]; p < ro[k + 1]; p++) { keys[w++] = base + ci[p]; }
        }
    }
}

// Candidate values: one Jacobi update from the current factors on S
template <typename ValueType>
__global__ void candidateValuesKernel(const int *c_rows, const int *c_cols, int m, int64_t n, const int64_t *A_keys,
                                      const int *A_pos, int A_nnz, const ValueType *A_vals, const int *ro, const int *ci,
                                      const int *diag, const ValueType *F, const double *row_norms, ValueType *vals)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < m; c += gridDim.x * blockDim.x)
    {
        const int i = c_rows[c], j = c_cols[c];
        const int ap = aPosition(A_keys, A_pos, A_nnz, (int64_t) i * n + j);
        const ValueType a = ap >= 0 ? A_vals[ap] : ValueType(0);
        const ValueType s = a - lowerUpperDot(i, j, ro, ci, F);

        if (j < i)
        {
            const ValueType u = F[diag[j]];
            const int old = findInRow(ci, ro[i], ro[i + 1], j);
            vals[c] = u != ValueType(0) ? s / u : (old >= 0 ? F[old] : ValueType(0));
        }
        else
        {
            vals[c] = (j == i) ? guardPivot(s, row_norms[i]) : s;
        }
    }
}

// Keep the diagonal and the largest entries above the threshold, at most fill_factor * nnz(part of a_i)
// per row part; rank by magnitude, the column breaks ties
template <typename ValueType>
__global__ void candidateSelectKernel(const int *c_ro, const int *c_rows, const int *c_cols, const ValueType *vals, int m,
                                      const double *thresholds, const int *len_lower, const int *len_upper,
                                      double fill_factor, int *keep)
{
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < m; c += gridDim.x * blockDim.x)
    {
        const int i = c_rows[c], j = c_cols[c];

        if (j == i) { keep[c] = 1; continue; }

        const bool lower = j < i;
        const double v = fabs((double) vals[c]);
        const int limit = (int)((lower ? len_lower[i] : len_upper[i]) * fill_factor);
        int rank = 0;

        for (int o = c_ro[i]; o < c_ro[i + 1] && rank < limit; o++)
        {
            const int jo = c_cols[o];

            if (o == c || jo == i || (jo < i) != lower) { continue; }

            const double vo = fabs((double) vals[o]);

            if (vo > v || (vo == v && jo < j)) { rank++; }
        }

        keep[c] = (rank < limit && v > thresholds[i]) ? 1 : 0;
    }
}

__global__ void keepCountKernel(const int *c_ro, const int *keep, int n, int *counts)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int c = 0;

        for (int o = c_ro[i]; o < c_ro[i + 1]; o++) { c += keep[o]; }

        counts[i] = c;
    }
}

template <typename ValueType>
__global__ void compactKernel(const int *c_ro, const int *c_cols, const ValueType *vals, const int *keep, int n,
                              const int *ro, int *rows, int *ci, ValueType *F)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int w = ro[i];

        for (int o = c_ro[i]; o < c_ro[i + 1]; o++)
        {
            if (keep[o]) { rows[w] = i; ci[w] = c_cols[o]; F[w] = vals[o]; w++; }
        }
    }
}

// ---------------- Host-side helpers ----------------

template <class TConfig>
struct PatternOps
{
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueType;
    DEFINE_VECTOR_TYPES

    // keys/values of a scalar CSR matrix, sorted
    static void toKeys(const Matrix<TConfig> &M, I64Vector &keys, MVector &vals, bool unit_values = false)
    {
        const int nnz = M.get_num_nz();
        keys.resize(nnz);
        vals.resize(nnz);

        if (M.get_num_rows() > 0)
        {
            csrToKeysKernel <<< piluNumBlocks(M.get_num_rows()), PILU_BLOCK_SIZE>>>(M.row_offsets.raw(), M.col_indices.raw(),
                    M.values.raw(), M.get_num_rows(), (int64_t) M.get_num_cols(), unit_values, keys.raw(), vals.raw());
            cudaCheckError();
        }

        thrust_wrapper::sort_by_key<TConfig::memSpace>(keys.begin(), keys.end(), vals.begin());
        cudaCheckError();
    }

    // Scalar n x n CSR matrix from sorted keys
    static void fromKeys(const I64Vector &keys, const MVector &vals, int n, Matrix<TConfig> &M)
    {
        const int nnz = keys.size();
        M.set_initialized(0);
        M.addProps(CSR);
        M.resize(n, n, nnz, 1, 1, 1);
        IVector rows(nnz);

        if (nnz > 0)
        {
            splitKeysKernel <<< piluNumBlocks(nnz), PILU_BLOCK_SIZE>>>(keys.raw(), nnz, (int64_t) n, rows.raw(), M.col_indices.raw());
            cudaCheckError();
            thrust_wrapper::copy<TConfig::memSpace>(vals.begin(), vals.end(), M.values.begin());
        }

        amgx::thrust::lower_bound(rows.begin(), rows.end(), amgx::thrust::counting_iterator<int>(0),
                                  amgx::thrust::counting_iterator<int>(n + 1), M.row_offsets.begin());
        cudaCheckError();
        M.computeDiagonal();
        M.set_initialized(1);
    }

    // Entries of (keys, vals) selected by pred
    template <class Pred>
    static void filter(const I64Vector &keys, const MVector &vals, Pred pred, I64Vector &out_keys, MVector &out_vals)
    {
        const int count = (int) thrust_wrapper::count_if<TConfig::memSpace>(keys.begin(), keys.end(), pred);
        out_keys.resize(count);
        out_vals.resize(count);
        amgx::thrust::copy_if(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(keys.begin(), vals.begin())),
                              amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(keys.end(), vals.end())),
                              keys.begin(),
                              amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(out_keys.begin(), out_vals.begin())),
                              pred);
        cudaCheckError();
    }

    // Sorted union of two sorted key sets
    static void merge(const I64Vector &a, const I64Vector &b, I64Vector &out)
    {
        I64Vector merged(a.size() + b.size());
        amgx::thrust::merge(a.begin(), a.end(), b.begin(), b.end(), merged.begin());
        const int count = amgx::thrust::unique(merged.begin(), merged.end()) - merged.begin();
        out.resize(count);
        thrust_wrapper::copy<TConfig::memSpace>(merged.begin(), merged.begin() + count, out.begin());
        cudaCheckError();
    }

    // L (strictly lower), U (upper with diagonal) and Lhat = L + I as matrices
    static void split(const I64Vector &keys, const MVector &vals, int n, Matrix<TConfig> &L, Matrix<TConfig> &U,
                      Matrix<TConfig> *Lhat)
    {
        I64Vector k_part;
        MVector v_part;
        filter(keys, vals, is_strictly_lower(n), k_part, v_part);
        fromKeys(k_part, v_part, n, L);

        if (Lhat != NULL)
        {
            I64Vector diag_keys(n), lhat_keys;
            MVector lhat_vals;
            diagonalKeysKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(n, diag_keys.raw());
            cudaCheckError();
            merge(k_part, diag_keys, lhat_keys);
            lhat_vals.resize(lhat_keys.size());
            // diagonal entries become 1, strictly lower ones keep their value
            gatherFactorKernel <<< piluNumBlocks(std::max<int>(1, lhat_keys.size())), PILU_BLOCK_SIZE>>>(lhat_keys.raw(),
                    (int) lhat_keys.size(), k_part.raw(), v_part.raw(), (int) k_part.size(), lhat_vals.raw());
            cudaCheckError();
            amgx::thrust::transform_if(lhat_vals.begin(), lhat_vals.end(), lhat_keys.begin(), lhat_vals.begin(),
                                       amgx::thrust::placeholders::_1 * ValueType(0) + ValueType(1), is_upper(n));
            cudaCheckError();
            fromKeys(lhat_keys, lhat_vals, n, *Lhat);
        }

        filter(keys, vals, is_upper(n), k_part, v_part);
        fromKeys(k_part, v_part, n, U);
    }
};

} // anonymous namespace

// ---------------- ParILU ----------------

namespace
{

// State shared by the fixed-pattern and threshold factorizations
template <class TConfig>
struct FactorState
{
    typedef PatternOps<TConfig> Ops;
    typedef typename Ops::I64Vector I64Vector;
    typedef typename Ops::MVector MVector;
    typedef typename Ops::IVector IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecDouble>::Type> DVector;

    int n;
    I64Vector A_keys;
    MVector A_vals, A_diag;
    DVector row_norms, row_abs_sums;
    IVector len_lower, len_upper;
    void *wk;

    FactorState(const Matrix<TConfig> &A) : n(A.get_num_rows())
    {
        if (A.get_block_size() != 1 || A.hasProps(DIAG))
        {
            FatalError("ParILU requires a scalar matrix without external diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        }

        Ops::toKeys(A, A_keys, A_vals);
        A_diag.resize(n);
        row_norms.resize(n);
        row_abs_sums.resize(n);
        len_lower.resize(n);
        len_upper.resize(n);

        if (n > 0)
        {
            rowStatsKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), n,
                    row_norms.raw(), row_abs_sums.raw(), A_diag.raw(), len_lower.raw(), len_upper.raw());
            cudaCheckError();
        }

        wk = CSR_Multiply<TConfig>::csr_workspace_create();
    }

    ~FactorState() { CSR_Multiply<TConfig>::csr_workspace_delete(wk); }

    // Factor values on `keys`, initialized from A
    void init(const I64Vector &keys, MVector &F)
    {
        F.resize(keys.size());

        if (keys.size() > 0)
        {
            initFactorKernel <<< piluNumBlocks(keys.size()), PILU_BLOCK_SIZE>>>(keys.raw(), (int) keys.size(), (int64_t) n,
                    A_keys.raw(), A_vals.raw(), (int) A_keys.size(), A_diag.raw(), F.raw());
            cudaCheckError();
        }
    }

    // T = (L + I) U for the factors stored on `keys`
    void product(const I64Vector &keys, const MVector &F, I64Vector &T_keys, MVector &T_vals)
    {
        Matrix<TConfig> L, U, Lhat, T;
        Ops::split(keys, F, n, L, U, &Lhat);
        T.set_initialized(0);
        T.addProps(CSR);
        CSR_Multiply<TConfig>::csr_multiply(Lhat, U, T, wk);
        Ops::toKeys(T, T_keys, T_vals);
    }

    // One Jacobi-style ParILU sweep of the factors on `keys` (which must contain the diagonal)
    void sweep(const I64Vector &keys, MVector &F, const I64Vector &T_keys, const MVector &T_vals)
    {
        const int nnz = keys.size();
        IVector diag_pos(n);
        MVector F_new(nnz);
        diagonalPositionKernel<typename Ops::ValueType> <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(keys.raw(), nnz, n, diag_pos.raw());
        cudaCheckError();
        parILUSweepKernel <<< piluNumBlocks(nnz), PILU_BLOCK_SIZE>>>(keys.raw(), F.raw(), nnz, (int64_t) n,
                A_keys.raw(), A_vals.raw(), (int) A_keys.size(), T_keys.raw(), T_vals.raw(), (int) T_keys.size(),
                diag_pos.raw(), F_new.raw());
        cudaCheckError();
        pivotGuardKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(n, diag_pos.raw(), row_norms.raw(), F_new.raw());
        cudaCheckError();
        thrust_wrapper::copy<TConfig::memSpace>(F_new.begin(), F_new.end(), F.begin());
    }

    void sweep(const I64Vector &keys, MVector &F)
    {
        I64Vector T_keys;
        MVector T_vals;
        product(keys, F, T_keys, T_vals);
        sweep(keys, F, T_keys, T_vals);
    }

    void withDiagonal(const I64Vector &keys, I64Vector &out)
    {
        I64Vector diag_keys(n);
        diagonalKeysKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(n, diag_keys.raw());
        cudaCheckError();
        Ops::merge(keys, diag_keys, out);
    }
};

} // anonymous namespace

// ---------------- ParILUFactor ----------------

namespace
{

template <class Vec>
void ensureSize(Vec &v, size_t size)
{
    if (v.size() < size) { v.resize(size); }
}

} // anonymous namespace

template <class T_Config>
void ParILUFactor<T_Config>::prepareA(const Matrix<TConfig> &A)
{
    if (A.get_block_size() != 1 || A.hasProps(DIAG))
    {
        FatalError("ParILU requires a scalar matrix without external diagonal", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    m_n = A.get_num_rows();
    const int nnz = A.get_num_nz();
    m_A_keys.resize(nnz);
    m_A_pos.resize(nnz);
    m_A_diag.resize(m_n);
    m_row_norms.resize(m_n);
    m_row_abs_sums.resize(m_n);
    m_len_lower.resize(m_n);
    m_len_upper.resize(m_n);

    if (m_n > 0)
    {
        csrKeysPositionsKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), m_n,
                (int64_t) m_n, m_A_keys.raw(), m_A_pos.raw());
        cudaCheckError();
        rowStatsKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), m_n,
                m_row_norms.raw(), m_row_abs_sums.raw(), m_A_diag.raw(), m_len_lower.raw(), m_len_upper.raw());
        cudaCheckError();
    }

    thrust_wrapper::sort_by_key<TConfig::memSpace>(m_A_keys.begin(), m_A_keys.end(), m_A_pos.begin());
    cudaCheckError();
    m_A_ro.resize(m_n + 1);
    m_A_ci.resize(nnz);
    thrust_wrapper::copy<TConfig::memSpace>(A.row_offsets.begin(), A.row_offsets.begin() + m_n + 1, m_A_ro.begin());
    thrust_wrapper::copy<TConfig::memSpace>(A.col_indices.begin(), A.col_indices.begin() + nnz, m_A_ci.begin());
    cudaCheckError();
}

// S from sorted keys i * n + j: rows, columns, row offsets, diagonal and A positions
template <class T_Config>
void ParILUFactor<T_Config>::patternFromKeys(const I64Vector &keys, int nnz)
{
    m_nnz = nnz;
    m_row.resize(nnz);
    m_ci.resize(nnz);
    m_apos.resize(nnz);
    m_ro.resize(m_n + 1);
    m_diag.resize(m_n);

    if (nnz > 0)
    {
        patternRowsKernel <<< piluNumBlocks(nnz), PILU_BLOCK_SIZE>>>(keys.raw(), nnz, (int64_t) m_n, m_row.raw(), m_ci.raw());
        cudaCheckError();
    }

    amgx::thrust::lower_bound(m_row.begin(), m_row.begin() + nnz, amgx::thrust::counting_iterator<int>(0),
                              amgx::thrust::counting_iterator<int>(m_n + 1), m_ro.begin());
    cudaCheckError();

    if (m_n > 0)
    {
        diagonalOfRowsKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(m_ro.raw(), m_ci.raw(), m_n, m_diag.raw());
        cudaCheckError();
    }

    if (nnz > 0)
    {
        aPositionsKernel <<< piluNumBlocks(nnz), PILU_BLOCK_SIZE>>>(m_row.raw(), m_ci.raw(), nnz, (int64_t) m_n,
                m_A_keys.raw(), m_A_pos.raw(), (int) m_A_keys.size(), m_apos.raw());
        cudaCheckError();
    }
}

template <class T_Config>
void ParILUFactor<T_Config>::gatherAValues(const Matrix<TConfig> &A)
{
    m_aval.resize(m_nnz);

    if (m_nnz > 0)
    {
        gatherAValuesKernel <<< piluNumBlocks(m_nnz), PILU_BLOCK_SIZE>>>(m_apos.raw(), m_nnz, A.values.raw(), m_aval.raw());
        cudaCheckError();
    }
}

template <class T_Config>
void ParILUFactor<T_Config>::initValues()
{
    m_F.resize(m_nnz);
    m_F_old.resize(m_nnz);

    if (m_nnz > 0)
    {
        initPatternKernel <<< piluNumBlocks(m_nnz), PILU_BLOCK_SIZE>>>(m_row.raw(), m_ci.raw(), m_aval.raw(), m_nnz,
                m_A_diag.raw(), m_F.raw());
        cudaCheckError();
    }
}

template <class T_Config>
void ParILUFactor<T_Config>::sweep()
{
    m_F_old.resize(m_nnz);
    m_F.swap(m_F_old);

    if (m_nnz > 0)
    {
        patternSweepKernel <<< piluNumBlocks(m_nnz), PILU_BLOCK_SIZE>>>(m_row.raw(), m_ro.raw(), m_ci.raw(), m_diag.raw(),
                m_aval.raw(), m_nnz, m_row_norms.raw(), m_F_old.raw(), m_F.raw());
        cudaCheckError();
    }
}

// L and U as sorted CSR matrices; their storage is kept when the sizes do not change, so the
// triangular solve can refresh its cuSPARSE analysis in place
template <class T_Config>
void ParILUFactor<T_Config>::splitLU(Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    IVector L_ro(m_n + 1), U_ro(m_n + 1);

    if (m_n > 0)
    {
        splitCountsKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(m_ro.raw(), m_diag.raw(), m_n, L_ro.raw(), U_ro.raw());
        cudaCheckError();
    }

    thrust_wrapper::exclusive_scan<TConfig::memSpace>(L_ro.begin(), L_ro.end(), L_ro.begin());
    thrust_wrapper::exclusive_scan<TConfig::memSpace>(U_ro.begin(), U_ro.end(), U_ro.begin());
    cudaCheckError();
    const int nnz_L = L_ro[m_n], nnz_U = U_ro[m_n];
    Matrix<TConfig> *factors[2] = {&L, &U};
    const int nnz[2] = {nnz_L, nnz_U};

    for (int f = 0; f < 2; f++)
    {
        Matrix<TConfig> &M = *factors[f];
        const bool same = M.get_num_rows() == m_n && M.get_num_nz() == nnz[f] && (int) M.row_offsets.size() == m_n + 1;
        M.set_initialized(0);
        M.addProps(CSR);

        if (!same) { M.resize(m_n, m_n, nnz[f], 1, 1, 1); }

        thrust_wrapper::copy<TConfig::memSpace>((f == 0 ? L_ro : U_ro).begin(), (f == 0 ? L_ro : U_ro).end(), M.row_offsets.begin());
        cudaCheckError();
    }

    if (m_n > 0)
    {
        splitRowsKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(m_ro.raw(), m_ci.raw(), m_diag.raw(), m_F.raw(), m_n,
                L.row_offsets.raw(), U.row_offsets.raw(), L.col_indices.raw(), L.values.raw(), U.col_indices.raw(),
                U.values.raw());
        cudaCheckError();
    }

    for (int f = 0; f < 2; f++)
    {
        factors[f]->computeDiagonal();
        factors[f]->set_initialized(1);
    }
}

template <class T_Config>
void ParILUFactor<T_Config>::factorPower(const Matrix<TConfig> &A, int level, int sweeps,
        Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    typedef PatternOps<TConfig> Ops;
    prepareA(A);
    // Pattern of A^(level+1) with the diagonal, built once
    I64Vector keys, diag_keys(m_n), S;
    powerKeys(A, level, keys);

    if (m_n > 0)
    {
        diagonalKeysKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(m_n, diag_keys.raw());
        cudaCheckError();
    }

    Ops::merge(keys, diag_keys, S);
    patternFromKeys(S, (int) S.size());
    gatherAValues(A);
    initValues();

    for (int s = 0; s < sweeps; s++) { sweep(); }

    splitLU(L, U);
    m_power_valid = true;
}

template <class T_Config>
void ParILUFactor<T_Config>::powerKeys(const Matrix<TConfig> &A, int level, I64Vector &keys)
{
    typedef PatternOps<TConfig> Ops;
    MVector unit_vals;
    Ops::toKeys(A, keys, unit_vals, true);

    if (level > 0)
    {
        void *wk = CSR_Multiply<TConfig>::csr_workspace_create();
        Matrix<TConfig> pattern, power;
        Ops::fromKeys(keys, unit_vals, m_n, pattern);

        for (int l = 0; l < level; l++)
        {
            Matrix<TConfig> next;
            next.set_initialized(0);
            next.addProps(CSR);
            CSR_Multiply<TConfig>::csr_multiply(l == 0 ? pattern : power, pattern, next, wk);
            power.set_initialized(0);
            power.copy(next);
            power.set_initialized(1);
        }

        CSR_Multiply<TConfig>::csr_workspace_delete(wk);
        Ops::toKeys(power, keys, unit_vals);
    }
}

template <class T_Config>
bool ParILUFactor<T_Config>::samePattern(const Matrix<TConfig> &A) const
{
    if (!m_power_valid || A.get_num_rows() != m_n || A.get_num_nz() != (int) m_A_ci.size() ||
            A.get_block_size() != 1 || A.hasProps(DIAG))
    {
        return false;
    }

    return amgx::thrust::equal(A.row_offsets.begin(), A.row_offsets.begin() + m_n + 1, m_A_ro.begin()) &&
           amgx::thrust::equal(A.col_indices.begin(), A.col_indices.begin() + A.get_num_nz(), m_A_ci.begin());
}

template <class T_Config>
void ParILUFactor<T_Config>::refactorPower(const Matrix<TConfig> &A, int sweeps, Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    if (m_n > 0)
    {
        rowStatsKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), m_n,
                m_row_norms.raw(), m_row_abs_sums.raw(), m_A_diag.raw(), m_len_lower.raw(), m_len_upper.raw());
        cudaCheckError();
    }

    gatherAValues(A);
    initValues();

    for (int s = 0; s < sweeps; s++) { sweep(); }

    splitLU(L, U);
}

template <class T_Config>
void ParILUFactor<T_Config>::factorThreshold(const Matrix<TConfig> &A, double drop_tol, double fill_factor, int sweeps,
        Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    typedef PatternOps<TConfig> Ops;
    prepareA(A);
    m_power_valid = false;
    const int n = m_n;
    m_thresholds.resize(n);

    if (n > 0)
    {
        dropToleranceKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(n, drop_tol, m_row_abs_sums.raw(), m_len_lower.raw(),
                m_len_upper.raw(), m_thresholds.raw());
        cudaCheckError();
    }

    // Start from ILU(0) values on the pattern of A (or of A^(k+1), ilut_initial_fill_level) with the
    // diagonal; a richer start needs fewer selection steps to reach the fill of deep rows
    I64Vector diag_keys(n), S, start_keys;

    if (n > 0)
    {
        diagonalKeysKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(n, diag_keys.raw());
        cudaCheckError();
    }

    if (m_initial_level > 0) { powerKeys(A, m_initial_level, start_keys); }

    Ops::merge(m_initial_level > 0 ? start_keys : m_A_keys, diag_keys, S);
    patternFromKeys(S, (int) S.size());
    gatherAValues(A);
    initValues();

    // the extra fill starts at zero: one sweep gives it values before the first selection
    if (m_initial_level > 0) { sweep(); }
    m_counts.resize(n + 1);
    m_cand_ro.resize(n + 1);

    for (int s = 0; s < sweeps && n > 0; s++)
    {
        // 1. Candidates A_i, S_i and U_k (k in L_i), valued by one Jacobi update of the factors
        candidateCountKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(A.row_offsets.raw(), m_ro.raw(), m_ci.raw(),
                m_diag.raw(), n, m_counts.raw());
        cudaCheckError();
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(m_counts.begin(), m_counts.begin() + n + 1, m_counts.begin());
        cudaCheckError();
        const int total = m_counts[n];
        ensureSize(m_cand_keys, total);
        candidateFillKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(A.row_offsets.raw(), A.col_indices.raw(), m_ro.raw(),
                m_ci.raw(), m_diag.raw(), n, m_counts.raw(), m_cand_keys.raw());
        cudaCheckError();
        // Keys i * n + j need only the low bits of n^2: a radix sort over that range
        ensureSize(m_cand_keys_alt, total);
        {
            int end_bit = 1;

            while (end_bit < 63 && ((int64_t) 1 << end_bit) < (int64_t) n * n) { end_bit++; }

            amgx::cub::DoubleBuffer<int64_t> buffers(m_cand_keys.raw(), m_cand_keys_alt.raw());
            size_t tmp_bytes = 0;
            amgx::cub::DeviceRadixSort::SortKeys(NULL, tmp_bytes, buffers, total, 0, end_bit);
            ensureSize(m_sort_tmp, (tmp_bytes + sizeof(int64_t) - 1) / sizeof(int64_t) + 1);
            amgx::cub::DeviceRadixSort::SortKeys(m_sort_tmp.raw(), tmp_bytes, buffers, total, 0, end_bit);
            cudaCheckError();

            if (buffers.Current() != m_cand_keys.raw()) { m_cand_keys.swap(m_cand_keys_alt); }
        }
        const int m = amgx::thrust::unique(m_cand_keys.begin(), m_cand_keys.begin() + total) - m_cand_keys.begin();
        cudaCheckError();
        ensureSize(m_cand_row, m);
        ensureSize(m_cand_ci, m);
        ensureSize(m_cand_vals, m);
        ensureSize(m_keep, m);
        patternRowsKernel <<< piluNumBlocks(m), PILU_BLOCK_SIZE>>>(m_cand_keys.raw(), m, (int64_t) n, m_cand_row.raw(), m_cand_ci.raw());
        cudaCheckError();
        amgx::thrust::lower_bound(m_cand_row.begin(), m_cand_row.begin() + m, amgx::thrust::counting_iterator<int>(0),
                                  amgx::thrust::counting_iterator<int>(n + 1), m_cand_ro.begin());
        cudaCheckError();
        candidateValuesKernel <<< piluNumBlocks(m), PILU_BLOCK_SIZE>>>(m_cand_row.raw(), m_cand_ci.raw(), m, (int64_t) n,
                m_A_keys.raw(), m_A_pos.raw(), (int) m_A_keys.size(), A.values.raw(), m_ro.raw(), m_ci.raw(), m_diag.raw(),
                m_F.raw(), m_row_norms.raw(), m_cand_vals.raw());
        cudaCheckError();
        // 2. Keep the diagonal and the largest entries per row part (AMGCL rule)
        candidateSelectKernel <<< piluNumBlocks(m), PILU_BLOCK_SIZE>>>(m_cand_ro.raw(), m_cand_row.raw(), m_cand_ci.raw(),
                m_cand_vals.raw(), m, m_thresholds.raw(), m_len_lower.raw(), m_len_upper.raw(), fill_factor, m_keep.raw());
        cudaCheckError();
        keepCountKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(m_cand_ro.raw(), m_keep.raw(), n, m_counts.raw());
        cudaCheckError();
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(m_counts.begin(), m_counts.begin() + n + 1, m_counts.begin());
        cudaCheckError();
        const int kept = m_counts[n];
        m_nnz = kept;
        thrust_wrapper::copy<TConfig::memSpace>(m_counts.begin(), m_counts.begin() + n + 1, m_ro.begin());
        m_row.resize(kept);
        m_ci.resize(kept);
        m_F.resize(kept);
        m_apos.resize(kept);
        compactKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(m_cand_ro.raw(), m_cand_ci.raw(), m_cand_vals.raw(), m_keep.raw(),
                n, m_ro.raw(), m_row.raw(), m_ci.raw(), m_F.raw());
        cudaCheckError();
        diagonalOfRowsKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(m_ro.raw(), m_ci.raw(), n, m_diag.raw());
        cudaCheckError();

        if (kept > 0)
        {
            aPositionsKernel <<< piluNumBlocks(kept), PILU_BLOCK_SIZE>>>(m_row.raw(), m_ci.raw(), kept, (int64_t) n,
                    m_A_keys.raw(), m_A_pos.raw(), (int) m_A_keys.size(), m_apos.raw());
            cudaCheckError();
        }

        gatherAValues(A);
        // 3. Sweep on the reduced pattern
        sweep();
    }

    splitLU(L, U);
}

// ---------------- ParILU (static entry points) ----------------

template <class T_Config>
void ParILU<T_Config>::factorPowerPattern(const Matrix<TConfig> &A, int level, int sweeps,
        Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    ParILUFactor<TConfig> factor;
    factor.factorPower(A, level, sweeps, L, U);
}

template <class T_Config>
void ParILU<T_Config>::factorThreshold(const Matrix<TConfig> &A, double drop_tol, double fill_factor, int sweeps,
                                       Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    ParILUFactor<TConfig> factor;
    factor.factorThreshold(A, drop_tol, fill_factor, sweeps, L, U);
}

template <class T_Config>
double ParILU<T_Config>::patternResidual(const Matrix<TConfig> &A, const Matrix<TConfig> &L, const Matrix<TConfig> &U)
{
    typedef PatternOps<TConfig> Ops;
    FactorState<TConfig> state(A);
    I64Vector L_keys, U_keys, keys, T_keys;
    MVector L_vals, U_vals, F, T_vals, R;
    Ops::toKeys(L, L_keys, L_vals);
    Ops::toKeys(U, U_keys, U_vals);
    Ops::merge(L_keys, U_keys, keys);
    F.resize(keys.size());
    R.resize(keys.size());
    // L and U have disjoint patterns: F = L + U
    MVector part(keys.size());
    gatherFactorKernel <<< piluNumBlocks(std::max<int>(1, keys.size())), PILU_BLOCK_SIZE>>>(keys.raw(), (int) keys.size(),
            L_keys.raw(), L_vals.raw(), (int) L_keys.size(), F.raw());
    gatherFactorKernel <<< piluNumBlocks(std::max<int>(1, keys.size())), PILU_BLOCK_SIZE>>>(keys.raw(), (int) keys.size(),
            U_keys.raw(), U_vals.raw(), (int) U_keys.size(), part.raw());
    cudaCheckError();
    thrust_wrapper::transform<TConfig::memSpace>(F.begin(), F.end(), part.begin(), F.begin(), amgx::thrust::plus<ValueTypeA>());
    state.product(keys, F, T_keys, T_vals);
    patternResidualKernel <<< piluNumBlocks(std::max<int>(1, keys.size())), PILU_BLOCK_SIZE>>>(keys.raw(), (int) keys.size(),
            state.A_keys.raw(), state.A_vals.raw(), (int) state.A_keys.size(), T_keys.raw(), T_vals.raw(), (int) T_keys.size(), R.raw());
    cudaCheckError();
    return std::sqrt(thrust_wrapper::transform_reduce<TConfig::memSpace>(R.begin(), R.end(), square_of<ValueTypeA>(), 0.0,
                     amgx::thrust::plus<double>()));
}

// ---------------- Triangular solve ----------------

template <class T_Config>
void ILUTriangularSolve<T_Config>::release()
{
    if (m_L_spsv) { cusparseSpSV_destroyDescr(m_L_spsv); m_L_spsv = NULL; }

    if (m_U_spsv) { cusparseSpSV_destroyDescr(m_U_spsv); m_U_spsv = NULL; }

    if (m_L_descr) { cusparseDestroySpMat(m_L_descr); m_L_descr = NULL; }

    if (m_U_descr) { cusparseDestroySpMat(m_U_descr); m_U_descr = NULL; }

    if (m_in_descr) { cusparseDestroyDnVec(m_in_descr); m_in_descr = NULL; }

    if (m_mid_descr) { cusparseDestroyDnVec(m_mid_descr); m_mid_descr = NULL; }

    if (m_out_descr) { cusparseDestroyDnVec(m_out_descr); m_out_descr = NULL; }
}

template <class T_Config>
bool ILUTriangularSolve<T_Config>::refresh(const Matrix<TConfig> &L, const Matrix<TConfig> &U)
{
    if (m_L != &L || m_U != &U || L.values.raw() != m_L_values || U.values.raw() != m_U_values ||
            L.get_num_nz() != m_L_nnz || U.get_num_nz() != m_U_nnz || U.get_num_rows() != m_n)
    {
        return false;
    }

    if (m_n > 0)
    {
        inverseDiagonalKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(m_n, U.row_offsets.raw(), U.col_indices.raw(),
                U.values.raw(), m_dinv.raw());
        cudaCheckError();
    }

    if (m_method == CUSPARSE && m_n > 0)
    {
#if CUSPARSE_VERSION >= 12100
        cusparseHandle_t handle = Cusparse::get_instance().get_handle();
        cusparseCheckError(cusparseSpSV_updateMatrix(handle, m_L_spsv, const_cast<ValueTypeA *>(L.values.raw()),
                           CUSPARSE_SPSV_UPDATE_GENERAL));
        cusparseCheckError(cusparseSpSV_updateMatrix(handle, m_U_spsv, const_cast<ValueTypeA *>(U.values.raw()),
                           CUSPARSE_SPSV_UPDATE_GENERAL));
#else
        return false;
#endif
    }

    return true;
}

template <class T_Config>
void ILUTriangularSolve<T_Config>::setup(const Matrix<TConfig> &L, const Matrix<TConfig> &U, Method method, int iters,
        double damping)
{
    release();
    m_L = &L;
    m_U = &U;
    m_L_values = L.values.raw();
    m_U_values = U.values.raw();
    m_L_nnz = L.get_num_nz();
    m_U_nnz = U.get_num_nz();
    m_method = method;
    m_iters = iters;
    m_damping = damping;
    m_n = U.get_num_rows();
    m_dinv.resize(m_n);

    if (m_n > 0)
    {
        inverseDiagonalKernel <<< piluNumBlocks(m_n), PILU_BLOCK_SIZE>>>(m_n, U.row_offsets.raw(), U.col_indices.raw(),
                U.values.raw(), m_dinv.raw());
        cudaCheckError();
    }

    m_t0.resize(m_n);
    m_t1.resize(m_n);
    m_t2.resize(m_n);

    if (method != CUSPARSE || m_n == 0) { return; }

    const cudaDataType dtype = sizeof(ValueTypeA) == 4 ? CUDA_R_32F : CUDA_R_64F;
    Cusparse::get_instance().create_handle();
    cusparseHandle_t handle = Cusparse::get_instance().get_handle();
    m_in.resize(m_n);
    m_mid.resize(m_n);
    m_out.resize(m_n);
    cusparseCheckError(cusparseCreateDnVec(&m_in_descr, m_n, m_in.raw(), dtype));
    cusparseCheckError(cusparseCreateDnVec(&m_mid_descr, m_n, m_mid.raw(), dtype));
    cusparseCheckError(cusparseCreateDnVec(&m_out_descr, m_n, m_out.raw(), dtype));
    const Matrix<TConfig> *factors[2] = {&L, &U};
    cusparseSpMatDescr_t *descrs[2] = {&m_L_descr, &m_U_descr};
    cusparseSpSVDescr_t *spsv[2] = {&m_L_spsv, &m_U_spsv};
    MVector *buffers[2] = {&m_L_buffer, &m_U_buffer};
    cusparseDnVecDescr_t in_descr[2] = {m_in_descr, m_mid_descr};
    cusparseDnVecDescr_t out_descr[2] = {m_mid_descr, m_out_descr};
    const ValueTypeA one = ValueTypeA(1);

    for (int f = 0; f < 2; f++)
    {
        const Matrix<TConfig> &M = *factors[f];
        cusparseCheckError(cusparseCreateCsr(descrs[f], m_n, m_n, M.get_num_nz(), const_cast<IndexType *>(M.row_offsets.raw()),
                                             const_cast<IndexType *>(M.col_indices.raw()), const_cast<ValueTypeA *>(M.values.raw()),
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dtype));
        cusparseFillMode_t fill = f == 0 ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag = f == 0 ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseCheckError(cusparseSpMatSetAttribute(*descrs[f], CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
        cusparseCheckError(cusparseSpMatSetAttribute(*descrs[f], CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));
        cusparseCheckError(cusparseSpSV_createDescr(spsv[f]));
        size_t bytes = 0;
        cusparseCheckError(cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, *descrs[f], in_descr[f],
                           out_descr[f], dtype, CUSPARSE_SPSV_ALG_DEFAULT, *spsv[f], &bytes));
        buffers[f]->resize(std::max<size_t>(1, (bytes + sizeof(ValueTypeA) - 1) / sizeof(ValueTypeA)));
        cusparseCheckError(cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, *descrs[f], in_descr[f],
                           out_descr[f], dtype, CUSPARSE_SPSV_ALG_DEFAULT, *spsv[f], buffers[f]->raw()));
    }
}

template <class T_Config>
void ILUTriangularSolve<T_Config>::apply(const VVector &r, VVector &z)
{
    const int n = m_n;

    if (n == 0) { return; }

    const int nb = piluNumBlocks(n);

    if (m_method == CUSPARSE)
    {
        cusparseHandle_t handle = Cusparse::get_instance().get_handle();
        const cudaDataType dtype = sizeof(ValueTypeA) == 4 ? CUDA_R_32F : CUDA_R_64F;
        const ValueTypeA one = ValueTypeA(1);
        convertKernel <<< nb, PILU_BLOCK_SIZE>>>(n, r.raw(), m_in.raw());
        cusparseCheckError(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, m_L_descr, m_in_descr, m_mid_descr,
                                              dtype, CUSPARSE_SPSV_ALG_DEFAULT, m_L_spsv));
        cusparseCheckError(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, m_U_descr, m_mid_descr, m_out_descr,
                                              dtype, CUSPARSE_SPSV_ALG_DEFAULT, m_U_spsv));
        convertKernel <<< nb, PILU_BLOCK_SIZE>>>(n, m_out.raw(), z.raw());
        cudaCheckError();
        return;
    }

    // Damped Jacobi iterations: lower solve ping-pongs between t0 and t1, upper between t2 and z
    const ValueTypeB d = ValueTypeB(m_damping);
    ValueTypeB *y_old = m_t0.raw(), *y_new = m_t1.raw();
    scaleKernel <<< nb, PILU_BLOCK_SIZE>>>(n, d, r.raw(), y_old);

    for (int it = 0; it < m_iters; it++)
    {
        lowerJacobiKernel <<< nb, PILU_BLOCK_SIZE>>>(n, m_L->row_offsets.raw(), m_L->col_indices.raw(), m_L->values.raw(),
                r.raw(), y_old, d, y_new);
        std::swap(y_old, y_new);
    }

    ValueTypeB *x_old = m_t2.raw(), *x_new = y_new;
    scaleDiagKernel <<< nb, PILU_BLOCK_SIZE>>>(n, d, m_dinv.raw(), y_old, x_old);

    for (int it = 0; it < m_iters; it++)
    {
        upperJacobiKernel <<< nb, PILU_BLOCK_SIZE>>>(n, m_U->row_offsets.raw(), m_U->col_indices.raw(), m_U->values.raw(),
                m_dinv.raw(), y_old, x_old, d, x_new);
        std::swap(x_old, x_new);
    }

    cudaMemcpyAsync(z.raw(), x_old, sizeof(ValueTypeB) * n, cudaMemcpyDeviceToDevice);
    cudaCheckError();
}

#define AMGX_CASE_LINE(CASE) template struct ParILU<TemplateMode<CASE>::Type>; template class ParILUFactor<TemplateMode<CASE>::Type>; template class ILUTriangularSolve<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
