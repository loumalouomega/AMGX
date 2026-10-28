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

template <class T_Config>
void ParILU<T_Config>::factorPowerPattern(const Matrix<TConfig> &A, int level, int sweeps,
        Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    typedef PatternOps<TConfig> Ops;
    FactorState<TConfig> state(A);
    const int n = state.n;
    // Pattern of A^(level+1), from products of the unit-valued pattern matrix
    I64Vector keys;
    MVector unit_vals, F;
    Ops::toKeys(A, keys, unit_vals, true);
    Matrix<TConfig> pattern, power;
    Ops::fromKeys(keys, unit_vals, n, pattern);

    for (int l = 0; l < level; l++)
    {
        Matrix<TConfig> next;
        next.set_initialized(0);
        next.addProps(CSR);
        CSR_Multiply<TConfig>::csr_multiply(l == 0 ? pattern : power, pattern, next, state.wk);
        power.set_initialized(0);
        power.copy(next);
        power.set_initialized(1);
    }

    if (level > 0) { Ops::toKeys(power, keys, unit_vals); }

    I64Vector S;
    state.withDiagonal(keys, S);
    state.init(S, F);

    for (int s = 0; s < sweeps; s++) { state.sweep(S, F); }

    Ops::split(S, F, n, L, U, NULL);
}

template <class T_Config>
void ParILU<T_Config>::factorThreshold(const Matrix<TConfig> &A, double drop_tol, double fill_factor, int sweeps,
                                       Matrix<TConfig> &L, Matrix<TConfig> &U)
{
    typedef PatternOps<TConfig> Ops;
    typedef typename FactorState<TConfig>::DVector DVector;
    FactorState<TConfig> state(A);
    const int n = state.n;
    DVector thresholds(n);

    if (n > 0)
    {
        dropToleranceKernel <<< piluNumBlocks(n), PILU_BLOCK_SIZE>>>(n, drop_tol, state.row_abs_sums.raw(), state.len_lower.raw(),
                state.len_upper.raw(), thresholds.raw());
        cudaCheckError();
    }

    // Start from ILU(0) values on the pattern of A
    I64Vector keys;
    MVector F;
    state.withDiagonal(state.A_keys, keys);
    state.init(keys, F);

    for (int s = 0; s < sweeps; s++)
    {
        // 1. Candidates: pattern of A and of (L + I) U, then a sweep on the extended pattern
        I64Vector T_keys, candidates, ext_keys;
        MVector T_vals, F_ext;
        state.product(keys, F, T_keys, T_vals);
        Ops::merge(state.A_keys, T_keys, candidates);
        Ops::merge(candidates, keys, ext_keys);
        F_ext.resize(ext_keys.size());
        gatherFactorKernel <<< piluNumBlocks(ext_keys.size()), PILU_BLOCK_SIZE>>>(ext_keys.raw(), (int) ext_keys.size(),
                keys.raw(), F.raw(), (int) keys.size(), F_ext.raw());
        cudaCheckError();
        state.sweep(ext_keys, F_ext, T_keys, T_vals);
        // 2. Drop small entries and keep the largest fill_factor * nnz(a_i) per row and part
        const int nnz = ext_keys.size();
        I64Vector segment(nnz);
        DVector neg_abs(nnz);
        IVector order(nnz), rank(nnz), keep(nnz);
        selectionKeysKernel <<< piluNumBlocks(nnz), PILU_BLOCK_SIZE>>>(ext_keys.raw(), F_ext.raw(), nnz, (int64_t) n,
                segment.raw(), neg_abs.raw());
        cudaCheckError();
        thrust_wrapper::sequence<TConfig::memSpace>(order.begin(), order.end());
        amgx::thrust::stable_sort_by_key(neg_abs.begin(), neg_abs.end(),
                                         amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(order.begin(), segment.begin())));
        amgx::thrust::stable_sort_by_key(segment.begin(), segment.end(),
                                         amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(order.begin(), neg_abs.begin())));
        amgx::thrust::exclusive_scan_by_key(segment.begin(), segment.end(), amgx::thrust::constant_iterator<int>(1), rank.begin());
        cudaCheckError();
        selectionKeepKernel <<< piluNumBlocks(nnz), PILU_BLOCK_SIZE>>>(segment.raw(), neg_abs.raw(), rank.raw(), order.raw(),
                nnz, (int64_t) n, thresholds.raw(), state.len_lower.raw(), state.len_upper.raw(), fill_factor, keep.raw());
        cudaCheckError();
        const int kept = (int) thrust_wrapper::count_if<TConfig::memSpace>(keep.begin(), keep.end(), is_kept());
        keys.resize(kept);
        F.resize(kept);
        amgx::thrust::copy_if(amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(ext_keys.begin(), F_ext.begin())),
                              amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(ext_keys.end(), F_ext.end())),
                              keep.begin(),
                              amgx::thrust::make_zip_iterator(amgx::thrust::make_tuple(keys.begin(), F.begin())),
                              is_kept());
        cudaCheckError();
        // 3. Sweep on the reduced pattern
        state.sweep(keys, F);
    }

    Ops::split(keys, F, n, L, U, NULL);
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
void ILUTriangularSolve<T_Config>::setup(const Matrix<TConfig> &L, const Matrix<TConfig> &U, Method method, int iters,
        double damping)
{
    release();
    m_L = &L;
    m_U = &U;
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

#define AMGX_CASE_LINE(CASE) template struct ParILU<TemplateMode<CASE>::Type>; template class ILUTriangularSolve<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
