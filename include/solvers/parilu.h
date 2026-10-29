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

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>
#include <cusparse.h>

namespace amgx
{

/***************************************************
 * Factors are sorted scalar CSR matrices: L strictly lower (unit diagonal implied)
 * and U upper triangular including the diagonal. A must be scalar, without external diagonal.
 **************************************************/
template <class T_Config>
struct ParILU
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;

    // ILU on the fixed pattern of A^(level+1) with `sweeps` fixed-point sweeps
    static void factorPowerPattern(const Matrix<TConfig> &A, int level, int sweeps,
                                   Matrix<TConfig> &L, Matrix<TConfig> &U);

    // ParILUT: keep entries with |v| > drop_tol * sum_j |a_ij| / (off-diagonal entries of a_i) and at most
    // fill_factor * nnz(lower/upper part of a_i) of them per row and part (AMGCL dropping rule)
    static void factorThreshold(const Matrix<TConfig> &A, double drop_tol, double fill_factor, int sweeps,
                                Matrix<TConfig> &L, Matrix<TConfig> &U);

    // || (A - (L + I) U) restricted to the pattern of L and U ||_F
    static double patternResidual(const Matrix<TConfig> &A, const Matrix<TConfig> &L, const Matrix<TConfig> &U);
};

/***************************************************
 * Persistent factorization state. The pattern S of L + U is a row-sorted CSR (diagonal always
 * present); a sweep is one kernel, one thread per entry of S, computing
 *   l_ij = (a_ij - sum_{k<j} l_ik u_kj) / u_jj,  u_ij = a_ij - sum_{k<i} l_ik u_kj
 * from the previous sweep's values (Chow-Patel), the sums by binary search in the rows of S, with
 * a fixed summation order: deterministic, no sort, SpGEMM, allocation or host synchronization.
 * ILUP keeps S and the position of each a_ij, so a refactorization on the same pattern of A only
 * redoes the values. ParILUT steps generate the candidates A_i, S_i and U_k (k in L_i) with one
 * key sort per step and select per row and part (rank by magnitude, column as tie-break).
 **************************************************/
template <class T_Config>
class ParILUFactor
{
    public:
        typedef T_Config TConfig;
        DEFINE_VECTOR_TYPES
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecDouble>::Type> DVector;

        ParILUFactor() : m_n(0), m_nnz(0), m_power_valid(false), m_initial_level(0) {}

        // ParILUT starts from the pattern of A^(initial_level+1) (0: the pattern of A)
        void setInitialFillLevel(int initial_level) { m_initial_level = initial_level; }

        // ILU on the pattern of A^(level+1); the pattern is kept for refactorPower
        void factorPower(const Matrix<TConfig> &A, int level, int sweeps, Matrix<TConfig> &L, Matrix<TConfig> &U);
        // True if A has the pattern of the last factorPower (values may differ)
        bool samePattern(const Matrix<TConfig> &A) const;
        // factorPower on the kept pattern: values and sweeps only
        void refactorPower(const Matrix<TConfig> &A, int sweeps, Matrix<TConfig> &L, Matrix<TConfig> &U);
        // ParILUT (AMGCL dropping rule)
        void factorThreshold(const Matrix<TConfig> &A, double drop_tol, double fill_factor, int sweeps,
                             Matrix<TConfig> &L, Matrix<TConfig> &U);

    private:
        int m_n, m_nnz;
        bool m_power_valid;
        int m_initial_level;
        // A: sorted keys with their positions in A, row statistics, a copy of its pattern
        I64Vector m_A_keys;
        IVector m_A_pos, m_A_ro, m_A_ci, m_len_lower, m_len_upper;
        MVector m_A_diag;
        DVector m_row_norms, m_row_abs_sums;
        // Pattern S and per-entry data
        IVector m_ro, m_ci, m_row, m_diag, m_apos;
        MVector m_F, m_F_old, m_aval;
        // ParILUT work buffers (grow only)
        I64Vector m_cand_keys, m_cand_keys_alt;
        Vector<typename TConfig::template setVecPrec<AMGX_vecInt64>::Type> m_sort_tmp;
        IVector m_cand_ro, m_cand_row, m_cand_ci, m_keep, m_counts;
        MVector m_cand_vals;
        DVector m_thresholds;

        void prepareA(const Matrix<TConfig> &A);
        // Sorted keys of the pattern of A^(level+1) (unit-valued SpGEMM powers)
        void powerKeys(const Matrix<TConfig> &A, int level, I64Vector &keys);
        void patternFromKeys(const I64Vector &keys, int nnz);
        void gatherAValues(const Matrix<TConfig> &A);
        void initValues();
        void sweep();
        void splitLU(Matrix<TConfig> &L, Matrix<TConfig> &U);
};

/***************************************************
 * Applies z = U^{-1} L^{-1} r with the factors above, either approximately with damped
 * Jacobi iterations or exactly with cuSPARSE SpSV. All buffers are allocated in setup.
 **************************************************/
template <class T_Config>
class ILUTriangularSolve
{
    public:
        typedef T_Config TConfig;
        DEFINE_VECTOR_TYPES
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::VecPrec ValueTypeB;

        enum Method { JACOBI, CUSPARSE };

        ILUTriangularSolve() : m_L(NULL), m_U(NULL), m_L_values(NULL), m_U_values(NULL), m_L_nnz(0), m_U_nnz(0),
            m_method(JACOBI), m_iters(2), m_damping(0.72), m_n(0),
            m_L_descr(NULL), m_U_descr(NULL), m_in_descr(NULL), m_mid_descr(NULL), m_out_descr(NULL),
            m_L_spsv(NULL), m_U_spsv(NULL) {}
        ~ILUTriangularSolve() { release(); }

        // L and U are kept by reference and must outlive the solver
        void setup(const Matrix<TConfig> &L, const Matrix<TConfig> &U, Method method, int iters, double damping);
        // New values in the same L and U (same pattern and storage): cuSPARSE updates its analysis
        // in place (cusparseSpSV_updateMatrix); returns false if a full setup is needed
        bool refresh(const Matrix<TConfig> &L, const Matrix<TConfig> &U);
        void apply(const VVector &r, VVector &z);

    private:
        void release();

        const Matrix<TConfig> *m_L, *m_U;
        const ValueTypeA *m_L_values, *m_U_values;
        int m_L_nnz, m_U_nnz;
        Method m_method;
        int m_iters;
        double m_damping;
        int m_n;
        MVector m_dinv;
        VVector m_t0, m_t1, m_t2;
        // cuSPARSE SpSV state, in matrix precision
        MVector m_in, m_mid, m_out, m_L_buffer, m_U_buffer;
        cusparseSpMatDescr_t m_L_descr, m_U_descr;
        cusparseDnVecDescr_t m_in_descr, m_mid_descr, m_out_descr;
        cusparseSpSVDescr_t m_L_spsv, m_U_spsv;
};

} // namespace amgx
