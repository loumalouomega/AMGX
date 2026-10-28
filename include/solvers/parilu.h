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

        ILUTriangularSolve() : m_method(JACOBI), m_iters(2), m_damping(0.72), m_n(0),
            m_L_descr(NULL), m_U_descr(NULL), m_in_descr(NULL), m_mid_descr(NULL), m_out_descr(NULL),
            m_L_spsv(NULL), m_U_spsv(NULL) {}
        ~ILUTriangularSolve() { release(); }

        // L and U are kept by reference and must outlive the solver
        void setup(const Matrix<TConfig> &L, const Matrix<TConfig> &U, Method method, int iters, double damping);
        void apply(const VVector &r, VVector &z);

    private:
        void release();

        const Matrix<TConfig> *m_L, *m_U;
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
