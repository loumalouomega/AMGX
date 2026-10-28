// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Deflated preconditioner adapted from AMGCL (amgcl/deflated_solver.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).
// J. M. Tang, R. Nabben, C. Vuik, Y. A. Erlangga, "Comparison of two-level preconditioners derived
// from deflation, domain decomposition and multigrid methods", J. Sci. Comput. 39, 2009.

#pragma once

#include <solvers/solver.h>
#include <cusolverDn.h>

namespace amgx
{

/***************************************************
 * Coarse-space deflation around a preconditioner M (A-DEF2, as AMGCL apply()):
 *   y = M b,  y += Z E^{-1} Z^T (b - A y),  E = Z^T A Z
 * Z holds the k deflation vectors of AMGX_matrix_set_deflation_vectors (or the near-nullspace with
 * deflation_use_nullspace = 1). E is LU-factorized on the device with cuSOLVER at setup; the
 * apply path is allocation- and synchronization-free.
 **************************************************/
template<class T_Config>
class Deflation_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        typedef typename Base::VVector VVector;
        typedef typename Base::ValueTypeA ValueTypeA;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename Matrix<TConfig>::IVector IVector;

    private:
        bool m_use_nullspace;
        int m_n, m_k;
        VVector m_Z;            // n x k, column-major
        VVector m_E;            // LU factors of Z^T A Z, k x k column-major
        IVector m_ipiv, m_info;
        VVector m_f;            // coarse right-hand side
        VVector m_r;            // residual
        VVector m_t;            // preconditioned residual
        cusolverDnHandle_t m_cusolver;

        Solver<T_Config> *m_preconditioner;

    public:
        Deflation_Solver(AMG_Config &cfg, const std::string &cfg_scope);
        ~Deflation_Solver();

        bool is_residual_needed() const { return false; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return m_preconditioner != NULL && m_preconditioner->isColoringNeeded(); }
        void getColoringScope(std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }
        bool getReorderColsByColorDesired() const { return m_preconditioner != NULL && m_preconditioner->getReorderColsByColorDesired(); }
        bool getInsertDiagonalDesired() const { return m_preconditioner != NULL && m_preconditioner->getInsertDiagonalDesired(); }

        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);

        int numVectors() const { return m_k; }
};

template<class T_Config>
class Deflation_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new Deflation_Solver<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx
