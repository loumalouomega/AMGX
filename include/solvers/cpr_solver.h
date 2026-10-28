// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Constrained pressure residual preconditioner adapted from AMGCL (amgcl/preconditioner/cpr.hpp,
// amgcl/preconditioner/cpr_drs.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).
// J. R. Wallis, "Incomplete Gaussian elimination as a preconditioning for generalized conjugate
// gradient acceleration", SPE 12265, 1983; S. Gries et al., "Preconditioning for efficiently
// applying algebraic multigrid in fully implicit reservoir simulations", SPE J. 19(4), 2014.

#pragma once

#include <solvers/solver.h>
#include <vector>

namespace amgx
{

/***************************************************
 * Two-stage CPR preconditioner for systems with B interleaved unknowns per cell, pressure first
 * (BSR blocks or scalar rows with cpr_block_dim):
 *   x = S(b),  x += Scatter P(W (b - A x))
 * S is a global smoother on A (cpr_global_smoother), P a solver on the pressure matrix
 * App(I, J) = sum_k w_I[k] A(I B + k, J B) (cpr_pressure_solver). The weights w are the dynamic
 * row-sum (DRS) decoupling or quasi-IMPES (first column of D_I^{-T}, D_I the diagonal block).
 **************************************************/
template<class T_Config>
class CPR_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        typedef typename Base::VVector VVector;
        typedef typename Base::ValueTypeA ValueTypeA;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename Matrix<TConfig>::MVector MVector;

    private:
        bool m_drs;
        int m_block_dim_param;
        double m_eps_dd, m_eps_ps;
        std::vector<double> m_component_weights;

        int m_B, m_np;
        Matrix<TConfig> m_App;      // pressure matrix
        MVector m_weights;          // restriction weights, one per scalar row
        VVector m_r, m_t;           // residual and smoother correction
        VVector m_rp, m_xp;         // pressure residual and correction

        Solver<T_Config> *m_smoother;
        Solver<T_Config> *m_pressure_solver;

    public:
        CPR_Solver(AMG_Config &cfg, const std::string &cfg_scope);
        ~CPR_Solver();

        bool is_residual_needed() const { return false; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return m_smoother != NULL && m_smoother->isColoringNeeded(); }
        void getColoringScope(std::string &cfg_scope_for_coloring) const { if (m_smoother != NULL) m_smoother->getColoringScope(cfg_scope_for_coloring); }
        bool getReorderColsByColorDesired() const { return m_smoother != NULL && m_smoother->getReorderColsByColorDesired(); }
        bool getInsertDiagonalDesired() const { return m_smoother != NULL && m_smoother->getInsertDiagonalDesired(); }

        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);

        // Read access for tests
        const Matrix<TConfig> &pressureMatrix() const { return m_App; }
        const MVector &weights() const { return m_weights; }
        int blockDim() const { return m_B; }
};

template<class T_Config>
class CPR_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new CPR_Solver<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx
