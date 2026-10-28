// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// BiCGStab(L) adapted from AMGCL (amgcl/solver/bicgstabl.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).
// G. L. G. Sleijpen, D. R. Fokkema, "BiCGstab(l) for linear equations involving unsymmetric
// matrices with complex spectrum", ETNA 1, 1993.

#pragma once

#include <solvers/solver.h>
#include <vector>

namespace amgx
{

/***************************************************
 * BiCGStab(L), right preconditioned: L BiCG steps followed by a minimal residual
 * polynomial of degree L. One iteration is one BiCG step (two operator applications),
 * as in AMGCL, so iteration counts compare with BiCGStab. Real and complex modes.
 **************************************************/
template<class T_Config>
class BiCGStabL_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename types::PODTypes<ValueTypeB>::type PodTypeB;

    private:
        int m_L;
        int m_step;                      // BiCG step within the current cycle
        std::vector<VVector> m_R, m_U;   // residuals and search directions, L + 1 each
        VVector m_Rt;                    // shadow residual
        VVector m_X;                     // correction in preconditioned space
        VVector m_T;                     // preconditioned vector
        ValueTypeB m_alpha, m_rho0, m_omega;
        std::vector<ValueTypeB> m_G, m_y; // polynomial part, (L+1) x (L+1) and L + 1

        Solver<T_Config> *m_preconditioner;

        // dst = A P^{-1} src
        void applyOperator(VVector &src, VVector &dst);
        // x += P^{-1} X, X = 0
        void updateSolution(VVector &x);
        // Restart the BiCG recurrences from the residual in R_0
        void restart();

    public:
        BiCGStabL_Solver(AMG_Config &cfg, const std::string &cfg_scope);
        ~BiCGStabL_Solver();

        bool is_residual_needed() const { return true; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return m_preconditioner != NULL && m_preconditioner->isColoringNeeded(); }
        void getColoringScope(std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }
        bool getReorderColsByColorDesired() const { return m_preconditioner != NULL && m_preconditioner->getReorderColsByColorDesired(); }
        bool getInsertDiagonalDesired() const { return m_preconditioner != NULL && m_preconditioner->getInsertDiagonalDesired(); }

        void solve_init(VVector &b, VVector &x, bool xIsZero);
        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);
        void solve_finalize(VVector &b, VVector &x);
};

template<class T_Config>
class BiCGStabL_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new BiCGStabL_Solver<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx
