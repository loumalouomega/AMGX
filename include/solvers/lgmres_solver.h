// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// LGMRES adapted from AMGCL (amgcl/solver/lgmres.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).
// A. H. Baker, E. R. Jessup, T. Manteuffel, "A technique for accelerating the convergence of
// restarted GMRES", SIAM J. Matrix Anal. Appl. 26(4), 2005.

#pragma once

#include <solvers/solver.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <vector>

namespace amgx
{

/***************************************************
 * LGMRES(M, K), right preconditioned: restarted GMRES whose M + K basis columns end with
 * the (normalized) corrections dx of the last K restart cycles, as AMGCL. M is
 * gmres_n_restart (Krylov steps per cycle), K is lgmres_augmentation_k; K = 0 is GMRES(M).
 * m_M holds the total basis size M + K. One iteration is one Arnoldi step.
 * Real and complex modes, L2 norm only.
 **************************************************/
template<class T_Config>
class LGMRES_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename types::PODTypes<ValueTypeB>::type PodTypeB;

    private:
        int m_M, m_K;
        std::vector<VVector> m_V;      // Arnoldi basis, M + 1 vectors
        std::vector<VVector> m_aug;    // augmentation vectors, K slots
        std::vector<int> m_aug_order;  // slots from oldest to newest
        int m_num_aug_cycle;           // augmentation columns used in the current cycle
        VVector m_Z, m_dx;
        cusp::array2d<ValueTypeB, cusp::host_memory, cusp::column_major> m_H;
        cusp::array1d<ValueTypeB, cusp::host_memory> m_s, m_cs, m_sn;

        Solver<T_Config> *m_preconditioner;

        // Column j of W: the Krylov vector V_j or an augmentation vector
        VVector &column(int j);
        // dst = A P^{-1} src
        void applyOperator(VVector &src, VVector &dst);

    public:
        LGMRES_Solver(AMG_Config &cfg, const std::string &cfg_scope);
        ~LGMRES_Solver();

        bool is_residual_needed() const { return false; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded() const { return m_preconditioner != NULL && m_preconditioner->isColoringNeeded(); }
        void getColoringScope(std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }
        bool getReorderColsByColorDesired() const { return m_preconditioner != NULL && m_preconditioner->getReorderColsByColorDesired(); }
        bool getInsertDiagonalDesired() const { return m_preconditioner != NULL && m_preconditioner->getInsertDiagonalDesired(); }

        void solve_init(VVector &b, VVector &x, bool xIsZero);
        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);
        void solve_finalize(VVector &b, VVector &x) {}
};

template<class T_Config>
class LGMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new LGMRES_Solver<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx
