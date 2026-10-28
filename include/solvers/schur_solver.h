// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Schur complement pressure correction adapted from AMGCL
// (amgcl/preconditioner/schur_pressure_correction.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#pragma once

#include <solvers/solver.h>

namespace amgx
{

/***************************************************
 * Block preconditioner for saddle-point systems [Kuu Kup; Kpu Kpp] [u; p] = [fu; fp]. The pressure
 * unknowns come from AMGX_matrix_set_row_markers or schur_pmask_pattern.
 *   variant 1 (pressure correction): u = U(fu); p = P(fp - Kpu u); u = U(fu - Kup p)
 *   variant 2 (block triangular):    p = P(fp); u = U(fu - Kup p)
 * U solves with Kuu (schur_velocity_solver), P with the assembled pressure matrix
 * (schur_pressure_solver): Kpp, Kpp - diag(Kpu M Kup) or Kpp - Kpu M Kup (schur_adjust_p 0/1/2),
 * M = 1 / sum_j |Kuu_ij| (SIMPLEC) or diag(Kuu)^-1.
 **************************************************/
template<class T_Config>
class SchurPressureCorrection_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        typedef typename Base::VVector VVector;
        typedef typename Base::ValueTypeA ValueTypeA;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename Matrix<TConfig>::MVector MVector;
        typedef typename Matrix<TConfig>::IVector IVector;

    private:
        std::string m_pattern;
        int m_variant, m_adjust_p;
        bool m_simplec;

        int m_nu, m_np;
        IVector m_idx_u, m_idx_p;           // scalar unknowns of each field
        Matrix<TConfig> m_Kuu, m_Kup, m_Kpu, m_Kpp, m_S;
        VVector m_r;                        // outer residual
        VVector m_fu, m_fp, m_u, m_p, m_tu, m_tp;

        Solver<T_Config> *m_velocity_solver;
        Solver<T_Config> *m_pressure_solver;

        void buildMask(const Matrix<TConfig> &A, int n, IVector &mask) const;

    public:
        SchurPressureCorrection_Solver(AMG_Config &cfg, const std::string &cfg_scope);
        ~SchurPressureCorrection_Solver();

        bool is_residual_needed() const { return false; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        // The inner solvers run on sub-matrices, which they set up themselves
        bool isColoringNeeded() const { return false; }
        bool getReorderColsByColorDesired() const { return false; }
        bool getInsertDiagonalDesired() const { return false; }

        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);

        // Read access for tests
        const Matrix<TConfig> &velocityMatrix() const { return m_Kuu; }
        const Matrix<TConfig> &pressureMatrix() const { return m_S; }
        const Matrix<TConfig> &Kup() const { return m_Kup; }
        const Matrix<TConfig> &Kpu() const { return m_Kpu; }
        const Matrix<TConfig> &Kpp() const { return m_Kpp; }
        int numPressure() const { return m_np; }
};

template<class T_Config>
class SchurPressureCorrection_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new SchurPressureCorrection_Solver<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx
