// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Sparse approximate inverse smoothers adapted from AMGCL (amgcl/relaxation/spai0.hpp,
// amgcl/relaxation/spai1.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#pragma once

#include <solvers/solver.h>

namespace amgx
{

template <class T_Config> class SPAISolver;

/***************************************************
 * Sparse approximate inverse smoothers, x += w M (b - A x)
 *  SPAI0: M = diag(m), m_i = a_ii / sum_j |a_ij|^2 (block rows: M_i = A_ii / sum_j ||A_ij||_F^2)
 *  SPAI1: M has the pattern of A, row i minimizes || m_i^T A - e_i^T ||_2 (scalar matrices)
 * No coloring is needed and the solve path allocates nothing.
 **************************************************/
template<class T_Config>
class SPAISolver_Base : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef typename T_Config::MatPrec ValueTypeA;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef Vector<T_Config> VVector;
        typedef typename Matrix<TConfig>::MVector MVector;

        // Rows of SPAI1 longer than this fall back to the SPAI0 row
        static const int SPAI1_MAX_ROW_LENGTH = 64;

        SPAISolver_Base(AMG_Config &cfg, const std::string &cfg_scope, int level);
        ~SPAISolver_Base() {}

        bool is_residual_needed() const { return false; }
        void solver_setup(bool reuse_matrix_structure);
        bool isColoringNeeded() const { return false; }
        bool getReorderColsByColorDesired() const { return false; }
        bool getInsertDiagonalDesired() const { return false; }
        void solve_init(VVector &b, VVector &x, bool xIsZero) {}
        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);
        void solve_finalize(VVector &b, VVector &x) {}

        // Read access for tests: SPAI0 blocks or the SPAI1 matrix
        const MVector &getDiagonalInverse() const { return m_spai0; }
        const Matrix<TConfig> &getApproximateInverse() const { return m_spai1; }
        int getNumFallbackRows() const { return m_num_fallback_rows; }

    protected:
        virtual void computeSPAI0(const Matrix<TConfig> &A) = 0;
        virtual void computeSPAI1(const Matrix<TConfig> &A) = 0;
        virtual void smoothSPAI0(const Matrix<TConfig> &A, VVector &b, VVector &x, bool xIsZero) = 0;

        Matrix<TConfig> *m_explicit_A;
        int m_level;
        ValueTypeB m_weight;
        MVector m_spai0;           // SPAI0: one b x b block per row
        Matrix<TConfig> m_spai1;   // SPAI1: approximate inverse with the pattern of A
        int m_num_fallback_rows;
        VVector m_y;               // A x
        VVector m_r;               // residual (SPAI1)
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class SPAISolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public SPAISolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef SPAISolver_Base<TConfig> Base;
        typedef typename Base::VVector VVector;
        SPAISolver(AMG_Config &cfg, const std::string &cfg_scope, int level) : Base(cfg, cfg_scope, level) {}
    private:
        void computeSPAI0(const Matrix<TConfig> &A);
        void computeSPAI1(const Matrix<TConfig> &A);
        void smoothSPAI0(const Matrix<TConfig> &A, VVector &b, VVector &x, bool xIsZero);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class SPAISolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public SPAISolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef SPAISolver_Base<TConfig> Base;
        typedef typename Base::VVector VVector;
        SPAISolver(AMG_Config &cfg, const std::string &cfg_scope, int level) : Base(cfg, cfg_scope, level) {}
    private:
        void computeSPAI0(const Matrix<TConfig> &A);
        void computeSPAI1(const Matrix<TConfig> &A);
        void smoothSPAI0(const Matrix<TConfig> &A, VVector &b, VVector &x, bool xIsZero);
};

template<class T_Config>
class SPAI0SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new SPAISolver<T_Config>(cfg, cfg_scope, 0); }
};

template<class T_Config>
class SPAI1SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new SPAISolver<T_Config>(cfg, cfg_scope, 1); }
};

} // namespace amgx
