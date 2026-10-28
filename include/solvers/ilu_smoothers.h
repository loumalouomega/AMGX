// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// ILUT and ILUP smoothers modelled on AMGCL (amgcl/relaxation/ilut.hpp, amgcl/relaxation/ilup.hpp),
// Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License
// (see LICENSES/MIT.txt).

#pragma once

#include <solvers/solver.h>
#include <solvers/parilu.h>

namespace amgx
{

template <class T_Config> class ParILUSolver;

/***************************************************
 * Incomplete LU smoothers, x += w (LU)^{-1} (b - A x)
 *  ILUP: ILU on the pattern of A^(ilup_fill_level+1)
 *  ILUT: threshold ILU (ilut_drop_tol, ilut_fill_factor)
 * Factors are computed on the GPU with ilu_factor_sweeps parallel sweeps, triangular
 * solves use ilu_solve_method (JACOBI or CUSPARSE). Scalar matrices only.
 **************************************************/
template<class T_Config>
class ParILUSolver_Base : public Solver<T_Config>
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

        enum Variant { ILUT, ILUP };

        ParILUSolver_Base(AMG_Config &cfg, const std::string &cfg_scope, Variant variant);
        ~ParILUSolver_Base() {}

        bool is_residual_needed() const { return false; }
        void solver_setup(bool reuse_matrix_structure);
        bool isColoringNeeded() const { return false; }
        bool getReorderColsByColorDesired() const { return false; }
        bool getInsertDiagonalDesired() const { return false; }
        void solve_init(VVector &b, VVector &x, bool xIsZero) {}
        AMGX_STATUS solve_iteration(VVector &b, VVector &x, bool xIsZero);
        void solve_finalize(VVector &b, VVector &x) {}

        // Read access for tests
        const Matrix<TConfig> &getL() const { return m_L; }
        const Matrix<TConfig> &getU() const { return m_U; }

    protected:
        virtual void factorize(const Matrix<TConfig> &A) = 0;
        virtual void applyFactors(VVector &r, VVector &z) = 0;

        Matrix<TConfig> *m_explicit_A;
        Variant m_variant;
        ValueTypeB m_weight;
        double m_drop_tol;
        double m_fill_factor;
        int m_fill_level;
        int m_sweeps;
        int m_solve_method;
        int m_solve_iters;
        double m_solve_damping;
        Matrix<TConfig> m_L, m_U;
        ParILUFactor<TConfig> m_factor;   // kept between setups: ILUP refactors on an unchanged pattern
        VVector m_r, m_z;
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class ParILUSolver< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public ParILUSolver_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef ParILUSolver_Base<TConfig> Base;
        typedef typename Base::VVector VVector;
        ParILUSolver(AMG_Config &cfg, const std::string &cfg_scope, typename Base::Variant variant) : Base(cfg, cfg_scope, variant) {}
    private:
        void factorize(const Matrix<TConfig> &A);
        void applyFactors(VVector &r, VVector &z);
};

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class ParILUSolver< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public ParILUSolver_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef ParILUSolver_Base<TConfig> Base;
        typedef typename Base::VVector VVector;
        ParILUSolver(AMG_Config &cfg, const std::string &cfg_scope, typename Base::Variant variant) : Base(cfg, cfg_scope, variant) {}
    private:
        void factorize(const Matrix<TConfig> &A);
        void applyFactors(VVector &r, VVector &z);
        ILUTriangularSolve<TConfig> m_solve;
};

template<class T_Config>
class ILUTSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new ParILUSolver<T_Config>(cfg, cfg_scope, ParILUSolver_Base<T_Config>::ILUT); }
};

template<class T_Config>
class ILUPSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng) { return new ParILUSolver<T_Config>(cfg, cfg_scope, ParILUSolver_Base<T_Config>::ILUP); }
};

} // namespace amgx
