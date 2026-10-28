// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// ILUT and ILUP smoothers modelled on AMGCL (amgcl/relaxation/ilut.hpp, amgcl/relaxation/ilup.hpp),
// Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License
// (see LICENSES/MIT.txt).

#include <solvers/ilu_smoothers.h>
#include <amgx_timer.h>
#include <aggregation/pointwise.h>
#include <basic_types.h>
#include <blas.h>
#include <cutil.h>
#include <multiply.h>
#include <util.h>

namespace amgx
{

template<class T_Config>
ParILUSolver_Base<T_Config>::ParILUSolver_Base(AMG_Config &cfg, const std::string &cfg_scope, Variant variant)
    : Solver<T_Config>(cfg, cfg_scope), m_explicit_A(NULL), m_variant(variant)
{
    m_weight = cfg.AMG_Config::template getParameter<double>("relaxation_factor", cfg_scope);

    if (m_weight == 0)
    {
        m_weight = 1.;
    }

    m_drop_tol = cfg.AMG_Config::template getParameter<double>("ilut_drop_tol", cfg_scope);
    m_fill_factor = cfg.AMG_Config::template getParameter<double>("ilut_fill_factor", cfg_scope);
    m_fill_level = cfg.AMG_Config::template getParameter<int>("ilup_fill_level", cfg_scope);
    m_sweeps = cfg.AMG_Config::template getParameter<int>("ilu_factor_sweeps", cfg_scope);

    // Variant defaults (decision 14): ParILU sweeps are cheap and hard problems need many; ParILUT
    // sweeps cost more and saturate earlier
    if (m_sweeps < 1) { m_sweeps = variant == ILUP ? 30 : 10; }

    m_solve_method = cfg.AMG_Config::template getParameter<std::string>("ilu_solve_method", cfg_scope) == "CUSPARSE" ? 1 : 0;
    m_solve_iters = cfg.AMG_Config::template getParameter<int>("ilu_solve_iters", cfg_scope);
    m_solve_damping = cfg.AMG_Config::template getParameter<double>("ilu_solve_damping", cfg_scope);
}

template<class T_Config>
void ParILUSolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!m_explicit_A)
    {
        FatalError("ILUT/ILUP smoothers only work with explicit matrices", AMGX_ERR_INTERNAL);
    }

    const Matrix<T_Config> &A = *m_explicit_A;

    if (A.get_block_size() != 1)
    {
        FatalError("ILUT/ILUP smoothers only support scalar matrices", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (!A.is_matrix_singleGPU())
    {
        FatalError("ILUT/ILUP smoothers are not implemented for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // ILUP keeps its pattern and refactors with new values when A's pattern is unchanged; the ILUT
    // pattern depends on the values and is rebuilt.
    factorize(A);
    m_r.resize(A.get_num_rows());
    m_z.resize(A.get_num_rows());
}

template<class T_Config>
AMGX_STATUS ParILUSolver_Base<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    if (xIsZero) { x.dirtybit = 0; }

    Matrix<T_Config> &A = *m_explicit_A;
    const int n = A.get_num_rows();

    if (xIsZero)
    {
        // x = w (LU)^{-1} b
        applyFactors(b, m_z);
        axpby(m_z, m_z, x, m_weight, ValueTypeB(0), 0, n);
    }
    else
    {
        // x += w (LU)^{-1} (b - A x)
        multiply(A, x, m_z);
        axpby(b, m_z, m_r, ValueTypeB(1), ValueTypeB(-1), 0, n);
        applyFactors(m_r, m_z);
        axpy(m_z, x, m_weight, 0, n);
    }

    x.dirtybit = 1;
    return this->converged(b, x);
}

// --------------------------- Host specialization ------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ParILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::factorize(const Matrix<TConfig> &A)
{
    FatalError("ILUT/ILUP smoothers are not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ParILUSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::applyFactors(VVector &r, VVector &z)
{
    FatalError("ILUT/ILUP smoothers are not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

// --------------------------- Device specialization ------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ParILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::factorize(const Matrix<TConfig> &A)
{
    // Plain scalar CSR without external diagonal
    Matrix<TConfig> A_expanded;

    if (A.hasProps(DIAG))
    {
        aggregation::PointwiseAggregation<TConfig>::blockToScalar(A, A_expanded);
    }

    const Matrix<TConfig> &As = A.hasProps(DIAG) ? A_expanded : A;

    nvtxRange *factor_nr = new nvtxRange("ilu_factorize");
    // ILUP on the pattern of the previous setup (e.g. after replace_coefficients): values only
    const bool refactor = this->m_variant == Base::ILUP && this->m_factor.samePattern(As);

    if (refactor)
    {
        this->m_factor.refactorPower(As, this->m_sweeps, this->m_L, this->m_U);
    }
    else if (this->m_variant == Base::ILUP)
    {
        this->m_factor.factorPower(As, this->m_fill_level, this->m_sweeps, this->m_L, this->m_U);
    }
    else
    {
        this->m_factor.factorThreshold(As, this->m_drop_tol, this->m_fill_factor, this->m_sweeps, this->m_L, this->m_U);
    }

    delete factor_nr;
    nvtxRange solve_nr("ilu_solve_setup");
    const typename ILUTriangularSolve<TConfig>::Method method =
        this->m_solve_method == 1 ? ILUTriangularSolve<TConfig>::CUSPARSE : ILUTriangularSolve<TConfig>::JACOBI;

    if (!(refactor && m_solve.refresh(this->m_L, this->m_U)))
    {
        m_solve.setup(this->m_L, this->m_U, method, this->m_solve_iters, this->m_solve_damping);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ParILUSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::applyFactors(VVector &r, VVector &z)
{
    m_solve.apply(r, z);
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class ParILUSolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class ParILUSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
