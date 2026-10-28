// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoothed aggregation coarsening adapted from AMGCL
// (amgcl/coarsening/smoothed_aggregation.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <aggregation/sa_amg_level.h>
#include <aggregation/strength/symmetric_scaled.h>
#include <aggregation/selectors/mis2_aggregates.h>
#include <aggregation/sa_prolongation.h>
#include <amg.h>
#include <blas.h>
#include <csr_multiply.h>
#include <multiply.h>
#include <transpose.h>
#include <truncate.h>
#include <thrust_wrapper.h>
#include <cutil.h>
#include <util.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace amgx
{
namespace aggregation
{

template <class T_Config>
SA_AMG_Level_Base<T_Config>::SA_AMG_Level_Base(AMG_Class *amg, ThreadManager *tmng)
    : AMG_Level<T_Config>(amg, tmng), m_num_aggregates(0), m_num_removed(0), m_num_coarse_vertices(0),
      m_level_eps(0.0), m_rho(0.0), m_omega(0.0)
{
    AMG_Config &cfg = *(amg->m_cfg);
    const std::string &scope = amg->m_cfg_scope;
    m_eps_strong = cfg.AMG_Config::template getParameter<double>("aggregation_eps_strong", scope);
    m_eps_decay = cfg.AMG_Config::template getParameter<double>("aggregation_eps_decay", scope);
    m_relax = cfg.AMG_Config::template getParameter<double>("sa_relaxation_factor", scope);
    m_power_iters = cfg.AMG_Config::template getParameter<int>("sa_spectral_radius_iters", scope);
    m_drop_tol = cfg.AMG_Config::template getParameter<double>("sa_prolongator_drop_tol", scope);
    m_print_aggregation_info = cfg.AMG_Config::template getParameter<int>("print_aggregation_info", scope) != 0;
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    Matrix<TConfig> &A = this->getA();

    if (!A.is_matrix_singleGPU())
    {
        FatalError("SMOOTHED_AGGREGATION is not implemented for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    if (A.get_block_size() != 1)
    {
        FatalError("SMOOTHED_AGGREGATION only supports scalar matrices (block_size 1)", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (A.hasProps(DIAG))
    {
        FatalError("SMOOTHED_AGGREGATION does not support matrices with external diagonal", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // eps_l = max(eps_strong * decay^l, 1e-4)
    m_level_eps = std::max(m_eps_strong * std::pow(m_eps_decay, this->getLevelIndex()), 1e-4);
    this->Profile.tic("aggregates");
    computeAggregates();

    // No strong connection at all (e.g. 27-point stencils, where |a_ij| / sqrt(a_ii a_jj) = 1/26):
    // keep decaying eps instead of ending the hierarchy on a large level.
    while (m_num_aggregates == 0 && m_level_eps > 1e-4 && m_eps_decay < 1.0)
    {
        m_level_eps = std::max(m_level_eps * m_eps_decay, 1e-4);
        computeAggregates();
    }

    this->Profile.toc("aggregates");
    // Stop coarsening when nothing can be aggregated: reporting the fine size makes the hierarchy end here.
    m_num_coarse_vertices = (m_num_aggregates > 0) ? m_num_aggregates : A.get_num_rows();
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    // As in classical AMG, P and R are kept when the level structure is reused
    if (!this->isReuseLevel())
    {
        computeProlongationOperator();
        computeRestrictionOperator();
    }

    computeAOperator();
    Matrix<TConfig> &Ac = this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA();
    Ac.copyAuxData(&this->getA());
    this->m_next_level_size = Ac.get_num_rows() * Ac.get_block_dimy();
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::computeRestrictionOperator()
{
    this->Profile.tic("computeR");
    R.set_initialized(0);
    P.setView(OWNED);
    transpose(P, R, P.get_num_rows());

    if (!R.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        R.setInteriorView(OWNED);
        R.setExteriorView(OWNED);
    }

    R.set_initialized(1);
    this->Profile.toc("computeR");
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
    multiply(R, r, rr);
    rr.dirtybit = 1;
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp)
{
    e.dirtybit = 1;

    if (e.size() > 0)
    {
        multiply(P, e, tmp);
    }

    axpby(x, tmp, x, ValueTypeB(1), ValueTypeB(1), 0, (int) x.size());
    x.dirtybit = 1;
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::printLevelInfo(int nnz_P_untruncated)
{
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename Matrix<TConfig_h>::IVector IVector_h;
    IVector_h aggregates = m_aggregates;
    std::vector<int> sizes(m_num_aggregates, 0);

    for (int i = 0; i < (int) aggregates.size(); i++)
    {
        if (aggregates[i] >= 0) { sizes[aggregates[i]]++; }
    }

    int min_size = 0, max_size = 0;

    if (m_num_aggregates > 0)
    {
        min_size = *std::min_element(sizes.begin(), sizes.end());
        max_size = *std::max_element(sizes.begin(), sizes.end());
    }

    const int n = this->getA().get_num_rows();
    std::stringstream ss;
    ss << "SA level " << this->getLevelIndex() << ": rows " << n
       << ", aggregates " << m_num_aggregates
       << " (size min " << min_size << ", avg " << (m_num_aggregates > 0 ? double(n - m_num_removed) / m_num_aggregates : 0.0)
       << ", max " << max_size << "), removed " << m_num_removed
       << ", eps_strong " << m_level_eps << ", rho " << m_rho << ", omega " << m_omega
       << ", nnz(P) " << nnz_P_untruncated << " -> " << P.get_num_nz() << std::endl;
    amgx_output(ss.str().c_str(), ss.str().length());
}

// --------------------------- Host specialization ------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAggregates()
{
    FatalError("SMOOTHED_AGGREGATION setup is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeProlongationOperator()
{
    FatalError("SMOOTHED_AGGREGATION setup is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator()
{
    FatalError("SMOOTHED_AGGREGATION setup is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

// --------------------------- Device specialization ------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAggregates()
{
    Matrix<TConfig> &A = this->getA();
    SymmetricScaledStrength<TConfig>::computeStrongConnections(A, this->m_level_eps, this->m_strong, this->m_filtered_diag);
    MIS2Aggregates<TConfig>::compute(A, this->m_strong, this->m_aggregates, this->m_num_aggregates, this->m_num_removed);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeProlongationOperator()
{
    this->Profile.tic("computeP");
    Matrix<TConfig> &A = this->getA();
    void *wk = this->amg->getCsrWorkspace();

    if (wk == NULL)
    {
        wk = CSR_Multiply<TConfig>::csr_workspace_create(*(this->amg->m_cfg), this->amg->m_cfg_scope);
        this->amg->setCsrWorkspace(wk);
    }

    // S = D_f^{-1} A^F
    Matrix<TConfig> S;
    SymmetricScaledStrength<TConfig>::buildFilteredScaledOperator(A, this->m_strong, this->m_filtered_diag, S);
    this->m_strong.clear();
    this->m_strong.shrink_to_fit();
    this->m_filtered_diag.clear();
    this->m_filtered_diag.shrink_to_fit();
    // omega = relax * 4 / (3 rho(D_f^{-1} A^F))
    this->m_rho = SAProlongation<TConfig>::estimateSpectralRadius(S, this->m_power_iters);
    this->m_omega = (this->m_rho > 0.0) ? this->m_relax * 4.0 / (3.0 * this->m_rho) : 0.0;
    // P = (I - omega S) P_tent
    Matrix<TConfig> P_tent;
    SAProlongation<TConfig>::buildTentativeProlongator(this->m_aggregates, this->m_num_aggregates, P_tent);
    SAProlongation<TConfig>::smoothProlongator(S, this->m_omega, P_tent, this->P, wk);
    const int nnz_P_untruncated = this->P.get_num_nz();

    if (this->m_drop_tol > 0.0 && this->P.get_num_rows() > 0)
    {
        Truncate<TConfig>::truncateByFactor(this->P, this->m_drop_tol, AMGX_TruncateByMaxCoefficient);
    }

    if (!this->P.isLatencyHidingEnabled(*this->amg->m_cfg))
    {
        this->P.setInteriorView(OWNED);
        this->P.setExteriorView(OWNED);
    }

    if (this->m_print_aggregation_info)
    {
        this->printLevelInfo(nnz_P_untruncated);
    }

    this->Profile.toc("computeP");
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator()
{
    this->Profile.tic("computeA");
    Matrix<TConfig> &Ac = this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA();
    void *wk = this->amg->getCsrWorkspace();

    if (wk == NULL)
    {
        wk = CSR_Multiply<TConfig>::csr_workspace_create(*(this->amg->m_cfg), this->amg->m_cfg_scope);
        this->amg->setCsrWorkspace(wk);
    }

    Ac.set_initialized(0);
    Ac.addProps(CSR);
    Ac.set_block_dimx(1);
    Ac.set_block_dimy(1);
    this->R.set_initialized(0);
    this->R.addProps(CSR);
    this->R.set_initialized(1);
    this->P.set_initialized(0);
    this->P.addProps(CSR);
    this->P.set_initialized(1);
    CSR_Multiply<TConfig>::csr_galerkin_product(this->R, this->getA(), this->P, Ac,
            NULL, NULL, NULL, NULL, NULL, NULL, wk);
    Ac.set_initialized(1);
    this->Profile.toc("computeA");
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class SA_AMG_Level_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class SA_AMG_Level<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace aggregation
} // namespace amgx
