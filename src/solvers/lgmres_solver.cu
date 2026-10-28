// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// LGMRES adapted from AMGCL (amgcl/solver/lgmres.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).
// A. H. Baker, E. R. Jessup, T. Manteuffel, "A technique for accelerating the convergence of
// restarted GMRES", SIAM J. Matrix Anal. Appl. 26(4), 2005.

#include <solvers/lgmres_solver.h>
#include <solvers/givens_rotation.h>
#include <blas.h>
#include <norm.h>
#include <amgx_types/util.h>
#include <amgx_types/math.h>

namespace amgx
{

template<class T_Config>
LGMRES_Solver<T_Config>::LGMRES_Solver(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope), m_num_aug_cycle(0), m_preconditioner(NULL)
{
    std::string solverName, new_scope;
    cfg.getParameter<std::string>("preconditioner", solverName, cfg_scope, new_scope);

    if (solverName.compare("NOSOLVER") != 0)
    {
        m_preconditioner = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "preconditioner");
    }

    // Total basis size: gmres_n_restart Krylov columns plus K augmentation columns (AMGCL's M + K)
    m_K = cfg.AMG_Config::template getParameter<int>("lgmres_augmentation_k", cfg_scope);
    m_M = cfg.AMG_Config::template getParameter<int>("gmres_n_restart", cfg_scope) + m_K;

    if (this->m_norm_type != L2)
    {
        FatalError("LGMRES only works with L2 norm", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    m_H.resize(m_M + 1, m_M);
    m_s.resize(m_M + 1);
    m_cs.resize(m_M);
    m_sn.resize(m_M);
}

template<class T_Config>
LGMRES_Solver<T_Config>::~LGMRES_Solver()
{
    delete m_preconditioner;
}

template<class T_Config>
void LGMRES_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "gmres_n_restart=" << m_M - m_K << ", lgmres_augmentation_k=" << m_K << std::endl;

    if (m_preconditioner != NULL)
    {
        std::cout << "preconditioner: " << m_preconditioner->getName() << " with scope name: "
                  << m_preconditioner->getScope() << std::endl;
    }
}

template<class T_Config>
void LGMRES_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    if (this->m_A->get_block_dimy() != 1 && !this->m_use_scalar_norm)
    {
        FatalError("LGMRES solver only works on block matrix if configuration parameter use_scalar_norm=1", AMGX_ERR_NOT_SUPPORTED_TARGET);
    }

    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    const int N = static_cast<int>(this->m_A->get_num_cols() * this->m_A->get_block_dimy());
    m_V.resize(m_M + 1);
    m_aug.resize(m_K);
    std::vector<VVector *> vectors;

    for (int i = 0; i <= m_M; i++) { vectors.push_back(&m_V[i]); }

    for (int i = 0; i < m_K; i++) { vectors.push_back(&m_aug[i]); }

    vectors.push_back(&m_Z);
    vectors.push_back(&m_dx);

    for (size_t k = 0; k < vectors.size(); k++)
    {
        vectors[k]->resize(N);
        vectors[k]->set_block_dimy(this->m_A->get_block_dimy());
        vectors[k]->set_block_dimx(1);
        vectors[k]->dirtybit = 1;
        vectors[k]->delayed_send = 1;
        vectors[k]->tag = this->tag * 100 + (int) k;
    }

    if (m_preconditioner != NULL)
    {
        m_preconditioner->setup(*this->m_A, reuse_matrix_structure);
    }

    this->m_A->setView(oldView);
}

template<class T_Config>
typename LGMRES_Solver<T_Config>::VVector &LGMRES_Solver<T_Config>::column(int j)
{
    const int first_aug = m_M - m_num_aug_cycle;
    return j < first_aug ? m_V[j] : m_aug[m_aug_order[j - first_aug]];
}

template<class T_Config>
void LGMRES_Solver<T_Config>::applyOperator(VVector &src, VVector &dst)
{
    Operator<T_Config> &A = *this->m_A;

    if (m_preconditioner == NULL)
    {
        int offset, size;
        A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
        copy(src, m_Z, offset, size);
    }
    else
    {
        src.delayed_send = 1;
        m_Z.delayed_send = 1;
        m_preconditioner->solve(src, m_Z, true);
        src.delayed_send = 1;
        m_Z.delayed_send = 1;
    }

    A.apply(m_Z, dst);
}

template<class T_Config>
void LGMRES_Solver<T_Config>::solve_init(VVector &b, VVector &x, bool xIsZero)
{
    // A new solve starts without augmentation vectors (AMGCL always_reset)
    m_aug_order.clear();
    m_num_aug_cycle = 0;
}

template<class T_Config>
AMGX_STATUS LGMRES_Solver<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    typedef types::util<ValueTypeB> util;
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    const int j = this->m_curr_iter % m_M;
    AMGX_STATUS conv_stat = AMGX_ST_NOT_CONVERGED;

    if (j == 0)
    {
        // Restart: V_0 = (b - A x) / beta, s = beta e_1
        axmb(A, x, b, m_V[0], offset, size);
        const PodTypeB beta = get_norm(A, m_V[0], L2);

        if (this->m_monitor_convergence)
        {
            this->m_nrm[0] = beta;
            conv_stat = this->converged();

            if (isDone(conv_stat))
            {
                A.setView(oldView);
                return conv_stat;
            }
        }

        // axmb computes b - A x
        scal(m_V[0], util::get_one() / (util::get_one() * beta), offset, size);

        for (int k = 0; k <= m_M; k++) { m_s[k] = util::get_zero(); }

        m_s[0] = util::get_one() * beta;
        m_num_aug_cycle = (int) m_aug_order.size();
    }

    // Arnoldi step on column j of W
    applyOperator(column(j), m_V[j + 1]);

    for (int k = 0; k <= j; k++)
    {
        m_H(k, j) = dot(A, m_V[k], m_V[j + 1]);
        axpy(m_V[k], m_V[j + 1], util::invert(m_H(k, j)), offset, size);
    }

    const PodTypeB h = get_norm(A, m_V[j + 1], L2);
    m_H(j + 1, j) = util::get_one() * h;

    if (h != PodTypeB(0))
    {
        scal(m_V[j + 1], util::get_one() / m_H(j + 1, j), offset, size);
    }

    PlaneRotation(m_H, m_cs, m_sn, m_s, j);

    if (this->m_monitor_convergence)
    {
        this->m_nrm[0] = util::abs(m_s[j + 1]);
        conv_stat = this->converged();
    }

    // End of the cycle: x += P^{-1} dx with dx = W y, keep dx / ||dx|| for the next cycles
    if (j == m_M - 1 || this->is_last_iter() || isDone(conv_stat))
    {
        for (int i = j; i >= 0; i--)
        {
            m_s[i] = m_s[i] / m_H(i, i);

            for (int k = i - 1; k >= 0; k--) { m_s[k] = m_s[k] - m_H(k, i) * m_s[i]; }
        }

        fill(m_dx, util::get_zero());

        for (int i = 0; i <= j; i++) { axpy(column(i), m_dx, m_s[i], offset, size); }

        if (m_preconditioner == NULL)
        {
            axpy(m_dx, x, util::get_one(), offset, size);
        }
        else
        {
            m_dx.delayed_send = 1;
            m_Z.delayed_send = 1;
            m_preconditioner->solve(m_dx, m_Z, true);
            axpy(m_Z, x, util::get_one(), offset, size);
        }

        x.dirtybit = 1;
        const PodTypeB norm_dx = get_norm(A, m_dx, L2);

        if (m_K > 0 && norm_dx != PodTypeB(0))
        {
            // Overwrite the oldest slot once all K are in use
            int slot;

            if ((int) m_aug_order.size() < m_K)
            {
                slot = (int) m_aug_order.size();
            }
            else
            {
                slot = m_aug_order.front();
                m_aug_order.erase(m_aug_order.begin());
            }

            copy(m_dx, m_aug[slot], offset, size);
            scal(m_aug[slot], util::get_one() / (util::get_one() * norm_dx), offset, size);
            m_aug_order.push_back(slot);
        }
    }

    A.setView(oldView);
    return this->m_monitor_convergence ? conv_stat : AMGX_ST_CONVERGED;
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class LGMRES_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
