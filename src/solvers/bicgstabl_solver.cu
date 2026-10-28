// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// BiCGStab(L) adapted from AMGCL (amgcl/solver/bicgstabl.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).
// G. L. G. Sleijpen, D. R. Fokkema, "BiCGstab(l) for linear equations involving unsymmetric
// matrices with complex spectrum", ETNA 1, 1993.

#include <solvers/bicgstabl_solver.h>
#include <blas.h>
#include <amgx_types/util.h>
#include <amgx_types/math.h>

namespace amgx
{

namespace
{

// Gaussian elimination with partial pivoting on a small dense row-major system; false if singular
template <typename ValueType>
bool solveSmallDense(int n, std::vector<ValueType> &A, std::vector<ValueType> &b)
{
    typedef types::util<ValueType> util;

    for (int c = 0; c < n; c++)
    {
        int p = c;

        for (int r = c + 1; r < n; r++)
            if (util::abs(A[r * n + c]) > util::abs(A[p * n + c])) { p = r; }

        if (util::is_zero(A[p * n + c])) { return false; }

        if (p != c)
        {
            for (int k = 0; k < n; k++) { std::swap(A[p * n + k], A[c * n + k]); }

            std::swap(b[p], b[c]);
        }

        for (int r = c + 1; r < n; r++)
        {
            const ValueType f = A[r * n + c] / A[c * n + c];

            for (int k = c; k < n; k++) { A[r * n + k] = A[r * n + k] - f * A[c * n + k]; }

            b[r] = b[r] - f * b[c];
        }
    }

    for (int r = n - 1; r >= 0; r--)
    {
        ValueType s = b[r];

        for (int k = r + 1; k < n; k++) { s = s - A[r * n + k] * b[k]; }

        b[r] = s / A[r * n + r];
    }

    return true;
}

} // anonymous namespace

template<class T_Config>
BiCGStabL_Solver<T_Config>::BiCGStabL_Solver(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope), m_preconditioner(NULL)
{
    m_L = cfg.AMG_Config::template getParameter<int>("bicgstab_l", cfg_scope);
    std::string solverName, new_scope;
    cfg.getParameter<std::string>("preconditioner", solverName, cfg_scope, new_scope);

    if (solverName.compare("NOSOLVER") != 0)
    {
        m_preconditioner = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "preconditioner");
    }
}

template<class T_Config>
BiCGStabL_Solver<T_Config>::~BiCGStabL_Solver()
{
    delete m_preconditioner;
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "bicgstab_l: " << m_L << std::endl;

    if (m_preconditioner != NULL)
    {
        std::cout << "preconditioner: " << m_preconditioner->getName() << " with scope name: "
                  << m_preconditioner->getScope() << std::endl;
    }
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    const int N = static_cast<int>(this->m_A->get_num_cols() * this->m_A->get_block_dimy());
    m_R.resize(m_L + 1);
    m_U.resize(m_L + 1);
    std::vector<VVector *> vectors;

    for (int i = 0; i <= m_L; i++)
    {
        vectors.push_back(&m_R[i]);
        vectors.push_back(&m_U[i]);
    }

    vectors.push_back(&m_Rt);
    vectors.push_back(&m_X);
    vectors.push_back(&m_T);

    for (size_t k = 0; k < vectors.size(); k++)
    {
        vectors[k]->resize(N);
        vectors[k]->set_block_dimy(this->m_A->get_block_dimy());
        vectors[k]->set_block_dimx(1);
        vectors[k]->dirtybit = 1;
        vectors[k]->delayed_send = 1;
        vectors[k]->tag = this->tag * 100 + 3 + (int) k;
    }

    m_G.resize((m_L + 1) * (m_L + 1));
    m_y.resize(m_L + 1);

    if (m_preconditioner != NULL)
    {
        m_preconditioner->setup(*this->m_A, reuse_matrix_structure);
    }

    this->m_A->setView(oldView);
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::applyOperator(VVector &src, VVector &dst)
{
    Operator<T_Config> &A = *this->m_A;

    if (m_preconditioner == NULL)
    {
        int offset, size;
        A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
        copy(src, m_T, offset, size);
    }
    else
    {
        src.delayed_send = 1;
        m_T.delayed_send = 1;
        m_preconditioner->solve(src, m_T, true);
        src.delayed_send = 1;
        m_T.delayed_send = 1;
    }

    A.apply(m_T, dst);
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::solve_init(VVector &b, VVector &x, bool xIsZero)
{
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    copy(*this->m_r, m_R[0], offset, size);
    fill(m_X, types::util<ValueTypeB>::get_zero());
    restart();
    A.setView(oldView);
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::restart()
{
    Operator<T_Config> &A = *this->m_A;
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    copy(m_R[0], m_Rt, offset, size);
    fill(m_U[0], types::util<ValueTypeB>::get_zero());
    m_alpha = types::util<ValueTypeB>::get_zero();
    m_rho0 = types::util<ValueTypeB>::get_one();
    m_omega = types::util<ValueTypeB>::get_one();
    m_step = 0;
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::updateSolution(VVector &x)
{
    Operator<T_Config> &A = *this->m_A;
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);

    if (m_preconditioner == NULL)
    {
        axpy(m_X, x, types::util<ValueTypeB>::get_one(), offset, size);
    }
    else
    {
        m_X.delayed_send = 1;
        m_T.delayed_send = 1;
        m_preconditioner->solve(m_X, m_T, true);
        axpy(m_T, x, types::util<ValueTypeB>::get_one(), offset, size);
    }

    fill(m_X, types::util<ValueTypeB>::get_zero());
    x.dirtybit = 1;
}

template<class T_Config>
AMGX_STATUS BiCGStabL_Solver<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    typedef types::util<ValueTypeB> util;
    const ValueTypeB one = util::get_one();
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    const int j = m_step;

    // BiCG part, step j
    if (j == 0) { m_rho0 = util::invert(m_omega) * m_rho0; }

    const ValueTypeB rho1 = dot(A, m_Rt, m_R[j]);

    if (util::is_zero(rho1) || util::is_zero(m_rho0))
    {
        A.setView(oldView);
        return AMGX_ST_DIVERGED;
    }

    const ValueTypeB beta = m_alpha * (rho1 / m_rho0);
    m_rho0 = rho1;

    for (int i = 0; i <= j; i++)
    {
        axpby(m_R[i], m_U[i], m_U[i], one, util::invert(beta), offset, size);
    }

    applyOperator(m_U[j], m_U[j + 1]);
    const ValueTypeB sigma = dot(A, m_Rt, m_U[j + 1]);

    if (util::is_zero(sigma))
    {
        A.setView(oldView);
        return AMGX_ST_DIVERGED;
    }

    m_alpha = rho1 / sigma;
    axpy(m_U[0], m_X, m_alpha, offset, size);

    for (int i = 0; i <= j; i++)
    {
        axpy(m_U[i + 1], m_R[i], util::invert(m_alpha), offset, size);
    }

    applyOperator(m_R[j], m_R[j + 1]);

    // Minimal residual polynomial: min || R_0 - sum_k y_k R_k ||, normal equations <R_i, R_k> y = <R_i, R_0>
    if (j == m_L - 1)
    {
        const int L = m_L;

        for (int i = 1; i <= L; i++)
        {
            for (int k = i; k <= L; k++)
            {
                const ValueTypeB g = dot(A, m_R[i], m_R[k]);
                m_G[(i - 1) * L + (k - 1)] = g;
                m_G[(k - 1) * L + (i - 1)] = util::conjugate(g);
            }

            m_y[i - 1] = dot(A, m_R[i], m_R[0]);
        }

        std::vector<ValueTypeB> G(m_G.begin(), m_G.begin() + L * L), y(m_y.begin(), m_y.begin() + L);

        if (!solveSmallDense(L, G, y))
        {
            A.setView(oldView);
            return AMGX_ST_DIVERGED;
        }

        m_omega = y[L - 1];

        for (int h = L - 1; h > 0 && util::is_zero(m_omega); h--) { m_omega = y[h - 1]; }

        if (util::is_zero(m_omega))
        {
            A.setView(oldView);
            return AMGX_ST_DIVERGED;
        }

        for (int k = 1; k <= L; k++)
        {
            axpy(m_R[k - 1], m_X, y[k - 1], offset, size);
            axpy(m_U[k], m_U[0], util::invert(y[k - 1]), offset, size);
            axpy(m_R[k], m_R[0], util::invert(y[k - 1]), offset, size);
        }
    }

    m_step = (j + 1) % m_L;
    AMGX_STATUS conv_stat = AMGX_ST_NOT_CONVERGED;

    if (this->m_monitor_convergence)
    {
        // Recursive residual of the preconditioned correction, as AMGCL
        this->compute_norm(m_R[0], this->m_nrm);
        conv_stat = this->converged();

        // In finite precision the recursive residual can drift away from b - A x: confirm with the
        // true residual and restart the recurrences from it if needed (residual replacement)
        if (isDone(conv_stat) && conv_stat == AMGX_ST_CONVERGED)
        {
            updateSolution(x);
            axmb(A, x, b, m_R[0], offset, size);
            this->compute_norm(m_R[0], this->m_nrm);
            conv_stat = this->converged();

            if (!isDone(conv_stat)) { restart(); }
        }
    }

    A.setView(oldView);
    return this->m_monitor_convergence ? conv_stat : AMGX_ST_CONVERGED;
}

template<class T_Config>
void BiCGStabL_Solver<T_Config>::solve_finalize(VVector &b, VVector &x)
{
    Operator<T_Config> &A = *this->m_A;
    ViewType oldView = A.currentView();
    A.setViewExterior();
    updateSolution(x);
    A.setView(oldView);
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class BiCGStabL_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
