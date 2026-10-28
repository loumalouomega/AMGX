// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Deflated preconditioner adapted from AMGCL (amgcl/deflated_solver.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <solvers/deflation_solver.h>
#include <composite/submatrix.h>
#include <amgx_cusolverDn.h>
#include <blas.h>
#include <multiply.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>

namespace amgx
{

template<class T_Config>
Deflation_Solver<T_Config>::Deflation_Solver(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope), m_n(0), m_k(0), m_cusolver(NULL), m_preconditioner(NULL)
{
    m_use_nullspace = cfg.AMG_Config::template getParameter<int>("deflation_use_nullspace", cfg_scope) != 0;
    std::string name, new_scope;
    cfg.getParameter<std::string>("preconditioner", name, cfg_scope, new_scope);

    if (name != "NOSOLVER")
    {
        m_preconditioner = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "preconditioner");
    }
}

template<class T_Config>
Deflation_Solver<T_Config>::~Deflation_Solver()
{
    delete m_preconditioner;

    if (m_cusolver != NULL)
    {
        cusolverDnDestroy(m_cusolver);
    }
}

template<class T_Config>
void Deflation_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "deflation vectors: " << m_k << (m_use_nullspace ? " (near-nullspace)" : "") << std::endl;

    if (m_preconditioner != NULL)
    {
        std::cout << "preconditioner: " << m_preconditioner->getName() << " with scope name: " << m_preconditioner->getScope() << std::endl;
    }
}

template<class T_Config>
void Deflation_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    composite::SubmatrixOps<TConfig>::requireDevice("DEFLATION");
    Matrix<TConfig> *A = dynamic_cast<Matrix<TConfig>*>(this->m_A);

    if (A == NULL)
    {
        FatalError("DEFLATION only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (!A->is_matrix_singleGPU())
    {
        FatalError("DEFLATION is not implemented for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    m_n = A->get_num_rows() * A->get_block_dimy();
    // Deflation vectors, column-major n x k
    std::vector<double> Z_h;

    if (m_use_nullspace)
    {
        m_k = A->m_nullspace_cols;
        Z_h.resize((size_t) m_n * m_k);

        for (int j = 0; j < m_k; j++)
            for (int i = 0; i < m_n; i++) { Z_h[(size_t) j * m_n + i] = A->m_nullspace[(size_t) i * m_k + j]; }
    }
    else
    {
        m_k = A->m_deflation_cols;
        Z_h = A->m_deflation;
    }

    if (m_k < 1 || Z_h.size() != (size_t) m_n * m_k)
    {
        FatalError("DEFLATION: set the deflation vectors with AMGX_matrix_set_deflation_vectors (or AMGX_matrix_set_nullspace and deflation_use_nullspace=1) after uploading the matrix", AMGX_ERR_BAD_PARAMETERS);
    }

    m_Z.resize((size_t) m_n * m_k);
    amgx::thrust::copy(Z_h.begin(), Z_h.end(), m_Z.begin());
    // Work vectors (sized once; the apply path does not allocate)
    const int n_ext = A->get_num_cols() * A->get_block_dimy();
    VVector *full[] = { &m_r, &m_t };

    for (int k = 0; k < 2; k++)
    {
        full[k]->resize(n_ext);
        full[k]->set_block_dimy(A->get_block_dimy());
        full[k]->set_block_dimx(1);
        full[k]->dirtybit = 1;
        full[k]->delayed_send = 1;
        full[k]->tag = this->tag * 100 + 3 + k;
    }

    m_f.resize(m_k);
    m_E.resize((size_t) m_k * m_k);
    // E = Z^T A Z, column by column: E(:, j) = Z^T (A Z(:, j))
    const ValueTypeB one = types::util<ValueTypeB>::get_one();
    const ValueTypeB zero = types::util<ValueTypeB>::get_zero();

    for (int j = 0; j < m_k; j++)
    {
        thrust_wrapper::copy<TConfig::memSpace>(m_Z.begin() + (size_t) j * m_n, m_Z.begin() + (size_t)(j + 1) * m_n, m_t.begin());
        m_t.dirtybit = 1;
        multiply(*A, m_t, m_r);
        gemv_extnd(true, m_Z, m_r, m_E, m_n, m_k, one, zero, 1, 1, m_n, 0, 0, j * m_k);
    }

    // LU factorization of E on the device
    if (m_cusolver == NULL)
    {
        if (cusolverDnCreate(&m_cusolver) != CUSOLVER_STATUS_SUCCESS)
        {
            FatalError("DEFLATION: could not create the cuSOLVER handle", AMGX_ERR_CUDA_FAILURE);
        }
    }

    if (cusolverDnSetStream(m_cusolver, amgx::thrust::global_thread_handle::get_stream()) != CUSOLVER_STATUS_SUCCESS)
    {
        FatalError("DEFLATION: could not set the cuSOLVER stream", AMGX_ERR_CUDA_FAILURE);
    }

    int lwork = 0;
    cusolverDnXgetrf_bufferSize(m_cusolver, m_k, m_k, m_E.raw(), m_k, &lwork);
    VVector work(std::max(lwork, 1));
    m_ipiv.resize(m_k);
    m_info.resize(1);

    if (cusolverDnXgetrf(m_cusolver, m_k, m_k, m_E.raw(), m_k, work.raw(), m_ipiv.raw(), m_info.raw()) != CUSOLVER_STATUS_SUCCESS)
    {
        FatalError("DEFLATION: LU factorization of Z^T A Z failed", AMGX_ERR_CUDA_FAILURE);
    }

    const int info = m_info[0];

    if (info != 0)
    {
        FatalError("DEFLATION: Z^T A Z is singular (the deflation vectors are linearly dependent or span a null space of A)", AMGX_ERR_BAD_PARAMETERS);
    }

    if (m_preconditioner != NULL)
    {
        m_preconditioner->setup(*A, reuse_matrix_structure);
    }
}

template<class T_Config>
AMGX_STATUS Deflation_Solver<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    Matrix<TConfig> &A = *dynamic_cast<Matrix<TConfig>*>(this->m_A);
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    const ValueTypeB one = types::util<ValueTypeB>::get_one();
    const ValueTypeB zero = types::util<ValueTypeB>::get_zero();

    // x += M (b - A x)
    if (xIsZero)
    {
        if (m_preconditioner != NULL) { m_preconditioner->solve(b, x, true); }
        else { fill(x, zero); }
    }
    else if (m_preconditioner != NULL)
    {
        axmb(A, x, b, m_r, offset, size);
        m_preconditioner->solve(m_r, m_t, true);
        axpy(m_t, x, one, offset, size);
    }

    // x += Z E^{-1} Z^T (b - A x)
    axmb(A, x, b, m_r, offset, size);
    gemv_extnd(true, m_Z, m_r, m_f, m_n, m_k, one, zero, 1, 1, m_n, 0, 0, 0);
    cusolverDnXgetrs(m_cusolver, CUBLAS_OP_N, m_k, 1, m_E.raw(), m_k, m_ipiv.raw(), m_f.raw(), m_k, m_info.raw());
    gemv_extnd(false, m_Z, m_f, x, m_n, m_k, one, one, 1, 1, m_n, 0, 0, 0);
    x.dirtybit = 1;
    A.setView(oldView);
    return this->converged(b, x);
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Deflation_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
