// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amg_solver.h>
#include <amg_config.h>
#include <amg.h>
#include <basic_types.h>
#include <misc.h>
#include <assert.h>
#include <util.h>
#include <reorder/rcm.h>
#include <sstream>

using std::string;

namespace amgx
{

// Permuted copy of the user's matrix for reordering=RCM. The user's matrix is left untouched
// (AMGX_matrix_replace_coefficients writes into it in its own order), so values are re-gathered
// through nnz_map on every (re)setup.
template <class T_Config>
struct ReorderedSystem
{
    typedef reorder::RCM<T_Config> Ops;
    typedef typename Ops::IVector IVector;

    Matrix<T_Config> A;
    IVector perm, iperm, nnz_map;
    Vector<T_Config> b, x;
    const Matrix<T_Config> *source;
    int rows, nnz, bandwidth_before, bandwidth_after;
    bool active;

    ReorderedSystem() : source(NULL), rows(0), nnz(0), bandwidth_before(0), bandwidth_after(0), active(false) {}

    bool matches(const Matrix<T_Config> &user) const
    {
        return active && source == &user && rows == user.get_num_rows() && nnz == user.get_num_nz();
    }

    void build(const Matrix<T_Config> &user)
    {
        A = Matrix<T_Config>();
        A.setResources(user.getResources());
        Ops::permutation(user, perm);
        Ops::inverse(perm, iperm);
        Ops::permuteStructure(user, perm, iperm, A, nnz_map);
        source = &user;
        rows = user.get_num_rows();
        nnz = user.get_num_nz();
        bandwidth_before = Ops::bandwidth(user);
        bandwidth_after = Ops::bandwidth(A);
        const int n = rows * user.get_block_dimy();
        b.resize(n);
        x.resize(n);
        copyAttachedData(user);
        active = true;
    }

    void refresh(const Matrix<T_Config> &user)
    {
        Ops::permuteValues(user, nnz_map, A);
        copyAttachedData(user);
    }

    // Per-unknown data attached through the C API follows the permutation
    void copyAttachedData(const Matrix<T_Config> &user)
    {
        const int bdim = user.get_block_dimy(), n = rows * bdim;
        std::vector<int> p(perm.size());
        amgx::thrust::copy(perm.begin(), perm.end(), p.begin());
        A.m_nullspace_cols = user.m_nullspace_cols;
        A.m_nullspace.resize(user.m_nullspace.size());
        A.m_deflation_cols = user.m_deflation_cols;
        A.m_deflation.resize(user.m_deflation.size());
        A.m_row_markers.resize(user.m_row_markers.size());

        for (int i = 0; i < rows; i++)
            for (int c = 0; c < bdim; c++)
            {
                const size_t dst = (size_t) i * bdim + c, src = (size_t) p[i] * bdim + c;

                for (int k = 0; k < user.m_nullspace_cols && !user.m_nullspace.empty(); k++)
                {
                    A.m_nullspace[dst * user.m_nullspace_cols + k] = user.m_nullspace[src * user.m_nullspace_cols + k];
                }

                for (int k = 0; k < user.m_deflation_cols && !user.m_deflation.empty(); k++)
                {
                    A.m_deflation[(size_t) k * n + dst] = user.m_deflation[(size_t) k * n + src];
                }

                if (!user.m_row_markers.empty()) { A.m_row_markers[dst] = user.m_row_markers[src]; }
            }
    }
};

template< class T_Config >
void AMG_Solver<T_Config>::process_config(AMG_Config &in_cfg, std::string solver_scope)
{
    if (in_cfg.getParameter<int>("print_config", solver_scope) == 1)
    {
        in_cfg.printAMGConfig();
    }
}

template< class T_Config >
void AMG_Solver<T_Config>::init()
{
    std::string solver_value, solver_scope;
    m_cfg->template getParameter<std::string>("solver", solver_value, "default", solver_scope);
    process_config(*m_cfg, solver_scope);
    m_rcm = m_cfg->template getParameter<std::string>("reordering", solver_scope) == "RCM" ||
            m_cfg->template getParameter<std::string>("reordering", "default") == "RCM";
    // pass thread manager to solver
    solver = SolverFactory<T_Config>::allocate(*m_cfg, "default", "solver", m_resources->get_tmng());
    structure_reuse_levels_scope = "";

    // Reusing structure
    if (solver_value == "AMG") // AMG is used as main solver
    {
        structure_reuse_levels_scope = solver_scope;
    }
    else
    {
        // AMG might be used as preconditioner of the main solver    std::string preconditioner_value, preconditioner_scope;
        std::string preconditioner_value, preconditioner_scope;
        m_cfg->template getParameter<std::string>("preconditioner", preconditioner_value, solver_scope, preconditioner_scope);

        if (preconditioner_value == "AMG")
        {
            structure_reuse_levels_scope = preconditioner_scope;
        }
        else
        {
            structure_reuse_levels_scope = "";
        }
    }

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaCheckError();
        cudaEventCreate(&m_setup_stop);
        cudaCheckError();
        cudaEventCreate(&m_solve_start);
        cudaCheckError();
        cudaEventCreate(&m_solve_stop);
        cudaCheckError();
    }
}

template< class T_Config >
AMG_Solver<T_Config>::AMG_Solver(Resources *res, AMG_Configuration *cfg) : m_with_timings(false), m_resources(res), m_cfg_self(false), m_rcm(false)
{
    if (cfg)
    {
        m_cfg = cfg->getConfigObject();
    }
    else
    {
        m_cfg = res->getResourcesConfig();
    }

    init();
}

template< class T_Config >
AMG_Solver<T_Config>::AMG_Solver(Resources *res, AMG_Configuration &cfg) : m_with_timings(false), m_resources(res), m_cfg_self(true), m_rcm(false)
{
    m_cfg = new AMG_Config;
    *m_cfg = *(cfg.getConfigObject());
    init();
}

template< class T_Config >
AMG_Solver<T_Config>::AMG_Solver(const AMG_Solver<T_Config> &amg_solver)
{
    solver = amg_solver.solver;
    m_resources = amg_solver.getResources();
    m_cfg = amg_solver.getConfig();
    m_cfg_self = false;
    solver->incr_ref_count();
    m_ptrA = amg_solver.m_ptrA;
    m_rcm = amg_solver.m_rcm;
    m_reorder = amg_solver.m_reorder;
    m_with_timings = amg_solver.m_with_timings;

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaCheckError();
        cudaEventCreate(&m_setup_stop);
        cudaCheckError();
        cudaEventCreate(&m_solve_start);
        cudaCheckError();
        cudaEventCreate(&m_solve_stop);
        cudaCheckError();
    }
}

template< class T_Config >
AMG_Solver<T_Config> &AMG_Solver<T_Config>::operator=(const AMG_Solver<T_Config> &amg_solver)
{
    solver = amg_solver.solver;
    m_resources = amg_solver.getResources();
    m_cfg = amg_solver.getConfig();
    m_cfg_self = false;
    solver->incr_ref_count();
    m_ptrA = amg_solver.m_ptrA;
    m_rcm = amg_solver.m_rcm;
    m_reorder = amg_solver.m_reorder;
    m_with_timings = amg_solver.m_with_timings;

    if ( m_with_timings )
    {
        cudaEventCreate(&m_setup_start);
        cudaCheckError();
        cudaEventCreate(&m_setup_stop);
        cudaCheckError();
        cudaEventCreate(&m_solve_start);
        cudaCheckError();
        cudaEventCreate(&m_solve_stop);
        cudaCheckError();
    }

    return *this;
}

template< class T_Config >
AMG_Solver<T_Config>::~AMG_Solver()
{
    if (m_cfg_self)
    {
        delete m_cfg;
    }

    if ( solver->decr_ref_count() )
    {
        delete solver;

        if ( !m_with_timings )
        {
            return;
        }

        std::cerr << std::endl;
        float elapsed_time = 0.0f;
        cudaEventElapsedTime(&elapsed_time, m_setup_start, m_setup_stop);
        std::cerr << "AMG_Solver::setup time: " << 1.0e-3 * elapsed_time << "s" << std::endl;
        cudaEventElapsedTime(&elapsed_time, m_solve_start, m_solve_stop);
        std::cerr << "AMG_Solver::solve time: " << 1.0e-3 * elapsed_time << "s" << std::endl;
        cudaEventDestroy(m_setup_start);
        cudaCheckError();
        cudaEventDestroy(m_setup_stop);
        cudaCheckError();
        cudaEventDestroy(m_solve_start);
        cudaCheckError();
        cudaEventDestroy(m_solve_stop);
        cudaCheckError();
    }
}

template < class T_Config >
const typename AMG_Solver<T_Config>::PODVector_h   &AMG_Solver<T_Config>::get_residual( int res_num ) const
{
    return solver->get_residual(res_num);
}

/****************************************************
* Sets A as the matrix for the AMG system
****************************************************/
template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::setup( Matrix<T_Config> &A_user)//&A0)
{
    // With structure reuse, a repeated setup on the same matrix keeps the permutation
    Matrix<T_Config> &A = prepareReordered(A_user, getStructureReuseLevels() != 0);
    bool reuse_fine_matrix = (getStructureReuseLevels() > 0) && A.is_matrix_setup();
    bool reuse_all = (getStructureReuseLevels() == -1) && A.is_matrix_setup();

    if (reuse_all)
    {
        solver->reset_setup_timer();
        return AMGX_OK;
    }

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_start);
        cudaCheckError();
    }

    // postpone free syncs, use device pool
    memory::setAsyncFreeFlag(true);
    AMGX_ERROR e = solver->setup_no_throw(A, reuse_fine_matrix);
    m_resources->get_tmng()->wait_threads();
    amgx::thrust::global_thread_handle::joinDevicePools();
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    // free postponed objects
    amgx::thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_stop);
        cudaCheckError();
        cudaEventSynchronize(m_setup_stop);
        cudaCheckError();
    }

    return e;
}

template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::resetup( Matrix<T_Config> &A_user)//&A0 )
{
    Matrix<T_Config> &A = prepareReordered(A_user, true);

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_start);
        cudaCheckError();
    }

    // postpone free syncs, use device pool
    memory::setAsyncFreeFlag(true);
    AMGX_ERROR e = solver->setup_no_throw(A, true);
    m_resources->get_tmng()->wait_threads();
    amgx::thrust::global_thread_handle::joinDevicePools();
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    // free postponed objects
    amgx::thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_setup_stop);
        cudaCheckError();
        cudaEventSynchronize(m_setup_stop);
        cudaCheckError();
    }

    return e;
}

template< class T_Config >
Matrix<T_Config> &AMG_Solver<T_Config>::prepareReordered(Matrix<T_Config> &A, bool keep_permutation)
{
    if (!m_rcm)
    {
        return A;
    }

    if (A.manager != NULL)
    {
        std::string msg = "reordering=RCM is ignored for distributed matrices\n";
        amgx_output(msg.c_str(), msg.length());

        if (m_reorder) { m_reorder->active = false; }

        return A;
    }

    if (!m_reorder) { m_reorder.reset(new ReorderedSystem<T_Config>()); }

    ReorderedSystem<T_Config> &R = *m_reorder;

    if (keep_permutation && R.matches(A))
    {
        R.refresh(A);
    }
    else
    {
        R.build(A);

        std::string solver_value, solver_scope;
        m_cfg->template getParameter<std::string>("solver", solver_value, "default", solver_scope);

        if (m_cfg->template getParameter<int>("print_grid_stats", solver_scope) || m_cfg->template getParameter<int>("print_grid_stats", "default"))
        {
            std::stringstream ss;
            ss << "RCM reordering: bandwidth " << R.bandwidth_before << " -> " << R.bandwidth_after << std::endl;
            amgx_output(ss.str().c_str(), ss.str().length());
        }
    }

    return R.A;
}

template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::setup_capi( std::shared_ptr<Matrix<T_Config>> pA0)
{
    m_ptrA = pA0;
    return setup(*m_ptrA);
}

template< class T_Config >
AMGX_ERROR AMG_Solver<T_Config>::resetup_capi( std::shared_ptr<Matrix<T_Config>> pA0)
{
    m_ptrA = pA0;
    return resetup(*m_ptrA);
}

/****************************************************
* Solves the AMG system Ax=b
***************************************************/
template<class T_Config>
AMGX_ERROR AMG_Solver<T_Config>::solve( Vector<T_Config> &b, Vector<T_Config> &x, AMGX_STATUS &status, bool xIsZero )
{
    if ( m_with_timings )
    {
        cudaEventRecord(m_solve_start);
        cudaCheckError();
    }

    AMGX_ERROR e;

    if (m_reorder && m_reorder->active)
    {
        // Solve in the permuted numbering: b_p = b[perm], x_p = x[perm], then x[perm] = x_p
        ReorderedSystem<T_Config> &R = *m_reorder;
        const int bsize = b.get_block_dimy();
        R.b.set_block_dimx(b.get_block_dimx());
        R.b.set_block_dimy(bsize);
        R.x.set_block_dimx(x.get_block_dimx());
        R.x.set_block_dimy(bsize);
        reorder::RCM<T_Config>::gather(b, R.perm, bsize, R.b);

        if (!xIsZero) { reorder::RCM<T_Config>::gather(x, R.perm, bsize, R.x); }

        e = solver->solve_no_throw( R.b, R.x, status, xIsZero );
        reorder::RCM<T_Config>::scatter(R.x, R.perm, bsize, x);
    }
    else
    {
        e = solver->solve_no_throw( b, x, status, xIsZero );
    }

    amgx::thrust::global_thread_handle::cudaFreeWait();

    if ( m_with_timings )
    {
        cudaEventRecord(m_solve_stop);
        cudaCheckError();
        cudaEventSynchronize(m_solve_stop);
        cudaCheckError();
    }

    return e;
}

template<class T_Config>
int AMG_Solver<T_Config>::get_num_iters()
{
    return solver->get_num_iters();
}

template< class T_Config >
int AMG_Solver<T_Config>::getStructureReuseLevels()
{
    int lvls = 0;

    if (structure_reuse_levels_scope != "") // AMG is used as main solver
    {
        lvls = m_cfg->template getParameter<int>("structure_reuse_levels", structure_reuse_levels_scope);
    }

    return lvls;
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class AMG_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
