// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Constrained pressure residual preconditioner adapted from AMGCL (amgcl/preconditioner/cpr.hpp,
// amgcl/preconditioner/cpr_drs.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <solvers/cpr_solver.h>
#include <composite/submatrix.h>
#include <blas.h>
#include <multiply.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>

#include <thrust/remove.h>

#include <sstream>

namespace amgx
{

namespace
{

const int CPR_BLOCK_SIZE = 256;
const int CPR_MAX_BLOCK_DIM = 8;

inline int cprNumBlocks(int n)
{
    return std::max(1, std::min<int>(4096, (n + CPR_BLOCK_SIZE - 1) / CPR_BLOCK_SIZE));
}

// One thread per cell: restriction weights of its B rows
template <typename IndexType, typename ValueType>
__global__ void cprWeightsKernel(const IndexType *row_offsets, const IndexType *col_indices, const ValueType *values,
                                 int np, int B, bool drs, double eps_dd, double eps_ps, const double *component_weights,
                                 ValueType *w)
{
    for (int I = blockIdx.x * blockDim.x + threadIdx.x; I < np; I += gridDim.x * blockDim.x)
    {
        if (drs)
        {
            // AMGCL cpr_drs: a_top[c] = sum |A(IB, JB+c)|, a_dia[i] = A(IB+i, IB), a_off[i] = sum_{J!=I} |A(IB+i, JB)|
            double a_top[CPR_MAX_BLOCK_DIM], a_dia[CPR_MAX_BLOCK_DIM], a_off[CPR_MAX_BLOCK_DIM];

            for (int i = 0; i < B; i++) { a_top[i] = a_dia[i] = a_off[i] = 0.0; }

            for (int i = 0; i < B; i++)
            {
                const int row = I * B + i;

                for (IndexType k = row_offsets[row]; k < row_offsets[row + 1]; k++)
                {
                    const IndexType j = col_indices[k];
                    const int c = j % B;
                    const double v = (double) values[k];

                    if (i == 0) { a_top[c] += fabs(v); }

                    if (c == 0)
                    {
                        if (j / B == I) { a_dia[i] = v; }
                        else { a_off[i] += fabs(v); }
                    }
                }
            }

            for (int i = 0; i < B; i++)
            {
                double delta = component_weights != NULL ? component_weights[i] : 1.0;

                if (i > 0 && (a_dia[i] < eps_dd * a_off[i] || a_top[i] < eps_ps * fabs(a_dia[0])))
                {
                    delta = 0.0;
                }

                w[I * B + i] = (ValueType) delta;
            }
        }
        else
        {
            // AMGCL cpr: w solves D_I^T w = e_0 (LU with partial pivoting; e_0 if D_I is singular)
            double M[CPR_MAX_BLOCK_DIM * CPR_MAX_BLOCK_DIM], y[CPR_MAX_BLOCK_DIM];

            for (int i = 0; i < B * B; i++) { M[i] = 0.0; }

            for (int i = 0; i < B; i++)
            {
                const int row = I * B + i;
                y[i] = (i == 0) ? 1.0 : 0.0;

                for (IndexType k = row_offsets[row]; k < row_offsets[row + 1]; k++)
                {
                    const IndexType j = col_indices[k];

                    if (j / B == I) { M[(j % B) * B + i] = (double) values[k]; }
                }
            }

            bool singular = false;

            for (int c = 0; c < B && !singular; c++)
            {
                int p = c;

                for (int r = c + 1; r < B; r++)
                    if (fabs(M[r * B + c]) > fabs(M[p * B + c])) { p = r; }

                if (M[p * B + c] == 0.0) { singular = true; break; }

                if (p != c)
                {
                    for (int k = 0; k < B; k++) { const double t = M[p * B + k]; M[p * B + k] = M[c * B + k]; M[c * B + k] = t; }

                    const double t = y[p]; y[p] = y[c]; y[c] = t;
                }

                for (int r = c + 1; r < B; r++)
                {
                    const double f = M[r * B + c] / M[c * B + c];

                    for (int k = c; k < B; k++) { M[r * B + k] -= f * M[c * B + k]; }

                    y[r] -= f * y[c];
                }
            }

            if (!singular)
            {
                for (int r = B - 1; r >= 0; r--)
                {
                    double s = y[r];

                    for (int k = r + 1; k < B; k++) { s -= M[r * B + k] * y[k]; }

                    y[r] = s / M[r * B + r];
                }
            }

            for (int i = 0; i < B; i++) { w[I * B + i] = (ValueType)(singular ? (i == 0 ? 1.0 : 0.0) : y[i]); }
        }
    }
}

// One thread per scalar row: COO entries w_row * A(row, JB) of App(I, J); other columns are marked
template <typename IndexType, typename ValueType>
__global__ void cprPressureEntriesKernel(const IndexType *row_offsets, const IndexType *col_indices,
        const ValueType *values, int n, int B, int64_t np, const ValueType *w, int64_t *keys, ValueType *out)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x)
    {
        const int64_t I = row / B;
        const ValueType wr = w[row];

        for (IndexType k = row_offsets[row]; k < row_offsets[row + 1]; k++)
        {
            const IndexType j = col_indices[k];

            if (j % B == 0)
            {
                keys[k] = I * np + j / B;
                out[k] = wr * values[k];
            }
            else
            {
                keys[k] = -1;
            }
        }
    }
}

struct negative_key
{
    __host__ __device__ bool operator()(int64_t k) const { return k < 0; }
};

// rp[I] = sum_k w[IB+k] r[IB+k]
template <typename ValueTypeA, typename ValueTypeB>
__global__ void cprRestrictKernel(const ValueTypeA *w, const ValueTypeB *r, int np, int B, ValueTypeB *rp)
{
    for (int I = blockIdx.x * blockDim.x + threadIdx.x; I < np; I += gridDim.x * blockDim.x)
    {
        ValueTypeB s = ValueTypeB(0);

        for (int k = 0; k < B; k++) { s += ValueTypeB(w[I * B + k]) * r[I * B + k]; }

        rp[I] = s;
    }
}

// x[IB] += xp[I]
template <typename ValueType>
__global__ void cprProlongKernel(const ValueType *xp, int np, int B, ValueType *x)
{
    for (int I = blockIdx.x * blockDim.x + threadIdx.x; I < np; I += gridDim.x * blockDim.x)
    {
        x[I * B] += xp[I];
    }
}

void parseWeights(const std::string &s, std::vector<double> &w)
{
    w.clear();
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        if (item.find_first_not_of(" \t") != std::string::npos) { w.push_back(atof(item.c_str())); }
    }
}

} // anonymous namespace

template<class T_Config>
CPR_Solver<T_Config>::CPR_Solver(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope), m_B(0), m_np(0), m_smoother(NULL), m_pressure_solver(NULL)
{
    m_drs = cfg.AMG_Config::template getParameter<std::string>("cpr_decoupling", cfg_scope) == "DRS";
    m_block_dim_param = cfg.AMG_Config::template getParameter<int>("cpr_block_dim", cfg_scope);
    m_eps_dd = cfg.AMG_Config::template getParameter<double>("cpr_diag_dominant_tol", cfg_scope);
    m_eps_ps = cfg.AMG_Config::template getParameter<double>("cpr_pressure_coupling_tol", cfg_scope);
    parseWeights(cfg.AMG_Config::template getParameter<std::string>("cpr_drs_weights", cfg_scope), m_component_weights);
    std::string name, new_scope;
    cfg.getParameter<std::string>("cpr_global_smoother", name, cfg_scope, new_scope);

    if (name != "NOSOLVER")
    {
        m_smoother = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "cpr_global_smoother");
    }

    m_pressure_solver = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "cpr_pressure_solver");
}

template<class T_Config>
CPR_Solver<T_Config>::~CPR_Solver()
{
    delete m_smoother;
    delete m_pressure_solver;
}

template<class T_Config>
void CPR_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "cpr_decoupling: " << (m_drs ? "DRS" : "QUASI_IMPES") << std::endl;
    std::cout << "cpr_block_dim: " << m_B << std::endl;

    if (m_drs)
    {
        std::cout << "cpr_diag_dominant_tol: " << m_eps_dd << std::endl;
        std::cout << "cpr_pressure_coupling_tol: " << m_eps_ps << std::endl;
    }

    if (m_smoother != NULL)
    {
        std::cout << "cpr_global_smoother: " << m_smoother->getName() << " with scope name: " << m_smoother->getScope() << std::endl;
    }

    std::cout << "cpr_pressure_solver: " << m_pressure_solver->getName() << " with scope name: " << m_pressure_solver->getScope() << std::endl;
}

template<class T_Config>
void CPR_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    typedef composite::SubmatrixOps<TConfig> Ops;
    typedef typename Ops::I64Vector I64Vector;
    Ops::requireDevice("CPR");
    Matrix<TConfig> *A = dynamic_cast<Matrix<TConfig>*>(this->m_A);

    if (A == NULL)
    {
        FatalError("CPR only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (!A->is_matrix_singleGPU())
    {
        FatalError("CPR is not implemented for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    m_B = A->get_block_dimy() > 1 ? A->get_block_dimy() : m_block_dim_param;

    if (m_B < 1 || m_B > CPR_MAX_BLOCK_DIM)
    {
        FatalError("CPR: block dimension must be between 1 and 8", AMGX_ERR_BAD_PARAMETERS);
    }

    if (!m_component_weights.empty() && (int) m_component_weights.size() != m_B)
    {
        FatalError("CPR: cpr_drs_weights must list one weight per unknown of a cell", AMGX_ERR_BAD_PARAMETERS);
    }

    Matrix<TConfig> storage;
    const Matrix<TConfig> &As = Ops::scalarView(*A, storage);
    const int N = As.get_num_rows();

    if (N % m_B != 0)
    {
        FatalError("CPR: the number of unknowns is not a multiple of the block dimension", AMGX_ERR_BAD_PARAMETERS);
    }

    m_np = N / m_B;
    // Restriction weights
    m_weights.resize(N);
    Vector<typename TConfig::template setVecPrec<AMGX_vecDouble>::Type> component_weights;

    if (!m_component_weights.empty())
    {
        component_weights.resize(m_B);
        amgx::thrust::copy(m_component_weights.begin(), m_component_weights.end(), component_weights.begin());
    }

    if (m_np > 0)
    {
        cprWeightsKernel <<< cprNumBlocks(m_np), CPR_BLOCK_SIZE>>>(As.row_offsets.raw(), As.col_indices.raw(),
                As.values.raw(), m_np, m_B, m_drs, m_eps_dd, m_eps_ps,
                m_component_weights.empty() ? (const double *) NULL : component_weights.raw(), m_weights.raw());
        cudaCheckError();
    }

    // Pressure matrix App(I, J) = sum_k w[IB+k] A(IB+k, JB)
    const int nnz = As.get_num_nz();
    I64Vector keys(nnz);
    MVector values(nnz);

    if (N > 0)
    {
        cprPressureEntriesKernel <<< cprNumBlocks(N), CPR_BLOCK_SIZE>>>(As.row_offsets.raw(), As.col_indices.raw(),
                As.values.raw(), N, m_B, (int64_t) m_np, m_weights.raw(), keys.raw(), values.raw());
        cudaCheckError();
    }

    const int kept = amgx::thrust::remove_if(values.begin(), values.end(), keys.begin(), negative_key()) - values.begin();
    amgx::thrust::remove_if(keys.begin(), keys.end(), negative_key());
    cudaCheckError();
    keys.resize(kept);
    values.resize(kept);
    m_App.setResources(A->getResources());
    Ops::fromCOO(keys, values, m_np, m_np, m_App);
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

    m_rp.resize(m_np);
    m_xp.resize(m_np);
    m_rp.set_block_dimx(1);
    m_rp.set_block_dimy(1);
    m_xp.set_block_dimx(1);
    m_xp.set_block_dimy(1);

    if (m_smoother != NULL)
    {
        m_smoother->setup(*A, reuse_matrix_structure);
    }

    m_pressure_solver->setup(m_App, false);
}

template<class T_Config>
AMGX_STATUS CPR_Solver<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    Matrix<TConfig> &A = *dynamic_cast<Matrix<TConfig>*>(this->m_A);
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    const ValueTypeB one = types::util<ValueTypeB>::get_one();

    if (xIsZero)
    {
        fill(x, types::util<ValueTypeB>::get_zero());
    }

    // Stage 1: global smoother, x += S(b - A x)
    if (m_smoother != NULL)
    {
        if (xIsZero)
        {
            m_smoother->solve(b, x, true);
        }
        else
        {
            axmb(A, x, b, m_r, offset, size);
            m_smoother->solve(m_r, m_t, true);
            axpy(m_t, x, one, offset, size);
        }
    }

    // Stage 2: pressure correction, x += Scatter P(W (b - A x))
    axmb(A, x, b, m_r, offset, size);

    if (m_np > 0)
    {
        cprRestrictKernel <<< cprNumBlocks(m_np), CPR_BLOCK_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            m_weights.raw(), m_r.raw(), m_np, m_B, m_rp.raw());
        cudaCheckError();
    }

    m_rp.dirtybit = 1;
    m_pressure_solver->solve(m_rp, m_xp, true);

    if (m_np > 0)
    {
        cprProlongKernel <<< cprNumBlocks(m_np), CPR_BLOCK_SIZE, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
            m_xp.raw(), m_np, m_B, x.raw());
        cudaCheckError();
    }

    x.dirtybit = 1;
    A.setView(oldView);
    return this->converged(b, x);
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class CPR_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
