// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Schur complement pressure correction adapted from AMGCL
// (amgcl/preconditioner/schur_pressure_correction.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <solvers/schur_solver.h>
#include <composite/submatrix.h>
#include <blas.h>
#include <multiply.h>
#include <thrust_wrapper.h>
#include <error.h>
#include <cutil.h>

#include <cstdlib>
#include <sstream>

namespace amgx
{

namespace
{

const int SCHUR_BLOCK_SIZE = 256;

inline int schurNumBlocks(int n)
{
    return std::max(1, std::min<int>(4096, (n + SCHUR_BLOCK_SIZE - 1) / SCHUR_BLOCK_SIZE));
}

// M_i = 1 / sum_j |Kuu_ij| (SIMPLEC) or 1 / Kuu_ii
template <typename IndexType, typename ValueType>
__global__ void schurVelocityScaleKernel(const IndexType *row_offsets, const IndexType *col_indices,
        const ValueType *values, int n, bool simplec, ValueType *M)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        ValueType s = ValueType(0);

        for (IndexType k = row_offsets[i]; k < row_offsets[i + 1]; k++)
        {
            if (simplec) { s += fabs(values[k]); }
            else if (col_indices[k] == i) { s = values[k]; }
        }

        M[i] = (s != ValueType(0)) ? ValueType(1) / s : ValueType(0);
    }
}

// COO diagonal -diag(Kpu M Kup)_i (AMGCL adjust_p = 1)
template <typename IndexType, typename ValueType>
__global__ void schurDiagonalCorrectionKernel(const IndexType *pu_offsets, const IndexType *pu_cols,
        const ValueType *pu_values, const IndexType *up_offsets, const IndexType *up_cols, const ValueType *up_values,
        const ValueType *M, int np, int64_t *keys, ValueType *out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < np; i += gridDim.x * blockDim.x)
    {
        ValueType s = ValueType(0);

        for (IndexType k = pu_offsets[i]; k < pu_offsets[i + 1]; k++)
        {
            const IndexType u = pu_cols[k];

            for (IndexType kk = up_offsets[u]; kk < up_offsets[u + 1]; kk++)
            {
                if (up_cols[kk] == i)
                {
                    s += pu_values[k] * M[u] * up_values[kk];
                    break;
                }
            }
        }

        keys[i] = (int64_t) i * np + i;
        out[i] = -s;
    }
}

// Parses the pressure mask pattern: AMGCL's "%start:stride" (every stride-th unknown from start),
// ">start" (unknowns from start on) and "<end" (unknowns before end), usable in JSON configs, or
// the config-string forms "stride.START.STRIDE", "tail.START" and "head.END" (config strings only
// accept [A-Za-z0-9=_.+-])
void patternMask(const std::string &pattern, int n, std::vector<int> &mask)
{
    const char *usage = "Schur pressure correction: schur_pmask_pattern must be %start:stride, >start, <end, stride.START.STRIDE, tail.START or head.END";
    mask.assign(n, 0);
    char kind = 0;
    std::string args;

    if (!pattern.empty() && (pattern[0] == '%' || pattern[0] == '>' || pattern[0] == '<'))
    {
        kind = pattern[0];
        args = pattern.substr(1);
    }
    else
    {
        const size_t dot = pattern.find('.');
        const std::string word = pattern.substr(0, dot);
        args = dot == std::string::npos ? "" : pattern.substr(dot + 1);
        kind = word == "stride" ? '%' : word == "tail" ? '>' : word == "head" ? '<' : 0;
    }

    if (kind == 0 || args.empty())
    {
        FatalError(usage, AMGX_ERR_BAD_PARAMETERS);
    }

    if (kind == '%')
    {
        const size_t sep = args.find_first_of(":.");

        if (sep == std::string::npos)
        {
            FatalError(usage, AMGX_ERR_BAD_PARAMETERS);
        }

        const int start = atoi(args.substr(0, sep).c_str());
        const int stride = atoi(args.substr(sep + 1).c_str());

        if (stride < 1 || start < 0)
        {
            FatalError(usage, AMGX_ERR_BAD_PARAMETERS);
        }

        for (int i = start; i < n; i += stride) { mask[i] = 1; }
    }
    else if (kind == '>')
    {
        for (int i = std::max(atoi(args.c_str()), 0); i < n; i++) { mask[i] = 1; }
    }
    else
    {
        for (int i = 0; i < std::min(atoi(args.c_str()), n); i++) { mask[i] = 1; }
    }
}

} // anonymous namespace

template<class T_Config>
SchurPressureCorrection_Solver<T_Config>::SchurPressureCorrection_Solver(AMG_Config &cfg, const std::string &cfg_scope)
    : Solver<T_Config>(cfg, cfg_scope), m_nu(0), m_np(0)
{
    m_pattern = cfg.AMG_Config::template getParameter<std::string>("schur_pmask_pattern", cfg_scope);
    m_variant = cfg.AMG_Config::template getParameter<int>("schur_variant", cfg_scope);
    m_adjust_p = cfg.AMG_Config::template getParameter<int>("schur_adjust_p", cfg_scope);
    m_simplec = cfg.AMG_Config::template getParameter<int>("schur_simplec_dia", cfg_scope) != 0;
    m_velocity_solver = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "schur_velocity_solver");
    m_pressure_solver = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "schur_pressure_solver");
}

template<class T_Config>
SchurPressureCorrection_Solver<T_Config>::~SchurPressureCorrection_Solver()
{
    delete m_velocity_solver;
    delete m_pressure_solver;
}

template<class T_Config>
void SchurPressureCorrection_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "schur_variant: " << m_variant << std::endl;
    std::cout << "schur_adjust_p: " << m_adjust_p << std::endl;
    std::cout << "schur_simplec_dia: " << (m_simplec ? 1 : 0) << std::endl;
    std::cout << "schur_velocity_solver: " << m_velocity_solver->getName() << " with scope name: " << m_velocity_solver->getScope() << std::endl;
    std::cout << "schur_pressure_solver: " << m_pressure_solver->getName() << " with scope name: " << m_pressure_solver->getScope() << std::endl;
}

template<class T_Config>
void SchurPressureCorrection_Solver<T_Config>::buildMask(const Matrix<TConfig> &A, int n, IVector &mask) const
{
    std::vector<int> h_mask;

    if (!A.m_row_markers.empty())
    {
        if ((int) A.m_row_markers.size() != n)
        {
            FatalError("Schur pressure correction: the row markers do not match the matrix size", AMGX_ERR_BAD_PARAMETERS);
        }

        h_mask = A.m_row_markers;
    }
    else if (!m_pattern.empty())
    {
        patternMask(m_pattern, n, h_mask);
    }
    else
    {
        FatalError("Schur pressure correction: set the pressure unknowns with AMGX_matrix_set_row_markers or schur_pmask_pattern", AMGX_ERR_BAD_PARAMETERS);
    }

    mask.resize(n);
    amgx::thrust::copy(h_mask.begin(), h_mask.end(), mask.begin());
}

template<class T_Config>
void SchurPressureCorrection_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    typedef composite::SubmatrixOps<TConfig> Ops;
    typedef typename Ops::I64Vector I64Vector;
    Ops::requireDevice("Schur pressure correction");
    Matrix<TConfig> *A = dynamic_cast<Matrix<TConfig>*>(this->m_A);

    if (A == NULL)
    {
        FatalError("Schur pressure correction only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    if (!A->is_matrix_singleGPU())
    {
        FatalError("Schur pressure correction is not implemented for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
    }

    Matrix<TConfig> storage;
    const Matrix<TConfig> &As = Ops::scalarView(*A, storage);
    const int n = As.get_num_rows();
    IVector mask, map_p, map_u;
    buildMask(*A, n, mask);
    m_np = Ops::splitMask(mask, map_p, map_u, m_idx_p, m_idx_u);
    m_nu = n - m_np;

    if (m_np == 0 || m_nu == 0)
    {
        FatalError("Schur pressure correction: both the pressure and the velocity sets must be non-empty", AMGX_ERR_BAD_PARAMETERS);
    }

    Matrix<TConfig> *blocks[] = { &m_Kuu, &m_Kup, &m_Kpu, &m_Kpp, &m_S };

    for (int k = 0; k < 5; k++) { blocks[k]->setResources(A->getResources()); }

    Ops::extract(As, map_u, map_u, m_nu, m_nu, m_Kuu);
    Ops::extract(As, map_u, map_p, m_nu, m_np, m_Kup);
    Ops::extract(As, map_p, map_u, m_np, m_nu, m_Kpu);
    Ops::extract(As, map_p, map_p, m_np, m_np, m_Kpp);
    // Pressure matrix
    MVector M(m_nu);
    schurVelocityScaleKernel <<< schurNumBlocks(m_nu), SCHUR_BLOCK_SIZE>>>(m_Kuu.row_offsets.raw(),
            m_Kuu.col_indices.raw(), m_Kuu.values.raw(), m_nu, m_simplec, M.raw());
    cudaCheckError();
    const ValueTypeA one = types::util<ValueTypeA>::get_one();

    if (m_adjust_p == 2)
    {
        // Kpp - Kpu M Kup
        Matrix<TConfig> Kup_hat, product;
        Kup_hat.setResources(A->getResources());
        product.setResources(A->getResources());
        Kup_hat.copy(m_Kup);
        Ops::scaleRows(Kup_hat, M);
        Ops::product(m_Kpu, Kup_hat, product);
        Ops::add(one, m_Kpp, types::util<ValueTypeA>::invert(one), product, m_S);
    }
    else if (m_adjust_p == 1)
    {
        // Kpp - diag(Kpu M Kup); the diagonal is inserted where Kpp has none
        I64Vector keys(m_np);
        MVector values(m_np);
        schurDiagonalCorrectionKernel <<< schurNumBlocks(m_np), SCHUR_BLOCK_SIZE>>>(m_Kpu.row_offsets.raw(),
                m_Kpu.col_indices.raw(), m_Kpu.values.raw(), m_Kup.row_offsets.raw(), m_Kup.col_indices.raw(),
                m_Kup.values.raw(), M.raw(), m_np, keys.raw(), values.raw());
        cudaCheckError();
        Matrix<TConfig> correction;
        correction.setResources(A->getResources());
        Ops::fromCOO(keys, values, m_np, m_np, correction);
        Ops::add(one, m_Kpp, one, correction, m_S);
    }
    else
    {
        m_S.copy(m_Kpp);
    }

    const int zero_pivots = Ops::countZeroDiagonal(m_S);

    if (zero_pivots > 0)
    {
        std::stringstream ss;
        ss << "Schur pressure correction: the pressure matrix has " << zero_pivots << " zero diagonal entries"
           << " (C = 0 with schur_adjust_p = 0?); smoothers and AMG need schur_adjust_p 1 or 2" << std::endl;
        amgx_output(ss.str().c_str(), ss.str().length());
    }

    // Work vectors
    m_r.resize(A->get_num_cols() * A->get_block_dimy());
    m_r.set_block_dimy(A->get_block_dimy());
    m_r.set_block_dimx(1);
    m_r.dirtybit = 1;
    m_r.delayed_send = 1;
    VVector *vu[] = { &m_fu, &m_u, &m_tu };
    VVector *vp[] = { &m_fp, &m_p, &m_tp };

    for (int k = 0; k < 3; k++)
    {
        vu[k]->resize(m_nu);
        vp[k]->resize(m_np);
        vu[k]->set_block_dimx(1);
        vu[k]->set_block_dimy(1);
        vp[k]->set_block_dimx(1);
        vp[k]->set_block_dimy(1);
    }

    m_velocity_solver->setup(m_Kuu, false);
    m_pressure_solver->setup(m_S, false);
}

template<class T_Config>
AMGX_STATUS SchurPressureCorrection_Solver<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    typedef composite::SubmatrixOps<TConfig> Ops;
    Matrix<TConfig> &A = *dynamic_cast<Matrix<TConfig>*>(this->m_A);
    ViewType oldView = A.currentView();
    A.setViewExterior();
    int offset, size;
    A.getOffsetAndSizeForView(A.getViewExterior(), &offset, &size);
    const ValueTypeB one = types::util<ValueTypeB>::get_one();
    const ValueTypeB minus_one = types::util<ValueTypeB>::invert(one);

    // Right-hand side of the correction: b, or b - A x
    VVector *rhs = &b;

    if (!xIsZero)
    {
        axmb(A, x, b, m_r, offset, size);
        rhs = &m_r;
    }

    Ops::gather(*rhs, m_idx_u, m_fu);
    Ops::gather(*rhs, m_idx_p, m_fp);

    if (m_variant == 1)
    {
        // u = U(fu); p = P(fp - Kpu u)
        m_velocity_solver->solve(m_fu, m_u, true);
        multiply(m_Kpu, m_u, m_tp);
        axpby(m_fp, m_tp, m_tp, one, minus_one);
        m_pressure_solver->solve(m_tp, m_p, true);
    }
    else
    {
        // p = P(fp)
        m_pressure_solver->solve(m_fp, m_p, true);
    }

    // u = U(fu - Kup p)
    multiply(m_Kup, m_p, m_tu);
    axpby(m_fu, m_tu, m_tu, one, minus_one);
    m_velocity_solver->solve(m_tu, m_u, true);

    if (xIsZero)
    {
        Ops::scatter(m_u, m_idx_u, x);
        Ops::scatter(m_p, m_idx_p, x);
    }
    else
    {
        Ops::scatter(m_u, m_idx_u, m_r);
        Ops::scatter(m_p, m_idx_p, m_r);
        axpy(m_r, x, one, offset, size);
    }

    x.dirtybit = 1;
    A.setView(oldView);
    return this->converged(b, x);
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class SchurPressureCorrection_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
