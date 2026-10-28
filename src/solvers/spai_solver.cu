// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Sparse approximate inverse smoothers adapted from AMGCL (amgcl/relaxation/spai0.hpp,
// amgcl/relaxation/spai1.hpp), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <solvers/spai_solver.h>
#include <aggregation/pointwise.h>
#include <basic_types.h>
#include <blas.h>
#include <csr_multiply.h>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <util.h>

#include <algorithm>
#include <sstream>

namespace amgx
{

// -----------
// Kernels
// -----------

namespace
{

const int SPAI_BLOCK_SIZE = 128;

inline int spaiNumBlocks(int n)
{
    return std::max(1, std::min<int>(4096, (n + SPAI_BLOCK_SIZE - 1) / SPAI_BLOCK_SIZE));
}

// M_i = A_ii / sum_j ||A_ij||_F^2 for each block row (the diagonal block may be stored externally)
template <typename IndexType, typename ValueTypeA>
__global__ void spai0SetupKernel(const IndexType *row_offsets, const ValueTypeA *values, const IndexType *diag,
                                 int num_rows, int bsize, bool external_diag, ValueTypeA *M)
{
    const int bb = bsize * bsize;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += gridDim.x * blockDim.x)
    {
        double den = 0.0;

        for (IndexType k = row_offsets[i] * bb; k < row_offsets[i + 1] * bb; k++)
        {
            den += (double) values[k] * (double) values[k];
        }

        const IndexType d = diag[i] * bb;

        if (external_diag)
            for (int q = 0; q < bb; q++) { den += (double) values[d + q] * (double) values[d + q]; }

        for (int q = 0; q < bb; q++)
        {
            M[(size_t) i * bb + q] = den > 0.0 ? ValueTypeA(values[d + q] / den) : ValueTypeA(0);
        }
    }
}

// x_i = (x_i if y given) + w M_i (rhs_i - y_i), one thread per unknown
template <typename ValueTypeA, typename ValueTypeB>
__global__ void spai0ApplyKernel(int num_unknowns, int bsize, const ValueTypeA *M, const ValueTypeB *rhs,
                                 const ValueTypeB *y, ValueTypeB w, ValueTypeB *x)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_unknowns; i += gridDim.x * blockDim.x)
    {
        const int I = i / bsize, r = i % bsize;
        const ValueTypeA *Mi = M + (size_t) I * bsize * bsize + r * bsize;
        ValueTypeB s = ValueTypeB(0);

        for (int c = 0; c < bsize; c++)
        {
            const ValueTypeB res = (y != NULL) ? rhs[I * bsize + c] - y[I * bsize + c] : rhs[I * bsize + c];
            s += ValueTypeB(Mi[c]) * res;
        }

        x[i] = (y != NULL) ? x[i] + w * s : w * s;
    }
}

template <typename IndexType, typename ValueType>
__device__ __forceinline__ double sortedRowLookup(const IndexType *row_offsets, const IndexType *cols,
        const ValueType *values, IndexType row, IndexType col)
{
    IndexType lo = row_offsets[row], hi = row_offsets[row + 1];

    while (lo < hi)
    {
        const IndexType mid = (lo + hi) / 2;

        if (cols[mid] < col) { lo = mid + 1; }
        else { hi = mid; }
    }

    return (lo < row_offsets[row + 1] && cols[lo] == col) ? (double) values[lo] : 0.0;
}

// One warp per row i of A: solve the normal equations G m = g of min || A(I,:)^T m - e_i ||, with
// G_pq = (A A^T)(I_p, I_q) read from the sorted product C and g_p = A(I_p, i).
template <typename IndexType, typename ValueTypeA>
__global__ void spai1RowKernel(const IndexType *A_offsets, const IndexType *A_cols, const ValueTypeA *A_vals,
                               const IndexType *C_offsets, const IndexType *C_cols, const ValueTypeA *C_vals,
                               int num_rows, int max_len, ValueTypeA *M_vals, int *num_fallback)
{
    extern __shared__ double smem[];
    const int lane = threadIdx.x & 31;
    double *G = smem;
    double *g = G + max_len * max_len;

    for (int i = blockIdx.x; i < num_rows; i += gridDim.x)
    {
        const IndexType beg = A_offsets[i];
        const int L = A_offsets[i + 1] - beg;

        if (L == 0) { continue; }

        bool fallback = L > max_len;

        if (!fallback)
        {
            // Normal equations (upper triangle, mirrored)
            for (int idx = lane; idx < L * L; idx += 32)
            {
                const int p = idx / L, q = idx % L;

                if (q >= p)
                {
                    const double v = sortedRowLookup(C_offsets, C_cols, C_vals, A_cols[beg + p], A_cols[beg + q]);
                    G[p * L + q] = v;
                    G[q * L + p] = v;
                }
            }

            for (int p = lane; p < L; p += 32)
            {
                const IndexType row = A_cols[beg + p];
                double v = 0.0;

                for (IndexType k = A_offsets[row]; k < A_offsets[row + 1]; k++)
                    if (A_cols[k] == i) { v += (double) A_vals[k]; }

                g[p] = v;
            }

            __syncwarp();
            double max_diag = 0.0;

            for (int p = 0; p < L; p++) { max_diag = fmax(max_diag, G[p * L + p]); }

            // Cholesky G = R^T R, R stored in the upper triangle
            for (int j = 0; j < L && !fallback; j++)
            {
                const double d = G[j * L + j];

                if (!(d > 1e-12 * max_diag))
                {
                    fallback = true;
                    break;
                }

                const double r = sqrt(d);
                __syncwarp();

                for (int q = j + lane; q < L; q += 32) { G[j * L + q] /= r; }

                __syncwarp();

                for (int idx = lane; idx < (L - j - 1) * (L - j - 1); idx += 32)
                {
                    const int p = j + 1 + idx / (L - j - 1), q = j + 1 + idx % (L - j - 1);

                    if (q >= p) { G[p * L + q] -= G[j * L + p] * G[j * L + q]; }
                }

                __syncwarp();
            }

            if (!fallback && lane == 0)
            {
                // R^T z = g, then R m = z
                for (int p = 0; p < L; p++)
                {
                    double s = g[p];

                    for (int k = 0; k < p; k++) { s -= G[k * L + p] * g[k]; }

                    g[p] = s / G[p * L + p];
                }

                for (int p = L - 1; p >= 0; p--)
                {
                    double s = g[p];

                    for (int k = p + 1; k < L; k++) { s -= G[p * L + k] * g[k]; }

                    g[p] = s / G[p * L + p];
                }
            }

            __syncwarp();
        }

        if (fallback)
        {
            // SPAI0 row: m_ii = a_ii / ||a_i||^2
            double den = 0.0, dia = 0.0;

            for (int p = 0; p < L; p++)
            {
                const double v = A_vals[beg + p];
                den += v * v;

                if (A_cols[beg + p] == i) { dia += v; }
            }

            for (int p = lane; p < L; p += 32)
            {
                M_vals[beg + p] = (A_cols[beg + p] == i && den > 0.0) ? ValueTypeA(dia / den) : ValueTypeA(0);
            }

            if (lane == 0) { atomicAdd(num_fallback, 1); }
        }
        else
        {
            for (int p = lane; p < L; p += 32) { M_vals[beg + p] = ValueTypeA(g[p]); }
        }

        __syncwarp();
    }
}

} // anonymous namespace

// -----------------
// Methods
// -----------------

template<class T_Config>
SPAISolver_Base<T_Config>::SPAISolver_Base(AMG_Config &cfg, const std::string &cfg_scope, int level)
    : Solver<T_Config>(cfg, cfg_scope), m_explicit_A(NULL), m_level(level), m_num_fallback_rows(0)
{
    m_weight = cfg.AMG_Config::template getParameter<double>("relaxation_factor", cfg_scope);

    if (m_weight == 0)
    {
        m_weight = 1.;
    }
}

template<class T_Config>
void SPAISolver_Base<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    m_explicit_A = dynamic_cast<Matrix<T_Config>*>(Base::m_A);

    if (!m_explicit_A)
    {
        FatalError("SPAI smoothers only work with explicit matrices", AMGX_ERR_INTERNAL);
    }

    const Matrix<T_Config> &A = *m_explicit_A;

    if (A.get_block_dimx() != A.get_block_dimy())
    {
        FatalError("SPAI smoothers require square blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (m_level == 0)
    {
        computeSPAI0(A);
    }
    else
    {
        if (A.get_block_size() != 1)
        {
            FatalError("SPAI1 only supports scalar matrices", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        }

        if (!A.is_matrix_singleGPU())
        {
            FatalError("SPAI1 is not implemented for distributed matrices", AMGX_ERR_NOT_IMPLEMENTED);
        }

        computeSPAI1(A);

        if (m_num_fallback_rows > 0)
        {
            std::stringstream ss;
            ss << "SPAI1: " << m_num_fallback_rows << " rows use the SPAI0 value (longer than "
               << SPAI1_MAX_ROW_LENGTH << " entries or singular normal equations)" << std::endl;
            amgx_output(ss.str().c_str(), ss.str().length());
        }
    }

    const int N = A.get_num_cols() * A.get_block_dimy();
    m_y.resize(N);
    m_r.resize(m_level == 1 ? N : 0);
}

template<class T_Config>
AMGX_STATUS SPAISolver_Base<T_Config>::solve_iteration(VVector &b, VVector &x, bool xIsZero)
{
    if (xIsZero) { x.dirtybit = 0; }

    Matrix<T_Config> &A = *m_explicit_A;

    if (m_level == 0)
    {
        smoothSPAI0(A, b, x, xIsZero);
    }
    else if (xIsZero)
    {
        // x = w M b
        multiply(m_spai1, b, x);
        scal(x, m_weight, 0, A.get_num_rows());
    }
    else
    {
        // x += w M (b - A x)
        m_y.set_block_dimx(b.get_block_dimx());
        m_y.set_block_dimy(b.get_block_dimy());
        m_r.set_block_dimx(b.get_block_dimx());
        m_r.set_block_dimy(b.get_block_dimy());
        multiply(A, x, m_y);
        axpby(b, m_y, m_r, ValueTypeB(1), ValueTypeB(-1), 0, A.get_num_rows());
        multiply(m_spai1, m_r, m_y);
        axpy(m_y, x, m_weight, 0, A.get_num_rows());
    }

    x.dirtybit = 1;
    return this->converged(b, x);
}

// --------------------------- Host specialization ------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SPAISolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeSPAI0(const Matrix<TConfig> &A)
{
    FatalError("SPAI smoothers are not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SPAISolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeSPAI1(const Matrix<TConfig> &A)
{
    FatalError("SPAI smoothers are not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SPAISolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::smoothSPAI0(const Matrix<TConfig> &A, VVector &b, VVector &x, bool xIsZero)
{
    FatalError("SPAI smoothers are not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
}

// --------------------------- Device specialization ------------------------------------

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SPAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeSPAI0(const Matrix<TConfig> &A)
{
    const int n = A.get_num_rows(), bs = A.get_block_dimy();

    if (bs > 1 && A.getBlockFormat() != ROW_MAJOR)
    {
        FatalError("SPAI0 requires row-major blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    this->m_spai0.resize((size_t) n * bs * bs);

    if (n > 0)
    {
        spai0SetupKernel <<< spaiNumBlocks(n), SPAI_BLOCK_SIZE>>>(A.row_offsets.raw(), A.values.raw(), A.diag.raw(), n, bs,
                A.hasProps(DIAG), this->m_spai0.raw());
        cudaCheckError();
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SPAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::smoothSPAI0(const Matrix<TConfig> &A, VVector &b, VVector &x, bool xIsZero)
{
    const int bs = A.get_block_dimy();
    const int n = A.get_num_rows() * bs;

    if (!xIsZero)
    {
        this->m_y.set_block_dimx(b.get_block_dimx());
        this->m_y.set_block_dimy(b.get_block_dimy());
        multiply(const_cast<Matrix<TConfig> &>(A), x, this->m_y);
    }

    if (n > 0)
    {
        spai0ApplyKernel <<< spaiNumBlocks(n), SPAI_BLOCK_SIZE>>>(n, bs, this->m_spai0.raw(), b.raw(),
                xIsZero ? (const typename Base::ValueTypeB *) NULL : this->m_y.raw(), this->m_weight, x.raw());
        cudaCheckError();
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SPAISolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeSPAI1(const Matrix<TConfig> &A)
{
    // Scalar copy without external diagonal
    Matrix<TConfig> A_expanded;

    if (A.hasProps(DIAG))
    {
        aggregation::PointwiseAggregation<TConfig>::blockToScalar(A, A_expanded);
    }

    const Matrix<TConfig> &As = A.hasProps(DIAG) ? A_expanded : A;
    const int n = As.get_num_rows();
    // C = A A^T holds the entries of every normal-equation matrix
    Matrix<TConfig> At, C;
    transpose(As, At);
    void *wk = CSR_Multiply<TConfig>::csr_workspace_create();
    C.set_initialized(0);
    C.addProps(CSR);
    CSR_Multiply<TConfig>::csr_multiply(As, At, C, wk);
    CSR_Multiply<TConfig>::csr_workspace_delete(wk);
    C.sortByRowAndColumn();
    // M has the pattern of A
    this->m_spai1.set_initialized(0);
    this->m_spai1.copy(As);
    this->m_spai1.set_initialized(1);
    Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> fallback(1, 0);

    if (n > 0)
    {
        const int max_len = Base::SPAI1_MAX_ROW_LENGTH;
        const size_t shared = (size_t)(max_len * max_len + max_len) * sizeof(double);
        spai1RowKernel <<< std::min(n, 8192), 32, shared>>>(As.row_offsets.raw(), As.col_indices.raw(), As.values.raw(),
                C.row_offsets.raw(), C.col_indices.raw(), C.values.raw(), n, max_len, this->m_spai1.values.raw(),
                fallback.raw());
        cudaCheckError();
    }

    this->m_num_fallback_rows = fallback[0];
}

/****************************************
 * Explicit instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class SPAISolver_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class SPAISolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
