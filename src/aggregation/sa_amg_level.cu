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
#include <aggregation/pointwise.h>
#include <amg.h>
#include <blas.h>
#include <csr_multiply.h>
#include <multiply.h>
#include <transpose.h>
#include <truncate.h>
#include <amgx_timer.h>
#include <distributed/distributed_manager.h>
#include <distributed/distributed_arranger.h>
#include <thrust_wrapper.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <cutil.h>
#include <util.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace amgx
{
namespace aggregation
{

namespace
{

const int SAL_BLOCK_SIZE = 256;

inline int salNumBlocks(int n) { return std::max(1, std::min(4096, (n + SAL_BLOCK_SIZE - 1) / SAL_BLOCK_SIZE)); }

// dst[renumbering[i] * w + t] = src[i * w + t]
__global__ void permuteRowsKernel(const int *renumbering, int n, int w, const double *src, double *dst)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        for (int t = 0; t < w; t++) { dst[(size_t) renumbering[i] * w + t] = src[(size_t) i * w + t]; }
}

// Flags nodes whose b unknowns are not renumbered to consecutive rows starting at a multiple of b
__global__ void nodeContiguityKernel(const int *renumbering, int num_nodes, int b, int *bad)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_nodes; i += gridDim.x * blockDim.x)
    {
        const int first = renumbering[i * b];
        bool ok = first % b == 0;

        for (int k = 1; k < b; k++) { ok = ok && renumbering[i * b + k] == first + k; }

        if (!ok) { atomicExch(bad, 1); }
    }
}

// Rows of P_tent at a fixed width W (global columns, -1 pads), with the coarse nullspace rows of their
// columns (CB values each) when cb_src is given
template <typename IndexType, typename ValueType>
__global__ void packTentativeKernel(const IndexType *ro, const IndexType *ci, const ValueType *v, int n, int W, int CB,
                                    const double *cb_src, int64_t base, int64_t *gcol, ValueType *gval, double *gbc)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const int len = ro[i + 1] - ro[i];

        for (int t = 0; t < W; t++)
        {
            const int q = i * W + t;
            const IndexType c = t < len ? ci[ro[i] + t] : -1;
            gcol[q] = c >= 0 ? base + c : -1;
            gval[q] = c >= 0 ? v[ro[i] + t] : ValueType(0);

            if (cb_src != NULL)
                for (int s = 0; s < CB; s++) { gbc[(size_t) q * CB + s] = c >= 0 ? cb_src[(size_t) c * CB + s] : 0.0; }
        }
    }
}

__device__ __forceinline__ int lowerBound64(const int64_t *a, int n, int64_t key)
{
    int lo = 0, hi = n;

    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;

        if (a[mid] < key) { lo = mid + 1; }
        else { hi = mid; }
    }

    return lo;
}

// Local column of a global coarse column: owned first, then the sorted halo list
__device__ __forceinline__ int localCoarseColumn(int64_t g, int64_t base, int n_coarse, const int64_t *halo, int num_halo)
{
    return (g >= base && g < base + n_coarse) ? (int)(g - base) : n_coarse + lowerBound64(halo, num_halo, g);
}

__global__ void extendedRowCountKernel(const int64_t *gcol, int rows, int W, int *counts)
{
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x)
    {
        int c = 0;

        for (int t = 0; t < W; t++) { c += gcol[r * W + t] >= 0; }

        counts[r] = c;
    }
}

template <typename IndexType, typename ValueType>
__global__ void extendedFillKernel(const int64_t *gcol, const ValueType *gval, int rows, int W, int64_t base, int n_coarse,
                                   const int64_t *halo, int num_halo, const IndexType *ro, IndexType *ci, ValueType *v)
{
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x)
    {
        IndexType w = ro[r];

        for (int t = 0; t < W; t++)
        {
            const int64_t g = gcol[r * W + t];

            if (g < 0) { continue; }

            ci[w] = localCoarseColumn(g, base, n_coarse, halo, num_halo);
            v[w] = gval[r * W + t];
            w++;
        }
    }
}

// Coarse nullspace rows of the halo columns, from the received tentative rows
__global__ void haloNullspaceKernel(const int64_t *gcol, const double *gbc, int first_row, int rows, int W, int CB,
                                    int64_t base, int n_coarse, const int64_t *halo, int num_halo, double *bc)
{
    for (int r = first_row + blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x)
        for (int t = 0; t < W; t++)
        {
            const int64_t g = gcol[r * W + t];

            if (g < 0 || (g >= base && g < base + n_coarse)) { continue; }

            const int c = n_coarse + lowerBound64(halo, num_halo, g);

            for (int s = 0; s < CB; s++) { bc[(size_t) c * CB + s] = gbc[((size_t) r * W + t) * CB + s]; }
        }
}

template <typename IndexType>
__global__ void globalColumnsKernel(const IndexType *ci, int nnz, int n_coarse, int64_t base, const int64_t *halo, int64_t *g)
{
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < nnz; k += gridDim.x * blockDim.x)
    {
        const IndexType c = ci[k];
        g[k] = c < n_coarse ? base + c : halo[c - n_coarse];
    }
}

// Compression of the halo columns of RAP that only the dropped external rows referenced
// (as in the distributed classical path)
__global__ void flagHaloColumnsKernel(int n, const int *row_offsets, const int *col_indices, int *flags)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            if (col_indices[j] >= n) { flags[col_indices[j] - n] = 1; }
}

__global__ void compressHaloColumnsKernel(int n, const int *row_offsets, int *col_indices, const int *flags)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; j++)
            if (col_indices[j] >= n) { col_indices[j] = n + flags[col_indices[j] - n]; }
}

__global__ void compressLocalToGlobalKernel(int n, const int64_t *l2g_in, int64_t *l2g_out, const int *flags)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (flags[i] != flags[i + 1]) { l2g_out[flags[i]] = l2g_in[i]; }
}

// Rows of P (columns sorted) padded to whole blocks of cb coarse unknowns: each block a row touches
// gets all its cb columns, explicit zeros where truncation dropped them. Count pass (ci_out == NULL), fill.
template <typename IndexType, typename ValueType>
__global__ void padBlocksKernel(const IndexType *ro, const IndexType *ci, const ValueType *v, int n, int cb,
                                int *counts, const IndexType *ro_out, IndexType *ci_out, ValueType *v_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        int count = 0;
        IndexType w = ci_out != NULL ? ro_out[i] : 0;

        for (IndexType k = ro[i]; k < ro[i + 1];)
        {
            const IndexType block = ci[k] / cb;

            for (int t = 0; t < cb; t++)
            {
                const IndexType c = block * cb + t;
                const bool present = k < ro[i + 1] && ci[k] == c;

                if (ci_out != NULL)
                {
                    ci_out[w] = c;
                    v_out[w] = present ? v[k] : ValueType(0);
                    w++;
                }

                if (present) { k++; }
            }

            count += cb;
        }

        if (ci_out == NULL) { counts[i] = count; }
    }
}

struct halo_column
{
    int64_t base, end;
    halo_column(int64_t b, int64_t e) : base(b), end(e) {}
    __host__ __device__ bool operator()(int64_t g) const { return g < 0 || (g >= base && g < end); }
};

} // anonymous namespace

template <class T_Config>
SA_AMG_Level_Base<T_Config>::SA_AMG_Level_Base(AMG_Class *amg, ThreadManager *tmng)
    : AMG_Level<T_Config>(amg, tmng), m_use_scalar_copy(false), m_block_size(1), m_inherited_block_size(0),
      m_nullspace_cols(0), m_num_aggregates(0), m_num_removed(0), m_num_coarse_vertices(0),
      m_level_eps(0.0), m_rho(0.0), m_omega(0.0)
{
    AMG_Config &cfg = *(amg->m_cfg);
    const std::string &scope = amg->m_cfg_scope;
    m_eps_strong = cfg.AMG_Config::template getParameter<double>("aggregation_eps_strong", scope);
    m_eps_decay = cfg.AMG_Config::template getParameter<double>("aggregation_eps_decay", scope);
    m_mis_index_priority = cfg.AMG_Config::template getParameter<std::string>("aggregation_mis_priority", scope) == "INDEX";
    m_relax = cfg.AMG_Config::template getParameter<double>("sa_relaxation_factor", scope);
    m_power_iters = cfg.AMG_Config::template getParameter<int>("sa_spectral_radius_iters", scope);
    m_drop_tol = cfg.AMG_Config::template getParameter<double>("sa_prolongator_drop_tol", scope);
    m_aggregation_block_size = cfg.AMG_Config::template getParameter<int>("aggregation_block_size", scope);
    m_print_aggregation_info = cfg.AMG_Config::template getParameter<int>("print_aggregation_info", scope) != 0;
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    Matrix<TConfig> &A = this->getA();
    const bool distributed = isDistributed();

    if (distributed && (A.get_block_dimy() > 1 || A.hasProps(DIAG)))
    {
        FatalError("Distributed SMOOTHED_AGGREGATION needs a scalar matrix without external diagonal: upload systems as "
                   "scalar rows with aggregation_block_size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if (distributed) { A.setView(OWNED); }

    if (A.get_block_dimx() != A.get_block_dimy())
    {
        FatalError("SMOOTHED_AGGREGATION requires square blocks", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    // Unknowns per node: the BSR block, the user setting on the finest level, or what the
    // previous level produced
    if (A.get_block_dimy() > 1)
    {
        m_block_size = A.get_block_dimy();
    }
    else if (this->isFinest())
    {
        m_block_size = std::max(1, m_aggregation_block_size);
    }
    else
    {
        m_block_size = std::max(1, m_inherited_block_size);
    }

    // The near-nullspace of the finest level comes with the matrix
    if (this->isFinest())
    {
        m_nullspace_cols = A.m_nullspace_cols;
        m_nullspace.resize(A.m_nullspace.size());

        if (!A.m_nullspace.empty())
        {
            amgx::thrust::copy(A.m_nullspace.begin(), A.m_nullspace.end(), m_nullspace.begin());
        }

        if ((size_t) m_nullspace.size() != (size_t) A.get_num_rows() * A.get_block_dimy() * m_nullspace_cols)
        {
            FatalError("The near-nullspace set with AMGX_matrix_set_nullspace does not match the matrix size",
                       AMGX_ERR_BAD_PARAMETERS);
        }

        // The user's rows were renumbered into [interior | boundary] at upload: the nullspace follows,
        // and the unknowns of a node must stay consecutive for the pointwise aggregation
        const int n = A.get_num_rows();

        if (distributed && (int) A.manager->renumbering.size() >= n && n > 0)
        {
            if (m_block_size > 1)
            {
                IVector bad(1, 0);
                nodeContiguityKernel <<< salNumBlocks(n / m_block_size), SAL_BLOCK_SIZE>>>(A.manager->renumbering.raw(),
                        n / m_block_size, m_block_size, bad.raw());
                cudaCheckError();

                if (bad[0] != 0)
                {
                    FatalError("Distributed SMOOTHED_AGGREGATION: the interior/boundary renumbering split the unknowns of a "
                               "node (each node's unknowns need the same coupling pattern)", AMGX_ERR_BAD_PARAMETERS);
                }
            }

            if (m_nullspace_cols > 0)
            {
                DVector permuted(m_nullspace.size());
                permuteRowsKernel <<< salNumBlocks(n), SAL_BLOCK_SIZE>>>(A.manager->renumbering.raw(), n, m_nullspace_cols,
                        m_nullspace.raw(), permuted.raw());
                cudaCheckError();
                m_nullspace.swap(permuted);
            }
        }
    }

    // eps_l = max(eps_strong * decay^l, 1e-4)
    m_level_eps = std::max(m_eps_strong * std::pow(m_eps_decay, this->getLevelIndex()), 1e-4);
    this->Profile.tic("aggregates");
    computeAggregates();
    // Aggregates are rank-local, the decisions below are global so that all ranks take them together
    int total_aggregates = m_num_aggregates;

    if (distributed) { A.manager->global_reduce_sum(&total_aggregates); }

    // No strong connection at all (e.g. 27-point stencils, where |a_ij| / sqrt(a_ii a_jj) = 1/26):
    // keep decaying eps instead of ending the hierarchy on a large level.
    while (total_aggregates == 0 && m_level_eps > 1e-4 && m_eps_decay < 1.0)
    {
        m_level_eps = std::max(m_level_eps * m_eps_decay, 1e-4);
        computeAggregates();
        total_aggregates = m_num_aggregates;

        if (distributed) { A.manager->global_reduce_sum(&total_aggregates); }
    }

    this->Profile.toc("aggregates");
    // getNumCoarseVertices is compared with the number of rows of A in units of its block size.
    // Reporting the fine size when nothing can be aggregated ends the hierarchy here.
    const int coarse_unknowns = m_num_aggregates * coarseBlockSize();
    const int bdim = A.get_block_dimy();
    m_num_coarse_vertices = (total_aggregates > 0) ? (coarse_unknowns + bdim - 1) / bdim : A.get_num_rows();
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    const bool distributed = isDistributed();

    // As in classical AMG, P and R are kept when the level structure is reused (distributed: R is
    // formed inside the Galerkin product, after P's halo rows arrived)
    if (!this->isReuseLevel())
    {
        computeProlongationOperator();

        if (!distributed) { computeRestrictionOperator(); }
    }

    computeAOperator();
    AMG_Level<TConfig> *next = this->getNextLevel(typename Matrix<TConfig>::memory_space());
    Matrix<TConfig> &Ac = next->getA();

    if (distributed) { renumberCoarseLevel(); }

    Ac.copyAuxData(&this->getA());

    if (distributed)
    {
        // xc and bc carry the coarse halo
        int offset, size;
        Ac.getOffsetAndSizeForView(FULL, &offset, &size);
        this->m_next_level_size = size * Ac.get_block_dimy();
    }
    else
    {
        this->m_next_level_size = Ac.get_num_rows() * Ac.get_block_dimy();
    }

    SA_AMG_Level_Base<TConfig> *next_sa = dynamic_cast<SA_AMG_Level_Base<TConfig> *>(next);

    if (next_sa != NULL && !this->isReuseLevel())
    {
        next_sa->setInheritedStructure(coarseBlockSize(), m_nullspace_cols, m_coarse_nullspace);
    }

    // The scalar copy is only needed during setup
    m_A_scalar = Matrix<TConfig>();
    m_use_scalar_copy = false;
}

// The distributed classical post-processing of RAP: owned coarse rows renumbered into [interior |
// boundary], P's columns and R's rows along, the coarse halo rows and neighbours; the coarse
// nullspace (rows of coarse unknowns) follows the renumbering.
template <class T_Config>
void SA_AMG_Level_Base<T_Config>::renumberCoarseLevel()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA();
    Ac.set_initialized(0);
    Ac.manager->renumberMatrixOneRing(this->isReuseLevel());
    Ac.manager->renumber_P_R(P, R, A);
    Ac.manager->createOneRingHaloRows();
    Ac.manager->getComms()->set_neighbors(Ac.manager->num_neighbors());
    Ac.setView(OWNED);
    Ac.set_initialized(1);
    P.set_initialized(0);
    const int new_num_cols = P.get_num_nz() > 0 ? thrust_wrapper::reduce<TConfig::memSpace>(P.col_indices.begin(),
                             P.col_indices.end(), int(0), amgx::thrust::maximum<int>()) + 1 : 0;
    cudaCheckError();
    P.set_num_cols(std::max(new_num_cols, (int) Ac.get_num_rows()));
    P.set_initialized(1);
    const int nc = Ac.get_num_rows();
    const int cb = coarseBlockSize();

    if (this->isReuseLevel() || nc == 0) { return; }

    if (cb > 1)
    {
        IVector bad(1, 0);
        nodeContiguityKernel <<< salNumBlocks(nc / cb), SAL_BLOCK_SIZE>>>(Ac.manager->renumbering.raw(), nc / cb, cb, bad.raw());
        cudaCheckError();

        if (bad[0] != 0)
        {
            FatalError("Distributed SMOOTHED_AGGREGATION: the coarse renumbering split the unknowns of an aggregate",
                       AMGX_ERR_INTERNAL);
        }
    }

    if (!m_coarse_nullspace.empty())
    {
        DVector permuted(m_coarse_nullspace.size());
        permuteRowsKernel <<< salNumBlocks(nc), SAL_BLOCK_SIZE>>>(Ac.manager->renumbering.raw(), nc, cb,
                m_coarse_nullspace.raw(), permuted.raw());
        cudaCheckError();
        m_coarse_nullspace.swap(permuted);
    }
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

    if (P.is_matrix_distributed())
    {
        // Sizes of R's views for the masked restriction, through a stub manager
        if (R.manager == NULL) { R.manager = new DistributedManager<TConfig>(); }

        const int nrows_owned = P.manager->halo_offsets[0];
        const int nrows_full = P.manager->halo_offsets[P.manager->neighbors.size()];
        const int nz_full = R.row_offsets[nrows_full];
        const int nz_owned = R.row_offsets[nrows_owned];
        R.manager->setViewSizes(nrows_owned, nz_owned, nrows_owned, nz_owned, nrows_full, nz_full, R.get_num_rows(), R.get_num_nz());
    }

    R.set_initialized(1);
    this->Profile.toc("computeR");
}

// P and R are scalar while the vectors of a block level carry its block size: relabel them
// as scalar vectors around the products.
template <class T_Config>
void SA_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
    const bool distributed = !P.is_matrix_singleGPU();

    // rr holds the coarse halo and the rows of R sent to the neighbours
    if (distributed)
    {
        Matrix<TConfig> &Ac = this->getNextLevel(typename TConfig::MemSpace())->getA();
        rr.resize(std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]));
    }

    const int bx = r.get_block_dimx(), by = r.get_block_dimy();
    r.set_block_dimx(1);
    r.set_block_dimy(1);
    rr.set_block_dimx(1);
    rr.set_block_dimy(1);

    if (distributed) { multiply_with_mask_restriction(R, r, rr, P); }
    else { multiply(R, r, rr); }

    r.set_block_dimx(bx);
    r.set_block_dimy(by);
    rr.dirtybit = 1;
}

template <class T_Config>
void SA_AMG_Level_Base<T_Config>::prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp)
{
    e.dirtybit = 1;
    const bool distributed = !P.is_matrix_singleGPU();

    // e gets the coarse halo of P's columns
    if (distributed)
    {
        Matrix<TConfig> &Ac = this->getNextLevel(typename TConfig::MemSpace())->getA();
        const int e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]);

        if ((int) e.size() < e_size) { e.resize(e_size); }
    }

    const int bx = tmp.get_block_dimx(), by = tmp.get_block_dimy();
    tmp.set_block_dimx(1);
    tmp.set_block_dimy(1);
    e.set_block_dimx(1);
    e.set_block_dimy(1);

    if (distributed) { multiply_with_mask(P, e, tmp); }
    else if (e.size() > 0) { multiply(P, e, tmp); }

    tmp.set_block_dimx(bx);
    tmp.set_block_dimy(by);
    // the owned rows only when distributed (x keeps its halo), in block rows of x
    int owned_size = -1;

    if (P.is_matrix_distributed())
    {
        int owned_offset;
        P.manager->getOffsetAndSizeForView(OWNED, &owned_offset, &owned_size);
    }

    axpby(x, tmp, x, ValueTypeB(1), ValueTypeB(1), 0, owned_size);
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

    const int nodes = aggregates.size();
    std::stringstream ss;
    ss << "SA level " << this->getLevelIndex() << ": rows " << nodes * m_block_size
       << ", block " << m_block_size << ", nullspace " << m_nullspace_cols
       << ", aggregates " << m_num_aggregates
       << " (nodes min " << min_size << ", avg " << (m_num_aggregates > 0 ? double(nodes - m_num_removed) / m_num_aggregates : 0.0)
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
    const int b = this->m_block_size;
    // Work on a scalar matrix: BSR and external diagonal storage are expanded
    this->m_use_scalar_copy = A.get_block_dimy() > 1 || A.hasProps(DIAG);

    if (this->m_use_scalar_copy)
    {
        PointwiseAggregation<TConfig>::blockToScalar(A, this->m_A_scalar);
    }

    const Matrix<TConfig> &As = this->scalarA();

    if (b == 1)
    {
        {
            nvtxRange nr("sa_strength");
            SymmetricScaledStrength<TConfig>::computeStrongConnections(As, this->m_level_eps, this->m_strong, this->m_filtered_diag);
        }
        nvtxRange nr("sa_mis2");
        MIS2Aggregates<TConfig>::compute(As, this->m_strong, this->m_aggregates, this->m_num_aggregates, this->m_num_removed, this->m_mis_index_priority);
    }
    else
    {
        // Aggregate nodes on the point matrix, then expand the strong connections to the unknowns
        Matrix<TConfig> Ap;
        typename Base::BVector strong_points;
        typename Base::MVector point_diag;
        PointwiseAggregation<TConfig>::buildPointMatrix(As, b, Ap);
        SymmetricScaledStrength<TConfig>::computeStrongConnections(Ap, this->m_level_eps, strong_points, point_diag);
        MIS2Aggregates<TConfig>::compute(Ap, strong_points, this->m_aggregates, this->m_num_aggregates, this->m_num_removed, this->m_mis_index_priority);
        PointwiseAggregation<TConfig>::expandStrongConnections(As, b, Ap, strong_points, this->m_strong, this->m_filtered_diag);

        // The point matrix leaves the halo couplings out: their strength by the scalar criterion
        if (this->isDistributed())
        {
            SymmetricScaledStrength<TConfig>::patchHaloStrength(As, this->m_level_eps, this->m_strong, this->m_filtered_diag);
        }
    }

    // Each aggregate needs at least as many unknowns as nullspace vectors for its QR
    if (this->m_nullspace_cols > 0)
    {
        this->m_num_removed += PointwiseAggregation<TConfig>::removeSmallAggregates(this->m_aggregates, this->m_num_aggregates,
                               b, this->m_nullspace_cols);
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeProlongationOperator()
{
    this->Profile.tic("computeP");
    void *wk = this->amg->getCsrWorkspace();

    if (wk == NULL)
    {
        wk = CSR_Multiply<TConfig>::csr_workspace_create(*(this->amg->m_cfg), this->amg->m_cfg_scope);
        this->amg->setCsrWorkspace(wk);
    }

    const int b = this->m_block_size;
    const int K = this->m_nullspace_cols;
    // S = D_f^{-1} A^F
    Matrix<TConfig> S;
    nvtxRange *nr = new nvtxRange("sa_filtered_operator");
    SymmetricScaledStrength<TConfig>::buildFilteredScaledOperator(this->scalarA(), this->m_strong, this->m_filtered_diag, S);
    delete nr;
    this->m_strong.clear();
    this->m_strong.shrink_to_fit();
    this->m_filtered_diag.clear();
    this->m_filtered_diag.shrink_to_fit();
    // omega = relax * 4 / (3 rho(D_f^{-1} A^F))
    nr = new nvtxRange("sa_spectral_radius");
    const bool distributed = this->isDistributed();
    this->m_rho = SAProlongation<TConfig>::estimateSpectralRadius(S, this->m_power_iters, distributed ? &this->scalarA() : NULL);
    delete nr;
    nr = new nvtxRange("sa_tentative");
    this->m_omega = (this->m_rho > 0.0) ? this->m_relax * 4.0 / (3.0 * this->m_rho) : 0.0;
    // Tentative prolongator and coarse nullspace (explicit for blocks, to preserve it in the truncation)
    Matrix<TConfig> P_tent;

    if (K > 0)
    {
        SAProlongation<TConfig>::buildTentativeProlongatorNullspace(this->m_aggregates, this->m_num_aggregates, b, K,
                this->m_nullspace, P_tent, this->m_coarse_nullspace);
    }
    else if (b > 1)
    {
        SAProlongation<TConfig>::buildTentativeProlongatorBlock(this->m_aggregates, this->m_num_aggregates, b, P_tent);
        SAProlongation<TConfig>::blockIdentityNullspace(this->m_num_aggregates, b, this->m_coarse_nullspace);
    }
    else
    {
        SAProlongation<TConfig>::buildTentativeProlongator(this->m_aggregates, this->m_num_aggregates, P_tent);
        this->m_coarse_nullspace.clear();
    }

    delete nr;
    int nnz_P_untruncated = 0;

    if (distributed)
    {
        computeProlongationDistributed(S, P_tent, wk, nnz_P_untruncated);
    }
    else
    {
        // P = (I - omega S) P_tent
        nr = new nvtxRange("sa_smooth_p");
        SAProlongation<TConfig>::smoothProlongator(S, this->m_omega, P_tent, this->P, wk);
        delete nr;
        nr = new nvtxRange("sa_truncate");
        nnz_P_untruncated = this->P.get_num_nz();

        if (this->m_drop_tol > 0.0 && this->P.get_num_rows() > 0)
        {
            if (K == 0 && b == 1)
            {
                // Rescaling the kept entries to the row sum preserves the constant
                SAProlongation<TConfig>::truncateRowSum(this->P, this->m_drop_tol);
            }
            else if (this->coarseBlockSize() <= SAProlongation<TConfig>::MAX_TRUNCATION_K)
            {
                SAProlongation<TConfig>::truncateNullspace(this->P, this->m_drop_tol, this->coarseBlockSize(), this->m_coarse_nullspace);
            }
        }

        delete nr;
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

// Distributed prolongator. Aggregates are rank-local, so the tentative rows of the halo nodes carry
// only the neighbours' coarse columns: they are exchanged at a fixed width W (global columns and
// values, plus the coarse nullspace rows of their columns for the truncation), P_tent is extended to
// the FULL fine rows with the halo coarse columns numbered after the owned ones, smoothed and
// truncated locally, and P's manager is built from its global columns.
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeProlongationDistributed(
    Matrix<TConfig> &S, const Matrix<TConfig> &P_tent, void *wk, int &nnz_untruncated)
{
    typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type TConfig64;
    typedef Vector<TConfig64> I64Vector;
    typedef typename Base::DVector DVector;
    typedef typename Base::MVector MVector;
    typedef typename Base::IVector IVector;
    typedef typename Base::ValueTypeA ValueTypeA;
    typedef typename Base::IndexType IndexType;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &P = this->P;
    const int K = this->m_nullspace_cols;
    const int CB = (K > 0 || this->m_block_size > 1) ? this->coarseBlockSize() : 0;
    const int W = K > 0 ? K : 1;
    const int n = A.get_num_rows();
    const int n_coarse = P_tent.get_num_cols();
    int offset, full;
    A.getOffsetAndSizeForView(FULL, &offset, &full);
    // Coarse partition offsets (all-gather of the owned coarse counts)
    DistributedArranger<TConfig> prep;
    prep.initialize_manager(A, P, n_coarse);
    const int64_t base = P.manager->base_index();
    // Fixed-width tentative rows of the FULL fine rows
    nvtxRange *nr = new nvtxRange("sa_tentative_halo");
    I64Vector gcol((size_t) full * W, -1);
    MVector gval((size_t) full * W, ValueTypeA(0));
    DVector gbc((size_t) full * W * std::max(CB, 1), 0.0);

    if (n > 0)
    {
        packTentativeKernel <<< salNumBlocks(n), SAL_BLOCK_SIZE>>>(P_tent.row_offsets.raw(), P_tent.col_indices.raw(),
                P_tent.values.raw(), n, W, CB, CB > 0 ? this->m_coarse_nullspace.raw() : (const double *) NULL, base,
                gcol.raw(), gval.raw(), gbc.raw());
        cudaCheckError();
    }

    gcol.set_block_dimx(1);
    gcol.set_block_dimy(W);
    gcol.dirtybit = 1;
    A.manager->exchange_halo(gcol, gcol.tag);
    gval.set_block_dimx(1);
    gval.set_block_dimy(W);
    gval.dirtybit = 1;
    A.manager->exchange_halo(gval, gval.tag);

    if (CB > 0)
    {
        gbc.set_block_dimx(1);
        gbc.set_block_dimy(W * CB);
        gbc.dirtybit = 1;
        A.manager->exchange_halo(gbc, gbc.tag);
    }

    // Sorted global ids of the halo coarse columns
    I64Vector halo((size_t)(full - n) * W);
    amgx::thrust::copy(gcol.begin() + (size_t) n * W, gcol.begin() + (size_t) full * W, halo.begin());
    halo.resize(amgx::thrust::remove_if(halo.begin(), halo.end(), halo_column(base, base + n_coarse)) - halo.begin());
    thrust_wrapper::sort<AMGX_device>(halo.begin(), halo.end());
    halo.resize(amgx::thrust::unique(halo.begin(), halo.end()) - halo.begin());
    cudaCheckError();
    const int num_halo = halo.size();
    // P_tent over the FULL rows, local columns
    Matrix<TConfig> P_ext;
    IVector counts(full + 1, 0);

    if (full > 0)
    {
        extendedRowCountKernel <<< salNumBlocks(full), SAL_BLOCK_SIZE>>>(gcol.raw(), full, W, counts.raw());
        cudaCheckError();
    }

    thrust_wrapper::exclusive_scan<AMGX_device>(counts.begin(), counts.end(), counts.begin());
    cudaCheckError();
    P_ext.addProps(CSR);
    P_ext.resize(full, n_coarse + num_halo, counts[full], 1, 1, 1);
    thrust_wrapper::copy<AMGX_device>(counts.begin(), counts.end(), P_ext.row_offsets.begin());

    if (full > 0)
    {
        extendedFillKernel <<< salNumBlocks(full), SAL_BLOCK_SIZE>>>(gcol.raw(), gval.raw(), full, W, base, n_coarse,
                halo.raw(), num_halo, P_ext.row_offsets.raw(), P_ext.col_indices.raw(), P_ext.values.raw());
        cudaCheckError();
    }

    P_ext.computeDiagonal();
    P_ext.set_initialized(1);
    // Coarse nullspace rows of all local coarse columns, for the truncation
    DVector bc_ext;

    if (CB > 0)
    {
        bc_ext.resize((size_t)(n_coarse + num_halo) * CB, 0.0);
        amgx::thrust::copy(this->m_coarse_nullspace.begin(), this->m_coarse_nullspace.begin() + (size_t) n_coarse * CB, bc_ext.begin());

        if (full > n)
        {
            haloNullspaceKernel <<< salNumBlocks(full - n), SAL_BLOCK_SIZE>>>(gcol.raw(), gbc.raw(), n, full, W, CB, base,
                    n_coarse, halo.raw(), num_halo, bc_ext.raw());
        }

        cudaCheckError();
    }

    delete nr;
    // P = (I - omega S) P_tent on the owned rows
    nr = new nvtxRange("sa_smooth_p");
    S.set_initialized(0);
    S.set_num_cols(full);
    S.set_initialized(1);
    Matrix<TConfig> P_loc;
    SAProlongation<TConfig>::smoothProlongator(S, this->m_omega, P_ext, P_loc, wk);
    delete nr;
    nr = new nvtxRange("sa_truncate");
    nnz_untruncated = P_loc.get_num_nz();

    if (this->m_drop_tol > 0.0 && P_loc.get_num_rows() > 0)
    {
        if (CB == 0)
        {
            SAProlongation<TConfig>::truncateRowSum(P_loc, this->m_drop_tol);
        }
        else if (CB <= SAProlongation<TConfig>::MAX_TRUNCATION_K)
        {
            SAProlongation<TConfig>::truncateNullspace(P_loc, this->m_drop_tol, CB, bc_ext);
        }
    }

    // Truncation drops entries one by one: the unknowns of an aggregate would then get different
    // patterns in R A P, and the interior/boundary renumbering of the coarse level could separate them
    if (CB > 1 && P_loc.get_num_rows() > 0)
    {
        // the halo columns come in whole aggregates (tentative rows carry all of them), aligned to CB
        if (n_coarse % CB != 0 || num_halo % CB != 0)
        {
            FatalError("Distributed SMOOTHED_AGGREGATION: coarse columns are not whole aggregates", AMGX_ERR_INTERNAL);
        }

        P_loc.set_initialized(0);
        P_loc.sortByRowAndColumn();
        IVector pad_counts(n + 1, 0);
        padBlocksKernel <<< salNumBlocks(n), SAL_BLOCK_SIZE>>>(P_loc.row_offsets.raw(), P_loc.col_indices.raw(),
                P_loc.values.raw(), n, CB, pad_counts.raw(), (const IndexType *) NULL, (IndexType *) NULL, (ValueTypeA *) NULL);
        cudaCheckError();
        thrust_wrapper::exclusive_scan<AMGX_device>(pad_counts.begin(), pad_counts.end(), pad_counts.begin());
        cudaCheckError();
        Matrix<TConfig> P_pad;
        P_pad.addProps(CSR);
        P_pad.resize(n, P_loc.get_num_cols(), pad_counts[n], 1, 1, 1);
        thrust_wrapper::copy<AMGX_device>(pad_counts.begin(), pad_counts.end(), P_pad.row_offsets.begin());
        padBlocksKernel <<< salNumBlocks(n), SAL_BLOCK_SIZE>>>(P_loc.row_offsets.raw(), P_loc.col_indices.raw(),
                P_loc.values.raw(), n, CB, (int *) NULL, P_pad.row_offsets.raw(), P_pad.col_indices.raw(), P_pad.values.raw());
        cudaCheckError();
        P_loc.swap(P_pad);
    }

    delete nr;
    // P with the coarse manager: neighbours, halo numbering and B2L maps from the global columns
    const int nnz = P_loc.get_num_nz();
    I64Vector gcols_P(std::max(nnz, 1));

    if (nnz > 0)
    {
        globalColumnsKernel <<< salNumBlocks(nnz), SAL_BLOCK_SIZE>>>(P_loc.col_indices.raw(), nnz, n_coarse, base,
                halo.raw(), gcols_P.raw());
        cudaCheckError();
    }

    P.set_initialized(0);
    P.addProps(CSR);
    P.resize(n, n_coarse + num_halo, nnz, 1, 1, 1);
    P.row_offsets.swap(P_loc.row_offsets);
    P.col_indices.swap(P_loc.col_indices);
    P.values.swap(P_loc.values);
    P.set_num_nz(nnz);
    prep.initialize_manager_from_global_col_indices(P, gcols_P);
    prep.createRowsLists(P, true);
    P.set_initialized(0);
    P.set_num_cols(P.manager->halo_offsets[P.manager->neighbors.size()]);
    P.computeDiagonal();
    P.set_initialized(1);
}

// Numeric-only Galerkin rebuild: on with structure reuse, single GPU, unless numeric_rebuild=0
template <class TConfig>
static bool useNumericGalerkin(AMG_Config *cfg, const std::string &scope, const Matrix<TConfig> &A)
{
    return cfg->AMG_Config::template getParameter<int>("structure_reuse_levels", scope) != 0 &&
           cfg->AMG_Config::template getParameter<int>("numeric_rebuild", scope) != 0 && A.is_matrix_singleGPU();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator()
{
    this->Profile.tic("computeA");
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA();
    void *wk = this->amg->getCsrWorkspace();

    if (wk == NULL)
    {
        wk = CSR_Multiply<TConfig>::csr_workspace_create(*(this->amg->m_cfg), this->amg->m_cfg_scope);
        this->amg->setCsrWorkspace(wk);
    }

    // On reuse levels the coefficients of A changed since the scalar copy was made (or it was freed)
    if (A.get_block_dimy() > 1 || A.hasProps(DIAG))
    {
        PointwiseAggregation<TConfig>::blockToScalar(A, this->m_A_scalar);
        this->m_use_scalar_copy = true;
    }

    Ac.set_initialized(0);
    Ac.addProps(CSR);
    Ac.set_block_dimx(1);
    Ac.set_block_dimy(1);

    if (this->isDistributed())
    {
        computeAOperatorDistributed(wk);
        this->Profile.toc("computeA");
        return;
    }

    this->R.set_initialized(0);
    this->R.addProps(CSR);
    this->R.set_initialized(1);
    this->P.set_initialized(0);
    this->P.addProps(CSR);
    this->P.set_initialized(1);
    if (useNumericGalerkin(this->amg->m_cfg, this->amg->m_cfg_scope, A))
    {
        if (!(this->isReuseLevel() && this->m_galerkin.refill(this->R, this->scalarA(), this->P, Ac)))
        {
            this->m_galerkin.build(this->R, this->scalarA(), this->P, Ac, wk);
        }
    }
    else
    {
        this->m_galerkin.clear();
        CSR_Multiply<TConfig>::csr_galerkin_product(this->R, this->scalarA(), this->P, Ac,
                NULL, NULL, NULL, NULL, NULL, NULL, wk);
    }

    Ac.set_initialized(1);
    this->Profile.toc("computeA");
}

// Distributed Galerkin product (the CLASSICAL computeAOperator_1x1_distributed): P's rows of the
// fine halo come from the neighbours, R = P^T over the owned fine rows, the rows of R A P that
// belong to the neighbours' coarse unknowns are sent to them and added there.
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void SA_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperatorDistributed(void *wk)
{
    typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type TConfig64;
    typedef Vector<TConfig64> I64Vector;
    typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type TConfigI_h;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type TConfig64_h;
    typedef Vector<TConfigI_h> IVector_h;
    typedef Vector<TConfig64_h> I64Vector_h;
    typedef typename Base::IVector IVector;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &P = this->P;
    Matrix<TConfig> &RAP = this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA();
    const int num_parts = A.manager->get_num_partitions();
    const int my_rank = A.manager->global_id();
    A.setView(OWNED);
    const int num_owned_coarse_pts = P.manager->halo_offsets[0];
    const int num_owned_fine_pts = A.manager->halo_offsets[0];

    if (RAP.manager == NULL) { RAP.manager = new DistributedManager<TConfig>(); }

    RAP.manager->A = &RAP;
    RAP.manager->setComms(A.manager->getComms());
    RAP.manager->set_global_id(my_rank);
    RAP.manager->set_num_partitions(num_parts);
    RAP.manager->part_offsets_h = P.manager->part_offsets_h;
    RAP.manager->part_offsets = P.manager->part_offsets;
    RAP.manager->set_base_index(RAP.manager->part_offsets_h[my_rank]);
    RAP.manager->set_index_range(num_owned_coarse_pts);
    RAP.manager->num_rows_global = RAP.manager->part_offsets_h[num_parts];
    // P's manager stays as it is: the exchanges below work on copies
    IVector_h P_neighbors = P.manager->neighbors;
    I64Vector_h P_halo_ranges_h = P.manager->halo_ranges_h;
    I64Vector P_halo_ranges = P.manager->halo_ranges;
    RAP.manager->local_to_global_map = P.manager->local_to_global_map;
    IVector_h P_halo_offsets = P.manager->halo_offsets;
    DistributedArranger<TConfig> prep;
    prep.exchange_halo_rows_P(A, P, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges,
                              P_halo_offsets, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts,
                              RAP.manager->part_offsets_h[my_rank]);
    cudaCheckError();
    // The Galerkin product sees the owned fine rows of P
    P.set_initialized(0);
    P.set_num_rows(num_owned_fine_pts);
    P.addProps(CSR);
    P.set_initialized(1);

    if (!this->isReuseLevel())
    {
        this->R.set_initialized(0);
        this->R.addProps(CSR);
        this->computeRestrictionOperator();
        this->R.set_initialized(1);
    }

    Matrix<TConfig> RAP_full;
    RAP_full.set_initialized(0);
    CSR_Multiply<TConfig>::csr_galerkin_product(this->R, A, P, RAP_full, NULL, NULL, NULL, NULL, NULL, NULL, wk);
    RAP_full.set_initialized(1);
    prep.exchange_RAP_ext(RAP, RAP_full, A, P, P_halo_offsets, RAP.manager->local_to_global_map, P_neighbors,
                          P_halo_ranges_h, P_halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets,
                          num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank], wk);
    // Drop the halo columns only the neighbours' rows referenced, and their local_to_global entries
    const int nrow = RAP.get_num_rows();
    const int nl2g = RAP.get_num_cols() - nrow;

    if (nl2g > 0)
    {
        IVector flags(nl2g + 1, 0);
        I64Vector l2g(nl2g, 0);

        if (nrow > 0)
        {
            flagHaloColumnsKernel <<< salNumBlocks(nrow), SAL_BLOCK_SIZE>>>(nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), flags.raw());
            cudaCheckError();
        }

        thrust_wrapper::exclusive_scan<AMGX_device>(flags.begin(), flags.end(), flags.begin());
        cudaCheckError();
        const int new_nl2g = flags[nl2g];

        if (nrow > 0)
        {
            compressHaloColumnsKernel <<< salNumBlocks(nrow), SAL_BLOCK_SIZE>>>(nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), flags.raw());
            cudaCheckError();
        }

        RAP.set_initialized(0);
        RAP.set_num_cols(nrow + new_nl2g);
        RAP.set_initialized(1);
        compressLocalToGlobalKernel <<< salNumBlocks(nl2g), SAL_BLOCK_SIZE>>>(nl2g, RAP.manager->local_to_global_map.raw(), l2g.raw(), flags.raw());
        cudaCheckError();
        amgx::thrust::copy(l2g.begin(), l2g.begin() + new_nl2g, RAP.manager->local_to_global_map.begin());
        cudaCheckError();
        RAP.manager->local_to_global_map.resize(new_nl2g);
    }
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
