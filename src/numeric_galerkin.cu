// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <numeric_galerkin.h>
#include <csr_multiply.h>
#include <amgx_timer.h>
#include <error.h>
#include <cutil.h>

namespace amgx
{

namespace
{

const int NG_WARPS_PER_BLOCK = 8;
const int NG_ROW_CAPACITY = 256;

__device__ __forceinline__ void ngAtomicAdd(double *p, double v) { atomicAdd(p, v); }
__device__ __forceinline__ void ngAtomicAdd(float *p, float v) { atomicAdd(p, v); }

// C(i, :) = sum_k A(i, k) B(k, :) into the sorted pattern of C. GROUP lanes per row (sized from
// the B row lengths, as csr_galerkin_product does): the group walks A's row, its lanes split each B
// row, and the products are accumulated with atomics, in shared memory on a copy of C's row for
// whole-warp groups (rows up to NG_ROW_CAPACITY), in global memory otherwise.
template <int GROUP, typename IndexType, typename ValueType>
__global__ void __launch_bounds__(NG_WARPS_PER_BLOCK * 32) numericProductKernel(const IndexType *a_ro, const IndexType *a_ci,
        const ValueType *a_v, const IndexType *b_ro, const IndexType *b_ci, const ValueType *b_v,
        const IndexType *c_ro, const IndexType *c_ci, ValueType *c_v, int n)
{
    __shared__ IndexType s_cols[NG_WARPS_PER_BLOCK][GROUP == 32 ? NG_ROW_CAPACITY : 1];
    __shared__ ValueType s_vals[NG_WARPS_PER_BLOCK][GROUP == 32 ? NG_ROW_CAPACITY : 1];
    const int groups_per_block = NG_WARPS_PER_BLOCK * 32 / GROUP;
    const int group = threadIdx.x / GROUP, lane = threadIdx.x % GROUP, warp = threadIdx.x / 32;

    for (int i = blockIdx.x * groups_per_block + group; i < n; i += gridDim.x * groups_per_block)
    {
        const IndexType c_begin = c_ro[i], length = c_ro[i + 1] - c_begin;
        const bool in_shared = GROUP == 32 && length <= NG_ROW_CAPACITY;
        const IndexType *cols = in_shared ? s_cols[warp] : c_ci + c_begin;
        ValueType *vals = in_shared ? s_vals[warp] : c_v + c_begin;

        for (IndexType q = lane; q < length; q += GROUP)
        {
            if (in_shared) { s_cols[warp][q] = c_ci[c_begin + q]; }

            vals[q] = ValueType(0);
        }

        __syncwarp();

        for (IndexType ka = a_ro[i]; ka < a_ro[i + 1]; ka++)
        {
            const IndexType k = a_ci[ka];
            const ValueType a = a_v[ka];

            for (IndexType kb = b_ro[k] + lane; kb < b_ro[k + 1]; kb += GROUP)
            {
                const IndexType j = b_ci[kb];
                IndexType lo = 0, hi = length;

                while (lo < hi)
                {
                    const IndexType mid = (lo + hi) / 2;

                    if (cols[mid] < j) { lo = mid + 1; }
                    else { hi = mid; }
                }

                ngAtomicAdd(&vals[lo], a * b_v[kb]);
            }
        }

        __syncwarp();

        if (in_shared)
            for (IndexType q = lane; q < length; q += GROUP) { c_v[c_begin + q] = s_vals[warp][q]; }

        __syncwarp();
    }
}

template <int GROUP, typename IndexType, typename ValueType>
void launchNumericProduct(const IndexType *a_ro, const IndexType *a_ci, const ValueType *a_v, const IndexType *b_ro,
                          const IndexType *b_ci, const ValueType *b_v, const IndexType *c_ro, const IndexType *c_ci,
                          ValueType *c_v, int n)
{
    const int groups_per_block = NG_WARPS_PER_BLOCK * 32 / GROUP;
    const int blocks = std::max(1, std::min(16384, (n + groups_per_block - 1) / groups_per_block));
    numericProductKernel<GROUP> <<< blocks, NG_WARPS_PER_BLOCK * 32, 0, amgx::thrust::global_thread_handle::get_stream()>>>(
        a_ro, a_ci, a_v, b_ro, b_ci, b_v, c_ro, c_ci, c_v, n);
    cudaCheckError();
}

// csr_multiply is declared for every build, the workspace only on device
template <class TConfig, AMGX_MemorySpace M = TConfig::memSpace>
struct Workspace
{
    static void *create() { return CSR_Multiply<TConfig>::csr_workspace_create(); }
    static void destroy(void *wk) { CSR_Multiply<TConfig>::csr_workspace_delete(wk); }
};

template <class TConfig>
struct Workspace<TConfig, AMGX_host>
{
    static void *create() { FatalError("NumericGalerkin is only implemented on device", AMGX_ERR_NOT_IMPLEMENTED); return NULL; }
    static void destroy(void *wk) {}
};

} // anonymous namespace

template <class T_Config>
void NumericGalerkin<T_Config>::numericProduct(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C)
{
    if (TConfig::memSpace != AMGX_device)
    {
        FatalError("NumericGalerkin is only implemented on device", AMGX_ERR_NOT_IMPLEMENTED);
    }

    const int n = A.get_num_rows();

    if (n == 0) { return; }

    // lanes per row from the average B row length (csr_galerkin_product's choice)
    const double avg_b = B.get_num_rows() > 0 ? (double) B.get_num_nz() / B.get_num_rows() : 0.0;
    const int g = avg_b < 2 ? 2 : avg_b <= 8 ? 4 : avg_b <= 16 ? 8 : 32;
#define AMGX_NG_LAUNCH(G) launchNumericProduct<G>(A.row_offsets.raw(), A.col_indices.raw(), A.values.raw(), \
        B.row_offsets.raw(), B.col_indices.raw(), B.values.raw(), C.row_offsets.raw(), C.col_indices.raw(), C.values.raw(), n)

    if (g == 2) { AMGX_NG_LAUNCH(2); }
    else if (g == 4) { AMGX_NG_LAUNCH(4); }
    else if (g == 8) { AMGX_NG_LAUNCH(8); }
    else { AMGX_NG_LAUNCH(32); }

#undef AMGX_NG_LAUNCH
}

template <class T_Config>
void NumericGalerkin<T_Config>::build(const Matrix<TConfig> &R, const Matrix<TConfig> &A, const Matrix<TConfig> &P,
                                      Matrix<TConfig> &Ac, void *workspace)
{
    nvtxRange nr("galerkin_build");
    void *wk = workspace != NULL ? workspace : Workspace<TConfig>::create();
    m_AP = Matrix<TConfig>();
    m_RAP = Matrix<TConfig>();
    m_AP.set_initialized(0);
    m_AP.addProps(CSR);
    CSR_Multiply<TConfig>::csr_multiply(A, P, m_AP, wk);
    m_AP.set_initialized(1);
    m_RAP.set_initialized(0);
    m_RAP.addProps(CSR);
    CSR_Multiply<TConfig>::csr_multiply(R, m_AP, m_RAP, wk);
    m_RAP.set_initialized(1);

    if (workspace == NULL) { Workspace<TConfig>::destroy(wk); }

    // Sorted patterns for the binary searches of refill()
    m_AP.set_initialized(0);
    m_AP.sortByRowAndColumn();
    m_AP.set_initialized(1);
    m_RAP.set_initialized(0);
    m_RAP.sortByRowAndColumn();
    m_RAP.set_initialized(1);
    m_a_nnz = A.get_num_nz();
    m_p_nnz = P.get_num_nz();
    m_r_nnz = R.get_num_nz();
    m_a_rows = A.get_num_rows();
    m_valid = true;
    writeCoarse(Ac);
}

template <class T_Config>
bool NumericGalerkin<T_Config>::refill(const Matrix<TConfig> &R, const Matrix<TConfig> &A, const Matrix<TConfig> &P,
                                       Matrix<TConfig> &Ac)
{
    if (!m_valid || A.get_num_nz() != m_a_nnz || P.get_num_nz() != m_p_nnz || R.get_num_nz() != m_r_nnz ||
            A.get_num_rows() != m_a_rows)
    {
        return false;
    }

    nvtxRange nr("galerkin_numeric_refill");
    numericProduct(A, P, m_AP);
    numericProduct(R, m_AP, m_RAP);
    writeCoarse(Ac);
    return true;
}

// Ac takes the pattern and values of R A P as csr_multiply writes its output (Ac keeps its other
// properties, e.g. its coloring, as on the existing reuse path)
template <class T_Config>
void NumericGalerkin<T_Config>::writeCoarse(Matrix<TConfig> &Ac)
{
    Ac.set_initialized(0);
    Ac.addProps(CSR);
    Ac.set_num_rows(m_RAP.get_num_rows());
    Ac.set_num_cols(m_RAP.get_num_cols());
    Ac.row_offsets = m_RAP.row_offsets;
    Ac.col_indices = m_RAP.col_indices;
    Ac.values = m_RAP.values;
    Ac.set_num_nz(m_RAP.get_num_nz());
    Ac.diag.resize(m_RAP.get_num_rows());
    Ac.setColsReorderedByColor(false);
    Ac.computeDiagonal();
    Ac.set_initialized(1);
}

#define AMGX_CASE_LINE(CASE) template class NumericGalerkin<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
