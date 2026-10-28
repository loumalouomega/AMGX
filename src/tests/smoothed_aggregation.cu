// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <amgx_c.h>
#include "amg_solver.h"
#include "test_utils.h"
#include <matrix.h>
#include <multiply.h>
#include <blas.h>
#include <norm.h>
#include <transpose.h>
#include <truncate.h>
#include <csr_multiply.h>
#include <aggregation/strength/symmetric_scaled.h>
#include <aggregation/selectors/mis2_aggregates.h>
#include <aggregation/sa_prolongation.h>
#include <thrust/functional.h>

#include <cmath>
#include <map>
#include <sstream>
#include <vector>

namespace amgx
{

DECLARE_UNITTEST_BEGIN(SmoothedAggregationTests_Base);

typedef aggregation::SymmetricScaledStrength<TConfig_d> Strength_d;
typedef aggregation::MIS2Aggregates<TConfig_d> Aggregates_d;
typedef aggregation::SAProlongation<TConfig_d> Prolongation_d;
typedef typename Strength_d::BVector BVector_d;
typedef typename Strength_d::MVector MVector_d;
typedef typename Strength_d::IVector IVector_d;
typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector_h;

double tol() { return sizeof(ValueTypeA) == 4 ? 1e-4 : 1e-11; }

void loadCSR(Matrix_h &A, int n, int m, const std::vector<int> &ro, const std::vector<int> &ci, const std::vector<double> &v)
{
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(n, m, (int) ci.size(), 1);

    for (int i = 0; i <= n; i++) { A.row_offsets[i] = ro[i]; }

    for (size_t k = 0; k < ci.size(); k++)
    {
        A.col_indices[k] = ci[k];
        A.values[k] = ValueTypeA(v[k]);
    }

    A.computeDiagonal();
    A.set_initialized(1);
}

// 7-point anisotropic diffusion with Dirichlet boundaries eliminated
void makeAniso7pt(Matrix_h &A, int nx, int ny, int nz, double kx, double ky, double kz)
{
    std::vector<int> ro(1, 0), ci;
    std::vector<double> v;

    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                const int i = x + nx * (y + ny * z);
                const int nb[6][3] = {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}}; // {dz, dy, dx}
                const double k[6] = {kz, ky, kx, kx, ky, kz};

                for (int d = 0; d < 6; d++)
                {
                    if (d == 3) { ci.push_back(i); v.push_back(2 * (kx + ky + kz)); }

                    const int xx = x + nb[d][2], yy = y + nb[d][1], zz = z + nb[d][0];

                    if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) { continue; }

                    ci.push_back(xx + nx * (yy + ny * zz));
                    v.push_back(-k[d]);
                }

                ro.push_back((int) ci.size());
            }

    loadCSR(A, nx * ny * nz, nx * ny * nz, ro, ci, v);
}

// 5-point graph Laplacian with random symmetric edge weights plus a small shift
void makeRandom5pt(Matrix_h &A, int nx, int ny)
{
    const int n = nx * ny;
    std::vector<double> wx(n, 0.0), wy(n, 0.0); // weight of edge to +x / +y neighbour
    srand(1234);

    for (int i = 0; i < n; i++)
    {
        wx[i] = 0.01 + 10.0 * rand() / (double) RAND_MAX;
        wy[i] = 0.01 + 10.0 * rand() / (double) RAND_MAX;
    }

    std::vector<int> ro(1, 0), ci;
    std::vector<double> v;

    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            const int i = x + nx * y;
            double d = 0.1;
            std::map<int, double> row;

            if (x > 0) { row[i - 1] = -wx[i - 1]; d += wx[i - 1]; }

            if (x < nx - 1) { row[i + 1] = -wx[i]; d += wx[i]; }

            if (y > 0) { row[i - nx] = -wy[i - nx]; d += wy[i - nx]; }

            if (y < ny - 1) { row[i + nx] = -wy[i]; d += wy[i]; }

            row[i] = d;

            for (std::map<int, double>::iterator it = row.begin(); it != row.end(); ++it)
            {
                ci.push_back(it->first);
                v.push_back(it->second);
            }

            ro.push_back((int) ci.size());
        }

    loadCSR(A, n, n, ro, ci, v);
}

void makeLaplace1D(Matrix_h &A, int n)
{
    std::vector<int> ro(1, 0), ci;
    std::vector<double> v;

    for (int i = 0; i < n; i++)
    {
        if (i > 0) { ci.push_back(i - 1); v.push_back(-1.0); }

        ci.push_back(i);
        v.push_back(2.0);

        if (i < n - 1) { ci.push_back(i + 1); v.push_back(-1.0); }

        ro.push_back((int) ci.size());
    }

    loadCSR(A, n, n, ro, ci, v);
}

// Host reference of the AMGCL strength criterion (amgcl/coarsening/plain_aggregates.hpp)
void referenceStrength(const Matrix_h &A, double eps, std::vector<bool> &strong, std::vector<double> &d_f)
{
    const int n = A.get_num_rows();
    std::vector<double> dia(n, 0.0);
    strong.assign(A.get_num_nz(), false);
    d_f.assign(n, 0.0);

    for (int i = 0; i < n; i++)
        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
            if (A.col_indices[k] == i) { dia[i] = A.values[k]; }

    for (int i = 0; i < n; i++)
        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
        {
            const int c = A.col_indices[k];
            const double v = A.values[k];
            strong[k] = (c != i) && (eps * eps * dia[i] * dia[c] < v * v);

            if (!strong[k]) { d_f[i] += v; }
        }
}

typedef std::vector<std::map<int, double> > SparseRows;

SparseRows toRows(const Matrix_h &M)
{
    SparseRows rows(M.get_num_rows());

    for (int i = 0; i < M.get_num_rows(); i++)
        for (int k = M.row_offsets[i]; k < M.row_offsets[i + 1]; k++)
        {
            rows[i][M.col_indices[k]] += M.values[k];
        }

    return rows;
}

// max_ij |X_ij - Y_ij| over the union of both patterns
double maxDiff(const SparseRows &X, const SparseRows &Y)
{
    double diff = 0.0;

    for (size_t i = 0; i < X.size(); i++)
    {
        for (std::map<int, double>::const_iterator it = X[i].begin(); it != X[i].end(); ++it)
        {
            std::map<int, double>::const_iterator jt = Y[i].find(it->first);
            diff = std::max(diff, std::fabs(it->second - (jt == Y[i].end() ? 0.0 : jt->second)));
        }

        for (std::map<int, double>::const_iterator jt = Y[i].begin(); jt != Y[i].end(); ++jt)
            if (X[i].find(jt->first) == X[i].end()) { diff = std::max(diff, std::fabs(jt->second)); }
    }

    return diff;
}

// Runs strength + MIS(2) aggregation on the device and returns the host aggregates
int deviceAggregates(const Matrix_d &A_d, double eps, BVector_d &strong, MVector_d &d_f, IVector_h &agg_h, int &num_removed)
{
    IVector_d agg;
    int num_aggregates = 0;
    Strength_d::computeStrongConnections(A_d, eps, strong, d_f);
    Aggregates_d::compute(A_d, strong, agg, num_aggregates, num_removed);
    agg_h = agg;
    return num_aggregates;
}

DECLARE_UNITTEST_END(SmoothedAggregationTests_Base);

// ------------------------------------------------------------------------------------------------
// P0.1: strong connections and lumped diagonal match the AMGCL criterion
DECLARE_UNITTEST_BEGIN_EXTD(SAStrengthFilter, SmoothedAggregationTests_Base<T_Config>);

void check(const Matrix_h &A_h, double eps)
{
    Matrix_d A_d = A_h;
    typename SAStrengthFilter<T_Config>::BVector_d strong;
    typename SAStrengthFilter<T_Config>::MVector_d d_f;
    SAStrengthFilter<T_Config>::Strength_d::computeStrongConnections(A_d, eps, strong, d_f);
    typename SAStrengthFilter<T_Config>::BVector_h strong_h = strong;
    MVector_h d_f_h = d_f;
    std::vector<bool> strong_ref;
    std::vector<double> d_f_ref;
    this->referenceStrength(A_h, eps, strong_ref, d_f_ref);
    int mismatches = 0;

    for (size_t k = 0; k < strong_ref.size(); k++)
        if (bool(strong_h[k]) != strong_ref[k]) { mismatches++; }

    UNITTEST_ASSERT_EQUAL(mismatches, 0);

    for (size_t i = 0; i < d_f_ref.size(); i++)
    {
        UNITTEST_ASSERT_EQUAL_TOL(double(d_f_h[i]), d_f_ref[i], this->tol() * std::max(1.0, std::fabs(d_f_ref[i])));
    }
}

void run()
{
    Matrix_h A;
    this->makeRandom5pt(A, 20, 20);
    check(A, 0.08);
    check(A, 0.25);
    this->makeAniso7pt(A, 10, 10, 6, 1.0, 1.0, 1e-3);
    check(A, 0.08);
}

DECLARE_UNITTEST_END(SAStrengthFilter)

SAStrengthFilter<TemplateMode<AMGX_mode_dDDI>::Type> SAStrengthFilter_dDDI;
SAStrengthFilter<TemplateMode<AMGX_mode_dFFI>::Type> SAStrengthFilter_dFFI;

// ------------------------------------------------------------------------------------------------
// P0.2: every non-isolated row is aggregated, deterministically, along strong couplings only
DECLARE_UNITTEST_BEGIN_EXTD(SAMIS2Aggregates, SmoothedAggregationTests_Base<T_Config>);

void run()
{
    // Anisotropic: z couplings are weak, aggregates must stay inside x-y planes
    const int nx = 12, ny = 12, nz = 6;
    Matrix_h A_h;
    this->makeAniso7pt(A_h, nx, ny, nz, 1.0, 1.0, 1e-3);
    Matrix_d A_d = A_h;
    typename SAMIS2Aggregates<T_Config>::BVector_d strong;
    typename SAMIS2Aggregates<T_Config>::MVector_d d_f;
    IVector_h agg, agg2;
    int removed = -1, removed2 = -1;
    const int naggr = this->deviceAggregates(A_d, 0.08, strong, d_f, agg, removed);
    const int naggr2 = this->deviceAggregates(A_d, 0.08, strong, d_f, agg2, removed2);
    UNITTEST_ASSERT_TRUE(naggr > 0);
    UNITTEST_ASSERT_EQUAL(removed, 0);
    UNITTEST_ASSERT_EQUAL(naggr, naggr2);
    std::vector<int> size(naggr, 0), plane(naggr, -1);
    int bad_plane = 0, bad_id = 0, not_deterministic = 0;

    for (int i = 0; i < nx * ny * nz; i++)
    {
        if (agg[i] != agg2[i]) { not_deterministic++; }

        if (agg[i] < 0 || agg[i] >= naggr) { bad_id++; continue; }

        size[agg[i]]++;
        const int z = i / (nx * ny);

        if (plane[agg[i]] < 0) { plane[agg[i]] = z; }
        else if (plane[agg[i]] != z) { bad_plane++; }
    }

    UNITTEST_ASSERT_EQUAL(bad_id, 0);
    UNITTEST_ASSERT_EQUAL(bad_plane, 0);
    UNITTEST_ASSERT_EQUAL(not_deterministic, 0);

    for (int a = 0; a < naggr; a++) { UNITTEST_ASSERT_TRUE(size[a] > 0); }

    // Isotropic Poisson: aggregates have at least 2 nodes (a root and its strong 1-ring)
    Matrix_h P_h;
    generatePoissonForTest(P_h, 1, 0, 7, 16, 16, 16);
    Matrix_d P_d = P_h;
    const int naggr3 = this->deviceAggregates(P_d, 0.08, strong, d_f, agg, removed);
    UNITTEST_ASSERT_EQUAL(removed, 0);
    std::vector<int> size3(naggr3, 0);

    for (int i = 0; i < P_h.get_num_rows(); i++) { size3[agg[i]]++; }

    for (int a = 0; a < naggr3; a++) { UNITTEST_ASSERT_TRUE(size3[a] >= 2); }

    // Rows without strong connections are removed
    Matrix_h D_h;
    std::vector<int> ro(1, 0), ci;
    std::vector<double> v;

    for (int i = 0; i < 8; i++) { ci.push_back(i); v.push_back(1.0); ro.push_back(i + 1); }

    this->loadCSR(D_h, 8, 8, ro, ci, v);
    Matrix_d D_d = D_h;
    UNITTEST_ASSERT_EQUAL(this->deviceAggregates(D_d, 0.08, strong, d_f, agg, removed), 0);
    UNITTEST_ASSERT_EQUAL(removed, 8);
}

DECLARE_UNITTEST_END(SAMIS2Aggregates)

SAMIS2Aggregates<TemplateMode<AMGX_mode_dDDI>::Type> SAMIS2Aggregates_dDDI;
SAMIS2Aggregates<TemplateMode<AMGX_mode_dFFI>::Type> SAMIS2Aggregates_dFFI;

// ------------------------------------------------------------------------------------------------
// P0.6: spectral radius of D^-1 A for the 1D Laplacian
DECLARE_UNITTEST_BEGIN_EXTD(SASpectralRadius, SmoothedAggregationTests_Base<T_Config>);

void run()
{
    const int n = 32;
    Matrix_h A_h;
    this->makeLaplace1D(A_h, n);
    Matrix_d A_d = A_h, S;
    typename SASpectralRadius<T_Config>::BVector_d strong;
    typename SASpectralRadius<T_Config>::MVector_d d_f;
    SASpectralRadius<T_Config>::Strength_d::computeStrongConnections(A_d, 0.08, strong, d_f);
    SASpectralRadius<T_Config>::Strength_d::buildFilteredScaledOperator(A_d, strong, d_f, S);
    const double exact = 1.0 + std::cos(M_PI / (n + 1)); // lambda_max(D^-1 A)
    const double gershgorin = SASpectralRadius<T_Config>::Prolongation_d::estimateSpectralRadius(S, 0);
    UNITTEST_ASSERT_EQUAL_TOL(gershgorin, 2.0, this->tol());
    const double power4 = SASpectralRadius<T_Config>::Prolongation_d::estimateSpectralRadius(S, 4);
    UNITTEST_ASSERT_TRUE(power4 > 0.0 && power4 <= gershgorin + this->tol());
    const double power = SASpectralRadius<T_Config>::Prolongation_d::estimateSpectralRadius(S, 500);
    UNITTEST_ASSERT_TRUE(std::fabs(power - exact) < 0.01 * exact);
}

DECLARE_UNITTEST_END(SASpectralRadius)

SASpectralRadius<TemplateMode<AMGX_mode_dDDI>::Type> SASpectralRadius_dDDI;
SASpectralRadius<TemplateMode<AMGX_mode_dFFI>::Type> SASpectralRadius_dFFI;

// ------------------------------------------------------------------------------------------------
// P0.7 / P0.8 / P0.9: smoothed prolongator, truncation and Galerkin product
DECLARE_UNITTEST_BEGIN_EXTD(SASmoothedProlongator, SmoothedAggregationTests_Base<T_Config>);

// Host port of the AMGCL fused smoothing loop (amgcl/coarsening/smoothed_aggregation.hpp)
typename SASmoothedProlongator<T_Config>::SparseRows referenceP(const Matrix_h &A, const std::vector<bool> &strong,
        const IVector_h &agg, double omega)
{
    typename SASmoothedProlongator<T_Config>::SparseRows P(A.get_num_rows());

    for (int i = 0; i < A.get_num_rows(); i++)
    {
        double dia = 0.0;

        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
            if (A.col_indices[k] == i || !strong[k]) { dia += A.values[k]; }

        if (dia != 0.0) { dia = -omega / dia; }

        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; k++)
        {
            const int c = A.col_indices[k];

            if (c != i && !strong[k]) { continue; }

            const double va = (c == i) ? 1.0 - omega : dia * A.values[k];

            if (agg[c] >= 0) { P[i][agg[c]] += va; }
        }
    }

    return P;
}

void buildP(const Matrix_d &A_d, double eps, double relax, Matrix_d &P, IVector_h &agg_h, std::vector<bool> &strong_ref, double &omega)
{
    typename SASmoothedProlongator<T_Config>::BVector_d strong;
    typename SASmoothedProlongator<T_Config>::MVector_d d_f;
    typename SASmoothedProlongator<T_Config>::IVector_d agg;
    int naggr = 0, removed = 0;
    SASmoothedProlongator<T_Config>::Strength_d::computeStrongConnections(A_d, eps, strong, d_f);
    SASmoothedProlongator<T_Config>::Aggregates_d::compute(A_d, strong, agg, naggr, removed);
    Matrix_d S, P_tent;
    SASmoothedProlongator<T_Config>::Strength_d::buildFilteredScaledOperator(A_d, strong, d_f, S);
    const double rho = SASmoothedProlongator<T_Config>::Prolongation_d::estimateSpectralRadius(S, 4);
    omega = relax * 4.0 / (3.0 * rho);
    SASmoothedProlongator<T_Config>::Prolongation_d::buildTentativeProlongator(agg, naggr, P_tent);
    AMG_Config cfg;
    void *wk = CSR_Multiply<TConfig_d>::csr_workspace_create(cfg, "default");
    SASmoothedProlongator<T_Config>::Prolongation_d::smoothProlongator(S, omega, P_tent, P, wk);
    CSR_Multiply<TConfig_d>::csr_workspace_delete(wk);
    agg_h = agg;
    typename SASmoothedProlongator<T_Config>::BVector_h strong_h = strong;
    strong_ref.assign(strong_h.size(), false);

    for (size_t k = 0; k < strong_h.size(); k++) { strong_ref[k] = strong_h[k]; }
}

void run()
{
    typedef typename SASmoothedProlongator<T_Config>::SparseRows SparseRows;
    // Smoothed P against the AMGCL loop on a random-coefficient matrix
    Matrix_h A_h;
    this->makeRandom5pt(A_h, 24, 24);
    Matrix_d A_d = A_h, P_d;
    IVector_h agg;
    std::vector<bool> strong;
    double omega = 0.0;
    buildP(A_d, 0.08, 0.67, P_d, agg, strong, omega);
    Matrix_h P_h = P_d;
    UNITTEST_ASSERT_TRUE(omega > 0.0);
    UNITTEST_ASSERT_TRUE(this->maxDiff(this->toRows(P_h), referenceP(A_h, strong, agg, omega)) < this->tol());
    // Constant vector is preserved on interior rows of the Poisson matrix (zero row sum).
    // With eps_strong = 0.08 the 27-point stencil has no strong connection (1/26 < 0.08), use 0.02.
    Matrix_h L_h;
    generatePoissonForTest(L_h, 1, 0, 27, 10, 10, 10);
    Matrix_d L_d = L_h, PL_d;
    buildP(L_d, 0.02, 0.67, PL_d, agg, strong, omega);
    Matrix_h PL_h = PL_d;
    int interior_errors = 0;

    for (int i = 0; i < L_h.get_num_rows(); i++)
    {
        double row_sum_A = 0.0, row_sum_P = 0.0;

        for (int k = L_h.row_offsets[i]; k < L_h.row_offsets[i + 1]; k++) { row_sum_A += L_h.values[k]; }

        for (int k = PL_h.row_offsets[i]; k < PL_h.row_offsets[i + 1]; k++) { row_sum_P += PL_h.values[k]; }

        if (std::fabs(row_sum_A) < 1e-12 && std::fabs(row_sum_P - 1.0) > this->tol()) { interior_errors++; }
    }

    UNITTEST_ASSERT_EQUAL(interior_errors, 0);
    // Truncation keeps the row sums and only drops entries below eps * row max
    const double eps_drop = 0.1;
    SparseRows P_full = this->toRows(PL_h);
    Matrix_d PT_d = PL_d;
    Truncate<TConfig_d>::truncateByFactor(PT_d, eps_drop, AMGX_TruncateByMaxCoefficient);
    Matrix_h PT_h = PT_d;
    SparseRows P_trunc = this->toRows(PT_h);
    UNITTEST_ASSERT_TRUE(PT_h.get_num_nz() < PL_h.get_num_nz());
    int truncation_errors = 0;

    for (size_t i = 0; i < P_full.size(); i++)
    {
        double sum_full = 0.0, sum_trunc = 0.0, row_max = 0.0;

        for (std::map<int, double>::iterator it = P_full[i].begin(); it != P_full[i].end(); ++it)
        {
            sum_full += it->second;
            row_max = std::max(row_max, std::fabs(it->second));
        }

        for (std::map<int, double>::iterator it = P_trunc[i].begin(); it != P_trunc[i].end(); ++it)
        {
            sum_trunc += it->second;

            if (std::fabs(P_full[i][it->first]) < eps_drop * row_max * (1.0 - 1e-6)) { truncation_errors++; }
        }

        if (std::fabs(sum_full - sum_trunc) > this->tol() * std::max(1.0, std::fabs(sum_full))) { truncation_errors++; }
    }

    UNITTEST_ASSERT_EQUAL(truncation_errors, 0);
    // Galerkin product R A P against a host reference
    Matrix_d R_d, Ac_d;
    PT_d.set_initialized(0);
    transpose(PT_d, R_d, PT_d.get_num_rows());
    PT_d.set_initialized(1);
    AMG_Config cfg;
    void *wk = CSR_Multiply<TConfig_d>::csr_workspace_create(cfg, "default");
    Ac_d.set_initialized(0);
    Ac_d.addProps(CSR);
    CSR_Multiply<TConfig_d>::csr_galerkin_product(R_d, L_d, PT_d, Ac_d, NULL, NULL, NULL, NULL, NULL, NULL, wk);
    CSR_Multiply<TConfig_d>::csr_workspace_delete(wk);
    Matrix_h Ac_h = Ac_d;
    SparseRows A_rows = this->toRows(L_h);
    SparseRows AP(A_rows.size()), Ac_ref(PT_h.get_num_cols());

    for (size_t i = 0; i < A_rows.size(); i++)
        for (std::map<int, double>::iterator it = A_rows[i].begin(); it != A_rows[i].end(); ++it)
            for (std::map<int, double>::iterator jt = P_trunc[it->first].begin(); jt != P_trunc[it->first].end(); ++jt)
            {
                AP[i][jt->first] += it->second * jt->second;
            }

    for (size_t i = 0; i < P_trunc.size(); i++)
        for (std::map<int, double>::iterator it = P_trunc[i].begin(); it != P_trunc[i].end(); ++it)
            for (std::map<int, double>::iterator jt = AP[i].begin(); jt != AP[i].end(); ++jt)
            {
                Ac_ref[it->first][jt->first] += it->second * jt->second;
            }

    UNITTEST_ASSERT_EQUAL(Ac_h.get_num_rows(), PT_h.get_num_cols());
    UNITTEST_ASSERT_TRUE(this->maxDiff(this->toRows(Ac_h), Ac_ref) < 10 * this->tol());
}

DECLARE_UNITTEST_END(SASmoothedProlongator)

SASmoothedProlongator<TemplateMode<AMGX_mode_dDDI>::Type> SASmoothedProlongator_dDDI;
SASmoothedProlongator<TemplateMode<AMGX_mode_dFFI>::Type> SASmoothedProlongator_dFFI;

// ------------------------------------------------------------------------------------------------
// End to end through the C API: PCG preconditioned by SMOOTHED_AGGREGATION, then a structure
// reuse resetup with new coefficients (P and R are kept, only Ac is recomputed)
DECLARE_UNITTEST_BEGIN_EXTD(SAEndToEnd, SmoothedAggregationTests_Base<T_Config>);

void run()
{
    Matrix_h A_h;
    generatePoissonForTest(A_h, 1, 0, 7, 32, 32, 32);
    const int n = A_h.get_num_rows(), nnz = A_h.get_num_nz();
    std::vector<int> row_offsets(A_h.row_offsets.begin(), A_h.row_offsets.end());
    std::vector<int> col_indices(A_h.col_indices.begin(), A_h.col_indices.begin() + nnz);
    std::vector<double> values(A_h.values.begin(), A_h.values.begin() + nnz);
    std::vector<double> b_vec(n, 1.0), x_vec(n, 0.0);
    const char *params = "config_version=2, determinism_flag=1, solver(main)=PCG, main:max_iters=100,"
                         " main:tolerance=1e-8, main:convergence=RELATIVE_INI_CORE, main:norm=L2, main:monitor_residual=1,"
                         " main:preconditioner(amg)=AMG, amg:algorithm=SMOOTHED_AGGREGATION, amg:max_iters=1, amg:cycle=V,"
                         " amg:smoother(sm)=BLOCK_JACOBI, sm:relaxation_factor=0.8, amg:presweeps=1, amg:postsweeps=1,"
                         " amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, amg:structure_reuse_levels=-1";
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, params), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n, nnz, 1, 1, &row_offsets[0], &col_indices[0], &values[0], NULL), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(b, n, 1, &b_vec[0]), AMGX_OK);

    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 0)
        {
            UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, A), AMGX_OK);
        }
        else
        {
            for (int k = 0; k < nnz; k++) { values[k] *= 3.0; }

            UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(A, n, nnz, &values[0], NULL), AMGX_OK);
            UNITTEST_ASSERT_EQUAL(AMGX_solver_resetup(solver, A), AMGX_OK);
        }

        UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(x, n, 1, &x_vec[0]), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_solve_with_0_initial_guess(solver, b, x), AMGX_OK);
        AMGX_SOLVE_STATUS status;
        int iters = 0;
        UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &iters), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_SUCCESS);
        UNITTEST_ASSERT_TRUE(iters > 1 && iters < 30);
        // True residual check on the host
        std::vector<double> sol(n);
        UNITTEST_ASSERT_EQUAL(AMGX_vector_download(x, &sol[0]), AMGX_OK);
        double r2 = 0.0, b2 = 0.0;

        for (int i = 0; i < n; i++)
        {
            double ax = 0.0;

            for (int k = row_offsets[i]; k < row_offsets[i + 1]; k++) { ax += values[k] * sol[col_indices[k]]; }

            r2 += (b_vec[i] - ax) * (b_vec[i] - ax);
            b2 += b_vec[i] * b_vec[i];
        }

        UNITTEST_ASSERT_TRUE(std::sqrt(r2 / b2) < 1e-7);
    }

    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(x), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(b), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_OK);
}

DECLARE_UNITTEST_END(SAEndToEnd)

SAEndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> SAEndToEnd_dDDI;

} // namespace amgx
