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
#include <aggregation/pointwise.h>
#include <aggregation/rigid_body_modes.h>
#include <thrust/functional.h>

#include <cmath>
#include <map>
#include <sstream>
#include <vector>

namespace amgx
{

namespace
{

// 3D linear elasticity on nx x ny x nz unit-size trilinear hexahedra, 3 x 3 blocks per node pair.
// Nodes on the x = 0 face get identity rows when clamp is set.
struct ElasticityBSR
{
    int n;
    std::vector<int> ro, ci;
    std::vector<double> v, coords;
    std::vector<char> clamped;
};

void hex8Stiffness(double h, double nu, double Ke[24][24])
{
    const double lam = nu / ((1 + nu) * (1 - 2 * nu)), mu = 1.0 / (2 * (1 + nu));
    double D[6][6] = {{0}};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) { D[i][j] = lam; }

        D[i][i] = lam + 2 * mu;
        D[i + 3][i + 3] = mu;
    }

    for (int i = 0; i < 24; i++)
        for (int j = 0; j < 24; j++) { Ke[i][j] = 0; }

    const double g = 1.0 / std::sqrt(3.0);

    for (int q = 0; q < 8; q++)
    {
        const double xi = (q & 1 ? g : -g), eta = (q & 2 ? g : -g), zeta = (q & 4 ? g : -g);
        double B[6][24] = {{0}};

        for (int a = 0; a < 8; a++)
        {
            const double sa = (a & 1) ? 1 : -1, ta = (a & 2) ? 1 : -1, ua = (a & 4) ? 1 : -1;
            const double dx = 0.25 * sa * (1 + ta * eta) * (1 + ua * zeta) / h;
            const double dy = 0.25 * ta * (1 + sa * xi) * (1 + ua * zeta) / h;
            const double dz = 0.25 * ua * (1 + sa * xi) * (1 + ta * eta) / h;
            B[0][3 * a] = dx; B[1][3 * a + 1] = dy; B[2][3 * a + 2] = dz;
            B[3][3 * a] = dy; B[3][3 * a + 1] = dx;
            B[4][3 * a + 1] = dz; B[4][3 * a + 2] = dy;
            B[5][3 * a] = dz; B[5][3 * a + 2] = dx;
        }

        const double w = h * h * h / 8;

        for (int i = 0; i < 24; i++)
            for (int j = 0; j < 24; j++)
            {
                double s = 0;

                for (int k = 0; k < 6; k++)
                    for (int l = 0; l < 6; l++) { s += B[k][i] * D[k][l] * B[l][j]; }

                Ke[i][j] += w * s;
            }
    }
}

ElasticityBSR elasticity3d(int nx, int ny, int nz, bool clamp)
{
    static double Ke[24][24];
    hex8Stiffness(1.0, 0.3, Ke);
    const int px = nx + 1, py = ny + 1, pz = nz + 1, nn = px * py * pz;
    std::vector<double> slots((size_t) nn * 27 * 9, 0.0);

    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++)
                for (int a = 0; a < 8; a++)
                    for (int c = 0; c < 8; c++)
                    {
                        const int na = (i + (a & 1)) + px * ((j + ((a >> 1) & 1)) + py * (k + ((a >> 2) & 1)));
                        const int off = ((c & 1) - (a & 1) + 1) * 9 + (((c >> 1) & 1) - ((a >> 1) & 1) + 1) * 3 + (((c >> 2) & 1) - ((a >> 2) & 1) + 1);

                        for (int r = 0; r < 3; r++)
                            for (int s = 0; s < 3; s++) { slots[((size_t) na * 27 + off) * 9 + r * 3 + s] += Ke[3 * a + r][3 * c + s]; }
                    }

    ElasticityBSR A;
    A.n = nn;
    A.ro.push_back(0);

    for (int k = 0; k < pz; k++)
        for (int j = 0; j < py; j++)
            for (int i = 0; i < px; i++)
            {
                A.coords.push_back(i);
                A.coords.push_back(j);
                A.coords.push_back(k);
                A.clamped.push_back(clamp && i == 0);
            }

    for (int k = 0; k < pz; k++)
        for (int j = 0; j < py; j++)
            for (int i = 0; i < px; i++)
            {
                const int n = i + px * (j + py * k);

                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            const int ii = i + dx, jj = j + dy, kk = k + dz;

                            if (ii < 0 || jj < 0 || kk < 0 || ii >= px || jj >= py || kk >= pz) { continue; }

                            const int m = ii + px * (jj + py * kk);

                            if ((A.clamped[n] || A.clamped[m]) && m != n) { continue; }

                            A.ci.push_back(m);

                            for (int q = 0; q < 9; q++)
                            {
                                A.v.push_back(A.clamped[n] ? (q % 4 == 0 ? 1.0 : 0.0) : slots[((size_t) n * 27 + (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) * 9 + q]);
                            }
                        }

                A.ro.push_back((int) A.ci.size());
            }

    return A;
}

// Scalar CSR with interleaved unknowns
void elasticityToScalar(const ElasticityBSR &A, std::vector<int> &ro, std::vector<int> &ci, std::vector<double> &v)
{
    ro.assign(1, 0);
    ci.clear();
    v.clear();

    for (int I = 0; I < A.n; I++)
        for (int r = 0; r < 3; r++)
        {
            for (int k = A.ro[I]; k < A.ro[I + 1]; k++)
                for (int c = 0; c < 3; c++)
                {
                    ci.push_back(A.ci[k] * 3 + c);
                    v.push_back(A.v[(size_t) k * 9 + r * 3 + c]);
                }

            ro.push_back((int) ci.size());
        }
}

} // anonymous namespace

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
int deviceAggregates(const Matrix_d &A_d, double eps, BVector_d &strong, MVector_d &d_f, IVector_h &agg_h, int &num_removed,
                     bool index_priority = false)
{
    IVector_d agg;
    int num_aggregates = 0;
    Strength_d::computeStrongConnections(A_d, eps, strong, d_f);
    Aggregates_d::compute(A_d, strong, agg, num_aggregates, num_removed, index_priority);
    agg_h = agg;
    return num_aggregates;
}

// 3 x 3 block matrix; with diag_prop the diagonal blocks are stored separately (DIAG property)
void loadBSR(Matrix_h &A, const ElasticityBSR &E, bool diag_prop)
{
    std::vector<int> ro(1, 0), ci;
    std::vector<double> v, d((size_t) E.n * 9, 0.0);

    for (int I = 0; I < E.n; I++)
    {
        for (int k = E.ro[I]; k < E.ro[I + 1]; k++)
        {
            if (diag_prop && E.ci[k] == I)
            {
                std::copy(E.v.begin() + (size_t) k * 9, E.v.begin() + (size_t)(k + 1) * 9, d.begin() + (size_t) I * 9);
                continue;
            }

            ci.push_back(E.ci[k]);
            v.insert(v.end(), E.v.begin() + (size_t) k * 9, E.v.begin() + (size_t)(k + 1) * 9);
        }

        ro.push_back((int) ci.size());
    }

    A.set_initialized(0);
    A.addProps(CSR);

    if (diag_prop) { A.addProps(DIAG); }

    A.resize(E.n, E.n, (int) ci.size(), 3, 3, 1);

    for (int i = 0; i <= E.n; i++) { A.row_offsets[i] = ro[i]; }

    for (size_t k = 0; k < ci.size(); k++) { A.col_indices[k] = ci[k]; }

    for (size_t k = 0; k < v.size(); k++) { A.values[k] = ValueTypeA(v[k]); }

    if (diag_prop)
        for (size_t k = 0; k < d.size(); k++) { A.values[ci.size() * 9 + k] = ValueTypeA(d[k]); }

    A.computeDiagonal();
    A.set_initialized(1);
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
    // aggregation_mis_priority=INDEX: the roots are the greedy MIS(2) in index order (AMGCL's
    // plain_aggregates seeds); roots are numbered in index order, so the k-th root is in aggregate k
    Matrix_h G_h;
    generatePoissonForTest(G_h, 1, 0, 7, 11, 9, 5);
    Matrix_d G_d = G_h;
    const int ng = G_h.get_num_rows();
    const int naggr_index = this->deviceAggregates(G_d, 0.08, strong, d_f, agg, removed, true);
    std::vector<int> dist(ng, 3), roots;

    for (int i = 0; i < ng; i++)
    {
        if (dist[i] <= 2) { continue; }

        roots.push_back(i);
        dist[i] = 0;

        for (int k = G_h.row_offsets[i]; k < G_h.row_offsets[i + 1]; k++)
        {
            const int j = G_h.col_indices[k];

            if (j == i) { continue; }

            dist[j] = std::min(dist[j], 1);

            for (int kk = G_h.row_offsets[j]; kk < G_h.row_offsets[j + 1]; kk++) { dist[G_h.col_indices[kk]] = std::min(dist[G_h.col_indices[kk]], 2); }
        }
    }

    UNITTEST_ASSERT_EQUAL(naggr_index, (int) roots.size());

    for (size_t r = 0; r < roots.size(); r++) { UNITTEST_ASSERT_EQUAL(agg[roots[r]], (int) r); }

    for (int i = 0; i < ng; i++) { UNITTEST_ASSERT_TRUE(agg[i] >= 0); }
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

// ------------------------------------------------------------------------------------------------
// P0.5: rigid body modes are orthonormal and in the kernel of a free-floating stiffness matrix
DECLARE_UNITTEST_BEGIN_EXTD(SARigidBodyModes, SmoothedAggregationTests_Base<T_Config>);

void run()
{
    const ElasticityBSR E = elasticity3d(5, 4, 3, false);
    const size_t n = (size_t) E.n * 3;
    std::vector<double> modes(6 * n);
    UNITTEST_ASSERT_EQUAL(AMGX_generate_rigid_body_modes(3, E.n, &E.coords[0], &modes[0]), AMGX_RC_OK);
    double orthogonality = 0.0, energy = 0.0;

    for (int k = 0; k < 6; k++)
    {
        const double *u = &modes[k * n];

        for (int l = 0; l < 6; l++)
        {
            double dot = 0.0;

            for (size_t i = 0; i < n; i++) { dot += u[i] * modes[l * n + i]; }

            orthogonality = std::max(orthogonality, std::fabs(dot - (k == l ? 1.0 : 0.0)));
        }

        double uKu = 0.0;

        for (int I = 0; I < E.n; I++)
            for (int r = 0; r < 3; r++)
            {
                double Ku = 0.0;

                for (int q = E.ro[I]; q < E.ro[I + 1]; q++)
                    for (int c = 0; c < 3; c++) { Ku += E.v[(size_t) q * 9 + r * 3 + c] * u[E.ci[q] * 3 + c]; }

                uKu += u[I * 3 + r] * Ku;
            }

        energy = std::max(energy, std::fabs(uKu));
    }

    UNITTEST_ASSERT_TRUE(orthogonality < 1e-12);
    UNITTEST_ASSERT_TRUE(energy < 1e-10);
    // 2D: 3 modes; unsupported dimensions are rejected
    std::vector<double> c2(8), m2(3 * 8);

    for (int i = 0; i < 4; i++) { c2[2 * i] = i % 2; c2[2 * i + 1] = i / 2; }

    UNITTEST_ASSERT_EQUAL(AMGX_generate_rigid_body_modes(2, 4, &c2[0], &m2[0]), AMGX_RC_OK);
    UNITTEST_ASSERT_TRUE(AMGX_generate_rigid_body_modes(4, 4, &c2[0], &m2[0]) != AMGX_RC_OK);
}

DECLARE_UNITTEST_END(SARigidBodyModes)

SARigidBodyModes<TemplateMode<AMGX_mode_dDDI>::Type> SARigidBodyModes_dDDI;

// ------------------------------------------------------------------------------------------------
// P0.3 / P0.11: BSR to scalar expansion, point matrix and expanded strong connections
DECLARE_UNITTEST_BEGIN_EXTD(SAPointwiseAggregation, SmoothedAggregationTests_Base<T_Config>);

typedef aggregation::PointwiseAggregation<TConfig_d> Pointwise_d;

void run()
{
    const ElasticityBSR E = elasticity3d(6, 4, 3, true);
    std::vector<int> ro, ci;
    std::vector<double> v;
    elasticityToScalar(E, ro, ci, v);
    Matrix_h S_h;
    this->loadCSR(S_h, E.n * 3, E.n * 3, ro, ci, v);
    typename SAPointwiseAggregation<T_Config>::SparseRows ref = this->toRows(S_h);

    // Expansion of BSR, with the diagonal inside and stored separately
    for (int diag_prop = 0; diag_prop < 2; diag_prop++)
    {
        Matrix_h B_h;
        this->loadBSR(B_h, E, diag_prop != 0);
        Matrix_d B_d = B_h, S_d;
        Pointwise_d::blockToScalar(B_d, S_d);
        Matrix_h Sx_h = S_d;
        UNITTEST_ASSERT_EQUAL(Sx_h.get_num_rows(), E.n * 3);
        UNITTEST_ASSERT_TRUE(this->maxDiff(this->toRows(Sx_h), ref) == 0.0);
    }

    // Point matrix: max |a_ij| over each 3 x 3 block
    Matrix_d S_d = S_h, Ap_d;
    Pointwise_d::buildPointMatrix(S_d, 3, Ap_d);
    Matrix_h Ap_h = Ap_d;
    typename SAPointwiseAggregation<T_Config>::SparseRows ap_ref(E.n);

    for (int i = 0; i < S_h.get_num_rows(); i++)
        for (int k = S_h.row_offsets[i]; k < S_h.row_offsets[i + 1]; k++)
        {
            double &m = ap_ref[i / 3][S_h.col_indices[k] / 3];
            m = std::max(m, std::fabs((double) S_h.values[k]));
        }

    UNITTEST_ASSERT_TRUE(this->maxDiff(this->toRows(Ap_h), ap_ref) == 0.0);
    // Unknowns of a node are coupled as strongly as their nodes
    typename SAPointwiseAggregation<T_Config>::BVector_d strong_points, strong;
    typename SAPointwiseAggregation<T_Config>::MVector_d point_diag, d_f;
    SAPointwiseAggregation<T_Config>::Strength_d::computeStrongConnections(Ap_d, 0.08, strong_points, point_diag);
    Pointwise_d::expandStrongConnections(S_d, 3, Ap_d, strong_points, strong, d_f);
    typename SAPointwiseAggregation<T_Config>::BVector_h sp_h = strong_points, s_h = strong;
    std::map<std::pair<int, int>, bool> sp_map;

    for (int I = 0; I < Ap_h.get_num_rows(); I++)
        for (int k = Ap_h.row_offsets[I]; k < Ap_h.row_offsets[I + 1]; k++) { sp_map[std::make_pair(I, (int) Ap_h.col_indices[k])] = sp_h[k]; }

    int mismatches = 0;

    for (int i = 0; i < S_h.get_num_rows(); i++)
        for (int k = S_h.row_offsets[i]; k < S_h.row_offsets[i + 1]; k++)
        {
            const int j = S_h.col_indices[k];
            const bool expected = (i != j) && (i / 3 == j / 3 || sp_map[std::make_pair(i / 3, j / 3)]);

            if (bool(s_h[k]) != expected) { mismatches++; }
        }

    UNITTEST_ASSERT_EQUAL(mismatches, 0);
}

DECLARE_UNITTEST_END(SAPointwiseAggregation)

SAPointwiseAggregation<TemplateMode<AMGX_mode_dDDI>::Type> SAPointwiseAggregation_dDDI;
SAPointwiseAggregation<TemplateMode<AMGX_mode_dFFI>::Type> SAPointwiseAggregation_dFFI;

// ------------------------------------------------------------------------------------------------
// P0.4 / P0.8: nullspace tentative prolongator (QR per aggregate) and nullspace-preserving truncation
DECLARE_UNITTEST_BEGIN_EXTD(SANullspaceProlongator, SmoothedAggregationTests_Base<T_Config>);

typedef aggregation::PointwiseAggregation<TConfig_d> Pointwise_d;
typedef typename aggregation::SAProlongation<TConfig_d>::DVector DVector_d;
typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecDouble>::Type> DVector_h;

// max_i |(P Bc)_i - target_i| over the rows of P
double nullspaceError(const Matrix_h &P, const DVector_h &Bc, int K, const std::vector<double> &target)
{
    double err = 0.0;

    for (int i = 0; i < P.get_num_rows(); i++)
        for (int c = 0; c < K; c++)
        {
            double s = 0.0;

            for (int k = P.row_offsets[i]; k < P.row_offsets[i + 1]; k++) { s += (double) P.values[k] * Bc[(size_t) P.col_indices[k] * K + c]; }

            err = std::max(err, std::fabs(s - target[(size_t) i * K + c]));
        }

    return err;
}

void run()
{
    const int K = 6;
    const ElasticityBSR E = elasticity3d(8, 4, 4, true);
    const int n = E.n * 3;
    std::vector<int> ro, ci;
    std::vector<double> v, modes((size_t) K * n), B((size_t) n * K);
    elasticityToScalar(E, ro, ci, v);
    aggregation::rigidBodyModes(3, E.n, &E.coords[0], &modes[0]);

    for (int i = 0; i < n; i++)
        for (int c = 0; c < K; c++) { B[(size_t) i * K + c] = modes[(size_t) c * n + i]; }

    Matrix_h A_h;
    this->loadCSR(A_h, n, n, ro, ci, v);
    Matrix_d A_d = A_h, Ap, S, P_tent, P;
    typename SANullspaceProlongator<T_Config>::BVector_d strong_points, strong;
    typename SANullspaceProlongator<T_Config>::MVector_d point_diag, d_f;
    typename SANullspaceProlongator<T_Config>::IVector_d agg;
    int naggr = 0, removed = 0;
    Pointwise_d::buildPointMatrix(A_d, 3, Ap);
    SANullspaceProlongator<T_Config>::Strength_d::computeStrongConnections(Ap, 0.08, strong_points, point_diag);
    SANullspaceProlongator<T_Config>::Aggregates_d::compute(Ap, strong_points, agg, naggr, removed);
    removed += Pointwise_d::removeSmallAggregates(agg, naggr, 3, K);
    Pointwise_d::expandStrongConnections(A_d, 3, Ap, strong_points, strong, d_f);
    UNITTEST_ASSERT_TRUE(naggr > 1);
    DVector_d B_d((size_t) n * K), Bc_d;
    amgx::thrust::copy(B.begin(), B.end(), B_d.begin());
    SANullspaceProlongator<T_Config>::Prolongation_d::buildTentativeProlongatorNullspace(agg, naggr, 3, K, B_d, P_tent, Bc_d);
    Matrix_h Pt_h = P_tent;
    DVector_h Bc_h = Bc_d;
    IVector_h agg_h = agg;
    UNITTEST_ASSERT_EQUAL(Pt_h.get_num_cols(), naggr * K);
    // P_tent^T P_tent = I: the columns of each aggregate are orthonormal and aggregates are disjoint
    std::vector<double> PtP((size_t) naggr * K * K, 0.0);
    std::vector<double> target((size_t) n * K, 0.0);
    const double tol = sizeof(ValueTypeA) == 4 ? 1e-5 : 1e-14;

    for (int i = 0; i < n; i++)
    {
        const int len = Pt_h.row_offsets[i + 1] - Pt_h.row_offsets[i];
        UNITTEST_ASSERT_EQUAL(len, agg_h[i / 3] >= 0 ? K : 0);

        for (int k = Pt_h.row_offsets[i]; k < Pt_h.row_offsets[i + 1]; k++)
            for (int l = Pt_h.row_offsets[i]; l < Pt_h.row_offsets[i + 1]; l++)
            {
                const int a = Pt_h.col_indices[k] / K;
                UNITTEST_ASSERT_EQUAL(a, Pt_h.col_indices[l] / K);
                PtP[((size_t) a * K + Pt_h.col_indices[k] % K) * K + Pt_h.col_indices[l] % K] += (double) Pt_h.values[k] * Pt_h.values[l];
            }

        if (agg_h[i / 3] >= 0)
            for (int c = 0; c < K; c++) { target[(size_t) i * K + c] = B[(size_t) i * K + c]; }
    }

    double orth = 0.0;

    for (int a = 0; a < naggr; a++)
        for (int r = 0; r < K; r++)
            for (int c = 0; c < K; c++) { orth = std::max(orth, std::fabs(PtP[((size_t) a * K + r) * K + c] - (r == c ? 1.0 : 0.0))); }

    UNITTEST_ASSERT_TRUE(orth < tol);
    // P_tent Bc reproduces B on the aggregated unknowns
    UNITTEST_ASSERT_TRUE(this->nullspaceError(Pt_h, Bc_h, K, target) < (sizeof(ValueTypeA) == 4 ? 1e-4 : 1e-12));
    // Smoothed P, then truncation keeps P Bc unchanged row by row
    SANullspaceProlongator<T_Config>::Strength_d::buildFilteredScaledOperator(A_d, strong, d_f, S);
    const double rho = SANullspaceProlongator<T_Config>::Prolongation_d::estimateSpectralRadius(S, 0);
    AMG_Config cfg;
    void *wk = CSR_Multiply<TConfig_d>::csr_workspace_create(cfg, "default");
    SANullspaceProlongator<T_Config>::Prolongation_d::smoothProlongator(S, 0.67 * 4.0 / (3.0 * rho), P_tent, P, wk);
    CSR_Multiply<TConfig_d>::csr_workspace_delete(wk);
    Matrix_h P_h = P;
    std::vector<double> PBc((size_t) n * K, 0.0);

    for (int i = 0; i < n; i++)
        for (int k = P_h.row_offsets[i]; k < P_h.row_offsets[i + 1]; k++)
            for (int c = 0; c < K; c++) { PBc[(size_t) i * K + c] += (double) P_h.values[k] * Bc_h[(size_t) P_h.col_indices[k] * K + c]; }

    SANullspaceProlongator<T_Config>::Prolongation_d::truncateNullspace(P, 0.1, K, Bc_d);
    Matrix_h Ptr_h = P;
    UNITTEST_ASSERT_TRUE(Ptr_h.get_num_nz() < P_h.get_num_nz());
    UNITTEST_ASSERT_TRUE(this->nullspaceError(Ptr_h, Bc_h, K, PBc) < (sizeof(ValueTypeA) == 4 ? 1e-3 : 1e-10));
}

DECLARE_UNITTEST_END(SANullspaceProlongator)

SANullspaceProlongator<TemplateMode<AMGX_mode_dDDI>::Type> SANullspaceProlongator_dDDI;
SANullspaceProlongator<TemplateMode<AMGX_mode_dFFI>::Type> SANullspaceProlongator_dFFI;

// ------------------------------------------------------------------------------------------------
// Stage 2 milestone: clamped 3D elasticity with rigid body modes through the C API, as BSR and as
// a scalar matrix with aggregation_block_size = 3
DECLARE_UNITTEST_BEGIN_EXTD(SAElasticity, SmoothedAggregationTests_Base<T_Config>);

int solve(const ElasticityBSR &E, bool block, bool nullspace)
{
    const int n = E.n * 3;
    std::vector<int> ro, ci;
    std::vector<double> v, modes((size_t) 6 * n), b_vec(n, 1.0), x_vec(n, 0.0);

    if (block) { ro = E.ro; ci = E.ci; v = E.v; }
    else { elasticityToScalar(E, ro, ci, v); }

    for (int I = 0; I < E.n; I++)
        if (E.clamped[I])
            for (int r = 0; r < 3; r++) { b_vec[I * 3 + r] = 0.0; }

    std::stringstream params;
    params << "config_version=2, determinism_flag=1, solver(main)=PCG, main:max_iters=200, main:tolerance=1e-8,"
           << " main:convergence=RELATIVE_INI_CORE, main:norm=L2, main:monitor_residual=1, main:preconditioner(amg)=AMG,"
           << " amg:algorithm=SMOOTHED_AGGREGATION, amg:max_iters=1, amg:cycle=V, amg:smoother(gs)=MULTICOLOR_GS,"
           << " gs:symmetric_GS=1, amg:presweeps=1, amg:postsweeps=1, amg:coarse_solver=DENSE_LU_SOLVER,"
           << " amg:dense_lu_num_rows=64, amg:aggregation_block_size=" << (block ? 1 : 3);
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, params.str().c_str()), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&b, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&x, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_OK);
    const int bs = block ? 3 : 1;
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n / bs, (int) ci.size(), bs, bs, &ro[0], &ci[0], &v[0], NULL), AMGX_OK);

    if (nullspace)
    {
        UNITTEST_ASSERT_EQUAL(AMGX_generate_rigid_body_modes(3, E.n, &E.coords[0], &modes[0]), AMGX_RC_OK);
        const double *ptrs[6];

        for (int k = 0; k < 6; k++) { ptrs[k] = &modes[(size_t) k * n]; }

        UNITTEST_ASSERT_EQUAL(AMGX_matrix_set_nullspace(A, 6, ptrs), AMGX_RC_OK);
    }

    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(b, n / bs, bs, &b_vec[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(x, n / bs, bs, &x_vec[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_solve_with_0_initial_guess(solver, b, x), AMGX_OK);
    AMGX_SOLVE_STATUS status;
    int iters = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &iters), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_SUCCESS);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(x), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(b), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_OK);
    return iters;
}

void run()
{
    const ElasticityBSR E = elasticity3d(24, 6, 6, true);
    const int scalar_rbm = solve(E, false, true);
    const int block_rbm = solve(E, true, true);
    const int block_plain = solve(E, true, false);
    UNITTEST_ASSERT_TRUE(scalar_rbm < 30);
    UNITTEST_ASSERT_TRUE(block_rbm < 40);
    UNITTEST_ASSERT_TRUE(block_rbm < block_plain);
}

DECLARE_UNITTEST_END(SAElasticity)

SAElasticity<TemplateMode<AMGX_mode_dDDI>::Type> SAElasticity_dDDI;

} // namespace amgx
