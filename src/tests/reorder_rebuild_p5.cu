// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <amgx_c.h>
#include "test_utils.h"
#include <matrix.h>
#include <reorder/rcm.h>
#include <numeric_galerkin.h>
#include <csr_multiply.h>
#include <transpose.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

namespace amgx
{

namespace
{

typedef std::vector<std::map<int, double> > Rows;

// 7-point Laplacian on nx x ny x nz with the nodes relabelled by label (identity if empty)
Rows poisson3d(int nx, int ny, int nz, const std::vector<int> &label)
{
    const int n = nx * ny * nz;
    Rows R(n);

    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                const int i = x + nx * (y + ny * z), li = label.empty() ? i : label[i];
                const int nb[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
                R[li][li] = 6.0;

                for (int f = 0; f < 6; f++)
                {
                    const int X = x + nb[f][0], Y = y + nb[f][1], Z = z + nb[f][2];

                    if (X < 0 || X >= nx || Y < 0 || Y >= ny || Z < 0 || Z >= nz) { continue; }

                    const int j = X + nx * (Y + ny * Z);
                    R[li][label.empty() ? j : label[j]] = -1.0 + 0.01 * ((i + j) % 7);
                }
            }

    return R;
}

std::vector<int> shuffled(int n, unsigned seed)
{
    std::vector<int> p(n);

    for (int i = 0; i < n; i++) { p[i] = i; }

    srand(seed);

    for (int i = n - 1; i > 0; i--) { std::swap(p[i], p[rand() % (i + 1)]); }

    return p;
}

void toCSR(const Rows &A, std::vector<int> &ro, std::vector<int> &ci, std::vector<double> &v)
{
    ro.assign(1, 0);
    ci.clear();
    v.clear();

    for (size_t i = 0; i < A.size(); i++)
    {
        for (std::map<int, double>::const_iterator it = A[i].begin(); it != A[i].end(); ++it)
        {
            ci.push_back(it->first);
            v.push_back(it->second);
        }

        ro.push_back((int) ci.size());
    }
}

int hostBandwidth(const Rows &R)
{
    int b = 0;

    for (size_t i = 0; i < R.size(); i++)
        for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it) { b = std::max(b, std::abs(it->first - (int) i)); }

    return b;
}

} // anonymous namespace

DECLARE_UNITTEST_BEGIN(ReorderP5Tests_Base);

typedef reorder::RCM<TConfig> RCMOps;
typedef typename RCMOps::IVector IVector_d;

// Scalar, or b x b blocks scaled from the scalar entries (diagonal blocks SPD); with external
// diagonal storage when diag_prop
void load(Matrix_h &A, const Rows &rows, int b = 1, bool diag_prop = false)
{
    std::vector<int> ro, ci;
    std::vector<double> v;
    const int n = rows.size();
    Rows off = rows;

    if (diag_prop)
        for (int i = 0; i < n; i++) { off[i].erase(i); }

    toCSR(off, ro, ci, v);
    A.set_initialized(0);
    A.addProps(CSR);

    if (diag_prop) { A.addProps(DIAG); }

    A.resize(n, n, (int) ci.size(), b, b, 1);

    for (int i = 0; i <= n; i++) { A.row_offsets[i] = ro[i]; }

    for (size_t k = 0; k < ci.size(); k++)
    {
        A.col_indices[k] = ci[k];

        for (int q = 0; q < b * b; q++) { A.values[k * b * b + q] = ValueTypeA(v[k] * (q % (b + 1) == 0 ? 1.0 : 0.1 * (1 + q))); }
    }

    if (diag_prop)
        for (int i = 0; i < n; i++)
            for (int q = 0; q < b * b; q++) { A.values[(ci.size() + i) * b * b + q] = ValueTypeA(rows[i].at(i) * (q % (b + 1) == 0 ? 1.0 : 0.05)); }
    else
        for (int q = 0; q < b * b; q++) { A.values[ci.size() * b * b + q] = ValueTypeA(0); }

    A.computeDiagonal();
    A.set_initialized(1);
}

std::vector<int> permutationOf(const Rows &R)
{
    Matrix_h A_h;
    load(A_h, R);
    Matrix_d A = A_h;
    IVector_d perm;
    RCMOps::permutation(A, perm);
    std::vector<int> p(perm.size());
    amgx::thrust::copy(perm.begin(), perm.end(), p.begin());
    return p;
}

bool isPermutation(const std::vector<int> &p)
{
    std::vector<int> seen(p.size(), 0);

    for (size_t i = 0; i < p.size(); i++)
    {
        if (p[i] < 0 || p[i] >= (int) p.size() || seen[p[i]]) { return false; }

        seen[p[i]] = 1;
    }

    return true;
}

Rows permuted(const Rows &R, const std::vector<int> &p)
{
    std::vector<int> ip(p.size());

    for (size_t i = 0; i < p.size(); i++) { ip[p[i]] = i; }

    Rows B(R.size());

    for (size_t i = 0; i < R.size(); i++)
        for (std::map<int, double>::const_iterator it = R[p[i]].begin(); it != R[p[i]].end(); ++it) { B[i][ip[it->first]] = it->second; }

    return B;
}

// C API solve in dDDI (scalar, or block b with the scalar rows replicated per component);
// returns the iterations; replace != 0 scales the diagonal by replace and re-sets up before solving
int solve(const Rows &R, const std::string &params, double replace = 0.0, int block = 1)
{
    std::vector<int> ro, ci;
    std::vector<double> v;
    toCSR(R, ro, ci, v);
    const int n = R.size(), N = n * block, nnz = ci.size();
    // block b: A_IJ = a_IJ (I + 0.1 offdiag) per block, as load()
    std::vector<double> bv;

    for (int k = 0; k < nnz; k++)
        for (int q = 0; q < block * block; q++) { bv.push_back(v[k] * (q % (block + 1) == 0 ? 1.0 : (ci[k] == std::upper_bound(ro.begin(), ro.end(), k) - ro.begin() - 1 ? 0.02 : 0.0))); }

    std::vector<double> b(N), x(N, 0.0);

    for (int i = 0; i < N; i++) { b[i] = 1.0 + 0.3 * std::sin(0.11 * i); }

    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle bh, xh;
    AMGX_solver_handle solver;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, params.c_str()), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&bh, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&xh, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n, nnz, block, block, &ro[0], &ci[0], &bv[0], NULL), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, A), AMGX_OK);

    if (replace != 0.0)
    {
        // diagonal blocks scaled (stays SPD)
        for (int I = 0; I < n; I++)
            for (int k = ro[I]; k < ro[I + 1]; k++)
                if (ci[k] == I)
                    for (int q = 0; q < block * block; q++) { bv[(size_t) k * block * block + q] *= replace; }

        UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(A, n, nnz, &bv[0], NULL), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_resetup(solver, A), AMGX_OK);
    }

    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(bh, n, block, &b[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(xh, n, block, &x[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_solve(solver, bh, xh), AMGX_OK);
    AMGX_SOLVE_STATUS status;
    int iters = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &iters), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_SUCCESS);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_download(xh, &x[0]), AMGX_OK);
    // true residual in the original numbering
    double r2 = 0.0, b2 = 0.0;

    for (int I = 0; I < n; I++)
        for (int r = 0; r < block; r++)
        {
            double ax = 0.0;

            for (int k = ro[I]; k < ro[I + 1]; k++)
                for (int c = 0; c < block; c++) { ax += bv[(size_t) k * block * block + r * block + c] * x[ci[k] * block + c]; }

            const double d = b[I * block + r] - ax;
            r2 += d * d;
            b2 += b[I * block + r] * b[I * block + r];
        }

    std::cout << "  iterations " << iters << ", true residual " << std::sqrt(r2 / b2) << std::endl;
    UNITTEST_ASSERT_TRUE(std::sqrt(r2 / b2) < 1e-7);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(xh), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(bh), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_OK);
    return iters;
}

DECLARE_UNITTEST_END(ReorderP5Tests_Base);

// ------------------------------------------------------------------------------------------------
// P5.1: RCM returns a permutation; it orders a path with bandwidth 1, recovers a small bandwidth
// from a randomly relabelled grid, and numbers each connected component contiguously
DECLARE_UNITTEST_BEGIN_EXTD(RCMPermutation, ReorderP5Tests_Base<T_Config>);

void run()
{
    // path 0 - 1 - ... - 49, relabelled
    const int m = 50;
    const std::vector<int> lab = shuffled(m, 3);
    Rows path(m);

    for (int i = 0; i < m; i++)
    {
        path[lab[i]][lab[i]] = 2.0;

        if (i > 0) { path[lab[i]][lab[i - 1]] = -1.0; }

        if (i + 1 < m) { path[lab[i]][lab[i + 1]] = -1.0; }
    }

    std::vector<int> p = this->permutationOf(path);
    UNITTEST_ASSERT_TRUE(this->isPermutation(p));
    UNITTEST_ASSERT_EQUAL(hostBandwidth(this->permuted(path, p)), 1);
    // relabelled 3D grid: natural bandwidth nx * ny = 144
    const Rows grid = poisson3d(12, 12, 10, shuffled(1440, 5));
    p = this->permutationOf(grid);
    UNITTEST_ASSERT_TRUE(this->isPermutation(p));
    const int bw = hostBandwidth(this->permuted(grid, p));
    std::cout << "  relabelled grid bandwidth " << hostBandwidth(grid) << " -> " << bw << std::endl;
    UNITTEST_ASSERT_TRUE(bw <= 2 * 144);
    // two components (grids) interleaved by the labels, plus isolated nodes
    const Rows g1 = poisson3d(6, 5, 4, std::vector<int>()), g2 = poisson3d(5, 5, 5, std::vector<int>());
    const int n1 = g1.size(), n2 = g2.size(), n = n1 + n2 + 3;
    const std::vector<int> mix = shuffled(n, 9);
    Rows two(n);

    for (int i = 0; i < n1; i++)
        for (std::map<int, double>::const_iterator it = g1[i].begin(); it != g1[i].end(); ++it) { two[mix[i]][mix[it->first]] = it->second; }

    for (int i = 0; i < n2; i++)
        for (std::map<int, double>::const_iterator it = g2[i].begin(); it != g2[i].end(); ++it) { two[mix[n1 + i]][mix[n1 + it->first]] = it->second; }

    for (int i = n1 + n2; i < n; i++) { two[mix[i]][mix[i]] = 1.0; }

    p = this->permutationOf(two);
    UNITTEST_ASSERT_TRUE(this->isPermutation(p));
    // the new indices of each component form a contiguous range
    std::vector<int> comp(n, 2);

    for (int i = 0; i < n1; i++) { comp[mix[i]] = 0; }

    for (int i = 0; i < n2; i++) { comp[mix[n1 + i]] = 1; }

    int changes = 0;

    for (int i = 1; i < n; i++) { changes += comp[p[i]] != comp[p[i - 1]]; }

    UNITTEST_ASSERT_TRUE(changes <= 2);
}

DECLARE_UNITTEST_END(RCMPermutation)

RCMPermutation<TemplateMode<AMGX_mode_dDDI>::Type> RCMPermutation_dDDI;

// ------------------------------------------------------------------------------------------------
// P5.1: B = P A P^T entry by entry (scalar, 3x3 BSR, external diagonal), and the nnz_map refill
// after the values change
DECLARE_UNITTEST_BEGIN_EXTD(RCMPermuteMatrix, ReorderP5Tests_Base<T_Config>);

void run()
{
    const Rows R = poisson3d(7, 6, 5, shuffled(210, 11));
    const int n = R.size();

    for (int variant = 0; variant < 3; variant++)
    {
        const int b = variant == 1 ? 3 : 1;
        const bool diag_prop = variant == 2;
        Matrix_h A_h;
        this->load(A_h, R, b, diag_prop);

        for (int pass = 0; pass < 2; pass++)
        {
            Matrix_d A = A_h;
            typename reorder::RCM<TConfig>::IVector perm, iperm, map;
            Matrix_d B;
            reorder::RCM<TConfig>::permutation(A, perm);
            reorder::RCM<TConfig>::inverse(perm, iperm);
            reorder::RCM<TConfig>::permuteStructure(A, perm, iperm, B, map);

            if (pass == 1)
            {
                // new values in A, refilled through the map
                for (size_t k = 0; k < A_h.values.size(); k++) { A_h.values[k] = A_h.values[k] * ValueTypeA(1.5); }

                A = A_h;
                reorder::RCM<TConfig>::permuteValues(A, map, B);
            }

            Matrix_h B_h = B;
            std::vector<int> p(n);
            amgx::thrust::copy(perm.begin(), perm.end(), p.begin());
            UNITTEST_ASSERT_TRUE(this->isPermutation(p));
            UNITTEST_ASSERT_EQUAL(B_h.get_num_nz(), A_h.get_num_nz());
            UNITTEST_ASSERT_EQUAL(B_h.hasProps(DIAG), diag_prop);
            double err = 0.0;

            for (int i = 0; i < n; i++)
            {
                // B row i is A row p[i]
                std::map<int, int> a_cols;

                for (int k = A_h.row_offsets[p[i]]; k < A_h.row_offsets[p[i] + 1]; k++) { a_cols[A_h.col_indices[k]] = k; }

                UNITTEST_ASSERT_EQUAL(B_h.row_offsets[i + 1] - B_h.row_offsets[i], (int) a_cols.size());

                for (int k = B_h.row_offsets[i]; k < B_h.row_offsets[i + 1]; k++)
                {
                    const int j = B_h.col_indices[k];
                    UNITTEST_ASSERT_TRUE(a_cols.count(p[j]) == 1);

                    if (k > B_h.row_offsets[i]) { UNITTEST_ASSERT_TRUE(j > B_h.col_indices[k - 1]); }

                    for (int q = 0; q < b * b; q++) { err = std::max(err, std::fabs((double) B_h.values[k * b * b + q] - (double) A_h.values[a_cols[p[j]] * b * b + q])); }
                }

                if (diag_prop)
                    for (int q = 0; q < b * b; q++) { err = std::max(err, std::fabs((double) B_h.values[B_h.diag[i] * b * b + q] - (double) A_h.values[A_h.diag[p[i]] * b * b + q])); }
            }

            UNITTEST_ASSERT_TRUE(err == 0.0);
        }
    }
}

DECLARE_UNITTEST_END(RCMPermuteMatrix)

RCMPermuteMatrix<TemplateMode<AMGX_mode_dDDI>::Type> RCMPermuteMatrix_dDDI;
RCMPermuteMatrix<TemplateMode<AMGX_mode_dFFI>::Type> RCMPermuteMatrix_dFFI;

// ------------------------------------------------------------------------------------------------
// P5.1: reordering=RCM solves in the original numbering; permutation-invariant solvers keep their
// iteration counts; coefficient replacement plus resetup equals a fresh setup on the new values
DECLARE_UNITTEST_BEGIN_EXTD(RCMEndToEnd, ReorderP5Tests_Base<T_Config>);

void run()
{
    const Rows R = poisson3d(16, 14, 12, shuffled(16 * 14 * 12, 13));
    const std::string base = "config_version=2, determinism_flag=1, solver(main)=PCG, main:monitor_residual=1, main:max_iters=1000,"
                             " main:tolerance=1e-9, main:convergence=RELATIVE_INI_CORE, main:norm=L2,";
    const std::string jacobi = " main:preconditioner(j)=BLOCK_JACOBI, j:max_iters=1, j:relaxation_factor=1";
    const std::string sa = " main:preconditioner(amg)=AMG, amg:algorithm=SMOOTHED_AGGREGATION, amg:max_iters=1,"
                           " amg:smoother(s)=SPAI0, s:relaxation_factor=1, amg:max_levels=20";
    // Jacobi-PCG is invariant under symmetric permutations
    const int j_none = this->solve(R, base + jacobi);
    const int j_rcm = this->solve(R, base + jacobi + ", main:reordering=RCM");
    UNITTEST_ASSERT_TRUE(std::abs(j_none - j_rcm) <= 1);
    // AMG converges in both numberings
    const int a_none = this->solve(R, base + sa);
    const int a_rcm = this->solve(R, base + sa + ", main:reordering=RCM");
    UNITTEST_ASSERT_TRUE(a_rcm <= a_none + 5);
    // replace + resetup equals a fresh setup on the new values, with and without structure reuse
    Rows R2 = R;

    for (size_t i = 0; i < R2.size(); i++) { R2[i][i] *= 2.0; }

    const int fresh = this->solve(R2, base + sa + ", main:reordering=RCM");
    const int resetup = this->solve(R, base + sa + ", main:reordering=RCM", 2.0);
    UNITTEST_ASSERT_EQUAL(fresh, resetup);
    this->solve(R, base + sa + ", main:reordering=RCM, amg:structure_reuse_levels=-1", 2.0);
    // 3 x 3 blocks with SA
    this->solve(poisson3d(8, 8, 6, shuffled(384, 17)), base + sa + ", main:reordering=RCM", 0.0, 3);
}

DECLARE_UNITTEST_END(RCMEndToEnd)

RCMEndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> RCMEndToEnd_dDDI;

// ------------------------------------------------------------------------------------------------
// P5.2: after the coefficients of A change, the numeric-only refill of R A P equals a fresh
// csr_galerkin_product on the new values
DECLARE_UNITTEST_BEGIN_EXTD(NumericGalerkinRefill, ReorderP5Tests_Base<T_Config>);

void run()
{
    const Rows R_A = poisson3d(12, 11, 10, std::vector<int>());
    const int n = R_A.size(), nc = n / 6;
    // tall P: each fine row couples to 1-3 coarse columns
    Rows R_P(n);

    for (int i = 0; i < n; i++)
    {
        R_P[i][i % nc] = 1.0 + 0.1 * (i % 5);

        if (i % 3 != 0) { R_P[i][(i * 7 + 3) % nc] = 0.3; }

        if (i % 4 == 1) { R_P[i][(i * 13 + 5) % nc] = -0.2; }
    }

    Matrix_h A_h, P_h;
    this->load(A_h, R_A);
    std::vector<int> ro, ci;
    std::vector<double> v;
    toCSR(R_P, ro, ci, v);
    P_h.set_initialized(0);
    P_h.addProps(CSR);
    P_h.resize(n, nc, (int) ci.size(), 1);

    for (int i = 0; i <= n; i++) { P_h.row_offsets[i] = ro[i]; }

    for (size_t k = 0; k < ci.size(); k++) { P_h.col_indices[k] = ci[k]; P_h.values[k] = ValueTypeA(v[k]); }

    P_h.computeDiagonal();
    P_h.set_initialized(1);
    Matrix_d A = A_h, P = P_h, Rm, Ac, Ref;
    Rm.set_initialized(0);
    transpose(P, Rm);
    Rm.set_initialized(1);
    NumericGalerkin<TConfig> galerkin;
    galerkin.build(Rm, A, P, Ac, NULL);
    // new coefficients, same pattern
    for (size_t k = 0; k < A_h.values.size(); k++) { A_h.values[k] = A_h.values[k] * ValueTypeA(1.0 + 0.25 * (k % 4)); }

    A = A_h;
    UNITTEST_ASSERT_TRUE(galerkin.refill(Rm, A, P, Ac));
    void *wk = CSR_Multiply<TConfig>::csr_workspace_create();
    Ref.set_initialized(0);
    Ref.addProps(CSR);
    CSR_Multiply<TConfig>::csr_galerkin_product(Rm, A, P, Ref, NULL, NULL, NULL, NULL, NULL, NULL, wk);
    CSR_Multiply<TConfig>::csr_workspace_delete(wk);
    Matrix_h Ac_h = Ac, Ref_h = Ref;
    UNITTEST_ASSERT_EQUAL(Ac_h.get_num_nz(), Ref_h.get_num_nz());
    double err = 0.0, scale = 0.0;

    for (int i = 0; i < nc; i++)
    {
        std::map<int, double> ref;

        for (int k = Ref_h.row_offsets[i]; k < Ref_h.row_offsets[i + 1]; k++) { ref[Ref_h.col_indices[k]] += Ref_h.values[k]; }

        for (int k = Ac_h.row_offsets[i]; k < Ac_h.row_offsets[i + 1]; k++)
        {
            if (k > Ac_h.row_offsets[i]) { UNITTEST_ASSERT_TRUE(Ac_h.col_indices[k] > Ac_h.col_indices[k - 1]); }

            UNITTEST_ASSERT_TRUE(ref.count(Ac_h.col_indices[k]) == 1);
            err = std::max(err, std::fabs((double) Ac_h.values[k] - ref[Ac_h.col_indices[k]]));
            scale = std::max(scale, std::fabs(ref[Ac_h.col_indices[k]]));
        }
    }

    UNITTEST_ASSERT_TRUE(err <= (sizeof(ValueTypeA) == 4 ? 1e-5 : 1e-12) * scale);
    // a different pattern is refused
    Matrix_d A2 = A;
    A2.set_initialized(0);
    A2.set_num_nz(A.get_num_nz() - 1);
    UNITTEST_ASSERT_TRUE(!galerkin.refill(Rm, A2, P, Ac));
}

DECLARE_UNITTEST_END(NumericGalerkinRefill)

NumericGalerkinRefill<TemplateMode<AMGX_mode_dDDI>::Type> NumericGalerkinRefill_dDDI;
NumericGalerkinRefill<TemplateMode<AMGX_mode_dFFI>::Type> NumericGalerkinRefill_dFFI;

// ------------------------------------------------------------------------------------------------
// P5.2: with structure_reuse_levels=-1, replace + resetup cycles give the same iterations with the
// numeric-only Galerkin rebuild as with the full product (numeric_rebuild=0), for SA and classical
DECLARE_UNITTEST_BEGIN_EXTD(RebuildEndToEnd, ReorderP5Tests_Base<T_Config>);

void run()
{
    const Rows R = poisson3d(18, 16, 14, std::vector<int>());
    const std::string base = "config_version=2, determinism_flag=1, solver(main)=PCG, main:monitor_residual=1, main:max_iters=500,"
                             " main:tolerance=1e-9, main:convergence=RELATIVE_INI_CORE, main:norm=L2,"
                             " main:preconditioner(amg)=AMG, amg:max_iters=1, amg:max_levels=20, amg:structure_reuse_levels=-1,"
                             " amg:smoother(s)=BLOCK_JACOBI, s:relaxation_factor=0.8,";
    const char *hierarchies[] = {" amg:algorithm=SMOOTHED_AGGREGATION", " amg:algorithm=CLASSICAL, amg:selector=PMIS, amg:interpolator=D2"};

    for (int h = 0; h < 2; h++)
        for (int scale = 2; scale <= 4; scale++)
        {
            const int full = this->solve(R, base + hierarchies[h] + ", amg:numeric_rebuild=0", scale);
            const int numeric = this->solve(R, base + hierarchies[h] + ", amg:numeric_rebuild=1", scale);
            UNITTEST_ASSERT_EQUAL(full, numeric);
        }
}

DECLARE_UNITTEST_END(RebuildEndToEnd)

RebuildEndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> RebuildEndToEnd_dDDI;


} // namespace amgx
