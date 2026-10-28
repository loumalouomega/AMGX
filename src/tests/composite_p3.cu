// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <amgx_c.h>
#include "test_utils.h"
#include <matrix.h>
#include <solvers/solver.h>
#include <solvers/cpr_solver.h>
#include <solvers/schur_solver.h>
#include <solvers/deflation_solver.h>

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

// 2D two-phase (p, Sw) Jacobian, unknowns interleaved per cell with pressure first, equations
// (water, oil): upwinded TPFA fluxes in a heterogeneous permeability field, as the SPE10 probe
Rows twoPhase(int nx, int ny, double dt)
{
    const int nc = nx * ny;
    Rows R(2 * nc);
    std::vector<double> k(nc), p(nc), S(nc);

    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            const int i = x + nx * y;
            k[i] = std::exp(3.0 * std::sin(0.7 * x) * std::cos(0.45 * y) + (y % 5 == 0 ? -4.0 : 0.0));
            p[i] = 100.0 - 50.0 * std::log(1.0 + x + 0.5 * y);
            S[i] = 0.2 + 0.6 / (1.0 + std::exp((x + y - 0.5 * (nx + ny)) / 2.0));
        }

    const double muw = 0.3, muo = 3.0, ct = 1e-4;

    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            const int i = x + nx * y;
            const double acc = 0.2 / dt;
            R[2 * i][2 * i] += acc * ct * S[i];
            R[2 * i][2 * i + 1] += acc;
            R[2 * i + 1][2 * i] += acc * ct * (1 - S[i]);
            R[2 * i + 1][2 * i + 1] -= acc;
            const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

            for (int f = 0; f < 4; f++)
            {
                const int X = x + nb[f][0], Y = y + nb[f][1];

                if (X < 0 || X >= nx || Y < 0 || Y >= ny) { continue; }

                const int j = X + nx * Y;
                const double T = 2.0 / (1.0 / k[i] + 1.0 / k[j]), dP = p[i] - p[j];
                const int up = dP >= 0 ? i : j;
                const double lw = S[up] * S[up] / muw, lo = (1 - S[up]) * (1 - S[up]) / muo;
                R[2 * i][2 * i] += T * lw;
                R[2 * i][2 * j] -= T * lw;
                R[2 * i + 1][2 * i] += T * lo;
                R[2 * i + 1][2 * j] -= T * lo;
                R[2 * i][2 * up + 1] += T * 2 * S[up] / muw * dP;
                R[2 * i + 1][2 * up + 1] -= T * 2 * (1 - S[up]) / muo * dP;
            }

            if ((x == 0 && y == 0) || (x == nx - 1 && y == ny - 1))
            {
                R[2 * i][2 * i] += 10.0 * S[i] * S[i] / muw;
                R[2 * i + 1][2 * i] += 10.0 * (1 - S[i]) * (1 - S[i]) / muo;
            }
        }

    return R;
}

// 2D lid-driven cavity Stokes on a MAC grid: [A B^T; B -C] with B = -div, velocities first.
// c > 0 adds C = c h^2 I (stabilized); c == 0 pins the first pressure.
Rows stokes2d(int n, double c, int &nu)
{
    const double h = 1.0 / n, h2 = h * h;
    const int nf = (n - 1) * n;
    nu = 2 * nf;
    Rows R(nu + n * n);

    for (int comp = 0; comp < 2; comp++)
    {
        const int dim[2] = {comp == 0 ? n - 1 : n, comp == 1 ? n - 1 : n};

        for (int j = 0; j < dim[1]; j++)
            for (int i = 0; i < dim[0]; i++)
            {
                const int r = comp * nf + i + dim[0] * j;
                const int idx[2] = {i, j};
                double diag = 0.0;

                for (int d = 0; d < 2; d++)
                    for (int s = -1; s <= 1; s += 2)
                    {
                        int q[2] = {i, j};
                        q[d] += s;

                        if (q[d] >= 0 && q[d] < dim[d]) { R[r][comp * nf + q[0] + dim[0] * q[1]] -= 1.0 / h2; diag += 1.0 / h2; }
                        else { diag += (d == comp ? 1.0 : 2.0) / h2; }
                    }

                R[r][r] += diag;
                int lo[2] = {idx[0], idx[1]}, hi[2] = {idx[0], idx[1]};
                hi[comp] += 1;
                R[r][nu + hi[0] + n * hi[1]] += 1.0 / h;
                R[r][nu + lo[0] + n * lo[1]] -= 1.0 / h;
            }
    }

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
        {
            const int r = nu + i + n * j;

            if (c == 0.0 && i == 0 && j == 0) { R[r][r] = -1.0; continue; }

            for (int comp = 0; comp < 2; comp++)
            {
                const int idx[2] = {i, j}, dimx = comp == 0 ? n - 1 : n;

                if (idx[comp] < n - 1) { R[r][comp * nf + i + dimx * j] -= 1.0 / h; }

                if (idx[comp] > 0) { int f[2] = {i, j}; f[comp] -= 1; R[r][comp * nf + f[0] + dimx * f[1]] += 1.0 / h; }
            }

            if (c > 0.0) { R[r][r] -= c * h2; }
        }

    return R;
}

// 2D Poisson on s x s boxes coupled by eps, Dirichlet on x = 0; Z = box indicators
Rows subdomains2d(int n, int s, double eps, std::vector<std::vector<double> > &Z)
{
    Rows R(n * n);
    Z.assign(s * s, std::vector<double>(n * n, 0.0));

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            const int i = x + n * y, b = x * s / n + s * (y * s / n);
            Z[b][i] = 1.0;
            const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

            for (int f = 0; f < 4; f++)
            {
                const int X = x + nb[f][0], Y = y + nb[f][1];

                if (X < 0) { R[i][i] += 2.0; continue; }

                if (X >= n || Y < 0 || Y >= n) { continue; }

                const double w = (X * s / n + s * (Y * s / n)) == b ? 1.0 : eps;
                R[i][X + n * Y] -= w;
                R[i][i] += w;
            }
        }

    return R;
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

// Host reference of the CPR restriction weights (AMGCL cpr_drs / cpr)
std::vector<double> cprWeights(const Rows &R, int B, bool drs, double eps_dd, double eps_ps)
{
    const int np = R.size() / B;
    std::vector<double> w(R.size(), 0.0);

    for (int I = 0; I < np; I++)
    {
        if (drs)
        {
            std::vector<double> top(B, 0.0), dia(B, 0.0), off(B, 0.0);

            for (int i = 0; i < B; i++)
                for (std::map<int, double>::const_iterator it = R[I * B + i].begin(); it != R[I * B + i].end(); ++it)
                {
                    const int c = it->first % B;

                    if (i == 0) { top[c] += std::fabs(it->second); }

                    if (c == 0) { if (it->first / B == I) { dia[i] = it->second; } else { off[i] += std::fabs(it->second); } }
                }

            for (int i = 0; i < B; i++)
            {
                w[I * B + i] = (i > 0 && (dia[i] < eps_dd * off[i] || top[i] < eps_ps * std::fabs(dia[0]))) ? 0.0 : 1.0;
            }
        }
        else
        {
            // B == 2: w = D^-T e_0
            const std::map<int, double> &r0 = R[I * B], &r1 = R[I * B + 1];
            const double d00 = r0.count(2 * I) ? r0.at(2 * I) : 0.0, d01 = r0.count(2 * I + 1) ? r0.at(2 * I + 1) : 0.0;
            const double d10 = r1.count(2 * I) ? r1.at(2 * I) : 0.0, d11 = r1.count(2 * I + 1) ? r1.at(2 * I + 1) : 0.0;
            const double det = d00 * d11 - d01 * d10;
            // D^T = [d00 d10; d01 d11], (D^T)^-1 e_0 = [d11; -d01] / det
            w[I * B] = d11 / det;
            w[I * B + 1] = -d01 / det;
        }
    }

    return w;
}

} // anonymous namespace

DECLARE_UNITTEST_BEGIN(CompositeP3Tests_Base);

double tol() { return sizeof(ValueTypeA) == 4 ? 1e-4 : 1e-10; }

void load(Matrix_h &A, const Rows &rows)
{
    std::vector<int> ro, ci;
    std::vector<double> v;
    toCSR(rows, ro, ci, v);
    const int n = rows.size();
    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(n, n, (int) ci.size(), 1);

    for (int i = 0; i <= n; i++) { A.row_offsets[i] = ro[i]; }

    for (size_t k = 0; k < ci.size(); k++)
    {
        A.col_indices[k] = ci[k];
        A.values[k] = ValueTypeA(v[k]);
    }

    A.computeDiagonal();
    A.set_initialized(1);
}

// Interleaved scalar rows -> b x b BSR with the union block pattern
void loadBlocks(Matrix_h &A, const Rows &rows, int b)
{
    const int nb = rows.size() / b;
    std::vector<std::map<int, std::vector<double> > > blocks(nb);

    for (int r = 0; r < (int) rows.size(); r++)
        for (std::map<int, double>::const_iterator it = rows[r].begin(); it != rows[r].end(); ++it)
        {
            std::vector<double> &blk = blocks[r / b][it->first / b];
            blk.resize(b * b, 0.0);
            blk[(r % b) * b + it->first % b] += it->second;
        }

    int nnz = 0;

    for (int I = 0; I < nb; I++) { nnz += blocks[I].size(); }

    A.set_initialized(0);
    A.addProps(CSR);
    A.resize(nb, nb, nnz, b, b, 1);
    int k = 0;
    A.row_offsets[0] = 0;

    for (int I = 0; I < nb; I++)
    {
        for (std::map<int, std::vector<double> >::const_iterator it = blocks[I].begin(); it != blocks[I].end(); ++it, ++k)
        {
            A.col_indices[k] = it->first;

            for (int q = 0; q < b * b; q++) { A.values[k * b * b + q] = ValueTypeA(it->second[q]); }
        }

        A.row_offsets[I + 1] = k;
    }

    A.computeDiagonal();
    A.set_initialized(1);
}

Rows rowsOf(const Matrix_h &M)
{
    Rows R(M.get_num_rows());

    for (int i = 0; i < M.get_num_rows(); i++)
        for (int k = M.row_offsets[i]; k < M.row_offsets[i + 1]; k++) { R[i][M.col_indices[k]] += M.values[k]; }

    return R;
}

// max |X - Y| / max |Y|
double relDiff(const Rows &X, const Rows &Y)
{
    double d = 0.0, s = 0.0;

    for (size_t i = 0; i < Y.size(); i++)
    {
        for (std::map<int, double>::const_iterator jt = Y[i].begin(); jt != Y[i].end(); ++jt)
        {
            s = std::max(s, std::fabs(jt->second));
            std::map<int, double>::const_iterator it = X[i].find(jt->first);
            d = std::max(d, std::fabs(jt->second - (it == X[i].end() ? 0.0 : it->second)));
        }

        for (std::map<int, double>::const_iterator it = X[i].begin(); it != X[i].end(); ++it)
            if (Y[i].find(it->first) == Y[i].end()) { d = std::max(d, std::fabs(it->second)); }
    }

    return X.size() == Y.size() ? d / std::max(s, 1e-300) : 1e300;
}

// The solver keeps a reference to cfg, which must outlive it
Solver<TConfig> *allocate(AMG_Config &cfg, const std::string &params, Matrix_d &A)
{
    cfg.parseParameterString(params.c_str());
    Solver<TConfig> *s = SolverFactory<TConfig>::allocate(cfg, "default", "solver");
    s->setup(A, false);
    return s;
}

// C API solve in dDDI with a host check of the true residual; returns the iterations
// Returns the iterations, or max_iters + 1 when a baseline (may_fail) does not converge
int solve(const Rows &R, const std::string &params, int block, const std::vector<int> *markers,
          const std::vector<std::vector<double> > *deflation, bool as_nullspace, bool may_fail = false)
{
    std::vector<int> ro, ci;
    std::vector<double> v;
    toCSR(R, ro, ci, v);
    const int n = R.size();
    std::vector<double> b(n), x(n, 0.0);

    for (int i = 0; i < n; i++) { b[i] = 1.0 + 0.5 * std::sin(0.37 * i); }

    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle bv, xv;
    AMGX_solver_handle solver;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, params.c_str()), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&bv, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&xv, rsrc, AMGX_mode_dDDI), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, rsrc, AMGX_mode_dDDI, cfg), AMGX_OK);

    if (block > 1)
    {
        Matrix_h B;
        loadBlocks(B, R, block);
        std::vector<int> bro(B.row_offsets.begin(), B.row_offsets.end()), bci(B.col_indices.begin(), B.col_indices.begin() + B.get_num_nz());
        std::vector<double> bvals(B.values.begin(), B.values.begin() + B.get_num_nz() * block * block);
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, B.get_num_rows(), B.get_num_nz(), block, block, &bro[0], &bci[0], &bvals[0], NULL), AMGX_OK);
    }
    else
    {
        UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n, (int) ci.size(), 1, 1, &ro[0], &ci[0], &v[0], NULL), AMGX_OK);
    }

    if (markers != NULL) { UNITTEST_ASSERT_EQUAL(AMGX_matrix_set_row_markers(A, n, &(*markers)[0]), AMGX_OK); }

    if (deflation != NULL)
    {
        std::vector<const double *> z;

        for (size_t k = 0; k < deflation->size(); k++) { z.push_back(&(*deflation)[k][0]); }

        if (as_nullspace) { UNITTEST_ASSERT_EQUAL(AMGX_matrix_set_nullspace(A, (int) z.size(), &z[0]), AMGX_OK); }
        else { UNITTEST_ASSERT_EQUAL(AMGX_matrix_set_deflation_vectors(A, (int) z.size(), &z[0]), AMGX_OK); }
    }

    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(bv, n / block, block, &b[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(xv, n / block, block, &x[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_solve(solver, bv, xv), AMGX_OK);
    AMGX_SOLVE_STATUS status;
    int iters = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &iters), AMGX_OK);
    const bool failed = status != AMGX_SOLVE_SUCCESS;

    if (!may_fail) { UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_SUCCESS); }

    UNITTEST_ASSERT_EQUAL(AMGX_vector_download(xv, &x[0]), AMGX_OK);
    double r2 = 0.0, b2 = 0.0;

    for (int i = 0; i < n; i++)
    {
        double ax = 0.0;

        for (int k = ro[i]; k < ro[i + 1]; k++) { ax += v[k] * x[ci[k]]; }

        r2 += (b[i] - ax) * (b[i] - ax);
        b2 += b[i] * b[i];
    }

    std::cout << "  iterations " << iters << (failed ? " (not converged)" : "") << ", true residual " << std::sqrt(r2 / b2) << std::endl;

    if (!failed) { UNITTEST_ASSERT_TRUE(std::sqrt(r2 / b2) < 1e-6); }

    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(xv), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(bv), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_OK);
    return failed ? iters + 1 : iters;
}

DECLARE_UNITTEST_END(CompositeP3Tests_Base);

// ------------------------------------------------------------------------------------------------
// P3.1: CPR weights and pressure matrix App(I, J) = sum_k w_I[k] A(2I + k, 2J) match the AMGCL
// definitions for DRS and quasi-IMPES, from scalar interleaved and BSR input
DECLARE_UNITTEST_BEGIN_EXTD(CPRPressureMatrix, CompositeP3Tests_Base<T_Config>);

void run()
{
    const Rows R = twoPhase(12, 10, 0.5);
    const int np = R.size() / 2;
    const std::string inner = ", c:cpr_pressure_solver(p)=BLOCK_JACOBI, c:cpr_global_smoother(g)=BLOCK_JACOBI";
    const char *names[] = {"DRS", "QUASI_IMPES"};

    for (int d = 0; d < 2; d++)
    {
        const std::vector<double> w = cprWeights(R, 2, d == 0, 0.2, 0.02);
        Rows App(np);

        for (int r = 0; r < (int) R.size(); r++)
            for (std::map<int, double>::const_iterator it = R[r].begin(); it != R[r].end(); ++it)
                if (it->first % 2 == 0) { App[r / 2][it->first / 2] += w[r] * it->second; }

        for (int bsr = 0; bsr < 2; bsr++)
        {
            Matrix_h A_h;

            if (bsr) { this->loadBlocks(A_h, R, 2); }
            else { this->load(A_h, R); }

            Matrix_d A = A_h;
            std::stringstream params;
            params << "config_version=2, solver(c)=CPR, c:cpr_block_dim=2, c:cpr_decoupling=" << names[d] << inner;
            AMG_Config cfg;
            Solver<TConfig> *s = this->allocate(cfg, params.str(), A);
            CPR_Solver<TConfig> *cpr = dynamic_cast<CPR_Solver<TConfig> *>(s);
            UNITTEST_ASSERT_TRUE(cpr != NULL);
            UNITTEST_ASSERT_EQUAL(cpr->blockDim(), 2);
            MVector_h wd = cpr->weights();
            double werr = 0.0, wmax = 0.0;

            for (size_t i = 0; i < w.size(); i++) { werr = std::max(werr, std::fabs((double) wd[i] - w[i])); wmax = std::max(wmax, std::fabs(w[i])); }

            Matrix_h P = cpr->pressureMatrix();
            UNITTEST_ASSERT_TRUE(werr <= this->tol() * wmax);
            UNITTEST_ASSERT_TRUE(this->relDiff(this->rowsOf(P), App) < 10 * this->tol());
            delete s;
        }
    }
}

DECLARE_UNITTEST_END(CPRPressureMatrix)

CPRPressureMatrix<TemplateMode<AMGX_mode_dDDI>::Type> CPRPressureMatrix_dDDI;
CPRPressureMatrix<TemplateMode<AMGX_mode_dFFI>::Type> CPRPressureMatrix_dFFI;

// ------------------------------------------------------------------------------------------------
// P3.1: FGMRES + CPR (DRS and quasi-IMPES, scalar interleaved and BSR input) converges on the
// two-phase Jacobian where single-stage ILU(0) needs far more iterations
DECLARE_UNITTEST_BEGIN_EXTD(CPREndToEnd, CompositeP3Tests_Base<T_Config>);

void run()
{
    const Rows R = twoPhase(40, 40, 5.0);
    const std::string base = "config_version=2, determinism_flag=1, solver(main)=FGMRES, main:monitor_residual=1, main:max_iters=400, main:tolerance=1e-8,"
                             " main:convergence=RELATIVE_INI_CORE, main:gmres_n_restart=50, main:norm=L2,";
    const std::string ilu = " ilu:max_iters=1, ilu:ilup_fill_level=0, ilu:ilu_solve_method=CUSPARSE, ilu:ilu_factor_sweeps=30";
    const std::string cpr = base + " main:preconditioner(c)=CPR, c:max_iters=1, c:cpr_block_dim=2, c:cpr_global_smoother(ilu)=ILUP,"
                            + ilu + ", c:cpr_pressure_solver(p)=AMG, p:algorithm=CLASSICAL, p:selector=PMIS, p:interpolator=D2,"
                            " p:max_iters=1, p:smoother(ps)=BLOCK_JACOBI, ps:relaxation_factor=0.8, p:max_levels=20";
    const int ilu_iters = this->solve(R, base + " main:preconditioner(ilu)=ILUP," + ilu, 1, NULL, NULL, false, true);
    const int drs = this->solve(R, cpr, 1, NULL, NULL, false);
    // ILUP is scalar: the BSR system uses block DILU as global smoother (App itself is checked
    // against the scalar path in CPRPressureMatrix)
    std::string cpr_bsr = cpr;
    cpr_bsr.replace(cpr_bsr.find("c:cpr_global_smoother(ilu)=ILUP,"), std::string("c:cpr_global_smoother(ilu)=ILUP,").size(), "c:cpr_global_smoother(ilu)=MULTICOLOR_DILU,");
    const int drs_bsr = this->solve(R, cpr_bsr, 2, NULL, NULL, false);
    const int impes = this->solve(R, cpr + ", c:cpr_decoupling=QUASI_IMPES", 1, NULL, NULL, false);
    UNITTEST_ASSERT_TRUE(drs < ilu_iters);
    UNITTEST_ASSERT_TRUE(impes < ilu_iters);
    UNITTEST_ASSERT_TRUE(drs_bsr < ilu_iters);
}

DECLARE_UNITTEST_END(CPREndToEnd)

CPREndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> CPREndToEnd_dDDI;

// ------------------------------------------------------------------------------------------------
// P3.2: the extracted blocks and the pressure matrices of schur_adjust_p 0/1/2 (SIMPLEC or
// diagonal M) match the AMGCL definitions
DECLARE_UNITTEST_BEGIN_EXTD(SchurSubmatrices, CompositeP3Tests_Base<T_Config>);

void run()
{
    int nu = 0;
    const Rows R = stokes2d(10, 1.0, nu);
    const int n = R.size(), np = n - nu;
    Rows Kuu(nu), Kup(nu), Kpu(np), Kpp(np);

    for (int i = 0; i < n; i++)
        for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it)
        {
            const int j = it->first;

            if (i < nu) { (j < nu ? Kuu[i][j] : Kup[i][j - nu]) = it->second; }
            else { (j < nu ? Kpu[i - nu][j] : Kpp[i - nu][j - nu]) = it->second; }
        }

    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h;

    for (int simplec = 0; simplec < 2; simplec++)
    {
        std::vector<double> M(nu, 0.0);

        for (int i = 0; i < nu; i++)
        {
            double s = 0.0;

            for (std::map<int, double>::const_iterator it = Kuu[i].begin(); it != Kuu[i].end(); ++it) { s += simplec ? std::fabs(it->second) : (it->first == i ? it->second : 0.0); }

            M[i] = 1.0 / s;
        }

        // Kpu M Kup
        Rows prod(np);

        for (int i = 0; i < np; i++)
            for (std::map<int, double>::const_iterator it = Kpu[i].begin(); it != Kpu[i].end(); ++it)
                for (std::map<int, double>::const_iterator jt = Kup[it->first].begin(); jt != Kup[it->first].end(); ++jt)
                {
                    prod[i][jt->first] += it->second * M[it->first] * jt->second;
                }

        for (int adjust = 0; adjust <= 2; adjust++)
        {
            Rows S = Kpp;

            for (int i = 0; i < np; i++)
            {
                if (adjust == 1) { S[i][i] -= prod[i].count(i) ? prod[i].at(i) : 0.0; }

                if (adjust == 2)
                    for (std::map<int, double>::const_iterator it = prod[i].begin(); it != prod[i].end(); ++it) { S[i][it->first] -= it->second; }
            }

            std::stringstream params;
            params << "config_version=2, solver(s)=SCHUR_PRESSURE_CORRECTION, s:schur_pmask_pattern=tail." << nu
                   << ", s:schur_adjust_p=" << adjust << ", s:schur_simplec_dia=" << simplec
                   << ", s:schur_velocity_solver(u)=BLOCK_JACOBI, s:schur_pressure_solver(p)=BLOCK_JACOBI";
            AMG_Config cfg;
            Solver<TConfig> *s = this->allocate(cfg, params.str(), A);
            SchurPressureCorrection_Solver<TConfig> *sc = dynamic_cast<SchurPressureCorrection_Solver<TConfig> *>(s);
            UNITTEST_ASSERT_TRUE(sc != NULL);
            UNITTEST_ASSERT_EQUAL(sc->numPressure(), np);
            Matrix_h Kuu_d = sc->velocityMatrix(), Kup_d = sc->Kup(), Kpu_d = sc->Kpu(), Kpp_d = sc->Kpp(), S_d = sc->pressureMatrix();
            UNITTEST_ASSERT_TRUE(this->relDiff(this->rowsOf(Kuu_d), Kuu) < this->tol());
            UNITTEST_ASSERT_TRUE(this->relDiff(this->rowsOf(Kup_d), Kup) < this->tol());
            UNITTEST_ASSERT_TRUE(this->relDiff(this->rowsOf(Kpu_d), Kpu) < this->tol());
            UNITTEST_ASSERT_TRUE(this->relDiff(this->rowsOf(Kpp_d), Kpp) < this->tol());
            UNITTEST_ASSERT_TRUE(this->relDiff(this->rowsOf(S_d), S) < 10 * this->tol());
            delete s;
        }
    }
}

DECLARE_UNITTEST_END(SchurSubmatrices)

SchurSubmatrices<TemplateMode<AMGX_mode_dDDI>::Type> SchurSubmatrices_dDDI;
SchurSubmatrices<TemplateMode<AMGX_mode_dFFI>::Type> SchurSubmatrices_dFFI;

// ------------------------------------------------------------------------------------------------
// P3.2: FGMRES + Schur pressure correction solves the C = 0 cavity (zero pressure diagonal);
// the config pattern and the C API markers select the same unknowns
DECLARE_UNITTEST_BEGIN_EXTD(SchurEndToEnd, CompositeP3Tests_Base<T_Config>);

void run()
{
    int nu = 0;
    const Rows R = stokes2d(24, 0.0, nu);
    std::vector<int> markers(R.size(), 0);

    for (size_t i = nu; i < R.size(); i++) { markers[i] = 1; }

    std::stringstream base;
    base << "config_version=2, determinism_flag=1, solver(main)=FGMRES, main:monitor_residual=1, main:max_iters=300, main:tolerance=1e-8,"
         " main:convergence=RELATIVE_INI_CORE, main:gmres_n_restart=50, main:norm=L2,"
         " main:preconditioner(s)=SCHUR_PRESSURE_CORRECTION, s:max_iters=1,"
         " s:schur_velocity_solver(u)=AMG, u:algorithm=SMOOTHED_AGGREGATION, u:max_iters=1, u:smoother(us)=SPAI0, us:relaxation_factor=1,"
         " s:schur_pressure_solver(p)=AMG, p:algorithm=SMOOTHED_AGGREGATION, p:max_iters=1, p:smoother(ps)=SPAI0, ps:relaxation_factor=1";
    std::stringstream pattern;
    pattern << ", s:schur_pmask_pattern=tail." << nu;
    const int with_pattern = this->solve(R, base.str() + pattern.str(), 1, NULL, NULL, false);
    const int with_markers = this->solve(R, base.str(), 1, &markers, NULL, false);
    const int triangular = this->solve(R, base.str() + pattern.str() + ", s:schur_variant=2", 1, NULL, NULL, false);
    UNITTEST_ASSERT_EQUAL(with_pattern, with_markers);
    UNITTEST_ASSERT_TRUE(with_pattern < 150);
    UNITTEST_ASSERT_TRUE(triangular < 150);
}

DECLARE_UNITTEST_END(SchurEndToEnd)

SchurEndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> SchurEndToEnd_dDDI;

// ------------------------------------------------------------------------------------------------
// P3.3: without a preconditioner the deflation correction is the Galerkin projection,
// Z^T (b - A y) = 0
DECLARE_UNITTEST_BEGIN_EXTD(DeflationProjection, CompositeP3Tests_Base<T_Config>);

void run()
{
    std::vector<std::vector<double> > Z;
    const Rows R = subdomains2d(24, 4, 1e-3, Z);
    const int n = R.size(), k = Z.size();
    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h;
    A.m_deflation_cols = k;

    for (int j = 0; j < k; j++) { A.m_deflation.insert(A.m_deflation.end(), Z[j].begin(), Z[j].end()); }

    AMG_Config cfg;
    Solver<TConfig> *s = this->allocate(cfg, "config_version=2, solver(d)=DEFLATION, d:preconditioner=NOSOLVER, d:max_iters=1, d:monitor_residual=0", A);
    UNITTEST_ASSERT_EQUAL(dynamic_cast<Deflation_Solver<TConfig> *>(s)->numVectors(), k);
    Vector_h b_h(n), y_h(n, 0.0);

    for (int i = 0; i < n; i++) { b_h[i] = ValueTypeB(1.0 + std::cos(0.3 * i)); }

    VVector b = b_h, y = y_h;
    b.set_block_dimx(1);
    b.set_block_dimy(1);
    y.set_block_dimx(1);
    y.set_block_dimy(1);
    s->solve(b, y, true);
    y_h = y;
    double worst = 0.0, scale = 0.0;

    for (int j = 0; j < k; j++)
    {
        double zr = 0.0, zb = 0.0;

        for (int i = 0; i < n; i++)
        {
            if (Z[j][i] == 0.0) { continue; }

            double ay = 0.0;

            for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it) { ay += it->second * (double) y_h[it->first]; }

            zr += Z[j][i] * ((double) b_h[i] - ay);
            zb += std::fabs(Z[j][i] * (double) b_h[i]);
        }

        worst = std::max(worst, std::fabs(zr));
        scale = std::max(scale, zb);
    }

    delete s;
    UNITTEST_ASSERT_TRUE(worst < (sizeof(ValueTypeB) == 4 ? 1e-3 : 1e-8) * scale);
}

DECLARE_UNITTEST_END(DeflationProjection)

DeflationProjection<TemplateMode<AMGX_mode_dDDI>::Type> DeflationProjection_dDDI;
DeflationProjection<TemplateMode<AMGX_mode_dFFI>::Type> DeflationProjection_dFFI;

// ------------------------------------------------------------------------------------------------
// P3.3: deflating the subdomain indicators removes the near-zero eigenvalues of weakly coupled
// subdomains: far fewer PCG iterations with Jacobi; the near-nullspace path is equivalent
DECLARE_UNITTEST_BEGIN_EXTD(DeflationEndToEnd, CompositeP3Tests_Base<T_Config>);

void run()
{
    std::vector<std::vector<double> > Z;
    const Rows R = subdomains2d(48, 6, 1e-5, Z);
    const std::string base = "config_version=2, determinism_flag=1, solver(main)=FGMRES, main:monitor_residual=1, main:max_iters=2000, main:tolerance=1e-8,"
                             " main:convergence=RELATIVE_INI_CORE, main:gmres_n_restart=100, main:norm=L2,";
    const std::string jacobi = " j:max_iters=1, j:relaxation_factor=1";
    const int plain = this->solve(R, base + " main:preconditioner(j)=BLOCK_JACOBI," + jacobi, 1, NULL, NULL, false, true);
    const std::string defl = base + " main:preconditioner(d)=DEFLATION, d:max_iters=1, d:preconditioner(j)=BLOCK_JACOBI," + jacobi;
    const int deflated = this->solve(R, defl, 1, NULL, &Z, false);
    const int nullspace = this->solve(R, defl + ", d:deflation_use_nullspace=1", 1, NULL, &Z, true);
    UNITTEST_ASSERT_TRUE(2 * deflated < plain);
    UNITTEST_ASSERT_EQUAL(deflated, nullspace);
}

DECLARE_UNITTEST_END(DeflationEndToEnd)

DeflationEndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> DeflationEndToEnd_dDDI;

} // namespace amgx
