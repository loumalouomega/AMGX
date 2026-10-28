// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <amgx_c.h>
#include "test_utils.h"
#include <blas.h>

#include <cmath>
#include <complex>
#include <map>
#include <sstream>
#include <vector>

namespace amgx
{

namespace
{

typedef std::complex<double> cplx;
typedef std::vector<std::map<int, cplx> > CRows;

// 5-point -Laplacian + centered convection (cx, cy) on an n x n grid, scaled by h^2
CRows convection(int n, double cx, double cy)
{
    CRows A(n * n);
    const double h = 1.0 / (n + 1);

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            const int i = x + n * y;
            A[i][i] = 4.0;

            if (x > 0) { A[i][i - 1] = -1.0 - 0.5 * h * cx; }

            if (x < n - 1) { A[i][i + 1] = -1.0 + 0.5 * h * cx; }

            if (y > 0) { A[i][i - n] = -1.0 - 0.5 * h * cy; }

            if (y < n - 1) { A[i][i + n] = -1.0 + 0.5 * h * cy; }
        }

    return A;
}

// Upwind convection-diffusion with a recirculating velocity field pe (sin(pi x) cos(pi y), -cos(pi x) sin(pi y))
CRows recirculation(int n, double pe)
{
    CRows A(n * n);
    const double h = 1.0 / (n + 1), pi = 3.14159265358979323846;

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            const int i = x + n * y;
            const double px = (x + 1) * h, py = (y + 1) * h;
            const double cx = pe * std::sin(pi * px) * std::cos(pi * py), cy = -pe * std::cos(pi * px) * std::sin(pi * py);
            const double wm = 1.0 + h * std::max(cx, 0.0), wp = 1.0 + h * std::max(-cx, 0.0);
            const double sm = 1.0 + h * std::max(cy, 0.0), sp = 1.0 + h * std::max(-cy, 0.0);
            A[i][i] = wm + wp + sm + sp;

            if (x > 0) { A[i][i - 1] = -wm; }

            if (x < n - 1) { A[i][i + 1] = -wp; }

            if (y > 0) { A[i][i - n] = -sm; }

            if (y < n - 1) { A[i][i + n] = -sp; }
        }

    return A;
}

// -Laplace - k^2 with first-order absorbing boundary du/dn = i k u (ghost node u_g = (1 + i k h) u_i), scaled by h^2
CRows helmholtz(int n, double k)
{
    CRows A(n * n);
    const double h = 1.0 / (n + 1);
    const cplx ghost(1.0, k * h);

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            const int i = x + n * y;
            cplx d = 4.0 - k * k * h * h;

            if (x > 0) { A[i][i - 1] = -1.0; } else { d -= ghost; }

            if (x < n - 1) { A[i][i + 1] = -1.0; } else { d -= ghost; }

            if (y > 0) { A[i][i - n] = -1.0; } else { d -= ghost; }

            if (y < n - 1) { A[i][i + n] = -1.0; } else { d -= ghost; }

            A[i][i] = d;
        }

    return A;
}

} // anonymous namespace

// ------------------------------------------------------------------------------------------------
// Complex inner products conjugate their first argument on the device, as on the host
DECLARE_UNITTEST_BEGIN(ComplexDotc);

// The test launcher only runs real modes: build the complex vectors explicitly
typedef TemplateMode<AMGX_mode_dZZI>::Type TConfigZ_d;
typedef TemplateMode<AMGX_mode_hZZI>::Type TConfigZ_h;

void run()
{
    const int n = 1000;
    Vector<TConfigZ_h> a_h(n), b_h(n);
    cplx expected(0.0, 0.0), norm2(0.0, 0.0);

    for (int i = 0; i < n; i++)
    {
        const cplx a(std::sin(0.3 * i), std::cos(0.7 * i)), b(std::cos(0.2 * i), -std::sin(0.5 * i));
        a_h[i] = make_cuDoubleComplex(a.real(), a.imag());
        b_h[i] = make_cuDoubleComplex(b.real(), b.imag());
        expected += std::conj(a) * b;
        norm2 += std::conj(a) * a;
    }

    Vector<TConfigZ_d> a = a_h, b = b_h;
    const cuDoubleComplex d = dotc(a, b), h = dotc(a_h, b_h), aa = dotc(a, a);
    UNITTEST_ASSERT_TRUE(std::abs(cplx(d.x, d.y) - expected) < 1e-10 * std::abs(expected));
    UNITTEST_ASSERT_TRUE(std::abs(cplx(h.x, h.y) - expected) < 1e-10 * std::abs(expected));
    UNITTEST_ASSERT_TRUE(std::abs(aa.y) < 1e-10 * aa.x && std::abs(aa.x - norm2.real()) < 1e-10 * norm2.real());
}

DECLARE_UNITTEST_END(ComplexDotc)

ComplexDotc<TemplateMode<AMGX_mode_dDDI>::Type> ComplexDotc_dDDI;

// ------------------------------------------------------------------------------------------------
// BICGSTABL and LGMRES through the C API, real and complex modes
DECLARE_UNITTEST_BEGIN(KrylovP2Solvers);

// Solves A x = 1 and returns the number of iterations (-1 if it failed and failure is allowed);
// checks the true residual on the host
int solve(const CRows &R, bool complex_mode, const std::string &params, bool may_fail = false)
{
    const int n = R.size();
    std::vector<int> ro(1, 0), ci;
    std::vector<cplx> v;

    for (int i = 0; i < n; i++)
    {
        for (std::map<int, cplx>::const_iterator it = R[i].begin(); it != R[i].end(); ++it)
        {
            ci.push_back(it->first);
            v.push_back(it->second);
        }

        ro.push_back((int) ci.size());
    }

    const int nnz = ci.size();
    const AMGX_Mode mode = complex_mode ? AMGX_mode_dZZI : AMGX_mode_dDDI;
    std::vector<double> vr(nnz), br(n, 1.0), xr(n, 0.0);
    std::vector<cplx> bc(n, cplx(1.0, 0.0)), xc(n, cplx(0.0, 0.0));

    for (int k = 0; k < nnz; k++) { vr[k] = v[k].real(); }

    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle bv, xv;
    AMGX_solver_handle solver;
    UNITTEST_ASSERT_EQUAL(AMGX_config_create(&cfg, params.c_str()), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_create_simple(&rsrc, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_create(&A, rsrc, mode), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&bv, rsrc, mode), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_create(&xv, rsrc, mode), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_create(&solver, rsrc, mode, cfg), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n, nnz, 1, 1, &ro[0], &ci[0],
                          complex_mode ? (const void *) &v[0] : (const void *) &vr[0], NULL), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(bv, n, 1, complex_mode ? (const void *) &bc[0] : (const void *) &br[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(xv, n, 1, complex_mode ? (const void *) &xc[0] : (const void *) &xr[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_solve(solver, bv, xv), AMGX_OK);
    AMGX_SOLVE_STATUS status;
    int iters = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &iters), AMGX_OK);
    const bool failed = status != AMGX_SOLVE_SUCCESS;

    if (!may_fail) { UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_SUCCESS); }

    UNITTEST_ASSERT_EQUAL(AMGX_vector_download(xv, complex_mode ? (void *) &xc[0] : (void *) &xr[0]), AMGX_OK);
    double r2 = 0.0;

    for (int i = 0; i < n; i++)
    {
        cplx ax(0.0, 0.0);

        for (int k = ro[i]; k < ro[i + 1]; k++) { ax += (complex_mode ? v[k] : cplx(vr[k], 0.0)) * (complex_mode ? xc[ci[k]] : cplx(xr[ci[k]], 0.0)); }

        r2 += std::norm(cplx(1.0, 0.0) - ax);
    }

    if (!failed) { UNITTEST_ASSERT_TRUE(std::sqrt(r2 / n) < 1e-6); }

    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(xv), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(bv), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_OK);
    return failed ? -1 : iters;
}

std::string base(const std::string &solver, int max_iters = 2000)
{
    std::stringstream s;
    s << "config_version=2, determinism_flag=1, solver(main)=" << solver << ", main:max_iters=" << max_iters
      << ", main:tolerance=1e-8, main:convergence=RELATIVE_INI_CORE, main:norm=L2, main:monitor_residual=1";
    return s.str();
}

void run()
{
    const std::string jacobi = ", main:preconditioner(p)=BLOCK_JACOBI, p:max_iters=1, p:relaxation_factor=1";
    // BiCGStab(1) is BiCGStab
    const CRows P = convection(40, 0.0, 0.0);
    const int bicgstab = solve(P, false, base("PBICGSTAB") + jacobi);
    const int bicgstab1 = solve(P, false, base("BICGSTABL") + jacobi + ", main:bicgstab_l=1");
    UNITTEST_ASSERT_TRUE(std::abs(bicgstab1 - bicgstab) <= 2);
    // Centered convection with cell Peclet ~4 (strongly complex spectrum): BiCGStab(1) may break
    // down, L = 2 and 4 must converge
    const CRows C = convection(48, 400.0, 250.0);

    for (int L = 1; L <= 4; L *= 2)
    {
        std::stringstream p;
        p << base("BICGSTABL") << jacobi << ", main:bicgstab_l=" << L;
        solve(C, false, p.str(), L == 1);
    }

    // C1: BiCGStab must not report success on a drifted recursive residual (solve() checks the
    // true residual whenever the solver reports success)
    const CRows D = convection(128, 1000.0, 600.0);
    solve(D, false, base("PBICGSTAB", 5000) + ", main:preconditioner=NOSOLVER", true);
    solve(D, false, base("BICGSTAB", 5000), true);

    // LGMRES with K = 0 is GMRES(M); augmentation does not slow it down on recirculating flow
    const CRows F = recirculation(32, 200.0);
    const std::string restart = ", main:gmres_n_restart=30";
    const int gmres = solve(F, false, base("GMRES") + restart + jacobi);
    const int lgmres0 = solve(F, false, base("LGMRES") + restart + jacobi + ", main:lgmres_augmentation_k=0");
    const int lgmres3 = solve(F, false, base("LGMRES") + restart + jacobi + ", main:lgmres_augmentation_k=3");
    UNITTEST_ASSERT_TRUE(std::abs(lgmres0 - gmres) <= 1);
    UNITTEST_ASSERT_TRUE(lgmres3 <= gmres);
    // Complex Helmholtz with absorbing boundary: GMRES (conjugated inner products), BiCGStab(L), LGMRES
    const CRows H = helmholtz(24, 6.0);
    const std::string none = ", main:preconditioner=NOSOLVER";
    solve(H, true, base("GMRES") + ", main:gmres_n_restart=60" + none);
    solve(H, true, base("BICGSTABL") + ", main:bicgstab_l=2" + none);
    solve(H, true, base("LGMRES") + ", main:gmres_n_restart=30" + none);
}

DECLARE_UNITTEST_END(KrylovP2Solvers)

KrylovP2Solvers<TemplateMode<AMGX_mode_dDDI>::Type> KrylovP2Solvers_dDDI;

} // namespace amgx
