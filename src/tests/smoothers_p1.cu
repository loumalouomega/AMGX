// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include <amgx_c.h>
#include "test_utils.h"
#include <matrix.h>
#include <solvers/solver.h>
#include <solvers/spai_solver.h>
#include <solvers/ilu_smoothers.h>
#include <solvers/parilu.h>

#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace amgx
{

namespace
{

typedef std::vector<std::map<int, double> > Rows;

// 5-point non-symmetric matrix with random couplings, diagonally dominant
Rows randomNonsymmetric5pt(int nx, int ny, unsigned seed)
{
    Rows A(nx * ny);
    srand(seed);

    for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            const int i = x + nx * y;
            double sum = 0.0;
            const int nb[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

            for (int d = 0; d < 4; d++)
            {
                const int xx = x + nb[d][0], yy = y + nb[d][1];

                if (xx < 0 || yy < 0 || xx >= nx || yy >= ny) { continue; }

                const double v = -(0.1 + rand() / (double) RAND_MAX);
                A[i][xx + nx * yy] = v;
                sum += std::fabs(v);
            }

            A[i][i] = sum + 0.5;
        }

    return A;
}

// Upwind 2D convection-diffusion: -eps_x u_xx - u_yy + (cx, cy) . grad u on an n x n grid, scaled by h^2
Rows convectionDiffusion(int n, double eps_x, double cx, double cy)
{
    Rows A(n * n);
    const double h = 1.0 / (n + 1);

    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
        {
            const int i = x + n * y;
            const double dxm = eps_x / (h * h) + std::max(cx, 0.0) / h, dxp = eps_x / (h * h) + std::max(-cx, 0.0) / h;
            const double dym = 1.0 / (h * h) + std::max(cy, 0.0) / h, dyp = 1.0 / (h * h) + std::max(-cy, 0.0) / h;
            A[i][i] = dxm + dxp + dym + dyp;

            if (x > 0) { A[i][i - 1] = -dxm; }

            if (x < n - 1) { A[i][i + 1] = -dxp; }

            if (y > 0) { A[i][i - n] = -dym; }

            if (y < n - 1) { A[i][i + n] = -dyp; }
        }

    for (size_t i = 0; i < A.size(); i++)
        for (std::map<int, double>::iterator it = A[i].begin(); it != A[i].end(); ++it) { it->second *= h * h; }

    return A;
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

// Exact incomplete LU (IKJ) restricted to the pattern of A^(level+1); returns L + U in one map
Rows hostILU(const Rows &A, int level)
{
    const int n = A.size();
    std::vector<std::set<int> > S(n);

    for (int i = 0; i < n; i++)
        for (std::map<int, double>::const_iterator it = A[i].begin(); it != A[i].end(); ++it) { S[i].insert(it->first); }

    for (int l = 0; l < level; l++)
    {
        std::vector<std::set<int> > next(n);

        for (int i = 0; i < n; i++)
            for (std::set<int>::iterator k = S[i].begin(); k != S[i].end(); ++k)
                for (std::map<int, double>::const_iterator j = A[*k].begin(); j != A[*k].end(); ++j) { next[i].insert(j->first); }

        S.swap(next);
    }

    Rows F(n);

    for (int i = 0; i < n; i++)
    {
        S[i].insert(i);

        for (std::set<int>::iterator j = S[i].begin(); j != S[i].end(); ++j)
        {
            std::map<int, double>::const_iterator a = A[i].find(*j);
            F[i][*j] = a == A[i].end() ? 0.0 : a->second;
        }

        for (std::map<int, double>::iterator k = F[i].begin(); k != F[i].end() && k->first < i; ++k)
        {
            k->second /= F[k->first][k->first];

            for (std::map<int, double>::iterator j = F[k->first].upper_bound(k->first); j != F[k->first].end(); ++j)
            {
                std::map<int, double>::iterator target = F[i].find(j->first);

                if (target != F[i].end()) { target->second -= k->second * j->second; }
            }
        }
    }

    return F;
}

} // anonymous namespace

DECLARE_UNITTEST_BEGIN(SmootherP1Tests_Base);

typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecDouble>::Type> DVector_h;

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

Rows rowsOf(const Matrix_h &M)
{
    Rows R(M.get_num_rows());

    for (int i = 0; i < M.get_num_rows(); i++)
        for (int k = M.row_offsets[i]; k < M.row_offsets[i + 1]; k++) { R[i][M.col_indices[k]] += M.values[k]; }

    return R;
}

double maxDiff(const Rows &X, const Rows &Y)
{
    double d = 0.0;

    for (size_t i = 0; i < X.size(); i++)
    {
        for (std::map<int, double>::const_iterator it = X[i].begin(); it != X[i].end(); ++it)
        {
            std::map<int, double>::const_iterator jt = Y[i].find(it->first);
            d = std::max(d, std::fabs(it->second - (jt == Y[i].end() ? 0.0 : jt->second)));
        }

        for (std::map<int, double>::const_iterator jt = Y[i].begin(); jt != Y[i].end(); ++jt)
            if (X[i].find(jt->first) == X[i].end()) { d = std::max(d, std::fabs(jt->second)); }
    }

    return d;
}

Solver<TConfig> *smoother(const std::string &params, Matrix_d &A)
{
    AMG_Config cfg;
    cfg.parseParameterString(params.c_str());
    Solver<TConfig> *s = SolverFactory<TConfig>::allocate(cfg, "default", "smoother");
    s->setup(A, false);
    return s;
}

DECLARE_UNITTEST_END(SmootherP1Tests_Base);

// ------------------------------------------------------------------------------------------------
// P1.3: the persistent ParILU state refactors an unchanged pattern with new values exactly as a fresh
// factorization, and repeated factorizations (ILUP and ILUT) are bitwise identical
DECLARE_UNITTEST_BEGIN_EXTD(ParILURefactor, SmootherP1Tests_Base<T_Config>);

bool identical(const Matrix_h &X, const Matrix_h &Y)
{
    if (X.get_num_nz() != Y.get_num_nz() || X.get_num_rows() != Y.get_num_rows()) { return false; }

    for (int i = 0; i <= X.get_num_rows(); i++)
        if (X.row_offsets[i] != Y.row_offsets[i]) { return false; }

    for (int k = 0; k < X.get_num_nz(); k++)
        if (X.col_indices[k] != Y.col_indices[k] || X.values[k] != Y.values[k]) { return false; }

    return true;
}

void run()
{
    const Rows R = randomNonsymmetric5pt(20, 18, 11);
    Rows R2 = R;

    for (size_t i = 0; i < R2.size(); i++)
        for (std::map<int, double>::iterator it = R2[i].begin(); it != R2[i].end(); ++it) { it->second *= (it->first == (int) i ? 1.7 : 0.8); }

    Matrix_h A_h, A2_h;
    this->load(A_h, R);
    this->load(A2_h, R2);
    Matrix_d A = A_h, A2 = A2_h;

    for (int level = 0; level < 2; level++)
    {
        // refactor on new values == fresh factorization of the new matrix
        ParILUFactor<TConfig> kept, fresh;
        Matrix_d L, U, L2, U2;
        kept.factorPower(A, level, 12, L, U);
        UNITTEST_ASSERT_TRUE(kept.samePattern(A2));
        kept.refactorPower(A2, 12, L, U);
        fresh.factorPower(A2, level, 12, L2, U2);
        Matrix_h Lh = L, Uh = U, L2h = L2, U2h = U2;
        UNITTEST_ASSERT_TRUE(identical(Lh, L2h));
        UNITTEST_ASSERT_TRUE(identical(Uh, U2h));
    }

    // determinism of the threshold factorization
    Matrix_d L1, U1, L2, U2;
    ParILU<TConfig>::factorThreshold(A, 1e-3, 3.0, 6, L1, U1);
    ParILU<TConfig>::factorThreshold(A, 1e-3, 3.0, 6, L2, U2);
    Matrix_h L1h = L1, U1h = U1, L2h = L2, U2h = U2;
    UNITTEST_ASSERT_TRUE(identical(L1h, L2h));
    UNITTEST_ASSERT_TRUE(identical(U1h, U2h));
    // a different pattern is detected
    Rows R3 = R;
    R3[0][R3.size() - 1] = 0.5;
    Matrix_h A3_h;
    this->load(A3_h, R3);
    Matrix_d A3 = A3_h;
    ParILUFactor<TConfig> f;
    Matrix_d L, U;
    f.factorPower(A, 0, 4, L, U);
    UNITTEST_ASSERT_TRUE(!f.samePattern(A3));
}

DECLARE_UNITTEST_END(ParILURefactor)

ParILURefactor<TemplateMode<AMGX_mode_dDDI>::Type> ParILURefactor_dDDI;
ParILURefactor<TemplateMode<AMGX_mode_dFFI>::Type> ParILURefactor_dFFI;

// ------------------------------------------------------------------------------------------------
// P1.1: SPAI0 matches m_i = a_ii / sum_j a_ij^2 (scalar) and M_i = A_ii / sum_j ||A_ij||_F^2 (blocks)
DECLARE_UNITTEST_BEGIN_EXTD(SPAI0Setup, SmootherP1Tests_Base<T_Config>);

void run()
{
    const Rows R = randomNonsymmetric5pt(16, 12, 7);
    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h;
    Solver<TConfig> *s = this->smoother("smoother=SPAI0", A);
    MVector_h m = dynamic_cast<SPAISolver_Base<TConfig> *>(s)->getDiagonalInverse();
    double err = 0.0;

    for (size_t i = 0; i < R.size(); i++)
    {
        double den = 0.0;

        for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it) { den += it->second * it->second; }

        err = std::max(err, std::fabs((double) m[i] - R[i].at(i) / den) / (R[i].at(i) / den));
    }

    delete s;
    UNITTEST_ASSERT_TRUE(err < this->tol());
    // 4 x 4 blocks on the same pattern
    const int b = 4, n = R.size();
    std::vector<int> ro, ci;
    std::vector<double> v;
    toCSR(R, ro, ci, v);
    Matrix_h B_h;
    B_h.set_initialized(0);
    B_h.addProps(CSR);
    B_h.resize(n, n, (int) ci.size(), b, b, 1);
    srand(3);

    for (int i = 0; i <= n; i++) { B_h.row_offsets[i] = ro[i]; }

    for (size_t k = 0; k < ci.size(); k++)
    {
        B_h.col_indices[k] = ci[k];

        for (int q = 0; q < b * b; q++) { B_h.values[k * b * b + q] = ValueTypeA(v[k] * (q % (b + 1) == 0 ? 1.0 : 0.1 * rand() / (double) RAND_MAX)); }
    }

    B_h.computeDiagonal();
    B_h.set_initialized(1);
    Matrix_d B = B_h;
    s = this->smoother("smoother=SPAI0", B);
    MVector_h M = dynamic_cast<SPAISolver_Base<TConfig> *>(s)->getDiagonalInverse();
    err = 0.0;

    for (int i = 0; i < n; i++)
    {
        double den = 0.0;
        int d = -1;

        for (int k = ro[i]; k < ro[i + 1]; k++)
        {
            if (ci[k] == i) { d = k; }

            for (int q = 0; q < b * b; q++) { den += (double) B_h.values[k * b * b + q] * B_h.values[k * b * b + q]; }
        }

        for (int q = 0; q < b * b; q++) { err = std::max(err, std::fabs((double) M[i * b * b + q] - B_h.values[d * b * b + q] / den)); }
    }

    delete s;
    UNITTEST_ASSERT_TRUE(err < this->tol());
}

DECLARE_UNITTEST_END(SPAI0Setup)

SPAI0Setup<TemplateMode<AMGX_mode_dDDI>::Type> SPAI0Setup_dDDI;
SPAI0Setup<TemplateMode<AMGX_mode_dFFI>::Type> SPAI0Setup_dFFI;

// ------------------------------------------------------------------------------------------------
// P1.1: SPAI1 rows solve the least-squares problems of AMGCL and beat SPAI0 in || I - M A ||_F
DECLARE_UNITTEST_BEGIN_EXTD(SPAI1Setup, SmootherP1Tests_Base<T_Config>);

// Host solve of min || A(I,:)^T m - e_i || through the normal equations
Rows hostSPAI1(const Rows &A)
{
    const int n = A.size();
    Rows M(n);

    for (int i = 0; i < n; i++)
    {
        std::vector<int> I;

        for (std::map<int, double>::const_iterator it = A[i].begin(); it != A[i].end(); ++it) { I.push_back(it->first); }

        const int L = I.size();
        std::vector<double> G(L * L), g(L);

        for (int p = 0; p < L; p++)
        {
            for (int q = 0; q < L; q++)
            {
                double s = 0.0;

                for (std::map<int, double>::const_iterator it = A[I[p]].begin(); it != A[I[p]].end(); ++it)
                {
                    std::map<int, double>::const_iterator jt = A[I[q]].find(it->first);

                    if (jt != A[I[q]].end()) { s += it->second * jt->second; }
                }

                G[p * L + q] = s;
            }

            std::map<int, double>::const_iterator a = A[I[p]].find(i);
            g[p] = a == A[I[p]].end() ? 0.0 : a->second;
        }

        // Gaussian elimination
        for (int c = 0; c < L; c++)
            for (int r = c + 1; r < L; r++)
            {
                const double f = G[r * L + c] / G[c * L + c];

                for (int k = c; k < L; k++) { G[r * L + k] -= f * G[c * L + k]; }

                g[r] -= f * g[c];
            }

        for (int r = L - 1; r >= 0; r--)
        {
            double s = g[r];

            for (int k = r + 1; k < L; k++) { s -= G[r * L + k] * g[k]; }

            g[r] = s / G[r * L + r];
        }

        for (int p = 0; p < L; p++) { M[i][I[p]] = g[p]; }
    }

    return M;
}

double inverseError(const Rows &M, const Rows &A)
{
    double s = 0.0;

    for (size_t i = 0; i < A.size(); i++)
    {
        std::map<int, double> row;

        for (std::map<int, double>::const_iterator m = M[i].begin(); m != M[i].end(); ++m)
            for (std::map<int, double>::const_iterator a = A[m->first].begin(); a != A[m->first].end(); ++a) { row[a->first] += m->second * a->second; }

        row[i] -= 1.0;

        for (std::map<int, double>::iterator it = row.begin(); it != row.end(); ++it) { s += it->second * it->second; }
    }

    return std::sqrt(s);
}

void run()
{
    const Rows R = randomNonsymmetric5pt(14, 10, 11);
    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h;
    Solver<TConfig> *s = this->smoother("smoother=SPAI1", A);
    SPAISolver_Base<TConfig> *spai = dynamic_cast<SPAISolver_Base<TConfig> *>(s);
    Matrix_h M_h = spai->getApproximateInverse();
    const Rows M = this->rowsOf(M_h), M_ref = hostSPAI1(R);
    UNITTEST_ASSERT_EQUAL(spai->getNumFallbackRows(), 0);
    UNITTEST_ASSERT_TRUE(this->maxDiff(M, M_ref) < (sizeof(ValueTypeA) == 4 ? 1e-3 : 1e-9));
    Rows M0(R.size());

    for (size_t i = 0; i < R.size(); i++)
    {
        double den = 0.0;

        for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it) { den += it->second * it->second; }

        M0[i][i] = R[i].at(i) / den;
    }

    UNITTEST_ASSERT_TRUE(inverseError(M, R) < inverseError(M0, R));
    delete s;
    // A row longer than the shared-memory limit falls back to SPAI0
    Rows W = randomNonsymmetric5pt(10, 10, 5);

    for (int j = 0; j < 80; j++) { W[0][j] = (j == 0) ? 200.0 : -0.5; }

    Matrix_h W_h;
    this->load(W_h, W);
    Matrix_d Wd = W_h;
    s = this->smoother("smoother=SPAI1", Wd);
    UNITTEST_ASSERT_TRUE(dynamic_cast<SPAISolver_Base<TConfig> *>(s)->getNumFallbackRows() >= 1);
    delete s;
}

DECLARE_UNITTEST_END(SPAI1Setup)

SPAI1Setup<TemplateMode<AMGX_mode_dDDI>::Type> SPAI1Setup_dDDI;
SPAI1Setup<TemplateMode<AMGX_mode_dFFI>::Type> SPAI1Setup_dFFI;

// ------------------------------------------------------------------------------------------------
// P1.2: ParILU on the pattern of A^(k+1) converges to the exact ILU on that pattern
DECLARE_UNITTEST_BEGIN_EXTD(ParILUPowerPattern, SmootherP1Tests_Base<T_Config>);

void run()
{
    const Rows R = randomNonsymmetric5pt(12, 12, 13);
    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h;

    for (int level = 0; level <= 1; level++)
    {
        Matrix_d L, U;
        ParILU<TConfig>::factorPowerPattern(A, level, 60, L, U);
        Matrix_h L_h = L, U_h = U;
        Rows F = this->rowsOf(L_h);
        const Rows Uh = this->rowsOf(U_h);

        for (size_t i = 0; i < F.size(); i++) { F[i].insert(Uh[i].begin(), Uh[i].end()); }

        UNITTEST_ASSERT_TRUE(this->maxDiff(F, hostILU(R, level)) < (sizeof(ValueTypeA) == 4 ? 1e-4 : 1e-8));
    }

    // The pattern residual decreases with the number of sweeps
    double previous = 1e300;

    for (int sweeps = 1; sweeps <= 5; sweeps++)
    {
        Matrix_d L, U;
        ParILU<TConfig>::factorPowerPattern(A, 1, sweeps, L, U);
        const double residual = ParILU<TConfig>::patternResidual(A, L, U);
        UNITTEST_ASSERT_TRUE(residual < previous);
        previous = residual;
    }
}

DECLARE_UNITTEST_END(ParILUPowerPattern)

ParILUPowerPattern<TemplateMode<AMGX_mode_dDDI>::Type> ParILUPowerPattern_dDDI;
ParILUPowerPattern<TemplateMode<AMGX_mode_dFFI>::Type> ParILUPowerPattern_dFFI;

// ------------------------------------------------------------------------------------------------
// P1.2: ParILUT keeps the diagonal, respects the fill limits, adds fill-in and lowers the residual
DECLARE_UNITTEST_BEGIN_EXTD(ParILUTSetup, SmootherP1Tests_Base<T_Config>);

void run()
{
    const Rows R = convectionDiffusion(24, 1e-2, 50.0, 20.0);
    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h;
    const double fill = 2.0;
    double norm_A = 0.0;

    for (size_t i = 0; i < R.size(); i++)
        for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it) { norm_A += it->second * it->second; }

    norm_A = std::sqrt(norm_A);

    for (int sweeps = 1; sweeps <= 4; sweeps++)
    {
        Matrix_d L, U;
        ParILU<TConfig>::factorThreshold(A, 1e-3, fill, sweeps, L, U);
        Matrix_h L_h = L, U_h = U;
        const Rows Lr = this->rowsOf(L_h), Ur = this->rowsOf(U_h);
        int fill_in = 0, violations = 0;

        for (size_t i = 0; i < R.size(); i++)
        {
            int lower = 0, upper = 0;

            for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it)
            {
                if (it->first < (int) i) { lower++; }
                else if (it->first > (int) i) { upper++; }
            }

            if (Ur[i].find(i) == Ur[i].end()) { violations++; }

            if ((int) Lr[i].size() > (int)(lower * fill) || (int) Ur[i].size() - 1 > (int)(upper * fill)) { violations++; }

            for (std::map<int, double>::const_iterator it = Lr[i].begin(); it != Lr[i].end(); ++it)
                if (R[i].find(it->first) == R[i].end()) { fill_in++; }

            for (std::map<int, double>::const_iterator it = Ur[i].begin(); it != Ur[i].end(); ++it)
                if (R[i].find(it->first) == R[i].end()) { fill_in++; }
        }

        UNITTEST_ASSERT_EQUAL(violations, 0);
        UNITTEST_ASSERT_TRUE(fill_in > 0);
        // The pattern changes with every sweep: from the second sweep on, check the residual against the size of A
        if (sweeps > 1) { UNITTEST_ASSERT_TRUE(ParILU<TConfig>::patternResidual(A, L, U) < 1e-2 * norm_A); }
    }

    // ilut_initial_fill_level = 1 starts from the pattern of A^2: at equal steps it reaches at least the
    // fill of the A start, within the same row caps and residual bound
    int nnz[2];

    for (int level = 0; level <= 1; level++)
    {
        ParILUFactor<TConfig> factor;
        factor.setInitialFillLevel(level);
        Matrix_d L, U;
        factor.factorThreshold(A, 1e-3, fill, 2, L, U);
        Matrix_h L_h = L, U_h = U;
        const Rows Lr = this->rowsOf(L_h), Ur = this->rowsOf(U_h);
        int violations = 0;

        for (size_t i = 0; i < R.size(); i++)
        {
            int lower = 0, upper = 0;

            for (std::map<int, double>::const_iterator it = R[i].begin(); it != R[i].end(); ++it)
            {
                if (it->first < (int) i) { lower++; }
                else if (it->first > (int) i) { upper++; }
            }

            if ((int) Lr[i].size() > (int)(lower * fill) || (int) Ur[i].size() - 1 > (int)(upper * fill)) { violations++; }
        }

        UNITTEST_ASSERT_EQUAL(violations, 0);
        UNITTEST_ASSERT_TRUE(ParILU<TConfig>::patternResidual(A, L, U) < 1e-2 * norm_A);
        nnz[level] = L.get_num_nz() + U.get_num_nz();
    }

    UNITTEST_ASSERT_TRUE(nnz[1] >= nnz[0]);
}

DECLARE_UNITTEST_END(ParILUTSetup)

ParILUTSetup<TemplateMode<AMGX_mode_dDDI>::Type> ParILUTSetup_dDDI;

// ------------------------------------------------------------------------------------------------
// P1.2: SpSV triangular solves are exact, Jacobi iterations converge to them
DECLARE_UNITTEST_BEGIN_EXTD(ILUTriangularSolves, SmootherP1Tests_Base<T_Config>);

void run()
{
    const Rows R = randomNonsymmetric5pt(10, 9, 17);
    Matrix_h A_h;
    this->load(A_h, R);
    Matrix_d A = A_h, L, U;
    ParILU<TConfig>::factorPowerPattern(A, 1, 40, L, U);
    Matrix_h L_h = L, U_h = U;
    const Rows Lr = this->rowsOf(L_h), Ur = this->rowsOf(U_h);
    const int n = R.size();
    std::vector<double> r(n), y(n), z(n);

    for (int i = 0; i < n; i++) { r[i] = std::sin(1.0 + i); }

    // Host forward and backward substitution
    for (int i = 0; i < n; i++)
    {
        double s = r[i];

        for (std::map<int, double>::const_iterator it = Lr[i].begin(); it != Lr[i].end(); ++it) { s -= it->second * y[it->first]; }

        y[i] = s;
    }

    for (int i = n - 1; i >= 0; i--)
    {
        double s = y[i];

        for (std::map<int, double>::const_iterator it = Ur[i].begin(); it != Ur[i].end(); ++it)
            if (it->first != i) { s -= it->second * z[it->first]; }

        z[i] = s / Ur[i].at(i);
    }

    Vector_h r_h(n);

    for (int i = 0; i < n; i++) { r_h[i] = r[i]; }

    VVector r_d = r_h, z_d(n);
    typename ILUTriangularSolve<TConfig>::Method methods[2] = {ILUTriangularSolve<TConfig>::CUSPARSE, ILUTriangularSolve<TConfig>::JACOBI};

    for (int m = 0; m < 2; m++)
    {
        ILUTriangularSolve<TConfig> solve;
        solve.setup(L, U, methods[m], 60, 1.0);
        solve.apply(r_d, z_d);
        Vector_h z_h = z_d;
        double err = 0.0, nrm = 0.0;

        for (int i = 0; i < n; i++)
        {
            err = std::max(err, std::fabs(z_h[i] - z[i]));
            nrm = std::max(nrm, std::fabs(z[i]));
        }

        UNITTEST_ASSERT_TRUE(err < (sizeof(ValueTypeA) == 4 ? 1e-4 : 1e-11) * nrm);
    }
}

DECLARE_UNITTEST_END(ILUTriangularSolves)

ILUTriangularSolves<TemplateMode<AMGX_mode_dDDI>::Type> ILUTriangularSolves_dDDI;
ILUTriangularSolves<TemplateMode<AMGX_mode_dFFI>::Type> ILUTriangularSolves_dFFI;

// ------------------------------------------------------------------------------------------------
// End to end through the C API: SPAI smoothers in SA-PCG, ILUT/ILUP preconditioners in GMRES
DECLARE_UNITTEST_BEGIN_EXTD(SmootherP1EndToEnd, SmootherP1Tests_Base<T_Config>);

int solve(const Rows &R, const std::string &params, bool resetup)
{
    std::vector<int> ro, ci;
    std::vector<double> v;
    toCSR(R, ro, ci, v);
    const int n = R.size(), nnz = ci.size();
    std::vector<double> b(n, 1.0), x(n, 0.0);
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
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_upload_all(A, n, nnz, 1, 1, &ro[0], &ci[0], &v[0], NULL), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(bv, n, 1, &b[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_setup(solver, A), AMGX_OK);

    if (resetup)
    {
        for (int k = 0; k < nnz; k++) { v[k] *= 2.0; }

        UNITTEST_ASSERT_EQUAL(AMGX_matrix_replace_coefficients(A, n, nnz, &v[0], NULL), AMGX_OK);
        UNITTEST_ASSERT_EQUAL(AMGX_solver_resetup(solver, A), AMGX_OK);
    }

    UNITTEST_ASSERT_EQUAL(AMGX_vector_upload(xv, n, 1, &x[0]), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_solve_with_0_initial_guess(solver, bv, xv), AMGX_OK);
    AMGX_SOLVE_STATUS status;
    int iters = 0;
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_status(solver, &status), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_get_iterations_number(solver, &iters), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(status, AMGX_SOLVE_SUCCESS);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_download(xv, &x[0]), AMGX_OK);
    double r2 = 0.0, b2 = 0.0;

    for (int i = 0; i < n; i++)
    {
        double ax = 0.0;

        for (int k = ro[i]; k < ro[i + 1]; k++) { ax += v[k] * x[ci[k]]; }

        r2 += (b[i] - ax) * (b[i] - ax);
        b2 += b[i] * b[i];
    }

    UNITTEST_ASSERT_TRUE(std::sqrt(r2 / b2) < 1e-6);
    UNITTEST_ASSERT_EQUAL(AMGX_solver_destroy(solver), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(xv), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_vector_destroy(bv), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_matrix_destroy(A), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_resources_destroy(rsrc), AMGX_OK);
    UNITTEST_ASSERT_EQUAL(AMGX_config_destroy(cfg), AMGX_OK);
    return iters;
}

void run()
{
    const std::string base = "config_version=2, determinism_flag=1, main:max_iters=300, main:tolerance=1e-8,"
                             " main:convergence=RELATIVE_INI_CORE, main:norm=L2, main:monitor_residual=1, ";
    // SPAI smoothers inside smoothed aggregation
    const Rows P = convectionDiffusion(48, 1.0, 0.0, 0.0);

    for (int level = 0; level < 2; level++)
    {
        std::stringstream params;
        params << base << "solver(main)=PCG, main:preconditioner(amg)=AMG, amg:algorithm=SMOOTHED_AGGREGATION, amg:max_iters=1,"
               << " amg:smoother(sm)=SPAI" << level << ", sm:relaxation_factor=1, amg:presweeps=1, amg:postsweeps=1,"
               << " amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64";
        UNITTEST_ASSERT_TRUE(solve(P, params.str(), level == 1) < 40);
    }

    // ILU preconditioners on convection-diffusion
    const Rows C = convectionDiffusion(48, 1e-1, 100.0, 40.0);
    const char *variants[] = {"ILUP, ilu:ilup_fill_level=0", "ILUP, ilu:ilup_fill_level=1", "ILUT, ilu:ilut_drop_tol=1e-3"};
    const char *methods[] = {"JACOBI", "CUSPARSE"};

    for (int v = 0; v < 3; v++)
        for (int m = 0; m < 2; m++)
        {
            std::stringstream params;
            params << base << "solver(main)=FGMRES, main:gmres_n_restart=50, main:preconditioner(ilu)=" << variants[v]
                   << ", ilu:max_iters=1, ilu:relaxation_factor=1, ilu:ilu_solve_method=" << methods[m];
            UNITTEST_ASSERT_TRUE(solve(C, params.str(), v == 2 && m == 0) < 200);
        }
}

DECLARE_UNITTEST_END(SmootherP1EndToEnd)

SmootherP1EndToEnd<TemplateMode<AMGX_mode_dDDI>::Type> SmootherP1EndToEnd_dDDI;

} // namespace amgx
