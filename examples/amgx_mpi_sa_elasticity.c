// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/* Distributed SMOOTHED_AGGREGATION example: 3D linear elasticity (trilinear hexahedra, clamped
   x = 0 face) with its six rigid body modes as near-nullspace, or a 7-point Poisson problem
   (-poisson). The (nx+1) x (ny+1) x (nz+1) nodes are split into z slabs across the ranks; each rank
   assembles its rows with global column indices, uploads them with contiguous partition offsets
   and, for elasticity, sets the rigid body modes of its nodes. The displacement unknowns are
   interleaved scalar rows: the AMG scope of the config needs aggregation_block_size=3 (see
   src/configs/PCG_SA_ELASTICITY.json). After the solve the true residual ||b - A x|| / ||b|| is
   computed. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <mpi.h>
#include "cuda_runtime.h"
#include "amgx_c.h"

static void errAndExit(const char *err)
{
    printf("%s\n", err);
    fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static void print_callback(const char *msg, int length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { printf("%s", msg); }
}

static int findParamIndex(char **argv, int argc, const char *parm)
{
    for (int i = 0; i < argc; i++)
        if (strncmp(argv[i], parm, 100) == 0) { return i; }

    return -1;
}

/* Element stiffness of a cube of side h (E = 1, Poisson ratio nu), 2x2x2 Gauss points; local node a
   has coordinates ((a & 1), (a >> 1) & 1, (a >> 2) & 1) */
static void hex8Stiffness(double h, double nu, double Ke[24][24])
{
    const double lam = nu / ((1 + nu) * (1 - 2 * nu)), mu = 1.0 / (2 * (1 + nu));
    double D[6][6] = {{0}};

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++) { D[i][j] = lam; }

        D[i][i] = lam + 2 * mu;
        D[i + 3][i + 3] = mu;
    }

    memset(Ke, 0, sizeof(double) * 24 * 24);
    const double g = 1.0 / sqrt(3.0);

    for (int q = 0; q < 8; q++)
    {
        const double xi = (q & 1 ? 1 : -1) * g, eta = (q & 2 ? 1 : -1) * g, zeta = (q & 4 ? 1 : -1) * g;
        double B[6][24] = {{0}}, DB[6][24];

        for (int a = 0; a < 8; a++)
        {
            const double sa = (a & 1) ? 1 : -1, ta = (a & 2) ? 1 : -1, ua = (a & 4) ? 1 : -1;
            const double dx = 0.125 * sa * (1 + ta * eta) * (1 + ua * zeta) * 2 / h;
            const double dy = 0.125 * ta * (1 + sa * xi) * (1 + ua * zeta) * 2 / h;
            const double dz = 0.125 * ua * (1 + sa * xi) * (1 + ta * eta) * 2 / h;
            B[0][3 * a] = dx; B[1][3 * a + 1] = dy; B[2][3 * a + 2] = dz;
            B[3][3 * a] = dy; B[3][3 * a + 1] = dx;
            B[4][3 * a + 1] = dz; B[4][3 * a + 2] = dy;
            B[5][3 * a] = dz; B[5][3 * a + 2] = dx;
        }

        const double w = pow(h / 2, 3);

        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 24; j++)
            {
                double s = 0;

                for (int k = 0; k < 6; k++) { s += D[i][k] * B[k][j]; }

                DB[i][j] = s;
            }

        for (int i = 0; i < 24; i++)
            for (int j = 0; j < 24; j++)
            {
                double s = 0;

                for (int k = 0; k < 6; k++) { s += B[k][i] * DB[k][j]; }

                Ke[i][j] += w * s;
            }
    }
}

int main(int argc, char **argv)
{
    int rank, nranks;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    int pidx, nx = 16, ny = 16, nz = 16;
    const int poisson = findParamIndex(argv, argc, "-poisson") != -1;
    const int use_nullspace = !poisson && findParamIndex(argv, argc, "-no-nullspace") == -1;

    if ((pidx = findParamIndex(argv, argc, "-n")) != -1 && pidx + 3 < argc)
    {
        nx = atoi(argv[pidx + 1]);
        ny = atoi(argv[pidx + 2]);
        nz = atoi(argv[pidx + 3]);
    }

    if ((pidx = findParamIndex(argv, argc, "-c")) == -1)
    {
        if (rank == 0)
        {
            printf("Usage: mpirun -n N ./amgx_mpi_sa_elasticity -c config.json [-n nx ny nz] [-poisson] [-no-nullspace] [-resetup] [-amg \"param=value ...\"]\n");
        }

        MPI_Finalize();
        return 0;
    }

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
    AMGX_config_handle cfg;
    const int sidx = findParamIndex(argv, argc, "-amg");

    if (sidx != -1) { AMGX_SAFE_CALL(AMGX_config_create_from_file_and_string(&cfg, argv[pidx + 1], argv[sidx + 1])); }
    else { AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, argv[pidx + 1])); }

    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    int ndev = 1;
    cudaGetDeviceCount(&ndev);
    int device = rank % (ndev > 0 ? ndev : 1);
    AMGX_resources_handle rsrc;
    AMGX_resources_create(&rsrc, cfg, &comm, 1, &device);
    const AMGX_Mode mode = AMGX_mode_dDDI;
    AMGX_matrix_handle A;
    AMGX_vector_handle x, b, y;
    AMGX_solver_handle solver;
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_vector_create(&y, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);
    /* nodes and their z slabs */
    const int px = nx + 1, py = ny + 1, pz = nz + 1, bs = poisson ? 1 : 3;
    const int k0 = (int)((int64_t) pz * rank / nranks), k1 = (int)((int64_t) pz * (rank + 1) / nranks);
    const int n_nodes = (k1 - k0) * px * py;
    const int n = n_nodes * bs;
    const int n_global = px * py * pz * bs;
    const double h = 1.0 / nz;
    static double Ke[24][24];
    hex8Stiffness(h, 0.3, Ke);
    int *row_ptrs = (int *) malloc((n + 1) * sizeof(int));
    int64_t *cols = (int64_t *) malloc((size_t) n * 27 * bs * sizeof(int64_t));
    double *vals = (double *) malloc((size_t) n * 27 * bs * sizeof(double));
    double *rhs = (double *) malloc((n > 0 ? n : 1) * sizeof(double));
    double *coords = (double *) malloc((n_nodes > 0 ? 3 * n_nodes : 1) * sizeof(double));
    int nnz = 0;
    row_ptrs[0] = 0;

    for (int k = k0; k < k1; k++)
        for (int j = 0; j < py; j++)
            for (int i = 0; i < px; i++)
            {
                const int64_t node = i + (int64_t) px * (j + (int64_t) py * k);
                const int local = (int)(node - (int64_t) k0 * px * py);
                const int clamped = !poisson && i == 0;
                coords[3 * local] = i * h;
                coords[3 * local + 1] = j * h;
                coords[3 * local + 2] = k * h;

                for (int r = 0; r < bs; r++)
                {
                    /* neighbours in increasing global order */
                    for (int dz = -1; dz <= 1; dz++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                const int ii = i + dx, jj = j + dy, kk = k + dz;

                                if (ii < 0 || jj < 0 || kk < 0 || ii >= px || jj >= py || kk >= pz) { continue; }

                                const int64_t m = ii + (int64_t) px * (jj + (int64_t) py * kk);

                                if (poisson)
                                {
                                    const int dist = abs(dx) + abs(dy) + abs(dz);

                                    if (dist > 1) { continue; }

                                    cols[nnz] = m;
                                    vals[nnz++] = dist == 0 ? 6.0 : -1.0;
                                    continue;
                                }

                                if ((clamped || ii == 0) && m != node) { continue; }

                                for (int s = 0; s < 3; s++)
                                {
                                    double v = 0.0;

                                    if (clamped) { v = (r == s) ? 1.0 : 0.0; }
                                    else
                                        /* elements around the node that also contain m */
                                        for (int a = 0; a < 8; a++)
                                        {
                                            const int ax = a & 1, ay = (a >> 1) & 1, az = (a >> 2) & 1;
                                            const int ex = i - ax, ey = j - ay, ez = k - az;
                                            const int cx = ax + dx, cy = ay + dy, cz = az + dz;

                                            if (ex < 0 || ey < 0 || ez < 0 || ex >= nx || ey >= ny || ez >= nz) { continue; }

                                            if (cx < 0 || cx > 1 || cy < 0 || cy > 1 || cz < 0 || cz > 1) { continue; }

                                            v += Ke[3 * a + r][3 * (cx + 2 * cy + 4 * cz) + s];
                                        }

                                    cols[nnz] = m * 3 + s;
                                    vals[nnz++] = v;
                                }
                            }

                    rhs[local * bs + r] = clamped ? 0.0 : 1.0;
                    row_ptrs[local * bs + r + 1] = nnz;
                }
            }

    int64_t *offsets = (int64_t *) malloc((nranks + 1) * sizeof(int64_t));

    for (int p = 0; p <= nranks; p++) { offsets[p] = (int64_t)((int64_t) pz * p / nranks) * px * py * bs; }

    AMGX_distribution_handle dist;
    AMGX_distribution_create(&dist, cfg);
    AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, offsets);
    AMGX_matrix_upload_distributed(A, n_global, n, nnz, 1, 1, row_ptrs, cols, vals, NULL, dist);
    AMGX_distribution_destroy(dist);

    if (use_nullspace)
    {
        double *modes = (double *) malloc((size_t)(n > 0 ? n : 1) * 6 * sizeof(double));
        const double *vectors[6];
        AMGX_generate_rigid_body_modes(3, n_nodes, coords, modes);

        for (int q = 0; q < 6; q++) { vectors[q] = modes + (size_t) q * n; }

        AMGX_matrix_set_nullspace(A, 6, vectors);
        free(modes);
    }

    double *zeros = (double *) calloc(n > 0 ? n : 1, sizeof(double));
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    AMGX_vector_bind(y, A);
    AMGX_vector_upload(x, n, 1, zeros);
    AMGX_vector_upload(b, n, 1, rhs);
    AMGX_vector_upload(y, n, 1, zeros);
    /* -resetup: a second solve after the coefficients changed (diagonal shift), with AMGX_solver_resetup */
    const int num_solves = findParamIndex(argv, argc, "-resetup") != -1 ? 2 : 1;
    const int64_t first_row = offsets[rank];
    double *Ax = (double *) malloc((n > 0 ? n : 1) * sizeof(double));
    AMGX_SOLVE_STATUS status = AMGX_SOLVE_SUCCESS;

    for (int solve = 0; solve < num_solves; solve++)
    {
        if (solve == 0) { AMGX_solver_setup(solver, A); }
        else
        {
            for (int i = 0; i < n; i++)
                for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++)
                    if (cols[k] == first_row + i) { vals[k] *= 1.5; }

            AMGX_matrix_replace_coefficients(A, n, nnz, vals, NULL);
            AMGX_vector_upload(x, n, 1, zeros);
            AMGX_solver_resetup(solver, A);
        }

        AMGX_solver_solve(solver, b, x);
        int iters = 0;
        AMGX_solver_get_status(solver, &status);
        AMGX_solver_get_iterations_number(solver, &iters);
        /* true residual */
        AMGX_matrix_vector_multiply(A, x, y);
        AMGX_vector_download(y, Ax);
        double norms[2] = {0.0, 0.0}, global[2];

        for (int i = 0; i < n; i++)
        {
            norms[0] += (rhs[i] - Ax[i]) * (rhs[i] - Ax[i]);
            norms[1] += rhs[i] * rhs[i];
        }

        MPI_Allreduce(norms, global, 2, MPI_DOUBLE, MPI_SUM, comm);

        if (rank == 0)
        {
            printf("%s%s %d x %d x %d elements, %d rows, %d ranks: status %d, iterations %d, true relative residual %.3e\n",
                   solve > 0 ? "resetup: " : "",
                   poisson ? "poisson" : (use_nullspace ? "elasticity (6 rigid body modes)" : "elasticity (no nullspace)"),
                   nx, ny, nz, n_global, nranks, (int) status, iters, sqrt(global[0] / global[1]));
        }
    }

    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_vector_destroy(y);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize());
    free(row_ptrs); free(cols); free(vals); free(rhs); free(coords); free(offsets); free(zeros); free(Ax);
    MPI_Finalize();
    return status == AMGX_SOLVE_SUCCESS ? 0 : 1;
}
