// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted from AMGCL (amgcl/coarsening/rigid_body_modes.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#include <aggregation/rigid_body_modes.h>
#include <error.h>

#include <cmath>
#include <vector>

namespace amgx
{
namespace aggregation
{

int rigidBodyModes(int ndim, int num_nodes, const double *coords, double *modes)
{
    if (ndim != 2 && ndim != 3)
    {
        FatalError("Rigid body modes are only defined for 2D or 3D problems", AMGX_ERR_BAD_PARAMETERS);
    }

    if (num_nodes < 1 || coords == NULL || modes == NULL)
    {
        FatalError("Rigid body modes need at least one node", AMGX_ERR_BAD_PARAMETERS);
    }

    const size_t n = (size_t) num_nodes * ndim;
    const int num_modes = (ndim == 2) ? 3 : 6;
    // Rotations about the centroid span the same space and are better conditioned
    double centroid[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < num_nodes; i++)
        for (int d = 0; d < ndim; d++) { centroid[d] += coords[(size_t) i * ndim + d] / num_nodes; }

    for (size_t k = 0; k < n * num_modes; k++) { modes[k] = 0.0; }

    for (int i = 0; i < num_nodes; i++)
    {
        const double x = coords[(size_t) i * ndim] - centroid[0];
        const double y = coords[(size_t) i * ndim + 1] - centroid[1];
        const size_t r = (size_t) i * ndim;

        for (int d = 0; d < ndim; d++) { modes[d * n + r + d] = 1.0; }

        if (ndim == 2)
        {
            modes[2 * n + r] = -y;
            modes[2 * n + r + 1] = x;
        }
        else
        {
            const double z = coords[(size_t) i * ndim + 2] - centroid[2];
            modes[3 * n + r + 1] = -z;
            modes[3 * n + r + 2] = y;
            modes[4 * n + r] = z;
            modes[4 * n + r + 2] = -x;
            modes[5 * n + r] = -y;
            modes[5 * n + r + 1] = x;
        }
    }

    // Modified Gram-Schmidt, applied twice for orthogonality to machine precision
    for (int k = 0; k < num_modes; k++)
    {
        double *mk = modes + k * n;
        double norm0 = 0.0;

        for (size_t i = 0; i < n; i++) { norm0 += mk[i] * mk[i]; }

        for (int pass = 0; pass < 2; pass++)
        {
            for (int j = 0; j < k; j++)
            {
                const double *mj = modes + j * n;
                double dot = 0.0;

                for (size_t i = 0; i < n; i++) { dot += mj[i] * mk[i]; }

                for (size_t i = 0; i < n; i++) { mk[i] -= dot * mj[i]; }
            }
        }

        double norm = 0.0;

        for (size_t i = 0; i < n; i++) { norm += mk[i] * mk[i]; }

        norm = std::sqrt(norm);

        if (!(norm > 1e-10 * std::sqrt(norm0)))
        {
            FatalError("Rigid body modes are linearly dependent (degenerate node coordinates)", AMGX_ERR_BAD_PARAMETERS);
        }

        for (size_t i = 0; i < n; i++) { mk[i] /= norm; }
    }

    return num_modes;
}

} // namespace aggregation
} // namespace amgx
