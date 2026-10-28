// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Adapted from AMGCL (amgcl/coarsening/rigid_body_modes.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#pragma once

namespace amgx
{
namespace aggregation
{

// Writes the orthonormalized rigid body modes of a 2D (3 modes) or 3D (6 modes) elasticity
// problem: translations, then rotations (-y, x) in 2D and (0, -z, y), (z, 0, -x), (-y, x, 0)
// in 3D. coords holds num_nodes * ndim coordinates, mode k is stored in
// modes[k * num_nodes * ndim, (k + 1) * num_nodes * ndim). Returns the number of modes.
int rigidBodyModes(int ndim, int num_nodes, const double *coords, double *modes);

} // namespace aggregation
} // namespace amgx
