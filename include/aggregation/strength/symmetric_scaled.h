// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Strength-of-connection criterion adapted from AMGCL
// (amgcl/coarsening/plain_aggregates.hpp, amgcl/coarsening/smoothed_aggregation.hpp),
// Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License
// (see LICENSES/MIT.txt).

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>

namespace amgx
{
namespace aggregation
{

/***************************************************
 * Symmetric scaled strength of connection (SYMMETRIC_SCALED)
 *
 * Connection i-j (i != j) is strong if
 *     a_ij^2 > eps^2 * a_ii * a_jj
 * The filtered matrix A^F keeps the diagonal and the strong
 * connections; weak connections are lumped into the diagonal:
 *     d_f[i] = a_ii + sum_{j weak} a_ij
 * Only scalar (block_size 1) CSR matrices are supported.
 **************************************************/
template <class T_Config>
struct SymmetricScaledStrength
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

    // strong[k] marks the k-th nonzero of A as a strong off-diagonal connection.
    // filtered_diag[i] is the diagonal of the filtered matrix A^F.
    static void computeStrongConnections(const Matrix<TConfig> &A, double eps_strong,
                                         BVector &strong, MVector &filtered_diag);

    // Builds S = D_f^{-1} A^F: the diagonal entry (always 1) and the strong
    // off-diagonals a_ij / d_f[i] of each row (0 when d_f[i] is zero, as in AMGCL).
    // Distributed pointwise aggregation: strong flags of the halo couplings (which the point matrix
    // leaves out) by the scalar criterion with the halo diagonal; the filtered diagonal is recomputed.
    static void patchHaloStrength(const Matrix<TConfig> &A, double eps_strong, BVector &strong, MVector &filtered_diag);

    static void buildFilteredScaledOperator(const Matrix<TConfig> &A, const BVector &strong,
                                            const MVector &filtered_diag, Matrix<TConfig> &S);
};

} // namespace aggregation
} // namespace amgx
