// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Reverse Cuthill-McKee reordering, after AMGCL (amgcl/reorder/cuthill_mckee.hpp,
// amgcl/adapter/reorder.hpp), Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>,
// MIT License (see LICENSES/MIT.txt). E. Cuthill, J. McKee, "Reducing the bandwidth of sparse
// symmetric matrices", ACM 1969; A. George, J. W. H. Liu, "An implementation of a pseudoperipheral
// node finder", ACM TOMS 5(3), 1979.

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>

namespace amgx
{
namespace reorder
{

/***************************************************
 * Device Reverse Cuthill-McKee on the (block) pattern of A, symmetrized and without the
 * diagonal. Level-synchronous: each Cuthill-McKee level is ordered by (position of the first
 * parent, degree, index) with sorts, so no serial loop runs on the device. Every connected
 * component starts from a George-Liu pseudo-peripheral node.
 * Convention (as AMGCL): perm[new] = old, iperm[old] = new, B = P A P^T with B(i, j) =
 * A(perm[i], perm[j]).
 **************************************************/
template <class T_Config>
struct RCM
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;

    // RCM permutation of the block rows of A
    static void permutation(const Matrix<TConfig> &A, IVector &perm);

    static void inverse(const IVector &perm, IVector &iperm);

    // B = P A P^T with sorted columns; nnz_map[k] is the source block of B's k-th stored block
    // (off-diagonal entries, then the external diagonal or the zero sentinel)
    static void permuteStructure(const Matrix<TConfig> &A, const IVector &perm, const IVector &iperm,
                                 Matrix<TConfig> &B, IVector &nnz_map);

    // Refills the values of B from A through nnz_map (same pattern as at permuteStructure)
    static void permuteValues(const Matrix<TConfig> &A, const IVector &nnz_map, Matrix<TConfig> &B);

    // max |i - j| over the stored blocks of A
    static int bandwidth(const Matrix<TConfig> &A);

    // y[i] = x[perm[i]] and x[perm[i]] = y[i], block rows of size bsize
    static void gather(const VVector &x, const IVector &perm, int bsize, VVector &y);
    static void scatter(const VVector &y, const IVector &perm, int bsize, VVector &x);
};

} // namespace reorder
} // namespace amgx
