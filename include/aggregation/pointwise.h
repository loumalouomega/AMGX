// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Pointwise aggregation adapted from AMGCL (amgcl/coarsening/pointwise_aggregates.hpp,
// amgcl/backend/builtin.hpp pointwise_matrix), Copyright (c) 2012-2022 Denis Demidov
// <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>

namespace amgx
{
namespace aggregation
{

/***************************************************
 * Pointwise aggregation for systems with b interleaved unknowns per node
 * (BSR matrices or scalar matrices with aggregation_block_size = b).
 * Nodes are aggregated on the point matrix Ap(I, J) = max |a_ij| over the
 * b x b block (I, J); all unknowns of a node share its aggregate.
 **************************************************/
template <class T_Config>
struct PointwiseAggregation
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

    // Expands a square-block (and/or external diagonal) matrix into a scalar CSR matrix.
    static void blockToScalar(const Matrix<TConfig> &A, Matrix<TConfig> &A_scalar);

    // Point matrix of a scalar matrix with b interleaved unknowns per node (sorted columns).
    static void buildPointMatrix(const Matrix<TConfig> &A, int block_size, Matrix<TConfig> &Ap);

    // strong[k] for the nonzeros of A: off-diagonal entries are strong when their nodes are the
    // same or strongly connected in Ap. filtered_diag is the lumped diagonal of A^F.
    static void expandStrongConnections(const Matrix<TConfig> &A, int block_size, const Matrix<TConfig> &Ap,
                                        const BVector &strong_points, BVector &strong, MVector &filtered_diag);

    // Removes aggregates with fewer than min_unknowns unknowns (block_size * nodes) and renumbers
    // the rest; returns the number of nodes removed.
    static int removeSmallAggregates(IVector &aggregates, int &num_aggregates, int block_size, int min_unknowns);
};

} // namespace aggregation
} // namespace amgx
