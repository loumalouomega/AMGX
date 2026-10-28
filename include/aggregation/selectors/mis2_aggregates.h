// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>

namespace amgx
{
namespace aggregation
{

/***************************************************
 * Parallel distance-2 maximal independent set aggregation
 *
 * Aggregates are built on the graph of strong connections:
 *  1. Rows without strong connections are removed (aggregate id -1),
 *     they get an empty row in the prolongator.
 *  2. A distance-2 MIS of the strong graph is computed with
 *     deterministic hash priorities; MIS nodes are aggregate roots.
 *  3. Every remaining node joins the aggregate of an adjacent root,
 *     or else of an adjacent node that already joined one.
 **************************************************/
template <class T_Config>
struct MIS2Aggregates
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

    // index_priority: lexicographically first MIS(2) (AMGCL's greedy seed order) instead of hashed priorities
    static void compute(const Matrix<TConfig> &A, const BVector &strong,
                        IVector &aggregates, int &num_aggregates, int &num_removed, bool index_priority = false);
};

} // namespace aggregation
} // namespace amgx
