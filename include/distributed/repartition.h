// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <distributed/distributed_manager.h>
#include <vector>

namespace amgx
{

/***************************************************
 * Graph-partitioned coarse-level consolidation (matrix_consolidation_partitioner=SCOTCH).
 * The ranks of the manager's communicator are the vertices of a graph (weight: the rows each
 * rank brings to the consolidated level, rows_per_part), its neighbours the edges (weight: the
 * halo exchanged with each); PT-Scotch splits it into new_num_parts balanced groups of
 * neighbouring ranks. Every rank merges into the lowest rank of its group, which belongs to the
 * group as the consolidation requires (and keeps rank 0 active). Collective over the communicator.
 * Returns false (destination untouched) when AMGX is built without PT-Scotch or the partitioning
 * fails, so that the caller keeps the rank-number map; num_roots is the number of non-empty groups.
 **************************************************/
template <class TConfig>
bool computeDestinationPartitionsScotch(DistributedManagerBase<TConfig> &manager, const std::vector<int> &rows_per_part,
                                        int new_num_parts, typename DistributedManagerBase<TConfig>::IVector_h &destination,
                                        int &num_roots);

} // namespace amgx
