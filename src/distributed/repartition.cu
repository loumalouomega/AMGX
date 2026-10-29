// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <distributed/repartition.h>
#include <distributed/distributed_comms.h>
#include <basic_types.h>

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#ifdef AMGX_WITH_PTSCOTCH
#include <cstdio>
#include <stdint.h>
#include <ptscotch.h>
#endif
#endif

namespace amgx
{

template <class TConfig>
bool computeDestinationPartitionsScotch(DistributedManagerBase<TConfig> &manager, const std::vector<int> &rows_per_part,
                                        int new_num_parts, typename DistributedManagerBase<TConfig>::IVector_h &destination,
                                        int &num_roots)
{
#if defined(AMGX_WITH_MPI) && defined(AMGX_WITH_PTSCOTCH)
    MPI_Comm comm = manager.getComms()->get_mpi_comm();
    int my_id, num_parts;
    MPI_Comm_rank(comm, &my_id);
    MPI_Comm_size(comm, &num_parts);

    if ((int) rows_per_part.size() != num_parts || new_num_parts < 1 || new_num_parts >= num_parts) { return false; }

    // This rank's vertex: its rows, its neighbours weighted by the halo sent and received
    const int num_neighbors = manager.num_neighbors();
    std::vector<SCOTCH_Num> vertloctab(2, 0), veloloctab(1, std::max(rows_per_part[my_id], 1));
    std::vector<SCOTCH_Num> edgeloctab(std::max(num_neighbors, 1)), edloloctab(std::max(num_neighbors, 1));

    const std::vector<typename DistributedManagerBase<TConfig>::IVector> &B2L_maps = manager.getB2Lmaps();
    const typename DistributedManagerBase<TConfig>::IVector_h &halo_offsets = manager.getHaloOffsets();

    for (int i = 0; i < num_neighbors; i++)
    {
        const int sent = B2L_maps.size() > (size_t) i ? (int) B2L_maps[i].size() : 0;
        const int received = halo_offsets.size() > (size_t)(i + 1) ? halo_offsets[i + 1] - halo_offsets[i] : 0;
        edgeloctab[i] = manager.neighbors[i];
        edloloctab[i] = std::max(sent + received, 1);
    }

    vertloctab[1] = num_neighbors;
    SCOTCH_Dgraph graph;
    SCOTCH_Strat strat;
    SCOTCH_Num part = 0;
    int failed = SCOTCH_dgraphInit(&graph, comm) != 0;

    if (!failed)
    {
        failed = SCOTCH_dgraphBuild(&graph, 0, 1, 1, &vertloctab[0], &vertloctab[1], &veloloctab[0], NULL, num_neighbors,
                                    num_neighbors, &edgeloctab[0], NULL, &edloloctab[0]) != 0;
        // the neighbour lists must be symmetric (a one-sided coupling fails the check)
        int any_failed = 0;
        MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
        failed = any_failed || SCOTCH_dgraphCheck(&graph) != 0;
        MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
        failed = any_failed;

        if (!failed)
        {
            SCOTCH_stratInit(&strat);
            failed = SCOTCH_dgraphPart(&graph, new_num_parts, &strat, &part) != 0;
            SCOTCH_stratExit(&strat);
            MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm);
            failed = any_failed;
        }

        SCOTCH_dgraphExit(&graph);
    }

    if (failed) { return false; }

    // Groups everywhere, then the root of each: its lowest rank (rank 0 stays active, as with RANK,
    // so that its output covers the whole hierarchy)
    int my_part = (int) part;
    std::vector<int> parts(num_parts);
    MPI_Allgather(&my_part, 1, MPI_INT, &parts[0], 1, MPI_INT, comm);
    std::vector<int> root(new_num_parts, -1);

    for (int i = 0; i < num_parts; i++)
    {
        const int g = parts[i];

        if (g < 0 || g >= new_num_parts) { return false; }

        if (root[g] < 0) { root[g] = i; }
    }

    destination.resize(num_parts);
    num_roots = 0;

    for (int g = 0; g < new_num_parts; g++) { num_roots += root[g] >= 0; }

    for (int i = 0; i < num_parts; i++) { destination[i] = root[parts[i]]; }

    return true;
#else
    return false;
#endif
}

#define AMGX_CASE_LINE(CASE) template bool computeDestinationPartitionsScotch<TemplateMode<CASE>::Type>( \
        DistributedManagerBase<TemplateMode<CASE>::Type> &, const std::vector<int> &, int, \
        DistributedManagerBase<TemplateMode<CASE>::Type>::IVector_h &, int &);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
