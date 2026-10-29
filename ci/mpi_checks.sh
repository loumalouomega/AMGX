#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Distributed regression checks: distributed SMOOTHED_AGGREGATION and coarse-level consolidation,
# run with mpirun (ranks may share one GPU). The test launcher has no MPI tests.
#
#   cd build && bash ../ci/mpi_checks.sh [filter]
#
# Each check runs an example and compares its iteration count with a band and, where the example
# prints it, the true relative residual with a bound. Only checks whose name contains [filter] run.
# Exits non-zero if any check fails. MPIRUN overrides the launcher command.

set -u

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${SRC_DIR}/src/configs"
CI_CFG="${SRC_DIR}/ci/mpi_configs"
MPIRUN="${MPIRUN:-mpirun --oversubscribe}"
FILTER="${1:-}"
LOG_DIR="$(mktemp -d)"
num_pass=0
num_fail=0

for exe in examples/amgx_mpi_sa_elasticity examples/amgx_mpi_poisson7; do
    if [ ! -x "${exe}" ]; then
        echo "missing ${exe}: run from the build directory of an MPI build"
        exit 2
    fi
done

report() # name status detail
{
    printf '%-44s %s  %s\n' "$1" "$2" "$3"

    if [ "$2" = "PASS" ]; then num_pass=$((num_pass + 1)); else num_fail=$((num_fail + 1)); fi
}

# check_sa NAME RANKS MIN MAX MAX_RES ARGS...: amgx_mpi_sa_elasticity, every solve it reports
# (two with -resetup) must succeed within [MIN, MAX] iterations below MAX_RES true residual
check_sa()
{
    local name=$1 ranks=$2 lo=$3 hi=$4 max_res=$5
    shift 5
    case "${name}" in *"${FILTER}"*) ;; *) return ;; esac
    local log="${LOG_DIR}/${name}.log"
    ${MPIRUN} -n "${ranks}" examples/amgx_mpi_sa_elasticity "$@" > "${log}" 2>&1
    local lines
    lines=$(grep "true relative residual" "${log}")

    if [ -z "${lines}" ]; then
        report "${name}" FAIL "no result (log ${log})"
        return
    fi

    local status=PASS detail=""

    while read -r line; do
        local st it res
        st=$(sed -E 's/.*status ([0-9]+).*/\1/' <<< "${line}")
        it=$(sed -E 's/.*iterations ([0-9]+).*/\1/' <<< "${line}")
        res=$(sed -E 's/.*true relative residual ([^ ]+).*/\1/' <<< "${line}")
        detail="${detail}${it} its ${res}; "

        if [ "${st}" != "0" ] || [ "${it}" -lt "${lo}" ] || [ "${it}" -gt "${hi}" ] ||
                ! awk -v r="${res}" -v m="${max_res}" 'BEGIN { exit !(r + 0 < m + 0) }'; then
            status=FAIL
        fi
    done <<< "${lines}"

    report "${name}" "${status}" "${detail}band ${lo}-${hi}, res < ${max_res}"
}

# check_p7 NAME RANKS MIN MAX EXPECT_REGEX ARGS...: amgx_mpi_poisson7, iterations in [MIN, MAX] and
# EXPECT_REGEX (e.g. a consolidation line) present in the output when not empty
check_p7()
{
    local name=$1 ranks=$2 lo=$3 hi=$4 expect=$5
    shift 5
    case "${name}" in *"${FILTER}"*) ;; *) return ;; esac
    local log="${LOG_DIR}/${name}.log"
    ${MPIRUN} -n "${ranks}" examples/amgx_mpi_poisson7 -mode dDDI "$@" > "${log}" 2>&1
    local it
    it=$(grep "Total Iterations" "${log}" | sed -E 's/.*: *([0-9]+).*/\1/' | tail -1)

    if [ -z "${it}" ] || ! grep -q "Solver Converged" "${log}"; then
        report "${name}" FAIL "no converged solve (log ${log})"
    elif [ "${it}" -lt "${lo}" ] || [ "${it}" -gt "${hi}" ]; then
        report "${name}" FAIL "${it} its, band ${lo}-${hi}"
    elif [ -n "${expect}" ] && ! grep -qE "${expect}" "${log}"; then
        report "${name}" FAIL "${it} its, missing '${expect}'"
    else
        report "${name}" PASS "${it} its, band ${lo}-${hi}"
    fi
}

SA_P="${CI_CFG}/PCG_SA_POISSON_EXACT.json"
SA_E="${CFG}/PCG_SA_ELASTICITY.json"
AGG_C="${CI_CFG}/PCG_AGGREGATION_CONSOLIDATION_SMALL.json"
SCOTCH="config_version=2, amg:matrix_consolidation_partitioner=SCOTCH"

# Distributed SA (decision 15): iterations within one or two of a single rank
for n in 1 2 4; do
    check_sa "sa_poisson63_${n}ranks" "${n}" 19 21 1e-8 -c "${SA_P}" -n 63 63 63 -poisson
    check_sa "sa_elasticity24_rbm_${n}ranks" "${n}" 29 35 1e-8 -c "${SA_E}" -n 24 24 24
done

check_sa "sa_elasticity24_nonullspace_4ranks" 4 55 62 1e-8 -c "${SA_E}" -n 24 24 24 -no-nullspace

# The same system uploaded as 3 x 3 blocks (BSR): dense_lu_num_rows is divided by the fine block size,
# x3 gives the hierarchy of the interleaved scalar upload
BSR_LU="config_version=2, amg:dense_lu_num_rows=192"

for n in 1 4; do
    check_sa "sa_elasticity24_bsr_${n}ranks" "${n}" 29 35 1e-8 -c "${SA_E}" -n 24 24 24 -bsr -amg "${BSR_LU}"
done

check_sa "sa_elasticity24_bsr_resetup_reuse_4ranks" 4 8 35 1e-8 -c "${SA_E}" -n 24 24 24 -bsr -resetup \
    -amg "${BSR_LU}, amg:structure_reuse_levels=-1"
check_sa "sa_elasticity24_resetup_4ranks" 4 8 35 1e-8 -c "${SA_E}" -n 24 24 24 -resetup
check_sa "sa_elasticity24_resetup_reuse_4ranks" 4 8 35 1e-8 -c "${SA_E}" -n 24 24 24 -resetup \
    -amg "config_version=2, amg:structure_reuse_levels=-1"

# check_same NAME RANKS PARAMS_A PARAMS_B ARGS...: two runs of amgx_mpi_sa_elasticity that must report
# the same iterations for every solve (e.g. numeric-only rebuild against full rebuild)
check_same()
{
    local name=$1 ranks=$2 pa=$3 pb=$4
    shift 4
    case "${name}" in *"${FILTER}"*) ;; *) return ;; esac
    local la="${LOG_DIR}/${name}_a.log" lb="${LOG_DIR}/${name}_b.log"
    ${MPIRUN} -n "${ranks}" examples/amgx_mpi_sa_elasticity "$@" -amg "${pa}" > "${la}" 2>&1
    ${MPIRUN} -n "${ranks}" examples/amgx_mpi_sa_elasticity "$@" -amg "${pb}" > "${lb}" 2>&1
    local ia ib
    ia=$(grep "true relative residual" "${la}" | sed -E 's/.*status ([0-9]+), iterations ([0-9]+).*/\1:\2/' | tr '\n' ' ')
    ib=$(grep "true relative residual" "${lb}" | sed -E 's/.*status ([0-9]+), iterations ([0-9]+).*/\1:\2/' | tr '\n' ' ')

    if [ -n "${ia}" ] && [ "${ia}" = "${ib}" ] && ! grep -q " [1-9][0-9]*:" <<< " ${ia}"; then
        report "${name}" PASS "status:iterations ${ia}"
    else
        report "${name}" FAIL "'${ia}' vs '${ib}' (logs ${la} ${lb})"
    fi
}

# Numeric-only distributed rebuild on structure-reuse levels: from the third setup on, Ac is refilled
REUSE="config_version=2, amg:structure_reuse_levels=-1"
check_same "sa_refill_elasticity24_4ranks" 4 "${REUSE}, amg:numeric_rebuild=1" "${REUSE}, amg:numeric_rebuild=0" \
    -c "${SA_E}" -n 24 24 24 -resetup 4
check_same "sa_refill_poisson63_4ranks" 4 "${REUSE}, amg:numeric_rebuild=1" "${REUSE}, amg:numeric_rebuild=0" \
    -c "${SA_P}" -n 63 63 63 -poisson -resetup 3
check_same "sa_refill_bsr_4ranks" 4 "${BSR_LU}, amg:structure_reuse_levels=-1, amg:numeric_rebuild=1" \
    "${BSR_LU}, amg:structure_reuse_levels=-1, amg:numeric_rebuild=0" -c "${SA_E}" -n 24 24 24 -bsr -resetup 3

# AGGREGATION consolidation onto several roots (decision 15)
check_p7 "agg_consolidation_rank_2x2x1" 4 21 28 "consolidated: 4 --> 2 partitions \(RANK\)" \
    -p 32 32 32 2 2 1 -c "${AGG_C}"
check_p7 "agg_consolidation_scotch_2x2x1" 4 21 28 "consolidated: 4 --> 2 partitions \(SCOTCH\)" \
    -p 32 32 32 2 2 1 -c "${AGG_C}" -amg "${SCOTCH}"
check_p7 "agg_consolidation_scotch_strip" 4 22 30 "partitions \(SCOTCH\), destinations 0 0 2 2" \
    -p 32 32 32 4 1 1 -c "${AGG_C}" -amg "${SCOTCH}"
check_sa "agg_consolidation_rank_chain8" 8 20 28 1e-6 -c "${AGG_C}" -n 47 47 47 -poisson
check_sa "agg_consolidation_scotch_chain8" 8 20 28 1e-6 -c "${AGG_C}" -n 47 47 47 -poisson -amg "${SCOTCH}"

echo "passed ${num_pass}, failed ${num_fail} (logs in ${LOG_DIR})"
[ "${num_fail}" -eq 0 ]
