// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoothed aggregation prolongator adapted from AMGCL
// (amgcl/coarsening/smoothed_aggregation.hpp, amgcl/coarsening/tentative_prolongation.hpp,
// amgcl/backend/builtin.hpp), Copyright (c) 2012-2022 Denis Demidov
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
 * Building blocks of the smoothed aggregation prolongator
 *     P = (I - omega D_f^{-1} A^F) P_tent,  omega = relax * 4 / (3 rho(D_f^{-1} A^F))
 **************************************************/
template <class T_Config>
struct SAProlongation
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;

    // Estimates the spectral radius of S. power_iters == 0 returns the
    // Gershgorin bound (max absolute row sum), otherwise the Rayleigh quotient
    // after power_iters power iterations.
    // comm: a distributed matrix with S's row and column numbering (S has its owned rows and halo
    // columns); the vectors are halo-exchanged and the reductions are global.
    static double estimateSpectralRadius(const Matrix<TConfig> &S, int power_iters, const Matrix<TConfig> *comm = NULL);

    // Piecewise constant prolongator: P_tent(i, aggregates[i]) = 1, rows with a
    // negative aggregate id are empty.
    static void buildTentativeProlongator(const IVector &aggregates, int num_aggregates,
                                          Matrix<TConfig> &P_tent);

    // Overwrites S with J = I - omega S and computes P = J P_tent. When P_tent has at most one entry
    // per row (no nullspace) P is formed row by row without a sparse product.
    static void smoothProlongator(Matrix<TConfig> &S, double omega, const Matrix<TConfig> &P_tent,
                                  Matrix<TConfig> &P, void *csr_workspace);

    // Drops |p_ij| < drop_tol * max_k |p_ik| and rescales the kept entries of each row to its former
    // row sum (Truncate::truncateByFactor with AMGX_TruncateByMaxCoefficient, fused in two kernels)
    static void truncateRowSum(Matrix<TConfig> &P, double drop_tol);

    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecDouble>::Type> DVector;

    // Block prolongator without nullspace: unknown k of a node in aggregate a is interpolated
    // from coarse unknown a * block_size + k (aggregates are given per node).
    static void buildTentativeProlongatorBlock(const IVector &aggregates, int num_aggregates, int block_size,
            Matrix<TConfig> &P_tent);

    // Nullspace prolongator: for each aggregate the rows of B (row-major, n x K, n = nodes * block_size)
    // are factorized as B_a = Q_a R_a; P_tent holds Q_a and the coarse nullspace Bc (row-major,
    // num_aggregates * K x K) holds R_a.
    static void buildTentativeProlongatorNullspace(const IVector &aggregates, int num_aggregates, int block_size,
            int K, const DVector &B, Matrix<TConfig> &P_tent, DVector &Bc);

    // Coarse nullspace of buildTentativeProlongatorBlock: row a * block_size + k is e_k.
    static void blockIdentityNullspace(int num_aggregates, int block_size, DVector &Bc);

    // Largest K supported by truncateNullspace
    static const int MAX_TRUNCATION_K = 8;

    // Drops |p_ij| < drop_tol * max_k |p_ik| and corrects the kept entries of each row with the
    // minimum-norm change that preserves (P Bc)_i. Rows where that is impossible are kept whole.
    static void truncateNullspace(Matrix<TConfig> &P, double drop_tol, int K, const DVector &Bc);
};

} // namespace aggregation
} // namespace amgx
