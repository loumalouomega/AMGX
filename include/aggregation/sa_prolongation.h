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
    static double estimateSpectralRadius(const Matrix<TConfig> &S, int power_iters);

    // Piecewise constant prolongator: P_tent(i, aggregates[i]) = 1, rows with a
    // negative aggregate id are empty.
    static void buildTentativeProlongator(const IVector &aggregates, int num_aggregates,
                                          Matrix<TConfig> &P_tent);

    // Overwrites S with J = I - omega S and computes P = J P_tent.
    static void smoothProlongator(Matrix<TConfig> &S, double omega, const Matrix<TConfig> &P_tent,
                                  Matrix<TConfig> &P, void *csr_workspace);
};

} // namespace aggregation
} // namespace amgx
