// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Sub-matrix utilities for the composite preconditioners (CPR, Schur pressure correction),
// following AMGCL (amgcl/preconditioner/cpr.hpp, schur_pressure_correction.hpp),
// Copyright (c) 2012-2022 Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#pragma once

#include <basic_types.h>
#include <matrix.h>
#include <vector.h>

namespace amgx
{
namespace composite
{

/***************************************************
 * Device building blocks for physics-split preconditioners. All operations are massively
 * parallel (one thread per row or per entry, sorts and scans); matrices are scalar CSR.
 * Host instantiations exist for linking only and raise AMGX_ERR_NOT_IMPLEMENTED.
 **************************************************/
template <class T_Config>
struct SubmatrixOps
{
    typedef T_Config TConfig;
    DEFINE_VECTOR_TYPES
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;
    typedef typename TConfig::VecPrec ValueTypeB;

    // Scalar CSR view of A: A itself when it is scalar without external diagonal, otherwise
    // the expansion into storage (blockToScalar).
    static const Matrix<TConfig> &scalarView(const Matrix<TConfig> &A, Matrix<TConfig> &storage);

    // out = sum of the COO entries (row * m + col keys, duplicates summed); keys and values are
    // consumed. Columns are sorted.
    static void fromCOO(I64Vector &keys, MVector &values, int n, int m, Matrix<TConfig> &out);

    // out = A(rows, cols) for the rows/columns with map >= 0, renumbered by the maps. A is scalar
    // CSR without external diagonal; monotone maps keep sorted columns sorted.
    static void extract(const Matrix<TConfig> &A, const IVector &row_map, const IVector &col_map,
                        int n_rows, int n_cols, Matrix<TConfig> &out);

    // out = alpha A + beta B (same dimensions, union pattern)
    static void add(ValueTypeA alpha, const Matrix<TConfig> &A, ValueTypeA beta, const Matrix<TConfig> &B,
                    Matrix<TConfig> &out);

    // C = A B (cuSPARSE-free hash SpGEMM of csr_multiply.h)
    static void product(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C);

    // A = diag(d) A
    static void scaleRows(Matrix<TConfig> &A, const MVector &d);

    // Number of rows whose diagonal entry is missing or zero
    static int countZeroDiagonal(const Matrix<TConfig> &A);

    // y[k] = x[idx[k]]
    static void gather(const VVector &x, const IVector &idx, VVector &y);
    // x[idx[k]] = y[k]
    static void scatter(const VVector &y, const IVector &idx, VVector &x);

    // Old-to-new maps of a 0/1 mask: map_set[i] (index among the set entries or -1),
    // map_unset[i] (index among the others or -1) and the index lists of both; returns the
    // number of set entries.
    static int splitMask(const IVector &mask, IVector &map_set, IVector &map_unset, IVector &idx_set,
                         IVector &idx_unset);

    static void requireDevice(const char *who);
};

} // namespace composite
} // namespace amgx
