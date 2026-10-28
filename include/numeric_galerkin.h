// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <basic_types.h>
#include <matrix.h>

namespace amgx
{

/***************************************************
 * Galerkin product Ac = R (A P) with a numeric-only path for coefficient updates.
 * build() runs the two SpGEMMs of csr_galerkin_product and keeps the sorted patterns of A P and
 * R A P; refill() recomputes only their values (a thread group per row, no symbolic pass, allocation
 * or host synchronization) when A, P and R keep the same
 * pattern sizes, as on a structure-reuse level. Scalar CSR, single GPU, device only.
 **************************************************/
template <class T_Config>
class NumericGalerkin
{
    public:
        typedef T_Config TConfig;

        NumericGalerkin() : m_valid(false), m_a_nnz(0), m_p_nnz(0), m_r_nnz(0), m_a_rows(0) {}

        // Full product into Ac; keeps the patterns for refill()
        void build(const Matrix<TConfig> &R, const Matrix<TConfig> &A, const Matrix<TConfig> &P, Matrix<TConfig> &Ac,
                   void *workspace);

        // Values only; returns false (and leaves Ac untouched) if the operands' patterns changed
        bool refill(const Matrix<TConfig> &R, const Matrix<TConfig> &A, const Matrix<TConfig> &P, Matrix<TConfig> &Ac);

        bool valid() const { return m_valid; }
        void clear() { m_valid = false; m_AP = Matrix<TConfig>(); m_RAP = Matrix<TConfig>(); }

        // C = A B into the fixed sorted pattern of C (exposed for tests)
        static void numericProduct(const Matrix<TConfig> &A, const Matrix<TConfig> &B, Matrix<TConfig> &C);

    private:
        Matrix<TConfig> m_AP, m_RAP;
        bool m_valid;
        int m_a_nnz, m_p_nnz, m_r_nnz, m_a_rows;

        void writeCoarse(Matrix<TConfig> &Ac);
};

} // namespace amgx
