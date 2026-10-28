// SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoothed aggregation coarsening adapted from AMGCL
// (amgcl/coarsening/smoothed_aggregation.hpp), Copyright (c) 2012-2022
// Denis Demidov <dennis.demidov@gmail.com>, MIT License (see LICENSES/MIT.txt).

#pragma once

namespace amgx
{
namespace aggregation
{
template <class T_Config> class SA_AMG_Level;
}
}

#include <basic_types.h>
#include <amg_level.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{

/***************************************************
 * Smoothed aggregation AMG level (SMOOTHED_AGGREGATION)
 *  - aggregates: parallel MIS(2) on the SYMMETRIC_SCALED strong graph
 *  - P = (I - omega D_f^{-1} A^F) P_tent, truncated with sa_prolongator_drop_tol
 *  - R = P^T, Ac = R A P
 * Scalar, single GPU only.
 **************************************************/
template <class T_Config>
class SA_AMG_Level_Base : public AMG_Level<T_Config>
{
    public:
        typedef T_Config TConfig;
        static const AMGX_VecPrecision vecPrec = TConfig::vecPrec;
        static const AMGX_MatPrecision matPrec = TConfig::matPrec;
        static const AMGX_IndPrecision indPrec = TConfig::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::VecPrec ValueTypeB;
        typedef Vector<TConfig> VVector;
        typedef typename Matrix<TConfig>::IVector IVector;
        typedef typename Matrix<TConfig>::MVector MVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;

        static const AMGX_MemorySpace other_memspace = MemorySpaceMap<opposite_memspace<TConfig::memSpace>::memspace>::id;
        typedef TemplateConfig<other_memspace, vecPrec, matPrec, indPrec> TConfig1;

        SA_AMG_Level_Base(AMG_Class *amg, ThreadManager *tmng);
        virtual ~SA_AMG_Level_Base() {}

        virtual void transfer_level(AMG_Level<TConfig1> *ref_lvl)
        {
            FatalError("transfer_level is not implemented for SMOOTHED_AGGREGATION", AMGX_ERR_NOT_IMPLEMENTED);
        }

        void createCoarseVertices();
        void createCoarseMatrices();
        bool isClassicalAMGLevel() { return false; }
        IndexType getNumCoarseVertices() { return m_num_coarse_vertices; }

        void restrictResidual(VVector &r, VVector &rr);
        void prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp);

        void prepareNextLevelMatrix(const Matrix<TConfig> &A, Matrix<TConfig> &Ac) {}
        void consolidateVector(VVector &x) {}
        void unconsolidateVector(VVector &x) {}

        // Read access for tests
        const Matrix<TConfig> &getP() const { return P; }
        const Matrix<TConfig> &getR() const { return R; }
        const IVector &getAggregates() const { return m_aggregates; }
        int getNumAggregates() const { return m_num_aggregates; }
        double getOmega() const { return m_omega; }

    protected:
        virtual void computeAggregates() = 0;
        virtual void computeProlongationOperator() = 0;
        virtual void computeAOperator() = 0;
        void computeRestrictionOperator();
        void printLevelInfo(int nnz_P_untruncated);

        Matrix<TConfig> P, R;
        IVector m_aggregates;
        BVector m_strong;
        MVector m_filtered_diag;
        int m_num_aggregates;
        int m_num_removed;
        IndexType m_num_coarse_vertices;

        double m_eps_strong;
        double m_eps_decay;
        double m_relax;
        int m_power_iters;
        double m_drop_tol;
        bool m_print_aggregation_info;

        double m_level_eps;
        double m_rho;
        double m_omega;
};

// specialization for host: not implemented, setup must run on the GPU
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class SA_AMG_Level< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public SA_AMG_Level_Base< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
        SA_AMG_Level(AMG_Class *amg, ThreadManager *tmng) : SA_AMG_Level_Base<TConfig>(amg, tmng) {}
    protected:
        void computeAggregates();
        void computeProlongationOperator();
        void computeAOperator();
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class SA_AMG_Level< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public SA_AMG_Level_Base< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef AMG<t_vecPrec, t_matPrec, t_indPrec> AMG_Class;
        SA_AMG_Level(AMG_Class *amg, ThreadManager *tmng) : SA_AMG_Level_Base<TConfig>(amg, tmng) {}
    protected:
        void computeAggregates();
        void computeProlongationOperator();
        void computeAOperator();
};

} // namespace aggregation

template<typename T_Config>
class SA_AMG_LevelFactory : public AMG_LevelFactory<T_Config>
{
    public:
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;
        AMG_Level<T_Config> *create(AMG_Class *amg, ThreadManager *tmng) { return new aggregation::SA_AMG_Level<T_Config>(amg, tmng); }
};

} // namespace amgx
