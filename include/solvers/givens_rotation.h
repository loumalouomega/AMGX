// SPDX-FileCopyrightText: 2011 - 2026 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <amgx_types/util.h>
#include <amgx_types/math.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cmath>

namespace amgx
{

// Givens rotation [c s; -conj(s) c] that zeroes dy in (dx, dy) and is applied to the GMRES
// right-hand side (rhsx, rhsy).
template <typename ValueType, bool IsComplex>
struct GeneratePlaneRotation;

template <typename ValueType>
struct GeneratePlaneRotation<ValueType, false>
{
    static __host__ void generate( ValueType &dx, ValueType &dy, ValueType &cs, ValueType &sn, ValueType &rhsx, ValueType &rhsy )
    {
        ValueType tmp;

        if (dy == ValueType(0.0))
        {
            cs = 1.0;
            sn = 0.0;
        }
        else if (std::abs(dy) > std::abs(dx))
        {
            tmp = dx / dy;
            sn = ValueType(1.0) / std::sqrt(ValueType(1.0) + tmp * tmp);
            cs = tmp * sn;
        }
        else
        {
            tmp = dy / dx;
            cs = ValueType(1.0) / std::sqrt(ValueType(1.0) + tmp * tmp);
            sn = tmp * cs;
        }

        tmp   =  cs * rhsx;
        rhsy  = -sn * rhsx;
        rhsx  = tmp;
    }
};

// Complex case: c = |dx| / r (real), s = (dx / |dx|) conj(dy) / r, with r = sqrt(|dx|^2 + |dy|^2),
// so that c dx + s dy = (dx / |dx|) r and -conj(s) dx + c dy = 0.
template <typename ValueType>
struct GeneratePlaneRotation<ValueType, true>
{
    static __host__ void generate( ValueType &dx, ValueType &dy, ValueType &cs, ValueType &sn, ValueType &rhsx, ValueType &rhsy )
    {
        typedef typename types::PODTypes<ValueType>::type PodTypeB;
        const PodTypeB adx = types::util<ValueType>::abs(dx);
        const PodTypeB ady = types::util<ValueType>::abs(dy);

        if (ady == PodTypeB(0))
        {
            cs = types::util<ValueType>::get_one();
            sn = types::util<ValueType>::get_zero();
        }
        else if (adx == PodTypeB(0))
        {
            cs = types::util<ValueType>::get_zero();
            sn = types::util<ValueType>::conjugate(dy) * (PodTypeB(1) / ady);
        }
        else
        {
            const PodTypeB r = std::sqrt(adx * adx + ady * ady);
            cs = types::util<ValueType>::get_one() * (adx / r);
            sn = dx * types::util<ValueType>::conjugate(dy) * (PodTypeB(1) / (adx * r));
        }

        ValueType tmp = cs * rhsx;
        rhsy  = types::util<ValueType>::invert(types::util<ValueType>::conjugate(sn)) * rhsx; //-conjugate(sin)
        rhsx  = tmp;
    }
};

// Applies the previous rotations to column i of H, then generates and applies rotation i
template <typename ValueType>
static __host__ void PlaneRotation( cusp::array2d<ValueType, cusp::host_memory, cusp::column_major> &H,
                                    cusp::array1d<ValueType, cusp::host_memory> &cs,
                                    cusp::array1d<ValueType, cusp::host_memory> &sn,
                                    cusp::array1d<ValueType, cusp::host_memory> &s,
                                    int i)
{
    ValueType temp;

    for (int k = 0; k < i; k++)
    {
        temp     =  cs[k] * H(k,  i) +                                   sn[k] * H(k + 1, i);
        H(k + 1, i) =  cs[k] * H(k + 1, i) - types::util<ValueType>::conjugate(sn[k]) * H(k, i);
        H(k, i)   = temp;
    }

    GeneratePlaneRotation<ValueType, types::util<ValueType>::is_complex>::generate(H(i, i), H(i + 1, i), cs[i], sn[i], s[i], s[i + 1]);
    H(i, i) = cs[i] * H(i, i) + sn[i] * H(i + 1, i);
    H(i + 1, i) = types::util<ValueType>::get_zero();
}

} // namespace amgx
