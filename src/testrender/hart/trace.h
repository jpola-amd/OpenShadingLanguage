// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "../hartparams.h"
#include <amd/hart/hart_device.h>

extern "C" __global__ void
__closesthit__osl_hart()
{
    const float2 bary  = hartGetTriangleBarycentrics();
    const auto* record = static_cast<const OSL::HartHitRecordData*>(
        hartGetSbtDataPointer());
    hartSetPayload_0(hartGetPrimitiveIndex());
    hartSetPayload_1(__float_as_uint(hartGetRayTmax()));
    hartSetPayload_2(__float_as_uint(bary.x));
    hartSetPayload_3(__float_as_uint(bary.y));
    hartSetPayload_4(record->material);
}



extern "C" __global__ void
__miss__osl_hart()
{
    hartSetPayload_0(~0u);
    hartSetPayload_4(~0u);
}



struct HartTraceHit {
    unsigned primitive, material;
    float t, u, v;
};



__device__ HartTraceHit
hart_trace(uint64_t handle, const OSL::Vec3& origin, const OSL::Vec3& direction,
           float tmin, float tmax)
{
    unsigned primitive = ~0u, material = ~0u;
    unsigned t = __float_as_uint(tmax), u = 0, v = 0;
    if (handle)
        hartTrace(handle, make_float3(origin.x, origin.y, origin.z),
                  make_float3(direction.x, direction.y, direction.z), tmin,
                  tmax, 0.0f, HartVisibilityMask(255),
                  HART_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, primitive, t, u, v,
                  material);
    return { primitive, material, __uint_as_float(t), __uint_as_float(u),
             __uint_as_float(v) };
}
