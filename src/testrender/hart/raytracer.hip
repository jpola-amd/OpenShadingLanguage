// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_device.h>

#include "../hartparams.h"

extern "C" __constant__ OSL::HartProbeParams osl_hart_render_params;

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



extern "C" __global__ void
__raygen__osl_hart_probe()
{
    const auto& params = osl_hart_render_params;
    const uint3 index  = hartGetLaunchIndex();
    const uint3 size   = hartGetLaunchDimensions();
    const uint64_t i   = uint64_t(index.y) * size.x + index.x;
    if (i >= params.ray_count)
        return;

    const auto& ray    = params.rays[i];
    unsigned primitive = ~0u, material = ~0u;
    unsigned t = __float_as_uint(ray.tmax), u = 0, v = 0;
    if (params.traversable)
        hartTrace(params.traversable,
                  make_float3(ray.origin.x, ray.origin.y, ray.origin.z),
                  make_float3(ray.direction.x, ray.direction.y, ray.direction.z),
                  ray.tmin, ray.tmax, 0.0f, HartVisibilityMask(255),
                  HART_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, primitive, t, u, v,
                  material);

    OSL::Vec3 normal(0.0f);
    if (primitive != ~0u) {
        const auto& tri = params.triangles[primitive];
        const auto& a   = params.vertices[tri.a];
        const auto& b   = params.vertices[tri.b];
        const auto& c   = params.vertices[tri.c];
        normal          = (b - a).cross(c - a).normalized();
    }
    params.hits[i] = { primitive,          material,
                       __uint_as_float(t), __uint_as_float(u),
                       __uint_as_float(v), normal };
}
