// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "trace.h"

extern "C" __constant__ OSL::HartProbeParams osl_hart_render_params;

extern "C" __global__ void
__raygen__osl_hart_probe()
{
    const auto& params = osl_hart_render_params;
    const uint3 index  = hartGetLaunchIndex();
    const uint3 size   = hartGetLaunchDimensions();
    const uint64_t i   = uint64_t(index.y) * size.x + index.x;
    if (i >= params.ray_count)
        return;

    const auto& ray = params.rays[i];
    const auto hit  = hart_trace(params.traversable, ray.origin, ray.direction,
                                 ray.tmin, ray.tmax);

    OSL::Vec3 normal(0.0f);
    if (hit.primitive != ~0u) {
        const auto& tri = params.triangles[hit.primitive];
        const auto& a   = params.vertices[tri.a];
        const auto& b   = params.vertices[tri.b];
        const auto& c   = params.vertices[tri.c];
        normal          = (b - a).cross(c - a).normalized();
    }
    params.hits[i]
        = { hit.primitive, hit.material, hit.t, hit.u, hit.v, normal };
}
