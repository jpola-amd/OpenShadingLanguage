// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_optix_compat.h>

#include <OSL/oslconfig.h>
#include <OSL/shaderglobals.h>

#include "hartgeneratedparams.h"

extern "C" __constant__ testshade::HartGeneratedParams testshade_hart_params;

extern "C" __global__ void
__raygen__testshade_generated()
{
    const uint3 index         = optixGetLaunchIndex();
    const uint3 size          = optixGetLaunchDimensions();
    const uint64_t shadeindex = uint64_t(index.y) * size.x + index.x;
    const auto& params        = testshade_hart_params;
    if (shadeindex >= params.point_count || shadeindex > 0x7fffffff
        || !params.group_stride
        || params.point_count > params.scratch_bytes / params.group_stride)
        return;

    OSL::ShaderGlobals sg { };
    sg.u    = size.x == 1 ? 0.5f : float(index.x) / float(size.x - 1);
    sg.v    = size.y == 1 ? 0.5f : float(index.y) / float(size.y - 1);
    sg.dudx = 1.0f / float(size.x == 1 ? 1 : size.x - 1);
    sg.dvdy = 1.0f / float(size.y == 1 ? 1 : size.y - 1);
    sg.P    = OSL::Vec3(sg.u, sg.v, 1.0f);
    sg.dPdx = OSL::Vec3(sg.dudx, 0.0f, 0.0f);
    sg.dPdy = OSL::Vec3(0.0f, sg.dvdy, 0.0f);
    sg.dPdu = OSL::Vec3(1.0f, 0.0f, 0.0f);
    sg.dPdv = OSL::Vec3(0.0f, 1.0f, 0.0f);
    sg.N = sg.Ng   = OSL::Vec3(0.0f, 0.0f, 1.0f);
    sg.surfacearea = 1.0f;
    sg.raytype     = params.raytype;
    void* group    = params.scratch + shadeindex * params.group_stride;
    optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int, void*>(
        0u, &sg, group, nullptr, params.output, int(shadeindex), nullptr);
    optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int, void*>(
        1u, &sg, group, nullptr, params.output, int(shadeindex), nullptr);
}
