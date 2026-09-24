// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_optix_compat.h>

#include "callable_smoke.h"
#include "hartgridparams.h"

extern "C" __constant__ testshade::HartGridParams testshade_hart_params;

extern "C" __global__ void
__raygen__testshade()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 size  = optixGetLaunchDimensions();
    OSL::ShaderGlobals sg { };
    sg.u = size.x == 1 ? 0.5f : float(index.x) / float(size.x - 1);
    sg.v = size.y == 1 ? 0.5f : float(index.y) / float(size.y - 1);
    const int shadeindex = index.y * size.x + index.x;
    testshade::CallableSmokeGroupData group { };
    float userdata    = 0.25f;
    float interactive = 2.0f;
    optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int, void*>(
        0u, &sg, &group, &userdata, testshade_hart_params.output, shadeindex,
        &interactive);
    optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int, void*>(
        1u, &sg, &group, &userdata, testshade_hart_params.output, shadeindex,
        &interactive);
}
