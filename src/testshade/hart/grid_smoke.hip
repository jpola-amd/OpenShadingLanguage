// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_optix_compat.h>

#include "hartgridparams.h"

extern "C" __constant__ testshade::HartGridParams testshade_hart_params;

extern "C" __global__ void
__raygen__testshade()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 size  = optixGetLaunchDimensions();
    const float u     = size.x == 1 ? 0.5f : float(index.x) / float(size.x - 1);
    const float v     = size.y == 1 ? 0.5f : float(index.y) / float(size.y - 1);
    float* pixel      = testshade_hart_params.output
                        + (size_t(index.y) * size.x + index.x) * 3;
    pixel[0]          = u;
    pixel[1]          = v;
    pixel[2]          = u + v;
}
