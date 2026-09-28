// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_device.h>
#include <hip/hip_runtime.h>

#include "../hartmaterialtest.h"
#include "trace.h"

extern "C" __constant__ OSL::HartMaterialTestParams osl_hart_render_params;

extern "C" __global__ void
__raygen__osl_hart_material_test()
{
    const auto index = hartGetLaunchIndex();
    osl_hart_render_params.results[index.y * OSL::HartMaterialCases + index.x]
        = OSL::material_probe(index.x, index.y);
}
