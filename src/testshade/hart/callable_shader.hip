// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HART validates the device ABI marker emitted by this header.
#include <amd/hart/hart_device.h>

#include "callable_smoke.h"

extern "C" __device__ float
osl_sin_ff(float);

extern "C" __device__ void
__direct_callable__testshade_init(OSL::ShaderGlobals* sg, void* groupdata,
                                  void* userdata, void*, int, void*)
{
    auto& group   = *static_cast<testshade::CallableSmokeGroupData*>(groupdata);
    group.input   = sg->u + sg->v;
    group.Cout[0] = sg->u;
    group.Cout[1] = sg->v;
    group.Cout[2] = *static_cast<float*>(userdata);
}



extern "C" __device__ void
__direct_callable__testshade_entry(OSL::ShaderGlobals*, void* groupdata, void*,
                                   void* output, int shadeindex,
                                   void* interactive)
{
    auto& group = *static_cast<testshade::CallableSmokeGroupData*>(groupdata);
    group.Cout[2] += *static_cast<float*>(interactive)
                     * osl_sin_ff(group.input);
    float* pixel = static_cast<float*>(output) + size_t(shadeindex) * 3;
    for (int c = 0; c < 3; ++c)
        pixel[c] = group.Cout[c];
}
