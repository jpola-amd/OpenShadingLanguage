// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <hip/hip_runtime.h>

#include <OSL/rs_free_function.h>
#include <OSL/shaderglobals.h>

#include "../hartrenderstate.h"

OSL_RSOP OSL_HOSTDEVICE void
rs_hart_noise_error(OSL::OpaqueExecContextPtr ec)
{
    const auto* sg    = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* state = static_cast<const testshade::HartRenderState*>(
        sg->renderstate);
    atomicOr(state->textures->errors, testshade::HartInvalidNoiseArguments);
}
