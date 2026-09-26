// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <hip/hip_runtime.h>

#include <OSL/rs_free_function.h>
#include <OSL/shaderglobals.h>

#include "../hartrenderstate.h"

OSL_RSOP OSL_HOSTDEVICE void*
rs_allocate_closure(OSL::OpaqueExecContextPtr ec, size_t size, size_t alignment)
{
    const auto* sg = static_cast<const OSL::ShaderGlobals*>(ec);
    auto* state    = static_cast<testshade::HartRenderState*>(sg->renderstate);
    void* result   = state && state->closure_pool
                         ? state->closure_pool->allocate(size, alignment)
                         : nullptr;
    if (!result && state && state->textures && state->textures->errors)
        atomicOr(state->textures->errors,
                 testshade::HartClosureAllocationFailed);
    return result;
}
