// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <hip/hip_runtime.h>

#include <OSL/rs_free_function.h>

#include "../harttextureparams.h"

namespace {

__device__ bool
hart_get_matrix(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                OSL::ustringhash space, bool inverse)
{
    if (space == OSL::Hashes::common) {
        result.makeIdentity();
        return true;
    }
    const auto* sg       = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* matrices = static_cast<const OSL::Matrix44*>(
        space == OSL::Hashes::object   ? sg->object2common
        : space == OSL::Hashes::shader ? sg->shader2common
                                       : nullptr);
    if (matrices) {
        result = matrices[inverse ? 1 : 0];
        return true;
    }
    const auto* state = static_cast<const testshade::HartTextureState*>(
        sg->renderstate);
    if (state && state->errors)
        atomicOr(state->errors, testshade::HartInvalidTransform);
    result = OSL::Matrix44(__builtin_nanf(""));
    return false;
}

}  // namespace



OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_space_time(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                         OSL::ustringhash from, float)
{ return hart_get_matrix(ec, result, from, false); }



OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_space_time(OSL::OpaqueExecContextPtr ec,
                                 OSL::Matrix44& result, OSL::ustringhash to,
                                 float)
{ return hart_get_matrix(ec, result, to, true); }
