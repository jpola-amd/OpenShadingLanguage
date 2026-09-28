// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <hip/hip_runtime.h>

#include <OSL/rs_free_function.h>

#include "../hartrenderstate.h"

namespace {

__device__ bool
hart_get_matrix(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                OSL::ustringhash space, bool inverse)
{
    const auto* sg     = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* render = static_cast<const testshade::HartRenderState*>(
        sg->renderstate);
    const auto* state = render ? render->textures : nullptr;
    const auto* named = state ? state->transforms : nullptr;
    auto fail         = [&](bool invalid) {
        if (state && state->errors
            && (invalid || !named || named->unknown_error))
            atomicOr(state->errors, testshade::HartInvalidTransform);
        result.makeIdentity();
        return false;
    };
    if (named
        && (named->reserved || named->unknown_error > 1
            || (named->count && !named->entries)))
        return fail(true);
    if (space == OSL::Hashes::common
        || (named && space.hash() == named->commonspace)) {
        result.makeIdentity();
        return true;
    }
    const auto* matrices = static_cast<const OSL::Matrix44*>(
        space == OSL::Hashes::object   ? sg->object2common
        : space == OSL::Hashes::shader ? sg->shader2common
                                       : nullptr);
    if (matrices) {
        result = matrices[inverse ? 1 : 0];
        return true;
    }
    if (space == OSL::Hashes::object || space == OSL::Hashes::shader)
        return fail(true);
    if (named) {
        for (uint64_t i = 0; i < named->count; ++i) {
            const auto& entry = named->entries[i];
            if (entry.name != space.hash())
                continue;
            if (entry.reserved || (entry.directions & ~3u))
                return fail(true);
            if (!(entry.directions & (inverse ? 2u : 1u)))
                return fail(false);
            const auto& values = inverse ? entry.inverse : entry.forward;
            for (int j = 0; j < 16; ++j) {
                if (!__builtin_isfinite(values[j]))
                    return fail(true);
                result[j / 4][j % 4] = values[j];
            }
            return true;
        }
    }
    return fail(false);
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
