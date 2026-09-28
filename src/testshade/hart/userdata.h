// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "../hartrenderstate.h"
#include "../userdata.h"
#include <hip/hip_runtime.h>

extern "C" __device__ bool
rs_hart_get_userdata(void* ec, int shade_index, OSL::ustringhash_pod name,
                     long long encoded_type, bool derivatives, void* result)
{
    const auto* sg     = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* render = static_cast<const testshade::HartRenderState*>(
        sg->renderstate);
    const auto* textures = render ? render->textures : nullptr;
    const auto* state    = textures ? textures->userdata : nullptr;
    auto fail            = [&]() {
        if (textures && textures->errors)
            atomicOr(textures->errors, testshade::HartInvalidUserdata);
        return false;
    };
    if (!state || shade_index < 0 || uint64_t(shade_index) >= state->points
        || (state->count && (!state->entries || !state->data)))
        return fail();
    OSL::TypeDesc type;
    __builtin_memcpy(&type, &encoded_type, sizeof(type));
    if (type.arraylen < 0 || !type.aggregate
        || (type.basetype != OSL::TypeDesc::INT
            && type.basetype != OSL::TypeDesc::FLOAT
            && type.basetype != OSL::TypeDesc::STRING)
        || (derivatives && type.basetype != OSL::TypeDesc::FLOAT))
        return fail();
    if (state->grid_defaults
        && testshade::get_default_userdata(*sg, OSL::ustringhash_from(name),
                                           type, derivatives, result))
        return true;
    for (uint64_t i = 0; i < state->count; ++i) {
        const auto& entry = state->entries[i];
        if (entry.name != OSL::ustringhash_from(name).hash()
            || entry.type != uint64_t(encoded_type))
            continue;
        const uint64_t size
            = uint64_t(type.arraylen ? type.arraylen : 1) * type.aggregate
              * (type.basetype == OSL::TypeDesc::STRING ? 8 : 4);
        if (!size || size != entry.size || entry.derivatives > 1
            || (entry.derivatives && type.basetype != OSL::TypeDesc::FLOAT))
            return fail();
        if (entry.presence != UINT64_MAX) {
            if (entry.presence > state->bytes
                || uint64_t(shade_index) >= state->bytes - entry.presence)
                return fail();
            const auto present = state->data[entry.presence + shade_index];
            if (present > 1)
                return fail();
            if (!present)
                return false;
        }
        const uint64_t bytes = size * (entry.derivatives ? 3 : 1);
        if (entry.offset > state->bytes || bytes > state->bytes - entry.offset
            || (entry.stride
                && (entry.stride < bytes
                    || uint64_t(shade_index)
                           > (state->bytes - entry.offset - bytes)
                                 / entry.stride)))
            return fail();
        const auto* source    = state->data + entry.offset
                                + uint64_t(shade_index) * entry.stride;
        auto* destination     = static_cast<unsigned char*>(result);
        const uint64_t copied = size
                                * (derivatives && entry.derivatives ? 3 : 1);
        for (uint64_t j = 0; j < copied; ++j)
            destination[j] = source[j];
        if (derivatives && !entry.derivatives)
            for (uint64_t j = size; j < 3 * size; ++j)
                destination[j] = 0;
        return true;
    }
    return false;
}
