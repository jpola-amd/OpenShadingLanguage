// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "../render_state.h"
#include "userdata.h"

namespace testshade {
namespace AttributeHashes {
#define RS_STRDECL(str, name) constexpr uint64_t name = OSL::strhash(str);
#include "../rs_strdecls.h"
#undef RS_STRDECL
}  // namespace AttributeHashes



template<class T, size_t N>
__device__ inline bool
hart_attribute_values(OSL::TypeDesc type, bool derivatives, void* result,
                      const T (&values)[N])
{
    constexpr auto base = std::is_same<T, float>::value ? OSL::TypeDesc::FLOAT
                          : std::is_same<T, int>::value ? OSL::TypeDesc::INT
                                                        : OSL::TypeDesc::STRING;
    if (type.basetype != base || type.aggregate != OSL::TypeDesc::SCALAR
        || type.arraylen != (N == 1 ? 0 : int(N))
        || (derivatives && base != OSL::TypeDesc::FLOAT))
        return false;
    __builtin_memcpy(result, values, sizeof(values));
    if (derivatives) {
        auto* bytes = static_cast<unsigned char*>(result);
        for (size_t i = sizeof(values); i < 3 * sizeof(values); ++i)
            bytes[i] = 0;
    }
    return true;
}



template<class T>
__device__ inline bool
hart_attribute_scalar(OSL::TypeDesc type, bool derivatives, void* result,
                      T value)
{
    const T values[] = { value };
    return hart_attribute_values(type, derivatives, result, values);
}

}  // namespace testshade



extern "C" __device__ bool
rs_hart_get_attribute(void* ec, int shade_index, OSL::ustringhash_pod object,
                      OSL::ustringhash_pod attribute, long long encoded_type,
                      bool derivatives, int index, void* result)
{
    namespace H = testshade::AttributeHashes;
    using testshade::hart_attribute_scalar;
    using testshade::hart_attribute_values;
    const auto name = OSL::ustringhash_from(attribute).hash();
    const auto obj  = OSL::ustringhash_from(object).hash();
    OSL::TypeDesc type;
    __builtin_memcpy(&type, &encoded_type, sizeof(type));
    if (name == H::osl_version)
        return hart_attribute_scalar(type, derivatives, result, OSL_VERSION);
    if (name == H::shading_index && !obj)
        return hart_attribute_scalar(type, derivatives, result, shade_index);
    if (obj == H::options && name == H::blahblah)
        return hart_attribute_scalar(type, derivatives, result, 3.14159f);

    const auto* sg     = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* render = static_cast<const testshade::HartRenderState*>(
        sg->renderstate);
    const auto* textures = render ? render->textures : nullptr;
    const auto* context  = textures ? textures->attributes : nullptr;
    if (context) {
        if (name == H::camera_resolution) {
            const int resolution[] = { context->xres, context->yres };
            return hart_attribute_values(type, derivatives, result, resolution);
        }
        if (name == H::camera_projection)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->projection.hash());
        if (name == H::camera_pixelaspect)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->pixelaspect);
        if (name == H::camera_screen_window)
            return hart_attribute_values(type, derivatives, result,
                                         context->screen_window);
        if (name == H::camera_fov)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->fov);
        if (name == H::camera_clip) {
            const float clip[] = { context->hither, context->yon };
            return hart_attribute_values(type, derivatives, result, clip);
        }
        if (name == H::camera_clip_near)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->hither);
        if (name == H::camera_clip_far)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->yon);
        if (name == H::camera_shutter)
            return hart_attribute_values(type, derivatives, result,
                                         context->shutter);
        if (name == H::camera_shutter_open)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->shutter[0]);
        if (name == H::camera_shutter_close)
            return hart_attribute_scalar(type, derivatives, result,
                                         context->shutter[1]);
    }
    if (!obj && index == -1 && textures && textures->userdata)
        return rs_hart_get_userdata(ec, shade_index, attribute, encoded_type,
                                    derivatives, result);
    return false;
}
