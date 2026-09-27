// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/hashes.h>
#include <OSL/shaderglobals.h>
#include <amd/hart/hart_device.h>

extern "C" __device__ bool
rs_hart_get_userdata(void* ec, int shade_index, OSL::ustringhash_pod name,
                     long long encoded_type, bool derivatives, void* result)
{
    constexpr OSL::ustringhash requested(OSL::strhash("renderer_value"));
    OSL::TypeDesc type;
    __builtin_memcpy(&type, &encoded_type, sizeof(type));
    if (OSL::ustringhash_from(name) != requested
        || type.basetype != OSL::TypeDesc::FLOAT
        || type.aggregate != OSL::TypeDesc::SCALAR || type.arraylen || !ec
        || !result || shade_index < 0)
        return false;
    const auto& sg = *static_cast<const OSL::ShaderGlobals*>(ec);
    auto* data     = static_cast<float*>(result);
    data[0] = HART_USERDATA_BIAS + sg.u + 2.0f * sg.v + 0.125f * shade_index;
    if (derivatives) {
        data[1] = sg.dudx + 2.0f * sg.dvdx;
        data[2] = sg.dudy + 2.0f * sg.dvdy;
    }
    return true;
}
