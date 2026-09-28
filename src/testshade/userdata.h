// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/shaderglobals.h>

namespace testshade {

OSL_HOSTDEVICE inline bool
get_default_userdata(const OSL::ShaderGlobals& sg, OSL::ustringhash name,
                     OSL::TypeDesc type, bool derivatives, void* data)
{
    constexpr OSL::ustringhash face_idx(OSL::strhash("face_idx"));
    constexpr OSL::ustringhash s(OSL::strhash("s"));
    constexpr OSL::ustringhash t(OSL::strhash("t"));
    constexpr OSL::ustringhash red(OSL::strhash("red"));
    constexpr OSL::ustringhash green(OSL::strhash("green"));
    constexpr OSL::ustringhash blue(OSL::strhash("blue"));
    if (name == face_idx && type == OSL::TypeInt) {
        static_cast<int*>(data)[0] = int(4 * sg.u);
        return true;
    }
    if (type != OSL::TypeFloat)
        return false;
    float value, dx, dy;
    if (name == s || (name == red && sg.P.x > 0.5f)) {
        value = sg.u;
        dx    = sg.dudx;
        dy    = sg.dudy;
    } else if (name == t || (name == green && sg.P.x < 0.5f)) {
        value = sg.v;
        dx    = sg.dvdx;
        dy    = sg.dvdy;
    } else if (name == blue && int(sg.P.y * 12) % 2 == 0) {
        value = 1.0f - sg.u;
        dx    = -sg.dudx;
        dy    = -sg.dudy;
    } else {
        return false;
    }
    auto* result = static_cast<float*>(data);
    result[0]    = value;
    if (derivatives) {
        result[1] = dx;
        result[2] = dy;
    }
    return true;
}

}  // namespace testshade
