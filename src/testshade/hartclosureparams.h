// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cstdint>

namespace testshade {

// The inspection harness returns numbers, never device tree pointers.
struct HartClosureSummary {
    float diffuse[3];
    float emission[3];
    float normal[3];
    uint32_t diffuse_count;
    uint32_t emission_count;
    uint32_t used;
};

constexpr unsigned int HartClosureCapacity = 1024;
static_assert(sizeof(HartClosureSummary) == 48,
              "Unexpected HART closure inspection layout");

}  // namespace testshade
