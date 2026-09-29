// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cstdint>

namespace testshade {

constexpr int HartDiffuseRampId = 101;
constexpr int HartPhongRampId   = 102;

// Test-only Blender-shaped payloads, with numeric keyword fields after the
// phong ramp to catch array stores that overwrite subsequent parameters.
struct HartDiffuseRampParams {
    float N[3];
    float colors[8][3];
};

struct HartPhongRampParams {
    float N[3];
    float exponent;
    float colors[8][3];
    float knots[4];
    float marker;
};

// The inspection harness returns numbers, never device tree pointers.
struct HartClosureSummary {
    float diffuse[3];
    float emission[3];
    float normal[3];
    uint32_t diffuse_count;
    uint32_t emission_count;
    uint32_t used;
    HartDiffuseRampParams diffuse_ramp;
    HartPhongRampParams phong_ramp;
    float diffuse_ramp_weight[3];
    float phong_ramp_weight[3];
    uint32_t diffuse_ramp_count;
    uint32_t phong_ramp_count;
};

constexpr unsigned int HartClosureCapacity = 1024;
static_assert(sizeof(HartDiffuseRampParams) == 108
                  && sizeof(HartPhongRampParams) == 132,
              "Unexpected HART ramp parameter layout");
static_assert(sizeof(HartClosureSummary) == 320,
              "Unexpected HART closure inspection layout");

}  // namespace testshade
