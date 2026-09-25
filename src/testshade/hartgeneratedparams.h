// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <cstdint>

namespace testshade {

struct HartTextureState;

// Separate from the stable external HartGridParams ABI.
struct HartGeneratedParams {
    float* output;
    unsigned char* scratch;
    uint64_t group_stride;
    uint64_t scratch_bytes;
    uint64_t point_count;
    int raytype;
    const HartTextureState* textures;
    const OSL::Matrix44* transforms;
};

static_assert(sizeof(void*) == 8 && sizeof(HartGeneratedParams) == 64,
              "The generated HART grid ABI requires 64-bit pointers");
static_assert(sizeof(OSL::Matrix44) == 16 * sizeof(float),
              "The HART transform ABI requires packed 4x4 float matrices");

}  // namespace testshade
