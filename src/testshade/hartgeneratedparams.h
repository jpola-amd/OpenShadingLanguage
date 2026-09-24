// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cstdint>

namespace testshade {

// Separate from the stable external HartGridParams ABI.
struct HartGeneratedParams {
    float* output;
    unsigned char* scratch;
    uint64_t group_stride;
    uint64_t scratch_bytes;
    uint64_t point_count;
    int raytype;
};

static_assert(sizeof(void*) == 8 && sizeof(HartGeneratedParams) == 48,
              "The generated HART grid ABI requires 64-bit pointers");

}  // namespace testshade
