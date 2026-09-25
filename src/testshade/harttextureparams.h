// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace testshade {

enum HartTextureError : unsigned int {
    HartTextureInvalidHandle        = 1,
    HartTextureNonfiniteCoordinates = 2,
    HartTextureInvalidOptions       = 4
};

// Texture IDs are one-based indices into the launch-time descriptor table.
struct HartTextureDesc {
    uint64_t object;
    int width, height, levels, channels;
};

struct HartTextureState {
    const HartTextureDesc* textures;
    uint64_t count;
    unsigned int* errors;
};

static_assert(
    sizeof(void*) == 8 && sizeof(HartTextureDesc) == 24
        && sizeof(HartTextureState) == 24,
    "The HART texture ABI requires 64-bit pointers and 24-byte records");
static_assert(offsetof(HartTextureDesc, object) == 0
                  && offsetof(HartTextureDesc, width) == 8
                  && offsetof(HartTextureDesc, height) == 12
                  && offsetof(HartTextureDesc, levels) == 16
                  && offsetof(HartTextureDesc, channels) == 20
                  && offsetof(HartTextureState, textures) == 0
                  && offsetof(HartTextureState, count) == 8
                  && offsetof(HartTextureState, errors) == 16,
              "Unexpected HART texture ABI offsets");
static_assert(std::is_trivial<HartTextureDesc>::value
                  && std::is_standard_layout<HartTextureDesc>::value
                  && std::is_trivial<HartTextureState>::value
                  && std::is_standard_layout<HartTextureState>::value,
              "HART texture records must be POD");

}  // namespace testshade
