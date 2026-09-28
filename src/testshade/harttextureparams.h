// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/hart_diagnostics.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

struct RenderContext;

namespace testshade {

enum HartDeviceError : unsigned int {
    HartTextureInvalidHandle        = 1,
    HartTextureNonfiniteCoordinates = 2,
    HartTextureInvalidOptions       = 4,
    HartInvalidTransform            = 8,
    HartClosureAllocationFailed     = 16,
    HartClosureInvalidTree          = 32,
    HartInvalidRayHit               = 64,
    HartArrayIndexOutOfBounds       = 128,
    HartInvalidSpline               = 256,
    HartUnsupportedColorTransform   = 512,
    HartInvalidNoiseArguments       = 1024,
    HartDiagnosticOverflow          = 2048,
    HartInvalidDiagnostic           = 4096,
    HartShaderError                 = 8192,
    HartInvalidUserdata             = 16384
};

struct HartUserdataDesc {
    uint64_t name, type;
    uint64_t offset, stride, presence;
    uint32_t size, derivatives;
};

struct HartUserdataState {
    const HartUserdataDesc* entries;
    uint64_t count;
    const unsigned char* data;
    uint64_t bytes, points;
    uint32_t grid_defaults, reserved;
};

struct HartTransformDesc {
    uint64_t name;
    uint32_t directions, reserved;
    float forward[16], inverse[16];
};

struct HartTransformState {
    const HartTransformDesc* entries;
    uint64_t count, commonspace;
    uint32_t unknown_error, reserved;
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
    const void* colorsystem;
    OSL::HartDiagnosticBuffer* diagnostics;
    const HartUserdataState* userdata;
    const RenderContext* attributes;
    const HartTransformState* transforms;
};

static_assert(
    sizeof(void*) == 8 && sizeof(HartTextureDesc) == 24
        && sizeof(HartTextureState) == 64 && sizeof(HartUserdataDesc) == 48
        && sizeof(HartUserdataState) == 48 && sizeof(HartTransformDesc) == 144
        && sizeof(HartTransformState) == 32,
    "The HART texture ABI requires 64-bit pointers and fixed record sizes");
static_assert(offsetof(HartTextureDesc, object) == 0
                  && offsetof(HartTextureDesc, width) == 8
                  && offsetof(HartTextureDesc, height) == 12
                  && offsetof(HartTextureDesc, levels) == 16
                  && offsetof(HartTextureDesc, channels) == 20
                  && offsetof(HartTextureState, textures) == 0
                  && offsetof(HartTextureState, count) == 8
                  && offsetof(HartTextureState, errors) == 16
                  && offsetof(HartTextureState, colorsystem) == 24
                  && offsetof(HartTextureState, diagnostics) == 32
                  && offsetof(HartTextureState, userdata) == 40
                  && offsetof(HartTextureState, attributes) == 48
                  && offsetof(HartTextureState, transforms) == 56
                  && offsetof(HartTransformDesc, name) == 0
                  && offsetof(HartTransformDesc, directions) == 8
                  && offsetof(HartTransformDesc, reserved) == 12
                  && offsetof(HartTransformDesc, forward) == 16
                  && offsetof(HartTransformDesc, inverse) == 80
                  && offsetof(HartTransformState, entries) == 0
                  && offsetof(HartTransformState, count) == 8
                  && offsetof(HartTransformState, commonspace) == 16
                  && offsetof(HartTransformState, unknown_error) == 24
                  && offsetof(HartTransformState, reserved) == 28,
              "Unexpected HART texture ABI offsets");
static_assert(std::is_trivial<HartTextureDesc>::value
                  && std::is_standard_layout<HartTextureDesc>::value
                  && std::is_trivial<HartTextureState>::value
                  && std::is_standard_layout<HartTextureState>::value
                  && std::is_trivial<HartUserdataDesc>::value
                  && std::is_standard_layout<HartUserdataDesc>::value
                  && std::is_trivial<HartUserdataState>::value
                  && std::is_standard_layout<HartUserdataState>::value
                  && std::is_trivial<HartTransformDesc>::value
                  && std::is_standard_layout<HartTransformDesc>::value
                  && std::is_trivial<HartTransformState>::value
                  && std::is_standard_layout<HartTransformState>::value,
              "HART texture records must be POD");

}  // namespace testshade
