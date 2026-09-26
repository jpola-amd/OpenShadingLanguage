// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <cstdint>

OSL_NAMESPACE_BEGIN

struct HartTriangle {
    unsigned a, b, c;
};

struct HartHitRecordData {
    unsigned material;
};

struct HartProbeRay {
    Vec3 origin, direction;
    float tmin, tmax;
};

struct HartProbeHit {
    unsigned primitive, material;
    float t, u, v;
    Vec3 normal;
};

struct HartProbeParams {
    uint64_t traversable;
    const HartProbeRay* rays;
    HartProbeHit* hits;
    const Vec3* vertices;
    const HartTriangle* triangles;
    uint64_t ray_count;
};

static_assert(sizeof(HartTriangle) == 12 && sizeof(HartHitRecordData) == 4,
              "HART triangle and SBT layouts must match on host and device");
static_assert(sizeof(HartProbeRay) == 32 && sizeof(HartProbeHit) == 32
                  && sizeof(HartProbeParams) == 48,
              "HART probe layouts must match on host and device");

OSL_NAMESPACE_END
