// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "../testshade/harttextureparams.h"
#include "raytracer.h"

OSL_NAMESPACE_BEGIN

struct HartMaterialBinding {
    unsigned callable;
    unsigned local;
};

struct HartPathParams {
    uint64_t traversable;
    Camera camera;
    const Vec3* vertices;
    const Vec3* normals;
    const Vec2* uvs;
    const TriangleIndices* triangles;
    const TriangleIndices* normal_indices;
    const TriangleIndices* uv_indices;
    const float* surfaceareas;
    const HartMaterialBinding* materials;
    const testshade::HartTextureState* textures;
    unsigned char* scratch;
    uint64_t scratch_stride;
    Color3* output;
    unsigned material_count;
    unsigned max_bounces;
    unsigned aa;
    unsigned no_jitter;
    unsigned fused;
};

OSL_NAMESPACE_END
