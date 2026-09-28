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
    void* interactive;
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
    const int* shader_ids;
    const unsigned* light_primitives;
    const unsigned* material_is_light;
    Vec3* background_values;
    float* background_rows;
    float* background_cols;
    unsigned triangle_count;
    unsigned light_count;
    int background_material;
    unsigned background_resolution;
    unsigned background_offset;
    int rr_depth;
};

OSL_NAMESPACE_END
