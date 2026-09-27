// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "hart/material.h"

#include <limits>

OSL_NAMESPACE_BEGIN

struct HartMaterialResult {
    float values[22];
    uint32_t valid, extent;
};

struct HartMaterialTestParams {
    HartMaterialResult* results;
};

constexpr unsigned HartMaterialCases = 21;
constexpr unsigned HartMaterialRows  = 3;
static_assert(sizeof(HartMaterialResult) == 96
                  && sizeof(HartMaterialTestParams) == 8,
              "Unexpected material probe ABI");

namespace {

template<class T>
OSL_HOSTDEVICE ClosureComponent*
material_component(testshade::HartClosurePool& pool, int id, const T& params,
                   Color3 weight = Color3(.25f, .5f, .75f))
{
    static_assert(alignof(T) <= alignof(ClosureComponent),
                  "Closure parameters require a more aligned arena");
    auto* ptr = pool.allocate(sizeof(ClosureComponent) + sizeof(T),
                              alignof(ClosureComponent));
    if (!ptr)
        return nullptr;
    auto* result = new (ptr) ClosureComponent;
    result->id   = id;
    result->w    = weight;
    new (result->data()) T(params);
    return result;
}



OSL_HOSTDEVICE ClosureColor*
material_add(testshade::HartClosurePool& pool, const ClosureColor* a,
             const ClosureColor* b)
{
    auto* ptr = pool.allocate(sizeof(ClosureAdd), alignof(ClosureAdd));
    if (!ptr)
        return nullptr;
    auto* result     = new (ptr) ClosureAdd;
    result->id       = ClosureColor::ADD;
    result->closureA = a;
    result->closureB = b;
    return result;
}



OSL_HOSTDEVICE HartMaterialResult
material_probe(unsigned test, unsigned row)
{
    HartMaterialResult result { };
    alignas(16) unsigned char memory[4096];
    testshade::HartClosurePool pool { memory, sizeof(memory) };
    const Vec3 N(0, 0, 1), T(1, 0, 0);
    const Vec3 wo = Vec3(.2f + .1f * row, .15f - .05f * row, 1).normalized();
    const Vec3 wi = Vec3(-.2f, .3f + .05f * row,
                         test >= 7 && test <= 9 ? -1 : 1)
                        .normalized();
    const DiffuseParams diffuse { N };
    const PhongParams phong { N, 8 };
    MicrofacetParams micro {
        ustringhash(strhash("ggx")), N, T, .25f, .4f, 1.5f, 0
    };
    ClosureColor* root = nullptr;
    switch (test) {
    case 0: root = material_component(pool, DIFFUSE_ID, diffuse); break;
    case 1:
        root = material_component(pool, OREN_NAYAR_ID,
                                  OrenNayarParams { N, .3f });
        break;
    case 2: root = material_component(pool, PHONG_ID, phong); break;
    case 3:
        root = material_component(pool, WARD_ID, WardParams { N, T, .3f, .4f });
        break;
    case 4:
    case 5:
        if (test == 5)
            micro.dist = ustringhash(strhash("beckmann"));
        root = material_component(pool, MICROFACET_ID, micro);
        break;
    case 6:
        root = material_component(pool, FRESNEL_REFLECTION_ID,
                                  ReflectionParams { N, 1.5f });
        break;
    case 7:
        root = material_component(pool, REFRACTION_ID,
                                  RefractionParams { N, 1.5f });
        break;
    case 8: root = material_component(pool, TRANSLUCENT_ID, diffuse); break;
    case 9:
        root = material_component(pool, TRANSPARENT_ID, EmptyParams { });
        break;
    case 10: {
        MxBurleyDiffuse::Data params { };
        params.N         = N;
        params.albedo    = Color3(.6f, .4f, .2f);
        params.roughness = .3f;
        root = material_component(pool, MX_BURLEY_DIFFUSE_ID, params);
        break;
    }
    case 11:
    case 12: {
        MxSheen::Data params { };
        params.N         = N;
        params.albedo    = Color3(.2f, .4f, .6f);
        params.roughness = .35f;
        params.mode      = int(row % 2);
        root             = material_component(pool, MX_SHEEN_ID, params);
        if (test == 12) {
            auto* base          = material_component(pool, DIFFUSE_ID, diffuse);
            const Color3 weight = row == 0   ? Color3(1)
                                  : row == 1 ? Color3(.25f, .5f, .75f)
                                             : Color3(0);
            root = material_component(pool, MX_LAYER_ID,
                                      MxLayerParams { root, base }, weight);
        }
        break;
    }
    case 13: {
        auto* a = material_component(pool, DIFFUSE_ID, diffuse);
        auto* b = material_component(pool, PHONG_ID, phong);
        auto* c = material_component(pool, MICROFACET_ID, micro);
        root    = material_add(pool, material_add(pool, a, b), c);
        break;
    }
    case 14: root = material_component(pool, 9999, diffuse); break;
    case 15:
        micro.dist = ustringhash(strhash("invalid_distribution"));
        root       = material_component(pool, MICROFACET_ID, micro);
        break;
    case 16:
        root = material_component(pool, DIFFUSE_ID, diffuse);
        --pool.used;
        break;
    case 17: {
        auto* add = static_cast<ClosureAdd*>(
            material_add(pool, nullptr, nullptr));
        add->closureA = add;
        root          = add;
        break;
    }
    case 18:
        root = material_component(pool, DIFFUSE_ID, diffuse);
        for (int i = 0; i < 18; ++i)
            root = material_add(pool, root, nullptr);
        break;
    case 19: {
        CompositeBSDF bsdf;
        for (unsigned i = 0; i < 33; ++i) {
            if (!bsdf.add_bsdf<Diffuse<0>>(Color3(1), diffuse)) {
                result.valid = i != 32;
                return result;
            }
        }
        result.valid = 1;
        return result;
    }
    case 20:
        root = material_component(pool, DIFFUSE_ID, diffuse,
                                  Color3(std::numeric_limits<float>::max()));
        break;
    }
    if (!hart_valid_closure(root, pool))
        return result;
    ShaderGlobals sg { };
    sg.N = sg.Ng = N;
    sg.I         = -wo;
    ShadingResult shading;
    MediumStack medium;
    if (!process_closure(sg, .2f, shading, medium, root, false))
        return result;
    if (!shading.bsdf.prepare(wo, Color3(1), false))
        return result;
    const auto albedo = shading.bsdf.get_albedo(wo);
    const auto eval   = shading.bsdf.eval(wo, wi);
    const auto sample = shading.bsdf.sample(wo, .2f + .2f * row, .37f, .61f);
    for (int c = 0; c < 3; ++c) {
        result.values[c]      = albedo[c];
        result.values[3 + c]  = eval.weight[c];
        result.values[8 + c]  = eval.wi[c];
        result.values[11 + c] = sample.weight[c];
        result.values[16 + c] = sample.wi[c];
        result.values[19 + c] = shading.Le[c];
    }
    result.values[6]  = eval.pdf;
    result.values[7]  = eval.roughness;
    result.values[14] = sample.pdf;
    result.values[15] = sample.roughness;
    result.valid      = 1;
    result.extent     = uint32_t(pool.used);
    return result;
}

}  // namespace

OSL_NAMESPACE_END
