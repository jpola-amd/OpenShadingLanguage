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

enum class MediumProbeCase : unsigned {
    Absorption,
    Vacuum,
    InfiniteAbsorption,
    Scatter,
    ZeroSegment,
    Overlap,
    Priority,
    LayerTop,
    LayerBase,
    Medium,
    VacuumIor,
    Absorbed,
    InvalidCoefficients,
    InvalidPhaseIor,
    Capacity,
    Overflow,
    InvalidSegment,
    InvalidMedium,
    InvalidAnisotropic,
    Count
};

constexpr unsigned HartSurfaceCases = 27;
constexpr unsigned HartMaterialCases
    = HartSurfaceCases + static_cast<unsigned>(MediumProbeCase::Count);
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
medium_probe(MediumProbeCase test, unsigned row)
{
    HartMaterialResult result { };
    MediumStack stack;
    MediumParams params;
    params.sigma_t = Color3(0, .5f, 1);
    Color3 weight(1);
    Ray ray(Vec3(1, 2, 3), Vec3(0, 0, 1), .125f, .025f, .2f, Ray::REFRACTION);
    Sampler sampler(13, int(row), 7 + int(row));
    float pdf = .25f, distance = .5f + .25f * row;
    auto expected_event = MediumStack::Event::Surface;
    switch (test) {
    case MediumProbeCase::Absorption:
        params.sigma_t = Color3(0, .4f, .6f);
        break;
    case MediumProbeCase::Vacuum: params.sigma_t = Color3(0); break;
    case MediumProbeCase::InfiniteAbsorption:
        distance = std::numeric_limits<float>::infinity();
        break;
    case MediumProbeCase::Scatter:
        params.sigma_t = params.sigma_s = row == 0   ? Color3(0, 2, 4)
                                          : row == 1 ? Color3(2, 0, 4)
                                                     : Color3(2, 4, 0);
        distance       = std::numeric_limits<float>::infinity();
        expected_event = MediumStack::Event::Scatter;
        break;
    case MediumProbeCase::ZeroSegment:
        params.sigma_s = .5f * params.sigma_t;
        distance       = 0;
        break;
    case MediumProbeCase::Overlap:
        if (!stack.add_medium(params))
            return result;
        params.sigma_s  = .5f * params.sigma_t;
        params.medium_g = .2f * (int(row) - 1);
        if (!stack.add_medium(params) || stack.cdf[0] != 0 || stack.cdf[1] != 1)
            return result;
        distance       = std::numeric_limits<float>::infinity();
        expected_event = MediumStack::Event::Scatter;
        break;
    case MediumProbeCase::Priority: {
        params.priority       = 1;
        params.refraction_ior = 1.25f;
        auto high             = params;
        high.priority         = 3;
        high.refraction_ior   = 1.75f;
        auto low              = params;
        low.priority          = 0;
        if (!stack.add_medium(params) || !stack.add_medium(high)
            || !stack.add_medium(low)
            || stack.get_current_params()->refraction_ior != 1.75f
            || !stack.false_intersection_with(high)
            || stack.false_intersection_with(low) || !stack.pop_medium()
            || stack.get_current_params()->priority != 3 || !stack.pop_medium()
            || stack.get_current_params()->refraction_ior != 1.25f
            || !stack.pop_medium() || !stack.pop_medium())
            return result;
        for (int i = 0; i < 2 * MediumStack::MaxEntries; ++i)
            if (!stack.add_medium(params) || !stack.pop_medium()
                || stack.in_medium() || stack.pool_size != 0)
                return result;
        break;
    }
    case MediumProbeCase::LayerTop:
    case MediumProbeCase::LayerBase:
    case MediumProbeCase::Medium:
    case MediumProbeCase::VacuumIor: {
        alignas(16) unsigned char memory[1024];
        testshade::HartClosurePool pool { memory, sizeof(memory) };
        ClosureColor* root = nullptr;
        if (test == MediumProbeCase::LayerTop
            || test == MediumProbeCase::LayerBase) {
            MxAnisotropicVdfParams vdf { };
            vdf.albedo     = Color3(.5f);
            vdf.extinction = Color3(.2f, .4f, .6f);
            vdf.anisotropy = .3f;
            auto* volume = material_component(pool, MX_ANISOTROPIC_VDF_ID, vdf,
                                              Color3(1));
            if (!volume)
                return result;
            const bool top     = test == MediumProbeCase::LayerTop;
            const Color3 scale = row == 0   ? Color3(0)
                                 : row == 1 ? Color3(.2f, .4f, .6f)
                                            : Color3(1);
            root = material_component(pool, MX_LAYER_ID,
                                      MxLayerParams { top ? volume : nullptr,
                                                      top ? nullptr : volume },
                                      scale);
            distance = 0;
        } else {
            MxMediumVdfParams vdf { };
            vdf.albedo             = Color3(0);
            vdf.transmission_color = test == MediumProbeCase::VacuumIor
                                         ? Color3(0)
                                         : Color3(1, .5f, .25f);
            vdf.transmission_depth = test == MediumProbeCase::VacuumIor ? 0 : 2;
            vdf.ior                = 1.5f;
            vdf.priority           = 2 + int(row);
            root = material_component(pool, MX_MEDIUM_VDF_ID, vdf, Color3(1));
        }
        auto* boundary = material_component(pool, TRANSPARENT_ID,
                                            EmptyParams { }, Color3(1));
        if (!root || !boundary)
            return result;
        root = material_add(pool, boundary, root);
        if (!root || !hart_valid_closure(root, pool))
            return result;
        ShaderGlobals sg { };
        sg.N = sg.Ng  = Vec3(0, 0, 1);
        sg.I          = -sg.N;
        sg.backfacing = row == 1;
        ShadingResult shading;
        if (!process_closure(sg, .2f, shading, stack, root, false)
            || !shading.bsdf.prepare(sg.N, Color3(1), false))
            return result;
        const auto sample = shading.bsdf.sample(sg.N, .2f, .3f, .4f);
        if (sample.wi != -sg.N || sample.weight != Color3(1)
            || sample.pdf != std::numeric_limits<float>::infinity())
            return result;
        params        = shading.medium_data;
        result.extent = uint32_t(pool.used);
        break;
    }
    case MediumProbeCase::Absorbed:
        params.sigma_t = Color3(1);
        distance       = std::numeric_limits<float>::infinity();
        expected_event = MediumStack::Event::Absorbed;
        break;
    case MediumProbeCase::InvalidCoefficients:
    case MediumProbeCase::InvalidPhaseIor:
        params.sigma_t = Color3(1);
        if (test == MediumProbeCase::InvalidCoefficients) {
            if (row == 0)
                params.sigma_t.x = -1;
            if (row == 1)
                params.sigma_s.y = 2;
            if (row == 2)
                params.sigma_t.z = std::numeric_limits<float>::infinity();
        } else {
            if (row == 0)
                params.medium_g = 1;
            if (row == 1)
                params.medium_g = std::numeric_limits<float>::quiet_NaN();
            if (row == 2)
                params.refraction_ior = 0;
        }
        result.valid = params.valid() || stack.add_medium(params)
                       || stack.size() != 0 || stack.pool_size != 0;
        return result;
    case MediumProbeCase::Capacity:
        result.valid = 1;
        for (int i = 0; i < MediumStack::MaxEntries; ++i)
            if (!stack.add_medium(params))
                return result;
        result.valid = stack.add_medium(params)
                       || stack.size() != MediumStack::MaxEntries
                       || stack.pool_size != MediumStack::MaxEntries
                       || stack.current_params.sigma_t
                              != float(MediumStack::MaxEntries)
                                     * params.sigma_t;
        return result;
    case MediumProbeCase::Overflow:
        result.valid   = 1;
        params.sigma_t = Color3(.75f * std::numeric_limits<float>::max());
        if (!stack.add_medium(params))
            return result;
        result.valid = stack.add_medium(params) || stack.size() != 1
                       || stack.pool_size != 1
                       || stack.current_params.sigma_t != params.sigma_t
                       || !stack.pop_medium() || stack.in_medium()
                       || stack.pool_size != 0;
        return result;
    case MediumProbeCase::InvalidSegment: {
        result.valid = 1;
        if (!stack.add_medium(params))
            return result;
        if (row == 0)
            distance = std::numeric_limits<float>::quiet_NaN();
        if (row == 1)
            weight.x = -1;
        if (row == 2)
            ray.direction = Vec3(0);
        const Vec3 origin = ray.origin, direction = ray.direction;
        const Color3 original_weight = weight;
        result.valid = stack.integrate(ray, sampler, distance, weight, pdf)
                           != MediumStack::Event::Error
                       || ray.origin != origin || ray.direction != direction
                       || weight != original_weight || pdf != .25f;
        return result;
    }
    case MediumProbeCase::InvalidMedium:
    case MediumProbeCase::InvalidAnisotropic: {
        result.valid = 1;
        alignas(16) unsigned char memory[256];
        testshade::HartClosurePool pool { memory, sizeof(memory) };
        ClosureColor* root = nullptr;
        if (test == MediumProbeCase::InvalidMedium) {
            MxMediumVdfParams vdf { };
            vdf.albedo             = Color3(0);
            vdf.transmission_color = Color3(.5f);
            vdf.transmission_depth = row == 0 ? 0 : 1;
            vdf.ior                = row == 2 ? 0 : 1.5f;
            if (row == 1)
                vdf.transmission_color.x = 1.1f;
            root = material_component(pool, MX_MEDIUM_VDF_ID, vdf, Color3(1));
        } else {
            MxAnisotropicVdfParams vdf { };
            vdf.extinction = Color3(row == 0 ? -1 : 1);
            vdf.albedo     = Color3(
                row == 1 ? std::numeric_limits<float>::quiet_NaN() : .5f);
            vdf.anisotropy = row == 2 ? -1 : 0;
            root = material_component(pool, MX_ANISOTROPIC_VDF_ID, vdf,
                                      Color3(1));
        }
        if (!root)
            return result;
        ShaderGlobals sg { };
        ShadingResult shading, light;
        result.valid = hart_valid_closure(root, pool)
                       || process_closure(sg, 0, shading, stack, root, false)
                       || process_closure(sg, 0, light, stack, root, true);
        return result;
    }
    case MediumProbeCase::Count: return result;
    }
    if (!stack.in_medium() && !stack.add_medium(params))
        return result;
    const auto event = stack.integrate(ray, sampler, distance, weight, pdf);
    if (event != expected_event || ray.radius != .125f || ray.spread != .025f
        || ray.roughness != .2f || ray.raytype != Ray::REFRACTION)
        return result;
    // Medium values: throughput, origin, phase PDF, direction, stack size/IOR,
    // aggregate coefficients, selected phase g/priority, event, pool size.
    for (int c = 0; c < 3; ++c) {
        result.values[c]      = weight[c];
        result.values[3 + c]  = ray.origin[c];
        result.values[7 + c]  = ray.direction[c];
        result.values[12 + c] = stack.current_params.sigma_t[c];
        result.values[15 + c] = stack.current_params.sigma_s[c];
    }
    result.values[6]  = pdf;
    result.values[10] = float(stack.size());
    result.values[11] = stack.get_current_params()->refraction_ior;
    result.values[18] = params.medium_g;
    result.values[19] = float(stack.get_current_params()->priority);
    result.values[20] = float(event);
    result.values[21] = float(stack.pool_size);
    result.valid      = 1;
    return result;
}



OSL_HOSTDEVICE HartMaterialResult
grazing_microfacet_probe(unsigned test, unsigned row)
{
    HartMaterialResult result { };
    alignas(16) unsigned char memory[1024];
    testshade::HartClosurePool pool { memory, sizeof(memory) };
    const Vec3 N(0, 0, 1), T(1, 0, 0);
    const float z = row == 0 ? 1e-12f : row == 1 ? 1e-20f : 1e-21f;
    const Vec3 wo = Vec3(1, 0, z).normalized();
    const Vec3 wi = Vec3(-.8f, .6f, 2 * z).normalized();
    const MicrofacetParams params { test == 21
                                        ? ustringhash(strhash("beckmann"))
                                        : ustringhash(strhash("ggx")),
                                    N,
                                    T,
                                    .001f,
                                    .01f,
                                    1.5f,
                                    0 };
    auto* root = material_component(pool, MICROFACET_ID, params);
    ShaderGlobals sg { };
    sg.N = sg.Ng = N;
    sg.I         = -wo;
    ShadingResult shading;
    MediumStack medium;
    if (!hart_valid_closure(root, pool)
        || !process_closure(sg, .2f, shading, medium, root, false)
        || !shading.bsdf.prepare(wo, Color3(1), false))
        return result;
    const auto eval  = shading.bsdf.eval(wo, wi);
    result.values[0] = eval.pdf;
    for (int c = 0; c < 3; ++c)
        result.values[1 + c] = eval.weight[c];
    result.values[4] = eval.roughness;
    result.valid     = 1;
    result.extent    = uint32_t(pool.used);
    return result;
}



OSL_HOSTDEVICE HartMaterialResult
mis_density_probe(unsigned row)
{
    HartMaterialResult result { };
    const float scale = row == 0   ? std::numeric_limits<float>::denorm_min()
                        : row == 1 ? std::numeric_limits<float>::min()
                                   : 1.0f;
    // Keep division operands live; constant folding would bypass device math.
    volatile float inputs[] = { 4 * scale, 8 * scale, 0 };
    const float a = inputs[0], b = inputs[1], zero = inputs[2];
    Color3 weight(0);
    float pdf = zero;
    MIS::update_eval(&weight, &pdf, Color3(.125f, .25f, .375f), a, .25f);
    result.values[21] = pdf;
    MIS::update_eval(&weight, &pdf, Color3(.5f, .25f, .125f), b, .5f);
    result.values[0] = pdf;
    for (int c = 0; c < 3; ++c)
        result.values[1 + c] = weight[c];
    const float pairs[][2]
        = { { a, b }, { b, a }, { zero, b }, { b, zero }, { a, a } };
    for (unsigned i = 0; i < std::size(pairs); ++i) {
        result.values[4 + 3 * i]
            = MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(pairs[i][0],
                                                       pairs[i][1]);
        result.values[5 + 3 * i]
            = MIS::power_heuristic<MIS::WEIGHT_EVAL>(pairs[i][0], pairs[i][1]);
        result.values[6 + 3 * i]
            = MIS::power_heuristic<MIS::EVAL_WEIGHT>(pairs[i][0], pairs[i][1]);
    }
    result.values[19] = a;
    result.values[20] = b;
    result.valid      = 1;
    return result;
}



OSL_HOSTDEVICE HartMaterialResult
captured_microfacet_probe(unsigned row)
{
    HartMaterialResult result { };
    alignas(16) unsigned char memory[1024];
    testshade::HartClosurePool pool { memory, sizeof(memory) };
    const Vec3 N(.652908266f, .259998649f, .711415112f);
    const Vec3 U(.736752808f, 0, -.676162183f);
    const Vec3 wo(-.114152454f, .362606257f, .924924791f);
    const Vec3 wi(.580448151f, .402788311f, .707701564f);
    const float alpha = .05f + .001f * row;
    MicrofacetParams params {
        ustringhash(strhash("beckmann")), N, U, alpha, alpha, 1.5f, 1
    };
    const auto* transmission = material_component(pool, MICROFACET_ID, params,
                                                  Color3(1));
    params.refract           = 0;
    const auto* reflection   = material_component(pool, MICROFACET_ID, params,
                                                  Color3(1));
    const auto* root         = material_add(pool, transmission, reflection);
    ShaderGlobals sg { };
    sg.N  = N;
    sg.Ng = Vec3(.668885171f, .242914349f, .702556312f);
    sg.I  = -wo;
    ShadingResult shading;
    MediumStack medium;
    if (!hart_valid_closure(root, pool)
        || !process_closure(sg, 0, shading, medium, root, false)
        || !shading.bsdf.prepare(wo, Color3(1), false))
        return result;
    const auto eval  = shading.bsdf.eval(wo, wi);
    result.values[0] = eval.pdf;
    result.values[4] = eval.roughness;
    for (int c = 0; c < 3; ++c) {
        result.values[1 + c]  = eval.weight[c];
        result.values[5 + c]  = wo[c];
        result.values[8 + c]  = wi[c];
        result.values[11 + c] = N[c];
    }
    result.values[14] = params.eta;
    result.valid      = 1;
    result.extent     = uint32_t(pool.used);
    return result;
}



OSL_HOSTDEVICE HartMaterialResult
captured_microfacet_sample(unsigned test, unsigned row)
{
    HartMaterialResult result { };
    alignas(16) unsigned char memory[1024];
    testshade::HartClosurePool pool { memory, sizeof(memory) };
    volatile float inputs[] = { -.223643467f, .347753167f, .91052264f,
                                .971134841f,  0,           .238531098f,
                                -.309215724f, .345641434f, .88595587f };
    const Vec3 N(inputs[0], inputs[1], inputs[2]);
    const Vec3 U(inputs[3], inputs[4], inputs[5]);
    const Vec3 wo(inputs[6], inputs[7], inputs[8]);
    MicrofacetParams params { test == 25 ? ustringhash(strhash("beckmann"))
                                         : ustringhash(strhash("ggx")),
                              N,
                              U,
                              .001f,
                              .01f,
                              1.5f,
                              1 };
    const auto* transmission = material_component(pool, MICROFACET_ID, params,
                                                  Color3(1));
    params.refract           = 0;
    const auto* reflection   = material_component(pool, MICROFACET_ID, params,
                                                  Color3(1));
    const auto* root         = material_add(pool, transmission, reflection);
    ShaderGlobals sg { };
    sg.N = sg.Ng = N;
    sg.I         = -wo;
    ShadingResult shading;
    MediumStack medium;
    if (!hart_valid_closure(root, pool)
        || !process_closure(sg, 0, shading, medium, root, false)
        || !shading.bsdf.prepare(wo, Color3(1), false))
        return result;
    const auto sample = shading.bsdf.sample(wo, .279518962f + .1f * row,
                                            .655218065f, .120993137f);
    const auto eval   = shading.bsdf.eval(wo, sample.wi);
    result.values[0]  = sample.pdf;
    result.values[4]  = sample.roughness;
    result.values[8]  = eval.pdf;
    result.values[12] = eval.roughness;
    for (int c = 0; c < 3; ++c) {
        result.values[1 + c] = sample.weight[c];
        result.values[5 + c] = sample.wi[c];
        result.values[9 + c] = eval.weight[c];
    }
    Vec3 stretched = TangentFrame::from_normal_and_tangent(N, U).tolocal(wo);
    stretched.x *= params.xalpha;
    stretched.y *= params.yalpha;
    result.values[13] = stretched.normalized().z;
    result.valid      = 1;
    result.extent     = uint32_t(pool.used);
    return result;
}



OSL_HOSTDEVICE HartMaterialResult
material_probe(unsigned test, unsigned row)
{
    if (test >= HartSurfaceCases)
        return medium_probe(static_cast<MediumProbeCase>(test
                                                         - HartSurfaceCases),
                            row);
    if (test == 21 || test == 22)
        return grazing_microfacet_probe(test, row);
    if (test == 23)
        return mis_density_probe(row);
    if (test == 24)
        return captured_microfacet_probe(row);
    if (test == 25 || test == 26)
        return captured_microfacet_sample(test, row);
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
