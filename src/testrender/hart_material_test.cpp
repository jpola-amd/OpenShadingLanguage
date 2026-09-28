// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <array>
#include <cmath>

#include <OpenImageIO/unittest.h>

#include "hart_material_bitcode.h"
#include "hartcontext.h"
#include "hartmaterialtest.h"

using namespace OSL;

namespace {

class MaterialDiagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR
            || (code & 0xffff0000) == EH_SEVERE) {
            ++errors;
            last_error = message;
        }
        ErrorHandler::operator()(code, message);
    }

    int errors = 0;
    std::string last_error;
};



void
check_boundary_offsets()
{
    for (float scale : { .001f, 1.f, 1000.f }) {
        Scene scene;
        scene.verts     = { Vec3(scale, 0, 0), Vec3(0, scale, 0),
                            Vec3(0, 0, scale) };
        scene.triangles = { { 0, 1, 2 } };
        const Vec3 point(scale / 3);
        const Vec3 normal = Vec3(1).normalized();
        for (float side : { -1.f, 1.f }) {
            const Vec3 direction   = side * normal;
            const Vec3 origin      = scene.offset_ray_origin(0, point, normal,
                                                             direction);
            const float separation = (origin - point).dot(direction);
            OIIO_CHECK_ASSERT(separation > 0);
            OIIO_CHECK_ASSERT(separation < 1e-5f * scale);
        }
    }
}



void
check_layer_weights()
{
    const Vec3 N(0, 0, 1);
    ShaderGlobals sg { };
    sg.N = sg.Ng = N;
    sg.I         = -N;
    for (const Color3 weight :
         { Color3(0), Color3(.2f, .4f, .6f), Color3(1) }) {
        alignas(16) unsigned char memory[4096];
        testshade::HartClosurePool pool { memory, sizeof(memory) };
        auto* diffuse = material_component(pool, DIFFUSE_ID,
                                           DiffuseParams { N }, Color3(1));
        MxSheen::Data sheen { };
        sheen.N         = N;
        sheen.albedo    = Color3(.2f, .3f, .5f);
        sheen.roughness = .35f;
        sheen.mode      = 1;
        auto* coat  = material_component(pool, MX_SHEEN_ID, sheen, Color3(1));
        auto* layer = material_component(pool, MX_LAYER_ID,
                                         MxLayerParams { coat, diffuse },
                                         Color3(1));
        for (auto* root : { diffuse, layer }) {
            const Color3 opacity = evaluate_layer_opacity(sg, .2f, root);
            ShadingResult plain, scaled;
            MediumStack medium;
            OIIO_CHECK_ASSERT(
                process_closure(sg, .2f, plain, medium, root, false));
            root->w                     = weight;
            const Color3 scaled_opacity = evaluate_layer_opacity(sg, .2f, root);
            OIIO_CHECK_ASSERT(
                process_closure(sg, .2f, scaled, medium, root, false));
            const Color3 albedo        = plain.bsdf.get_albedo(N);
            const Color3 scaled_albedo = scaled.bsdf.get_albedo(N);
            for (int c = 0; c < 3; ++c) {
                OIIO_CHECK_EQUAL_THRESH(scaled_opacity[c],
                                        weight[c] * opacity[c], 2e-6f);
                OIIO_CHECK_EQUAL_THRESH(scaled_albedo[c], weight[c] * albedo[c],
                                        2e-6f);
            }
            root->w = Color3(1);
        }
        MxGeneralizedSchlick::Data schlick { };
        schlick.refr_tint = Color3(1);
        auto* opaque      = material_component(pool, MX_GENERALIZED_SCHLICK_ID,
                                               schlick, weight);
        const Color3 opacity = evaluate_layer_opacity(sg, .2f, opaque);
        for (int c = 0; c < 3; ++c)
            OIIO_CHECK_EQUAL_THRESH(opacity[c], weight[c], 2e-6f);

        MxAnisotropicVdfParams volume { };
        volume.albedo     = Color3(.5f);
        volume.extinction = Color3(.2f, .4f, .6f);
        auto* vdf = material_component(pool, MX_ANISOTROPIC_VDF_ID, volume,
                                       Color3(1));
        for (bool top : { false, true }) {
            auto* medium_layer
                = material_component(pool, MX_LAYER_ID,
                                     MxLayerParams { top ? vdf : nullptr,
                                                     top ? nullptr : vdf },
                                     weight);
            ShadingResult shading;
            MediumStack medium;
            OIIO_CHECK_ASSERT(process_medium_closure(sg, .2f, shading, medium,
                                                     medium_layer, Color3(1)));
            for (int c = 0; c < 3; ++c) {
                OIIO_CHECK_EQUAL_THRESH(shading.medium_data.sigma_t[c],
                                        weight[c] * volume.extinction[c],
                                        2e-6f);
                OIIO_CHECK_EQUAL_THRESH(shading.medium_data.sigma_s[c],
                                        weight[c] * volume.extinction[c] * .5f,
                                        2e-6f);
            }
        }
    }
}



bool
material_probe_rejection(unsigned test)
{
    return (test >= 14 && test <= 20)
           || test >= HartSurfaceCases
                          + static_cast<unsigned>(
                              MediumProbeCase::InvalidCoefficients);
}



void
check_grazing_microfacets()
{
    for (unsigned row = 0; row < HartMaterialRows; ++row) {
        const auto beckmann = material_probe(21, row);
        OIIO_CHECK_EQUAL(beckmann.valid, 1);
        OIIO_CHECK_EQUAL(beckmann.values[0], 0);
        for (int c = 1; c < 4; ++c) {
            OIIO_CHECK_ASSERT(std::isfinite(beckmann.values[c])
                              && beckmann.values[c] >= 0);
            OIIO_CHECK_EQUAL(beckmann.values[0] * beckmann.values[c], 0);
        }
        const float z = row == 0 ? 1e-12f : row == 1 ? 1e-20f : 1e-21f;
        const Vec3 wo = Vec3(1, 0, z).normalized();
        const Vec3 wi = Vec3(-.8f, .6f, 2 * z).normalized();
        const Imath::V3d o(wo), i(wi);
        const Imath::V3d h = (i + o).normalized();
        const double ax = .001f, ay = .01f;
        const double q   = h.z * h.z + (h.x / ax) * (h.x / ax)
                           + (h.y / ay) * (h.y / ay);
        const double d   = 1 / (M_PI * ax * ay * q * q);
        const double a2  = (ax * o.x) * (ax * o.x) + (ay * o.y) * (ay * o.y);
        const double g1  = 2 / (1 + std::sqrt(1 + a2 / (o.z * o.z)));
        const double pdf = .25 * g1 * d / o.z;
        const auto ggx   = material_probe(22, row);
        OIIO_CHECK_EQUAL(ggx.valid, 1);
        OIIO_CHECK_ASSERT(pdf > 0);
        OIIO_CHECK_EQUAL_THRESH(ggx.values[0], float(pdf), float(pdf) * 2e-5f);
    }
}



void
check_mis_values(const HartMaterialResult& actual, unsigned row)
{
    const float scale = row == 0   ? std::numeric_limits<float>::denorm_min()
                        : row == 1 ? std::numeric_limits<float>::min()
                                   : 1.0f;
    OIIO_CHECK_EQUAL(actual.valid, 1);
    OIIO_CHECK_EQUAL(actual.values[0], 5 * scale);
    OIIO_CHECK_EQUAL(actual.values[19], 4 * scale);
    OIIO_CHECK_EQUAL(actual.values[20], 8 * scale);
    OIIO_CHECK_EQUAL(actual.values[21], scale);
    const float weight[] = { .9f, .6f, .5f };
    for (int c = 0; c < 3; ++c)
        OIIO_CHECK_EQUAL_THRESH(actual.values[1 + c], weight[c], 2e-6f);
    const double pairs[][2] = { { 4 * double(scale), 8 * double(scale) },
                                { 8 * double(scale), 4 * double(scale) },
                                { 0, 8 * double(scale) },
                                { 8 * double(scale), 0 },
                                { 4 * double(scale), 4 * double(scale) } };
    for (unsigned i = 0; i < std::size(pairs); ++i) {
        const double a = pairs[i][0], b = pairs[i][1];
        const double mis       = a * a / (a * a + b * b);
        const float expected[] = { float(b * mis), float(mis),
                                   float(a == 0 ? 0 : mis * b / a) };
        for (unsigned c = 0; c < 3; ++c) {
            const float value = actual.values[4 + 3 * i + c];
            OIIO_CHECK_ASSERT(std::isfinite(value));
            if (expected[c] == 0
                || (expected[c] > 0
                    && expected[c] < std::numeric_limits<float>::min()))
                OIIO_CHECK_EQUAL(value, expected[c]);
            else
                OIIO_CHECK_EQUAL_THRESH(value, expected[c], 2e-6f);
        }
    }
}



void
check_captured_microfacet(const HartMaterialResult& actual, unsigned row)
{
    OIIO_CHECK_EQUAL(actual.valid, 1);
    OIIO_CHECK_ASSERT(actual.values[0] > 0);
    if (row == 0)
        OIIO_CHECK_ASSERT(actual.values[0] < std::numeric_limits<float>::min());
    const Imath::V3d wo(actual.values[5], actual.values[6], actual.values[7]);
    const Imath::V3d wi(actual.values[8], actual.values[9], actual.values[10]);
    const Imath::V3d N(actual.values[11], actual.values[12], actual.values[13]);
    const double eta = actual.values[14];
    auto fresnel     = [eta](double cosine) {
        const double transmitted = std::sqrt(
            1 - (1 - cosine * cosine) / (eta * eta));
        const double s = (cosine - eta * transmitted)
                         / (cosine + eta * transmitted);
        const double p = (eta * cosine - transmitted)
                         / (eta * cosine + transmitted);
        return .5 * (s * s + p * p);
    };
    const double expected = fresnel(wo.dot((wo + wi).normalized()))
                            / fresnel(wo.dot(N));
    for (int c = 1; c < 4; ++c) {
        OIIO_CHECK_ASSERT(std::isfinite(actual.values[c]));
        OIIO_CHECK_EQUAL_THRESH(actual.values[c], float(expected), 2e-5f);
    }
}



void
check_sampled_microfacet(const HartMaterialResult& actual)
{
    OIIO_CHECK_EQUAL(actual.valid, 1);
    OIIO_CHECK_ASSERT(std::isfinite(actual.values[0]) && actual.values[0] > 0);
    OIIO_CHECK_ASSERT(std::isfinite(actual.values[8]) && actual.values[8] > 0);
    const Vec3 wi(actual.values[5], actual.values[6], actual.values[7]);
    OIIO_CHECK_EQUAL_THRESH(wi.length2(), 1, 3e-6f);
    OIIO_CHECK_ASSERT(wi.dot(Vec3(-.223643467f, .347753167f, .91052264f)) < 0);
    for (int c = 0; c < 3; ++c) {
        OIIO_CHECK_EQUAL_THRESH(actual.values[1 + c], 1, .005f);
        OIIO_CHECK_EQUAL_THRESH(actual.values[9 + c], 1, .005f);
    }
}



void
check_medium_values()
{
    for (unsigned row = 0; row < HartMaterialRows; ++row)
        for (unsigned index = HartSurfaceCases; index < HartMaterialCases;
             ++index) {
            const auto actual = material_probe(index, row);
            if (material_probe_rejection(index)) {
                OIIO_CHECK_EQUAL(actual.valid, 0);
                OIIO_CHECK_EQUAL(actual.extent, 0);
                for (float value : actual.values)
                    OIIO_CHECK_EQUAL(value, 0);
                continue;
            }
            OIIO_CHECK_EQUAL(actual.valid, 1);
            for (float value : actual.values)
                OIIO_CHECK_ASSERT(std::isfinite(value));
            const auto test = static_cast<MediumProbeCase>(index
                                                           - HartSurfaceCases);
            const float distance = .5f + .25f * row;
            Color3 expected(1), sigma_t(0, .5f, 1), sigma_s(0);
            const bool scatter = test == MediumProbeCase::Scatter
                                 || test == MediumProbeCase::Overlap;
            if (scatter) {
                sigma_t = test == MediumProbeCase::Scatter
                              ? (row == 0   ? Color3(0, 2, 4)
                                 : row == 1 ? Color3(2, 0, 4)
                                            : Color3(2, 4, 0))
                              : Color3(0, 1, 2);
                sigma_s = test == MediumProbeCase::Scatter
                              ? sigma_t
                              : Color3(0, .25f, .5f);
                Sampler sampler(13, int(row), 7 + int(row));
                const Vec3 random = sampler.get();
                const int first   = test == MediumProbeCase::Overlap || row == 0
                                        ? 1
                                        : 0;
                const int last    = test == MediumProbeCase::Scatter && row == 2
                                        ? 1
                                        : 2;
                const int channel = random.y < .5f ? first : last;
                const float t     = -std::log(1 - random.x) / sigma_t[channel];
                const Color3 transmittance(std::exp(-sigma_t.x * t),
                                           std::exp(-sigma_t.y * t),
                                           std::exp(-sigma_t.z * t));
                const float density = .5f
                                      * (sigma_t.x * transmittance.x
                                         + sigma_t.y * transmittance.y
                                         + sigma_t.z * transmittance.z);
                expected            = transmittance * sigma_s / density;
                OIIO_CHECK_EQUAL_THRESH(actual.values[5], 3 + t, 2e-6f);
                const Vec3 direction(actual.values[7], actual.values[8],
                                     actual.values[9]);
                OIIO_CHECK_EQUAL_THRESH(direction.length2(), 1, 2e-6f);
                OIIO_CHECK_ASSERT(actual.values[6] > 0);
                if (test == MediumProbeCase::Scatter)
                    OIIO_CHECK_EQUAL_THRESH(actual.values[6],
                                            float(M_1_PI) * .25f, 2e-6f);
            } else {
                OIIO_CHECK_EQUAL(actual.values[5], 3);
                OIIO_CHECK_EQUAL(actual.values[6], .25f);
                OIIO_CHECK_EQUAL(actual.values[7], 0);
                OIIO_CHECK_EQUAL(actual.values[8], 0);
                OIIO_CHECK_EQUAL(actual.values[9], 1);
                switch (test) {
                case MediumProbeCase::Absorption:
                    sigma_t  = Color3(0, .4f, .6f);
                    expected = Color3(1, std::exp(-.4f * distance),
                                      std::exp(-.6f * distance));
                    break;
                case MediumProbeCase::Priority:
                    expected = Color3(1, std::exp(-.5f * distance),
                                      std::exp(-distance));
                    break;
                case MediumProbeCase::Vacuum:
                case MediumProbeCase::VacuumIor: sigma_t = Color3(0); break;
                case MediumProbeCase::InfiniteAbsorption:
                    expected = Color3(1, 0, 0);
                    break;
                case MediumProbeCase::ZeroSegment:
                    sigma_s = .5f * sigma_t;
                    break;
                case MediumProbeCase::LayerTop:
                case MediumProbeCase::LayerBase: {
                    const Color3 scale = row == 0   ? Color3(0)
                                         : row == 1 ? Color3(.2f, .4f, .6f)
                                                    : Color3(1);
                    sigma_t            = scale * Color3(.2f, .4f, .6f);
                    sigma_s            = .5f * sigma_t;
                    OIIO_CHECK_EQUAL(actual.values[18], .3f);
                    OIIO_CHECK_ASSERT(actual.extent > 0);
                    break;
                }
                case MediumProbeCase::Medium:
                    sigma_t  = Color3(0, -std::log(.5f) / 2,
                                      -std::log(.25f) / 2);
                    expected = Color3(1, std::exp(-sigma_t.y * distance),
                                      std::exp(-sigma_t.z * distance));
                    break;
                case MediumProbeCase::Absorbed:
                    sigma_t  = Color3(1);
                    expected = Color3(0);
                    break;
                default: OIIO_CHECK_ASSERT(false); break;
                }
            }
            const auto event = scatter ? MediumStack::Event::Scatter
                               : test == MediumProbeCase::Absorbed
                                   ? MediumStack::Event::Absorbed
                                   : MediumStack::Event::Surface;
            OIIO_CHECK_EQUAL(actual.values[20], float(event));
            OIIO_CHECK_EQUAL(actual.values[3], 1);
            OIIO_CHECK_EQUAL(actual.values[4], 2);
            OIIO_CHECK_EQUAL(actual.values[10],
                             test == MediumProbeCase::Overlap ? 2 : 1);
            OIIO_CHECK_EQUAL(actual.values[21], actual.values[10]);
            float ior = 1, priority = 0;
            if (test == MediumProbeCase::Medium
                || test == MediumProbeCase::VacuumIor) {
                ior      = row == 1 ? 1 / 1.5f : 1.5f;
                priority = 2 + float(row);
                OIIO_CHECK_ASSERT(actual.extent > 0);
            } else if (test == MediumProbeCase::Priority) {
                ior      = 1.25f;
                priority = 1;
            }
            OIIO_CHECK_EQUAL(actual.values[11], ior);
            OIIO_CHECK_EQUAL(actual.values[19], priority);
            for (int c = 0; c < 3; ++c) {
                OIIO_CHECK_EQUAL_THRESH(actual.values[c], expected[c], 2e-6f);
                OIIO_CHECK_EQUAL_THRESH(actual.values[12 + c], sigma_t[c],
                                        2e-6f);
                OIIO_CHECK_EQUAL_THRESH(actual.values[15 + c], sigma_s[c],
                                        2e-6f);
            }
        }
}



bool
run_cases(MaterialDiagnostics& errors)
{
    HartContext context(errors);
    std::string arch;
    if (!context.init(0, arch))
        return false;
    cspan<unsigned char> bitcode;
    for (const auto& module : hart_material_modules)
        if (arch == module.arch && *module.size > 0)
            bitcode = { module.data, size_t(*module.size) };
    if (bitcode.empty()) {
        errors.errorfmt("No HART material probe for {}", arch);
        return false;
    }
    if (!context.build_accel({ }, { }, { }, 1)
        || !context.create_pipeline(bitcode, "__raygen__osl_hart_material_test",
                                    1))
        return false;
    std::array<HartMaterialResult, HartMaterialCases * HartMaterialRows> output;
    HartMaterialTestParams params { static_cast<HartMaterialResult*>(
        context.alloc(sizeof(output))) };
    if (!params.results)
        return false;
    for (unsigned index : { 1u, 2u }) {
        OIIO_CHECK_ASSERT(!context.launch(&params, sizeof(params),
                                          HartMaterialCases, HartMaterialRows,
                                          index));
        OIIO_CHECK_EQUAL(errors.last_error,
                         fmtformat("Invalid HART raygen index {}", index));
    }
    if (!context.launch(&params, sizeof(params), HartMaterialCases,
                        HartMaterialRows)
        || !context.download({ reinterpret_cast<unsigned char*>(output.data()),
                               sizeof(output) },
                             params.results))
        return false;
    for (unsigned row = 0; row < HartMaterialRows; ++row)
        for (unsigned test = 0; test < HartMaterialCases; ++test) {
            const auto& actual = output[row * HartMaterialCases + test];
            if (material_probe_rejection(test)) {
                OIIO_CHECK_EQUAL(actual.valid, 0);
                OIIO_CHECK_EQUAL(actual.extent, 0);
                for (float value : actual.values)
                    OIIO_CHECK_EQUAL(value, 0);
                continue;
            }
            const auto expected = material_probe(test, row);
            OIIO_CHECK_EQUAL(expected.valid, 1);
            OIIO_CHECK_EQUAL(actual.valid, 1);
            OIIO_CHECK_EQUAL(actual.extent, expected.extent);
            if (test == 23)
                check_mis_values(actual, row);
            if (test == 24)
                check_captured_microfacet(actual, row);
            if (test == 25 || test == 26)
                check_sampled_microfacet(actual);
            for (unsigned i = 0; i < std::size(actual.values); ++i) {
                const float a = actual.values[i], b = expected.values[i];
                // Re-evaluating a sampled direction amplifies CPU/GPU direction
                // rounding in these .001-roughness lobes (measured below 9e-5).
                const float relative = (test == 25 || test == 26) && i == 8
                                           ? 1e-4f
                                           : 2e-5f;
                if (std::isinf(b) && (i == 6 || i == 14)) {
                    OIIO_CHECK_ASSERT(a == b && a > 0);
                } else if (!std::isfinite(a) || !std::isfinite(b)
                           || std::abs(a - b)
                                  > 2e-6f + relative * std::abs(b)) {
                    errors.errorfmt("Material {} row {} field {}: GPU {} CPU {}",
                                    test, row, i, a, b);
                    OIIO_CHECK_ASSERT(false);
                }
            }
            if (test == 0) {
                for (int c = 0; c < 3; ++c)
                    OIIO_CHECK_EQUAL_THRESH(actual.values[c], .25f * (c + 1),
                                            2e-6f);
                const Vec3 wi(actual.values[16], actual.values[17],
                              actual.values[18]);
                OIIO_CHECK_EQUAL_THRESH(wi.length2(), 1, 2e-6f);
                OIIO_CHECK_EQUAL_THRESH(actual.values[14], wi.z * float(M_1_PI),
                                        2e-6f);
            }
        }
    return context.clear();
}

}  // namespace



int
main()
{
    MaterialDiagnostics errors;
    check_boundary_offsets();
    check_grazing_microfacets();
    for (unsigned row = 0; row < HartMaterialRows; ++row) {
        check_mis_values(material_probe(23, row), row);
        check_captured_microfacet(material_probe(24, row), row);
        check_sampled_microfacet(material_probe(25, row));
        check_sampled_microfacet(material_probe(26, row));
    }
    check_layer_weights();
    check_medium_values();
    OIIO_CHECK_ASSERT(run_cases(errors));
    OIIO_CHECK_EQUAL(errors.errors, 2);
    unsigned rejected = 0;
    for (unsigned test = 0; test < HartMaterialCases; ++test)
        rejected += material_probe_rejection(test);
    print("HART material probe: {} value cases and {} rejection cases, "
          "{} rows\n",
          HartMaterialCases - rejected, rejected, HartMaterialRows);
    return unit_test_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
