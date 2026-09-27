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
        if ((code & 0xffff0000) == EH_ERROR || (code & 0xffff0000) == EH_SEVERE)
            ++errors;
        ErrorHandler::operator()(code, message);
    }

    int errors = 0;
};



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

        // Host-only medium invariance does not qualify HART volume rendering.
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
            process_medium_closure(sg, .2f, shading, medium, medium_layer,
                                   Color3(1));
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
run_cases(ErrorHandler& errors)
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
    if (!params.results
        || !context.launch(&params, sizeof(params), HartMaterialCases,
                           HartMaterialRows)
        || !context.download({ reinterpret_cast<unsigned char*>(output.data()),
                               sizeof(output) },
                             params.results))
        return false;
    for (unsigned row = 0; row < HartMaterialRows; ++row)
        for (unsigned test = 0; test < HartMaterialCases; ++test) {
            const auto& actual = output[row * HartMaterialCases + test];
            if (test >= 14) {
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
            for (unsigned i = 0; i < std::size(actual.values); ++i) {
                const float a = actual.values[i], b = expected.values[i];
                if (std::isinf(b) && (i == 6 || i == 14)) {
                    OIIO_CHECK_ASSERT(a == b && a > 0);
                } else if (!std::isfinite(a) || !std::isfinite(b)
                           || std::abs(a - b) > 2e-6f + 2e-5f * std::abs(b)) {
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
    check_layer_weights();
    OIIO_CHECK_ASSERT(run_cases(errors));
    OIIO_CHECK_EQUAL(errors.errors, 0);
    print(
        "HART material probe: 14 value cases and 7 rejection cases, 3 rows\n");
    return unit_test_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
