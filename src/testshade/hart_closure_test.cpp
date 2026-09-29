// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/unittest.h>

#include "hartgridrender.h"
#include "hartrenderstate.h"
#include "simplerend.h"

using namespace OSL;
using testshade::HartClosureSummary;
using testshade::HartDiffuseRampParams;
using testshade::HartPhongRampParams;

namespace {

class Diagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR || (code & 0xffff0000) == EH_SEVERE)
            ++errors;
        ErrorHandler::operator()(code, message);
    }
    int errors = 0;
};



struct TextureFixture {
    std::string filename = OIIO::Filesystem::unique_path(
        "hart-closure-%%%%%%%%-%%%%-%%%%.exr");
    bool written = false;

    bool create()
    {
        OIIO::ImageBuf image(OIIO::ImageSpec(2, 2, 3, TypeDesc::FLOAT));
        const float value[] = { 0.2f, 0.3f, 0.4f };
        written             = OIIO::ImageBufAlgo::fill(image, value)
                              && image.write(filename);
        if (!written)
            print(stderr, "Cannot write closure texture: {}\n",
                  image.geterror());
        return written;
    }

    ~TextureFixture()
    {
        if (written)
            OIIO_CHECK_ASSERT(OIIO::Filesystem::remove(filename));
    }
};



void
register_ramp_closures(ShadingSystem& ss)
{
    const ClosureParam diffuse[]
        = { CLOSURE_VECTOR_PARAM(HartDiffuseRampParams, N),
            CLOSURE_COLOR_ARRAY_PARAM(HartDiffuseRampParams, colors, 8),
            CLOSURE_FINISH_PARAM(HartDiffuseRampParams) };
    const ClosureParam phong[]
        = { CLOSURE_VECTOR_PARAM(HartPhongRampParams, N),
            CLOSURE_FLOAT_PARAM(HartPhongRampParams, exponent),
            CLOSURE_COLOR_ARRAY_PARAM(HartPhongRampParams, colors, 8),
            { TypeDesc(TypeDesc::FLOAT, 4),
              int(offsetof(HartPhongRampParams, knots)), "knots",
              sizeof(HartPhongRampParams::knots) },
            CLOSURE_FLOAT_KEYPARAM(HartPhongRampParams, marker, "marker"),
            CLOSURE_FINISH_PARAM(HartPhongRampParams) };
    ss.register_closure("diffuse_ramp", testshade::HartDiffuseRampId, diffuse,
                        nullptr, nullptr);
    ss.register_closure("phong_ramp", testshade::HartPhongRampId, phong,
                        nullptr, nullptr);
}



ShaderGroupRef
make_group(ShadingSystem& ss, string_view stdosl, string_view texture,
           Diagnostics& errors)
{
    const std::array<std::string, 3> sources { fmtformat(R"OSL(
shader weights(output color weight=0) {{
    weight = texture("{}", u, v, "interp", "closest", "wrap", "clamp")
             + color(u, 0.1, 0.2);
}})OSL",
                                                         texture),
                                               R"OSL(
shader material(color weight=0, output closure color lobes=0) {
    lobes = weight * diffuse(normal(u, 1, 2))
            + color(0.04, 0.05, 0.06) * emission();
})OSL",
                                               R"OSL(
closure color diffuse_ramp(normal N, color colors[8]) [[int builtin=1]];
closure color phong_ramp(normal N, float exponent, color colors[8])
    [[int builtin=1]];
shader inspect(closure color lobes=0) {
    int row = int(8*v + 0.5);
    if (row == 0)
        Ci = color(0.8, 0.3, 0.1)*diffuse(N) + color(0.02)*emission();
    else if (row == 1)
        Ci = 0;
    else if (row == 2)
        Ci = (u + 0.5)*diffuse(normal(1,2,3));
    else if (row == 3)
        Ci = color(0)*diffuse(N) + 0*emission();
    else if (row == 4) {
        closure color tree = 2*diffuse(N) + 0.25*emission();
        Ci = color(u + 0.2, 0.4, 0.6)*tree;
    } else if (row == 5) {
        if (u < 0.5) Ci = diffuse(N);
        else Ci = emission();
    } else if (row == 6)
        Ci = lobes;
    else if (row == 7) {
        color colors[8] = {color(1,2,3), color(4,5,6), color(7,8,9),
                           color(10,11,12), color(13,14,15), color(16,17,18),
                           color(19,20,21), color(22,23,24)};
        float knots[4] = {0.125, 0.375, 0.625, 0.875};
        Ci = diffuse_ramp(normal(3,2,1), colors)
             + phong_ramp(normal(-1,4,2), 7.5, colors,
                          "knots", knots, "marker", 19.25);
    } else {
        color colors[8];
        for (int i = 0; i < 8; ++i)
            colors[i] = color(i+1+u, 2*i+2+v, -i-3+u*v);
        float knots[4];
        for (int i = 0; i < 4; ++i)
            knots[i] = u + 0.25*i;
        Ci = color(u+0.25,0.5,1.5)
                 * diffuse_ramp(normal(u,2+u,-3), colors)
             + color(0.75,1+0.5*u,0.25+0.25*u)
                 * phong_ramp(normal(1+u,-2,3*u), 2+4*u, colors,
                              "knots", knots, "marker", 31+2*u);
    }
})OSL" };
    const char* names[] = { "weights", "material", "inspect" };
    for (size_t i = 0; i < sources.size(); ++i) {
        OSLCompiler compiler(&errors);
        std::string oso;
        if (!compiler.compile_buffer(sources[i], oso, { }, stdosl)
            || !ss.LoadMemoryCompiledShader(names[i], oso))
            return { };
    }
    auto group = ss.ShaderGroupBegin("hart_closure_test");
    if (!group)
        return { };
    bool ok = true;
    for (const char* name : names)
        ok = ss.Shader("surface", name, name) && ok;
    ok = ss.ConnectShaders("weights", "weight", "material", "weight") && ok;
    ok = ss.ConnectShaders("material", "lobes", "inspect", "lobes") && ok;
    ok = ss.ShaderGroupEnd() && ok;
    return ok ? group : ShaderGroupRef();
}



void
summarize_cpu(const ClosureColor* closure, const Color3& weight,
              HartClosureSummary& result)
{
    if (!closure)
        return;
    if (closure->id == ClosureColor::ADD) {
        summarize_cpu(closure->as_add()->closureA, weight, result);
        summarize_cpu(closure->as_add()->closureB, weight, result);
    } else if (closure->id == ClosureColor::MUL) {
        summarize_cpu(closure->as_mul()->closure,
                      weight * closure->as_mul()->weight, result);
    } else {
        OIIO_CHECK_ASSERT(closure->id == 1 || closure->id == 3
                          || closure->id == testshade::HartDiffuseRampId
                          || closure->id == testshade::HartPhongRampId);
        const auto* component = closure->as_comp();
        const Color3 combined = weight * component->w;
        if (component->id == testshade::HartDiffuseRampId) {
            result.diffuse_ramp = *component->as<HartDiffuseRampParams>();
            for (int c = 0; c < 3; ++c)
                result.diffuse_ramp_weight[c] = combined[c];
            OIIO_CHECK_EQUAL(++result.diffuse_ramp_count, 1u);
            return;
        }
        if (component->id == testshade::HartPhongRampId) {
            result.phong_ramp = *component->as<HartPhongRampParams>();
            for (int c = 0; c < 3; ++c)
                result.phong_ramp_weight[c] = combined[c];
            OIIO_CHECK_EQUAL(++result.phong_ramp_count, 1u);
            return;
        }
        for (int c = 0; c < 3; ++c) {
            if (component->id == 3) {
                result.diffuse[c] += combined[c];
                result.normal[c] += (*component->as<Vec3>())[c] * combined.x;
            } else {
                result.emission[c] += combined[c];
            }
        }
        if (component->id == 3)
            ++result.diffuse_count;
        else
            ++result.emission_count;
    }
}



void
compare_ramps(const HartClosureSummary& actual,
              const HartClosureSummary& expected)
{
    OIIO_CHECK_EQUAL(actual.diffuse_ramp_count, expected.diffuse_ramp_count);
    OIIO_CHECK_EQUAL(actual.phong_ramp_count, expected.phong_ramp_count);
    for (int c = 0; c < 3; ++c) {
        OIIO_CHECK_EQUAL_THRESH(actual.diffuse_ramp_weight[c],
                                expected.diffuse_ramp_weight[c], 2.0e-6f);
        OIIO_CHECK_EQUAL_THRESH(actual.phong_ramp_weight[c],
                                expected.phong_ramp_weight[c], 2.0e-6f);
        OIIO_CHECK_EQUAL_THRESH(actual.diffuse_ramp.N[c],
                                expected.diffuse_ramp.N[c], 2.0e-6f);
        OIIO_CHECK_EQUAL_THRESH(actual.phong_ramp.N[c],
                                expected.phong_ramp.N[c], 2.0e-6f);
        for (int i = 0; i < 8; ++i) {
            OIIO_CHECK_EQUAL_THRESH(actual.diffuse_ramp.colors[i][c],
                                    expected.diffuse_ramp.colors[i][c],
                                    2.0e-6f);
            OIIO_CHECK_EQUAL_THRESH(actual.phong_ramp.colors[i][c],
                                    expected.phong_ramp.colors[i][c], 2.0e-6f);
        }
    }
    OIIO_CHECK_EQUAL_THRESH(actual.phong_ramp.exponent,
                            expected.phong_ramp.exponent, 2.0e-6f);
    for (int i = 0; i < 4; ++i)
        OIIO_CHECK_EQUAL_THRESH(actual.phong_ramp.knots[i],
                                expected.phong_ramp.knots[i], 2.0e-6f);
    OIIO_CHECK_EQUAL_THRESH(actual.phong_ramp.marker,
                            expected.phong_ramp.marker, 2.0e-6f);
}



bool
compare_cpu(string_view stdosl, string_view texture,
            cspan<HartClosureSummary> gpu, int width, int height,
            string_view mode, Diagnostics& errors)
{
    SimpleRenderer renderer;
    ShadingSystem ss(&renderer, nullptr, &errors);
    renderer.init_shadingsys(&ss);
    register_closures(&ss);
    register_ramp_closures(ss);
    if (!ss.attribute("optimize", mode == "unoptimized" ? 0 : 2)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3))
        return false;
    auto group = make_group(ss, stdosl, texture, errors);
    if (!group)
        return false;
    auto* thread  = ss.create_thread_info();
    auto* context = ss.get_context(thread);
    bool ok       = true;
    for (int y = 0; y < height && ok; ++y) {
        for (int x = 0; x < width && ok; ++x) {
            ShaderGlobals sg { };
            sg.u    = float(x) / (width - 1);
            sg.v    = float(y) / (height - 1);
            sg.dudx = 1.0f / (width - 1);
            sg.dvdy = 1.0f / (height - 1);
            sg.N = sg.Ng = Vec3(0, 0, 1);
            ok           = ss.execute(*context, *group, sg);
            if (!ok)
                break;
            HartClosureSummary cpu { };
            summarize_cpu(sg.Ci, Color3(1), cpu);
            const auto& actual = gpu[y * width + x];
            for (int c = 0; c < 3; ++c) {
                OIIO_CHECK_EQUAL_THRESH(actual.diffuse[c], cpu.diffuse[c],
                                        2.0e-6f);
                OIIO_CHECK_EQUAL_THRESH(actual.emission[c], cpu.emission[c],
                                        2.0e-6f);
                OIIO_CHECK_EQUAL_THRESH(actual.normal[c], cpu.normal[c],
                                        2.0e-6f);
            }
            OIIO_CHECK_EQUAL(actual.diffuse_count, cpu.diffuse_count);
            OIIO_CHECK_EQUAL(actual.emission_count, cpu.emission_count);
            compare_ramps(actual, cpu);
        }
    }
    ss.release_context(context);
    ss.destroy_thread_info(thread);
    ss.texturesys()->invalidate(ustring(texture), true);
    return ok && errors.errors == 0;
}



void
check_summary(const HartClosureSummary& result, int x, int row, int width)
{
    const float u = width == 1 ? 0.5f : float(x) / (width - 1);
    Color3 diffuse(0), emission(0);
    Vec3 normal(0, 0, 1);
    if (row == 0) {
        diffuse  = Color3(0.8f, 0.3f, 0.1f);
        emission = Color3(0.02f);
    } else if (row == 2) {
        diffuse = Color3(u + 0.5f);
        normal  = Vec3(1, 2, 3);
    } else if (row == 4) {
        diffuse  = 2.0f * Color3(u + 0.2f, 0.4f, 0.6f);
        emission = 0.25f * Color3(u + 0.2f, 0.4f, 0.6f);
    } else if (row == 5) {
        if (u < 0.5f)
            diffuse = Color3(1);
        else
            emission = Color3(1);
    } else if (row == 6) {
        diffuse  = Color3(u + 0.2f, 0.4f, 0.6f);
        emission = Color3(0.04f, 0.05f, 0.06f);
        normal   = Vec3(u, 1, 2);
    }
    for (int c = 0; c < 3; ++c) {
        OIIO_CHECK_EQUAL_THRESH(result.diffuse[c], diffuse[c], 2.0e-6f);
        OIIO_CHECK_EQUAL_THRESH(result.emission[c], emission[c], 2.0e-6f);
        OIIO_CHECK_EQUAL_THRESH(result.normal[c], normal[c] * diffuse.x,
                                2.0e-6f);
    }
    OIIO_CHECK_EQUAL(result.diffuse_count, diffuse.x > 0 ? 1u : 0u);
    OIIO_CHECK_EQUAL(result.emission_count, emission.x > 0 ? 1u : 0u);
    HartClosureSummary ramps { };
    if (row == 7 || row == 8) {
        ramps.diffuse_ramp_count = ramps.phong_ramp_count = 1;
        const Vec3 diffuse_N = row == 7 ? Vec3(3, 2, 1) : Vec3(u, 2 + u, -3);
        const Vec3 phong_N = row == 7 ? Vec3(-1, 4, 2) : Vec3(1 + u, -2, 3 * u);
        const Color3 diffuse_weight = row == 7 ? Color3(1)
                                               : Color3(u + 0.25f, 0.5f, 1.5f);
        const Color3 phong_weight   = row == 7 ? Color3(1)
                                               : Color3(0.75f, 1 + 0.5f * u,
                                                        0.25f + 0.25f * u);
        for (int c = 0; c < 3; ++c) {
            ramps.diffuse_ramp.N[c]      = diffuse_N[c];
            ramps.phong_ramp.N[c]        = phong_N[c];
            ramps.diffuse_ramp_weight[c] = diffuse_weight[c];
            ramps.phong_ramp_weight[c]   = phong_weight[c];
        }
        for (int i = 0; i < 8; ++i) {
            const Color3 value = row == 7
                                     ? Color3(3 * i + 1, 3 * i + 2, 3 * i + 3)
                                     : Color3(i + 1 + u, 2 * i + 3, -i - 3 + u);
            for (int c = 0; c < 3; ++c) {
                ramps.diffuse_ramp.colors[i][c] = value[c];
                ramps.phong_ramp.colors[i][c]   = value[c];
            }
        }
        ramps.phong_ramp.exponent = row == 7 ? 7.5f : 2 + 4 * u;
        for (int i = 0; i < 4; ++i)
            ramps.phong_ramp.knots[i] = (row == 7 ? 0.125f : u) + 0.25f * i;
        ramps.phong_ramp.marker = row == 7 ? 19.25f : 31 + 2 * u;
    }
    compare_ramps(result, ramps);
    OIIO_CHECK_ASSERT(result.used <= testshade::HartClosureCapacity);
    if (diffuse.x > 0 || emission.x > 0 || ramps.diffuse_ramp_count)
        OIIO_CHECK_ASSERT(result.used > 0);
}



bool
run(string_view stdosl, string_view mode, bool exhaust, Diagnostics& errors)
{
    TextureFixture texture;
    if (!texture.create())
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch, true);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &errors);
    renderer->init_shadingsys(&ss);
    register_closures(&ss);
    register_ramp_closures(ss);
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? std::numeric_limits<int>::max()
                                               : 0)
        || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3))
        return false;
    auto group = make_group(ss, stdosl, texture.filename, errors);
    if (!group)
        return false;
    HartOptions options;
    options.fused       = mode == "fused" || mode == "fused-local";
    constexpr int width = 5, height = 9;
    std::vector<HartClosureSummary> summaries(width * height);
    if (exhaust) {
        // Failure must occur after launch, not from the shader validator.
        return testshade_hart_closure_test(*renderer, ss, group.get(), options,
                                           arch, width, height, summaries, 1);
    }
    if (!testshade_hart_closure_test(*renderer, ss, nullptr, options, arch,
                                     width, height, summaries))
        return false;
    for (const auto& result : summaries) {
        for (int c = 0; c < 3; ++c) {
            OIIO_CHECK_EQUAL(result.diffuse[c], 1);
            OIIO_CHECK_EQUAL(result.emission[c], 1);
            OIIO_CHECK_EQUAL(result.normal[c], 1);
        }
        OIIO_CHECK_EQUAL(result.diffuse_count, 1);
        OIIO_CHECK_EQUAL(result.emission_count, 1);
        OIIO_CHECK_EQUAL(result.used, 16);
        compare_ramps(result, HartClosureSummary { });
    }
    if (!testshade_hart_closure_test(*renderer, ss, group.get(), options, arch,
                                     width, height, summaries))
        return false;
    uint32_t used = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const auto& result = summaries[y * width + x];
            check_summary(result, x, y, width);
            used = std::max(used, result.used);
        }
    int local = -1;
    OIIO_CHECK_ASSERT(
        ss.getattribute(group.get(), "hart_groupdata_alloc", local));
    OIIO_CHECK_EQUAL(local > 0, mode == "fused-local");
    OIIO_CHECK_ASSERT(used > 0);
    const auto original = summaries;
    if (!testshade_hart_closure_test(*renderer, ss, group.get(), options, arch,
                                     width, height, summaries, used))
        return false;
    for (size_t i = 0; i < summaries.size(); ++i) {
        OIIO_CHECK_EQUAL(summaries[i].used, original[i].used);
        check_summary(summaries[i], int(i % width), int(i / width), width);
    }
    if (!compare_cpu(stdosl, texture.filename, summaries, width, height, mode,
                     errors))
        return false;
    print(
        "HART closure inspection verified: {} points, mode {}, exact pool {} bytes, all 24 ramp components\n",
        summaries.size(), mode, used);
    return errors.errors == 0;
}

}  // namespace



int
main(int argc, char* argv[])
{
    const string_view mode(argc >= 3 ? argv[2] : "split");
    if (argc < 2 || argc > 4
        || (mode != "split" && mode != "fused" && mode != "fused-local"
            && mode != "unoptimized")
        || (argc == 4 && string_view(argv[3]) != "exhaust")) {
        print(
            stderr,
            "Usage: hart_closure_test stdosl.h [split|fused|fused-local|unoptimized] [exhaust]\n");
        return 1;
    }
    Diagnostics errors;
    OIIO_CHECK_ASSERT(run(argv[1], mode, argc == 4, errors));
    return unit_test_failures;
}
