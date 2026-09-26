// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslclosure.h>
#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>

#include <algorithm>
#include <array>
#include <cmath>
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
shader inspect(closure color lobes=0) {
    int row = int(6*v + 0.5);
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
    } else
        Ci = lobes;
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
        OIIO_CHECK_ASSERT(closure->id == 1 || closure->id == 3);
        const auto* component = closure->as_comp();
        const Color3 combined = weight * component->w;
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



bool
compare_cpu(string_view stdosl, string_view texture,
            cspan<HartClosureSummary> gpu, int width, int height,
            Diagnostics& errors)
{
    SimpleRenderer renderer;
    ShadingSystem ss(&renderer, nullptr, &errors);
    renderer.init_shadingsys(&ss);
    register_closures(&ss);
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
    OIIO_CHECK_ASSERT(result.used <= testshade::HartClosureCapacity);
    if (diffuse.x > 0 || emission.x > 0)
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
    constexpr int width = 5, height = 7;
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
    if (!compare_cpu(stdosl, texture.filename, summaries, width, height, errors))
        return false;
    print(
        "HART closure inspection verified: {} points, mode {}, exact pool {} bytes\n",
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
