// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslcomp.h>

#include <OpenImageIO/unittest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "hartraytracer.h"
#include "shading.h"

using namespace OSL;

namespace {

struct Artifact {
    const void* address = nullptr;
    std::vector<unsigned char> bytes;
    void* interactive = nullptr;
};



void
check_resources(const HartContext::ResourceUsage& actual,
                const HartContext::ResourceUsage& expected = { })
{
    OIIO_CHECK_EQUAL(actual.allocations, expected.allocations);
    OIIO_CHECK_EQUAL(actual.bytes, expected.bytes);
    OIIO_CHECK_EQUAL(actual.modules, expected.modules);
    OIIO_CHECK_EQUAL(actual.program_groups, expected.program_groups);
    OIIO_CHECK_EQUAL(actual.context, expected.context);
    OIIO_CHECK_EQUAL(actual.stream, expected.stream);
    OIIO_CHECK_EQUAL(actual.pipeline, expected.pipeline);
}



bool
artifact(ShadingSystem& ss, ShaderGroup& group, Artifact& result)
{
    uint64_t size = 0;
    if (!ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &result.address)
        || !ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size)
        || !ss.getattribute(&group, "device_interactive_params", TypeDesc::PTR,
                            &result.interactive)
        || !result.address || !size || size > std::numeric_limits<size_t>::max()
        || !result.interactive) {
        print(stderr, "Native lifecycle material has no compiled artifact "
                      "or interactive binding\n");
        return false;
    }
    const auto* begin = static_cast<const unsigned char*>(result.address);
    result.bytes.assign(begin, begin + size_t(size));
    return true;
}



class NativeFixture {
public:
    ~NativeFixture() { clear(); }

    bool open(string_view bytecode, bool fused, size_t budget, int optimize)
    {
        if (!renderer.initialize(0, fused, budget))
            return false;
        ss = std::make_unique<ShadingSystem>(&renderer, nullptr,
                                             &renderer.errhandler());
        renderer.shadingsys = ss.get();
        register_closures(ss.get());
        if (!ss->attribute("optimize", optimize)
            || !ss->attribute("llvm_optimize", optimize ? 3 : 10)
            || !ss->attribute("error_repeats", 1)
            || !ss->LoadMemoryCompiledShader("lifecycle_surface", bytecode))
            return false;
        renderer.attribute("aa", 1);
        renderer.attribute("no_jitter", 1);
        renderer.attribute("max_bounces", 0);
        renderer.camera.resolution(8, 4);
        // Deliberately reuse shader, group and layer names in every renderer.
        // The oversized panels cover every camera sample with no edge pixels.
        renderer.parse_scene_xml(
            "<World><Camera eye=\"0,0,4\" dir=\"0,0,-1\" fov=\"90\"/>"
            "<ShaderGroup>color tint 0.125 0.25 0.5 [[int interactive=1]]; "
            "shader lifecycle_surface m;</ShaderGroup>"
            "<Quad corner=\"-10,-10,0\" edge_x=\"10,0,0\" edge_y=\"0,20,0\"/>"
            "<ShaderGroup>color tint 0.5 0.25 0.125 [[int interactive=1]]; "
            "shader lifecycle_surface m;</ShaderGroup>"
            "<Quad corner=\"0,-10,0\" edge_x=\"10,0,0\" edge_y=\"0,20,0\"/>"
            "</World>");
        OIIO_CHECK_EQUAL(renderer.shaders().size(), size_t(2));
        return !renderer.failed() && !renderer.had_error()
               && renderer.shaders().size() == 2;
    }

    bool prepare()
    {
        renderer.prepare_render();
        if (renderer.failed() || renderer.had_error())
            return false;
        const auto resources = renderer.resource_usage();
        OIIO_CHECK_ASSERT(resources.allocations > 0 && resources.bytes > 0);
        OIIO_CHECK_ASSERT(resources.context && resources.stream
                          && resources.pipeline);
        for (size_t i = 0; i < artifacts.size(); ++i)
            if (!artifact(*ss, *renderer.shaders()[i].surf, artifacts[i]))
                return false;
        OIIO_CHECK_ASSERT(artifacts[0].interactive != artifacts[1].interactive);
        return true;
    }

    bool render(int width, int height, const Color3& left, const Color3& right)
    {
        const int failures = unit_test_failures;
        renderer.render(width, height);
        renderer.finalize_pixel_buffer();
        if (renderer.failed() || renderer.had_error())
            return false;
        OIIO_CHECK_ASSERT(renderer.pixelbuf.initialized());
        OIIO_CHECK_EQUAL(renderer.pixelbuf.spec().width, width);
        OIIO_CHECK_EQUAL(renderer.pixelbuf.spec().height, height);
        OIIO_CHECK_EQUAL(renderer.pixelbuf.spec().nchannels, 3);
        std::vector<float> pixels(size_t(width) * height * 3);
        if (!renderer.pixelbuf.get_pixels(OIIO::ROI(0, width, 0, height, 0, 1,
                                                    0, 3),
                                          TypeDesc::FLOAT, pixels.data())) {
            print(stderr, "Cannot read lifecycle pixels: {}\n",
                  renderer.pixelbuf.geterror());
            return false;
        }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c) {
                    const float value = pixels[(y * width + x) * 3 + c];
                    OIIO_CHECK_ASSERT(std::isfinite(value));
                    OIIO_CHECK_EQUAL(value, (x < width / 2 ? left : right)[c]);
                }
        for (size_t i = 0; i < artifacts.size(); ++i) {
            Artifact current;
            if (!artifact(*ss, *renderer.shaders()[i].surf, current))
                return false;
            OIIO_CHECK_EQUAL(current.address, artifacts[i].address);
            OIIO_CHECK_EQUAL(current.interactive, artifacts[i].interactive);
            OIIO_CHECK_ASSERT(current.bytes == artifacts[i].bytes);
        }
        return unit_test_failures == failures;
    }

    bool rebind(size_t material, const Color3& tint)
    {
        return ss->ReParameter(*renderer.shaders()[material].surf, "m", "tint",
                               TypeColor, &tint);
    }

    void clear()
    {
        // Match testrender: drop groups while both renderer and shading
        // system live, then destroy the shading system before its services.
        const bool failed    = renderer.failed();
        const bool had_error = renderer.had_error();
        renderer.clear();
        OIIO_CHECK_ASSERT(renderer.shaders().empty());
        OIIO_CHECK_ASSERT(!renderer.pixelbuf.initialized());
        check_resources(renderer.resource_usage());
        ss.reset();
        renderer.shadingsys = nullptr;
        OIIO_CHECK_EQUAL(renderer.failed(), failed);
        OIIO_CHECK_EQUAL(renderer.had_error(), had_error);
    }

    HartRaytracer renderer;
    std::unique_ptr<ShadingSystem> ss;
    std::array<Artifact, 2> artifacts;
};



bool
check_native_lifecycle(string_view stdosl, bool fused, size_t budget,
                       int optimize)
{
    std::array<std::string, 2> bytecode;
    for (size_t i = 0; i < bytecode.size(); ++i) {
        OSLCompiler compiler;
        const auto source = fmtformat(
            "shader lifecycle_surface(color tint=1 [[int interactive=1]], "
            "output color Cout=0) {{ Cout=tint; Ci={}*tint*emission(); }}",
            i + 1);
        if (!compiler.compile_buffer(source, bytecode[i], { }, stdosl))
            return false;
    }
    const Color3 left(.125f, .25f, .5f), right(.5f, .25f, .125f);
    const Color3 changed(.25f, .5f, .75f);
    NativeFixture resident;
    if (!resident.open(bytecode[1], fused, budget, optimize)
        || !resident.prepare()
        || !resident.render(8, 4, 2.0f * left, 2.0f * right))
        return false;
    for (int cycle = 0; cycle < 3; ++cycle) {
        NativeFixture current;
        const int variant = cycle % 2;
        const float scale = float(variant + 1);
        if (!current.open(bytecode[variant], fused, budget, optimize)
            || !current.prepare())
            return false;
        OIIO_CHECK_ASSERT(current.artifacts[0].interactive
                          != resident.artifacts[0].interactive);
        // Grow once, reuse, then shrink; every pixel must be freshly written.
        HartContext::ResourceUsage grown;
        const auto resident_usage = resident.renderer.resource_usage();
        for (int width : { 8, 12, 12, 4, 8 }) {
            if (!current.render(width, 4, scale * left, scale * right)
                || !resident.render(8, 4, 2.0f * left, 2.0f * right))
                return false;
            check_resources(resident.renderer.resource_usage(), resident_usage);
            if (grown.allocations)
                check_resources(current.renderer.resource_usage(), grown);
            else if (width == 12)
                grown = current.renderer.resource_usage();
        }
        if (!current.rebind(0, changed)
            || !current.render(8, 4, scale * changed, scale * right)
            || !resident.render(8, 4, 2.0f * left, 2.0f * right)
            || !current.rebind(0, left)
            || !current.render(8, 4, scale * left, scale * right))
            return false;
        if (cycle == 1) {
            // A failed preparation must not leave the last successful image
            // visible, even before render/finalize is called again.
            current.renderer.prepare_render();
            OIIO_CHECK_ASSERT(current.renderer.failed());
            OIIO_CHECK_ASSERT(current.renderer.had_error());
            OIIO_CHECK_ASSERT(!current.renderer.pixelbuf.initialized());
        }
        current.clear();
        current.clear();
        OIIO_CHECK_EQUAL(current.renderer.failed(), cycle == 1);
        OIIO_CHECK_EQUAL(current.renderer.had_error(), cycle == 1);
        if (!resident.render(8, 4, 2.0f * left, 2.0f * right))
            return false;
    }
    {
        NativeFixture partial;
        if (!partial.open(bytecode[0], fused, budget, optimize))
            return false;
        // The first group owns compiled code and an interactive allocation
        // before the second group's unsupported output layout is rejected.
        if (!partial.ss->attribute(partial.renderer.shaders()[1].surf.get(),
                                   "renderer_outputs", "Cout"))
            return false;
        partial.renderer.prepare_render();
        OIIO_CHECK_ASSERT(partial.renderer.failed());
        OIIO_CHECK_ASSERT(partial.renderer.had_error());
        OIIO_CHECK_ASSERT(!partial.renderer.pixelbuf.initialized());
        int optimized = 0;
        OIIO_CHECK_ASSERT(
            partial.ss->getattribute(partial.renderer.shaders()[0].surf.get(),
                                     "is_optimized", optimized));
        OIIO_CHECK_EQUAL(optimized, 1);
        optimized = -1;
        OIIO_CHECK_ASSERT(
            partial.ss->getattribute(partial.renderer.shaders()[1].surf.get(),
                                     "is_optimized", optimized));
        OIIO_CHECK_EQUAL(optimized, 0);
        Artifact first;
        if (!artifact(*partial.ss, *partial.renderer.shaders()[0].surf, first))
            return false;
        partial.renderer.render(8, 4);
        partial.renderer.finalize_pixel_buffer();
        OIIO_CHECK_ASSERT(!partial.renderer.pixelbuf.initialized());
        partial.clear();
        partial.clear();
    }
    // Renderer errors are terminal; a fresh renderer is the supported
    // same-process recovery path, with no shader-name or binding carryover.
    {
        NativeFixture recovered;
        if (!recovered.open(bytecode[0], fused, budget, optimize)
            || !recovered.prepare() || !recovered.render(8, 4, left, right))
            return false;
    }
    if (!resident.render(8, 4, 2.0f * left, 2.0f * right))
        return false;
    resident.clear();
    OIIO_CHECK_ASSERT(!resident.renderer.failed());
    OIIO_CHECK_ASSERT(!resident.renderer.had_error());
    return true;
}

}  // namespace



int
main(int argc, const char* argv[])
{
    if (argc != 3) {
        print(stderr, "Usage: hart_lifecycle_test stdosl "
                      "{split|fused|fused-local|unoptimized}\n");
        return EXIT_FAILURE;
    }
    const string_view mode(argv[2]);
    if (mode != "split" && mode != "fused" && mode != "fused-local"
        && mode != "unoptimized") {
        print(stderr, "Unknown native lifecycle mode '{}'\n", mode);
        return EXIT_FAILURE;
    }
    OIIO_CHECK_ASSERT(check_native_lifecycle(
        argv[1], mode == "fused" || mode == "fused-local",
        mode == "fused-local" ? 4096 : 0, mode == "unoptimized" ? 0 : 2));
    if (!unit_test_failures)
        print("HART native renderer lifecycle {}: isolated materials, "
              "A-B-A bindings, resize/reuse and failed-prepare teardown\n",
              mode);
    return unit_test_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
