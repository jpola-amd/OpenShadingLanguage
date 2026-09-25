// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/unittest.h>

#include "hartgridrender.h"
#include "simplerend.h"

using namespace OSL;

namespace {

constexpr int width = 5, height = 3;

const char* source = R"OSL(
shader hart_transform_rebind(output color Cout = 0)
{
    int column = int(4*u + 0.5);
    vector result = 0;
    if (v < 0.25) {
        point q = point(u + 0.25, 2*v - 0.5, u + v + 1);
        if (column == 0) q = transform("object", "common", q);
        else if (column == 1) q = transform("common", "object", q);
        else if (column == 2) q = transform("shader", "common", q);
        else if (column == 3) q = transform("common", "shader", q);
        else q = transform("object", "shader", q);
        result = vector(q);
    } else if (v < 0.75) {
        vector q = vector(u + 0.25, 2*v - 0.5, u + v + 1);
        if (column == 0) q = transform("object", "common", q);
        else if (column == 1) q = transform("common", "object", q);
        else if (column == 2) q = transform("shader", "common", q);
        else if (column == 3) q = transform("common", "shader", q);
        else q = transform("object", "shader", q);
        result = q;
    } else {
        normal q = normal(u + 0.25, 2*v - 0.5, u + v + 1);
        if (column == 0) q = transform("object", "common", q);
        else if (column == 1) q = transform("common", "object", q);
        else if (column == 2) q = transform("shader", "common", q);
        else if (column == 3) q = transform("common", "shader", q);
        else q = transform("object", "shader", q);
        result = vector(q);
    }
    Cout = color(dot(result, vector(1,2,3)),
                 dot(Dx(result), vector(3,1,2)),
                 dot(Dy(result), vector(2,3,1)));
}
)OSL";



class Diagnostics final : public ErrorHandler {
public:
    Diagnostics() { verbosity(VERBOSE); }

    void operator()(int code, const std::string& message) override
    {
        const int category = code & 0xffff0000;
        if (category == EH_WARNING)
            ++warnings;
        if (category == EH_ERROR || category == EH_SEVERE)
            ++errors;
        ErrorHandler::operator()(code, message);
    }

    int warnings = 0, errors = 0;
};



struct OutputFiles {
    bool create(Diagnostics& diagnostics)
    {
        const auto candidate = OIIO::Filesystem::unique_path(
            "hart-transform-%%%%%%%%-%%%%-%%%%-%%%%-%%%%%%%%%%%%");
        std::error_code error;
        // Do not adopt or remove a directory created by another process.
        if (!std::filesystem::create_directory(candidate, error)) {
            diagnostics.errorfmt(
                "Cannot create private output directory '{}': {}", candidate,
                error ? error.message() : "already exists");
            return false;
        }
        directory = candidate;
        // EXR avoids PFM's bottom-to-top scanline convention on readback.
        for (size_t i = 0; i < files.size(); ++i)
            files[i] = (std::filesystem::path(directory)
                        / fmtformat("pass{}.exr", i))
                           .string();
        return true;
    }

    ~OutputFiles()
    {
        if (directory.empty())
            return;
        for (const auto& file : files) {
            std::error_code error;
            std::filesystem::remove(file, error);
            if (error)
                print(stderr, "Cannot remove '{}': {}\n", file,
                      error.message());
            OIIO_CHECK_ASSERT(!error);
        }
        std::error_code error;
        std::filesystem::remove(directory, error);
        if (error)
            print(stderr, "Cannot remove '{}': {}\n", directory,
                  error.message());
        OIIO_CHECK_ASSERT(!error);
    }

    std::string directory;
    std::array<std::string, 3> files;
};



using Triple = std::array<double, 3>;

// Upper-triangular linear part plus row-vector translation. The oracle solves
// these transforms directly, independently of OSL/Imath matrix implementations.
struct Affine {
    double xx, xy, xz, yy, yz, zz;
    Triple translation;

    Matrix44 matrix() const
    {
        return Matrix44(float(xx), float(xy), float(xz), 0, 0, float(yy),
                        float(yz), 0, 0, 0, float(zz), 0, float(translation[0]),
                        float(translation[1]), float(translation[2]), 1);
    }

    Triple apply(Triple q, int kind, bool inverse) const
    {
        if (kind == 2) {
            if (inverse)
                return { xx * q[0] + xy * q[1] + xz * q[2],
                         yy * q[1] + yz * q[2], zz * q[2] };
            // Normals use the inverse transpose, without normalization.
            const double z = q[2] / zz;
            const double y = (q[1] - yz * z) / yy;
            return { (q[0] - xy * y - xz * z) / xx, y, z };
        }
        if (inverse) {
            if (kind == 0)
                for (int c = 0; c < 3; ++c)
                    q[c] -= translation[c];
            const double x = q[0] / xx;
            const double y = (q[1] - xy * x) / yy;
            return { x, y, (q[2] - xz * x - yz * y) / zz };
        }
        Triple result { xx * q[0], xy * q[0] + yy * q[1],
                        xz * q[0] + yz * q[1] + zz * q[2] };
        if (kind == 0)
            for (int c = 0; c < 3; ++c)
                result[c] += translation[c];
        return result;
    }
};



Triple
transform(const Affine& object, const Affine& shader, Triple q, int kind,
          int column)
{
    switch (column) {
    case 0: return object.apply(q, kind, false);
    case 1: return object.apply(q, kind, true);
    case 2: return shader.apply(q, kind, false);
    case 3: return shader.apply(q, kind, true);
    default: return shader.apply(object.apply(q, kind, false), kind, true);
    }
}



bool
check_image(string_view filename, const Affine& object, const Affine& shader,
            std::vector<float>& pixels, Diagnostics& diagnostics)
{
    OIIO::ImageBuf image(filename);
    if (!image.read(0, 0, true, TypeDesc::FLOAT)) {
        diagnostics.errorfmt("Cannot read HART transform output '{}': {}",
                             filename, image.geterror());
        return false;
    }
    if (image.spec().width != width || image.spec().height != height
        || image.nchannels() != 3) {
        diagnostics.errorfmt("HART transform output '{}' must be {}x{} RGB",
                             filename, width, height);
        return false;
    }
    bool ok = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double u     = double(x) / (width - 1);
            const double v     = double(y) / (height - 1);
            const Triple value = transform(object, shader,
                                           { u + 0.25, 2 * v - 0.5, u + v + 1 },
                                           y, x);
            const int derivative_kind = y == 0 ? 1 : y;
            const double du = 1.0 / (width - 1), dv = 1.0 / (height - 1);
            const Triple dx = transform(object, shader, { du, 0, du },
                                        derivative_kind, x);
            const Triple dy = transform(object, shader, { 0, 2 * dv, dv },
                                        derivative_kind, x);
            const Triple expected { value[0] + 2 * value[1] + 3 * value[2],
                                    3 * dx[0] + dx[1] + 2 * dx[2],
                                    2 * dy[0] + 3 * dy[1] + dy[2] };
            for (int c = 0; c < 3; ++c) {
                const float actual = image.getchannel(x, y, 0, c);
                pixels.push_back(actual);
                if (!std::isfinite(actual)
                    || std::abs(double(actual) - expected[c])
                           > 2.0e-6 + 1.0e-6 * std::abs(expected[c])) {
                    diagnostics.errorfmt(
                        "{} pixel ({}, {}) channel {}: expected {:.9g}, got {:.9g}",
                        filename, x, y, c, expected[c], actual);
                    ok = false;
                }
            }
        }
    }
    return ok;
}



bool
artifact(ShadingSystem& ss, ShaderGroup& group, std::string& result,
         const void*& address, Diagnostics& diagnostics)
{
    uint64_t bytes = 0;
    if (!ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &address)
        || !ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &bytes)
        || !address || bytes < 4 || bytes > result.max_size()) {
        diagnostics.errorfmt(
            "Cannot retrieve the optimized HART transform artifact");
        return false;
    }
    result.assign(static_cast<const char*>(address), size_t(bytes));
    return true;
}



bool
run(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    OutputFiles outputs;
    if (!outputs.create(diagnostics))
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer) {
        diagnostics.errorfmt("Cannot create the HART transform test renderer");
        return false;
    }
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    for (const char* capability :
         { "HART", "HARTTextures", "HARTTransforms" }) {
        if (!renderer->supports(capability)) {
            diagnostics.errorfmt("HART test renderer does not advertise {}",
                                 capability);
            return false;
        }
    }
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? std::numeric_limits<int>::max()
                                               : 0)
        || !ss.attribute("llvm_optimize", 3) || !ss.attribute("optimize", 2)) {
        diagnostics.errorfmt("Cannot configure HART transform compilation");
        return false;
    }
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(source, oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_transform_rebind", oso)) {
        diagnostics.errorfmt("Cannot compile/load the HART transform shader");
        return false;
    }
    auto group = ss.ShaderGroupBegin("hart_transform_rebind_group");
    if (!group || !ss.Shader("surface", "hart_transform_rebind", "transform")
        || !ss.ShaderGroupEnd()) {
        diagnostics.errorfmt(
            "Cannot construct the HART transform shader group");
        return false;
    }
    const SymLocationDesc output("transform.Cout", TypeColor, false,
                                 SymArena::Outputs, 0, 3 * sizeof(float));
    ss.add_symlocs(group.get(), { &output, 1 });
    ss.optimize_group(group.get(), nullptr);
    std::string original_bitcode;
    const void* original_address = nullptr;
    if (!artifact(ss, *group, original_bitcode, original_address, diagnostics))
        return false;
    int local_bytes = 0;
    if (!ss.getattribute(group.get(), "hart_groupdata_alloc", local_bytes)
        || (local_bytes > 0) != (mode == "fused-local")) {
        diagnostics.errorfmt("Unexpected HART transform group storage mode");
        return false;
    }

    const Affine object_a { 2, 0.5, -0.25, 3, 0.75, 4, { 5, -2, 1.5 } };
    const Affine shader_a { 1.5, -0.75, 0.5, 2.5, -0.25, 0.5, { -3, 4, 2 } };
    const Affine object_b { -1.25, 0.25, 1, 0.75, -0.5, 2, { -4, 2, 7 } };
    const Affine shader_b { 0.5, 1.5, -0.75, -2, 0.25, 1.25, { 6, -3, -2 } };
    HartOptions options;
    options.no_cache = false;
    options.fused    = mode != "split";
    std::array<std::vector<float>, 3> images;
    for (int pass = 0; pass < 3; ++pass) {
        const Affine& object = pass == 1 ? object_b : object_a;
        const Affine& shader = pass == 1 ? shader_b : shader_a;
        print("HART transform pass {}\n", pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      width, height, 1, false, true, 0, false,
                                      outputs.files[pass], "float",
                                      object.matrix(), shader.matrix())) {
            diagnostics.errorfmt("HART transform launch {} failed", pass);
            return false;
        }
        if (!check_image(outputs.files[pass], object, shader, images[pass],
                         diagnostics))
            return false;
        std::string current_bitcode;
        const void* current_address = nullptr;
        if (!artifact(ss, *group, current_bitcode, current_address, diagnostics))
            return false;
        if (current_address != original_address
            || current_bitcode != original_bitcode) {
            diagnostics.errorfmt("HART transform artifact changed on launch {}",
                                 pass);
            return false;
        }
    }
    if (images[0] == images[1] || images[0] != images[2]) {
        diagnostics.errorfmt(
            "HART transform images did not follow A -> B -> A");
        return false;
    }
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}

}  // namespace



int
main(int argc, char* argv[])
{
    const string_view mode(argc == 3 ? argv[2] : "split");
    if ((argc != 2 && argc != 3)
        || (mode != "split" && mode != "fused" && mode != "fused-local")) {
        print(stderr,
              "Usage: hart_transform_test stdosl.h [split|fused|fused-local]\n");
        return 1;
    }
    Diagnostics diagnostics;
    OIIO_CHECK_ASSERT(run(argv[1], mode, diagnostics));
    OIIO_CHECK_EQUAL(diagnostics.errors, 0);
    OIIO_CHECK_EQUAL(diagnostics.warnings, 0);
    return unit_test_failures;
}
