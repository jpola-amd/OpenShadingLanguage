// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>

#include <algorithm>
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
constexpr int color_width = 8;

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

const char* color_source = R"OSL(
shader hart_color_rebind(output color Cout = 0)
{
    int column = int(7*u + 0.5);
    color c = color(u + 0.25, 2*v - 0.5, u + v + 1);
    color q = 0;
    if (column == 0) q = transformc("rgb", "XYZ", color(.25, .5, .75));
    else if (column == 1) q = transformc("XYZ", "rgb", c);
    else if (column == 2)
        q = color(luminance(color(1,0,0)), luminance(color(0,1,0)),
                  luminance(color(0,0,1)));
    else if (column == 3) q = color(luminance(c));
    else if (column == 4) q = transformc("rgb", "XYZ", c);
    else if (column == 5) q = blackbody(4256)/1000000;
    else if (column == 6) q = wavelength_color(500);
    else q = color("XYZ", u+0.25, 2*v-0.5, u+v+1);
    Cout = v < 0.25 ? q : (v < 0.75 ? Dx(q) : Dy(q));
}
)OSL";

const char* interactive_source = R"OSL(
shader hart_interactive_rebind(
    float gain = 2 [[int interactive=1]],
    float weights[2] = {3,5} [[int interactive=1]],
    int code = 16777219 [[int interactive=1]],
    matrix M = 1 [[int interactive=1]],
    string label = "red" [[int interactive=1]],
    string names[2] = {"red",""} [[int interactive=1]],
    output color Cout = 0)
{
    int i = int(u + 0.5);
    float q = gain * weights[i] * (u + 2*v) + M[0][0] + code % 97;
    Cout = color(q + (label == names[i] ? 10 : 0)
                   + (names[1] == "" ? 100 : 0), Dx(q), Dy(q));
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
    std::array<std::string, 5> files;
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
check_color_image(string_view filename, bool xyz, cspan<float> initial,
                  std::vector<float>& pixels, Diagnostics& diagnostics)
{
    // Source primaries and existing D65 xy=(.3127,.3291), solved in double.
    const std::array<Triple, 3> to_xyz {
        Triple { .412135323427, .357675002654, .180356796374 },
        Triple { .212507276142, .715350005308, .072142718550 },
        Triple { .019318843286, .119225000885, .949879127571 }
    };
    const std::array<Triple, 3> from_xyz {
        Triple { 3.242978965321, -1.538336175857, -.498919840819 },
        Triple { -.968997952917, 1.875491982259, .041544524053 },
        Triple { .055668324368, -.204117189350, 1.057698162996 }
    };
    auto transform_color = [&](Triple q, bool inverse) {
        if (xyz)
            return q;
        Triple result { };
        const auto& matrix = inverse ? from_xyz : to_xyz;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result[i] += matrix[i][j] * q[j];
        return result;
    };
    const Triple weights = xyz ? Triple { 0, 1, 0 } : to_xyz[1];
    OIIO::ImageBuf image(filename);
    if (!image.read(0, 0, true, TypeDesc::FLOAT)) {
        diagnostics.errorfmt("Cannot read HART color output '{}': {}", filename,
                             image.geterror());
        return false;
    }
    if (image.spec().width != color_width || image.spec().height != height
        || image.nchannels() != 3) {
        diagnostics.errorfmt("HART color output '{}' must be {}x{} RGB",
                             filename, color_width, height);
        return false;
    }
    bool ok = true;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < color_width; ++x) {
            const double u  = double(x) / (color_width - 1);
            const double v  = double(y) / (height - 1);
            const double du = 1.0 / (color_width - 1), dv = 1.0 / (height - 1);
            const Triple q = y == 0 ? Triple { u + .25, 2 * v - .5, u + v + 1 }
                             : y == 1 ? Triple { du, 0, du }
                                      : Triple { 0, 2 * dv, dv };
            Triple expected { };
            if (x == 0 && y == 0)
                expected = transform_color({ .25, .5, .75 }, false);
            else if (x == 1)
                expected = transform_color(q, true);
            else if (x == 2 && y == 0)
                expected = weights;
            else if (x == 3) {
                const double l = weights[0] * q[0] + weights[1] * q[1]
                                 + weights[2] * q[2];
                expected       = { l, l, l };
            } else if (x == 4)
                expected = transform_color(q, false);
            else if (x == 5 && y == 0 && !initial.empty()) {
                // 4256K is a table knot: compare a linear basis change, not
                // the space-dependent approximation between lookup entries.
                Triple rgb { initial[15], initial[16], initial[17] };
                if (xyz)
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            expected[i] += to_xyz[i][j] * rgb[j];
                else
                    expected = rgb;
            } else if (x == 6 && y == 0) {
                expected = transform_color({ .0049, .3230, .2720 }, true);
                for (auto& component : expected)
                    component = std::max(0.0, component / 2.52);
            } else if (x == 7 && y == 0)
                expected = transform_color(q, true);
            for (int c = 0; c < 3; ++c) {
                const float actual = image.getchannel(x, y, 0, c);
                pixels.push_back(actual);
                if (x == 5 && y == 0 && initial.empty()) {
                    if (!std::isfinite(actual) || actual <= 0) {
                        diagnostics.errorfmt("Invalid blackbody baseline {}",
                                             actual);
                        ok = false;
                    }
                    continue;
                }
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
            "Cannot retrieve the optimized HART group artifact");
        return false;
    }
    result.assign(static_cast<const char*>(address), size_t(bytes));
    return true;
}



bool
check_color_rejection(SimpleRenderer& renderer, ShadingSystem& ss,
                      const HartOptions& options, string_view arch,
                      string_view stdosl, string_view filename,
                      Diagnostics& diagnostics)
{
    const char* shader = R"OSL(
shader color_rebound_error(output color Cout=0)
{ Cout=color("Rec709",u+.25,v+.5,1); }
)OSL";
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(shader, oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("color_rebound_error", oso))
        return false;
    auto group = ss.ShaderGroupBegin("color_rebound_error_group");
    if (!group || !ss.Shader("surface", "color_rebound_error", "c")
        || !ss.ShaderGroupEnd())
        return false;
    const SymLocationDesc output("c.Cout", TypeColor, false, SymArena::Outputs,
                                 0, 3 * sizeof(float));
    ss.add_symlocs(group.get(), { &output, 1 });
    ss.optimize_group(group.get(), nullptr);
    std::string original;
    const void* address = nullptr;
    if (!artifact(ss, *group, original, address, diagnostics)
        || !ss.attribute("colorspace", "XYZ"))
        return false;
    Matrix44 identity;
    identity.makeIdentity();
    print("HART color unsupported rebound\n");
    std::fflush(stdout);
    if (testshade_hart_generated(renderer, ss, *group, options, arch,
                                 color_width, height, 1, false, true, 0, false,
                                 filename, "float", identity, identity)
        || OIIO::Filesystem::exists(filename)) {
        diagnostics.errorfmt("Unsupported rebound color was published");
        return false;
    }
    if (!ss.attribute("colorspace", "Rec709")
        || !testshade_hart_generated(renderer, ss, *group, options, arch,
                                     color_width, height, 1, false, true, 0,
                                     false, filename, "float", identity,
                                     identity))
        return false;
    OIIO::ImageBuf image(filename);
    if (!image.read(0, 0, true, TypeDesc::FLOAT)) {
        diagnostics.errorfmt("Cannot read recovered color image: {}",
                             image.geterror());
        return false;
    }
    if (image.spec().width != color_width || image.spec().height != height
        || image.nchannels() != 3) {
        diagnostics.errorfmt("Invalid recovered color image dimensions");
        return false;
    }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < color_width; ++x) {
            const Triple expected { double(x) / (color_width - 1) + .25,
                                    double(y) / (height - 1) + .5, 1 };
            for (int c = 0; c < 3; ++c) {
                const float actual = image.getchannel(x, y, 0, c);
                if (!std::isfinite(actual)
                    || std::abs(actual - expected[c]) > 2e-6) {
                    diagnostics.errorfmt("Incorrect recovered color pixel");
                    return false;
                }
            }
        }
    std::string current;
    const void* current_address = nullptr;
    if (!artifact(ss, *group, current, current_address, diagnostics)
        || current_address != address || current != original) {
        diagnostics.errorfmt("Color artifact changed during error recovery");
        return false;
    }
    print("HART color error recovery passed\n");
    return true;
}



bool
check_interactive_image(string_view filename, bool changed,
                        std::vector<float>& pixels, Diagnostics& diagnostics)
{
    OIIO::ImageBuf image(filename);
    if (!image.read(0, 0, true, TypeFloat) || image.spec().width != width
        || image.spec().height != height || image.nchannels() != 3) {
        diagnostics.errorfmt("Invalid interactive output '{}': {}", filename,
                             image.geterror());
        return false;
    }
    pixels.resize(width * height * 3);
    if (!image.get_pixels(image.roi(), TypeFloat, pixels.data())) {
        diagnostics.errorfmt("Cannot read interactive pixels: {}",
                             image.geterror());
        return false;
    }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const float u      = float(x) / (width - 1);
            const float v      = float(y) / (height - 1);
            const int i        = int(u + .5f);
            const float gain   = changed ? 4 : 2;
            const float weight = changed ? (i ? -2 : 7) : (i ? 5 : 3);
            const float q      = gain * weight * (u + 2 * v) + (changed ? 3 : 1)
                                 + (changed ? 16777223 : 16777219) % 97;
            const float expected[] = {
                q + ((changed ? i == 1 : i == 0) ? 10 : 0)
                    + (changed ? 0 : 100),
                gain * weight / (width - 1), 2 * gain * weight / (height - 1)
            };
            for (int c = 0; c < 3; ++c)
                OIIO_CHECK_EQUAL(pixels[(y * width + x) * 3 + c], expected[c]);
        }
    return true;
}



bool
run(string_view stdosl, string_view mode, bool color, bool interactive,
    Diagnostics& diagnostics)
{
    const char* kind        = interactive ? "interactive"
                              : color     ? "color"
                                          : "transform";
    const char* shader_name = interactive ? "hart_interactive_rebind"
                              : color     ? "hart_color_rebind"
                                          : "hart_transform_rebind";
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
    if (color
        && (!renderer->supports("HARTColorSystem")
            || !ss.attribute("colorspace", "Rec709"))) {
        diagnostics.errorfmt("Cannot configure HART color-system bindings");
        return false;
    }
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? std::numeric_limits<int>::max()
                                               : 0)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2)) {
        diagnostics.errorfmt("Cannot configure HART transform compilation");
        return false;
    }
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(interactive ? interactive_source
                                 : color     ? color_source
                                             : source,
                                 oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader(shader_name, oso)) {
        diagnostics.errorfmt("Cannot compile/load the HART transform shader");
        return false;
    }
    auto group = ss.ShaderGroupBegin(fmtformat("{}_group", shader_name));
    if (!group || !ss.Shader("surface", shader_name, kind)
        || !ss.ShaderGroupEnd()) {
        diagnostics.errorfmt(
            "Cannot construct the HART transform shader group");
        return false;
    }
    const SymLocationDesc output(fmtformat("{}.Cout", kind), TypeColor, false,
                                 SymArena::Outputs, 0, 3 * sizeof(float));
    ss.add_symlocs(group.get(), { &output, 1 });
    ss.optimize_group(group.get(), nullptr);
    std::string original_bitcode;
    const void* original_address = nullptr;
    if (!artifact(ss, *group, original_bitcode, original_address, diagnostics))
        return false;
    void* original_interactive = nullptr;
    if (interactive
        && (!ss.getattribute(group.get(), "device_interactive_params",
                             TypeDesc::PTR, &original_interactive)
            || !original_interactive)) {
        diagnostics.errorfmt("Missing interactive device allocation");
        return false;
    }
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
    options.fused    = mode != "split" && mode != "unoptimized";
    std::array<std::vector<float>, 3> images;
    for (int pass = 0; pass < 3; ++pass) {
        const Affine& object = pass == 1 ? object_b : object_a;
        const Affine& shader = pass == 1 ? shader_b : shader_a;
        if (interactive && pass) {
            const bool changed    = pass == 1;
            const float weights[] = { changed ? 7.0f : 3.0f,
                                      changed ? -2.0f : 5.0f };
            const ustring names[] = { ustring(changed ? "green" : "red"),
                                      ustring(changed ? "blue" : "") };
            const Matrix44 matrix(changed ? 3.0f : 1.0f);
            if (!ss.ReParameter(*group, kind, "gain", changed ? 4.0f : 2.0f)
                || !ss.ReParameter(*group, kind, "weights",
                                   TypeDesc(TypeDesc::FLOAT, 2), weights)
                || !ss.ReParameter(*group, kind, "code",
                                   changed ? 16777223 : 16777219)
                || !ss.ReParameter(*group, kind, "M", TypeMatrix, &matrix)
                || !ss.ReParameter(*group, kind, "label",
                                   std::string(changed ? "blue" : "red"))
                || !ss.ReParameter(*group, kind, "names",
                                   TypeDesc(TypeDesc::STRING, 2), names)) {
                diagnostics.errorfmt("Cannot update interactive values");
                return false;
            }
        }
        if (color
            && !ss.attribute("colorspace", pass == 1 ? "XYZ" : "Rec709")) {
            diagnostics.errorfmt("Cannot rebind HART color system");
            return false;
        }
        print("HART {} pass {}\n", kind, pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      color ? color_width : width, height, 1,
                                      false, true, 0, false,
                                      outputs.files[pass], "float",
                                      object.matrix(), shader.matrix())) {
            diagnostics.errorfmt("HART transform launch {} failed", pass);
            return false;
        }
        if (!(interactive
                  ? check_interactive_image(outputs.files[pass], pass == 1,
                                            images[pass], diagnostics)
              : color ? check_color_image(outputs.files[pass], pass == 1,
                                          pass ? cspan<float>(images[0])
                                               : cspan<float>(),
                                          images[pass], diagnostics)
                      : check_image(outputs.files[pass], object, shader,
                                    images[pass], diagnostics)))
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
        if (interactive) {
            void* current = nullptr;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "device_interactive_params",
                                              TypeDesc::PTR, &current));
            OIIO_CHECK_EQUAL(current, original_interactive);
        }
    }
    if (images[0] == images[1] || images[0] != images[2]) {
        diagnostics.errorfmt(
            "HART transform images did not follow A -> B -> A");
        return false;
    }
    if (color
        && !check_color_rejection(*renderer, ss, options, arch, stdosl,
                                  outputs.files[3], diagnostics))
        return false;
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}

bool
run_outputs(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    OutputFiles files;
    if (!files.create(diagnostics))
        return false;
    for (auto& file : files.files)
        file += ".tif";
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? 1048576 : 0)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2))
        return false;
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(R"OSL(
        shader hart_output_rebind(output int index = 0,
                                  output float values[2] = {0,0},
                                  output color Cout = 0)
        {
            index = 16777217 + int(2*u);
            values[0] = u + 2*v;
            values[1] = u - v;
            Cout = color(u,v,1);
        })OSL",
                                 oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_output_rebind", oso))
        return false;
    auto group = ss.ShaderGroupBegin("hart_output_rebind_group");
    if (!group || !ss.Shader("surface", "hart_output_rebind", "out")
        || !ss.ShaderGroupEnd())
        return false;
    HartOptions options;
    options.fused          = mode == "fused" || mode == "fused-local";
    options.shader_entries = { "out" };
    const Matrix44 identity(1);
    std::string original_bitcode;
    const void* original_address = nullptr;
    for (int pass = 0; pass < 3; ++pass) {
        const int columns = pass == 1 ? 1 : 3;
        const std::array<HartOutputRequest, 3> requests { {
            { "out.index", files.files[pass] },
            { "index", "null" },
            { "values", files.files[3] },
        } };
        print("HART output rebind pass {}\n", pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      columns, 2, 1, false, true, 0, false,
                                      "null", "", identity, identity, requests))
            return false;
        OIIO::ImageBuf indices(files.files[pass]), values(files.files[3]);
        std::vector<int> integer_pixels(size_t(columns) * 2);
        std::vector<float> float_pixels(size_t(columns) * 4);
        if (!indices.read(0, 0, true) || !values.read(0, 0, true)
            || indices.spec().nchannels != 1 || values.spec().nchannels != 2) {
            diagnostics.errorfmt("Cannot read HART output rebind images: {} {}",
                                 indices.geterror(), values.geterror());
            return false;
        }
#if OIIO_VERSION_GREATER_EQUAL(3, 1, 0)
        const bool copied = indices.get_pixels(OIIO::ROI::All(),
                                               OIIO::make_span(integer_pixels))
                            && values.get_pixels(OIIO::ROI::All(),
                                                 OIIO::make_span(float_pixels));
#else
        const bool copied = indices.get_pixels(OIIO::ROI::All(), TypeInt,
                                               integer_pixels.data())
                            && values.get_pixels(OIIO::ROI::All(), TypeFloat,
                                                 float_pixels.data());
#endif
        if (!copied) {
            diagnostics.errorfmt("Cannot copy HART output rebind pixels");
            return false;
        }
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < columns; ++x) {
                const float u      = columns == 1 ? 0.5f : float(x) / 2;
                const size_t index = size_t(y) * columns + x;
                OIIO_CHECK_EQUAL(integer_pixels[index], 16777217 + int(2 * u));
                OIIO_CHECK_EQUAL(float_pixels[2 * index], u + 2 * y);
                OIIO_CHECK_EQUAL(float_pixels[2 * index + 1], u - y);
            }
        std::string bitcode;
        const void* address = nullptr;
        if (!artifact(ss, *group, bitcode, address, diagnostics))
            return false;
        if (pass == 0) {
            original_bitcode = bitcode;
            original_address = address;
        } else {
            OIIO_CHECK_EQUAL(address, original_address);
            OIIO_CHECK_EQUAL(bitcode, original_bitcode);
        }
    }
    // Subsets, reordered layouts and newly requested stores cannot reinterpret
    // the allocation still used by the compiled callable.
    const std::array<std::vector<HartOutputRequest>, 3> invalid { {
        { { "index", "null" } },
        { { "values", "null" }, { "index", "null" } },
        { { "index", "null" }, { "values", "null" }, { "Cout", "null" } },
    } };
    for (const auto& requests : invalid) {
        OIIO_CHECK_ASSERT(
            !testshade_hart_generated(*renderer, ss, *group, options, arch, 3,
                                      2, 1, false, true, 0, false, "null", "",
                                      identity, identity, requests));
    }
    std::string bitcode;
    const void* address = nullptr;
    if (!artifact(ss, *group, bitcode, address, diagnostics))
        return false;
    OIIO_CHECK_EQUAL(address, original_address);
    OIIO_CHECK_EQUAL(bitcode, original_bitcode);
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



bool
run_userdata(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    OutputFiles files;
    if (!files.create(diagnostics))
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? 1048576 : 0)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2))
        return false;
    const char* sources[] = {
        "shader hart_ud_p(float field=2 [[int interpolated=1]],"
        "output float value=0) { value=field; }",
        "shader hart_ud_c(float value=0, float field=7 [[int interpolated=1]],"
        "output color Cout=0) { Cout=color(value+field,Dx(field),Dy(field)); }"
    };
    for (int i = 0; i < 2; ++i) {
        OSLCompiler compiler(&diagnostics);
        std::string oso;
        if (!compiler.compile_buffer(sources[i], oso, { }, stdosl)
            || !ss.LoadMemoryCompiledShader(i ? "hart_ud_c" : "hart_ud_p", oso))
            return false;
    }
    auto group = ss.ShaderGroupBegin("hart_userdata_rebind");
    if (!group || !ss.Shader("surface", "hart_ud_p", "producer")
        || !ss.Shader("surface", "hart_ud_c", "consumer")
        || !ss.ConnectShaders("producer", "value", "consumer", "value")
        || !ss.ShaderGroupEnd())
        return false;
    const SymLocationDesc output("consumer.Cout", TypeColor, false,
                                 SymArena::Outputs, 0, 12);
    ss.add_symlocs(group.get(), { &output, 1 });
    ss.optimize_group(group.get(), nullptr);
    std::string original;
    const void* original_address = nullptr;
    if (!artifact(ss, *group, original, original_address, diagnostics))
        return false;
    constexpr size_t count = width * height;
    std::array<std::array<float, 4>, count> data;
    std::array<uint8_t, count> present;
    HartOptions options;
    options.fused = mode == "fused" || mode == "fused-local";
    const HartUserdataBinding binding {
        "field",
        TypeFloat,
        true,
        sizeof(data[0]),
        { reinterpret_cast<const std::byte*>(data.data()), sizeof(data) },
        present
    };
    options.userdata_bindings = { binding };
    const Matrix44 identity;
    std::array<std::vector<float>, 3> images;
    for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < count; ++i) {
            const float n = float(i + 1);
            data[i] = pass == 1
                          ? std::array<float, 4> { 20 - n, -.5f * n, .75f * n,
                                                   999 }
                          : std::array<float, 4> { n, .25f * n, -.5f * n, 999 };
            present[i] = i % 3 != size_t(pass == 1);
        }
        print("HART userdata pass {}\n", pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      width, height, 1, false, true, 0, false,
                                      files.files[pass], "float", identity,
                                      identity))
            return false;
        OIIO::ImageBuf image(files.files[pass]);
        if (!image.read(0, 0, true, TypeFloat) || image.nchannels() != 3
            || image.spec().width != width || image.spec().height != height) {
            diagnostics.errorfmt("Invalid userdata output: {}",
                                 image.geterror());
            return false;
        }
        images[pass].resize(count * 3);
        if (!image.get_pixels(image.roi(), TypeFloat, images[pass].data())) {
            diagnostics.errorfmt("Cannot read userdata output: {}",
                                 image.geterror());
            return false;
        }
        for (size_t i = 0; i < count; ++i)
            for (int c = 0; c < 3; ++c) {
                const float expected = present[i] ? data[i][c] * (c ? 1 : 2)
                                                  : (c ? 0.0f : 9.0f);
                OIIO_CHECK_EQUAL(images[pass][3 * i + c], expected);
            }
        std::string current;
        const void* address = nullptr;
        if (!artifact(ss, *group, current, address, diagnostics))
            return false;
        OIIO_CHECK_EQUAL(current, original);
        OIIO_CHECK_EQUAL(address, original_address);
    }
    OIIO_CHECK_ASSERT(images[0] != images[1] && images[0] == images[2]);
    for (int invalid = 0; invalid < 7; ++invalid) {
        present[0]                = 0;
        options.userdata_bindings = { binding };
        auto& entry               = options.userdata_bindings[0];
        if (invalid == 0)
            entry.data = { binding.data.data(), sizeof(float) };
        else if (invalid == 1)
            entry.stride = sizeof(float);
        else if (invalid == 2)
            entry.present = { present.data(), 1 };
        else if (invalid == 3)
            entry.type = TypeString;
        else if (invalid == 4)
            entry.type = TypeDesc(TypeDesc::BASETYPE(255));
        else if (invalid == 5)
            present[0] = 2;
        else
            options.userdata_bindings.push_back(binding);
        OIIO_CHECK_ASSERT(
            !testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      width, height, 1, false, true, 0, false,
                                      "null", "float", identity, identity));
    }
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



bool
run_renderer_library(string_view stdosl, string_view mode,
                     string_view library_dir, Diagnostics& diagnostics)
{
    OutputFiles files;
    if (!files.create(diagnostics))
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? 1048576 : 0)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2))
        return false;
    std::string libraries[2];
    for (int i = 0; i < 2; ++i) {
        const auto path = std::filesystem::path(std::string(library_dir.data(),
                                                            library_dir.size()))
                          / arch
                          / fmtformat("hart_renderer_services_test_{}_{}.bc",
                                      i ? "b" : "a", arch);
        OIIO::Filesystem::IOFile input(path.string(),
                                       OIIO::Filesystem::IOProxy::Read);
        if (!input.opened() || !input.size()
            || input.size() > size_t(std::numeric_limits<int>::max())) {
            diagnostics.errorfmt("Cannot open renderer library '{}'",
                                 path.string());
            return false;
        }
        libraries[i].resize(input.size());
        if (input.read(libraries[i].data(), libraries[i].size())
            != libraries[i].size()) {
            diagnostics.errorfmt("Cannot read renderer library '{}'",
                                 path.string());
            return false;
        }
    }
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(
            "shader hart_renderer_library(float renderer_value=-99 "
            "[[int lockgeom=0]], output color Cout=0) { "
            "Cout=color(renderer_value,Dx(renderer_value),Dy(renderer_value)); }",
            oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_renderer_library", oso))
        return false;
    ShaderGroupRef groups[2];
    std::string artifacts[2];
    const void* addresses[2] = { nullptr, nullptr };
    for (int i = 0; i < 2; ++i) {
        if (!ss.attribute("lib_bitcode",
                          TypeDesc(TypeDesc::UINT8, int(libraries[i].size())),
                          libraries[i].data()))
            return false;
        groups[i] = ss.ShaderGroupBegin(i ? "hart_library_b"
                                          : "hart_library_a");
        if (!groups[i] || !ss.Shader("surface", "hart_renderer_library", "out")
            || !ss.ShaderGroupEnd())
            return false;
        const SymLocationDesc output("out.Cout", TypeColor, false,
                                     SymArena::Outputs, 0, 12);
        ss.add_symlocs(groups[i].get(), { &output, 1 });
        ss.optimize_group(groups[i].get(), nullptr);
        if (!artifact(ss, *groups[i], artifacts[i], addresses[i], diagnostics))
            return false;
    }
    OIIO_CHECK_ASSERT(artifacts[0] != artifacts[1]);
    HartOptions options;
    options.fused = mode == "fused" || mode == "fused-local";
    const Matrix44 identity;
    std::array<std::vector<float>, 3> images;
    for (int pass = 0; pass < 3; ++pass) {
        const int version = pass == 1 ? 1 : 0;
        print("HART library pass {}\n", pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *groups[version], options,
                                      arch, width, height, 1, false, true, 0,
                                      false, files.files[pass], "float",
                                      identity, identity))
            return false;
        OIIO::ImageBuf image(files.files[pass]);
        if (!image.read(0, 0, true, TypeFloat) || image.nchannels() != 3
            || image.spec().width != width || image.spec().height != height) {
            diagnostics.errorfmt("Invalid renderer-library output: {}",
                                 image.geterror());
            return false;
        }
        images[pass].resize(width * height * 3);
        if (!image.get_pixels(image.roi(), TypeFloat, images[pass].data())) {
            diagnostics.errorfmt("Cannot read renderer-library output: {}",
                                 image.geterror());
            return false;
        }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const int index      = width * y + x;
                const float expected = (version ? -2.5f : 1.25f)
                                       + float(x) / (width - 1)
                                       + 2.0f * y / (height - 1)
                                       + 0.125f * index;
                OIIO_CHECK_EQUAL(images[pass][3 * index], expected);
                OIIO_CHECK_EQUAL(images[pass][3 * index + 1],
                                 1.0f / (width - 1));
                OIIO_CHECK_EQUAL(images[pass][3 * index + 2],
                                 2.0f / (height - 1));
            }
        for (int i = 0; i < 2; ++i) {
            std::string current;
            const void* address = nullptr;
            if (!artifact(ss, *groups[i], current, address, diagnostics))
                return false;
            OIIO_CHECK_EQUAL(current, artifacts[i]);
            OIIO_CHECK_EQUAL(address, addresses[i]);
        }
    }
    OIIO_CHECK_ASSERT(images[0] != images[1] && images[0] == images[2]);
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



bool
run_attributes(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    constexpr int columns = 33, rows = 3;
    OutputFiles files;
    if (!files.create(diagnostics))
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    const int optimize = mode == "unoptimized" ? 0 : 2;
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? 1048576 : 0)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("optimize", optimize))
        return false;
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(R"OSL(
shader hart_attributes(output color Cout=-1000)
{
    int column = int(32*u + .5);
    float q = u + 2*v;
    int ok = 0;
    if (column == 0) {
        ok = getattribute("camera", "camera:fov", q);
    } else if (column == 1) {
        string obj = v < .25 ? "" : "my_camera";
        string name = v < .25 ? "camera:clip_near" : "camera:clip_far";
        ok = getattribute(obj, name, int(2*v)-1, q);
    } else if (column == 2) {
        int resolution[2] = {-1,-2};
        if (getattribute("camera:resolution", resolution))
            Cout = color(resolution[0],resolution[1],1);
    } else if (column == 3) {
        string projection = "initial";
        if (getattribute("camera:projection", projection))
            Cout = color(projection=="perspective",projection=="orthographic",1);
    } else if (column == 4) {
        ok = getattribute("camera:pixelaspect", q);
    } else if (column == 5) {
        float window[4] = {u,v,u+v,u+2*v};
        if (getattribute("camera:screen_window", window)) {
            if (v < .25) Cout = color(window[0],window[1],window[2]);
            else if (v < .75) Cout = color(window[3],Dx(window[0]),Dy(window[3]));
            else Cout = color(Dx(window[1])+Dx(window[2])+Dx(window[3]),
                              Dy(window[0])+Dy(window[1])+Dy(window[2]),1);
        }
    } else if (column == 6 || column == 7 || column == 19) {
        float values[2] = {u+v,u+2*v};
        string name = column == 6 ? "camera:clip" :
                      column == 7 ? "camera:shutter" : "array_values";
        if (getattribute(name, -1, values))
            Cout = v < .25 ? color(values[0],values[1],1) :
                   v < .75 ? color(Dx(values[0]),Dx(values[1]),1) :
                             color(Dy(values[0]),Dy(values[1]),1);
    } else if (column == 8) {
        ok = getattribute("camera:shutter_open", q);
    } else if (column == 9) {
        ok = getattribute("camera:shutter_close", q);
    } else if (column == 10) {
        ok = getattribute("camera:clip_near", 3, q);
    } else if (column == 11) {
        ok = getattribute("camera:clip_far", q);
    } else if (column == 12) {
        string name = v < .25 ? "osl:version" : "missing";
        int version = -99;
        int found = getattribute(name, version);
        Cout = color(found,version,0);
    } else if (column == 13) {
        int index = -1;
        if (getattribute("", "shading:index", 0, index))
            Cout = color(index,1,0);
    } else if (column == 14 || column == 21) {
        int found = column == 14 ? getattribute("missing", q) :
                                  getattribute("object", "s", q);
        ok = !found;
    } else if (column == 15 || column == 31) {
        int value = 7;
        int found = column == 15 ? getattribute("camera:fov", value) :
                                  getattribute("object", "shading:index", value);
        Cout = color(value,found,0);
    } else if (column == 16) {
        float values[3] = {q,7,9};
        if (!getattribute("camera:clip", values))
            Cout = v < .25 ? color(values[0],values[1],values[2]) :
                   v < .75 ? color(Dx(values[0]),Dx(values[1]),Dx(values[2])) :
                             color(Dy(values[0]),Dy(values[1]),Dy(values[2]));
    } else if (column == 17) {
        ok = getattribute("s", q);
    } else if (column == 18) {
        ok = getattribute("renderer_weight", q);
    } else if (column == 20) {
        string label = "initial";
        if (getattribute("label", label))
            Cout = color(label=="hello",label=="world",1);
    } else if (column == 22) {
        int found = getattribute("s", int(2*v)-1, q);
        ok = found == (v < .25);
    } else if (column == 23) {
        string a="initial", b="initial", c="initial";
        int aa=getattribute("shader:shadername",a);
        int bb=getattribute("shader:layername",b);
        int cc=getattribute("shader:groupname",c);
        Cout=color(aa ? (a=="hart_attributes" ? 1 : -1000) :
                       (a=="initial" ? 0 : -1000),
                   bb ? (b=="out" ? 1 : -1000) : (b=="initial" ? 0 : -1000),
                   cc ? (c=="hart_attributes" ? 1 : -1000) :
                       (c=="initial" ? 0 : -1000));
    } else if (column == 24) {
        string name=v < .25 ? "shader:shadername" :
                    v < .75 ? "shader:layername" : "shader:groupname";
        string value="initial";
        int found=getattribute(name,value);
        Cout=color(found,value=="initial",0);
    } else if (column == 25) {
        matrix M=1;
        if (getattribute("matrix_value",M)) {
            float sum=0, diagonal=0;
            for (int i=0; i<4; ++i)
                for (int j=0; j<4; ++j) {
                    sum += M[i][j];
                    if (i==j) diagonal += M[i][j];
                }
            Cout=color(sum,diagonal,1);
        }
    } else if (column == 26) {
        color tint=0;
        if (getattribute("tint",tint)) Cout=tint;
    } else if (column == 27) {
        normal n=normal(q,7,9);
        if (!getattribute("tint",n))
            Cout = color(v < .25 ? n : v < .75 ? Dx(n) : Dy(n));
    } else if (column == 28 || column == 30) {
        string labels[2]={"sentinel",""};
        if (column == 28) {
            if (getattribute("labels",labels))
                Cout=color(labels[0]=="hello",labels[1]=="world",1);
        } else {
            int found=getattribute("never",int(v),labels);
            Cout=color(found,labels[0]=="sentinel",labels[1]=="");
        }
    } else if (column == 29) {
        ok=getattribute("options","blahblah",q);
    } else if (column == 32) {
        string object=v < .25 ? "" : "scene";
        int found=getattribute(object,"renderer_weight",-1,q);
        ok=found==(v < .25);
    }
    if (ok) Cout=color(q,Dx(q),Dy(q));
}
)OSL",
                                 oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_attributes", oso))
        return false;
    auto group = ss.ShaderGroupBegin("hart_attributes");
    if (!group || !ss.Shader("surface", "hart_attributes", "out")
        || !ss.ShaderGroupEnd())
        return false;
    const SymLocationDesc output("out.Cout", TypeColor, false,
                                 SymArena::Outputs, 0, 12);
    ss.add_symlocs(group.get(), { &output, 1 });
    HartOptions options;
    options.fused = mode == "fused" || mode == "fused-local";
    const Matrix44 identity(1);
    std::string original;
    const void* original_address = nullptr;
    std::array<std::vector<float>, 3> images;
    for (int pass = 0; pass < 3; ++pass) {
        const bool b = pass == 1;
        auto bind    = [&](SimpleRenderer& r) {
            r.camera_params(identity,
                            ustringhash(b ? "orthographic" : "perspective"),
                            b ? 72 : 45, b ? .5f : .125f, b ? 128 : 64,
                            b ? 300 : 640, b ? 600 : 320);
            r.userdata.clear();
            auto add = [&](const char* name, TypeDesc type, const void* data) {
                r.userdata.emplace_back(ustring(name), type, 1, data);
            };
            const float weight   = b ? -1.5f : 2.5f;
            const float values[] = { b ? -2.0f : 3.0f, b ? 7.0f : -4.0f };
            const float tint[]   = { b ? .5f : .125f, b ? 1.0f : .25f,
                                     b ? 2.0f : .5f };
            const ustring label(b ? "world" : "hello");
            const ustring labels[] = { ustring(b ? "" : "hello"),
                                       ustring(b ? "world" : "") };
            Matrix44 matrix;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    matrix[i][j] = (4 * i + j + 1) * (b ? -.125f : .0625f);
            add("renderer_weight", TypeFloat, &weight);
            add("array_values", TypeDesc(TypeDesc::FLOAT, 2), values);
            add("label", TypeString, &label);
            add("labels", TypeDesc(TypeDesc::STRING, 2), labels);
            add("matrix_value", TypeMatrix, &matrix);
            add("tint", TypeColor, tint);
        };
        bind(*renderer);
        print("HART attributes pass {}\n", pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      columns, rows, 1, false, true, 0, false,
                                      files.files[pass], "float", identity,
                                      identity))
            return false;
        OIIO::ImageBuf image(files.files[pass]);
        if (!image.read(0, 0, true, TypeFloat) || image.nchannels() != 3
            || image.spec().width != columns || image.spec().height != rows) {
            diagnostics.errorfmt("Invalid attribute output: {}",
                                 image.geterror());
            return false;
        }
        images[pass].resize(columns * rows * 3);
        if (!image.get_pixels(image.roi(), TypeFloat, images[pass].data())) {
            diagnostics.errorfmt("Cannot read attribute output: {}",
                                 image.geterror());
            return false;
        }
        SimpleRenderer cpu_renderer;
        bind(cpu_renderer);
        ShadingSystem cpu(&cpu_renderer, nullptr, &diagnostics);
        cpu_renderer.init_shadingsys(&cpu);
        if (!cpu.attribute("optimize", optimize)
            || !cpu.attribute("llvm_debugging_symbols", 0)
            || !cpu.LoadMemoryCompiledShader("hart_attributes", oso))
            return false;
        auto cpu_group = cpu.ShaderGroupBegin("hart_attributes");
        if (!cpu_group || !cpu.Shader("surface", "hart_attributes", "out")
            || !cpu.ShaderGroupEnd())
            return false;
        cpu.add_symlocs(cpu_group.get(), { &output, 1 });
        auto free_thread = [&](PerThreadInfo* p) {
            cpu.destroy_thread_info(p);
        };
        std::unique_ptr<PerThreadInfo, decltype(free_thread)> thread(
            cpu.create_thread_info(), free_thread);
        auto free_context = [&](ShadingContext* p) { cpu.release_context(p); };
        std::unique_ptr<ShadingContext, decltype(free_context)> context(
            cpu.get_context(thread.get()), free_context);
        std::vector<float> reference(columns * rows * 3, -1000);
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < columns; ++x) {
                const int index = y * columns + x;
                ShaderGlobals sg { };
                sg.u    = float(x) / (columns - 1);
                sg.v    = float(y) / (rows - 1);
                sg.dudx = 1.0f / (columns - 1);
                sg.dvdy = 1.0f / (rows - 1);
                if (!cpu.execute(*context, *cpu_group, 0, index, sg, nullptr,
                                 reference.data()))
                    return false;
                const float q         = sg.u + 2 * sg.v;
                const float clip_near = b ? .5f : .125f;
                const float clip_far  = b ? 128 : 64;
                const float weight    = b ? -1.5f : 2.5f;
                std::array<float, 3> expected { };
                switch (x) {
                case 0: expected = { b ? 72.0f : 45.0f, 0, 0 }; break;
                case 1:
                    expected = { y == 0 ? clip_near : clip_far, 0, 0 };
                    break;
                case 2:
                    expected = { b ? 300.0f : 640.0f, b ? 600.0f : 320.0f, 1 };
                    break;
                case 3:
                case 20:
                case 28:
                    expected = { b ? 0.0f : 1.0f, b ? 1.0f : 0.0f, 1 };
                    break;
                case 4:
                case 9: expected = { 1, 0, 0 }; break;
                case 5:
                    expected = y == 0 ? std::array<float, 3> { b ? -.5f : -2,
                                                               -1, b ? .5f : 2 }
                               : y == 1 ? std::array<float, 3> { 1, 0, 0 }
                                        : std::array<float, 3> { 0, 0, 1 };
                    break;
                case 6:
                case 7:
                case 19:
                    expected = { 0, 0, 1 };
                    if (y == 0) {
                        expected[0] = x == 6   ? clip_near
                                      : x == 7 ? 0
                                      : b      ? -2
                                               : 3;
                        expected[1] = x == 6   ? clip_far
                                      : x == 7 ? 1
                                      : b      ? 7
                                               : -4;
                    }
                    break;
                case 8: break;
                case 10: expected = { clip_near, 0, 0 }; break;
                case 11: expected = { clip_far, 0, 0 }; break;
                case 12:
                    expected = { y == 0 ? 1.0f : 0.0f,
                                 y == 0 ? float(OSL_VERSION) : -99, 0 };
                    break;
                case 13: expected = { float(index), 1, 0 }; break;
                case 14:
                case 21: expected = { q, sg.dudx, 1 }; break;
                case 15:
                case 31: expected = { 7, 0, 0 }; break;
                case 16:
                case 27:
                    expected = y == 0 ? std::array<float, 3> { q, 7, 9 }
                                      : std::array<float, 3> { y == 1 ? sg.dudx
                                                                      : 1,
                                                               0, 0 };
                    break;
                case 17: expected = { sg.u, sg.dudx, 0 }; break;
                case 18: expected = { weight, 0, 0 }; break;
                case 22:
                    expected = { y == 0 ? sg.u : q, sg.dudx,
                                 y == 0 ? 0.0f : 1.0f };
                    break;
                case 23: expected.fill(optimize ? 1.0f : 0.0f); break;
                case 24: expected = { 0, 1, 0 }; break;
                case 25:
                    expected = { b ? -17.0f : 8.5f, b ? -4.25f : 2.125f, 1 };
                    break;
                case 26:
                    expected = { b ? .5f : .125f, b ? 1.0f : .25f,
                                 b ? 2.0f : .5f };
                    break;
                case 29: expected = { 3.14159f, 0, 0 }; break;
                case 30: expected = { 0, 1, 1 }; break;
                case 32:
                    expected = { y == 0 ? weight : q, y == 0 ? 0.0f : sg.dudx,
                                 y == 0 ? 0.0f : 1.0f };
                    break;
                }
                for (int c = 0; c < 3; ++c) {
                    const float actual    = images[pass][3 * index + c];
                    const float cpu_value = reference[3 * index + c];
                    if (actual != expected[c] || cpu_value != expected[c]) {
                        diagnostics.errorfmt(
                            "Attribute pass {} pixel ({},{}) channel {}: "
                            "expected {}, GPU {}, CPU {}",
                            pass, x, y, c, expected[c], actual, cpu_value);
                        return false;
                    }
                }
            }
        }
        std::string current;
        const void* address = nullptr;
        if (!artifact(ss, *group, current, address, diagnostics))
            return false;
        if (pass == 0) {
            original         = current;
            original_address = address;
        } else {
            OIIO_CHECK_EQUAL(current, original);
            OIIO_CHECK_EQUAL(address, original_address);
        }
    }
    OIIO_CHECK_ASSERT(images[0] != images[1] && images[0] == images[2]);
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



bool
run_spaces(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    constexpr int columns = 33, rows = 3;
    OutputFiles files;
    if (!files.create(diagnostics))
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    const int optimize = mode == "unoptimized" ? 0 : 2;
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("optimize", optimize)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? 4096 : 0)
        || !ss.attribute("unknown_coordsys_error", 0))
        return false;
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(R"OSL(
shader hart_spaces(output color Cout=999999)
{
    int column=int(32*u+.5);
    point p=point(u+.25,2*v-.5,2+u+v);
    vector q=0;
    matrix M=1;
    int matrix_result=0, ok=1;
    if (column < 15) {
        string from=column < 3 ? "myspace" : column < 6 ? "common" :
                    column < 9 ? "object" : column < 12 ? "myspace" : "common";
        string to=column < 3 ? "common" : column < 9 ? "myspace" :
                  column < 12 ? "shader" :
                  v < .25 ? "camera" : v < .75 ? "screen" : "raster";
        if (column%3==0) q=vector(transform(from,to,p));
        else if (column%3==1) q=transform(from,to,vector(p));
        else q=vector(transform(from,to,normal(p)));
    } else if (column == 15) {
        ok=getmatrix("common","NDC",M); matrix_result=1;
    } else if (column == 16) {
        string missing=v < .25 ? "missing_a" : "missing_b";
        ok=getmatrix(missing,"myspace",M); matrix_result=1;
    } else if (column == 17) {
        ok=getmatrix("camera","common",M); matrix_result=1;
    } else if (column == 18) {
        q=vector(point("world",u+.25,2*v-.5,2+u+v));
    } else if (column == 19) {
        M=matrix("world",1); matrix_result=1;
    } else if (column == 20) {
        M=matrix("world","common"); matrix_result=1;
    } else if (column == 21) {
        ok=getmatrix("common","world",M); matrix_result=1;
    } else if (column == 22) {
        q=vector(transform("world","common",p));
    } else if (column == 23) {
        q=transform("common","world",vector(p));
    } else if (column == 24) {
        q=vector("world",u+.25,2*v-.5,2+u+v);
    } else if (column == 25) {
        q=vector(normal("world",u+.25,2*v-.5,2+u+v));
    } else if (column == 26) {
        string names[4]={"world","stage","myspace","common"};
        int index=int(2*v);
        q=vector(transform(names[index],names[index+1],normal(p)));
    } else if (column == 27) {
        q=vector(point("stage",u+.25,2*v-.5,2+u+v));
    } else if (column == 28) {
        M=matrix("stage",1); matrix_result=1;
    } else if (column == 29) {
        ok=getmatrix("camera","stage",M); matrix_result=1;
    } else if (column == 30) {
        string moving=u>v ? "myspace" : "stage";
        M=v < .5 ? matrix(moving,"$unknown1$") : matrix("$unknown2$",moving);
        matrix_result=1;
    } else if (column == 31) {
        ok=getmatrix("object","shader",M); matrix_result=1;
    } else {
        M=matrix("myspace",1,0,0,0,0,2,0,0,0,0,3,0,.5,.25,0,1);
        matrix_result=1;
    }
    if (matrix_result) {
        float sum=0, diagonal=0;
        for (int i=0;i<4;++i)
            for (int j=0;j<4;++j) {
                sum+=(4*i+j+1)*M[i][j];
                if (i==j) diagonal+=M[i][j];
            }
        Cout=color(sum,diagonal,ok);
    } else {
        Cout=color(v < .25 ? q : v < .75 ? Dx(q) : Dy(q));
    }
}
)OSL",
                                 oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_spaces", oso))
        return false;
    auto group = ss.ShaderGroupBegin("hart_spaces");
    if (!group || !ss.Shader("surface", "hart_spaces", "out")
        || !ss.ShaderGroupEnd())
        return false;
    const SymLocationDesc output("out.Cout", TypeColor, false,
                                 SymArena::Outputs, 0, 12);
    ss.add_symlocs(group.get(), { &output, 1 });
    Matrix44 object(1), shader(1);
    object.scale(Vec3(2, .5f, 4));
    object.translate(Vec3(.25f, -.5f, 1));
    shader.scale(Vec3(.5f, 2, 1));
    shader[0][1] = .25f;
    auto bind    = [&](SimpleRenderer& r, bool b) {
        Matrix44 named(1), world(1), stage(1), camera(1);
        named.scale(b ? Vec3(.5f, 2, -4) : Vec3(2, 4, .5f));
        named[0][1] = b ? -.5f : .25f;
        named.translate(b ? Vec3(-1, .5f, 2) : Vec3(.25f, 1, -.5f));
        world.scale(Vec3(4, 2, .5f));
        world.translate(Vec3(.5f, -.25f, 1));
        stage.scale(Vec3(2, .5f, 4));
        stage.translate(Vec3(-.5f, 1, .25f));
        camera.scale(b ? Vec3(.5f, 2, 1) : Vec3(2, 1, .5f));
        camera.translate(Vec3(.25f, -.5f, 2));
        r.name_transform("myspace", named);
        r.name_transform("world", world);
        r.name_transform("stage", stage);
        r.name_transform("$unknown1$", world);
        r.name_transform("$unknown2$", stage);
        Matrix44 unused(1);
        unused[0][0] = std::numeric_limits<float>::quiet_NaN();
        r.name_transform("unused_bad", unused);
        r.camera_params(camera, ustringhash(b ? "orthographic" : "perspective"),
                        b ? 60 : 90, b ? .5f : .125f, b ? 128 : 64,
                        b ? 300 : 640, b ? 600 : 320);
    };
    HartOptions options;
    options.fused = mode == "fused" || mode == "fused-local";
    auto render   = [&](string_view path) {
        return testshade_hart_generated(*renderer, ss, *group, options, arch,
                                        columns, rows, 1, false, true, 0, false,
                                        path, "float", object, shader);
    };
    auto read = [&](string_view path, std::vector<float>& pixels) {
        OIIO::ImageBuf image(path);
        if (!image.read(0, 0, true, TypeFloat) || image.nchannels() != 3
            || image.spec().width != columns || image.spec().height != rows) {
            diagnostics.errorfmt("Invalid named-space output: {}",
                                 image.geterror());
            return false;
        }
        pixels.resize(columns * rows * 3);
        if (!image.get_pixels(image.roi(), TypeFloat, pixels.data())) {
            diagnostics.errorfmt("Cannot read named-space output: {}",
                                 image.geterror());
            return false;
        }
        return true;
    };
    std::array<std::vector<float>, 3> images;
    std::string original;
    const void* original_address = nullptr;
    for (int pass = 0; pass < 3; ++pass) {
        const bool b = pass == 1;
        bind(*renderer, b);
        if (!ss.attribute("commonspace", b ? "stage" : "world"))
            return false;
        print("HART spaces pass {}\n", pass);
        std::fflush(stdout);
        if (!render(files.files[pass])
            || !read(files.files[pass], images[pass]))
            return false;
        SimpleRenderer cpu_renderer;
        bind(cpu_renderer, b);
        ShadingSystem cpu(&cpu_renderer, nullptr, &diagnostics);
        cpu_renderer.init_shadingsys(&cpu);
        if (!cpu.attribute("optimize", optimize)
            || !cpu.attribute("llvm_debugging_symbols", 0)
            || !cpu.attribute("commonspace", b ? "stage" : "world")
            || !cpu.attribute("unknown_coordsys_error", 0)
            || !cpu.LoadMemoryCompiledShader("hart_spaces", oso))
            return false;
        auto cpu_group = cpu.ShaderGroupBegin("hart_spaces");
        if (!cpu_group || !cpu.Shader("surface", "hart_spaces", "out")
            || !cpu.ShaderGroupEnd())
            return false;
        cpu.add_symlocs(cpu_group.get(), { &output, 1 });
        auto free_thread = [&](PerThreadInfo* p) {
            cpu.destroy_thread_info(p);
        };
        std::unique_ptr<PerThreadInfo, decltype(free_thread)> thread(
            cpu.create_thread_info(), free_thread);
        auto free_context = [&](ShadingContext* p) { cpu.release_context(p); };
        std::unique_ptr<ShadingContext, decltype(free_context)> context(
            cpu.get_context(thread.get()), free_context);
        std::vector<float> reference(columns * rows * 3, 999999);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < columns; ++x) {
                const int index = y * columns + x;
                ShaderGlobals sg { };
                sg.u             = float(x) / (columns - 1);
                sg.v             = float(y) / (rows - 1);
                sg.dudx          = 1.0f / (columns - 1);
                sg.dvdy          = 1.0f / (rows - 1);
                sg.time          = float(index);
                sg.object2common = &object;
                sg.shader2common = &shader;
                if (!cpu.execute(*context, *cpu_group, 0, index, sg, nullptr,
                                 reference.data()))
                    return false;
                if (x == 30) {
                    const ustringhash moving(sg.u > sg.v ? "myspace" : "stage");
                    Matrix44 from, to;
                    OIIO_CHECK_ASSERT(cpu_renderer.get_matrix(
                        nullptr, from,
                        y == 0 ? moving : ustringhash("$unknown2$"), 0));
                    OIIO_CHECK_ASSERT(cpu_renderer.get_inverse_matrix(
                        nullptr, to,
                        y == 0 ? ustringhash("$unknown1$") : moving, 0));
                    const Matrix44 composed = from * to;
                    float sum = 0, diagonal = 0;
                    for (int i = 0; i < 16; ++i) {
                        sum += (i + 1) * composed[i / 4][i % 4];
                        if (i / 4 == i % 4)
                            diagonal += composed[i / 4][i % 4];
                    }
                    OIIO_CHECK_EQUAL(reference[3 * index], sum);
                    OIIO_CHECK_EQUAL(reference[3 * index + 1], diagonal);
                }
                for (int c = 0; c < 3; ++c) {
                    const float actual   = images[pass][3 * index + c];
                    const float expected = reference[3 * index + c];
                    if (!std::isfinite(actual) || !std::isfinite(expected)
                        || actual == 999999 || expected == 999999
                        || std::abs(actual - expected)
                               > 2e-6f + 2e-6f * std::abs(expected)) {
                        diagnostics.errorfmt(
                            "Spaces pass {} pixel ({},{}) channel {}: GPU {} CPU {}",
                            pass, x, y, c, actual, expected);
                        return false;
                    }
                }
                if (x == 16 || x == 17 || x == 29)
                    OIIO_CHECK_EQUAL(images[pass][3 * index + 2], 0);
                if (x == 19 || x == 20 || x == 28 || x == 30 || x == 31
                    || x == 32 || x == 15 || x == 21)
                    OIIO_CHECK_EQUAL(images[pass][3 * index + 2], 1);
            }
        std::string current;
        const void* address = nullptr;
        if (!artifact(ss, *group, current, address, diagnostics))
            return false;
        if (pass == 0) {
            original         = current;
            original_address = address;
        } else {
            OIIO_CHECK_EQUAL(current, original);
            OIIO_CHECK_EQUAL(address, original_address);
        }
    }
    OIIO_CHECK_ASSERT(images[0] != images[1] && images[0] == images[2]);
    print("HART spaces expected unknown\n");
    OIIO_CHECK_ASSERT(ss.attribute("unknown_coordsys_error", 1));
    OIIO_CHECK_ASSERT(!render(files.files[3]));
    OIIO_CHECK_ASSERT(!OIIO::Filesystem::exists(files.files[3]));
    OIIO_CHECK_ASSERT(ss.attribute("unknown_coordsys_error", 0));
    if (!render(files.files[3]))
        return false;
    std::vector<float> recovered;
    if (!read(files.files[3], recovered))
        return false;
    OIIO_CHECK_ASSERT(recovered == images[0]);
    Matrix44 invalid(1);
    invalid[0][0] = std::numeric_limits<float>::quiet_NaN();
    renderer->name_transform("myspace", invalid);
    print("HART spaces expected invalid\n");
    OIIO_CHECK_ASSERT(!render(files.files[4]));
    OIIO_CHECK_ASSERT(!OIIO::Filesystem::exists(files.files[4]));
    bind(*renderer, false);
    if (!render(files.files[4]) || !read(files.files[4], recovered))
        return false;
    OIIO_CHECK_ASSERT(recovered == images[0]);
    std::string current;
    const void* address = nullptr;
    if (!artifact(ss, *group, current, address, diagnostics))
        return false;
    OIIO_CHECK_EQUAL(current, original);
    OIIO_CHECK_EQUAL(address, original_address);
    print("HART spaces error recovery passed\n");
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



bool
run_raytypes(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    OutputFiles files;
    if (!files.create(diagnostics))
        return false;
    std::string arch;
    auto renderer = testshade_hart_renderer(0, arch);
    if (!renderer)
        return false;
    renderer->errhandler().verbosity(ErrorHandler::VERBOSE);
    ShadingSystem ss(renderer.get(), nullptr, &diagnostics);
    renderer->init_shadingsys(&ss);
    if (!ss.attribute("hart_arch", arch)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("max_hart_groupdata_alloc",
                         mode == "fused-local" ? 1048576 : 0)
        || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
        || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2))
        return false;
    std::array<std::string, 32> storage;
    std::array<const char*, 32> names;
    for (size_t i = 0; i < names.size(); ++i) {
        storage[i] = fmtformat("ray{}", i);
        names[i]   = storage[i].c_str();
    }
    names[1] = names[2] = "duplicate";
    names[30]           = "";
    names[31]           = "high";
    if (!ss.attribute("raytypes", TypeDesc(TypeDesc::STRING, 32), names.data()))
        return false;
    const char* source
        = "shader hart_raytypes(output color Cout=0) {"
          "string name=v>.5 ? \"\" : u<.5 ? \"high\""
          ": u<.75 ? \"duplicate\" : \"unknown\";"
          "Cout=color(raytype(name),raytype(\"high\"),raytype(\"duplicate\")); }";
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(source, oso, { }, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_raytypes", oso))
        return false;
    auto group = ss.ShaderGroupBegin("hart_raytype_rebind");
    if (!group || !ss.Shader("surface", "hart_raytypes", "out")
        || !ss.ShaderGroupEnd())
        return false;
    const SymLocationDesc output("out.Cout", TypeColor, false,
                                 SymArena::Outputs, 0, 12);
    ss.add_symlocs(group.get(), { &output, 1 });
    ss.optimize_group(group.get(), nullptr);
    std::string original;
    const void* original_address = nullptr;
    if (!artifact(ss, *group, original, original_address, diagnostics))
        return false;
    HartOptions options;
    options.fused = mode == "fused" || mode == "fused-local";
    const Matrix44 identity;
    const int masks[] = { std::numeric_limits<int32_t>::min(), (1 << 30) | 4,
                          std::numeric_limits<int32_t>::min() };
    std::array<std::vector<float>, 3> images;
    for (int pass = 0; pass < 3; ++pass) {
        print("HART raytypes pass {}\n", pass);
        std::fflush(stdout);
        if (!testshade_hart_generated(*renderer, ss, *group, options, arch,
                                      width, height, 1, false, true,
                                      masks[pass], false, files.files[pass],
                                      "float", identity, identity))
            return false;
        OIIO::ImageBuf image(files.files[pass]);
        if (!image.read(0, 0, true, TypeFloat) || image.nchannels() != 3
            || image.spec().width != width || image.spec().height != height) {
            diagnostics.errorfmt("Invalid ray-type output: {}",
                                 image.geterror());
            return false;
        }
        images[pass].resize(width * height * 3);
        if (!image.get_pixels(image.roi(), TypeFloat, images[pass].data())) {
            diagnostics.errorfmt("Cannot read ray-type output: {}",
                                 image.geterror());
            return false;
        }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const int offset = 3 * (width * y + x);
                const float u    = float(x) / (width - 1);
                const float v    = float(y) / (height - 1);
                OIIO_CHECK_EQUAL(images[pass][offset],
                                 float(pass == 1 ? v > .5f
                                                 : u < .5f && v <= .5f));
                OIIO_CHECK_EQUAL(images[pass][offset + 1],
                                 pass == 1 ? 0.0f : 1.0f);
                OIIO_CHECK_EQUAL(images[pass][offset + 2], 0.0f);
            }
        std::string current;
        const void* address = nullptr;
        if (!artifact(ss, *group, current, address, diagnostics))
            return false;
        OIIO_CHECK_EQUAL(current, original);
        OIIO_CHECK_EQUAL(address, original_address);
    }
    OIIO_CHECK_ASSERT(images[0] != images[1] && images[0] == images[2]);
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



bool
run_lifecycle(string_view stdosl, string_view mode, Diagnostics& diagnostics)
{
    const Matrix44 identity(1);
    for (int cycle = 0; cycle < 3; ++cycle) {
        OutputFiles files;
        if (!files.create(diagnostics))
            return false;
        print("HART lifecycle cycle {}\n", cycle);
        std::fflush(stdout);
        // Groups must die before their systems, and systems before renderers.
        std::array<std::unique_ptr<SimpleRenderer>, 2> renderers;
        std::array<std::unique_ptr<ShadingSystem>, 2> systems;
        std::array<ShaderGroupRef, 2> groups;
        std::array<std::string, 2> arches, bitcode;
        std::array<const void*, 2> addresses { };
        std::array<void*, 2> interactive { };
        std::array<int, 2> factors { cycle == 1 ? 5 : 3, cycle == 1 ? 11 : 7 };
        for (int context = 0; context < 2; ++context) {
            renderers[context] = testshade_hart_renderer(0, arches[context]);
            if (!renderers[context])
                return false;
            auto& renderer = *renderers[context];
            renderer.errhandler().verbosity(ErrorHandler::VERBOSE);
            systems[context] = std::make_unique<ShadingSystem>(&renderer,
                                                               nullptr,
                                                               &diagnostics);
            auto& ss         = *systems[context];
            renderer.init_shadingsys(&ss);
            if (!ss.attribute("hart_arch", arches[context])
                || !ss.attribute("llvm_debugging_symbols", 0)
                || !ss.attribute("llvm_profiling_events", 0)
                || !ss.attribute("max_hart_groupdata_alloc",
                                 mode == "fused-local" ? 1048576 : 0)
                || !ss.attribute("llvm_optimize", mode == "unoptimized" ? 10 : 3)
                || !ss.attribute("optimize", mode == "unoptimized" ? 0 : 2))
                return false;
            const auto source
                = fmtformat("shader hart_lifecycle("
                            "int index=0 [[int interactive=1]],"
                            "float values[2]={{2,4}} [[int interactive=1]],"
                            "output color Cout=0) {{"
                            "Cout=color({}*values[index]+u, Dx(u), Dy(v)); }}",
                            factors[context]);
            OSLCompiler compiler(&diagnostics);
            std::string oso;
            if (!compiler.compile_buffer(source, oso, { }, stdosl)
                || !ss.LoadMemoryCompiledShader("hart_lifecycle", oso))
                return false;
            groups[context] = ss.ShaderGroupBegin("hart_lifecycle_group");
            if (!groups[context]
                || !ss.Shader("surface", "hart_lifecycle", "out")
                || !ss.ShaderGroupEnd())
                return false;
            const SymLocationDesc output("out.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(groups[context].get(), { &output, 1 });
            ss.optimize_group(groups[context].get(), nullptr);
            if (!artifact(ss, *groups[context], bitcode[context],
                          addresses[context], diagnostics)
                || !ss.getattribute(groups[context].get(),
                                    "device_interactive_params", TypeDesc::PTR,
                                    &interactive[context])
                || !interactive[context])
                return false;
        }
        OIIO_CHECK_ASSERT(interactive[0] != interactive[1]);
        OIIO_CHECK_ASSERT(bitcode[0] != bitcode[1]);
        HartOptions options;
        options.fused = mode == "fused" || mode == "fused-local";
        auto render   = [&](int context, int iterations, string_view filename) {
            return testshade_hart_generated(*renderers[context],
                                            *systems[context], *groups[context],
                                            options, arches[context], width,
                                            height, iterations, false, true, 0,
                                            false, filename, "float", identity,
                                            identity);
        };
        auto check = [&](int context, float value, string_view filename) {
            OIIO::ImageBuf image(filename);
            if (!image.read(0, 0, true, TypeFloat)
                || image.spec().width != width || image.spec().height != height
                || image.nchannels() != 3) {
                diagnostics.errorfmt("Invalid lifecycle output: {}",
                                     image.geterror());
                return false;
            }
            std::array<float, width * height * 3> pixels;
            if (!image.get_pixels(image.roi(), TypeFloat, pixels.data())) {
                diagnostics.errorfmt("Cannot read lifecycle output: {}",
                                     image.geterror());
                return false;
            }
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const size_t offset = 3 * (width * y + x);
                    OIIO_CHECK_EQUAL(pixels[offset],
                                     factors[context] * value
                                         + float(x) / (width - 1));
                    OIIO_CHECK_EQUAL(pixels[offset + 1], 1.0f / (width - 1));
                    OIIO_CHECK_EQUAL(pixels[offset + 2], 1.0f / (height - 1));
                }
            return true;
        };
        print("HART lifecycle initial {}\n", cycle);
        if (!render(0, 1, files.files[0]) || !check(0, 2, files.files[0]))
            return false;
        if (!systems[0]->ReParameter(*groups[0], "out", "index", 2))
            return false;
        print("HART lifecycle bounded failure {}\n", cycle);
        OIIO_CHECK_ASSERT(!render(0, 1, files.files[2]));
        OIIO_CHECK_ASSERT(!OIIO::Filesystem::exists(files.files[2]));
        // A's failure must not poison B, even with identical shader/group names.
        print("HART lifecycle independent {}\n", cycle);
        if (!systems[1]->ReParameter(*groups[1], "out", "index", 1)
            || !render(1, 1, files.files[1]) || !check(1, 4, files.files[1]))
            return false;
        if (!systems[0]->ReParameter(*groups[0], "out", "index", 0))
            return false;
        int updates               = 0;
        options.update_parameters = [&]() {
            ++updates;
            return false;
        };
        print("HART lifecycle update failure {}\n", cycle);
        OIIO_CHECK_ASSERT(!render(0, 2, files.files[3]));
        OIIO_CHECK_EQUAL(updates, 1);
        OIIO_CHECK_ASSERT(!OIIO::Filesystem::exists(files.files[3]));
        options.update_parameters = { };
        const float values[]      = { 13, 17 };
        print("HART lifecycle recovered {}\n", cycle);
        if (!systems[0]->ReParameter(*groups[0], "out", "values",
                                     TypeDesc(TypeDesc::FLOAT, 2), values)
            || !render(0, 1, files.files[4]) || !check(0, 13, files.files[4]))
            return false;
        for (int context = 0; context < 2; ++context) {
            std::string current;
            const void* address = nullptr;
            void* arena         = nullptr;
            if (!artifact(*systems[context], *groups[context], current, address,
                          diagnostics)
                || !systems[context]->getattribute(groups[context].get(),
                                                   "device_interactive_params",
                                                   TypeDesc::PTR, &arena))
                return false;
            OIIO_CHECK_EQUAL(current, bitcode[context]);
            OIIO_CHECK_EQUAL(address, addresses[context]);
            OIIO_CHECK_EQUAL(arena, interactive[context]);
        }
        groups    = { };
        systems   = { };
        renderers = { };
        print("HART lifecycle destroyed {}\n", cycle);
        std::fflush(stdout);
    }
    return diagnostics.errors == 0 && diagnostics.warnings == 0;
}



}  // namespace



int
main(int argc, char* argv[])
{
    const bool color       = argc == 4 && string_view(argv[3]) == "color";
    const bool outputs     = argc == 4 && string_view(argv[3]) == "outputs";
    const bool interactive = argc == 4 && string_view(argv[3]) == "interactive";
    const bool userdata    = argc == 4 && string_view(argv[3]) == "userdata";
    const bool raytypes    = argc == 4 && string_view(argv[3]) == "raytypes";
    const bool attributes  = argc == 4 && string_view(argv[3]) == "attributes";
    const bool spaces      = argc == 4 && string_view(argv[3]) == "spaces";
    const bool lifecycle   = argc == 4 && string_view(argv[3]) == "lifecycle";
    const bool library     = argc == 5 && string_view(argv[3]) == "library";
    const string_view mode(argc >= 3 ? argv[2] : "split");
    if ((argc != 2 && argc != 3 && !color && !outputs && !interactive
         && !userdata && !raytypes && !library && !attributes && !spaces
         && !lifecycle)
        || (mode != "split" && mode != "fused" && mode != "fused-local"
            && !((color || outputs || interactive || userdata || raytypes
                  || library || attributes || spaces || lifecycle)
                 && mode == "unoptimized"))) {
        print(
            stderr,
            "Usage: hart_transform_test stdosl.h "
            "[split|fused|fused-local|unoptimized] "
            "[color|outputs|interactive|userdata|raytypes|attributes|spaces|lifecycle|library BITCODE_DIR]\n");
        return 1;
    }
    Diagnostics diagnostics;
    OIIO_CHECK_ASSERT(
        lifecycle    ? run_lifecycle(argv[1], mode, diagnostics)
        : spaces     ? run_spaces(argv[1], mode, diagnostics)
        : attributes ? run_attributes(argv[1], mode, diagnostics)
        : library    ? run_renderer_library(argv[1], mode, argv[4], diagnostics)
        : raytypes   ? run_raytypes(argv[1], mode, diagnostics)
        : userdata   ? run_userdata(argv[1], mode, diagnostics)
        : outputs    ? run_outputs(argv[1], mode, diagnostics)
                     : run(argv[1], mode, color, interactive, diagnostics));
    OIIO_CHECK_EQUAL(diagnostics.errors, 0);
    OIIO_CHECK_EQUAL(diagnostics.warnings, 0);
    return unit_test_failures;
}
