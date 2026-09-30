// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HIP must precede OIIO's __has_attribute fallback on MSVC.
#include <hip/hip_runtime.h>

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#include <OpenImageIO/unittest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "../testrender/hartcontext.h"
#include "hart_texture_colorspace_bitcode.h"
#include "hart_texture_colorspace_params.h"

#if !OSL_ARNOLD_COMPAT
#    error This fixture requires the Arnold texture colorspace ABI
#endif

using namespace OSL;
using namespace texture_colorspace_test;

namespace {

static_assert(value_count == 108 && slots == 9);

const char* source = R"OSL(
void store_sample(int slot, color c, float a, output float values[108])
{
    int base = slot * 12;
    color dx = Dx(c), dy = Dy(c);
    for (int channel = 0; channel < 3; ++channel) {
        values[base + channel] = c[channel];
        values[base + 4 + channel] = dx[channel];
        values[base + 7 + channel] = dy[channel];
    }
    values[base + 3] = a;
    values[base + 10] = Dx(a);
    values[base + 11] = Dy(a);
}

shader hart_texture_colorspace(
    string filename = "unbound-file",
    string bound_space = "raw",
    string live_space = "sRGB" [[int interactive=1]],
    output float values[108] = {0})
{
    float a = -1;
    color c = texture(filename, u, v, "interp", "closest",
                      "wrap", "clamp", "alpha", a);
    store_sample(0, c, a, values);
    c = texture(filename, 1+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", "", "alpha", a);
    store_sample(1, c, a, values);
    c = texture(filename, 2+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", "raw", "alpha", a);
    store_sample(2, c, a, values);
    c = texture(filename, 3+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", "sRGB", "alpha", a);
    store_sample(3, c, a, values);
    c = texture(filename, 4+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", bound_space, "alpha", a);
    store_sample(4, c, a, values);
    string selected = u < 0.5 ? "raw" : "sRGB";
    c = texture(filename, 5+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", selected, "alpha", a);
    store_sample(5, c, a, values);
    c = texture(filename, 6+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", live_space, "alpha", a);
    store_sample(6, c, a, values);
    c = texture(filename, 7+u, v, "interp", "closest", "wrap", "clamp",
                "colorspace", "invalid-source-space", "alpha", a);
    store_sample(7, c, a, values);
    float only_a = -1;
    color only_c = texture(filename, 8+u, v, "interp", "closest",
                           "wrap", "clamp", "colorspace", "sRGB",
                           "alpha", only_a);
    // Derivative demand from alpha alone must still reach the sampler.
    values[96] = only_c[0]; values[97] = only_c[1]; values[98] = only_c[2];
    values[99] = only_a;
    values[106] = Dx(only_a); values[107] = Dy(only_a);
}
)OSL";



class Diagnostics final : public ErrorHandler {
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
    unsigned errors = 0;
    std::string last_error;
};



bool
hip_check(Diagnostics& diagnostics, hipError_t status, string_view operation)
{
    if (status == hipSuccess)
        return true;
    diagnostics.errorfmt("{} failed: {} ({})", operation,
                         hipGetErrorName(status), hipGetErrorString(status));
    return false;
}



class FixedTexture {
public:
    explicit FixedTexture(Diagnostics& diagnostics) : m_diagnostics(diagnostics)
    {
    }
    ~FixedTexture() { clear(); }

    bool create()
    {
        const auto channels = hipCreateChannelDesc(32, 32, 32, 32,
                                                   hipChannelFormatKindFloat);
        const float pixel[] = { 0.5f, 0.5f, 0.5f, 0.75f };
        if (!hip_check(m_diagnostics, hipMallocArray(&m_array, &channels, 1, 1),
                       "hipMallocArray")
            || !hip_check(m_diagnostics,
                          hipMemcpy2DToArray(m_array, 0, 0, pixel,
                                             sizeof(pixel), sizeof(pixel), 1,
                                             hipMemcpyHostToDevice),
                          "hipMemcpy2DToArray"))
            return false;
        hipResourceDesc resource {};
        resource.resType         = hipResourceTypeArray;
        resource.res.array.array = m_array;
        hipTextureDesc sampler {};
        sampler.addressMode[0] = sampler.addressMode[1] = hipAddressModeClamp;
        sampler.filterMode                              = hipFilterModePoint;
        sampler.readMode         = hipReadModeElementType;
        sampler.normalizedCoords = 1;
        return hip_check(m_diagnostics,
                         hipCreateTextureObject(&m_object, &resource, &sampler,
                                                nullptr),
                         "hipCreateTextureObject");
    }
    bool clear()
    {
        if (m_object) {
            if (!hip_check(m_diagnostics, hipDestroyTextureObject(m_object),
                           "hipDestroyTextureObject"))
                return false;
            m_object = {};
        }
        if (m_array) {
            if (!hip_check(m_diagnostics, hipFreeArray(m_array), "hipFreeArray"))
                return false;
            m_array = nullptr;
        }
        return true;
    }
    uint64_t object() const { return uint64_t(uintptr_t(m_object)); }

private:
    Diagnostics& m_diagnostics;
    hipArray_t m_array = nullptr;
    hipTextureObject_t m_object {};
};



class Services final : public RendererServices {
public:
    Services(Diagnostics& diagnostics, bool textures = true,
             bool colorspaces = true)
        : m_diagnostics(diagnostics)
        , m_textures(textures)
        , m_colorspaces(colorspaces)
    {
    }
    ~Services() override
    {
        for (void* ptr : m_allocations)
            hip_check(m_diagnostics, hipFree(ptr), "hipFree remaining binding");
    }

    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTInteractive"
               || feature == "HARTArrayBounds"
               || (m_textures && feature == "HARTTextures")
               || (m_colorspaces && feature == "HARTTextureColorSpaces");
    }
    TextureHandle* get_texture_handle(ustring filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        ++handle_calls;
        return filename == ustring("hart-colorspace-fixed")
                   ? reinterpret_cast<TextureHandle*>(uintptr_t(texture_id))
                   : nullptr;
    }
    bool good(TextureHandle* handle) override
    {
        return reinterpret_cast<uintptr_t>(handle) == texture_id;
    }
    bool texture(ustringhash, TextureHandle*, TexturePerthread*, TextureOpt&,
                 ShaderGlobals*, float, float, float, float, float, float, int,
                 float*, float*, float*, ustringhash*) override
    {
        ++host_sampler_calls;
        return false;
    }
    bool texture(ustringhash, ustringhash, TextureHandle*, TexturePerthread*,
                 TextureOpt&, ShaderGlobals*, float, float, float, float, float,
                 float, int, float*, float*, float*, ustringhash*) override
    {
        ++host_sampler_calls;
        return false;
    }
    void* device_alloc(size_t size) override
    {
        void* ptr = nullptr;
        if (!hip_check(m_diagnostics, hipMalloc(&ptr, size),
                       "hipMalloc interactive"))
            return nullptr;
        m_allocations.push_back(ptr);
        return ptr;
    }
    void device_free(void* ptr) override
    {
        if (hip_check(m_diagnostics, hipFree(ptr), "hipFree interactive")) {
            const auto found = std::find(m_allocations.begin(),
                                         m_allocations.end(), ptr);
            if (found != m_allocations.end())
                m_allocations.erase(found);
        }
    }
    void* copy_to_device(void* dst, const void* src, size_t size) override
    {
        return hip_check(m_diagnostics,
                         hipMemcpy(dst, src, size, hipMemcpyHostToDevice),
                         "hipMemcpy interactive")
                   ? dst
                   : nullptr;
    }
    size_t allocations() const { return m_allocations.size(); }

    unsigned host_sampler_calls = 0, handle_calls = 0;

private:
    Diagnostics& m_diagnostics;
    bool m_textures, m_colorspaces;
    std::vector<void*> m_allocations;
};



ShaderGroupRef
make_group(ShadingSystem& ss, Diagnostics& diagnostics, string_view stdosl,
           string_view arch, int optimize)
{
    if (!ss.attribute("hart_arch", arch) || !ss.attribute("optimize", optimize)
        || !ss.attribute("llvm_optimize", optimize)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("llvm_profiling_events", 0)
        || !ss.attribute("profile", 0))
        return {};
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(source, oso, {}, stdosl)
        || !ss.LoadMemoryCompiledShader("hart_texture_colorspace", oso))
        return {};
    auto group = ss.ShaderGroupBegin("hart_texture_colorspace");
    if (!group
        || !ss.Parameter(*group, "filename", ustring("hart-colorspace-fixed"),
                         ParamHints::none)
        || !ss.Parameter(*group, "bound_space", ustring("sRGB"),
                         ParamHints::none)
        || !ss.Parameter(*group, "live_space", ustring("raw"),
                         ParamHints::interactive)
        || !ss.Shader("surface", "hart_texture_colorspace", "material")
        || !ss.ShaderGroupEnd())
        return {};
    const SymLocationDesc output("material.values",
                                 TypeDesc(TypeDesc::FLOAT, value_count), false,
                                 SymArena::Outputs, offsetof(Output, values),
                                 sizeof(Output));
    ss.add_symlocs(group.get(), { &output, 1 });
    ss.optimize_group(group.get(), nullptr);
    return group;
}



void
check_output(const Output& output, const Result& result, unsigned point,
             string_view live_space)
{
    // The oracle uses host double precision, independently of device powf.
    const double srgb_linear     = std::pow((0.5 + 0.055) / 1.055, 2.4);
    const uint64_t raw_hash      = ustringhash("raw").hash();
    const uint64_t srgb_hash     = ustringhash("sRGB").hash();
    const uint64_t hashes[slots] = { 0,
                                     0,
                                     raw_hash,
                                     srgb_hash,
                                     srgb_hash,
                                     point ? srgb_hash : raw_hash,
                                     ustringhash(live_space).hash(),
                                     ustringhash("invalid-source-space").hash(),
                                     srgb_hash };
    OIIO_CHECK_EQUAL(output.head, guard);
    OIIO_CHECK_EQUAL(output.tail, guard);
    OIIO_CHECK_EQUAL(result.done, completed);
    OIIO_CHECK_EQUAL(result.errors, invalid_source);
    for (unsigned slot = 0; slot < slots; ++slot) {
        const bool failed     = slot == invalid;
        const double expected = failed                      ? 0.0
                                : hashes[slot] == srgb_hash ? srgb_linear
                                                            : 0.5;
        OIIO_CHECK_EQUAL(result.calls[slot], 1u);
        OIIO_CHECK_EQUAL(result.gradients[slot], 3u);
        OIIO_CHECK_EQUAL(result.colorspaces[slot], hashes[slot]);
        OIIO_CHECK_EQUAL(result.filenames[slot],
                         ustringhash("hart-colorspace-fixed").hash());
        OIIO_CHECK_EQUAL(result.handles[slot], texture_id);
        OIIO_CHECK_EQUAL(
            result.diagnostics[slot],
            failed ? ustringhash("HART texture test: unknown source colorspace")
                         .hash()
                   : uint64_t(0));
        const float* values = output.values + slot * values_per_slot;
        for (unsigned c = 0; c < 3; ++c) {
            OIIO_CHECK_ASSERT(std::isfinite(values[c]));
            OIIO_CHECK_ASSERT(std::abs(double(values[c]) - expected) < 2e-6);
        }
        OIIO_CHECK_EQUAL(values[3], failed ? 0.0f : 0.75f);
        for (unsigned c = 4; c < values_per_slot; ++c)
            OIIO_CHECK_EQUAL(values[c], 0.0f);
    }
}



bool
run(string_view stdosl, int optimize)
{
    Diagnostics diagnostics;
    HartContext context(diagnostics);
    std::string arch;
    if (!context.init(0, arch, true, true))
        return false;
    cspan<unsigned char> raygen;
    for (const auto& module : hart_texture_colorspace_modules)
        if (module.arch && arch == module.arch && *module.size > 0)
            raygen = { module.data, size_t(*module.size) };
    if (raygen.empty()) {
        diagnostics.errorfmt("No texture colorspace test module for {}", arch);
        return false;
    }
    for (bool textures : { false, true }) {
        Services unsupported(diagnostics, textures, !textures);
        ShadingSystem ss(&unsupported, nullptr, &diagnostics);
        const unsigned before = diagnostics.errors;
        const auto group = make_group(ss, diagnostics, stdosl, arch, optimize);
        OIIO_CHECK_ASSERT(group);
        const void* bitcode = nullptr;
        OIIO_CHECK_ASSERT(group
                          && !ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR, &bitcode));
        OIIO_CHECK_ASSERT(!bitcode);
        uint64_t bitcode_size = 0;
        OIIO_CHECK_ASSERT(group
                          && (!ss.getattribute(group.get(), "hart_bitcode_size",
                                               TypeUInt64, &bitcode_size)
                              || bitcode_size == 0));
        OIIO_CHECK_ASSERT(diagnostics.errors > before);
        OIIO_CHECK_ASSERT(
            diagnostics.last_error.find(textures ? "HARTTextureColorSpaces"
                                                 : "HARTTextures")
            != std::string::npos);
        OIIO_CHECK_EQUAL(unsupported.host_sampler_calls, 0u);
        OIIO_CHECK_EQUAL(context.statistics().launches, size_t(0));
    }
    const unsigned expected_errors = diagnostics.errors;
    FixedTexture texture(diagnostics);
    if (!texture.create())
        return false;
    Services services(diagnostics);
    {
        ShadingSystem ss(&services, nullptr, &diagnostics);
        const auto group = make_group(ss, diagnostics, stdosl, arch, optimize);
        if (!group || diagnostics.errors != expected_errors)
            return false;
        const void* data = nullptr;
        uint64_t bytes   = 0;
        int group_size = 0, alignment = 0;
        HartCallable callable;
        callable.entries.resize(2);
        Params params {};
        if (!ss.getattribute(group.get(), "hart_bitcode", TypeDesc::PTR, &data)
            || !ss.getattribute(group.get(), "hart_bitcode_size", TypeUInt64,
                                &bytes)
            || !data || !bytes
            || !ss.getattribute(group.get(), "llvm_groupdata_size", group_size)
            || !ss.getattribute(group.get(), "llvm_groupdata_alignment",
                                alignment)
            || group_size < 0 || alignment <= 0 || alignment > 256
            || (alignment & (alignment - 1))
            || !ss.getattribute(group.get(), "group_init_name",
                                callable.entries[0])
            || !ss.getattribute(group.get(), "group_entry_name",
                                callable.entries[1])
            || !ss.getattribute(group.get(), "device_interactive_params",
                                TypeDesc::PTR, &params.interactive)
            || !params.interactive)
            return false;
        callable.bitcode = { static_cast<const unsigned char*>(data),
                             size_t(bytes) };
        const std::vector<unsigned char> original(callable.bitcode.begin(),
                                                  callable.bitcode.end());
        if (!context.build_accel({}, {}, {}, 1)
            || !context.create_pipeline(raygen, "__raygen__texture_colorspaces",
                                        1, { &callable, 1 }))
            return false;
        params.group_stride = (unsigned(group_size) + 255) & ~255u;
        if (!params.group_stride)
            params.group_stride = 256;
        params.groupdata = static_cast<unsigned char*>(
            context.alloc(params.group_stride * points));
        params.outputs = static_cast<Output*>(
            context.alloc(sizeof(Output) * points));
        params.results = static_cast<Result*>(
            context.alloc(sizeof(Result) * points));
        params.texture = texture.object();
        if (!params.groupdata || !params.outputs || !params.results)
            return false;
        const char* bindings[] = { "raw", "sRGB", "", "raw" };
        for (unsigned pass = 0; pass < 4; ++pass) {
            if (pass
                && !ss.ReParameter(*group, "material", "live_space",
                                   ustring(bindings[pass])))
                return false;
            void* interactive_params = nullptr;
            const void* current      = nullptr;
            uint64_t current_bytes   = 0;
            if (!ss.getattribute(group.get(), "device_interactive_params",
                                 TypeDesc::PTR, &interactive_params)
                || !ss.getattribute(group.get(), "hart_bitcode", TypeDesc::PTR,
                                    &current)
                || !ss.getattribute(group.get(), "hart_bitcode_size",
                                    TypeUInt64, &current_bytes))
                return false;
            OIIO_CHECK_EQUAL(interactive_params, params.interactive);
            OIIO_CHECK_EQUAL(current, data);
            OIIO_CHECK_EQUAL(current_bytes, bytes);
            if (current != data || current_bytes != bytes
                || std::memcmp(current, original.data(), original.size()))
                return false;
            std::array<Output, points> outputs;
            for (auto& output : outputs) {
                output.head = output.tail = guard;
                std::fill(std::begin(output.values), std::end(output.values),
                          std::numeric_limits<float>::quiet_NaN());
            }
            std::array<Result, points> results {};
            if (!context.upload(params.outputs,
                                { reinterpret_cast<const unsigned char*>(
                                      outputs.data()),
                                  sizeof(outputs) })
                || !context.upload(params.results,
                                   { reinterpret_cast<const unsigned char*>(
                                         results.data()),
                                     sizeof(results) })
                || !context.launch(&params, sizeof(params), points, 1)
                || !context.download({ reinterpret_cast<unsigned char*>(
                                           outputs.data()),
                                       sizeof(outputs) },
                                     params.outputs)
                || !context.download({ reinterpret_cast<unsigned char*>(
                                           results.data()),
                                       sizeof(results) },
                                     params.results))
                return false;
            for (unsigned point = 0; point < points; ++point)
                check_output(outputs[point], results[point], point,
                             bindings[pass]);
            print("HART texture colorspace O{} pass {}: live='{}' points={} "
                  "raw={} sRGB={} alpha={} invalid-source-diagnostics={}\n",
                  optimize, pass, bindings[pass], points,
                  outputs[0].values[raw * values_per_slot],
                  outputs[0].values[srgb * values_per_slot],
                  outputs[0].values[srgb * values_per_slot + 3], points);
        }
        OIIO_CHECK_ASSERT(services.handle_calls > 0);
        OIIO_CHECK_EQUAL(services.host_sampler_calls, 0u);
        OIIO_CHECK_EQUAL(context.statistics().launches, size_t(4));
    }
    OIIO_CHECK_EQUAL(services.allocations(), size_t(0));
    OIIO_CHECK_ASSERT(texture.clear());
    OIIO_CHECK_ASSERT(context.clear());
    OIIO_CHECK_EQUAL(diagnostics.errors, expected_errors);
    print("HART texture colorspace acceptance: arch={} O{} launches=4 "
          "host-sampler-calls=0 unchanged-bitcode=true arnold-profile=true\n",
          arch, optimize);
    return true;
}

}  // namespace



int
main(int argc, char* argv[])
{
    if (argc != 3 || (std::strcmp(argv[2], "0") && std::strcmp(argv[2], "2"))) {
        print(stderr, "Usage: hart_texture_colorspace_test stdosl.h 0|2\n");
        return 1;
    }
    OIIO_CHECK_ASSERT(run(argv[1], argv[2][0] - '0'));
    return unit_test_failures;
}
