// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "hartraytracer.h"

#include "../testshade/harttexture.h"
#include "hart_pathtracer_bitcode.h"
#include "hartcontext.h"
#include "hartparams.h"
#include "hartpathparams.h"

#include <OpenImageIO/timer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

OSL_NAMESPACE_BEGIN
namespace {

bool
finite_vec(const Vec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y)
           && std::isfinite(value.z);
}



bool
valid_indices(const TriangleIndices& indices, size_t count, bool optional)
{
    if (optional && indices.a == -1 && indices.b == -1 && indices.c == -1)
        return true;
    return indices.a >= 0 && size_t(indices.a) < count && indices.b >= 0
           && size_t(indices.b) < count && indices.c >= 0
           && size_t(indices.c) < count;
}



template<typename T>
cspan<unsigned char>
as_bytes(cspan<T> values)
{
    return { reinterpret_cast<const unsigned char*>(values.data()),
             values.size() * sizeof(T) };
}

}  // namespace



struct HartRaytracer::Impl {
    explicit Impl(OIIO::ErrorHandler& err)
        : m_err(err), m_textures(err), m_context(err)
    {
    }

    bool fail(string_view message)
    {
        m_err.errorfmt("{}", message);
        m_failed = true;
        return false;
    }

    bool render_options(HartRaytracer& renderer)
    {
        const int aa      = renderer.options.get_int("aa", 1);
        const int bounces = renderer.options.get_int("max_bounces", 4);
        if (aa < 1 || aa > 64 || bounces < 0 || bounces > 64)
            return fail("HART requires aa in [1,64] and max_bounces in [0,64]");
        if (renderer.options.get_int("show_globals") != 0
            || renderer.options.get_float("show_albedo_scale") != 0.0f)
            return fail("HART does not support show_globals or albedo display");
        const int background = renderer.getBackgroundShaderID();
        if (background < -1
            || (background >= 0
                && size_t(background) >= renderer.shaders().size()))
            return fail("Invalid HART background material");
        int resolution = background >= 0 ? renderer.getBackgroundResolution()
                                         : 0;
        resolution     = resolution > 0 ? std::max(32, resolution) : 0;
        if (resolution
            && resolution > std::numeric_limits<int>::max() / resolution)
            return fail("HART background table exceeds INT_MAX texels");
        for (const auto& material : renderer.shaders())
            if (material.disp)
                return fail("HART does not support displacement shaders");
        m_params.aa          = unsigned(aa);
        m_params.max_bounces = unsigned(bounces);
        m_params.no_jitter   = renderer.options.get_int("no_jitter") != 0;
        m_params.fused       = m_fused;
        m_params.rr_depth    = renderer.options.get_int("rr_depth", 5);
        m_params.background_material   = background;
        m_params.background_resolution = unsigned(resolution);
        m_params.background_offset     = 0;
        return true;
    }

    bool configure_camera(const Camera& camera, int width, int height)
    {
        if (width <= 0 || height <= 0
            || width > std::numeric_limits<int>::max() / height)
            return fail("HART requires positive image dimensions with at "
                        "most INT_MAX pixels");
        if (!finite_vec(camera.eye) || !finite_vec(camera.dir)
            || !finite_vec(camera.up) || !std::isfinite(camera.fov)
            || camera.fov <= 0 || camera.fov >= 180
            || !std::isfinite(camera.dir.length2())
            || camera.dir.length2() == 0)
            return fail("Invalid HART camera");
        // Copy the specified camera values, then recompute the derived fields.
        // A default Camera does not initialize its Imath vector members.
        auto& view = m_params.camera;
        view.eye   = camera.eye;
        view.dir   = camera.dir;
        view.up    = camera.up;
        view.fov   = camera.fov;
        view.resolution(width, height);
        if (!finite_vec(view.cx) || !finite_vec(view.cy)
            || !std::isfinite(view.cx.length2())
            || !std::isfinite(view.cy.length2()) || view.cx.length2() == 0
            || view.cy.length2() == 0)
            return fail("HART camera has invalid or parallel view/up vectors");
        return true;
    }

    bool geometry(HartRaytracer& renderer, std::vector<HartTriangle>& triangles,
                  std::vector<unsigned>& material_ids,
                  std::vector<float>& surfaceareas)
    {
        const auto& scene        = renderer.scene;
        const size_t count       = scene.triangles.size();
        const size_t index_limit = std::numeric_limits<int>::max();
        if (count > index_limit || scene.verts.size() > index_limit
            || scene.normals.size() > index_limit
            || scene.uvs.size() > index_limit
            || scene.n_triangles.size() != count
            || scene.uv_triangles.size() != count
            || scene.shaderids.size() != count
            || renderer.shaders().size()
                   > (std::numeric_limits<unsigned>::max() - 3)
                         / (m_fused ? 1 : 2))
            return fail("Invalid HART mesh array sizes or material count");
        for (const auto& material : renderer.shaders())
            if (!material.surf)
                return fail("HART material is missing a surface shader");
        for (const auto& vertex : scene.verts)
            if (!finite_vec(vertex))
                return fail("HART mesh vertex is not finite");
        for (const auto& normal : scene.normals)
            if (!finite_vec(normal))
                return fail("HART mesh normal is not finite");
        for (const auto& uv : scene.uvs)
            if (!std::isfinite(uv.x) || !std::isfinite(uv.y))
                return fail("HART mesh UV is not finite");
        for (size_t i = 0; i < count; ++i) {
            if (!valid_indices(scene.triangles[i], scene.verts.size(), false))
                return fail("HART mesh vertex index is out of range");
            if (!valid_indices(scene.n_triangles[i], scene.normals.size(), true))
                return fail(
                    "HART mesh normal index is out of range or partial");
            if (!valid_indices(scene.uv_triangles[i], scene.uvs.size(), true))
                return fail("HART mesh UV index is out of range or partial");
            if (scene.shaderids[i] < 0
                || size_t(scene.shaderids[i]) >= renderer.shaders().size())
                return fail("HART mesh material index is out of range");
        }
        size_t first = 0;
        for (int last : scene.last_index) {
            if (last < 0 || size_t(last) < first || size_t(last) > count)
                return fail(
                    "HART mesh boundaries are out of range or unordered");
            first = size_t(last);
        }
        if (first != count)
            return fail("HART mesh boundaries do not cover every triangle");
        surfaceareas.resize(count);
        first = 0;
        for (int last : scene.last_index) {
            float area = 0;
            for (size_t i = first; i < size_t(last); ++i)
                area += scene.primitivearea(int(i));
            if (!std::isfinite(area))
                return fail("HART mesh surface area is not finite");
            std::fill(surfaceareas.begin() + first,
                      surfaceareas.begin() + size_t(last), area);
            first = size_t(last);
        }
        triangles.reserve(count);
        material_ids.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& triangle = scene.triangles[i];
            triangles.push_back({ unsigned(triangle.a), unsigned(triangle.b),
                                  unsigned(triangle.c) });
            material_ids.push_back(unsigned(scene.shaderids[i]));
        }
        return true;
    }

    bool compile_materials(HartRaytracer& renderer,
                           std::vector<HartCallable>& callables,
                           std::vector<HartMaterialBinding>& bindings)
    {
        auto& ss = *renderer.shadingsys;
        if (!ss.attribute("hart_arch", m_arch)
            || !ss.attribute("max_hart_groupdata_alloc", int(m_local_budget)))
            return fail("Cannot configure HART shader compilation");
        size_t max_size        = 1;
        size_t max_alignment   = 1;
        bool all_local         = true;
        unsigned next_callable = 0;
        std::unordered_map<ShaderGroup*, HartMaterialBinding> compiled;
        for (const auto& material : renderer.shaders()) {
            auto* group      = material.surf.get();
            const auto found = compiled.find(group);
            if (found != compiled.end()) {
                bindings.push_back(found->second);
                continue;
            }
            int outputs = 0;
            if (!ss.getattribute(group, "num_renderer_outputs", outputs)
                || outputs != 0)
                return fail("HART path tracing does not support user outputs");
            // XML groups may share names, but their exported functions are
            // linked into one device module by HART.
            std::string name;
            if (!ss.getattribute(group, "groupname", name)
                || !ss.attribute(group, "groupname",
                                 fmtformat("hart_material_{}_{}",
                                           compiled.size(), name)))
                return fail("Cannot assign a unique HART material name");
            ss.optimize_group(group, nullptr);
            if (m_failed)
                return false;
            const void* data  = nullptr;
            uint64_t bytes    = 0;
            void* interactive = nullptr;
            int size = 0, alignment = 0, local = 0;
            HartCallable callable;
            callable.entries.resize(m_fused ? 1 : 2);
            if (!ss.getattribute(group, "hart_bitcode", TypeDesc::PTR, &data)
                || !ss.getattribute(group, "hart_bitcode_size", TypeUInt64,
                                    &bytes)
                || !data || !bytes || bytes > std::numeric_limits<size_t>::max()
                || !ss.getattribute(group, "llvm_groupdata_size", size)
                || !ss.getattribute(group, "llvm_groupdata_alignment", alignment)
                || !ss.getattribute(group, "hart_groupdata_alloc", local)
                || !ss.getattribute(group, "device_interactive_params",
                                    TypeDesc::PTR, &interactive)
                || size < 0 || alignment <= 0 || (alignment & (alignment - 1))
                || local < 0 || (local && local != size)
                || !ss.getattribute(group,
                                    m_fused ? "group_fused_name"
                                            : "group_init_name",
                                    callable.entries[0])
                || (!m_fused
                    && !ss.getattribute(group, "group_entry_name",
                                        callable.entries[1])))
                return fail("Cannot retrieve a compiled HART material; CPU "
                            "fallback is not supported");
            callable.bitcode = { static_cast<const unsigned char*>(data),
                                 size_t(bytes) };
            const HartMaterialBinding binding { next_callable,
                                                unsigned(m_fused && local),
                                                interactive };
            next_callable += unsigned(callable.entries.size());
            max_size      = std::max(max_size, size_t(size));
            max_alignment = std::max(max_alignment, size_t(alignment));
            all_local &= binding.local != 0;
            compiled.emplace(group, binding);
            bindings.push_back(binding);
            callables.push_back(std::move(callable));
        }
        if (max_size > std::numeric_limits<size_t>::max() - max_alignment + 1)
            return fail("HART shader scratch stride overflows");
        m_scratch_alignment     = max_alignment;
        m_params.scratch_stride = all_local ? 0
                                            : ((max_size + max_alignment - 1)
                                               & ~(max_alignment - 1));
        m_err.infofmt("HART compiled {} materials, {} callables, {} caller "
                      "Groupdata bytes per pixel",
                      callables.size(), next_callable, m_params.scratch_stride);
        return true;
    }

    bool prepare_lights(HartRaytracer& renderer)
    {
        if (renderer.shader_is_light().size() != renderer.shaders().size())
            return fail("Invalid HART light material metadata");
        renderer.prepare_lights();
        std::vector<unsigned> is_light;
        is_light.reserve(renderer.shader_is_light().size());
        for (bool light : renderer.shader_is_light())
            is_light.push_back(unsigned(light));
        for (unsigned primitive : renderer.lightprims())
            if (primitive >= renderer.scene.triangles.size()
                || !(renderer.scene.primitivearea(int(primitive)) > 0))
                return fail("HART light triangles must have positive area");
        m_params.material_is_light = upload<unsigned>(is_light);
        m_params.light_primitives  = upload<unsigned>(renderer.lightprims());
        m_params.light_count       = unsigned(renderer.lightprims().size());
        return !m_failed;
    }

    template<typename T> const T* upload(cspan<T> values)
    {
        if (m_failed || values.empty())
            return nullptr;
        if (values.size() > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fail("HART mesh upload size overflows");
            return nullptr;
        }
        void* device = m_context.alloc(values.size() * sizeof(T));
        if (!device || !m_context.upload(device, as_bytes(values))) {
            m_failed = true;
            return nullptr;
        }
        return static_cast<const T*>(device);
    }

    bool frame_storage(size_t count)
    {
        if (count > m_pixels.max_size())
            return fail("HART image allocation size overflows");
        m_pixels.assign(count, Color3(std::numeric_limits<float>::quiet_NaN()));
        if (count > m_output_capacity) {
            auto* output = static_cast<Color3*>(
                m_context.alloc(count * sizeof(Color3)));
            if (!output) {
                m_failed = true;
                return false;
            }
            m_params.output   = output;
            m_output_capacity = count;
        }
        const size_t stride = size_t(m_params.scratch_stride);
        if (stride) {
            const size_t limit = std::numeric_limits<size_t>::max();
            if (count > (limit - m_scratch_alignment + 1) / stride)
                return fail("HART shader scratch allocation size overflows");
            const size_t bytes = count * stride;
            if (bytes > m_scratch_capacity) {
                std::vector<unsigned char> zeros;
                if (bytes > zeros.max_size())
                    return fail(
                        "HART shader scratch exceeds the host size limit");
                zeros.resize(bytes, 0);
                auto* storage = static_cast<unsigned char*>(
                    m_context.alloc(bytes + m_scratch_alignment - 1));
                if (!storage) {
                    m_failed = true;
                    return false;
                }
                const size_t offset = (m_scratch_alignment
                                       - (reinterpret_cast<uintptr_t>(storage)
                                          & (m_scratch_alignment - 1)))
                                      & (m_scratch_alignment - 1);
                m_params.scratch    = storage + offset;
                if (!m_context.upload(m_params.scratch, zeros)) {
                    m_failed = true;
                    return false;
                }
                m_scratch_capacity = bytes;
            }
        } else {
            m_params.scratch = nullptr;
        }
        if (!m_context.upload(m_params.output,
                              as_bytes(cspan<Color3>(m_pixels)))) {
            m_failed = true;
            return false;
        }
        return true;
    }

    bool prepare_background()
    {
        const size_t res = m_params.background_resolution;
        if (!res)
            return true;
        const size_t count = res * res;
        if (count > m_background_values.max_size()
            || count > m_background_cols.max_size()
            || res > m_background_rows.max_size())
            return fail("HART background table allocation size overflows");
        try {
            m_background_values.resize(count);
            m_background_rows.resize(res);
            m_background_cols.resize(count);
        } catch (const std::bad_alloc&) {
            return fail("Cannot allocate HART background importance tables");
        }
        // Bound callable scratch while shading whole rows, never one launch
        // per texel. Table values are refreshed after interactive uploads.
        const size_t rows_per_batch = std::max(size_t(1), size_t(65536) / res);
        for (size_t y = 0; y < res; y += rows_per_batch) {
            const size_t rows = std::min(rows_per_batch, res - y);
            if (!frame_storage(res * rows) || !m_textures.reset_errors())
                return false;
            auto params              = m_params;
            params.background_offset = unsigned(y * res);
            if (!m_context.launch(&params, sizeof(params), unsigned(res),
                                  unsigned(rows), 1)
                || !m_textures.check_errors()
                || !m_context.download({ reinterpret_cast<unsigned char*>(
                                             m_pixels.data()),
                                         res * rows * sizeof(Color3) },
                                       params.output))
                return false;
            std::copy(m_pixels.begin(), m_pixels.end(),
                      m_background_values.begin() + y * res);
        }
        for (const auto& value : m_background_values)
            if (!finite_vec(value) || value.x < 0 || value.y < 0 || value.z < 0)
                return fail("HART background importance sampling requires "
                            "finite nonnegative radiance");
        Background::prepare_cdf(int(res), m_background_values,
                                m_background_rows, m_background_cols);
        for (const auto& value : m_background_values)
            if (!finite_vec(value))
                return fail(
                    "HART background importance weights are not finite");
        for (const auto& cdf : { cspan<float>(m_background_rows),
                                 cspan<float>(m_background_cols) })
            for (float value : cdf)
                if (!std::isfinite(value) || value < 0 || value > 1)
                    return fail("Invalid HART background importance CDF");
        if (count > m_background_capacity) {
            auto* values = static_cast<Vec3*>(
                m_context.alloc(count * sizeof(Vec3)));
            if (!values)
                return false;
            auto* rows = static_cast<float*>(
                m_context.alloc(res * sizeof(float)));
            if (!rows)
                return false;
            auto* cols = static_cast<float*>(
                m_context.alloc(count * sizeof(float)));
            if (!cols)
                return false;
            m_params.background_values = values;
            m_params.background_rows   = rows;
            m_params.background_cols   = cols;
            m_background_capacity      = count;
        }
        return m_context.upload(m_params.background_values,
                                as_bytes(cspan<Vec3>(m_background_values)))
               && m_context.upload(m_params.background_rows,
                                   as_bytes(cspan<float>(m_background_rows)))
               && m_context.upload(m_params.background_cols,
                                   as_bytes(cspan<float>(m_background_cols)));
    }

    OIIO::ErrorHandler& m_err;
    HartTextureStore m_textures;
    HartContext m_context;
    std::string m_arch;
    int m_device                 = 0;
    bool m_initialized           = false;
    bool m_prepared              = false;
    bool m_published             = false;
    bool m_failed                = false;
    bool m_fused                 = false;
    bool m_collect_statistics    = false;
    bool m_cache_enabled         = true;
    double m_compile_seconds     = 0;
    size_t m_local_budget        = 0;
    size_t m_scratch_alignment   = 1;
    size_t m_scratch_capacity    = 0;
    size_t m_output_capacity     = 0;
    size_t m_background_capacity = 0;
    HartPathParams m_params { };
    std::vector<Color3> m_pixels;
    std::vector<Vec3> m_background_values;
    std::vector<float> m_background_rows, m_background_cols;
};



HartRaytracer::HartRaytracer()
    : m_impl(std::make_unique<Impl>(errhandler())) { }



HartRaytracer::~HartRaytracer()
{ clear(); }



bool
HartRaytracer::initialize(int device, bool fused, size_t local_budget,
                          bool cache_enabled, bool statistics)
{
    auto& impl = *m_impl;
    if (impl.m_failed)
        return false;
    if (impl.m_initialized || shadingsys)
        return impl.fail(
            "Initialize HART once, before creating the shading system");
    if (local_budget > size_t(std::numeric_limits<int>::max())
        || (!fused && local_budget))
        return impl.fail("HART local groupdata requires fused callables and a "
                         "budget in [0,INT_MAX]");
    if (!impl.m_context.init(device, impl.m_arch, cache_enabled, statistics)) {
        impl.m_failed = true;
        return false;
    }
    impl.m_fused              = fused;
    impl.m_device             = device;
    impl.m_local_budget       = local_budget;
    impl.m_collect_statistics = statistics;
    impl.m_cache_enabled      = cache_enabled;
    impl.m_initialized        = true;
    return true;
}



bool
HartRaytracer::failed() const
{ return m_impl->m_failed; }



HartContext::ResourceUsage
HartRaytracer::resource_usage() const
{ return m_impl->m_context.resource_usage(); }



void
HartRaytracer::reset_launch_statistics()
{ m_impl->m_context.reset_launch_statistics(); }



void
HartRaytracer::print_statistics() const
{
    const auto& impl = *m_impl;
    const auto stats = impl.m_context.statistics();
    print(
        "HART native pipeline cache: {} (SDK policy; hit status unavailable)\n",
        impl.m_cache_enabled ? "enabled" : "disabled");
    print("HART native preparation: {:.6f} ms OSL groups, {:.6f} ms pipeline\n",
          impl.m_compile_seconds * 1000.0, stats.pipeline_seconds * 1000.0);
    print("HART native synchronized launches: {} launches, {:.6f} ms total\n",
          stats.launches, stats.launch_seconds * 1000.0);
    print("HART native memory: {} context-owned bytes, {} groupdata stride "
          "bytes, {} local budget bytes\n",
          impl.m_context.resource_usage().bytes, impl.m_params.scratch_stride,
          impl.m_local_budget);
    print("HART native stack estimates: traversal {} bytes, state {} bytes, "
          "continuation {} bytes\n",
          stats.traversal_stack, stats.state_stack, stats.continuation_stack);
}



int
HartRaytracer::supports(string_view feature) const
{
    return feature == "HART" || feature == "HARTClosures"
           || feature == "HARTClosureParameters" || feature == "HARTTextures"
           || feature == "HARTTextureDefaults" || feature == "HARTArrayBounds"
           || feature == "HARTSplineErrors" || feature == "HARTColorSystem"
           || feature == "HARTNoiseErrors" || feature == "HARTDiagnostics"
           || feature == "HARTInteractive" || feature == "HARTGeometry";
}



void*
HartRaytracer::device_alloc(size_t size)
{ return m_impl->m_textures.device_alloc(m_impl->m_device, size); }



void
HartRaytracer::device_free(void* ptr)
{ m_impl->m_textures.device_free(m_impl->m_device, ptr); }



void*
HartRaytracer::copy_to_device(void* dst, const void* src, size_t size)
{ return m_impl->m_textures.copy_to_device(m_impl->m_device, dst, src, size); }



RendererServices::TextureHandle*
HartRaytracer::get_texture_handle(ustring filename, ShadingContext*,
                                  const TextureOpt*)
{
    auto& impl = *m_impl;
    if (impl.m_failed)
        return nullptr;
    if (!impl.m_initialized) {
        impl.fail("Initialize HART before loading textures");
        return nullptr;
    }
    // The texture store uses the current HIP device, including compiler threads.
    if (!impl.m_context.make_current()) {
        impl.m_failed = true;
        return nullptr;
    }
    const uint64_t id = impl.m_textures.load(filename);
    if (!id)
        impl.m_failed = true;
    return reinterpret_cast<TextureHandle*>(uintptr_t(id));
}



bool
HartRaytracer::good(TextureHandle* handle)
{ return handle && m_impl->m_initialized && !m_impl->m_failed; }



void
HartRaytracer::prepare_render()
{
    auto& impl       = *m_impl;
    impl.m_published = false;
    pixelbuf.clear();
    if (impl.m_failed)
        return;
    if (!impl.m_initialized || !shadingsys || impl.m_prepared) {
        impl.fail("HART prepare_render requires an initialized, unprepared "
                  "renderer with a shading system");
        return;
    }
    std::vector<HartTriangle> triangles;
    std::vector<unsigned> material_ids;
    std::vector<float> surfaceareas;
    if (!impl.render_options(*this)
        || !impl.configure_camera(camera, camera.xres, camera.yres)
        || !impl.geometry(*this, triangles, material_ids, surfaceareas))
        return;
    std::vector<HartCallable> callables;
    std::vector<HartMaterialBinding> bindings;
    OIIO::Timer compile_timer(impl.m_collect_statistics);
    if (!impl.compile_materials(*this, callables, bindings))
        return;
    if (impl.m_collect_statistics)
        impl.m_compile_seconds = compile_timer();
    cspan<unsigned char> raygen;
    for (const auto& module : hart_pathtracer_modules) {
        if (module.arch && impl.m_arch == module.arch && module.data
            && module.size && *module.size > 0) {
            raygen = { module.data, size_t(*module.size) };
            break;
        }
    }
    if (raygen.empty()) {
        impl.fail("No embedded HART path tracer for the selected architecture");
        return;
    }
    const unsigned material_count = unsigned(bindings.size());
    if (!impl.m_context.build_accel(scene.verts, triangles, material_ids,
                                    material_count)
        || !impl.m_context.create_pipeline(raygen, "__raygen__osl_hart_path",
                                           material_count, callables,
                                           "__raygen__osl_hart_background")
        || !impl.m_textures.prepare(*shadingsys)) {
        impl.m_failed = true;
        return;
    }
    auto& params          = impl.m_params;
    params.vertices       = impl.upload<Vec3>(scene.verts);
    params.normals        = impl.upload<Vec3>(scene.normals);
    params.uvs            = impl.upload<Vec2>(scene.uvs);
    params.triangles      = impl.upload<TriangleIndices>(scene.triangles);
    params.shader_ids     = impl.upload<int>(scene.shaderids);
    params.normal_indices = impl.upload<TriangleIndices>(scene.n_triangles);
    params.uv_indices     = impl.upload<TriangleIndices>(scene.uv_triangles);
    params.surfaceareas   = impl.upload<float>(surfaceareas);
    params.materials      = impl.upload<HartMaterialBinding>(bindings);
    params.material_count = material_count;
    params.triangle_count = unsigned(scene.triangles.size());
    params.traversable    = impl.m_context.traversable();
    params.textures       = impl.m_textures.device_state();
    if (impl.m_failed)
        return;
    if (!params.textures) {
        impl.fail("HART device service state is missing");
        return;
    }
    if (!impl.prepare_lights(*this))
        return;
    impl.m_prepared = true;
}



void
HartRaytracer::render(int xres, int yres)
{
    auto& impl       = *m_impl;
    impl.m_published = false;
    if (impl.m_failed || had_error()) {
        impl.m_failed = true;
        pixelbuf.clear();
        return;
    }
    if (!impl.m_prepared) {
        impl.fail("Prepare the HART renderer before rendering");
        pixelbuf.clear();
        return;
    }
    for (const auto& material : shaders()) {
        void* interactive = nullptr;
        if (!shadingsys->getattribute(material.surf.get(),
                                      "device_interactive_params",
                                      TypeDesc::PTR, &interactive)) {
            impl.fail("Cannot retrieve HART material interactive bindings");
            pixelbuf.clear();
            return;
        }
    }
    if (!impl.render_options(*this)
        || !impl.configure_camera(camera, xres, yres)) {
        pixelbuf.clear();
        return;
    }
    if (!impl.m_textures.prepare(*shadingsys)) {
        impl.m_failed = true;
        pixelbuf.clear();
        return;
    }
    impl.m_params.textures = impl.m_textures.device_state();
    if (!impl.prepare_background()
        || !impl.frame_storage(size_t(xres) * size_t(yres))
        || !impl.m_textures.reset_errors()
        || !impl.m_context.launch(&impl.m_params, sizeof(impl.m_params),
                                  unsigned(xres), unsigned(yres))
        || !impl.m_textures.check_errors()
        || !impl.m_context.download({ reinterpret_cast<unsigned char*>(
                                          impl.m_pixels.data()),
                                      impl.m_pixels.size() * sizeof(Color3) },
                                    impl.m_params.output)) {
        impl.m_failed = true;
        pixelbuf.clear();
        return;
    }
    if (had_error()) {
        impl.m_failed = true;
        pixelbuf.clear();
        return;
    }
    for (const auto& pixel : impl.m_pixels) {
        if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)
            || !std::isfinite(pixel.z)) {
            impl.fail("HART returned nonfinite or unwritten RGB pixels");
            pixelbuf.clear();
            return;
        }
    }
    static_assert(sizeof(Color3) == 3 * sizeof(float),
                  "HART output requires packed RGB floats");
    pixelbuf.reset(OIIO::ImageSpec(xres, yres, 3, TypeDesc::FLOAT));
    if (!pixelbuf.set_pixels(OIIO::ROI(0, xres, 0, yres, 0, 1, 0, 3),
                             TypeDesc::FLOAT, impl.m_pixels.data())) {
        errhandler().errorfmt("Cannot publish HART pixels: {}",
                              pixelbuf.geterror());
        impl.m_failed = true;
        pixelbuf.clear();
        return;
    }
    impl.m_published = true;
    errhandler().infofmt("HART path tracer rendered {}x{} with {} samples "
                         "per pixel",
                         xres, yres, impl.m_params.aa * impl.m_params.aa);
}



void
HartRaytracer::warmup()
{ render(camera.xres, camera.yres); }



void
HartRaytracer::finalize_pixel_buffer()
{
    auto& impl = *m_impl;
    if (!impl.m_failed && !impl.m_published)
        impl.fail("HART has no completed pixel buffer to publish");
    if (impl.m_failed)
        pixelbuf.clear();
}



void
HartRaytracer::clear()
{
    auto& impl                  = *m_impl;
    const bool context_cleared  = impl.m_context.clear();
    const bool textures_cleared = context_cleared && impl.m_textures.clear();
    impl.m_failed |= !context_cleared || !textures_cleared;
    impl.m_initialized         = false;
    impl.m_prepared            = false;
    impl.m_published           = false;
    impl.m_compile_seconds     = 0;
    impl.m_output_capacity     = 0;
    impl.m_scratch_capacity    = 0;
    impl.m_background_capacity = 0;
    impl.m_params              = { };
    impl.m_pixels.clear();
    impl.m_background_values.clear();
    impl.m_background_rows.clear();
    impl.m_background_cols.clear();
    pixelbuf.clear();
    SimpleRaytracer::clear();
}

OSL_NAMESPACE_END
