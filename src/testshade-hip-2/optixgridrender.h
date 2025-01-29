// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once
#include <hip/hip_runtime.h>
#include <OpenImageIO/ustring.h>

#include <OSL/oslexec.h>

#include <OSL/hashes.h>

#include "hip_compat.h"
#include "simplerend.h"


OSL_NAMESPACE_ENTER


class OptixGridRenderer final : public SimpleRenderer {
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixGridRenderer();
    virtual ~OptixGridRenderer();

    int supports(string_view feature) const override
    {
        if (feature == "HIP")
            return true;
        return SimpleRenderer::supports(feature);
    }

    std::string load_ptx_file(string_view filename);
    bool synch_attributes();

    void init_shadingsys(ShadingSystem* ss) final;
    bool init_optix_context(int xres, int yres);
    bool make_optix_materials();
    bool finalize_scene();
    void prepare_render() final;
    void warmup() final;
    void render(int xres, int yres, RenderState& renderState) final;
    void finalize_pixel_buffer() final;
    void clear() final;

    virtual void set_transforms(const OSL::Matrix44& object2common,
                                const OSL::Matrix44& shader2common);

    virtual void register_named_transforms();

    /// Register "shadeops" functions that should or should not be inlined
    /// during ShaderGroup optimization.
    void register_inline_functions();

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    bool good(TextureHandle* handle) override;

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    TextureHandle* get_texture_handle(ustring filename,
                                      ShadingContext* shading_context,
                                      const TextureOpt* options) override;

    optix::OptixDeviceContext optix_ctx() { return m_optix_ctx; }
    optix::OptixDeviceContext context() { return m_optix_ctx; }
    optix::OptixDeviceContext operator->() { return context(); }

    void processPrintfBuffer(void* buffer_data, size_t buffer_size);

    virtual void* device_alloc(size_t size) override;
    virtual void device_free(void* ptr) override;
    virtual void* copy_to_device(void* dst_device, const void* src_host,
                                 size_t size) override;

private:
    optix::Context m_optix_ctx = nullptr;

    hipStream_t m_cuda_stream;
    // OptixShaderBindingTable m_optix_sbt            = {};
    // OptixShaderBindingTable m_setglobals_optix_sbt = {};
    // OptixPipeline m_optix_pipeline                 = {};
    bool m_fused_callable                          = false;
    hipDeviceptr_t d_output_buffer;
    hipDeviceptr_t d_launch_params = 0;
    hipDeviceptr_t d_osl_printf_buffer;
    hipDeviceptr_t d_color_system;
    hipDeviceptr_t d_object2common;
    hipDeviceptr_t d_shader2common;
    uint64_t m_num_named_xforms;
    hipDeviceptr_t d_xform_name_buffer;
    hipDeviceptr_t d_xform_buffer;
    uint64_t test_str_1;
    uint64_t test_str_2;
    const unsigned long OSL_PRINTF_BUFFER_SIZE = 8 * 1024 * 1024;

    std::unordered_map<ustringhash, optix::TextureSampler> m_samplers;
    hipModule_t m_module { nullptr };
    hipFunction_t m_function_shade { nullptr };

    OSL::Matrix44 m_shader2common;  // "shader" space to "common" space matrix
    OSL::Matrix44 m_object2common;  // "object" space to "common" space matrix

    // CUdeviceptrs that need to be freed after we are done
    std::vector<hipDeviceptr_t> m_ptrs_to_free;
    std::vector<hipArray_t> m_arrays_to_free;
};



OSL_NAMESPACE_EXIT
