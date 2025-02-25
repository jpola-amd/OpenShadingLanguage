// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <hip/hip_runtime.h>
#include "osl_wrapper.hip.h"
#include "rend_lib.hip.h"
#include "render_params.hip.h"


OSL_NAMESPACE_ENTER
namespace pvt {
__device__ hipDeviceptr_t s_color_system    = 0;
__device__ uint64_t osl_printf_buffer_start = 0;
__device__ uint64_t osl_printf_buffer_end   = 0;
__device__ hipDeviceptr_t osl_printf_buffer = 0;
__device__ uint64_t osl_printf_buffer_size  = 0;
__device__ uint64_t test_str_1              = 0;
__device__ uint64_t test_str_2              = 0;
__device__ uint64_t num_named_xforms        = 0;
__device__ hipDeviceptr_t xform_name_buffer = 0;
__device__ hipDeviceptr_t xform_buffer      = 0;
}  // namespace pvt
OSL_NAMESPACE_EXIT


extern "C" __global__ void
__miss__()
{
    // do nothing
}

extern "C" __global__ void
__closesthit__()
{
    // do nothing
}

extern "C" __global__ void
__anyhit__()
{
    // do nothing
}

extern "C"  __device__ void
osl_printf(void* sg_, OSL::ustringhash_pod fmt_str_hash, void* args);

extern "C" __device__ void 
osl_printf(void* sg_, char *fmt_str, void* args);



extern "C" __global__ void
__raygen__setglobals(testshade::RenderParams* lp)
{
    testshade::RenderParams& render_params = *lp;
    printf("Test string before global: %lu render param: %lu\n", OSL::pvt::test_str_1, render_params.test_str_1);
    // Set global variables
    OSL::pvt::osl_printf_buffer_start = render_params.osl_printf_buffer_start;
    OSL::pvt::osl_printf_buffer_end   = render_params.osl_printf_buffer_end;

    OSL::pvt::osl_printf_buffer       = render_params.osl_printf_buffer;
    OSL::pvt::osl_printf_buffer_size  = render_params.osl_printf_buffer_size;
    
    OSL::pvt::s_color_system          = render_params.color_system;
    OSL::pvt::test_str_1              = render_params.test_str_1;
    OSL::pvt::test_str_2              = render_params.test_str_2;
    OSL::pvt::num_named_xforms        = render_params.num_named_xforms;
    OSL::pvt::xform_name_buffer       = render_params.xform_name_buffer;
    OSL::pvt::xform_buffer            = render_params.xform_buffer;

    printf("Test string before global: %lu render param: %lu\n", OSL::pvt::test_str_1, render_params.test_str_1);
    printf("Testing the printf params: %lu, %lu, %p, %lu, %lu, %lu\n", 
        OSL::pvt::osl_printf_buffer_start, 
        OSL::pvt::osl_printf_buffer_end, 
        OSL::pvt::osl_printf_buffer, 
        OSL::pvt::osl_printf_buffer_size,
        uint64_t(OSL::pvt::osl_printf_buffer),
        uint64_t( (uint64_t) OSL::pvt::osl_printf_buffer  + OSL::pvt::osl_printf_buffer_size)
        );

   

}



extern "C" __global__ void
__miss__setglobals()
{
}

 

extern "C" __global__ void
__raygen__(testshade::RenderParams* lp)
{
    testshade::RenderParams& render_params = *lp;

    const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;

    int w = 256;
    int h = 256;
	const uint32_t index = w * y + x;
    
    // uint3 launch_dims  = optixGetLaunchDimensions();
    // uint3 launch_index = optixGetLaunchIndex();

    // auto sbtdata = reinterpret_cast<GenericData*>(optixGetSbtDataPointer());

    const float invw      = render_params.invw;
    const float invh      = render_params.invh;
    bool flipv            = render_params.flipv;
    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);

    const uint3 dims = make_uint3(
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y,
        gridDim.z * blockDim.z
    );

    // Compute the pixel coordinates
    // Matching testshade's setup_shaderglobals for !pixelcenters
    float2 d = make_float2((dims.x == 1) ? 0.5f : invw * x,
                           (dims.y == 1) ? 0.5f : invh * y);

    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params[256];

    OSL_CUDA::ShaderGlobals sg;
    // Setup the ShaderGlobals
    sg.I  = make_float3(0, 0, 1);
    sg.N  = make_float3(0, 0, 1);
    sg.Ng = make_float3(0, 0, 1);
    sg.P  = make_float3(d.x, d.y, 0);
    sg.u  = d.x;
    sg.v  = d.y;
    if (flipv)
        sg.v = 1.f - sg.v;

    sg.dudx = invw;
    sg.dudy = 0;
    sg.dvdx = 0;
    sg.dvdy = invh;

    // Matching testshade's setup_shaderglobals
    sg.dPdu = make_float3(1.f, 0.f, 0.f);
    sg.dPdv = make_float3(0.f, 1.f, 0.f);

    sg.dPdx = make_float3(1.f, 0.f, 0.f);
    sg.dPdy = make_float3(0.f, 1.f, 0.f);
    sg.dPdz = make_float3(0.f, 0.f, 0.f);

    sg.Ci          = NULL;
    sg.surfacearea = 0;
    sg.backfacing  = 0;

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype        = OSL::Ray::CAMERA;
    sg.flipHandedness = 0;

    sg.shader2common = reinterpret_cast<void*>(render_params.shader2common);
    sg.object2common = reinterpret_cast<void*>(render_params.object2common);

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*)&closure_pool[0] = 0;
    sg.renderstate          = &closure_pool[0];

    // if (render_params.fused_callable)
    // {
    //     osl_fused(&sg, params, nullptr, nullptr, 0, nullptr);
    // }
    // else
    // {
    //     osl_init(&sg, params, nullptr, nullptr, 0, nullptr);
    //     osl_entry(&sg, params, nullptr, nullptr, 0, nullptr);
    // }
   
    
    // Run the OSL group and init functions
    // if (render_params.fused_callable)
    //     // call osl_init_func 
    //     optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*,
    //                     int, void*>(0u, &sg /*shaderglobals_ptr*/,
    //                                 params /*groupdata_ptr*/,
    //                                 nullptr /*userdata_base_ptr*/,
    //                                 nullptr /*output_base_ptr*/,
    //                                 0 /*shadeindex - unused*/,
    //                                 sbtdata->data /*interactive_params_ptr*/);
    // else {
    //     // call osl_init_func
    //     optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*,
    //                     int, void*>(0u, &sg /*shaderglobals_ptr*/,
    //                                 params /*groupdata_ptr*/,
    //                                 nullptr /*userdata_base_ptr*/,
    //                                 nullptr /*output_base_ptr*/,
    //                                 0 /*shadeindex - unused*/,
    //                                 sbtdata->data /*interactive_params_ptr*/);
    //     // call osl_group_func
    //     optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*,
    //                     int, void*>(1u, &sg /*shaderglobals_ptr*/,
    //                                 params /*groupdata_ptr*/,
    //                                 nullptr /*userdata_base_ptr*/,
    //                                 nullptr /*output_base_ptr*/,
    //                                 0 /*shadeindex - unused*/,
    //                                 sbtdata->data /*interactive_params_ptr*/);
    // }

    float* f_output      = (float*)params;
    int pixel            = index;
    // f_output[1]          = 1.0f;
    // f_output[2]          = 1.0f;
    // f_output[3]          = 1.0f;

    output_buffer[pixel] = { f_output[1], f_output[2], f_output[3] };
    // if (x < 2 && y < 1 )
    // {
    //     printf("Pixel: %d: (%f, %f, %f) \n", pixel, f_output[1], f_output[2], f_output[3]);
    //     printf(" buffer values %f, %f, %f\n", output_buffer[pixel].x, output_buffer[pixel].y, output_buffer[pixel].z);
    // }

     //OSL::ustring fmt_str = OSL::ustring("Hello from OptiX!\n");
    int data = 10;
    int data1 = 20;
    double data2 = 30.0f;
    // int data2 = 20;
    // float data3 = 30.0f;
    uint64_t size = sizeof(data) + sizeof(data1) + sizeof(data2);
    void * args[] = {&size, &data, &data1, &data2};
    

    osl_printf(nullptr, OSL::pvt::test_str_1, args);
}

// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t, float dsdx, float dtdx,
                float dsdy, float dtdy)
{
    const float2 dx           = { dsdx, dtdx };
    const float2 dy           = { dsdy, dtdy };
    hipTextureObject_t texID = hipTextureObject_t(handle);
    return tex2DGrad<float4>(texID, s, t, dx, dy);
}
