



#include <hip/hip_runtime.h>

#include "rend_lib.hip.hpp"
#include "../render_params.hpp"


// Definition is in the hip_grid_renderer.hip.cpp
OSL_NAMESPACE_ENTER

namespace pvt {
    __device__ hipDeviceptr_t s_color_system {nullptr};
    __device__ uint64_t osl_printf_buffer_start {0};
    __device__ uint64_t osl_printf_buffer_end {0};
    __device__ uint64_t test_str_1 {0};
    __device__ uint64_t test_str_2 {0};
    __device__ uint64_t num_named_xforms {0};
    __device__ hipDeviceptr_t xform_name_buffer {0};
    __device__ hipDeviceptr_t xform_buffer {0};
}  // namespace pvt
OSL_NAMESPACE_EXIT


// these trampolines will be linked in by the renderer
extern "C" __device__ void
__osl__init(ShaderGlobals*, void*);
extern "C" __device__ void
__osl__entry(ShaderGlobals*, void*);

extern "C" {
__device__ __constant__ testshadeHIP::RenderParams gc_render_params;
}


extern "C" __device__  void __prepare_globals(testshadeHIP::RenderParams& render_params)
{
    // Set global variables
    OSL::pvt::osl_printf_buffer_start = render_params.osl_printf_buffer_start;
    OSL::pvt::osl_printf_buffer_end   = render_params.osl_printf_buffer_end;
    OSL::pvt::s_color_system          = render_params.color_system;
    OSL::pvt::test_str_1              = render_params.test_str_1;
    OSL::pvt::test_str_2              = render_params.test_str_2;
    OSL::pvt::num_named_xforms        = render_params.num_named_xforms;
    OSL::pvt::xform_name_buffer       = render_params.xform_name_buffer;
    OSL::pvt::xform_buffer            = render_params.xform_buffer;
}

extern "C" __global__ void shade(float3* Cout, int w, int h, testshadeHIP::RenderParams* ptr_render_params)
{
    

    testshadeHIP::RenderParams& render_params = *ptr_render_params;

   // Get thread indices (equivalent to launch index in OptiX)
    const uint3 thread_idx = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    );

    if (thread_idx.x < 1)
    {
        __prepare_globals(render_params);
        printf("Thread idx: %d, w %d, h %d c: %p\n", thread_idx.x, w, h, render_params.color_system);
    }
    //return;



    // Get dimensions (equivalent to launch dims in OptiX)
    const uint3 dims = make_uint3(
        gridDim.x * blockDim.x,
        gridDim.y * blockDim.y,
        gridDim.z * blockDim.z
    );

    const float invw      = render_params.invw;
    const float invh      = render_params.invh;
    bool flipv            = render_params.flipv;

    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);


    // Compute pixel coordinates (same logic as original)
    float2 d = make_float2(
        (dims.x == 1) ? 0.5f : invw * thread_idx.x,
        (dims.y == 1) ? 0.5f : invh * thread_idx.y
    );

     // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params[256];

    ShaderGlobals sg;
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
    sg.raytype        = 1; //OSL::Ray::CAMERA;
    sg.flipHandedness = 0;

    sg.shader2common = reinterpret_cast<void*>(render_params.shader2common);
    sg.object2common = reinterpret_cast<void*>(render_params.object2common);

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*)&closure_pool[0] = 0;
    sg.renderstate          = &closure_pool[0];

     // call the osl_init
    __osl__init(&sg, params);
    __osl__entry(&sg, params);

    // call the osl_entry


    float* f_output      = (float*)params;
    int pixel            = thread_idx.y * thread_idx.x + thread_idx.x;
    output_buffer[pixel] = { f_output[1], f_output[2], f_output[3] };
   


}


extern "C" __global__ void ShadeExplicit(float3* Cout, int w, int h, ShaderGlobals* shaderGlobals)
{

}