



#include <hip/hip_runtime.h>

#include "rend_lib.hip.hpp"
#include "render_params.hpp"


// Definition is in the hip_grid_renderer.hip.cpp
OSL_NAMESPACE_ENTER
namespace pvt {
    __device__ hipDeviceptr_t s_color_system;
    __device__ hipDeviceptr_t osl_printf_buffer_start;
    __device__ hipDeviceptr_t osl_printf_buffer_end;
    __device__ uint64_t test_str_1;
    __device__ uint64_t test_str_2;
    __device__ uint64_t num_named_xforms;
    __device__ hipDeviceptr_t xform_name_buffer;
    __device__ hipDeviceptr_t xform_buffer;
}  // namespace pvt
OSL_NAMESPACE_EXIT

extern "C" {
__device__ __constant__ testshadeHIP::RenderParams render_params;
}


extern "C" __global__ void shade(float3* Cout, int w, int h)
{

}


extern "C" __global__ void ShadeExplicit(float3* Cout, int w, int h, OSL_HIP::ShaderGlobals* shaderGlobals)
{

}