



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

extern "C" {
__device__ __constant__ testshadeHIP::RenderParams render_params;
}


extern "C" __global__ void shade(float3* Cout, int w, int h)
{

}


extern "C" __global__ void ShadeExplicit(float3* Cout, int w, int h, OSL_HIP::ShaderGlobals* shaderGlobals)
{

}