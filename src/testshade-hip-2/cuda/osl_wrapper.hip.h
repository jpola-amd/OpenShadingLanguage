#ifndef OSL_WRAPPER_HIP_H
#define OSL_WRAPPER_HIP_H


extern "C" __device__ void osl_init(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
extern "C" __device__ void osl_entry(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
extern "C" __device__ void osl_fused(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);


#endif // OSL_WRAPPER_HIP_H