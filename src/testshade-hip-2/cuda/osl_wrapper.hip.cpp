

#include "osl_wrapper.hip.h"


// here we define the extern "C" functions that are called from the OSL generated code

extern "C" __device__ void __direct_callable__osl_init_group_unnamed_group_1(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
extern "C" __device__ void __direct_callable__osl_entry_group_unnamed_group_1_name_test_0(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
extern "C" __device__ void __direct_callable__fused_unnamed_group_1_name_test_0(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);


extern "C" __device__ __constant__ OslDeviceFunction init_func = __direct_callable__osl_init_group_unnamed_group_1;
extern "C" __device__ __constant__ OslDeviceFunction entry_func = __direct_callable__osl_entry_group_unnamed_group_1_name_test_0;
extern "C" __device__ __constant__ OslDeviceFunction fused_func = __direct_callable__fused_unnamed_group_1_name_test_0;




