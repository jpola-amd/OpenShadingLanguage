

#include "osl_wrapper.hip.h"


// here we define the extern "C" functions that are called from the OSL generated code

extern "C" __device__ void __direct_callable__osl_init_group_unnamed_group_1(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
extern "C" __device__ void __direct_callable__osl_entry_group_unnamed_group_1_name_test_0(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
extern "C" __device__ void __direct_callable__fused_unnamed_group_1_name_test_0(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);



extern "C" __device__ osl_init(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive) {
    __direct_callable__osl_init_group_unnamed_group_1(sg, params, userdata, outdata, idx, interactive);
}

extern "C" __device__ osl_entry(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive) {
    __direct_callable__osl_entry_group_unnamed_group_1_name_test_0(sg, params, userdata, outdata, idx, interactive);
}

extern "C" __device__ osl_fused(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive) {
    __direct_callable__fused_unnamed_group_1_name_test_0(sg, params, userdata, outdata, idx, interactive);
}




