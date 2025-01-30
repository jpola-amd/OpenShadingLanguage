#ifndef OSL_WRAPPER_HIP_H
#define OSL_WRAPPER_HIP_H


using OslDeviceFunction = void (*)(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);

struct OslDeviceShaderLayer {
    OslDeviceFunction init_func {nullptr};
    OslDeviceFunction entry_func {nullptr};
    OslDeviceFunction fused_func {nullptr};
};

struct OslHostShaderLayer{
    hipDeviceptr_t init_func {nullptr};
    hipDeviceptr_t entry_func {nullptr};
    hipDeviceptr_t fused_func {nullptr};
};

struct OslDeviceFunctionTable {
    OslDeviceShaderLayer* layers {nullptr};
    int num_layers {0};
};

struct OslHostFunctionTable {
    OslHostShaderLayer* layers;
    int num_layers;
};


#endif // OSL_WRAPPER_HIP_H