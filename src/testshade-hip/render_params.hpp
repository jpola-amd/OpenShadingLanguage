
namespace testshadeHIP 
{
struct RenderParams {
    float invw;
    float invh;
    hipDeviceptr_t output_buffer;
    bool flipv;
    int fused_callable;
    hipDeviceptr_t osl_printf_buffer_start;
    hipDeviceptr_t osl_printf_buffer_end;
    hipDeviceptr_t color_system;

    // for transforms
    hipDeviceptr_t object2common;
    hipDeviceptr_t shader2common;
    uint64_t num_named_xforms;
    hipDeviceptr_t xform_name_buffer;
    hipDeviceptr_t xform_buffer;

    // for used-data tests
    uint64_t test_str_1;
    uint64_t test_str_2;
};

}  // namespace testshadeHIP