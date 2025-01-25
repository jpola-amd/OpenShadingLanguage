
namespace testshadeHIP 
{
struct RenderParams {
    float invw{0.f};
    float invh{0.f};
    hipDeviceptr_t output_buffer{nullptr};
    bool flipv {false};
    int fused_callable {0};
    uint64_t osl_printf_buffer_start {0};
    uint64_t osl_printf_buffer_end {0};
    hipDeviceptr_t color_system {nullptr};

    // for transforms
    hipDeviceptr_t object2common {nullptr};
    hipDeviceptr_t shader2common {nullptr};
    uint64_t num_named_xforms {0};
    hipDeviceptr_t xform_name_buffer {nullptr};
    hipDeviceptr_t xform_buffer {nullptr};

    // for used-data tests
    uint64_t test_str_1 {0};
    uint64_t test_str_2 {0};
};

}  // namespace testshadeHIP