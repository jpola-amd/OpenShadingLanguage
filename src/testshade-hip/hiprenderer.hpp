#ifndef HIPRENDERER_HPP
#define HIPRENDERER_HPP

#include "simplerenderer.hpp"

#include <hip/hip_runtime.h>
#include <memory>
#include <OpenImageIO/errorhandler.h>

//TODO: Add finalize_pixel_buffer
/*
std::string buffer_name = "output_buffer";
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    CUDA_CHECK(cudaMemcpy(tmp_buff.data(),
                          reinterpret_cast<void*>(d_output_buffer),
                          m_xres * m_yres * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    OIIO::ImageBuf* buf = outputbuf(0);
    if (buf)
        buf->set_pixels(OIIO::ROI::All(), OIIO::TypeFloat, tmp_buff.data());
*/
class HIPRenderer : public SimpleRenderer
{
public:
    HIPRenderer();
    virtual ~HIPRenderer();

    int supports(OIIO::string_view feature) const override;

    std::vector<char> load_file(OIIO::string_view filename) const;

    // actually not required because the base class has it.
    void init_shadingsys(OSL::ShadingSystem* shadingsys) override final;
    // using options from the base renderer
    bool init_renderer_options() override final;

    // instead of launching the kernel twice we get the globals by name and copy to symbol
    void prepare_render(RenderState& renderState) override final;

    void render(int xres , int yres, RenderState& renderState) override final;

    void set_transforms(const OSL::Matrix44& object2common, const OSL::Matrix44& shader2common) override final;
    
    void register_named_transforms() override final;

    void finalize_pixel_buffer() override final;


    void camera_params(const OSL::Matrix44& world_to_camera,
                       OSL::ustringhash projection, float hfov, float hither,
                       float yon, int xres, int yres);


    // required by the shaders to be execute on the device 
    virtual hipDeviceptr_t device_alloc(size_t size) override;
    virtual void device_free(hipDeviceptr_t ptr) override;
    virtual hipDeviceptr_t copy_to_device(hipDeviceptr_t dst_device, const void* src_host, size_t size) override;

    bool initialize_render_parameters() override;

private:
    hipDeviceProp_t m_deviceProperties;
    hipStream_t m_stream { nullptr };
    hipModule_t m_module { nullptr };

    hipFunction_t m_function_osl_init { nullptr };
    hipFunction_t m_function_osl_entry { nullptr };
    hipFunction_t m_function_shade { nullptr };

    // render parameters
    int m_xres { 0 };
    int m_yres { 0 };

    uint64_t test_str_1;
    uint64_t test_str_2;

    const size_t OSL_PRINTF_BUFFER_SIZE {8 * 1024 * 1024};


    // device memory
    hipDeviceptr_t d_output_buffer { nullptr };
    hipDeviceptr_t d_launch_params { nullptr };    
    hipDeviceptr_t d_osl_printf_buffer { nullptr };
    hipDeviceptr_t d_color_system { nullptr };






    // named transformations
    uint64_t m_num_named_xforms { 0 };
    hipDeviceptr_t d_xform_name_buffer;
    hipDeviceptr_t d_xform_buffer;


    // transformations and the device counterparts
    OSL::Matrix44 m_shader2common;  // "shader" space to "common" space matrix
    hipDeviceptr_t d_shader2common;
    OSL::Matrix44 m_object2common;  // "object" space to "common" space matrix
    hipDeviceptr_t d_object2common;

    // TODO: maybe implement switching between devices
    int m_deviceId { 0 };

    OSL::ShadingSystem* m_shadingSystem { nullptr };
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler { new OIIO::ErrorHandler };
   

    // device memory management, garbage collection
    std::vector<hipDeviceptr_t> m_ptrs_to_free;
    std::vector<hipArray_t> m_arrays_to_free;


    // some utils

     std::vector<std::string> m_search_paths;

    void processPrintfBuffer(void* buffer_data, size_t buffer_size);

};


#endif // HIPRENDERER_HPP