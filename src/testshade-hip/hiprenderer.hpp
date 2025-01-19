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

    std::vector<uint8_t> load_bitcode_file(OIIO::string_view filename);


    void init_shadingsys(OSL::ShadingSystem* shadingsys) final;
    // using options from the base renderer
    bool init_renderer_options();

    // instead of launching the kernel twice we get the globals by name and copy to symbol
    void prepare_render() override final;

    void render(int xres , int yres, RenderState& renderState) override final;

    void set_transforms(const OSL::Matrix44& object2common, const OSL::Matrix44& shader2common);
    
    void register_named_transforms();


    // required by the shaders to be execute on the device 
    virtual hipDeviceptr_t device_alloc(size_t size) override;
    virtual void device_free(void* ptr) override;
    virtual hipDeviceptr_t copy_to_device(hipDeviceptr_t dst_device, const void* src_host, size_t size) override;

private:
    hipDeviceProp_t m_deviceProperties;
    hipStream_t m_stream;
    hipModule_t m_module;
    hipFunction_t m_function;

    //device data


    // named transformations
    uint64_t m_num_named_xforms { 0 };
    hipDeviceptr_t d_xform_name_buffer;
    hipDeviceptr_t d_xform_buffer;


    // transformations and the device counterparts
    OSL::Matrix44 m_shader2common;  // "shader" space to "common" space matrix
    hipDeviceptr_t m_device_shader2common;
    OSL::Matrix44 m_object2common;  // "object" space to "common" space matrix
    hipDeviceptr_t m_device_object2common;

    // TODO: maybe implement switching between devices
    int m_deviceId { 0 };

    OSL::ShadingSystem* m_shadingSystem { nullptr };
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler { new OIIO::ErrorHandler };
   

    // device memory management, garbage collection
    std::vector<hipDeviceptr_t> m_ptrs_to_free;
    std::vector<hipArray_t> m_arrays_to_free;



};


#endif // HIPRENDERER_HPP