#include "hiprenderer.hpp"

#include "assert_hip.hpp"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

//TODO: Add assert to check HIP CALLS
HIPRenderer::HIPRenderer()
{
    HIP_CHECK(hipInit(0));
    
    // Get the device count
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No HIP capable devices found" << std::endl;
        exit(1);
    }

    HIP_CHECK(hipGetDeviceProperties(&m_deviceProperties, m_deviceId));
    std::cout << "Using device " << m_deviceId << ": " << m_deviceProperties.name << std::endl;

    HIP_CHECK(hipStreamCreate(&m_stream));
}

HIPRenderer::~HIPRenderer()
{
    HIP_CHECK(hipStreamDestroy(m_stream));

    for(hipDeviceptr_t ptr : m_ptrs_to_free)
    {
        HIP_CHECK(hipFree(ptr));
    }

    for(hipArray_t arr : m_arrays_to_free)
    {
        HIP_CHECK(hipFreeArray(arr));
    }
}


int HIPRenderer::supports(OIIO::string_view feature) const
{
    if (feature == "HIP")
        return true;
    return SimpleRenderer::supports(feature);
}

std::vector<uint8_t>
HIPRenderer::load_bitcode_file(OIIO::string_view filename)
{
     std::vector<std::string> paths = {
        OIIO::Sysutil::this_program_path(),
        OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path())
    };

    std::string filepath = OIIO::Filesystem::searchpath_find(filename, paths,
                                                             false);

    std::cout << "Loading bitcode file from: " << filepath << std::endl;
    if (OIIO::Filesystem::exists(filepath)) 
    {
        const auto size = OIIO::Filesystem::file_size(filepath);
        std::vector<uint8_t> bitcode_data(size);
        if (OIIO::Filesystem::read_bytes(filepath, bitcode_data.data(), size))
        {
            return bitcode_data;
        }
    }

    return {};
}

void
HIPRenderer::init_shadingsys(OSL::ShadingSystem* shadingsys)
{
    m_shadingSystem = shadingsys;
}

bool
HIPRenderer::init_renderer_options()
{
    if (!m_shadingSystem)
    {
        errhandler().error("ShadingSystem is not initialized");
        return false;
    }

    if (!options.get_int("no_rend_lib_bitcode"))
    {
        std::cout << "Loading rend_lib bitcode" << std::endl;
        // m_shadingSystem->attribute("lib_bitcode",
        //                            { OSL::TypeDesc::UINT8, rend_lib_hip_llvm_compiled_ops_size },
        //                             rend_lib_hip_llvm_compiled_ops_block);
    };
    return false;
}

void
HIPRenderer::prepare_render()
{
    // use the hiprtcAddNameExpression mechanism to get the global variables
    //hipModuleGetGlobal(&m_function, &m_module, "test", 0);
}

void
HIPRenderer::render(int xres, int yres)
{


}


void HIPRenderer::set_transforms(const OSL::Matrix44& object2common, const OSL::Matrix44& shader2common)
{
    m_object2common = object2common;
    m_shader2common = shader2common;
}

void HIPRenderer::register_named_transforms()
{
    std::vector<uint64_t> xform_name_buffer;
    std::vector<OSL::Matrix44> xform_buffer;

    // Gather:
    //   1) All of the named transforms
    //   2) The "string" value associated with the transform name, which is
    //      actually the ustring hash of the transform name.
    // populated by name_transform
    for (const auto& item : m_named_xforms) {
        const uint64_t addr = item.first.hash();
        xform_name_buffer.push_back(addr);
        xform_buffer.push_back(*item.second);
    }

    // Push the names and transforms to the device
    size_t sz           = sizeof(uint64_t) * xform_name_buffer.size();
    d_xform_name_buffer = device_alloc(sz);
    copy_to_device(d_xform_name_buffer, xform_name_buffer.data(), sz);
    
    sz             = sizeof(OSL::Matrix44) * xform_buffer.size();
    d_xform_buffer = device_alloc(sz);
    copy_to_device(d_xform_buffer, xform_buffer.data(), sz);

    m_num_named_xforms = xform_name_buffer.size();
}

hipDeviceptr_t
HIPRenderer::device_alloc(size_t size)
{
    hipDeviceptr_t ptr = nullptr;
    hipError_t result = hipMalloc(&ptr, size);
    if ( result != hipSuccess)
    {
        errhandler().errorfmt("hipMalloc({}) failed with error: {}\n", size, hipGetErrorString(result));
    }
    m_ptrs_to_free.push_back(ptr);
    return ptr;
}

void
HIPRenderer::device_free(hipDeviceptr_t ptr)
{
    hipError_t result = hipFree(ptr);
    if (result != hipSuccess)
    {
        errhandler().errorfmt("hipFree() failed with error: {}\n", hipGetErrorString(result));
    }
}

hipDeviceptr_t
HIPRenderer::copy_to_device(hipDeviceptr_t dst_device, const void* src_host, size_t size)
{
    hipError_t result = hipMemcpy(dst_device, src_host, size, hipMemcpyHostToDevice);
    if (result != hipSuccess)
    {
        errhandler().errorfmt("hipMemcpy host->device of size {} failed with error: {}\n", size, hipGetErrorString(result));
    }
    return dst_device;
}
