#include "hiprenderer.hpp"

#include "assert_hip.hpp"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <hip/hiprtc.h>

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

    m_search_paths = {
        OIIO::Sysutil::this_program_path(),
        OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()),
        HIP_BC_DIR,
        HIP_SRC_DIR
    };
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

// this can be a free function
std::vector<uint8_t>
HIPRenderer::load_file(OIIO::string_view filename) const 
{
    std::string filepath = OIIO::Filesystem::searchpath_find(filename, m_search_paths, false);

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


static void
test_group_attributes(OSL::ShaderGroup* group, OSL::ShadingSystem* shadingsys)
{
    int nt = 0;
    if (shadingsys->getattribute(group, "num_textures_needed", nt)) {
        std::cout << "Need " << nt << " textures:\n";
        OSL::ustring* tex = NULL;
        shadingsys->getattribute(group, "textures_needed", OSL::TypeDesc::PTR, &tex);
        for (int i = 0; i < nt; ++i)
            std::cout << "    " << tex[i] << "\n";
        int unk = 0;
        shadingsys->getattribute(group, "unknown_textures_needed", unk);
        if (unk)
            std::cout << "    and unknown textures\n";
    }
    int nclosures = 0;
    if (shadingsys->getattribute(group, "num_closures_needed", nclosures)) {
        std::cout << "Need " << nclosures << " closures:\n";
        OSL::ustring* closures = NULL;
        shadingsys->getattribute(group, "closures_needed", OSL::TypeDesc::PTR,
                                 &closures);
        for (int i = 0; i < nclosures; ++i)
            std::cout << "    " << closures[i] << "\n";
        int unk = 0;
        shadingsys->getattribute(group, "unknown_closures_needed", unk);
        if (unk)
            std::cout << "    and unknown closures\n";
    }
    int nglobals = 0;
    if (shadingsys->getattribute(group, "num_globals_needed", nglobals)) {
        std::cout << "Need " << nglobals << " globals: ";
        OSL::ustring* globals = NULL;
        shadingsys->getattribute(group, "globals_needed", OSL::TypeDesc::PTR,
                                 &globals);
        for (int i = 0; i < nglobals; ++i)
            std::cout << " " << globals[i];
        std::cout << "\n";
    }

    int globals_read  = 0;
    int globals_write = 0;
    shadingsys->getattribute(group, "globals_read", globals_read);
    shadingsys->getattribute(group, "globals_write", globals_write);
    std::cout << "Globals read: (" << globals_read << ") ";
    // for (int i = 1; i < int(OSL::SGBits::last); i <<= 1)
    //     if (globals_read & i)
    //         std::cout << ' ' << shadingsys->globals_name(SGBits(i));
    // std::cout << "\nGlobals written: (" << globals_write << ") ";
    // for (int i = 1; i < int(SGBits::last); i <<= 1)
    //     if (globals_write & i)
    //         std::cout << ' ' << shadingsys->globals_name(SGBits(i));
    // std::cout << "\n";

    int nuser = 0;
    if (shadingsys->getattribute(group, "num_userdata", nuser) && nuser) {
        std::cout << "Need " << nuser << " user data items:\n";
        OSL::ustring* userdata_names  = NULL;
        OSL::TypeDesc* userdata_types = NULL;
        int* userdata_offsets    = NULL;
        bool* userdata_derivs    = NULL;
        shadingsys->getattribute(group, "userdata_names", OSL::TypeDesc::PTR,
                                 &userdata_names);
        shadingsys->getattribute(group, "userdata_types", OSL::TypeDesc::PTR,
                                 &userdata_types);
        shadingsys->getattribute(group, "userdata_offsets", OSL::TypeDesc::PTR,
                                 &userdata_offsets);
        shadingsys->getattribute(group, "userdata_derivs", OSL::TypeDesc::PTR,
                                 &userdata_derivs);
        OSL_DASSERT(userdata_names && userdata_types && userdata_offsets);
        for (int i = 0; i < nuser; ++i)
            std::cout << "    " << userdata_names[i] << ' ' << userdata_types[i]
                      << "  offset=" << userdata_offsets[i]
                      << " deriv=" << userdata_derivs[i] << "\n";
    }
    int nattr = 0;
    if (shadingsys->getattribute(group, "num_attributes_needed", nattr)
        && nattr) {
        std::cout << "Need " << nattr << " attributes:\n";
        OSL::ustring* names  = NULL;
        OSL::ustring* scopes = NULL;
        OSL::TypeDesc* types = NULL;
        shadingsys->getattribute(group, "attributes_needed", OSL::TypeDesc::PTR,
                                 &names);
        shadingsys->getattribute(group, "attribute_scopes", OSL::TypeDesc::PTR,
                                 &scopes);
        shadingsys->getattribute(group, "attribute_types", OSL::TypeDesc::PTR,
                                 &types);
        OSL_DASSERT(names && scopes && types);
        for (int i = 0; i < nattr; ++i)
            std::cout << "    " << names[i] << ' ' << scopes[i] << ' '
                      << types[i] << "\n";

        int unk = 0;
        shadingsys->getattribute(group, "unknown_attributes_needed", unk);
        if (unk)
            std::cout << "    and unknown attributes\n";
    }
    int raytype_queries = 0;
    shadingsys->getattribute(group, "raytype_queries", raytype_queries);
    std::cout << "raytype() query mask: " << raytype_queries << "\n";
}


void
HIPRenderer::prepare_render(RenderState& renderState)
{
    typedef std::vector<uint8_t> Bitcode;

    export_state(renderState);

    auto groupref = renderState.shaderGroup;
    if (!groupref) {
        errhandler().error("No shader group");
        return;
    }
    test_group_attributes(groupref, m_shadingSystem);
    


    // // Load the llvm from the shading system and create the module for further processing
    std::string group_name, init_name, entry_name, fused_name;
    
    m_shadingSystem->getattribute(groupref, "groupname", group_name);
    m_shadingSystem->getattribute(groupref, "group_init_name", init_name);
    m_shadingSystem->getattribute(groupref, "group_entry_name", entry_name);
    m_shadingSystem->getattribute(groupref, "group_fused_name", fused_name);

    std::cout << "Group name: " << group_name << std::endl;
    std::cout << "Init name: " << init_name << std::endl;
    std::cout << "Entry name: " << entry_name << std::endl;
    std::cout << "Fused name: " << fused_name << std::endl;

    // if (options.get_int("saveptx")) {
    //     std::string filename
    //         = OIIO::Strutil::fmt::format("{}_{}.ptx", group_name, mtl_id++);
    //     OIIO::ofstream out;
    //     OIIO::Filesystem::open(out, filename);
    //     out << hip_llvm_gcn;
    // }

    m_shadingSystem->optimize_group(groupref, nullptr, true);

    // Retrieve the compiled ShaderGroup PTX
    std::string hip_llvm_gcn;
    m_shadingSystem->getattribute(groupref, "hip_compiled_version",
                                OSL::TypeDesc::PTR, &hip_llvm_gcn);

    if (hip_llvm_gcn.empty()) {
        errhandler().errorfmt("Failed to generate hip_llvm_gcn for ShaderGroup {}",
                                group_name);
        return;
    }

    const auto& grid_renderer_bc = load_file("hip_grid_renderer.bc");
    const auto& rend_lib_bc = load_file("rend_lib.bc");


    // std::map<std::string, Bitcode> bitcode_map;

    // {

    // }


    
    // use the hiprtcAddNameExpression mechanism to get the global variables
    //hipModuleGetGlobal(&m_function, &m_module, "test", 0);
}

void
HIPRenderer::render(int xres, int yres, RenderState& renderState)
{
    // Launch the kernel
    //hipLaunchKernelGGL(m_function, dim3(1), dim3(1), 0, m_stream);
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

void
HIPRenderer::finalize_pixel_buffer()
{
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    HIP_CHECK(hipMemcpy(tmp_buff.data(),
                        reinterpret_cast<void*>(d_output_buffer),
                        m_xres * m_yres * 3 * sizeof(float),
                        hipMemcpyDeviceToHost));

    OIIO::ImageBuf* buf = outputbuf(0);
    if (buf)
        buf->set_pixels(OIIO::ROI::All(), OIIO::TypeFloat, tmp_buff.data());
}

void
HIPRenderer::camera_params(const OSL::Matrix44& world_to_camera,
                           OSL::ustringhash projection, float hfov,
                           float hither, float yon, int xres, int yres)
{
    //m_world_to_camera  = world_to_camera;
    //m_projection       = projection;
    // m_fov              = hfov;
    // m_pixelaspect      = 1.0f;  // hard-coded
    // m_hither           = hither;
    // m_yon              = yon;
    // m_shutter[0]       = 0.0f;
    // m_shutter[1]       = 1.0f;  // hard-coded
    // float frame_aspect = float(xres) / float(yres) * m_pixelaspect;
    // m_screen_window[0] = -frame_aspect;
    // m_screen_window[1] = -1.0f;
    // m_screen_window[2] = frame_aspect;
    // m_screen_window[3] = 1.0f;
    m_xres             = xres;
    m_yres             = yres;
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
