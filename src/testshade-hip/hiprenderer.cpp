#include "hiprenderer.hpp"
#include "render_params.hpp"
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
std::vector<char>
HIPRenderer::load_file(OIIO::string_view filename) const 
{
    std::string filepath = OIIO::Filesystem::searchpath_find(filename, m_search_paths, false);

    std::cout << "Loading bitcode file from: " << filepath << std::endl;
    if (OIIO::Filesystem::exists(filepath)) 
    {
        const auto size = OIIO::Filesystem::file_size(filepath);
        std::vector<char> bitcode_data(size);
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


static std::vector<const char*> outputs { "Cout" };
void
HIPRenderer::prepare_render(RenderState& renderState)
{
    typedef std::vector<uint8_t> Bitcode;

    export_state(renderState);

    OSL::ShaderGroup* groupref = renderState.shaderGroup;
    if (!groupref) {
        errhandler().error("No shader group");
        return;
    }
    m_shadingSystem->optimize_group(groupref, nullptr, false);
    test_group_attributes(groupref, m_shadingSystem);
    
    {
        m_shadingSystem->attribute(groupref, "renderer_outputs",
                              OSL::TypeDesc(OSL::TypeDesc::STRING, outputs.size()),
                              outputs.data());
    }

    m_shadingSystem->optimize_group(groupref, nullptr, true);

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

    std::string hip_llvm_gcn;
    if (!m_shadingSystem->getattribute(groupref, "hip_compiled_version", OSL::TypeDesc::PTR, &hip_llvm_gcn))
    {
        errhandler().errorfmt("Failed to generate hip_llvm_gcn for ShaderGroup {}",
                                group_name);
        return;

    }

    // save the llvm module for further analysis
    {
        //save the code for further analysis
        std::ofstream outFile("hip_shader_gcn.ll", std::ios::out);
        if (outFile.is_open())
        {
            outFile.write(reinterpret_cast<const char*>(hip_llvm_gcn.data()), hip_llvm_gcn.size());
            outFile.close();
        }
    }


    size_t trampoiline_bitcode_size;
    std::vector<char> trampoline_bitcode;
    {
        std::string init_name_addr = "&" + init_name;
        std::string entry_name_addr = "&" + entry_name;
        std::string fused_name_addr = "&" + fused_name;

        std::stringstream ss;
     ss << "class ShaderGlobals;\n";
    // init 
    ss << "extern \"C\" __device__ void " << init_name
       << "(ShaderGlobals*,void*);\n";
    
    // // entry
    ss << "extern \"C\" __device__ void " << entry_name
       << "(ShaderGlobals*,void*, void*, void*, int, void*);\n";
    
    // // fused
    // ss << "extern \"C\" __device__ void " << fused_name
    //    << "(ShaderGlobals*,void*);\n";

    ss << "extern \"C\" __device__ void __osl__init(ShaderGlobals* sg, void* params)\n"
    << "{\n"
      //  << "printf(\"executing init\\n\");\n"
        << init_name << "(sg, params);\n"
      //  << "printf(\"done wiht init\\n\");\n";
    << "}\n";

    ss << "extern \"C\" __device__ void __osl__entry(ShaderGlobals* sg, void* params, void* userdata, void* outdata, int idx, void* interactive)\n"
    << "{\n"
       // << "printf(\"executing layer\\n\");\n"
        << entry_name << "(sg, params, userdata, outdata, idx, interactive);\n"
    //    << "printf(\"done wiht layer\\n\");\n"
    << "}\n";

    std::cout << "Generated code: " << std::endl;
    std::cout << ss.str() << std::endl;

    
    // dummy kernel
    //ss << "extern \"C\" __global__ void dummy_trampoline() { ShaderGlobals sg; __osl__init(sg, nullptr); __osl_entry(sg, nullptr); __osl_fused(sg); }\n"; 
    


        auto code = ss.str();

        hiprtcProgram program;
        HIPRTC_CHECK(hiprtcCreateProgram(&program, code.data(), "dummy_code", 0, nullptr, nullptr));
        // if (!init_name.empty())
        //     HIPRTC_CHECK(hiprtcAddNameExpression(program, init_name.c_str()));
        // if (!entry_name.empty())    
        //     HIPRTC_CHECK(hiprtcAddNameExpression(program, entry_name.c_str()));
        // if (!fused_name.empty())
        //     HIPRTC_CHECK(hiprtcAddNameExpression(program, fused_name.c_str()));
        // HIPRTC_CHECK(hiprtcAddNameExpression(program, init_name_addr.c_str()));
        // HIPRTC_CHECK(hiprtcAddNameExpression(program, entry_name_addr.c_str()));
         // Add the name expressions to the HIPRTC program
    // hiprtcAddNameExpression(program, init_name_addr.c_str());
    // hiprtcAddNameExpression(program, entry_name_addr.c_str());
    //hiprtcAddNameExpression(program, fused_name_addr.c_str());
        // HIPRTC_CHECK(hiprtcAddNameExpression(program, fused_name_addr.c_str()));

        const char* hip_compile_options[] = { 
                "--offload-arch=gfx1036",
                "-ffast-math", "-fgpu-rdc", "-emit-llvm", "-c",
                "-D__HIP_PLATFORM_AMD",
                "--std=c++17"
                //,"-O0", "-ggdb",
        };

        const int num_compile_flags = int(sizeof(hip_compile_options) / sizeof(hip_compile_options[0]));
        hiprtcResult compileResult = hiprtcCompileProgram(program, num_compile_flags, hip_compile_options);
        if (compileResult != HIPRTC_SUCCESS)
        {
            size_t hip_log_size;
            HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &hip_log_size));
            std::vector<char> hip_log(hip_log_size + 1);
            HIPRTC_CHECK(hiprtcGetProgramLog(program, hip_log.data()));
            hip_log.back() = 0;
            std::stringstream ss;
            ss << "hiprtcCompileProgram failure for: \n"
                << hip_log.data();
            errhandler().errorfmt("Failed to compile the llvm code\n {} ", ss.str());        

            HIPRTC_CHECK(hiprtcDestroyProgram(&program));
            return;

        }

       
        HIPRTC_CHECK(hiprtcGetBitcodeSize(program, &trampoiline_bitcode_size));
        trampoline_bitcode.resize(trampoiline_bitcode_size);
        HIPRTC_CHECK(hiprtcGetBitcode(program, trampoline_bitcode.data()));
        HIPRTC_CHECK(hiprtcDestroyProgram(&program));        

    }
    
    // Workaround because this should be linked to the hip_llvm_gcn during the optimization / jit stage of the group ref.
    char* hip_llvm_shaderops {nullptr};
    int   hip_llvm_shaderops_size {0};
    if (!m_shadingSystem->getattribute("shadeops_hip_llvm", OSL::TypeDesc::PTR, &hip_llvm_shaderops))
    {
        errhandler().errorfmt("Failed to generate hip_llvm_shaderps for ShaderGroup {}",
                                group_name);
        return;
    }

     if (!m_shadingSystem->getattribute("shadeops_hip_llvm_size", OSL::TypeDesc::INT, &hip_llvm_shaderops_size))
    {
        errhandler().errorfmt("Error getting hip llvm ops size");
        return;

    };

    

    const auto& grid_renderer_bc = load_file("hip_grid_renderer.bc");
    const auto& rend_lib_bc = load_file("rend_lib.bc");

    // can we linki it here?
    hiprtcLinkState linkState;
    HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkState));

    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE,  hip_llvm_shaderops, hip_llvm_shaderops_size, "hip_llvm_ops", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void*)grid_renderer_bc.data(), grid_renderer_bc.size(), "grid_renderer", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void*)rend_lib_bc.data(), rend_lib_bc.size(), "my_render_lib", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, trampoline_bitcode.data(), trampoline_bitcode.size(), "trampoline", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void*)hip_llvm_gcn.data(), hip_llvm_gcn.size(), "osl_shader", 0, nullptr, nullptr));
    //HIPRTC_CHECK(hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "hip_shader_gcn_used.ll", 0, nullptr, nullptr));

    //link everything together
    std::vector<uint8_t> hip_fatbin; 

    {
        void* code { nullptr };
        size_t size { 0 };

        HIPRTC_CHECK(hiprtcLinkComplete(linkState, &code, &size));

        //JPA: This is stupid and not intuitive. the owner of the code_ptr and size is the linker. 
        // Copy the data before destroying the linker
        if (size == 0)
        {
            errhandler().errorfmt("HIPRTC Error: the HIP fatbin size is 0");
            HIPRTC_CHECK(hiprtcLinkDestroy(linkState)); 
            return;
        }

        std::cout << "Program compiled successfully size: " << size << " B" << std::endl;

        hip_fatbin.resize(size);
        memcpy(hip_fatbin.data(), code, size);

        //save the code for further analysis
        std::ofstream outFile("hip_fatbin.bin", std::ios::out | std::ios::binary);
        if (outFile.is_open())
        {
            outFile.write(reinterpret_cast<const char*>(hip_fatbin.data()), hip_fatbin.size());
            outFile.close();
        }
        else{
            errhandler().errorfmt("Proble with opening hip_fatbin.bin file for writing compiled code");
        }

        HIPRTC_CHECK(hiprtcLinkDestroy(linkState)); 
    }


    HIP_CHECK(hipModuleLoadData(&m_module, hip_fatbin.data()));

    HIP_CHECK(hipModuleGetFunction(&m_function_shade, m_module, "shade"));
}

void
HIPRenderer::render(int xres, int yres, RenderState& renderState)
{
    d_output_buffer = device_alloc(xres * yres * 4 * sizeof(float));
    d_launch_params = device_alloc(sizeof(testshadeHIP::RenderParams));

    m_xres = xres;
    m_yres = yres;

    testshadeHIP::RenderParams params;
    params.invw  = 1.0f / std::max(1, m_xres - 1);
    params.invh  = 1.0f / std::max(1, m_yres - 1);
    params.flipv = false; /* I don't see flipv being initialized anywhere */
    params.output_buffer           = d_output_buffer;
    params.osl_printf_buffer_start = reinterpret_cast<uint64_t>(d_osl_printf_buffer);
    // maybe send buffer size to CUDA instead of the buffer 'end'
    params.osl_printf_buffer_end = reinterpret_cast<uint64_t>(d_osl_printf_buffer) + OSL_PRINTF_BUFFER_SIZE;
    params.color_system          = d_color_system;
    params.test_str_1            = test_str_1;
    params.test_str_2            = test_str_2;
    params.object2common         = d_object2common;
    params.shader2common         = d_shader2common;
    params.num_named_xforms      = m_num_named_xforms;
    params.xform_name_buffer     = d_xform_name_buffer;
    params.xform_buffer          = d_xform_buffer;
    //params.fused_callable        = m_fused_callable;

    copy_to_device(d_launch_params, &params, sizeof(testshadeHIP::RenderParams));

    hipDeviceptr_t d_render_globals;
    size_t bytes {0};
    HIP_CHECK(hipModuleGetGlobal(&d_render_globals, &bytes, m_module, "gc_render_params"));

    HIP_CHECK(hipMemcpy(d_render_globals, &d_launch_params, sizeof(hipDeviceptr_t), hipMemcpyHostToDevice));

    void* args[] = { &d_output_buffer, &m_xres, &m_yres, &d_launch_params};

    uint3 blockSize {8, 8, 1};

    uint3 gridSize = { (m_xres + blockSize.x - 1) / blockSize.x,
                       (m_yres + blockSize.y - 1) / blockSize.y,
                       1};

    std::cout << "Xres: " << m_xres << " Yres: " << m_yres << std::endl;
    std::cout << "Block size: " << blockSize.x << " " << blockSize.y << " " << blockSize.z << std::endl;
    std::cout << "Grid size: " << gridSize.x << " " << gridSize.y << " " << gridSize.z << std::endl;


    hipModuleLaunchKernel(m_function_shade, 
        gridSize.x, gridSize.y, gridSize.z, 
        blockSize.x, blockSize.y, blockSize.z, 
        0, 
        m_stream, args, nullptr);

    std::vector<uint8_t> printf_buffer(OSL_PRINTF_BUFFER_SIZE);
    HIP_CHECK(hipMemcpy(printf_buffer.data(), d_osl_printf_buffer, OSL_PRINTF_BUFFER_SIZE, hipMemcpyDeviceToHost));

    processPrintfBuffer(printf_buffer.data(), OSL_PRINTF_BUFFER_SIZE);


}

void HIPRenderer::processPrintfBuffer(void* buffer_data, size_t buffer_size)
{
    const uint8_t* ptr = reinterpret_cast<uint8_t*>(buffer_data);
    // process until
    std::string fmt_string;
    size_t total_read = 0;
    while (total_read < buffer_size) {
        size_t src = 0;
        // set max size of each output string
        const size_t BufferSize = 4096;
        char buffer[BufferSize];
        size_t dst = 0;
        // get hash of the format string
        uint64_t fmt_str_hash = *reinterpret_cast<const uint64_t*>(&ptr[src]);
        src += sizeof(uint64_t);
        // get sizeof the argument stack
        uint64_t args_size = *reinterpret_cast<const uint64_t*>(&ptr[src]);
        src += sizeof(size_t);
        uint64_t next_args = src + args_size;

        // have we reached the end?
        if (fmt_str_hash == 0)
            break;
        const char* format = OSL::ustring::from_hash(fmt_str_hash).c_str();
        OSL_ASSERT(format != nullptr
                   && "The string should have been a valid ustring");
        const size_t len = strlen(format);

        for (size_t j = 0; j < len; j++) {
            // If we encounter a '%', then we'll copy the format string to 'fmt_string'
            // and provide that to printf() directly along with a pointer to the argument
            // we're interested in printing.
            if (format[j] == '%') {
                fmt_string            = "%";
                bool format_end_found = false;
                for (size_t i = 0; !format_end_found; i++) {
                    j++;
                    fmt_string += format[j];
                    switch (format[j]) {
                    case '%':
                        // seems like a silly to print a '%', but it keeps the logic parallel with the other cases
                        dst += snprintf(&buffer[dst], BufferSize - dst, "%s",
                                        fmt_string.c_str());
                        format_end_found = true;
                        break;
                    case 'd':
                    case 'i':
                    case 'o':
                    case 'x':
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(),
                                        *reinterpret_cast<const int*>(
                                            &ptr[src]));
                        src += sizeof(int);
                        format_end_found = true;
                        break;
                    case 'f':
                    case 'g':
                    case 'e':
                        // TODO:  For OptiX llvm_gen_printf() aligns doubles on sizeof(double) boundaries -- since we're not
                        // printing from the device anymore, maybe we don't need this alignment?
                        src = (src + sizeof(double) - 1)
                              & ~(sizeof(double) - 1);
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(),
                                        *reinterpret_cast<const double*>(
                                            &ptr[src]));
                        src += sizeof(double);
                        format_end_found = true;
                        break;
                    case 's':
                        src = (src + sizeof(uint64_t) - 1)
                              & ~(sizeof(uint64_t) - 1);
                        uint64_t str_hash = *reinterpret_cast<const uint64_t*>(
                            &ptr[src]);
                        OSL::ustring str = OSL::ustring::from_hash(str_hash);
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(), str.c_str());
                        src += sizeof(uint64_t);
                        format_end_found = true;
                        break;

                        break;
                    }
                }
            } else {
                buffer[dst++] = format[j];
            }
        }
        // realign
        ptr = ptr + next_args;
        total_read += next_args;

        buffer[dst++] = '\0';
        OIIO::print("{}", buffer);
    }
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
    size_t sz = sizeof(uint64_t) * xform_name_buffer.size();
    d_xform_name_buffer = device_alloc(sz);
    copy_to_device(d_xform_name_buffer, xform_name_buffer.data(), sz);
    
    sz = sizeof(OSL::Matrix44) * xform_buffer.size();
    d_xform_buffer = device_alloc(sz);
    copy_to_device(d_xform_buffer, xform_buffer.data(), sz);

    m_num_named_xforms = xform_name_buffer.size();
}

void
HIPRenderer::finalize_pixel_buffer()
{
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    HIP_CHECK(hipMemcpy(tmp_buff.data(),
                        d_output_buffer,
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

bool
HIPRenderer::initialize_render_parameters()
{
    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    OSL::ustring userdata_str1("ud_str_1");
    OSL::ustring userdata_str2("userdata string");

    // Store the user-data
    test_str_1 = userdata_str1.hash();
    test_str_2 = userdata_str2.hash();

    {
        char* colorSys            = nullptr;
        long long cpuDataSizes[2] = { 0, 0 };
        // TODO: utilize opaque shading state uniform data structure
        // which has a device friendly representation this data
        // and is already accessed directly by opcolor and opmatrix for
        // the cpu (just remove optix special casing)
        if (!m_shadingSystem->getattribute("colorsystem", OSL::TypeDesc::PTR,
                                      (void*)&colorSys)
            || !m_shadingSystem->getattribute("colorsystem:sizes",
                                         OSL::TypeDesc(OSL::TypeDesc::LONGLONG, 2),
                                         (void*)&cpuDataSizes)
            || !colorSys || !cpuDataSizes[0]) {
            errhandler().errorfmt("No colorsystem available.");
            return false;
        }

        auto cpuDataSize = cpuDataSizes[0];
        auto numStrings  = cpuDataSizes[1];
        
        // Get the size data-size, minus the ustring size
        const size_t podDataSize = cpuDataSize
                                   - sizeof(OSL::ustringhash) * numStrings;

        d_color_system = device_alloc(podDataSize + sizeof(uint64_t) * numStrings);
        copy_to_device(d_color_system, colorSys, podDataSize);

        d_osl_printf_buffer = device_alloc(OSL_PRINTF_BUFFER_SIZE);
        HIP_CHECK(hipMemset(d_osl_printf_buffer, 0, OSL_PRINTF_BUFFER_SIZE));
        
        
        // Transforms
        d_object2common = device_alloc(sizeof(OSL::Matrix44));
        copy_to_device(d_object2common, &m_object2common, sizeof(OSL::Matrix44));
        
        d_shader2common = device_alloc(sizeof(OSL::Matrix44));
        copy_to_device(d_shader2common, &m_shader2common, sizeof(OSL::Matrix44));
        
        // then copy the device string to the end, first strings starting at dataPtr - (numStrings)
        // FIXME -- Should probably handle alignment better.
        const OSL::ustringhash* cpuStringHash
            = (const OSL::ustringhash*)(colorSys + (cpuDataSize - sizeof(OSL::ustringhash) * numStrings));

        hipDeviceptr_t gpuStrings = static_cast<hipDeviceptr_t>(static_cast<char*>(d_color_system) + podDataSize);

        for (const OSL::ustringhash* end = cpuStringHash + numStrings; cpuStringHash < end; ++cpuStringHash) {
                OSL::ustringhash_pod devStr = cpuStringHash->hash();
                copy_to_device(gpuStrings, &devStr, sizeof(devStr));
                gpuStrings = static_cast<hipDeviceptr_t>(static_cast<char*>(gpuStrings) + sizeof(devStr));
        }
    }
    return true;
}
