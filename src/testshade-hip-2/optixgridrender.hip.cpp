// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <hip/hip_runtime.h>
#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslconfig.h>

#include "optixgridrender.h"
#include "render_params.hip.h"

#include <hip/hiprtc.h>
#include <filesystem>

// The pre-compiled renderer support library LLVM bitcode is embedded
// into the executable and made available through these variables.
extern int rend_lib_llvm_compiled_ops_size;
extern unsigned char rend_lib_llvm_compiled_ops_block[];

// The entry point for OptiX Module creation changed in OptiX 7.7
// #if OPTIX_VERSION < 70700
// const auto optixModuleCreateFn = optixModuleCreateFromPTX;
// #else
// const auto optixModuleCreateFn = optixModuleCreate;
// #endif


using namespace testshade;


OSL_NAMESPACE_ENTER


#define CUDA_CHECK(call)                                               \
    {                                                                  \
        hipError_t res = call;                                        \
        if (res != hipSuccess) {                                      \
            print(stderr,                                              \
                  "[CUDA ERROR] Cuda call '{}' failed with error:"     \
                  " {} ({}:{})\n",                                     \
                  #call, hipGetErrorString(res), __FILE__, __LINE__); \
        }                                                              \
    }

// Define optix_check as CUDA_CHECK for now
#define OPTIX_CHECK(call) CUDA_CHECK(call)

#define HIPRTC_CHECK(call)                                              \
    {                                                                   \
        hiprtcResult res = call;                                        \
        if (res != HIPRTC_SUCCESS) {                                     \
            print(stderr,                                               \
                  "[HIPRTC ERROR] HIPRTC call '{}' failed with error:"   \
                  " {} ({}:{})\n",                                       \
                  #call, hiprtcGetErrorString(res), __FILE__, __LINE__); \
        }                                                               \
    }

#define OPTIX_CHECK_MSG(call, msg)                                         \
    {                                                                      \
        hipError_t res = call;                                            \
        if (res != hipSuccess) {                                        \
            print(stderr,                                                  \
                  "[OPTIX ERROR] OptiX call '{}' failed with error:"       \
                  " {} ({}:{})\nMessage: {}\n",                            \
                  #call, hipGetErrorString(res), __FILE__, __LINE__, msg); \
            exit(1);                                                       \
        }                                                                  \
    }

#define CUDA_SYNC_CHECK()                                                  \
    {                                                                      \
        hipDeviceSynchronize();                                           \
        hipError_t error = hipGetLastError();                            \
        if (error != hipSuccess) {                                        \
            print(stderr, "error ({}: line {}): {}\n", __FILE__, __LINE__, \
                  hipGetErrorString(error));                              \
            exit(1);                                                       \
        }                                                                  \
    }


// static void
// context_log_cb(unsigned int level, const char* tag, const char* message,
//                void* /*cbdata */)
// {
//         std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
// }



OptixGridRenderer::OptixGridRenderer()
{
    // Initialize CUDA
    hipInit(0);

    hipCtx_t cuCtx = nullptr;  // zero means take the current context
    hipCtxCreate(&cuCtx, 0, 0);

    // optix::OptixDeviceContextOptions ctx_options = {};
    // ctx_options.logCallbackFunction       = context_log_cb;
    // ctx_options.logCallbackLevel          = 4;

    // OPTIX_CHECK(optixInit());
    // OPTIX_CHECK(optix::OptixDeviceContextCreate(cuCtx, &ctx_options, &m_optix_ctx));

    CUDA_CHECK(hipSetDevice(0));
    CUDA_CHECK(hipStreamCreate(&m_cuda_stream));

    m_fused_callable = false;
    if (const char* fused_env = getenv("TESTSHADE_FUSED"))
        m_fused_callable = atoi(fused_env);
}



hipDeviceptr_t
OptixGridRenderer::device_alloc(size_t size)
{
    hipDeviceptr_t ptr       = nullptr;
    hipError_t res = hipMalloc(&ptr, size);
    if (res != hipSuccess) {
        errhandler().errorfmt("hipMalloc({}) failed with error: {}\n", size,
                              hipGetErrorString(res));
    }
    m_ptrs_to_free.push_back(ptr);
    return ptr;
}


void
OptixGridRenderer::device_free(hipDeviceptr_t ptr)
{
    hipError_t res = hipFree(ptr);
    if (res != hipSuccess) {
        errhandler().errorfmt("hipFree() failed with error: {}\n",
                              hipGetErrorString(res));
    }
}


hipDeviceptr_t
OptixGridRenderer::copy_to_device(hipDeviceptr_t dst_device, const void* src_host,
                                  size_t size)
{
    hipError_t res = hipMemcpy(dst_device, src_host, size,
                                 hipMemcpyHostToDevice);
    if (res != hipSuccess) {
        errhandler().errorfmt(
            "hipMemcpy host->device of size {} failed with error: {}\n", size,
            hipGetErrorString(res));
    }
    return dst_device;
}



std::string
OptixGridRenderer::load_ptx_file(string_view filename, const std::vector<std::string>& search_paths)
{
    
    std::vector<std::string> paths
        = { OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()),
            PTX_PATH };

    paths.insert(paths.end(), search_paths.begin(), search_paths.end());
    
    std::string filepath = OIIO::Filesystem::searchpath_find(filename, paths, false);
    if (OIIO::Filesystem::exists(filepath)) {
        std::string ptx_string;
        if (OIIO::Filesystem::read_text_file(filepath, ptx_string))
            return ptx_string;
    }
    errhandler().severefmt("Unable to load {}", filename);
    return {};
}



OptixGridRenderer::~OptixGridRenderer()
{
    if (m_optix_ctx)
        CUDA_CHECK(hipCtxDestroy(m_optix_ctx));
    for (hipDeviceptr_t ptr : m_ptrs_to_free)
        hipFree(reinterpret_cast<void*>(ptr));
    for (hipArray_t arr : m_arrays_to_free)
        hipFreeArray(arr);
}



void
OptixGridRenderer::init_shadingsys(ShadingSystem* ss)
{
    shadingsys = ss;
}



bool
OptixGridRenderer::init_optix_context(int xres OSL_MAYBE_UNUSED,
                                      int yres OSL_MAYBE_UNUSED)
{
    if (!options.get_int("no_rend_lib_bitcode")) {
        shadingsys->attribute("lib_bitcode",
                              { OSL::TypeDesc::UINT8,
                                rend_lib_llvm_compiled_ops_size },
                              rend_lib_llvm_compiled_ops_block);
    }
    if (options.get_int("optix_register_inline_funcs")) {
        register_inline_functions();
    }
    return true;
}



bool
OptixGridRenderer::synch_attributes()
{
    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    ustring userdata_str1("ud_str_1");
    ustring userdata_str2("userdata string");

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
        if (!shadingsys->getattribute("colorsystem", TypeDesc::PTR,
                                      (void*)&colorSys)
            || !shadingsys->getattribute("colorsystem:sizes",
                                         TypeDesc(TypeDesc::LONGLONG, 2),
                                         (void*)&cpuDataSizes)
            || !colorSys || !cpuDataSizes[0]) {
            errhandler().errorfmt("No colorsystem available.");
            return false;
        }

        auto cpuDataSize = cpuDataSizes[0];
        auto numStrings  = cpuDataSizes[1];

        // Get the size data-size, minus the ustring size
        const size_t podDataSize = cpuDataSize
                                   - sizeof(ustringhash) * numStrings;

        d_color_system = device_alloc(podDataSize
                                      + sizeof(uint64_t) * numStrings);
        CUDA_CHECK(hipMemcpy(reinterpret_cast<void*>(d_color_system), colorSys,
                              podDataSize, hipMemcpyHostToDevice));
        d_osl_printf_buffer = device_alloc(OSL_PRINTF_BUFFER_SIZE);
        CUDA_CHECK(hipMemset(reinterpret_cast<void*>(d_osl_printf_buffer), 0,
                              OSL_PRINTF_BUFFER_SIZE));

        // Transforms
        d_object2common = device_alloc(sizeof(OSL::Matrix44));
        CUDA_CHECK(hipMemcpy(reinterpret_cast<void*>(d_object2common),
                              &m_object2common, sizeof(OSL::Matrix44),
                              hipMemcpyHostToDevice));
        d_shader2common = device_alloc(sizeof(OSL::Matrix44));
        CUDA_CHECK(hipMemcpy(reinterpret_cast<void*>(d_shader2common),
                              &m_shader2common, sizeof(OSL::Matrix44),
                              hipMemcpyHostToDevice));

        // then copy the device string to the end, first strings starting at dataPtr - (numStrings)
        // FIXME -- Should probably handle alignment better.
        const ustringhash* cpuStringHash
            = (const ustringhash*)(colorSys
                                   + (cpuDataSize
                                      - sizeof(ustringhash) * numStrings));
        // compute offset 
        std::ptrdiff_t offset = podDataSize;
        char* d_color_system_char = reinterpret_cast<char*>(d_color_system);
        char* gpuStrings = d_color_system_char + offset;
        // hipDeviceptr_t gpuStrings = reinterpret_cast<hipDeviceptr_t>(gpuStrings_char);

        for (const ustringhash* end = cpuStringHash + numStrings;
             cpuStringHash < end; ++cpuStringHash) {
            ustringhash_pod devStr = cpuStringHash->hash();
            CUDA_CHECK(hipMemcpy(reinterpret_cast<void*>(gpuStrings), &devStr,
                                  sizeof(devStr), hipMemcpyHostToDevice));
            gpuStrings += sizeof(ustringhash_pod);
        }
    }
    return true;
}

// static bool hiprtcCompileRenderer(const char* source, const char* name, 
//                 int num_headers, std::vector<const char*> headers, 
//                 std::vector<const char*> include_names, 
//                 std::vector<std::string> include_paths, 
//                 OIIO::ErrorHandler& errhandler)
// {
//     hiprtcProgram prog;
//     hiprtcResult res = hiprtcCreateProgram(&prog, source, name, num_headers, headers.data(), include_names.data());
//     if (res != HIPRTC_SUCCESS) {
//         errhandler.errorfmt("hiprtcCreateProgram failed with error: {}\n", hiprtcGetErrorString(res));
//         return false;
//     }

//     // Prepare compiler options

//     std::vector<std::string> include_options;

//     std::transform(include_paths.begin(), include_paths.end(), std::back_inserter(include_options),
//                [](const std::string& path) { return std::string("-I") + path; });
    
   
//     std::vector<const char*> options { 
//         "-std=c++17",
//         "-fgpu-rdc",
//         "-emit-llvm",
//         "-ffast-math",
//         "--offload-arch=" TEST_SHADE_TARGET,
//         "--offload-device-only",
//         "-D__HIP_PLATFORM_AMD__",
//         "-D__HIP_DEVICE_COMPILE__",
//         "-DUSE_HIP",
//         "-DHIP",
//         "-DOSL_USE_FAST_MATH=1"
//     };

//     std::transform(include_options.begin(), include_options.end(), std::back_inserter(options),
//                [](const std::string& option) { return option.c_str(); });

//     hiprtcResult compile_res = hiprtcCompileProgram(prog, options.size(), options.data());
//     if (compile_res != HIPRTC_SUCCESS) {
//         size_t log_size;
//         hiprtcGetProgramLogSize(prog, &log_size);
//         std::vector<char> log(log_size);
//         hiprtcGetProgramLog(prog, log.data());
//         errhandler.errorfmt("hiprtcCompileProgram failed with error: {}\n{}", hiprtcGetErrorString(compile_res), log.data());
//         return false;
//     }
//     return true;
// }
static std::string generate_source(const std::string& init_name, const std::string& layer_name, const std::string& fused_name, const std::string& group_name)
{
    /**
     *  The generated header should look like this:
        #include "osl_wrapper.hip.h"


        // here we define the extern "C" functions that are called from the OSL generated code

        extern "C" __device__ void __direct_callable__osl_init_group_unnamed_group_1(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
        extern "C" __device__ void __direct_callable__osl_entry_group_unnamed_group_1_name_test_0(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);
        extern "C" __device__ void __direct_callable__fused_unnamed_group_1_name_test_0(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);


        extern "C" __device__ __constant__ OslDeviceFunction init_func = __direct_callable__osl_init_group_unnamed_group_1;
        extern "C" __device__ __constant__ OslDeviceFunction entry_func = __direct_callable__osl_entry_group_unnamed_group_1_name_test_0;
        extern "C" __device__ __constant__ OslDeviceFunction fused_func = __direct_callable__fused_unnamed_group_1_name_test_0;

     */
    std::string source = "#include \"osl_wrapper.hip.h\"\n\n";
    source += "\n\n";
    source += "extern \"C\" __device__ void " + init_name + "(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);\n";
    source += "extern \"C\" __device__ void " + layer_name + "(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);\n";
    source += "extern \"C\" __device__ void " + fused_name + "(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);\n";
    source += "\n\n";
    source += "extern \"C\" __device__ __constant__ OslDeviceFunction "+ group_name + "_init_func = " + init_name + ";\n";
    source += "extern \"C\" __device__ __constant__ OslDeviceFunction "+ group_name + "_entry_func = " + layer_name + ";\n";
    source += "extern \"C\" __device__ __constant__ OslDeviceFunction "+ group_name + "_fused_func = " + fused_name + ";\n";


    return source;
}

static std::string generate_header(const std::string& init_name, const std::string& layer_name, const std::string& fused_name, const std::string& group_name)
{
    /**
     *  The generated header should look like this:
        #ifndef OSL_WRAPPER_HIP_H
        #define OSL_WRAPPER_HIP_H

        using OslDeviceFunction = void (*)(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);

        struct OslShaderLayer {
            OslDeviceFunction init_func {nullptr};
            OslDeviceFunction entry_func {nullptr};
            OslDeviceFunction fused_func {nullptr};
        };

        struct OslFunctionTable {
            OslShaderLayer* layers;
            int num_layers;
        }
    */

    std::string header = R"(
#ifndef OSL_WRAPPER_HIP_H
#define OSL_WRAPPER_HIP_H

using OslDeviceFunction = void (*)(void* sg, void* params, void* userdata, void* outdata, int idx, void* interactive);

struct OslShaderLayer {
    OslDeviceFunction init_func {nullptr};
    OslDeviceFunction entry_func {nullptr};
    OslDeviceFunction fused_func {nullptr};
};

struct OslFunctionTable {
    OslShaderLayer* layers;
    int num_layers;
};

#endif // OSL_WRAPPER_HIP_H
)";

    return header;
}


static bool compile(const std::string& source, const std::string& header, const std::string& name, std::string& out_bc)
{
    std::string source_name = name + ".hip.cpp";
    std::string header_name = "osl_wrapper.hip.h";

    std::ofstream source_file(source_name);
    source_file << source;
    source_file.close();

    std::ofstream header_file(header_name);
    header_file << header;
    header_file.close();
 
    std::string include_path = "-I" + OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path());
    std::string command = "hipcc -std=c++17 -fgpu-rdc -emit-llvm -ffast-math --offload-arch=" TEST_SHADE_TARGET " --offload-device-only -D__HIP_PLATFORM_AMD__ -D__HIP_DEVICE_COMPILE__ -DUSE_HIP -DHIP -DOSL_USE_FAST_MATH=1 " + source_name + " -o " + name + ".bc " + include_path;
    int status = system(command.c_str());

    std::ifstream file(name + ".bc");
    if (!file.good()) {
        std::cerr << "Failed to compile " << name << std::endl;
        return false;
    }

    //read the file
    std::string bc((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    out_bc = bc;
    
    return status == 0;
}

static bool compile_wrapper(const std::string& init_name, const std::string& layer_name, const std::string& fused_name, const std::string& group_name, std::string& out_bc)
{
    std::string header = generate_header(init_name, layer_name, fused_name, group_name);
    std::string source = generate_source(init_name, layer_name, fused_name, group_name);

    std::string name = "wrapper_" + group_name + ".hip.cpp";
    bool status = compile(source, header, name, out_bc);
    return status;
}


bool
OptixGridRenderer::make_optix_materials()
{

    // Optimize each ShaderGroup in the scene, and use the resulting
    // PTX to create OptiX Programs which can be called by the closest
    // hit program in the wrapper to execute the compiled OSL shader.
   
    // char msg_log[8192];
    // size_t sizeof_msg_log;
    std::cout << "Compiling renderer source code with HIPRTC\n";
    std::cout << "Current directory:" << OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()) << std::endl;
    {
        
        // prepare search paths
        std::vector<std::string> paths
            = { OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()),
                TEST_SHADE_SRC_DIR,
                TEST_SHADE_SRC_DIR "/cuda",
                TEST_SHADE_SRC_DIR "/../include",
                TEST_SHADE_INSTALL_DIR  "/include",
                TEST_SHADE_INSTALL_DIR  "/include/OSL",
                TEST_SHADE_3RD_PARTY_DIR,
                PTX_PATH };
        

        std::cout << "Searching path for includes is:\n";
        for (const auto& path : paths) {
            if (std::filesystem::exists(path)) {
                std::cout << path << std::endl;
            }
            else {
                std::cout << "Path does not exist: " << path << std::endl;
            }
        }

        // load the optix grid renderer source code
        std::string program_source = load_ptx_file("optix_grid_renderer.hip.cu", paths);
        if (program_source.empty()) {
            errhandler().severefmt("Could not find the source code for the renderer");
            return false;
        }
        // it is easier to provide the include paths instead of reading all the sources and headers
        //hiprtcCompileRenderer(program_source.c_str(), "optix_grid_renderer.hip.cpp", 0, {}, {}, paths, errhandler());
        
    }


    // Renderer
    std::string name = "optix_grid_renderer.bc";
    std::string program_ptx = load_ptx_file(name);

    if (program_ptx.empty()) {
        errhandler().severefmt("Could not find PTX for the raygen program");
        return false;
    }

    // Shadeops
    const char* shadeops_ptx = nullptr;
    // jpa this should be changed to shadeops_hip_bitcode
    shadingsys->getattribute("shadeops_hip_llvm", OSL::TypeDesc::PTR,
                             &shadeops_ptx);

    int shadeops_ptx_size = 0;
    shadingsys->getattribute("shadeops_hip_llvm_size", OSL::TypeDesc::INT,
                             &shadeops_ptx_size);

    if (shadeops_ptx == nullptr || shadeops_ptx_size == 0) {
        errhandler().severefmt(
            "Could not retrieve bitcode for the shadeops library");
        return false;
    }

    /* Render Library 
    It is created based on the 
        - cuda/rend_lib.hip.cu - this overrides the osl functions and multiple other required functions i.e: closure_component_allot
        - cuda/rend_lib.hip.h - defines the shader globals struct
        - rs_simplerend.hip.cpp - this defines multiple rs_* functions
        - raytracer.h - camera, sphere, shape, scene etc.
    */
    std::string rend_libName = "rend_lib_testshade.bc";
    std::string rend_lib_ptx = load_ptx_file(rend_libName);

    if (rend_lib_ptx.empty()) {
        errhandler().severefmt("Could not find BC for the renderer library");
        return false;
    }

    
    //int callables = m_fused_callable ? 1 : 2;

    std::vector<void*> material_interactive_params;

    // Stand-in: names of shader outputs to preserve the code
    std::vector<const char*> outputs { "Cout" };
    // here create a shader bc for each shader group
    std::vector<std::string> shader_bitcodes(shaders().size());
    std::vector<std::string> shader_wrappers(shaders().size());
    std::vector<std::string> shader_names(shaders().size());
    std::vector<std::string> shader_init_names(shaders().size());
    std::vector<std::string> shader_entry_names(shaders().size());
    std::vector<std::string> shader_fused_names(shaders().size());

    int material_layer_id = 0;
    for (const auto& groupref : shaders()) 
    {
        shadingsys->attribute(groupref.get(), "renderer_outputs",
                              TypeDesc(TypeDesc::STRING, outputs.size()),
                              outputs.data());

        shadingsys->optimize_group(groupref.get(), nullptr);

        if (!shadingsys->find_symbol(*groupref.get(), ustring(outputs[0]))) 
        {
            // FIXME: This is for cases where testshade is run with 1x1 resolution
            //        Those tests may not have a Cout parameter to write to.
            if (m_xres > 1 && m_yres > 1) {
                errhandler().warningfmt(
                    "Requested output '{}', which wasn't found", outputs[0]);
            }
        }
        std::string group_name, init_name, entry_name, fused_name;
        shadingsys->getattribute(groupref.get(), "groupname", group_name);
        shadingsys->getattribute(groupref.get(), "group_init_name", init_name);
        shadingsys->getattribute(groupref.get(), "group_entry_name", entry_name);
        shadingsys->getattribute(groupref.get(), "group_fused_name", fused_name);

        std::cout << "Init: " << init_name << std::endl;
        std::cout << "Entry: " << entry_name << std::endl;
        std::cout << "Fused: " << fused_name << std::endl;

        std::string shader_wrapper_bc;
        compile_wrapper(init_name, entry_name, fused_name, group_name, shader_wrapper_bc);
        shader_wrappers[material_layer_id] = std::move(shader_wrapper_bc);

        ShaderWrapperInfo info;
        info.init_name = init_name;
        info.entry_name = entry_name;
        info.fused_name = fused_name;
        info.group_name = group_name;
        m_shader_wrappers.push_back(info);

        // Retrieve the compiled ShaderGroup PTX
        std::string osl_ptx;
        shadingsys->getattribute(groupref.get(), "hip_compiled_version", OSL::TypeDesc::PTR, &osl_ptx);
        if (osl_ptx.empty()) {
        errhandler().errorfmt("Failed to generate PTX for ShaderGroup {}",
                                group_name);
        return false;
        }
        
        if (options.get_int("saveptx")) {
        std::string filename
            = OIIO::Strutil::fmt::format("{}_{}.bc", group_name, material_layer_id++);
        OIIO::ofstream out;
        OIIO::Filesystem::open(out, filename);
        out << osl_ptx;
        }
      
        shader_bitcodes[material_layer_id] = std::move(osl_ptx);
        shader_names[material_layer_id] = std::move(group_name);
        shader_init_names[material_layer_id] = std::move(init_name);
        shader_entry_names[material_layer_id] = std::move(entry_name);
        shader_fused_names[material_layer_id] = std::move(fused_name);

        void* interactive_params = nullptr;
        shadingsys->getattribute(groupref.get(), "device_interactive_params",
                                 TypeDesc::PTR, &interactive_params);
        if (nullptr != interactive_params)
        {
            std::cout << "Interactive params found for " << group_name << std::endl;
        }
        material_interactive_params.push_back(interactive_params);
    }

    // link everything together
    std::vector<uint8_t> hip_fatbin; 
    {
        hiprtcLinkState linkState;
        HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkState));

        HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, program_ptx.data(), program_ptx.size(), name.c_str(), 0, nullptr, nullptr));
        HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void*)shadeops_ptx, shadeops_ptx_size, "hip_llvm_ops", 0, nullptr, nullptr));
        HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, (void*)rend_lib_ptx.data(), rend_lib_ptx.size(), rend_libName.c_str(), 0, nullptr, nullptr));

        for (size_t i = 0; i < shader_bitcodes.size(); ++i)
        {
            HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, shader_bitcodes[i].data(), shader_bitcodes[i].size(), shader_names[i].c_str(), 0, nullptr, nullptr));
        }

        for (size_t i = 0; i < shader_wrappers.size(); ++i)
        {
            HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, shader_wrappers[i].data(), shader_wrappers[i].size(), shader_names[i].c_str(), 0, nullptr, nullptr));
        }

      

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
                return false;
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
    }

    CUDA_CHECK(hipModuleLoadData(&m_module, hip_fatbin.data()));

    CUDA_CHECK(hipModuleGetFunction(&m_function_init_globals, m_module, "__raygen__setglobals"));
    CUDA_CHECK(hipModuleGetFunction(&m_function_shade, m_module, "__raygen__"));



    
    size_t bytes {0};
    for (const auto& shaderInfo : m_shader_wrappers)
    {
        OslHostShaderLayer layer;
        CUDA_CHECK(hipModuleGetGlobal(&layer.init_func, &bytes, m_module, shaderInfo.GetInitName().c_str()));
        CUDA_CHECK(hipModuleGetGlobal(&layer.entry_func, &bytes, m_module, shaderInfo.GetEntryName().c_str()));
        CUDA_CHECK(hipModuleGetGlobal(&layer.fused_func, &bytes, m_module, shaderInfo.GetFusedName().c_str()));

        m_shader_layers.push_back(layer);
    }

    m_function_table.layers = m_shader_layers.data();
    m_function_table.num_layers = m_shader_layers.size();

    // walk trough the m_function table and print the function addresses
    for (int i = 0; i < m_function_table.num_layers; ++i)
    {
        std::cout << "Layer " << i << " init: " << m_function_table.layers[i].init_func << std::endl;
        std::cout << "Layer " << i << " entry: " << m_function_table.layers[i].entry_func << std::endl;
        std::cout << "Layer " << i << " fused: " << m_function_table.layers[i].fused_func << std::endl;
    }

    static_assert(sizeof(OslDeviceFunctionTable) == sizeof(OslHostFunctionTable), "Size of OslDeviceFunctionTable and OslHostFunctionTable must be the same");
    hipDeviceptr_t device_function_table = device_alloc(sizeof(OslDeviceFunctionTable));
    copy_to_device(device_function_table, &m_function_table, sizeof(OslDeviceFunctionTable));
    
    CUDA_CHECK(hipModuleGetGlobal(&m_device_function_table, &bytes, m_module, "oslDeviceFunctionTable"));
    std::cout << "Device function table global size: " << bytes << std::endl;
    assert(bytes == sizeof(OslDeviceFunctionTable));
    CUDA_CHECK(hipMemcpy(m_device_function_table, &device_function_table, sizeof(OslDeviceFunctionTable), hipMemcpyHostToDevice));
    
    return true;
}


bool
OptixGridRenderer::finalize_scene()
{
    return make_optix_materials();
}



/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool
OptixGridRenderer::good(TextureHandle* handle OSL_MAYBE_UNUSED)
{
    return handle != nullptr;
}



/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle*
OptixGridRenderer::get_texture_handle(ustring filename,
                                      ShadingContext* /*shading_context*/,
                                      const TextureOpt* /*options*/)
{
    auto itr = m_samplers.find(filename);
    if (itr == m_samplers.end()) {
        // Open image to check the number of mip levels
        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            errhandler().errorfmt("Could not load: {} (hash {})", filename,
                                  filename);
            return (TextureHandle*)nullptr;
        }
        int32_t nmiplevels = std::max(image.nmiplevels(), 1);
        int32_t img_width  = image.xmax() + 1;
        int32_t img_height = image.ymax() + 1;

        // hard-code textures to 4 channels
        hipChannelFormatDesc channel_desc
            = hipCreateChannelDesc(32, 32, 32, 32, hipChannelFormatKindFloat);

        hipMipmappedArray_t mipmapArray;
        hipExtent extent = make_hipExtent(img_width, img_height, 0);
        CUDA_CHECK(hipMallocMipmappedArray(&mipmapArray, &channel_desc, extent,
                                            nmiplevels));

        // Copy the pixel data for each mip level
        std::vector<std::vector<float>> level_pixels(nmiplevels);
        for (int32_t level = 0; level < nmiplevels; ++level) {
            image.reset(filename, 0, level);
            OIIO::ROI roi = OIIO::get_roi_full(image.spec());
            if (!roi.defined()) {
                errhandler().errorfmt(
                    "Could not load mip level {}: {} (hash {})", level,
                    filename, filename);
                return (TextureHandle*)nullptr;
            }

            int32_t width = roi.width(), height = roi.height();
            level_pixels[level].resize(width * height * 4);
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    image.getpixel(i, j, 0,
                                   &level_pixels[level][((j * width) + i) * 4]);
                }
            }

            hipArray_t miplevelArray;
            CUDA_CHECK(
                hipGetMipmappedArrayLevel(&miplevelArray, mipmapArray, level));

            // Copy the texel data into the miplevel array
            int32_t pitch = width * 4 * sizeof(float);
            CUDA_CHECK(hipMemcpy2DToArray(miplevelArray, 0, 0,
                                           level_pixels[level].data(), pitch,
                                           pitch, height,
                                           hipMemcpyHostToDevice));
        }

        int32_t pitch = img_width * 4 * sizeof(float);
        hipArray_t pixelArray;
        CUDA_CHECK(
            hipMallocArray(&pixelArray, &channel_desc, img_width, img_height));
        CUDA_CHECK(hipMemcpy2DToArray(pixelArray, 0, 0, level_pixels[0].data(),
                                       pitch, pitch, img_height,
                                       hipMemcpyHostToDevice));
        m_arrays_to_free.push_back(pixelArray);

        hipResourceDesc res_desc  = {};
        res_desc.resType           = hipResourceTypeMipmappedArray;
        res_desc.res.mipmap.mipmap = mipmapArray;

        hipTextureDesc tex_desc     = {};
        tex_desc.addressMode[0]      = hipAddressModeWrap;
        tex_desc.addressMode[1]      = hipAddressModeWrap;
        tex_desc.filterMode          = hipFilterModeLinear;
        tex_desc.readMode            = hipReadModeElementType;
        tex_desc.normalizedCoords    = 1;
        tex_desc.maxAnisotropy       = 1;
        tex_desc.maxMipmapLevelClamp = float(nmiplevels - 1);
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = hipFilterModeLinear;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;

        // Create texture object
        hipTextureObject_t cuda_tex = 0;
        CUDA_CHECK(
            hipCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        itr = m_samplers
                  .emplace(std::move(filename.hash()), std::move(cuda_tex))
                  .first;
    }
    return reinterpret_cast<RendererServices::TextureHandle*>(itr->second);
}



void
OptixGridRenderer::prepare_render()
{
    // Set up the OptiX Context
    init_optix_context(m_xres, m_yres);

    // Set up the OptiX scene graph
    finalize_scene();
}



void
OptixGridRenderer::warmup()
{
    // Perform a tiny launch to warm up the OptiX context
    // OPTIX_CHECK(optixLaunch(m_optix_pipeline, m_cuda_stream, d_launch_params,
    //                         sizeof(RenderParams), &m_optix_sbt, 0, 0, 1));
    CUDA_SYNC_CHECK();
}


//extern "C" void setTestshadeGlobals(float h_invw, float h_invh, hipDeviceptr_t d_output_buffer, bool h_flipv);

void
OptixGridRenderer::render(int xres OSL_MAYBE_UNUSED, int yres OSL_MAYBE_UNUSED, RenderState& renderStae)
{
    d_output_buffer = device_alloc(xres * yres * 4 * sizeof(float));
    d_launch_params = device_alloc(sizeof(RenderParams));
    printf("Output buffer: %p\n", d_output_buffer);
    m_xres = xres;
    m_yres = yres;

    RenderParams params;
    params.invw  = 1.0f / std::max(1, m_xres - 1);
    params.invh  = 1.0f / std::max(1, m_yres - 1);
    params.flipv = false; /* I don't see flipv being initialized anywhere */
    params.output_buffer           = d_output_buffer;
    // get the address of the printf buffer
    params.osl_printf_buffer_start = (uint64_t)d_osl_printf_buffer;
    // maybe send buffer size to CUDA instead of the buffer 'end'
    params.osl_printf_buffer_end = params.osl_printf_buffer_start + OSL_PRINTF_BUFFER_SIZE;
    params.color_system          = d_color_system;
    params.test_str_1            = test_str_1;
    params.test_str_2            = test_str_2;
    params.object2common         = d_object2common;
    params.shader2common         = d_shader2common;
    params.num_named_xforms      = m_num_named_xforms;
    params.xform_name_buffer     = d_xform_name_buffer;
    params.xform_buffer          = d_xform_buffer;
    params.fused_callable        = m_fused_callable;

    copy_to_device(d_launch_params, &params, sizeof(RenderParams));
    // hipDeviceptr_t render_params{nullptr};
    // CUDA_CHECK( hipModuleGetGlobal(&render_params, nullptr, m_module, "render_params") );
    // CUDA_CHECK( hipMemcpy(render_params, &d_launch_params, sizeof(RenderParams), hipMemcpyHostToDevice) );

    void *args[] = { &d_launch_params };
    //setup globals
    {
        CUDA_CHECK(hipModuleLaunchKernel(m_function_init_globals, 
            1, 1, 1, 
            1, 1, 1, 
            0, 
            m_cuda_stream,
            args,
            nullptr));
    }

    // Launch the OptiX kernel
    {
         uint3 blockSize {8, 8, 1};

        uint3 gridSize = {  (m_xres + blockSize.x - 1) / blockSize.x,
                            (m_yres + blockSize.y - 1) / blockSize.y,
                            1};
        
        std::cout << "Xres: " << m_xres << " Yres: " << m_yres << std::endl;
        std::cout << "Block size: " << blockSize.x << " " << blockSize.y << " " << blockSize.z << std::endl;
        std::cout << "Grid size: " << gridSize.x << " " << gridSize.y << " " << gridSize.z << std::endl;

        CUDA_CHECK(
            hipModuleLaunchKernel(m_function_shade, 
                gridSize.x, gridSize.y, gridSize.z, 
                blockSize.x, blockSize.y, blockSize.z, 
                0, 
                m_cuda_stream, args, nullptr));
    }

    // CUDA_CHECK(hipMemcpy(d_launch_params, render_params, sizeof(RenderParams), hipMemcpyDeviceToHost));

   
    
    //Let's print some basic stuff
    
    std::vector<uint8_t> printf_buffer(OSL_PRINTF_BUFFER_SIZE);
    CUDA_CHECK(hipMemcpy(printf_buffer.data(),
                          reinterpret_cast<void*>(d_osl_printf_buffer),
                          OSL_PRINTF_BUFFER_SIZE, hipMemcpyDeviceToHost));

    processPrintfBuffer(printf_buffer.data(), OSL_PRINTF_BUFFER_SIZE);
}



void
OptixGridRenderer::processPrintfBuffer(void* buffer_data, size_t buffer_size)
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
        const char* format = ustring::from_hash(fmt_str_hash).c_str();
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
                        ustring str = ustring::from_hash(str_hash);
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
        print("{}", buffer);
    }
}



void
OptixGridRenderer::finalize_pixel_buffer()
{
    std::string buffer_name = "output_buffer";
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    CUDA_CHECK(hipMemcpy(tmp_buff.data(),
                          d_output_buffer,
                          m_xres * m_yres * 3 * sizeof(float),
                          hipMemcpyDeviceToHost));
    OIIO::ImageBuf* buf = outputbuf(0);
    if (buf)
        buf->set_pixels(OIIO::ROI::All(), OIIO::TypeFloat, tmp_buff.data());
    
    // // check if any value is non zero:
    // for (int i = 0; i < m_xres * m_yres * 3; ++i) {
    //     if (tmp_buff[i] != 0.0f) {
    //         std::cout << "Buffer value: " << tmp_buff[i] << std::endl;
    //     }
    // }
}



void
OptixGridRenderer::clear()
{
    shaders().clear();
    if (m_optix_ctx) {
        OPTIX_CHECK(hipCtxDestroy(m_optix_ctx));
        m_optix_ctx = 0;
    }
}



void
OptixGridRenderer::set_transforms(const OSL::Matrix44& object2common,
                                  const OSL::Matrix44& shader2common)
{
    m_object2common = object2common;
    m_shader2common = shader2common;
}



void
OptixGridRenderer::register_named_transforms()
{
    std::vector<uint64_t> xform_name_buffer;
    std::vector<OSL::Matrix44> xform_buffer;

    // Gather:
    //   1) All of the named transforms
    //   2) The "string" value associated with the transform name, which is
    //      actually the ustring hash of the transform name.
    for (const auto& item : m_named_xforms) {
        const uint64_t addr = item.first.hash();
        xform_name_buffer.push_back(addr);
        xform_buffer.push_back(*item.second);
    }

    // Push the names and transforms to the device
    size_t sz           = sizeof(uint64_t) * xform_name_buffer.size();
    d_xform_name_buffer = device_alloc(sz);
    CUDA_CHECK(hipMemcpy(reinterpret_cast<void*>(d_xform_name_buffer),
                          xform_name_buffer.data(), sz,
                          hipMemcpyHostToDevice));
    sz             = sizeof(OSL::Matrix44) * xform_buffer.size();
    d_xform_buffer = device_alloc(sz);
    CUDA_CHECK(hipMemcpy(reinterpret_cast<void*>(d_xform_buffer),
                          xform_buffer.data(), sz, hipMemcpyHostToDevice));
    m_num_named_xforms = xform_name_buffer.size();
}

void
OptixGridRenderer::register_inline_functions()
{
    // clang-format off

    // Depending on the inlining options and optimization level, some functions
    // might not be inlined even when it would be beneficial to do so. We can
    // register such functions with the ShadingSystem to ensure that they are
    // inlined regardless of the other inlining options or the optimization
    // level.
    //
    // Conversely, there are some functions which should rarely be inlined. If that
    // is known in advance, we can register those functions with the ShadingSystem
    // so they can be excluded before running the ShaderGroup optimization, which
    // can help speed up the optimization and JIT stages.
    //
    // The default behavior of the optimizer should be sufficient for most
    // cases, and the inline/noinline thresholds available through the
    // ShadingSystem attributes enable some degree of fine tuning. This
    // mechanism has been added to offer a finer degree of control
    //
    // Please refer to doc/app_integration/OptiX-Inlining-Options.md for more
    // details about the inlining options.

    // These functions are all 5 instructions or less in the PTX, with most of
    // those instructions related to reading the parameters and writing out the
    // return value. It would be beneficial to inline them in all cases. We can
    // register them to ensure that they are inlined regardless of the other
    // compile options.
    shadingsys->register_inline_function(ustring("osl_abs_ff"));
    shadingsys->register_inline_function(ustring("osl_abs_ii"));
    shadingsys->register_inline_function(ustring("osl_ceil_ff"));
    shadingsys->register_inline_function(ustring("osl_cos_ff"));
    shadingsys->register_inline_function(ustring("osl_exp2_ff"));
    shadingsys->register_inline_function(ustring("osl_exp_ff"));
    shadingsys->register_inline_function(ustring("osl_fabs_ff"));
    shadingsys->register_inline_function(ustring("osl_fabs_ii"));
    shadingsys->register_inline_function(ustring("osl_floor_ff"));
    shadingsys->register_inline_function(ustring("osl_init_texture_options"));
    shadingsys->register_inline_function(ustring("osl_getchar_isi"));
    shadingsys->register_inline_function(ustring("osl_hash_is"));
    shadingsys->register_inline_function(ustring("osl_log10_ff"));
    shadingsys->register_inline_function(ustring("osl_log2_ff"));
    shadingsys->register_inline_function(ustring("osl_log_ff"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_anisotropic"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_bandwidth"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_do_filter"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_impulses"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_ff"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_fff"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_fv"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_fvf"));
    shadingsys->register_inline_function(ustring("osl_sin_ff"));
    shadingsys->register_inline_function(ustring("osl_strlen_is"));
    shadingsys->register_inline_function(ustring("osl_texture_set_interp_code"));
    shadingsys->register_inline_function(ustring("osl_texture_set_stwrap_code"));
    shadingsys->register_inline_function(ustring("osl_trunc_ff"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_ff"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_fff"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_fv"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_fvf"));

    // These large functions are unlikely to ever been inlined. In such cases,
    // we may be able to speed up ShaderGroup compilation by registering these
    // functions as "noinline" so they can be excluded from the ShaderGroup
    // module prior to optimization/JIT.
    shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdfdf"));
    shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdv"));
    shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdvdf"));
    shadingsys->register_noinline_function(ustring("osl_gaborpnoise_dfdfdfff"));
    shadingsys->register_noinline_function(ustring("osl_gaborpnoise_dfdvdfvf"));
    shadingsys->register_noinline_function(ustring("osl_gaborpnoise_dfdvv"));
    shadingsys->register_noinline_function(ustring("osl_genericnoise_dfdvdf"));
    shadingsys->register_noinline_function(ustring("osl_genericpnoise_dfdvv"));
    shadingsys->register_noinline_function(ustring("osl_get_inverse_matrix"));
    shadingsys->register_noinline_function(ustring("osl_noise_dfdfdf"));
    shadingsys->register_noinline_function(ustring("osl_noise_dfdff"));
    shadingsys->register_noinline_function(ustring("osl_noise_dffdf"));
    shadingsys->register_noinline_function(ustring("osl_noise_fv"));
    shadingsys->register_noinline_function(ustring("osl_noise_vff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_dfdfdfff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_dfdffff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_dffdfff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_fffff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_vffff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_dfdfdfff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_dfdffff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_dffdfff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_fffff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_vffff"));
    shadingsys->register_noinline_function(ustring("osl_simplexnoise_dvdf"));
    shadingsys->register_noinline_function(ustring("osl_simplexnoise_vf"));
    shadingsys->register_noinline_function(ustring("osl_simplexnoise_vff"));
    shadingsys->register_noinline_function(ustring("osl_snoise_dfdfdf"));
    shadingsys->register_noinline_function(ustring("osl_snoise_dfdff"));
    shadingsys->register_noinline_function(ustring("osl_snoise_dffdf"));
    shadingsys->register_noinline_function(ustring("osl_snoise_fv"));
    shadingsys->register_noinline_function(ustring("osl_snoise_vff"));
    shadingsys->register_noinline_function(ustring("osl_transform_triple"));
    shadingsys->register_noinline_function(ustring("osl_transformn_dvmdv"));
    shadingsys->register_noinline_function(ustring("osl_usimplexnoise_dvdf"));
    shadingsys->register_noinline_function(ustring("osl_usimplexnoise_vf"));
    shadingsys->register_noinline_function(ustring("osl_usimplexnoise_vff"));

    // It's also possible to unregister functions to restore the default
    // inlining behavior when needed.
    shadingsys->unregister_inline_function(ustring("osl_init_texture_options"));
    shadingsys->unregister_noinline_function(ustring("osl_get_inverse_matrix"));

    // clang-format on
}

OSL_NAMESPACE_EXIT
