
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>

#include <OSL/device_string.h>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <hip/hiprtc.h>

#include "HipRenderer.hpp"
#include "ClosureIDs.hpp"

#include "Assert.hpp"

using namespace OIIO;


// anonymous namespace
namespace {

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams {};
struct DiffuseParams {
    OSL::Vec3 N;
    OSL::ustring label;
};
struct OrenNayarParams {
    OSL::Vec3 N;
    float sigma;
};
struct PhongParams {
    OSL::Vec3 N;
    float exponent;
    OSL::ustring label;
};
struct WardParams {
    OSL::Vec3 N, T;
    float ax, ay;
};
struct ReflectionParams {
    OSL::Vec3 N;
    float eta;
};
struct RefractionParams {
    OSL::Vec3 N;
    float eta;
};
struct MicrofacetParams {
    OSL::ustring dist;
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};
struct DebugParams {
    OSL::ustring tag;
};
}  // anonymous namespace

bool RegisterClosures(OSL::ShadingSystem &shadingSystem);

std::string build_trampoline(OSL::ShaderGroup& group, std::string init_name, std::string entry_name);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <oso shader>\n";
        return 1;
    }

    constexpr int deviceId = 0;

    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(deviceId));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, deviceId));

    std::cout << "Using device id: " << deviceId << " " << properties.name << std::endl;

    HipRenderer renderer;
    auto textureSystem = OIIO::TextureSystem::create();

    OSL::ShadingSystem shadingSystem(&renderer, textureSystem.get());
    if (!RegisterClosures(shadingSystem))
    {
        std::cerr << "Could not register closures\n";
        return 1;
    }

    shadingSystem.attribute("lockgeom", 1);
    shadingSystem.attribute("debug", 3);
    shadingSystem.attribute("statistics:level", 3);
    shadingSystem.attribute("llvm_debug", 3);
    shadingSystem.attribute("llvm_debug_layers", 3);
    shadingSystem.attribute("llvm_debug_ops", 3);
    shadingSystem.attribute("llvm_output_bitcode", 3);
    

    OIIO::string_view shader_path = argv[1];
    // name of the shader can't have any special chars like / or \ or .
    OIIO::string_view layer_name = OIIO::Filesystem::filename(shader_path);
    //remove the extension
    layer_name = OIIO::Filesystem::replace_extension(layer_name, "");

    OSL::ShaderGroupRef shaderGroup = shadingSystem.ShaderGroupBegin("");
    shadingSystem.Shader(*shaderGroup, "surface", shader_path, layer_name);
    shadingSystem.ShaderGroupEnd(*shaderGroup);

    OSL::PerThreadInfo* thread_info = shadingSystem.create_thread_info();
    OSL::ShadingContext* ctx        = shadingSystem.get_context(thread_info);
    
    OSL::ShaderGlobals sg;
    memset((char*)&sg, 0, sizeof(OSL::ShaderGlobals));
    shadingSystem.optimize_group(shaderGroup.get(), nullptr, true);

    
    constexpr bool run = false;
    if (!shadingSystem.execute(*ctx, *shaderGroup, sg, run))
    {
        std::cerr << "Could not execute shader\n";
        return 1;
    }



    shadingSystem.release_context(ctx);  // don't need this anymore for now
    shadingSystem.destroy_thread_info(thread_info);

    
    std::string hip_gcn_shader;
    if (!shadingSystem.getattribute(shaderGroup.get(), "hip_compiled_version",
                         TypeDesc::PTR, &hip_gcn_shader)) {
        std::cerr << "Error getting shader group bc" << std::endl;
        return 1;
    }
    // JPA: We can query the shader group

    // Get the entry points from the ShadingSystem. The names are
    // auto-generated and mangled, so will be different for every shader group.
    std::string init_name, entry_name;
    if (!shadingSystem.getattribute(shaderGroup.get(), "group_init_name", init_name)) {
        std::cerr << "Error getting shader group init name" << std::endl;
        return 1;
    }

    if (!shadingSystem.getattribute(shaderGroup.get(), "group_entry_name", entry_name)) {
        std::cerr << "Error getting shader group entry name" << std::endl;
        return 1;
    }

    std::cout << "init_name: " << init_name << std::endl;
    std::cout << "entry_name: " << entry_name << std::endl;

    // JPA: Here is how we can query the shading system itself.
    int hip_llvm_ops_size {0};
    if (!shadingSystem.getattribute("shadeops_hip_llvm_size", TypeDesc::INT, (void*)&hip_llvm_ops_size))
    {
        std::cerr << "Error getting hip llvm ops size" << std::endl;
        return 1;

    };

    char* hip_llvm_ops {nullptr};
    if (!shadingSystem.getattribute("shadeops_hip_llvm", TypeDesc::PTR, &hip_llvm_ops))
    {
        std::cerr << "Error getting hip llvm ops" << std::endl;
        return 1;
    };


    std::string trampoline_bc = build_trampoline(*shaderGroup, init_name, entry_name);

    //link everything together
    std::vector<uint8_t> hip_fatbin; 
    // there is a problem with many functions. The hip_gcn_shader should be linked internally with hip_llvm_ops. 
    // but due to problem with executin llvm::Linker::linkModules() llvm_instance.cpp:L2582
    // i have to link it here manually.
    // There are also some missing definitions. The critical are the printf and variations of this function. 
    // It must be working in order to parse strings and capture string params. 
    hiprtcLinkState linkState;
    HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkState));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, hip_llvm_ops, hip_llvm_ops_size, "hip_llvm_ops", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "HipRenderer.bc", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "RenderLib.bc", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, trampoline_bc.data(), trampoline_bc.size(), "trampoline.bc", 0, nullptr, nullptr));
    HIPRTC_CHECK(hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, hip_gcn_shader.data(), hip_gcn_shader.size(), "shader.bc", 0, nullptr, nullptr));

    {
        void* code { nullptr };
        size_t size { 0 };

        HIPRTC_CHECK(hiprtcLinkComplete(linkState, &code, &size));

        //JPA: This is stupid and not intuitive. the owner of the code_ptr and size is the linker. 
        // Copy the data before destroying the linker
        if (size == 0)
        {
            std::cerr << "HIPRTC Error: the HIP fatbin size is 0" << std::endl;
            HIPRTC_CHECK(hiprtcLinkDestroy(linkState)); 
            return EXIT_FAILURE;
        }

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
            std::cerr << "Proble with opening hip_fatbin.bin file for writing compiled code" << std::endl;
        }

        HIPRTC_CHECK(hiprtcLinkDestroy(linkState)); 
    }

    
    hipModule_t module {nullptr};
    HIP_CHECK(hipModuleLoadData(&module, hip_fatbin.data()));
    
    hipFunction_t kernel_render_entry{nullptr};
    HIP_CHECK(hipModuleGetFunction(&kernel_render_entry, module, "shade"));

    // now we can render
    size_t stack_size {0};
    HIP_CHECK(hipDeviceGetLimit(&stack_size, hipLimitStackSize));
    HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, 4096));

    int w = 512;
    int h = 512;
    hipDeviceptr_t d_outputBuffer {nullptr};
    
    HIP_CHECK(hipMalloc(&d_outputBuffer, w*h*3*sizeof(float)));
    HIP_SYNC_CHECK();
    unsigned int  block_size {8};
    unsigned int num_blocks_x = static_cast<unsigned int>(w) / block_size;
    unsigned int num_blocks_y = static_cast<unsigned int>(h) / block_size;

    dim3 gridSize(num_blocks_x, num_blocks_y, 1u);
    dim3 blockSize(block_size, block_size, 1u);

    void* params [] = {
        &d_outputBuffer,
        &w,
        &h
    };

    HIP_CHECK(hipLaunchKernel(kernel_render_entry, gridSize, blockSize, params, 0, stream));
        
    HIP_SYNC_CHECK();


    std::cout << "\n\n-----------End of the program---------\n\n" << std::endl << std::flush;
    HIP_CHECK(hipFree(d_outputBuffer));                 
    

    return 0;
}

const char* hip_compile_options[] = { 
    "--offload-arch=gfx1030",
    "-ffast-math", "-fgpu-rdc", "-emit-llvm", "-c", 
    "-D__HIP_PLATFORM_AMD",
    "--std=c++17" 
};



std::string
build_trampoline(OSL::ShaderGroup& group, std::string init_name,
                     std::string entry_name)
{

    std::stringstream ss;
    ss << "class ShaderGlobals;\n";
    ss << "extern \"C\" __device__ void " << init_name
       << "(ShaderGlobals*,void*);\n";
    ss << "extern \"C\" __device__ void " << entry_name
       << "(ShaderGlobals*,void*);\n";
    ss << "extern \"C\" __device__ void __osl__init(ShaderGlobals* sg, void* "
          "params) { "
       << init_name << "(sg, params); }\n";
    ss << "extern \"C\" __device__ void __osl__entry(ShaderGlobals* sg, void* "
          "params) { "
       << entry_name << "(sg, params); }\n";

    auto code = ss.str();

    hiprtcProgram trampolineProgram;

    constexpr int num_compile_flags = int(sizeof(hip_compile_options) / sizeof(hip_compile_options[0]));
    size_t hip_log_size;
    HIPRTC_CHECK(hiprtcCreateProgram(&trampolineProgram, code.c_str(),
                                   "trampoline", 0, nullptr, nullptr));
    auto compileResult = hiprtcCompileProgram(trampolineProgram, num_compile_flags,
                                             hip_compile_options);

    if (compileResult != HIPRTC_SUCCESS) {
        HIPRTC_CHECK(hiprtcGetProgramLogSize(trampolineProgram, &hip_log_size));
        std::vector<char> hip_log(hip_log_size + 1);
        HIPRTC_CHECK(hiprtcGetProgramLog(trampolineProgram, hip_log.data()));
        hip_log.back() = 0;
        std::stringstream ss;
        ss << "hiprtcCompileProgram failure for: " << code
           << "====================================\n"
           << hip_log.data();
        throw std::runtime_error(ss.str());
    }


    size_t bitcode_size;
    HIPRTC_CHECK(hiprtcGetBitcodeSize(trampolineProgram, &bitcode_size));
    std::vector<char> bitcode(bitcode_size);
    HIPRTC_CHECK(hiprtcGetBitcode(trampolineProgram, bitcode.data()));
    HIPRTC_CHECK(hiprtcDestroyProgram(&trampolineProgram));

    std::string trampoline_bc(bitcode.begin(), bitcode.end());
    return trampoline_bc;
}

bool RegisterClosures(OSL::ShadingSystem &shadingSystem)
{
    // Describe the memory layout of each closure type to the OSL runtime
    enum
    {
        MAX_PARAMS = 32
    };
    struct BuiltinClosures
    {
        const char *name;
        int id;
        OSL::ClosureParam params[MAX_PARAMS]; // upper bound
    };

    BuiltinClosures builtins[] = {
        {"emission", EMISSION_ID, {CLOSURE_FINISH_PARAM(EmptyParams)}},
        {"background", BACKGROUND_ID, {CLOSURE_FINISH_PARAM(EmptyParams)}},
        {"diffuse",
         DIFFUSE_ID,
         {CLOSURE_VECTOR_PARAM(DiffuseParams, N),
          CLOSURE_STRING_KEYPARAM(DiffuseParams, label,
                                  "label"), // example of custom key param
          CLOSURE_FINISH_PARAM(DiffuseParams)}},
        {"oren_nayar",
         OREN_NAYAR_ID,
         {CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
          CLOSURE_FLOAT_PARAM(OrenNayarParams, sigma),
          CLOSURE_FINISH_PARAM(OrenNayarParams)}},
        {"translucent",
         TRANSLUCENT_ID,
         {CLOSURE_VECTOR_PARAM(DiffuseParams, N),
          CLOSURE_FINISH_PARAM(DiffuseParams)}},
        {"phong",
         PHONG_ID,
         {CLOSURE_VECTOR_PARAM(PhongParams, N),
          CLOSURE_FLOAT_PARAM(PhongParams, exponent),
          CLOSURE_STRING_KEYPARAM(PhongParams, label,
                                  "label"), // example of custom key param
          CLOSURE_FINISH_PARAM(PhongParams)}},
        {"ward",
         WARD_ID,
         {CLOSURE_VECTOR_PARAM(WardParams, N),
          CLOSURE_VECTOR_PARAM(WardParams, T),
          CLOSURE_FLOAT_PARAM(WardParams, ax),
          CLOSURE_FLOAT_PARAM(WardParams, ay),
          CLOSURE_FINISH_PARAM(WardParams)}},
        {"microfacet",
         MICROFACET_ID,
         {CLOSURE_STRING_PARAM(MicrofacetParams, dist),
          CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
          CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
          CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
          CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
          CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
          CLOSURE_INT_PARAM(MicrofacetParams, refract),
          CLOSURE_FINISH_PARAM(MicrofacetParams)}},
        {"reflection",
         REFLECTION_ID,
         {CLOSURE_VECTOR_PARAM(ReflectionParams, N),
          CLOSURE_FINISH_PARAM(ReflectionParams)}},
        {"reflection",
         FRESNEL_REFLECTION_ID,
         {CLOSURE_VECTOR_PARAM(ReflectionParams, N),
          CLOSURE_FLOAT_PARAM(ReflectionParams, eta),
          CLOSURE_FINISH_PARAM(ReflectionParams)}},
        {"refraction",
         REFRACTION_ID,
         {CLOSURE_VECTOR_PARAM(RefractionParams, N),
          CLOSURE_FLOAT_PARAM(RefractionParams, eta),
          CLOSURE_FINISH_PARAM(RefractionParams)}},
        {"transparent", TRANSPARENT_ID, {CLOSURE_FINISH_PARAM(EmptyParams)}},
        {"debug",
         DEBUG_ID,
         {CLOSURE_STRING_PARAM(DebugParams, tag),
          CLOSURE_FINISH_PARAM(DebugParams)}},
        {"holdout", HOLDOUT_ID, {CLOSURE_FINISH_PARAM(EmptyParams)}}};

    for (const auto &b : builtins)
    {
        shadingSystem.register_closure(b.name, b.id, b.params, nullptr, nullptr);
    }
    return true;
}