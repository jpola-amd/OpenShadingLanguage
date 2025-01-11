
#include <iostream>
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



int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <oso shader>\n";
        return 1;
    }

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
    

    OIIO::string_view shader_name = argv[1];
    OIIO::string_view layer_name = shader_name;


    OSL::ShaderGroupRef shaderGroup = shadingSystem.ShaderGroupBegin("");
    shadingSystem.Shader(*shaderGroup, "surface", shader_name, layer_name);
    shadingSystem.ShaderGroupEnd(*shaderGroup);

    OSL::PerThreadInfo* thread_info = shadingSystem.create_thread_info();
    OSL::ShadingContext* ctx        = shadingSystem.get_context(thread_info);
    
    OSL::ShaderGlobals sg;
    memset((char*)&sg, 0, sizeof(OSL::ShaderGlobals));
    shadingSystem.optimize_group(shaderGroup.get(), nullptr, true);

    // now we should have it JITted

    shadingSystem.release_context(ctx);  // don't need this anymore for now
    shadingSystem.destroy_thread_info(thread_info);


    return 0;
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