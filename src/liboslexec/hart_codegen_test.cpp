// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// GPU-independent compiler regression tests for the HART backend. Compile small
// OSL shaders and inspect their AMDGPU bitcode before and after optimization,
// checking target metadata, ordered entries, split/fused callable ABI, address
// spaces, group-data alignment, output placement, string hash storage, packed
// diagnostics, interactive uploads, userdata caching, geometry state, renderer
// attributes/libraries, immutable texture bindings, named transforms, material
// closure records, linked shadeops, control flow, HART provenance, and rejection
// of unsupported operations.
// This allows testing every configured architecture without its physical GPU.
//
// Built as a separate test executable, not part of the runtime library, only
// when OSL_BUILD_TESTS, BUILD_TESTING, OSL_USE_HART, and USE_LLVM_BITCODE are on.
// It lives beside the other liboslexec unit tests; actual GPU execution and
// image comparisons are covered separately by check-generated.py.
//
// Keep and extend this coverage as HART and LLVM evolve. Replace individual
// rejection cases with positive tests when their features become supported.

#include <OSL/genclosure.h>
#include <OSL/hart_diagnostics.h>
#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>

#include "../testrender/bsdl_config.h"
#include "../testshade/hartgeneratedparams.h"
#include "../testshade/hartrenderstate.h"
#include "../testshade/render_state.h"
#include "hart_bitcode.h"
#include "oslexec_pvt.h"
#include <BSDL/MTX/bsdf_dielectric_decl.h>
#include <BSDL/MTX/bsdf_oren_nayar_diffuse_decl.h>
#include <BSDL/MTX/bsdf_sheen_decl.h>
#include "opcolor.h"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/unittest.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <utility>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/GlobalAlias.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#if LLVM_VERSION_MAJOR >= 17
#    include <llvm/TargetParser/Triple.h>
#else
#    include <llvm/ADT/Triple.h>
#endif

using namespace OSL;

namespace {

class HartServices : public RendererServices {
public:
    explicit HartServices(bool textures = false, bool transforms = false,
                          bool closures = false, bool arrays = false,
                          bool splines = false, bool colors = false,
                          bool noise = false, bool diagnostics = false,
                          bool texture_defaults = false)
        : m_textures(textures)
        , m_transforms(transforms)
        , m_closures(closures)
        , m_arrays(arrays)
        , m_splines(splines)
        , m_colors(colors)
        , m_noise(noise)
        , m_diagnostics(diagnostics)
        , m_texture_defaults(texture_defaults)
    {
    }
    int supports(string_view feature) const override
    {
        return feature == "HART" || (m_textures && feature == "HARTTextures")
               || (m_transforms && feature == "HARTTransforms")
               || (m_closures && feature == "HARTClosures")
               || (m_arrays && feature == "HARTArrayBounds")
               || (m_splines && feature == "HARTSplineErrors")
               || (m_colors && feature == "HARTColorSystem")
               || (m_noise && feature == "HARTNoiseErrors")
               || (m_diagnostics && feature == "HARTDiagnostics")
               || (m_texture_defaults && feature == "HARTTextureDefaults");
    }
    TextureHandle* get_texture_handle(ustring filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        ++texture_requests;
        texture_filenames.push_back(filename);
        return m_textures
                       && (filename == "hart-test-texture.exr"
                           || filename == "hart_texture_alpha_4.exr")
                   ? reinterpret_cast<TextureHandle*>(uintptr_t(1))
                   : nullptr;
    }
    bool good(TextureHandle* handle) override
    {
        return m_textures
               && handle == reinterpret_cast<TextureHandle*>(uintptr_t(1));
    }

    int texture_requests = 0;
    std::vector<ustring> texture_filenames;

private:
    bool m_textures;
    bool m_transforms;
    bool m_closures;
    bool m_arrays;
    bool m_splines;
    bool m_colors;
    bool m_noise;
    bool m_diagnostics;
    bool m_texture_defaults;
};



struct EmptyClosureParams { };
struct DiffuseClosureParams {
    Vec3 N;
    ustring label;
};

const ClosureParam emission_params[]
    = { CLOSURE_FINISH_PARAM(EmptyClosureParams) };
const ClosureParam diffuse_params[]
    = { CLOSURE_VECTOR_PARAM(DiffuseClosureParams, N),
        CLOSURE_STRING_KEYPARAM(DiffuseClosureParams, label, "label"),
        CLOSURE_FINISH_PARAM(DiffuseClosureParams) };

int closure_callback_calls = 0;



void
host_closure_callback(RendererServices*, int, void*)
{ ++closure_callback_calls; }



void
register_hart_closures(ShadingSystem& ss)
{
    ss.register_closure("emission", 1, emission_params, nullptr, nullptr);
    ss.register_closure("diffuse", 3, diffuse_params, nullptr, nullptr);
}



// These small records follow testrender/shading.h without importing its
// raytracer/CUDA dependencies. BSDL records use their actual Data and entry().
struct MaterialMicrofacetParams {
    ustringhash dist;
    Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};



struct MaterialLayerParams {
    ClosureColor* top;
    ClosureColor* base;
};



struct MaterialUniformParams {
    Color3 emittance;
    ustringhash label;
};



struct MaterialProbeParams {
    Matrix44 basis;
    ClosureColor* input;
    ClosureColor* fallback;
    float gain;
    int mode;
    ustringhash label;
};



struct MaterialDiffuseRampParams {
    ustringhash label;
    Vec3 N;
    Color3 colors[8];
};



struct MaterialPhongRampParams {
    ustringhash label;
    Vec3 N;
    float exponent;
    Color3 colors[8];
};



struct MaterialArrayParams {
    int integers[2];
    float floats[2];
    Vec3 points[2], vectors[2], normals[2];
    Matrix44 matrices[2];
    Color3 palette[8];
    float gain;
};



struct MaterialBsdfRoot { };
using MaterialOren       = bsdl::mtx::OrenNayarDiffuseLobe<MaterialBsdfRoot>;
using MaterialSheen      = bsdl::mtx::SheenLobe<MaterialBsdfRoot>;
using MaterialDielectric = bsdl::mtx::DielectricLobe<MaterialBsdfRoot>;



struct MaterialClosure {
    const char* name;
    int id;
    std::vector<ClosureParam> params;
};



template<typename Lobe>
MaterialClosure
material_bsdl_closure(int id)
{
    const auto entry = Lobe::template entry<typename Lobe::Data>();
    MaterialClosure result { entry.name, id, { } };
    for (const auto& p : entry.params) {
        TypeDesc type;
        switch (p.type) {
        case bsdl::ParamType::NONE: break;
        case bsdl::ParamType::VECTOR: type = TypeVector; break;
        case bsdl::ParamType::COLOR: type = TypeColor; break;
        case bsdl::ParamType::INT: type = TypeInt; break;
        case bsdl::ParamType::FLOAT: type = TypeFloat; break;
        case bsdl::ParamType::STRING: type = TypeString; break;
        case bsdl::ParamType::CLOSURE: type = TypeDesc::PTR; break;
        }
        result.params.push_back({ type, p.offset, p.key, p.type_size });
        if (p.type == bsdl::ParamType::NONE)
            break;
    }
    OIIO_CHECK_ASSERT(!result.params.empty()
                      && result.params.back().type == TypeDesc());
    return result;
}



const std::vector<MaterialClosure>&
material_closures()
{
    static const std::vector<MaterialClosure> entries {
        { "emission", 1, { CLOSURE_FINISH_PARAM(EmptyClosureParams) } },
        { "microfacet",
          8,
          { CLOSURE_STRING_PARAM(MaterialMicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MaterialMicrofacetParams, N),
            CLOSURE_VECTOR_PARAM(MaterialMicrofacetParams, U),
            CLOSURE_FLOAT_PARAM(MaterialMicrofacetParams, xalpha),
            CLOSURE_FLOAT_PARAM(MaterialMicrofacetParams, yalpha),
            CLOSURE_FLOAT_PARAM(MaterialMicrofacetParams, eta),
            CLOSURE_INT_PARAM(MaterialMicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MaterialMicrofacetParams) } },
        material_bsdl_closure<MaterialOren>(15),
        material_bsdl_closure<MaterialDielectric>(17),
        material_bsdl_closure<MaterialSheen>(23),
        { "uniform_edf",
          24,
          { CLOSURE_COLOR_PARAM(MaterialUniformParams, emittance),
            CLOSURE_STRING_KEYPARAM(MaterialUniformParams, label, "label"),
            CLOSURE_FINISH_PARAM(MaterialUniformParams) } },
        { "layer",
          27,
          { CLOSURE_CLOSURE_PARAM(MaterialLayerParams, top),
            CLOSURE_CLOSURE_PARAM(MaterialLayerParams, base),
            CLOSURE_FINISH_PARAM(MaterialLayerParams) } },
        { "hart_material_probe",
          101,
          { { TypeMatrix, offsetof(MaterialProbeParams, basis), nullptr,
              sizeof(Matrix44) },
            CLOSURE_CLOSURE_PARAM(MaterialProbeParams, input),
            { TypeDesc::PTR, offsetof(MaterialProbeParams, fallback),
              "fallback", sizeof(ClosureColor*) },
            CLOSURE_FLOAT_KEYPARAM(MaterialProbeParams, gain, "gain"),
            CLOSURE_INT_KEYPARAM(MaterialProbeParams, mode, "mode"),
            CLOSURE_STRING_KEYPARAM(MaterialProbeParams, label, "label"),
            CLOSURE_FINISH_PARAM(MaterialProbeParams) } },
        { "diffuse_ramp",
          102,
          { CLOSURE_VECTOR_PARAM(MaterialDiffuseRampParams, N),
            CLOSURE_COLOR_ARRAY_PARAM(MaterialDiffuseRampParams, colors, 8),
            CLOSURE_STRING_KEYPARAM(MaterialDiffuseRampParams, label, "label"),
            CLOSURE_FINISH_PARAM(MaterialDiffuseRampParams) } },
        { "phong_ramp",
          103,
          { CLOSURE_VECTOR_PARAM(MaterialPhongRampParams, N),
            CLOSURE_FLOAT_PARAM(MaterialPhongRampParams, exponent),
            CLOSURE_COLOR_ARRAY_PARAM(MaterialPhongRampParams, colors, 8),
            CLOSURE_STRING_KEYPARAM(MaterialPhongRampParams, label, "label"),
            CLOSURE_FINISH_PARAM(MaterialPhongRampParams) } },
        { "hart_array_probe",
          104,
          { CLOSURE_INT_ARRAY_PARAM(MaterialArrayParams, integers, 2),
            CLOSURE_FLOAT_ARRAY_PARAM(MaterialArrayParams, floats, 2),
            { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::POINT, 2),
              offsetof(MaterialArrayParams, points), nullptr, sizeof(Vec3[2]) },
            CLOSURE_VECTOR_ARRAY_PARAM(MaterialArrayParams, vectors, 2),
            { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::NORMAL, 2),
              offsetof(MaterialArrayParams, normals), nullptr, sizeof(Vec3[2]) },
            { TypeDesc(TypeDesc::FLOAT, TypeDesc::MATRIX44, 2),
              offsetof(MaterialArrayParams, matrices), nullptr,
              sizeof(Matrix44[2]) },
            { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR, 8),
              offsetof(MaterialArrayParams, palette), "palette",
              sizeof(Color3[8]) },
            CLOSURE_FLOAT_KEYPARAM(MaterialArrayParams, gain, "gain"),
            CLOSURE_FINISH_PARAM(MaterialArrayParams) } },
    };
    return entries;
}



class HartMaterialServices final : public RendererServices {
public:
    int supports(string_view feature) const override
    {
        return feature == "HART" || (arrays && feature == "HARTArrayBounds")
               || (closures && feature == "HARTClosures")
               || (parameters && feature == "HARTClosureParameters");
    }
    bool closures = true, parameters = true, arrays = true;
};



class Diagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR) {
            ++errors;
            last_error = message;
            messages += message;
            messages += '\n';
        }
    }
    int errors = 0;
    std::string last_error;
    std::string messages;
};



class HartInteractiveServices final : public RendererServices {
public:
    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTArrayBounds"
               || (interactive && feature == "HARTInteractive");
    }
    void* device_alloc(size_t size) override
    {
        ++allocations;
        requested_size = size;
        OIIO_CHECK_ASSERT(!storage);
        if (fail_allocation || storage)
            return nullptr;
        storage = std::make_unique<uint8_t[]>(size);
        std::memset(storage.get(), 0xa5, size);
        ++successful_allocations;
        return storage.get();
    }
    void device_free(void* pointer) override
    {
        OIIO_CHECK_ASSERT(storage && pointer == storage.get());
        if (storage && pointer == storage.get()) {
            ++frees;
            storage.reset();
        }
    }
    void* copy_to_device(void* destination, const void* source,
                         size_t size) override
    {
        const auto address = reinterpret_cast<uintptr_t>(destination);
        const auto base    = reinterpret_cast<uintptr_t>(storage.get());
        const bool valid   = storage && source && size && address >= base
                             && address - base <= requested_size
                             && size <= requested_size - (address - base);
        OIIO_CHECK_ASSERT(valid);
        if (!valid)
            return nullptr;
        copies.push_back({ size_t(address - base), size });
        if (fail_copy) {
            // Deliberately damage only a prefix; rollback must repair the rest
            // of this allocation even when the next update targets elsewhere.
            std::memcpy(destination, source, std::max(size_t(1), size / 2));
            return nullptr;
        }
        std::memcpy(destination, source, size);
        return destination;
    }

    struct Copy {
        size_t offset;
        size_t size;
    };
    bool interactive     = true;
    bool fail_allocation = false;
    bool fail_copy       = false;
    int allocations = 0, successful_allocations = 0, frees = 0;
    size_t requested_size = 0;
    std::unique_ptr<uint8_t[]> storage;
    std::vector<Copy> copies;
};



class HartMutableServices final : public HartServices {
public:
    using HartServices::HartServices;

    int supports(string_view feature) const override
    {
        return feature == "HARTUserdata" || feature == "HARTInteractive"
               || HartServices::supports(feature);
    }
    void* device_alloc(size_t size) override
    { return arena.device_alloc(size); }
    void device_free(void* pointer) override { arena.device_free(pointer); }
    void* copy_to_device(void* destination, const void* source,
                         size_t size) override
    { return arena.copy_to_device(destination, source, size); }

    HartInteractiveServices arena;
};



class HartUserdataServices final : public RendererServices {
public:
    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTArrayBounds"
               || (arena.interactive && feature == "HARTInteractive")
               || feature == "HARTClosures"
               || (getter && feature == "build_interpolated_getter")
               || (userdata && feature == "HARTUserdata");
    }
    void* device_alloc(size_t size) override
    { return arena.device_alloc(size); }
    void device_free(void* pointer) override { arena.device_free(pointer); }
    void* copy_to_device(void* destination, const void* source,
                         size_t size) override
    { return arena.copy_to_device(destination, source, size); }
    void build_interpolated_getter(const ShaderGroup&, const ustring& name,
                                   TypeDesc type, bool derivatives,
                                   InterpolatedGetterSpec& spec) override
    {
        requests.push_back({ name, type, derivatives });
        if (!missing_spec)
            spec.set(ustring("osl_hart_get_userdata"),
                     InterpolatedSpecBuiltinArg::OpaqueExecutionContext,
                     InterpolatedSpecBuiltinArg::ShadeIndex,
                     InterpolatedSpecBuiltinArg::ParamName,
                     InterpolatedSpecBuiltinArg::Type,
                     InterpolatedSpecBuiltinArg::Derivatives);
    }
    bool get_userdata(bool, ustringhash, TypeDesc, ShaderGlobals*,
                      void*) override
    {
        ++host_lookups;
        OIIO_CHECK_ASSERT(false);
        return false;
    }

    struct Request {
        ustring name;
        TypeDesc type;
        bool derivatives;
    };
    bool userdata = true, getter = true, missing_spec = false;
    int host_lookups = 0;
    std::vector<Request> requests;
    HartInteractiveServices arena;
};



class HartGeometryServices final : public RendererServices {
public:
    explicit HartGeometryServices(bool geometry = true) : m_geometry(geometry)
    {
    }
    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTArrayBounds"
               || (m_geometry && feature == "HARTGeometry");
    }

private:
    bool m_geometry;
};



class HartNamedTransformServices final : public RendererServices {
public:
    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTArrayBounds"
               || (transforms && feature == "HARTTransforms")
               || (named && feature == "HARTNamedTransforms");
    }
    bool get_matrix(ShaderGlobals*, Matrix44& result, ustringhash) override
    { return lookup(result); }
    bool get_matrix(ShaderGlobals*, Matrix44& result, ustringhash,
                    float) override
    { return lookup(result); }
    bool get_inverse_matrix(ShaderGlobals*, Matrix44& result,
                            ustringhash) override
    { return lookup(result); }
    bool get_inverse_matrix(ShaderGlobals*, Matrix44& result, ustringhash,
                            float) override
    { return lookup(result); }
    bool get_matrix(ShaderGlobals*, Matrix44& result,
                    TransformationPtr) override
    { return lookup(result); }
    bool get_matrix(ShaderGlobals*, Matrix44& result, TransformationPtr,
                    float) override
    { return lookup(result); }
    bool get_inverse_matrix(ShaderGlobals*, Matrix44& result,
                            TransformationPtr) override
    { return lookup(result); }
    bool get_inverse_matrix(ShaderGlobals*, Matrix44& result, TransformationPtr,
                            float) override
    { return lookup(result); }
    bool transform_points(ShaderGlobals*, ustringhash, ustringhash, float,
                          const Vec3*, Vec3*, int,
                          TypeDesc::VECSEMANTICS) override
    {
        ++nonlinear_queries;
        return true;
    }

    bool transforms = true, named = true;
    int matrix_queries = 0, nonlinear_queries = 0;

private:
    bool lookup(Matrix44& result)
    {
        // Successful nonidentity answers expose accidental compile-time folds.
        ++matrix_queries;
        result.makeIdentity();
        result[0][0] = 7;
        result[3][0] = 19;
        return true;
    }
};



class HartAttributeServices final : public RendererServices {
public:
    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTArrayBounds"
               || feature == "HARTClosures"
               || feature == "build_attribute_getter"
               || (attributes && feature == "HARTAttributes");
    }
    void build_attribute_getter(const ShaderGroup&, bool, const ustring*,
                                const ustring*, bool, const int*, TypeDesc,
                                bool, AttributeGetterSpec&) override
    {
        ++builders;
    }
    bool get_attribute(ShaderGlobals*, bool, ustringhash, TypeDesc type,
                       ustringhash, void* data) override
    {
        ++host_queries;
        const float value = 123.0f;
        if (type == TypeFloat && data)
            std::memcpy(data, &value, sizeof(value));
        return type == TypeFloat && data;
    }
    bool get_array_attribute(ShaderGlobals*, bool, ustringhash, TypeDesc type,
                             ustringhash, int, void* data) override
    {
        ++host_array_queries;
        const float value = 456.0f;
        if (type == TypeFloat && data)
            std::memcpy(data, &value, sizeof(value));
        return type == TypeFloat && data;
    }

    bool attributes  = true;
    int host_queries = 0, host_array_queries = 0, builders = 0;
};



ShaderGroupRef
make_group(ShadingSystem& ss, string_view oso, int layers = 1,
           bool load_shader = true)
{
    if (load_shader)
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_test", oso));
    auto group = ss.ShaderGroupBegin("hart_test_group");
    for (int i = 0; i < layers; ++i)
        OIIO_CHECK_ASSERT(
            ss.Shader("surface", "hart_test", fmtformat("layer{}", i)));
    OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
    const SymLocationDesc output("Cout", TypeColor, false, SymArena::Outputs, 0,
                                 3 * sizeof(float));
    ss.add_symlocs(group.get(), { &output, 1 });
    return group;
}



void
check_rejected_group(ShadingSystem& ss, ShaderGroup& group,
                     const Diagnostics& errors, string_view expected)
{
    int allocated = 0;
    OIIO_CHECK_ASSERT(
        !ss.getattribute(&group, "hart_groupdata_alloc", allocated));
    ss.optimize_group(&group, nullptr);
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        !ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        !ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    OIIO_CHECK_ASSERT(
        !ss.getattribute(&group, "hart_groupdata_alloc", allocated));
    OIIO_CHECK_ASSERT(bytes == nullptr && size == 0);
    OIIO_CHECK_ASSERT(errors.errors > 0);
    const bool matched = OIIO::Strutil::contains(errors.last_error, expected);
    if (!matched)
        print(stderr, "Expected diagnostic '{}', got '{}'\n", expected,
              errors.last_error);
    OIIO_CHECK_ASSERT(matched);
}



ShaderGroupRef
make_connected_group(ShadingSystem& ss, string_view producer,
                     string_view consumer, bool load_shaders = true)
{
    if (load_shaders) {
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("hart_producer", producer));
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("hart_consumer", consumer));
    }
    auto group = ss.ShaderGroupBegin("hart_test_group");
    OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_producer", "producer"));
    OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_consumer", "consumer"));
    OIIO_CHECK_ASSERT(
        ss.ConnectShaders("producer", "value", "consumer", "value"));
    OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
    const SymLocationDesc output("consumer.Cout", TypeColor, false,
                                 SymArena::Outputs, 0, 3 * sizeof(float));
    ss.add_symlocs(group.get(), { &output, 1 });
    return group;
}



void
check_function_abi(const llvm::Function& function, string_view arch)
{
    OIIO_CHECK_ASSERT(!function.isDeclaration());
    OIIO_CHECK_ASSERT(function.getReturnType()->isVoidTy());
    OIIO_CHECK_ASSERT(!function.isVarArg());
    OIIO_CHECK_EQUAL(function.arg_size(), 6);
    OIIO_CHECK_EQUAL(
        function.getFnAttribute("target-cpu").getValueAsString().str(),
        std::string(arch));
    for (const auto& arg : function.args()) {
        if (arg.getArgNo() == 4)
            OIIO_CHECK_ASSERT(arg.getType()->isIntegerTy(32));
        else
            OIIO_CHECK_ASSERT(arg.getType()->isPointerTy()
                              && arg.getType()->getPointerAddressSpace() == 0);
    }
}



void
check_wrapper(const llvm::Function* wrapper,
              cspan<const llvm::Function*> targets,
              const llvm::StructType* storage = nullptr, int alignment = 0)
{
    OIIO_CHECK_ASSERT(wrapper);
    if (!wrapper)
        return;
    OIIO_CHECK_EQUAL(wrapper->size(), 1);
    const llvm::AllocaInst* allocation  = nullptr;
    const llvm::AddrSpaceCastInst* cast = nullptr;
    const llvm::Value* groupdata        = wrapper->getArg(1);
    size_t calls                        = 0;
    for (const auto& block : *wrapper)
        for (const auto& inst : block) {
            if (const auto* local = llvm::dyn_cast<llvm::AllocaInst>(&inst)) {
                OIIO_CHECK_ASSERT(storage && !allocation);
                OIIO_CHECK_EQUAL(local->getAllocatedType(), storage);
                OIIO_CHECK_EQUAL(local->getAddressSpace(), 5);
                OIIO_CHECK_EQUAL(local->getAlign().value(),
                                 uint64_t(alignment));
                const auto* count = llvm::dyn_cast<llvm::ConstantInt>(
                    local->getArraySize());
                OIIO_CHECK_ASSERT(count && count->isOne());
                allocation = local;
                continue;
            }
            if (const auto* flat = llvm::dyn_cast<llvm::AddrSpaceCastInst>(
                    &inst)) {
                OIIO_CHECK_ASSERT(storage && allocation && !cast);
                OIIO_CHECK_EQUAL(flat->getOperand(0), allocation);
                OIIO_CHECK_EQUAL(flat->getSrcAddressSpace(), 5);
                OIIO_CHECK_EQUAL(flat->getDestAddressSpace(), 0);
                cast      = flat;
                groupdata = flat;
                continue;
            }
            const auto* call = llvm::dyn_cast<llvm::CallInst>(&inst);
            if (!call) {
                OIIO_CHECK_ASSERT(llvm::isa<llvm::ReturnInst>(inst));
                continue;
            }
            const auto* target = calls < targets.size()
                                     ? *(targets.begin() + calls)
                                     : nullptr;
            ++calls;
            OIIO_CHECK_ASSERT(target);
            OIIO_CHECK_EQUAL(call->getCalledFunction(), target);
            if (target) {
                OIIO_CHECK_ASSERT(target->getName().find("__direct_callable__")
                                  != 0);
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 target->getCallingConv());
                OIIO_CHECK_EQUAL(wrapper->getFunctionType(),
                                 target->getFunctionType());
            }
            OIIO_CHECK_EQUAL(call->arg_size(), 6);
            if (storage)
                OIIO_CHECK_ASSERT(cast);
            for (unsigned int arg = 0;
                 arg < call->arg_size() && arg < wrapper->arg_size(); ++arg)
                OIIO_CHECK_EQUAL(call->getArgOperand(arg),
                                 arg == 1 ? groupdata : wrapper->getArg(arg));
        }
    OIIO_CHECK_EQUAL(calls, targets.size());
    OIIO_CHECK_EQUAL(allocation != nullptr, storage != nullptr);
    OIIO_CHECK_EQUAL(cast != nullptr, storage != nullptr);
}



void
check_wrapper(const llvm::Function* wrapper,
              std::initializer_list<const llvm::Function*> targets,
              const llvm::StructType* storage = nullptr, int alignment = 0)
{
    check_wrapper(wrapper,
                  cspan<const llvm::Function*>(targets.begin(), targets.size()),
                  storage, alignment);
}



void
check_noise_abi(llvm::Module& module, int optimize, int expected_guard_flags)
{
    const struct {
        const char* name;
        const char* args;
        char result = 'v';
    } signatures[] = {
        { "osl_noise_dfdv", "pp" },
        { "osl_snoise_dfdv", "pp" },
        { "osl_pnoise_dfdvv", "ppp" },
        { "osl_psnoise_dfdvv", "ppp" },
        { "osl_gabornoise_dfdf", "lpppp" },
        { "osl_gabornoise_dfdfdf", "lppppp" },
        { "osl_gabornoise_dfdv", "lpppp" },
        { "osl_gabornoise_dfdvdf", "lppppp" },
        { "osl_gabornoise_dvdf", "lpppp" },
        { "osl_gabornoise_dvdfdf", "lppppp" },
        { "osl_gabornoise_dvdv", "lpppp" },
        { "osl_gabornoise_dvdvdf", "lppppp" },
        { "osl_gaborpnoise_dfdff", "lppfpp" },
        { "osl_gaborpnoise_dfdfdfff", "lpppffpp" },
        { "osl_gaborpnoise_dfdvv", "lppppp" },
        { "osl_gaborpnoise_dfdvdfvf", "lppppfpp" },
        { "osl_gaborpnoise_dvdff", "lppfpp" },
        { "osl_gaborpnoise_dvdfdfff", "lpppffpp" },
        { "osl_gaborpnoise_dvdvv", "lppppp" },
        { "osl_gaborpnoise_dvdvdfvf", "lppppfpp" },
        { "osl_genericnoise_dfdf", "lpppp" },
        { "osl_genericnoise_dfdfdf", "lppppp" },
        { "osl_genericnoise_dfdv", "lpppp" },
        { "osl_genericnoise_dfdvdf", "lppppp" },
        { "osl_genericnoise_dvdf", "lpppp" },
        { "osl_genericnoise_dvdfdf", "lppppp" },
        { "osl_genericnoise_dvdv", "lpppp" },
        { "osl_genericnoise_dvdvdf", "lppppp" },
        { "osl_genericpnoise_dfdff", "lppfpp" },
        { "osl_genericpnoise_dfdfdfff", "lpppffpp" },
        { "osl_genericpnoise_dfdvv", "lppppp" },
        { "osl_genericpnoise_dfdvdfvf", "lppppfpp" },
        { "osl_genericpnoise_dvdff", "lppfpp" },
        { "osl_genericpnoise_dvdfdfff", "lpppffpp" },
        { "osl_genericpnoise_dvdvv", "lppppp" },
        { "osl_genericpnoise_dvdvdfvf", "lppppfpp" },
        { "osl_hart_noise_validate", "lpii", 'i' },
        { "osl_hash_is", "l", 'i' },
        { "osl_hash_ii", "i", 'i' },
        { "osl_hash_if", "f", 'i' },
        { "osl_hash_iff", "ff", 'i' },
        { "osl_hash_iv", "p", 'i' },
        { "osl_hash_ivf", "pf", 'i' },
        { "osl_init_noise_options", "pp" },
        { "osl_noiseparams_set_anisotropic", "pi" },
        { "osl_noiseparams_set_do_filter", "pi" },
        { "osl_noiseparams_set_direction", "pp" },
        { "osl_noiseparams_set_bandwidth", "pf" },
        { "osl_noiseparams_set_impulses", "pf" },
        { "rs_hart_noise_error", "p" },
    };
    const auto& layout       = module.getDataLayout();
    const auto* init_options = module.getFunction("osl_init_noise_options");
    const auto* validate     = module.getFunction("osl_hart_noise_validate");
    auto same_pointer        = [&](const llvm::Value* a, const llvm::Value* b) {
        int64_t a_offset = 0, b_offset = 0;
        const auto* a_base = llvm::GetPointerBaseWithConstantOffset(a, a_offset,
                                                                           layout);
        const auto* b_base = llvm::GetPointerBaseWithConstantOffset(b, b_offset,
                                                                           layout);
        return a_base == b_base && a_offset == b_offset;
    };
    auto same_selector = [&](const llvm::Value* a, const llvm::Value* b) {
        if (a == b)
            return true;
        const auto* a_load = llvm::dyn_cast<llvm::LoadInst>(a);
        const auto* b_load = llvm::dyn_cast<llvm::LoadInst>(b);
        return a_load && b_load && a_load->getType() == b_load->getType()
               && same_pointer(a_load->getPointerOperand(),
                               b_load->getPointerOperand());
    };
    std::vector<const llvm::CallBase*> used_resets;
    std::vector<const llvm::CallBase*> used_guards;
    int dual_calls = 0, initializers = 0, guard_flags = 0;
    for (const auto& signature : signatures) {
        auto* function = module.getFunction(signature.name);
        if (!function || function->use_empty())
            continue;
        const string_view name(signature.name), args(signature.args);
        const bool callback = name == "rs_hart_noise_error";
        const bool gabor    = OIIO::Strutil::starts_with(name, "osl_gabor");
        const bool generic  = OIIO::Strutil::starts_with(name, "osl_generic");
        bool original_call  = false;
        if (!callback && optimize == 10)
            for (auto* user : function->users()) {
                auto* call = llvm::dyn_cast<llvm::CallBase>(user);
                if (!call || call->getCalledFunction() != function)
                    continue;
                const auto caller = call->getFunction()->getName();
                if (caller.find("osl_layer_group_") != 0
                    && caller.find("osl_init_group_") != 0)
                    continue;
                original_call = true;
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 function->getCallingConv());
                initializers += name == "osl_init_noise_options";
                if (!gabor && !generic)
                    continue;
                ++dual_calls;
                OIIO_CHECK_EQUAL(call->arg_size(), args.size());
                if (call->arg_size() != args.size())
                    continue;
                const auto* selector = llvm::dyn_cast<llvm::ConstantInt>(
                    call->getArgOperand(0));
                OIIO_CHECK_ASSERT(
                    call->getArgOperand(0)->getType()->isIntegerTy(64));
                if (gabor)
                    OIIO_CHECK_ASSERT(selector);
                if (gabor && selector)
                    OIIO_CHECK_EQUAL(selector->getZExtValue(),
                                     ustringhash("gabor").hash());
                OIIO_CHECK_EQUAL(
                    call->getArgOperand(args.size() - 2)->stripPointerCasts(),
                    call->getFunction()->getArg(0));
                const auto* options = llvm::dyn_cast<llvm::AllocaInst>(
                    call->getArgOperand(args.size() - 1)->stripPointerCasts());
                OIIO_CHECK_ASSERT(options);
                if (options) {
                    OIIO_CHECK_EQUAL(options->getAddressSpace(), 5);
                    OIIO_CHECK_EQUAL(options->getAllocatedType(),
                                     llvm::StructType::getTypeByName(
                                         module.getContext(), "NoiseOptions"));
                    llvm::DominatorTree dominators(*call->getFunction());
                    const llvm::CallBase* reset = nullptr;
                    if (init_options)
                        for (const auto* init_user : init_options->users()) {
                            const auto* candidate
                                = llvm::dyn_cast<llvm::CallBase>(init_user);
                            if (!candidate
                                || candidate->getCalledFunction()
                                       != init_options
                                || candidate->getFunction()
                                       != call->getFunction()
                                || candidate->arg_size() != 2
                                || candidate->getArgOperand(1)
                                           ->stripPointerCasts()
                                       != options
                                || !dominators.dominates(candidate, call))
                                continue;
                            if (!reset
                                || dominators.dominates(reset, candidate))
                                reset = candidate;
                        }
                    if (!reset)
                        print(stderr,
                              "Missing dominating NoiseOptions reset "
                              "for {} in {}\n",
                              name, caller.str());
                    OIIO_CHECK_ASSERT(reset);
                    if (reset) {
                        OIIO_CHECK_EQUAL(
                            reset->getArgOperand(0)->stripPointerCasts(),
                            call->getFunction()->getArg(0));
                        // Each original call must reset its own options, not
                        // inherit a preceding call's option setters.
                        const bool fresh = std::find(used_resets.begin(),
                                                     used_resets.end(), reset)
                                           == used_resets.end();
                        if (!fresh)
                            print(stderr,
                                  "Reused NoiseOptions reset for {} "
                                  "in {}\n",
                                  name, caller.str());
                        OIIO_CHECK_ASSERT(fresh);
                        used_resets.push_back(reset);
                    }
                    if (generic) {
                        const llvm::CallBase* guard = nullptr;
                        if (validate)
                            for (const auto* guard_user : validate->users()) {
                                const auto* candidate
                                    = llvm::dyn_cast<llvm::CallBase>(
                                        guard_user);
                                if (!candidate
                                    || candidate->getCalledFunction()
                                           != validate
                                    || candidate->getFunction()
                                           != call->getFunction()
                                    || candidate->arg_size() != 4
                                    || !dominators.dominates(candidate, call)
                                    || !same_selector(candidate->getArgOperand(
                                                          0),
                                                      call->getArgOperand(0)))
                                    continue;
                                if (!guard
                                    || dominators.dominates(guard, candidate))
                                    guard = candidate;
                            }
                        if (!guard)
                            print(stderr,
                                  "Missing selector guard for {} in {}\n", name,
                                  caller.str());
                        OIIO_CHECK_ASSERT(guard);
                        if (!guard)
                            continue;
                        OIIO_CHECK_ASSERT(std::find(used_guards.begin(),
                                                    used_guards.end(), guard)
                                          == used_guards.end());
                        used_guards.push_back(guard);
                        OIIO_CHECK_EQUAL(
                            guard->getArgOperand(1)->stripPointerCasts(),
                            call->getFunction()->getArg(0));
                        const auto* periodic = llvm::dyn_cast<llvm::ConstantInt>(
                            guard->getArgOperand(2));
                        const auto* only_gabor
                            = llvm::dyn_cast<llvm::ConstantInt>(
                                guard->getArgOperand(3));
                        OIIO_CHECK_ASSERT(periodic && only_gabor);
                        if (periodic)
                            OIIO_CHECK_EQUAL(periodic->getZExtValue(),
                                             uint64_t(OIIO::Strutil::starts_with(
                                                 name, "osl_genericpnoise_")));
                        bool has_options = false;
                        if (reset)
                            for (const auto& block : *call->getFunction())
                                for (const auto& inst : block) {
                                    const auto* setter
                                        = llvm::dyn_cast<llvm::CallBase>(&inst);
                                    const auto* callee
                                        = setter ? setter->getCalledFunction()
                                                 : nullptr;
                                    if (callee
                                        && callee->getName().find(
                                               "osl_noiseparams_set_")
                                               == 0
                                        && setter->arg_size() > 0
                                        && setter->getArgOperand(0)
                                                   ->stripPointerCasts()
                                               == options
                                        && dominators.dominates(reset, setter)
                                        && dominators.dominates(setter, call))
                                        has_options = true;
                                }
                        if (only_gabor) {
                            OIIO_CHECK_EQUAL(only_gabor->getZExtValue(),
                                             uint64_t(has_options));
                            if (only_gabor->getZExtValue() <= 1)
                                guard_flags |= 1 << only_gabor->getZExtValue();
                        }
                        const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                            guard->getParent()->getTerminator());
                        const auto* cmp = branch && branch->isConditional()
                                              ? llvm::dyn_cast<llvm::ICmpInst>(
                                                    branch->getCondition())
                                              : nullptr;
                        OIIO_CHECK_ASSERT(cmp && cmp->isEquality());
                        if (!cmp || !cmp->isEquality())
                            continue;
                        const auto* zero = llvm::dyn_cast<llvm::ConstantInt>(
                            cmp->getOperand(cmp->getOperand(0) == guard ? 1
                                                                        : 0));
                        OIIO_CHECK_ASSERT(zero && zero->isZero()
                                          && (cmp->getOperand(0) == guard
                                              || cmp->getOperand(1) == guard));
                        const bool valid_on_true = cmp->getPredicate()
                                                   == llvm::CmpInst::ICMP_NE;
                        const auto* valid_block = branch->getSuccessor(
                            valid_on_true ? 0 : 1);
                        const auto* error_block = branch->getSuccessor(
                            valid_on_true ? 1 : 0);
                        OIIO_CHECK_ASSERT(
                            dominators.dominates(valid_block,
                                                 call->getParent()));
                        OIIO_CHECK_ASSERT(
                            !dominators.dominates(error_block,
                                                  call->getParent()));
                        const auto* error_exit
                            = llvm::dyn_cast<llvm::BranchInst>(
                                error_block->getTerminator());
                        const auto* valid_exit
                            = llvm::dyn_cast<llvm::BranchInst>(
                                call->getParent()->getTerminator());
                        OIIO_CHECK_ASSERT(
                            error_exit && error_exit->isUnconditional()
                            && valid_exit && valid_exit->isUnconditional());
                        if (error_exit && error_exit->isUnconditional()
                            && valid_exit && valid_exit->isUnconditional()) {
                            OIIO_CHECK_EQUAL(error_exit->getSuccessor(0),
                                             valid_exit->getSuccessor(0));
                            OIIO_CHECK_ASSERT(error_exit->getSuccessor(0)
                                              != call->getParent());
                        }
                        int clears = 0;
                        for (const auto& inst : *error_block)
                            if (const auto* clear
                                = llvm::dyn_cast<llvm::MemSetInst>(&inst)) {
                                ++clears;
                                const auto* value
                                    = llvm::dyn_cast<llvm::ConstantInt>(
                                        clear->getValue());
                                const auto* length
                                    = llvm::dyn_cast<llvm::ConstantInt>(
                                        clear->getLength());
                                OIIO_CHECK_ASSERT(value && value->isZero()
                                                  && length);
                                const auto suffix = name.substr(
                                    name.find_last_of('_') + 1);
                                const uint64_t bytes
                                    = (suffix[1] == 'f' ? 4 : 12)
                                      * (same_pointer(clear->getRawDest(),
                                                      call->getArgOperand(1))
                                             ? 3
                                             : 1);
                                if (length)
                                    OIIO_CHECK_EQUAL(length->getZExtValue(),
                                                     bytes);
                            }
                        OIIO_CHECK_EQUAL(clears, 1);
                    }
                }
                // Value-only results and constant/partial coordinates still
                // need full dual temporaries for generic/Gabor filtering.
                const auto suffix = name.substr(name.find_last_of('_') + 1);
                for (size_t i = 0; i + 1 < suffix.size() && suffix[i] == 'd';
                     i += 2) {
                    const auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(
                        call->getArgOperand(1 + i / 2)->stripPointerCasts());
                    if (!allocation)
                        continue;  // ShaderGlobals or Groupdata field.
                    const auto* count = llvm::dyn_cast<llvm::ConstantInt>(
                        allocation->getArraySize());
                    OIIO_CHECK_ASSERT(count);
                    if (count)
                        OIIO_CHECK_ASSERT(
                            layout.getTypeAllocSize(
                                      allocation->getAllocatedType())
                                    .getFixedValue()
                                * count->getZExtValue()
                            >= uint64_t(suffix[i + 1] == 'f' ? 12 : 36));
                }
            }
        // Only original calls retain the full shadeop ABI after specialization.
        if (!callback && !original_call)
            continue;
        OIIO_CHECK_EQUAL(function->isDeclaration(), callback);
        OIIO_CHECK_ASSERT(!function->isVarArg());
        OIIO_CHECK_ASSERT(signature.result == 'i'
                              ? function->getReturnType()->isIntegerTy(32)
                              : function->getReturnType()->isVoidTy());
        OIIO_CHECK_EQUAL(function->arg_size(), args.size());
        for (const auto& arg : function->args()) {
            if (arg.getArgNo() >= args.size())
                continue;
            const char kind = args[arg.getArgNo()];
            OIIO_CHECK_ASSERT(
                kind == 'p'
                    ? arg.getType()->isPointerTy()
                          && arg.getType()->getPointerAddressSpace() == 0
                    : (kind == 'f' ? arg.getType()->isFloatTy()
                                   : arg.getType()->isIntegerTy(
                                         kind == 'i' ? 32 : 64)));
        }
        if (name == "osl_hash_is" && function->arg_size() == 1
            && function->getArg(0)->getType()->isIntegerTy(64)
            && function->getReturnType()->isIntegerTy(32)) {
            auto* input = function->getArg(0)->getType();
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(input).getFixedValue(), 8);
            OIIO_CHECK_EQUAL(layout.getABITypeAlign(input).value(), 8);
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(function->getReturnType())
                                 .getFixedValue(),
                             4);
            int returns = 0;
            for (const auto& block : *function)
                if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(
                        block.getTerminator())) {
                    ++returns;
                    const auto* low = llvm::dyn_cast_or_null<llvm::TruncInst>(
                        ret->getReturnValue());
                    OIIO_CHECK_ASSERT(low && low->getType()->isIntegerTy(32));
                    if (low)
                        OIIO_CHECK_EQUAL(low->getOperand(0),
                                         function->getArg(0));
                }
            OIIO_CHECK_EQUAL(returns, 1);
        }
    }
    if (optimize == 10 && expected_guard_flags >= 0)
        OIIO_CHECK_EQUAL(guard_flags, expected_guard_flags);
    if (!dual_calls)
        return;
    OIIO_CHECK_EQUAL(initializers, dual_calls);
    auto* options = llvm::StructType::getTypeByName(module.getContext(),
                                                    "NoiseOptions");
    OIIO_CHECK_ASSERT(options && options->getNumElements() == 5);
    if (!options || options->getNumElements() != 5)
        return;
    const uint64_t offsets[] = { 0, 4, 8, 20, 24 };
    for (unsigned int i = 0; i < std::size(offsets); ++i)
        OIIO_CHECK_EQUAL(layout.getStructLayout(options)->getElementOffset(i),
                         offsets[i]);
    OIIO_CHECK_EQUAL(layout.getTypeAllocSize(options).getFixedValue(), 28);
    OIIO_CHECK_EQUAL(layout.getABITypeAlign(options).value(), 4);
    OIIO_CHECK_ASSERT(options->getElementType(0)->isIntegerTy(32)
                      && options->getElementType(1)->isIntegerTy(32)
                      && options->getElementType(3)->isFloatTy()
                      && options->getElementType(4)->isFloatTy());
    const auto* direction = llvm::dyn_cast<llvm::StructType>(
        options->getElementType(2));
    OIIO_CHECK_ASSERT(direction && direction->getNumElements() == 3);
    if (direction)
        for (const auto* component : direction->elements())
            OIIO_CHECK_ASSERT(component->isFloatTy());
}



void
check_color_abi(const llvm::Module& module, int optimize,
                int expected_transforms)
{
    int transforms = 0;
    const struct {
        const char* name;
        const char* args;
        char result;
    } signatures[] = {
        { "osl_blackbody_vf", "ppf", 'v' },
        { "osl_wavelength_color_vf", "ppf", 'v' },
        { "osl_luminance_fv", "ppp", 'v' },
        { "osl_luminance_dfdv", "ppp", 'v' },
        { "osl_prepend_color_from", "ppl", 'v' },
        { "osl_transformc", "ppipill", 'i' },
        { "rs_hart_get_colorsystem", "p", 'p' },
        { "rs_hart_color_error", "p", 'v' },
    };
    for (const auto& signature : signatures) {
        const auto* function = module.getFunction(signature.name);
        if (!function || function->use_empty())
            continue;
        const bool callback = OIIO::Strutil::starts_with(signature.name, "rs_");
        bool original_call  = false;
        if (!callback && optimize == 10)
            for (const auto* user : function->users())
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(user))
                    if (call->getCalledFunction() == function) {
                        const auto caller = call->getFunction()->getName();
                        original_call |= caller.find("osl_layer_group_") == 0
                                         || caller.find("osl_init_group_") == 0;
                        if (string_view(signature.name) == "osl_transformc"
                            && (caller.find("osl_layer_group_") == 0
                                || caller.find("osl_init_group_") == 0))
                            ++transforms;
                    }
        if (!callback && !original_call)
            continue;  // Internal specializations may drop original arguments.
        OIIO_CHECK_EQUAL(function->isDeclaration(), callback);
        OIIO_CHECK_ASSERT(!function->isVarArg());
        const auto* result = function->getReturnType();
        OIIO_CHECK_ASSERT(
            signature.result == 'p'
                ? result->isPointerTy() && result->getPointerAddressSpace() == 0
                : (signature.result == 'i' ? result->isIntegerTy(32)
                                           : result->isVoidTy()));
        const string_view args(signature.args);
        OIIO_CHECK_EQUAL(function->arg_size(), args.size());
        for (const auto& arg : function->args()) {
            if (arg.getArgNo() >= args.size())
                continue;
            const char kind = args[arg.getArgNo()];
            OIIO_CHECK_ASSERT(
                kind == 'p'
                    ? arg.getType()->isPointerTy()
                          && arg.getType()->getPointerAddressSpace() == 0
                    : (kind == 'f' ? arg.getType()->isFloatTy()
                                   : arg.getType()->isIntegerTy(
                                         kind == 'i' ? 32 : 64)));
        }
    }
    if (optimize == 10 && expected_transforms >= 0)
        OIIO_CHECK_EQUAL(transforms, expected_transforms);
    const auto* getter = module.getFunction("rs_hart_get_colorsystem");
    if (!getter || getter->use_empty())
        return;
    const auto* legacy = module.getFunction("rend_get_userdata");
    OIIO_CHECK_ASSERT(!legacy || legacy->use_empty());
}



bool
check_color_layout(string_view arch, string_view filename)
{
    auto file = llvm::MemoryBuffer::getFile(std::string(filename));
    if (!file) {
        print(stderr, "Cannot read color ABI probe '{}': {}\n", filename,
              file.getError().message());
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile((*file)->getMemBufferRef(), context);
    if (!parsed) {
        print(stderr, "Cannot parse color ABI probe: {}\n",
              llvm::toString(parsed.takeError()));
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    auto& module = **parsed;
    std::string error;
    llvm::raw_string_ostream diagnostic(error);
    OIIO_CHECK_ASSERT(!llvm::verifyModule(module, &diagnostic));
    if (!error.empty())
        print(stderr, "{}\n", error);
    const llvm::Triple triple(module.getTargetTriple());
    OIIO_CHECK_EQUAL(triple.getArch(), llvm::Triple::amdgcn);
    OIIO_CHECK_EQUAL(triple.getOS(), llvm::Triple::AMDHSA);
    const auto* address = module.getFunction("osl_hart_color_abi_address");
    OIIO_CHECK_ASSERT(address && !address->isDeclaration());
    if (address)
        OIIO_CHECK_EQUAL(
            address->getFnAttribute("target-cpu").getValueAsString().str(),
            std::string(arch));
    const auto* object = module.getNamedGlobal("osl_hart_color_abi_probe");
    OIIO_CHECK_ASSERT(object && object->isDeclaration()
                      && !object->use_empty());
    if (!object)
        return false;
    auto* type = llvm::dyn_cast<llvm::StructType>(object->getValueType());
    OIIO_CHECK_ASSERT(type && !type->isOpaque() && type->getNumElements() > 0);
    if (!type || type->isOpaque() || !type->getNumElements())
        return false;
    OIIO_CHECK_EQUAL(object->getAddressSpace(), 1);
    const auto& layout = module.getDataLayout();
    OIIO_CHECK_EQUAL(layout.getAllocaAddrSpace(), 5);
    OIIO_CHECK_EQUAL(layout.getPointerSize(0), 8);
    auto renderer_record = [&](const char* name, uint64_t size,
                               std::initializer_list<uint64_t> offsets) {
        const auto* object = module.getNamedGlobal(name);
        OIIO_CHECK_ASSERT(object && object->isDeclaration()
                          && !object->use_empty());
        if (!object)
            return;
        auto* record = llvm::dyn_cast<llvm::StructType>(object->getValueType());
        OIIO_CHECK_ASSERT(record && !record->isOpaque()
                          && record->getNumElements() == offsets.size());
        if (!record || record->isOpaque()
            || record->getNumElements() != offsets.size())
            return;
        OIIO_CHECK_EQUAL(object->getAddressSpace(), 1);
        OIIO_CHECK_EQUAL(layout.getTypeAllocSize(record).getFixedValue(), size);
        OIIO_CHECK_EQUAL(layout.getABITypeAlign(record).value(), 8);
        const auto* fields = layout.getStructLayout(record);
        unsigned index     = 0;
        for (uint64_t offset : offsets)
            OIIO_CHECK_EQUAL(fields->getElementOffset(index++), offset);
    };
    renderer_record("osl_hart_texture_abi_probe",
                    sizeof(testshade::HartTextureState),
                    { 0, 8, 16, 24, 32, 40, 48, 56 });
    renderer_record("osl_hart_transform_abi_probe",
                    sizeof(testshade::HartTransformState),
                    { 0, 8, 16, 24, 28 });
    renderer_record("osl_hart_transform_desc_abi_probe",
                    sizeof(testshade::HartTransformDesc), { 0, 8, 12, 16, 80 });
    for (const char* name : { "osl_hart_transform_abi_probe",
                              "osl_hart_transform_desc_abi_probe" }) {
        const auto* object = module.getNamedGlobal(name);
        auto* record       = object ? llvm::dyn_cast<llvm::StructType>(
                                          object->getValueType())
                                    : nullptr;
        if (!record || record->isOpaque() || record->getNumElements() != 5)
            continue;  // renderer_record reports missing/malformed records.
        const bool descriptor = string_view(name)
                                == "osl_hart_transform_desc_abi_probe";
        for (unsigned i = 0; i < 5; ++i) {
            auto* field = record->getElementType(i);
            if (descriptor && i >= 3) {
                const auto* matrix = llvm::dyn_cast<llvm::ArrayType>(field);
                OIIO_CHECK_ASSERT(matrix && matrix->getNumElements() == 16
                                  && matrix->getElementType()->isFloatTy());
            } else if (!descriptor && i == 0) {
                OIIO_CHECK_ASSERT(field->isPointerTy()
                                  && field->getPointerAddressSpace() == 0);
            } else
                OIIO_CHECK_ASSERT(field->isIntegerTy(
                    descriptor ? (i == 0 ? 64 : 32) : (i < 3 ? 64 : 32)));
        }
    }
    renderer_record("osl_hart_userdata_abi_probe",
                    sizeof(testshade::HartUserdataState),
                    { 0, 8, 16, 24, 32, 40, 44 });
    renderer_record("osl_hart_userdata_desc_abi_probe",
                    sizeof(testshade::HartUserdataDesc),
                    { 0, 8, 16, 24, 32, 40, 44 });
    renderer_record("osl_hart_attributes_abi_probe", sizeof(RenderContext),
                    { 0, 4, 8, 72, 80, 84, 100, 108, 112, 116, 120 });
    const auto* attributes = module.getNamedGlobal(
        "osl_hart_attributes_abi_probe");
    if (attributes) {
        auto* record = llvm::dyn_cast<llvm::StructType>(
            attributes->getValueType());
        if (record && record->getNumElements() == 11) {
            OIIO_CHECK_ASSERT(record->getElementType(0)->isIntegerTy(32)
                              && record->getElementType(1)->isIntegerTy(32));
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(record->getElementType(2))
                                 .getFixedValue(),
                             64);
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(record->getElementType(3))
                                 .getFixedValue(),
                             8);
            for (unsigned i : { 4, 7, 8, 9 })
                OIIO_CHECK_ASSERT(record->getElementType(i)->isFloatTy());
            for (unsigned i : { 5, 6 }) {
                const auto* array = llvm::dyn_cast<llvm::ArrayType>(
                    record->getElementType(i));
                OIIO_CHECK_ASSERT(array
                                  && array->getElementType()->isFloatTy());
                if (array)
                    OIIO_CHECK_EQUAL(array->getNumElements(), i == 5 ? 4 : 2);
            }
            OIIO_CHECK_ASSERT(
                record->getElementType(10)->isPointerTy()
                && record->getElementType(10)->getPointerAddressSpace() == 0);
        }
    }
    const uint64_t target_size  = layout.getTypeAllocSize(type).getFixedValue();
    const uint64_t target_align = layout.getABITypeAlign(type).value();
    const unsigned int last     = type->getNumElements() - 1;
    const uint64_t target_hash = layout.getStructLayout(type)->getElementOffset(
        last);
    OIIO_CHECK_ASSERT(object->getAlign());
    if (object->getAlign())
        OIIO_CHECK_EQUAL(object->getAlign()->value(), target_align);

    // Inspect the real target class, not a fabricated layout or host-size
    // assumption. The public upload protocol supplies the independent CPU ABI.
    RendererServices renderer;
    Diagnostics errors;
    ShadingSystem ss(&renderer, nullptr, &errors);
    const void* host_data = nullptr;
    long long sizes[2]    = { };
    OIIO_CHECK_ASSERT(
        ss.getattribute("colorsystem", TypeDesc::PTR, &host_data));
    OIIO_CHECK_ASSERT(ss.getattribute("colorsystem:sizes",
                                      TypeDesc(TypeDesc::LONGLONG, 2), sizes));
    OIIO_CHECK_ASSERT(host_data && sizes[0] > 0 && sizes[1] == 1);
    if (!host_data || sizes[0] <= 0 || sizes[1] != 1)
        return false;
    const auto* host       = static_cast<const pvt::ColorSystem*>(host_data);
    const auto hash_offset = reinterpret_cast<const char*>(&host->colorspace())
                             - static_cast<const char*>(host_data);
    OIIO_CHECK_EQUAL(uint64_t(hash_offset),
                     uint64_t(sizes[0] - sizeof(ustringhash)));
    const auto pointer_free = [](const auto& self, llvm::Type* type) -> bool {
        if (auto* record = llvm::dyn_cast<llvm::StructType>(type)) {
            if (record->isOpaque())
                return false;
            for (auto* field : record->elements())
                if (!self(self, field))
                    return false;
            return true;
        }
        if (auto* array = llvm::dyn_cast<llvm::ArrayType>(type))
            return self(self, array->getElementType());
        if (auto* vector = llvm::dyn_cast<llvm::VectorType>(type))
            return self(self, vector->getElementType());
        return type->isIntegerTy() || type->isFloatingPointTy();
    };
    OIIO_CHECK_ASSERT(pointer_free(pointer_free, type));
    OIIO_CHECK_EQUAL(target_size, uint64_t(sizes[0]));
    OIIO_CHECK_EQUAL(target_align, uint64_t(alignof(pvt::ColorSystem)));
    OIIO_CHECK_EQUAL(target_hash, uint64_t(hash_offset));
    OIIO_CHECK_EQUAL(
        layout.getTypeAllocSize(type->getElementType(last)).getFixedValue(),
        sizeof(ustringhash));

    const auto* diagnostic_address = module.getFunction(
        "osl_hart_diagnostic_abi_address");
    const auto* diagnostic_object = module.getNamedGlobal(
        "osl_hart_diagnostic_abi_probe");
    OIIO_CHECK_ASSERT(diagnostic_address
                      && !diagnostic_address->isDeclaration());
    OIIO_CHECK_ASSERT(diagnostic_object && diagnostic_object->isDeclaration()
                      && !diagnostic_object->use_empty());
    if (!diagnostic_address || !diagnostic_object)
        return false;
    OIIO_CHECK_EQUAL(diagnostic_address->getFnAttribute("target-cpu")
                         .getValueAsString()
                         .str(),
                     std::string(arch));
    OIIO_CHECK_EQUAL(diagnostic_object->getAddressSpace(), 1);
    auto* buffer = llvm::dyn_cast<llvm::StructType>(
        diagnostic_object->getValueType());
    OIIO_CHECK_ASSERT(buffer && !buffer->isOpaque()
                      && buffer->getNumElements() == 3);
    if (!buffer || buffer->isOpaque() || buffer->getNumElements() != 3)
        return false;
    auto* records = llvm::dyn_cast<llvm::ArrayType>(buffer->getElementType(2));
    auto* record  = records ? llvm::dyn_cast<llvm::StructType>(
                                 records->getElementType())
                            : nullptr;
    OIIO_CHECK_ASSERT(record && !record->isOpaque()
                      && record->getNumElements() == 10);
    if (!record || record->isOpaque() || record->getNumElements() != 10)
        return false;
    OIIO_CHECK_EQUAL(HartDiagnosticCapacity, 256);
    OIIO_CHECK_EQUAL(HartDiagnosticMaxArgs, 256);
    OIIO_CHECK_EQUAL(HartDiagnosticMaxValues, 2048);
    OIIO_CHECK_EQUAL(HartDiagnosticMaxFormat, 4096);
    OIIO_CHECK_EQUAL(HartDiagnosticMaxMessage, 4096);
    OIIO_CHECK_EQUAL(HartDiagnosticMaxField, 1024);
    OIIO_CHECK_EQUAL(int(HartDiagnosticSeverity::Print), 0);
    OIIO_CHECK_EQUAL(int(HartDiagnosticSeverity::Warning), 1);
    OIIO_CHECK_EQUAL(int(HartDiagnosticSeverity::Error), 2);
    OIIO_CHECK_ASSERT(pointer_free(pointer_free, buffer));
    OIIO_CHECK_ASSERT(buffer->getElementType(0)->isIntegerTy(32)
                      && buffer->getElementType(1)->isIntegerTy(32));
    const uint64_t buffer_offsets[] = { 0, 4, 8 };
    for (unsigned i = 0; i < std::size(buffer_offsets); ++i)
        OIIO_CHECK_EQUAL(layout.getStructLayout(buffer)->getElementOffset(i),
                         buffer_offsets[i]);
    OIIO_CHECK_EQUAL(records->getNumElements(), 256);
    OIIO_CHECK_EQUAL(layout.getTypeAllocSize(buffer).getFixedValue(), 602120);
    OIIO_CHECK_EQUAL(layout.getABITypeAlign(buffer).value(), 8);
    OIIO_CHECK_EQUAL(layout.getTypeAllocSize(buffer).getFixedValue(),
                     sizeof(HartDiagnosticBuffer));
    const uint64_t record_offsets[] = { 0, 8, 16, 24, 28, 32, 40, 44, 48, 304 };
    for (unsigned i = 0; i < std::size(record_offsets); ++i)
        OIIO_CHECK_EQUAL(layout.getStructLayout(record)->getElementOffset(i),
                         record_offsets[i]);
    const unsigned widths[] = { 64, 64, 64, 32, 32, 64, 32, 32 };
    for (unsigned i = 0; i < std::size(widths); ++i)
        OIIO_CHECK_ASSERT(record->getElementType(i)->isIntegerTy(widths[i]));
    for (unsigned i : { 8, 9 }) {
        auto* array = llvm::dyn_cast<llvm::ArrayType>(
            record->getElementType(i));
        OIIO_CHECK_ASSERT(array && array->getElementType()->isIntegerTy(8));
        if (array)
            OIIO_CHECK_EQUAL(array->getNumElements(), i == 8 ? 256 : 2048);
    }
    OIIO_CHECK_EQUAL(layout.getTypeAllocSize(record).getFixedValue(), 2352);
    OIIO_CHECK_EQUAL(layout.getABITypeAlign(record).value(), 8);
    OIIO_CHECK_EQUAL(layout.getTypeAllocSize(record).getFixedValue(),
                     sizeof(HartDiagnosticRecord));
    OIIO_CHECK_ASSERT(diagnostic_object->getAlign());
    if (diagnostic_object->getAlign())
        OIIO_CHECK_EQUAL(diagnostic_object->getAlign()->value(), 8);

    // Cross-check the probe against the actual embedded shadeops, whose record
    // type may already have disappeared during HIP frontend optimization.
    const auto bytes = pvt::hart_shadeops_bitcode(arch, errors);
    OIIO_CHECK_EQUAL(errors.errors, 0);
    OIIO_CHECK_ASSERT(!bytes.empty());
    if (bytes.empty())
        return false;
    llvm::LLVMContext production_context;
    const llvm::StringRef data(reinterpret_cast<const char*>(bytes.data()),
                               bytes.size());
    auto production
        = llvm::parseBitcodeFile(llvm::MemoryBufferRef(data, "hart_shadeops"),
                                 production_context);
    if (!production) {
        print(stderr, "{}\n", llvm::toString(production.takeError()));
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    const auto& shadeops = **production;
    OIIO_CHECK_EQUAL(triple.str(),
                     llvm::Triple(shadeops.getTargetTriple()).str());
    OIIO_CHECK_EQUAL(module.getDataLayoutStr(), shadeops.getDataLayoutStr());
    OIIO_CHECK_ASSERT(!shadeops.getNamedGlobal("osl_hart_color_abi_probe"));
    OIIO_CHECK_ASSERT(!shadeops.getFunction("osl_hart_color_abi_address"));
    OIIO_CHECK_ASSERT(!shadeops.getNamedGlobal("osl_hart_diagnostic_abi_probe"));
    OIIO_CHECK_ASSERT(!shadeops.getFunction("osl_hart_diagnostic_abi_address"));
    OIIO_CHECK_ASSERT(!shadeops.getNamedGlobal("osl_hart_texture_abi_probe"));
    OIIO_CHECK_ASSERT(!shadeops.getNamedGlobal("osl_hart_userdata_abi_probe"));
    OIIO_CHECK_ASSERT(
        !shadeops.getNamedGlobal("osl_hart_userdata_desc_abi_probe"));
    OIIO_CHECK_ASSERT(
        !shadeops.getNamedGlobal("osl_hart_attributes_abi_probe"));
    OIIO_CHECK_ASSERT(!shadeops.getFunction("osl_hart_attributes_abi_address"));
    OIIO_CHECK_ASSERT(!shadeops.getNamedGlobal("osl_hart_transform_abi_probe"));
    OIIO_CHECK_ASSERT(
        !shadeops.getNamedGlobal("osl_hart_transform_desc_abi_probe"));
    OIIO_CHECK_ASSERT(
        !shadeops.getFunction("osl_hart_transform_abi_addresses"));
    OIIO_CHECK_ASSERT(!shadeops.getFunction("osl_hart_userdata_abi_addresses"));
    int setters = 0;
    for (const auto& setter : shadeops) {
        if (setter.getName().find("11ColorSystem14set_colorspace")
            == llvm::StringRef::npos)
            continue;
        ++setters;
        OIIO_CHECK_ASSERT(!setter.isDeclaration());
        OIIO_CHECK_EQUAL(setter.arg_size(), 2);
        if (setter.arg_size() != 2)
            continue;
        OIIO_CHECK_EQUAL(
            setter.getFnAttribute("target-cpu").getValueAsString().str(),
            std::string(arch));
        const auto alignment
            = setter.getAttributes().getParamAttr(0,
                                                  llvm::Attribute::Alignment);
        const auto size = setter.getAttributes().getParamAttr(
            0, llvm::Attribute::Dereferenceable);
        OIIO_CHECK_ASSERT(alignment.isIntAttribute() && size.isIntAttribute());
        if (alignment.isIntAttribute())
            OIIO_CHECK_EQUAL(alignment.getValueAsInt(), target_align);
        if (size.isIntAttribute())
            OIIO_CHECK_EQUAL(size.getValueAsInt(), target_size);
        bool hash_load = false;
        for (const auto& block : setter)
            for (const auto& inst : block) {
                const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(
                    &inst);
                if (!gep
                    || gep->getPointerOperand()->stripPointerCasts()
                           != setter.getArg(0))
                    continue;
                llvm::APInt offset(shadeops.getDataLayout().getIndexSizeInBits(
                                       gep->getPointerAddressSpace()),
                                   0);
                if (!gep->accumulateConstantOffset(shadeops.getDataLayout(),
                                                   offset)
                    || offset.getZExtValue() != target_hash)
                    continue;
                for (const auto* user : gep->users())
                    if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(user))
                        hash_load |= load->getPointerOperand() == gep
                                     && load->getType()->isIntegerTy(64);
            }
        OIIO_CHECK_ASSERT(hash_load);
    }
    OIIO_CHECK_EQUAL(setters, 1);
    return true;
}



struct DiagnosticExpectation {
    std::string format;
    std::vector<EncodedType> types;
    ustring shader, source;
    int line;
    HartDiagnosticSeverity severity;
    std::vector<std::pair<unsigned, uint64_t>> literals;
};



void
check_diagnostic_abi(llvm::Module& module, int optimize,
                     cspan<DiagnosticExpectation> expected)
{
    auto* bridge                = module.getFunction("osl_hart_diagnostic");
    auto* callback              = module.getFunction("rs_hart_diagnostic");
    const string_view signature = "plipipillii";
    for (auto* function : { bridge, callback }) {
        if (!function || function->use_empty()
            || (function == bridge && optimize != 10))
            continue;
        OIIO_CHECK_EQUAL(function->isDeclaration(), function == callback);
        OIIO_CHECK_ASSERT(function->getReturnType()->isVoidTy()
                          && !function->isVarArg());
        OIIO_CHECK_EQUAL(function->arg_size(), signature.size());
        for (const auto& arg : function->args()) {
            if (arg.getArgNo() >= signature.size())
                continue;
            const char kind = signature[arg.getArgNo()];
            OIIO_CHECK_ASSERT(
                kind == 'p'
                    ? arg.getType()->isPointerTy()
                          && arg.getType()->getPointerAddressSpace() == 0
                    : arg.getType()->isIntegerTy(kind == 'l' ? 64 : 32));
        }
    }
    if (!expected.empty())
        OIIO_CHECK_ASSERT(callback && !callback->use_empty());
    if (optimize != 10 || expected.empty())
        return;
    OIIO_CHECK_ASSERT(bridge && !bridge->isDeclaration()
                      && bridge->arg_size() == signature.size() && callback
                      && callback->isDeclaration());
    if (!bridge || bridge->isDeclaration()
        || bridge->arg_size() != signature.size() || !callback)
        return;
    int forwards = 0;
    for (const auto& block : *bridge)
        for (const auto& inst : block)
            if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                if (call->getCalledFunction() == callback) {
                    ++forwards;
                    OIIO_CHECK_EQUAL(call->getCallingConv(),
                                     callback->getCallingConv());
                    OIIO_CHECK_EQUAL(call->arg_size(), signature.size());
                    OIIO_CHECK_EQUAL(call->getCallingConv(),
                                     callback->getCallingConv());
                    for (unsigned i = 0;
                         i < call->arg_size() && i < bridge->arg_size(); ++i)
                        OIIO_CHECK_EQUAL(call->getArgOperand(i),
                                         bridge->getArg(i));
                }
    OIIO_CHECK_EQUAL(forwards, 1);
    const auto& layout = module.getDataLayout();
    std::vector<int> seen(expected.size(), 0);
    for (auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0
            && function.getName().find("osl_init_group_") != 0)
            continue;
        llvm::DominatorTree dominators(function);
        for (const auto& block : function)
            for (const auto& inst : block) {
                const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst);
                if (!call || call->getCalledFunction() != bridge)
                    continue;
                OIIO_CHECK_EQUAL(call->arg_size(), signature.size());
                OIIO_CHECK_EQUAL(function.arg_size(), 6);
                if (call->arg_size() != signature.size()
                    || function.arg_size() != 6)
                    continue;
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 bridge->getCallingConv());
                OIIO_CHECK_EQUAL(call->getArgOperand(0)->stripPointerCasts(),
                                 function.getArg(0));
                OIIO_CHECK_EQUAL(call->getArgOperand(10), function.getArg(4));
                const auto* shader = llvm::dyn_cast<llvm::ConstantInt>(
                    call->getArgOperand(7));
                const auto* line = llvm::dyn_cast<llvm::ConstantInt>(
                    call->getArgOperand(9));
                OIIO_CHECK_ASSERT(shader && line);
                if (!shader || !line)
                    continue;
                const auto found = std::find_if(
                    expected.begin(), expected.end(), [&](const auto& test) {
                        return shader->getZExtValue()
                                   == ustringhash(test.shader).hash()
                               && line->getSExtValue() == test.line;
                    });
                if (found == expected.end()) {
                    print(stderr,
                          "Unexpected diagnostic at shader hash {}:{}\n",
                          shader->getZExtValue(), line->getSExtValue());
                    OIIO_CHECK_ASSERT(false);
                    continue;
                }
                ++seen[found - expected.begin()];
                const auto& test = *found;
                std::vector<uint64_t> offsets;
                uint64_t bytes = 0;
                for (auto type : test.types) {
                    offsets.push_back(bytes);
                    bytes += type == EncodedType::kUstringHash ? 8 : 4;
                }
                const std::pair<unsigned, uint64_t> constants[] = {
                    { 1, ustringhash(test.format).hash() },
                    { 2, test.types.size() },
                    { 4, bytes },
                    { 6, uint64_t(test.severity) },
                    { 7, ustringhash(test.shader).hash() },
                    { 8, ustringhash(test.source).hash() },
                    { 9, uint64_t(test.line) },
                };
                for (const auto& constant : constants) {
                    const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                        call->getArgOperand(constant.first));
                    OIIO_CHECK_ASSERT(value);
                    if (value)
                        OIIO_CHECK_EQUAL(value->getZExtValue(),
                                         constant.second);
                }
                const llvm::AllocaInst* buffers[2] = {};
                for (unsigned i = 0; i < 2; ++i) {
                    const auto* pointer = call->getArgOperand(i ? 5 : 3);
                    OIIO_CHECK_ASSERT(
                        pointer->getType()->isPointerTy()
                        && pointer->getType()->getPointerAddressSpace() == 0);
                    buffers[i] = llvm::dyn_cast<llvm::AllocaInst>(
                        pointer->stripPointerCasts());
                    OIIO_CHECK_ASSERT(buffers[i]);
                    if (!buffers[i])
                        continue;
                    OIIO_CHECK_EQUAL(buffers[i]->getAddressSpace(), 5);
                    OIIO_CHECK_ASSERT(
                        buffers[i]->getAllocatedType()->isIntegerTy(8));
                    const auto* count = llvm::dyn_cast<llvm::ConstantInt>(
                        buffers[i]->getArraySize());
                    OIIO_CHECK_ASSERT(count);
                    if (count)
                        OIIO_CHECK_EQUAL(count->getZExtValue(),
                                         i ? bytes : test.types.size());
                }
                if (!buffers[0] || !buffers[1])
                    continue;
                std::vector<int> type_stores(test.types.size(), 0);
                std::vector<const llvm::StoreInst*> value_stores(
                    test.types.size(), nullptr);
                for (const auto& store_block : function)
                    for (const auto& candidate : store_block) {
                        const auto* store = llvm::dyn_cast<llvm::StoreInst>(
                            &candidate);
                        if (!store || !dominators.dominates(store, call))
                            continue;
                        int64_t offset = 0;
                        const auto* base
                            = llvm::GetPointerBaseWithConstantOffset(
                                store->getPointerOperand(), offset, layout);
                        if (base == buffers[0]) {
                            OIIO_CHECK_ASSERT(offset >= 0
                                              && uint64_t(offset)
                                                     < test.types.size());
                            if (offset < 0
                                || uint64_t(offset) >= test.types.size())
                                continue;
                            ++type_stores[offset];
                            const auto* type = llvm::dyn_cast<llvm::ConstantInt>(
                                store->getValueOperand());
                            OIIO_CHECK_ASSERT(type && type->getBitWidth() == 8);
                            if (type)
                                OIIO_CHECK_EQUAL(type->getZExtValue(),
                                                 uint64_t(test.types[offset]));
                        } else if (base == buffers[1]) {
                            const auto slot = std::find(offsets.begin(),
                                                        offsets.end(),
                                                        uint64_t(offset));
                            OIIO_CHECK_ASSERT(slot != offsets.end());
                            if (slot == offsets.end())
                                continue;
                            const size_t index = slot - offsets.begin();
                            OIIO_CHECK_ASSERT(!value_stores[index]);
                            value_stores[index] = store;
                            OIIO_CHECK_EQUAL(store->getAlign().value(), 1);
                            auto* type = store->getValueOperand()->getType();
                            const auto encoded = test.types[index];
                            OIIO_CHECK_ASSERT(
                                encoded == EncodedType::kFloat
                                    ? type->isFloatTy()
                                    : type->isIntegerTy(
                                          encoded == EncodedType::kUstringHash
                                              ? 64
                                              : 32));
                            OIIO_CHECK_EQUAL(
                                layout.getTypeAllocSize(type).getFixedValue(),
                                encoded == EncodedType::kUstringHash ? 8 : 4);
                        }
                    }
                for (size_t i = 0; i < test.types.size(); ++i) {
                    OIIO_CHECK_EQUAL(type_stores[i], 1);
                    OIIO_CHECK_ASSERT(value_stores[i]);
                }
                for (const auto& literal : test.literals) {
                    OIIO_CHECK_ASSERT(literal.first < value_stores.size());
                    if (literal.first >= value_stores.size()
                        || !value_stores[literal.first])
                        continue;
                    const auto* value
                        = value_stores[literal.first]->getValueOperand();
                    if (const auto* integer = llvm::dyn_cast<llvm::ConstantInt>(
                            value)) {
                        OIIO_CHECK_EQUAL(integer->getZExtValue(),
                                         literal.second);
                    } else if (const auto* fp
                               = llvm::dyn_cast<llvm::ConstantFP>(value)) {
                        OIIO_CHECK_EQUAL(
                            fp->getValueAPF().bitcastToAPInt().getZExtValue(),
                            literal.second);
                    } else {
                        OIIO_CHECK_ASSERT(false);
                    }
                }
            }
    }
    for (int calls : seen)
        OIIO_CHECK_EQUAL(calls, 1);
}



void
check_module(ShadingSystem& ss, ShaderGroup& group, string_view arch,
             std::initializer_list<string_view> shadeops, int optimize,
             bool connected = false, bool branching = false,
             bool looping = false, int used_layers = 0, bool closures = false,
             bool aggregates = false, int spline_arraylen = 0,
             int color_transforms = -1, int noise_guard_flags = -1,
             cspan<DiagnosticExpectation> diagnostics   = { },
             bool eager_layers                          = false,
             cspan<std::pair<int, int>> closure_layouts = { },
             ustring spline_basis                       = ustring())
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    OIIO_CHECK_ASSERT(bytes && size > 4);
    if (!bytes || size <= 4)
        return;
    llvm::LLVMContext context;
    const llvm::StringRef data(static_cast<const char*>(bytes), size);
    auto parsed
        = llvm::parseBitcodeFile(llvm::MemoryBufferRef(data, "hart_test"),
                                 context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        OIIO_CHECK_ASSERT(false);
        return;
    }
    auto& module = **parsed;
    const llvm::Triple triple(module.getTargetTriple());
    OIIO_CHECK_EQUAL(triple.getArch(), llvm::Triple::amdgcn);
    OIIO_CHECK_EQUAL(triple.getOS(), llvm::Triple::AMDHSA);
    std::string error;
    llvm::raw_string_ostream diagnostic(error);
    OIIO_CHECK_ASSERT(!llvm::verifyModule(module, &diagnostic));
    if (!error.empty())
        print(stderr, "{}\n", error);
    OIIO_CHECK_EQUAL(module.getDataLayout().getAllocaAddrSpace(), 5);
    check_color_abi(module, optimize, color_transforms);
    check_noise_abi(module, optimize, noise_guard_flags);
    check_diagnostic_abi(module, optimize, diagnostics);
    if (closures) {
        const auto& layout = module.getDataLayout();
        OIIO_CHECK_EQUAL(layout.getPointerSize(0), 8);
        if (optimize == 10) {
            auto* component
                = llvm::StructType::getTypeByName(context, "ClosureComponent");
            if (module.getFunction("osl_allocate_closure_component")
                || module.getFunction("osl_allocate_weighted_closure_component"))
                OIIO_CHECK_ASSERT(component);
            if (component) {
                OIIO_CHECK_EQUAL(component->getNumElements(), 3);
                const auto* fields = layout.getStructLayout(component);
                OIIO_CHECK_EQUAL(fields->getElementOffset(0), 0);
                OIIO_CHECK_EQUAL(fields->getElementOffset(1), 4);
                OIIO_CHECK_EQUAL(fields->getElementOffset(2), 16);
            }
        }
        const struct {
            const char* name;
            const char* args;
        } signatures[] = {
            { "osl_allocate_closure_component", "pii" },
            { "osl_allocate_weighted_closure_component", "piip" },
            { "osl_add_closure_closure", "ppp" },
            { "osl_mul_closure_color", "ppp" },
            { "osl_mul_closure_float", "ppf" },
            { "rs_allocate_closure", "pll" },
        };
        for (const auto& signature : signatures) {
            const auto* function = module.getFunction(signature.name);
            if (!function)
                continue;  // Null trees and optimization can remove shadeops.
            OIIO_CHECK_EQUAL(function->isDeclaration(),
                             string_view(signature.name)
                                 == "rs_allocate_closure");
            OIIO_CHECK_ASSERT(!function->isVarArg());
            OIIO_CHECK_ASSERT(function->getReturnType()->isPointerTy());
            OIIO_CHECK_EQUAL(function->getReturnType()->getPointerAddressSpace(),
                             0);
            OIIO_CHECK_EQUAL(function->arg_size(),
                             string_view(signature.args).size());
            for (const auto& arg : function->args()) {
                if (arg.getArgNo() >= string_view(signature.args).size())
                    continue;
                const char kind = signature.args[arg.getArgNo()];
                if (kind == 'p')
                    OIIO_CHECK_ASSERT(arg.getType()->isPointerTy()
                                      && arg.getType()->getPointerAddressSpace()
                                             == 0);
                else if (kind == 'f')
                    OIIO_CHECK_ASSERT(arg.getType()->isFloatTy());
                else
                    OIIO_CHECK_ASSERT(
                        arg.getType()->isIntegerTy(kind == 'i' ? 32 : 64));
            }
            if (optimize != 10 || function->isDeclaration())
                continue;
            const bool component = string_view(signature.name).find("component")
                                   != string_view::npos;
            int allocations      = 0;
            for (const auto& block : *function)
                for (const auto& inst : block) {
                    const auto* call   = llvm::dyn_cast<llvm::CallBase>(&inst);
                    const auto* callee = call ? call->getCalledFunction()
                                              : nullptr;
                    if (!callee || callee->getName() != "rs_allocate_closure")
                        continue;
                    ++allocations;
                    const auto* alignment = llvm::dyn_cast<llvm::ConstantInt>(
                        call->getArgOperand(2));
                    OIIO_CHECK_ASSERT(alignment);
                    if (alignment)
                        OIIO_CHECK_EQUAL(alignment->getZExtValue(),
                                         component ? 16 : 8);
                    if (!component) {
                        const auto* bytes = llvm::dyn_cast<llvm::ConstantInt>(
                            call->getArgOperand(1));
                        OIIO_CHECK_ASSERT(bytes);
                        if (bytes)
                            OIIO_CHECK_EQUAL(bytes->getZExtValue(), 24);
                    }
                }
            OIIO_CHECK_EQUAL(allocations, 1);
        }
        for (const auto& function : module) {
            if (function.getName().find("osl_layer_group_") != 0
                && function.getName().find("osl_init_group_") != 0
                && function.getName().find("__direct_callable__") != 0)
                continue;
            for (const auto& block : function)
                for (const auto& inst : block) {
                    if (const auto* cast = llvm::dyn_cast<llvm::IntToPtrInst>(
                            &inst)) {
                        // offset_ptr uses runtime pointer arithmetic; only
                        // literal non-null addresses would embed host memory.
                        const auto* address = llvm::dyn_cast<llvm::ConstantInt>(
                            cast->getOperand(0));
                        OIIO_CHECK_ASSERT(!address || address->isZero());
                    }
                    if (const auto* call = llvm::dyn_cast<llvm::CallBase>(
                            &inst)) {
                        OIIO_CHECK_ASSERT(!call->isIndirectCall());
                        const auto* callee = call->getCalledFunction();
                        if (callee
                            && (callee->getName()
                                    == "osl_allocate_closure_component"
                                || callee->getName()
                                       == "osl_allocate_weighted_closure_component")) {
                            const auto* id = llvm::dyn_cast<llvm::ConstantInt>(
                                call->getArgOperand(1));
                            const auto* size = llvm::dyn_cast<llvm::ConstantInt>(
                                call->getArgOperand(2));
                            OIIO_CHECK_ASSERT(id && size);
                            if (id && size) {
                                if (closure_layouts.empty()) {
                                    OIIO_CHECK_ASSERT(id->getZExtValue() == 1
                                                      || id->getZExtValue()
                                                             == 3);
                                    OIIO_CHECK_EQUAL(size->getZExtValue(),
                                                     id->getZExtValue() == 1
                                                         ? 1
                                                         : 24);
                                } else {
                                    const auto layout = std::find_if(
                                        closure_layouts.begin(),
                                        closure_layouts.end(),
                                        [&](const auto& p) {
                                            return uint64_t(p.first)
                                                   == id->getZExtValue();
                                        });
                                    OIIO_CHECK_ASSERT(layout
                                                      != closure_layouts.end());
                                    if (layout != closure_layouts.end())
                                        OIIO_CHECK_EQUAL(size->getZExtValue(),
                                                         uint64_t(
                                                             layout->second));
                                }
                            }
                        }
                    }
                    for (const auto& operand : inst.operands())
                        if (const auto* expr
                            = llvm::dyn_cast<llvm::ConstantExpr>(operand.get()))
                            OIIO_CHECK_ASSERT(expr->getOpcode()
                                              != llvm::Instruction::IntToPtr);
                }
        }
    }
    bool provenance = false;
    for (const auto& global : module.globals()) {
        provenance |= global.getName().contains("__hart_device_storage_abi");
        OIIO_CHECK_ASSERT(global.isDeclaration()
                          || !global.hasExternalLinkage());
    }
    OIIO_CHECK_ASSERT(provenance);
    OIIO_CHECK_ASSERT(module.getNamedGlobal("llvm.compiler.used"));
    int size_bytes = 0, alignment = 0, allocated = -1;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_size", size_bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_alignment", alignment));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_groupdata_alloc", allocated));
    OIIO_CHECK_ASSERT(size_bytes > 0 && alignment > 0);
    OIIO_CHECK_EQUAL(size_bytes % alignment, 0);
    OIIO_CHECK_ASSERT(allocated == 0 || allocated == size_bytes);
    if (aggregates && optimize == 10) {
        auto* storage = llvm::StructType::getTypeByName(context, "Groupdata");
        OIIO_CHECK_ASSERT(storage);
        if (storage) {
            const auto& layout = module.getDataLayout();
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(storage).getFixedValue(),
                             uint64_t(size_bytes));
            OIIO_CHECK_EQUAL(layout.getABITypeAlign(storage).value(),
                             uint64_t(alignment));
        }
        const auto* callback = module.getFunction("rs_hart_range_error");
        if (callback && !callback->use_empty()) {
            OIIO_CHECK_ASSERT(callback->isDeclaration());
            OIIO_CHECK_ASSERT(callback->getReturnType()->isVoidTy());
            OIIO_CHECK_ASSERT(!callback->isVarArg());
            OIIO_CHECK_EQUAL(callback->arg_size(), 3);
            for (const auto& arg : callback->args())
                OIIO_CHECK_ASSERT(
                    arg.getArgNo() == 0
                        ? arg.getType()->isPointerTy()
                              && arg.getType()->getPointerAddressSpace() == 0
                        : arg.getType()->isIntegerTy(32));
        }
    }
    const char* queries[]       = { "group_init_name", "group_entry_name",
                                    "group_fused_name" };
    llvm::Function* wrappers[3] = { };
    for (size_t i = 0; i < std::size(queries); ++i) {
        ustring name;
        OIIO_CHECK_ASSERT(ss.getattribute(&group, queries[i], name));
        auto* function = module.getFunction(name.c_str());
        wrappers[i]    = function;
        OIIO_CHECK_ASSERT(function && !function->isDeclaration());
        if (!function)
            continue;
        OIIO_CHECK_ASSERT(function->hasExternalLinkage());
        OIIO_CHECK_ASSERT(function->getName().find("__direct_callable__") == 0);
        check_function_abi(*function, arch);
    }
    if (wrappers[0])
        OIIO_CHECK_EQUAL(wrappers[0]->getName().str(),
                         "__direct_callable__osl_init_group_hart_test_group");
    if (wrappers[1]) {
        const llvm::StringRef entry_prefix(
            "__direct_callable__osl_layer_group_");
        const bool has_prefix = wrappers[1]->getName().find(entry_prefix) == 0;
        OIIO_CHECK_ASSERT(has_prefix);
        if (has_prefix && wrappers[2])
            OIIO_CHECK_EQUAL(wrappers[2]->getName().str(),
                             fmtformat("__direct_callable__fused_{}",
                                       wrappers[1]
                                           ->getName()
                                           .drop_front(entry_prefix.size())
                                           .str()));
    }
    llvm::Function* bodies[2] = { };
    if (optimize == 10) {
        const llvm::StringRef prefix("__direct_callable__");
        for (size_t i = 0; i < std::size(bodies); ++i) {
            if (!wrappers[i] || wrappers[i]->getName().find(prefix) != 0)
                continue;
            auto* body = module.getFunction(
                wrappers[i]->getName().drop_front(prefix.size()));
            bodies[i] = body;
            OIIO_CHECK_ASSERT(body);
            if (body) {
                OIIO_CHECK_ASSERT(body->hasLocalLinkage());
                check_function_abi(*body, arch);
            }
        }
        check_wrapper(wrappers[0], { bodies[0] });
        check_wrapper(wrappers[1], { bodies[1] });
        auto* storage = allocated > 0
                            ? llvm::StructType::getTypeByName(context,
                                                              "Groupdata")
                            : nullptr;
        if (allocated > 0) {
            OIIO_CHECK_ASSERT(storage);
            if (storage)
                OIIO_CHECK_EQUAL(module.getDataLayout()
                                     .getTypeAllocSize(storage)
                                     .getFixedValue(),
                                 uint64_t(allocated));
        }
        check_wrapper(wrappers[2], { bodies[0], bodies[1] }, storage,
                      alignment);
    }
    auto* init    = bodies[0];
    auto* entry   = bodies[1];
    int callables = 0;
    for (const auto& function : module) {
        if (function.getName().find("__direct_callable__") == 0)
            ++callables;
        if (function.getName().find("osl_") == 0 && !function.use_empty()) {
            if (function.isDeclaration())
                print("Undefined used shadeop '{}' (arch {}, LLVM {})\n",
                      function.getName().str(), arch, optimize);
            OIIO_CHECK_ASSERT(!function.isDeclaration());
        }
        const bool spline = function.getName().find("osl_spline_") == 0
                            || function.getName().find("osl_splineinverse_")
                                   == 0;
        const bool spline_validator = function.getName()
                                      == "osl_hart_spline_validate";
        const bool spline_error = function.getName() == "rs_hart_spline_error";
        bool original_spline_call = false;
        if (optimize == 10 && (spline || spline_validator))
            for (const auto* user : function.users())
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(user))
                    if (call->getCalledFunction() == &function) {
                        const auto caller = call->getFunction()->getName();
                        original_spline_call
                            |= caller.find("osl_layer_group_") == 0
                               || caller.find("osl_init_group_") == 0;
                    }
        // Optimized internal specializations may drop arguments. Only the
        // original shadeop calls have this ABI; renderer callbacks always do.
        if (!function.use_empty() && (original_spline_call || spline_error)) {
            OIIO_CHECK_EQUAL(function.isDeclaration(), spline_error);
            OIIO_CHECK_ASSERT(!function.isVarArg());
            OIIO_CHECK_ASSERT(spline_validator
                                  ? function.getReturnType()->isIntegerTy(32)
                                  : function.getReturnType()->isVoidTy());
            OIIO_CHECK_EQUAL(function.arg_size(),
                             spline ? 6 : (spline_validator ? 5 : 1));
            for (const auto& arg : function.args()) {
                const unsigned int n = arg.getArgNo();
                if (spline && n == 1)
                    OIIO_CHECK_ASSERT(arg.getType()->isIntegerTy(64));
                else if ((spline && n >= 4) || (spline_validator && n < 3))
                    OIIO_CHECK_ASSERT(arg.getType()->isIntegerTy(32));
                else if (spline_validator && n == 3)
                    OIIO_CHECK_ASSERT(arg.getType()->isFloatTy());
                else
                    OIIO_CHECK_ASSERT(arg.getType()->isPointerTy()
                                      && arg.getType()->getPointerAddressSpace()
                                             == 0);
            }
        }
        OIIO_CHECK_ASSERT(!function.hasFnAttribute("nvptx-f32ftz"));
        const auto cpu = function.getFnAttribute("target-cpu");
        if (cpu.isStringAttribute())
            OIIO_CHECK_EQUAL(cpu.getValueAsString().str(), std::string(arch));
        for (const auto& block : function)
            for (const auto& inst : block) {
                if (const auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(
                        &inst))
                    OIIO_CHECK_EQUAL(allocation->getAddressSpace(), 5);
                if (const auto* cast = llvm::dyn_cast<llvm::IntToPtrInst>(&inst))
                    if (const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                            cast->getOperand(0)))
                        OIIO_CHECK_ASSERT(value->isZero());
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                    if (const auto* callee = call->getCalledFunction())
                        OIIO_CHECK_ASSERT(
                            callee->getName().find("__direct_callable__") != 0);
            }
    }
    OIIO_CHECK_EQUAL(callables, 3);
    if (spline_arraylen && optimize == 10) {
        int spline_calls  = 0;
        int spline_checks = 0;
        for (auto& function : module) {
            if (function.getName().find("osl_layer_group_") != 0)
                continue;
            llvm::DominatorTree dominators(function);
            for (const auto& block : function)
                for (const auto& inst : block) {
                    const auto* call   = llvm::dyn_cast<llvm::CallBase>(&inst);
                    const auto* callee = call ? call->getCalledFunction()
                                              : nullptr;
                    if (callee && !spline_basis.empty()
                        && callee->getName() == "osl_hart_spline_validate") {
                        ++spline_checks;
                        OIIO_CHECK_EQUAL(call->arg_size(), 5);
                        if (call->arg_size() == 5) {
                            const auto* step = llvm::dyn_cast<llvm::ConstantInt>(
                                call->getArgOperand(2));
                            OIIO_CHECK_ASSERT(step);
                            if (step)
                                OIIO_CHECK_EQUAL(
                                    step->getSExtValue(),
                                    spline_basis == ustring("bezier")    ? 3
                                    : spline_basis == ustring("hermite") ? 2
                                                                         : 1);
                        }
                    }
                    if (!callee
                        || (callee->getName().find("osl_spline_") != 0
                            && callee->getName().find("osl_splineinverse_")
                                   != 0))
                        continue;
                    ++spline_calls;
                    OIIO_CHECK_EQUAL(call->arg_size(), 6);
                    if (call->arg_size() != 6)
                        continue;
                    if (!spline_basis.empty()) {
                        const auto* basis = llvm::dyn_cast<llvm::ConstantInt>(
                            call->getArgOperand(1));
                        OIIO_CHECK_ASSERT(basis);
                        if (basis)
                            OIIO_CHECK_EQUAL(basis->getLimitedValue(),
                                             ustringhash(spline_basis).hash());
                    }
                    const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                        call->getArgOperand(5));
                    OIIO_CHECK_ASSERT(length);
                    if (length)
                        OIIO_CHECK_EQUAL(length->getSExtValue(),
                                         spline_arraylen);
                    bool guarded = false;
                    for (const auto& candidate : function) {
                        const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                            candidate.getTerminator());
                        if (!branch || !branch->isConditional()
                            || !dominators.dominates(&candidate, &block))
                            continue;
                        guarded
                            |= dominators.dominates(branch->getSuccessor(0),
                                                    &block)
                               != dominators.dominates(branch->getSuccessor(1),
                                                       &block);
                    }
                    OIIO_CHECK_ASSERT(guarded);
                }
        }
        OIIO_CHECK_ASSERT(spline_calls > 0);
        if (!spline_basis.empty())
            OIIO_CHECK_EQUAL(spline_checks, spline_calls);
    }
    if (looping && optimize == 10) {
        OIIO_CHECK_ASSERT(entry);
        if (entry) {
            llvm::DominatorTree dominators(*entry);
            llvm::LoopInfo loops(dominators);
            OIIO_CHECK_ASSERT(!loops.empty());
            bool loop_shadeop  = false;
            bool loop_producer = false;
            for (const auto& block : *entry) {
                if (!loops.getLoopFor(&block))
                    continue;
                for (const auto& inst : block) {
                    const auto* call   = llvm::dyn_cast<llvm::CallInst>(&inst);
                    const auto* callee = call ? call->getCalledFunction()
                                              : nullptr;
                    if (!callee)
                        continue;
                    loop_shadeop |= callee->getName().find("osl_sin_") == 0;
                    loop_producer
                        |= callee->getName()
                           == "osl_layer_group_hart_test_group_name_producer";
                }
            }
            OIIO_CHECK_ASSERT(loop_shadeop);
            if (connected)
                OIIO_CHECK_ASSERT(loop_producer);
        }
    }
    if (branching && optimize == 10) {
        OIIO_CHECK_ASSERT(entry);
        if (entry) {
            llvm::DominatorTree dominators(*entry);
            bool varying_branch       = false;
            bool conditional_producer = false;
            const auto* producer      = module.getFunction(
                "osl_layer_group_hart_test_group_name_producer");
            for (const auto& block : *entry) {
                for (const auto& inst : block) {
                    const auto* cmp    = llvm::dyn_cast<llvm::FCmpInst>(&inst);
                    const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                        block.getTerminator());
                    if (!cmp
                        || (cmp->getPredicate() != llvm::CmpInst::FCMP_OGT
                            && cmp->getPredicate() != llvm::CmpInst::FCMP_UGT)
                        || !branch || !branch->isConditional())
                        continue;
                    varying_branch = true;
                    if (producer)
                        for (const auto* user : producer->users())
                            if (const auto* call
                                = llvm::dyn_cast<llvm::CallInst>(user))
                                if (call->getFunction() == entry) {
                                    conditional_producer = true;
                                    OIIO_CHECK_ASSERT(dominators.dominates(
                                        branch->getSuccessor(0),
                                        call->getParent()));
                                }
                }
            }
            OIIO_CHECK_ASSERT(varying_branch);
            if (connected)
                OIIO_CHECK_ASSERT(conditional_producer);
        }
    }
    if (connected && optimize == 10) {
        const auto* producer = module.getFunction(
            "osl_layer_group_hart_test_group_name_producer");
        OIIO_CHECK_ASSERT(producer && !producer->isDeclaration()
                          && producer->hasLocalLinkage());
        if (producer) {
            OIIO_CHECK_EQUAL(producer->arg_size(), 6);
            OIIO_CHECK_EQUAL(
                producer->getFnAttribute("target-cpu").getValueAsString().str(),
                std::string(arch));
            bool calls_producer = false;
            if (entry)
                for (const auto& block : *entry)
                    for (const auto& inst : block)
                        if (const auto* call = llvm::dyn_cast<llvm::CallInst>(
                                &inst))
                            if (call->getCalledFunction() == producer) {
                                calls_producer = true;
                                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                                 producer->getCallingConv());
                                OIIO_CHECK_EQUAL(call->arg_size(), 6);
                                for (unsigned int arg = 0; arg < 6; ++arg)
                                    OIIO_CHECK_EQUAL(call->getArgOperand(arg),
                                                     entry->getArg(arg));
                            }
            OIIO_CHECK_ASSERT(calls_producer);
        }
    }
    for (const auto name : shadeops) {
        if (name.empty() || optimize != 10)
            continue;
        const auto* shadeop = module.getFunction(std::string(name));
        const bool callback = OIIO::Strutil::starts_with(name, "rs_");
        const bool used     = shadeop && shadeop->isDeclaration() == callback
                              && !shadeop->use_empty();
        if (!used) {
            const char* state = !shadeop ? "missing"
                                : shadeop->isDeclaration() != callback
                                    ? "wrong declaration/definition kind"
                                    : "unused";
            print("Expected a used {} '{}': {} (arch {}, LLVM {})\n",
                  callback ? "renderer callback declaration" : "linked shadeop",
                  name, state, arch, optimize);
        }
        OIIO_CHECK_ASSERT(used);
        const bool scalar_derivs = name == "osl_sin_dfdf"
                                   || name == "osl_filterwidth_fdf"
                                   || OIIO::Strutil::contains(name, "noise_df");
        const bool vector_derivs = name == "osl_normalize_dvdv"
                                   || name == "osl_filterwidth_vdv"
                                   || OIIO::Strutil::contains(name, "noise_dv");
        if (connected && (scalar_derivs || vector_derivs)) {
            const auto* storage = llvm::StructType::getTypeByName(context,
                                                                  "Groupdata");
            OIIO_CHECK_ASSERT(storage);
            bool dual_storage = false;
            if (storage)
                for (const auto* field : storage->elements())
                    if (const auto* array = llvm::dyn_cast<llvm::ArrayType>(
                            field)) {
                        const auto* element = array->getElementType();
                        const auto* triple  = llvm::dyn_cast<llvm::StructType>(
                            element);
                        const bool vector
                            = triple && triple->getNumElements() == 3
                              && triple->getElementType(0)->isFloatTy()
                              && triple->getElementType(1)->isFloatTy()
                              && triple->getElementType(2)->isFloatTy();
                        dual_storage |= array->getNumElements() == 3
                                        && (scalar_derivs ? element->isFloatTy()
                                                          : vector);
                    }
            OIIO_CHECK_ASSERT(dual_storage);
        }
    }
    if (used_layers && optimize == 10) {
        auto* storage = llvm::StructType::getTypeByName(context, "Groupdata");
        OIIO_CHECK_ASSERT(storage);
        if (storage) {
            const auto& layout = module.getDataLayout();
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(storage).getFixedValue(),
                             uint64_t(size_bytes));
            OIIO_CHECK_EQUAL(layout.getABITypeAlign(storage).value(),
                             uint64_t(alignment));
            const auto* flags = llvm::dyn_cast<llvm::ArrayType>(
                storage->getElementType(0));
            OIIO_CHECK_ASSERT(flags);
            if (flags) {
                OIIO_CHECK_ASSERT(flags->getElementType()->isIntegerTy(1));
                OIIO_CHECK_EQUAL(flags->getNumElements(),
                                 uint64_t((used_layers + 3) & ~3));
            }
        }
        int internal_layers = 0;
        auto flag_index     = [&](const llvm::Value* pointer,
                                  const llvm::Function& function) {
            if (pointer->stripPointerCasts() == function.getArg(1))
                return 0;
            const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(
                pointer->stripPointerCasts());
            if (!gep || gep->getSourceElementType() != storage
                || gep->getNumIndices() != 3
                || gep->getPointerOperand()->stripPointerCasts()
                       != function.getArg(1))
                return -1;
            const auto* base = llvm::dyn_cast<llvm::ConstantInt>(
                gep->getOperand(1));
            const auto* field = llvm::dyn_cast<llvm::ConstantInt>(
                gep->getOperand(2));
            const auto* index = llvm::dyn_cast<llvm::ConstantInt>(
                gep->getOperand(3));
            return base && base->isZero() && field && field->isZero() && index
                           && index->getLimitedValue() < uint64_t(used_layers)
                       ? int(index->getZExtValue())
                       : -1;
        };
        std::vector<bool> flags_written(used_layers - 1, false);
        for (auto& function : module) {
            if (function.getName().find("osl_layer_group_hart_test_group_name_")
                != 0)
                continue;
            OIIO_CHECK_ASSERT(function.hasLocalLinkage());
            check_function_abi(function, arch);
            if (&function == entry)
                continue;
            ++internal_layers;
            int flag = -1, stores = 0;
            for (const auto& block : function)
                for (const auto& inst : block) {
                    const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst);
                    if (!store)
                        continue;
                    const int index = flag_index(store->getPointerOperand(),
                                                 function);
                    if (index < 0)
                        continue;
                    ++stores;
                    flag              = index;
                    const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                        store->getValueOperand());
                    OIIO_CHECK_ASSERT(value && value->isOne());
                    OIIO_CHECK_EQUAL(store->getParent(),
                                     &function.getEntryBlock());
                }
            OIIO_CHECK_EQUAL(stores, 1);
            OIIO_CHECK_ASSERT(flag >= 0 && flag < used_layers - 1);
            if (flag >= 0 && flag < used_layers - 1) {
                OIIO_CHECK_ASSERT(!flags_written[flag]);
                flags_written[flag] = true;
            }
            int calls = 0;
            for (auto* user : function.users()) {
                auto* call = llvm::dyn_cast<llvm::CallInst>(user);
                OIIO_CHECK_ASSERT(call);
                if (!call)
                    continue;
                ++calls;
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 function.getCallingConv());
                OIIO_CHECK_EQUAL(call->arg_size(), 6);
                for (unsigned int arg = 0; arg < 6; ++arg)
                    OIIO_CHECK_EQUAL(call->getArgOperand(arg),
                                     call->getFunction()->getArg(arg));
                llvm::DominatorTree dominators(*call->getFunction());
                bool guarded = false;
                for (const auto& block : *call->getFunction()) {
                    const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                        block.getTerminator());
                    if (!branch || !branch->isConditional())
                        continue;
                    const auto* cmp = llvm::dyn_cast<llvm::ICmpInst>(
                        branch->getCondition());
                    if (!cmp || cmp->getPredicate() != llvm::CmpInst::ICMP_NE)
                        continue;
                    const auto* load = llvm::dyn_cast<llvm::LoadInst>(
                        cmp->getOperand(0));
                    const auto* ran = llvm::dyn_cast<llvm::ConstantInt>(
                        cmp->getOperand(1));
                    guarded |= flag >= 0 && load && ran && ran->isOne()
                               && flag_index(load->getPointerOperand(),
                                             *call->getFunction())
                                      == flag
                               && dominators.dominates(branch->getSuccessor(0),
                                                       call->getParent());
                }
                if (eager_layers) {
                    for (const auto& block : *call->getFunction())
                        if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(
                                block.getTerminator()))
                            OIIO_CHECK_ASSERT(dominators.dominates(call, ret));
                } else {
                    OIIO_CHECK_ASSERT(guarded);
                }
            }
            OIIO_CHECK_ASSERT(calls > 0);
        }
        OIIO_CHECK_EQUAL(internal_layers, used_layers - 1);
        bool resets_flags = false;
        if (init)
            for (const auto& block : *init)
                for (const auto& inst : block) {
                    const auto* clear = llvm::dyn_cast<llvm::MemSetInst>(&inst);
                    if (!clear || flag_index(clear->getRawDest(), *init) != 0)
                        continue;
                    const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                        clear->getValue());
                    const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                        clear->getLength());
                    resets_flags |= value && value->isZero() && length
                                    && length->getZExtValue()
                                           == uint64_t((used_layers + 3) & ~3);
                }
        OIIO_CHECK_ASSERT(resets_flags);
    }
    const void* again = nullptr;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &again));
    OIIO_CHECK_EQUAL(bytes, again);
}



void
check_rejection(string_view arch, string_view oso, string_view expected,
                int layers = 1, bool instrument = false, bool textures = false)
{
    HartServices renderer(textures);
    Diagnostics errors;
    ShadingSystem ss(&renderer, nullptr, &errors);
    OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
    if (instrument)
        ss.attribute("debug_nan", 1);
    auto group = make_group(ss, oso, layers);
    check_rejected_group(ss, *group, errors, expected);
}



void
check_groupdata_alloc_settings()
{
    const string_view option("max_hart_groupdata_alloc");
    {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("error_repeats", 1));
        int budget = -1;
        OIIO_CHECK_ASSERT(ss.getattribute(option, budget));
        OIIO_CHECK_EQUAL(budget, 0);
        OIIO_CHECK_ASSERT(ss.attribute(option, 64));
        OIIO_CHECK_ASSERT(ss.getattribute(option, budget));
        OIIO_CHECK_EQUAL(budget, 64);
        auto reject = [&](TypeDesc type, const void* value) {
            const int previous_errors = errors.errors;
            OIIO_CHECK_ASSERT(!ss.attribute(option, type, value));
            OIIO_CHECK_EQUAL(errors.errors, previous_errors + 1);
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, option));
            OIIO_CHECK_ASSERT(ss.getattribute(option, budget));
            OIIO_CHECK_EQUAL(budget, 64);
        };
        const int negative   = -1;
        const float floating = 64.0f;
        const char* text     = "64";
        const int array[]    = { 64, 64 };
        reject(TypeDesc::INT, &negative);
        reject(TypeDesc::FLOAT, &floating);
        reject(TypeDesc::STRING, &text);
        reject(TypeDesc(TypeDesc::INT, 2), array);
        OIIO_CHECK_ASSERT(ss.attribute(option, 0));
        OIIO_CHECK_ASSERT(ss.getattribute(option, budget));
        OIIO_CHECK_EQUAL(budget, 0);
        float wrong_type = -1.0f;
        OIIO_CHECK_ASSERT(!ss.getattribute(option, wrong_type));
    }
    {
        RendererServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("error_repeats", 1));
        for (int budget : { 0, 64 }) {
            const int previous_errors = errors.errors;
            OIIO_CHECK_ASSERT(!ss.attribute(option, budget));
            OIIO_CHECK_EQUAL(errors.errors, previous_errors + 1);
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, option));
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, "HART"));
            int unchanged = -1;
            OIIO_CHECK_ASSERT(ss.getattribute(option, unchanged));
            OIIO_CHECK_EQUAL(unchanged, 0);
        }
    }
}



void
check_groupdata_alloc_modules(string_view arch, string_view producer,
                              string_view consumer)
{
    for (int osl_optimize : { 0, 2 })
        for (int optimize : { 10, 3 }) {
            int required = 0;
            {
                HartServices renderer;
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
                OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
                auto group    = make_connected_group(ss, producer, consumer);
                int allocated = -1;
                OIIO_CHECK_ASSERT(!ss.getattribute(group.get(),
                                                   "hart_groupdata_alloc",
                                                   allocated));
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "{}\n", errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "llvm_groupdata_size",
                                                  required));
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "hart_groupdata_alloc",
                                                  allocated));
                OIIO_CHECK_EQUAL(allocated, 0);
                check_module(ss, *group, arch, { "osl_sin_dfdf" }, optimize,
                             true, false, false, 2);
            }
            OIIO_CHECK_ASSERT(required > 0);
            if (required <= 0)
                continue;
            for (int budget : { 0, required - 1, required, required + 1 }) {
                HartServices renderer;
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
                OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
                OIIO_CHECK_ASSERT(
                    ss.attribute("max_hart_groupdata_alloc", budget));
                int configured = -1;
                OIIO_CHECK_ASSERT(
                    ss.getattribute("max_hart_groupdata_alloc", configured));
                OIIO_CHECK_EQUAL(configured, budget);
                const int expected = budget > 0 && required <= budget ? required
                                                                      : 0;
                auto check_compiled = [&](ShaderGroup& group, int allocation) {
                    if (errors.errors)
                        print(stderr, "{}\n", errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    int size = 0, allocated = -1;
                    OIIO_CHECK_ASSERT(
                        ss.getattribute(&group, "llvm_groupdata_size", size));
                    OIIO_CHECK_EQUAL(size, required);
                    OIIO_CHECK_ASSERT(ss.getattribute(&group,
                                                      "hart_groupdata_alloc",
                                                      allocated));
                    OIIO_CHECK_EQUAL(allocated, allocation);
                    float wrong_type = -1.0f;
                    OIIO_CHECK_ASSERT(!ss.getattribute(&group,
                                                       "hart_groupdata_alloc",
                                                       wrong_type));
                    check_module(ss, group, arch, { "osl_sin_dfdf" }, optimize,
                                 true, false, false, 2);
                };
                auto group    = make_connected_group(ss, producer, consumer);
                int allocated = -1;
                OIIO_CHECK_ASSERT(!ss.getattribute(group.get(),
                                                   "hart_groupdata_alloc",
                                                   allocated));
                ss.optimize_group(group.get(), nullptr);
                check_compiled(*group, expected);
                const void* bytes = nullptr;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                                  TypeDesc::PTR, &bytes));
                auto future_group = make_connected_group(ss, producer, consumer,
                                                         false);
                OIIO_CHECK_ASSERT(!ss.getattribute(future_group.get(),
                                                   "hart_groupdata_alloc",
                                                   allocated));
                const int future_budget = expected ? 0 : required;
                OIIO_CHECK_ASSERT(
                    ss.attribute("max_hart_groupdata_alloc", future_budget));
                OIIO_CHECK_ASSERT(
                    ss.getattribute("max_hart_groupdata_alloc", configured));
                OIIO_CHECK_EQUAL(configured, future_budget);
                // A budget change affects compilation, not an existing module.
                ss.optimize_group(group.get(), nullptr);
                check_compiled(*group, expected);
                OIIO_CHECK_ASSERT(!ss.getattribute(future_group.get(),
                                                   "hart_groupdata_alloc",
                                                   allocated));
                ss.optimize_group(future_group.get(), nullptr);
                check_compiled(*future_group, future_budget);
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "hart_groupdata_alloc",
                                                  allocated));
                OIIO_CHECK_EQUAL(allocated, expected);
                const void* again = nullptr;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                                  TypeDesc::PTR, &again));
                OIIO_CHECK_EQUAL(bytes, again);
            }
        }
}



bool
check_chain_modules(string_view arch, string_view stdosl)
{
    for (string_view type :
         { "float", "color", "point", "vector", "normal", "matrix" }) {
        const auto expression
            = type == "matrix" ? "incoming*matrix(1+u*v+u+2*v)"
                               : fmtformat("0.5*incoming+{}(u*v+u+2*v)", type);
        const auto source = fmtformat(
            "shader hart_chain({0} incoming={1},output {0} value=0) {{ "
            "value={2}; }}",
            type, type == "matrix" ? 1 : 0, expression);
        const auto terminal = fmtformat(
            "shader hart_chain_end({} value=0,output color Cout=0) {{ "
            "float x={}; Cout=color(x,Dx(x),Dy(x)); }}",
            type,
            type == "matrix"  ? "value[0][0]"
            : type == "float" ? "value"
                              : "dot(vector(value),vector(1,2,3))");
        OSLCompiler relay_compiler, output_compiler;
        std::string relay, output;
        if (!relay_compiler.compile_buffer(source, relay, { }, stdosl)
            || !output_compiler.compile_buffer(terminal, output, { }, stdosl))
            return false;
        for (int layers : { 3, 5, 9 })
            for (int osl_optimize : { 0, 2 })
                for (int optimize : { 10, 3 }) {
                    HartServices renderer;
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("optimize", osl_optimize);
                    ss.attribute("llvm_optimize", optimize);
                    OIIO_CHECK_ASSERT(
                        ss.LoadMemoryCompiledShader("hart_chain", relay));
                    OIIO_CHECK_ASSERT(
                        ss.LoadMemoryCompiledShader("hart_chain_end", output));
                    auto group = ss.ShaderGroupBegin("hart_test_group");
                    for (int i = 0; i < layers; ++i) {
                        const bool last = i == layers - 1;
                        OIIO_CHECK_ASSERT(
                            ss.Shader("surface",
                                      last ? "hart_chain_end" : "hart_chain",
                                      fmtformat("layer{}", i)));
                        if (i)
                            OIIO_CHECK_ASSERT(
                                ss.ConnectShaders(fmtformat("layer{}", i - 1),
                                                  "value",
                                                  fmtformat("layer{}", i),
                                                  last ? "value" : "incoming"));
                    }
                    OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
                    const SymLocationDesc result(
                        fmtformat("layer{}.Cout", layers - 1), TypeColor, false,
                        SymArena::Outputs, 0, 3 * sizeof(float));
                    ss.add_symlocs(group.get(), { &result, 1 });
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "{}\n", errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    check_module(ss, *group, arch, { }, optimize, false, false,
                                 false, layers);
                }
    }
    return true;
}



bool
check_topology_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_root(output float value=0) { value=sin(u*v); }",
        "shader hart_left(float value=0,output float result=0) { result=2*value+v; }",
        "shader hart_right(float value=0,output float result=0) { result=3*value-u; }",
        "shader hart_join(float a=0,float b=0,int reuse=0,output color Cout=0) { "
        "float x=0; if(u>v) x=a; else x=b; if(reuse) x+=a+b; "
        "Cout=color(x,Dx(x),Dy(x)); }",
        "shader hart_unused(output float value=0) { value=u*v; }",
        "shader hart_bad_unused(output float value=0) { "
        "if (u<0) printf(\"unsupported\"); value=u*v; }",
    };
    std::string bytecode[6];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
            return false;
    }
    for (int osl_optimize : { 0, 2 })
        for (int optimize : { 10, 3 })
            for (int reuse : { 0, 1 })
                for (bool reject : { false, true }) {
                    HartServices renderer;
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("optimize", osl_optimize);
                    ss.attribute("llvm_optimize", optimize);
                    const char* names[] = { "root", "left", "right", "join",
                                            "unused" };
                    for (int i = 0; i < 5; ++i)
                        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader(
                            names[i], bytecode[i == 4 && reject ? 5 : i]));
                    auto group = ss.ShaderGroupBegin("hart_test_group");
                    for (int i : { 0, 4, 1, 2, 3 }) {
                        if (i == 3)
                            OIIO_CHECK_ASSERT(
                                ss.Parameter("reuse", TypeDesc::INT, &reuse));
                        OIIO_CHECK_ASSERT(
                            ss.Shader("surface", names[i], names[i]));
                    }
                    OIIO_CHECK_ASSERT(
                        ss.ConnectShaders("root", "value", "left", "value"));
                    OIIO_CHECK_ASSERT(
                        ss.ConnectShaders("root", "value", "right", "value"));
                    OIIO_CHECK_ASSERT(
                        ss.ConnectShaders("left", "result", "join", "a"));
                    OIIO_CHECK_ASSERT(
                        ss.ConnectShaders("right", "result", "join", "b"));
                    OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
                    const SymLocationDesc output("join.Cout", TypeColor, false,
                                                 SymArena::Outputs, 0,
                                                 3 * sizeof(float));
                    ss.add_symlocs(group.get(), { &output, 1 });
                    if (reject) {
                        check_rejected_group(ss, *group, errors,
                                             "renderer lacks HARTDiagnostics");
                        continue;
                    }
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "{}\n", errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    check_module(ss, *group, arch, { "osl_sin_dfdf" }, optimize,
                                 false, false, false, 4);
                }
    return true;
}



bool
check_material_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_coord(output point value=0) { value=point(u,v,u*v); }",
        "shader hart_space(point incoming=0,output point value=0) { "
        "value=transform(\"object\",\"shader\",incoming); }",
        "shader hart_warp(point incoming=0,output point value=0) { "
        "float n=psnoise(incoming,point(2)); "
        "value=incoming+vector(0.025*n,-0.05*n,0); }",
        "shader hart_sample(point incoming=0,output color value=0,"
        "output float alpha=0) { "
        "float data=texture(\"hart_texture_alpha_4.exr\",incoming[0],incoming[1],"
        "\"interp\",\"linear\",\"wrap\",\"clamp\",\"firstchannel\",2,"
        "\"alpha\",alpha); "
        "value=texture(\"hart_texture_alpha_4.exr\",incoming[0],incoming[1],"
        "\"interp\",\"linear\",\"wrap\",\"clamp\",\"firstchannel\",1)"
        "+color(0.125*data); }",
        "shader hart_mask(point incoming=0,output float value=0) { "
        "float n=psnoise(incoming,point(2)); value=smoothstep(-0.1,0.1,n); }",
        "shader hart_combine(color albedo=0,float mask=0,float strength=1,"
        "float texture_alpha=1,output color Cout=0) { "
        "color c=mix(color(0.2),albedo,strength*mask*texture_alpha); "
        "Cout=c+Dx(c)+Dy(c); }",
    };
    const char* names[]
        = { "coord", "space", "warp", "sample", "mask", "result" };
    std::string bytecode[6];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
            return false;
    }
    for (int osl_optimize : { 0, 2 })
        for (int optimize : { 10, 3 }) {
            HartServices renderer(true, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", osl_optimize);
            ss.attribute("llvm_optimize", optimize);
            const bool local_groupdata = osl_optimize == 0 && optimize == 10;
            if (local_groupdata)
                OIIO_CHECK_ASSERT(
                    ss.attribute("max_hart_groupdata_alloc", 4096));
            for (size_t i = 0; i < std::size(names); ++i)
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader(names[i], bytecode[i]));
            auto group = ss.ShaderGroupBegin("hart_test_group");
            for (const auto* name : names)
                OIIO_CHECK_ASSERT(ss.Shader("surface", name, name));
            for (int i = 1; i < 5; ++i)
                OIIO_CHECK_ASSERT(ss.ConnectShaders(names[i == 4 ? 2 : i - 1],
                                                    "value", names[i],
                                                    "incoming"));
            OIIO_CHECK_ASSERT(
                ss.ConnectShaders("sample", "value", "result", "albedo"));
            OIIO_CHECK_ASSERT(ss.ConnectShaders("sample", "alpha", "result",
                                                "texture_alpha"));
            OIIO_CHECK_ASSERT(
                ss.ConnectShaders("mask", "value", "result", "mask"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("result.Cout", TypeColor, false,
                                         SymArena::Outputs, 0,
                                         3 * sizeof(float));
            ss.add_symlocs(group.get(), { &output, 1 });
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            int allocated = -1;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_EQUAL(allocated > 0, local_groupdata);
            check_module(ss, *group, arch,
                         { "osl_transform_triple", "osl_psnoise_dfdvv",
                           "osl_texture", "osl_texture_set_firstchannel" },
                         optimize, false, false, false, 6);
        }
    return true;
}



bool
check_division_ir(ShadingSystem& ss, ShaderGroup& group, int optimize,
                  bool safe)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    OIIO_CHECK_ASSERT(bytes && size);
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_division"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    std::vector<const llvm::Value*> divisions, guarded;
    int reciprocals = 0;
    for (const auto& function : **parsed) {
        if (function.getName().find("osl_layer_group_") != 0
            && (optimize == 10
                || function.getName().find("__direct_callable__") != 0))
            continue;
        for (const char* name : { "unsafe-fp-math", "approx-func-fp-math",
                                  "no-nans-fp-math", "no-infs-fp-math" }) {
            const auto attr = function.getFnAttribute(name);
            OIIO_CHECK_ASSERT(!attr.isStringAttribute()
                              || attr.getValueAsString() != "true");
        }
        for (const auto& block : function)
            for (const auto& inst : block) {
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                    if (const auto* callee = call->getCalledFunction())
                        OIIO_CHECK_ASSERT(callee->getName()
                                          != "osl_safe_div_fff");
                if (inst.getOpcode() == llvm::Instruction::FDiv) {
                    OIIO_CHECK_ASSERT(!inst.getFastMathFlags().any());
                    divisions.push_back(&inst);
                    const auto* numerator = llvm::dyn_cast<llvm::ConstantFP>(
                        inst.getOperand(0));
                    reciprocals += numerator && numerator->isExactlyValue(1.0);
                }
                if (optimize != 10)
                    continue;
                const auto* select = llvm::dyn_cast<llvm::SelectInst>(&inst);
                const auto* cmp    = select ? llvm::dyn_cast<llvm::ICmpInst>(
                                                  select->getCondition())
                                            : nullptr;
                const auto* finite = cmp ? llvm::dyn_cast<llvm::CallBase>(
                                               cmp->getOperand(0))
                                         : nullptr;
                const auto* callee = finite ? finite->getCalledFunction()
                                            : nullptr;
                if (!callee || callee->getName() != "osl_isfinite_if")
                    continue;
                OIIO_CHECK_EQUAL(finite->arg_size(), 1);
                OIIO_CHECK_EQUAL(cmp->getPredicate(), llvm::CmpInst::ICMP_NE);
                const auto* test_zero = llvm::dyn_cast<llvm::ConstantInt>(
                    cmp->getOperand(1));
                OIIO_CHECK_ASSERT(test_zero && test_zero->isZero());
                const auto* result_zero = llvm::dyn_cast<llvm::ConstantFP>(
                    select->getFalseValue());
                OIIO_CHECK_ASSERT(result_zero
                                  && result_zero->getValueAPF().isZero()
                                  && !result_zero->getValueAPF().isNegative());
                if (finite->arg_size() != 1)
                    continue;
                OIIO_CHECK_EQUAL(select->getTrueValue(),
                                 finite->getArgOperand(0));
                guarded.push_back(select->getTrueValue());
            }
    }
    OIIO_CHECK_ASSERT(!divisions.empty());
    if (optimize == 10) {
        for (const auto* division : divisions)
            OIIO_CHECK_EQUAL(std::find(guarded.begin(), guarded.end(), division)
                                 != guarded.end(),
                             safe);
        if (safe)
            OIIO_CHECK_ASSERT(reciprocals > 0);
        else
            OIIO_CHECK_ASSERT(guarded.empty());
    }
    return !divisions.empty();
}



bool
check_division_modules(string_view arch, string_view stdosl)
{
    auto check = [&](string_view source, bool safe) {
        OSLCompiler compiler;
        std::string bytecode;
        const std::vector<std::string> options {
            "-I" + OIIO::Filesystem::parent_path(stdosl)
        };
        if (!compiler.compile_buffer(source, bytecode, options, stdosl))
            return false;
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 }) {
                HartServices renderer;
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
                OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
                auto group = make_group(ss, bytecode);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "Division (OSL {}, LLVM {}): {}\n",
                          osl_optimize, optimize, errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                check_module(ss, *group, arch, { }, optimize);
                OIIO_CHECK_ASSERT(
                    check_division_ir(ss, *group, optimize, safe));
            }
        return true;
    };
    if (!check("shader scalar_division(float a=2, float b=1.5, "
               "output color Cout=0) { float q=(a+u)/(b+v); "
               "Cout=color(q,Dx(q),Dy(q)); }",
               true)
        || !check("shader color_division(output color Cout=0) { "
                  "color a=color(2+u,4+u,8+u), b=color(.5+v,1.5+v,2.5+v); "
                  "color q=a/b; Cout=q+Dx(q)+Dy(q); }",
                  true)
        || !check("shader literal_divisor(float a=2, output color Cout=0) { "
                  "float q=(a+u)/1.5; Cout=color(q,Dx(q),Dy(q)); }",
                  false))
        return false;
    const struct {
        string_view type, initial, expected, component;
    } aggregates[] = {
        { "color2", "color2(.5,1.5)", "color2(4,2.0/1.5)", "a" },
        { "color4", "color4(color(.5,1.5,2.5),1.5)",
          "color4(color(4,2.0/1.5,2.0/2.5),2.0/1.5)", "a" },
        { "vector2", "vector2(.5,1.5)", "vector2(4,2.0/1.5)", "y" },
        { "vector4", "vector4(.5,1.5,2.5,3.5)",
          "vector4(4,2.0/1.5,2.0/2.5,2.0/3.5)", "y" },
    };
    for (const auto& test : aggregates) {
        // Keep a varying quotient live even when OSL2 folds the exact
        // default-parameter comparison from the original standard fixtures.
        const auto source = fmtformat(
            "#include \"{0}.h\"\n"
            "shader aggregate_division({0} param1={1}, output color Cout=0) {{ "
            "{0} q=2/(param1+v); "
            "int exact=(2/param1=={2}) && (2.0/param1=={2}); "
            "Cout=color(exact,q.{3},Dx(q.{3})+Dy(q.{3})); }}",
            test.type, test.initial, test.expected, test.component);
        if (!check(source, true))
            return false;
    }
    return true;
}



bool
check_math_modules(string_view arch, string_view stdosl)
{
    if (!check_division_modules(arch, stdosl))
        return false;
    for (string_view type : { "float", "color", "vector" }) {
        const auto body = fmtformat(
            "{0} x={0}(1.7*u-0.7), y={0}(v+0.4); "
            "value=abs(x)+min(x,y)+max(x,y)+clamp(x,{0}(-0.5),{0}(0.5))"
            "+mix(x,y,{0}(u))+step({0}(0.2),x)"
            "+smoothstep(x-{0}(0.4),y+{0}(0.7),x)+floor(x)+ceil(x)"
            "+fmod(x,y)+cos(x)+sqrt(abs(x)+{0}(0.25))"
            "+pow(abs(x)+{0}(0.5),y); ",
            type);
        const std::string sources[] = {
            fmtformat("shader hart_math(output color Cout=0) {{ "
                      "{} value=0; {} Cout=color(value); }}",
                      type, body),
            fmtformat("shader hart_math_producer(output {} value=0) {{ {} }}",
                      type, body),
            fmtformat(
                "shader hart_math_consumer({} value=0, output color Cout=0) {{ "
                "Cout=color(value+Dx(value)+Dy(value)); }}",
                type),
        };
        std::string bytecode[3];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
                return false;
        }
        for (int optimize : { 10, 3 }) {
            for (int osl_optimize : { 0, 2 }) {
                for (bool connected : { false, true }) {
                    HartServices renderer;
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("llvm_optimize", optimize);
                    ss.attribute("optimize", osl_optimize);
                    auto group = connected
                                     ? make_connected_group(ss, bytecode[1],
                                                            bytecode[2])
                                     : make_group(ss, bytecode[0]);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "Math {}: {}\n", type, errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    const auto signature = type == "float"
                                               ? (connected ? "dfdf" : "ff")
                                               : (connected ? "dvdv" : "vv");
                    check_module(ss, *group, arch,
                                 { fmtformat("osl_abs_{}", signature),
                                   fmtformat("osl_cos_{}", signature),
                                   fmtformat("osl_sqrt_{}", signature),
                                   "osl_fmod_fff",
                                   connected ? "osl_smoothstep_dfdfdfdf"
                                             : "osl_smoothstep_ffff" },
                                 optimize, connected);
                }
            }
        }
    }
    const struct {
        string_view source;
        string_view error;
    } rejected[] = {
        { "float hidden(float x) { printf(\"no\"); return x; } "
          "shader bad(output color Cout=0) { Cout=color(hidden(u)); }",
          "renderer lacks HARTDiagnostics" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, test.error);
    }
    return true;
}



bool
check_isconstant_modules(string_view arch, string_view stdosl)
{
    const string_view source = R"osl(
shader hart_constants(float A=1, string label="literal", output color Cout=0) {
    float twice=2*A;
    string selected_label=u>v ? "left" : "right";
    Cout=color(isconstant(3)+2*isconstant(2.0)+4*isconstant("literal"),
               isconstant(u)+2*isconstant(P)+4*isconstant(selected_label),
               isconstant(A)+2*isconstant(twice)+4*isconstant(label));
}
)osl";
    OSLCompiler compiler;
    std::string bytecode;
    if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
        return false;
    for (int osl_optimize : { 0, 2 }) {
        for (int optimize : { 10, 3 }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("llvm_optimize", optimize);
            ss.attribute("optimize", osl_optimize);
            auto group = make_group(ss, bytecode);
            ss.optimize_group(group.get(), nullptr);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch, { }, optimize);
        }
    }
    return true;
}



bool
check_numeric_math_modules(string_view arch, string_view stdosl)
{
    const struct {
        int osl_optimize;
        int llvm_optimize;
        bool connected;
        bool local;
    } variants[] = {
        { 0, 10, false, false },
        { 2, 10, true, true },
        { 2, 3, true, false },
    };
    auto check = [&](string_view label, string_view producer,
                     string_view consumer,
                     std::initializer_list<string_view> shadeops,
                     int osl_optimize, int optimize, bool local = false) {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", local ? 4096 : 0));
        const bool connected = !consumer.empty();
        auto group = connected ? make_connected_group(ss, producer, consumer)
                               : make_group(ss, producer);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr,
                  "Numeric math {} (OSL {}, LLVM {}, connected {}): {}\n",
                  label, osl_optimize, optimize, connected, errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int allocated = -1, size = 0;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, local ? size : 0);
        check_module(ss, *group, arch, shadeops, optimize, connected, false,
                     false, connected ? 2 : 0);
    };

    // Three representative modes, not a Cartesian product for every opcode.
    for (string_view type : { "float", "color", "vector" }) {
        const auto coordinate
            = type == "float"
                  ? std::string("0.7*u-0.35")
                  : fmtformat("{}(0.7*u-0.35,0.5*v-0.2,0.4*u*v-0.1)", type);
        const auto body = fmtformat(
            "{0} x={1}, y={0}(v+0.8), p=fabs(x)+{0}(0.75); "
            "value=tan(x)+asin(x)+acos(x+{0}(0.1))+atan(y)+atan2(x,y)"
            "+atan2(x,{0}(1.25))+atan2({0}(0.3),y)"
            "+sinh(x)+cosh(y)+tanh(x+y)"
            "+log(p)+log2(p+{0}(0.1))+log10(p+{0}(0.2))"
            "+exp(x)+exp2(y)+expm1(x-y)+cbrt(x-{0}(0.6))"
            "+inversesqrt(p)+fabs(x-{0}(0.2))"
            "+0.01*degrees(x)+2*radians(y); "
            "{0} z=logb(p)+round(4*x)+trunc(3*y)+sign(x); "
            "value+=z+Dx(z)+Dy(z); ",
            type, coordinate);
        const std::string sources[] = {
            fmtformat("shader numeric_components(output color Cout=0) {{ "
                      "{} value=0; {} Cout=color(value); }}",
                      type, body),
            fmtformat("shader numeric_producer(output {} value=0) {{ {} }}",
                      type, body),
            fmtformat(
                "shader numeric_consumer({0} value=0, output color Cout=0) {{ "
                "Cout=color(value+Dx(value)+Dy(value))"
                "+color(filterwidth({1})); }}",
                type, type == "float" ? "value" : "vector(value)"),
        };
        std::string bytecode[3];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
                return false;
        }
        for (const auto& variant : variants) {
            const bool dual   = variant.connected;
            const auto unary  = type == "float" ? (dual ? "dfdf" : "ff")
                                                : (dual ? "dvdv" : "vv");
            const auto binary = type == "float" ? (dual ? "dfdfdf" : "fff")
                                                : (dual ? "dvdvdv" : "vvv");
            const auto plain  = type == "float" ? "ff" : "vv";
            check(type, bytecode[dual ? 1 : 0], dual ? bytecode[2] : "",
                  { fmtformat("osl_tan_{}", unary),
                    fmtformat("osl_asin_{}", unary),
                    fmtformat("osl_acos_{}", unary),
                    fmtformat("osl_atan_{}", unary),
                    fmtformat("osl_atan2_{}", binary),
                    dual ? (type == "float" ? "osl_atan2_dfdff"
                                            : "osl_atan2_dvdvv")
                         : "",
                    dual ? (type == "float" ? "osl_atan2_dffdf"
                                            : "osl_atan2_dvvdv")
                         : "",
                    fmtformat("osl_sinh_{}", unary),
                    fmtformat("osl_cosh_{}", unary),
                    fmtformat("osl_tanh_{}", unary),
                    fmtformat("osl_log_{}", unary),
                    fmtformat("osl_log2_{}", unary),
                    fmtformat("osl_log10_{}", unary),
                    fmtformat("osl_exp_{}", unary),
                    fmtformat("osl_exp2_{}", unary),
                    fmtformat("osl_expm1_{}", unary),
                    fmtformat("osl_cbrt_{}", unary),
                    fmtformat("osl_inversesqrt_{}", unary),
                    fmtformat("osl_fabs_{}", unary),
                    fmtformat("osl_logb_{}", plain),
                    fmtformat("osl_round_{}", plain),
                    fmtformat("osl_trunc_{}", plain),
                    fmtformat("osl_sign_{}", plain),
                    dual ? (type == "float" ? "osl_filterwidth_fdf"
                                            : "osl_filterwidth_vdv")
                         : "" },
                  variant.osl_optimize, variant.llvm_optimize, variant.local);
        }

        // Independently demand sine/cosine derivatives, then alias the input
        // with either output, both outputs together, and all three arguments.
        const auto sincos_source
            = fmtformat("shader numeric_sincos(output color Cout=0) {{ "
                        "{0} x={1}, sv=0, cv=0; sincos(x,sv,cv); "
                        "{0} xs=x+{0}(0.1), ss=0, cs=0; sincos(xs,ss,cs); "
                        "{0} xc=x+{0}(0.2), sc=0, cc=0; sincos(xc,sc,cc); "
                        "{0} a=x+{0}(0.3), b=0; sincos(a,a,b); "
                        "{0} c=x+{0}(0.4), d=0; sincos(c,d,c); "
                        "{0} shared=0; sincos(x+{0}(0.5),shared,shared); "
                        "{0} all=x+{0}(0.6); sincos(all,all,all); "
                        "{0} zs=0, zc=0; sincos({0}(time),zs,zc); "
                        "Cout=color(sv+2*cv+ss+3*cs+Dx(ss)+2*sc+cc+Dy(cc)"
                        "+a+2*b+Dx(a)+Dy(b)+3*c+d+Dx(d)+Dy(c)"
                        "+shared+Dx(shared)+Dy(shared)+all+Dx(all)+Dy(all)"
                        "+zs+zc+Dx(zs)+Dy(zc)); }}",
                        type, coordinate);
        OSLCompiler sincos_compiler;
        std::string sincos_bytecode;
        if (!sincos_compiler.compile_buffer(sincos_source, sincos_bytecode, { },
                                            stdosl))
            return false;
        const auto code = type == "float" ? "f" : "v";
        for (int optimize : { 10, 3 })
            check(fmtformat("sincos {}", type), sincos_bytecode, "",
                  { fmtformat("osl_sincos_{0}{0}{0}", code),
                    fmtformat("osl_sincos_d{0}d{0}{0}", code),
                    fmtformat("osl_sincos_d{0}{0}d{0}", code),
                    fmtformat("osl_sincos_d{0}d{0}d{0}", code) },
                  optimize == 10 ? 0 : 2, optimize);
    }

    // Classification, erf/erfc, and hypot have scalar-only public overloads.
    const char* scalar_source
        = "shader numeric_scalar(output color Cout=0) { "
          "float x=u-v, q=erf(x)+erfc(0.5*x)"
          "+hypot(x,v+0.3)+hypot(x,u+0.2,v-0.4); "
          "float plain=erf(u+0.2)+erfc(v-0.1)"
          "+hypot(u+0.1,v+0.2)+hypot(u+0.3,v+0.4,u*v+0.5); "
          "int flags=isnan(x)+2*isinf(x)+4*isfinite(x)+fabs(int(4*x)); "
          "float z=flags+round(3*x)+trunc(4*x)+sign(x)"
          "+logb(fabs(x)+0.5); "
          "Cout=color(plain+q+z,Dx(q)+Dx(z),Dy(q)+Dy(z)); }";
    OSLCompiler scalar_compiler;
    std::string scalar_bytecode;
    if (!scalar_compiler.compile_buffer(scalar_source, scalar_bytecode, { },
                                        stdosl))
        return false;
    for (int optimize : { 10, 3 })
        check("scalar classification", scalar_bytecode, "",
              { "osl_erf_ff", "osl_erfc_ff", "osl_erf_dfdf", "osl_erfc_dfdf",
                "osl_sqrt_ff", "osl_sqrt_dfdf", "osl_isnan_if", "osl_isinf_if",
                "osl_isfinite_if", "osl_fabs_ii", "osl_logb_ff", "osl_round_ff",
                "osl_trunc_ff", "osl_sign_ff" },
              0, optimize);

    // The stdosl geometric wrappers lower to existing arithmetic, branches,
    // dot/sqrt, sincos, and matrix transforms, not new wrapper opcodes.
    const string_view geometry_body
        = "point q=point(P[0]+u,P[1]+v,P[2]+u*v); "
          "point r=point(v+0.25,u-0.5,1+u); "
          "vector a=vector(q), b=vector(r); "
          "value=cross(a,b)+2*cross(a,N)+3*cross(Ng,b); "
          "float dist=distance(q,r)+2*distance(q,point(N))"
          "+3*distance(point(Ng),r); "
          "float ar=area(q), zero=area(point(time)); "
          "normal n=calculatenormal(q), zn=calculatenormal(point(N)); "
          "value+=vector(dist+ar+zero+Dx(ar)+Dy(ar)+Dx(zero)+Dy(zero))"
          "+vector(n+zn+Dx(n)+Dy(n)+Dx(zn)+Dy(zn)); "
          "vector nn=normalize(vector(0.2+u,0.3+v,1)); "
          "vector ii=normalize(I+vector(0.1+u,0.2-v,-1)); "
          "value+=reflect(ii,nn)+refract(ii,nn,0.6+0.2*u)"
          "+faceforward(nn,ii)+faceforward(nn,ii,Ng)"
          "+vector(rotate(q,u+0.3,point(0.1,0.2,0.3),point(0.8,0.9,1.1)))"
          "+vector(rotate(r,v+0.1,vector(1,0.2+u,0.5))); ";
    const std::string geometry_sources[] = {
        fmtformat("shader numeric_geometry(output color Cout=0) {{ "
                  "vector value=0; {} Cout=color(value); }}",
                  geometry_body),
        fmtformat("shader numeric_geometry_producer(output vector value=0) {{ "
                  "{} }}",
                  geometry_body),
        "shader numeric_geometry_consumer(vector value=0, output color Cout=0) "
        "{ Cout=color(value+Dx(value)+Dy(value)+filterwidth(value)); }",
    };
    std::string geometry_bytecode[3];
    for (size_t i = 0; i < std::size(geometry_sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(geometry_sources[i], geometry_bytecode[i],
                                     { }, stdosl))
            return false;
    }
    for (const auto& variant : variants) {
        const bool dual = variant.connected;
        check("geometry", geometry_bytecode[dual ? 1 : 0],
              dual ? geometry_bytecode[2] : "",
              { dual ? "osl_cross_dvdvdv" : "osl_cross_vvv",
                dual ? "osl_cross_dvdvv" : "", dual ? "osl_cross_dvvdv" : "",
                dual ? "osl_distance_dfdvdv" : "osl_distance_fvv",
                dual ? "osl_distance_dfdvv" : "",
                dual ? "osl_distance_dfvdv" : "", "osl_area",
                "osl_calculatenormal",
                dual ? "osl_normalize_dvdv" : "osl_normalize_vv",
                dual ? "osl_filterwidth_vdv" : "" },
              variant.osl_optimize, variant.llvm_optimize, variant.local);
    }
    return true;
}



bool
check_noise_modules(string_view arch, string_view stdosl)
{
    const string_view coordinates = "float x=1.7*u-0.23; float y=2.3*v+0.31; "
                                    "point p=point(x,y,u*v+0.7); ";
    const struct {
        string_view call;
        string_view shadeop;
        bool periodic;
        bool derivatives;
    } families[] = {
        { "noise(", "noise", false, true },
        { "snoise(", "snoise", false, true },
        { "pnoise(", "pnoise", true, true },
        { "psnoise(", "psnoise", true, true },
        { "cellnoise(", "cellnoise", false, false },
        { "hashnoise(", "hashnoise", false, false },
        { "noise(\"simplex\",", "simplexnoise", false, true },
        { "noise(\"usimplex\",", "usimplexnoise", false, true },
        { "pnoise(\"cell\",", "pcellnoise", true, false },
        { "pnoise(\"hash\",", "phashnoise", true, false },
    };
    for (const auto& family : families) {
        for (string_view type : { "float", "color", "vector" }) {
            const auto assignment = fmtformat(
                "{0} a={1}x{2}); {0} b={1}x,y{3}); {0} c={1}p{4}); "
                "{0} d={1}p,u+v{5}); {0} e={1}x,0.31{3}); {0} f={1}p,0.19{5}); "
                "value=a+b+c+d+e+f; ",
                type, family.call, family.periodic ? ",2.0" : "",
                family.periodic ? ",2.0,3.0" : "",
                family.periodic ? ",point(2,3,4)" : "",
                family.periodic ? ",point(2,3,4),5.0" : "");
            const std::string sources[] = {
                fmtformat("shader hart_noise_test(output color Cout=0) {{ "
                          "{} {} value=0; {} Cout=color(value); }}",
                          coordinates, type, assignment),
                fmtformat(
                    "shader hart_noise_producer(output {} value=0) {{ {} {} }}",
                    type, coordinates, assignment),
                fmtformat(
                    "shader hart_noise_consumer({} value=0, output color Cout=0) {{ "
                    "Cout=color(value+Dx(value)+Dy(value)+filterwidth({})); }}",
                    type, type == "float" ? "value" : "vector(value)"),
            };
            std::string bytecode[3];
            for (size_t i = 0; i < std::size(sources); ++i) {
                OSLCompiler compiler;
                if (!compiler.compile_buffer(sources[i], bytecode[i], { },
                                             stdosl))
                    return false;
            }
            for (int optimize : { 10, 3 }) {
                for (bool connected : { false, true }) {
                    HartServices renderer;
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                    ss.attribute("llvm_optimize", optimize);
                    auto group = connected
                                     ? make_connected_group(ss, bytecode[1],
                                                            bytecode[2])
                                     : make_group(ss, bytecode[0]);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "{} {} (LLVM {}): {}\n", family.call,
                              type, optimize, errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    const bool derivs = connected && family.derivatives;
                    const auto prefix = fmtformat("osl_{}_{}{}", family.shadeop,
                                                  derivs ? "d" : "",
                                                  type == "float" ? "f" : "v");
                    const std::string names[] = {
                        prefix + (derivs ? "df" : "f")
                            + (family.periodic ? "f" : ""),
                        prefix + (derivs ? "dfdf" : "ff")
                            + (family.periodic ? "ff" : ""),
                        prefix + (derivs ? "dv" : "v")
                            + (family.periodic ? "v" : ""),
                        prefix + (derivs ? "dvdf" : "vf")
                            + (family.periodic ? "vf" : ""),
                    };
                    check_module(ss, *group, arch,
                                 { names[0], names[1], names[2], names[3] },
                                 optimize, connected);
                }
            }
        }
    }
    const struct {
        string_view expression;
        string_view error;
    } rejected[] = {
        { "noise(\"gabor\",point(0.25))", "HARTNoiseErrors" },
        { "noise(\"\",P)", "unsupported noise type ''" },
        { "noise(\"unknown\",P)", "unsupported noise type 'unknown'" },
        { "noise(\"perlin\",P,\"bandwidth\",1.0)",
          "noise options require gabor" },
        { "pnoise(\"simplex\",P,point(2))", "unsupported noise type 'simplex'" },
        { "pnoise(\"usimplex\",P,point(2))",
          "unsupported noise type 'usimplex'" },
        { "pnoise(\"gabor\",P,point(2))", "HARTNoiseErrors" },
        { "noise(Ps)", "unsupported shader global 'Ps'" },
    };
    for (const auto& test : rejected) {
        const auto source = fmtformat(
            "shader hart_noise_unsupported(output color Cout=0) {{ Cout=color({}); }}",
            test.expression);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, test.error);
    }
    for (string_view operation : { "noise", "pnoise" }) {
        const auto source = fmtformat(
            "shader hart_dynamic_noise(string kind=\"perlin\", output color Cout=0) {{ "
            "Cout=color({}(kind,P{})); }}",
            operation, operation == "pnoise" ? ",point(2)" : "");
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, "HARTNoiseErrors");
        for (string_view name :
             { "perlin", "uperlin", "cell", "hash", "noise", "snoise" }) {
            OSLCompiler named_compiler;
            const auto named_source = fmtformat(
                "shader hart_named_noise(output color Cout=0) {{ "
                "float n={}(\"{}\",P{}); Cout=color(n,Dx(n),Dy(n)); }}",
                operation, name, operation == "pnoise" ? ",point(2)" : "");
            if (!named_compiler.compile_buffer(named_source, bytecode, { },
                                               stdosl))
                return false;
            for (int optimize : { 10, 3 }) {
                HartServices renderer;
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                ss.attribute("hart_arch", arch);
                ss.attribute("llvm_optimize", optimize);
                const bool alias = name == "noise" || name == "snoise";
                if (alias)
                    ss.attribute("optimize", optimize == 10 ? 0 : 2);
                auto group = make_group(ss, bytecode);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "{}\n", errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                const bool periodic = operation == "pnoise";
                const bool derivs   = name == "perlin" || name == "uperlin"
                                      || alias;
                const auto family   = name == "perlin" || name == "snoise"
                                          ? "snoise"
                                      : name == "uperlin" || name == "noise"
                                          ? "noise"
                                      : name == "cell" ? "cellnoise"
                                                       : "hashnoise";
                const auto shadeop
                    = fmtformat("osl_{}{}_{}{}", periodic ? "p" : "", family,
                                derivs ? "dfdv" : "fv", periodic ? "v" : "");
                check_module(ss, *group, arch, { shadeop }, optimize);
            }
        }
    }
    return true;
}



struct OutputPlacement {
    SymLocationDesc location;
    uint64_t bytes;
};



bool
check_output_placement_ir(ShadingSystem& ss, ShaderGroup& group,
                          cspan<OutputPlacement> expected,
                          bool scheduled_producer = false)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_output_placement"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    auto& module        = **parsed;
    auto split_constant = [](const llvm::Value* value, unsigned opcode,
                             const llvm::ConstantInt*& constant,
                             const llvm::Value*& other) {
        const auto* operation = llvm::dyn_cast<llvm::BinaryOperator>(value);
        if (!operation || operation->getOpcode() != opcode)
            return false;
        for (unsigned i = 0; i < 2; ++i)
            if (const auto* number = llvm::dyn_cast<llvm::ConstantInt>(
                    operation->getOperand(i))) {
                constant = number;
                other    = operation->getOperand(1 - i);
                return true;
            }
        return false;
    };
    std::vector<int> copies(expected.size(), 0);
    std::vector<const llvm::MemCpyInst*> consumer_copies;
    for (const auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0)
            continue;
        OIIO_CHECK_EQUAL(function.arg_size(), 6);
        if (function.arg_size() != 6)
            continue;
        for (const auto& block : function)
            for (const auto& inst : block) {
                const auto* copy = llvm::dyn_cast<llvm::MemCpyInst>(&inst);
                if (!copy)
                    continue;
                const auto* address = llvm::dyn_cast<llvm::IntToPtrInst>(
                    copy->getRawDest()->stripPointerCasts());
                const auto* sum = address
                                      ? llvm::dyn_cast<llvm::BinaryOperator>(
                                            address->getOperand(0))
                                      : nullptr;
                if (!sum || sum->getOpcode() != llvm::Instruction::Add)
                    continue;
                const llvm::Value* relative = nullptr;
                for (unsigned i = 0; i < 2; ++i)
                    if (const auto* base = llvm::dyn_cast<llvm::PtrToIntInst>(
                            sum->getOperand(i)))
                        if (base->getOperand(0)->stripPointerCasts()
                            == function.getArg(3)) {
                            OIIO_CHECK_ASSERT(base->getType()->isIntegerTy(64));
                            relative = sum->getOperand(1 - i);
                        }
                if (!relative)
                    continue;  // Not a renderer-output copy.
                OIIO_CHECK_ASSERT(address->getType()->getPointerAddressSpace()
                                  == 0);
                const llvm::ConstantInt* offset = nullptr;
                const llvm::Value* product      = relative;
                split_constant(relative, llvm::Instruction::Add, offset,
                               product);
                const llvm::ConstantInt* stride = nullptr;
                const llvm::Value* index        = nullptr;
                OIIO_CHECK_ASSERT(split_constant(product,
                                                 llvm::Instruction::Mul, stride,
                                                 index));
                if (!stride || !index)
                    continue;
                const auto* extended = llvm::dyn_cast<llvm::SExtInst>(index);
                OIIO_CHECK_ASSERT(
                    stride->getType()->isIntegerTy(64)
                    && (!offset || offset->getType()->isIntegerTy(64))
                    && extended && extended->getType()->isIntegerTy(64)
                    && extended->getOperand(0) == function.getArg(4));
                const auto found = std::find_if(
                    expected.begin(), expected.end(), [&](const auto& test) {
                        const string_view name(test.location.name);
                        const auto dot = name.find('.');
                        return dot != string_view::npos
                               && function.getName()
                                      == fmtformat(
                                          "osl_layer_group_hart_test_group_name_{}",
                                          name.substr(0, dot))
                               && test.location.offset != -1
                               && test.location.offset
                                      == (offset ? offset->getSExtValue() : 0)
                               && test.location.stride
                                      == stride->getSExtValue();
                    });
                if (found == expected.end()) {
                    print(stderr,
                          "Unexpected output copy in {} at {} + {}*index\n",
                          function.getName().str(),
                          offset ? offset->getSExtValue() : 0,
                          stride->getSExtValue());
                    OIIO_CHECK_ASSERT(false);
                    continue;
                }
                ++copies[found - expected.begin()];
                const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                    copy->getLength());
                OIIO_CHECK_ASSERT(length);
                if (length)
                    OIIO_CHECK_EQUAL(length->getZExtValue(), found->bytes);
                OIIO_CHECK_ASSERT(!copy->isVolatile());
                if (function.getName()
                    == "osl_layer_group_hart_test_group_name_consumer")
                    consumer_copies.push_back(copy);
            }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (copies[i] != (expected[i].location.offset == -1 ? 0 : 1))
            print(stderr, "Wrong copy count for output '{}'\n",
                  expected[i].location.name);
        OIIO_CHECK_EQUAL(copies[i], expected[i].location.offset == -1 ? 0 : 1);
    }
    if (scheduled_producer) {
        auto* producer = module.getFunction(
            "osl_layer_group_hart_test_group_name_producer");
        auto* consumer = module.getFunction(
            "osl_layer_group_hart_test_group_name_consumer");
        OIIO_CHECK_ASSERT(producer && consumer && !consumer_copies.empty());
        if (!producer || !consumer)
            return false;
        llvm::DominatorTree dominators(*consumer);
        bool scheduled = false;
        for (const auto* user : producer->users()) {
            const auto* call = llvm::dyn_cast<llvm::CallBase>(user);
            if (!call || call->getCalledFunction() != producer
                || call->getFunction() != consumer)
                continue;
            OIIO_CHECK_EQUAL(call->arg_size(), 6);
            if (call->arg_size() != 6)
                continue;
            for (unsigned i = 0; i < 6; ++i)
                OIIO_CHECK_EQUAL(call->getArgOperand(i), consumer->getArg(i));
            bool dominates = true;
            for (const auto* copy : consumer_copies)
                dominates &= dominators.dominates(call, copy);
            for (const auto& block : *consumer)
                if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(
                        block.getTerminator()))
                    dominates &= dominators.dominates(call, ret);
            scheduled |= dominates;
        }
        OIIO_CHECK_ASSERT(scheduled);
    }
    return true;
}



bool
check_output_placement_modules(string_view arch, string_view stdosl,
                               string_view basic, string_view producer,
                               string_view consumer)
{
    const string_view source = R"osl(
shader hart_output_values(output int integer=0, output float scalar=0,
                         output color Cout=0, output vector direction=0,
                         output matrix matrix_out=1, output int integers[3]={},
                         output float floats[2]={}, output vector vectors[2]={},
                         output float disabled=0) {
    integer=int(10*u)+3; scalar=u-2*v;
    Cout=color(u,v,u+v); direction=vector(u+1,v+2,u-v);
    matrix_out=matrix(1+u,2,3,4,5,6+v,7,8,9,10,11,12,13,14,15,16);
    integers[0]=integer; integers[1]=integer+1; integers[2]=integer+2;
    floats[0]=scalar; floats[1]=u+v;
    vectors[0]=direction; vectors[1]=vector(Cout);
    disabled=u+42;
}
)osl";
    OSLCompiler compiler;
    std::string bytecode;
    if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
        return false;
    TypeDesc vectors                 = TypeVector;
    vectors.arraylen                 = 2;
    const OutputPlacement original[] = {
        { { "layer0.integer", TypeInt, false, SymArena::Outputs, 8, 256 }, 4 },
        { { "layer0.scalar", TypeFloat, false, SymArena::Outputs, 16, 256 }, 4 },
        { { "layer0.Cout", TypeColor, false, SymArena::Outputs, 32, 256 }, 12 },
        { { "layer0.direction", TypeVector, false, SymArena::Outputs, 48, 256 },
          12 },
        { { "layer0.matrix_out", TypeMatrix, false, SymArena::Outputs, 64, 256 },
          64 },
        { { "layer0.integers", TypeDesc(TypeDesc::INT, 3), false,
            SymArena::Outputs, 136, 256 },
          12 },
        { { "layer0.floats", TypeDesc(TypeDesc::FLOAT, 2), false,
            SymArena::Outputs, 152, 256 },
          8 },
        { { "layer0.vectors", vectors, false, SymArena::Outputs, 176, 256 },
          24 },
        { { "layer0.disabled", TypeFloat, false, SymArena::Outputs, -1, 4 }, 4 },
    };
    const struct {
        int osl, llvm;
        bool local;
    } variants[] = { { 0, 10, false }, { 2, 10, true }, { 2, 3, true } };
    for (const auto& variant : variants) {
        HartServices renderer(false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", variant.osl);
        ss.attribute("llvm_optimize", variant.llvm);
        ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
        auto group = make_group(ss, bytecode);
        ss.clear_symlocs(group.get());
        std::vector<OutputPlacement> expected(std::begin(original),
                                              std::end(original));
        std::vector<SymLocationDesc> locations;
        std::vector<ustring> names;
        for (size_t i = 0; i < expected.size(); ++i) {
            auto& location = expected[i].location;
            if (variant.local && location.offset != -1)
                location = SymLocationDesc(location.name, location.type, false,
                                           SymArena::Outputs,
                                           int64_t(65536 * (i + 1)),
                                           i ? int64_t(expected[i].bytes
                                                       + 4 * (i + 1))
                                             : SymLocationDesc::AutoStride);
            locations.push_back(location);
            names.push_back(location.name);
        }
        OIIO_CHECK_ASSERT(
            ss.attribute(group.get(), "renderer_outputs",
                         TypeDesc(TypeDesc::STRING, int(names.size())),
                         names.data()));
        auto provisional = locations;
        for (auto& location : provisional)
            if (location.offset != -1) {
                location.offset += 1024;
                location.stride += 64;
            }
        ss.add_symlocs(group.get(), provisional);
        for (int query = 0; query < 2; ++query) {
            int optimized = -1;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "is_optimized", optimized));
            OIIO_CHECK_EQUAL(optimized, 0);
        }
        ss.optimize_group(group.get(), nullptr, false);
        int optimized = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "is_optimized", optimized));
        OIIO_CHECK_EQUAL(optimized, 1);
        const void* bytes = nullptr;
        uint64_t size     = 0;
        OIIO_CHECK_ASSERT(!ss.getattribute(group.get(), "hart_bitcode",
                                           TypeDesc::PTR, &bytes));
        OIIO_CHECK_ASSERT(!ss.getattribute(group.get(), "hart_bitcode_size",
                                           TypeUInt64, &size));
        OIIO_CHECK_ASSERT(!bytes && !size);
        // Optimization preserves the named outputs; placement remains mutable
        // until code generation bakes the final mappings into the artifact.
        ss.clear_symlocs(group.get());
        for (const auto& location : locations)
            OIIO_CHECK_ASSERT(!ss.find_symloc(group.get(), location.name));
        ss.add_symlocs(group.get(), locations);
        OIIO_CHECK_ASSERT(
            ss.attribute(group.get(), "renderer_outputs",
                         TypeDesc(TypeDesc::STRING, int(names.size())),
                         names.data()));
        for (const auto& location : locations) {
            const auto* symbol = ss.find_symbol(*group, location.name);
            OIIO_CHECK_ASSERT(symbol);
            if (symbol)
                OIIO_CHECK_ASSERT(ss.symbol_typedesc(symbol) == location.type);
        }
        OIIO_CHECK_ASSERT(!ss.find_symloc(group.get(),
                                          ustring("layer0.disabled"),
                                          SymArena::Outputs));
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Typed outputs (OSL {}, LLVM {}, local={}):\n{}",
                  variant.osl, variant.llvm, variant.local, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int group_size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", group_size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
        OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
        check_module(ss, *group, arch, { }, variant.llvm, false, false, false,
                     0, false, true);
        if (variant.llvm == 10)
            OIIO_CHECK_ASSERT(check_output_placement_ir(ss, *group, expected));
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                          TypeDesc::PTR, &bytes));
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                          TypeUInt64, &size));
        if (!bytes || !size)
            return false;
        const void* original_bytes = bytes;
        const std::string artifact(static_cast<const char*>(bytes), size);
        auto unchanged = [&]() {
            const void* current = nullptr;
            uint64_t length     = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR, &current));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                              TypeUInt64, &length));
            OIIO_CHECK_EQUAL(current, original_bytes);
            OIIO_CHECK_EQUAL(length, artifact.size());
            if (current && length == artifact.size())
                OIIO_CHECK_ASSERT(
                    string_view(static_cast<const char*>(current), length)
                    == string_view(artifact));
            for (const auto& location : locations) {
                const auto* found = ss.find_symloc(group.get(), location.name);
                OIIO_CHECK_ASSERT(found);
                if (!found)
                    continue;
                OIIO_CHECK_EQUAL(found->name, location.name);
                OIIO_CHECK_ASSERT(found->type == location.type);
                OIIO_CHECK_EQUAL(found->offset, location.offset);
                OIIO_CHECK_EQUAL(found->stride, location.stride);
                OIIO_CHECK_ASSERT(found->arena == location.arena);
                OIIO_CHECK_EQUAL(found->derivs, location.derivs);
            }
            OIIO_CHECK_ASSERT(
                !ss.find_symloc(group.get(), ustring("late_output")));
            int count = -1;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "num_renderer_outputs", count));
            OIIO_CHECK_EQUAL(count, names.size());
            std::vector<ustring> current_names(names.size());
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "renderer_outputs",
                                TypeDesc(TypeDesc::STRING,
                                         int(current_names.size())),
                                current_names.data()));
            OIIO_CHECK_ASSERT(current_names == names);
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "is_optimized", optimized));
            OIIO_CHECK_EQUAL(optimized, 1);
            int current_size = 0, current_allocated = -1;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "llvm_groupdata_size",
                                              current_size));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              current_allocated));
            OIIO_CHECK_EQUAL(current_size, group_size);
            OIIO_CHECK_EQUAL(current_allocated, allocated);
        };
        const SymLocationDesc changed[] = {
            { "layer0.scalar", TypeFloat, false, SymArena::Outputs, 512, 1024 },
            { "late_output", TypeFloat, false, SymArena::Outputs, 768, 1024 },
        };
        int before = errors.errors;
        ss.add_symlocs(group.get(), changed);
        OIIO_CHECK_EQUAL(errors.errors, before + 1);
        OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
            errors.last_error,
            "Cannot change symbol locations of a compiled HART group"));
        unchanged();
        before = errors.errors;
        ss.clear_symlocs(group.get());
        OIIO_CHECK_EQUAL(errors.errors, before + 1);
        OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
            errors.last_error,
            "Cannot clear symbol locations of a compiled HART group"));
        unchanged();
        before = errors.errors;
        const ustring changed_name("late_output");
        OIIO_CHECK_ASSERT(!ss.attribute(group.get(), "renderer_outputs",
                                        TypeString, &changed_name));
        OIIO_CHECK_EQUAL(errors.errors, before + 1);
        OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
            errors.last_error,
            "Cannot change renderer outputs of a compiled HART group"));
        unchanged();
    }
    const OutputPlacement layered[] = {
        { { "producer.value", TypeFloat, false, SymArena::Outputs, 4, 64 }, 4 },
        { { "consumer.Cout", TypeColor, false, SymArena::Outputs, 16, 64 }, 12 },
    };
    for (bool connected : { false, true })
        for (int optimize : { 10, 3 }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", optimize == 10 ? 0 : 2);
            ss.attribute("llvm_optimize", optimize);
            ss.attribute("lazyunconnected", 1);
            ss.attribute("max_hart_groupdata_alloc", optimize == 3 ? 4096 : 0);
            ShaderGroupRef group;
            if (connected) {
                group = make_connected_group(ss, producer, consumer);
            } else {
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_producer", producer));
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_consumer", basic));
                group = ss.ShaderGroupBegin("hart_test_group");
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_producer", "producer"));
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_consumer", "consumer"));
                OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            }
            ss.clear_symlocs(group.get());
            const SymLocationDesc locations[] = { layered[0].location,
                                                  layered[1].location };
            ss.add_symlocs(group.get(), locations);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "Mapped producer (connected={}, LLVM={}):\n{}",
                      connected, optimize, errors.messages);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            // Renderer outputs force these producers to run unconditionally.
            check_module(ss, *group, arch, { }, optimize, connected, false,
                         false, 2, false, true, 0, -1, -1, { }, true);
            int size = 0, allocated = -1;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "llvm_groupdata_size", size));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
            OIIO_CHECK_EQUAL(allocated, optimize == 3 ? size : 0);
            if (optimize == 10)
                OIIO_CHECK_ASSERT(
                    check_output_placement_ir(ss, *group, layered, true));
        }
    TypeDesc unresolved = vectors, oversized = vectors;
    unresolved.arraylen = -1;
    // Only the descriptor is oversized; the shader still has two elements.
    oversized.arraylen = std::numeric_limits<int>::max() / 12 + 1;
    const struct {
        SymLocationDesc location;
        const char* diagnostic;
    } rejected[] = {
        { { "layer0.integer", TypeFloat, false, SymArena::Outputs, 0, 4 },
          "does not match location type" },
        { { "layer0.Cout", TypeFloat, false, SymArena::Outputs, 0, 4 },
          "does not match location type" },
        { { "layer0.floats", TypeDesc(TypeDesc::FLOAT, 3), false,
            SymArena::Outputs, 0, 12 },
          "does not match location type" },
        { { "layer0.vectors", oversized, false, SymArena::Outputs, 0, 24 },
          "does not match location type" },
        { { "layer0.vectors", unresolved, false, SymArena::Outputs, 0, 24 },
          "Invalid HART output location size or stride" },
        { { "layer0.scalar", TypeFloat, false, SymArena::Outputs, 0, -4 },
          "Invalid HART output location size or stride" },
        { { "layer0.scalar", TypeFloat, false, SymArena::Outputs, 0, 0 },
          "Invalid HART output location size or stride" },
        { { "layer0.scalar", TypeFloat, false, SymArena::Outputs, 0, 3 },
          "Invalid HART output location size or stride" },
        { { "layer0.Cout", TypeColor, false, SymArena::Outputs, 0, 11 },
          "Invalid HART output location size or stride" },
        { { "layer0.matrix_out", TypeMatrix, false, SymArena::Outputs, 0, 63 },
          "Invalid HART output location size or stride" },
        { { "layer0.floats", TypeDesc(TypeDesc::FLOAT, 2), false,
            SymArena::Outputs, 0, 7 },
          "Invalid HART output location size or stride" },
        { { "layer0.scalar", TypeFloat, true, SymArena::Outputs, 0, 8 },
          "Invalid HART output location size or stride" },
    };
    for (const auto& test : rejected) {
        HartServices renderer(false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        auto group = make_group(ss, bytecode);
        ss.clear_symlocs(group.get());
        ss.add_symlocs(group.get(), { &test.location, 1 });
        const int previous_failures = unit_test_failures;
        check_rejected_group(ss, *group, errors, test.diagnostic);
        if (unit_test_failures != previous_failures)
            print(stderr, "Output rejection failed for '{}' (stride={})\n",
                  test.location.name, test.location.stride);
    }
    return true;
}



bool
check_hart_entry_module(ShadingSystem& ss, ShaderGroup& group, string_view arch,
                        int optimize, cspan<ustring> sequence, bool local,
                        bool unused_tail)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_entry_sequence"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    auto& module = **parsed;
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    OIIO_CHECK_ASSERT(!llvm::verifyModule(module, &stream));
    if (!diagnostic.empty())
        print(stderr, "{}\n", diagnostic);
    const llvm::Triple triple(module.getTargetTriple());
    OIIO_CHECK_EQUAL(triple.getArch(), llvm::Triple::amdgcn);
    OIIO_CHECK_EQUAL(triple.getOS(), llvm::Triple::AMDHSA);
    const auto& layout = module.getDataLayout();
    OIIO_CHECK_EQUAL(layout.getAllocaAddrSpace(), 5);
    OIIO_CHECK_EQUAL(layout.getPointerSize(0), 8);
    bool provenance = false;
    for (const auto& global : module.globals()) {
        provenance |= global.getName().contains("__hart_device_storage_abi");
        OIIO_CHECK_ASSERT(global.isDeclaration()
                          || !global.hasExternalLinkage());
    }
    OIIO_CHECK_ASSERT(provenance);
    OIIO_CHECK_ASSERT(module.getNamedGlobal("llvm.compiler.used"));
    int exports = 0;
    for (const auto& function : module) {
        exports += function.getName().find("__direct_callable__") == 0;
        if (function.getName().find("osl_") == 0 && !function.use_empty()) {
            if (function.isDeclaration())
                print(stderr, "Undefined entry-group shadeop '{}' (arch {})\n",
                      function.getName().str(), arch);
            OIIO_CHECK_ASSERT(!function.isDeclaration());
        }
        OIIO_CHECK_ASSERT(!function.hasFnAttribute("nvptx-f32ftz"));
        const auto cpu = function.getFnAttribute("target-cpu");
        if (cpu.isStringAttribute())
            OIIO_CHECK_EQUAL(cpu.getValueAsString().str(), std::string(arch));
        for (const auto& block : function)
            for (const auto& inst : block) {
                if (const auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(
                        &inst))
                    OIIO_CHECK_EQUAL(allocation->getAddressSpace(), 5);
                if (const auto* cast = llvm::dyn_cast<llvm::IntToPtrInst>(&inst))
                    if (const auto* address = llvm::dyn_cast<llvm::ConstantInt>(
                            cast->getOperand(0)))
                        OIIO_CHECK_ASSERT(address->isZero());
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                    if (const auto* callee = call->getCalledFunction())
                        OIIO_CHECK_ASSERT(
                            callee->getName().find("__direct_callable__") != 0);
                if (function.getName().find("osl_layer_group_") == 0
                    || function.getName().find("osl_init_group_") == 0
                    || function.getName().find("__direct_callable__") == 0)
                    for (const auto& operand : inst.operands())
                        if (const auto* cast
                            = llvm::dyn_cast<llvm::ConstantExpr>(operand.get()))
                            if (cast->getOpcode()
                                == llvm::Instruction::IntToPtr) {
                                const auto* address
                                    = llvm::dyn_cast<llvm::ConstantInt>(
                                        cast->getOperand(0));
                                OIIO_CHECK_ASSERT(address && address->isZero());
                            }
            }
    }
    OIIO_CHECK_EQUAL(exports, 3);
    const char* queries[]       = { "group_init_name", "group_entry_name",
                                    "group_fused_name" };
    llvm::Function* wrappers[3] = { };
    for (size_t i = 0; i < std::size(queries); ++i) {
        ustring name;
        OIIO_CHECK_ASSERT(ss.getattribute(&group, queries[i], name));
        wrappers[i] = module.getFunction(name.c_str());
        OIIO_CHECK_ASSERT(wrappers[i] && wrappers[i]->hasExternalLinkage()
                          && wrappers[i]->getName().find("__direct_callable__")
                                 == 0);
        if (!wrappers[i])
            return false;
        check_function_abi(*wrappers[i], arch);
    }
    OIIO_CHECK_ASSERT(wrappers[0] != wrappers[1] && wrappers[0] != wrappers[2]
                      && wrappers[1] != wrappers[2]);
    int group_size = 0, alignment = 0, allocated = -1;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_size", group_size));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_alignment", alignment));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_groupdata_alloc", allocated));
    OIIO_CHECK_ASSERT(group_size > 0 && alignment > 0);
    if (alignment > 0)
        OIIO_CHECK_EQUAL(group_size % alignment, 0);
    OIIO_CHECK_EQUAL(allocated, local ? group_size : 0);
    if (local)
        OIIO_CHECK_ASSERT(group_size <= 4096);
    if (unused_tail) {
        const auto* tail = module.getFunction(
            "osl_layer_group_hart_test_group_name_tail");
        OIIO_CHECK_ASSERT(!tail || tail->use_empty());
    }
    if (optimize != 10)
        return true;

    auto* storage  = llvm::StructType::getTypeByName(context, "Groupdata");
    auto* init     = module.getFunction("osl_init_group_hart_test_group");
    auto* producer = module.getFunction(
        "osl_layer_group_hart_test_group_name_producer");
    OIIO_CHECK_ASSERT(storage && init && producer);
    if (!storage || !init || !producer)
        return false;
    OIIO_CHECK_EQUAL(layout.getTypeAllocSize(storage).getFixedValue(),
                     uint64_t(group_size));
    OIIO_CHECK_EQUAL(layout.getABITypeAlign(storage).value(),
                     uint64_t(alignment));
    for (const auto* body : { init, producer }) {
        OIIO_CHECK_ASSERT(body->hasLocalLinkage());
        check_function_abi(*body, arch);
    }
    std::vector<const llvm::Function*> targets;
    std::vector<llvm::Function*> unique_targets;
    for (ustring name : sequence) {
        auto* body = module.getFunction(
            fmtformat("osl_layer_group_hart_test_group_name_{}", name));
        OIIO_CHECK_ASSERT(body && body->hasLocalLinkage());
        if (!body)
            return false;
        check_function_abi(*body, arch);
        targets.push_back(body);
        if (std::find(unique_targets.begin(), unique_targets.end(), body)
            == unique_targets.end())
            unique_targets.push_back(body);
    }
    check_wrapper(wrappers[0], { init });
    check_wrapper(wrappers[1], targets);
    targets.insert(targets.begin(), init);
    check_wrapper(wrappers[2], targets, local ? storage : nullptr, alignment);

    // The producer is the first layer, whether a shared dependency or itself
    // an explicit entry.
    const bool producer_is_entry = std::find(unique_targets.begin(),
                                             unique_targets.end(), producer)
                                   != unique_targets.end();
    OIIO_CHECK_ASSERT(storage->getNumElements() > 0);
    if (!storage->getNumElements())
        return false;
    auto* flags = llvm::dyn_cast<llvm::ArrayType>(storage->getElementType(0));
    OIIO_CHECK_ASSERT(flags && flags->getElementType()->isIntegerTy(1));
    if (!flags)
        return false;
    OIIO_CHECK_EQUAL(layout.getStructLayout(storage)->getElementOffset(0), 0);
    OIIO_CHECK_EQUAL(flags->getNumElements(), 4);
    const auto producer_flag = [&](const llvm::Value* pointer,
                                   const llvm::Function& function) {
        int64_t offset = 0;
        const auto* base
            = llvm::GetPointerBaseWithConstantOffset(pointer, offset, layout);
        return base == function.getArg(1) && offset == 0;
    };
    const auto guarded_by_producer_flag = [&](const llvm::Instruction* target,
                                              llvm::Function& function) {
        llvm::DominatorTree dominators(function);
        for (const auto& block : function) {
            const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                block.getTerminator());
            const auto* cmp = branch && branch->isConditional()
                                  ? llvm::dyn_cast<llvm::ICmpInst>(
                                        branch->getCondition())
                                  : nullptr;
            if (!cmp || !cmp->isEquality())
                continue;
            for (unsigned i = 0; i < 2; ++i) {
                const auto* load = llvm::dyn_cast<llvm::LoadInst>(
                    cmp->getOperand(i));
                const auto* ran = llvm::dyn_cast<llvm::ConstantInt>(
                    cmp->getOperand(1 - i));
                if (!load || !load->getType()->isIntegerTy(1) || !ran
                    || !ran->isOne()
                    || !producer_flag(load->getPointerOperand(), function))
                    continue;
                const unsigned unran
                    = cmp->getPredicate() == llvm::CmpInst::ICMP_NE ? 0 : 1;
                if (dominators.dominates(branch->getSuccessor(unran),
                                         target->getParent())
                    && !dominators.dominates(branch->getSuccessor(1 - unran),
                                             target->getParent()))
                    return true;
            }
        }
        return false;
    };
    int resets = 0, marks = 0;
    for (const auto& function : module) {
        const bool layer = function.getName().find("osl_layer_group_") == 0;
        if (&function != init && !layer)
            continue;
        for (const auto& block : function)
            for (const auto& inst : block) {
                if (const auto* clear = llvm::dyn_cast<llvm::MemSetInst>(&inst))
                    if (producer_flag(clear->getRawDest(), function)) {
                        ++resets;
                        OIIO_CHECK_EQUAL(&function, init);
                        const auto* zero = llvm::dyn_cast<llvm::ConstantInt>(
                            clear->getValue());
                        const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                            clear->getLength());
                        OIIO_CHECK_ASSERT(zero && zero->isZero() && length);
                        if (length)
                            OIIO_CHECK_EQUAL(
                                length->getZExtValue(),
                                layout.getTypeAllocSize(flags).getFixedValue());
                    }
                if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst))
                    if (producer_flag(store->getPointerOperand(), function)) {
                        ++marks;
                        OIIO_CHECK_EQUAL(&function, producer);
                        const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                            store->getValueOperand());
                        OIIO_CHECK_ASSERT(value && value->isOne());
                        if (producer_is_entry)
                            OIIO_CHECK_ASSERT(
                                guarded_by_producer_flag(store, *producer));
                        else
                            OIIO_CHECK_EQUAL(store->getParent(),
                                             &producer->getEntryBlock());
                    }
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                    OIIO_CHECK_ASSERT(call->getCalledFunction() != init);
            }
    }
    OIIO_CHECK_EQUAL(resets, 1);
    OIIO_CHECK_EQUAL(marks, 1);
    for (auto* entry : unique_targets) {
        int calls = 0;
        for (const auto* user : producer->users()) {
            const auto* call = llvm::dyn_cast<llvm::CallBase>(user);
            if (!call || call->getCalledFunction() != producer
                || call->getFunction() != entry)
                continue;
            ++calls;
            OIIO_CHECK_EQUAL(call->arg_size(), 6);
            OIIO_CHECK_EQUAL(call->getCallingConv(),
                             producer->getCallingConv());
            for (unsigned i = 0; i < call->arg_size() && i < 6; ++i)
                OIIO_CHECK_EQUAL(call->getArgOperand(i), entry->getArg(i));
            OIIO_CHECK_ASSERT(guarded_by_producer_flag(call, *entry));
        }
        if (entry == producer)
            OIIO_CHECK_EQUAL(calls, 0);
        else if (!producer_is_entry)
            OIIO_CHECK_ASSERT(calls > 0);
    }
    return true;
}



void
check_hart_entry_selection(ShadingSystem& ss, ShaderGroup& group,
                           cspan<ustring> declared, cspan<ustring> sequence)
{
    int layers = 0, count = -1;
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "num_layers", layers));
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "num_entry_layers", count));
    OIIO_CHECK_EQUAL(count, int(declared.size()));
    std::vector<ustring> declarations(layers + 1, ustring("unwritten"));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "entry_layers",
                        TypeDesc(TypeDesc::STRING, int(declarations.size())),
                        declarations.data()));
    for (size_t i = 0; i < declarations.size(); ++i)
        if (i < declared.size())
            OIIO_CHECK_EQUAL(declarations[i], declared[i]);
        else
            OIIO_CHECK_ASSERT(declarations[i].empty());
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "num_hart_entry_layers", count));
    OIIO_CHECK_EQUAL(count, int(sequence.size()));
    std::vector<ustring> selected(sequence.size() + 2, ustring("unwritten"));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_entry_layers",
                        TypeDesc(TypeDesc::STRING, int(selected.size())),
                        selected.data()));
    for (size_t i = 0; i < selected.size(); ++i)
        if (i < sequence.size())
            OIIO_CHECK_EQUAL(selected[i], sequence[i]);
        else
            OIIO_CHECK_ASSERT(selected[i].empty());
}



bool
check_explicit_entry_modules(string_view arch, string_view stdosl,
                             string_view basic, string_view producer)
{
    OSLCompiler compiler;
    std::string entry;
    if (!compiler.compile_buffer(
            "shader hart_numeric_entry(float value=0,float scale=1,"
            "output color Cout=0) { Cout=color(scale*value,u,v); }",
            entry, { }, stdosl))
        return false;
    const ustring left("left"), right("right");
    auto set_entries = [&](ShadingSystem& ss, ShaderGroup& group,
                           string_view attribute, cspan<ustring> names) {
        OIIO_CHECK_ASSERT(!names.empty());
        return ss.attribute(&group, attribute,
                            TypeDesc(TypeDesc::STRING, int(names.size())),
                            names.data());
    };
    auto make = [&](ShadingSystem& ss, cspan<ustring> declared, bool tail) {
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("hart_producer", producer));
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_entry", entry));
        if (tail)
            OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_tail", basic));
        auto group = ss.ShaderGroupBegin("hart_test_group");
        OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_producer", "producer"));
        for (int i : { 1, 2 }) {
            const float scale = float(i);
            OIIO_CHECK_ASSERT(ss.Parameter("scale", TypeFloat, &scale));
            OIIO_CHECK_ASSERT(
                ss.Shader("surface", "hart_entry", i == 1 ? "left" : "right"));
        }
        if (tail)
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_tail", "tail"));
        OIIO_CHECK_ASSERT(
            ss.ConnectShaders("producer", "value", "left", "value"));
        OIIO_CHECK_ASSERT(
            ss.ConnectShaders("producer", "value", "right", "value"));
        OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        if (!declared.empty())
            OIIO_CHECK_ASSERT(
                set_entries(ss, *group, "entry_layers", declared));
        for (ustring name : { left, right })
            if ((declared.empty() && name == right)
                || std::find(declared.begin(), declared.end(), name)
                       != declared.end()) {
                const SymLocationDesc output(fmtformat("{}.Cout", name),
                                             TypeColor, false,
                                             SymArena::Outputs,
                                             name == left ? 0 : 16, 32);
                ss.add_symlocs(group.get(), { &output, 1 });
            }
        return group;
    };
    const struct {
        std::vector<ustring> declared, selected;
        bool after_optimization, tail;
        int osl, llvm;
        bool local;
    } variants[] = {
        { { left, right }, { }, false, true, 0, 10, false },
        { { right, left, right }, { }, false, true, 2, 10, true },
        { { left, right }, { right, left, right }, true, true, 2, 10, false },
        { { left, right }, { right }, false, true, 2, 10, true },
        { { left }, { }, false, true, 2, 10, false },
        { { left, right }, { right, left, right }, false, true, 2, 3, true },
        { { }, { }, false, false, 0, 10, false },
        { { }, { right, right }, false, false, 2, 10, true },
    };
    for (const auto& variant : variants) {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", variant.osl);
        ss.attribute("llvm_optimize", variant.llvm);
        ss.attribute("lazyunconnected", 1);
        ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
        auto group = make(ss, variant.declared, variant.tail);
        std::vector<ustring> declarations;
        for (ustring name : { left, right })
            if (std::find(variant.declared.begin(), variant.declared.end(), name)
                != variant.declared.end())
                declarations.push_back(name);
        std::vector<ustring> sequence = variant.declared.empty()
                                            ? std::vector<ustring> { right }
                                            : variant.declared;
        check_hart_entry_selection(ss, *group, declarations, sequence);
        int optimized = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "is_optimized", optimized));
        OIIO_CHECK_EQUAL(optimized, 0);
        if (variant.after_optimization) {
            ss.optimize_group(group.get(), nullptr, false);
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "is_optimized", optimized));
            OIIO_CHECK_EQUAL(optimized, 1);
        }
        if (!variant.selected.empty()) {
            OIIO_CHECK_ASSERT(
                set_entries(ss, *group, "hart_entry_layers", variant.selected));
            sequence = variant.selected;
        }
        check_hart_entry_selection(ss, *group, declarations, sequence);
        std::vector<std::string> entry_names;
        for (ustring name : sequence)
            entry_names.push_back(name.string());
        const auto label = OIIO::Strutil::join(entry_names, ",");
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "HART entries {} (OSL {}, LLVM {}, local={}):\n{}",
                  label, variant.osl, variant.llvm, variant.local,
                  errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        const int previous_failures = unit_test_failures;
        OIIO_CHECK_ASSERT(check_hart_entry_module(ss, *group, arch,
                                                  variant.llvm, sequence,
                                                  variant.local, variant.tail));
        check_hart_entry_selection(ss, *group, declarations, sequence);
        if (unit_test_failures != previous_failures)
            print(stderr,
                  "Entry module checks failed: {} (LLVM {}, local={})\n", label,
                  variant.llvm, variant.local);
    }

    // Failed setters are transactional, not failed shader groups: preserve the
    // valid declaration/sequence and prove that it can still be compiled.
    for (bool explicit_entries : { false, true }) {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("error_repeats", 1));
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        ss.attribute("llvm_optimize", 10);
        ss.attribute("max_hart_groupdata_alloc", 4096);
        const std::vector<ustring> declarations
            = explicit_entries ? std::vector<ustring> { left, right }
                               : std::vector<ustring> { };
        std::vector<ustring> sequence = explicit_entries
                                            ? declarations
                                            : std::vector<ustring> { right };
        auto group                   = make(ss, declarations, explicit_entries);
        const void* artifact_pointer = nullptr;
        std::string artifact;
        int optimized         = 0;
        const char* queries[] = { "group_init_name", "group_entry_name",
                                  "group_fused_name" };
        ustring names[3];
        for (unsigned i = 0; i < 3; ++i)
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), queries[i], names[i]));
        auto unchanged = [&]() {
            check_hart_entry_selection(ss, *group, declarations, sequence);
            int state = -1;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "is_optimized", state));
            OIIO_CHECK_EQUAL(state, optimized);
            for (unsigned i = 0; i < 3; ++i) {
                ustring name;
                OIIO_CHECK_ASSERT(
                    ss.getattribute(group.get(), queries[i], name));
                OIIO_CHECK_EQUAL(name, names[i]);
            }
            const void* current = nullptr;
            uint64_t length     = 0;
            OIIO_CHECK_EQUAL(ss.getattribute(group.get(), "hart_bitcode",
                                             TypeDesc::PTR, &current),
                             !artifact.empty());
            OIIO_CHECK_EQUAL(ss.getattribute(group.get(), "hart_bitcode_size",
                                             TypeUInt64, &length),
                             !artifact.empty());
            OIIO_CHECK_EQUAL(current, artifact_pointer);
            OIIO_CHECK_EQUAL(length, artifact.size());
            if (current && length == artifact.size())
                OIIO_CHECK_ASSERT(
                    string_view(static_cast<const char*>(current), length)
                    == string_view(artifact));
        };
        auto reject = [&](string_view attribute,
                          std::initializer_list<ustring> attempted,
                          string_view message) {
            const int before = errors.errors;
            OIIO_CHECK_ASSERT(!set_entries(ss, *group, attribute,
                                           cspan<ustring>(attempted.begin(),
                                                          attempted.size())));
            OIIO_CHECK_EQUAL(errors.errors, before + 1);
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, message));
            unchanged();
        };
        unchanged();
        reject("entry_layers", { right, ustring("missing") },
               "not a shader layer");
        reject("hart_entry_layers", { right, ustring("missing") },
               "not a declared entry layer");
        reject("hart_entry_layers",
               { right, explicit_entries ? ustring("producer") : left },
               "not a declared entry layer");
        if (explicit_entries)
            reject("hart_entry_layers", { left, ustring("tail") },
                   "not a declared entry layer");
        const int before_optimization = errors.errors;
        ss.optimize_group(group.get(), nullptr, false);
        OIIO_CHECK_EQUAL(errors.errors, before_optimization);
        optimized = 1;
        unchanged();
        reject("entry_layers", { left }, "after optimization");
        sequence = explicit_entries
                       ? std::vector<ustring> { right, left, right }
                       : std::vector<ustring> { right, right };
        const int before_selection = errors.errors;
        OIIO_CHECK_ASSERT(
            set_entries(ss, *group, "hart_entry_layers", sequence));
        OIIO_CHECK_EQUAL(errors.errors, before_selection);
        unchanged();
        ss.optimize_group(group.get(), nullptr);
        OIIO_CHECK_EQUAL(errors.errors, before_selection);
        uint64_t size = 0;
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                          TypeDesc::PTR, &artifact_pointer));
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                          TypeUInt64, &size));
        if (!artifact_pointer || !size)
            return false;
        artifact.assign(static_cast<const char*>(artifact_pointer), size);
        OIIO_CHECK_ASSERT(check_hart_entry_module(ss, *group, arch, 10,
                                                  sequence, true,
                                                  explicit_entries));
        unchanged();
        reject("entry_layers", { right }, "after optimization");
        reject("hart_entry_layers", { right }, "after compilation");
    }
    return true;
}



bool
interactive_address(const llvm::Value* value, const llvm::Value* arena,
                    const llvm::DataLayout& layout, int64_t& offset,
                    unsigned depth = 0)
{
    if (depth > 16)
        return false;
    if (value == arena) {
        offset = 0;
        return true;
    }
    if (const auto* cast = llvm::dyn_cast<llvm::CastInst>(value))
        if (cast->getOpcode() == llvm::Instruction::BitCast
            || cast->getOpcode() == llvm::Instruction::AddrSpaceCast
            || cast->getOpcode() == llvm::Instruction::PtrToInt
            || cast->getOpcode() == llvm::Instruction::IntToPtr)
            return interactive_address(cast->getOperand(0), arena, layout,
                                       offset, depth + 1);
    if (const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(value)) {
        llvm::APInt delta(layout.getIndexTypeSizeInBits(gep->getType()), 0);
        if (gep->accumulateConstantOffset(layout, delta)
            && interactive_address(gep->getPointerOperand(), arena, layout,
                                   offset, depth + 1)) {
            offset += delta.getSExtValue();
            return true;
        }
    }
    if (const auto* sum = llvm::dyn_cast<llvm::BinaryOperator>(value))
        if (sum->getOpcode() == llvm::Instruction::Add)
            for (unsigned i = 0; i < 2; ++i)
                if (const auto* constant = llvm::dyn_cast<llvm::ConstantInt>(
                        sum->getOperand(i)))
                    if (interactive_address(sum->getOperand(1 - i), arena,
                                            layout, offset, depth + 1)) {
                        offset += constant->getSExtValue();
                        return true;
                    }
    return false;
}



struct InteractiveField {
    const char* name;
    TypeDesc type;
    size_t offset = 0, extent = 0;
};



std::vector<uint8_t>
interactive_payload(TypeDesc type, const void* value)
{
    std::vector<uint8_t> bytes(type.size());
    if (type.basetype == TypeDesc::STRING) {
        const auto* strings = static_cast<const ustring*>(value);
        OIIO_CHECK_EQUAL(type.size(), 8 * type.numelements());
        for (int i = 0; i < type.numelements(); ++i) {
            const uint64_t hash = strings[i].hash();
            std::memcpy(bytes.data() + 8 * i, &hash, 8);
        }
    } else {
        std::memcpy(bytes.data(), value, bytes.size());
    }
    return bytes;
}



bool
check_interactive_ir(ShadingSystem& ss, ShaderGroup& group,
                     cspan<InteractiveField> fields)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_interactive"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module       = **parsed;
    const auto& layout = module.getDataLayout();
    OIIO_CHECK_ASSERT(layout.isLittleEndian());
    OIIO_CHECK_EQUAL(layout.getPointerSize(0), 8);
    std::vector<unsigned> planes(fields.size(), 0);
    std::vector<unsigned> string_elements(fields.size(), 0);
    for (const auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0)
            continue;
        OIIO_CHECK_EQUAL(function.arg_size(), 6);
        if (function.arg_size() != 6)
            continue;
        for (const auto& block : function)
            for (const auto& inst : block) {
                int64_t offset = -1;
                if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst))
                    OIIO_CHECK_ASSERT(
                        !interactive_address(store->getPointerOperand(),
                                             function.getArg(5), layout,
                                             offset));
                const auto* load = llvm::dyn_cast<llvm::LoadInst>(&inst);
                if (!load
                    || !interactive_address(load->getPointerOperand(),
                                            function.getArg(5), layout, offset))
                    continue;
                bool matched = false;
                for (size_t i = 0; i < fields.size(); ++i) {
                    const auto& field = fields[i];
                    if (offset < 0 || size_t(offset) < field.offset
                        || size_t(offset) >= field.offset + field.extent)
                        continue;
                    matched               = true;
                    const size_t relative = size_t(offset) - field.offset;
                    const bool string = field.type.basetype == TypeDesc::STRING;
                    const size_t scalar_size = string ? 8 : 4;
                    OIIO_CHECK_EQUAL(relative % scalar_size, 0);
                    OIIO_CHECK_EQUAL(size_t(offset) % scalar_size, 0);
                    OIIO_CHECK_EQUAL(layout.getTypeStoreSize(load->getType())
                                         .getFixedValue(),
                                     scalar_size);
                    OIIO_CHECK_ASSERT(string ? load->getType()->isIntegerTy(64)
                                      : field.type.basetype == TypeDesc::INT
                                          ? load->getType()->isIntegerTy(32)
                                          : load->getType()->isFloatTy());
                    OIIO_CHECK_ASSERT(relative + scalar_size <= field.extent);
                    planes[i] |= 1u << (relative / field.type.size());
                    if (string)
                        string_elements[i] |= 1u << (relative / 8);
                }
                OIIO_CHECK_ASSERT(matched);
            }
    }
    for (size_t i = 0; i < fields.size(); ++i) {
        OIIO_CHECK_ASSERT(planes[i] & 1u);
        if (fields[i].type.basetype == TypeDesc::STRING)
            OIIO_CHECK_EQUAL(string_elements[i],
                             (1u << fields[i].type.numelements()) - 1);
        // The fixture directly demands both gradients of gain and tint. Any
        // retained derivative planes must also be read through argument 5.
        if (fields[i].extent > fields[i].type.size()
            && (string_view(fields[i].name) == "gain"
                || string_view(fields[i].name) == "tint"))
            OIIO_CHECK_EQUAL(planes[i], 7u);
    }
    return true;
}



bool
check_interactive_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_interactive("
        "int count=7 [[int interactive=1]], float gain=1.25, "
        "color tint=color(0.25,0.5,0.75) [[int interactive=1]], "
        "matrix basis=1 [[int interactive=1]], "
        "int indices[3]={7,11,13} [[int interactive=1]], "
        "float weights[]={0.125,0.25} [[int interactive=1]], "
        "string label=\"alpha\" [[int interactive=1]], "
        "string tags[2]={\"beta\",\"\"} [[int interactive=1]], "
        "output float value=0, output color Cout=0) { "
        "color dx=Dx(tint), dy=Dy(tint); "
        "value=count+gain*(1+u)+tint[0]+tint[1]+tint[2]"
        "+basis[0][1]+basis[3][2]+indices[0]+indices[1]+indices[2]"
        "+weights[0]+weights[1]+arraylength(weights)"
        "+(label==\"alpha\")+2*(tags[0]==label)+4*(tags[1]==\"\")"
        "+Dx(gain)+Dy(gain)+dx[0]+dy[1]; "
        "Cout=color(value,Dx(gain)+dx[0],Dy(gain)+dy[1]); }",
        "shader hart_interactive_consumer(float value=0, output color Cout=0) "
        "{ Cout=color(value,Dx(value),Dy(value)); }",
        "shader hart_interpolated(float gain=1 [[int interpolated=1]], "
        "output color Cout=0) { Cout=color(gain*u); }",
    };
    std::string oso[3];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, stdosl))
            return false;
    }
    const struct {
        int osl, llvm, length;
        bool local, connected;
        int failure;  // 1: allocation; 2: copy/recover; 3: copy/abandon.
    } variants[] = {
        { 0, 10, 2, false, false, 0 }, { 2, 10, 5, true, false, 0 },
        { 2, 3, 3, true, false, 0 },   { 0, 10, 3, false, true, 0 },
        { 2, 3, 5, true, true, 0 },    { 2, 10, 3, false, false, 1 },
        { 2, 10, 3, false, false, 2 }, { 2, 10, 3, false, false, 3 },
    };
    for (const auto& variant : variants) {
        HartInteractiveServices renderer;
        auto run = [&]() {
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", variant.osl);
            ss.attribute("llvm_optimize", variant.llvm);
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
            OIIO_CHECK_ASSERT(ss.attribute("error_repeats", 1));
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_producer", oso[0]));
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_consumer", oso[1]));
            const char* layer = variant.connected ? "producer" : "layer0";
            const int count = 7, indices[] = { 7, 11, 13 };
            const float gain = 1.25f, tint[] = { .25f, .5f, .75f };
            const float basis[]   = { 1, 0, 0, 0, 0, 1, 0, 0,
                                      0, 0, 1, 0, 0, 0, 0, 1 };
            const float weights[] = { .125f, .25f, .5f, .75f, 1 };
            const ustring label("alpha"),
                tags[] = { ustring("beta"), ustring("") };
            std::vector<InteractiveField> fields = {
                { "count", TypeInt },
                { "gain", TypeFloat },
                { "tint", TypeColor },
                { "basis", TypeMatrix },
                { "indices", TypeDesc(TypeDesc::INT, 3) },
                { "weights", TypeDesc(TypeDesc::FLOAT, variant.length) },
                { "label", TypeString },
                { "tags", TypeDesc(TypeDesc::STRING, 2) },
            };
            const void* initial[] = { &count,  &gain,   tint,   basis,
                                      indices, weights, &label, tags };
            auto group            = ss.ShaderGroupBegin("hart_test_group");
            OIIO_CHECK_ASSERT(ss.Parameter("gain", TypeFloat, &gain,
                                           ParamHints::interactive));
            if (variant.length != 2)
                OIIO_CHECK_ASSERT(ss.Parameter("weights", fields[5].type,
                                               weights,
                                               ParamHints::interactive));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_producer", layer));
            if (variant.connected) {
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_consumer", "consumer"));
                OIIO_CHECK_ASSERT(ss.ConnectShaders("producer", "value",
                                                    "consumer", "value"));
            }
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output(variant.connected ? "consumer.Cout"
                                                           : "layer0.Cout",
                                         TypeColor, false, SymArena::Outputs, 0,
                                         12);
            ss.add_symlocs(group.get(), { &output, 1 });
            auto no_artifact = [&]() {
                const void* bytes = nullptr;
                uint64_t size     = 0;
                OIIO_CHECK_ASSERT(!ss.getattribute(group.get(), "hart_bitcode",
                                                   TypeDesc::PTR, &bytes));
                OIIO_CHECK_ASSERT(!ss.getattribute(group.get(),
                                                   "hart_bitcode_size",
                                                   TypeUInt64, &size));
                OIIO_CHECK_ASSERT(!bytes && !size);
            };
            auto invalid_binding = [&]() {
                void* pointer    = &renderer;
                const int before = errors.errors;
                OIIO_CHECK_ASSERT(!ss.getattribute(group.get(),
                                                   "device_interactive_params",
                                                   TypeDesc::PTR, &pointer));
                OIIO_CHECK_ASSERT(!pointer);
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
                    errors.last_error,
                    "interactive parameter device storage is invalid"));
            };
            int optimized = -1;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "is_optimized", optimized));
            OIIO_CHECK_EQUAL(optimized, 0);
            renderer.fail_allocation = variant.failure == 1;
            renderer.fail_copy       = variant.failure >= 2;
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors != (variant.failure ? 1 : 0)
                || renderer.allocations != 1)
                print(stderr, "Interactive compilation:\n{}", errors.messages);
            OIIO_CHECK_EQUAL(errors.errors, variant.failure ? 1 : 0);
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "is_optimized", optimized));
            OIIO_CHECK_EQUAL(optimized, 1);
            OIIO_CHECK_EQUAL(renderer.allocations, 1);
            OIIO_CHECK_EQUAL(renderer.successful_allocations,
                             variant.failure == 1 ? 0 : 1);
            OIIO_CHECK_EQUAL(renderer.copies.size(),
                             variant.failure == 1 ? 0 : 1);

            // Read actual optimized symbol extents and arena offsets. Do not
            // guess a host C++ struct layout or mutate private group state.
            size_t extent = 0;
            for (auto& field : fields) {
                const int offset
                    = group->interactive_param_offset(0, ustring(field.name));
                OIIO_CHECK_ASSERT(offset >= 0);
                if (offset < 0)
                    return false;
                field.offset = size_t(offset);
                for (const auto& symbol : group->layer(0)->symbols())
                    if (symbol.name() == field.name) {
                        OIIO_CHECK_ASSERT(symbol.interactive());
                        OIIO_CHECK_ASSERT(symbol.typespec().simpletype()
                                          == field.type);
                        field.extent = field.type.size()
                                       * (symbol.has_derivs() ? 3 : 1);
                    }
                OIIO_CHECK_ASSERT(field.extent > 0);
                if (!field.extent)
                    return false;
                extent = std::max(extent, field.offset + field.extent);
            }
            for (size_t i = 0; i < fields.size(); ++i)
                for (size_t j = i + 1; j < fields.size(); ++j)
                    OIIO_CHECK_ASSERT(fields[i].offset + fields[i].extent
                                          <= fields[j].offset
                                      || fields[j].offset + fields[j].extent
                                             <= fields[i].offset);
            OIIO_CHECK_EQUAL(renderer.requested_size, extent);
            if (renderer.requested_size != extent || !extent || extent > 4096)
                return false;
            std::vector<uint8_t> expected(extent, 0);
            for (size_t i = 0; i < fields.size(); ++i) {
                auto payload = interactive_payload(fields[i].type, initial[i]);
                std::memcpy(expected.data() + fields[i].offset, payload.data(),
                            payload.size());
            }
            auto host_matches = [&]() {
                void* host = nullptr;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "interactive_params",
                                                  TypeDesc::PTR, &host));
                OIIO_CHECK_ASSERT(host);
                if (host)
                    OIIO_CHECK_EQUAL(std::memcmp(host, expected.data(), extent),
                                     0);
            };
            auto device_matches = [&]() {
                void* device = nullptr;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "device_interactive_params",
                                                  TypeDesc::PTR, &device));
                OIIO_CHECK_ASSERT(device && device == renderer.storage.get());
                if (device && device == renderer.storage.get())
                    OIIO_CHECK_EQUAL(std::memcmp(device, expected.data(),
                                                 extent),
                                     0);
                host_matches();
            };
            host_matches();
            if (variant.failure) {
                OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
                    errors.last_error, variant.failure == 1
                                           ? "failed to allocate interactive"
                                           : "failed to upload interactive"));
                no_artifact();
                invalid_binding();
                invalid_binding();
                if (renderer.storage)
                    OIIO_CHECK_ASSERT(std::memcmp(renderer.storage.get(),
                                                  expected.data(), extent)
                                      != 0);
                if (variant.failure == 3)
                    return true;
                const size_t copies = renderer.copies.size();
                const int before    = errors.errors;
                ss.optimize_group(group.get(), nullptr);
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_EQUAL(renderer.copies.size(), copies);
                no_artifact();
                renderer.fail_allocation = renderer.fail_copy = false;
                OIIO_CHECK_ASSERT(ss.ReParameter(*group, layer, "gain", gain));
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_EQUAL(renderer.allocations,
                                 variant.failure == 1 ? 2 : 1);
                OIIO_CHECK_EQUAL(renderer.successful_allocations, 1);
                OIIO_CHECK_EQUAL(renderer.copies.size(), copies + 1);
                OIIO_CHECK_EQUAL(renderer.copies.back().offset, 0);
                OIIO_CHECK_EQUAL(renderer.copies.back().size, extent);
                device_matches();
                ss.optimize_group(group.get(), nullptr);
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_EQUAL(renderer.copies.size(), copies + 1);
            }
            device_matches();
            check_module(ss, *group, arch, { }, variant.llvm, variant.connected,
                         false, false, 0, false, true);
            if (variant.llvm == 10)
                OIIO_CHECK_ASSERT(check_interactive_ir(ss, *group, fields));
            int allocated = -1, group_size = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "llvm_groupdata_size",
                                              group_size));
            OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
            OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
            const void* artifact_pointer = nullptr;
            uint64_t size                = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR,
                                              &artifact_pointer));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                              TypeUInt64, &size));
            if (!artifact_pointer || !size)
                return false;
            const std::string artifact(static_cast<const char*>(
                                           artifact_pointer),
                                       size);
            auto unchanged = [&]() {
                const void* current = nullptr;
                uint64_t length     = 0;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                                  TypeDesc::PTR, &current));
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "hart_bitcode_size",
                                                  TypeUInt64, &length));
                OIIO_CHECK_EQUAL(current, artifact_pointer);
                OIIO_CHECK_EQUAL(length, artifact.size());
                if (current && length == artifact.size())
                    OIIO_CHECK_ASSERT(
                        string_view(static_cast<const char*>(current), length)
                        == string_view(artifact));
                OIIO_CHECK_EQUAL(renderer.successful_allocations, 1);
                OIIO_CHECK_EQUAL(renderer.frees, 0);
                host_matches();
            };
            bool valid  = true;
            auto update = [&](size_t index, const void* data, bool success) {
                const auto& field = fields[index];
                auto payload      = interactive_payload(field.type, data);
                const bool upload
                    = !valid
                      || std::memcmp(expected.data() + field.offset,
                                     payload.data(), payload.size());
                const size_t copies = renderer.copies.size();
                const int before    = errors.errors;
                OIIO_CHECK_EQUAL(ss.ReParameter(*group, layer, field.name,
                                                field.type, data),
                                 success);
                OIIO_CHECK_EQUAL(errors.errors, before + (success ? 0 : 1));
                OIIO_CHECK_EQUAL(renderer.copies.size(),
                                 copies + (upload ? 1 : 0));
                if (upload && renderer.copies.size() > copies) {
                    OIIO_CHECK_EQUAL(renderer.copies.back().offset,
                                     valid ? field.offset : 0);
                    OIIO_CHECK_EQUAL(renderer.copies.back().size,
                                     valid ? payload.size() : extent);
                }
                if (success) {
                    std::memcpy(expected.data() + field.offset, payload.data(),
                                payload.size());
                    device_matches();
                } else {
                    OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
                        errors.last_error, "failed to upload interactive"));
                    invalid_binding();
                }
                valid = success;
                unchanged();
            };
            for (size_t i = 0; i < fields.size(); ++i)
                update(i, initial[i], true);
            const int new_count = 17, new_indices[] = { 101, 202, 303 };
            const float new_gain = 2.75f, new_tint[] = { 3, 4, 5 };
            const float new_basis[]   = { 2,  3,  4,  5,  6,  7,  8,  9,
                                          10, 11, 12, 13, 14, 15, 16, 17 };
            const float new_weights[] = { 5, 4, 3, 2, 1 };
            const ustring empty, new_tags[] = { ustring(""), ustring("alpha") };
            const void* changed[] = { &new_count, &new_gain,   new_tint,
                                      new_basis,  new_indices, new_weights,
                                      &empty,     new_tags };
            for (size_t i = 0; i < fields.size(); ++i)
                update(i, changed[i], true);
            renderer.fail_copy = true;
            update(4, indices, false);
            if (renderer.storage)
                OIIO_CHECK_ASSERT(
                    std::memcmp(renderer.storage.get(), expected.data(), extent)
                    != 0);
            // Repeating an equal-valued, different field still repairs every
            // byte. A second partial repair failure must remain recoverable.
            update(1, &new_gain, false);
            renderer.fail_copy = false;
            update(1, &new_gain, true);
            renderer.fail_copy = true;
            update(7, tags, false);
            if (renderer.storage)
                OIIO_CHECK_ASSERT(
                    std::memcmp(renderer.storage.get(), expected.data(), extent)
                    != 0);
            renderer.fail_copy = false;
            update(7, new_tags, true);
            update(7, new_tags, true);
            renderer.fail_copy = true;
            update(0, &count, false);
            renderer.fail_copy       = false;
            const int repaired_count = -19;
            update(0, &repaired_count, true);
            auto reject = [&](string_view target_layer, string_view name,
                              TypeDesc type, const void* data,
                              string_view diagnostic) {
                const size_t copies = renderer.copies.size();
                const int before    = errors.errors;
                OIIO_CHECK_ASSERT(
                    !ss.ReParameter(*group, target_layer, name, type, data));
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_ASSERT(
                    OIIO::Strutil::contains(errors.last_error, diagnostic));
                OIIO_CHECK_EQUAL(renderer.copies.size(), copies);
                device_matches();
                unchanged();
            };
            reject("missing", "gain", TypeFloat, &new_gain, "unknown layer");
            reject(layer, "missing", TypeFloat, &new_gain, "unknown parameter");
            reject(variant.connected ? "consumer" : layer, "Cout", TypeColor,
                   new_tint, "was not declared interactive");
            reject(layer, "gain", TypeInt, &new_count, "type mismatch");
            reject(layer, "indices", TypeDesc(TypeDesc::INT, 2), indices,
                   "type mismatch");
            const ustring oversized_tags[] = { ustring("a"), ustring("b"),
                                               ustring("c") };
            reject(layer, "tags", TypeDesc(TypeDesc::STRING, 3), oversized_tags,
                   "type mismatch");
            reject(layer, "weights", TypeDesc(TypeDesc::FLOAT, -1), weights,
                   "type mismatch");
            reject(layer, "tint", TypeFloat, &new_gain, "invalid data or size");
            reject(layer, "gain", TypeFloat, nullptr, "invalid data or size");
            reject(layer, "tags", fields[7].type, nullptr,
                   "invalid data or size");
            OIIO_CHECK_EQUAL(renderer.allocations,
                             variant.failure == 1 ? 2 : 1);
            OIIO_CHECK_EQUAL(renderer.copies.size(),
                             variant.failure == 2 ? 17 : 16);
            for (const std::string& text :
                 { std::string(128, 'x'), std::string() }) {
                const size_t copies = renderer.copies.size();
                const int before    = errors.errors;
                OIIO_CHECK_ASSERT(ss.ReParameter(*group, layer, "label", text));
                OIIO_CHECK_EQUAL(errors.errors, before);
                OIIO_CHECK_EQUAL(renderer.copies.size(), copies + 1);
                const ustring value(text);
                const auto payload = interactive_payload(TypeString, &value);
                std::memcpy(expected.data() + fields[6].offset, payload.data(),
                            payload.size());
                device_matches();
                unchanged();
            }
            return true;
        };
        const int before = unit_test_failures;
        OIIO_CHECK_ASSERT(run());
        OIIO_CHECK_EQUAL(renderer.frees, renderer.successful_allocations);
        OIIO_CHECK_ASSERT(!renderer.storage);
        if (unit_test_failures != before)
            print(stderr,
                  "Interactive checks failed (OSL {}, LLVM {}, length {}, "
                  "local {}, connected {}, failure {})\n",
                  variant.osl, variant.llvm, variant.length, variant.local,
                  variant.connected, variant.failure);
    }
    for (bool interpolated : { false, true }) {
        HartInteractiveServices renderer;
        renderer.interactive = interpolated;
        {
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", 2);
            auto group = make_group(ss, oso[interpolated ? 2 : 0]);
            check_rejected_group(ss, *group, errors,
                                 interpolated ? "interpolated parameter"
                                              : "interactive parameter");
        }
        OIIO_CHECK_EQUAL(renderer.allocations, 0);
        OIIO_CHECK_EQUAL(renderer.copies.size(), 0);
        OIIO_CHECK_EQUAL(renderer.frees, 0);
        OIIO_CHECK_ASSERT(!renderer.storage);
    }
    return true;
}



void
check_userdata_host_layout()
{
    using namespace testshade;
    OIIO_CHECK_EQUAL(sizeof(HartTextureState), 64);
    OIIO_CHECK_EQUAL(alignof(HartTextureState), 8);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, textures), 0);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, count), 8);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, errors), 16);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, colorsystem), 24);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, diagnostics), 32);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, userdata), 40);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, attributes), 48);
    OIIO_CHECK_EQUAL(offsetof(HartTextureState, transforms), 56);
    OIIO_CHECK_ASSERT((std::is_same<decltype(HartTextureState::transforms),
                                    const HartTransformState*>::value));
    OIIO_CHECK_EQUAL(sizeof(HartTransformState), 32);
    OIIO_CHECK_EQUAL(alignof(HartTransformState), 8);
    OIIO_CHECK_EQUAL(offsetof(HartTransformState, entries), 0);
    OIIO_CHECK_EQUAL(offsetof(HartTransformState, count), 8);
    OIIO_CHECK_EQUAL(offsetof(HartTransformState, commonspace), 16);
    OIIO_CHECK_EQUAL(offsetof(HartTransformState, unknown_error), 24);
    OIIO_CHECK_EQUAL(offsetof(HartTransformState, reserved), 28);
    OIIO_CHECK_ASSERT((std::is_same<decltype(HartTransformState::entries),
                                    const HartTransformDesc*>::value));
    OIIO_CHECK_ASSERT((
        std::is_same<decltype(HartTransformState::count), uint64_t>::value
        && std::is_same<decltype(HartTransformState::commonspace), uint64_t>::value
        && std::is_same<decltype(HartTransformState::unknown_error),
                        uint32_t>::value
        && std::is_same<decltype(HartTransformState::reserved),
                        uint32_t>::value));
    OIIO_CHECK_EQUAL(sizeof(HartTransformDesc), 144);
    OIIO_CHECK_EQUAL(alignof(HartTransformDesc), 8);
    OIIO_CHECK_EQUAL(offsetof(HartTransformDesc, name), 0);
    OIIO_CHECK_EQUAL(offsetof(HartTransformDesc, directions), 8);
    OIIO_CHECK_EQUAL(offsetof(HartTransformDesc, reserved), 12);
    OIIO_CHECK_EQUAL(offsetof(HartTransformDesc, forward), 16);
    OIIO_CHECK_EQUAL(offsetof(HartTransformDesc, inverse), 80);
    OIIO_CHECK_ASSERT((
        std::is_same<decltype(HartTransformDesc::name), uint64_t>::value
        && std::is_same<decltype(HartTransformDesc::directions), uint32_t>::value
        && std::is_same<decltype(HartTransformDesc::reserved), uint32_t>::value));
    OIIO_CHECK_ASSERT(
        (std::is_same<decltype(HartTransformDesc::forward), float[16]>::value));
    OIIO_CHECK_ASSERT(
        (std::is_same<decltype(HartTransformDesc::inverse), float[16]>::value));
    OIIO_CHECK_ASSERT((std::is_same<decltype(HartTextureState::attributes),
                                    const RenderContext*>::value));
    OIIO_CHECK_EQUAL(sizeof(RenderContext), 128);
    OIIO_CHECK_EQUAL(alignof(RenderContext), 8);
    OIIO_CHECK_ASSERT((
        std::is_same<decltype(RenderContext::projection), ustringhash>::value));
    OIIO_CHECK_ASSERT((std::is_same<decltype(RenderContext::world_to_camera),
                                    Matrix44>::value));
    OIIO_CHECK_EQUAL(offsetof(RenderContext, xres), 0);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, yres), 4);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, world_to_camera), 8);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, projection), 72);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, pixelaspect), 80);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, screen_window), 84);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, shutter), 100);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, fov), 108);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, hither), 112);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, yon), 116);
    OIIO_CHECK_EQUAL(offsetof(RenderContext, journal_buffer), 120);
    OIIO_CHECK_ASSERT((std::is_same<decltype(HartTextureState::userdata),
                                    const HartUserdataState*>::value));
    OIIO_CHECK_EQUAL(sizeof(HartUserdataDesc), 48);
    OIIO_CHECK_EQUAL(alignof(HartUserdataDesc), 8);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, name), 0);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, type), 8);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, offset), 16);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, stride), 24);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, presence), 32);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, size), 40);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataDesc, derivatives), 44);
    OIIO_CHECK_EQUAL(sizeof(HartUserdataState), 48);
    OIIO_CHECK_EQUAL(alignof(HartUserdataState), 8);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, entries), 0);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, count), 8);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, data), 16);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, bytes), 24);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, points), 32);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, grid_defaults), 40);
    OIIO_CHECK_EQUAL(offsetof(HartUserdataState, reserved), 44);
    OIIO_CHECK_EQUAL(sizeof(HartRenderState), 16);
    OIIO_CHECK_EQUAL(alignof(HartRenderState), 8);
    OIIO_CHECK_EQUAL(offsetof(HartRenderState, textures), 0);
    OIIO_CHECK_EQUAL(offsetof(HartRenderState, closure_pool), 8);
    OIIO_CHECK_EQUAL(sizeof(HartGeneratedParams), 80);
    OIIO_CHECK_EQUAL(alignof(HartGeneratedParams), 8);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, output), 0);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, scratch), 8);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, group_stride), 16);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, scratch_bytes), 24);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, point_count), 32);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, raytype), 40);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, pixelcenters), 44);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, textures), 48);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, transforms), 56);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, closure_capacity), 64);
    OIIO_CHECK_EQUAL(offsetof(HartGeneratedParams, interactive), 72);
}



bool
check_userdata_ir(ShadingSystem& ss, ShaderGroup& group,
                  cspan<HartUserdataServices::Request> requests, bool connected,
                  bool lazy, bool missing_spec, bool linked_renderer = false)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_userdata"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module       = **parsed;
    const auto& layout = module.getDataLayout();
    OIIO_CHECK_ASSERT(layout.isLittleEndian());
    int count               = 0;
    const ustring* names    = nullptr;
    const TypeDesc* types   = nullptr;
    const int* offsets      = nullptr;
    const char* derivatives = nullptr;
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "num_userdata", count));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "userdata_names", TypeDesc::PTR, &names));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "userdata_types", TypeDesc::PTR, &types));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "userdata_offsets", TypeDesc::PTR, &offsets));
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "userdata_derivs", TypeDesc::PTR,
                                      &derivatives));
    OIIO_CHECK_ASSERT(count > 0 && names && types && offsets && derivatives);
    if (count <= 0 || !names || !types || !offsets || !derivatives)
        return false;
    auto* storage = llvm::StructType::getTypeByName(context, "Groupdata");
    OIIO_CHECK_ASSERT(storage
                      && storage->getNumElements() >= unsigned(2 + count));
    if (!storage || storage->getNumElements() < unsigned(2 + count))
        return false;
    const auto* fields = layout.getStructLayout(storage);
    auto* flags = llvm::dyn_cast<llvm::ArrayType>(storage->getElementType(1));
    OIIO_CHECK_ASSERT(flags && flags->getElementType()->isIntegerTy(8));
    if (!flags)
        return false;
    const uint64_t flag_bytes = (count + 3) & ~3;
    OIIO_CHECK_EQUAL(flags->getNumElements(), flag_bytes);
    const uint64_t flags_offset = fields->getElementOffset(1);
    auto at = [&](const llvm::Value* pointer, const llvm::Function& function,
                  int64_t expected) {
        int64_t offset = 0;
        return llvm::GetPointerBaseWithConstantOffset(pointer, offset, layout)
                   == function.getArg(1)
               && offset == expected;
    };
    auto peel = [](const llvm::Value* value) {
        while (const auto* cast = llvm::dyn_cast<llvm::CastInst>(value))
            value = cast->getOperand(0);
        return value;
    };
    auto integer = [](const llvm::Value* value, uint64_t expected) {
        const auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value);
        return constant && constant->getZExtValue() == expected;
    };
    auto status_call =
        [&](const llvm::StoreInst* store) -> const llvm::CallBase* {
        const auto* sum = llvm::dyn_cast<llvm::BinaryOperator>(
            peel(store->getValueOperand()));
        if (sum && sum->getOpcode() == llvm::Instruction::Add)
            for (unsigned a = 0; a < 2; ++a)
                if (integer(sum->getOperand(a), 1))
                    return llvm::dyn_cast<llvm::CallBase>(
                        peel(sum->getOperand(1 - a)));
        return nullptr;
    };
    for (int i = 0; i < count; ++i) {
        OIIO_CHECK_ASSERT(names[i] != "never");
        OIIO_CHECK_EQUAL(fields->getElementOffset(2 + i), uint64_t(offsets[i]));
        auto* data = storage->getElementType(2 + i);
        OIIO_CHECK_EQUAL(layout.getTypeAllocSize(data).getFixedValue(),
                         types[i].size()
                             * (types[i].basetype == TypeDesc::FLOAT ? 3 : 1));
        const auto typed = [&](const auto& self, llvm::Type* type) -> bool {
            if (auto* array = llvm::dyn_cast<llvm::ArrayType>(type))
                return self(self, array->getElementType());
            if (auto* record = llvm::dyn_cast<llvm::StructType>(type)) {
                if (record->isOpaque())
                    return false;
                for (auto* element : record->elements())
                    if (!self(self, element))
                        return false;
                return true;
            }
            return types[i].basetype == TypeDesc::FLOAT
                       ? type->isFloatTy()
                       : type->isIntegerTy(
                             types[i].basetype == TypeDesc::STRING ? 64 : 32);
        };
        OIIO_CHECK_ASSERT(typed(typed, data));
        if (names[i] == "shared" || names[i] == "gain" || names[i] == "tint")
            OIIO_CHECK_EQUAL(int(derivatives[i]), 1);
        if (types[i].basetype != TypeDesc::FLOAT)
            OIIO_CHECK_EQUAL(int(derivatives[i]), 0);
    }
    const auto* wrapper  = module.getFunction("osl_hart_get_userdata");
    const auto* callback = module.getFunction("rs_hart_get_userdata");
    for (const auto* function : { wrapper, callback }) {
        if (missing_spec) {
            OIIO_CHECK_ASSERT(!function || function->use_empty());
            continue;
        }
        OIIO_CHECK_ASSERT(function && !function->use_empty());
        if (!function)
            return false;
        OIIO_CHECK_EQUAL(function->isDeclaration(),
                         function == callback && !linked_renderer);
        if (function == callback && linked_renderer)
            OIIO_CHECK_ASSERT(function->hasLocalLinkage());
        OIIO_CHECK_ASSERT(function->getReturnType()->isIntegerTy(1));
        OIIO_CHECK_ASSERT(!function->isVarArg());
        OIIO_CHECK_EQUAL(function->arg_size(), 6);
        for (const auto& arg : function->args()) {
            const unsigned index = arg.getArgNo();
            OIIO_CHECK_ASSERT(
                index == 0 || index == 5
                    ? arg.getType()->isPointerTy()
                          && arg.getType()->getPointerAddressSpace() == 0
                    : arg.getType()->isIntegerTy(index == 1   ? 32
                                                 : index == 4 ? 1
                                                              : 64));
        }
    }
    for (const char* legacy : { "osl_bind_interpolated_param",
                                "rend_get_userdata", "rs_get_userdata" }) {
        const auto* function = module.getFunction(legacy);
        OIIO_CHECK_ASSERT(!function || function->use_empty());
    }
    if (!missing_spec && wrapper && callback) {
        int forwards = 0;
        for (const auto& block : *wrapper)
            for (const auto& inst : block)
                if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                    if (call->getCalledFunction() == callback) {
                        ++forwards;
                        OIIO_CHECK_EQUAL(call->arg_size(), 6);
                        for (unsigned a = 0; a < call->arg_size() && a < 6; ++a)
                            OIIO_CHECK_EQUAL(call->getArgOperand(a),
                                             wrapper->getArg(a));
                        const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(
                            block.getTerminator());
                        OIIO_CHECK_ASSERT(ret && ret->getReturnValue() == call);
                    }
        OIIO_CHECK_EQUAL(forwards, 1);
    }

    int resets = 0, calls = 0;
    std::vector<int> slot_calls(count, 0);
    std::vector<unsigned> slot_layers(count, 0);
    for (auto& function : module) {
        const bool init  = function.getName().find("osl_init_group_") == 0;
        const bool layer = function.getName().find("osl_layer_group_") == 0;
        if ((!init && !layer) || function.arg_size() != 6)
            continue;
        OIIO_CHECK_ASSERT(!function.getName().contains("_name_unused"));
        llvm::DominatorTree dominators(function);
        // LLVM10 preserves the i1 -> i32 spill/load -> nonzero test used by if.
        auto comparison_source =
            [&](const llvm::Value* condition) -> const llvm::FCmpInst* {
            const llvm::Value* value = condition;
            if (const auto* cmp = llvm::dyn_cast<llvm::ICmpInst>(value)) {
                if (cmp->getPredicate() != llvm::CmpInst::ICMP_NE)
                    return nullptr;
                if (integer(cmp->getOperand(1), 0))
                    value = cmp->getOperand(0);
                else if (integer(cmp->getOperand(0), 0))
                    value = cmp->getOperand(1);
                else
                    return nullptr;
            }
            value = peel(value);
            if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(value)) {
                const auto* temporary = llvm::dyn_cast<llvm::AllocaInst>(
                    load->getPointerOperand());
                if (!temporary
                    || temporary->getAllocatedType() != load->getType())
                    return nullptr;
                const llvm::StoreInst* writer = nullptr;
                for (const auto* user : temporary->users()) {
                    if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(
                            user)) {
                        if (writer || store->getPointerOperand() != temporary)
                            return nullptr;
                        writer = store;
                    } else if (!llvm::isa<llvm::LoadInst>(user)) {
                        return nullptr;
                    }
                }
                if (!writer || !dominators.dominates(writer, load))
                    return nullptr;
                value = peel(writer->getValueOperand());
            }
            return llvm::dyn_cast<llvm::FCmpInst>(value);
        };
        const llvm::BranchInst* varying_branch = nullptr;
        if (connected && layer) {
            for (const auto& candidate : function) {
                const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                    candidate.getTerminator());
                const auto* comparison = branch && branch->isConditional()
                                             ? comparison_source(
                                                   branch->getCondition())
                                             : nullptr;
                if (!comparison)
                    continue;
                const auto* threshold = llvm::dyn_cast<llvm::ConstantFP>(
                    comparison->getOperand(1));
                OIIO_CHECK_ASSERT(
                    (comparison->getPredicate() == llvm::CmpInst::FCMP_UGT
                     || comparison->getPredicate() == llvm::CmpInst::FCMP_OGT)
                    && threshold
                    && threshold->getValueAPF().convertToFloat() == 0.25f);
                OIIO_CHECK_ASSERT(!varying_branch);
                varying_branch = branch;
            }
            OIIO_CHECK_ASSERT(varying_branch);
        }
        for (const auto& block : function)
            for (const auto& inst : block) {
                int64_t arena_offset = 0;
                if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst))
                    OIIO_CHECK_ASSERT(
                        !interactive_address(store->getPointerOperand(),
                                             function.getArg(5), layout,
                                             arena_offset));
                if (const auto* memory = llvm::dyn_cast<llvm::MemIntrinsic>(
                        &inst))
                    OIIO_CHECK_ASSERT(!interactive_address(memory->getRawDest(),
                                                           function.getArg(5),
                                                           layout,
                                                           arena_offset));
                if (const auto* clear = llvm::dyn_cast<llvm::MemSetInst>(
                        &inst)) {
                    int64_t offset = 0;
                    if (llvm::GetPointerBaseWithConstantOffset(
                            clear->getRawDest(), offset, layout)
                            == function.getArg(1)
                        && offset >= int64_t(flags_offset)
                        && offset < int64_t(flags_offset + flag_bytes)) {
                        ++resets;
                        OIIO_CHECK_ASSERT(init);
                        OIIO_CHECK_EQUAL(offset, flags_offset);
                        OIIO_CHECK_ASSERT(integer(clear->getValue(), 0));
                        OIIO_CHECK_ASSERT(
                            integer(clear->getLength(), flag_bytes));
                    }
                }
                if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
                    int64_t offset = 0;
                    if (llvm::GetPointerBaseWithConstantOffset(
                            store->getPointerOperand(), offset, layout)
                            == function.getArg(1)
                        && offset >= int64_t(flags_offset)
                        && offset < int64_t(flags_offset + flag_bytes)) {
                        const auto* lookup = status_call(store);
                        OIIO_CHECK_ASSERT(
                            layer && lookup
                            && lookup->getCalledFunction() == wrapper
                            && lookup->getParent() == store->getParent()
                            && dominators.dominates(lookup, store)
                            && store->getValueOperand()->getType()->isIntegerTy(
                                8));
                    }
                }
                const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst);
                if (!call || call->getCalledFunction() != wrapper || !wrapper)
                    continue;
                ++calls;
                OIIO_CHECK_ASSERT(layer && !missing_spec);
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 wrapper->getCallingConv());
                OIIO_CHECK_EQUAL(call->arg_size(), 6);
                if (call->arg_size() != 6)
                    continue;
                OIIO_CHECK_EQUAL(call->getArgOperand(0)->stripPointerCasts(),
                                 function.getArg(0));
                OIIO_CHECK_EQUAL(call->getArgOperand(1), function.getArg(4));
                int slot = -1;
                for (int i = 0; i < count; ++i) {
                    uint64_t encoded_type = 0;
                    static_assert(sizeof(TypeDesc) == sizeof(encoded_type));
                    std::memcpy(&encoded_type, &types[i], sizeof(encoded_type));
                    if (integer(call->getArgOperand(2), names[i].hash())
                        && integer(call->getArgOperand(3), encoded_type))
                        slot = i;
                }
                OIIO_CHECK_ASSERT(slot >= 0);
                if (slot < 0)
                    continue;
                ++slot_calls[slot];
                if (connected) {
                    const bool producer = function.getName().contains(
                        "_name_producer");
                    OIIO_CHECK_ASSERT(
                        producer
                        || function.getName().contains("_name_consumer"));
                    slot_layers[slot] |= producer ? 1u : 2u;
                }
                OIIO_CHECK_ASSERT(
                    integer(call->getArgOperand(4), derivatives[slot] ? 1 : 0));
                OIIO_CHECK_ASSERT(
                    at(call->getArgOperand(5), function, offsets[slot]));
                const uint64_t flag_offset = flags_offset + slot;
                auto status_is = [&](const llvm::Value* value, int status) {
                    const auto* cmp = llvm::dyn_cast<llvm::ICmpInst>(value);
                    if (!cmp || cmp->getPredicate() != llvm::CmpInst::ICMP_EQ)
                        return false;
                    for (unsigned a = 0; a < 2; ++a)
                        if (integer(cmp->getOperand(a), status))
                            if (const auto* load
                                = llvm::dyn_cast<llvm::LoadInst>(
                                    peel(cmp->getOperand(1 - a))))
                                if (load->getType()->isIntegerTy(8)
                                    && at(load->getPointerOperand(), function,
                                          flag_offset))
                                    return true;
                    return false;
                };
                // Pair this binding with its own flag-0, flag-2 and default
                // diamonds, not other valid bindings of the same cache slot.
                const auto* predecessor
                    = call->getParent()->getSinglePredecessor();
                const auto* lookup_guard
                    = predecessor ? llvm::dyn_cast<llvm::BranchInst>(
                                        predecessor->getTerminator())
                                  : nullptr;
                const auto* lookup_exit = llvm::dyn_cast<llvm::BranchInst>(
                    call->getParent()->getTerminator());
                OIIO_CHECK_ASSERT(lookup_guard && lookup_guard->isConditional()
                                  && lookup_exit
                                  && lookup_exit->isUnconditional());
                if (!lookup_guard || !lookup_guard->isConditional()
                    || !lookup_exit || !lookup_exit->isUnconditional())
                    continue;
                OIIO_CHECK_ASSERT(status_is(lookup_guard->getCondition(), 0));
                OIIO_CHECK_EQUAL(lookup_guard->getSuccessor(0),
                                 call->getParent());
                OIIO_CHECK_EQUAL(lookup_guard->getSuccessor(1),
                                 lookup_exit->getSuccessor(0));
                int marks = 0;
                for (const auto& operation : *call->getParent())
                    if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(
                            &operation))
                        if (status_call(store) == call) {
                            ++marks;
                            OIIO_CHECK_ASSERT(at(store->getPointerOperand(),
                                                 function, flag_offset));
                            OIIO_CHECK_ASSERT(
                                dominators.dominates(call, store));
                        }
                OIIO_CHECK_EQUAL(marks, 1);
                if (varying_branch) {
                    if (lazy) {
                        OIIO_CHECK_ASSERT(dominators.dominates(
                            varying_branch->getSuccessor(0), predecessor));
                        OIIO_CHECK_ASSERT(!dominators.dominates(
                            varying_branch->getSuccessor(1), call->getParent()));
                    } else {
                        OIIO_CHECK_ASSERT(
                            dominators.dominates(predecessor,
                                                 varying_branch->getParent()));
                    }
                }
                const auto* hit_guard = llvm::dyn_cast<llvm::BranchInst>(
                    lookup_exit->getSuccessor(0)->getTerminator());
                OIIO_CHECK_ASSERT(hit_guard && hit_guard->isConditional());
                if (!hit_guard || !hit_guard->isConditional())
                    continue;
                const auto* hit_condition = hit_guard->getCondition();
                OIIO_CHECK_ASSERT(status_is(hit_condition, 2));
                const auto* hit_block = hit_guard->getSuccessor(0);
                OIIO_CHECK_EQUAL(hit_block->getSinglePredecessor(),
                                 hit_guard->getParent());
                const llvm::MemCpyInst* hit_copy = nullptr;
                for (const auto& operation : *hit_block)
                    if (const auto* copy = llvm::dyn_cast<llvm::MemCpyInst>(
                            &operation)) {
                        OIIO_CHECK_ASSERT(!hit_copy);
                        OIIO_CHECK_ASSERT(
                            at(copy->getRawSource(), function, offsets[slot]));
                        hit_copy = copy;
                    }
                OIIO_CHECK_ASSERT(hit_copy);
                const auto* hit_exit = llvm::dyn_cast<llvm::BranchInst>(
                    hit_block->getTerminator());
                OIIO_CHECK_ASSERT(hit_exit && hit_exit->isUnconditional());
                if (!hit_copy || !hit_exit || !hit_exit->isUnconditional())
                    continue;
                OIIO_CHECK_EQUAL(hit_exit->getSuccessor(0),
                                 hit_guard->getSuccessor(1));
                int64_t symbol_offset = 0;
                OIIO_CHECK_EQUAL(llvm::GetPointerBaseWithConstantOffset(
                                     hit_copy->getRawDest(), symbol_offset,
                                     layout),
                                 function.getArg(1));
                const pvt::Symbol* symbol = nullptr;
                int symbol_layer          = -1;
                for (int l = 0; l < group.nlayers(); ++l)
                    if (function.getName().str()
                        == fmtformat("osl_layer_group_hart_test_group_name_{}",
                                     group.layer(l)->layername()))
                        for (const auto& candidate : group.layer(l)->symbols())
                            if (candidate.name() == names[slot]) {
                                symbol       = &candidate;
                                symbol_layer = l;
                            }
                OIIO_CHECK_ASSERT(symbol);
                if (!symbol)
                    continue;
                const size_t value_size = types[slot].size();
                OIIO_CHECK_ASSERT(symbol_offset != offsets[slot]);
                const size_t cache_size = value_size
                                          * (derivatives[slot] ? 3 : 1);
                if (connected && names[slot] == "shared") {
                    const bool producer = function.getName().contains(
                        "_name_producer");
                    OIIO_CHECK_EQUAL(symbol->has_derivs(), !producer);
                    OIIO_CHECK_EQUAL(value_size, 4);
                    OIIO_CHECK_EQUAL(cache_size, 12);
                    OIIO_CHECK_ASSERT(
                        integer(hit_copy->getLength(), producer ? 4 : 12));
                }
                OIIO_CHECK_ASSERT(
                    integer(hit_copy->getLength(),
                            std::min(size_t(symbol->derivsize()), cache_size)));
                const auto* default_branch = llvm::dyn_cast<llvm::BranchInst>(
                    hit_guard->getSuccessor(1)->getTerminator());
                const auto* miss_test
                    = default_branch && default_branch->isConditional()
                          ? llvm::dyn_cast<llvm::ICmpInst>(
                                default_branch->getCondition())
                          : nullptr;
                bool only_on_miss = false;
                if (miss_test
                    && miss_test->getPredicate() == llvm::CmpInst::ICMP_EQ)
                    for (unsigned a = 0; a < 2; ++a)
                        only_on_miss |= integer(miss_test->getOperand(a), 0)
                                        && peel(miss_test->getOperand(1 - a))
                                               == hit_condition;
                OIIO_CHECK_ASSERT(only_on_miss);
                if (!only_on_miss)
                    continue;
                const auto* missing = default_branch->getSuccessor(0);
                OIIO_CHECK_EQUAL(missing->getSinglePredecessor(),
                                 default_branch->getParent());
                OIIO_CHECK_ASSERT(
                    !dominators.dominates(missing,
                                          default_branch->getSuccessor(1)));
                bool default_write = false, scalar_default = false;
                bool zero_derivatives    = false;
                bool interactive_default = false;
                for (const auto& candidate : function) {
                    if (!dominators.dominates(missing, &candidate))
                        continue;
                    for (const auto& operation : candidate) {
                        if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(
                                &operation))
                            if (at(store->getPointerOperand(), function,
                                   symbol_offset)) {
                                default_write = true;
                                if (names[slot] == "shared"
                                    || names[slot] == "gain") {
                                    const auto* value
                                        = llvm::dyn_cast<llvm::ConstantFP>(
                                            store->getValueOperand());
                                    const float expected
                                        = names[slot] == "gain" ? 1.25f
                                          : function.getName().contains(
                                                "_name_producer")
                                              ? 2.0f
                                              : 5.0f;
                                    scalar_default
                                        |= value
                                           && value->getValueAPF()
                                                      .convertToFloat()
                                                  == expected;
                                } else if (names[slot] == "tag") {
                                    const ustring expected(
                                        function.getName().contains(
                                            "_name_producer")
                                            ? "producer-default"
                                            : "consumer-default");
                                    scalar_default
                                        |= integer(store->getValueOperand(),
                                                   expected.hash());
                                }
                            }
                        if (const auto* copy = llvm::dyn_cast<llvm::MemCpyInst>(
                                &operation)) {
                            if (at(copy->getRawDest(), function,
                                   symbol_offset)) {
                                default_write = true;
                                if (symbol->interactive()) {
                                    int64_t offset = -1;
                                    OIIO_CHECK_ASSERT(interactive_address(
                                        copy->getRawSource(),
                                        function.getArg(5), layout, offset));
                                    OIIO_CHECK_EQUAL(
                                        offset,
                                        group.interactive_param_offset(
                                            symbol_layer, symbol->name()));
                                    OIIO_CHECK_ASSERT(
                                        integer(copy->getLength(), value_size));
                                    interactive_default = true;
                                }
                            }
                        }
                        if (const auto* clear
                            = llvm::dyn_cast<llvm::MemSetInst>(&operation))
                            zero_derivatives |= at(clear->getRawDest(),
                                                   function,
                                                   symbol_offset + value_size)
                                                && integer(clear->getValue(), 0)
                                                && integer(clear->getLength(),
                                                           2 * value_size);
                    }
                }
                OIIO_CHECK_ASSERT(default_write);
                if (symbol->interactive()) {
                    OIIO_CHECK_ASSERT(interactive_default);
                    if (symbol->has_derivs())
                        OIIO_CHECK_ASSERT(zero_derivatives);
                } else if (names[slot] == "shared" || names[slot] == "gain"
                           || names[slot] == "tag") {
                    OIIO_CHECK_ASSERT(scalar_default);
                    if (symbol->has_derivs())
                        OIIO_CHECK_ASSERT(zero_derivatives);
                }
            }
    }
    OIIO_CHECK_EQUAL(resets, 1);
    OIIO_CHECK_EQUAL(calls, missing_spec ? 0 : requests.size());
    if (missing_spec) {
        int defaults = 0;
        for (const auto& function : module) {
            if (function.getName().find("osl_layer_group_") != 0)
                continue;
            for (const auto& block : function)
                for (const auto& inst : block) {
                    const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst);
                    const auto* value = store
                                            ? llvm::dyn_cast<llvm::ConstantFP>(
                                                  store->getValueOperand())
                                            : nullptr;
                    if (!value
                        || value->getValueAPF().convertToFloat() != 1.75f)
                        continue;
                    ++defaults;
                    int64_t offset = 0;
                    OIIO_CHECK_EQUAL(llvm::GetPointerBaseWithConstantOffset(
                                         store->getPointerOperand(), offset,
                                         layout),
                                     function.getArg(1));
                    OIIO_CHECK_ASSERT(offset != offsets[0]);
                    bool zero_derivatives = false;
                    for (const auto& operation : block)
                        if (const auto* clear
                            = llvm::dyn_cast<llvm::MemSetInst>(&operation))
                            zero_derivatives
                                |= at(clear->getRawDest(), function, offset + 4)
                                   && integer(clear->getValue(), 0)
                                   && integer(clear->getLength(), 8);
                    OIIO_CHECK_ASSERT(zero_derivatives);
                }
        }
        OIIO_CHECK_EQUAL(defaults, 1);
    }
    if (connected)
        for (int i = 0; i < count; ++i) {
            unsigned expected = 0;
            for (int l = 0; l < group.nlayers(); ++l) {
                const auto* instance = group.layer(l);
                if (instance->unused())
                    continue;
                for (const auto& symbol : instance->symbols())
                    if (symbol.name() == names[i] && symbol.interpolated()
                        && !symbol.connected()
                        && symbol.typespec().simpletype() == types[i])
                        expected |= instance->layername() == "producer" ? 1u
                                                                        : 2u;
            }
            if (names[i] == "shared" || names[i] == "tag")
                OIIO_CHECK_EQUAL(expected, 3u);
            OIIO_CHECK_EQUAL(slot_layers[i], expected);
        }
    std::vector<int> requested_sites(count, 0);
    for (const auto& request : requests) {
        bool matched = false;
        for (int i = 0; i < count; ++i)
            if (names[i] == request.name && types[i] == request.type) {
                matched = true;
                ++requested_sites[i];
                OIIO_CHECK_EQUAL(request.derivatives, derivatives[i] != 0);
            }
        OIIO_CHECK_ASSERT(matched);
    }
    for (int i = 0; i < count; ++i)
        OIIO_CHECK_EQUAL(slot_calls[i], missing_spec ? 0 : requested_sites[i]);
    return true;
}



bool
check_userdata_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_userdata("
        "int count=7 [[int interpolated=1]], "
        "float gain=1.25 [[int interpolated=1]], "
        "color tint=color(0.25,0.5,0.75) [[int interpolated=1]], "
        "matrix basis=1 [[int interpolated=1]], "
        "int indices[3]={7,11,13} [[int interpolated=1]], "
        "float weights[]={0.125,0.25} [[int interpolated=1]], "
        "string label=\"alpha\" [[int interpolated=1]], "
        "string tags[2]={\"beta\",\"\"} [[int interpolated=1]], "
        "string choices[]={\"alpha\",\"\"} [[int interpolated=1]], "
        "float fallback=u+2*v [[int interpolated=1]], output color Cout=0) { "
        "color dx=Dx(tint), dy=Dy(tint); "
        "float value=count+gain+tint[0]+tint[1]+tint[2]+basis[0][1]"
        "+basis[3][2]+indices[0]+indices[1]+indices[2]+weights[0]+weights[1]"
        "+arraylength(weights)+(label==\"alpha\")+2*(tags[0]==label)"
        "+4*(tags[1]==\"\")+8*(choices[0]==label)+16*(choices[1]==\"\")"
        "+arraylength(choices)+fallback; "
        "Cout=color(value,Dx(gain)+dx[0],Dy(gain)+dy[1]); }",
        "shader hart_userdata_producer("
        "float shared=2 [[int interpolated=1]], "
        "string tag=\"producer-default\" [[int interpolated=1]], "
        "output float value=0) { "
        "if(u>0.25) value=shared+(tag==\"hit\"); else value=u; }",
        "shader hart_userdata_consumer("
        "float value=0 [[int interpolated=1]], "
        "float shared=5 [[int interpolated=1]], "
        "string tag=\"consumer-default\" [[int interpolated=1]], "
        "output color Cout=0) { "
        "if(v>0.25) Cout=color(value+shared+(tag==\"hit\"),"
        "Dx(shared),Dy(shared)); else Cout=0; }",
        "shader hart_userdata_unused(float never=13 [[int interpolated=1]], "
        "output color Cout=0) { Cout=color(never*u); }",
        "shader hart_userdata_default("
        "float fallback=1.75 [[int interpolated=1]], output color Cout=0) "
        "{ Cout=color(fallback,Dx(fallback),Dy(fallback)); }",
        "shader hart_userdata_conflict("
        "float gain=1 [[int interpolated=1,int interactive=1]], "
        "output color Cout=0) { Cout=color(gain*u); }",
        "shader hart_userdata_closure("
        "closure color c=0 [[int interpolated=1]], output color Cout=0) "
        "{ Cout=color(u,v,0); }",
    };
    std::string oso[std::size(sources)];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, stdosl))
            return false;
    }
    const struct {
        int osl, llvm, length;
        bool local, lazy, connected, missing;
    } variants[] = {
        { 0, 10, 2, false, true, false, false },
        { 2, 10, 5, true, false, false, false },
        { 2, 3, 3, true, true, false, false },
        { 0, 10, 0, false, true, true, false },
        { 0, 10, 0, true, false, true, false },
        { 2, 3, 0, true, true, true, false },
        { 0, 10, 0, false, true, false, true },
        { 2, 3, 0, true, false, false, true },
    };
    for (const auto& variant : variants) {
        HartUserdataServices renderer;
        renderer.missing_spec = variant.missing;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", variant.osl);
        ss.attribute("llvm_optimize", variant.llvm);
        ss.attribute("lazy_userdata", int(variant.lazy));
        ss.attribute("lazy_unconnected", 1);
        ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
        ShaderGroupRef group;
        if (variant.connected) {
            for (int i : { 1, 2, 3 })
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader(fmtformat("userdata{}", i),
                                                oso[i]));
            group = ss.ShaderGroupBegin("hart_test_group");
            OIIO_CHECK_ASSERT(ss.Shader("surface", "userdata3", "unused"));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "userdata1", "producer"));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "userdata2", "consumer"));
            OIIO_CHECK_ASSERT(
                ss.ConnectShaders("producer", "value", "consumer", "value"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        } else {
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_test",
                                            oso[variant.missing ? 4 : 0]));
            group = ss.ShaderGroupBegin("hart_test_group");
            if (!variant.missing && variant.length != 2) {
                const float weights[]   = { 1, 2, 3, 4, 5 };
                const ustring choices[] = { ustring("chosen"), ustring(""),
                                            ustring("alpha"), ustring("beta"),
                                            ustring("") };
                OIIO_CHECK_ASSERT(
                    ss.Parameter("weights",
                                 TypeDesc(TypeDesc::FLOAT, variant.length),
                                 weights, ParamHints::interpolated));
                OIIO_CHECK_ASSERT(
                    ss.Parameter("choices",
                                 TypeDesc(TypeDesc::STRING, variant.length),
                                 choices, ParamHints::interpolated));
            }
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        }
        const SymLocationDesc output(variant.connected ? "consumer.Cout"
                                                       : "layer0.Cout",
                                     TypeColor, false, SymArena::Outputs, 0,
                                     12);
        ss.add_symlocs(group.get(), { &output, 1 });
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr,
                  "Userdata (OSL {}, LLVM {}, lazy {}, connected {}, "
                  "missing {}):\n{}",
                  variant.osl, variant.llvm, variant.lazy, variant.connected,
                  variant.missing, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
        const std::vector<string_view> expected_names
            = variant.connected ? std::vector<string_view> { "shared", "tag" }
              : variant.missing
                  ? std::vector<string_view> { "fallback" }
                  : std::vector<string_view> { "count",   "gain",    "tint",
                                               "basis",   "indices", "weights",
                                               "label",   "tags",    "choices",
                                               "fallback" };
        // Lazy value/Dx/Dy uses can have separate guarded binding sites.
        // Require complete name coverage; the IR check accounts for every site.
        std::vector<ustring> requested_names;
        for (const auto& request : renderer.requests) {
            OIIO_CHECK_ASSERT(request.name != "never"
                              && request.name != "value");
            OIIO_CHECK_ASSERT(std::find(expected_names.begin(),
                                        expected_names.end(),
                                        string_view(request.name.c_str()))
                              != expected_names.end());
            if (std::find(requested_names.begin(), requested_names.end(),
                          request.name)
                == requested_names.end())
                requested_names.push_back(request.name);
            if (request.name == "weights")
                OIIO_CHECK_ASSERT(request.type
                                  == TypeDesc(TypeDesc::FLOAT, variant.length));
            if (request.name == "choices")
                OIIO_CHECK_ASSERT(
                    request.type == TypeDesc(TypeDesc::STRING, variant.length));
            if (request.name == "shared" || request.name == "gain"
                || request.name == "tint")
                OIIO_CHECK_ASSERT(request.derivatives);
        }
        OIIO_CHECK_EQUAL(requested_names.size(), expected_names.size());
        check_module(
            ss, *group, arch,
            variant.missing
                ? std::initializer_list<string_view> { }
                : std::initializer_list<string_view> { "osl_hart_get_userdata",
                                                       "rs_hart_get_userdata" },
            variant.llvm, variant.connected, variant.connected, false,
            variant.connected ? 2 : 0, false, true);
        if (variant.llvm == 10)
            OIIO_CHECK_ASSERT(check_userdata_ir(ss, *group, renderer.requests,
                                                variant.connected, variant.lazy,
                                                variant.missing));
        int allocated = -1, group_size = 0;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", group_size));
        OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
        OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
        const void* artifact = nullptr;
        uint64_t size        = 0;
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                          TypeDesc::PTR, &artifact));
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                          TypeUInt64, &size));
        if (!artifact || !size)
            return false;
        const std::string original(static_cast<const char*>(artifact), size);
        const size_t requests = renderer.requests.size();
        renderer.missing_spec = !renderer.missing_spec;
        ss.optimize_group(group.get(), nullptr);
        const void* current   = nullptr;
        uint64_t current_size = 0;
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                          TypeDesc::PTR, &current));
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                          TypeUInt64, &current_size));
        OIIO_CHECK_EQUAL(current, artifact);
        OIIO_CHECK_EQUAL(current_size, size);
        if (current && current_size == size)
            OIIO_CHECK_ASSERT(
                string_view(static_cast<const char*>(current), current_size)
                == string_view(original));
        OIIO_CHECK_EQUAL(renderer.requests.size(), requests);
        OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
        OIIO_CHECK_EQUAL(errors.errors, 0);
    }
    for (int failure = 0; failure < 6; ++failure) {
        HartUserdataServices renderer;
        renderer.userdata          = failure != 0;
        renderer.getter            = failure != 5;
        renderer.arena.interactive = failure != 1 && failure != 2;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        const int source = failure == 0 || failure == 5 ? 0
                           : failure <= 2               ? 5
                           : failure == 3               ? 6
                                                        : 4;
        auto group       = make_group(ss, oso[source]);
        if (failure == 2) {
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("userdata_valid", oso[4]));
            group = ss.ShaderGroupBegin("hart_test_group");
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "unused"));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "userdata_valid", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("layer0.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(group.get(), { &output, 1 });
        }
        if (failure == 4) {
            const SymLocationDesc input("layer0.fallback", TypeFloat, false,
                                        SymArena::UserData, 0, 4);
            ss.add_symlocs(group.get(), { &input, 1 });
        }
        check_rejected_group(ss, *group, errors,
                             failure == 4                   ? "pre-placement"
                             : failure == 1 || failure == 2 ? "interactive"
                                                            : "interpolated");
        if (failure == 1 || failure == 2)
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, "interactive"));
        if (failure == 3)
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, "closure"));
        OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
        OIIO_CHECK_ASSERT(renderer.requests.empty());
    }
    return true;
}



bool
check_interpolated_interactive_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_combined_producer("
        "float shared=2 [[int interpolated=1]], "
        "int count=3 [[int interpolated=1,int interactive=1]], "
        "color tint=color(.125,.25,.5) [[int interpolated=1,int interactive=1]], "
        "float weights[2]={1,2} [[int interpolated=1,int interactive=1]], "
        "matrix basis=1 [[int interpolated=1,int interactive=1]], "
        "output float value=0) { "
        "if(u>.25) value=shared+count+tint[0]+tint[1]+tint[2]"
        "+Dx(tint[0])+Dy(tint[1])+weights[0]+weights[1]"
        "+Dx(weights[0])+Dy(weights[1])+basis[0][0]; else value=u; }",
        "shader hart_combined_consumer(float value=0, "
        "float shared=5 [[int interpolated=1,int interactive=1]], "
        "output color Cout=0) { "
        "if(v>.25) Cout=color(value+shared,Dx(shared),Dy(shared)); "
        "else Cout=0; }",
        "shader hart_combined_initializer("
        "float gain=u [[int interpolated=1,int interactive=1]], "
        "output color Cout=0) { Cout=color(gain,Dx(gain),Dy(gain)); }",
    };
    std::string oso[std::size(sources)];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, stdosl))
            return false;
    }
    const struct {
        int osl, llvm;
        bool lazy, local;
    } variants[] = { { 0, 10, true, false },
                     { 0, 10, false, false },
                     { 2, 10, true, true },
                     { 2, 3, false, true } };
    for (const auto& variant : variants) {
        HartUserdataServices renderer;
        auto check = [&]() {
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", variant.osl);
            ss.attribute("llvm_optimize", variant.llvm);
            ss.attribute("lazy_userdata", int(variant.lazy));
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
            ss.attribute("error_repeats", 1);
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_producer", oso[0]));
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_consumer", oso[1]));
            const float shared = 2, other_shared = 5, changed_shared = 11;
            const int count = 3, changed_count = 17;
            const float tint[]         = { .125f, .25f, .5f };
            const float changed_tint[] = { 2, 4, 8 };
            const float weights[] = { 1, 2 }, changed_weights[] = { 4, 9 };
            const float basis[]         = { 1, 0, 0, 0, 0, 1, 0, 0,
                                            0, 0, 1, 0, 0, 0, 0, 1 };
            const float changed_basis[] = { 2, 0, 0, 0, 0, 3, 0, 0,
                                            0, 0, 4, 0, 0, 0, 0, 5 };
            struct {
                int layer;
                InteractiveField field;
                const void* initial;
                const void* changed;
            } fields[] = {
                { 0, { "shared", TypeFloat }, &shared, &changed_shared },
                { 0, { "count", TypeInt }, &count, &changed_count },
                { 0, { "tint", TypeColor }, tint, changed_tint },
                { 0,
                  { "weights", TypeDesc(TypeDesc::FLOAT, 2) },
                  weights,
                  changed_weights },
                { 0, { "basis", TypeMatrix }, basis, changed_basis },
                { 1, { "shared", TypeFloat }, &other_shared, &changed_shared },
            };
            auto group = ss.ShaderGroupBegin("hart_test_group");
            // The master supplies interpolation, the instance adds interaction.
            OIIO_CHECK_ASSERT(ss.Parameter("shared", TypeFloat, &shared,
                                           ParamHints::interactive));
            OIIO_CHECK_ASSERT(
                ss.Shader("surface", "hart_producer", "producer"));
            OIIO_CHECK_ASSERT(
                ss.Shader("surface", "hart_consumer", "consumer"));
            OIIO_CHECK_ASSERT(
                ss.ConnectShaders("producer", "value", "consumer", "value"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("consumer.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(group.get(), { &output, 1 });
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "Combined userdata compilation:\n{}",
                      errors.messages);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
            OIIO_CHECK_EQUAL(renderer.arena.allocations, 1);
            OIIO_CHECK_EQUAL(renderer.arena.copies.size(), 1);
            check_module(ss, *group, arch,
                         { "osl_hart_get_userdata", "rs_hart_get_userdata" },
                         variant.llvm, true, true, false, 2, false, true);
            if (variant.llvm == 10)
                OIIO_CHECK_ASSERT(check_userdata_ir(ss, *group,
                                                    renderer.requests, true,
                                                    variant.lazy, false));
            int allocated = -1, group_size = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "llvm_groupdata_size",
                                              group_size));
            OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
            OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
            size_t extent = 0;
            for (auto& entry : fields) {
                auto& field = entry.field;
                const int offset
                    = group->interactive_param_offset(entry.layer,
                                                      ustring(field.name));
                OIIO_CHECK_ASSERT(offset >= 0);
                if (offset < 0)
                    return false;
                field.offset = size_t(offset);
                for (const auto& symbol : group->layer(entry.layer)->symbols())
                    if (symbol.name() == field.name) {
                        OIIO_CHECK_ASSERT(
                            symbol.interpolated() && symbol.interactive()
                            && !symbol.lockgeom() && !symbol.is_constant());
                        OIIO_CHECK_ASSERT(symbol.typespec().simpletype()
                                          == field.type);
                        field.extent = symbol.derivsize();
                    }
                OIIO_CHECK_ASSERT(field.extent >= field.type.size());
                extent = std::max(extent, field.offset + field.extent);
            }
            OIIO_CHECK_ASSERT(fields[0].field.offset != fields[5].field.offset);
            OIIO_CHECK_EQUAL(renderer.arena.requested_size, extent);
            if (!extent || extent > 4096
                || renderer.arena.requested_size != extent)
                return false;
            std::vector<uint8_t> expected(extent, 0);
            for (const auto& entry : fields)
                std::memcpy(expected.data() + entry.field.offset, entry.initial,
                            entry.field.type.size());
            const void* artifact = nullptr;
            uint64_t size        = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR, &artifact));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                              TypeUInt64, &size));
            if (!artifact || !size)
                return false;
            const std::string original(static_cast<const char*>(artifact),
                                       size);
            const size_t requests = renderer.requests.size();
            auto unchanged        = [&]() {
                const void* current   = nullptr;
                uint64_t current_size = 0;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                                  TypeDesc::PTR, &current));
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "hart_bitcode_size",
                                                  TypeUInt64, &current_size));
                OIIO_CHECK_EQUAL(current, artifact);
                OIIO_CHECK_EQUAL(current_size, size);
                if (current && current_size == size)
                    OIIO_CHECK_ASSERT(
                        string_view(static_cast<const char*>(current), size)
                        == string_view(original));
                OIIO_CHECK_EQUAL(renderer.requests.size(), requests);
                OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
                void* host = nullptr;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "interactive_params",
                                                  TypeDesc::PTR, &host));
                OIIO_CHECK_ASSERT(host);
                if (host)
                    OIIO_CHECK_EQUAL(std::memcmp(host, expected.data(), extent),
                                     0);
            };
            auto device_matches = [&]() {
                void* device = nullptr;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "device_interactive_params",
                                                  TypeDesc::PTR, &device));
                OIIO_CHECK_EQUAL(device, renderer.arena.storage.get());
                OIIO_CHECK_ASSERT(device);
                if (device)
                    OIIO_CHECK_EQUAL(std::memcmp(device, expected.data(),
                                                 extent),
                                     0);
                unchanged();
            };
            auto update = [&](size_t index, const void* value) {
                const auto& entry = fields[index];
                OIIO_CHECK_ASSERT(
                    ss.ReParameter(*group,
                                   entry.layer ? "consumer" : "producer",
                                   entry.field.name, entry.field.type, value));
                std::memcpy(expected.data() + entry.field.offset, value,
                            entry.field.type.size());
                device_matches();
            };
            device_matches();
            for (size_t i = 0; i < std::size(fields); ++i)
                update(i, fields[i].changed);
            for (size_t i = 0; i < std::size(fields); ++i)
                update(i, fields[i].initial);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            // Damage a default upload, then repair it by updating another layer.
            const int before         = errors.errors;
            renderer.arena.fail_copy = true;
            OIIO_CHECK_ASSERT(!ss.ReParameter(*group, "producer", "weights",
                                              fields[3].field.type,
                                              changed_weights));
            OIIO_CHECK_EQUAL(errors.errors, before + 1);
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error,
                                        "failed to upload interactive"));
            OIIO_CHECK_ASSERT(std::memcmp(renderer.arena.storage.get(),
                                          expected.data(), extent)
                              != 0);
            unchanged();
            void* invalid = &renderer;
            OIIO_CHECK_ASSERT(!ss.getattribute(group.get(),
                                               "device_interactive_params",
                                               TypeDesc::PTR, &invalid));
            OIIO_CHECK_ASSERT(!invalid);
            renderer.arena.fail_copy = false;
            update(5, &changed_shared);
            OIIO_CHECK_EQUAL(renderer.arena.copies.back().offset, 0);
            OIIO_CHECK_EQUAL(renderer.arena.copies.back().size, extent);
            update(5, &other_shared);
            ss.optimize_group(group.get(), nullptr);
            device_matches();
            OIIO_CHECK_EQUAL(errors.errors, before + 2);
            OIIO_CHECK_EQUAL(renderer.arena.allocations, 1);
            return true;
        };
        const int before = unit_test_failures;
        OIIO_CHECK_ASSERT(check());
        OIIO_CHECK_EQUAL(renderer.arena.frees,
                         renderer.arena.successful_allocations);
        OIIO_CHECK_ASSERT(!renderer.arena.storage);
        if (before != unit_test_failures)
            print(stderr,
                  "Combined checks failed (OSL {}, LLVM {}, lazy {}, "
                  "local {})\n",
                  variant.osl, variant.llvm, variant.lazy, variant.local);
    }
    for (int optimize : { 0, 2 }) {
        HartUserdataServices renderer;
        {
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", optimize);
            ss.attribute("llvm_optimize", 10);
            OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_test", oso[2]));
            auto group       = ss.ShaderGroupBegin("hart_test_group");
            const float gain = 2.5f;
            OIIO_CHECK_ASSERT(ss.Parameter("gain", TypeFloat, &gain,
                                           ParamHints::interactive));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("layer0.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(group.get(), { &output, 1 });
            ss.optimize_group(group.get(), nullptr);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch,
                         { "osl_hart_get_userdata", "rs_hart_get_userdata" },
                         10, false, false, false, 0, false, true);
            OIIO_CHECK_ASSERT(check_userdata_ir(ss, *group, renderer.requests,
                                                false, false, false));
        }
        OIIO_CHECK_EQUAL(renderer.arena.frees,
                         renderer.arena.successful_allocations);
        OIIO_CHECK_ASSERT(!renderer.arena.storage);
    }
    const struct {
        string_view declaration, error;
    } rejected[] = {
        { "string value=\"x\" [[int interpolated=1,int interactive=1]]",
          "must be a numeric input" },
        { "output float value=1 [[int interpolated=1,int interactive=1]]",
          "must be a numeric input" },
        { "closure color value=0 [[int interpolated=1,int interactive=1]]",
          "cannot be closure-based" },
        { "float value=u [[int interpolated=1,int interactive=1]]",
          "requires a constant default or an instance value" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat("shader hart_combined_bad({},output color Cout=0)"
                          "{{ Cout=color(u,v,0); }}",
                          test.declaration),
                bytecode, { }, stdosl))
            return false;
        HartUserdataServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
        OIIO_CHECK_EQUAL(renderer.arena.allocations, 0);
        OIIO_CHECK_ASSERT(renderer.requests.empty());
    }
    return true;
}



struct AttributeExpectation {
    const char* object;
    const char* name;
    TypeDesc type;
    bool derivatives;
    int index          = -1;
    bool dynamic_index = false;
};



bool
check_attribute_ir(ShadingSystem& ss, ShaderGroup& group,
                   cspan<AttributeExpectation> expected, int optimize)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_attributes"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module       = **parsed;
    const auto& layout = module.getDataLayout();
    for (const char* name : { "osl_get_attribute", "rs_get_attribute" }) {
        const auto* legacy = module.getFunction(name);
        OIIO_CHECK_ASSERT(!legacy || legacy->use_empty());
    }
    const auto* wrapper  = module.getFunction("osl_hart_get_attribute");
    const auto* callback = module.getFunction("rs_hart_get_attribute");
    if (expected.empty()) {
        OIIO_CHECK_ASSERT(!wrapper || wrapper->use_empty());
        OIIO_CHECK_ASSERT(!callback || callback->use_empty());
        return true;
    }
    OIIO_CHECK_ASSERT(callback && !callback->use_empty()
                      && callback->isDeclaration());
    if (optimize != 10)
        return true;
    for (const auto* function : { wrapper, callback }) {
        OIIO_CHECK_ASSERT(function && !function->use_empty());
        if (!function)
            return false;
        OIIO_CHECK_EQUAL(function->isDeclaration(), function == callback);
        OIIO_CHECK_ASSERT(!function->isVarArg()
                          && function->getReturnType()->isIntegerTy(1));
        OIIO_CHECK_EQUAL(function->arg_size(), 8);
        if (function->arg_size() != 8)
            return false;
        for (const auto& arg : function->args()) {
            const unsigned i = arg.getArgNo();
            OIIO_CHECK_ASSERT(
                i == 0 || i == 7
                    ? arg.getType()->isPointerTy()
                          && arg.getType()->getPointerAddressSpace() == 0
                    : arg.getType()->isIntegerTy(i == 1 || i == 6 ? 32
                                                 : i == 5         ? 1
                                                                  : 64));
        }
    }
    int forwards = 0;
    for (const auto& block : *wrapper)
        for (const auto& inst : block)
            if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                if (call->getCalledFunction() == callback) {
                    ++forwards;
                    OIIO_CHECK_EQUAL(call->arg_size(), 8);
                    OIIO_CHECK_EQUAL(call->getCallingConv(),
                                     callback->getCallingConv());
                    for (unsigned i = 0; i < call->arg_size() && i < 8; ++i)
                        OIIO_CHECK_EQUAL(call->getArgOperand(i),
                                         wrapper->getArg(i));
                    const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(
                        block.getTerminator());
                    OIIO_CHECK_ASSERT(ret && ret->getReturnValue() == call);
                }
    OIIO_CHECK_EQUAL(forwards, 1);
    auto hash_matches = [](const llvm::Value* value, const char* text) {
        const auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value);
        return text
                   ? constant
                         && constant->getZExtValue() == ustringhash(text).hash()
                   : !llvm::isa<llvm::Constant>(value);
    };
    std::vector<int> seen(expected.size(), 0);
    int calls = 0;
    for (const auto* user : wrapper->users()) {
        const auto* call = llvm::dyn_cast<llvm::CallBase>(user);
        if (!call || call->getCalledFunction() != wrapper)
            continue;
        const auto* function = call->getFunction();
        if (function->getName().find("osl_layer_group_") != 0)
            continue;
        ++calls;
        OIIO_CHECK_EQUAL(function->arg_size(), 6);
        OIIO_CHECK_EQUAL(call->arg_size(), 8);
        if (function->arg_size() != 6 || call->arg_size() != 8)
            continue;
        OIIO_CHECK_EQUAL(call->getCallingConv(), wrapper->getCallingConv());
        OIIO_CHECK_EQUAL(call->getArgOperand(0)->stripPointerCasts(),
                         function->getArg(0));
        OIIO_CHECK_EQUAL(call->getArgOperand(1), function->getArg(4));
        const auto* type = llvm::dyn_cast<llvm::ConstantInt>(
            call->getArgOperand(4));
        const auto* derivatives = llvm::dyn_cast<llvm::ConstantInt>(
            call->getArgOperand(5));
        const auto* index = llvm::dyn_cast<llvm::ConstantInt>(
            call->getArgOperand(6));
        OIIO_CHECK_ASSERT(type && derivatives);
        if (!type || !derivatives)
            continue;
        int matches = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const auto& test      = expected[i];
            uint64_t encoded_type = 0;
            static_assert(sizeof(TypeDesc) == sizeof(encoded_type));
            std::memcpy(&encoded_type, &test.type, sizeof(encoded_type));
            if (type->getZExtValue() != encoded_type
                || !hash_matches(call->getArgOperand(2), test.object)
                || !hash_matches(call->getArgOperand(3), test.name)
                || (test.dynamic_index
                        ? index != nullptr
                        : !index || index->getSExtValue() != test.index))
                continue;
            ++matches;
            ++seen[i];
            OIIO_CHECK_EQUAL(derivatives->getZExtValue(),
                             uint64_t(test.derivatives));
            int64_t offset         = 0;
            const auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(
                llvm::GetPointerBaseWithConstantOffset(call->getArgOperand(7),
                                                       offset, layout));
            OIIO_CHECK_ASSERT(allocation && offset >= 0);
            if (!allocation || offset < 0)
                continue;
            OIIO_CHECK_EQUAL(allocation->getAddressSpace(), 5);
            const auto* count = llvm::dyn_cast<llvm::ConstantInt>(
                allocation->getArraySize());
            OIIO_CHECK_ASSERT(count);
            if (count) {
                const uint64_t available
                    = layout.getTypeAllocSize(allocation->getAllocatedType())
                          .getFixedValue()
                      * count->getZExtValue();
                const uint64_t required = test.type.size()
                                          * (test.derivatives ? 3 : 1);
                OIIO_CHECK_ASSERT(uint64_t(offset) <= available
                                  && required <= available - uint64_t(offset));
            }
            auto typed = [&](const auto& self, llvm::Type* storage) -> bool {
                if (auto* array = llvm::dyn_cast<llvm::ArrayType>(storage))
                    return self(self, array->getElementType());
                if (auto* record = llvm::dyn_cast<llvm::StructType>(storage)) {
                    if (record->isOpaque())
                        return false;
                    for (auto* field : record->elements())
                        if (!self(self, field))
                            return false;
                    return true;
                }
                return test.type.basetype == TypeDesc::FLOAT
                           ? storage->isFloatTy()
                           : storage->isIntegerTy(
                                 test.type.basetype == TypeDesc::STRING ? 64
                                                                        : 32);
            };
            OIIO_CHECK_ASSERT(typed(typed, allocation->getAllocatedType()));
        }
        OIIO_CHECK_EQUAL(matches, 1);
        bool integer_status = false;
        for (const auto* user : call->users())
            if (const auto* extend = llvm::dyn_cast<llvm::ZExtInst>(user))
                integer_status |= extend->getType()->isIntegerTy(32)
                                  && !extend->use_empty();
        OIIO_CHECK_ASSERT(integer_status);
    }
    OIIO_CHECK_EQUAL(calls, expected.size());
    for (int count : seen)
        OIIO_CHECK_EQUAL(count, 1);
    return true;
}



bool
check_attribute_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_attribute_types(output color Cout=0) { "
        "float f=1, fa[2]={2,3}; int i=4, ia[3]={5,6,7}; "
        "color c=color(8); string s=\"old\", sa[2]={\"left\",\"right\"}; "
        "matrix m=1; vector va[2]={vector(9),vector(10)}; "
        "int ok=getattribute(\"attr:f\",f); "
        "ok+=getattribute(\"attr:fa\",fa); "
        "ok+=getattribute(\"attr:i\",2,i); "
        "ok+=getattribute(\"attr:ia\",3,ia); "
        "ok+=getattribute(\"camera\",\"attr:c\",c); "
        "ok+=getattribute(\"scene\",\"attr:sa\",sa); "
        "ok+=getattribute(\"camera\",\"attr:m\",4,m); "
        "ok+=getattribute(\"scene\",\"attr:va\",5,va); "
        "ok+=getattribute(\"attr:s\",s); "
        "Cout=c+color(f+fa[0]+fa[1]+i+ia[0]+ia[1]+ia[2]+ok,"
        "m[0][0]+m[3][3]+(s==\"new\")+(sa[0]==sa[1]),"
        "va[0][0]+va[1][2]); "
        "Cout+=Dx(c)+Dy(c)+color(Dx(f)+Dy(f)+Dx(fa[0])+Dy(fa[1]))"
        "+color(Dx(va[0])+Dy(va[1])); }",
        "shader hart_attribute_dynamic(output color Cout=0) { "
        "string name=u>v?\"attr:left\":\"attr:right\"; "
        "string object=u>0.5?\"camera\":\"scene\"; int index=int(3*u)-1; "
        "float a=1,b=2,c=3; int ok=getattribute(name,a); "
        "ok+=getattribute(name,index,b); "
        "ok+=getattribute(object,name,index,c); "
        "Cout=color(a+b+c+ok,Dx(a)+Dx(b)+Dx(c),Dy(a)+Dy(b)+Dy(c)); }",
        "shader hart_test(output color Cout=0) { "
        "int version=0; string shader_name=\"\",layer_name=\"\",group_name=\"\"; "
        "int ok=getattribute(\"osl:version\",version); "
        "ok+=getattribute(\"shader:shadername\",shader_name); "
        "ok+=getattribute(\"shader:layername\",layer_name); "
        "ok+=getattribute(\"shader:groupname\",group_name); "
        "Cout=color(version,(shader_name==\"hart_test\")+(layer_name==\"layer0\"),"
        "(group_name==\"hart_test_group\")+ok); }",
        "shader hart_attribute_mutable(output color Cout=0) { "
        "float a=1,b=2; int ok=getattribute(\"camera\",\"mutable:value\",a); "
        "ok+=getattribute(\"camera\",\"mutable:array\",-2,b); "
        "Cout=color(a,b,ok)+color(Dx(a),Dy(b),0); }",
        "struct HartAttributeBad { float member; }; "
        "shader hart_attribute_bad(string object=\"camera\", string name=\"attr\", "
        "string names[2]={\"a\",\"b\"}, int index=1, float bad_index=0, "
        "HartAttributeBad structure={0}, closure color closure_value=0, "
        "output float value=0, output int ok=0, output color Cout=0) { "
        "ok=getattribute(object,name,1,value); "
        "Cout=color(value+bad_index+index+structure.member+ok); }",
        "shader hart_attribute_clean(output color Cout=0) { Cout=color(u,v,1); }",
    };
    std::string oso[std::size(sources)];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], {}, stdosl))
            return false;
    }
    const AttributeExpectation typed[] = {
        { "", "attr:f", TypeFloat, true },
        { "", "attr:fa", TypeDesc(TypeDesc::FLOAT, 2), true },
        { "", "attr:i", TypeInt, false, 2 },
        { "", "attr:ia", TypeDesc(TypeDesc::INT, 3), false, 3 },
        { "camera", "attr:c", TypeColor, true },
        { "scene", "attr:sa", TypeDesc(TypeDesc::STRING, 2), false },
        { "camera", "attr:m", TypeMatrix, false, 4 },
        { "scene", "attr:va",
          TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::VECTOR, 2), true,
          5 },
        { "", "attr:s", TypeString, false },
    };
    const AttributeExpectation dynamic[] = {
        { "", nullptr, TypeFloat, true },
        { "", nullptr, TypeFloat, true, 0, true },
        { nullptr, nullptr, TypeFloat, true, 0, true },
    };
    const AttributeExpectation metadata[] = {
        { "", "osl:version", TypeInt, false },
        { "", "shader:shadername", TypeString, false },
        { "", "shader:layername", TypeString, false },
        { "", "shader:groupname", TypeString, false },
    };
    const AttributeExpectation mutable_attributes[] = {
        { "camera", "mutable:value", TypeFloat, true },
        { "camera", "mutable:array", TypeFloat, true, -2 },
    };
    const cspan<AttributeExpectation> expectations[]
        = { typed, dynamic, metadata, mutable_attributes };
    const struct {
        int source, osl, llvm;
        bool local;
    } variants[] = {
        { 0, 0, 10, false }, { 0, 2, 10, true }, { 0, 2, 3, true },
        { 1, 0, 10, false }, { 1, 2, 10, true }, { 1, 2, 3, false },
        { 2, 0, 10, false }, { 2, 2, 10, true }, { 2, 2, 3, false },
        { 3, 2, 10, false },
    };
    for (const auto& variant : variants) {
        HartAttributeServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", variant.osl);
        ss.attribute("llvm_optimize", variant.llvm);
        OIIO_CHECK_ASSERT(ss.attribute("max_hart_groupdata_alloc",
                                       variant.local ? 4096 : 0));
        auto group = make_group(ss, oso[variant.source]);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "HART getattribute source {} OSL{} LLVM{}: {}\n",
                  variant.source, variant.osl, variant.llvm, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        const bool folded = variant.source == 2 && variant.osl == 2;
        check_module(
            ss, *group, arch,
            folded
                ? std::initializer_list<string_view> {}
                : std::initializer_list<string_view> { "osl_hart_get_attribute",
                                                       "rs_hart_get_attribute" },
            variant.llvm, false, variant.source == 1, false, 0, false, true);
        if (!check_attribute_ir(ss, *group,
                                folded ? cspan<AttributeExpectation> {}
                                       : expectations[variant.source],
                                variant.llvm))
            return false;
        int allocated = -1, group_size = 0;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", group_size));
        OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
        OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
        OIIO_CHECK_EQUAL(renderer.host_queries, 0);
        OIIO_CHECK_EQUAL(renderer.host_array_queries, 0);
        OIIO_CHECK_EQUAL(renderer.builders, 0);
    }
    for (bool unused : { false, true }) {
        HartAttributeServices renderer;
        renderer.attributes = false;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        auto group = make_group(ss, oso[3]);
        if (unused) {
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("attribute_clean", oso[5]));
            group = ss.ShaderGroupBegin("hart_test_group");
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "unused"));
            OIIO_CHECK_ASSERT(
                ss.Shader("surface", "attribute_clean", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("layer0.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(group.get(), { &output, 1 });
        }
        check_rejected_group(ss, *group, errors,
                             "renderer lacks HARTAttributes");
        OIIO_CHECK_EQUAL(renderer.host_queries, 0);
        OIIO_CHECK_EQUAL(renderer.host_array_queries, 0);
        OIIO_CHECK_EQUAL(renderer.builders, 0);
    }
    const auto op    = oso[4].find("\tgetattribute\t");
    const auto end   = oso[4].find('\n', op);
    const auto hints = oso[4].find('%', op);
    OIIO_CHECK_ASSERT(op != std::string::npos && end != std::string::npos
                      && hints < end);
    if (op == std::string::npos || end == std::string::npos || hints >= end)
        return false;
    std::vector<std::string> original;
    OIIO::Strutil::split(string_view(oso[4]).substr(op, hints - op), original,
                         "", -1);
    OIIO_CHECK_EQUAL(original.size(), 6);
    if (original.size() != 6)
        return false;
    const std::string original_hints = oso[4].substr(hints, end - hints);
    const std::string rw_prefix      = "%argrw{\"";
    const auto rw                    = original_hints.find(rw_prefix);
    const auto rw_end                = rw == std::string::npos
                                           ? std::string::npos
                                           : original_hints.find('"', rw + rw_prefix.size());
    OIIO_CHECK_ASSERT(rw != std::string::npos && rw_end != std::string::npos);
    if (rw == std::string::npos || rw_end == std::string::npos)
        return false;
    const struct {
        unsigned operand;
        const char* replacement;
    } replacements[] = {
        { 1, "value" },     { 2, "bad_index" }, { 2, "names" },
        { 3, "bad_index" }, { 4, "bad_index" }, { 5, "closure_value" },
        { 5, "structure" },
    };
    std::vector<std::vector<std::string>> malformed;
    for (const auto& test : replacements) {
        malformed.push_back(original);
        malformed.back()[test.operand] = test.replacement;
    }
    malformed.push_back({ original[0], original[1], original[2] });
    malformed.push_back(original);
    malformed.back().push_back("value");
    malformed.push_back({ original[0], "ok", "bad_index", "value" });
    for (unsigned operand : { 1, 5 }) {
        malformed.push_back(original);
        malformed.back()[operand] = original[4];  // The literal integer index.
    }
    for (const auto& words : malformed) {
        auto bytecode = oso[4];
        auto op_hints = original_hints;
        // Keep valid reader metadata so arity failures reach HART validation.
        std::string access(words.size() - 1, 'r');
        access.front() = access.back() = 'w';
        op_hints.replace(rw + rw_prefix.size(), rw_end - rw - rw_prefix.size(),
                         access);
        bytecode.replace(op, end - op,
                         fmtformat("\t{}\t{}", OIIO::Strutil::join(words, "\t"),
                                   op_hints));
        HartAttributeServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors,
                             "invalid getattribute operands");
        OIIO_CHECK_EQUAL(renderer.host_queries, 0);
        OIIO_CHECK_EQUAL(renderer.host_array_queries, 0);
        OIIO_CHECK_EQUAL(renderer.builders, 0);
    }
    return true;
}



bool
check_renderer_library_ir(ShadingSystem& ss, ShaderGroup& group,
                          string_view library, int optimize, bool used,
                          float bias)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_renderer_library_group"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    const auto& module   = **parsed;
    const auto* identity = module.getNamedMetadata("osl.hart.renderer_library");
    if (library.empty()) {
        OIIO_CHECK_ASSERT(!identity);
    } else {
        const auto digest = llvm::SHA256::hash(
            { reinterpret_cast<const uint8_t*>(library.data()),
              library.size() });
        OIIO_CHECK_ASSERT(identity && identity->getNumOperands() == 1);
        if (!identity || identity->getNumOperands() != 1)
            return false;
        const auto* node = identity->getOperand(0);
        OIIO_CHECK_EQUAL(node->getNumOperands(), 1);
        if (node->getNumOperands() != 1)
            return false;
        const auto* hash = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
        OIIO_CHECK_ASSERT(hash);
        if (hash) {
            OIIO_CHECK_EQUAL(hash->getString().size(), 64);
            OIIO_CHECK_EQUAL(hash->getString().str(), llvm::toHex(digest));
        }
    }
    const auto* callback = module.getFunction("rs_hart_get_userdata");
    if (!used) {
        OIIO_CHECK_ASSERT(!callback);
        return true;
    }
    for (const auto& function : module)
        if (function.getName().find("rs_hart_get_userdata") == 0
            && !function.use_empty()) {
            OIIO_CHECK_EQUAL(function.isDeclaration(), library.empty());
            if (!library.empty())
                OIIO_CHECK_ASSERT(function.hasLocalLinkage());
        }
    if (optimize != 10)
        return true;  // Optimized callbacks can be inlined or specialized.
    OIIO_CHECK_ASSERT(callback && !callback->use_empty());
    if (!callback)
        return false;
    OIIO_CHECK_ASSERT(callback->getReturnType()->isIntegerTy(1));
    OIIO_CHECK_EQUAL(callback->arg_size(), 6);
    for (const auto& arg : callback->args()) {
        const unsigned i = arg.getArgNo();
        OIIO_CHECK_ASSERT(
            i == 0 || i == 5
                ? arg.getType()->isPointerTy()
                      && arg.getType()->getPointerAddressSpace() == 0
                : arg.getType()->isIntegerTy(i == 1 ? 32 : (i == 4 ? 1 : 64)));
    }
    if (!library.empty()) {
        bool found_bias = false;
        for (const auto& block : *callback)
            for (const auto& inst : block)
                for (const auto& operand : inst.operands())
                    if (const auto* value = llvm::dyn_cast<llvm::ConstantFP>(
                            operand.get()))
                        found_bias |= value->getValueAPF().convertToFloat()
                                      == bias;
        OIIO_CHECK_ASSERT(found_bias);
    }
    return true;
}



bool
check_renderer_library_modules(string_view arch, string_view stdosl,
                               string_view filename_a, string_view filename_b,
                               string_view basic)
{
    std::string libraries[2];
    const string_view filenames[] = { filename_a, filename_b };
    const float biases[]          = { 1.25f, -2.5f };
    for (unsigned i = 0; i < 2; ++i) {
        auto file = llvm::MemoryBuffer::getFile(std::string(filenames[i]));
        if (!file) {
            print(stderr, "Cannot read renderer library '{}': {}\n",
                  filenames[i], file.getError().message());
            return false;
        }
        libraries[i] = (*file)->getBuffer().str();
        OIIO_CHECK_ASSERT(!libraries[i].empty()
                          && libraries[i].size()
                                 <= size_t(std::numeric_limits<int>::max()));
        if (libraries[i].empty()
            || libraries[i].size() > size_t(std::numeric_limits<int>::max()))
            return false;
    }
    OIIO_CHECK_ASSERT(libraries[0] != libraries[1]);
    auto digest = [](string_view bytes) {
        return llvm::toHex(llvm::SHA256::hash(
            { reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size() }));
    };
    OIIO_CHECK_ASSERT(digest(libraries[0]) != digest(libraries[1]));
    auto parse = [](string_view bytes, llvm::LLVMContext& context) {
        auto parsed = llvm::parseBitcodeFile(
            llvm::MemoryBufferRef(llvm::StringRef(bytes.data(), bytes.size()),
                                  "hart_renderer_fixture"),
            context);
        if (!parsed) {
            print(stderr, "{}\n", llvm::toString(parsed.takeError()));
            OIIO_CHECK_ASSERT(false);
            return std::unique_ptr<llvm::Module>();
        }
        return std::move(*parsed);
    };
    auto serialize = [](const llvm::Module& module) {
        std::string bytes;
        llvm::raw_string_ostream stream(bytes);
        llvm::WriteBitcodeToFile(module, stream);
        stream.flush();
        return bytes;
    };
    auto set_library = [](ShadingSystem& ss, string_view bytes) {
        OIIO_CHECK_ASSERT(bytes.size()
                          <= size_t(std::numeric_limits<int>::max()));
        return ss.attribute("lib_bitcode",
                            TypeDesc(TypeDesc::UINT8, int(bytes.size())),
                            bytes.empty() ? nullptr : bytes.data());
    };
    OSLCompiler compiler;
    std::string userdata;
    if (!compiler.compile_buffer(
            "shader hart_renderer_library("
            "float renderer_value=7 [[int interpolated=1]],output color Cout=0) { "
            "Cout=color(renderer_value,Dx(renderer_value),Dy(renderer_value)); }",
            userdata, { }, stdosl))
        return false;
    const struct {
        int osl, llvm;
        bool local, used;
    } variants[] = {
        { 0, 10, false, true },
        { 2, 10, true, true },
        { 2, 3, true, true },
        { 2, 3, false, false },
    };
    for (const auto& variant : variants) {
        HartUserdataServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
        OIIO_CHECK_ASSERT(ss.attribute("lazy_userdata", 1));
        OIIO_CHECK_ASSERT(ss.attribute("error_repeats", 1));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0));
        struct Snapshot {
            ShaderGroupRef group;
            const void* pointer;
            std::string bytes;
        };
        std::vector<Snapshot> snapshots;
        auto unchanged = [&]() {
            const int before = errors.errors;
            for (const auto& saved : snapshots) {
                ss.optimize_group(saved.group.get(), nullptr);
                const void* pointer = nullptr;
                uint64_t size       = 0;
                OIIO_CHECK_ASSERT(ss.getattribute(saved.group.get(),
                                                  "hart_bitcode", TypeDesc::PTR,
                                                  &pointer));
                OIIO_CHECK_ASSERT(ss.getattribute(saved.group.get(),
                                                  "hart_bitcode_size",
                                                  TypeUInt64, &size));
                OIIO_CHECK_EQUAL(pointer, saved.pointer);
                OIIO_CHECK_EQUAL(size, saved.bytes.size());
                if (pointer && size == saved.bytes.size())
                    OIIO_CHECK_EQUAL(std::memcmp(pointer, saved.bytes.data(),
                                                 size),
                                     0);
            }
            OIIO_CHECK_EQUAL(errors.errors, before);
        };
        bool load_shader = true;
        auto compile     = [&](string_view library, float bias) {
            const int previous_failures = unit_test_failures;
            renderer.requests.clear();
            const int before = errors.errors;
            auto group
                = make_group(ss, variant.used ? string_view(userdata) : basic,
                             1, load_shader);
            load_shader = false;
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors != before)
                print(stderr, "Renderer library (LLVM {}, used {}): {}\n",
                      variant.llvm, variant.used, errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, before);
            OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
            if (variant.used) {
                OIIO_CHECK_ASSERT(!renderer.requests.empty());
                for (const auto& request : renderer.requests) {
                    OIIO_CHECK_EQUAL(request.name, ustring("renderer_value"));
                    OIIO_CHECK_ASSERT(request.type == TypeFloat
                                      && request.derivatives);
                }
            } else {
                OIIO_CHECK_ASSERT(renderer.requests.empty());
            }
            check_module(ss, *group, arch,
                         variant.used
                             ? std::initializer_list<
                                   string_view> { "osl_hart_get_userdata" }
                             : std::initializer_list<string_view> { },
                         variant.llvm);
            OIIO_CHECK_ASSERT(check_renderer_library_ir(ss, *group, library,
                                                        variant.llvm,
                                                        variant.used, bias));
            if (variant.used && variant.llvm == 10)
                OIIO_CHECK_ASSERT(
                    check_userdata_ir(ss, *group, renderer.requests, false,
                                      true, false, !library.empty()));
            int allocated = -1, group_size = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "llvm_groupdata_size",
                                              group_size));
            OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
            OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
            const void* pointer = nullptr;
            uint64_t size       = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR, &pointer));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                              TypeUInt64, &size));
            if (!pointer || !size)
                return false;
            snapshots.push_back(
                { group, pointer,
                  std::string(static_cast<const char*>(pointer), size) });
            unchanged();
            if (unit_test_failures != previous_failures)
                print(stderr,
                      "Renderer library checks failed (OSL {}, LLVM {}, "
                      "local {}, used {}, bias {}, bytes {})\n",
                      variant.osl, variant.llvm, variant.local, variant.used,
                      bias, library.size());
            return true;
        };
        OIIO_CHECK_ASSERT(set_library(ss, libraries[0]));
        if (!compile(libraries[0], biases[0]))
            return false;
        if (variant.osl == 0) {
            const struct {
                TypeDesc type;
                const void* data;
            } invalid[] = {
                { TypeDesc(TypeDesc::UINT8, -1), libraries[1].data() },
                { TypeDesc(TypeDesc::UINT8, int(libraries[1].size())), nullptr },
                { TypeDesc(TypeDesc::UINT8, TypeDesc::VEC3, 1),
                  libraries[1].data() },
            };
            for (const auto& bad : invalid) {
                const int before = errors.errors;
                OIIO_CHECK_ASSERT(
                    !ss.attribute("lib_bitcode", bad.type, bad.data));
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_ASSERT(
                    OIIO::Strutil::contains(errors.last_error,
                                            "Invalid bitcode size:"));
                unchanged();
                // A fresh group must still link A, not just retain an old artifact.
                if (!compile(libraries[0], biases[0]))
                    return false;
            }
        }
        OIIO_CHECK_ASSERT(set_library(ss, libraries[1]));
        unchanged();
        if (!compile(libraries[1], biases[1]))
            return false;
        if (variant.osl == 0) {
            OIIO_CHECK_ASSERT(ss.attribute("lib_bitcode",
                                           TypeDesc(TypeDesc::UINT8, 0),
                                           nullptr));
            unchanged();
            if (!compile({ }, 0))
                return false;
        }
    }

    Diagnostics shadeops_errors;
    const auto shadeops_bytes = pvt::hart_shadeops_bitcode(arch,
                                                           shadeops_errors);
    OIIO_CHECK_ASSERT(shadeops_errors.errors == 0 && !shadeops_bytes.empty());
    if (shadeops_bytes.empty())
        return false;
    enum class Mutation {
        Triple,
        Layout,
        MissingProvenance,
        ProvenanceValue,
        ConflictingProvenance,
        Cpu,
        MissingCpu,
        Type,
        Convention,
        ABIAttribute,
        StrongDefinition,
        Import,
        LinkFlags,
        Constructor,
        Kernel,
        Callable,
        Alias,
        Assembly,
        IndirectCall
    };
    const struct {
        Mutation kind;
        const char* diagnostic;
    } mutations[] = {
        { Mutation::Triple, "target triple or data layout does not match" },
        { Mutation::Layout, "target triple or data layout does not match" },
        { Mutation::MissingProvenance, "missing device-storage ABI provenance" },
        { Mutation::ProvenanceValue,
          "incompatible device-storage ABI provenance" },
        { Mutation::ConflictingProvenance,
          "incompatible device-storage ABI provenance" },
        { Mutation::Cpu, "function 'rs_hart_get_userdata' does not target" },
        { Mutation::MissingCpu,
          "function 'rs_hart_get_userdata' does not target" },
        { Mutation::Type, "incompatible type for 'rs_hart_get_userdata'" },
        { Mutation::Convention,
          "incompatible calling convention for 'rs_hart_get_userdata'" },
        { Mutation::ABIAttribute,
          "incompatible ABI attributes for 'rs_hart_get_userdata'" },
        { Mutation::StrongDefinition, "duplicate definition of 'osl_sin_ff'" },
        { Mutation::Import, "unresolved import 'hart_renderer_missing'" },
        { Mutation::LinkFlags, "cannot link device bitcode" },
        { Mutation::Constructor,
          "assembly, aliases and global initialization are unsupported" },
        { Mutation::Kernel,
          "libraries must not define kernels or callable exports" },
        { Mutation::Callable,
          "libraries must not define kernels or callable exports" },
        { Mutation::Alias,
          "assembly, aliases and global initialization are unsupported" },
        { Mutation::Assembly,
          "assembly, aliases and global initialization are unsupported" },
        { Mutation::IndirectCall,
          "indirect calls and inline assembly are unsupported" },
    };
    auto reject = [&](string_view bytes, string_view diagnostic,
                      bool host_bitcode = false) {
        HartUserdataServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", 10));
        if (host_bitcode)
            OIIO_CHECK_ASSERT(
                ss.attribute("rs_bitcode",
                             TypeDesc(TypeDesc::UINT8, int(bytes.size())),
                             bytes.data()));
        else
            OIIO_CHECK_ASSERT(set_library(ss, bytes));
        auto group = make_group(ss, userdata);
        check_rejected_group(ss, *group, errors, diagnostic);
        if (!host_bitcode)
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error,
                                        "HART renderer library:"));
        OIIO_CHECK_EQUAL(renderer.host_lookups, 0);
    };
    reject("not LLVM bitcode", "cannot read bitcode:");
    reject(libraries[0], "host renderer bitcode", true);
    for (const auto& test : mutations) {
        llvm::LLVMContext context;
        auto library  = parse(libraries[0], context);
        auto shadeops = parse(string_view(reinterpret_cast<const char*>(
                                              shadeops_bytes.data()),
                                          shadeops_bytes.size()),
                              context);
        if (!library || !shadeops)
            return false;
        auto* callback = library->getFunction("rs_hart_get_userdata");
        auto* sine     = shadeops->getFunction("osl_sin_ff");
        OIIO_CHECK_ASSERT(callback && !callback->isDeclaration() && sine);
        if (!callback || callback->isDeclaration() || !sine)
            return false;
        auto* void_type
            = llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
        auto define_void = [&](const char* name) {
            auto* function
                = llvm::Function::Create(void_type,
                                         llvm::GlobalValue::ExternalLinkage,
                                         name, library.get());
            function->addFnAttr("target-cpu", std::string(arch));
            llvm::IRBuilder<> builder(
                llvm::BasicBlock::Create(context, "entry", function));
            builder.CreateRetVoid();
            return function;
        };
        switch (test.kind) {
        case Mutation::Triple:
#if LLVM_VERSION_MAJOR >= 21
            library->setTargetTriple(llvm::Triple("nvptx64-nvidia-cuda"));
#else
            library->setTargetTriple("nvptx64-nvidia-cuda");
#endif
            break;
        case Mutation::Layout: {
            auto layout = library->getDataLayout().getStringRepresentation();
            OIIO_CHECK_ASSERT(!layout.empty());
            if (layout.empty())
                return false;
            layout[0] = layout[0] == 'e' ? 'E' : 'e';
            library->setDataLayout(layout);
            break;
        }
        case Mutation::MissingProvenance:
        case Mutation::ProvenanceValue:
        case Mutation::ConflictingProvenance: {
            const llvm::StringRef prefix("__hart_device_storage_abi_");
            std::vector<llvm::GlobalVariable*> claims;
            for (auto& global : library->globals())
                if (global.getName().find(prefix) == 0)
                    claims.push_back(&global);
            OIIO_CHECK_ASSERT(!claims.empty());
            if (claims.empty())
                return false;
            if (test.kind == Mutation::MissingProvenance) {
                llvm::removeFromUsedLists(*library, [&](llvm::Constant* value) {
                    const auto* global = llvm::dyn_cast<llvm::GlobalVariable>(
                        value->stripPointerCasts());
                    return global && global->getName().find(prefix) == 0;
                });
                for (auto* claim : claims) {
                    claim->removeDeadConstantUsers();
                    OIIO_CHECK_ASSERT(claim->use_empty());
                    if (!claim->use_empty())
                        return false;
                    claim->eraseFromParent();
                }
            } else {
                auto* claim = claims.front();
                OIIO_CHECK_ASSERT(claim->isConstant()
                                  && claim->hasInitializer());
                if (!claim->isConstant() || !claim->hasInitializer())
                    return false;
                const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                    claim->getInitializer());
                OIIO_CHECK_ASSERT(value);
                if (!value)
                    return false;
                auto* wrong = llvm::ConstantInt::get(
                    context,
                    value->getValue() ^ llvm::APInt(value->getBitWidth(), 1));
                OIIO_CHECK_ASSERT(wrong != value);
                if (test.kind == Mutation::ProvenanceValue) {
                    claim->setInitializer(wrong);
                } else {
                    const auto name = claim->getName().split('.').first.str()
                                      + ".999";
                    OIIO_CHECK_ASSERT(!library->getNamedGlobal(name));
                    new llvm::GlobalVariable(
                        *library, claim->getValueType(), true,
                        llvm::GlobalValue::InternalLinkage, wrong, name,
                        nullptr, llvm::GlobalVariable::NotThreadLocal,
                        claim->getAddressSpace());
                    OIIO_CHECK_EQUAL(claim->getInitializer(), value);
                }
            }
            break;
        }
        case Mutation::Cpu: callback->addFnAttr("target-cpu", "gfx0000"); break;
        case Mutation::MissingCpu: callback->removeFnAttr("target-cpu"); break;
        case Mutation::Type: {
            std::vector<llvm::Type*> args;
            for (const auto& arg : callback->args())
                args.push_back(arg.getType());
            callback->setName("hart_original_userdata");
            callback->setLinkage(llvm::GlobalValue::InternalLinkage);
            auto* wrong = llvm::Function::Create(
                llvm::FunctionType::get(llvm::Type::getInt32Ty(context), args,
                                        false),
                llvm::GlobalValue::ExternalLinkage, "rs_hart_get_userdata",
                library.get());
            wrong->addFnAttr("target-cpu", std::string(arch));
            llvm::IRBuilder<> builder(
                llvm::BasicBlock::Create(context, "entry", wrong));
            builder.CreateRet(builder.getInt32(0));
            break;
        }
        case Mutation::Convention:
            callback->setCallingConv(callback->getCallingConv()
                                             == llvm::CallingConv::Fast
                                         ? llvm::CallingConv::C
                                         : llvm::CallingConv::Fast);
            break;
        case Mutation::ABIAttribute:
            if (callback->hasRetAttribute(llvm::Attribute::ZExt))
                callback->removeRetAttr(llvm::Attribute::ZExt);
            else if (callback->hasRetAttribute(llvm::Attribute::SExt))
                callback->removeRetAttr(llvm::Attribute::SExt);
            else
                callback->addRetAttr(llvm::Attribute::ZExt);
            break;
        case Mutation::StrongDefinition: {
            OIIO_CHECK_ASSERT(!library->getNamedValue("osl_sin_ff"));
            auto* duplicate
                = llvm::Function::Create(sine->getFunctionType(),
                                         llvm::GlobalValue::ExternalLinkage,
                                         "osl_sin_ff", library.get());
            duplicate->setAttributes(sine->getAttributes());
            duplicate->setCallingConv(sine->getCallingConv());
            llvm::IRBuilder<> builder(
                llvm::BasicBlock::Create(context, "entry", duplicate));
            builder.CreateRet(duplicate->getArg(0));
            break;
        }
        case Mutation::Import: {
            auto* missing = llvm::Function::Create(
                void_type, llvm::GlobalValue::ExternalLinkage,
                "hart_renderer_missing", library.get());
            llvm::IRBuilder<> builder(
                &*callback->getEntryBlock().getFirstInsertionPt());
            builder.CreateCall(missing);
            break;
        }
        case Mutation::LinkFlags: {
            const auto* flag = llvm::dyn_cast_or_null<llvm::ConstantAsMetadata>(
                shadeops->getModuleFlag("wchar_size"));
            const auto* size = flag ? llvm::dyn_cast<llvm::ConstantInt>(
                                          flag->getValue())
                                    : nullptr;
            OIIO_CHECK_ASSERT(size);
            if (!size)
                return false;
            library->setModuleFlag(llvm::Module::Error, "wchar_size",
                                   size->getZExtValue() == 2 ? uint32_t(4)
                                                             : uint32_t(2));
            break;
        }
        case Mutation::Constructor: {
            auto* startup = define_void("hart_renderer_startup");
            auto* pointer = llvm::cast<llvm::PointerType>(
                callback->getArg(0)->getType());
            auto* record
                = llvm::StructType::get(context,
                                        { llvm::Type::getInt32Ty(context),
                                          startup->getType(), pointer });
            llvm::Constant* fields[] = {
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 65535),
                startup, llvm::ConstantPointerNull::get(pointer)
            };
            auto* element = llvm::ConstantStruct::get(record, fields);
            auto* array   = llvm::ArrayType::get(record, 1);
            new llvm::GlobalVariable(*library, array, false,
                                     llvm::GlobalValue::AppendingLinkage,
                                     llvm::ConstantArray::get(array,
                                                              { element }),
                                     "llvm.global_ctors");
            break;
        }
        case Mutation::Kernel:
            define_void("hart_renderer_kernel")
                ->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
            break;
        case Mutation::Callable:
            define_void("__direct_callable__hart_renderer");
            break;
        case Mutation::Alias:
            llvm::GlobalAlias::create(callback->getValueType(),
                                      callback->getAddressSpace(),
                                      llvm::GlobalValue::ExternalLinkage,
                                      "hart_renderer_alias", callback,
                                      library.get());
            break;
        case Mutation::Assembly:
            library->setModuleInlineAsm("// unsupported renderer assembly");
            break;
        case Mutation::IndirectCall: {
            llvm::IRBuilder<> builder(
                &*callback->getEntryBlock().getFirstInsertionPt());
#if LLVM_VERSION_MAJOR >= 15
            auto* pointer_type = llvm::PointerType::getUnqual(context);
#else
            auto* pointer_type = llvm::PointerType::getUnqual(void_type);
#endif
            auto* pointer = builder.CreateBitCast(callback->getArg(0),
                                                  pointer_type);
            builder.CreateCall(void_type, pointer);
            break;
        }
        }
        std::string diagnostic;
        llvm::raw_string_ostream stream(diagnostic);
        const bool invalid = llvm::verifyModule(*library, &stream);
        if (invalid)
            print(stderr, "Invalid mutation for '{}': {}\n", test.diagnostic,
                  diagnostic);
        OIIO_CHECK_ASSERT(!invalid);
        if (invalid)
            return false;
        reject(serialize(*library), test.diagnostic);
    }
    return true;
}



bool
check_diagnostic_modules(string_view arch, string_view stdosl)
{
    const string_view filename = "hart_diagnostic_fixture.osl";
    auto expectation =
        [&](string_view format, string_view types, int line,
            HartDiagnosticSeverity severity = HartDiagnosticSeverity::Print,
            string_view shader              = "hart_test") {
            DiagnosticExpectation result { std::string(format),
                                           { },
                                           ustring(shader),
                                           ustring(filename),
                                           line,
                                           severity,
                                           { } };
            for (char type : types) {
                OIIO_CHECK_ASSERT(type == 's' || type == 'i' || type == 'u'
                                  || type == 'f');
                result.types.push_back(type == 's'   ? EncodedType::kUstringHash
                                       : type == 'i' ? EncodedType::kInt32
                                       : type == 'u' ? EncodedType::kUInt32
                                                     : EncodedType::kFloat);
            }
            return result;
        };
    auto repeat = [](string_view field, size_t count) {
        return OIIO::Strutil::join(std::vector<std::string>(count,
                                                            std::string(field)),
                                   " ");
    };
    auto check = [&](string_view label, string_view producer_source,
                     string_view consumer_source,
                     cspan<DiagnosticExpectation> expected) {
        std::string bytecode[2];
        const bool connected = !consumer_source.empty();
        OSLCompiler producer_compiler;
        // The lexer adjusts main-file line numbers; use a distinct #line file.
        const string_view input_file = "hart_diagnostic_input.osl";
        if (!producer_compiler.compile_buffer(producer_source, bytecode[0], { },
                                              stdosl, input_file))
            return false;
        if (connected) {
            OSLCompiler consumer_compiler;
            if (!consumer_compiler.compile_buffer(consumer_source, bytecode[1],
                                                  { }, stdosl, input_file))
                return false;
        }
        for (int optimize : { 10, 3 }) {
            HartServices renderer(false, false, false, true, false, false,
                                  false, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", optimize == 10 ? 0 : 2);
            ss.attribute("llvm_optimize", optimize);
            ss.attribute("max_hart_groupdata_alloc", optimize == 3 ? 4096 : 0);
            auto group = connected ? make_connected_group(ss, bytecode[0],
                                                          bytecode[1])
                                   : make_group(ss, bytecode[0]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "Diagnostic {} (LLVM {}):\n{}", label, optimize,
                      errors.messages);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            int size = 0, allocated = -1;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "llvm_groupdata_size", size));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
            OIIO_CHECK_EQUAL(allocated, optimize == 3 ? size : 0);
            const int previous_failures = unit_test_failures;
            check_module(ss, *group, arch,
                         { "osl_hart_diagnostic", "rs_hart_diagnostic" },
                         optimize, connected, false, false, connected ? 2 : 0,
                         false, true, 0, -1, -1, expected);
            if (unit_test_failures != previous_failures)
                print(stderr, "Diagnostic module checks failed: {} (LLVM {})\n",
                      label, optimize);
        }
        return true;
    };
    const string_view scalar_source = R"osl(
shader hart_test(string label="default", output color Cout=0) {
    int n=int(7*u)-1;
#line 100 "hart_diagnostic_fixture.osl"
    printf("diagnostic {literal} 100%%\n");
    printf("");
    printf("%d %s %f %o %x %X %i\n",7,"literal",1.25,n,n,n,3);
    warning("label=%s",label);
    error("error=%d",n);
    printf("flags=%-+08.2f|%#010x|% 8.2f",1.25,n,2.5);
    Cout=color(u,v,1);
}
)osl";
    DiagnosticExpectation scalars[] = {
        expectation("diagnostic {{literal}} 100%\n", "", 100),
        expectation("", "", 101),
        expectation("{:d} {:s} {:f} {:o} {:x} {:X} {:d}\n", "isfuuui", 102),
        expectation("label={:s}", "s", 103, HartDiagnosticSeverity::Warning),
        expectation("error={:d}", "i", 104, HartDiagnosticSeverity::Error),
        expectation("flags={:<+08.2f}|{:#010x}|{: 8.2f}", "fuf", 105),
    };
    scalars[2].literals = { { 0, 7 },
                            { 1, ustringhash("literal").hash() },
                            { 2, 0x3fa00000 },
                            { 6, 3 } };
    if (!check("empty, scalar, octal/hex and all severities", scalar_source, "",
               scalars))
        return false;
    const string_view array_source       = R"osl(
shader hart_test(output color Cout=0) {
    int integers[2]={-1,5};
    float floats[2]={1.25,2.5};
    string words[2]={"alpha",""};
    color colors[2]={color(1,2,3),color(4,5,6)};
    matrix m=matrix(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
#line 100 "hart_diagnostic_fixture.osl"
    printf("arrays=%d|%g|%s|%v|%m",integers,floats,words,colors,m);
    warning("geometry=%c|%n|%p|%v|%e",color(1,2,3),normal(0,0,1),point(u,v,1),vector(I),1.25);
    Cout=color(u,v,1);
}
)osl";
    const DiagnosticExpectation arrays[] = {
        expectation("arrays=" + repeat("{:d}", 2) + "|" + repeat("{:g}", 2)
                        + "|" + repeat("{:s}", 2) + "|" + repeat("{:f}", 6)
                        + "|" + repeat("{:f}", 16),
                    "iiffss" + std::string(22, 'f'), 100),
        expectation("geometry=" + repeat("{:f}", 3) + "|" + repeat("{:f}", 3)
                        + "|" + repeat("{:f}", 3) + "|" + repeat("{:f}", 3)
                        + "|{:e}",
                    std::string(13, 'f'), 101, HartDiagnosticSeverity::Warning),
    };
    if (!check("arrays, triples and matrix flattening", array_source, "",
               arrays))
        return false;
    const string_view producer_source       = R"osl(
shader hart_producer(string label="default", output string value="") {
    value=u>v ? label : "connected";
#line 200 "hart_diagnostic_fixture.osl"
    printf("producer=%s",value);
}
)osl";
    const string_view consumer_source       = R"osl(
shader hart_consumer(string value="", output color Cout=0) {
#line 300 "hart_diagnostic_fixture.osl"
    warning("consumer=%s",value);
    Cout=color(value=="default",u,v);
}
)osl";
    const DiagnosticExpectation connected[] = {
        expectation("producer={:s}", "s", 200, HartDiagnosticSeverity::Print,
                    "hart_producer"),
        expectation("consumer={:s}", "s", 300, HartDiagnosticSeverity::Warning,
                    "hart_consumer"),
    };
    if (!check("default and connected strings", producer_source,
               consumer_source, connected))
        return false;
    const std::string literal_limit(4096, 'x');
    const std::string expanded_prefix(3777, 'y');
    const std::string spec_limit = "%" + std::string(118, '-') + "d";
    const auto boundary_source
        = fmtformat("shader hart_test(string words[256]={{}},"
                    "float numbers[64]={{}},output color Cout=0) {{\n"
                    "#line 100 \"hart_diagnostic_fixture.osl\"\n"
                    "printf(\"%s\",words);\n"
                    "printf(\"{}%f\",numbers);\n"
                    "printf(\"{}\");\n"
                    "printf(\"%1024.128f\",1.25);\n"
                    "printf(\"{}\",7);\n"
                    "Cout=color(u,v,1); }}",
                    expanded_prefix, literal_limit, spec_limit);
    const DiagnosticExpectation boundaries[] = {
        expectation(repeat("{:s}", 256), std::string(256, 's'), 100),
        expectation(expanded_prefix + repeat("{:f}", 64), std::string(64, 'f'),
                    101),
        expectation(literal_limit, "", 102),
        expectation("{:1024.128f}", "f", 103),
        expectation("{:<d}", "i", 104),
    };
    OIIO_CHECK_EQUAL(boundaries[1].format.size(), 4096);
    OIIO_CHECK_EQUAL(spec_limit.size(), 120);
    if (!check("inclusive argument, payload and format limits", boundary_source,
               "", boundaries))
        return false;
    auto reject = [&](string_view bytecode, string_view diagnostic,
                      bool capability = true) {
        HartServices renderer(false, false, true, true, false, false, false,
                              capability);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        register_hart_closures(ss);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        ss.attribute("llvm_optimize", 3);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, diagnostic);
    };
    for (string_view operation : { "printf", "warning", "error" }) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat("shader bad(output color Cout=0) {{ "
                          "if(u<0) {}(\"unreachable on the grid\"); "
                          "Cout=color(u,v,1); }}",
                          operation),
                bytecode, { }, stdosl))
            return false;
        reject(bytecode, "renderer lacks HARTDiagnostics", false);
    }
    {
        // HART must not broaden the common frontend's integer conversions.
        Diagnostics errors;
        OSLCompiler compiler(&errors);
        std::string bytecode;
        OIIO_CHECK_ASSERT(
            !compiler.compile_buffer("shader bad(output color Cout=0) { "
                                     "printf(\"%u\",int(u)); Cout=color(u); }",
                                     bytecode, { }, stdosl));
        OIIO_CHECK_ASSERT(errors.errors > 0);
        OIIO_CHECK_ASSERT(
            OIIO::Strutil::contains(errors.messages,
                                    "needs %d, %i, %o, %x, or %X"));
    }
    // Mutate a valid literal to exercise the original-master validator, not
    // the frontend's separate printf checker (including malformed specifiers).
    OSLCompiler format_compiler;
    std::string format_bytecode;
    if (!format_compiler.compile_buffer(
            "shader bad(output color Cout=0) { "
            "printf(\"HART_DIAGNOSTIC_FORMAT %d\",int(u)); Cout=color(u,v,1); }",
            format_bytecode, { }, stdosl))
        return false;
    const std::string marker = "\"HART_DIAGNOSTIC_FORMAT %d\"";
    const auto format_offset = format_bytecode.find(marker);
    OIIO_CHECK_ASSERT(format_offset != std::string::npos);
    if (format_offset == std::string::npos)
        return false;
    const std::pair<std::string, string_view> bad_formats[] = {
        { "%", "unsupported diagnostic format specification" },
        { "%q", "unsupported diagnostic format specification" },
        { "%u", "unsupported diagnostic format specification" },
        { "%*d", "unsupported diagnostic format specification" },
        { "%.*f", "unsupported diagnostic format specification" },
        { "%ld", "unsupported diagnostic format specification" },
        { "%lld", "unsupported diagnostic format specification" },
        { "%hd", "unsupported diagnostic format specification" },
        { "%Lf", "unsupported diagnostic format specification" },
        { "%1$d", "unsupported diagnostic format specification" },
        { "%1025d", "diagnostic width exceeds 1024" },
        { "%999999999999999999999d", "diagnostic width exceeds 1024" },
        { "%.129f", "diagnostic precision exceeds 128" },
        { "%" + std::string(119, '-') + "d",
          "unsupported diagnostic format specification" },
        { std::string(4095, 'x') + "%d",
          "diagnostic format exceeds 4096 bytes" },
        { "no fields", "diagnostic format/argument count mismatch" },
        { "%d%d", "diagnostic format/argument count mismatch" },
    };
    for (const auto& test : bad_formats) {
        std::string bytecode = format_bytecode;
        bytecode.replace(format_offset, marker.size(),
                         fmtformat("\"{}\"", test.first));
        const int previous_failures = unit_test_failures;
        reject(bytecode, test.second);
        if (unit_test_failures != previous_failures)
            print(stderr, "Diagnostic format rejection failed ({} chars): {}\n",
                  test.first.size(), test.first.substr(0, 128));
    }
    const std::pair<std::string, string_view> bad_arguments[] = {
        { "shader bad(string fmt=\"%d\",output color Cout=0) { "
          "printf(fmt,int(u)); Cout=color(u); }",
          "diagnostic formats must be literal strings" },
        { "shader bad(output color Cout=0) { "
          "printf(u>v ? \"%d\" : \"%i\",int(u)); Cout=color(u); }",
          "diagnostic formats must be literal strings" },
        { "shader bad(closure color c=0,output color Cout=0) { "
          "printf(\"%s\",c); Cout=color(u); }",
          "unsupported diagnostic argument type" },
        { "shader bad(float a[257]={},output color Cout=0) { "
          "printf(\"%f\",a); Cout=color(u); }",
          "diagnostic exceeds 256 scalar arguments" },
        { "shader bad(string a[257]={},output color Cout=0) { "
          "printf(\"%s\",a); Cout=color(u); }",
          "diagnostic exceeds 256 scalar arguments" },
        { "shader bad(vector a[86]={},output color Cout=0) { "
          "printf(\"%v\",a); Cout=color(u); }",
          "diagnostic exceeds 256 scalar arguments" },
        { fmtformat("shader bad(float a[64]={{}},output color Cout=0) {{ "
                    "printf(\"{}%f\",a); Cout=color(u); }}",
                    std::string(3778, 'y')),
          "expanded diagnostic format or payload exceeds its limit" },
        { "shader bad(output color Cout=0) { "
          "fprintf(\"unsupported.txt\",\"%f\",u); Cout=color(u); }",
          "unsupported operation 'fprintf'" },
        { "shader bad(output color Cout=0) { "
          "Cout=color(format(\"%f\",u)==\"unsupported\",u,v); }",
          "unsupported operation 'format'" },
    };
    for (const auto& test : bad_arguments) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.first, bytecode, { }, stdosl))
            return false;
        reject(bytecode, test.second);
    }
    OSLCompiler struct_compiler;
    std::string struct_bytecode;
    if (!struct_compiler.compile_buffer(
            "struct HartDiagnosticRejectedStruct { float x; }; "
            "shader bad(HartDiagnosticRejectedStruct value={1},float n=1,"
            "output color Cout=0) { printf(\"%f\",n); "
            "Cout=color(value.x,u,v); }",
            struct_bytecode, { }, stdosl))
        return false;
    const auto op    = struct_bytecode.find("\tprintf\t");
    const auto end   = struct_bytecode.find('\n', op);
    const auto hints = struct_bytecode.find('%', op);
    OIIO_CHECK_ASSERT(op != std::string::npos && end != std::string::npos
                      && hints < end);
    if (op == std::string::npos || end == std::string::npos || hints >= end)
        return false;
    std::vector<std::string> words;
    OIIO::Strutil::split(string_view(struct_bytecode).substr(op, hints - op),
                         words, "", -1);
    OIIO_CHECK_EQUAL(words.size(), 3);
    if (words.size() != 3)
        return false;
    OIIO_CHECK_EQUAL(words[2], "n");
    words[2] = "value";
    struct_bytecode.replace(op, hints - op,
                            fmtformat("\t{}\t",
                                      OIIO::Strutil::join(words, "\t")));
    reject(struct_bytecode, "unsupported diagnostic argument type");
    return true;
}



bool
check_hash_modules(string_view arch, string_view stdosl)
{
    const string_view encode
        = "Cout=color(value&65535,(value>>16)&65535,value<0);";
    OSLCompiler consumer_compiler;
    std::string consumer;
    if (!consumer_compiler.compile_buffer(
            fmtformat("shader hash_consumer(int value=0, output color Cout=0) "
                      "{{ {} }}",
                      encode),
            consumer, {}, stdosl))
        return false;
    auto check = [&](string_view label, string_view producer, string_view input,
                     std::initializer_list<string_view> helpers,
                     int osl_optimize, int optimize, bool local) {
        HartServices renderer(false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", osl_optimize);
        ss.attribute("llvm_optimize", optimize);
        ss.attribute("max_hart_groupdata_alloc", local ? 4096 : 0);
        const bool connected = !input.empty();
        auto group = connected ? make_connected_group(ss, producer, input)
                               : make_group(ss, producer);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Hash {} (OSL {}, LLVM {}):\n{}", label, osl_optimize,
                  optimize, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, local ? size : 0);
        check_module(ss, *group, arch, helpers, optimize, connected, false,
                     false, connected ? 2 : 0, false, true);
    };
    for (bool strings : { false, true }) {
        const string_view params
            = strings
                  ? "string label=\"hart-hash-default\", string empty=\"\", "
                  : "";
        const string_view body
            = strings ? "string names[4]={\"\",\"hart-hash-alpha\","
                        "\"hart-hash-beta\",\"hart-hash-gamma\"}; "
                        "value=hash(names[int(3*u)])^hash(label)^hash(empty); "
                      : "float x=1.3*u-0.25, y=0.1+1.7*v; "
                        "point p=point(x,y,u*v); "
                        "value=hash(int(65535*u)-32768)^hash(x)^hash(x,y)"
                        "^hash(p)^hash(p,u+v); ";
        const std::string sources[] = {
            fmtformat("shader hash_values({}output color Cout=0) {{ "
                      "int value=0; {} {} }}",
                      params, body, encode),
            fmtformat("shader hash_producer({}output int value=0) {{ {} }}",
                      params, body),
        };
        std::string bytecode[2];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], {}, stdosl))
                return false;
        }
        const std::initializer_list<string_view> helpers = {
            strings ? "osl_hash_is" : "osl_hash_ii",
            strings ? "" : "osl_hash_if",
            strings ? "" : "osl_hash_iff",
            strings ? "" : "osl_hash_iv",
            strings ? "" : "osl_hash_ivf",
        };
        check(strings ? "string array and defaults" : "numeric overloads",
              bytecode[0], "", helpers, 0, 10, false);
        for (int optimize : { 10, 3 })
            check(strings ? "connected string hash results"
                          : "connected numeric results",
                  bytecode[1], consumer, helpers, 2, optimize, true);
    }
    {
        const string_view sources[] = {
            "shader hash_string_producer(output string value=\"\") { "
            "value=u>v?\"hart-hash-connected\":\"\"; }",
            "shader hash_string_consumer(string value=\"default\", "
            "output color Cout=0) { int h=hash(value); "
            "Cout=color(h&65535,(h>>16)&65535,h<0); }",
        };
        std::string bytecode[2];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], {}, stdosl))
                return false;
        }
        for (int optimize : { 10, 3 })
            check("connected string operand", bytecode[0], bytecode[1],
                  { "osl_hash_is" }, optimize == 10 ? 0 : 2, optimize,
                  optimize == 3);
    }
    const struct {
        bool pair;
        unsigned operand;
        const char* replacement;
    } malformed[] = {
        { false, 1, "x" }, { false, 2, "m" },   { false, 2, "a" },
        { true, 2, "n" },  { true, 2, "word" }, { true, 3, "word" },
    };
    for (const auto& test : malformed) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source = fmtformat(
            "shader bad_hash(float x=0.2, int n=7, string word=\"alpha\", "
            "matrix m=1, float a[2]={{0,1}}, output int value=0, "
            "output color Cout=0) {{ value=hash(x{}); "
            "Cout=color(value+n,x+m[0][0],a[0]); }}",
            test.pair ? ",x" : "");
        if (!compiler.compile_buffer(source, bytecode, {}, stdosl))
            return false;
        const auto op    = bytecode.find("\thash\t");
        const auto end   = bytecode.find('\n', op);
        const auto hints = bytecode.find('%', op);
        OIIO_CHECK_ASSERT(op != std::string::npos && end != std::string::npos
                          && hints < end);
        if (op == std::string::npos || end == std::string::npos || hints >= end)
            return false;
        std::vector<std::string> words;
        OIIO::Strutil::split(string_view(bytecode).substr(op, hints - op),
                             words, "", -1);
        OIIO_CHECK_EQUAL(words.size(), test.pair ? 4 : 3);
        if (words.size() != (test.pair ? 4 : 3))
            return false;
        words[test.operand] = test.replacement;
        bytecode.replace(op, hints - op,
                         fmtformat("\t{}\t", OIIO::Strutil::join(words, "\t")));
        HartServices renderer(false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, "invalid hash operands");
    }
    return true;
}



bool
check_dynamic_noise_modules(string_view arch, string_view stdosl)
{
    auto check = [&](string_view label, string_view producer,
                     string_view consumer,
                     std::initializer_list<string_view> helpers,
                     int osl_optimize, int optimize, bool local,
                     int guard_flags) {
        HartServices renderer(false, false, false, true, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", osl_optimize);
        ss.attribute("llvm_optimize", optimize);
        ss.attribute("max_hart_groupdata_alloc", local ? 4096 : 0);
        const bool connected = !consumer.empty();
        auto group = connected ? make_connected_group(ss, producer, consumer)
                               : make_group(ss, producer);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Dynamic {} (OSL {}, LLVM {}):\n{}", label,
                  osl_optimize, optimize, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, local ? size : 0);
        const int previous_failures = unit_test_failures;
        check_module(ss, *group, arch, helpers, optimize, connected, false,
                     false, connected ? 2 : 0, false, true, 0, -1, guard_flags);
        if (unit_test_failures != previous_failures)
            print(stderr,
                  "Dynamic noise module checks failed: {} "
                  "(OSL {}, LLVM {})\n",
                  label, osl_optimize, optimize);
    };
    // Invalid names in storage must compile; the dynamic guard rejects them
    // only when selected. They are not literal-selector exceptions.
    const string_view selection
        = "string names[10]={\"perlin\",\"uperlin\",\"noise\",\"snoise\","
          "\"cell\",\"hash\",\"gabor\",\"simplex\",\"usimplex\",\"unknown\"}; "
          "string kind=names[int(9*u)]; ";
    const string_view coordinates
        = "float x=0.31+1.3*u, y=0.27+0.7*v, t=0.2+u*v; "
          "point p=point(x,y,0.4+u*v); ";
    const string_view record
        = "struct HartDynamicNoiseInput { string kind; float x; float y; "
          "point p; float t; }; ";
    for (bool periodic : { false, true }) {
        const string_view operation = periodic ? "pnoise" : "noise";
        const string_view family = periodic ? "genericpnoise" : "genericnoise";
        const string_view per1   = periodic ? ",3.0" : "";
        const string_view per2   = periodic ? ",3.0,4.0" : "";
        const string_view per3   = periodic ? ",point(3,4,5)" : "";
        const string_view per4   = periodic ? ",point(3,4,5),7.0" : "";
        for (string_view type : { "float", "color" }) {
            const auto body
                = fmtformat("{0} a={1}(kind,x{2}), b={1}(kind,x,y{3}), "
                            "c={1}(kind,p{4}), d={1}(kind,p,t{5}), "
                            "e={1}(kind,x,0.31{3}), "
                            "f={1}(kind,point(0.2,0.3,0.4),t{5}); "
                            "{0} result=a+0.7*b+0.6*c+0.5*d+0.4*e+0.3*f; ",
                            type, operation, per1, per2, per3, per4);
            const std::string sources[] = {
                fmtformat("shader dynamic_noise_values(output color Cout=0) "
                          "{{ {} {} {} Cout=color(result); }}",
                          selection, coordinates, body),
                fmtformat("{} shader dynamic_selector("
                          "output HartDynamicNoiseInput value={{\"\",0,0,0,0}}) "
                          "{{ {} {} value.kind=kind; value.x=x; value.y=y; "
                          "value.p=p; value.t=t; }}",
                          record, selection, coordinates),
                fmtformat("{} shader dynamic_noise_consumer("
                          "HartDynamicNoiseInput value={{\"gabor\",0,0,0,0}}, "
                          "output color Cout=0) {{ string kind=value.kind; "
                          "float x=value.x, y=value.y, t=value.t; "
                          "point p=value.p; {} "
                          "Cout=color(result+Dx(result)+Dy(result)); }}",
                          record, body),
            };
            std::string bytecode[3];
            for (size_t i = 0; i < std::size(sources); ++i) {
                OSLCompiler compiler;
                if (!compiler.compile_buffer(sources[i], bytecode[i], {},
                                             stdosl))
                    return false;
            }
            const auto prefix           = fmtformat("osl_{}_{}", family,
                                          type == "float" ? "df" : "dv");
            const std::string helpers[] = {
                prefix + "df" + (periodic ? "f" : ""),
                prefix + "dfdf" + (periodic ? "ff" : ""),
                prefix + "dv" + (periodic ? "v" : ""),
                prefix + "dvdf" + (periodic ? "vf" : ""),
            };
            const auto required = { string_view(helpers[0]),
                                    string_view(helpers[1]),
                                    string_view(helpers[2]),
                                    string_view(helpers[3]),
                                    string_view("osl_hart_noise_validate"),
                                    string_view("osl_init_noise_options"),
                                    string_view("rs_hart_noise_error") };
            check(fmtformat("{} {} value only", operation, type), bytecode[0],
                  "", required, 0, 10, false, 1);
            for (int optimize : { 10, 3 })
                check(fmtformat("{} {} connected derivatives", operation, type),
                      bytecode[1], bytecode[2], required, 2, optimize, true, 1);
        }
        {
            const auto source = fmtformat(
                "shader dynamic_options(output color Cout=0) {{ "
                "string names[2]={{\"gabor\",\"perlin\"}}; "
                "string kind=names[int(u>v)]; {4} "
                "float a={0}(kind,p{1},\"bandwidth\",0.9+0.2*u,"
                "\"impulses\",8.0+v); "
                "color b={0}(kind,p,t{2},\"anisotropic\",1,"
                "\"do_filter\",int(u<0.75),\"direction\",vector(1,0.2*v,0.3),"
                "\"bandwidth\",2,\"impulses\",8); "
                "float c={0}(kind,x{3}); "
                "Cout=color(a+Dx(a)+Dy(a)+c)+b+Dx(b)+Dy(b); }}",
                operation, per3, per4, per1, coordinates);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, {}, stdosl))
                return false;
            for (int optimize : { 10, 3 })
                check(fmtformat("{} options and reset", operation), bytecode,
                      "",
                      { fmtformat("osl_{}_dfdv{}", family, periodic ? "v" : ""),
                        fmtformat("osl_{}_dvdvdf{}", family,
                                  periodic ? "vf" : ""),
                        fmtformat("osl_{}_dfdf{}", family, periodic ? "f" : ""),
                        "osl_hart_noise_validate", "osl_init_noise_options",
                        "osl_noiseparams_set_anisotropic",
                        "osl_noiseparams_set_do_filter",
                        "osl_noiseparams_set_direction",
                        "osl_noiseparams_set_bandwidth",
                        "osl_noiseparams_set_impulses", "rs_hart_noise_error" },
                      optimize == 10 ? 0 : 2, optimize, optimize == 3, 3);
        }
        for (string_view option : { "\"anisotropic\",1.0", "\"direction\",0.5",
                                    "\"impulses\",color(2)", "option,1.0" }) {
            const auto source = fmtformat(
                "shader bad_dynamic_noise(string kind=\"gabor\", "
                "string option=\"bandwidth\", output color Cout=0) {{ "
                "Cout={}(kind,P{},{}); }}",
                operation, per3, option);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, {}, stdosl))
                return false;
            HartServices renderer(false, false, false, false, false, false,
                                  true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", 2);
            auto group = make_group(ss, bytecode);
            check_rejected_group(
                ss, *group, errors,
                option == "option,1.0"
                    ? "noise option names must be literal strings"
                    : "unsupported noise option");
        }
        {
            OSLCompiler compiler;
            std::string bytecode;
            const auto source
                = fmtformat("shader specialized_noise(string kind=\"perlin\", "
                            "output color Cout=0) {{ Cout={}(kind,P{},"
                            "\"do_filter\",0); }}",
                            operation, per3);
            if (!compiler.compile_buffer(source, bytecode, {}, stdosl))
                return false;
            HartServices renderer(false, false, false, false, false, false,
                                  true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", 2);
            auto group = make_group(ss, bytecode);
            check_rejected_group(ss, *group, errors,
                                 "noise options require gabor");
        }
        const struct {
            const char* name;
            bool options;
        } overrides[] = {
            { "", false },       { "cellnoise", false }, { "hashnoise", false },
            { "pnoise", false }, { "psnoise", false },   { "perlin", true },
            { "cell", true },    { "hash", true },
        };
        for (const auto& test : overrides) {
            // The master selector is a parameter; only OSL2 specialization
            // exposes the invalid override, with coordinates eligible to fold.
            OSLCompiler compiler;
            std::string bytecode;
            const auto source = fmtformat(
                "shader folded_noise_override(string kind=\"gabor\" "
                "[[int lockgeom=1]], output color Cout=0) {{ "
                "Cout={}(kind,point(0.25,0.5,0.75){}{}); }}",
                operation, per3, test.options ? ",\"do_filter\",0" : "");
            if (!compiler.compile_buffer(source, bytecode, {}, stdosl))
                return false;
            HartServices renderer(false, false, false, false, false, false,
                                  true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", 3));
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_test", bytecode));
            auto group = ss.ShaderGroupBegin("hart_test_group");
            const ustring name(test.name);
            OIIO_CHECK_ASSERT(
                ss.Parameter("kind", TypeString, &name, ParamHints::none));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("Cout", TypeColor, false,
                                         SymArena::Outputs, 0,
                                         3 * sizeof(float));
            ss.add_symlocs(group.get(), { &output, 1 });
            const int previous_failures = unit_test_failures;
            check_rejected_group(ss, *group, errors,
                                 test.options ? "noise options require gabor"
                                              : "unsupported noise type");
            if (unit_test_failures != previous_failures)
                print(stderr,
                      "OSL2 constant-coordinate {} override '{}' "
                      "(options={}):\n{}",
                      operation, test.name, test.options, errors.messages);
        }
        for (string_view name :
             { "null", "unull", "simplexnoise", "unknown" }) {
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(
                    fmtformat("shader bad_literal_noise(output color Cout=0) "
                              "{{ Cout={}(\"{}\",P{}); }}",
                              operation, name, per3),
                    bytecode, {}, stdosl))
                return false;
            HartServices renderer(false, false, false, false, false, false,
                                  true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            auto group = make_group(ss, bytecode);
            check_rejected_group(ss, *group, errors, "unsupported noise type");
        }
    }
    return true;
}



bool
check_gabor_modules(string_view arch, string_view stdosl)
{
    auto check = [&](string_view label, string_view producer,
                     string_view consumer,
                     std::initializer_list<string_view> shadeops,
                     int osl_optimize, int optimize, bool local = false) {
        HartServices renderer(false, false, false, false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", local ? 4096 : 0));
        const bool connected = !consumer.empty();
        auto group = connected ? make_connected_group(ss, producer, consumer)
                               : make_group(ss, producer);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Gabor {} (OSL {}, LLVM {}, connected {}): {}\n",
                  label, osl_optimize, optimize, connected, errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, local ? size : 0);
        const int previous_failures = unit_test_failures;
        check_module(ss, *group, arch, shadeops, optimize, connected, false,
                     false, connected ? 2 : 0);
        if (unit_test_failures != previous_failures)
            print("Gabor module checks failed: {} (OSL {}, LLVM {}, "
                  "connected {})\n",
                  label, osl_optimize, optimize, connected);
    };
    const struct {
        int osl_optimize;
        int llvm_optimize;
        bool connected;
        bool local;
    } variants[] = {
        { 0, 10, false, false },
        { 2, 10, true, true },
        { 2, 3, true, false },
    };
    for (bool periodic : { false, true }) {
        const string_view operation = periodic ? "pnoise" : "noise";
        const string_view family    = periodic ? "gaborpnoise" : "gabornoise";
        const string_view per1      = periodic ? ",2.0" : "";
        const string_view per2      = periodic ? ",2.0,3.0" : "";
        const string_view per3      = periodic ? ",point(2,3,4)" : "";
        const string_view per4      = periodic ? ",point(2,3,4),5.0" : "";
        for (string_view type : { "float", "color" }) {
            // Every dimension has a live result. The extra 2D/4D calls force
            // each partial-coordinate promotion; 4D still ignores time.
            const auto assignment = fmtformat(
                "{0} a={1}(\"gabor\",x{2}); "
                "{0} b={1}(\"gabor\",x,y{3}); "
                "{0} c={1}(\"gabor\",p{4}); "
                "{0} d={1}(\"gabor\",p,t{5}); "
                "{0} e={1}(\"gabor\",x,0.31{3}); "
                "{0} f={1}(\"gabor\",0.23,y{3}); "
                "{0} g={1}(\"gabor\",p,0.19{5}); "
                "{0} h={1}(\"gabor\",point(0.3,0.4,0.5),t{5}); "
                "value=a+0.7*b+0.6*c+0.5*d+0.4*e+0.3*f+0.2*g+0.1*h; ",
                type, operation, per1, per2, per3, per4);
            const string_view coordinates
                = "float x=1.7*u-0.23, y=2.3*v+0.31, t=0.2+u*v; "
                  "point p=point(x,y,0.7+u*v); ";
            const std::string sources[] = {
                fmtformat("shader gabor_values(output color Cout=0) {{ "
                          "{} {} value=0; {} Cout=color(value); }}",
                          coordinates, type, assignment),
                fmtformat("shader gabor_producer(output {} value=0) {{ "
                          "{} {} }}",
                          type, coordinates, assignment),
                fmtformat("shader gabor_consumer({0} value=0, "
                          "output color Cout=0) {{ "
                          "Cout=color(value+Dx(value)+Dy(value))"
                          "+color(filterwidth({1})); }}",
                          type, type == "float" ? "value" : "vector(value)"),
            };
            std::string bytecode[3];
            for (size_t i = 0; i < std::size(sources); ++i) {
                OSLCompiler compiler;
                if (!compiler.compile_buffer(sources[i], bytecode[i], { },
                                             stdosl))
                    return false;
            }
            const auto prefix = fmtformat("osl_{}_{}", family,
                                          type == "float" ? "df" : "dv");
            const std::string helpers[] = {
                prefix + "df" + (periodic ? "f" : ""),
                prefix + "dfdf" + (periodic ? "ff" : ""),
                prefix + "dv" + (periodic ? "v" : ""),
                prefix + "dvdf" + (periodic ? "vf" : ""),
            };
            for (const auto& variant : variants)
                check(fmtformat("{} {}", operation, type),
                      bytecode[variant.connected ? 1 : 0],
                      variant.connected ? bytecode[2] : "",
                      { helpers[0], helpers[1], helpers[2], helpers[3],
                        "osl_init_noise_options", "rs_hart_noise_error" },
                      variant.osl_optimize, variant.llvm_optimize,
                      variant.local);
        }
        {
            const auto source
                = fmtformat("shader gabor_options(output color Cout=0) {{ "
                            "point p=point(0.3+u,0.4+v,0.5+u*v); "
                            "vector direction=vector(1+0.1*u,0.2*v,0.3); "
                            "float a={0}(\"gabor\",p{1},"
                            "\"anisotropic\",int(u>v),\"do_filter\",int(u<0.75),"
                            "\"direction\",direction,\"bandwidth\",0.9+0.2*u,"
                            "\"impulses\",8.0+v); "
                            "color b={0}(\"gabor\",p,0.3+u{2},"
                            "\"anisotropic\",2,\"do_filter\",0,"
                            "\"direction\",color(0.2+v,0.5,0.9),"
                            "\"bandwidth\",2,\"impulses\",8); "
                            "float c={0}(\"gabor\",0.37{3}); "
                            "color d={0}(\"gabor\",point(0.2,0.3,0.4){1},"
                            "\"direction\",normal(1,0,0),\"do_filter\",1,"
                            "\"bandwidth\",-1.0,\"impulses\",1000); "
                            "Cout=color(a+Dx(a)+Dy(a)+c)+b+Dx(b)+Dy(b)+d; }}",
                            operation, per3, per4, per1);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            for (int optimize : { 10, 3 })
                check(fmtformat("{} options and reset", operation), bytecode,
                      "",
                      { fmtformat("osl_{}_dfdv{}", family, periodic ? "v" : ""),
                        fmtformat("osl_{}_dvdvdf{}", family,
                                  periodic ? "vf" : ""),
                        fmtformat("osl_{}_dfdf{}", family, periodic ? "f" : ""),
                        fmtformat("osl_{}_dvdv{}", family, periodic ? "v" : ""),
                        "osl_init_noise_options",
                        "osl_noiseparams_set_anisotropic",
                        "osl_noiseparams_set_do_filter",
                        "osl_noiseparams_set_direction",
                        "osl_noiseparams_set_bandwidth",
                        "osl_noiseparams_set_impulses", "rs_hart_noise_error" },
                      optimize == 10 ? 0 : 2, optimize);
        }
        {
            const auto source
                = fmtformat("shader gabor_defaults(float x=0.25, float y=0.5, "
                            "point p=point(0.2,0.3,0.4), float t=0.75, "
                            "output color Cout=0) {{ "
                            "float a={0}(\"gabor\",x{1}); "
                            "float b={0}(\"gabor\",x,y{2}); "
                            "float c={0}(\"gabor\",p{3}); "
                            "float d={0}(\"gabor\",p,t{4}); "
                            "color e={0}(\"gabor\",x{1}); "
                            "color f={0}(\"gabor\",x,y{2}); "
                            "color g={0}(\"gabor\",p{3}); "
                            "color h={0}(\"gabor\",p,t{4}); "
                            "Cout=color(a+b+c+d)+e+f+g+h; }}",
                            operation, per1, per2, per3, per4);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            for (int optimize : { 10, 3 })
                check(fmtformat("{} constant defaults", operation), bytecode,
                      "",
                      { fmtformat("osl_{}_dfdf{}", family, periodic ? "f" : ""),
                        fmtformat("osl_{}_dfdfdf{}", family,
                                  periodic ? "ff" : ""),
                        fmtformat("osl_{}_dfdv{}", family, periodic ? "v" : ""),
                        fmtformat("osl_{}_dfdvdf{}", family,
                                  periodic ? "vf" : ""),
                        fmtformat("osl_{}_dvdf{}", family, periodic ? "f" : ""),
                        fmtformat("osl_{}_dvdfdf{}", family,
                                  periodic ? "ff" : ""),
                        fmtformat("osl_{}_dvdv{}", family, periodic ? "v" : ""),
                        fmtformat("osl_{}_dvdvdf{}", family,
                                  periodic ? "vf" : ""),
                        "osl_init_noise_options", "rs_hart_noise_error" },
                      2, optimize);
        }
    }
    {
        // Keep these independent branches together: the isolated periodic
        // embedding did not reproduce the GPU seed-wrapping mismatch.
        const string_view source
            = "shader gabor_compound_periodic(float offset_u=0, "
              "float offset_v=0, output color Cout=0) { "
              "float U=u+offset_u, V=v+offset_v; "
              "point p=point(.317+1.3*U,.127+.7*V,.233+.4*U+.2*V); "
              "int probe=(int(8*u+.5)+9*int(4*v+.5))%16; float d=0; "
              "if (probe==9) { "
              "float a=pnoise(\"gabor\",p[0],p[1],3,4); "
              "color b=pnoise(\"gabor\",p[0],p[1],3,4); d=a-b[0]; } "
              "if (probe==10) { "
              "float a=pnoise(\"gabor\",p,point(3,4,5)); "
              "color b=pnoise(\"gabor\",p,point(3,4,5)); d=a-b[0]; } "
              "if (probe==11) { "
              "float a=pnoise(\"gabor\",p,.61+.3*U-.2*V,point(3,4,5),7); "
              "color b=pnoise(\"gabor\",p,.61+.3*U-.2*V,point(3,4,5),7); "
              "d=a-b[0]; } "
              "if (probe==12) { "
              "color a=pnoise(\"gabor\",p[0],3), "
              "b=pnoise(\"gabor\",point(p[0],0,0),point(3,1,1)); "
              "d=a[0]-b[0]; } "
              "if (probe==14) { "
              "color a=pnoise(\"gabor\",p,point(3,4,5)), "
              "b=pnoise(\"gabor\",p,.61+.3*U-.2*V,point(3,4,5),7); "
              "d=a[0]-b[0]; } "
              "if (probe==15) { "
              "color a=pnoise(\"gabor\",p,point(3,4,5)), "
              "b=pnoise(\"gabor\",p,point(3,4,5),\"anisotropic\",0,"
              "\"do_filter\",1,\"direction\",vector(1,0,0),"
              "\"bandwidth\",1,\"impulses\",16); d=a[0]-b[0]; } "
              "Cout=color(d,Dx(d),Dy(d)); }";
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        for (int optimize : { 10, 3 })
            check("compound periodic branches", bytecode, "",
                  { "osl_gaborpnoise_dfdfdfff", "osl_gaborpnoise_dvdfdfff",
                    "osl_gaborpnoise_dfdvv", "osl_gaborpnoise_dvdvv",
                    "osl_gaborpnoise_dfdvdfvf", "osl_gaborpnoise_dvdvdfvf",
                    "osl_gaborpnoise_dvdff", "osl_init_noise_options",
                    "osl_noiseparams_set_anisotropic",
                    "osl_noiseparams_set_do_filter",
                    "osl_noiseparams_set_direction",
                    "osl_noiseparams_set_bandwidth",
                    "osl_noiseparams_set_impulses", "rs_hart_noise_error" },
                  optimize == 10 ? 0 : 2, optimize, optimize == 3);
    }
    const struct {
        const char* body;
        const char* error;
    } rejected[] = {
        { "Cout=noise(\"gabor\",P,\"unknown\",1);", "unsupported noise option" },
        { "Cout=pnoise(\"gabor\",P,point(2),\"\",1);",
          "unsupported noise option" },
        { "Cout=noise(\"gabor\",P,\"anisotropic\",1.0);",
          "unsupported noise option" },
        { "Cout=pnoise(\"gabor\",P,point(2),\"do_filter\",1.0);",
          "unsupported noise option" },
        { "Cout=noise(\"gabor\",P,\"direction\",0.5);",
          "unsupported noise option" },
        { "Cout=pnoise(\"gabor\",P,point(2),\"bandwidth\",color(1));",
          "unsupported noise option" },
        { "Cout=noise(\"gabor\",P,\"impulses\",vector(1));",
          "unsupported noise option" },
        { "Cout=noise(\"gabor\",P,\"do_filter\",\"true\");",
          "unsupported noise option" },
        { "float a[2]={1,2}; Cout=noise(\"gabor\",P,\"bandwidth\",a);",
          "unsupported noise option" },
        { "Cout=noise(\"noise\",P,\"do_filter\",0);",
          "noise options require gabor" },
        { "Cout=pnoise(\"snoise\",P,point(2),\"bandwidth\",1.0);",
          "noise options require gabor" },
        { "string option=u>v?\"bandwidth\":\"impulses\"; "
          "Cout=noise(\"gabor\",P,option,1.0);",
          "noise option names must be literal strings" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source
            = fmtformat("shader bad_gabor(output color Cout=0) {{ {} }}",
                        test.body);
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, false, false, true, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
    }
    for (string_view operation : { "noise", "pnoise" }) {
        const string_view period = operation == "pnoise" ? ",point(2)" : "";
        for (bool selector : { false, true }) {
            const auto source
                = fmtformat("shader dynamic_gabor(string token=\"{0}\", "
                            "output color Cout=0) {{ Cout={1}({2},P{3}{4}); }}",
                            selector ? "gabor" : "bandwidth", operation,
                            selector ? "token" : "\"gabor\"", period,
                            selector ? "" : ",token,1.0");
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            HartServices renderer(false, false, false, false, false, false,
                                  !selector);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            auto group = make_group(ss, bytecode);
            check_rejected_group(
                ss, *group, errors,
                selector ? "HARTNoiseErrors"
                         : "noise option names must be literal strings");
        }
        {
            Diagnostics errors;
            OSLCompiler compiler(&errors);
            std::string bytecode;
            const auto source
                = fmtformat("shader odd_gabor(output color Cout=0) {{ "
                            "Cout={}(\"gabor\",P{},\"bandwidth\"); }}",
                            operation, period);
            OIIO_CHECK_ASSERT(
                !compiler.compile_buffer(source, bytecode, { }, stdosl));
            OIIO_CHECK_ASSERT(errors.errors > 0);
        }
        {
            // Exercise the preoptimization check too, bypassing the source
            // compiler's token/value-pair requirement without false built-ins.
            OSLCompiler compiler;
            std::string bytecode;
            const auto source
                = fmtformat("shader malformed_gabor(output color Cout=0) {{ "
                            "Cout={}(\"gabor\",P{},\"bandwidth\",0.91); }}",
                            operation, period);
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            const auto op    = bytecode.find(fmtformat("\t{}\t", operation));
            const auto end   = bytecode.find('\n', op);
            const auto hints = bytecode.find('%', op);
            OIIO_CHECK_ASSERT(op != std::string::npos
                              && end != std::string::npos && hints < end);
            if (op == std::string::npos || end == std::string::npos
                || hints >= end)
                return false;
            std::vector<std::string> words;
            OIIO::Strutil::split(string_view(bytecode).substr(op, hints - op),
                                 words, "", -1);
            OIIO_CHECK_EQUAL(words.size(), operation == "noise" ? 6 : 7);
            if (words.size() != (operation == "noise" ? 6 : 7))
                return false;
            words.pop_back();
            bytecode.replace(op, end - op,
                             fmtformat("\t{}", OIIO::Strutil::join(words, " ")));
            HartServices renderer(false, false, false, false, false, false,
                                  true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            auto group = make_group(ss, bytecode);
            check_rejected_group(ss, *group, errors,
                                 "invalid noise option list");
        }
    }
    return true;
}



bool
check_matrix_modules(string_view arch, string_view stdosl)
{
    for (string_view type : { "point", "vector", "normal" }) {
        const auto body = fmtformat(
            "matrix m=matrix(1+u,0.25,0,0, 0,2+v,0,0, 0,0,0.5,0, 1,2,3,1); "
            "m[3][0]=u; matrix q=transpose(m); matrix n=(m*q)/q; "
            "value=transform(n,{0}(u,v,1))+{0}(determinant(m)/64);",
            type);
        const std::string sources[] = {
            fmtformat("shader hart_matrix_test(output color Cout=0) {{ "
                      "{} value=0; {} Cout=color(value); }}",
                      type, body),
            fmtformat("shader hart_matrix_producer(output {} value=0) {{ {} }}",
                      type, body),
            fmtformat(
                "shader hart_matrix_consumer({0} value=0, output color Cout=0) {{ "
                "Cout=color(value+Dx(value)+Dy(value)); }}",
                type),
        };
        std::string bytecode[3];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
                return false;
        }
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 })
                for (bool connected : { false, true }) {
                    HartServices renderer;
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("optimize", osl_optimize);
                    ss.attribute("llvm_optimize", optimize);
                    auto group = connected
                                     ? make_connected_group(ss, bytecode[1],
                                                            bytecode[2])
                                     : make_group(ss, bytecode[0]);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "Matrix {}: {}\n", type,
                              errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    const auto transform = fmtformat("osl_transform{}_{}",
                                                     type == "point"    ? ""
                                                     : type == "vector" ? "v"
                                                                        : "n",
                                                     connected ? "dvmdv"
                                                               : "vmv");
                    check_module(ss, *group, arch,
                                 { transform, "osl_mul_mmm", "osl_div_mmm",
                                   "osl_transpose_mm", "osl_determinant_fm" },
                                 optimize, connected);
                }
    }
    const char* sources[] = {
        "shader hart_matrix_producer(output matrix value=1) { "
        "value=matrix(1+u,0,0,0, 0,2+v,0,0, 0,0,0.5,0, 1,2,3,1); }",
        "shader hart_matrix_consumer(matrix value=1, output color Cout=0) { "
        "point p=transform(value,P); Cout=color(p+Dx(p)+Dy(p)+value[3][1]); }",
    };
    std::string bytecode[2];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
            return false;
    }
    for (int optimize : { 10, 3 }) {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("llvm_optimize", optimize);
        auto group = make_connected_group(ss, bytecode[0], bytecode[1]);
        ss.optimize_group(group.get(), nullptr);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        check_module(ss, *group, arch, { "osl_transform_dvmdv" }, optimize,
                     true);
    }
    OSLCompiler compiler;
    std::string rejected;
    if (!compiler.compile_buffer(
            "shader hart_matrix_index(output color Cout=0) { matrix m=matrix(1); "
            "if (u<0) m[int(3*u)][1]=v; Cout=color(m[0][0]); }",
            rejected, { }, stdosl))
        return false;
    check_rejection(arch, rejected, "HARTArrayBounds");
    return true;
}



bool
check_space_modules(string_view arch, string_view stdosl)
{
    for (string_view type : { "point", "vector", "normal" }) {
        const auto body = fmtformat(
            "{0} p={0}(\"object\",u,v,1); "
            "matrix a=matrix(\"shader\",\"common\"); "
            "matrix b=matrix(\"shader\",1.0); "
            "matrix c=matrix(\"object\",1,0,0,0,0,1,0,0,0,0,1,0,u,v,0,1); "
            "matrix d=1; int ok=getmatrix(\"common\",\"object\",d); "
            "value=transform(\"common\",\"shader\",p)"
            "+transform(a*b*c*d,{0}(u,v,1))+{0}(ok);",
            type);
        const std::string sources[] = {
            fmtformat("shader hart_space_test(output color Cout=0) {{ "
                      "{} value=0; {} Cout=color(value+Dx(value)+Dy(value)); }}",
                      type, body),
            fmtformat("shader hart_space_producer(output {} value=0) {{ {} }}",
                      type, body),
            fmtformat(
                "shader hart_space_consumer({0} value=0,output color Cout=0) {{ "
                "Cout=color(value+Dx(value)+Dy(value)); }}",
                type),
        };
        std::string bytecode[3];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
                return false;
        }
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 })
                for (bool connected : { false, true }) {
                    HartServices renderer(false, true);
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("optimize", osl_optimize);
                    ss.attribute("llvm_optimize", optimize);
                    auto group = connected
                                     ? make_connected_group(ss, bytecode[1],
                                                            bytecode[2])
                                     : make_group(ss, bytecode[0]);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "Spaces {}: {}\n", type,
                              errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    check_module(ss, *group, arch,
                                 { "osl_transform_triple",
                                   "osl_get_from_to_matrix",
                                   "osl_prepend_matrix_from",
                                   "rs_get_matrix_space_time",
                                   "rs_get_inverse_matrix_space_time" },
                                 optimize, connected);
                }
    }
    for (string_view body :
         { "Cout=color(point(\"myspace\",u,v,1));",
           "Cout=color(transform(\"bogus\",\"bogus\",P));",
           "matrix m=matrix(\"camera\",\"common\"); Cout=color(m[0][0]);",
           "if (u<0) Cout=color(vector(\"\",u,v,1));" }) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat("shader hart_bad_space(output color Cout=0) {{ {} }}",
                          body),
                bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors,
                             "unsupported coordinate space");
    }
    for (string_view body :
         { "Cout=color(transform(space,P));",
           "matrix m=matrix(space,\"common\"); Cout=color(m[0][0]);",
           "Cout=color(point(space,u,v,1));" }) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat("shader dynamic_space(output color Cout=0) {{ "
                          "string space=u>v?\"object\":\"shader\"; {} }}",
                          body),
                bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors,
                             "coordinate spaces must be literal strings");
    }
    return true;
}



struct NamedTransformExpectation {
    const char* helper;
    const char* from;
    const char* to;
    int semantic = -1;
    int calls    = 1;
};



bool
check_named_transform_ir(ShadingSystem& ss, ShaderGroup& group,
                         cspan<NamedTransformExpectation> expected,
                         int optimize)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_named_transforms"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module = **parsed;
    for (const char* name :
         { "osl_transform_triple_nonlinear", "rs_transform_points" }) {
        const auto* function = module.getFunction(name);
        OIIO_CHECK_ASSERT(!function || function->use_empty());
    }
    const struct {
        const char* name;
        const char* args;
        unsigned result;
        bool callback;
    } signatures[] = {
        { "osl_transform_triple", "ppipilli", 32, false },
        { "osl_get_from_to_matrix", "ppll", 32, false },
        { "osl_prepend_matrix_from", "ppl", 32, false },
        { "rs_get_matrix_space_time", "pplf", 1, true },
        { "rs_get_inverse_matrix_space_time", "pplf", 1, true },
    };
    for (const auto& signature : signatures) {
        const auto* function = module.getFunction(signature.name);
        if (expected.empty()) {
            OIIO_CHECK_ASSERT(!function || function->use_empty());
            continue;
        }
        if (!function || function->use_empty())
            continue;
        if (optimize != 10 && !signature.callback)
            continue;
        OIIO_CHECK_EQUAL(function->isDeclaration(), signature.callback);
        OIIO_CHECK_ASSERT(
            !function->isVarArg()
            && function->getReturnType()->isIntegerTy(signature.result));
        const string_view args(signature.args);
        OIIO_CHECK_EQUAL(function->arg_size(), args.size());
        for (const auto& arg : function->args()) {
            if (arg.getArgNo() >= args.size())
                continue;
            const char kind = args[arg.getArgNo()];
            OIIO_CHECK_ASSERT(
                kind == 'p'
                    ? arg.getType()->isPointerTy()
                          && arg.getType()->getPointerAddressSpace() == 0
                : kind == 'f'
                    ? arg.getType()->isFloatTy()
                    : arg.getType()->isIntegerTy(kind == 'l' ? 64 : 32));
        }
    }
    if (!expected.empty())
        for (const char* name : { "rs_get_matrix_space_time",
                                  "rs_get_inverse_matrix_space_time" }) {
            const auto* function = module.getFunction(name);
            OIIO_CHECK_ASSERT(function && !function->use_empty());
        }
    if (optimize != 10)
        return true;
    auto hash_matches = [](const llvm::Value* value, const char* name) {
        const auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value);
        return name
                   ? constant
                         && constant->getZExtValue() == ustringhash(name).hash()
                   : !llvm::isa<llvm::Constant>(value);
    };
    std::vector<int> seen(expected.size(), 0);
    for (const auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0)
            continue;
        for (const auto& block : function)
            for (const auto& inst : block) {
                const auto* call   = llvm::dyn_cast<llvm::CallBase>(&inst);
                const auto* helper = call ? call->getCalledFunction() : nullptr;
                if (!helper)
                    continue;
                const bool triple = helper->getName() == "osl_transform_triple";
                const bool pair = helper->getName() == "osl_get_from_to_matrix";
                const bool prepend = helper->getName()
                                     == "osl_prepend_matrix_from";
                if (!triple && !pair && !prepend)
                    continue;
                OIIO_CHECK_EQUAL(call->arg_size(), triple ? 8 : pair ? 4 : 3);
                if (call->arg_size() != (triple ? 8 : pair ? 4 : 3))
                    continue;
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 helper->getCallingConv());
                OIIO_CHECK_EQUAL(call->getArgOperand(0)->stripPointerCasts(),
                                 function.getArg(0));
                int semantic = -1;
                if (triple) {
                    const auto* kind = llvm::dyn_cast<llvm::ConstantInt>(
                        call->getArgOperand(7));
                    OIIO_CHECK_ASSERT(kind);
                    if (kind)
                        semantic = int(kind->getSExtValue());
                    for (unsigned a : { 2, 4 }) {
                        const auto* derivatives
                            = llvm::dyn_cast<llvm::ConstantInt>(
                                call->getArgOperand(a));
                        OIIO_CHECK_ASSERT(derivatives && derivatives->isOne());
                    }
                } else {
                    const auto* allocation = llvm::dyn_cast<llvm::AllocaInst>(
                        call->getArgOperand(1)->stripPointerCasts());
                    OIIO_CHECK_ASSERT(allocation);
                    if (allocation) {
                        const auto* count = llvm::dyn_cast<llvm::ConstantInt>(
                            allocation->getArraySize());
                        OIIO_CHECK_EQUAL(allocation->getAddressSpace(), 5);
                        OIIO_CHECK_ASSERT(count);
                        if (count)
                            OIIO_CHECK_ASSERT(
                                module.getDataLayout()
                                        .getTypeAllocSize(
                                            allocation->getAllocatedType())
                                        .getFixedValue()
                                    * count->getZExtValue()
                                >= sizeof(Matrix44));
                    }
                }
                int matches = 0;
                for (size_t i = 0; i < expected.size(); ++i) {
                    const auto& test = expected[i];
                    if (helper->getName() != test.helper
                        || semantic != test.semantic
                        || !hash_matches(call->getArgOperand(triple ? 5 : 2),
                                         test.from)
                        || (!prepend
                            && !hash_matches(call->getArgOperand(triple ? 6 : 3),
                                             test.to)))
                        continue;
                    ++matches;
                    ++seen[i];
                }
                OIIO_CHECK_EQUAL(matches, 1);
            }
    }
    for (size_t i = 0; i < expected.size(); ++i)
        OIIO_CHECK_EQUAL(seen[i], expected[i].calls);
    return true;
}



bool
check_named_transform_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader named_literal(output color Cout=0) { "
        "point p=point(\"model\",u,v,1); vector q=vector(\"basis\",u,1,v); "
        "normal n=normal(\"world\",1+u,1+v,1); "
        "matrix a=matrix(\"model\",\"camera\"), b=matrix(\"basis\",1+u); "
        "matrix c=matrix(\"model\",1,0,0,0,0,1,0,0,0,0,1,0,u,v,0,1), d=1; "
        "int ok=getmatrix(\"common\",\"screen\",d); "
        "point pp=transform(\"model\",\"camera\",p); "
        "vector qq=transform(\"basis\",\"NDC\",q); "
        "normal nn=transform(\"world\",\"raster\",n); "
        "Cout=color(pp+Dx(pp)+Dy(pp))+color(qq+Dx(qq)+Dy(qq))"
        "+color(nn+Dx(nn)+Dy(nn))+color(a[0][0]+b[0][0]+c[3][0]+d[1][1]+ok); }",
        "shader named_dynamic(output color Cout=0) { "
        "string spaces[3]={\"model\",\"basis\",\"world\"}; "
        "string from=spaces[int(2*u)], to=u>v?\"camera\":\"common\"; "
        "point p=point(from,u,v,1); vector q=vector(from,u,1,v); "
        "normal n=normal(from,1,u,v); matrix a=matrix(from,to), b=matrix(from,1); "
        "matrix c=matrix(from,1,0,0,0,0,1,0,0,0,0,1,0,u,v,0,1), d=1; "
        "int ok=getmatrix(from,to,d); point pp=transform(from,to,p); "
        "vector qq=transform(from,to,q); normal nn=transform(from,to,n); "
        "Cout=color(pp+Dx(pp)+Dy(pp))+color(qq+Dx(qq)+Dy(qq))"
        "+color(nn+Dx(nn)+Dy(nn))+color(a[0][0]+b[0][0]+c[3][0]+d[1][1]+ok); }",
        "shader named_alias(output color Cout=0) { "
        "matrix a=matrix(\"world\",\"common\"), b=matrix(\"common\",\"world\"); "
        "matrix c=matrix(\"world\",1); "
        "matrix d=matrix(\"world\",1,0,0,0,0,1,0,0,0,0,1,0,1,2,3,1), e=1, f=1; "
        "int ok=getmatrix(\"world\",\"common\",e); "
        "ok+=getmatrix(\"common\",\"world\",f); "
        "point p=point(\"world\",1,2,3), pp=transform(\"world\",\"common\",P); "
        "vector q=vector(\"world\",2,3,4); "
        "vector qq=transform(\"common\",\"world\",vector(u,v,1)); "
        "normal n=normal(\"world\",3,4,5); "
        "normal nn=transform(\"world\",\"common\",normal(u,v,1)); "
        "Cout=color(p+Dx(p)+Dy(p)+pp+Dx(pp)+Dy(pp))"
        "+color(q+Dx(q)+Dy(q)+qq+Dx(qq)+Dy(qq))"
        "+color(n+Dx(n)+Dy(n)+nn+Dx(nn)+Dy(nn))"
        "+color(a[0][0]+b[0][0]+c[0][0]+d[3][0]+e[0][0]+f[0][0]+ok); }",
        "shader named_identity(output color Cout=0) { "
        "matrix a=matrix(\"model\",\"model\"), b=matrix(\"common\",1); "
        "matrix c=matrix(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1), d=1; "
        "int ok=getmatrix(\"model\",\"model\",d); "
        "point p=point(\"common\",1,2,3); "
        "point q=transform(\"model\",\"model\",p); "
        "vector direction=transform(matrix(1),vector(u,v,1)); "
        "Cout=color(q)+color(transform(a*b*c*d,direction))+color(ok); }",
        "shader named_unusual(output color Cout=0) { "
        "string name=u>v?\"model\":\"world\"; "
        "matrix a=matrix(name,\"$unknown1$\"), b=matrix(\"$unknown2$\",name); "
        "matrix c=matrix(\"\",name), d=matrix(\"unregistered\",\"common\"); "
        "Cout=color(a[0][0]+b[0][0]+c[0][0]+d[0][0]); }",
        "shader named_clean(output color Cout=0) { Cout=color(u,v,1); }",
    };
    std::string oso[std::size(sources)];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, stdosl))
            return false;
    }
    const NamedTransformExpectation literal[] = {
        { "osl_transform_triple", "model", "common", TypeDesc::POINT },
        { "osl_transform_triple", "basis", "common", TypeDesc::VECTOR },
        { "osl_transform_triple", "world", "common", TypeDesc::NORMAL },
        { "osl_transform_triple", "model", "camera", TypeDesc::POINT },
        { "osl_transform_triple", "basis", "NDC", TypeDesc::VECTOR },
        { "osl_transform_triple", "world", "raster", TypeDesc::NORMAL },
        { "osl_get_from_to_matrix", "model", "camera" },
        { "osl_get_from_to_matrix", "common", "screen" },
        { "osl_prepend_matrix_from", "basis", nullptr },
        { "osl_prepend_matrix_from", "model", nullptr },
    };
    const NamedTransformExpectation dynamic[] = {
        { "osl_transform_triple", nullptr, "common", TypeDesc::POINT },
        { "osl_transform_triple", nullptr, "common", TypeDesc::VECTOR },
        { "osl_transform_triple", nullptr, "common", TypeDesc::NORMAL },
        { "osl_transform_triple", nullptr, nullptr, TypeDesc::POINT },
        { "osl_transform_triple", nullptr, nullptr, TypeDesc::VECTOR },
        { "osl_transform_triple", nullptr, nullptr, TypeDesc::NORMAL },
        { "osl_get_from_to_matrix", nullptr, nullptr, -1, 2 },
        { "osl_prepend_matrix_from", nullptr, nullptr, -1, 2 },
    };
    const NamedTransformExpectation alias[] = {
        { "osl_transform_triple", "world", "common", TypeDesc::POINT, 2 },
        { "osl_transform_triple", "world", "common", TypeDesc::VECTOR },
        { "osl_transform_triple", "world", "common", TypeDesc::NORMAL, 2 },
        { "osl_transform_triple", "common", "world", TypeDesc::VECTOR },
        { "osl_get_from_to_matrix", "world", "common", -1, 2 },
        { "osl_get_from_to_matrix", "common", "world", -1, 2 },
        { "osl_prepend_matrix_from", "world", nullptr, -1, 2 },
    };
    const NamedTransformExpectation unusual[] = {
        { "osl_get_from_to_matrix", nullptr, "$unknown1$" },
        { "osl_get_from_to_matrix", "$unknown2$", nullptr },
        { "osl_get_from_to_matrix", "", nullptr },
        { "osl_get_from_to_matrix", "unregistered", "common" },
    };
    const cspan<NamedTransformExpectation> expectations[]
        = { literal, dynamic, alias, { }, unusual };
    const struct {
        int source, osl, llvm;
        bool local;
        const char* common = "world";
    } variants[] = {
        { 0, 0, 10, false }, { 0, 2, 10, true },
        { 0, 2, 3, false },  { 1, 0, 10, false },
        { 1, 2, 10, true },  { 1, 2, 3, true },
        { 2, 0, 10, false }, { 2, 2, 10, true },
        { 2, 2, 3, true },   { 2, 2, 10, false, "other_common" },
        { 3, 2, 10, false }, { 3, 2, 3, true },
        { 4, 0, 10, false }, { 4, 2, 10, true },
    };
    for (const auto& variant : variants) {
        HartNamedTransformServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
        OIIO_CHECK_ASSERT(ss.attribute("commonspace", variant.common));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0));
        auto group = make_group(ss, oso[variant.source]);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Named transforms source {} OSL{} LLVM{}: {}\n",
                  variant.source, variant.osl, variant.llvm, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        check_module(ss, *group, arch, { }, variant.llvm, false, false, false,
                     0, false, true);
        if (!check_named_transform_ir(ss, *group, expectations[variant.source],
                                      variant.llvm))
            return false;
        int allocated = -1, group_size = 0;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", group_size));
        OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
        OIIO_CHECK_EQUAL(allocated, variant.local ? group_size : 0);
        OIIO_CHECK_EQUAL(renderer.matrix_queries, 0);
        OIIO_CHECK_EQUAL(renderer.nonlinear_queries, 0);
    }
    for (int failure = 0; failure < 3; ++failure) {
        HartNamedTransformServices renderer;
        renderer.transforms = failure != 0;
        renderer.named      = failure == 0;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
        auto group = make_group(ss, oso[0]);
        if (failure == 2) {
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("named_clean", oso[5]));
            group = ss.ShaderGroupBegin("hart_test_group");
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "unused"));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "named_clean", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("layer0.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(group.get(), { &output, 1 });
        }
        check_rejected_group(ss, *group, errors,
                             failure == 0 ? "renderer lacks HARTTransforms"
                                          : "unsupported coordinate space");
        OIIO_CHECK_EQUAL(renderer.matrix_queries, 0);
        OIIO_CHECK_EQUAL(renderer.nonlinear_queries, 0);
    }
    const struct {
        const char* body;
        const char* opcode;
        unsigned operand;
        const char* replacement;
    } malformed[] = {
        { "m=matrix(name,\"common\");", "matrix", 2, "names" },
        { "m=matrix(name,\"common\");", "matrix", 1, "p" },
        { "m=matrix(name,x);", "matrix", 2, "x" },
        { "m=matrix(name,x);", "matrix", 3, "floats" },
        { "m=matrix(x);", "assign", 0, "transform" },
        { "m=matrix(name,\"common\");", "matrix", 0, "getmatrix" },
        { "m=matrix(1,0,0,0,0,1,0,0,0,0,1,0,x,0,0,1);", "matrix", 2, "floats" },
        { "m=matrix(name,1,0,0,0,0,1,0,0,0,0,1,0,x,0,0,1);", "matrix", 2,
          "names" },
        { "ok=getmatrix(name,\"common\",m);", "getmatrix", 4, "p" },
        { "ok=getmatrix(name,\"common\",m);", "getmatrix", 1, "x" },
        { "p=point(name,x,1,2);", "point", 2, "names" },
        { "direction=vector(name,x,1,2);", "vector", 3, "floats" },
        { "n=normal(name,x,1,2);", "normal", 1, "m" },
        { "p=transform(name,\"common\",p);", "transform", 2, "names" },
        { "direction=transform(name,\"common\",direction);", "transformv", 4,
          "floats" },
        { "n=transform(name,\"common\",n);", "transformn", 1, "name" },
        { "p=transform(m,p);", "transform", 2, "names" },
    };
    for (const auto& test : malformed) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source = fmtformat(
            "shader named_bad(string name=\"model\", string names[2]={{\"a\",\"b\"}}, "
            "float x=0.2, float floats[2]={{1,2}}, output matrix m=1, output point p=0, "
            "output vector direction=0, output normal n=0, output int ok=0, output color Cout=0) {{ {} "
            "Cout=color(p)+color(direction)+color(n)+color(m[0][0]+ok+x); }}",
            test.body);
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        // Matrix and triple type declarations also contain the opcode names.
        const auto line  = bytecode.find(fmtformat("\n\t{}\t", test.opcode));
        const auto op    = line == std::string::npos ? line : line + 1;
        const auto end   = bytecode.find('\n', op);
        const auto hints = bytecode.find('%', op);
        OIIO_CHECK_ASSERT(op != std::string::npos && end != std::string::npos
                          && hints < end);
        if (op == std::string::npos || end == std::string::npos
            || hints >= end) {
            print(stderr, "Missing '{}' in named fixture '{}':\n{}\n",
                  test.opcode, test.body, bytecode);
            return false;
        }
        std::vector<std::string> words;
        OIIO::Strutil::split(string_view(bytecode).substr(op, hints - op),
                             words, "", -1);
        OIIO_CHECK_ASSERT(test.operand < words.size());
        if (test.operand >= words.size())
            return false;
        words[test.operand] = test.replacement;
        bytecode.replace(op, hints - op,
                         fmtformat("\t{}\t", OIIO::Strutil::join(words, "\t")));
        HartNamedTransformServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
        if (!ss.LoadMemoryCompiledShader("hart_test", bytecode)) {
            print(stderr,
                  "Named fixture '{}' operand {} replacement '{}': {}\n",
                  test.body, test.operand, test.replacement, errors.messages);
            return false;
        }
        auto group = make_group(ss, bytecode, 1, false);
        check_rejected_group(ss, *group, errors,
                             "invalid coordinate transform operands");
        OIIO_CHECK_EQUAL(renderer.matrix_queries, 0);
        OIIO_CHECK_EQUAL(renderer.nonlinear_queries, 0);
    }
    return true;
}



bool
check_geometry_state_ir(ShadingSystem& ss, ShaderGroup& group,
                        bool zero_derivatives, bool connected)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_geometry_state"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module       = **parsed;
    const auto& layout = module.getDataLayout();
    auto* sg = llvm::StructType::getTypeByName(context, "ShaderGlobals");
    const struct {
        unsigned index;
        size_t offset;
        unsigned words;
        bool writable;
    } fields[] = {
        { 0, offsetof(ShaderGlobals, P), 9, true },
        { 2, offsetof(ShaderGlobals, I), 9, true },
        { 3, offsetof(ShaderGlobals, N), 3, true },
        { 4, offsetof(ShaderGlobals, Ng), 3, true },
        { 5, offsetof(ShaderGlobals, u), 3, true },
        { 6, offsetof(ShaderGlobals, v), 3, true },
        { 7, offsetof(ShaderGlobals, dPdu), 3, true },
        { 8, offsetof(ShaderGlobals, dPdv), 3, true },
        { 9, offsetof(ShaderGlobals, time), 1, false },
        { 10, offsetof(ShaderGlobals, dtime), 1, false },
        { 11, offsetof(ShaderGlobals, dPdtime), 3, false },
        { 24, offsetof(ShaderGlobals, surfacearea), 1, false },
        { 27, offsetof(ShaderGlobals, backfacing), 1, false },
    };
    if (!zero_derivatives) {
        const auto* ray = module.getFunction("osl_raytype_bit");
        OIIO_CHECK_ASSERT(ray && !ray->isDeclaration() && !ray->use_empty());
        OIIO_CHECK_ASSERT(sg && sg->getNumElements() == 28);
        if (!sg || sg->getNumElements() != 28)
            return false;
        const auto* offsets = layout.getStructLayout(sg);
        OIIO_CHECK_EQUAL(layout.getTypeAllocSize(sg).getFixedValue(),
                         sizeof(ShaderGlobals));
        OIIO_CHECK_EQUAL(offsets->getElementOffset(25),
                         offsetof(ShaderGlobals, raytype));
        OIIO_CHECK_ASSERT(sg->getElementType(24)->isFloatTy()
                          && sg->getElementType(25)->isIntegerTy(32)
                          && sg->getElementType(27)->isIntegerTy(32));
        for (const auto& field : fields) {
            OIIO_CHECK_EQUAL(offsets->getElementOffset(field.index),
                             field.offset);
            OIIO_CHECK_EQUAL(
                layout.getTypeAllocSize(sg->getElementType(field.index))
                    .getFixedValue(),
                field.words * sizeof(float));
        }
    }
    struct Access {
        const llvm::Instruction* instruction;
        int64_t offset;
        uint64_t length;
        bool write;
    };
    int writers = 0, readers = 0, zero_outputs = 0;
    for (auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0)
            continue;
        OIIO_CHECK_EQUAL(function.arg_size(), 6);
        if (function.arg_size() != 6)
            continue;
        std::vector<Access> accesses;
        auto access = [&](const llvm::Instruction& instruction,
                          const llvm::Value* pointer, uint64_t length,
                          bool write) {
            int64_t offset = 0;
            if (!interactive_address(pointer, function.getArg(0), layout,
                                     offset))
                return;
            accesses.push_back({ &instruction, offset, length, write });
            bool allowed = false;
            for (const auto& field : fields)
                allowed |= (!write || field.writable)
                           && offset >= int64_t(field.offset)
                           && uint64_t(offset) + length
                                  <= field.offset + field.words * sizeof(float);
            OIIO_CHECK_ASSERT(allowed);
            // A derivative address must stay inside its actual SG field,
            // not alias the next writable global.
            const llvm::Value* base = pointer;
            for (unsigned depth = 0; base && depth < 16; ++depth) {
                if (const auto* cast = llvm::dyn_cast<llvm::CastInst>(base)) {
                    base = cast->getOperand(0);
                } else if (const auto* gep
                           = llvm::dyn_cast<llvm::GetElementPtrInst>(base)) {
                    if (sg && gep->getSourceElementType() == sg
                        && gep->getNumIndices() >= 2) {
                        const auto* index = llvm::dyn_cast<llvm::ConstantInt>(
                            gep->getOperand(2));
                        OIIO_CHECK_ASSERT(index
                                          && index->getZExtValue()
                                                 < sg->getNumElements());
                        if (index
                            && index->getZExtValue() < sg->getNumElements()) {
                            const unsigned i = unsigned(index->getZExtValue());
                            const auto start
                                = layout.getStructLayout(sg)->getElementOffset(
                                    i);
                            const auto extent
                                = layout.getTypeAllocSize(sg->getElementType(i))
                                      .getFixedValue();
                            OIIO_CHECK_ASSERT(offset >= int64_t(start)
                                              && uint64_t(offset) + length
                                                     <= start + extent);
                        }
                    }
                    base = gep->getPointerOperand();
                } else {
                    break;
                }
            }
        };
        for (const auto& block : function)
            for (const auto& inst : block) {
                if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&inst))
                    access(inst, load->getPointerOperand(),
                           layout.getTypeStoreSize(load->getType())
                               .getFixedValue(),
                           false);
                if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
                    access(inst, store->getPointerOperand(),
                           layout
                               .getTypeStoreSize(
                                   store->getValueOperand()->getType())
                               .getFixedValue(),
                           true);
                }
                if (const auto* copy = llvm::dyn_cast<llvm::MemTransferInst>(
                        &inst)) {
                    const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                        copy->getLength());
                    OIIO_CHECK_ASSERT(length);
                    if (!length)
                        continue;
                    access(inst, copy->getRawSource(), length->getZExtValue(),
                           false);
                    access(inst, copy->getRawDest(), length->getZExtValue(),
                           true);
                    if (!zero_derivatives)
                        continue;
                    // Prove the live output is zero, following the lowering's
                    // unoptimized temporary stores rather than assuming folding.
                    const auto* address = llvm::dyn_cast<llvm::IntToPtrInst>(
                        copy->getRawDest()->stripPointerCasts());
                    const auto* sum
                        = address ? llvm::dyn_cast<llvm::BinaryOperator>(
                                        address->getOperand(0))
                                  : nullptr;
                    bool output = false;
                    if (sum && sum->getOpcode() == llvm::Instruction::Add)
                        for (unsigned i = 0; i < 2; ++i)
                            if (const auto* base
                                = llvm::dyn_cast<llvm::PtrToIntInst>(
                                    sum->getOperand(i)))
                                output
                                    |= base->getOperand(0)->stripPointerCasts()
                                       == function.getArg(3);
                    if (!output)
                        continue;
                    ++zero_outputs;
                    OIIO_CHECK_EQUAL(length->getZExtValue(), 3 * sizeof(float));
                    int64_t source_offset = 0;
                    const auto* source = llvm::GetPointerBaseWithConstantOffset(
                        copy->getRawSource(), source_offset, layout);
                    if (const auto* global
                        = llvm::dyn_cast<llvm::GlobalVariable>(source)) {
                        OIIO_CHECK_ASSERT(
                            global->hasInitializer()
                            && global->getInitializer()->isNullValue());
                        OIIO_CHECK_ASSERT(
                            source_offset >= 0
                            && uint64_t(source_offset) + length->getZExtValue()
                                   <= layout
                                          .getTypeAllocSize(
                                              global->getValueType())
                                          .getFixedValue());
                    } else {
                        llvm::DominatorTree dominators(function);
                        auto last_store = [&](const llvm::Value* base,
                                              int64_t offset,
                                              const llvm::Instruction* before) {
                            const llvm::StoreInst* latest = nullptr;
                            for (const auto& candidate_block : function)
                                for (const auto& candidate : candidate_block) {
                                    const auto* store
                                        = llvm::dyn_cast<llvm::StoreInst>(
                                            &candidate);
                                    if (!store
                                        || !dominators.dominates(store, before))
                                        continue;
                                    int64_t position = 0;
                                    const auto* target
                                        = llvm::GetPointerBaseWithConstantOffset(
                                            store->getPointerOperand(),
                                            position, layout);
                                    if (target != base || position != offset)
                                        continue;
                                    if (!latest
                                        || dominators.dominates(latest, store))
                                        latest = store;
                                    else if (!dominators.dominates(store,
                                                                   latest))
                                        return static_cast<
                                            const llvm::StoreInst*>(nullptr);
                                }
                            return latest;
                        };
                        auto zero_value = [&](auto&& self,
                                              const llvm::Value* value,
                                              unsigned depth) -> bool {
                            if (depth > 128)
                                return false;
                            if (const auto* number
                                = llvm::dyn_cast<llvm::ConstantFP>(value))
                                return number->isZero();
                            if (const auto* sum
                                = llvm::dyn_cast<llvm::BinaryOperator>(value))
                                return sum->getOpcode()
                                           == llvm::Instruction::FAdd
                                       && self(self, sum->getOperand(0),
                                               depth + 1)
                                       && self(self, sum->getOperand(1),
                                               depth + 1);
                            if (const auto* load
                                = llvm::dyn_cast<llvm::LoadInst>(value)) {
                                int64_t offset = 0;
                                const auto* base
                                    = llvm::GetPointerBaseWithConstantOffset(
                                        load->getPointerOperand(), offset,
                                        layout);
                                const auto* store = last_store(base, offset,
                                                               load);
                                return store
                                       && self(self, store->getValueOperand(),
                                               depth + 1);
                            }
                            return false;
                        };
                        for (int c = 0; c < 3; ++c) {
                            const auto* store = last_store(
                                source, source_offset + c * int(sizeof(float)),
                                copy);
                            OIIO_CHECK_ASSERT(
                                store
                                && zero_value(zero_value,
                                              store->getValueOperand(), 0));
                        }
                    }
                }
                if (const auto* clear = llvm::dyn_cast<llvm::MemSetInst>(
                        &inst)) {
                    const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                        clear->getLength());
                    OIIO_CHECK_ASSERT(length);
                    if (length)
                        access(inst, clear->getRawDest(),
                               length->getZExtValue(), true);
                }
            }
        if (zero_derivatives) {
            OIIO_CHECK_ASSERT(accesses.empty());
            continue;
        }
        const bool writes = std::any_of(accesses.begin(), accesses.end(),
                                        [](const Access& a) { return a.write; });
        if (!writes && !connected)
            continue;
        if (writes) {
            ++writers;
        } else {
            ++readers;
        }
        llvm::DominatorTree dominators(function);
        for (const auto& field : fields) {
            if (!writes && !field.writable)
                continue;
            for (unsigned c = 0; c < field.words; ++c) {
                const auto offset = field.offset + c * sizeof(float);
                bool read = false, write = false, read_before_write = false;
                auto covers = [&](const Access& a) {
                    return a.offset <= int64_t(offset)
                           && int64_t(offset + sizeof(float))
                                  <= a.offset + int64_t(a.length);
                };
                for (const auto& a : accesses) {
                    if (!covers(a))
                        continue;
                    read |= !a.write;
                    write |= a.write;
                    if (a.write)
                        for (const auto& b : accesses)
                            read_before_write
                                |= !b.write && covers(b)
                                   && dominators.dominates(b.instruction,
                                                           a.instruction);
                }
                OIIO_CHECK_ASSERT(read);
                OIIO_CHECK_EQUAL(write, writes && field.writable);
                if (writes && field.writable)
                    OIIO_CHECK_ASSERT(read_before_write);
            }
        }
    }
    OIIO_CHECK_EQUAL(writers, zero_derivatives ? 0 : 1);
    OIIO_CHECK_EQUAL(readers, connected ? 1 : 0);
    if (zero_derivatives)
        OIIO_CHECK_EQUAL(zero_outputs, 1);
    return true;
}



uint32_t
geometry_ray_bit(cspan<ustring> names, ustring name)
{
    OIIO_CHECK_ASSERT(names.size() <= 32);
    if (names.size() > 32)
        return 0;
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == name || (names[i].empty() && name.empty()))
            return uint32_t(1) << i;
    return 0;
}



bool
check_geometry_ray_ir(ShadingSystem& ss, ShaderGroup& group,
                      cspan<ustring> names, bool dynamic, bool constants)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_geometry_rays"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module       = **parsed;
    const auto* legacy = module.getFunction("osl_raytype_name");
    OIIO_CHECK_ASSERT(!legacy || legacy->use_empty());
    const auto* helper = module.getFunction("osl_raytype_bit");
    OIIO_CHECK_ASSERT(helper && !helper->isDeclaration()
                      && !helper->use_empty());
    if (helper)
        OIIO_CHECK_EQUAL(helper->arg_size(), 2);
    if (!helper || helper->isDeclaration() || helper->arg_size() != 2)
        return false;
    OIIO_CHECK_ASSERT(helper->getReturnType()->isIntegerTy(32));
    OIIO_CHECK_ASSERT(!helper->isVarArg());
    OIIO_CHECK_ASSERT(helper->getArg(0)->getType()->isPointerTy()
                      && helper->getArg(0)->getType()->getPointerAddressSpace()
                             == 0);
    OIIO_CHECK_ASSERT(helper->getArg(1)->getType()->isIntegerTy(32));
    bool ray_load = false, bit_and = false;
    for (const auto& block : *helper)
        for (const auto& inst : block) {
            if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
                int64_t offset = 0;
                ray_load
                    |= load->getType()->isIntegerTy(32)
                       && interactive_address(load->getPointerOperand(),
                                              helper->getArg(0),
                                              module.getDataLayout(), offset)
                       && offset == int64_t(offsetof(ShaderGlobals, raytype));
            }
            if (const auto* op = llvm::dyn_cast<llvm::BinaryOperator>(&inst))
                if (op->getOpcode() == llvm::Instruction::And)
                    for (unsigned i = 0; i < 2; ++i) {
                        const auto* load = llvm::dyn_cast<llvm::LoadInst>(
                            op->getOperand(i));
                        int64_t offset = 0;
                        bit_and
                            |= load && load->getType()->isIntegerTy(32)
                               && op->getOperand(1 - i) == helper->getArg(1)
                               && interactive_address(load->getPointerOperand(),
                                                      helper->getArg(0),
                                                      module.getDataLayout(),
                                                      offset)
                               && offset
                                      == int64_t(
                                          offsetof(ShaderGlobals, raytype));
                    }
            if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst))
                OIIO_CHECK_ASSERT(call->getCalledFunction()
                                  && call->getCalledFunction()->isIntrinsic());
        }
    OIIO_CHECK_ASSERT(ray_load && bit_and);
    std::vector<uint32_t> masks;
    int dynamic_calls              = 0;
    const ustring constant_names[] = {
        ustring("camera"),           ustring("edge31"), ustring("duplicate"),
        ustring("hart-missing-ray"), ustring(),         ustring("")
    };
    for (const auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0)
            continue;
        for (const auto& block : function)
            for (const auto& inst : block) {
                const auto* call = llvm::dyn_cast<llvm::CallBase>(&inst);
                if (!call || call->getCalledFunction() != helper)
                    continue;
                OIIO_CHECK_EQUAL(call->arg_size(), 2);
                OIIO_CHECK_EQUAL(call->getCallingConv(),
                                 helper->getCallingConv());
                if (call->arg_size() != 2)
                    continue;
                OIIO_CHECK_EQUAL(call->getArgOperand(0), function.getArg(0));
                const auto* mask = call->getArgOperand(1);
                OIIO_CHECK_ASSERT(mask->getType()->isIntegerTy(32));
                if (const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                        mask)) {
                    masks.push_back(uint32_t(value->getZExtValue()));
                    continue;
                }
                ++dynamic_calls;
                const llvm::Value* selector = nullptr;
                const llvm::Value* tail     = mask;
                std::vector<std::pair<uint64_t, uint32_t>> choices;
                for (size_t i = 0; i < names.size(); ++i) {
                    const auto* select = llvm::dyn_cast<llvm::SelectInst>(tail);
                    OIIO_CHECK_ASSERT(select);
                    if (!select)
                        break;
                    const auto* cmp = llvm::dyn_cast<llvm::ICmpInst>(
                        select->getCondition());
                    const auto* bit = llvm::dyn_cast<llvm::ConstantInt>(
                        select->getTrueValue());
                    OIIO_CHECK_ASSERT(
                        cmp && cmp->getPredicate() == llvm::CmpInst::ICMP_EQ
                        && bit);
                    if (!cmp || !bit)
                        break;
                    const auto* hash = llvm::dyn_cast<llvm::ConstantInt>(
                        cmp->getOperand(1));
                    const auto* name = cmp->getOperand(0);
                    if (!hash) {
                        hash = llvm::dyn_cast<llvm::ConstantInt>(name);
                        name = cmp->getOperand(1);
                    }
                    OIIO_CHECK_ASSERT(hash && hash->getType()->isIntegerTy(64)
                                      && name->getType()->isIntegerTy(64)
                                      && !llvm::isa<llvm::Constant>(name));
                    if (!hash)
                        break;
                    if (!selector)
                        selector = name;
                    OIIO_CHECK_EQUAL(name, selector);
                    OIIO_CHECK_EQUAL(hash->getZExtValue(), names[i].hash());
                    OIIO_CHECK_EQUAL(bit->getZExtValue(), uint32_t(1) << i);
                    choices.emplace_back(hash->getZExtValue(),
                                         uint32_t(bit->getZExtValue()));
                    tail = select->getFalseValue();
                }
                OIIO_CHECK_EQUAL(choices.size(), names.size());
                const auto* fallback = llvm::dyn_cast<llvm::ConstantInt>(tail);
                OIIO_CHECK_ASSERT(fallback && fallback->isZero());
                auto selected_bit = [&](ustring name) {
                    for (const auto& choice : choices)
                        if (choice.first == name.hash())
                            return choice.second;
                    return uint32_t(0);
                };
                for (ustring name : names)
                    OIIO_CHECK_EQUAL(selected_bit(name),
                                     geometry_ray_bit(names, name));
                for (ustring name : constant_names) {
                    OIIO_CHECK_EQUAL(selected_bit(name),
                                     geometry_ray_bit(names, name));
                    OIIO_CHECK_EQUAL(selected_bit(name),
                                     uint32_t(ss.raytype_bit(name)));
                }
            }
    }
    OIIO_CHECK_EQUAL(dynamic_calls > 0, dynamic);
    if (constants) {
        for (ustring name : constant_names)
            OIIO_CHECK_ASSERT(std::find(masks.begin(), masks.end(),
                                        geometry_ray_bit(names, name))
                              != masks.end());
        for (uint32_t mask : masks)
            OIIO_CHECK_ASSERT(
                std::any_of(std::begin(constant_names),
                            std::end(constant_names), [&](ustring name) {
                                return mask == geometry_ray_bit(names, name);
                            }));
    } else {
        // Dynamic fixtures also query the literal empty name in the same
        // module, so both lowering paths must select the first empty slot.
        OIIO_CHECK_ASSERT(!masks.empty());
        for (uint32_t mask : masks)
            OIIO_CHECK_EQUAL(mask, geometry_ray_bit(names, ustring()));
    }
    return true;
}



bool
check_geometry_state_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_geometry_state(output float value=0,output color Cout=0) { "
        "float uu=2*u+v, vv=v-0.5*u; u=uu; v=vv; "
        "P=P+vector(u,v,u*v)+dPdtime*(time+dtime); P[2]+=u*v; I=I+vector(P); "
        "N=N+normal(u,v,1)+normal(backfacing(),surfacearea(),raytype(\"camera\")); "
        "Ng=Ng+normal(v,u,2); "
        "dPdu=dPdu+vector(N)+vector(P); dPdv=dPdv+vector(Ng)+I; "
        "vector dxP=Dx(P),dyP=Dy(P),dxI=Dx(I),dyI=Dy(I); "
        "value=u; float sum=u+v+P[0]+P[1]+P[2]+I[0]+I[1]+I[2]"
        "+N[0]+N[1]+N[2]+Ng[0]+Ng[1]+Ng[2]"
        "+dPdu[0]+dPdu[1]+dPdu[2]+dPdv[0]+dPdv[1]+dPdv[2]"
        "+time+dtime+dPdtime[0]+dPdtime[1]+dPdtime[2]"
        "+backfacing()+surfacearea()+raytype(\"camera\"); "
        "Cout=color(sum,Dx(u)+Dy(u)+Dx(v)+Dy(v)"
        "+dxP[0]+dxP[1]+dxP[2]+dyP[0]+dyP[1]+dyP[2],"
        "dxI[0]+dxI[1]+dxI[2]+dyI[0]+dyI[1]+dyI[2]); }",
        "shader hart_geometry_state_consumer(float value=0,output color Cout=0) { "
        "Cout=color(value+u+v)+color(P)+color(I)+color(N)+color(Ng)"
        "+color(dPdu)+color(dPdv)+color(Dx(P))+color(Dy(P))"
        "+color(Dx(I))+color(Dy(I))"
        "+color(Dx(u)+Dy(u)+Dx(v)+Dy(v)+Dx(value)+Dy(value)); }",
        "shader hart_geometry_zero_derivs(output color Cout=1) { "
        "Cout=color(Dx(time)+Dy(time)+Dx(dtime)+Dy(dtime))"
        "+color(Dx(dPdtime)+Dy(dPdtime)+Dx(N)+Dy(N)"
        "+Dx(Ng)+Dy(Ng)+Dx(dPdu)+Dy(dPdu)+Dx(dPdv)+Dy(dPdv)); }",
        "shader hart_geometry_ray_producer(output string value=\"\") { "
        "string names[5]={\"camera\",\"edge31\",\"duplicate\","
        "\"hart-missing-ray\",\"\"}; value=names[min(4,max(0,int(5*u)))]; }",
        "shader hart_geometry_ray_consumer(string value=\"\",output color Cout=0) { "
        "Cout=color(raytype(value),backfacing(),surfacearea()+raytype(\"\")); }",
        "shader hart_geometry_ray_dynamic(output color Cout=0) { "
        "string names[5]={\"camera\",\"edge31\",\"duplicate\","
        "\"hart-missing-ray\",\"\"}; "
        "Cout=color(raytype(names[min(4,max(0,int(5*u)))]),"
        "backfacing(),surfacearea()+raytype(\"\")); }",
        "shader hart_geometry_ray_constant(output color Cout=0) { "
        "Cout=color(raytype(\"camera\")+2*raytype(\"edge31\"),"
        "raytype(\"duplicate\")+2*raytype(\"hart-missing-ray\"),"
        "raytype(\"\")); }",
        "shader hart_geometry_plain(output color Cout=0) { Cout=color(u,v,u*v); }",
    };
    std::string bytecode[std::size(sources)];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
            return false;
    }
    const struct {
        int osl, llvm;
        bool connected, local, zero;
    } states[] = {
        { 0, 10, false, false, false }, { 2, 10, false, true, false },
        { 2, 3, false, true, false },   { 0, 10, true, false, false },
        { 2, 10, true, true, false },   { 2, 3, true, true, false },
        { 2, 10, false, true, true },
    };
    const ustring entries[] = { ustring("producer"), ustring("consumer") };
    for (const auto& test : states) {
        HartGeometryServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", test.osl));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", test.llvm));
        OIIO_CHECK_ASSERT(ss.attribute("lazyglobals", 1));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", test.local ? 4096 : 0));
        auto group = test.connected
                         ? make_connected_group(ss, bytecode[0], bytecode[1])
                         : make_group(ss, bytecode[test.zero ? 2 : 0]);
        if (test.connected) {
            // OSL O2 may alias value directly to u and remove the connection's
            // lazy call. Global side effects therefore need explicit entries.
            OIIO_CHECK_ASSERT(ss.attribute(group.get(), "entry_layers",
                                           TypeDesc(TypeDesc::STRING, 2),
                                           entries));
            check_hart_entry_selection(ss, *group, entries, entries);
        }
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Geometry state: {}\n", errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        if (test.connected) {
            OIIO_CHECK_ASSERT(check_hart_entry_module(ss, *group, arch,
                                                      test.llvm, entries,
                                                      test.local, false));
            check_hart_entry_selection(ss, *group, entries, entries);
        } else {
            check_module(
                ss, *group, arch,
                test.zero
                    ? std::initializer_list<string_view> { }
                    : std::initializer_list<string_view> { "osl_raytype_bit" },
                test.llvm);
        }
        int allocated = -1, group_size = 0;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", group_size));
        OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
        OIIO_CHECK_EQUAL(allocated, test.local ? group_size : 0);
        if (test.llvm == 10)
            OIIO_CHECK_ASSERT(
                check_geometry_state_ir(ss, *group, test.zero, test.connected));
    }
    const ustring defaults[] = {
        ustring("camera"),     ustring("shadow"),      ustring("reflection"),
        ustring("refraction"), ustring("diffuse"),     ustring("glossy"),
        ustring("subsurface"), ustring("displacement")
    };
    const ustring empty_names[] = { ustring(), ustring("") };
    OIIO_CHECK_ASSERT(empty_names[0].empty() && empty_names[1].empty());
    OIIO_CHECK_ASSERT(!empty_names[0].c_str() && empty_names[1].c_str());
    for (ustring name : empty_names)
        OIIO_CHECK_EQUAL(name.hash(), uint64_t(0));
    std::vector<ustring> custom;
    for (int i = 0; i < 32; ++i)
        custom.emplace_back(fmtformat("hart-ray-{}", i));
    custom[0] = ustring("camera");
    custom[1] = custom[7] = ustring("duplicate");
    // Exercise both orders without changing the first empty-name bit.
    custom[12] = empty_names[1];
    custom[19] = empty_names[0];
    custom[31] = ustring("edge31");
    const struct {
        int osl, llvm;
        bool custom, connected, local, constant;
        bool null_first = false;
    } rays[] = {
        { 0, 10, false, false, false, false },
        { 2, 3, false, false, true, false },
        { 0, 10, true, false, false, false },
        { 2, 3, true, false, true, false, true },
        { 2, 10, true, true, true, false, true },
        { 0, 10, false, false, false, true },
        { 2, 10, true, false, true, true },
    };
    for (const auto& test : rays) {
        HartGeometryServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", test.osl));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", test.llvm));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", test.local ? 4096 : 0));
        OIIO_CHECK_ASSERT(ss.attribute("error_repeats", 1));
        auto configured = custom;
        if (test.null_first)
            std::swap(configured[12], configured[19]);
        const cspan<ustring> names = test.custom ? cspan<ustring>(configured)
                                                 : cspan<ustring>(defaults);
        std::vector<const char*> strings;
        for (ustring name : names)
            strings.push_back(name.c_str());
        if (test.custom)
            OIIO_CHECK_ASSERT(
                ss.attribute("raytypes",
                             TypeDesc(TypeDesc::STRING, int(strings.size())),
                             strings.data()));
        auto unchanged = [&]() {
            for (ustring name : names)
                OIIO_CHECK_EQUAL(uint32_t(ss.raytype_bit(name)),
                                 geometry_ray_bit(names, name));
            OIIO_CHECK_EQUAL(ss.raytype_bit(ustring("hart-missing-ray")), 0);
            for (ustring empty : empty_names)
                OIIO_CHECK_EQUAL(uint32_t(ss.raytype_bit(empty)),
                                 test.custom ? uint32_t(1) << 12 : 0);
            if (test.custom)
                OIIO_CHECK_EQUAL(ss.raytype_bit(ustring("duplicate")), 2);
        };
        unchanged();
        if (test.custom && test.osl == 0) {
            auto too_many = strings;
            too_many.push_back("hart-overflow-ray");
            const struct {
                TypeDesc type;
                const void* data;
            } invalid[] = {
                { TypeDesc(TypeDesc::STRING, 33), too_many.data() },
                { TypeDesc(TypeDesc::STRING, -1), strings.data() },
                { TypeDesc(TypeDesc::STRING, TypeDesc::VEC3, 2),
                  strings.data() },
                { TypeDesc(TypeDesc::STRING, 32), nullptr },
                { TypeString, nullptr },
            };
            for (const auto& bad : invalid) {
                const int before = errors.errors;
                OIIO_CHECK_ASSERT(
                    !ss.attribute("raytypes", bad.type, bad.data));
                OIIO_CHECK_EQUAL(errors.errors, before + 1);
                OIIO_CHECK_ASSERT(OIIO::Strutil::contains(
                    errors.last_error,
                    "raytypes requires at most 32 string names"));
                unchanged();
            }
            for (ustring single : { ustring("hart-single-ray"), empty_names[0],
                                    empty_names[1] }) {
                const char* value = single.c_str();
                OIIO_CHECK_ASSERT(ss.attribute("raytypes", TypeString, &value));
                OIIO_CHECK_EQUAL(ss.raytype_bit(single), 1);
                for (ustring empty : empty_names)
                    OIIO_CHECK_EQUAL(ss.raytype_bit(empty),
                                     single.empty() ? 1 : 0);
                OIIO_CHECK_EQUAL(ss.raytype_bit(ustring("camera")), 0);
                OIIO_CHECK_EQUAL(ss.raytype_bit(ustring("edge31")), 0);
            }
            OIIO_CHECK_ASSERT(ss.attribute("raytypes",
                                           TypeDesc(TypeDesc::STRING, 32),
                                           strings.data()));
            unchanged();
            OIIO_CHECK_EQUAL(errors.errors, std::size(invalid));
        }
        const int before = errors.errors;
        auto group = test.connected
                         ? make_connected_group(ss, bytecode[3], bytecode[4])
                         : make_group(ss, bytecode[test.constant ? 6 : 5]);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors != before)
            print(stderr, "Geometry raytype: {}\n", errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, before);
        if (!test.constant) {
            // All three queries are live; no known ray masks are supplied.
            int count              = 0;
            const ustring* globals = nullptr;
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "num_globals_needed", count));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "globals_needed",
                                              TypeDesc::PTR, &globals));
            OIIO_CHECK_ASSERT(globals && count >= 3);
            if (globals && count >= 3)
                for (ustring name : { ustring("raytype"), ustring("backfacing"),
                                      ustring("surfacearea") })
                    OIIO_CHECK_ASSERT(std::find(globals, globals + count, name)
                                      != globals + count);
        }
        check_module(ss, *group, arch, { "osl_raytype_bit" }, test.llvm,
                     test.connected, false, false, test.connected ? 2 : 0);
        int allocated = -1, group_size = 0;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", group_size));
        OIIO_CHECK_ASSERT(group_size > 0 && group_size <= 4096);
        OIIO_CHECK_EQUAL(allocated, test.local ? group_size : 0);
        if (test.llvm == 10)
            OIIO_CHECK_ASSERT(check_geometry_ray_ir(ss, *group, names,
                                                    !test.constant,
                                                    test.constant));
    }
    const struct {
        const char* body;
        const char* diagnostic;
        bool geometry;
    } negatives[] = {
        { "Cout=color(raytype(\"camera\"));",
          "renderer lacks HARTGeometry for 'raytype'", false },
        { "Cout=color(backfacing());",
          "renderer lacks HARTGeometry for 'backfacing'", false },
        { "Cout=color(surfacearea());",
          "renderer lacks HARTGeometry for 'surfacearea'", false },
        { "Cout=color(dtime);", "unsupported shader global 'dtime'", false },
        { "Cout=color(dPdtime);", "unsupported shader global 'dPdtime'", false },
        { "N=normal(u,v,1);", "writing shader global 'N'", false },
        { "P[0]=u;", "writing shader global 'P'", false },
        { "time=u;", "writing shader global 'time'", true },
        { "dtime=u;", "writing shader global 'dtime'", true },
        { "dPdtime=vector(P);", "writing shader global 'dPdtime'", true },
        { "Cout=color(Ps);", "unsupported shader global 'Ps'", true },
        { "Ps=P;", "writing shader global 'Ps'", true },
    };
    for (const auto& test : negatives) {
        OSLCompiler compiler;
        std::string bad;
        if (!compiler.compile_buffer(
                fmtformat("shader hart_geometry_rejected(int enable=0,"
                          "output color Cout=0) {{ if(enable) {{ {} }} }}",
                          test.body),
                bad, { }, stdosl))
            return false;
        for (bool unused : { false, true }) {
            HartGeometryServices renderer(test.geometry);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", 10));
            OIIO_CHECK_ASSERT(ss.attribute("lazyunconnected", 1));
            OIIO_CHECK_ASSERT(ss.attribute("lazyglobals", 1));
            ShaderGroupRef group;
            if (!unused) {
                group = make_group(ss, bad);
            } else {
                OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_bad", bad));
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_plain", bytecode[7]));
                group = ss.ShaderGroupBegin("hart_geometry_rejected_group");
                OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_bad", "unused"));
                OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_plain", "last"));
                OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
                const SymLocationDesc output("last.Cout", TypeColor, false,
                                             SymArena::Outputs, 0,
                                             3 * sizeof(float));
                ss.add_symlocs(group.get(), { &output, 1 });
            }
            check_rejected_group(ss, *group, errors, test.diagnostic);
        }
    }
    return true;
}



bool
check_geometry_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_geometry_producer(output vector value=0) { "
        "value=I+vector(u+time,v-time,u*v)+Dx(I)+Dy(I); }",
        "shader hart_geometry_consumer(vector value=0,output color Cout=0) { "
        "point p=transform(\"common\",\"shader\",point(value)); "
        "color c=texture(\"hart-test-texture.exr\",p[0],p[1],"
        "\"interp\",\"linear\",\"wrap\",\"clamp\"); Cout=c+Dx(c)+Dy(c); }",
        "shader hart_geometry_test(output color Cout=0) { "
        "Cout=color(I+Dx(I)+Dy(I)+filterwidth(I))"
        "+color(time+Dx(time)+Dy(time)+filterwidth(time)); }",
    };
    std::string bytecode[3];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
            return false;
    }
    for (int osl_optimize : { 0, 2 })
        for (int optimize : { 10, 3 })
            for (bool connected : { false, true }) {
                HartServices renderer(true, true);
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                ss.attribute("hart_arch", arch);
                ss.attribute("optimize", osl_optimize);
                ss.attribute("llvm_optimize", optimize);
                auto group = connected ? make_connected_group(ss, bytecode[0],
                                                              bytecode[1])
                                       : make_group(ss, bytecode[2]);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "{}\n", errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                check_module(ss, *group, arch,
                             connected
                                 ? std::initializer_list<
                                       string_view> { "osl_texture",
                                                      "osl_transform_triple" }
                                 : std::initializer_list<
                                       string_view> { "osl_filterwidth_vdv" },
                             optimize, connected);
            }
    for (string_view body :
         { "if (enable) I=vector(u,v,1);", "if (enable) time=u;" }) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat(
                    "shader hart_geometry_write(int enable=0,output color Cout=0) {{ {} }}",
                    body),
                bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, "writing shader global");
    }
    return true;
}



bool
check_texture_alpha_modules(string_view arch, string_view stdosl)
{
    const struct {
        string_view output;
        bool connected;
        bool alpha_derivs;
        bool result_derivs;
    } tests[] = {
        { "color(sampled)+color(alpha)", false, false, false },
        { "color(alpha,Dx(alpha),Dy(alpha))", false, true, false },
        { "color(sampled)+color(alpha,Dx(alpha),Dy(alpha))", false, true,
          false },
        { "color(sampled+Dx(sampled)+Dy(sampled))"
          "+color(alpha,Dx(alpha),Dy(alpha))",
          false, true, true },
        { "color(value)", true, false, false },
        { "color(value,Dx(value),Dy(value))", true, true, false },
    };
    for (string_view type : { "float", "color" }) {
        const auto body = fmtformat(
            "float alpha_a=0, alpha_b=0; "
            "{0} a=texture(\"hart-test-texture.exr\",u,v,"
            "\"interp\",\"linear\",\"wrap\",\"periodic\",\"alpha\",alpha_a); "
            "{0} b=texture(\"hart-test-texture.exr\",u,v,0.25,0,0,0.5,"
            "\"interp\",\"closest\",\"wrap\",\"clamp\",\"alpha\",alpha_b); "
            "{0} sampled=a+b; float alpha=alpha_a+alpha_b; ",
            type);
        OSLCompiler producer_compiler;
        std::string producer;
        if (!producer_compiler.compile_buffer(
                fmtformat(
                    "shader hart_alpha_producer(output float value=0) {{ "
                    "float extra=0; "
                    "{0} a=texture(\"hart-test-texture.exr\",u,v,"
                    "\"interp\",\"linear\",\"wrap\",\"periodic\","
                    "\"alpha\",value); "
                    "{0} b=texture(\"hart-test-texture.exr\",u,v,0.25,0,0,0.5,"
                    "\"interp\",\"closest\",\"wrap\",\"clamp\","
                    "\"alpha\",extra); value+=extra; }}",
                    type),
                producer, { }, stdosl))
            return false;
        for (const auto& test : tests) {
            const auto source
                = test.connected
                      ? fmtformat("shader hart_alpha_consumer(float value=0,"
                                  "output color Cout=0) {{ Cout={}; }}",
                                  test.output)
                      : fmtformat("shader hart_alpha(output color Cout=0) {{ "
                                  "{} Cout={}; }}",
                                  body, test.output);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            for (int osl_optimize : { 0, 2 })
                for (int optimize : { 10, 3 }) {
                    HartServices renderer(true);
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("optimize", osl_optimize);
                    ss.attribute("llvm_optimize", optimize);
                    auto group = test.connected
                                     ? make_connected_group(ss, producer,
                                                            bytecode)
                                     : make_group(ss, bytecode);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "Texture alpha {}: {}\n", type,
                              errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    OIIO_CHECK_ASSERT(renderer.texture_requests > 0);
                    check_module(ss, *group, arch, { "osl_texture" }, optimize,
                                 test.connected, false, false,
                                 test.connected ? 2 : 0);
                    if (optimize != 10)
                        continue;
                    const void* bytes = nullptr;
                    uint64_t size     = 0;
                    OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                      "hart_bitcode",
                                                      TypeDesc::PTR, &bytes));
                    OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                      "hart_bitcode_size",
                                                      TypeUInt64, &size));
                    if (!bytes || !size)
                        continue;
                    llvm::LLVMContext context;
                    const llvm::StringRef data(static_cast<const char*>(bytes),
                                               size);
                    auto parsed = llvm::parseBitcodeFile(
                        llvm::MemoryBufferRef(data, "hart_alpha"), context);
                    if (!parsed) {
                        print(stderr, "{}\n",
                              llvm::toString(parsed.takeError()));
                        OIIO_CHECK_ASSERT(false);
                        continue;
                    }
                    const auto* texture = (*parsed)->getFunction("osl_texture");
                    OIIO_CHECK_ASSERT(texture);
                    int calls = 0;
                    if (texture)
                        for (const auto* user : texture->users()) {
                            const auto* call = llvm::dyn_cast<llvm::CallInst>(
                                user);
                            if (!call || call->getCalledFunction() != texture)
                                continue;
                            ++calls;
                            OIIO_CHECK_EQUAL(call->arg_size(), 18);
                            if (call->arg_size() != 18)
                                continue;
                            const auto* channels
                                = llvm::dyn_cast<llvm::ConstantInt>(
                                    call->getArgOperand(10));
                            OIIO_CHECK_ASSERT(channels);
                            if (channels)
                                OIIO_CHECK_EQUAL(channels->getZExtValue(),
                                                 type == "float" ? 1 : 3);
                            // Result (11..13) and alpha (14..16) have
                            // independent derivative demand.
                            for (int arg = 11; arg <= 16; ++arg) {
                                const auto* pointer = call->getArgOperand(arg);
                                OIIO_CHECK_ASSERT(
                                    pointer->getType()->isPointerTy());
                                const bool nonnull
                                    = arg == 11 || arg == 14
                                      || (arg < 14 ? test.result_derivs
                                                   : test.alpha_derivs);
                                OIIO_CHECK_EQUAL(
                                    llvm::isa<llvm::ConstantPointerNull>(
                                        pointer->stripPointerCasts()),
                                    !nonnull);
                            }
                        }
                    OIIO_CHECK_EQUAL(calls, 2);
                }
        }
    }
    for (string_view type : { "int", "color" }) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source = fmtformat(
            "shader hart_bad_alpha(output color Cout=0) {{ {} alpha=0; "
            "Cout=texture(\"hart-test-texture.exr\",u,v,\"interp\",\"linear\","
            "\"wrap\",\"clamp\",\"alpha\",alpha); }}",
            type);
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, "texture alpha requires a float output",
                        1, false, true);
    }
    return true;
}



bool
check_texture_firstchannel_modules(string_view arch, string_view stdosl)
{
    OSLCompiler consumer_compiler;
    std::string consumer;
    if (!consumer_compiler.compile_buffer(
            "shader hart_channel_consumer(float value=0,output color Cout=0) { "
            "Cout=color(value,Dx(value),Dy(value)); }",
            consumer, { }, stdosl))
        return false;
    const struct {
        int firstchannel;
        bool reset;
        bool connected;
    } tests[] = {
        { 0, false, false }, { 1, false, false }, { 2, false, false },
        { 3, false, true },  { 4, false, false }, { 2147483647, false, true },
        { 1, true, false },
    };
    for (const auto& test : tests) {
        const int settings_per_call = test.reset ? 2
                                                 : (test.firstchannel ? 1 : 0);
        const auto options          = fmtformat(
            "\"interp\",\"linear\",\"wrap\",\"periodic\",\"firstchannel\",{}{}",
            test.firstchannel, test.reset ? ",\"firstchannel\",0" : "");
        const auto body = fmtformat(
            "float extra=0; "
            "color c=texture(\"hart-test-texture.exr\",u,v,{0},\"alpha\",value); "
            "float f=texture(\"hart-test-texture.exr\",v,u,0.25,0,0,0.5,"
            "{0},\"alpha\",extra); "
            "color plain_c=texture(\"hart-test-texture.exr\",0.5*u,v,{0}); "
            "float plain_f=texture(\"hart-test-texture.exr\",u,0.5*v,{0}); "
            "value+=extra+dot(vector(c+plain_c),vector(1))+f+plain_f; ",
            options);
        const auto source
            = test.connected
                  ? fmtformat("shader hart_channel_producer("
                              "output float value=0) {{ {} }}",
                              body)
                  : fmtformat("shader hart_channel(output color Cout=0) {{ "
                              "float value=0; {} "
                              "Cout=color(value,Dx(value),Dy(value)); }}",
                              body);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 }) {
                HartServices renderer(true);
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                ss.attribute("hart_arch", arch);
                ss.attribute("optimize", osl_optimize);
                ss.attribute("llvm_optimize", optimize);
                auto group = test.connected
                                 ? make_connected_group(ss, bytecode, consumer)
                                 : make_group(ss, bytecode);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "Texture firstchannel {}: {}\n",
                          test.firstchannel, errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                OIIO_CHECK_ASSERT(renderer.texture_requests > 0);
                check_module(ss, *group, arch,
                             { "osl_texture",
                               test.firstchannel
                                   ? "osl_texture_set_firstchannel"
                                   : "" },
                             optimize, test.connected, false, false,
                             test.connected ? 2 : 0);
                if (optimize != 10)
                    continue;
                const void* bytes = nullptr;
                uint64_t size     = 0;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                                  TypeDesc::PTR, &bytes));
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "hart_bitcode_size",
                                                  TypeUInt64, &size));
                if (!bytes || !size)
                    continue;
                llvm::LLVMContext context;
                const llvm::StringRef data(static_cast<const char*>(bytes),
                                           size);
                auto parsed = llvm::parseBitcodeFile(
                    llvm::MemoryBufferRef(data, "hart_firstchannel"), context);
                if (!parsed) {
                    print(stderr, "{}\n", llvm::toString(parsed.takeError()));
                    OIIO_CHECK_ASSERT(false);
                    continue;
                }
                const auto* shader = (*parsed)->getFunction(
                    test.connected
                        ? "osl_layer_group_hart_test_group_name_producer"
                        : "osl_layer_group_hart_test_group_name_layer0");
                OIIO_CHECK_ASSERT(shader);
                if (!shader)
                    continue;
                int calls = 0, alpha_calls = 0, scalar_calls = 0;
                std::vector<int64_t> settings;
                const llvm::Value* options_ptr = nullptr;
                for (const auto& block : *shader)
                    for (const auto& inst : block) {
                        const auto* call = llvm::dyn_cast<llvm::CallInst>(
                            &inst);
                        const auto* callee = call ? call->getCalledFunction()
                                                  : nullptr;
                        if (!callee)
                            continue;
                        if (callee->getName()
                            == "osl_texture_set_firstchannel") {
                            OIIO_CHECK_ASSERT(!callee->isDeclaration());
                            OIIO_CHECK_ASSERT(
                                callee->getReturnType()->isVoidTy());
                            OIIO_CHECK_EQUAL(call->arg_size(), 2);
                            if (call->arg_size() != 2)
                                continue;
                            const auto* value
                                = llvm::dyn_cast<llvm::ConstantInt>(
                                    call->getArgOperand(1));
                            OIIO_CHECK_ASSERT(value);
                            if (value) {
                                OIIO_CHECK_ASSERT(
                                    value->getType()->isIntegerTy(32));
                                settings.push_back(value->getSExtValue());
                            }
                            if (options_ptr)
                                OIIO_CHECK_EQUAL(call->getArgOperand(0),
                                                 options_ptr);
                            options_ptr = call->getArgOperand(0);
                        }
                        if (callee->getName() != "osl_texture")
                            continue;
                        ++calls;
                        OIIO_CHECK_EQUAL(call->arg_size(), 18);
                        if (call->arg_size() != 18)
                            continue;
                        OIIO_CHECK_EQUAL(settings.size(),
                                         size_t(settings_per_call));
                        if (!settings.empty())
                            OIIO_CHECK_EQUAL(settings[0], test.firstchannel);
                        if (test.reset && settings.size() == 2)
                            OIIO_CHECK_EQUAL(settings[1], 0);
                        if (options_ptr)
                            OIIO_CHECK_EQUAL(call->getArgOperand(3),
                                             options_ptr);
                        settings.clear();
                        options_ptr = nullptr;
                        const auto* channels = llvm::dyn_cast<llvm::ConstantInt>(
                            call->getArgOperand(10));
                        OIIO_CHECK_ASSERT(channels);
                        if (channels) {
                            OIIO_CHECK_ASSERT(channels->getZExtValue() == 1
                                              || channels->getZExtValue() == 3);
                            scalar_calls += channels->getZExtValue() == 1;
                        }
                        const bool alpha = !llvm::isa<llvm::ConstantPointerNull>(
                            call->getArgOperand(14)->stripPointerCasts());
                        alpha_calls += alpha;
                        for (int arg = 11; arg <= 16; ++arg)
                            OIIO_CHECK_EQUAL(
                                llvm::isa<llvm::ConstantPointerNull>(
                                    call->getArgOperand(arg)
                                        ->stripPointerCasts()),
                                arg >= 14 && !alpha);
                    }
                OIIO_CHECK_ASSERT(settings.empty());
                OIIO_CHECK_EQUAL(calls, 4);
                OIIO_CHECK_EQUAL(alpha_calls, 2);
                OIIO_CHECK_EQUAL(scalar_calls, 2);
            }
    }
    const struct {
        string_view parameter;
        string_view channel;
        string_view error;
    } rejected[] = {
        { "", "-1", "firstchannel requires a literal nonnegative integer" },
        { "", "channel", "firstchannel requires a literal nonnegative integer" },
        { "", "int(4*u)",
          "firstchannel requires a literal nonnegative integer" },
        { "", "1.0", "firstchannel requires a literal nonnegative integer" },
        { "", "color(1)",
          "firstchannel requires a literal nonnegative integer" },
        { "int channels[2]={1,2},", "channels",
          "firstchannel requires a literal nonnegative integer" },
    };
    // Rejection must precede folding default parameters or pruning layers.
    for (const auto& test : rejected)
        for (int mode : { 0, 1, 2 }) {
            const auto source = fmtformat(
                "shader hart_bad_channel(int channel=0,int enable=0,"
                "{}output color Cout=0) {{ {} "
                "Cout=texture(\"hart-test-texture.exr\",u,v,\"interp\",\"linear\","
                "\"wrap\",\"clamp\",\"firstchannel\",{}); }}",
                test.parameter, mode == 1 ? "if(enable)" : "", test.channel);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            HartServices renderer(true, false, false, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", 2);
            ShaderGroupRef group;
            if (mode == 2) {
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_bad_channel", bytecode));
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_channel_consumer",
                                                consumer));
                group = ss.ShaderGroupBegin("hart_test_group");
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_bad_channel", "unused"));
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_channel_consumer", "consumer"));
                OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
                const SymLocationDesc output("consumer.Cout", TypeColor, false,
                                             SymArena::Outputs, 0,
                                             3 * sizeof(float));
                ss.add_symlocs(group.get(), { &output, 1 });
            } else {
                group = make_group(ss, bytecode);
            }
            check_rejected_group(ss, *group, errors, test.error);
        }
    return true;
}



struct TextureOptionExpectation {
    ustring filename;
    OIIO::TextureOpt::Wrap swrap, twrap;
    OIIO::TextureOpt::InterpMode interp;
    int channels = 3;
};



bool
check_texture_options_ir(ShadingSystem& ss, ShaderGroup& group,
                         cspan<TextureOptionExpectation> expected)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_texture_options"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    const auto* shader = (*parsed)->getFunction(
        "osl_layer_group_hart_test_group_name_layer0");
    OIIO_CHECK_ASSERT(shader);
    if (!shader)
        return false;
    const OIIO::TextureOpt initial;
    int swrap = -1, twrap = -1, interp = -1;
    size_t calls = 0, initializations = 0;
    const llvm::Value* options = nullptr;
    for (const auto& block : *shader)
        for (const auto& inst : block) {
            const auto* call   = llvm::dyn_cast<llvm::CallInst>(&inst);
            const auto* callee = call ? call->getCalledFunction() : nullptr;
            if (!callee)
                continue;
            const auto name = callee->getName();
            if (name == "osl_init_texture_options") {
                OIIO_CHECK_ASSERT(!callee->isDeclaration());
                OIIO_CHECK_EQUAL(call->arg_size(), 2);
                if (call->arg_size() != 2)
                    return false;
                options = call->getArgOperand(1)->stripPointerCasts();
                swrap   = int(initial.swrap);
                twrap   = int(initial.twrap);
                interp  = int(initial.interpmode);
                ++initializations;
                continue;
            }
            if (name == "osl_texture_set_interp_code"
                || name == "osl_texture_set_stwrap_code"
                || name == "osl_texture_set_swrap_code"
                || name == "osl_texture_set_twrap_code") {
                OIIO_CHECK_ASSERT(!callee->isDeclaration());
                OIIO_CHECK_EQUAL(call->arg_size(), 2);
                if (call->arg_size() != 2)
                    return false;
                OIIO_CHECK_EQUAL(call->getArgOperand(0)->stripPointerCasts(),
                                 options);
                const auto* code = llvm::dyn_cast<llvm::ConstantInt>(
                    call->getArgOperand(1));
                OIIO_CHECK_ASSERT(code && code->getType()->isIntegerTy(32));
                if (!code || !code->getType()->isIntegerTy(32))
                    return false;
                const int value = int(code->getSExtValue());
                if (name == "osl_texture_set_interp_code")
                    interp = value;
                if (name == "osl_texture_set_stwrap_code"
                    || name == "osl_texture_set_swrap_code")
                    swrap = value;
                if (name == "osl_texture_set_stwrap_code"
                    || name == "osl_texture_set_twrap_code")
                    twrap = value;
                continue;
            }
            if (name.find("osl_texture_set_") == 0) {
                print(stderr, "Unexpected texture option setter '{}'\n",
                      name.str());
                return false;
            }
            if (name != "osl_texture")
                continue;
            OIIO_CHECK_EQUAL(call->arg_size(), 18);
            OIIO_CHECK_ASSERT(options && calls < expected.size());
            if (call->arg_size() != 18 || !options || calls >= expected.size())
                return false;
            const auto& test = expected[calls++];
            OIIO_CHECK_EQUAL(call->getArgOperand(3)->stripPointerCasts(),
                             options);
            OIIO_CHECK_EQUAL(swrap, int(test.swrap));
            OIIO_CHECK_EQUAL(twrap, int(test.twrap));
            OIIO_CHECK_EQUAL(interp, int(test.interp));
            const auto* channels = llvm::dyn_cast<llvm::ConstantInt>(
                call->getArgOperand(10));
            OIIO_CHECK_ASSERT(channels);
            if (channels)
                OIIO_CHECK_EQUAL(channels->getLimitedValue(),
                                 uint64_t(test.channels));
            const auto* handle = llvm::dyn_cast<llvm::ConstantExpr>(
                call->getArgOperand(2));
            OIIO_CHECK_ASSERT(
                handle && handle->getOpcode() == llvm::Instruction::IntToPtr);
            if (handle && handle->getOpcode() == llvm::Instruction::IntToPtr) {
                const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                    handle->getOperand(0));
                OIIO_CHECK_ASSERT(value);
                if (value)
                    OIIO_CHECK_EQUAL(value->getLimitedValue(), uint64_t(1));
            }
            OIIO_CHECK_ASSERT(
                call->getArgOperand(1)->getType()->isIntegerTy(64));
            if (const auto* filename = llvm::dyn_cast<llvm::ConstantInt>(
                    call->getArgOperand(1)))
                OIIO_CHECK_EQUAL(filename->getLimitedValue(),
                                 ustringhash(test.filename).hash());
            options = nullptr;
        }
    OIIO_CHECK_EQUAL(calls, expected.size());
    OIIO_CHECK_EQUAL(initializations, calls);
    return calls == expected.size() && initializations == calls;
}



bool
check_texture_compatibility_modules(string_view arch, string_view stdosl)
{
    using Tex = OIIO::TextureOpt;
    const struct {
        int osl, llvm, handles;
        bool local;
    } variants[]
        = { { 0, 10, 0, false }, { 2, 10, 1, false }, { 2, 3, 0, true } };
    auto overridden_group = [](ShadingSystem& ss, string_view bytecode,
                               const char* value, ParamHints hints) {
        if (!value)
            return make_group(ss, bytecode);
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_test", bytecode));
        auto group = ss.ShaderGroupBegin("hart_test_group");
        const ustring filename(value);
        OIIO_CHECK_ASSERT(
            ss.Parameter("filename", TypeString, &filename, hints));
        OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
        OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        const SymLocationDesc output("Cout", TypeColor, false,
                                     SymArena::Outputs, 0, 3 * sizeof(float));
        ss.add_symlocs(group.get(), { &output, 1 });
        return group;
    };
    auto check = [&](ShadingSystem& ss, ShaderGroup& group,
                     const HartServices& renderer, const Diagnostics& errors,
                     int optimize, bool local,
                     cspan<TextureOptionExpectation> expected) {
        ss.optimize_group(&group, nullptr);
        if (errors.errors)
            print(stderr, "Texture compatibility: {}\n", errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        OIIO_CHECK_ASSERT(renderer.texture_requests > 0);
        for (const auto& texture : expected)
            OIIO_CHECK_ASSERT(std::find(renderer.texture_filenames.begin(),
                                        renderer.texture_filenames.end(),
                                        texture.filename)
                              != renderer.texture_filenames.end());
        for (const auto filename : renderer.texture_filenames)
            OIIO_CHECK_ASSERT(
                std::any_of(expected.begin(), expected.end(),
                            [&](const TextureOptionExpectation& texture) {
                                return texture.filename == filename;
                            }));
        check_module(ss, group, arch,
                     { "osl_texture", "osl_init_texture_options" }, optimize);
        int allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(&group, "hart_groupdata_alloc", allocated));
        OIIO_CHECK_EQUAL(allocated > 0, local);
        // These are newly emitted layer calls, not HIP-inlined helper bodies.
        if (optimize == 10)
            OIIO_CHECK_ASSERT(check_texture_options_ir(ss, group, expected));
    };
    const struct {
        string_view declaration;
        const char* override_name;
        bool defaults, derivatives;
        string_view options;
        Tex::Wrap swrap, twrap;
        Tex::InterpMode interp;
    } filenames[] = {
        { "string filename=\"hart-test-texture.exr\"", nullptr, false, false,
          ",\"interp\",\"linear\",\"wrap\",\"periodic\"", Tex::WrapPeriodic,
          Tex::WrapPeriodic, Tex::InterpBilinear },
        { "string filename=\"\"", "hart_texture_alpha_4.exr", false, false,
          ",\"interp\",\"closest\",\"wrap\",\"clamp\"", Tex::WrapClamp,
          Tex::WrapClamp, Tex::InterpClosest },
        { "string filename=\"hart-test-texture.exr\"",
          "hart_texture_alpha_4.exr", true, false, "", Tex::WrapPeriodic,
          Tex::WrapPeriodic, Tex::InterpBilinear },
        { "string filename=\"\"", "hart_texture_alpha_4.exr", true, false, "",
          Tex::WrapPeriodic, Tex::WrapPeriodic, Tex::InterpBilinear },
        { "string filename=\"hart-test-texture.exr\" [[int lockgeom=1]]",
          nullptr, true, true, ",\"interp\",\"closest\"", Tex::WrapPeriodic,
          Tex::WrapPeriodic, Tex::InterpClosest },
    };
    for (const auto& test : filenames) {
        const auto source = fmtformat(
            "shader hart_texture_filename({},output color Cout=0) {{ "
            "color sampled=texture(filename,u,v{}{}); "
            "Cout=sampled+Dx(sampled)+Dy(sampled); }}",
            test.declaration, test.derivatives ? ",.25,0,0,.5" : "",
            test.options);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        const TextureOptionExpectation expected[]
            = { { ustring(test.override_name ? test.override_name
                                             : "hart-test-texture.exr"),
                  test.swrap, test.twrap, test.interp } };
        for (const auto& variant : variants) {
            HartServices renderer(true, false, false, false, false, false,
                                  false, false, test.defaults);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
            OIIO_CHECK_ASSERT(
                ss.attribute("opt_texture_handle", variant.handles));
            OIIO_CHECK_ASSERT(ss.attribute("max_hart_groupdata_alloc",
                                           variant.local ? 4096 : 0));
            auto group = overridden_group(ss, bytecode, test.override_name,
                                          ParamHints::none);
            check(ss, *group, renderer, errors, variant.llvm, variant.local,
                  expected);
        }
    }
    const struct {
        string_view options;
        Tex::Wrap swrap, twrap;
        Tex::InterpMode interp;
        int channels = 3;
    } options[] = {
        { "", Tex::WrapPeriodic, Tex::WrapPeriodic, Tex::InterpBilinear },
        { ",\"interp\",\"closest\"", Tex::WrapPeriodic, Tex::WrapPeriodic,
          Tex::InterpClosest, 1 },
        { ",\"wrap\",\"clamp\"", Tex::WrapClamp, Tex::WrapClamp,
          Tex::InterpBilinear },
        { ",\"swrap\",\"black\"", Tex::WrapBlack, Tex::WrapPeriodic,
          Tex::InterpBilinear },
        { ",\"twrap\",\"clamp\"", Tex::WrapPeriodic, Tex::WrapClamp,
          Tex::InterpBilinear },
        { ",\"interp\",\"closest\",\"swrap\",\"clamp\",\"twrap\",\"black\"",
          Tex::WrapClamp, Tex::WrapBlack, Tex::InterpClosest },
        { ",\"wrap\",\"clamp\",\"swrap\",\"periodic\","
          "\"interp\",\"closest\",\"interp\",\"linear\"",
          Tex::WrapPeriodic, Tex::WrapClamp, Tex::InterpBilinear },
        { ",\"swrap\",\"clamp\",\"wrap\",\"periodic\",\"twrap\",\"black\"",
          Tex::WrapPeriodic, Tex::WrapBlack, Tex::InterpBilinear },
    };
    std::string source = "shader hart_texture_defaults(output color Cout=0) { ";
    std::vector<TextureOptionExpectation> expected;
    for (size_t i = 0; i < std::size(options); ++i) {
        const auto& test     = options[i];
        const char* filename = i == 1 ? "hart_texture_alpha_4.exr"
                                      : "hart-test-texture.exr";
        source += fmtformat("{} sampled{}=texture(\"{}\",u*{}.0,v{}{}); "
                            "Cout+=color(sampled{})*{}.0; ",
                            test.channels == 1 ? "float" : "color", i, filename,
                            i + 1, i == 5 ? ",.25,0,0,.5" : "", test.options, i,
                            i + 1);
        expected.push_back({ ustring(filename), test.swrap, test.twrap,
                             test.interp, test.channels });
    }
    source += "}";
    OSLCompiler options_compiler;
    std::string options_bytecode;
    if (!options_compiler.compile_buffer(source, options_bytecode, { }, stdosl))
        return false;
    for (const auto& variant : variants) {
        HartServices renderer(true, false, false, false, false, false, false,
                              false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
        OIIO_CHECK_ASSERT(ss.attribute("opt_texture_handle", variant.handles));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0));
        auto group = make_group(ss, options_bytecode);
        check(ss, *group, renderer, errors, variant.llvm, variant.local,
              expected);
    }

    // Reach the filename guard rather than rejecting the renderer capability.
    OSLCompiler producer_compiler;
    std::string producer;
    if (!producer_compiler.compile_buffer(
            "shader hart_texture_name_source(output string value=\"\") { "
            "if(u>.5) value=\"hart-test-texture.exr\"; "
            "else value=\"hart_texture_alpha_4.exr\"; }",
            producer, { }, stdosl))
        return false;
    const struct {
        string_view declaration, body, name;
        const char* override_name = nullptr;
        ParamHints hints          = ParamHints::none;
        bool connected            = false;
        string_view error         = "texture requires a literal filename";
        bool prepare              = false;
        bool written_input        = false;
    } rejected_names[] = {
        { "string filename=\"hart-test-texture.exr\"", "", "\"\"" },
        { "string filename=\"\"", "", "filename" },
        { "string filename=\"hart-test-texture.exr\"", "", "filename", "" },
        { "string filename=(u>.5 ? \"hart-test-texture.exr\" : \"hart_texture_alpha_4.exr\")",
          "", "filename" },
        { "string filename=(u>.5 ? \"hart-test-texture.exr\" : \"hart_texture_alpha_4.exr\")",
          "", "filename", "hart-test-texture.exr" },
        { "output string filename=\"hart-test-texture.exr\"",
          "if(u>.5) filename=\"hart_texture_alpha_4.exr\";", "filename",
          nullptr, ParamHints::none, false,
          "texture requires a literal filename", false, true },
        { "output string filename=\"hart-test-texture.exr\"", "", "filename" },
        { "string filename=\"hart-test-texture.exr\"",
          "string selected=filename; if(u>.5) selected=\"hart_texture_alpha_4.exr\";",
          "selected" },
        { "string filenames[2]={\"hart-test-texture.exr\",\"hart_texture_alpha_4.exr\"}",
          "string selected=filenames[int(2*u)];", "selected" },
        { "string filename=\"hart-test-texture.exr\" [[int interpolated=1]]",
          "", "filename" },
        { "string filename=\"hart-test-texture.exr\" [[int interactive=1]]", "",
          "filename" },
        { "string filename=\"hart-test-texture.exr\"", "", "filename",
          "hart_texture_alpha_4.exr", ParamHints::interpolated },
        { "string filename=\"hart-test-texture.exr\"", "", "filename",
          "hart_texture_alpha_4.exr", ParamHints::interactive },
        { "string filename=\"hart-test-texture.exr\" "
          "[[int interpolated=1,int interactive=1]]",
          "", "filename", nullptr, ParamHints::none, false,
          "must be a numeric input" },
        { "string filename=\"hart-test-texture.exr\"", "", "filename",
          "hart_texture_alpha_4.exr",
          ParamHints::interpolated | ParamHints::interactive, false,
          "must be a numeric input" },
        { "string value=\"hart-test-texture.exr\"", "", "value", nullptr,
          ParamHints::none, true },
        { "string filename=\"missing.exr\"", "", "filename", nullptr,
          ParamHints::none, false, "cannot prepare texture 'missing.exr'",
          true },
    };
    for (const auto& test : rejected_names) {
        const auto rejected_source = fmtformat(
            "shader hart_rejected_filename({},output color Cout=0) {{ {} "
            "Cout=texture({},u,v,\"interp\",\"linear\",\"wrap\",\"periodic\"); }}",
            test.declaration, test.body, test.name);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(rejected_source, bytecode, { }, stdosl))
            return false;
        if (test.written_input) {
            // The source compiler forbids input writes. Retain the real body
            // and write hints, changing only the parameter's OSO classification.
            const std::string declaration = "\noparam\tstring\tfilename\t";
            const auto offset             = bytecode.find(declaration);
            OIIO_CHECK_ASSERT(offset != std::string::npos);
            if (offset == std::string::npos)
                return false;
            OIIO_CHECK_EQUAL(bytecode.find(declaration,
                                           offset + declaration.size()),
                             std::string::npos);
            OIIO_CHECK_ASSERT(bytecode.find("\n\tassign\t\tfilename ")
                              != std::string::npos);
            bytecode.replace(offset + 1, sizeof("oparam") - 1, "param");
        }
        for (int optimize : { 0, 2 }) {
            HartMutableServices renderer(true, false, false, true, false,
                                         false, false, false, true);
            {
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
                OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", 10));
                auto group = test.connected
                                 ? make_connected_group(ss, producer, bytecode)
                                 : overridden_group(ss, bytecode,
                                                    test.override_name,
                                                    test.hints);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                if (test.written_input) {
                    OIIO_CHECK_EQUAL(group->nlayers(), 1);
                    if (group->nlayers() == 1) {
                        const auto* instance             = group->layer(0);
                        const OSL::pvt::Symbol* filename = nullptr;
                        for (int p = instance->firstparam();
                             p < instance->lastparam(); ++p) {
                            const auto* symbol = instance->mastersymbol(p);
                            if (symbol->name() == ustring("filename"))
                                filename = symbol;
                        }
                        OIIO_CHECK_ASSERT(filename);
                        if (filename) {
                            OIIO_CHECK_ASSERT(filename->symtype()
                                              == OSL::pvt::SymTypeParam);
                            OIIO_CHECK_ASSERT(
                                filename->typespec().is_string()
                                && !filename->typespec().is_array());
                            OIIO_CHECK_ASSERT(!filename->interpolated()
                                              && !filename->interactive());
                            OIIO_CHECK_EQUAL(filename->get_string(),
                                             ustring("hart-test-texture.exr"));
                            OIIO_CHECK_ASSERT(filename->everwritten());
                            OIIO_CHECK_ASSERT(!filename->has_init_ops());
                        }
                    }
                }
                check_rejected_group(ss, *group, errors, test.error);
                OIIO_CHECK_EQUAL(renderer.texture_requests > 0, test.prepare);
                for (auto filename : renderer.texture_filenames)
                    OIIO_CHECK_EQUAL(filename, ustring("missing.exr"));
            }
            OIIO_CHECK_EQUAL(renderer.arena.frees,
                             renderer.arena.successful_allocations);
            OIIO_CHECK_ASSERT(!renderer.arena.storage);
        }
    }
    const struct {
        string_view options, error;
        bool defaults;
        string_view declaration = "";
    } rejected_options[] = {
        { "", "requires explicit closest or linear", false },
        { ",\"interp\",\"closest\"", "requires explicit wrap", false },
        { ",\"wrap\",\"clamp\"", "requires explicit closest or linear", false },
        { ",\"interp\",\"linear\",\"swrap\",\"clamp\"",
          "requires explicit wrap", false },
        { ",\"interp\",\"linear\",\"twrap\",\"black\"",
          "requires explicit wrap", false },
        { ",\"interp\",\"cubic\"", "unsupported texture interpolation 'cubic'",
          true },
        { ",\"interp\",\"smartbicubic\"",
          "unsupported texture interpolation 'smartbicubic'", true },
        { ",\"wrap\",\"default\"", "unsupported texture wrap mode 'default'",
          true },
        { ",\"wrap\",\"mirror\"", "unsupported texture wrap mode 'mirror'",
          true },
        { ",\"width\",1", "unsupported texture option 'width'", true },
        { ",\"wrap\",mode", "texture option values must be literal strings",
          true, "string mode=\"periodic\"," },
        { ",option,mode", "texture option names must be literal strings", true,
          "string option=\"wrap\",output string mode=\"clamp\"," },
    };
    for (const auto& test : rejected_options) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat(
                    "shader hart_rejected_defaults({}output color Cout=0) {{ "
                    "Cout=texture(\"hart-test-texture.exr\",u,v{}); }}",
                    test.declaration, test.options),
                bytecode, { }, stdosl))
            return false;
        for (int optimize : { 0, 2 }) {
            HartServices renderer(true, false, false, false, false, false,
                                  false, false, test.defaults);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
            auto group = make_group(ss, bytecode);
            check_rejected_group(ss, *group, errors, test.error);
            OIIO_CHECK_EQUAL(renderer.texture_requests, 0);
        }
    }
    {
        HartServices renderer(false, false, false, false, false, false, false,
                              false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        auto group = make_group(ss, options_bytecode);
        check_rejected_group(ss, *group, errors, "renderer lacks HARTTextures");
        OIIO_CHECK_EQUAL(renderer.texture_requests, 0);
    }
    return true;
}



bool
check_texture_modules(string_view arch, string_view stdosl)
{
    for (string_view type : { "float", "color" }) {
        const auto body = fmtformat(
            "{0} a=texture(\"hart-test-texture.exr\",u,v,"
            "\"interp\",\"linear\",\"wrap\",\"periodic\"); "
            "{0} b=texture(\"hart-test-texture.exr\",u,v,0.25,0,0,0.5,"
            "\"interp\",\"closest\",\"swrap\",\"clamp\",\"twrap\",\"black\"); "
            "value=a+b; ",
            type);
        const std::string sources[] = {
            fmtformat("shader hart_texture_test(output color Cout=0) {{ "
                      "{} value=0; {} Cout=color(value); }}",
                      type, body),
            fmtformat("shader hart_texture_producer(output {} value=0) {{ {} }}",
                      type, body),
            fmtformat(
                "shader hart_texture_consumer({} value=0, output color Cout=0) {{ "
                "Cout=color(value+Dx(value)+Dy(value)); }}",
                type),
        };
        std::string bytecode[3];
        for (size_t i = 0; i < std::size(sources); ++i) {
            OSLCompiler compiler;
            if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
                return false;
        }
        for (int optimize : { 10, 3 }) {
            for (int handles : { 0, 1 }) {
                for (bool connected : { false, true }) {
                    HartServices renderer(true);
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("llvm_optimize", optimize);
                    ss.attribute("opt_texture_handle", handles);
                    auto group = connected
                                     ? make_connected_group(ss, bytecode[1],
                                                            bytecode[2])
                                     : make_group(ss, bytecode[0]);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "Texture {}: {}\n", type,
                              errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    OIIO_CHECK_ASSERT(renderer.texture_requests > 0);
                    check_module(ss, *group, arch,
                                 { "osl_texture", "osl_init_texture_options",
                                   "osl_texture_set_interp_code",
                                   "osl_texture_set_stwrap_code" },
                                 optimize, connected);
                }
            }
        }
    }
    const struct {
        string_view expression;
        string_view error;
    } rejected[] = {
        { "texture(\"x.exr\",u,v)", "requires explicit closest or linear" },
        { "texture(\"x.exr\",u,v,\"interp\",\"linear\")",
          "requires explicit wrap" },
        { "texture(\"\",u,v,\"interp\",\"linear\",\"wrap\",\"clamp\")",
          "requires a literal filename" },
        { "texture(\"x.exr\",u,v,\"interp\",\"cubic\",\"wrap\",\"clamp\")",
          "unsupported texture interpolation 'cubic'" },
        { "texture(\"x.exr\",u,v,\"interp\",\"linear\",\"wrap\",\"default\")",
          "unsupported texture wrap mode 'default'" },
        { "texture(\"x.exr\",u,v,\"interp\",\"linear\",\"wrap\",\"clamp\",\"width\",1)",
          "unsupported texture option 'width'" },
        { "texture(\"x.exr\",u,v,\"interp\",\"linear\",\"wrap\",\"clamp\",\"missingalpha\",alpha)",
          "unsupported texture option 'missingalpha'" },
        { "texture(\"missing.exr\",u,v,\"interp\",\"linear\",\"wrap\",\"clamp\")",
          "cannot prepare texture 'missing.exr'" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source = fmtformat(
            "shader hart_bad_texture(output color Cout=0) {{ float alpha=0; Cout=color({}); }}",
            test.expression);
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, test.error, 1, false, true);
    }
    for (string_view control :
         { "if (u > 0.5)", "for (int i=0; i<int(3*u); ++i)" }) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source = fmtformat(
            "shader hart_missing_texture(output color Cout=0) {{ {} {{ "
            "Cout=texture(\"missing.exr\",u,v,\"interp\",\"linear\","
            "\"wrap\",\"clamp\"); }} }}",
            control);
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, "cannot prepare texture 'missing.exr'",
                        1, false, true);
    }
    return check_texture_compatibility_modules(arch, stdosl)
           && check_texture_alpha_modules(arch, stdosl)
           && check_texture_firstchannel_modules(arch, stdosl);
}



struct MaterialFieldExpectation {
    int id;
    const char* key;
    int writes       = 1;
    const char* text = nullptr;
    double number    = 0;
    bool numeric     = false;
};



bool
check_material_closure_ir(ShadingSystem& ss, ShaderGroup& group,
                          cspan<std::pair<int, int>> counts,
                          cspan<MaterialFieldExpectation> fields, bool weighted,
                          bool null_input)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size)
        return false;
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_material_closures"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        return false;
    }
    auto& module        = **parsed;
    const auto& layout  = module.getDataLayout();
    auto* groupdata     = llvm::StructType::getTypeByName(context, "Groupdata");
    auto* shaderglobals = llvm::StructType::getTypeByName(context,
                                                          "ShaderGlobals");
    auto leaf_type      = [&](const auto& self, llvm::Type* type,
                              uint64_t offset) -> llvm::Type* {
        if (auto* array = llvm::dyn_cast<llvm::ArrayType>(type)) {
            const uint64_t stride
                = layout.getTypeAllocSize(array->getElementType())
                      .getFixedValue();
            return stride && offset / stride < array->getNumElements()
                       ? self(self, array->getElementType(), offset % stride)
                       : nullptr;
        }
        if (auto* record = llvm::dyn_cast<llvm::StructType>(type)) {
            if (record->isOpaque() || !record->getNumElements()
                || offset >= layout.getTypeAllocSize(record).getFixedValue())
                return nullptr;
            const auto* members = layout.getStructLayout(record);
            const unsigned i    = members->getElementContainingOffset(offset);
            return self(self, record->getElementType(i),
                        offset - members->getElementOffset(i));
        }
        return offset == 0 ? type : nullptr;
    };
    auto source_type = [&](const llvm::Value* pointer,
                           const llvm::Function& function) -> llvm::Type* {
        int64_t offset = 0;
        const auto* base
            = llvm::GetPointerBaseWithConstantOffset(pointer, offset, layout);
        llvm::Type* type = nullptr;
        if (const auto* global = llvm::dyn_cast<llvm::GlobalVariable>(base))
            type = global->getValueType();
        if (const auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(base))
            type = alloca->getAllocatedType();
        if (!type && groupdata
            && interactive_address(pointer, function.getArg(1), layout, offset))
            type = groupdata;
        if (!type && shaderglobals
            && interactive_address(pointer, function.getArg(0), layout, offset))
            type = shaderglobals;
        return type && offset >= 0 ? leaf_type(leaf_type, type, offset)
                                   : nullptr;
    };
    auto constant_at = [&](const llvm::Value* pointer,
                           unsigned component) -> const llvm::Constant* {
        int64_t offset = 0;
        const auto* base
            = llvm::GetPointerBaseWithConstantOffset(pointer, offset, layout);
        const auto* global = llvm::dyn_cast<llvm::GlobalVariable>(base);
        if (!global || !global->hasInitializer() || offset < 0)
            return nullptr;
        const auto* array = llvm::dyn_cast<llvm::ArrayType>(
            global->getValueType());
        if (!array)
            return nullptr;
        const auto stride
            = layout.getTypeAllocSize(array->getElementType()).getFixedValue();
        const uint64_t index = stride ? uint64_t(offset) / stride + component
                                      : 0;
        return stride && uint64_t(offset) % stride == 0
                       && index < array->getNumElements()
                   ? global->getInitializer()->getAggregateElement(
                         unsigned(index))
                   : nullptr;
    };
    std::vector<int> seen(counts.size(), 0);
    int weighted_calls = 0, closure_copies = 0, null_copies = 0;
    for (auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0)
            continue;
        llvm::DominatorTree dominators(function);
        for (const auto& block : function)
            for (const auto& inst : block) {
                const auto* allocation = llvm::dyn_cast<llvm::CallBase>(&inst);
                const auto* helper     = allocation
                                             ? allocation->getCalledFunction()
                                             : nullptr;
                if (!helper)
                    continue;
                const bool is_weighted
                    = helper->getName()
                      == "osl_allocate_weighted_closure_component";
                if (!is_weighted
                    && helper->getName() != "osl_allocate_closure_component")
                    continue;
                weighted_calls += is_weighted;
                OIIO_CHECK_EQUAL(allocation->arg_size(), is_weighted ? 4 : 3);
                OIIO_CHECK_EQUAL(
                    allocation->getArgOperand(0)->stripPointerCasts(),
                    function.getArg(0));
                const auto* id = llvm::dyn_cast<llvm::ConstantInt>(
                    allocation->getArgOperand(1));
                const auto* size = llvm::dyn_cast<llvm::ConstantInt>(
                    allocation->getArgOperand(2));
                OIIO_CHECK_ASSERT(id && size);
                if (!id || !size)
                    continue;
                const auto entry = std::find_if(
                    material_closures().begin(), material_closures().end(),
                    [&](const auto& e) {
                        return uint64_t(e.id) == id->getZExtValue();
                    });
                OIIO_CHECK_ASSERT(entry != material_closures().end());
                if (entry == material_closures().end())
                    continue;
                OIIO_CHECK_EQUAL(size->getZExtValue(),
                                 uint64_t(entry->params.back().offset));
                int matched = 0;
                for (size_t i = 0; i < counts.size(); ++i)
                    if (counts[i].first == entry->id) {
                        ++seen[i];
                        ++matched;
                    }
                OIIO_CHECK_EQUAL(matched, 1);
                auto guarded = [&](const llvm::Instruction* write) {
                    for (const auto& candidate : function) {
                        const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
                            candidate.getTerminator());
                        const auto* cmp = branch && branch->isConditional()
                                              ? llvm::dyn_cast<llvm::ICmpInst>(
                                                    branch->getCondition())
                                              : nullptr;
                        if (!cmp || !cmp->isEquality())
                            continue;
                        for (unsigned a = 0; a < 2; ++a) {
                            if (cmp->getOperand(a) != allocation
                                || !llvm::isa<llvm::ConstantPointerNull>(
                                    cmp->getOperand(1 - a)))
                                continue;
                            const unsigned good
                                = cmp->getPredicate() == llvm::CmpInst::ICMP_NE
                                      ? 0
                                      : 1;
                            if (dominators.dominates(branch->getSuccessor(good),
                                                     write->getParent())
                                && !dominators.dominates(branch->getSuccessor(
                                                             1 - good),
                                                         write->getParent()))
                                return true;
                        }
                    }
                    return false;
                };
                const llvm::MemSetInst* clear = nullptr;
                std::vector<const llvm::MemCpyInst*> copies;
                for (const auto& body : function)
                    for (const auto& write : body) {
                        int64_t offset = 0;
                        if (const auto* zero = llvm::dyn_cast<llvm::MemSetInst>(
                                &write)) {
                            if (!interactive_address(zero->getDest(),
                                                     allocation, layout,
                                                     offset))
                                continue;
                            OIIO_CHECK_ASSERT(!clear);
                            clear = zero;
                            OIIO_CHECK_EQUAL(offset, 16);
                            const auto* length
                                = llvm::dyn_cast<llvm::ConstantInt>(
                                    zero->getLength());
                            const auto* value
                                = llvm::dyn_cast<llvm::ConstantInt>(
                                    zero->getValue());
                            OIIO_CHECK_ASSERT(length && value
                                              && value->isZero());
                            if (length)
                                OIIO_CHECK_EQUAL(length->getZExtValue(),
                                                 size->getZExtValue());
                            OIIO_CHECK_ASSERT(guarded(zero));
                        }
                        if (const auto* copy = llvm::dyn_cast<llvm::MemCpyInst>(
                                &write))
                            if (interactive_address(copy->getDest(), allocation,
                                                    layout, offset))
                                copies.push_back(copy);
                    }
                OIIO_CHECK_ASSERT(clear);
                std::vector<int> writes(entry->params.size() - 1, 0);
                std::vector<const llvm::MemCpyInst*> last(writes.size(),
                                                          nullptr);
                for (const auto* copy : copies) {
                    int64_t offset = 0;
                    OIIO_CHECK_ASSERT(interactive_address(copy->getDest(),
                                                          allocation, layout,
                                                          offset));
                    OIIO_CHECK_ASSERT(guarded(copy));
                    if (clear)
                        OIIO_CHECK_ASSERT(dominators.dominates(clear, copy));
                    const auto* length = llvm::dyn_cast<llvm::ConstantInt>(
                        copy->getLength());
                    OIIO_CHECK_ASSERT(length);
                    int matches = 0;
                    for (size_t p = 0; length && p < writes.size(); ++p) {
                        const auto& param = entry->params[p];
                        if (offset != 16 + param.offset
                            || length->getZExtValue()
                                   != uint64_t(param.field_size))
                            continue;
                        ++matches;
                        ++writes[p];
                        if (last[p])
                            OIIO_CHECK_ASSERT(
                                dominators.dominates(last[p], copy));
                        last[p]          = copy;
                        const auto* type = source_type(copy->getSource(),
                                                       function);
                        OIIO_CHECK_ASSERT(type);
                        if (type)
                            OIIO_CHECK_ASSERT(
                                param.type == TypeDesc::PTR
                                    ? type->isPointerTy()
                                          && type->getPointerAddressSpace() == 0
                                : param.type == TypeString
                                    ? type->isIntegerTy(64)
                                : param.type.elementtype() == TypeInt
                                    ? type->isIntegerTy(32)
                                    : type->isFloatTy());
                        if (param.type == TypeDesc::PTR) {
                            ++closure_copies;
                            const llvm::StoreInst* latest = nullptr;
                            int64_t source_offset         = 0;
                            const auto* source
                                = llvm::GetPointerBaseWithConstantOffset(
                                    copy->getSource(), source_offset, layout);
                            // A self-nested assignment must not publish the new
                            // parent into its input slot before copying that slot.
                            for (const auto& b : function)
                                for (const auto& instruction : b) {
                                    const auto* store
                                        = llvm::dyn_cast<llvm::StoreInst>(
                                            &instruction);
                                    if (!store)
                                        continue;
                                    int64_t offset = 0;
                                    const auto* base
                                        = llvm::GetPointerBaseWithConstantOffset(
                                            store->getPointerOperand(), offset,
                                            layout);
                                    if (base != source
                                        || offset != source_offset
                                        || !dominators.dominates(store, copy))
                                        continue;
                                    OIIO_CHECK_ASSERT(store->getValueOperand()
                                                      != allocation);
                                    if (!latest
                                        || dominators.dominates(latest, store))
                                        latest = store;
                                }
                            if (latest
                                && llvm::isa<llvm::ConstantPointerNull>(
                                    latest->getValueOperand()))
                                ++null_copies;
                        }
                    }
                    OIIO_CHECK_EQUAL(matches, 1);
                }
                for (size_t p = 0; p < writes.size(); ++p) {
                    const auto& param = entry->params[p];
                    const auto field  = std::find_if(
                        fields.begin(), fields.end(), [&](const auto& f) {
                            return f.id == entry->id
                                   && (param.key ? f.key
                                                       && string_view(param.key)
                                                              == f.key
                                                 : f.key == nullptr);
                        });
                    const int expected = field != fields.end() ? field->writes
                                         : param.key           ? 0
                                                               : 1;
                    OIIO_CHECK_EQUAL(writes[p], expected);
                    if (!last[p] || field == fields.end())
                        continue;
                    if (field->text && param.type == TypeString) {
                        const auto* hash
                            = llvm::dyn_cast_or_null<llvm::ConstantInt>(
                                constant_at(last[p]->getSource(), 0));
                        OIIO_CHECK_ASSERT(hash && hash->getBitWidth() == 64);
                        if (hash)
                            OIIO_CHECK_EQUAL(hash->getZExtValue(),
                                             ustringhash(field->text).hash());
                    }
                    if (field->numeric && param.type.basetype != TypeDesc::PTR)
                        for (unsigned c = 0; c < param.type.aggregate; ++c) {
                            const auto* value
                                = constant_at(last[p]->getSource(), c);
                            if (param.type == TypeInt) {
                                const auto* number
                                    = llvm::dyn_cast_or_null<llvm::ConstantInt>(
                                        value);
                                OIIO_CHECK_ASSERT(number);
                                if (number)
                                    OIIO_CHECK_EQUAL(number->getSExtValue(),
                                                     int64_t(field->number));
                            } else {
                                const auto* number
                                    = llvm::dyn_cast_or_null<llvm::ConstantFP>(
                                        value);
                                OIIO_CHECK_ASSERT(number);
                                if (number)
                                    OIIO_CHECK_EQUAL(
                                        number->getValueAPF().convertToDouble(),
                                        field->number);
                            }
                        }
                }
            }
    }
    for (size_t i = 0; i < counts.size(); ++i)
        OIIO_CHECK_EQUAL(seen[i], counts[i].second);
    OIIO_CHECK_EQUAL(weighted_calls > 0, weighted);
    OIIO_CHECK_ASSERT(closure_copies > 0);
    if (null_input)
        OIIO_CHECK_ASSERT(null_copies > 0);
    return true;
}



bool
check_material_closure_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader material_old(output closure color value=0, output color Cout=0) { "
        "color weight=color(u,v,0.5); "
        "closure color base=oren_nayar_diffuse_bsdf(normal(0,0,1),color(0.5),0.25); "
        "value=microfacet(\"ggx\",normal(0,0,1),vector(1,0,0),u+0.125,0.25,1.5,1); "
        "value=weight*layer(value,base); "
        "value=value+uniform_edf(color(0.125),\"label\",\"lamp\"); "
        "Ci=value; Cout=color(u,v,0.5); }",
        "shader material_mx(output closure color value=0, output color Cout=0) { "
        "string dist=u>v?\"ggx\":\"beckmann\"; float rough=0.125+u; "
        "value=dielectric_bsdf(normal(0,0,1),vector(1,0,0),color(0.75),color(0.25),"
        "rough,0.25,1.5,dist,\"thinfilm_thickness\",0.5,\"thinfilm_ior\",1.25,"
        "\"absorption\",color(0.125),\"dispersion\",0.25); "
        "closure color coat=color(0.5,0.25,0.75)*"
        "sheen_bsdf(normal(0,0,1),color(0.5),0.25,"
        "\"label\",\"coat\",\"mode\",0,\"mode\",1); "
        "value=layer(coat,value); Ci=value; Cout=color(Dx(rough),v,1); }",
        "shader material_defaults(output closure color value=0, output color Cout=0) { "
        "closure color empty=0; "
        "value=layer(empty,sheen_bsdf(normal(0,0,1),color(0.5),0.25)); "
        "Ci=value; Cout=color(u,v,1); }",
        "closure color hart_material_probe(matrix basis, closure color input)"
        " [[int builtin=1]]; "
        "shader material_probe(output closure color value=0, output color Cout=0) { "
        "closure color empty=0; "
        "value=emission(); value=hart_material_probe(matrix(1),empty,"
        "\"fallback\",value,\"gain\",0.5,\"mode\",7,\"label\",\"\"); "
        "Ci=value; Cout=color(u,v,1); }",
        "closure color diffuse_ramp(normal n, color colors[8]) [[int builtin=1]]; "
        "shader material_diffuse_ramp(output closure color value=0, output color Cout=0) { "
        "color colors[8]; for(int i=0;i<8;i++) colors[i]=color(u+i,v-i,i+1); "
        "closure color empty=0; value=color(0.5,0.25,0.75)*"
        "diffuse_ramp(normal(0,0,1),colors,\"label\",\"diffuse ramp\"); "
        "value=layer(value,empty); Ci=value; Cout=Dx(colors[1]); }",
        "closure color phong_ramp(normal n, float exponent, color colors[8]) [[int builtin=1]]; "
        "shader material_phong_ramp("
        "color colors[8]={color(1),color(2),color(3),color(4),color(5),color(6),color(7),color(8)}, "
        "output closure color value=0, output color Cout=0) { "
        "closure color empty=0; value=phong_ramp(normal(0,0,1),u+4,colors,"
        "\"label\",\"phong ramp\"); value=layer(value,empty); Ci=value; Cout=color(u,v,1); }",
        "closure color hart_array_probe(int integers[2], float floats[2], point points[2], "
        "vector vectors[2], normal normals[2], matrix matrices[2]) [[int builtin=1]]; "
        "shader material_arrays(output closure color value=0, output color Cout=0) { "
        "int integers[2]={int(u),7}; float floats[2]={u,v}; "
        "point points[2]={point(u),point(v)}; vector vectors[2]={vector(u),vector(v)}; "
        "normal normals[2]={normal(u),normal(v)}; matrix matrices[2]={matrix(u),matrix(v)}; "
        "color palette[8]={color(1),color(2),color(3),color(4),color(5),color(6),color(7),color(8)}; "
        "closure color empty=0; value=hart_array_probe(integers,floats,points,vectors,"
        "normals,matrices,\"palette\",palette,\"gain\",0.5); "
        "value=layer(value,empty); Ci=value; Cout=color(u,v,1); }",
    };
    std::string oso[std::size(sources)];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, stdosl))
            return false;
    }
    const std::vector<std::pair<int, int>> counts[] = {
        { { 8, 1 }, { 15, 1 }, { 24, 1 }, { 27, 1 } },
        { { 17, 1 }, { 23, 1 }, { 27, 1 } },
        { { 23, 1 }, { 27, 1 } },
        { { 1, 1 }, { 101, 1 } },
        { { 102, 1 }, { 27, 1 } },
        { { 103, 1 }, { 27, 1 } },
        { { 104, 1 }, { 27, 1 } },
    };
    const std::vector<MaterialFieldExpectation> fields[] = {
        { { 8, nullptr, 1, "ggx" }, { 24, "label", 1, "lamp" } },
        { { 17, "thinfilm_thickness", 1, nullptr, 0.5, true },
          { 17, "thinfilm_ior", 1, nullptr, 1.25, true },
          { 17, "absorption", 1, nullptr, 0.125, true },
          { 17, "dispersion", 1, nullptr, 0.25, true },
          { 23, "label", 1, "coat" },
          { 23, "mode", 2, nullptr, 1, true } },
        { },
        { { 101, "fallback" },
          { 101, "gain", 1, nullptr, 0.5, true },
          { 101, "mode", 1, nullptr, 7, true },
          { 101, "label", 1, "" } },
        { { 102, "label", 1, "diffuse ramp" } },
        { { 103, "label", 1, "phong ramp" } },
        { { 104, "palette" }, { 104, "gain", 1, nullptr, 0.5, true } },
    };
    std::vector<std::pair<int, int>> layouts;
    for (const auto& entry : material_closures())
        layouts.emplace_back(entry.id, entry.params.back().offset);
    for (size_t source = 0; source < std::size(sources); ++source) {
        const struct {
            int osl, llvm, budget;
        } variants[] = { { 0, 10, 0 },
                         { 2, 10, 4096 },
                         { 2, 3, source % 2 ? 0 : 4096 } };
        for (const auto& variant : variants) {
            HartMaterialServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            for (const auto& entry : material_closures())
                ss.register_closure(entry.name, entry.id, entry.params.data(),
                                    nullptr, nullptr);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
            OIIO_CHECK_ASSERT(
                ss.attribute("max_hart_groupdata_alloc", variant.budget));
            auto group = make_group(ss, oso[source]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "Material closures {} OSL{} LLVM{}: {}\n", source,
                      variant.osl, variant.llvm, errors.messages);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch, { }, variant.llvm, false, false,
                         false, 0, true, false, 0, -1, -1, { }, false, layouts);
            if (variant.llvm == 10
                && !check_material_closure_ir(ss, *group, counts[source],
                                              fields[source],
                                              (source < 2 || source == 4)
                                                  && variant.osl == 2,
                                              source >= 2))
                return false;
            int allocated = -1, size = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                              "hart_groupdata_alloc",
                                              allocated));
            OIIO_CHECK_ASSERT(
                ss.getattribute(group.get(), "llvm_groupdata_size", size));
            OIIO_CHECK_EQUAL(allocated, variant.budget ? size : 0);
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
            OIIO_CHECK_EQUAL(closure_callback_calls, 0);
        }
    }
    enum class RegistryFailure {
        Missing,
        Prepare,
        Setup,
        Empty,
        Extent,
        Alignment,
        EndKey,
        NegativeOffset,
        Outside,
        Misaligned,
        Overlap,
        KeywordSize,
        UnsupportedType,
        Array,
        Order,
        DuplicateKey
    };
    const RegistryFailure failures[] = {
        RegistryFailure::Missing,         RegistryFailure::Prepare,
        RegistryFailure::Setup,           RegistryFailure::Empty,
        RegistryFailure::Extent,          RegistryFailure::Alignment,
        RegistryFailure::EndKey,          RegistryFailure::NegativeOffset,
        RegistryFailure::Outside,         RegistryFailure::Misaligned,
        RegistryFailure::Overlap,         RegistryFailure::KeywordSize,
        RegistryFailure::UnsupportedType, RegistryFailure::Array,
        RegistryFailure::Order,           RegistryFailure::DuplicateKey,
    };
    for (RegistryFailure failure : failures) {
        HartMaterialServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        for (const auto& entry : material_closures()) {
            if (entry.id != 23) {
                ss.register_closure(entry.name, entry.id, entry.params.data(),
                                    nullptr, nullptr);
                continue;
            }
            if (failure == RegistryFailure::Missing)
                continue;
            auto params = entry.params;
            switch (failure) {
            case RegistryFailure::Extent: params.back().offset = 16; break;
            case RegistryFailure::Alignment:
                params.back().field_size = 3;
                break;
            case RegistryFailure::EndKey:
                params.back().key = "not_an_end";
                break;
            case RegistryFailure::NegativeOffset: params[0].offset = -4; break;
            case RegistryFailure::Outside: params[0].offset = 40; break;
            case RegistryFailure::Misaligned: params[0].offset = 9; break;
            case RegistryFailure::Overlap:
                params[1].offset = params[0].offset;
                break;
            case RegistryFailure::KeywordSize: params[3].field_size = 4; break;
            case RegistryFailure::UnsupportedType:
                params[3].type = TypeDesc::DOUBLE;
                break;
            case RegistryFailure::Array:
                params[3].type       = TypeDesc(TypeDesc::STRING, 2);
                params[3].field_size = 16;
                break;
            case RegistryFailure::Order: std::swap(params[1], params[3]); break;
            case RegistryFailure::DuplicateKey:
                params[4].key = params[3].key;
                break;
            default: break;
            }
            ss.register_closure(
                entry.name, entry.id,
                failure == RegistryFailure::Empty ? nullptr : params.data(),
                failure == RegistryFailure::Prepare ? host_closure_callback
                                                    : nullptr,
                failure == RegistryFailure::Setup ? host_closure_callback
                                                  : nullptr);
        }
        OIIO_CHECK_EQUAL(errors.errors, 0);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
        auto group          = make_group(ss, oso[2]);
        const char* message = failure == RegistryFailure::Missing
                                  ? "not registered"
                              : failure == RegistryFailure::Prepare
                                      || failure == RegistryFailure::Setup
                                  ? "prepare/setup callbacks"
                              : failure == RegistryFailure::UnsupportedType
                                      || failure == RegistryFailure::Array
                                  ? "parameter type"
                                  : "parameter layout";
        check_rejected_group(ss, *group, errors, message);
        OIIO_CHECK_EQUAL(closure_callback_calls, 0);
    }
    OSLCompiler clean_compiler;
    std::string clean;
    if (!clean_compiler.compile_buffer(
            "shader material_clean(output color Cout=0) { Cout=color(u,v,1); }",
            clean, { }, stdosl))
        return false;
    for (int failure = 0; failure < 4; ++failure) {
        HartMaterialServices renderer;
        renderer.closures   = failure != 0;
        renderer.parameters = failure != 1 && failure != 2;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        for (const auto& entry : material_closures())
            ss.register_closure(entry.name, entry.id, entry.params.data(),
                                failure == 3 && entry.id == 23
                                    ? host_closure_callback
                                    : nullptr,
                                nullptr);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
        auto group = make_group(ss, oso[failure == 0 ? 0 : 2]);
        if (failure >= 2) {
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("material_clean", clean));
            group = ss.ShaderGroupBegin("hart_test_group");
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "unused"));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "material_clean", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("layer0.Cout", TypeColor, false,
                                         SymArena::Outputs, 0, 12);
            ss.add_symlocs(group.get(), { &output, 1 });
        }
        check_rejected_group(ss, *group, errors,
                             failure == 0   ? "unsupported type 'closure color'"
                             : failure == 3 ? "prepare/setup callbacks"
                                            : "unsupported closure");
        OIIO_CHECK_EQUAL(closure_callback_calls, 0);
    }
    OSLCompiler malformed_compiler;
    std::string original;
    if (!malformed_compiler.compile_buffer(
            "shader material_bad(string label=\"coat\", string words[2]={\"a\",\"b\"}, "
            "float gain=0.25, float numbers[2]={1,2}, closure color child=0, "
            "output closure color value=0, output color Cout=0) { "
            "value=sheen_bsdf(normal(0,0,1),color(0.5),gain,"
            "\"label\",\"coat\",\"mode\",1); Ci=value; Cout=color(u,v,1); }",
            original, { }, stdosl))
        return false;
    const auto marker = original.find("\n\tclosure\t");
    OIIO_CHECK_ASSERT(marker != std::string::npos);
    if (marker == std::string::npos)
        return false;
    const size_t op = marker + 1;
    const auto end  = original.find('\n', op);
    const auto hint = original.find('%', op);
    OIIO_CHECK_ASSERT(end != std::string::npos && hint < end);
    if (end == std::string::npos || hint >= end)
        return false;
    std::vector<std::string> operands;
    OIIO::Strutil::split(string_view(original).substr(op, hint - op), operands,
                         "", -1);
    OIIO_CHECK_EQUAL(operands.size(), 10);
    if (operands.size() != 10)
        return false;
    const auto original_hints   = original.substr(hint, end - hint);
    const std::string rw_prefix = "%argrw{\"";
    const auto rw               = original_hints.find(rw_prefix);
    const auto rw_end = rw == std::string::npos
                            ? std::string::npos
                            : original_hints.find('"', rw + rw_prefix.size());
    OIIO_CHECK_ASSERT(rw_end != std::string::npos);
    if (rw_end == std::string::npos)
        return false;
    const struct {
        unsigned operand;
        const char* replacement;
        const char* error;
    } bad_operands[] = {
        { 1, "gain", "invalid closure argument list" },
        { 2, "gain", "invalid closure weight" },
        { 2, "label", "closure names must be literal strings" },
        { 3, "numbers", "incompatible formal argument" },
        { 4, "child", "incompatible formal argument" },
        { 5, "child", "incompatible formal argument" },
        { 6, "label", "closure keyword names must be literal strings" },
        { 6, "gain", "closure keyword names must be literal strings" },
        { 7, "gain", "unsupported or incompatible keyword" },
        { 6, operands[7].c_str(), "unsupported or incompatible keyword" },
        { 9, "words", "unsupported or incompatible keyword" },
        { 0, nullptr, "invalid closure argument list" },
    };
    for (const auto& test : bad_operands) {
        auto words = operands;
        if (test.replacement)
            words[test.operand] = test.replacement;
        else
            words.pop_back();
        auto hints = original_hints;
        std::string access(words.size() - 1, 'r');
        access[0] = 'w';
        hints.replace(rw + rw_prefix.size(), rw_end - rw - rw_prefix.size(),
                      access);
        auto bytecode = original;
        bytecode.replace(op, end - op,
                         fmtformat("\t{}\t{}", OIIO::Strutil::join(words, "\t"),
                                   hints));
        HartMaterialServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        for (const auto& entry : material_closures())
            ss.register_closure(entry.name, entry.id, entry.params.data(),
                                nullptr, nullptr);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", 2));
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
        OIIO_CHECK_EQUAL(closure_callback_calls, 0);
    }
    for (const char* argument : { "gain", "label" }) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                "shader material_pointer(closure color child=0, float gain=0, "
                "string label=\"\", output closure color value=0, output color Cout=0) { "
                "value=layer(child,child); Ci=value; Cout=color(u,v,1); }",
                bytecode, { }, stdosl))
            return false;
        const auto marker = bytecode.find("\n\tclosure\t");
        OIIO_CHECK_ASSERT(marker != std::string::npos);
        if (marker == std::string::npos)
            return false;
        const auto hint = bytecode.find('%', marker);
        const auto end  = bytecode.find('\n', marker + 1);
        if (hint >= end) {
            OIIO_CHECK_ASSERT(false);
            return false;
        }
        std::vector<std::string> words;
        OIIO::Strutil::split(string_view(bytecode).substr(marker + 1,
                                                          hint - marker - 1),
                             words, "", -1);
        OIIO_CHECK_EQUAL(words.size(), 5);
        if (words.size() != 5)
            return false;
        words[3] = argument;
        bytecode.replace(marker + 1, hint - marker - 1,
                         fmtformat("\t{}\t", OIIO::Strutil::join(words, "\t")));
        HartMaterialServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        for (const auto& entry : material_closures())
            ss.register_closure(entry.name, entry.id, entry.params.data(),
                                nullptr, nullptr);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors,
                             "incompatible formal argument");
        OIIO_CHECK_EQUAL(closure_callback_calls, 0);
    }
    return true;
}



bool
check_closure_array_modules(string_view arch, string_view stdosl)
{
    const auto entry = std::find_if(material_closures().begin(),
                                    material_closures().end(),
                                    [](const auto& e) { return e.id == 102; });
    OIIO_CHECK_ASSERT(entry != material_closures().end());
    if (entry == material_closures().end())
        return false;
    const char* declaration
        = "closure color diffuse_ramp(normal n, color colors[]) [[int builtin=1]]; ";
    const char* body
        = "output color Cout=0) { Ci=diffuse_ramp(normal(0,0,1),colors); "
          "Cout=color(u,v,1); }";
    const char* values
        = "{color(1),color(2),color(3),color(4),color(5),color(6),color(7),color(8)}";
    OSLCompiler compiler;
    std::string good;
    if (!compiler.compile_buffer(
            fmtformat(
                "closure color diffuse_ramp(normal n) [[int builtin=1]]; "
                "shader ramp(color colors[8]={}, output color Cout=0) {{ "
                "Ci=diffuse_ramp(normal(0,0,1),\"colors\",colors); Cout=color(u,v,1); }}",
                values),
            good, { }, stdosl))
        return false;
    const TypeDesc color8(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR, 8);
    const struct {
        TypeDesc type;
        int size, offset;
        const char* error;
    } invalid_layouts[] = {
        { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR, -1), 96,
          20, "parameter type" },
        { TypeDesc(TypeDesc::STRING, 8), 64, 20, "parameter type" },
        { TypeDesc(TypeDesc::PTR, 8), 64, 20, "parameter type" },
        { TypeDesc(TypeDesc::DOUBLE, 8), 64, 20, "parameter type" },
        { color8, 12, 20, "parameter layout" },
        { color8, 96, 28, "parameter layout" },
        { color8, 96, 22, "parameter layout" },
        { color8, 96, 16, "parameter layout" },
        { color8, 96, -4, "parameter layout" },
        { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR,
                   std::numeric_limits<int>::max()),
          96, 20, "parameter layout" },
        { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR, 7), 84, 20,
          "unsupported or incompatible keyword" },
    };
    for (const auto& test : invalid_layouts) {
        auto params          = entry->params;
        params[1].type       = test.type;
        params[1].field_size = test.size;
        params[1].offset     = test.offset;
        // Keyword fields reach HART's checks rather than the host registry's
        // earlier formal-parameter size check.
        params[1].key = "colors";
        HartMaterialServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.register_closure(entry->name, entry->id, params.data(), nullptr,
                            nullptr);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        auto group = make_group(ss, good);
        check_rejected_group(ss, *group, errors, test.error);
    }
    for (bool missing_bounds : { false, true }) {
        HartMaterialServices renderer;
        renderer.arrays     = !missing_bounds;
        renderer.parameters = missing_bounds;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.register_closure(entry->name, entry->id, entry->params.data(),
                            nullptr, nullptr);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        auto group = make_group(ss, good);
        check_rejected_group(ss, *group, errors,
                             missing_bounds ? "HARTArrayBounds"
                                            : "unsupported closure");
    }
    for (bool keyword : { false, true }) {
        for (const char* type : { "color", "float", "closure color" }) {
            for (int length : { 0, 7, 8, 9, -1 }) {
                const std::string suffix = length == 0 ? ""
                                           : length < 0
                                               ? "[]"
                                               : fmtformat("[{}]", length);
                const std::string formal = fmtformat("{} colors{}", type,
                                                     suffix);
                const bool valid_type    = string_view(type) == "color";
                std::string initializer;
                for (int i = 0, n = length < 0 ? 8 : std::max(length, 1); i < n;
                     ++i) {
                    if (i)
                        initializer += ',';
                    initializer += valid_type ? "color(1)" : "0";
                }
                if (length)
                    initializer = "{" + initializer + "}";
                const std::string prototype = keyword ? "normal n"
                                                      : "normal n, " + formal;
                const char* argument = keyword ? "\"colors\",colors" : "colors";
                OSLCompiler compiler;
                std::string bytecode;
                if (!compiler.compile_buffer(
                        fmtformat(
                            "closure color diffuse_ramp({}) [[int builtin=1]]; "
                            "shader ramp({}={}, output color Cout=0) {{ "
                            "Ci=diffuse_ramp(normal(0,0,1),{}); Cout=color(u,v,1); }}",
                            prototype, formal, initializer, argument),
                        bytecode, { }, stdosl))
                    return false;
                for (int optimize : { 0, 2 }) {
                    HartMaterialServices renderer;
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    auto params = entry->params;
                    if (keyword)
                        params[1].key = "colors";
                    ss.register_closure(entry->name, entry->id, params.data(),
                                        nullptr, nullptr);
                    OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                    OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
                    auto group = make_group(ss, bytecode);
                    if (valid_type && (length == 8 || length < 0)) {
                        ss.optimize_group(group.get(), nullptr);
                        uint64_t size = 0;
                        OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                          "hart_bitcode_size",
                                                          TypeUInt64, &size));
                        OIIO_CHECK_ASSERT(size > 0);
                        OIIO_CHECK_EQUAL(errors.errors, 0);
                    } else {
                        check_rejected_group(
                            ss, *group, errors,
                            keyword ? "unsupported or incompatible keyword"
                                    : "incompatible formal argument");
                    }
                }
            }
        }
    }
    OSLCompiler unsized_compiler;
    std::string unsized;
    if (!unsized_compiler.compile_buffer(
            fmtformat("{} shader ramp(color colors[]={}, {}", declaration,
                      values, body),
            unsized, { }, stdosl))
        return false;
    for (int length : { 7, 8, 9 }) {
        Color3 colors[9];
        for (int i = 0; i < 9; ++i)
            colors[i] = Color3(float(i), float(i + 1), float(i + 2));
        for (int optimize : { 0, 2 }) {
            HartMaterialServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.register_closure(entry->name, entry->id, entry->params.data(),
                                nullptr, nullptr);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_test", unsized));
            auto group    = ss.ShaderGroupBegin("ramp_override");
            TypeDesc type = color8;
            type.arraylen = length;
            OIIO_CHECK_ASSERT(ss.Parameter("colors", type, colors));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            if (length == 8) {
                ss.optimize_group(group.get(), nullptr);
                uint64_t size = 0;
                OIIO_CHECK_ASSERT(ss.getattribute(group.get(),
                                                  "hart_bitcode_size",
                                                  TypeUInt64, &size));
                OIIO_CHECK_ASSERT(size > 0);
                OIIO_CHECK_EQUAL(errors.errors, 0);
            } else {
                check_rejected_group(ss, *group, errors,
                                     "incompatible formal argument");
            }
        }
    }
    return true;
}



bool
check_closure_modules(string_view arch, string_view stdosl)
{
    OSLCompiler consumer_compiler;
    std::string consumer;
    if (!consumer_compiler.compile_buffer(
            "shader hart_closure_consumer(closure color value=0, "
            "output closure color result=0, output color Cout=0) { "
            "result=value; if(u>v) Ci=result; else Ci=0; "
            "Cout=color(u,v,0.5); }",
            consumer, { }, stdosl))
        return false;
    const struct {
        const char* source;
        std::initializer_list<string_view> plain;
        std::initializer_list<string_view> optimized;
        bool connected;
    } tests[] = {
        { "shader hart_closure_smoke(output color Cout=0) { "
          "Ci=color(0.8,0.3,0.1)*diffuse(N)+color(0.02)*emission(); }",
          { "osl_allocate_closure_component", "osl_mul_closure_color",
            "osl_add_closure_closure", "rs_allocate_closure" },
          { "osl_allocate_weighted_closure_component",
            "osl_add_closure_closure", "rs_allocate_closure" },
          false },
        { "shader hart_closure_null(output color Cout=0) { Ci=0; }",
          { },
          { },
          false },
        { "shader hart_closure_string(string name=\"diffuse\", "
          "output color Cout=0) { Ci=diffuse(N); }",
          { "osl_allocate_closure_component", "rs_allocate_closure" },
          { "osl_allocate_closure_component", "rs_allocate_closure" },
          false },
        { "shader hart_closure_scalar(output color Cout=0) { "
          "Ci=(u+0.25)*diffuse(N); }",
          { "osl_allocate_closure_component", "osl_mul_closure_float",
            "rs_allocate_closure" },
          { "osl_allocate_closure_component", "osl_mul_closure_float",
            "rs_allocate_closure" },
          false },
        { "shader hart_closure_weighted(output color Cout=0) { "
          "Ci=color(u,v,0.5)*diffuse(N); }",
          { "osl_allocate_closure_component", "osl_mul_closure_color",
            "rs_allocate_closure" },
          { "osl_allocate_weighted_closure_component", "rs_allocate_closure" },
          false },
        { "shader hart_closure_zero(output color Cout=0) { "
          "Ci=color(0)*diffuse(N); }",
          { "osl_allocate_closure_component", "osl_mul_closure_color",
            "rs_allocate_closure" },
          { },
          false },
        { "shader hart_closure_conditional(output color Cout=0) { "
          "Ci=0; if(u>v) Ci=diffuse(N); "
          "if(v>0.5) Ci=Ci+emission(); Cout=color(u,v,0.5); }",
          { "osl_allocate_closure_component", "osl_add_closure_closure",
            "rs_allocate_closure" },
          { "osl_allocate_closure_component", "osl_add_closure_closure",
            "rs_allocate_closure" },
          false },
        { "shader hart_closure_parameter(closure color value=0, "
          "output closure color result=0, output color Cout=0) { "
          "result=value; Ci=result; }",
          { },
          { },
          false },
        { "shader hart_closure_producer(output closure color value=0) { "
          "value=color(u,v,0.5)*diffuse(N)+color(0.02)*emission(); }",
          { "osl_allocate_closure_component", "osl_mul_closure_color",
            "osl_add_closure_closure", "rs_allocate_closure" },
          { "osl_allocate_weighted_closure_component",
            "osl_add_closure_closure", "rs_allocate_closure" },
          true },
    };
    for (const auto& test : tests) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 })
                for (int budget : { 0, 4096 }) {
                    if (!test.connected
                        && (budget || ((osl_optimize == 0) != (optimize == 10))))
                        continue;
                    HartServices renderer(false, false, true);
                    Diagnostics errors;
                    ShadingSystem ss(&renderer, nullptr, &errors);
                    register_hart_closures(ss);
                    ss.attribute("hart_arch", arch);
                    ss.attribute("optimize", osl_optimize);
                    ss.attribute("llvm_optimize", optimize);
                    ss.attribute("max_hart_groupdata_alloc", budget);
                    auto group = test.connected
                                     ? make_connected_group(ss, bytecode,
                                                            consumer)
                                     : make_group(ss, bytecode);
                    ss.optimize_group(group.get(), nullptr);
                    if (errors.errors)
                        print(stderr, "{}: {}\n", test.source,
                              errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    check_module(ss, *group, arch,
                                 osl_optimize ? test.optimized : test.plain,
                                 optimize, test.connected, false, false,
                                 test.connected ? 2 : 0, true);
                }
    }

    enum Registry {
        Safe,
        Missing,
        Prepare,
        Setup,
        Both,
        WrongType,
        WrongOffset,
        NegativeOffset,
        WrongSize,
        WrongCount,
        WrongLabel,
    };
    const struct {
        const char* prefix;
        const char* expression;
        Registry registry;
        const char* error;
    } rejected[] = {
        { "", "diffuse(N)", Missing, "closure 'diffuse' is not registered" },
        { "", "emission()", Missing, "closure 'emission' is not registered" },
        { "", "background()", Safe, "unsupported closure 'background'" },
        { "", "diffuse(N)", Prepare, "prepare/setup callbacks" },
        { "", "diffuse(N)", Setup, "prepare/setup callbacks" },
        { "", "diffuse(N)", Both, "prepare/setup callbacks" },
        { "", "emission()", Prepare, "prepare/setup callbacks" },
        { "", "emission()", Setup, "prepare/setup callbacks" },
        { "", "emission()", Both, "prepare/setup callbacks" },
        { "", "diffuse(N,\"label\",\"test\")", Safe,
          "closure keyword arguments are unsupported" },
        { "", "diffuse(N,\"unknown\",1.0)", Safe,
          "closure keyword arguments are unsupported" },
        { "", "emission(\"label\",\"test\")", Safe,
          "closure keyword arguments are unsupported" },
        { "", "diffuse(N)", WrongType, "parameter layout" },
        { "", "diffuse(N)", WrongOffset, "parameter layout" },
        { "", "diffuse(N)", NegativeOffset, "parameter layout" },
        { "", "diffuse(N)", WrongSize, "parameter layout" },
        { "", "diffuse(N)", WrongCount, "parameter layout" },
        { "", "diffuse(N)", WrongLabel, "parameter layout" },
        { "closure color diffuse(float x) [[ int builtin=1 ]]; ", "diffuse(u)",
          Safe, "incompatible formal argument" },
        { "closure color diffuse() [[ int builtin=1 ]]; ", "diffuse()", Safe,
          "invalid closure argument list" },
    };
    for (const auto& test : rejected)
        for (int mode : { 0, 1, 2 }) {
            const auto source = fmtformat(
                "{} shader hart_bad_closure(int enable=0, "
                "output closure color value=0, output color Cout=0) {{ "
                "{} value={}; }}",
                test.prefix, mode == 1 ? "if(enable)" : "", test.expression);
            OSLCompiler compiler;
            std::string bytecode;
            if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
                return false;
            HartServices renderer(false, false, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", 2);
            ss.attribute("llvm_optimize", 3);
            if (test.registry != Missing) {
                std::vector<ClosureParam> params(std::begin(diffuse_params),
                                                 std::end(diffuse_params));
                if (test.registry == WrongType) {
                    params[0].type       = TypeFloat;
                    params[0].field_size = sizeof(float);
                }
                if (test.registry == WrongOffset)
                    params[0].offset = 4;
                if (test.registry == NegativeOffset)
                    params[0].offset = -4;
                if (test.registry == WrongSize)
                    params.back().offset = 8;
                if (test.registry == WrongCount)
                    params.erase(params.begin());
                if (test.registry == WrongLabel)
                    params[1].key = "other";
                const auto prepare = test.registry == Prepare
                                             || test.registry == Both
                                         ? host_closure_callback
                                         : nullptr;
                const auto setup   = test.registry == Setup
                                             || test.registry == Both
                                         ? host_closure_callback
                                         : nullptr;
                ss.register_closure("diffuse", 3, params.data(), prepare,
                                    setup);
                ss.register_closure("emission", 1, emission_params, prepare,
                                    setup);
            }
            OIIO_CHECK_EQUAL(errors.errors, 0);
            ShaderGroupRef group;
            if (mode == 2) {
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_bad_closure", bytecode));
                OIIO_CHECK_ASSERT(
                    ss.LoadMemoryCompiledShader("hart_closure_consumer",
                                                consumer));
                group = ss.ShaderGroupBegin("hart_test_group");
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_bad_closure", "unused"));
                OIIO_CHECK_ASSERT(
                    ss.Shader("surface", "hart_closure_consumer", "consumer"));
                OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            } else {
                group = make_group(ss, bytecode);
            }
            check_rejected_group(ss, *group, errors, test.error);
            OIIO_CHECK_EQUAL(closure_callback_calls, 0);
        }

    const struct {
        const char* source;
        const char* error;
    } types[] = {
        { "shader hart_closure_global(output color Cout=0) { "
          "N=normal(u,v,1); Ci=diffuse(N); }",
          "writing shader global 'N'" },
    };
    for (const auto& test : types) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        register_hart_closures(ss);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
    }
    // Without the opt-in, both constructors and plain Ci access stay rejected.
    for (const char* expression : { "diffuse(N)", "0" }) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                fmtformat("shader hart_no_closures(output color Cout=0) {{ "
                          "Ci={}; }}",
                          expression),
                bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode,
                        string_view(expression) == "0"
                            ? "unsupported type"
                            : "unsupported operation 'closure'");
    }
    return true;
}



bool
check_control_flow_ir(ShadingSystem& ss, ShaderGroup& group,
                      unsigned minimum_loop_depth, bool bitwise)
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    ustring entry_name;
    if (!ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes)
        || !ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size)
        || !ss.getattribute(&group, "group_entry_name", entry_name) || !bytes
        || !size) {
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_control_flow"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    const llvm::StringRef prefix("__direct_callable__");
    const llvm::StringRef name(entry_name.c_str());
    if (name.find(prefix) != 0) {
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    auto* entry = (*parsed)->getFunction(name.drop_front(prefix.size()));
    OIIO_CHECK_ASSERT(entry && !entry->isDeclaration());
    if (!entry || entry->isDeclaration())
        return false;

    llvm::DominatorTree dominators(*entry);
    llvm::LoopInfo loops(dominators);
    unsigned depth                   = 0;
    unsigned conditional_branches    = 0;
    const unsigned integer_opcodes[] = {
        llvm::Instruction::And, llvm::Instruction::Or, llvm::Instruction::Xor,
        llvm::Instruction::Shl, llvm::Instruction::AShr
    };
    bool seen[std::size(integer_opcodes)] = { };
    bool complement                       = false;
    for (const auto& block : *entry) {
        depth              = std::max(depth, loops.getLoopDepth(&block));
        const auto* branch = llvm::dyn_cast<llvm::BranchInst>(
            block.getTerminator());
        conditional_branches += branch && branch->isConditional();
        for (const auto& inst : block) {
            if (!inst.getType()->isIntegerTy(32))
                continue;
            for (size_t i = 0; i < std::size(integer_opcodes); ++i)
                seen[i] |= inst.getOpcode() == integer_opcodes[i];
            if (inst.getOpcode() == llvm::Instruction::Xor)
                for (const auto& operand : inst.operands())
                    if (const auto* value = llvm::dyn_cast<llvm::ConstantInt>(
                            operand.get()))
                        complement |= value->isMinusOne();
        }
    }
    OIIO_CHECK_ASSERT(conditional_branches > 0);
    OIIO_CHECK_ASSERT(depth >= minimum_loop_depth);
    if (bitwise) {
        for (size_t i = 0; i < std::size(integer_opcodes); ++i) {
            if (!seen[i])
                print(stderr, "Missing HART i32 instruction '{}'\n",
                      llvm::Instruction::getOpcodeName(integer_opcodes[i]));
            OIIO_CHECK_ASSERT(seen[i]);
        }
        OIIO_CHECK_ASSERT(complement);
    }
    return true;
}



bool
check_control_flow_modules(string_view arch, string_view stdosl,
                           cspan<std::string> basic_loops)
{
    OSLCompiler consumer_compiler;
    std::string consumer;
    if (!consumer_compiler.compile_buffer(
            "shader hart_flow_consumer(float value=42, output color Cout=0) { "
            "float sum=0; for(int i=0;i<3;i+=1) { "
            "if(u>v && i==1) continue; sum+=sin(value+i); "
            "if(v>0.75 && i==2) break; } "
            "Cout=color(sum,Dx(sum),Dy(sum)); }",
            consumer, { }, stdosl))
        return false;

    auto check = [&](string_view bytecode, string_view name, bool connected,
                     unsigned depth, bool bitwise, bool sine_loop,
                     string_view shadeop = { }) {
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 }) {
                HartServices renderer;
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
                OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
                OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
                auto group = connected
                                 ? make_connected_group(ss, bytecode, consumer)
                                 : make_group(ss, bytecode);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr,
                          "HART control flow '{}' (OSL {}, LLVM {}): {}\n",
                          name, osl_optimize, optimize, errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                check_module(ss, *group, arch, { shadeop }, optimize, connected,
                             false, sine_loop, connected ? 2 : 0);
                if (optimize == 10)
                    OIIO_CHECK_ASSERT(
                        check_control_flow_ir(ss, *group, depth, bitwise));
            }
    };
    for (size_t i = 0; i < basic_loops.size(); ++i)
        check(basic_loops[i], fmtformat("basic loop {}", i), false, 1, false,
              false);

    const struct {
        const char* name;
        const char* source;
        bool connected;
        unsigned depth;
        bool bitwise;
        const char* shadeop = "";
    } tests[] = {
        // Logical operators short-circuit to OSL if/assign, not and/or opcodes.
        { "logical short circuit and output side effects",
          "int hart_mark(int result, output int calls) { calls+=1; return result; } "
          "shader hart_logical(output color Cout=0) { "
          "int calls=0, a=int(8*u)-4, b=int(8*v)-4; "
          "int both=a && hart_mark(b,calls); "
          "int either=a || hart_mark(b,calls); "
          "int flags=(!a)+2*(a==0)+4*(a!=b)+8*(u<=v); "
          "Cout=color(both+2*either,calls,flags); }",
          false, 0, false },
        { "signed bitwise boundaries and masked shifts",
          "shader hart_bitwise(output color Cout=0) { "
          "int a=(u>0.5)?(-2147483647-1):2147483647; "
          "int b=(v>0.5)?-1:0; int shift=int(32*u)&31; "
          "Cout=color((a&b)^~a,a|b,(a<<shift)^(a>>shift)); }",
          false, 0, true },
        { "signed integer remainder",
          "shader hart_remainder(output color Cout=0) { "
          "int a=(u>v)?-17:17; int divisor=(u>0.5)?-3:3; "
          "Cout=color(a%divisor,a%3,a%-3); }",
          false, 0, false, "osl_safe_mod_iii" },
        { "safe integer remainder with zero divisor",
          "shader hart_zero_remainder(output color Cout=0) { "
          "int a=(u>v)?-17:17; int divisor=(v>0.5)?0:3; "
          "Cout=color(a%divisor,a%0,(a+1)%divisor); }",
          false, 0, false, "osl_safe_mod_iii" },
        { "all shift counts zero through thirty-one",
          "shader hart_shift_range(output color Cout=0) { "
          "int a=(u>v)?(-2147483647-1):-1; float sum=0; "
          "for(int shift=0;shift<32;shift+=1) { "
          "int bits=(a>>shift)^(1<<shift); "
          "sum+=float(bits&255)+sin(u+v+shift); } "
          "Cout=color(sum,Dx(sum),Dy(sum)); }",
          false, 1, false },
        { "nested divergent for and while",
          "shader hart_nested(output color Cout=0) { float sum=0; "
          "for(int outer=0;outer<2+int(2*u);outer+=1) { int inner=0; "
          "while(inner<3+int(v)) { inner+=1; "
          "if(inner==1 && u>v) continue; sum+=sin(u*v+outer+inner); "
          "if(inner>1 && v>u) break; } "
          "if(outer==1 && u<0.5) continue; "
          "if(sum>2 && outer>1) break; } "
          "Cout=color(sum,Dx(sum),Dy(sum)); }",
          false, 2, false },
        { "divergent do-while break and continue",
          "shader hart_do_flow(output color Cout=0) { "
          "int limit=(u>v)?3:0, i=0; float sum=0; do { i+=1; "
          "if(i==1 && v>0.5) continue; sum+=sin(u+v+i); "
          "if(i>1 && u>0.5) break; } while(i<limit); "
          "Cout=color(sum,Dx(sum),Dy(sum)); }",
          false, 1, false },
        { "conditional helper returns",
          "float hidden(float x) { if(x>0) return x; return -x; } "
          "shader hart_helper_return(output color Cout=0) { "
          "Cout=color(hidden(u)); }",
          false, 0, false },
        { "nested helper returns and output parameters",
          "float hart_partial(float x, output float extra) { extra=sin(x); "
          "for(int i=0;i<3+int(2*v);i+=1) { "
          "if(x+i>1.5) return extra; extra+=sin(x+i); } return extra; } "
          "void hart_assign(float x, output float result) { float extra=0; "
          "result=hart_partial(x,extra); if(x>0.5) return; result+=extra; } "
          "shader hart_returns(output color Cout=0) { float left=0,right=0; "
          "float sum=hart_partial(u,left); hart_assign(v,right); "
          "sum+=left+right; Cout=color(sum,Dx(sum),Dy(sum)); }",
          false, 1, false },
        { "shader return preserves prior output",
          "shader hart_shader_return(output color Cout=0) { "
          "float value=sin(u*v); Cout=color(value,Dx(value),Dy(value)); "
          "if(u>v) return; value+=sin(u+v); "
          "Cout=color(value,Dx(value),Dy(value)); }",
          false, 0, false },
        { "shader exit before and within a loop",
          "shader hart_exit(output color Cout=0) { float sum=sin(u+v); "
          "Cout=color(sum,Dx(sum),Dy(sum)); if(u>v) exit(); int i=0; "
          "do { sum+=sin(u*v+i); if(v>0.5) exit(); i+=1; "
          "} while(i<2+int(2*v)); Cout=color(sum,Dx(sum),Dy(sum)); }",
          false, 1, false },
        { "connected producer exit and lazy loop consumer",
          "float hart_flow_value(float x, output float extra) { extra=sin(x); "
          "if(x<0.25) return extra; extra+=sin(2*x); return 0.5*extra; } "
          "shader hart_flow_producer(output float value=0) { float extra=0; "
          "value=hart_flow_value(u*v,extra); if(u>v) exit(); int i=0; "
          "do { i+=1; if(i==2 && v>0.5) continue; "
          "value+=sin(extra+i); if(value>1 && i>1) break; } while(i<3); }",
          true, 1, false },
    };
    for (const auto& test : tests) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl)) {
            print(stderr, "Cannot compile HART control-flow case '{}'\n",
                  test.name);
            return false;
        }
        check(bytecode, test.name, test.connected, test.depth, test.bitwise,
              test.depth > 0, test.shadeop);
    }
    return true;
}



bool
check_aggregate_modules(string_view arch, string_view stdosl,
                        string_view output_array)
{
    check_rejection(arch, output_array, "HARTArrayBounds");
    const struct {
        const char* producer;
        const char* consumer;
        bool closure;
        bool checked;
    } tests[] = {
        { "shader array_local(output color Cout=0) { "
          "float a[3]={u,v,u*v}; float b[3]; b=a; int i=int(2*u); "
          "b[i]=a[i]+v; Cout=color(b[i],Dx(b[i]),Dy(b[i])); }",
          "", false, true },
        { "shader array_output(output float value[3]={0,0,0}) { "
          "value[0]=u; value[1]=v; value[2]=u*v; }",
          "shader array_input(float value[]={1,2,3}, output color Cout=0) { "
          "float q=0; for(int i=0;i<arraylength(value);++i) q+=value[i]; "
          "Cout=color(q,Dx(q),Dy(q)); }",
          false, true },
        { "struct Leaf { float f; color c; }; "
          "struct Packet { Leaf a[2]; float b[3]; }; "
          "shader struct_output(output Packet value={{{0,0},{0,0}},{0,0,0}}) { "
          "value.a[0].f=u; value.a[1].f=v; "
          "value.a[0].c=color(u,v,u*v); value.a[1].c=color(v,u,u+v); "
          "value.b[0]=u; value.b[1]=v; value.b[2]=u*v; }",
          "struct Leaf { float f; color c; }; "
          "struct Packet { Leaf a[2]; float b[3]; }; "
          "shader struct_input(Packet value={{{0,0},{0,0}},{0,0,0}}, "
          "output color Cout=0) { Packet copy=value; Leaf leaves[2]; "
          "leaves=copy.a; int i=int(u>v); Leaf selected=leaves[i]; "
          "float q=selected.f+selected.c[0]+copy.b[int(2*u)]; "
          "Cout=color(q,Dx(q),Dy(q)); }",
          false, true },
        { "struct Holder { closure color lobes[2]; }; "
          "shader closure_output(output Holder value={{0,0}}) { "
          "value.lobes[0]=u*diffuse(N); value.lobes[1]=v*emission(); }",
          "struct Holder { closure color lobes[2]; }; "
          "shader closure_input(Holder value={{0,0}}, output color Cout=0) { "
          "Holder copy=value; closure color c[2]; c=copy.lobes; "
          "Ci=c[int(u>v)]; Cout=color(u,v,1); }",
          true, true },
        { "shader empty_length(float a[]={}, output color Cout=0) { "
          "Cout=color(arraylength(a)); }",
          "", false, false },
        { "shader unchecked_static [[int range_checking=0]] "
          "(output color Cout=0) { float a[2]={u,v}; "
          "Cout=color(a[0]+P[1]); }",
          "", false, false },
    };
    for (const auto& test : tests) {
        OSLCompiler compiler, consumer_compiler;
        std::string producer, consumer;
        const bool connected = test.consumer[0] != '\0';
        if (!compiler.compile_buffer(test.producer, producer, { }, stdosl)
            || (connected
                && !consumer_compiler.compile_buffer(test.consumer, consumer,
                                                     { }, stdosl)))
            return false;
        for (int osl_optimize : { 0, 2 })
            for (int optimize : { 10, 3 }) {
                HartServices renderer(false, false, test.closure, true);
                Diagnostics errors;
                ShadingSystem ss(&renderer, nullptr, &errors);
                if (test.closure)
                    register_hart_closures(ss);
                ss.attribute("hart_arch", arch);
                ss.attribute("optimize", osl_optimize);
                ss.attribute("llvm_optimize", optimize);
                ss.attribute("max_hart_groupdata_alloc",
                             optimize == 3 ? 4096 : 0);
                auto group = connected
                                 ? make_connected_group(ss, producer, consumer)
                                 : make_group(ss, producer);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "{}: {}\n", test.producer, errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                check_module(ss, *group, arch,
                             test.checked && osl_optimize == 0
                                 ? std::initializer_list<
                                       string_view> { "rs_hart_range_error" }
                                 : std::initializer_list<string_view> { },
                             optimize, connected, false, false,
                             connected ? 2 : 0, test.closure, true);
            }
    }
    const struct {
        const char* source;
        const char* error;
    } rejected[] = {
        { "shader unchecked [[int range_checking=0]] (output color Cout=0) { "
          "float a[2]={u,v}; Cout=color(a[int(u)]); }",
          "range_checking" },
        { "shader unchecked [[int range_checking=0]] (output color Cout=0) { "
          "Cout=color(P[int(u)]); }",
          "range_checking" },
        { "shader unchecked [[int range_checking=0]] (output color Cout=0) { "
          "matrix m=matrix(1); m[int(u)][0]=v; Cout=color(m[0][0]); }",
          "range_checking" },
        { "shader interactive_array(float a[2]={0,0} [[int interactive=1]], "
          "output color Cout=0) { Cout=color(a[0]); }",
          "interactive" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
    }
    return true;
}



bool
check_string_storage_ir(ShadingSystem& ss, ShaderGroup& group, bool comparisons,
                        int arraylen, bool mixed,
                        cspan<ustring> constants = { })
{
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
    if (!bytes || !size) {
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    llvm::LLVMContext context;
    auto parsed = llvm::parseBitcodeFile(
        llvm::MemoryBufferRef(llvm::StringRef(static_cast<const char*>(bytes),
                                              size),
                              "hart_string_storage"),
        context);
    if (!parsed) {
        print(stderr, "{}\n", llvm::toString(parsed.takeError()));
        OIIO_CHECK_ASSERT(false);
        return false;
    }
    auto& module       = **parsed;
    const auto& layout = module.getDataLayout();
    int equal = 0, unequal = 0, loads = 0, stores = 0;
    for (const auto& function : module) {
        if (function.getName().find("osl_layer_group_") != 0
            && function.getName().find("osl_init_group_") != 0)
            continue;
        for (const auto& block : function)
            for (const auto& inst : block) {
                if (const auto* cmp = llvm::dyn_cast<llvm::ICmpInst>(&inst)) {
                    if (!cmp->getOperand(0)->getType()->isIntegerTy(64))
                        continue;
                    OIIO_CHECK_ASSERT(
                        cmp->getOperand(1)->getType()->isIntegerTy(64));
                    equal += cmp->getPredicate() == llvm::CmpInst::ICMP_EQ;
                    unequal += cmp->getPredicate() == llvm::CmpInst::ICMP_NE;
                }
                if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&inst))
                    loads += load->getType()->isIntegerTy(64);
                if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&inst))
                    stores += store->getValueOperand()->getType()->isIntegerTy(
                        64);
            }
    }
    if (comparisons) {
        OIIO_CHECK_ASSERT(equal > 0 && unequal > 0);
        OIIO_CHECK_ASSERT(loads > 0 && stores > 0);
    }
    if (arraylen || mixed) {
        auto* storage = llvm::StructType::getTypeByName(context, "Groupdata");
        OIIO_CHECK_ASSERT(storage);
        if (!storage)
            return false;
        const auto* fields = layout.getStructLayout(storage);
        bool found_array = false, scalar = false, pair = false, triple = false;
        bool integers = false, floats = false;
        for (unsigned i = 0; i < storage->getNumElements(); ++i) {
            auto* array = llvm::dyn_cast<llvm::ArrayType>(
                storage->getElementType(i));
            if (!array)
                continue;
            auto* element = array->getElementType();
            OIIO_CHECK_ASSERT(!element->isPointerTy());
            integers |= element->isIntegerTy(32);
            floats |= element->isFloatTy();
            if (!element->isIntegerTy(64))
                continue;
            // Read the emitted field's target layout, not sizeof(ustring).
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(element).getFixedValue(),
                             8);
            OIIO_CHECK_EQUAL(layout.getABITypeAlign(element).value(), 8);
            OIIO_CHECK_EQUAL(fields->getElementOffset(i) % 8, 0);
            OIIO_CHECK_EQUAL(layout.getTypeAllocSize(array).getFixedValue(),
                             8 * array->getNumElements());
            found_array |= array->getNumElements() == uint64_t(arraylen);
            scalar |= array->getNumElements() == 1;
            pair |= array->getNumElements() == 2;
            triple |= array->getNumElements() == 3;
            if (mixed)
                OIIO_CHECK_ASSERT(array->getNumElements() <= 3);
        }
        if (arraylen)
            OIIO_CHECK_ASSERT(found_array);
        if (mixed)
            OIIO_CHECK_ASSERT(scalar && pair && triple && integers && floats);
    }
    bool found_constants = false;
    for (const auto& global : module.globals()) {
        if (global.getName().find("hart_test_group") == llvm::StringRef::npos
            || !global.hasInitializer())
            continue;
        auto* array = llvm::dyn_cast<llvm::ArrayType>(global.getValueType());
        OIIO_CHECK_ASSERT(array);
        if (!array)
            continue;
        // Shader constants are flattened arrays. In particular, hashes must
        // not be disguised as pointer elements, including null pointers.
        OIIO_CHECK_ASSERT(!array->getElementType()->isPointerTy());
        if (constants.empty() || global.use_empty()
            || array->getNumElements() != constants.size()
            || !array->getElementType()->isIntegerTy(64))
            continue;
        bool matches = true;
        for (size_t i = 0; i < constants.size(); ++i) {
            const auto* value = llvm::dyn_cast_or_null<llvm::ConstantInt>(
                global.getInitializer()->getAggregateElement(unsigned(i)));
            matches &= value
                       && value->getZExtValue()
                              == ustringhash(constants[i]).hash();
        }
        if (!matches)
            continue;
        found_constants = true;
        OIIO_CHECK_ASSERT(global.isConstant());
        OIIO_CHECK_EQUAL(layout.getTypeAllocSize(array).getFixedValue(),
                         8 * constants.size());
        OIIO_CHECK_EQUAL(layout.getABITypeAlign(array).value(), 8);
    }
    if (!constants.empty()) {
        if (!found_constants)
            print(stderr, "Missing live i64[{}] string constant array\n",
                  constants.size());
        OIIO_CHECK_ASSERT(found_constants);
    }
    return true;
}



bool
check_string_modules(string_view arch, string_view stdosl)
{
    const struct {
        int osl_optimize;
        int llvm_optimize;
        bool local;
    } variants[] = { { 0, 10, false }, { 2, 10, true }, { 2, 3, true } };
    auto check   = [&](string_view label, ShadingSystem& ss, ShaderGroup& group,
                       const Diagnostics& errors, const auto& variant,
                       bool connected, bool checked) {
        ss.optimize_group(&group, nullptr);
        if (errors.errors)
            print(stderr, "String {} (OSL {}, LLVM {}):\n{}", label,
                  variant.osl_optimize, variant.llvm_optimize, errors.messages);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(ss.getattribute(&group, "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(&group, "hart_groupdata_alloc", allocated));
        if (variant.local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, variant.local ? size : 0);
        check_module(
            ss, group, arch,
            checked && variant.osl_optimize == 0
                ? std::initializer_list<string_view> { "rs_hart_range_error" }
                : std::initializer_list<string_view> { },
            variant.llvm_optimize, connected, false, false, connected ? 2 : 0,
            false, true);
    };
    const struct {
        const char* label;
        const char* producer;
        const char* consumer;
        int arraylen        = 0;
        bool mixed          = false;
        bool constant_array = false;
        bool inspect        = true;
    } tests[] = {
        { "locked defaults, literals, copies and helper return",
          "string copy_text(string text) { return text; } "
          "shader strings(string label=\"alpha\" [[int lockgeom=1]], "
          "string empty=\"\", output color Cout=0) { "
          "string selected=label; if(u>v) selected=\"beta\"; "
          "string copy=copy_text(selected); "
          "Cout=color(copy==label,copy!=empty,"
          "(copy==\"beta\")+2*(copy!=\"alpha\")); }",
          "" },
        { "constant arrays and indexed writes",
          "shader string_arrays(output color Cout=0) { "
          "string palette[3]={\"\",\"hart-string-alpha\",\"hart-string-beta\"}; "
          "string copy[3]; copy=palette; int i=int(2*u); "
          "string selected=palette[i]; "
          "copy[(i+1)%3]=(u>v)?\"hart-string-beta\":\"\"; "
          "Cout=color(selected==copy[i],copy[(i+1)%3]!=\"\",arraylength(copy)); }",
          "", 0, false, true },
        { "connected scalar",
          "shader string_producer(output string value=\"\") { "
          "value=u>v?\"alpha\":\"beta\"; }",
          "shader string_consumer(string value=\"unconnected\", "
          "output color Cout=0) { string copy=value; "
          "Cout=color(copy==\"alpha\",copy!=\"beta\",copy==value); }",
          1 },
        { "connected resolved string array",
          "shader string_array_producer("
          "output string value[4]={\"\",\"\",\"\",\"\"}) { "
          "value[0]=\"alpha\"; value[1]=\"beta\"; "
          "value[2]=u>v?\"alpha\":\"\"; value[3]=u<v?\"\":\"beta\"; }",
          "shader string_array_consumer(string value[]={\"unconnected\"}, "
          "output color Cout=0) { string copy[4]; copy=value; "
          "int i=int(3*u); copy[(i+1)%4]=value[i]; "
          "Cout=color(copy[i]==\"alpha\",copy[(i+1)%4]!=\"\",arraylength(value)); }",
          4 },
        { "mixed nested struct arrays and whole-struct connection",
          // Distinct names avoid the process-wide struct registry's earlier
          // numeric fixtures. Initialize mixed leaf fields in the body, not
          // through the compiler's row/field-confused nested initializers.
          "struct HartStringLeaf { int flag; string label; float weight; }; "
          "struct HartStringPacket { string tag; HartStringLeaf leaves[2]; "
          "string names[3]; color tint; }; "
          "shader string_packet_producer(output HartStringPacket value="
          "{\"\",{},{\"\",\"\",\"\"},0}) { "
          "value.tag=u>v?\"alpha\":\"beta\"; "
          "value.leaves[0].flag=int(u>v); value.leaves[1].flag=int(u<v); "
          "value.leaves[0].label=value.tag; value.leaves[1].label=\"\"; "
          "value.leaves[0].weight=u*v; value.leaves[1].weight=u+v; "
          "value.names[0]=\"alpha\"; value.names[1]=value.tag; "
          "value.names[2]=\"beta\"; "
          "value.tint=color(u,v,u*v); }",
          "struct HartStringLeaf { int flag; string label; float weight; }; "
          "struct HartStringPacket { string tag; HartStringLeaf leaves[2]; "
          "string names[3]; color tint; }; "
          "shader string_packet_consumer(HartStringPacket value="
          "{\"\",{},{\"\",\"\",\"\"},0},output color Cout=0) { "
          "HartStringPacket copy=value; HartStringLeaf leaves[2]; "
          "leaves=copy.leaves; "
          "int i=int(u>v); HartStringLeaf selected=leaves[i]; "
          "string names[3]; names=copy.names; names[(i+1)%3]=selected.label; "
          "float q=selected.weight+copy.tint[0]+copy.tint[1]+copy.tint[2]; "
          "Cout=color((selected.label==copy.tag)+2*(names[i]!=\"\")"
          "+selected.flag,q,Dx(q)+Dy(q)); }",
          2, true },
        { "unused string defaults and helper result",
          "string hidden() { return \"no\"; } "
          "struct HartUnusedStringHolder { string label; float a[2]; }; "
          "shader unused_strings(string unused=\"unused\", "
          "string labels[2]={\"alpha\",\"beta\"}, "
          "HartUnusedStringHolder h={\"bad\",{0,0}}, "
          "output color Cout=0) { string s=hidden(); Cout=color(u,v,1); }",
          "", 0, false, false, false },
    };
    const ustring palette[] = { ustring(""), ustring("hart-string-alpha"),
                                ustring("hart-string-beta") };
    for (const auto& test : tests) {
        OSLCompiler compiler, consumer_compiler;
        std::string producer, consumer;
        const bool connected = test.consumer[0] != '\0';
        if (!compiler.compile_buffer(test.producer, producer, { }, stdosl)
            || (connected
                && !consumer_compiler.compile_buffer(test.consumer, consumer,
                                                     { }, stdosl)))
            return false;
        if (test.constant_array)
            check_rejection(arch, producer, "HARTArrayBounds");
        for (const auto& variant : variants) {
            HartServices renderer(false, false, false, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", variant.osl_optimize);
            ss.attribute("llvm_optimize", variant.llvm_optimize);
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
            auto group = connected
                             ? make_connected_group(ss, producer, consumer)
                             : make_group(ss, producer);
            check(test.label, ss, *group, errors, variant, connected,
                  test.constant_array || test.arraylen > 1);
            if (variant.llvm_optimize == 10 && test.inspect)
                OIIO_CHECK_ASSERT(check_string_storage_ir(
                    ss, *group, variant.osl_optimize == 0, test.arraylen,
                    test.mixed,
                    test.constant_array && variant.osl_optimize == 2
                        ? cspan<ustring>(palette)
                        : cspan<ustring> { }));
        }
    }
    OSLCompiler override_compiler;
    std::string override_bytecode;
    if (!override_compiler.compile_buffer(
            "shader string_overrides(string label=\"default\" [[int lockgeom=1]], "
            "string labels[]={\"default\",\"other\"}, output color Cout=0) { "
            "int n=arraylength(labels); string selected=labels[int(u*(n-1))]; "
            "Cout=color(selected==label,selected!=\"\",n); }",
            override_bytecode, { }, stdosl))
        return false;
    const ustring overrides[] = { ustring("host-alpha"), ustring(""),
                                  ustring("host-beta"), ustring("host-alpha"),
                                  ustring("host-gamma") };
    for (int length : { 3, 5 })
        for (const auto& variant : variants) {
            HartServices renderer(false, false, false, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("optimize", variant.osl_optimize);
            ss.attribute("llvm_optimize", variant.llvm_optimize);
            ss.attribute("max_hart_groupdata_alloc", variant.local ? 4096 : 0);
            OIIO_CHECK_ASSERT(
                ss.LoadMemoryCompiledShader("hart_test", override_bytecode));
            auto group = ss.ShaderGroupBegin("hart_test_group");
            const ustring label(length == 3 ? "" : "host-alpha");
            OIIO_CHECK_ASSERT(
                ss.Parameter("label", TypeString, &label, ParamHints::none));
            OIIO_CHECK_ASSERT(ss.Parameter("labels",
                                           TypeDesc(TypeDesc::STRING, length),
                                           overrides, ParamHints::none));
            OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
            OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
            const SymLocationDesc output("Cout", TypeColor, false,
                                         SymArena::Outputs, 0,
                                         3 * sizeof(float));
            ss.add_symlocs(group.get(), { &output, 1 });
            check("locked host overrides", ss, *group, errors, variant, false,
                  true);
            if (variant.llvm_optimize == 10)
                OIIO_CHECK_ASSERT(check_string_storage_ir(
                    ss, *group, variant.osl_optimize == 0, 0, false,
                    variant.osl_optimize == 2
                        ? cspan<ustring>(overrides, length)
                        : cspan<ustring> { }));
        }
    const struct {
        const char* source;
        const char* error;
    } rejected[] = {
        { "shader bad(string s=\"alpha\" [[int interpolated=1]], "
          "output color Cout=0) { Cout=color(s==\"alpha\"); }",
          "interpolated parameter" },
        { "shader bad(string s=\"alpha\" [[int interactive=1]], "
          "output color Cout=0) { Cout=color(s==\"alpha\"); }",
          "interactive parameter" },
        { "shader bad(string s[2]={\"alpha\",\"beta\"} [[int interpolated=1]], "
          "output color Cout=0) { Cout=color(s[int(u)]==\"alpha\"); }",
          "interpolated parameter" },
        { "shader bad(string s[2]={\"alpha\",\"beta\"} [[int interactive=1]], "
          "output color Cout=0) { Cout=color(s[int(u)]==\"alpha\"); }",
          "interactive parameter" },
        { "shader bad [[int range_checking=0]] (output color Cout=0) { "
          "string s[2]={\"alpha\",\"beta\"}; Cout=color(s[int(u)]==\"alpha\"); }",
          "range_checking" },
        { "shader bad(output color Cout=0) { "
          "Cout=color(strlen(\"alpha\")); }",
          "unsupported operation 'strlen'" },
        { "shader bad(string s=\"alpha\", output color Cout=0) { "
          "Cout=color(strlen(s)); }",
          "unsupported operation 'strlen'" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
    }
    const struct {
        const char* operation;
        unsigned operand;
    } malformed[]
        = { { "eq", 3 }, { "neq", 3 }, { "assign", 1 }, { "assign", 2 } };
    for (const auto& test : malformed) {
        const bool comparison = string_view(test.operation) != "assign";
        const string_view source
            = comparison
                  ? "shader bad(string left=\"alpha\", string right=\"beta\", "
                    "int number=7, output color Cout=0) { "
                    "int result=(left==right)+(left!=right); "
                    "Cout=color(result,number,u); }"
                  : "shader bad(string text=\"alpha\", int number=7, "
                    "output string value=\"\", output color Cout=0) { "
                    "value=text; Cout=color(number,u,v); }";
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        const auto op    = bytecode.find(fmtformat("\t{}\t", test.operation));
        const auto end   = bytecode.find('\n', op);
        const auto hints = bytecode.find('%', op);
        OIIO_CHECK_ASSERT(op != std::string::npos && end != std::string::npos
                          && hints < end);
        if (op == std::string::npos || end == std::string::npos || hints >= end)
            return false;
        std::vector<std::string> words;
        OIIO::Strutil::split(string_view(bytecode).substr(op, hints - op),
                             words, "", -1);
        OIIO_CHECK_EQUAL(words.size(), comparison ? 4 : 3);
        if (words.size() != (comparison ? 4 : 3))
            return false;
        if (!comparison) {
            OIIO_CHECK_EQUAL(words[1], "value");
            OIIO_CHECK_EQUAL(words[2], "text");
            if (words[1] != "value" || words[2] != "text")
                return false;
        }
        words[test.operand] = "number";
        bytecode.replace(op, hints - op,
                         fmtformat("\t{}\t", OIIO::Strutil::join(words, "\t")));
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors,
                             fmtformat("unsupported string operands for '{}'",
                                       test.operation));
    }
    return true;
}



bool
check_spline_selector_modules(string_view arch, string_view stdosl)
{
    const struct {
        int osl, llvm;
    } variants[]          = { { 0, 10 }, { 0, 3 }, { 2, 10 }, { 2, 3 } };
    auto overridden_group = [](ShadingSystem& ss, string_view bytecode,
                               const char* value, ParamHints hints) {
        if (!value)
            return make_group(ss, bytecode);
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_test", bytecode));
        auto group = ss.ShaderGroupBegin("hart_test_group");
        const ustring basis(value);
        OIIO_CHECK_ASSERT(ss.Parameter("basis", TypeString, &basis, hints));
        OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
        OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        const SymLocationDesc output("Cout", TypeColor, false,
                                     SymArena::Outputs, 0, 3 * sizeof(float));
        ss.add_symlocs(group.get(), { &output, 1 });
        return group;
    };
    auto check = [&](string_view bytecode, const char* override_name,
                     ustring expected, int arraylen) {
        for (const auto& variant : variants) {
            HartServices renderer(false, false, false, true, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
            auto group = overridden_group(ss, bytecode, override_name,
                                          ParamHints::none);
            // Exercise resolution after overrides have been copied/released,
            // including OSL0 where parameters retain Groupdata storage.
            if (override_name)
                ss.optimize_group(group.get(), nullptr, false);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "Spline selector {} (OSL {}, LLVM {}): {}\n",
                      expected, variant.osl, variant.llvm, errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch, { "rs_hart_spline_error" },
                         variant.llvm, false, false, false, 0, false, true,
                         arraylen, -1, -1, { }, false, { }, expected);
        }
    };
    const struct {
        const char* initial;
        const char* override_name;
        bool alias;
    } selectors[] = {
        { "linear", nullptr, false },        { "bspline", nullptr, true },
        { "linear", "bezier", false },       { "bezier", "hermite", true },
        { "unknown", "catmull-rom", false }, { "", "constant", true },
    };
    for (const auto& test : selectors) {
        const ustring basis(test.override_name ? test.override_name
                                               : test.initial);
        const int step = basis == ustring("bezier")    ? 3
                         : basis == ustring("hermite") ? 2
                                                       : 1;
        const auto source
            = fmtformat("shader spline_selector(string basis=\"{}\", "
                        "output color Cout=0) {{ "
                        "float k[10]={{0,1,2,3,4,5,6,7,8,9}}; {} "
                        "int n=4+{}*int(u>v); "
                        "float f=spline({},u,k); "
                        "float inv=splineinverse({},1.2+.4*u,n,k); "
                        "Cout=color(f,inv,Dx(f)+Dy(inv)); }}",
                        test.initial,
                        test.alias ? "string selected=basis;" : "", step,
                        test.alias ? "selected" : "basis",
                        test.alias ? "selected" : "basis");
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check(bytecode, test.override_name, basis, 10);
    }
    {
        // spline-boundarybug calls this helper three times, reusing the
        // local's symbol for three identical literal initializations.
        const char* source = "float invspline(float x,float uu) { "
                             "string basis=\"bspline\"; float knots[8]; "
                             "for(int i=0;i<4;++i) knots[i]=x; "
                             "for(int i=4;i<8;++i) knots[i]=1.0; "
                             "return splineinverse(basis,uu,knots); } "
                             "shader spline_local(output color Cout=0) { "
                             "Cout=color(invspline(.6,u),invspline(.6,v),"
                             "invspline(.6,time)); }";
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check(bytecode, nullptr, ustring("bspline"), 8);
    }
    const struct {
        string_view declaration, body, selector;
        const char* override_name = nullptr;
        ParamHints hints          = ParamHints::none;
        int arraylen              = 4;
        string_view error         = "spline basis must be a nonempty immutable";
    } rejected[] = {
        { "string basis=\"unknown\"", "", "basis", nullptr, ParamHints::none, 4,
          "unsupported spline basis" },
        { "string basis=\"\"", "", "basis" },
        { "string basis=\"linear\"", "", "basis", "unknown", ParamHints::none,
          4, "unsupported spline basis" },
        { "string basis=\"linear\"", "", "basis", "" },
        { "string basis=(u>v?\"linear\":\"bezier\")", "", "basis" },
        { "string basis=(u>v?\"linear\":\"bezier\")", "", "basis", "linear" },
        { "string basis=\"linear\" [[int interpolated=1]]", "", "basis" },
        { "string basis=\"linear\" [[int interactive=1]]", "", "basis" },
        { "string basis=\"linear\"", "", "basis", "bezier",
          ParamHints::interpolated },
        { "string basis=\"linear\"", "", "basis", "bezier",
          ParamHints::interactive },
        { "output string basis=\"linear\"", "", "basis" },
        { "string basis=\"linear\"",
          "string selected=u>v?\"linear\":\"bezier\";", "selected" },
        { "string basis=\"linear\"",
          "string selected=basis; if(u>v) selected=\"bezier\";", "selected" },
        { "string basis=\"linear\"", "string selected; if(u>v) selected=basis;",
          "selected" },
        { "string basis=\"linear\"",
          "string names[2]={\"linear\",\"bezier\"}; "
          "string selected=names[int(u>v)];",
          "selected" },
        { "string basis=\"linear\"", "", "basis", "bezier", ParamHints::none, 5,
          "invalid spline knot count for array/basis" },
        { "string basis=\"linear\"", "", "basis", "hermite", ParamHints::none,
          5, "invalid spline knot count for array/basis" },
    };
    for (const auto& test : rejected) {
        const auto source = fmtformat(
            "shader rejected_selector({},output color Cout=0) {{ {} "
            "float k[{}]={{0,1,2,3{}}}; "
            "float f=spline({},u,k); float inv=splineinverse({},u,k); "
            "Cout=color(f+inv); }}",
            test.declaration, test.body, test.arraylen,
            test.arraylen == 5 ? ",4" : "", test.selector, test.selector);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        for (const auto& variant : variants) {
            HartMutableServices renderer(false, false, false, true, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
            auto group = overridden_group(ss, bytecode, test.override_name,
                                          test.hints);
            check_rejected_group(ss, *group, errors, test.error);
        }
    }
    {
        OSLCompiler producer_compiler, consumer_compiler;
        std::string producer, consumer;
        if (!producer_compiler.compile_buffer(
                "shader basis_source(output string value=\"linear\") { "
                "value=u>v?\"linear\":\"bezier\"; }",
                producer, { }, stdosl)
            || !consumer_compiler.compile_buffer(
                "shader basis_sink(string value=\"linear\", "
                "output color Cout=0) { float k[4]={0,1,2,3}; "
                "Cout=color(splineinverse(value,u,k)); }",
                consumer, { }, stdosl))
            return false;
        for (const auto& variant : variants) {
            HartServices renderer(false, false, false, true, true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            OIIO_CHECK_ASSERT(ss.attribute("optimize", variant.osl));
            OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", variant.llvm));
            auto group = make_connected_group(ss, producer, consumer);
            check_rejected_group(ss, *group, errors,
                                 "spline basis must be a nonempty immutable");
        }
    }
    return true;
}



bool
check_spline_modules(string_view arch, string_view stdosl)
{
    if (!check_spline_selector_modules(arch, stdosl))
        return false;
    auto check = [&](string_view label, string_view producer,
                     string_view consumer,
                     std::initializer_list<string_view> shadeops,
                     int osl_optimize, int optimize, int arraylen, bool local) {
        HartServices renderer(false, false, false, true, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", local ? 4096 : 0));
        const bool connected = !consumer.empty();
        auto group = connected ? make_connected_group(ss, producer, consumer)
                               : make_group(ss, producer);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Spline {} (OSL {}, LLVM {}): {}\n", label,
                  osl_optimize, optimize, errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, local ? size : 0);
        check_module(ss, *group, arch, shadeops, optimize, connected, false,
                     false, connected ? 2 : 0, false, true, arraylen);
    };
    const struct {
        const char* name;
        int step;
        const char* type;
    } bases[] = {
        { "catmull-rom", 1, "color" }, { "bezier", 3, "vector" },
        { "bspline", 1, "color" },     { "hermite", 2, "vector" },
        { "linear", 1, "color" },      { "constant", 1, "vector" },
    };
    // Inverse calls with knot derivatives cover the ABI only: the existing
    // inverse implementation deliberately ignores those gradients.
    for (const auto& basis : bases) {
        std::string fixed, moving, fixed_triple, moving_triple;
        for (int i = 0; i < 10; ++i) {
            const char* comma = i ? "," : "";
            fixed += fmtformat("{}{}", comma, i);
            moving += fmtformat("{}{}+0.02*{}*v", comma, i, i + 1);
            fixed_triple += fmtformat("{}{}({},{},{})", comma, basis.type, i,
                                      i + 1, i + 2);
            moving_triple += fmtformat("{}{}({}+0.02*u,{}+0.03*v,{}+0.01*u*v)",
                                       comma, basis.type, i, i + 1, i + 2);
        }
        // Ten knots fit every basis step. Explicit and dynamic counts use
        // fewer knots, retaining the full array length for derivative offsets.
        const auto source = fmtformat(
            "shader spline_family(int count=4, output color Cout=0) {{ "
            "float fixed[10]={{{2}}}, moving[10]={{{3}}}; "
            "{1} fixedt[10]={{{4}}}, movingt[10]={{{5}}}; "
            "int n=count+{6}*int(u>v); "
            "float plain=spline(\"{0}\",u,fixed); "
            "float coord=spline(\"{0}\",u,count,fixed); "
            "float knot=spline(\"{0}\",time,4,moving); "
            "float both=spline(\"{0}\",u,n,moving); "
            "{1} vplain=spline(\"{0}\",u,fixedt); "
            "{1} vcoord=spline(\"{0}\",u,4,fixedt); "
            "{1} vknot=spline(\"{0}\",time,4,movingt); "
            "{1} vboth=spline(\"{0}\",u,n,movingt); "
            "float ip=splineinverse(\"{0}\",1.2+0.4*u,fixed); "
            "float ix=splineinverse(\"{0}\",1.2+0.4*u,4,fixed); "
            "float ik=splineinverse(\"{0}\",time,4,moving); "
            "float ib=splineinverse(\"{0}\",1.2+0.4*u,n,moving); "
            "float f=coord+knot+both, inv=ix+ik+ib; "
            "{1} t=vcoord+vknot+vboth; "
            "Cout=color(plain+ip+f+Dx(f)+Dy(f)+inv+Dx(inv)+Dy(inv))"
            "+color(vplain+t+Dx(t)+Dy(t)); }}",
            basis.name, basis.type, fixed, moving, fixed_triple, moving_triple,
            basis.step);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        for (int optimize : { 10, 3 })
            check(basis.name, bytecode, "",
                  { "osl_spline_fff", "osl_spline_dffdf", "osl_spline_dfdfdf",
                    "osl_spline_vfv", "osl_spline_dvfdv", "osl_spline_dvdfdv",
                    "osl_splineinverse_fff", "osl_splineinverse_dffdf",
                    "osl_splineinverse_dfdfdf", "rs_hart_spline_error" },
                  optimize == 10 ? 0 : 2, optimize, 10, optimize == 3);
        // At OSL0 even constant-initialized knot arrays carry derivatives.
        // Forwarding helpers can inline in HIP, so check the direct dual-knot
        // calls above and the constant-knot overloads after OSL specialization.
        if (string_view(basis.name) == "bezier")
            check("specialized bezier", bytecode, "",
                  { "osl_spline_dfdff", "osl_spline_dvdfv",
                    "osl_splineinverse_dfdff", "rs_hart_spline_error" },
                  2, 10, 10, true);
    }

    const struct {
        int osl_optimize;
        int llvm_optimize;
        bool local;
    } variants[] = {
        { 0, 10, false },
        { 2, 10, true },
        { 2, 3, true },
    };
    for (string_view type : { "float", "vector" }) {
        const auto producer_source
            = fmtformat("shader spline_knots(output {0} value[10]="
                        "{{0,0,0,0,0,0,0,0,0,0}}) {{ "
                        "for(int i=0;i<10;++i) value[i]={0}(i)+{0}(u*v); }}",
                        type);
        const auto consumer_source
            = fmtformat("shader spline_consumer({0} value[]={{0,0,0,0}}, "
                        "output color Cout=0) {{ int n=4+3*int(u>v); "
                        "{0} q=spline(\"bezier\",u,value)"
                        "+spline(\"bezier\",u,n,value); "
                        "Cout=color(q+Dx(q)+Dy(q)); }}",
                        type);
        OSLCompiler producer_compiler, consumer_compiler;
        std::string producer, consumer;
        if (!producer_compiler.compile_buffer(producer_source, producer, { },
                                              stdosl)
            || !consumer_compiler.compile_buffer(consumer_source, consumer, { },
                                                 stdosl))
            return false;
        for (const auto& variant : variants)
            check(fmtformat("connected {}", type), producer, consumer,
                  { type == "float" ? "osl_spline_dfdfdf" : "osl_spline_dvdfdv",
                    "rs_hart_spline_error" },
                  variant.osl_optimize, variant.llvm_optimize, 10,
                  variant.local);
    }

    const char* override_source
        = "shader spline_overrides(float k[]={0,1,2,3}, int count=4, "
          "output color Cout=0) { float f=spline(\"bezier\",u,count,k)"
          "+splineinverse(\"bezier\",1.2+0.4*u,count,k); "
          "Cout=color(f,Dx(f),Dy(f)); }";
    OSLCompiler override_compiler;
    std::string override_bytecode;
    if (!override_compiler.compile_buffer(override_source, override_bytecode,
                                          { }, stdosl))
        return false;
    const struct {
        int arraylen;
        int count;
        int optimize;
        bool local;
    } overrides[] = {
        { 7, 7, 10, true },
        { 7, 7, 3, false },
        { 7, 3, 3, false },
        { 3, 4, 10, false },
    };
    for (const auto& test : overrides) {
        HartServices renderer(false, false, false, true, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", test.optimize == 10 ? 0 : 2);
        ss.attribute("llvm_optimize", test.optimize);
        ss.attribute("max_hart_groupdata_alloc", test.local ? 4096 : 0);
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("hart_test", override_bytecode));
        auto group          = ss.ShaderGroupBegin("hart_test_group");
        const float knots[] = { 0, 1, 2, 3, 4, 5, 6 };
        OIIO_CHECK_ASSERT(
            ss.Parameter("k", TypeDesc(TypeDesc::FLOAT, test.arraylen), knots));
        OIIO_CHECK_ASSERT(ss.Parameter("count", TypeInt, &test.count));
        OIIO_CHECK_ASSERT(ss.Shader("surface", "hart_test", "layer0"));
        OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        const SymLocationDesc output("Cout", TypeColor, false,
                                     SymArena::Outputs, 0, 3 * sizeof(float));
        ss.add_symlocs(group.get(), { &output, 1 });
        if (test.count != 7) {
            // The uniform count override is diagnosed after OSL2 specializes
            // it; at OSL0 a guarded runtime count remains valid codegen.
            check_rejected_group(ss, *group, errors, "spline knot");
            continue;
        }
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Spline overrides: {}\n", errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (test.local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, test.local ? size : 0);
        check_module(ss, *group, arch,
                     { "osl_spline_dfdfdf", "osl_splineinverse_dfdfdf",
                       "rs_hart_spline_error" },
                     test.optimize, false, false, false, 0, false, true, 7);
    }

    // No source-level branches: the conditional around the shadeop must be
    // the spline validation guard, even with ordinary range checking off.
    const char* unchecked_source
        = "shader spline_unchecked [[int range_checking=0]] "
          "(output color Cout=0) { "
          "float k[7]={0,1,2,3,4,5,6}; int n=4+3*int(u>v); "
          "float f=spline(\"bezier\",u,n,k); "
          "Cout=color(f,Dx(f),Dy(f)); }";
    OSLCompiler unchecked_compiler;
    std::string unchecked;
    if (!unchecked_compiler.compile_buffer(unchecked_source, unchecked, { },
                                           stdosl))
        return false;
    for (const auto& variant : variants)
        check("unchecked dynamic count", unchecked, "",
              { variant.osl_optimize == 0 ? "osl_spline_dfdfdf"
                                          : "osl_spline_dfdff",
                "rs_hart_spline_error" },
              variant.osl_optimize, variant.llvm_optimize, 7, variant.local);
    for (bool spline_capability : { false, true }) {
        HartServices renderer(false, false, false, !spline_capability,
                              spline_capability);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, unchecked);
        check_rejected_group(ss, *group, errors,
                             spline_capability ? "HARTArrayBounds"
                                               : "HARTSplineErrors");
    }

    const struct {
        const char* source;
        const char* error;
    } rejected[] = {
        { "shader bad(output color Cout=0) { float k[4]={0,1,2,3}; "
          "Cout=color(spline(\"unknown\",u,k)); }",
          "spline basis" },
        { "shader bad(output color Cout=0) { float k[4]={0,1,2,3}; "
          "string basis=u>v?\"linear\":\"bezier\"; "
          "Cout=color(splineinverse(basis,u,k)); }",
          "spline basis must be a nonempty immutable string" },
        { "shader bad [[int range_checking=0]] (output color Cout=0) { "
          "float k[4]={0,1,2,3}; Cout=color(spline(\"linear\",u,3,k)); }",
          "spline knot" },
        { "shader bad(output color Cout=0) { float k[4]={0,1,2,3}; "
          "Cout=color(splineinverse(\"linear\",u,5,k)); }",
          "spline knot" },
        { "shader bad(output color Cout=0) { float k[7]={0,1,2,3,4,5,6}; "
          "Cout=color(spline(\"bezier\",u,5,k)); }",
          "spline knot" },
        { "shader bad(output color Cout=0) { float k[6]={0,1,2,3,4,5}; "
          "Cout=color(splineinverse(\"hermite\",u,5,k)); }",
          "spline knot" },
        { "shader bad(int count=3, output color Cout=0) { "
          "float k[4]={0,1,2,3}; Cout=color(spline(\"linear\",u,count,k)); }",
          "spline knot" },
        { "shader bad [[int range_checking=0]] (output color Cout=0) { "
          "float k[3]={0,1,2}; Cout=color(spline(\"linear\",u,k)); }",
          "spline knot" },
        { "shader bad(output color Cout=0) { float k[5]={0,1,2,3,4}; "
          "Cout=color(spline(\"bezier\",u,k)); }",
          "spline knot" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, false, false, true, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
    }
    {
        OSLCompiler producer_compiler, consumer_compiler;
        std::string producer, consumer;
        if (!producer_compiler.compile_buffer(
                "shader short_knots(output float value[3]={0,1,2}) { "
                "value[1]=u; }",
                producer, { }, stdosl)
            || !consumer_compiler.compile_buffer(
                "shader short_consumer(float value[]={0,1,2,3}, "
                "output color Cout=0) { "
                "Cout=color(spline(\"linear\",u,value)); }",
                consumer, { }, stdosl))
            return false;
        HartServices renderer(false, false, false, true, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_connected_group(ss, producer, consumer);
        check_rejected_group(ss, *group, errors, "spline knot");
    }
    {
        // Bypass source type checking to exercise malformed bytecode rejection,
        // without declaring a nonexistent public splineinverse overload.
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(
                "shader malformed_spline(output color Cout=0) { "
                "color k[4]={0,1,2,3}; Cout=spline(\"linear\",u,k); }",
                bytecode, { }, stdosl))
            return false;
        const auto op = bytecode.find("\tspline\t");
        OIIO_CHECK_ASSERT(op != std::string::npos);
        if (op == std::string::npos)
            return false;
        bytecode.replace(op + 1, 6, "splineinverse");
        HartServices renderer(false, false, false, true, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, "spline knot");
    }
    for (string_view source : {
             "shader bad(output color Cout=0) { float k=1; "
             "Cout=color(spline(\"linear\",u,k)); }",
             "shader bad(output color Cout=0) { int k[4]={0,1,2,3}; "
             "Cout=color(spline(\"linear\",u,k)); }",
             "shader bad(output color Cout=0) { color k[4]={0,1,2,3}; "
             "Cout=color(splineinverse(\"linear\",u,k)); }",
         }) {
        Diagnostics errors;
        OSLCompiler compiler(&errors);
        std::string bytecode;
        OIIO_CHECK_ASSERT(
            !compiler.compile_buffer(source, bytecode, { }, stdosl));
        OIIO_CHECK_ASSERT(errors.errors > 0);
    }
    return true;
}



bool
check_color_modules(string_view arch, string_view stdosl)
{
    auto check = [&](string_view label, string_view producer,
                     string_view consumer, string_view colorspace,
                     std::initializer_list<string_view> shadeops,
                     int transforms, int osl_optimize, int optimize,
                     bool local = false) {
        HartServices renderer(false, false, false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
        OIIO_CHECK_ASSERT(ss.attribute("colorspace", colorspace));
        // A coordinate-space synonym must not erase a named color constructor.
        OIIO_CHECK_ASSERT(ss.attribute("commonspace", "XYZ"));
        OIIO_CHECK_ASSERT(ss.attribute("optimize", osl_optimize));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", optimize));
        OIIO_CHECK_ASSERT(
            ss.attribute("max_hart_groupdata_alloc", local ? 4096 : 0));
        const bool connected = !consumer.empty();
        auto group = connected ? make_connected_group(ss, producer, consumer)
                               : make_group(ss, producer);
        ss.optimize_group(group.get(), nullptr);
        if (errors.errors)
            print(stderr, "Color {} (OSL {}, LLVM {}, {}): {}\n", label,
                  osl_optimize, optimize, colorspace, errors.last_error);
        OIIO_CHECK_EQUAL(errors.errors, 0);
        int size = 0, allocated = -1;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "llvm_groupdata_size", size));
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "hart_groupdata_alloc", allocated));
        if (local)
            OIIO_CHECK_ASSERT(size > 0 && size <= 4096);
        OIIO_CHECK_EQUAL(allocated, local ? size : 0);
        const int previous_failures = unit_test_failures;
        check_module(ss, *group, arch, shadeops, optimize, connected, false,
                     false, connected ? 2 : 0, false, false, 0, transforms);
        if (unit_test_failures != previous_failures)
            print("Color module checks failed: {} (OSL {}, LLVM {}, {})\n",
                  label, osl_optimize, optimize, colorspace);

        // The GPU tests check numerical rebinding. Here even constant inputs
        // must retain runtime calls and the same published A-B-A artifact.
        const void* original   = nullptr;
        uint64_t original_size = 0;
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                          TypeDesc::PTR, &original));
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                          TypeUInt64, &original_size));
        if (!original || !original_size)
            return;
        const std::string snapshot(static_cast<const char*>(original),
                                   original_size);
        for (string_view binding : {
                 colorspace == "Rec709" ? string_view("ACEScg")
                                        : string_view("Rec709"),
                 colorspace,
             }) {
            OIIO_CHECK_ASSERT(ss.attribute("colorspace", binding));
            const void* bytes = nullptr;
            uint64_t size     = 0;
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR, &bytes));
            OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "hart_bitcode_size",
                                              TypeUInt64, &size));
            OIIO_CHECK_EQUAL(bytes, original);
            OIIO_CHECK_EQUAL(size, original_size);
            if (bytes && size == original_size)
                OIIO_CHECK_ASSERT(
                    snapshot
                    == std::string(static_cast<const char*>(bytes), size));
        }
    };
    const struct {
        const char* label;
        const char* source;
        std::initializer_list<string_view> shadeops;
        int transforms;
        int osl_optimize;
    } fixtures[] = {
        { "values",
          "shader color_values(output color Cout=0) { "
          "color c=color(0.2+0.1*u,0.3+0.1*v,0.4+0.1*u*v); "
          "color a=transformc(\"XYZ\",c); "
          "color b=transformc(\"XYZ\",\"RGB\",a); "
          "Cout=0.001*blackbody(4000+300*u)+wavelength_color(450+100*v)"
          "+b+color(luminance(c)); }",
          { "osl_blackbody_vf", "osl_wavelength_color_vf", "osl_luminance_fv",
            "osl_transformc", "rs_hart_get_colorsystem", "rs_hart_color_error" },
          2,
          0 },
        { "derivatives",
          "shader color_derivatives(output color Cout=0) { "
          "color c=color(0.15+0.3*u,0.25+0.2*v,0.45+0.1*u*v); "
          "color a=transformc(\"rgb\",\"hsv\",c); "
          "color b=transformc(\"hsv\",\"hsl\",a); "
          "color d=transformc(\"hsl\",\"YIQ\",b); "
          "color e=transformc(\"YIQ\",\"XYZ\",d); "
          "color f=transformc(\"XYZ\",\"xyY\",e); "
          "color g=transformc(\"xyY\",\"sRGB\",f); "
          "color h=transformc(\"sRGB\",\"linear\",g); "
          "color t=transformc(\"linear\",\"rgb\",h); "
          "float y=luminance(t); "
          "Cout=t+Dx(t)+Dy(t)+color(y+Dx(y)+Dy(y)); }",
          { "osl_transformc", "osl_luminance_dfdv", "rs_hart_get_colorsystem",
            "rs_hart_color_error" },
          8,
          0 },
        { "constructors",
          "shader color_constructors(output color Cout=0) { "
          "color c=color(\"RGB\",u,v,0.2)+color(\"rgb\",v,u,0.3)"
          "+color(\"hsv\",u,0.4,0.6)+color(\"hsl\",v,0.3,0.7)"
          "+color(\"YIQ\",0.4,u,v)+color(\"XYZ\",u,v,0.5)"
          "+color(\"xyY\",0.3,0.4,0.5+0.1*u); "
          "color s=blackbody(4000+300*u)+wavelength_color(450+100*v); "
          "Cout=c+Dx(c)+Dy(c)+0.001*(s+Dx(s)+Dy(s)); }",
          { "osl_prepend_color_from", "osl_blackbody_vf",
            "osl_wavelength_color_vf", "rs_hart_get_colorsystem",
            "rs_hart_color_error" },
          0,
          0 },
        { "constant inputs",
          "shader color_constants(output color Cout=0) { "
          "color x=transformc(\"XYZ\",\"rgb\",color(0.2,0.3,0.4)); "
          "color y=transformc(\"rgb\",\"XYZ\",color(0.3,0.4,0.5)); "
          "color z=transformc(\"sRGB\",\"rgb\",color(0.4,0.5,0.6)); "
          "Cout=0.001*blackbody(4000)+wavelength_color(520)+x+y+z"
          "+color(luminance(color(0.3,0.4,0.5)))"
          "+color(\"XYZ\",0.2,0.3,0.4); }",
          { "osl_blackbody_vf", "osl_wavelength_color_vf", "osl_luminance_fv",
            "osl_transformc", "osl_prepend_color_from",
            "rs_hart_get_colorsystem", "rs_hart_color_error" },
          3,
          2 },
    };
    for (const auto& fixture : fixtures) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(fixture.source, bytecode, { }, stdosl))
            return false;
        for (int optimize : { 10, 3 })
            check(fixture.label, bytecode, "", "Rec709", fixture.shadeops,
                  fixture.transforms, optimize == 10 ? fixture.osl_optimize : 2,
                  optimize);
    }
    for (string_view space : { "Rec709", "ACEScg" }) {
        const auto source
            = fmtformat("shader color_alias(output color Cout=0) {{ "
                        "Cout=transformc(\"{0}\",\"{0}\",color(0.2,0.3,0.4))"
                        "+transformc(\"{0}\",\"rgb\",color(0.3,0.4,0.5))"
                        "+transformc(\"rgb\",\"{0}\",color(0.4,0.5,0.6))"
                        "+color(\"{0}\",0.2,0.3,0.4); }}",
                        space);
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        for (int optimize : { 10, 3 })
            check("current-name constants", bytecode, "", space,
                  { "osl_transformc", "osl_prepend_color_from",
                    "rs_hart_get_colorsystem", "rs_hart_color_error" },
                  3, 2, optimize);
    }
    {
        OSLCompiler producer_compiler, consumer_compiler;
        std::string producer, consumer;
        if (!producer_compiler.compile_buffer(
                "shader color_producer(output color value=0) { "
                "value=transformc(\"rgb\",\"XYZ\","
                "color(0.2+0.3*u,0.3+0.2*v,0.4+0.1*u*v)); }",
                producer, { }, stdosl)
            || !consumer_compiler.compile_buffer(
                "shader color_consumer(color value=0, output color Cout=0) { "
                "color c=transformc(\"XYZ\",\"rgb\",value); "
                "float y=luminance(c); "
                "Cout=c+Dx(c)+Dy(c)+color(y+Dx(y)+Dy(y)); }",
                consumer, { }, stdosl))
            return false;
        for (const auto mode : { 0, 1, 2 })
            check("connected derivatives", producer, consumer, "Rec709",
                  { "osl_transformc", "osl_luminance_dfdv",
                    "rs_hart_get_colorsystem", "rs_hart_color_error" },
                  2, mode == 0 ? 0 : 2, mode == 2 ? 3 : 10, mode == 1);
    }
    const struct {
        const char* source;
        const char* error;
    } rejected[] = {
        { "shader bad(output color Cout=0) { "
          "Cout=transformc(\"custom\",\"custom\",color(0.2,0.3,0.4)); }",
          "unsupported color space" },
        { "shader bad(output color Cout=0) { "
          "Cout=transformc(\"rgb\",\"ACEScg\",color(0.2,0.3,0.4)); }",
          "unsupported color space" },
        { "shader bad(output color Cout=0) { "
          "Cout=color(\"custom\",0.2,0.3,0.4); }",
          "unsupported color space" },
        { "shader bad(output color Cout=0) { "
          "Cout=color(\"linear\",0.2,0.3,0.4); }",
          "unsupported color space" },
        { "shader bad(output color Cout=0) { "
          "Cout=color(\"sRGB\",0.2,0.3,0.4); }",
          "unsupported color space" },
        { "shader bad(output color Cout=0) { "
          "Cout=color(\"common\",0.2,0.3,0.4); }",
          "unsupported color space" },
        { "shader bad(string space=\"rgb\", output color Cout=0) { "
          "Cout=transformc(space,\"XYZ\",color(u)); }",
          "color spaces must be literal strings" },
        { "shader bad(output color Cout=0) { "
          "string space=u>v?\"rgb\":\"XYZ\"; Cout=color(space,u,v,0.5); }",
          "color spaces must be literal strings" },
    };
    for (const auto& test : rejected) {
        OSLCompiler compiler;
        std::string bytecode;
        if (!compiler.compile_buffer(test.source, bytecode, { }, stdosl))
            return false;
        HartServices renderer(false, false, false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("optimize", 2);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors, test.error);
    }
    for (string_view expression : {
             "blackbody(4000)",
             "wavelength_color(520)",
             "color(luminance(color(0.2,0.3,0.4)))",
             "transformc(\"rgb\",\"XYZ\",color(0.2,0.3,0.4))",
             "color(\"XYZ\",0.2,0.3,0.4)",
         }) {
        OSLCompiler compiler;
        std::string bytecode;
        const auto source = fmtformat(
            "shader missing_colors(output color Cout=0) {{ Cout={}; }}",
            expression);
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        check_rejection(arch, bytecode, "HARTColorSystem");
    }
    for (bool bad_arity : { false, true }) {
        OSLCompiler compiler;
        std::string bytecode;
        const char* source
            = bad_arity ? "shader malformed(output color Cout=0) { "
                          "Cout=transformc(\"rgb\",\"XYZ\",color(u)); }"
                        : "shader malformed(output color Cout=0) { "
                          "Cout=blackbody(4000+u); }";
        if (!compiler.compile_buffer(source, bytecode, { }, stdosl))
            return false;
        const auto op = bytecode.find(bad_arity ? "\ttransformc\t"
                                                : "\tblackbody\t");
        OIIO_CHECK_ASSERT(op != std::string::npos);
        if (op == std::string::npos)
            return false;
        bytecode.replace(op + 1, bad_arity ? 10 : 9,
                         bad_arity ? "blackbody" : "luminance");
        HartServices renderer(false, false, false, false, false, true);
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_group(ss, bytecode);
        check_rejected_group(ss, *group, errors,
                             bad_arity ? "color operation arguments"
                                       : "color operation result type");
    }
    return true;
}



bool
check_procedural_modules(string_view arch, string_view stdosl)
{
    const char* sources[] = {
        "shader hart_octaves(int octaves=3, output float value=0) { "
        "float amplitude=0.5, frequency=1; "
        "for(int i=0;i<octaves;i+=1) { "
        "value+=amplitude*psnoise(point(u*frequency,v*frequency,0.375),"
        "point(frequency,frequency,1)); amplitude*=0.5; frequency*=2; } }",
        "shader hart_ramp(float value=0, output color Cout=0) { "
        "float width=filterwidth(value); "
        "float a=smoothstep(-0.25-width,0.25+width,value); "
        "color c=mix(color(0.1,0.2,0.3),color(0.8,0.6,0.4),a); "
        "Cout=c+Dx(c)+Dy(c); }",
    };
    std::string bytecode[2];
    for (size_t i = 0; i < std::size(sources); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], bytecode[i], { }, stdosl))
            return false;
    }
    for (int optimize : { 10, 3 }) {
        for (int osl_optimize : { 0, 2 }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            ss.attribute("llvm_optimize", optimize);
            ss.attribute("optimize", osl_optimize);
            auto group = make_connected_group(ss, bytecode[0], bytecode[1]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch,
                         { "osl_psnoise_dfdvv", "osl_filterwidth_fdf" },
                         optimize, true);
        }
    }
    return true;
}

}  // namespace



int
main(int argc, char* argv[])
{
    if (argc != 6) {
        print(stderr, "Usage: hart_codegen_test architecture stdosl.h "
                      "color-abi-probe.bc renderer-library-a.bc "
                      "renderer-library-b.bc\n");
        return 1;
    }
    const string_view arch(argv[1]);
    check_userdata_host_layout();
    if (!check_color_layout(arch, argv[3]) || unit_test_failures)
        return 1;
    if (!check_named_transform_modules(arch, argv[2]) || unit_test_failures)
        return 1;
    if (!check_closure_modules(arch, argv[2])
        || !check_material_closure_modules(arch, argv[2])
        || !check_closure_array_modules(arch, argv[2]) || unit_test_failures)
        return 1;
    const char* sources[] = {
        "shader hart_test(output color Cout=0) { Cout=color(u,v,u+v); }",
        "shader hart_test(output color Cout=0) { Cout=color(u,v,sin(u+v)); }",
        "shader hart_test(output color Cout=0) { printf(\"not supported\"); Cout=color(u,v,0); }",
        "shader hart_test(output color Cout=0) { Cout=texture(\"missing.tx\",u,v); }",
        "shader hart_test(output color Cout=0) { Cout=color(P); }",
        "shader hart_producer(output float value=0) { value=u+v; }",
        "shader hart_consumer(float value=42, output color Cout=0) { Cout=color(u,v,sin(value)); }",
        "shader hart_bad_producer(output float value=0) { printf(\"not supported\"); value=u+v; }",
        "shader hart_bad_consumer(float value=42, output color Cout=0) { printf(\"not supported\"); Cout=color(u,v,sin(value)); }",
        "shader hart_userdata_producer(float scale=1 [[ int interpolated=1 ]], output float value=0) { value=scale*(u+v); }",
        "shader hart_userdata_consumer(float value=42 [[ int interpolated=1 ]], output color Cout=0) { Cout=color(u,v,sin(value)); }",
        "shader hart_array(output float values[2]={1,2}, output color Cout=0) { Cout=color(u,v,0); }",
        "shader hart_branch(float value=42, output color Cout=0) { if (u>v) Cout=color(u,v,sin(value)); else Cout=color(u,v,0); }",
        "shader hart_loop(float value=42, output color Cout=0) { int count=1; if(u>v) count=4; else if(u<v) count=0; float sum=0; int i=0; while(i<count) { sum+=sin(value+i); i+=1; } Cout=color(u,v,sum); }",
        "shader hart_branch_printf(output color Cout=0) { if (u>v) printf(\"not supported\"); Cout=color(u,v,0); }",
        "shader hart_branch_texture(output color Cout=0) { if (u>v) Cout=texture(\"missing.tx\",u,v); else Cout=0; }",
        "shader hart_compare(output color Cout=0) { int a=(u>0.5)-1; int b=(v>0.5)-1; Cout=color((u<v)+2*(u<=v)+4*(a<b)+8*(a<=b), (u>v)+2*(u>=v)+4*(a>b)+8*(a>=b), (u==v)+2*(u!=v)+4*(a==b)+8*(a!=b)); }",
        "shader hart_for(float value=42, output color Cout=0) { int count=1; if(u>v) count=4; else if(u<v) count=0; float sum=0; for(int i=0;i<count;i+=1) sum+=sin(value+i); Cout=color(u,v,sum); }",
        "shader hart_break(output color Cout=0) { for(int i=0;i<4;i+=1) { if(u>v) break; Cout+=color(u,v,0); } }",
        "shader hart_continue(output color Cout=0) { for(int i=0;i<4;i+=1) { if(u>v) continue; Cout+=color(u,v,0); } }",
        "shader hart_dowhile(output color Cout=0) { int i=0; do { Cout+=color(u,v,0); i+=1; } while(i<2); }",
        "shader hart_loop_printf(int count=0, output color Cout=0) { for(int i=0;i<count;i+=1) printf(\"not supported\"); Cout=color(u,v,0); }",
        "shader hart_loop_texture(int count=0, output color Cout=0) { for(int i=0;i<count;i+=1) Cout=texture(\"missing.tx\",u,v); }",
        "shader hart_deriv(output color Cout=0) { float value=sin(u*v); Cout=color(value,Dx(value),Dy(value)); }",
        "shader hart_deriv_producer(float scale=1, output float value=0) { value=sin(scale*u*v); }",
        "shader hart_deriv_consumer(float value=42, output color Cout=0) { Cout=color(value,Dx(value),Dy(value)); }",
        "shader hart_deriv_flow(output float value=0) { int count=1; if(u>v) count=3; else if(u<v) count=0; for(int i=0;i<count;i+=1) value+=sin(u*v+i); }",
        "shader hart_deriv_color(output color Cout=0) { color value=sin(color(u*v,u+v,u-v)); color dx=Dx(value); color dy=Dy(value); Cout=color(dx[0],dy[1],dx[2]+dy[2]); }",
        "shader hart_deriv_dz(output color Cout=0) { Cout=color(Dz(u)); }",
        "shader hart_deriv_filterwidth(output color Cout=0) { Cout=color(filterwidth(u)); }",
        "shader hart_deriv_arithmetic(output color Cout=0) { float value=-(u*v+u-v)/(v+1); Cout=color(value,Dx(value),Dy(value)); }",
        "shader hart_surface_globals(output color Cout=0) { "
        "Cout=color(P+Dx(P)+Dy(P)+N+Ng+dPdu+dPdv+Dx(N)+Dy(Ng)+Dx(dPdu)+Dy(dPdv)); }",
        "shader hart_surface_values(int planar=0, output color Cout=0) { "
        "vector q=vector(P); if(planar) q[2]=0; vector n=normalize(q); "
        "Cout=color(dot(q,N),length(q),n[0]); }",
        "shader hart_surface_producer(int planar=0, output vector value=0) { "
        "float z=P[2]+P[0]*P[1]; if(planar) z=0; "
        "point p=point(P[0],2*P[1],z); normal n=normal(p[0],p[1],p[2]); "
        "value=vector(n[0],n[1],n[2]); }",
        "shader hart_surface_consumer(vector value=0, int operation=-1, output color Cout=0) { "
        "int selected=operation; if(selected<0) selected=(u>0.5)+2*(v>0.5); float result=0; "
        "if(selected==0) { vector n=normalize(value); result=n[0]+2*n[1]+3*n[2]; } "
        "else if(selected==1) result=length(value); "
        "else if(selected==2) result=dot(value,value); "
        "else if(selected==3) result=dot(value,N); else result=dot(Ng,value); "
        "Cout=color(result,Dx(result),Dy(result)); }",
        "shader hart_surface_unsupported(output color Cout=0) { Cout=color(Ps); }",
        "shader hart_surface_write_normal(int enable=0, output color Cout=0) { "
        "if(enable) N=normal(u,v,1); Cout=color(u,v,0); }",
        "shader hart_surface_write_position(output color Cout=0) { P[0]=u; Cout=color(P); }",
        "shader hart_surface_transform(output color Cout=0) { Cout=color(transform(\"object\",P)); }",
        "shader hart_filterwidth_scalar(output color Cout=0) { "
        "float w=filterwidth(sin(u*v)); Cout=color(w,Dx(w),Dy(w)); }",
        "shader hart_filterwidth_consumer(float value=42, output color Cout=0) { "
        "float w=filterwidth(value); Cout=color(w,Dx(w),Dy(w)); }",
        "shader hart_filterwidth_triple(int kind=0, output color Cout=0) { "
        "vector q=vector(P[0]+P[1],P[0]-P[1],P[0]*P[1]); "
        "if(kind==0) Cout=color(filterwidth(q)); "
        "else if(kind==1) Cout=filterwidth(color(q)); "
        "else if(kind==2) Cout=color(filterwidth(point(q))); "
        "else if(kind==3) Cout=color(filterwidth(normal(q))); "
        "else if(kind==4) Cout=color(filterwidth(N)); else Cout=color(filterwidth(P)); }",
        "shader hart_filterwidth_vector_consumer(vector value=0, int derivative=0, output color Cout=0) { "
        "vector w=filterwidth(value); if(derivative==1) Cout=color(Dx(w)); "
        "else if(derivative==2) Cout=color(Dy(w)); else Cout=color(w); }",
        "shader hart_filterwidth_noise(output color Cout=0) { float value=noise(P); Cout=color(filterwidth(value)); }",
        "shader hart_filterwidth_unsupported(output color Cout=0) { Cout=color(filterwidth(Ps)); }",
    };
    std::vector<std::string> oso(std::size(sources));
    for (size_t i = 0; i < oso.size(); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, argv[2]))
            return 1;
    }
    // OSL level 10 skips passes; even O0 inlines alwaysinline HIP shadeops.
    for (int optimize : { 10, 3 }) {
        for (int i : { 0,  1,  4,  12, 13, 16, 17, 23, 25, 27,
                       29, 30, 31, 32, 34, 39, 40, 41, 42, 43 }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            ss.attribute("llvm_optimize", optimize);
            auto group = make_group(ss, oso[i]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            const bool looping = i == 13 || i == 17;
            const string_view sine_function = i == 23   ? "osl_sin_dfdf"
                                              : i == 27 ? "osl_sin_dvdv"
                                              : i == 1 || looping ? "osl_sin_ff"
                                                                  : "";
            if (i == 32)
                check_module(ss, *group, arch,
                             { "osl_dot_fvv", "osl_length_fv",
                               "osl_normalize_vv" },
                             optimize);
            else if (i == 29)
                check_module(ss, *group, arch, { "osl_filterwidth_fdf" },
                             optimize);
            else if (i == 39)
                check_module(ss, *group, arch,
                             { "osl_sin_dfdf", "osl_filterwidth_fdf" },
                             optimize);
            else if (i == 41)
                check_module(ss, *group, arch, { "osl_filterwidth_vdv" },
                             optimize);
            else if (i == 43)
                check_module(ss, *group, arch,
                             { "osl_noise_dfdv", "osl_filterwidth_fdf" },
                             optimize);
            else
                check_module(ss, *group, arch, { sine_function }, optimize,
                             false, i == 12, looping);
            auto* thread = ss.create_thread_info();
            auto* ctx    = ss.get_context(thread);
            ShaderGlobals globals { };
            OIIO_CHECK_ASSERT(!ss.execute(*ctx, *group, globals));
            ss.release_context(ctx);
            ss.destroy_thread_info(thread);
            OIIO_CHECK_EQUAL(errors.errors, 1);
            OIIO_CHECK_ASSERT(
                OIIO::Strutil::contains(errors.last_error, "not CPU execute"));
        }
        for (int consumer : { 6, 12, 13, 17 }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            ss.attribute("llvm_optimize", optimize);
            auto group = make_connected_group(ss, oso[5], oso[consumer]);
            // Instance parameter overrides are released during optimization.
            ss.optimize_group(group.get(), nullptr, false);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch, { "osl_sin_ff" }, optimize, true,
                         consumer == 12, consumer == 13 || consumer == 17);
        }
        for (int producer : { 24, 26 }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            ss.attribute("llvm_optimize", optimize);
            auto group = make_connected_group(ss, oso[producer], oso[25]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch, { "osl_sin_dfdf" }, optimize, true);
        }
        {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            ss.attribute("llvm_optimize", optimize);
            auto group = make_connected_group(ss, oso[33], oso[34]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch,
                         { "osl_normalize_dvdv", "osl_length_dfdv",
                           "osl_dot_dfdvdv", "osl_dot_dfdvv", "osl_dot_dfvdv" },
                         optimize, true);
        }
        for (bool scalar : { true, false }) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
            ss.attribute("llvm_optimize", optimize);
            auto group = make_connected_group(ss, oso[scalar ? 24 : 33],
                                              oso[scalar ? 40 : 42]);
            ss.optimize_group(group.get(), nullptr);
            if (errors.errors)
                print(stderr, "{}\n", errors.last_error);
            OIIO_CHECK_EQUAL(errors.errors, 0);
            check_module(ss, *group, arch,
                         { scalar ? "osl_filterwidth_fdf"
                                  : "osl_filterwidth_vdv" },
                         optimize, true);
        }
    }
    check_groupdata_alloc_settings();
    check_groupdata_alloc_modules(arch, oso[24], oso[25]);
    check_rejection("gfx9999", oso[0], "No embedded HART shadeops");
    check_rejection(arch, oso[2], "renderer lacks HARTDiagnostics", 3);
    check_rejection(arch, oso[0], "instrumentation", 1, true);
    check_rejection(arch, oso[2], "renderer lacks HARTDiagnostics");
    check_rejection(arch, oso[3], "unsupported operation 'texture'");
    check_rejection(arch, oso[14], "renderer lacks HARTDiagnostics");
    check_rejection(arch, oso[15], "unsupported operation 'texture'");
    check_rejection(arch, oso[21], "renderer lacks HARTDiagnostics");
    check_rejection(arch, oso[22], "unsupported operation 'texture'");
    check_rejection(arch, oso[28], "unsupported operation 'Dz'");
    check_rejection(arch, oso[44], "unsupported shader global 'Ps'");
    check_rejection(arch, oso[35], "unsupported shader global 'Ps'");
    check_rejection(arch, oso[36], "writing shader global 'N'");
    check_rejection(arch, oso[37], "writing shader global 'P'");
    {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        ss.attribute("profile", 1);
        auto group = make_group(ss, oso[43]);
        check_rejected_group(ss, *group, errors, "instrumentation");
    }
    // The two-argument transform is a stdosl wrapper.
    check_rejection(arch, oso[38], "renderer lacks HARTTransforms");
    for (string_view type : { "point", "vector", "normal" }) {
        for (string_view space : { "object", "common" }) {
            OSLCompiler compiler;
            std::string bytecode;
            const auto source
                = fmtformat("shader hart_space(output color Cout=0) {{ "
                            "Cout=color({}(\"{}\",u,v,1)); }}",
                            type, space);
            if (!compiler.compile_buffer(source, bytecode, { }, argv[2]))
                return 1;
            check_rejection(arch, bytecode, "renderer lacks HARTTransforms");
        }
    }
    for (bool userdata : { false, true }) {
        for (int unsupported_layer = 0; unsupported_layer < 2;
             ++unsupported_layer) {
            HartServices renderer;
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            auto group = make_connected_group(
                ss, oso[unsupported_layer == 0 ? (userdata ? 9 : 7) : 5],
                oso[unsupported_layer == 1 ? (userdata ? 10 : 8) : 6]);
            check_rejected_group(ss, *group, errors,
                                 userdata ? "interpolated parameter"
                                          : "renderer lacks HARTDiagnostics");
        }
    }
    if (!check_aggregate_modules(arch, argv[2], oso[11])
        || !check_string_modules(arch, argv[2])
        || !check_output_placement_modules(arch, argv[2], oso[0], oso[5], oso[6])
        || !check_explicit_entry_modules(arch, argv[2], oso[0], oso[5])
        || !check_interactive_modules(arch, argv[2])
        || !check_userdata_modules(arch, argv[2])
        || !check_interpolated_interactive_modules(arch, argv[2])
        || !check_attribute_modules(arch, argv[2])
        || !check_renderer_library_modules(arch, argv[2], argv[4], argv[5],
                                           oso[0])
        || !check_diagnostic_modules(arch, argv[2])
        || !check_control_flow_modules(arch, argv[2], { oso.data() + 18, 3 })
        || !check_chain_modules(arch, argv[2])
        || !check_topology_modules(arch, argv[2])
        || !check_material_modules(arch, argv[2])
        || !check_math_modules(arch, argv[2])
        || !check_isconstant_modules(arch, argv[2])
        || !check_numeric_math_modules(arch, argv[2])
        || !check_spline_modules(arch, argv[2])
        || !check_color_modules(arch, argv[2])
        || !check_noise_modules(arch, argv[2])
        || !check_hash_modules(arch, argv[2])
        || !check_dynamic_noise_modules(arch, argv[2])
        || !check_gabor_modules(arch, argv[2])
        || !check_procedural_modules(arch, argv[2])
        || !check_matrix_modules(arch, argv[2])
        || !check_space_modules(arch, argv[2])
        || !check_geometry_modules(arch, argv[2])
        || !check_geometry_state_modules(arch, argv[2])
        || !check_texture_modules(arch, argv[2]))
        return 1;
    return unit_test_failures;
}
