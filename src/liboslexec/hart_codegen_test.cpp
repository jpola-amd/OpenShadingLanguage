// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// GPU-independent compiler regression tests for the HART backend. Compile small
// OSL shaders and inspect their AMDGPU bitcode before and after optimization,
// checking target metadata, split/fused callable ABI, address spaces, group-data
// alignment, linked shadeops, control flow, HART provenance, and rejection of
// unsupported operations.
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
#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>

#include "hart_bitcode.h"
#include "opcolor.h"

#include <OpenImageIO/unittest.h>

#include <algorithm>
#include <initializer_list>

#include <llvm/ADT/APInt.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR >= 17
#    include <llvm/TargetParser/Triple.h>
#else
#    include <llvm/ADT/Triple.h>
#endif

using namespace OSL;

namespace {

class HartServices final : public RendererServices {
public:
    explicit HartServices(bool textures = false, bool transforms = false,
                          bool closures = false, bool arrays = false,
                          bool splines = false, bool colors = false,
                          bool noise = false)
        : m_textures(textures)
        , m_transforms(transforms)
        , m_closures(closures)
        , m_arrays(arrays)
        , m_splines(splines)
        , m_colors(colors)
        , m_noise(noise)
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
               || (m_noise && feature == "HARTNoiseErrors");
    }
    TextureHandle* get_texture_handle(ustring filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        ++texture_requests;
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

private:
    bool m_textures;
    bool m_transforms;
    bool m_closures;
    bool m_arrays;
    bool m_splines;
    bool m_colors;
    bool m_noise;
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



class Diagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR) {
            ++errors;
            last_error = message;
        }
    }
    int errors = 0;
    std::string last_error;
};



ShaderGroupRef
make_group(ShadingSystem& ss, string_view oso, int layers = 1)
{
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
              std::initializer_list<const llvm::Function*> targets,
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
check_noise_abi(llvm::Module& module, int optimize)
{
    const struct {
        const char* name;
        const char* args;
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
    std::vector<const llvm::CallBase*> used_resets;
    int gabor_calls = 0, initializers = 0;
    for (const auto& signature : signatures) {
        auto* function = module.getFunction(signature.name);
        if (!function || function->use_empty())
            continue;
        const string_view name(signature.name), args(signature.args);
        const bool callback = name == "rs_hart_noise_error";
        const bool gabor    = OIIO::Strutil::starts_with(name, "osl_gabor");
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
                if (!gabor)
                    continue;
                ++gabor_calls;
                OIIO_CHECK_EQUAL(call->arg_size(), args.size());
                if (call->arg_size() != args.size())
                    continue;
                const auto* selector = llvm::dyn_cast<llvm::ConstantInt>(
                    call->getArgOperand(0));
                OIIO_CHECK_ASSERT(selector);
                if (selector)
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
                }
                // Value-only results and constant/partial coordinates still
                // need full dual temporaries for Gabor's filtering ABI.
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
        OIIO_CHECK_ASSERT(function->getReturnType()->isVoidTy());
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
    if (!gabor_calls)
        return;
    OIIO_CHECK_EQUAL(initializers, gabor_calls);
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



void
check_module(ShadingSystem& ss, ShaderGroup& group, string_view arch,
             std::initializer_list<string_view> shadeops, int optimize,
             bool connected = false, bool branching = false,
             bool looping = false, int used_layers = 0, bool closures = false,
             bool aggregates = false, int spline_arraylen = 0,
             int color_transforms = -1)
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
    check_noise_abi(module, optimize);
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
                                OIIO_CHECK_ASSERT(id->getZExtValue() == 1
                                                  || id->getZExtValue() == 3);
                                OIIO_CHECK_EQUAL(size->getZExtValue(),
                                                 id->getZExtValue() == 1 ? 1
                                                                         : 24);
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
        int spline_calls = 0;
        for (auto& function : module) {
            if (function.getName().find("osl_layer_group_") != 0)
                continue;
            llvm::DominatorTree dominators(function);
            for (const auto& block : function)
                for (const auto& inst : block) {
                    const auto* call   = llvm::dyn_cast<llvm::CallBase>(&inst);
                    const auto* callee = call ? call->getCalledFunction()
                                              : nullptr;
                    if (!callee
                        || (callee->getName().find("osl_spline_") != 0
                            && callee->getName().find("osl_splineinverse_")
                                   != 0))
                        continue;
                    ++spline_calls;
                    OIIO_CHECK_EQUAL(call->arg_size(), 6);
                    if (call->arg_size() != 6)
                        continue;
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
                OIIO_CHECK_ASSERT(guarded);
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
                                             "unsupported operation 'printf'");
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
check_math_modules(string_view arch, string_view stdosl)
{
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
          "unsupported operation 'printf'" },
        { "string hidden() { return \"no\"; } "
          "shader bad(output color Cout=0) { string s=hidden(); Cout=color(u); }",
          "unsupported type 'string'" },
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
        check_rejection(arch, bytecode, "unsupported type 'string'");
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
          "unsupported type" },
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
                                  true);
            Diagnostics errors;
            ShadingSystem ss(&renderer, nullptr, &errors);
            ss.attribute("hart_arch", arch);
            auto group = make_group(ss, bytecode);
            check_rejected_group(ss, *group, errors, "unsupported type");
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
    OSLCompiler compiler;
    std::string bytecode;
    if (!compiler.compile_buffer(
            "shader hart_dynamic_texture(string filename=\"x.exr\",output color Cout=0) { "
            "Cout=texture(filename,u,v,\"interp\",\"linear\",\"wrap\",\"clamp\"); }",
            bytecode, { }, stdosl))
        return false;
    check_rejection(arch, bytecode, "unsupported type 'string'", 1, false,
                    true);
    return check_texture_alpha_modules(arch, stdosl)
           && check_texture_firstchannel_modules(arch, stdosl);
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
        { "shader hart_closure_string(string name=\"diffuse\", "
          "output color Cout=0) { Ci=diffuse(N); }",
          "unsupported type" },
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
        { "struct Holder { string label; float a[2]; }; "
          "shader unused_string(Holder h={\"bad\",{0,0}}, output color Cout=0) "
          "{ Cout=color(u,v,1); }",
          "unsupported type 'string'" },
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
check_spline_modules(string_view arch, string_view stdosl)
{
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
        { "shader bad(string basis=\"linear\", output color Cout=0) { "
          "float k[4]={0,1,2,3}; Cout=color(spline(basis,u,k)); }",
          "unsupported type" },
        { "shader bad(output color Cout=0) { float k[4]={0,1,2,3}; "
          "string basis=u>v?\"linear\":\"bezier\"; "
          "Cout=color(splineinverse(basis,u,k)); }",
          "unsupported type" },
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
          "unsupported type" },
        { "shader bad(output color Cout=0) { "
          "string space=u>v?\"rgb\":\"XYZ\"; Cout=color(space,u,v,0.5); }",
          "unsupported type" },
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
    if (argc != 4) {
        print(stderr, "Usage: hart_codegen_test architecture stdosl.h "
                      "color-abi-probe.bc\n");
        return 1;
    }
    const string_view arch(argv[1]);
    if (!check_color_layout(arch, argv[3]) || unit_test_failures)
        return 1;
    if (!check_closure_modules(arch, argv[2]) || unit_test_failures)
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
    check_rejection(arch, oso[2], "unsupported operation 'printf'", 3);
    check_rejection(arch, oso[0], "instrumentation", 1, true);
    check_rejection(arch, oso[2], "unsupported operation 'printf'");
    check_rejection(arch, oso[3], "unsupported operation 'texture'");
    check_rejection(arch, oso[14], "unsupported operation 'printf'");
    check_rejection(arch, oso[15], "unsupported operation 'texture'");
    check_rejection(arch, oso[21], "unsupported operation 'printf'");
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
                                 userdata ? "interpolated or interactive"
                                          : "unsupported operation 'printf'");
        }
    }
    {
        HartServices renderer;
        Diagnostics errors;
        ShadingSystem ss(&renderer, nullptr, &errors);
        ss.attribute("hart_arch", arch);
        auto group = make_connected_group(ss, oso[5], oso[6]);
        const ustring entry("producer");
        OIIO_CHECK_ASSERT(ss.attribute(group.get(), "entry_layers",
                                       TypeDesc(TypeDesc::STRING, 1), &entry));
        check_rejected_group(ss, *group, errors, "default entry point");
    }
    if (!check_aggregate_modules(arch, argv[2], oso[11])
        || !check_control_flow_modules(arch, argv[2], { oso.data() + 18, 3 })
        || !check_chain_modules(arch, argv[2])
        || !check_topology_modules(arch, argv[2])
        || !check_material_modules(arch, argv[2])
        || !check_math_modules(arch, argv[2])
        || !check_numeric_math_modules(arch, argv[2])
        || !check_spline_modules(arch, argv[2])
        || !check_color_modules(arch, argv[2])
        || !check_noise_modules(arch, argv[2])
        || !check_gabor_modules(arch, argv[2])
        || !check_procedural_modules(arch, argv[2])
        || !check_matrix_modules(arch, argv[2])
        || !check_space_modules(arch, argv[2])
        || !check_geometry_modules(arch, argv[2])
        || !check_texture_modules(arch, argv[2]))
        return 1;
    return unit_test_failures;
}
