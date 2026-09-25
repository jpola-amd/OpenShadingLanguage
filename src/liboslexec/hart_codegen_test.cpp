// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// GPU-independent compiler regression tests for the HART backend. Compile small
// OSL shaders and inspect their AMDGPU bitcode before and after optimization,
// checking target metadata, callable ABI, address spaces, group-data alignment,
// linked shadeops, HART provenance, and rejection of unsupported operations.
// This allows testing every configured architecture without its physical GPU.
//
// Built as a separate test executable, not part of the runtime library, only
// when OSL_BUILD_TESTS, BUILD_TESTING, OSL_USE_HART, and USE_LLVM_BITCODE are on.
// It lives beside the other liboslexec unit tests; actual GPU execution and
// image comparisons are covered separately by check-generated.py.
//
// Keep and extend this coverage as HART and LLVM evolve. Replace individual
// rejection cases with positive tests when their features become supported.

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>

#include <OpenImageIO/unittest.h>

#include <initializer_list>

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
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
    explicit HartServices(bool textures = false, bool transforms = false)
        : m_textures(textures), m_transforms(transforms)
    {
    }
    int supports(string_view feature) const override
    {
        return feature == "HART" || (m_textures && feature == "HARTTextures")
               || (m_transforms && feature == "HARTTransforms");
    }
    TextureHandle* get_texture_handle(ustring filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        ++texture_requests;
        return m_textures && filename == "hart-test-texture.exr"
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
};



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
    ss.optimize_group(&group, nullptr);
    const void* bytes = nullptr;
    uint64_t size     = 0;
    OIIO_CHECK_ASSERT(
        !ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &bytes));
    OIIO_CHECK_ASSERT(
        !ss.getattribute(&group, "hart_bitcode_size", TypeUInt64, &size));
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
                     string_view consumer)
{
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_producer", producer));
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("hart_consumer", consumer));
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
check_module(ShadingSystem& ss, ShaderGroup& group, string_view arch,
             std::initializer_list<string_view> shadeops, int optimize,
             bool connected = false, bool branching = false,
             bool looping = false, int used_layers = 0)
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
    bool provenance = false;
    for (const auto& global : module.globals())
        provenance |= global.getName().contains("__hart_device_storage_abi");
    OIIO_CHECK_ASSERT(provenance);
    for (const auto* query : { "group_init_name", "group_entry_name" }) {
        ustring name;
        OIIO_CHECK_ASSERT(ss.getattribute(&group, query, name));
        const auto* function = module.getFunction(name.c_str());
        OIIO_CHECK_ASSERT(function && !function->isDeclaration());
        if (!function)
            continue;
        OIIO_CHECK_ASSERT(function->hasExternalLinkage());
        OIIO_CHECK_ASSERT(function->getReturnType()->isVoidTy());
        OIIO_CHECK_ASSERT(!function->isVarArg());
        OIIO_CHECK_EQUAL(function->arg_size(), 6);
        OIIO_CHECK_EQUAL(
            function->getFnAttribute("target-cpu").getValueAsString().str(),
            std::string(arch));
        for (const auto& arg : function->args()) {
            if (arg.getArgNo() == 4)
                OIIO_CHECK_ASSERT(arg.getType()->isIntegerTy(32));
            else
                OIIO_CHECK_ASSERT(arg.getType()->isPointerTy()
                                  && arg.getType()->getPointerAddressSpace()
                                         == 0);
        }
    }
    int callables = 0;
    for (const auto& function : module) {
        if (function.getName().find("__direct_callable__") == 0)
            ++callables;
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
            }
    }
    OIIO_CHECK_EQUAL(callables, 2);
    if (looping && optimize == 10) {
        ustring entry_name;
        OIIO_CHECK_ASSERT(
            ss.getattribute(&group, "group_entry_name", entry_name));
        auto* entry = module.getFunction(entry_name.c_str());
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
        ustring entry_name;
        OIIO_CHECK_ASSERT(
            ss.getattribute(&group, "group_entry_name", entry_name));
        auto* entry = module.getFunction(entry_name.c_str());
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
            ustring entry_name;
            OIIO_CHECK_ASSERT(
                ss.getattribute(&group, "group_entry_name", entry_name));
            const auto* entry   = module.getFunction(entry_name.c_str());
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
        if (!used)
            print(stderr, "Expected a used {} '{}'\n",
                  callback ? "renderer callback declaration" : "linked shadeop",
                  name);
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
    int size_bytes = 0, alignment = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_size", size_bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_alignment", alignment));
    OIIO_CHECK_ASSERT(size_bytes > 0 && alignment > 0);
    OIIO_CHECK_EQUAL(size_bytes % alignment, 0);
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
        for (const auto& function : module) {
            if (function.getName().find("osl_layer_group_hart_test_group_name_")
                != 0)
                continue;
            ++internal_layers;
            OIIO_CHECK_ASSERT(function.hasLocalLinkage());
            OIIO_CHECK_ASSERT(!function.isDeclaration());
            OIIO_CHECK_EQUAL(function.arg_size(), 6);
            int calls = 0;
            for (const auto* user : function.users()) {
                const auto* call = llvm::dyn_cast<llvm::CallInst>(user);
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
            }
            OIIO_CHECK_ASSERT(calls > 0);
        }
        OIIO_CHECK_EQUAL(internal_layers, used_layers - 1);
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
        { "float hidden(float x) { if (x>0) return x; return -x; } "
          "shader bad(output color Cout=0) { Cout=color(hidden(u)); }",
          "unsupported operation 'return'" },
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
        { "noise(\"gabor\",point(0.25))", "unsupported noise type 'gabor'" },
        { "noise(\"\",P)", "unsupported noise type ''" },
        { "noise(\"unknown\",P)", "unsupported noise type 'unknown'" },
        { "noise(\"perlin\",P,\"bandwidth\",1.0)", "unsupported type 'string'" },
        { "pnoise(\"simplex\",P,point(2))", "unsupported noise type 'simplex'" },
        { "pnoise(\"usimplex\",P,point(2))",
          "unsupported noise type 'usimplex'" },
        { "pnoise(\"gabor\",P,point(2))", "unsupported noise type 'gabor'" },
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
        for (string_view name : { "perlin", "uperlin", "cell", "hash" }) {
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
                auto group = make_group(ss, bytecode);
                ss.optimize_group(group.get(), nullptr);
                if (errors.errors)
                    print(stderr, "{}\n", errors.last_error);
                OIIO_CHECK_EQUAL(errors.errors, 0);
                const bool periodic = operation == "pnoise";
                const bool derivs   = name == "perlin" || name == "uperlin";
                const auto family   = name == "perlin"    ? "snoise"
                                      : name == "uperlin" ? "noise"
                                      : name == "cell"    ? "cellnoise"
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
    check_rejection(arch, rejected, "matrix indices must be literal");
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
        { "texture(\"x.exr\",u,v,\"interp\",\"linear\",\"wrap\",\"clamp\",\"alpha\",alpha)",
          "unsupported texture option 'alpha'" },
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
    if (argc != 3) {
        print(stderr, "Usage: hart_codegen_test architecture stdosl.h\n");
        return 1;
    }
    const string_view arch(argv[1]);
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
    check_rejection("gfx9999", oso[0], "No embedded HART shadeops");
    check_rejection(arch, oso[2], "unsupported operation 'printf'", 3);
    check_rejection(arch, oso[0], "instrumentation", 1, true);
    check_rejection(arch, oso[2], "unsupported operation 'printf'");
    check_rejection(arch, oso[3], "unsupported operation 'texture'");
    check_rejection(arch, oso[11], "unsupported type");
    check_rejection(arch, oso[14], "unsupported operation 'printf'");
    check_rejection(arch, oso[15], "unsupported operation 'texture'");
    check_rejection(arch, oso[18], "unsupported operation 'break'");
    check_rejection(arch, oso[19], "unsupported operation 'continue'");
    check_rejection(arch, oso[20], "unsupported operation 'dowhile'");
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
    if (!check_chain_modules(arch, argv[2])
        || !check_math_modules(arch, argv[2])
        || !check_noise_modules(arch, argv[2])
        || !check_procedural_modules(arch, argv[2])
        || !check_matrix_modules(arch, argv[2])
        || !check_space_modules(arch, argv[2])
        || !check_geometry_modules(arch, argv[2])
        || !check_texture_modules(arch, argv[2]))
        return 1;
    return unit_test_failures;
}
