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
    int supports(string_view feature) const override
    { return feature == "HART"; }
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
    OIIO_CHECK_ASSERT(OIIO::Strutil::contains(errors.last_error, expected));
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
check_module(ShadingSystem& ss, ShaderGroup& group, string_view arch, bool sine,
             int optimize, bool connected = false, bool branching = false)
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
    if (sine && optimize == 10) {
        bool shadeop = false;
        for (const auto& function : module)
            shadeop |= function.getName().find("osl_sin_") == 0
                       && !function.isDeclaration() && !function.use_empty();
        OIIO_CHECK_ASSERT(shadeop);
    }
    int size_bytes = 0, alignment = 0;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_size", size_bytes));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "llvm_groupdata_alignment", alignment));
    OIIO_CHECK_ASSERT(size_bytes > 0 && alignment > 0);
    OIIO_CHECK_EQUAL(size_bytes % alignment, 0);
    const void* again = nullptr;
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &again));
    OIIO_CHECK_EQUAL(bytes, again);
}



void
check_rejection(string_view arch, string_view oso, string_view expected,
                int layers = 1, bool instrument = false)
{
    HartServices renderer;
    Diagnostics errors;
    ShadingSystem ss(&renderer, nullptr, &errors);
    OIIO_CHECK_ASSERT(ss.attribute("hart_arch", arch));
    if (instrument)
        ss.attribute("debug_nan", 1);
    auto group = make_group(ss, oso, layers);
    check_rejected_group(ss, *group, errors, expected);
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
        "shader hart_loop(output color Cout=0) { int i=0; while (i<2) { Cout+=color(u,v,0); i+=1; } }",
        "shader hart_branch_printf(output color Cout=0) { if (u>v) printf(\"not supported\"); Cout=color(u,v,0); }",
        "shader hart_branch_texture(output color Cout=0) { if (u>v) Cout=texture(\"missing.tx\",u,v); else Cout=0; }",
        "shader hart_compare(output color Cout=0) { int a=(u>0.5)-1; int b=(v>0.5)-1; Cout=color((u<v)+2*(u<=v)+4*(a<b)+8*(a<=b), (u>v)+2*(u>=v)+4*(a>b)+8*(a>=b), (u==v)+2*(u!=v)+4*(a==b)+8*(a!=b)); }",
    };
    std::vector<std::string> oso(std::size(sources));
    for (size_t i = 0; i < oso.size(); ++i) {
        OSLCompiler compiler;
        if (!compiler.compile_buffer(sources[i], oso[i], { }, argv[2]))
            return 1;
    }
    // OSL level 10 skips passes; even O0 inlines alwaysinline HIP shadeops.
    for (int optimize : { 10, 3 }) {
        for (int i : { 0, 1, 12, 16 }) {
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
            check_module(ss, *group, arch, i == 1, optimize, false, i == 12);
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
        for (int consumer : { 6, 12 }) {
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
            check_module(ss, *group, arch, true, optimize, true,
                         consumer == 12);
        }
    }
    check_rejection("gfx9999", oso[0], "No embedded HART shadeops");
    check_rejection(arch, oso[0], "one or two shader layers", 3);
    check_rejection(arch, oso[0], "instrumentation", 1, true);
    check_rejection(arch, oso[2], "unsupported operation 'printf'");
    check_rejection(arch, oso[3], "unsupported operation 'texture'");
    check_rejection(arch, oso[4], "shader globals u and v");
    check_rejection(arch, oso[11], "unsupported type");
    check_rejection(arch, oso[13], "unsupported operation 'while'");
    check_rejection(arch, oso[14], "unsupported operation 'printf'");
    check_rejection(arch, oso[15], "unsupported operation 'texture'");
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
    return unit_test_failures;
}
