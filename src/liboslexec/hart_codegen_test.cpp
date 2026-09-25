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
             bool looping = false)
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
        const bool linked   = shadeop && !shadeop->isDeclaration()
                              && !shadeop->use_empty();
        if (!linked)
            print(stderr, "Expected a linked, used shadeop '{}'\n", name);
        OIIO_CHECK_ASSERT(linked);
        const bool scalar_derivs
            = name == "osl_sin_dfdf" || name == "osl_filterwidth_fdf"
              || OIIO::Strutil::starts_with(name, "osl_noise_df")
              || OIIO::Strutil::starts_with(name, "osl_snoise_df");
        const bool vector_derivs
            = name == "osl_normalize_dvdv" || name == "osl_filterwidth_vdv"
              || OIIO::Strutil::starts_with(name, "osl_noise_dv")
              || OIIO::Strutil::starts_with(name, "osl_snoise_dv");
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



bool
check_noise_modules(string_view arch, string_view stdosl)
{
    const string_view coordinates = "float x=1.7*u-0.23; float y=2.3*v+0.31; "
                                    "point p=point(x,y,u*v+0.7); ";
    for (string_view operation : { "noise", "snoise" }) {
        for (string_view type : { "float", "color", "vector" }) {
            const auto assignment = fmtformat(
                "{0} a={1}(x); {0} b={1}(x,y); {0} c={1}(p); "
                "{0} d={1}(p,u+v); {0} e={1}(x,0.31); {0} f={1}(p,0.19); "
                "value=a+b+c+d+e+f; ",
                type, operation);
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
                        print(stderr, "{} {} (LLVM {}): {}\n", operation, type,
                              optimize, errors.last_error);
                    OIIO_CHECK_EQUAL(errors.errors, 0);
                    const auto prefix = fmtformat("osl_{}_{}{}", operation,
                                                  connected ? "d" : "",
                                                  type == "float" ? "f" : "v");
                    const std::string names[] = {
                        prefix + (connected ? "df" : "f"),
                        prefix + (connected ? "dfdf" : "ff"),
                        prefix + (connected ? "dv" : "v"),
                        prefix + (connected ? "dvdf" : "vf"),
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
        { "noise(\"perlin\",point(0.25))", "unsupported type 'string'" },
        { "noise(\"gabor\",P,\"bandwidth\",1.0)", "unsupported type 'string'" },
        { "pnoise(P,vector(2))", "unsupported operation 'pnoise'" },
        { "psnoise(P,vector(2))", "unsupported operation 'psnoise'" },
        { "cellnoise(P)", "unsupported operation 'cellnoise'" },
        { "hashnoise(P)", "unsupported operation 'hashnoise'" },
        { "noise(I)", "unsupported shader global 'I'" },
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
        "shader hart_surface_incident(output color Cout=0) { Cout=color(I); }",
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
        "shader hart_filterwidth_incident(output color Cout=0) { Cout=color(filterwidth(I)); }",
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
    check_rejection(arch, oso[0], "one or two shader layers", 3);
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
    check_rejection(arch, oso[44], "unsupported shader global 'I'");
    check_rejection(arch, oso[35], "unsupported shader global 'I'");
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
    check_rejection(arch, oso[38], "unsupported operation 'functioncall'");
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
            check_rejection(arch, bytecode, "unsupported type 'string'");
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
    if (!check_noise_modules(arch, argv[2]))
        return 1;
    return unit_test_failures;
}
