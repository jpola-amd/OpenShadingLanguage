// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HIP must precede OIIO: OIIO's __has_attribute fallback selects HIP's
// GNU vector implementation on MSVC.
#include <hip/hip_runtime_api.h>

// Keep these aliases out of headers shared with the real OptiX backend.
#define HART_OPTIX_COMPAT_DEFINE_FUNCTION_TABLE
#include <amd/hart/hart_optix_compat.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/strutil.h>

#include <OSL/oslquery.h>

#include "hart_generated_bitcode.h"
#include "hartgeneratedparams.h"
#include "hartgridparams.h"
#include "hartgridrender.h"
#include "harttexture.h"
#include "simplerend.h"

OSL_PRAGMA_WARNING_PUSH
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#if OSL_LLVM_VERSION >= 170
#    include <llvm/TargetParser/Triple.h>
#else
#    include <llvm/ADT/Triple.h>
#endif
OSL_PRAGMA_WARNING_POP

#ifdef LLVM_NAMESPACE
namespace llvm = LLVM_NAMESPACE;
#endif

OSL_NAMESPACE_BEGIN
namespace {

struct HartModuleInput {
    std::string filename;
    std::string bitcode;
    std::string arch;
};



struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HartSbtRecord {
    std::array<unsigned char, OPTIX_SBT_RECORD_HEADER_SIZE> header { };
};



bool
validate_module(HartModuleInput& input, ErrorHandler& err,
                cspan<std::string> callables = { })
{
    const auto& filename = input.filename;
    auto& bitcode        = input.bitcode;
    auto& arch           = input.arch;
    if (bitcode.compare(0, 4, "BC\xc0\xde", 4) != 0
        && bitcode.compare(0, 4, "\xde\xc0\x17\x0b", 4) != 0) {
        err.errorfmt("'{}' is not LLVM bitcode. Assemble textual LLVM IR with "
                     "llvm-as first; PTX and HIP code objects are not HART "
                     "module inputs.",
                     filename);
        return false;
    }
    llvm::LLVMContext context;
    llvm::MemoryBufferRef buffer(llvm::StringRef(bitcode.data(), bitcode.size()),
                                 llvm::StringRef(filename.data(),
                                                 filename.size()));
    auto parsed = llvm::parseBitcodeFile(buffer, context);
    if (!parsed) {
        err.errorfmt("Cannot parse HART bitcode '{}': {}", filename,
                     llvm::toString(parsed.takeError()));
        return false;
    }
    const auto& module = **parsed;
    std::string error;
    llvm::raw_string_ostream diagnostics(error);
    if (llvm::verifyModule(module, &diagnostics)) {
        diagnostics.flush();
        err.errorfmt("Invalid HART bitcode '{}': {}", filename, error);
        return false;
    }
    const llvm::Triple triple(module.getTargetTriple());
    if (triple.getArch() != llvm::Triple::amdgcn
        || triple.getOS() != llvm::Triple::AMDHSA) {
        err.errorfmt("HART bitcode '{}' must target amdgcn-amd-amdhsa, not '{}'",
                     filename, triple.str());
        return false;
    }

    arch.clear();
    for (const auto& function : module) {
        const auto cpu = function.getFnAttribute("target-cpu");
        if (!cpu.isStringAttribute() || cpu.getValueAsString().empty())
            continue;
        const std::string target = cpu.getValueAsString().str();
        if (!arch.empty() && arch != target) {
            err.errorfmt("HART bitcode '{}' mixes target-cpu '{}' and '{}' "
                         "(function '{}'); link modules for one architecture",
                         filename, arch, target, function.getName().str());
            return false;
        }
        arch = target;
    }
    for (const auto& name : callables) {
        const auto* function = module.getFunction(name);
        bool valid = function && !function->isDeclaration()
                     && function->hasExternalLinkage()
                     && function->getReturnType()->isVoidTy()
                     && !function->isVarArg() && function->arg_size() == 6
                     && function->getCallingConv() == llvm::CallingConv::C
                     && OIIO::Strutil::starts_with(name, "__direct_callable__");
        if (valid) {
            unsigned int index = 0;
            for (const auto& arg : function->args()) {
                const auto* pointer = llvm::dyn_cast<llvm::PointerType>(
                    arg.getType());
                valid &= index == 4
                             ? arg.getType()->isIntegerTy(32)
                             : pointer && pointer->getAddressSpace() == 0;
                ++index;
            }
        }
        if (!valid) {
            err.errorfmt("Generated HART callable '{}' must define the "
                         "six-argument OSL callable ABI in '{}'",
                         name, filename);
            return false;
        }
    }
    if (!callables.empty()) {
        if (arch.empty()) {
            err.errorfmt("Generated HART bitcode '{}' has no target-cpu",
                         filename);
            return false;
        }
        for (const auto& function : module) {
            if (function.isDeclaration() && !function.use_empty()
                && OIIO::Strutil::starts_with(function.getName().str(),
                                              "osl_")) {
                err.errorfmt("Generated HART bitcode has an undefined "
                             "shadeop '{}'",
                             function.getName().str());
                return false;
            }
        }
    }
    return true;
}



bool
read_module(HartModuleInput& input, ErrorHandler& err)
{
    const auto size = OIIO::Filesystem::file_size(input.filename);
    if (!size || size > input.bitcode.max_size()) {
        err.errorfmt("Cannot read HART bitcode file '{}' (missing, empty, or "
                     "too large)",
                     input.filename);
        return false;
    }
    input.bitcode.resize(size_t(size));
    if (OIIO::Filesystem::read_bytes(input.filename, input.bitcode.data(),
                                     input.bitcode.size())
        != input.bitcode.size()) {
        err.errorfmt("Cannot read complete HART bitcode file '{}'",
                     input.filename);
        return false;
    }
    return validate_module(input, err);
}



class GeneratedRenderer final : public SimpleRenderer {
public:
    GeneratedRenderer() : m_textures(errhandler()) { }

    int supports(string_view feature) const override
    {
        return feature == "HART" || feature == "HARTTextures"
               || feature == "HARTTransforms";
    }

    TextureHandle* get_texture_handle(ustring filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        return reinterpret_cast<TextureHandle*>(
            uintptr_t(m_textures.load(filename)));
    }
    bool good(TextureHandle* handle) override { return handle != nullptr; }

    HartTextureStore& textures() { return m_textures; }

private:
    HartTextureStore m_textures;
};



class HartGridRenderer {
public:
    explicit HartGridRenderer(ErrorHandler& err) : m_err(err) { }
    ~HartGridRenderer() { clear(); }
    HartGridRenderer(const HartGridRenderer&)            = delete;
    HartGridRenderer& operator=(const HartGridRenderer&) = delete;

    bool initialize(int device, bool verbose, cspan<HartModuleInput> inputs,
                    bool no_cache)
    {
        m_verbose = verbose;
        hipDeviceProp_t properties { };
        if (!hip_check(hipSetDevice(device), "hipSetDevice")
            || !hip_check(hipGetDeviceProperties(&properties, device),
                          "hipGetDeviceProperties"))
            return false;
        if (verbose)
            m_err.infofmt("HART device {}: {} ({})", device, properties.name,
                          properties.gcnArchName);

        // HIP target IDs may append features (e.g. ":xnack-") to the CPU name.
        const string_view device_target(properties.gcnArchName);
        const string_view device_arch
            = device_target.substr(0, device_target.find(':'));
        for (const auto& input : inputs) {
            if (!input.arch.empty() && input.arch != device_arch) {
                m_err.errorfmt(
                    "HART bitcode '{}' targets '{}', but HIP device {} ({}) is '{}'. "
                    "Use bitcode compiled for '{}'; refusing to create a HART pipeline.",
                    input.filename, input.arch, device, properties.name,
                    device_arch, device_arch);
                return false;
            }
        }
        if (!hip_check(hipFree(nullptr), "HIP context initialization"))
            return false;

        OptixDeviceContextOptions options { };
        options.logCallbackFunction = log_callback;
        options.logCallbackData     = &m_err;
        options.logCallbackLevel    = verbose ? 4 : 2;
        if (m_verbose)
            m_err.infofmt("Initializing HART runtime");
        if (!hart_check(optixInit(), "hartInit"))
            return false;
        if (m_verbose)
            m_err.infofmt("Creating HART context");
        if (!hart_check(optixDeviceContextCreate(nullptr, &options, &m_context),
                        "hartDeviceContextCreate"))
            return false;
        if (no_cache) {
            if (!hart_check(optixDeviceContextSetCacheEnabled(m_context, 0),
                            "hartDeviceContextSetCacheEnabled"))
                return false;
            if (m_verbose)
                m_err.infofmt("HART pipeline cache disabled");
        }
        return hip_check(hipStreamCreate(&m_stream), "hipStreamCreate");
    }

    bool load(cspan<HartModuleInput> inputs, const std::string& entry,
              cspan<std::string> callable_names = { })
    {
        OptixModuleCompileOptions module_options { };
        OptixPipelineCompileOptions compile_options { };
        compile_options.pipelineLaunchParamsVariableName
            = "testshade_hart_params";
        std::array<char, 8192> log { };
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (m_verbose)
                m_err.infofmt("Loading HART module '{}' ({} bytes)",
                              input.filename, input.bitcode.size());
            size_t log_size = log.size();
            if (!hart_check(optixModuleCreate(m_context, &module_options,
                                              &compile_options,
                                              input.bitcode.data(),
                                              input.bitcode.size(), log.data(),
                                              &log_size, &m_modules[i]),
                            "hartModuleCreate")) {
                log.back() = '\0';
                m_err.errorfmt("HART module '{}': {}", input.filename,
                               log.data());
                return false;
            }
        }

        std::array<OptixProgramGroupDesc, 3> desc { };
        desc[0].kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc[0].raygen.module            = m_modules[0];
        desc[0].raygen.entryFunctionName = entry.c_str();
        const std::array<std::string, 2> external_names {
            "__direct_callable__testshade_init",
            "__direct_callable__testshade_entry"
        };
        if (callable_names.empty())
            callable_names = external_names;
        if (callable_names.size() > 2) {
            m_err.errorfmt("HART grid supports at most two selected callables");
            return false;
        }
        m_group_count = inputs.size() == 2 ? 1 + callable_names.size() : 1;
        for (unsigned int i = 1; i < m_group_count; ++i) {
            desc[i].kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            desc[i].callables.moduleDC = m_modules[1];
            desc[i].callables.entryFunctionNameDC
                = callable_names[i - 1].c_str();
        }
        for (unsigned int i = 0; i < m_group_count; ++i) {
            size_t log_size = log.size();
            if (!hart_check(optixProgramGroupCreate(m_context, &desc[i], 1,
                                                    nullptr, log.data(),
                                                    &log_size, &m_groups[i]),
                            "hartProgramGroupCreate")) {
                log.back() = '\0';
                m_err.errorfmt("HART entry '{}': {}",
                               i == 0 ? entry : callable_names[i - 1],
                               log.data());
                return false;
            }
        }

        OptixPipelineLinkOptions link_options { };
        link_options.maxTraceDepth = 0;
        size_t log_size            = log.size();
        if (m_verbose)
            m_err.infofmt("Compiling HART pipeline");
        if (!hart_check(optixPipelineCreate(m_context, &compile_options,
                                            &link_options, m_groups.data(),
                                            m_group_count, log.data(),
                                            &log_size, &m_pipeline),
                        "hartPipelineCreate")) {
            log.back() = '\0';
            m_err.errorfmt("HART pipeline for entry '{}': {}", entry,
                           log.data());
            return false;
        }

        std::array<HartSbtRecord, 3> records { };
        for (unsigned int i = 0; i < m_group_count; ++i) {
            if (!hart_check(optixSbtRecordPackHeader(m_groups[i],
                                                     records[i].header.data()),
                            "hartSbtRecordPackHeader"))
                return false;
        }
        const size_t bytes = m_group_count * sizeof(HartSbtRecord);
        return hip_check(hipMalloc(&m_record, bytes), "hipMalloc SBT")
               && hip_check(hipMemcpy(m_record, records.data(), bytes,
                                      hipMemcpyHostToDevice),
                            "hipMemcpy SBT");
    }

    bool render(int width, int height, int iterations, bool warmup,
                span<float> pixels, size_t group_size = 0,
                size_t group_alignment = 0, int raytype = 0,
                HartTextureStore* textures = nullptr,
                cspan<Matrix44> transforms = { }, size_t local_groupdata = 0)
    {
        const size_t bytes       = pixels.size() * sizeof(float);
        const size_t params_size = group_alignment
                                       ? sizeof(testshade::HartGeneratedParams)
                                       : sizeof(testshade::HartGridParams);
        if (!hip_check(hipMalloc(&m_output, bytes), "hipMalloc output")
            || !hip_check(hipMalloc(&m_params, params_size),
                          "hipMalloc parameters"))
            return false;
        const testshade::HartGridParams params { static_cast<float*>(m_output) };
        testshade::HartGeneratedParams generated { };
        if (group_alignment) {
            if (transforms.size() != 4) {
                m_err.errorfmt(
                    "Generated HART mode requires four transform matrices");
                return false;
            }
            if (!hip_check(hipMalloc(&m_transforms, transforms.size_bytes()),
                           "hipMalloc transforms")
                || !hip_check(hipMemcpy(m_transforms, transforms.data(),
                                        transforms.size_bytes(),
                                        hipMemcpyHostToDevice),
                              "hipMemcpy transforms"))
                return false;
            const size_t limit = std::numeric_limits<size_t>::max();
            const size_t count = pixels.size() / 3;
            if ((group_alignment & (group_alignment - 1))
                || std::max(size_t(1), group_size) > limit - group_alignment
                || (local_groupdata && local_groupdata != group_size)
                || count == 0) {
                m_err.errorfmt("Invalid generated HART group storage layout");
                return false;
            }
            const size_t stride = (std::max(size_t(1), group_size)
                                   + group_alignment - 1)
                                  & ~(group_alignment - 1);
            if (!local_groupdata
                && count > (limit - group_alignment + 1) / stride) {
                m_err.errorfmt(
                    "HART grid exceeds the group storage size limit");
                return false;
            }
            const size_t scratch_bytes = local_groupdata ? 0 : count * stride;
            if (scratch_bytes
                && (!hip_check(hipMalloc(&m_scratch,
                                         scratch_bytes + group_alignment - 1),
                               "hipMalloc group storage")
                    || !hip_check(hipMemset(m_scratch, 0,
                                            scratch_bytes + group_alignment - 1),
                                  "hipMemset group storage")))
                return false;
            const uintptr_t aligned
                = !m_scratch ? 0
                             : (reinterpret_cast<uintptr_t>(m_scratch)
                                + group_alignment - 1)
                                   & ~(uintptr_t(group_alignment) - 1);
            generated = { static_cast<float*>(m_output),
                          reinterpret_cast<unsigned char*>(aligned),
                          local_groupdata ? 0 : stride,
                          scratch_bytes,
                          count,
                          raytype,
                          textures ? textures->device_state() : nullptr,
                          static_cast<const Matrix44*>(m_transforms) };
            if (m_verbose)
                m_err.infofmt(
                    "HART group storage: {} bytes, alignment {}, local {} bytes, scratch {} bytes",
                    group_size, group_alignment, local_groupdata,
                    scratch_bytes);
        }
        if (!hip_check(hipMemcpy(m_params,
                                 group_alignment
                                     ? static_cast<const void*>(&generated)
                                     : static_cast<const void*>(&params),
                                 params_size, hipMemcpyHostToDevice),
                       "hipMemcpy parameters"))
            return false;

        OptixShaderBindingTable sbt { };
        sbt.raygenRecord = m_record;
        if (m_group_count > 1) {
            sbt.callablesRecordBase          = static_cast<char*>(m_record)
                                               + sizeof(HartSbtRecord);
            sbt.callablesRecordStrideInBytes = sizeof(HartSbtRecord);
            sbt.callablesRecordCount         = m_group_count - 1;
        }
        auto launch = [&]() {
            if (textures && !textures->reset_errors())
                return false;
            // Reset every launch so warmup cannot conceal unwritten output.
            if (!hip_check(hipMemsetAsync(m_output, 0xff, bytes, m_stream),
                           "hipMemsetAsync output"))
                return false;
            if (m_verbose)
                m_err.infofmt("Launching HART grid {} x {}", width, height);
            if (!hart_check(optixLaunch(m_pipeline, m_stream, m_params,
                                        params_size, &sbt, width, height, 1),
                            "hartLaunch"))
                return false;
            if (m_verbose)
                m_err.infofmt("Waiting for HART grid completion");
            return hip_check(hipStreamSynchronize(m_stream),
                             "hipStreamSynchronize")
                   && (!textures || textures->check_errors());
        };
        if (warmup && !launch())
            return false;
        for (int i = 0; i < iterations; ++i)
            if (!launch())
                return false;
        return hip_check(hipMemcpy(pixels.data(), m_output, bytes,
                                   hipMemcpyDeviceToHost),
                         "hipMemcpy output");
    }

    bool clear()
    {
        bool ok = true;
        if (m_stream)
            ok = hip_check(hipStreamSynchronize(m_stream),
                           "hipStreamSynchronize during cleanup")
                 && ok;
        for (void** buffer :
             { &m_params, &m_output, &m_record, &m_scratch, &m_transforms }) {
            if (*buffer) {
                ok      = hip_check(hipFree(*buffer), "hipFree") && ok;
                *buffer = nullptr;
            }
        }
        if (m_pipeline) {
            ok         = hart_check(optixPipelineDestroy(m_pipeline),
                                    "hartPipelineDestroy")
                         && ok;
            m_pipeline = nullptr;
        }
        for (auto& group : m_groups) {
            if (group) {
                ok    = hart_check(optixProgramGroupDestroy(group),
                                   "hartProgramGroupDestroy")
                        && ok;
                group = nullptr;
            }
        }
        for (auto& module : m_modules) {
            if (module) {
                ok = hart_check(optixModuleDestroy(module), "hartModuleDestroy")
                     && ok;
                module = nullptr;
            }
        }
        if (m_context) {
            ok        = hart_check(optixDeviceContextDestroy(m_context),
                                   "hartDeviceContextDestroy")
                        && ok;
            m_context = nullptr;
        }
        if (m_stream) {
            ok       = hip_check(hipStreamDestroy(m_stream), "hipStreamDestroy")
                       && ok;
            m_stream = nullptr;
        }
        return ok;
    }

private:
    bool hip_check(hipError_t result, string_view operation)
    {
        if (result == hipSuccess)
            return true;
        m_err.errorfmt("{} failed: {} ({})", operation, hipGetErrorName(result),
                       hipGetErrorString(result));
        return false;
    }

    bool hart_check(OptixResult result, string_view operation)
    {
        if (result == OPTIX_SUCCESS)
            return true;
        m_err.errorfmt("{} failed: {} ({})", operation,
                       optixGetErrorName(result), optixGetErrorString(result));
        return false;
    }

    static void log_callback(unsigned int level, const char* tag,
                             const char* message, void* data)
    {
        auto& err = *static_cast<ErrorHandler*>(data);
        if (level <= 1)
            err.errorfmt("HART [{}]: {}", tag, message);
        else if (level == 2)
            err.warningfmt("HART [{}]: {}", tag, message);
        else
            err.infofmt("HART [{}]: {}", tag, message);
    }

    ErrorHandler& m_err;
    bool m_verbose               = false;
    hipStream_t m_stream         = nullptr;
    OptixDeviceContext m_context = nullptr;
    std::array<OptixModule, 2> m_modules { };
    std::array<OptixProgramGroup, 3> m_groups { };
    unsigned int m_group_count = 0;
    OptixPipeline m_pipeline   = nullptr;
    void* m_record             = nullptr;
    void* m_params             = nullptr;
    void* m_output             = nullptr;
    void* m_scratch            = nullptr;
    void* m_transforms         = nullptr;
};

}  // namespace



bool
testshade_hart_validate_generated(int argc, const char* argv[],
                                  const HartOptions& options, int width,
                                  int height, int iterations)
{
    ErrorHandler err;
    if (options.has_callables || options.has_entry) {
        err.errorfmt("--hart-callable-module and --hart-entry require "
                     "--hart-module; they cannot be mixed with an OSL shader");
        return false;
    }
    // Parse the supported subset separately, rather than silently ignoring
    // CPU-only switches accepted by the shared frontend.
    OIIO::ArgParse ap;
    ap.exit_on_error(false);
    std::string device = "0";
    std::vector<std::string> names, files;
    std::string format;
    bool parameter_hints = false, has_shader = false;
    // clang-format off
    ap.arg("filename")
      .action([&](cspan<const char*>) { has_shader = true; });
    ap.arg("--hart");
    ap.arg("--hart-device %s:INDEX", &device);
    ap.arg("--hart-no-cache");
    ap.arg("--hart-fused");
    ap.arg("--hart-local-groupdata %s:BYTES");
    ap.arg("--res %d:WIDTH %d:HEIGHT");
    ap.arg("-g %d:WIDTH %d:HEIGHT");
    ap.arg("--iters %d:COUNT");
    ap.arg("--warmup");
    ap.arg("--print");
    ap.arg("-v");
    ap.arg("--debug");
    ap.arg("-o %L:VARIABLE %L:FILE", &names, &files);
    ap.arg("-d %s:FORMAT", &format);
    ap.arg("--groupname %s:NAME");
    ap.arg("--layer %s:NAME");
    ap.arg("--shader %s:SHADER %s:LAYER")
      .action([&](cspan<const char*>) { has_shader = true; });
    ap.arg("--connect %s:FROMLAYER %s:FROMOUTPUT %s:TOLAYER %s:TOINPUT");
    ap.arg("--param %s:NAME %s:VALUE")
      .action([&](cspan<const char*> args) {
          const string_view option(args[0]);
          parameter_hints |= option.find("interpolated=") != string_view::npos
                             || option.find("interactive=") != string_view::npos;
      });
    ap.arg("-O0");
    ap.arg("-O1");
    ap.arg("-O2");
    ap.arg("--llvm_opt %d:LEVEL");
    // clang-format on
    if (ap.parse_args(argc, argv) < 0) {
        err.errorfmt("Generated HART mode: unsupported option or argument: {}",
                     ap.geterror());
        return false;
    }
    if (options.has_local_groupdata && !options.fused) {
        err.errorfmt("--hart-local-groupdata requires --hart-fused");
        return false;
    }
    const auto local_bytes = std::strtoll(options.local_groupdata.c_str(),
                                          nullptr, 10);
    if (options.local_groupdata.empty()
        || options.local_groupdata.find_first_not_of("0123456789")
               != std::string::npos
        || local_bytes > std::numeric_limits<int>::max()) {
        err.errorfmt("Invalid --hart-local-groupdata byte limit '{}'; "
                     "expected an integer in [0,2147483647]",
                     options.local_groupdata);
        return false;
    }
    char* device_end        = nullptr;
    const auto device_index = std::strtoll(device.c_str(), &device_end, 10);
    if (device.empty() || device_end != device.c_str() + device.size()
        || device_index < std::numeric_limits<int>::min()
        || device_index > std::numeric_limits<int>::max()) {
        err.errorfmt("Invalid HART device index '{}'", device);
        return false;
    }
    if (parameter_hints) {
        err.errorfmt("Generated HART mode does not support interpolated or "
                     "interactive parameters");
        return false;
    }
    if (!has_shader) {
        err.errorfmt("Generated HART mode requires an OSL shader; "
                     "use --hart-module for external bitcode");
        return false;
    }
    const char* batched = std::getenv("TESTSHADE_BATCHED");
    if (batched && OIIO::Strutil::stoi(batched)) {
        err.errorfmt("Generated HART mode does not support TESTSHADE_BATCHED");
        return false;
    }
    const char* rs_bitcode = std::getenv("TESTSHADE_RS_BITCODE");
    if (rs_bitcode && OIIO::Strutil::stoi(rs_bitcode)) {
        err.errorfmt("Generated HART mode does not support "
                     "TESTSHADE_RS_BITCODE");
        return false;
    }
    if (options.device < 0 || width <= 0 || height <= 0 || iterations <= 0) {
        err.errorfmt("HART device must be nonnegative; grid dimensions and "
                     "iteration count must be positive");
        return false;
    }
    if (names.size() > 1 || (!names.empty() && names[0] != "Cout")) {
        err.errorfmt("Generated HART mode supports one RGB output: Cout");
        return false;
    }
    if (!format.empty() && format != "float" && format != "half"
        && format != "uint8") {
        err.errorfmt("Unsupported HART output format '{}'", format);
        return false;
    }
    if (size_t(width) > std::vector<float>().max_size() / 3 / size_t(height)) {
        err.errorfmt(
            "HART grid dimensions exceed the output buffer size limit");
        return false;
    }
    if (size_t(width)
        > size_t(std::numeric_limits<int>::max()) / size_t(height)) {
        err.errorfmt("HART callable grid exceeds the int shade-index range");
        return false;
    }
    return true;
}



std::unique_ptr<SimpleRenderer>
testshade_hart_renderer(int device, std::string& arch)
{
    auto renderer = std::make_unique<GeneratedRenderer>();
    hipDeviceProp_t properties { };
    hipError_t status = hipSetDevice(device);
    if (status == hipSuccess)
        status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) {
        renderer->errhandler().errorfmt("Cannot select HIP device {}: {}",
                                        device, hipGetErrorString(status));
        return { };
    }
    const string_view target(properties.gcnArchName);
    arch = std::string(target.substr(0, target.find(':')));
    for (const auto& embedded : hart_generated_raygens)
        if (embedded.arch && arch == embedded.arch)
            return renderer;
    renderer->errhandler().errorfmt(
        "No embedded HART raygen for HIP architecture '{}'; configure "
        "USE_LLVM_BITCODE=ON and include it in HART_TARGET_ARCHITECTURES",
        arch);
    return { };
}



bool
testshade_hart_generated(SimpleRenderer& renderer, ShadingSystem& shadingsys,
                         ShaderGroup& group, const HartOptions& options,
                         string_view arch, int width, int height,
                         int iterations, bool warmup, bool verbose, int raytype,
                         bool print_pixels, string_view output_file,
                         string_view dataformat, const Matrix44& object2common,
                         const Matrix44& shader2common)
{
    auto& err  = renderer.errhandler();
    auto* generated = dynamic_cast<GeneratedRenderer*>(&renderer);
    if (!generated) {
        err.errorfmt("Generated HART mode requires its HART renderer");
        return false;
    }
    int layers = 0;
    if (!shadingsys.getattribute(&group, "num_layers", layers) || layers < 1) {
        err.errorfmt("Generated HART mode requires at least one shader layer");
        return false;
    }
    int outputs    = 0;
    bool has_cout  = false;
    for (int layer = 0; layer < layers; ++layer) {
        OSLQuery query = shadingsys.oslquery(group, layer);
        for (const auto& parameter : query) {
            if (layer == layers - 1 && parameter.isoutput) {
                ++outputs;
                has_cout |= parameter.name == "Cout"
                            && parameter.type == TypeColor
                            && !parameter.isclosure;
            }
            if (parameter.isclosure || parameter.isstruct
                || parameter.type.is_array()
                || parameter.type.basetype == TypeDesc::STRING) {
                err.errorfmt(
                    "Generated HART mode does not support parameter '{}' "
                    "of type '{}'",
                    parameter.name, parameter.type_name());
                return false;
            }
        }
    }
    if (!has_cout || outputs != 1) {
        err.errorfmt("Generated HART mode requires exactly one RGB color "
                     "output parameter on the last layer: Cout");
        return false;
    }
    std::vector<ustring> layer_names(layers);
    if (!shadingsys.getattribute(&group, "layer_names",
                                 TypeDesc(TypeDesc::STRING, layers),
                                 layer_names.data())) {
        err.errorfmt("Cannot retrieve HART shader layer names");
        return false;
    }
    const SymLocationDesc output(fmtformat("{}.Cout", layer_names.back()),
                                 TypeColor, false, SymArena::Outputs, 0,
                                 3 * sizeof(float));
    shadingsys.add_symlocs(&group, { &output, 1 });
    shadingsys.optimize_group(&group, nullptr);

    const void* data = nullptr;
    uint64_t bytes   = 0;
    int group_size = -1, group_alignment = 0, local_groupdata = 0;
    std::vector<std::string> callables(options.fused ? 1 : 2);
    if (!shadingsys.getattribute(&group, "hart_bitcode", TypeDesc::PTR, &data)
        || !shadingsys.getattribute(&group, "hart_bitcode_size", TypeUInt64,
                                    &bytes)
        || !data || !bytes || bytes > std::string().max_size()
        || !shadingsys.getattribute(&group, "llvm_groupdata_size", group_size)
        || !shadingsys.getattribute(&group, "llvm_groupdata_alignment",
                                    group_alignment)
        || !shadingsys.getattribute(&group, "hart_groupdata_alloc",
                                    local_groupdata)
        || group_size < 0 || group_alignment <= 0 || local_groupdata < 0
        || (local_groupdata && local_groupdata != group_size)
        || !shadingsys.getattribute(&group,
                                    options.fused ? "group_fused_name"
                                                  : "group_init_name",
                                    callables[0])
        || (!options.fused
            && !shadingsys.getattribute(&group, "group_entry_name",
                                        callables[1]))) {
        err.errorfmt("Cannot retrieve a compiled HART shader group; "
                     "CPU fallback is not supported");
        return false;
    }
    const auto* symbol = shadingsys.find_symbol(group, layer_names.back(),
                                                ustring("Cout"));
    if (!symbol || shadingsys.symbol_typedesc(symbol) != TypeColor) {
        err.errorfmt("Compiled HART group has no RGB color Cout symbol");
        return false;
    }

    std::array<HartModuleInput, 2> modules;
    modules[1].filename = "OSL-generated shader group";
    modules[1].bitcode.assign(static_cast<const char*>(data), size_t(bytes));
    for (const auto& embedded : hart_generated_raygens) {
        if (embedded.arch && arch == embedded.arch) {
            modules[0].filename = "embedded HART raygen " + std::string(arch);
            modules[0].bitcode.assign(reinterpret_cast<const char*>(
                                          embedded.data),
                                      *embedded.size);
            break;
        }
    }
    if (modules[0].bitcode.empty()) {
        err.errorfmt("No embedded HART raygen for architecture '{}'", arch);
        return false;
    }
    if (!validate_module(modules[0], err)
        || !validate_module(modules[1], err, callables))
        return false;
    auto& textures = generated->textures();
    if (!textures.prepare())
        return false;
    HartGridRenderer runtime(err);
    const char* entry = options.fused ? "__raygen__testshade_generated_fused"
                                      : "__raygen__testshade_generated";
    if (verbose)
        err.infofmt("HART callable mode: {}",
                    options.fused ? "fused" : "split");
    if (!runtime.initialize(options.device, verbose, modules, options.no_cache)
        || !runtime.load(modules, entry, callables))
        return false;
    std::vector<float> pixels(size_t(width) * size_t(height) * 3);
    const Matrix44 transforms[] = { object2common, object2common.inverse(),
                                    shader2common, shader2common.inverse() };
    const bool rendered
        = runtime.render(width, height, iterations, warmup, pixels,
                         size_t(group_size), size_t(group_alignment), raytype,
                         &textures, transforms,
                         options.fused ? size_t(local_groupdata) : 0);
    const bool cleared  = runtime.clear();
    if (!rendered || !cleared)
        return false;
    if (print_pixels) {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const size_t offset = (size_t(y) * width + x) * 3;
                print("Pixel ({}, {}): Cout = {:.9g} {:.9g} {:.9g}\n", x, y,
                      pixels[offset], pixels[offset + 1], pixels[offset + 2]);
            }
    } else if (output_file != "null") {
        OIIO::ImageBuf image(
            OIIO::ImageSpec(width, height, 3, TypeDesc::FLOAT));
#if OIIO_VERSION_GREATER_EQUAL(3, 1, 0)
        bool copied = image.set_pixels(OIIO::ROI::All(),
                                       OIIO::make_cspan(pixels));
#else
        bool copied = image.set_pixels(OIIO::ROI::All(), TypeDesc::FLOAT,
                                       pixels.data());
#endif
        if (OIIO::Strutil::iends_with(output_file, ".jpg")
            || OIIO::Strutil::iends_with(output_file, ".jpeg")
            || OIIO::Strutil::iends_with(output_file, ".gif")
            || OIIO::Strutil::iends_with(output_file, ".png"))
            image = OIIO::ImageBufAlgo::colorconvert(image, "linear", "sRGB");
        const TypeDesc format = dataformat == "half"    ? TypeDesc::HALF
                                : dataformat == "uint8" ? TypeDesc::UINT8
                                                        : TypeDesc::FLOAT;
        if (!copied || image.has_error() || !image.write(output_file, format)) {
            err.errorfmt("Cannot write HART output '{}': {}", output_file,
                         image.geterror());
            return false;
        }
    }
    return true;
}



int
testshade_hart(int argc, const char* argv[])
{
    ErrorHandler err;
    std::array<HartModuleInput, 2> inputs;
    std::string entry = "__raygen__testshade";
    std::vector<std::string> output_names, output_files;
    int device = 0, width = 1, height = 1, iterations = 1;
    bool enabled = false, print_pixels = false, verbose = false, warmup = false;
    bool has_shader    = false;
    bool has_callables = false, no_cache = false;
    OIIO::ArgParse ap;
    ap.exit_on_error(false);
    ap.intro("testshade --hart: external AMDGPU bitcode grid runner");
    ap.usage("testshade --hart --hart-module FILE.bc [options]");
    // clang-format off
    ap.arg("--hart", &enabled);
    ap.arg("--hart-module %s:FILE", &inputs[0].filename);
    ap.arg("--hart-callable-module %s:FILE")
      .action([&](cspan<const char*> args) {
          inputs[1].filename = args[1];
          has_callables = true;
      });
    ap.arg("--hart-entry %s:NAME", &entry);
    ap.arg("--hart-device %d:INDEX", &device);
    ap.arg("--hart-no-cache", &no_cache);
    ap.arg("--res %d:WIDTH %d:HEIGHT", &width, &height);
    ap.arg("-g %d:WIDTH %d:HEIGHT", &width, &height);
    ap.arg("--iters %d:COUNT", &iterations);
    ap.arg("--warmup", &warmup);
    ap.arg("--print", &print_pixels);
    ap.arg("-v", &verbose);
    ap.arg("--debug", &verbose);
    ap.arg("-o %L:VARIABLE %L:FILE", &output_names, &output_files);
    ap.arg("filename")
      .action([&](cspan<const char*>) { has_shader = true; });
    // clang-format on
    if (ap.parse_args(argc, argv) < 0) {
        err.errorfmt("HART mode: {}", ap.geterror());
        return EXIT_FAILURE;
    }
    if (has_shader) {
        err.errorfmt("HART mode accepts external bitcode, not OSL shaders; "
                     "use --hart-module FILE.bc");
        return EXIT_FAILURE;
    }
    if (inputs[0].filename.empty() || entry.empty()) {
        err.errorfmt("HART mode requires --hart-module FILE.bc and a nonempty "
                     "entry name");
        return EXIT_FAILURE;
    }
    if (has_callables && inputs[1].filename.empty()) {
        err.errorfmt("--hart-callable-module requires a nonempty filename");
        return EXIT_FAILURE;
    }
    if (device < 0 || width <= 0 || height <= 0 || iterations <= 0) {
        err.errorfmt("HART device must be nonnegative; grid dimensions and "
                     "iteration count must be positive");
        return EXIT_FAILURE;
    }
    if (output_names.size() > 1
        || (!output_names.empty() && output_names[0] != "Cout")) {
        err.errorfmt(
            "The external HART grid ABI supports one RGB output: Cout");
        return EXIT_FAILURE;
    }
    if (size_t(width) > std::vector<float>().max_size() / 3 / size_t(height)) {
        err.errorfmt(
            "HART grid dimensions exceed the output buffer size limit");
        return EXIT_FAILURE;
    }
    if (has_callables
        && size_t(width)
               > size_t(std::numeric_limits<int>::max()) / size_t(height)) {
        err.errorfmt("HART callable grid exceeds the int shade-index range");
        return EXIT_FAILURE;
    }

    if (verbose)
        err.verbosity(ErrorHandler::VERBOSE);
    span<HartModuleInput> modules(inputs.data(), has_callables ? 2 : 1);
    for (auto& input : modules) {
        if (!read_module(input, err))
            return EXIT_FAILURE;
    }
    HartGridRenderer renderer(err);
    if (!renderer.initialize(device, verbose, modules, no_cache)
        || !renderer.load(modules, entry))
        return EXIT_FAILURE;
    std::vector<float> pixels(size_t(width) * size_t(height) * 3);
    const bool rendered = renderer.render(width, height, iterations, warmup,
                                          pixels);
    const bool cleared  = renderer.clear();
    if (!rendered || !cleared)
        return EXIT_FAILURE;

    if (print_pixels) {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const float* pixel = pixels.data()
                                     + (size_t(y) * size_t(width) + size_t(x))
                                           * 3;
                print("Pixel ({}, {}): Cout = {:.9g} {:.9g} {:.9g}\n", x, y,
                      pixel[0], pixel[1], pixel[2]);
            }
    }
    if (!print_pixels && !output_files.empty() && output_files[0] != "null") {
        OIIO::ImageBuf image(
            OIIO::ImageSpec(width, height, 3, TypeDesc::FLOAT));
#if OIIO_VERSION_GREATER_EQUAL(3, 1, 0)
        const bool copied = image.set_pixels(OIIO::ROI::All(),
                                             OIIO::make_cspan(pixels));
#else
        const bool copied = image.set_pixels(OIIO::ROI::All(), TypeDesc::FLOAT,
                                             pixels.data());
#endif
        if (!copied || !image.write(output_files[0])) {
            err.errorfmt("Cannot write HART output '{}': {}", output_files[0],
                         image.geterror());
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

OSL_NAMESPACE_END
