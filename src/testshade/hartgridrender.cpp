// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HIP must precede OIIO: OIIO's __has_attribute fallback selects HIP's
// GNU vector implementation on MSVC.
#include <hip/hip_runtime_api.h>

// Keep these aliases out of headers shared with the real OptiX backend.
#define HART_OPTIX_COMPAT_DEFINE_FUNCTION_TABLE
#include <amd/hart/hart_optix_compat.h>

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/strutil.h>

#include "hartgridparams.h"
#include "hartgridrender.h"

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

bool
read_module_architecture(cspan<char> bitcode, string_view filename,
                         std::string& arch, ErrorHandler& err)
{
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
    return true;
}



class HartGridRenderer {
public:
    explicit HartGridRenderer(ErrorHandler& err) : m_err(err) { }
    ~HartGridRenderer() { clear(); }
    HartGridRenderer(const HartGridRenderer&)            = delete;
    HartGridRenderer& operator=(const HartGridRenderer&) = delete;

    bool initialize(int device, bool verbose, string_view module_arch,
                    string_view filename)
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
        if (!module_arch.empty() && module_arch != device_arch) {
            m_err.errorfmt(
                "HART bitcode '{}' targets '{}', but HIP device {} ({}) is '{}'. "
                "Use bitcode compiled for '{}'; refusing to create a HART pipeline.",
                filename, module_arch, device, properties.name, device_arch,
                device_arch);
            return false;
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
        return hart_check(optixDeviceContextCreate(nullptr, &options,
                                                   &m_context),
                          "hartDeviceContextCreate")
               && hip_check(hipStreamCreate(&m_stream), "hipStreamCreate");
    }

    bool load(cspan<char> bitcode, const std::string& entry)
    {
        if (m_verbose)
            m_err.infofmt("Loading HART module ({} bytes), entry '{}'",
                          bitcode.size(), entry);
        OptixModuleCompileOptions module_options { };
        OptixPipelineCompileOptions compile_options { };
        compile_options.pipelineLaunchParamsVariableName
            = "testshade_hart_params";
        std::array<char, 8192> log { };
        size_t log_size = log.size();
        if (!hart_check(optixModuleCreate(m_context, &module_options,
                                          &compile_options, bitcode.data(),
                                          bitcode.size(), log.data(), &log_size,
                                          &m_module),
                        "hartModuleCreate")) {
            log.back() = '\0';
            m_err.errorfmt("HART module: {}", log.data());
            return false;
        }

        OptixProgramGroupDesc desc { };
        desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module            = m_module;
        desc.raygen.entryFunctionName = entry.c_str();
        log_size                      = log.size();
        if (!hart_check(optixProgramGroupCreate(m_context, &desc, 1, nullptr,
                                                log.data(), &log_size, &m_group),
                        "hartProgramGroupCreate")) {
            log.back() = '\0';
            m_err.errorfmt("HART entry '{}': {}", entry, log.data());
            return false;
        }

        OptixPipelineLinkOptions link_options { };
        link_options.maxTraceDepth = 0;
        log_size                   = log.size();
        if (m_verbose)
            m_err.infofmt("Compiling HART pipeline");
        if (!hart_check(optixPipelineCreate(m_context, &compile_options,
                                            &link_options, &m_group, 1,
                                            log.data(), &log_size, &m_pipeline),
                        "hartPipelineCreate")) {
            log.back() = '\0';
            m_err.errorfmt("HART pipeline for entry '{}': {}", entry,
                           log.data());
            return false;
        }

        alignas(OPTIX_SBT_RECORD_ALIGNMENT)
            std::array<unsigned char, OPTIX_SBT_RECORD_HEADER_SIZE>
                record { };
        return hart_check(optixSbtRecordPackHeader(m_group, record.data()),
                          "hartSbtRecordPackHeader")
               && hip_check(hipMalloc(&m_record, record.size()),
                            "hipMalloc SBT")
               && hip_check(hipMemcpy(m_record, record.data(), record.size(),
                                      hipMemcpyHostToDevice),
                            "hipMemcpy SBT");
    }

    bool render(int width, int height, int iterations, bool warmup,
                span<float> pixels)
    {
        const size_t bytes = pixels.size() * sizeof(float);
        if (!hip_check(hipMalloc(&m_output, bytes), "hipMalloc output")
            || !hip_check(hipMalloc(&m_params,
                                    sizeof(testshade::HartGridParams)),
                          "hipMalloc parameters"))
            return false;
        const testshade::HartGridParams params { static_cast<float*>(m_output) };
        // Unwritten pixels remain NaNs rather than looking like valid black.
        if (!hip_check(hipMemset(m_output, 0xff, bytes), "hipMemset output")
            || !hip_check(hipMemcpy(m_params, &params, sizeof(params),
                                    hipMemcpyHostToDevice),
                          "hipMemcpy parameters"))
            return false;

        OptixShaderBindingTable sbt { };
        sbt.raygenRecord = m_record;
        auto launch      = [&]() {
            if (m_verbose)
                m_err.infofmt("Launching HART grid {} x {}", width, height);
            if (!hart_check(optixLaunch(m_pipeline, m_stream, m_params,
                                        sizeof(params), &sbt, width, height, 1),
                            "hartLaunch"))
                return false;
            if (m_verbose)
                m_err.infofmt("Waiting for HART grid completion");
            return hip_check(hipStreamSynchronize(m_stream),
                             "hipStreamSynchronize");
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
        for (void** buffer : { &m_params, &m_output, &m_record }) {
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
        if (m_group) {
            ok      = hart_check(optixProgramGroupDestroy(m_group),
                                 "hartProgramGroupDestroy")
                      && ok;
            m_group = nullptr;
        }
        if (m_module) {
            ok = hart_check(optixModuleDestroy(m_module), "hartModuleDestroy")
                 && ok;
            m_module = nullptr;
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
    OptixModule m_module         = nullptr;
    OptixProgramGroup m_group    = nullptr;
    OptixPipeline m_pipeline     = nullptr;
    void* m_record               = nullptr;
    void* m_params               = nullptr;
    void* m_output               = nullptr;
};

}  // namespace



int
testshade_hart(int argc, const char* argv[])
{
    ErrorHandler err;
    std::string module, entry = "__raygen__testshade";
    std::vector<std::string> output_names, output_files;
    int device = 0, width = 1, height = 1, iterations = 1;
    bool enabled = false, print_pixels = false, verbose = false, warmup = false;
    bool has_shader = false;
    OIIO::ArgParse ap;
    ap.exit_on_error(false);
    ap.intro("testshade --hart: external AMDGPU bitcode grid runner");
    ap.usage("testshade --hart --hart-module FILE.bc [options]");
    // clang-format off
    ap.arg("--hart", &enabled);
    ap.arg("--hart-module %s:FILE", &module);
    ap.arg("--hart-entry %s:NAME", &entry);
    ap.arg("--hart-device %d:INDEX", &device);
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
    if (module.empty() || entry.empty()) {
        err.errorfmt("HART mode requires --hart-module FILE.bc and a nonempty "
                     "entry name");
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

    const auto size = OIIO::Filesystem::file_size(module);
    if (!size || size > std::string().max_size()) {
        err.errorfmt("Cannot read HART bitcode file '{}' (missing, empty, or "
                     "too large)",
                     module);
        return EXIT_FAILURE;
    }
    std::string bitcode(size_t(size), '\0');
    if (OIIO::Filesystem::read_bytes(module, bitcode.data(), bitcode.size())
        != bitcode.size()) {
        err.errorfmt("Cannot read complete HART bitcode file '{}'", module);
        return EXIT_FAILURE;
    }
    if (bitcode.compare(0, 4, "BC\xc0\xde", 4) != 0
        && bitcode.compare(0, 4, "\xde\xc0\x17\x0b", 4) != 0) {
        err.errorfmt("'{}' is not LLVM bitcode. Assemble textual LLVM IR with "
                     "llvm-as first; PTX and HIP code objects are not HART "
                     "module inputs.",
                     module);
        return EXIT_FAILURE;
    }

    if (verbose)
        err.verbosity(ErrorHandler::VERBOSE);
    std::string module_arch;
    if (!read_module_architecture({ bitcode.data(), bitcode.size() }, module,
                                  module_arch, err))
        return EXIT_FAILURE;
    HartGridRenderer renderer(err);
    if (!renderer.initialize(device, verbose, module_arch, module)
        || !renderer.load({ bitcode.data(), bitcode.size() }, entry))
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
