// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// HIP must precede OIIO's __has_attribute fallback on MSVC.
#include <hip/hip_runtime_api.h>

#include <amd/hart/hart.h>
#include <amd/hart/hart_function_table_definition.h>
#include <amd/hart/hart_stack_size.h>
#include <amd/hart/hart_stubs.h>

#include "hartcontext.h"
#include "hartparams.h"

#include <OpenImageIO/timer.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

OSL_NAMESPACE_BEGIN
namespace {

struct alignas(HART_SBT_RECORD_ALIGNMENT) EmptyRecord {
    std::array<unsigned char, HART_SBT_RECORD_HEADER_SIZE> header { };
};



struct alignas(HART_SBT_RECORD_ALIGNMENT) HitRecord {
    std::array<unsigned char, HART_SBT_RECORD_HEADER_SIZE> header { };
    HartHitRecordData data { };
};

static_assert(sizeof(Vec3) == 3 * sizeof(float),
              "HART vertices require packed float3");
static_assert(sizeof(unsigned) == 4
                  && sizeof(HartTriangle) == 3 * sizeof(unsigned)
                  && offsetof(HartTriangle, a) == 0
                  && offsetof(HartTriangle, b) == sizeof(unsigned)
                  && offsetof(HartTriangle, c) == 2 * sizeof(unsigned),
              "HART indices require packed unsigned int3");
static_assert(offsetof(HitRecord, data) == HART_SBT_RECORD_HEADER_SIZE,
              "HART hit data must immediately follow the SBT header");
static_assert(sizeof(EmptyRecord) % HART_SBT_RECORD_ALIGNMENT == 0
                  && sizeof(HitRecord) % HART_SBT_RECORD_ALIGNMENT == 0,
              "HART SBT strides must preserve record alignment");



template<typename T>
cspan<unsigned char>
as_bytes(cspan<T> source)
{
    return { reinterpret_cast<const unsigned char*>(source.data()),
             source.size() * sizeof(T) };
}



bool
is_bitcode(cspan<unsigned char> bytes)
{
    return bytes.data() && bytes.size() >= 4
           && (std::memcmp(bytes.data(), "BC\xc0\xde", 4) == 0
               || std::memcmp(bytes.data(), "\xde\xc0\x17\x0b", 4) == 0);
}

}  // namespace



struct HartContext::Impl {
    explicit Impl(ErrorHandler& err) : m_err(err) { }

    bool hip_check(hipError_t result, string_view operation)
    {
        if (result == hipSuccess)
            return true;
        m_err.errorfmt("{} failed: {} ({})", operation, hipGetErrorName(result),
                       hipGetErrorString(result));
        return false;
    }

    bool hart_check(HartResult result, string_view operation)
    {
        if (result == HART_SUCCESS)
            return true;
        m_err.errorfmt("{} failed: {} ({})", operation,
                       hartGetErrorName(result), hartGetErrorString(result));
        return false;
    }

    bool compile_check(HartResult result, string_view operation, span<char> log,
                       size_t log_size)
    {
        if (hart_check(result, operation))
            return true;
        log.back() = '\0';
        m_err.errorfmt("{} compiler log{}: {}", operation,
                       log_size > log.size() ? " (truncated)" : "",
                       log[0] ? log.data() : "(not supplied by HART)");
        return false;
    }

    bool ready()
    {
        if (!m_initialized) {
            m_err.errorfmt("HART context is not initialized; call init first");
            return false;
        }
        return hip_check(hipSetDevice(m_device), "hipSetDevice");
    }

    bool buffer_size(size_t count, size_t stride)
    {
        if (count <= std::numeric_limits<size_t>::max() / stride)
            return true;
        m_err.errorfmt("HART buffer size overflows: {} elements of {} bytes",
                       count, stride);
        return false;
    }

    bool owns(const void* pointer, size_t bytes)
    {
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        for (const auto& allocation : m_allocations) {
            if (!allocation.pointer)
                continue;
            const auto base = reinterpret_cast<uintptr_t>(allocation.pointer);
            if (address >= base && address - base <= allocation.bytes
                && bytes <= allocation.bytes - (address - base))
                return true;
        }
        m_err.errorfmt("HART transfer exceeds an owned device allocation "
                       "({} bytes)",
                       bytes);
        return false;
    }

    static void log_callback(unsigned level, const char* tag,
                             const char* message, void* data)
    {
        auto& err = *static_cast<ErrorHandler*>(data);
        if (level <= 2)
            err.errorfmt("HART [{}]: {}", tag ? tag : "",
                         message ? message : "");
        else
            err.infofmt("HART [{}]: {}", tag ? tag : "",
                        message ? message : "");
    }

    struct Allocation {
        void* pointer;
        size_t bytes;
    };

    ErrorHandler& m_err;
    int m_device              = -1;
    bool m_initialized        = false;
    bool m_scene_ready        = false;
    bool m_pipeline_ready     = false;
    bool m_collect_statistics = false;
    Statistics m_statistics;
    unsigned m_scene_material_count    = 0;
    unsigned m_pipeline_material_count = 0;
    unsigned m_max_primitives          = 0;
    unsigned m_max_sbt_records         = 0;
    HartDeviceContext m_context        = nullptr;
    hipStream_t m_stream               = nullptr;
    HartModule m_module                = nullptr;
    std::vector<HartModule> m_callable_modules;
    std::vector<HartProgramGroup> m_groups;
    HartPipeline m_pipeline = nullptr;
    HartShaderBindingTable m_sbt { };
    void* m_secondary_raygen            = nullptr;
    HartTraversableHandle m_traversable = 0;
    void* m_params                      = nullptr;
    size_t m_param_capacity             = 0;
    std::vector<Allocation> m_allocations;
};



HartContext::HartContext(ErrorHandler& err)
    : m_impl(std::make_unique<Impl>(err))
{
}



HartContext::~HartContext()
{
    if (!clear())
        m_impl->m_err.errorfmt("HART context cleanup incomplete; resources "
                               "could not be safely released");
}



bool
HartContext::init(int device, std::string& arch, bool cache_enabled,
                  bool statistics)
{
    auto& ctx = *m_impl;
    arch.clear();
    if (ctx.m_device != -1) {
        ctx.m_err.errorfmt("Clear the HART context before initializing again");
        return false;
    }
    if (device < 0) {
        ctx.m_err.errorfmt("HART device ordinal must be nonnegative");
        return false;
    }
    hipDeviceProp_t properties { };
    if (!ctx.hip_check(hipSetDevice(device), "hipSetDevice")
        || !ctx.hip_check(hipGetDeviceProperties(&properties, device),
                          "hipGetDeviceProperties"))
        return false;
    const string_view target(properties.gcnArchName);
    const string_view selected_arch = target.substr(0, target.find(':'));
    if (selected_arch != "gfx1201" && selected_arch != "gfx1100"
        && selected_arch != "gfx1151") {
        ctx.m_err.errorfmt("Unsupported HART device {} architecture '{}'; "
                           "expected gfx1201, gfx1100, or gfx1151",
                           device, selected_arch);
        return false;
    }
    if (!ctx.hip_check(hipFree(nullptr), "HIP context initialization"))
        return false;
    // Link the native runtime rather than loading a second library instance.
    // The SDK validates the function-table ABI, once for all contexts.
    static const HartResult runtime_result
        = hartQueryFunctionTable(HART_ABI_VERSION, 0, nullptr, nullptr,
                                 &HART_FUNCTION_TABLE_SYMBOL,
                                 sizeof(HART_FUNCTION_TABLE_SYMBOL));
    if (!ctx.hart_check(runtime_result, "hartQueryFunctionTable"))
        return false;
    ctx.m_device = device;
    HartDeviceContextOptions options { };
    options.logCallbackFunction = Impl::log_callback;
    options.logCallbackData     = &ctx.m_err;
    options.logCallbackLevel    = 2;
    // Like the grid runner, use the supported context defaults. The SDK
    // rejects ALL; module provenance and pipeline stack checks still apply.
    if (!ctx.hart_check(hartDeviceContextCreate(nullptr, &options,
                                                &ctx.m_context),
                        "hartDeviceContextCreate")
        || !ctx.hart_check(hartDeviceContextGetProperty(
                               ctx.m_context,
                               HART_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
                               &ctx.m_max_primitives,
                               sizeof(ctx.m_max_primitives)),
                           "hartDeviceContextGetProperty max primitives")
        || !ctx.hart_check(
            hartDeviceContextGetProperty(
                ctx.m_context,
                HART_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,
                &ctx.m_max_sbt_records, sizeof(ctx.m_max_sbt_records)),
            "hartDeviceContextGetProperty max SBT records")
        || !ctx.hip_check(hipStreamCreate(&ctx.m_stream), "hipStreamCreate"))
        return false;
    if (!ctx.hart_check(hartDeviceContextSetCacheEnabled(ctx.m_context,
                                                         cache_enabled),
                        "hartDeviceContextSetCacheEnabled"))
        return false;
    ctx.m_collect_statistics = statistics;
    ctx.m_initialized        = true;
    arch.assign(selected_arch.data(), selected_arch.size());
    return true;
}



void*
HartContext::alloc(size_t bytes)
{
    auto& ctx = *m_impl;
    if (!bytes) {
        ctx.m_err.errorfmt("Cannot allocate a zero-byte HART device buffer");
        return nullptr;
    }
    if (!ctx.ready())
        return nullptr;
    ctx.m_allocations.push_back({ nullptr, bytes });
    auto& allocation = ctx.m_allocations.back();
    if (!ctx.hip_check(hipMalloc(&allocation.pointer, bytes), "hipMalloc")) {
        ctx.m_allocations.pop_back();
        return nullptr;
    }
    return allocation.pointer;
}



bool
HartContext::make_current()
{ return m_impl->ready(); }



bool
HartContext::upload(void* destination, cspan<unsigned char> source)
{
    auto& ctx = *m_impl;
    if (!ctx.ready())
        return false;
    if (source.empty())
        return true;
    if (!source.data() || !destination) {
        ctx.m_err.errorfmt("HART upload requires nonnull source and "
                           "destination");
        return false;
    }
    return ctx.owns(destination, source.size())
           && ctx.hip_check(hipMemcpy(destination, source.data(), source.size(),
                                      hipMemcpyHostToDevice),
                            "hipMemcpy upload");
}



bool
HartContext::download(span<unsigned char> destination, const void* source)
{
    auto& ctx = *m_impl;
    if (!ctx.ready())
        return false;
    if (destination.empty())
        return true;
    if (!destination.data() || !source) {
        ctx.m_err.errorfmt("HART download requires nonnull source and "
                           "destination");
        return false;
    }
    return ctx.owns(source, destination.size())
           && ctx.hip_check(hipStreamSynchronize(ctx.m_stream),
                            "hipStreamSynchronize before download")
           && ctx.hip_check(hipMemcpy(destination.data(), source,
                                      destination.size(),
                                      hipMemcpyDeviceToHost),
                            "hipMemcpy download");
}



bool
HartContext::build_accel(cspan<Vec3> vertices, cspan<HartTriangle> triangles,
                         cspan<unsigned> material_ids, unsigned material_count)
{
    auto& ctx = *m_impl;
    if (!ctx.ready())
        return false;
    if ((!vertices.empty() && !vertices.data())
        || (!triangles.empty() && !triangles.data())
        || (!material_ids.empty() && !material_ids.data())
        || material_ids.size() != triangles.size()
        || vertices.size() > std::numeric_limits<unsigned>::max()
        || triangles.size() > ctx.m_max_primitives
        || material_count > ctx.m_max_sbt_records
        || (!triangles.empty() && (!material_count || vertices.empty()))) {
        ctx.m_err.errorfmt("Invalid HART geometry counts, buffers, or "
                           "material IDs");
        return false;
    }
    if (!ctx.buffer_size(vertices.size(), sizeof(Vec3))
        || !ctx.buffer_size(triangles.size(), sizeof(HartTriangle))
        || !ctx.buffer_size(material_ids.size(), sizeof(unsigned)))
        return false;
    if (ctx.m_pipeline_ready
        && material_count != ctx.m_pipeline_material_count) {
        ctx.m_err.errorfmt("HART acceleration and pipeline material counts "
                           "must match");
        return false;
    }
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto& v = vertices[i];
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            ctx.m_err.errorfmt("HART vertex {} is not finite", i);
            return false;
        }
    }
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& t = triangles[i];
        if (t.a >= vertices.size() || t.b >= vertices.size()
            || t.c >= vertices.size()) {
            ctx.m_err.errorfmt("HART triangle {} has an out-of-range "
                               "vertex index",
                               i);
            return false;
        }
        if (material_ids[i] >= material_count) {
            ctx.m_err.errorfmt("HART triangle {} has an out-of-range "
                               "material ID",
                               i);
            return false;
        }
    }
    ctx.m_scene_ready = false;
    ctx.m_traversable = 0;
    if (triangles.empty()) {
        ctx.m_scene_material_count = material_count;
        ctx.m_scene_ready          = true;
        return true;
    }
    auto copy = [&](cspan<unsigned char> bytes) -> void* {
        void* pointer = alloc(bytes.size());
        return pointer && upload(pointer, bytes) ? pointer : nullptr;
    };
    hipDeviceptr_t device_vertices  = copy(as_bytes(vertices));
    hipDeviceptr_t device_triangles = copy(as_bytes(triangles));
    hipDeviceptr_t device_materials = copy(as_bytes(material_ids));
    if (!device_vertices || !device_triangles || !device_materials)
        return false;
    std::vector<unsigned> flags(material_count,
                                HART_GEOMETRY_FLAG_DISABLE_ANYHIT);
    HartBuildInput input { };
    input.type                       = HART_BUILD_INPUT_TYPE_TRIANGLES;
    auto& mesh                       = input.triangleArray;
    mesh.vertexBuffers               = &device_vertices;
    mesh.numVertices                 = static_cast<unsigned>(vertices.size());
    mesh.vertexFormat                = HART_VERTEX_FORMAT_FLOAT3;
    mesh.vertexStrideInBytes         = sizeof(Vec3);
    mesh.indexBuffer                 = device_triangles;
    mesh.numIndexTriplets            = static_cast<unsigned>(triangles.size());
    mesh.indexFormat                 = HART_INDICES_FORMAT_UNSIGNED_INT3;
    mesh.indexStrideInBytes          = sizeof(HartTriangle);
    mesh.flags                       = flags.data();
    mesh.numSbtRecords               = material_count;
    mesh.sbtIndexOffsetBuffer        = device_materials;
    mesh.sbtIndexOffsetSizeInBytes   = sizeof(unsigned);
    mesh.sbtIndexOffsetStrideInBytes = sizeof(unsigned);
    mesh.transformFormat             = HART_TRANSFORM_FORMAT_NONE;
    HartAccelBuildOptions options { };
    options.buildFlags = HART_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation  = HART_BUILD_OPERATION_BUILD;
    HartAccelBufferSizes sizes { };
    if (!ctx.hart_check(hartAccelComputeMemoryUsage(ctx.m_context, &options,
                                                    &input, 1, &sizes),
                        "hartAccelComputeMemoryUsage"))
        return false;
    if (!sizes.outputSizeInBytes) {
        ctx.m_err.errorfmt("HART returned empty triangle acceleration storage");
        return false;
    }
    void* output  = alloc(sizes.outputSizeInBytes);
    void* scratch = sizes.tempSizeInBytes ? alloc(sizes.tempSizeInBytes)
                                          : nullptr;
    if (!output || (sizes.tempSizeInBytes && !scratch))
        return false;
    HartTraversableHandle handle = 0;
    const bool built             = ctx.hart_check(
        hartAccelBuild(ctx.m_context, ctx.m_stream, &options, &input, 1,
                       scratch, sizes.tempSizeInBytes, output,
                       sizes.outputSizeInBytes, &handle, nullptr, 0),
        "hartAccelBuild");
    const bool synchronized = ctx.hip_check(hipStreamSynchronize(ctx.m_stream),
                                            "hipStreamSynchronize after build");
    if (!built || !synchronized)
        return false;
    if (!handle) {
        ctx.m_err.errorfmt("HART returned a null nonempty acceleration handle");
        return false;
    }
    ctx.m_traversable          = handle;
    ctx.m_scene_material_count = material_count;
    ctx.m_scene_ready          = true;
    return true;
}



uint64_t
HartContext::traversable() const
{ return m_impl->m_traversable; }



bool
HartContext::create_pipeline(cspan<unsigned char> bitcode,
                             string_view raygen_entry, unsigned material_count,
                             cspan<HartCallable> callables,
                             string_view secondary_raygen_entry)
{
    auto& ctx = *m_impl;
    OIIO::Timer pipeline_timer(ctx.m_collect_statistics);
    if (!ctx.ready())
        return false;
    if (ctx.m_module || ctx.m_pipeline || !ctx.m_groups.empty()) {
        ctx.m_err.errorfmt("Clear the HART context before replacing "
                           "a pipeline");
        return false;
    }
    auto valid_entry = [](string_view entry) {
        return entry.data() && entry.size() > 10
               && entry.find('\0') == string_view::npos
               && entry.substr(0, 10) == "__raygen__";
    };
    const bool secondary      = !secondary_raygen_entry.empty();
    const size_t fixed_groups = secondary ? 4 : 3;
    if (!is_bitcode(bitcode) || !valid_entry(raygen_entry)
        || (secondary
            && (!valid_entry(secondary_raygen_entry)
                || secondary_raygen_entry == raygen_entry))
        || material_count > ctx.m_max_sbt_records
        || (!callables.empty() && !callables.data())) {
        ctx.m_err.errorfmt("HART pipeline requires LLVM bitcode, a raygen "
                           "export, and a valid material count");
        return false;
    }
    if (!ctx.buffer_size(material_count, sizeof(HitRecord)))
        return false;
    if (ctx.m_scene_ready && material_count != ctx.m_scene_material_count) {
        ctx.m_err.errorfmt("HART pipeline and acceleration material counts "
                           "must match");
        return false;
    }
    size_t callable_count = 0;
    for (const auto& callable : callables) {
        if (!is_bitcode(callable.bitcode) || callable.entries.empty()
            || callable.entries.size() > std::numeric_limits<unsigned>::max()
                                             - fixed_groups - callable_count) {
            ctx.m_err.errorfmt("Invalid HART callable module or export count");
            return false;
        }
        for (const auto& name : callable.entries) {
            if (name.size() <= 19
                || name.compare(0, 19, "__direct_callable__") != 0
                || name.find('\0') != std::string::npos) {
                ctx.m_err.errorfmt("Invalid HART direct-callable export '{}'",
                                   name);
                return false;
            }
        }
        callable_count += callable.entries.size();
    }
    if (!ctx.buffer_size(callable_count, sizeof(EmptyRecord)))
        return false;
    ctx.m_groups.resize(fixed_groups + callable_count, nullptr);
    ctx.m_callable_modules.resize(callables.size(), nullptr);
    const std::string entry(raygen_entry.data(), raygen_entry.size());
    const std::string secondary_entry
        = secondary ? std::string(secondary_raygen_entry.data(),
                                  secondary_raygen_entry.size())
                    : std::string();
    HartPipelineCompileOptions options { };
    options.traversableGraphFlags = HART_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    options.numPayloadValues   = 5;
    options.numAttributeValues = 2;
    options.exceptionFlags     = HART_EXCEPTION_FLAG_STACK_OVERFLOW
                                 | HART_EXCEPTION_FLAG_TRACE_DEPTH;
    options.pipelineLaunchParamsVariableName = "osl_hart_render_params";
    options.usesPrimitiveTypeFlags = HART_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    HartModuleCompileOptions module_options { };
    module_options.optLevel   = HART_COMPILE_OPTIMIZATION_LEVEL_3;
    module_options.debugLevel = HART_COMPILE_DEBUG_LEVEL_NONE;
    std::array<char, 8192> log { };
    size_t log_size = log.size();
    HartResult result
        = hartModuleCreate(ctx.m_context, &module_options, &options,
                           reinterpret_cast<const char*>(bitcode.data()),
                           bitcode.size(), log.data(), &log_size,
                           &ctx.m_module);
    if (!ctx.compile_check(result, "hartModuleCreate", log, log_size))
        return false;
    std::array<HartProgramGroupDesc, 4> descriptions { };
    descriptions[0].kind                     = HART_PROGRAM_GROUP_KIND_RAYGEN;
    descriptions[0].raygen.module            = ctx.m_module;
    descriptions[0].raygen.entryFunctionName = entry.c_str();
    descriptions[1].kind                     = HART_PROGRAM_GROUP_KIND_MISS;
    descriptions[1].miss.module              = ctx.m_module;
    descriptions[1].miss.entryFunctionName   = "__miss__osl_hart";
    descriptions[2].kind                     = HART_PROGRAM_GROUP_KIND_HITGROUP;
    descriptions[2].hitgroup.moduleCH        = ctx.m_module;
    descriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__osl_hart";
    if (secondary) {
        descriptions[3].kind          = HART_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[3].raygen.module = ctx.m_module;
        descriptions[3].raygen.entryFunctionName = secondary_entry.c_str();
    }
    for (size_t i = 0; i < fixed_groups; ++i) {
        log.fill(0);
        log_size = log.size();
        result   = hartProgramGroupCreate(ctx.m_context, &descriptions[i], 1,
                                          nullptr, log.data(), &log_size,
                                          &ctx.m_groups[i]);
        if (!ctx.compile_check(result, "hartProgramGroupCreate", log, log_size))
            return false;
    }
    size_t group_index = fixed_groups;
    for (size_t i = 0; i < callables.size(); ++i) {
        const auto& callable = callables[i];
        log.fill(0);
        log_size = log.size();
        result   = hartModuleCreate(ctx.m_context, &module_options, &options,
                                    reinterpret_cast<const char*>(
                                        callable.bitcode.data()),
                                    callable.bitcode.size(), log.data(),
                                    &log_size, &ctx.m_callable_modules[i]);
        if (!ctx.compile_check(result, "hartModuleCreate callable", log,
                               log_size))
            return false;
        for (const auto& name : callable.entries) {
            HartProgramGroupDesc description { };
            description.kind               = HART_PROGRAM_GROUP_KIND_CALLABLES;
            description.callables.moduleDC = ctx.m_callable_modules[i];
            description.callables.entryFunctionNameDC = name.c_str();
            log.fill(0);
            log_size = log.size();
            result   = hartProgramGroupCreate(ctx.m_context, &description, 1,
                                              nullptr, log.data(), &log_size,
                                              &ctx.m_groups[group_index++]);
            if (!ctx.compile_check(result, "hartProgramGroupCreate callable",
                                   log, log_size)) {
                ctx.m_err.errorfmt("Cannot create HART callable '{}'", name);
                return false;
            }
        }
    }
    HartPipelineLinkOptions link_options { };
    link_options.maxTraceDepth = 1;
    log.fill(0);
    log_size = log.size();
    result   = hartPipelineCreate(ctx.m_context, &options, &link_options,
                                  ctx.m_groups.data(),
                                  static_cast<unsigned>(ctx.m_groups.size()),
                                  log.data(), &log_size, &ctx.m_pipeline);
    if (!ctx.compile_check(result, "hartPipelineCreate", log, log_size))
        return false;
    HartStackSizes stack { };
    for (auto group : ctx.m_groups)
        if (!ctx.hart_check(hartUtilAccumulateStackSizes(group, &stack,
                                                         ctx.m_pipeline),
                            "hartUtilAccumulateStackSizes"))
            return false;
    unsigned traversal_stack = 0, state_stack = 0, continuation_stack = 0;
    if (!ctx.hart_check(
            hartUtilComputeStackSizes(&stack, link_options.maxTraceDepth, 0,
                                      callable_count ? 1 : 0, &traversal_stack,
                                      &state_stack, &continuation_stack),
            "hartUtilComputeStackSizes")
        || !ctx.hart_check(hartPipelineSetStackSize(ctx.m_pipeline,
                                                    traversal_stack, state_stack,
                                                    continuation_stack, 1),
                           "hartPipelineSetStackSize"))
        return false;
    std::vector<EmptyRecord> records(secondary ? 3 : 2);
    for (size_t i = 0; i < records.size(); ++i)
        if (!ctx.hart_check(hartSbtRecordPackHeader(ctx.m_groups[i == 2 ? 3 : i],
                                                    records[i].header.data()),
                            "hartSbtRecordPackHeader"))
            return false;
    std::vector<HitRecord> hit_records(material_count);
    for (unsigned i = 0; i < material_count; ++i) {
        hit_records[i].data.material = i;
        if (!ctx.hart_check(hartSbtRecordPackHeader(
                                ctx.m_groups[2], hit_records[i].header.data()),
                            "hartSbtRecordPackHeader hit"))
            return false;
    }
    auto* device_records = static_cast<unsigned char*>(
        alloc(records.size() * sizeof(EmptyRecord)));
    if (!device_records
        || !upload(device_records, as_bytes(cspan<EmptyRecord>(records))))
        return false;
    void* device_hits = nullptr;
    if (material_count) {
        device_hits = alloc(hit_records.size() * sizeof(HitRecord));
        if (!device_hits
            || !upload(device_hits, as_bytes(cspan<HitRecord>(hit_records))))
            return false;
    }
    void* device_callables = nullptr;
    if (callable_count) {
        std::vector<EmptyRecord> callable_records(callable_count);
        for (size_t i = 0; i < callable_count; ++i)
            if (!ctx.hart_check(
                    hartSbtRecordPackHeader(ctx.m_groups[fixed_groups + i],
                                            callable_records[i].header.data()),
                    "hartSbtRecordPackHeader callable"))
                return false;
        device_callables = alloc(callable_count * sizeof(EmptyRecord));
        if (!device_callables
            || !upload(device_callables,
                       as_bytes(cspan<EmptyRecord>(callable_records))))
            return false;
    }
    ctx.m_sbt.raygenRecord   = device_records;
    ctx.m_secondary_raygen   = secondary
                                   ? device_records + 2 * sizeof(EmptyRecord)
                                   : nullptr;
    ctx.m_sbt.missRecordBase = device_records + sizeof(EmptyRecord);
    ctx.m_sbt.missRecordStrideInBytes      = sizeof(EmptyRecord);
    ctx.m_sbt.missRecordCount              = 1;
    ctx.m_sbt.hitgroupRecordBase           = device_hits;
    ctx.m_sbt.hitgroupRecordStrideInBytes  = material_count ? sizeof(HitRecord)
                                                            : 0;
    ctx.m_sbt.hitgroupRecordCount          = material_count;
    ctx.m_sbt.callablesRecordBase          = device_callables;
    ctx.m_sbt.callablesRecordStrideInBytes = callable_count
                                                 ? sizeof(EmptyRecord)
                                                 : 0;
    ctx.m_sbt.callablesRecordCount = static_cast<unsigned>(callable_count);
    ctx.m_pipeline_material_count  = material_count;
    ctx.m_pipeline_ready           = true;
    if (ctx.m_collect_statistics) {
        ctx.m_statistics.pipeline_seconds   = pipeline_timer();
        ctx.m_statistics.traversal_stack    = traversal_stack;
        ctx.m_statistics.state_stack        = state_stack;
        ctx.m_statistics.continuation_stack = continuation_stack;
    }
    return true;
}



bool
HartContext::launch(const void* params, size_t param_bytes, unsigned width,
                    unsigned height, unsigned raygen_index)
{
    auto& ctx = *m_impl;
    if (!ctx.ready())
        return false;
    if (!ctx.m_pipeline_ready || !ctx.m_scene_ready) {
        ctx.m_err.errorfmt("HART launch requires a pipeline and built scene");
        return false;
    }
    if (raygen_index > 1 || (raygen_index == 1 && !ctx.m_secondary_raygen)) {
        ctx.m_err.errorfmt("Invalid HART raygen index {}", raygen_index);
        return false;
    }
    if (!params || !param_bytes || !width || !height
        || width > std::numeric_limits<unsigned>::max() / height
        || !ctx.buffer_size(width, height)) {
        ctx.m_err.errorfmt("Invalid HART launch parameters or overflowing "
                           "launch dimensions {} x {}",
                           width, height);
        return false;
    }
    if (param_bytes > ctx.m_param_capacity) {
        void* buffer = alloc(param_bytes);
        if (!buffer)
            return false;
        ctx.m_params         = buffer;
        ctx.m_param_capacity = param_bytes;
    }
    if (!upload(ctx.m_params,
                { static_cast<const unsigned char*>(params), param_bytes }))
        return false;
    auto sbt = ctx.m_sbt;
    if (raygen_index == 1)
        sbt.raygenRecord = ctx.m_secondary_raygen;
    if (ctx.m_collect_statistics
        && !ctx.hip_check(hipStreamSynchronize(ctx.m_stream),
                          "hipStreamSynchronize before launch timing"))
        return false;
    OIIO::Timer launch_timer(ctx.m_collect_statistics);
    const bool launched
        = ctx.hart_check(hartLaunch(ctx.m_pipeline, ctx.m_stream, ctx.m_params,
                                    param_bytes, &sbt, width, height, 1),
                         "hartLaunch");
    const bool synchronized = ctx.hip_check(hipStreamSynchronize(ctx.m_stream),
                                            "hipStreamSynchronize after launch");
    if (launched && synchronized && ctx.m_collect_statistics) {
        ctx.m_statistics.launch_seconds += launch_timer();
        ++ctx.m_statistics.launches;
    }
    return launched && synchronized;
}



bool
HartContext::clear()
{
    auto& ctx = *m_impl;
    if (ctx.m_device < 0)
        return true;
    ctx.m_initialized    = false;
    ctx.m_scene_ready    = false;
    ctx.m_pipeline_ready = false;
    ctx.m_traversable    = 0;
    if (!ctx.hip_check(hipSetDevice(ctx.m_device), "hipSetDevice during cleanup")
        || (ctx.m_stream
            && !ctx.hip_check(hipStreamSynchronize(ctx.m_stream),
                              "hipStreamSynchronize during cleanup")))
        return false;
    if (ctx.m_pipeline) {
        if (!ctx.hart_check(hartPipelineDestroy(ctx.m_pipeline),
                            "hartPipelineDestroy"))
            return false;
        ctx.m_pipeline = nullptr;
    }
    bool ok = true;
    for (auto& group : ctx.m_groups) {
        if (group) {
            if (ctx.hart_check(hartProgramGroupDestroy(group),
                               "hartProgramGroupDestroy"))
                group = nullptr;
            else
                ok = false;
        }
    }
    if (!ok)
        return false;
    ctx.m_groups.clear();
    for (auto& module : ctx.m_callable_modules) {
        if (module) {
            if (ctx.hart_check(hartModuleDestroy(module),
                               "hartModuleDestroy callable"))
                module = nullptr;
            else
                ok = false;
        }
    }
    if (!ok)
        return false;
    ctx.m_callable_modules.clear();
    if (ctx.m_module) {
        if (!ctx.hart_check(hartModuleDestroy(ctx.m_module),
                            "hartModuleDestroy"))
            return false;
        ctx.m_module = nullptr;
    }
    if (ctx.m_context) {
        if (!ctx.hart_check(hartDeviceContextDestroy(ctx.m_context),
                            "hartDeviceContextDestroy"))
            return false;
        ctx.m_context = nullptr;
    }
    for (auto i = ctx.m_allocations.rbegin(); i != ctx.m_allocations.rend();
         ++i) {
        if (i->pointer) {
            if (ctx.hip_check(hipFree(i->pointer), "hipFree"))
                i->pointer = nullptr;
            else
                ok = false;
        }
    }
    if (!ok)
        return false;
    ctx.m_allocations.clear();
    if (ctx.m_stream) {
        if (!ctx.hip_check(hipStreamDestroy(ctx.m_stream), "hipStreamDestroy"))
            return false;
        ctx.m_stream = nullptr;
    }
    ctx.m_sbt                     = { };
    ctx.m_secondary_raygen        = nullptr;
    ctx.m_params                  = nullptr;
    ctx.m_param_capacity          = 0;
    ctx.m_scene_material_count    = 0;
    ctx.m_pipeline_material_count = 0;
    ctx.m_statistics              = { };
    ctx.m_collect_statistics      = false;
    ctx.m_device                  = -1;
    return true;
}



HartContext::Statistics
HartContext::statistics() const
{ return m_impl->m_statistics; }



void
HartContext::reset_launch_statistics()
{
    m_impl->m_statistics.launches       = 0;
    m_impl->m_statistics.launch_seconds = 0;
}



HartContext::ResourceUsage
HartContext::resource_usage() const
{
    const auto& ctx = *m_impl;
    ResourceUsage usage;
    for (const auto& allocation : ctx.m_allocations)
        if (allocation.pointer) {
            ++usage.allocations;
            usage.bytes += allocation.bytes;
        }
    usage.modules = ctx.m_module ? 1 : 0;
    for (const auto module : ctx.m_callable_modules)
        usage.modules += module != nullptr;
    for (const auto group : ctx.m_groups)
        usage.program_groups += group != nullptr;
    usage.context  = ctx.m_context != nullptr;
    usage.stream   = ctx.m_stream != nullptr;
    usage.pipeline = ctx.m_pipeline != nullptr;
    return usage;
}

OSL_NAMESPACE_END
