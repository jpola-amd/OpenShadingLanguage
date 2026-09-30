// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#include <OpenImageIO/unittest.h>

#include <array>
#include <cstring>

#include "../testrender/hartcontext.h"
#include "hart_custom_closure_bitcode.h"
#include "hartcustomclosureparams.h"

using namespace OSL;
using namespace custom_closure_test;

namespace {

unsigned host_allocator_calls = 0;

const char* source = R"OSL(
closure color custom32(int token, float gain, color tint) [[int builtin=1]];
closure color custom64(int token, float gain, color tint) [[int builtin=1]];
shader custom_closure_acceptance()
{
    int row = int(u);
    color tint = color(u+1, u+2, u+3);
    if (row == 0)
        Ci = custom32(row, u+0.5, tint);
    else if (row == 1)
        Ci = color(0.25,0.5,0.75)
             * custom64(row, u+0.5, tint, "keyword", 23.5);
    else if (row == 2)
        Ci = color(u-2)*custom32(row, u+0.5, tint);
    else if (row == 3) {
        closure color pair = custom32(row, u+0.5, tint)
            + custom64(row, u+0.5, tint, "keyword", 29.5);
        Ci = color(u, 2, 0.5)*pair;
    } else if (row == 4)
        Ci = custom64(row, u+0.5, tint, "keyword", 31.5);
    else if (row == 5)
        Ci = color(0)*custom32(row, u+0.5, tint);
    else if (row == 6)
        Ci = color(0,2,0)*custom64(row, u+0.5, tint);
    else if (row == 7)
        Ci = -2*custom32(row, u+0.5, tint, "keyword", 37.5);
    else
        Ci = custom64(row, u+0.5, tint);
}
)OSL";



class Diagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR
            || (code & 0xffff0000) == EH_SEVERE) {
            ++errors;
            last_error = message;
        }
        ErrorHandler::operator()(code, message);
    }
    unsigned errors = 0;
    std::string last_error;
};



class Services final : public RendererServices {
public:
    explicit Services(bool allocator) : m_allocator(allocator) {}

    int supports(string_view feature) const override
    {
        return feature == "HART"
               || (m_allocator && feature == "HARTClosureAllocator")
               || (!m_allocator
                   && (feature == "HARTClosures"
                       || feature == "HARTClosureParameters"));
    }

private:
    bool m_allocator;
};



ClosureComponent*
host_allocator(ShaderGlobals*, int, const Color3*)
{
    ++host_allocator_calls;
    return nullptr;
}



template<class PayloadT>
void
register_closure(ShadingSystem& ss, string_view name, int id)
{
    const ClosureParam params[] = {
        CLOSURE_INT_PARAM(PayloadT, token), CLOSURE_FLOAT_PARAM(PayloadT, gain),
        CLOSURE_COLOR_PARAM(PayloadT, tint),
        CLOSURE_FLOAT_KEYPARAM(PayloadT, keyword, "keyword"),
        CLOSURE_FINISH_PARAM(PayloadT)
    };
    ss.register_closure(name, id, params, host_allocator);
}



ShaderGroupRef
make_group(ShadingSystem& ss, Diagnostics& diagnostics, string_view stdosl,
           string_view arch, int optimize)
{
    if (!ss.attribute("hart_arch", arch) || !ss.attribute("optimize", optimize)
        || !ss.attribute("llvm_optimize", optimize)
        || !ss.attribute("llvm_debugging_symbols", 0)
        || !ss.attribute("profile", 0))
        return {};
    register_closure<Payload32>(ss, "custom32", id32);
    register_closure<Payload64>(ss, "custom64", id64);
    OSLCompiler compiler(&diagnostics);
    std::string oso;
    if (!compiler.compile_buffer(source, oso, {}, stdosl)
        || !ss.LoadMemoryCompiledShader("custom_closure_acceptance", oso))
        return {};
    auto group = ss.ShaderGroupBegin("custom_closure_acceptance");
    if (!group || !ss.Shader("surface", "custom_closure_acceptance", "material")
        || !ss.ShaderGroupEnd())
        return {};
    ss.optimize_group(group.get(), nullptr);
    return group;
}



void
check_leaf(const Leaf& leaf, unsigned row, int id, float keyword,
           const Color3& weight)
{
    OIIO_CHECK_EQUAL(leaf.id, id);
    OIIO_CHECK_EQUAL(leaf.guard, sentinel);
    OIIO_CHECK_EQUAL(leaf.end_guard, tail);
    OIIO_CHECK_EQUAL(leaf.payload_alignment, id == id32 ? 32u : 64u);
    OIIO_CHECK_EQUAL(leaf.payload_remainder, 0u);
    OIIO_CHECK_EQUAL(leaf.component_remainder, 0u);
    OIIO_CHECK_EQUAL(leaf.header_bytes, unsigned(sizeof(ClosureComponent)));
    OIIO_CHECK_EQUAL(leaf.token, int(row));
    OIIO_CHECK_EQUAL(leaf.gain, float(row) + 0.5f);
    OIIO_CHECK_EQUAL(leaf.keyword, keyword);
    for (int c = 0; c < 3; ++c) {
        OIIO_CHECK_EQUAL(leaf.tint[c], float(row + c + 1));
        OIIO_CHECK_EQUAL(leaf.weight[c], weight[c]);
    }
}



bool
run(string_view stdosl, int optimize)
{
    Diagnostics diagnostics;
    HartContext context(diagnostics);
    std::string arch;
    if (!context.init(0, arch, true, true))
        return false;
    cspan<unsigned char> raygen;
    for (const auto& module : hart_custom_closure_modules)
        if (module.arch && arch == module.arch && *module.size > 0)
            raygen = { module.data, size_t(*module.size) };
    if (raygen.empty()) {
        diagnostics.errorfmt("No custom closure test module for {}", arch);
        return false;
    }
    {
        Services unsupported(false);
        ShadingSystem ss(&unsupported, nullptr, &diagnostics);
        const auto group = make_group(ss, diagnostics, stdosl, arch, optimize);
        OIIO_CHECK_ASSERT(group);
        const void* bitcode = nullptr;
        OIIO_CHECK_ASSERT(group
                          && !ss.getattribute(group.get(), "hart_bitcode",
                                              TypeDesc::PTR, &bitcode));
        OIIO_CHECK_ASSERT(diagnostics.errors > 0);
        OIIO_CHECK_ASSERT(diagnostics.last_error.find("HARTClosureAllocator")
                          != std::string::npos);
        OIIO_CHECK_EQUAL(host_allocator_calls, 0u);
        OIIO_CHECK_EQUAL(context.statistics().launches, size_t(0));
    }
    const unsigned expected_errors = diagnostics.errors;
    Services services(true);
    ShadingSystem ss(&services, nullptr, &diagnostics);
    const auto group = make_group(ss, diagnostics, stdosl, arch, optimize);
    if (!group || diagnostics.errors != expected_errors)
        return false;
    const void* data = nullptr;
    uint64_t bytes   = 0;
    int group_size = 0, alignment = 0;
    HartCallable callable;
    callable.entries.resize(2);
    if (!ss.getattribute(group.get(), "hart_bitcode", TypeDesc::PTR, &data)
        || !ss.getattribute(group.get(), "hart_bitcode_size", TypeUInt64, &bytes)
        || !data || !bytes
        || !ss.getattribute(group.get(), "llvm_groupdata_size", group_size)
        || !ss.getattribute(group.get(), "llvm_groupdata_alignment", alignment)
        || group_size < 0 || alignment <= 0 || alignment > 256
        || (alignment & (alignment - 1))
        || !ss.getattribute(group.get(), "group_init_name", callable.entries[0])
        || !ss.getattribute(group.get(), "group_entry_name",
                            callable.entries[1]))
        return false;
    callable.bitcode = { static_cast<const unsigned char*>(data),
                         size_t(bytes) };
    if (!context.build_accel({}, {}, {}, 1)
        || !context.create_pipeline(raygen, "__raygen__custom_closures", 1,
                                    { &callable, 1 }))
        return false;
    std::array<Result, cases> results;
    std::memset(results.data(), 0xff, sizeof(results));
    Params params {};
    params.group_stride = (unsigned(group_size) + 255) & ~255u;
    if (!params.group_stride)
        params.group_stride = 256;
    params.groupdata = static_cast<unsigned char*>(
        context.alloc(params.group_stride * cases));
    params.results = static_cast<Result*>(context.alloc(sizeof(results)));
    if (!params.groupdata || !params.results
        || !context.upload(params.results,
                           { reinterpret_cast<const unsigned char*>(
                                 results.data()),
                             sizeof(results) })
        || !context.launch(&params, sizeof(params), cases, 1)
        || !context.download({ reinterpret_cast<unsigned char*>(results.data()),
                               sizeof(results) },
                             params.results))
        return false;
    OIIO_CHECK_EQUAL(context.statistics().launches, size_t(1));
    OIIO_CHECK_EQUAL(host_allocator_calls, 0u);
    unsigned weighted = 0, unweighted = 0, nodes = 0;
    for (unsigned row = 0; row < cases; ++row) {
        const auto& result = results[row];
        OIIO_CHECK_EQUAL(result.done, completed);
        OIIO_CHECK_EQUAL(result.errors, row == 4 ? allocation_failed : 0u);
        OIIO_CHECK_EQUAL(result.zero_probe_null, 1u);
        OIIO_CHECK_EQUAL(result.zero_probe_calls, 0u);
        OIIO_CHECK_EQUAL(result.sg_init_checked, 1u);
        OIIO_CHECK_EQUAL(result.sg_allocator_checks,
                         result.component_calls + result.node_calls);
        weighted += result.weighted_calls;
        unweighted += result.unweighted_calls;
        nodes += result.node_calls;
        if (row == 2 || row == 4 || row == 5) {
            OIIO_CHECK_EQUAL(result.leaves, 0u);
            if (row == 4)
                OIIO_CHECK_EQUAL(result.component_calls, 1u);
            else if (optimize == 2 && row == 5)
                OIIO_CHECK_EQUAL(result.component_calls, 0u);
        } else {
            OIIO_CHECK_EQUAL(result.leaves, row == 3 ? 2u : 1u);
            const int id = row == 0 || row == 3 || row == 7 ? id32 : id64;
            const float keyword = row == 1 ? 23.5f : row == 7 ? 37.5f : 17.5f;
            const Color3 weight = row == 1   ? Color3(.25f, .5f, .75f)
                                  : row == 3 ? Color3(3, 2, .5f)
                                  : row == 6 ? Color3(0, 2, 0)
                                  : row == 7 ? Color3(-2)
                                             : Color3(1);
            check_leaf(result.leaf[0], row, id, keyword, weight);
            if (row == 3) {
                check_leaf(result.leaf[1], row, id64, 29.5f, weight);
                OIIO_CHECK_EQUAL(result.add_nodes, 1u);
                OIIO_CHECK_ASSERT(result.mul_nodes > 0);
            }
        }
        print("HART custom closure O{} row {}: leaves={} allocations={} "
              "weighted={} unweighted={} nodes={} errors={} zero-suppressed={} "
              "sg-allocator-checks={}\n",
              optimize, row, result.leaves, result.component_calls,
              result.weighted_calls, result.unweighted_calls, result.node_calls,
              result.errors, result.zero_probe_null && !result.zero_probe_calls,
              result.sg_allocator_checks);
    }
    OIIO_CHECK_ASSERT(unweighted > 0 && nodes > 0);
    if (optimize == 2)
        OIIO_CHECK_ASSERT(weighted > 0);
    OIIO_CHECK_EQUAL(diagnostics.errors, expected_errors);
    print("HART custom closure acceptance: arch={} O{} launches={} points={} "
          "host-allocator-calls={} sg-init-checked=true arnold-profile={}\n",
          arch, optimize, context.statistics().launches, cases,
          host_allocator_calls, bool(OSL_ARNOLD_COMPAT));
    OIIO_CHECK_ASSERT(context.clear());
    return true;
}

}  // namespace



int
main(int argc, char* argv[])
{
    if (argc != 3 || (std::strcmp(argv[2], "0") && std::strcmp(argv[2], "2"))) {
        print(stderr, "Usage: hart_custom_closure_test stdosl.h 0|2\n");
        return 1;
    }
    OIIO_CHECK_ASSERT(run(argv[1], argv[2][0] - '0'));
    return unit_test_failures;
}
