// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Installed-consumer checks use only public APIs. The in-tree target also
// checks the raw JIT initializer, which has no public accessor.
#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include <OSL/shaderglobals.h>

#ifdef OSL_ARNOLD_TEST_INTERNAL
#    include "oslexec_pvt.h"
#endif

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/unittest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(EXPECT_STATIC_OSL) && !defined(OSL_STATIC_DEFINE)
#    error "Installed static targets must propagate OSL_STATIC_DEFINE"
#endif
#if OSL_ARNOLD_COMPAT && !defined(OSL_ARNOLD_MODIFIED_API)
#    error "The Arnold compatibility profile must advertise its API"
#endif
#if !OSL_ARNOLD_COMPAT && defined(OSL_ARNOLD_MODIFIED_API)
#    error "The default profile must not advertise the Arnold compatibility ABI"
#endif

using namespace OSL;

namespace {

constexpr int probe_id  = 91;
constexpr int legacy_id = 92;
constexpr int header_id = 1000 + probe_id;
constexpr int sentinel  = 0x12345678;
const Color3 header_weight(2, 3, 4);



struct alignas(64) ProbePayload {
    ProbePayload()
        : amount(-1)
        , direction(-1)
        , tag(-1)
        , roughness(0.75f)
        , omitted(0.875f)
        , marker(sentinel)
    {
    }
    float amount;
    Vec3 direction;
    int tag;
    float roughness;
    float omitted;
    int marker;
};

const ClosureParam probe_params[]
    = { CLOSURE_FLOAT_PARAM(ProbePayload, amount),
        CLOSURE_VECTOR_PARAM(ProbePayload, direction),
        CLOSURE_INT_PARAM(ProbePayload, tag),
        CLOSURE_FLOAT_KEYPARAM(ProbePayload, roughness, "roughness"),
        CLOSURE_FLOAT_KEYPARAM(ProbePayload, omitted, "omitted"),
        CLOSURE_FINISH_PARAM(ProbePayload) };



struct LegacyPayload {
    float amount;
    float roughness;
    int marker;
};

const ClosureParam legacy_params[]
    = { CLOSURE_FLOAT_PARAM(LegacyPayload, amount),
        CLOSURE_FLOAT_KEYPARAM(LegacyPayload, roughness, "roughness"),
        CLOSURE_FINISH_PARAM(LegacyPayload) };



struct Allocation {
    ClosureComponent* component = nullptr;
    bool weighted               = false;
    Color3 weight               = Color3(1);
};



struct AllocationState {
    // The payload, not the component, must be 64-byte aligned.
    static constexpr size_t slot_size = 64 + sizeof(ProbePayload);
    alignas(64) std::array<std::array<std::byte, slot_size>, 16> storage;
    std::vector<Allocation> allocations;
    ShaderGlobals* expected_globals = nullptr;
    bool fail                       = false;
    int calls                       = 0;
};



class TestRenderer final : public RendererServices {
public:
    int prepares = 0;
    int setups   = 0;
};



ClosureComponent*
allocate_probe(ShaderGlobals* globals, int id, const Color3* weight)
{
    auto& state = *static_cast<AllocationState*>(globals->renderstate);
    OIIO_CHECK_ASSERT(globals == state.expected_globals);
    OIIO_CHECK_ASSERT(globals->context);
    OIIO_CHECK_EQUAL(id, probe_id);
    ++state.calls;
    if (state.fail)
        return nullptr;
    if (state.allocations.size() >= state.storage.size()) {
        OIIO_CHECK_ASSERT(false);
        return nullptr;
    }
    auto* payload   = state.storage[state.allocations.size()].data() + 64;
    auto* component = new (payload - sizeof(ClosureComponent)) ClosureComponent;
    new (component->data()) ProbePayload;
    // Deliberately distinguish renderer-owned headers from OSL's defaults.
    component->id = header_id;
    component->w  = header_weight * (weight ? *weight : Color3(1));
    state.allocations.push_back(
        { component, weight != nullptr, weight ? *weight : Color3(1) });
    return component;
}



void
prepare_legacy(RendererServices* renderer, int id, void* data)
{
    OIIO_CHECK_EQUAL(id, legacy_id);
    ++static_cast<TestRenderer*>(renderer)->prepares;
    new (data) LegacyPayload { -1, 0.75f, sentinel };
}



void
setup_legacy(RendererServices* renderer, int id, void* data)
{
    OIIO_CHECK_EQUAL(id, legacy_id);
    ++static_cast<TestRenderer*>(renderer)->setups;
    const auto& payload = *static_cast<LegacyPayload*>(data);
    OIIO_CHECK_EQUAL(payload.amount, 9.0f);
    // Legacy setup precedes keyword assignment.
    OIIO_CHECK_EQUAL(payload.roughness, 0.75f);
    OIIO_CHECK_EQUAL(payload.marker, sentinel);
}



void
register_closures(ShadingSystem& ss)
{
    AllocClosureFunc allocator = allocate_probe;
    ss.register_closure("arnold_probe", probe_id, probe_params, allocator);
    ss.register_closure("arnold_legacy", legacy_id, legacy_params,
                        prepare_legacy, setup_legacy);
}



ShaderGroupRef
make_group(ShadingSystem& ss, string_view shader)
{
    auto group = ss.ShaderGroupBegin("arnold_acceptance");
    OIIO_CHECK_ASSERT(group);
    OIIO_CHECK_ASSERT(ss.Shader("surface", shader, "surface"));
    OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
    return group;
}



std::string
compile_shader(string_view source, string_view stdosl)
{
    OSLCompiler compiler;
    std::string bytecode;
    OIIO_CHECK_ASSERT(
        compiler.compile_buffer(source, bytecode, { "-O0" }, stdosl));
    return bytecode;
}



const char* declarations = R"OSL(
closure color arnold_probe(float amount, vector direction, int tag)
    [[ int builtin = 1 ]];
closure color arnold_legacy(float amount) [[ int builtin = 1 ]];
)OSL";

struct ClosureCase {
    const char* name;
    const char* body;
    bool fail = false;
};

const ClosureCase closure_cases[] = {
    { "unweighted", "Ci = arnold_probe(1.25, vector(1,2,3), 7);" },
    { "keyword",
      "Ci = arnold_probe(2.5, vector(1,2,3), 8, \"roughness\", 0.25);" },
    { "weighted",
      "Ci = color(0.25,0.5,0.75) * arnold_probe(3, vector(1,2,3), 9);" },
    { "tree", R"OSL(
        closure color a = arnold_probe(4, vector(1,2,3), 10);
        closure color b = arnold_probe(5, vector(1,2,3), 11,
                                      "roughness", 0.625);
        Ci = color(0.25,0.5,0.75) * (a + b) + 2 * a;
      )OSL" },
    { "zero", "Ci = color(0) * arnold_probe(6, vector(1,2,3), 12);" },
    { "dynamic_weight", "color weight = color(u, 2*u, 3*u);"
                        " Ci = weight * arnold_probe(6, vector(1,2,3), 12);" },
    { "null", "Ci = arnold_probe(7, vector(1,2,3), 13, \"roughness\", 0.5);",
      true },
    { "weighted_null",
      "Ci = color(0.25,0.5,0.75)"
      " * arnold_probe(7, vector(1,2,3), 13, \"roughness\", 0.5);",
      true },
    { "legacy", "Ci = arnold_legacy(9, \"roughness\", 0.5);" },
};



void
check_color(const Color3& actual, const Color3& expected)
{
    for (int channel = 0; channel < 3; ++channel)
        OIIO_CHECK_EQUAL(actual[channel], expected[channel]);
}



void
collect_weights(const ClosureColor* closure, const Color3& weight,
                std::array<Color3, 14>& weights)
{
    if (!closure)
        return;
    if (closure->id == ClosureColor::ADD) {
        collect_weights(closure->as_add()->closureA, weight, weights);
        collect_weights(closure->as_add()->closureB, weight, weights);
    } else if (closure->id == ClosureColor::MUL) {
        collect_weights(closure->as_mul()->closure,
                        weight * closure->as_mul()->weight, weights);
    } else {
        OIIO_CHECK_EQUAL(closure->id, header_id);
        if (closure->id != header_id)
            return;
        const auto* component = closure->as_comp();
        const int tag         = component->as<ProbePayload>()->tag;
        OIIO_CHECK_ASSERT(tag >= 0 && tag < int(weights.size()));
        if (tag >= 0 && tag < int(weights.size()))
            weights[tag] += weight * component->w;
    }
}



void
test_closures(string_view stdosl, int optimize, int llvm_optimize)
{
    TestRenderer renderer;
    ShadingSystem ss(&renderer);
    OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
    OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", llvm_optimize));
    register_closures(ss);
    auto* thread = ss.create_thread_info();
    auto* ctx    = ss.get_context(thread);
    OIIO_CHECK_ASSERT(ctx);
    for (const auto& test : closure_cases) {
        print(stderr, "  closure {}\n", test.name);
        const std::string source = std::string(declarations)
                                   + "shader arnold_acceptance() { " + test.body
                                   + " }";
        const auto bytecode      = compile_shader(source, stdosl);
        if (bytecode.empty())
            continue;
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader(test.name, bytecode));
        auto group = make_group(ss, test.name);
        if (!ctx || !group)
            continue;
        // Repeat execution to catch assumptions about allocator/pool ownership.
        for (int repeat = 0; repeat < 2; ++repeat) {
            AllocationState state;
            ShaderGlobals globals {};
            globals.renderstate = &state;
            if (std::string(test.name) == "dynamic_weight")
                globals.u = 0.5f * repeat;
            state.expected_globals = &globals;
            state.fail             = test.fail;
            OIIO_CHECK_ASSERT(
                ss.execute(*ctx, *group, 0, 0, globals, nullptr, nullptr));
            for (const auto& allocation : state.allocations) {
                const auto* component = allocation.component;
                const auto* payload   = component->as<ProbePayload>();
                OIIO_CHECK_EQUAL(component->id, header_id);
                check_color(component->w, header_weight * allocation.weight);
                OIIO_CHECK_EQUAL(uintptr_t(payload) % alignof(ProbePayload), 0);
                OIIO_CHECK_EQUAL(payload->marker, sentinel);
                OIIO_CHECK_EQUAL(payload->omitted, 0.875f);
                check_color(payload->direction, Vec3(1, 2, 3));
                const float amounts[] = { 1.25f, 2.5f, 3, 4, 5, 6 };
                OIIO_CHECK_ASSERT(payload->tag >= 7 && payload->tag <= 12);
                if (payload->tag >= 7 && payload->tag <= 12)
                    OIIO_CHECK_EQUAL(payload->amount,
                                     amounts[payload->tag - 7]);
                const float roughness = payload->tag == 8    ? 0.25f
                                        : payload->tag == 11 ? 0.625f
                                                             : 0.75f;
                OIIO_CHECK_EQUAL(payload->roughness, roughness);
            }
            if (std::string(test.name) == "legacy") {
                OIIO_CHECK_ASSERT(globals.Ci);
                if (globals.Ci) {
                    OIIO_CHECK_EQUAL(globals.Ci->id, legacy_id);
                    check_color(globals.Ci->as_comp()->w, Color3(1));
                    const auto* payload
                        = globals.Ci->as_comp()->as<LegacyPayload>();
                    OIIO_CHECK_EQUAL(payload->amount, 9.0f);
                    OIIO_CHECK_EQUAL(payload->roughness, 0.5f);
                    OIIO_CHECK_EQUAL(payload->marker, sentinel);
                }
                OIIO_CHECK_EQUAL(renderer.prepares, repeat + 1);
                OIIO_CHECK_EQUAL(renderer.setups, repeat + 1);
                OIIO_CHECK_EQUAL(state.calls, 0);
                continue;
            }
            if (test.fail) {
                OIIO_CHECK_EQUAL(state.calls, 1);
                OIIO_CHECK_ASSERT(!globals.Ci);
                continue;
            }
            if (std::string(test.name) == "unweighted") {
                OIIO_CHECK_EQUAL(state.calls, 1);
                if (!state.allocations.empty())
                    OIIO_CHECK_ASSERT(!state.allocations[0].weighted);
            }
            if (std::string(test.name) == "weighted" && optimize == 2) {
                OIIO_CHECK_EQUAL(state.calls, 1);
                if (!state.allocations.empty()) {
                    OIIO_CHECK_ASSERT(state.allocations[0].weighted);
                    check_color(state.allocations[0].weight,
                                Color3(0.25f, 0.5f, 0.75f));
                }
            }
            if (std::string(test.name) == "dynamic_weight" && optimize == 2) {
                // Unlike the default allocator, the CPU renderer callback must
                // receive a live weighted request even when the weight is zero.
                OIIO_CHECK_EQUAL(state.calls, 1);
                if (!state.allocations.empty()) {
                    OIIO_CHECK_ASSERT(state.allocations[0].weighted);
                    check_color(state.allocations[0].weight,
                                Color3(globals.u, 2 * globals.u, 3 * globals.u));
                }
            }
            std::array<Color3, 14> weights;
            weights.fill(Color3(0));
            collect_weights(globals.Ci, Color3(1), weights);
            for (int tag = 7; tag <= 12; ++tag) {
                Color3 expected(0);
                if (std::string(test.name) == "unweighted" && tag == 7)
                    expected = Color3(1);
                if (std::string(test.name) == "keyword" && tag == 8)
                    expected = Color3(1);
                if (std::string(test.name) == "weighted" && tag == 9)
                    expected = Color3(0.25f, 0.5f, 0.75f);
                if (std::string(test.name) == "tree" && tag == 10)
                    expected = Color3(2.25f, 2.5f, 2.75f);
                if (std::string(test.name) == "tree" && tag == 11)
                    expected = Color3(0.25f, 0.5f, 0.75f);
                if (std::string(test.name) == "dynamic_weight" && tag == 12)
                    expected = Color3(globals.u, 2 * globals.u, 3 * globals.u);
                check_color(weights[tag], header_weight * expected);
            }
        }
    }
    if (ctx)
        ss.release_context(ctx);
    ss.destroy_thread_info(thread);
}



void
test_loaded(string_view stdosl)
{
    TestRenderer renderer;
    ShadingSystem ss(&renderer);
    const auto bytecode = compile_shader("shader arnold_disk() {}", stdosl);
    if (bytecode.empty())
        return;
    OIIO_CHECK_ASSERT(!ss.ShaderLoaded(""));
    OIIO_CHECK_ASSERT(!ss.ShaderLoaded("arnold_memory"));
    OIIO_CHECK_ASSERT(!ss.ShaderLoaded("arnold_memory"));
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("arnold_memory", bytecode));
    OIIO_CHECK_ASSERT(ss.ShaderLoaded("arnold_memory"));
    OIIO_CHECK_ASSERT(!ss.LoadMemoryCompiledShader("arnold_memory", bytecode));
    OIIO_CHECK_ASSERT(ss.ShaderLoaded("arnold_memory"));
    OIIO_CHECK_ASSERT(
        !ss.LoadMemoryCompiledShader("arnold_invalid", "invalid"));
    // Compatibility queries key presence, including a failed-load cache entry.
    OIIO_CHECK_ASSERT(ss.ShaderLoaded("arnold_invalid"));
    OIIO_CHECK_ASSERT(ss.ShaderLoaded("arnold_invalid"));
    OIIO_CHECK_ASSERT(!ss.LoadMemoryCompiledShader("arnold_empty", ""));
    OIIO_CHECK_ASSERT(!ss.ShaderLoaded("arnold_empty"));

    const std::string filename = "arnold_compat_disk_only.oso";
    OIIO_CHECK_ASSERT(OIIO::Filesystem::write_text_file(filename, bytecode));
    OIIO_CHECK_ASSERT(ss.attribute("searchpath:shader", "."));
    OIIO_CHECK_ASSERT(!ss.ShaderLoaded("arnold_compat_disk_only"));
    OIIO_CHECK_ASSERT(!ss.ShaderLoaded("arnold_compat_disk_only"));
    auto group = make_group(ss, "arnold_compat_disk_only");
    OIIO_CHECK_ASSERT(ss.ShaderLoaded("arnold_compat_disk_only"));
    OIIO::Filesystem::remove(filename);
    OIIO_CHECK_ASSERT(ss.ShaderLoaded("arnold_compat_disk_only"));

    std::atomic<bool> stop { false };
    std::atomic<bool> valid { true };
    std::atomic<int> ready { 0 };
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t)
        readers.emplace_back([&]() {
            ++ready;
            while (!stop.load()) {
                if (!ss.ShaderLoaded("arnold_memory")
                    || ss.ShaderLoaded("arnold_never_loaded")
                    || !ss.ShaderLoaded("arnold_invalid"))
                    valid.store(false);
            }
        });
    while (ready.load() != 4)
        std::this_thread::yield();
    for (int i = 0; i < 16; ++i) {
        const std::string name = "arnold_concurrent_" + std::to_string(i);
        OIIO_CHECK_ASSERT(!ss.ShaderLoaded(name));
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader(name, bytecode));
        OIIO_CHECK_ASSERT(ss.ShaderLoaded(name));
    }
    stop.store(true);
    for (auto& reader : readers)
        reader.join();
    OIIO_CHECK_ASSERT(valid.load());
}



std::set<std::string>
shade_ops(ShadingSystem& ss, ShaderGroup& group, const ustring*& names)
{
    int count = -1;
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "num_shade_ops_needed", count));
    OIIO_CHECK_ASSERT(
        ss.getattribute(&group, "shade_ops_needed", TypeDesc::PTR, &names));
    OIIO_CHECK_ASSERT(count >= 0);
    OIIO_CHECK_ASSERT(count == 0 || names);
    std::set<std::string> result;
    if (count > 0 && names) {
        for (int i = 0; i < count; ++i)
            OIIO_CHECK_ASSERT(result.insert(names[i].string()).second);
    }
    return result;
}



void
test_metadata(string_view stdosl, int optimize, int llvm_optimize)
{
    print(stderr, "  optimized opcode metadata\n");
    TestRenderer renderer;
    ShadingSystem ss(&renderer);
    OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
    OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", llvm_optimize));
    register_closures(ss);
    const char* outputs[] = { "result" };
    OIIO_CHECK_ASSERT(ss.attribute("renderer_outputs",
                                   TypeDesc(TypeDesc::STRING, 1), outputs));
    const std::string source = std::string(declarations) + R"OSL(
        shader arnold_metadata(
            string filename = "" [[ int lockgeom = 0 ]],
            output float result = 0)
        {
            float attribute_value = 0, message_value = 0;
            int a = getattribute("arnold:test", attribute_value);
            int b = getmessage("trace", "arnold:test", message_value);
            color t = texture(filename, u, v);
            int hit = trace(P, I);
            int indices[4];
            float values[4];
            int found = pointcloud_search(filename, P, 1.0, 4, "index", indices);
            int fetched = pointcloud_get(filename, indices, found, "value", values);
            result = a + b + hit + found + fetched + attribute_value
                     + message_value + t[0] + values[0];
            if (0)
                result += sin(u) + cos(v);
            Ci = arnold_probe(u, vector(1,2,3), 7)
                 + arnold_probe(v, vector(1,2,3), 8);
        }
    )OSL";
    const auto bytecode      = compile_shader(source, stdosl);
    if (bytecode.empty())
        return;
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("arnold_metadata", bytecode));
    auto group   = make_group(ss, "arnold_metadata");
    auto* thread = ss.create_thread_info();
    auto* ctx    = ss.get_context(thread);
    if (ctx && group) {
        // Do not execute renderer-dependent operations or require their files.
        ss.optimize_group(group.get(), ctx, false);
        const ustring* names = nullptr;
        const auto expected  = shade_ops(ss, *group, names);
        for (const char* name :
             { "closure", "getattribute", "getmessage", "texture", "trace",
               "pointcloud_search", "pointcloud_get" })
            OIIO_CHECK_ASSERT(expected.count(name));
        if (optimize == 2) {
            OIIO_CHECK_ASSERT(!expected.count("sin"));
            OIIO_CHECK_ASSERT(!expected.count("cos"));
        } else {
            OIIO_CHECK_ASSERT(expected.count("end"));
            OIIO_CHECK_ASSERT(expected.count("useparam"));
        }
        int invalid = 0;
        OIIO_CHECK_ASSERT(!ss.getattribute(group.get(), "num_shade_ops_needed",
                                           TypeDesc::FLOAT, &invalid));
        OIIO_CHECK_ASSERT(!ss.getattribute(group.get(), "shade_ops_needed",
                                           TypeDesc::INT, &invalid));
        // A second group must not invalidate or overwrite the first's storage.
        const auto other_bytecode = compile_shader(
            "shader arnold_other_metadata(output float result=0)"
            " { result = cos(u); }",
            stdosl);
        OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("arnold_other_metadata",
                                                      other_bytecode));
        auto other = make_group(ss, "arnold_other_metadata");
        if (other) {
            ss.optimize_group(other.get(), ctx, false);
            const ustring* other_names = nullptr;
            const auto other_ops       = shade_ops(ss, *other, other_names);
            OIIO_CHECK_ASSERT(other_ops.count("cos"));
            OIIO_CHECK_ASSERT(!other_ops.count("closure"));
        }
        ss.optimize_group(group.get(), ctx, true);
        const ustring* repeated = nullptr;
        OIIO_CHECK_ASSERT(shade_ops(ss, *group, repeated) == expected);
        OIIO_CHECK_ASSERT(repeated == names);
        ss.release_context(ctx);
    } else {
        OIIO_CHECK_ASSERT(ctx && group);
        if (ctx)
            ss.release_context(ctx);
    }
    ss.destroy_thread_info(thread);
}



void
poison_globals(ShaderGlobals& sg, std::array<int, 3>& state)
{
    sg.P              = Vec3(1, 2, 3);
    sg.dPdx           = Vec3(4, 5, 6);
    sg.dPdy           = Vec3(7, 8, 9);
    sg.N              = Vec3(0, 0, 1);
    sg.u              = 0.25f;
    sg.v              = 0.75f;
    sg.time           = 11;
    sg.surfacearea    = 13;
    sg.raytype        = 17;
    sg.backfacing     = 1;
    sg.dtime          = 7;
    sg.dPdtime        = Vec3(8);
    sg.Ps             = Vec3(9);
    sg.dPsdx          = Vec3(10);
    sg.dPsdy          = Vec3(11);
    sg.flipHandedness = 1;
    sg.object2common  = &state[0];
    sg.shader2common  = &state[1];
    sg.renderstate    = &state[0];
    sg.tracedata      = &state[1];
    sg.objdata        = &state[2];
}



void
check_globals(const ShaderGlobals& sg, const std::array<int, 3>& state)
{
    check_color(sg.P, Vec3(1, 2, 3));
    check_color(sg.dPdx, Vec3(4, 5, 6));
    check_color(sg.dPdy, Vec3(7, 8, 9));
    check_color(sg.N, Vec3(0, 0, 1));
    OIIO_CHECK_EQUAL(sg.u, 0.25f);
    OIIO_CHECK_EQUAL(sg.v, 0.75f);
    OIIO_CHECK_EQUAL(sg.time, 11);
    OIIO_CHECK_EQUAL(sg.surfacearea, 13);
    OIIO_CHECK_EQUAL(sg.raytype, 17);
    OIIO_CHECK_EQUAL(sg.backfacing, 1);
    OIIO_CHECK_ASSERT(sg.renderstate == &state[0]);
    OIIO_CHECK_ASSERT(sg.tracedata == &state[1]);
    OIIO_CHECK_ASSERT(sg.objdata == &state[2]);
    OIIO_CHECK_EQUAL(sg.dtime, OSL_ARNOLD_COMPAT ? 0 : 7);
    check_color(sg.dPdtime, Vec3(OSL_ARNOLD_COMPAT ? 0 : 8));
    check_color(sg.Ps, Vec3(OSL_ARNOLD_COMPAT ? 0 : 9));
    check_color(sg.dPsdx, Vec3(OSL_ARNOLD_COMPAT ? 0 : 10));
    check_color(sg.dPsdy, Vec3(OSL_ARNOLD_COMPAT ? 0 : 11));
    OIIO_CHECK_EQUAL(sg.flipHandedness, OSL_ARNOLD_COMPAT ? 0 : 1);
    OIIO_CHECK_ASSERT(sg.object2common
                      == (OSL_ARNOLD_COMPAT ? nullptr : &state[0]));
    OIIO_CHECK_ASSERT(sg.shader2common
                      == (OSL_ARNOLD_COMPAT ? nullptr : &state[1]));
}



void
test_shaderglobals(string_view stdosl, int optimize, int llvm_optimize)
{
    print(stderr, "  ShaderGlobals initialization (profile {})\n",
          OSL_ARNOLD_COMPAT ? "ON" : "OFF");
    TestRenderer renderer;
    ShadingSystem ss(&renderer);
    OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
    OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", llvm_optimize));
    const char* outputs[] = { "result" };
    OIIO_CHECK_ASSERT(ss.attribute("renderer_outputs",
                                   TypeDesc(TypeDesc::STRING, 1), outputs));
    const auto bytecode
        = compile_shader("shader arnold_globals(output float result=0)"
                         " { result = u + dtime + dPdtime[0] + Ps[0]; }",
                         stdosl);
    if (bytecode.empty())
        return;
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("arnold_globals", bytecode));
    auto group   = make_group(ss, "arnold_globals");
    auto* thread = ss.create_thread_info();
    auto* ctx    = ss.get_context(thread);
    if (ctx && group) {
        std::array<int, 3> state {};
        ShaderGlobals sg {};
        for (bool init_only : { true, false }) {
            poison_globals(sg, state);
            if (init_only)
                OIIO_CHECK_ASSERT(ss.execute_init(*ctx, *group, 19, 23, sg,
                                                  nullptr, nullptr));
            else
                OIIO_CHECK_ASSERT(
                    ss.execute(*ctx, *group, 19, 23, sg, nullptr, nullptr));
            check_globals(sg, state);
            OIIO_CHECK_ASSERT(sg.context == ctx);
            OIIO_CHECK_ASSERT(sg.renderer == &renderer);
            OIIO_CHECK_ASSERT(sg.shadingStateUniform);
            OIIO_CHECK_EQUAL(sg.thread_index, 19);
            OIIO_CHECK_EQUAL(sg.shade_index, 23);
            if (init_only)
                OIIO_CHECK_ASSERT(ss.execute_cleanup(*ctx));
            else {
                TypeDesc type;
                const auto* result = static_cast<const float*>(
                    ss.get_symbol(*ctx, ustring("result"), type));
                OIIO_CHECK_ASSERT(result && type == TypeFloat);
                if (result && type == TypeFloat)
                    OIIO_CHECK_EQUAL(*result,
                                     OSL_ARNOLD_COMPAT ? 0.25f : 24.25f);
            }
        }
#ifdef OSL_ARNOLD_TEST_INTERNAL
        // Bypass execute_init's managed-field setup to isolate generated code.
        poison_globals(sg, state);
        ClosureColor closure {};
        sg.Ci                  = &closure;
        sg.thread_index        = 73;
        sg.shade_index         = 74;
        const auto uniform     = sg.shadingStateUniform;
        const size_t heap_size = group->llvm_groupdata_size();
        std::vector<std::max_align_t> heap(
            (heap_size + sizeof(std::max_align_t) - 1)
            / sizeof(std::max_align_t));
        const auto init = group->llvm_compiled_init();
        OIIO_CHECK_ASSERT(init);
        if (init) {
            init(&sg, heap.data(), nullptr, nullptr, 29,
                 group->interactive_arena_ptr());
            check_globals(sg, state);
            OIIO_CHECK_ASSERT(sg.context == ctx);
            OIIO_CHECK_ASSERT(sg.renderer == &renderer);
            OIIO_CHECK_ASSERT(sg.shadingStateUniform == uniform);
            OIIO_CHECK_ASSERT(sg.Ci == &closure);
            OIIO_CHECK_EQUAL(sg.thread_index, 73);
            OIIO_CHECK_EQUAL(sg.shade_index, 74);
        }
#endif
        ss.release_context(ctx);
    } else {
        OIIO_CHECK_ASSERT(ctx && group);
        if (ctx)
            ss.release_context(ctx);
    }
    ss.destroy_thread_info(thread);
}

#if OSL_USE_OPTIX

class OptixCodegenRenderer final : public RendererServices {
public:
    int supports(string_view feature) const override
    {
        return feature == "OptiX";
    }
};



class OptixDiagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR || (code & 0xffff0000) == EH_SEVERE)
            errors += message + "\n";
        ErrorHandler::operator()(code, message);
    }
    std::string errors;
};

int optix_host_allocations = 0;



ClosureComponent*
optix_host_allocator(ShaderGlobals*, int, const Color3*)
{
    ++optix_host_allocations;
    return nullptr;
}



void
check_ptx_parameters(const std::string& ptx, const std::string& name)
{
    OIIO_CHECK_ASSERT(!name.empty());
    OIIO_CHECK_ASSERT(name.find("__direct_callable__") == 0);
    OIIO_CHECK_ASSERT(std::regex_match(name, std::regex("[A-Za-z0-9_]+")));
    // Match an exported definition, not a declaration, call site, or metadata.
    const std::regex definition("\\.visible\\s+\\.func\\s+" + name
                                + "\\s*\\(([^)]*)\\)\\s*\\{");
    std::smatch match;
    const bool found = std::regex_search(ptx, match, definition);
    OIIO_CHECK_ASSERT(found);
    if (!found)
        return;
    const std::string parameters = match[1].str();
    const std::regex parameter("\\.param\\s+(\\.[A-Za-z0-9_]+)");
    std::vector<std::string> types;
    for (std::sregex_iterator
             p(parameters.begin(), parameters.end(), parameter),
         end;
         p != end; ++p)
        types.push_back((*p)[1].str());
    const size_t count = OSL_ARNOLD_COMPAT ? 5 : 6;
    OIIO_CHECK_EQUAL(types.size(), count);
    for (size_t i = 0; i < types.size(); ++i)
        OIIO_CHECK_EQUAL(types[i], i == 4 ? ".b32" : ".b64");
}



void
test_optix_codegen(string_view stdosl, int optimize, int llvm_optimize)
{
    print(stderr, "  OptiX PTX generation (no device/context)\n");
    OptixCodegenRenderer renderer;
    OIIO_CHECK_ASSERT(renderer.supports("OptiX"));
    OIIO_CHECK_ASSERT(!renderer.supports("HART"));
    OptixDiagnostics diagnostics;
    ShadingSystem ss(&renderer, nullptr, &diagnostics);
    OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
    OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", llvm_optimize));
    ss.register_closure("arnold_probe", probe_id, probe_params,
                        optix_host_allocator);
    const char* outputs[] = { "result" };
    OIIO_CHECK_ASSERT(ss.attribute("renderer_outputs",
                                   TypeDesc(TypeDesc::STRING, 1), outputs));
    const auto bytecode = compile_shader(std::string(declarations) + R"OSL(
            shader arnold_optix(output float result = 0)
            {
                result = sin(u) + 2 * v;
                Ci = color(0.25,0.5,0.75)
                     * arnold_probe(u + v, vector(P), 7, "roughness", 0.5);
            }
        )OSL",
                                         stdosl);
    if (bytecode.empty())
        return;
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("arnold_optix", bytecode));
    auto group = make_group(ss, "arnold_optix");
    if (!group)
        return;
    // Only the OSL LLVM/NVPTX backend is involved: no CUDA or OptiX context,
    // device enumeration, device allocations, or shader execution.
    // The export signature is checked below; external renderer launch and
    // execution behavior still require separate hardware integration tests.
    ss.optimize_group(group.get(), nullptr);
    std::string ptx, init, entry, fused;
    OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "ptx_compiled_version",
                                      TypeDesc::PTR, &ptx));
    OIIO_CHECK_ASSERT(!ptx.empty());
    OIIO_CHECK_ASSERT(ptx.find(".version") != std::string::npos);
    OIIO_CHECK_ASSERT(ptx.find(".target sm_") != std::string::npos);
    OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "group_init_name", init));
    OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "group_entry_name", entry));
    OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "group_fused_name", fused));
    for (const auto& name : { init, entry, fused })
        check_ptx_parameters(ptx, name);
    OIIO_CHECK_ASSERT(diagnostics.errors.empty());
    OIIO_CHECK_EQUAL(optix_host_allocations, 0);
    const ustring* names = nullptr;
    const auto ops       = shade_ops(ss, *group, names);
    OIIO_CHECK_ASSERT(ops.count("closure"));
    OIIO_CHECK_ASSERT(ops.count("sin"));
    int closure_count       = 0;
    const ustring* closures = nullptr;
    OIIO_CHECK_ASSERT(
        ss.getattribute(group.get(), "num_closures_needed", closure_count));
    OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "closures_needed",
                                      TypeDesc::PTR, &closures));
    OIIO_CHECK_EQUAL(closure_count, 1);
    OIIO_CHECK_ASSERT(closures);
    if (closure_count == 1 && closures)
        OIIO_CHECK_EQUAL(closures[0], ustring("arnold_probe"));
#    if OSL_ARNOLD_COMPAT
    const auto interactive_bytecode = compile_shader(
        "shader arnold_optix_interactive("
        "float gain=1 [[int interactive=1]], output float result=0)"
        " { result = gain*u; }",
        stdosl);
    if (interactive_bytecode.empty())
        return;
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("arnold_optix_interactive",
                                                  interactive_bytecode));
    auto interactive = make_group(ss, "arnold_optix_interactive");
    if (!interactive)
        return;
    ss.optimize_group(interactive.get(), nullptr);
    OIIO_CHECK_ASSERT(diagnostics.errors.find("interactive")
                      != std::string::npos);
    OIIO_CHECK_ASSERT(diagnostics.errors.find("OptiX") != std::string::npos);
    std::string rejected_ptx = "must be cleared";
    OIIO_CHECK_ASSERT(ss.getattribute(interactive.get(), "ptx_compiled_version",
                                      TypeDesc::PTR, &rejected_ptx));
    OIIO_CHECK_ASSERT(rejected_ptx.empty());
#    endif
}

#endif

}  // namespace



int
main(int argc, char* argv[])
{
    if (argc != 2) {
        print(stderr, "Usage: arnold_compat_test path-to-stdosl.h\n");
        return 1;
    }
    test_loaded(argv[1]);
    for (int optimize : { 0, 2 }) {
        for (int llvm_optimize : { 0, 2 }) {
            print(stderr, "Arnold acceptance: OSL O{}, LLVM O{}\n", optimize,
                  llvm_optimize);
            test_closures(argv[1], optimize, llvm_optimize);
            test_metadata(argv[1], optimize, llvm_optimize);
            test_shaderglobals(argv[1], optimize, llvm_optimize);
#if OSL_USE_OPTIX
            test_optix_codegen(argv[1], optimize, llvm_optimize);
#endif
        }
    }
    return unit_test_failures ? 1 : 0;
}
