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
#include <cmath>
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



#if OSL_ARNOLD_COMPAT
class ClosureInputRenderer final : public RendererServices {
public:
    ClosureColor* value = nullptr;
    bool found          = true;
    int calls           = 0;

    bool get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                      ShaderGlobals*, void* destination) override
    {
        OIIO_CHECK_EQUAL(name, ustringhash("input"));
        OIIO_CHECK_ASSERT(!derivatives);
        OIIO_CHECK_EQUAL(type, TypeDesc(TypeDesc::PTR));
        ++calls;
        if (found)
            *static_cast<ClosureColor**>(destination) = value;
        return found;
    }
};



void
test_closure_userdata(string_view stdosl, int optimize, int llvm_optimize)
{
    const auto bytecode = compile_shader(R"OSL(
        shader closure_input(closure color input = 0 [[ int lockgeom = 0 ]])
        {
            Ci = input;
        }
    )OSL",
                                         stdosl);
    if (bytecode.empty())
        return;
    for (int lazy : { 0, 1 }) {
        ClosureInputRenderer renderer;
        ShadingSystem ss(&renderer);
        ss.attribute("optimize", optimize);
        ss.attribute("llvm_optimize", llvm_optimize);
        ss.attribute("lazy_userdata", lazy);
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("closure_input", bytecode));
        auto group   = make_group(ss, "closure_input");
        auto* thread = ss.create_thread_info();
        auto* ctx    = ss.get_context(thread);
        ClosureComponent component {};
        component.id = header_id;
        component.w  = Color3(1);
        for (int sample = 0; sample < 3; ++sample) {
            renderer.value = sample == 0 ? &component : nullptr;
            renderer.found = sample != 2;
            renderer.calls = 0;
            ShaderGlobals globals {};
            OIIO_CHECK_ASSERT(
                ss.execute(*ctx, *group, 0, sample, globals, nullptr, nullptr));
            OIIO_CHECK_ASSERT(globals.Ci == renderer.value);
            OIIO_CHECK_EQUAL(renderer.calls, 1);
        }
        int num_userdata = 0;
        TypeDesc* types  = nullptr;
        OIIO_CHECK_ASSERT(
            ss.getattribute(group.get(), "num_userdata", num_userdata));
        OIIO_CHECK_EQUAL(num_userdata, 1);
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "userdata_types",
                                          TypeDesc::PTR, &types));
        OIIO_CHECK_ASSERT(types && types[0] == TypeDesc::PTR);
        ss.release_context(ctx);
        ss.destroy_thread_info(thread);
    }
    ClosureInputRenderer renderer;
    ShadingSystem ss(&renderer);
    ss.attribute("debug_output_cpp", 1);
    OIIO_CHECK_ASSERT(ss.LoadMemoryCompiledShader("closure_input", bytecode));
    auto group   = make_group(ss, "closure_input");
    auto* thread = ss.create_thread_info();
    auto* ctx    = ss.get_context(thread);
    ShaderGlobals globals {};
    OIIO_CHECK_ASSERT(
        !ss.execute(*ctx, *group, 0, 0, globals, nullptr, nullptr));
    ss.release_context(ctx);
    ss.destroy_thread_info(thread);
}
#endif



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

#if OSL_ARNOLD_COMPAT

class TextureDiagnostics final : public ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        if ((code & 0xffff0000) == EH_ERROR || (code & 0xffff0000) == EH_SEVERE)
            errors += message + "\n";
    }
    std::string errors;
};



class LegacyTextureRenderer : public RendererServices {
public:
    using RendererServices::environment;
    using RendererServices::texture;
    using RendererServices::texture3d;

    bool texture(ustringhash, TextureHandle*, TexturePerthread*, TextureOpt&,
                 ShaderGlobals*, float, float, float, float, float, float,
                 int nchannels, float* result, float* ds, float* dt,
                 ustringhash* error) override
    {
        const size_t channels = static_cast<size_t>(nchannels);
        return sample(0, { result, channels }, { ds, ds ? channels : 0 },
                      { dt, dt ? channels : 0 }, {}, error);
    }

    bool texture3d(ustringhash, TextureHandle*, TexturePerthread*, TextureOpt&,
                   ShaderGlobals*, const Vec3&, const Vec3&, const Vec3&,
                   const Vec3&, int nchannels, float* result, float* ds,
                   float* dt, float* dr, ustringhash* error) override
    {
        const size_t channels = static_cast<size_t>(nchannels);
        return sample(1, { result, channels }, { ds, ds ? channels : 0 },
                      { dt, dt ? channels : 0 }, { dr, dr ? channels : 0 },
                      error);
    }

    bool environment(ustringhash, TextureHandle*, TexturePerthread*,
                     TextureOpt&, ShaderGlobals*, const Vec3&, const Vec3&,
                     const Vec3&, int nchannels, float* result, float* ds,
                     float* dt, ustringhash* error) override
    {
        const size_t channels = static_cast<size_t>(nchannels);
        return sample(2, { result, channels }, { ds, ds ? channels : 0 },
                      { dt, dt ? channels : 0 }, {}, error);
    }

    std::array<int, 3> calls {};
    bool status = true;

private:
    bool sample(int operation, span<float> result, span<float> ds,
                span<float> dt, span<float> dr, ustringhash* error)
    {
        ++calls[operation];
        for (float& value : result)
            value = float(operation + 1);
        for (float& value : ds)
            value = 2;
        for (float& value : dt)
            value = 3;
        for (float& value : dr)
            value = 4;
        if (error)
            *error = status ? ustringhash()
                            : ustringhash("legacy texture error");
        return status;
    }
};



void
test_texture_defaults()
{
    LegacyTextureRenderer renderer;
    RendererServices& services = renderer;
    TextureDiagnostics diagnostics;
    ShadingSystem ss(&renderer, nullptr, &diagnostics);
    OIIO_CHECK_ASSERT(ss.attribute("buffer_printf", 0));
    auto* thread = ss.create_thread_info();
    auto* ctx    = ss.get_context(thread);
    ShaderGlobals sg {};
    if (ctx) {
        sg.context  = ctx;
        sg.renderer = &renderer;
        TextureOpt options;
        std::array<float, 4> result, ds, dt, dr;
        const Vec3 p(0), dp(1);
        const ustringhash filename("arnold_colorspace_fixture");
        for (int operation = 0; operation < 3; ++operation) {
            auto lookup = [&](ustringhash space, ustringhash* error,
                              bool derivatives) {
                if (operation == 0)
                    return services.texture(filename, space, nullptr, nullptr,
                                            options, &sg, 0, 0, 1, 0, 0, 1, 4,
                                            result.data(),
                                            derivatives ? ds.data() : nullptr,
                                            derivatives ? dt.data() : nullptr,
                                            error);
                if (operation == 1)
                    return services.texture3d(filename, space, nullptr, nullptr,
                                              options, &sg, p, dp, dp, dp, 4,
                                              result.data(),
                                              derivatives ? ds.data() : nullptr,
                                              derivatives ? dt.data() : nullptr,
                                              derivatives ? dr.data() : nullptr,
                                              error);
                return services.environment(filename, space, nullptr, nullptr,
                                            options, &sg, p, dp, dp, 4,
                                            result.data(),
                                            derivatives ? ds.data() : nullptr,
                                            derivatives ? dt.data() : nullptr,
                                            error);
            };
            for (bool status : { true, false }) {
                renderer.status = status;
                ustringhash error("not overwritten");
                OIIO_CHECK_EQUAL(lookup(ustringhash(), &error, true), status);
                OIIO_CHECK_EQUAL(error,
                                 status ? ustringhash()
                                        : ustringhash("legacy texture error"));
                for (int c = 0; c < 4; ++c) {
                    OIIO_CHECK_EQUAL(result[c], float(operation + 1));
                    OIIO_CHECK_EQUAL(ds[c], 2);
                    OIIO_CHECK_EQUAL(dt[c], 3);
                    if (operation == 1)
                        OIIO_CHECK_EQUAL(dr[c], 4);
                }
            }
            OIIO_CHECK_EQUAL(renderer.calls[operation], 2);
            for (auto space : { "raw", "sRGB", "unknown-space" }) {
                for (bool derivatives : { false, true }) {
                    result.fill(-1);
                    ds.fill(-1);
                    dt.fill(-1);
                    dr.fill(-1);
                    ustringhash error;
                    OIIO_CHECK_ASSERT(
                        !lookup(ustringhash(space), &error, derivatives));
                    const auto message = ustring_from(error).string();
                    OIIO_CHECK_ASSERT(message.find(space) != std::string::npos);
                    OIIO_CHECK_ASSERT(message.find("colorspace")
                                      != std::string::npos);
                    OIIO_CHECK_ASSERT(message.find("arnold_colorspace_fixture")
                                      != std::string::npos);
                    for (int c = 0; c < 4; ++c) {
                        OIIO_CHECK_EQUAL(result[c], 0);
                        OIIO_CHECK_EQUAL(ds[c], derivatives ? 0 : -1);
                        OIIO_CHECK_EQUAL(dt[c], derivatives ? 0 : -1);
                        OIIO_CHECK_EQUAL(dr[c], derivatives && operation == 1
                                                    ? 0
                                                    : -1);
                    }
                    OIIO_CHECK_EQUAL(renderer.calls[operation], 2);
                }
            }
            OIIO_CHECK_ASSERT(diagnostics.errors.empty());
            OIIO_CHECK_ASSERT(
                !lookup(ustringhash("reported-space"), nullptr, true));
            OIIO_CHECK_ASSERT(diagnostics.errors.find("reported-space")
                              != std::string::npos);
            diagnostics.errors.clear();
        }
    } else {
        OIIO_CHECK_ASSERT(ctx);
    }
    if (ctx)
        ss.release_context(ctx);
    ss.destroy_thread_info(thread);
}



struct TextureRequest {
    int operation;
    ustringhash space;
};



class ColorspaceRenderer final : public LegacyTextureRenderer {
public:
    using LegacyTextureRenderer::environment;
    using LegacyTextureRenderer::texture;
    using LegacyTextureRenderer::texture3d;

    TextureHandle* get_texture_handle(ustring filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        OIIO_CHECK_EQUAL(filename, ustring("arnold_colorspace_fixture"));
        return handle();
    }

    TextureHandle* get_texture_handle(ustringhash filename, ShadingContext*,
                                      const TextureOpt*) override
    {
        OIIO_CHECK_EQUAL(filename, ustringhash("arnold_colorspace_fixture"));
        return handle();
    }

    bool good(TextureHandle* texture) override { return texture == handle(); }
    bool is_udim(TextureHandle*) override { return false; }

    bool texture(ustringhash filename, ustringhash space,
                 TextureHandle* texture, TexturePerthread* thread,
                 TextureOpt& options, ShaderGlobals* sg, float s, float t,
                 float dsdx, float dtdx, float dsdy, float dtdy, int nchannels,
                 float* result, float* ds, float* dt,
                 ustringhash* error) override
    {
        check_request(0, filename, space, texture, nchannels);
        if (!supported(space))
            return RendererServices::texture(filename, space, texture, thread,
                                             options, sg, s, t, dsdx, dtdx,
                                             dsdy, dtdy, nchannels, result, ds,
                                             dt, error);
        const size_t channels = static_cast<size_t>(nchannels);
        return sample(space, Vec3(s, t, 0), { result, channels },
                      { ds, ds ? channels : 0 }, { dt, dt ? channels : 0 }, {},
                      error);
    }

    bool texture3d(ustringhash filename, ustringhash space,
                   TextureHandle* texture, TexturePerthread* thread,
                   TextureOpt& options, ShaderGlobals* sg, const Vec3& p,
                   const Vec3& dpdx, const Vec3& dpdy, const Vec3& dpdz,
                   int nchannels, float* result, float* ds, float* dt,
                   float* dr, ustringhash* error) override
    {
        check_request(1, filename, space, texture, nchannels);
        if (!supported(space))
            return RendererServices::texture3d(filename, space, texture, thread,
                                               options, sg, p, dpdx, dpdy, dpdz,
                                               nchannels, result, ds, dt, dr,
                                               error);
        const size_t channels = static_cast<size_t>(nchannels);
        return sample(space, p, { result, channels }, { ds, ds ? channels : 0 },
                      { dt, dt ? channels : 0 }, { dr, dr ? channels : 0 },
                      error);
    }

    bool environment(ustringhash filename, ustringhash space,
                     TextureHandle* texture, TexturePerthread* thread,
                     TextureOpt& options, ShaderGlobals* sg, const Vec3& r,
                     const Vec3& drdx, const Vec3& drdy, int nchannels,
                     float* result, float* ds, float* dt,
                     ustringhash* error) override
    {
        check_request(2, filename, space, texture, nchannels);
        if (!supported(space))
            return RendererServices::environment(filename, space, texture,
                                                 thread, options, sg, r, drdx,
                                                 drdy, nchannels, result, ds,
                                                 dt, error);
        const size_t channels = static_cast<size_t>(nchannels);
        return sample(space, r, { result, channels }, { ds, ds ? channels : 0 },
                      { dt, dt ? channels : 0 }, {}, error);
    }

    std::vector<TextureRequest> expected;
    size_t received = 0;

private:
    TextureHandle* handle()
    {
        return reinterpret_cast<TextureHandle*>(&m_resource);
    }

    void check_request(int operation, ustringhash filename, ustringhash space,
                       TextureHandle* texture, int nchannels)
    {
        // Verify the full request before any resource lookup or interpretation.
        OIIO_CHECK_EQUAL(filename, ustringhash("arnold_colorspace_fixture"));
        OIIO_CHECK_ASSERT(received < expected.size());
        if (received < expected.size()) {
            OIIO_CHECK_EQUAL(operation, expected[received].operation);
            OIIO_CHECK_EQUAL(space, expected[received].space);
        }
        ++received;
        OIIO_CHECK_ASSERT(!texture || texture == handle());
        OIIO_CHECK_EQUAL(nchannels, 4);
    }

    bool supported(ustringhash space) const
    {
        return space == ustringhash() || space == ustringhash("raw")
               || space == ustringhash("sRGB")
               || space == ustringhash("renderer-linear");
    }

    bool sample(ustringhash space, const Vec3& p, span<float> result,
                span<float> ds, span<float> dt, span<float> dr,
                ustringhash* error)
    {
        // A procedural RGBA fixture, not Arnold/OCIO integration. Its RGB
        // samples use the fixed sRGB transfer curve; alpha is always linear.
        // The empty selection deliberately means raw for this renderer only.
        const float base[] = { 0.02f, 0.25f, 0.5f, 0.625f };
        const float gs[]   = { 0.02f, 0.125f, 0.0625f, 0.125f };
        const float gt[]   = { 0.01f, 0.0625f, 0.125f, 0.0625f };
        const float gr[]   = { 0.005f, 0.03125f, 0.0625f, 0.03125f };
        for (int c = 0; c < int(result.size()); ++c) {
            const float raw = base[c] + gs[c] * p.x + gt[c] * p.y + gr[c] * p.z;
            float value = raw, slope = 1;
            if (c < 3 && space == ustringhash("sRGB")) {
                if (raw <= 0.04045f) {
                    value = raw / 12.92f;
                    slope = 1 / 12.92f;
                } else {
                    const float t = (raw + 0.055f) / 1.055f;
                    value         = std::pow(t, 2.4f);
                    slope         = (2.4f / 1.055f) * std::pow(t, 1.4f);
                }
            }
            result[c] = value;
            if (!ds.empty())
                ds[c] = slope * gs[c];
            if (!dt.empty())
                dt[c] = slope * gt[c];
            if (!dr.empty())
                dr[c] = slope * gr[c];
        }
        if (error)
            *error = ustringhash();
        return true;
    }

    int m_resource = 0;
};



double
reference_texture(int channel, const Vec3& p, bool srgb)
{
    // Independent double-precision formulas for the four fixture channels.
    double value
        = channel == 0
              ? (4 + 4 * double(p.x) + 2 * double(p.y) + double(p.z)) / 200
          : channel == 1
              ? (8 + 4 * double(p.x) + 2 * double(p.y) + double(p.z)) / 32
          : channel == 2
              ? (8 + double(p.x) + 2 * double(p.y) + double(p.z)) / 16
              : (20 + 4 * double(p.x) + 2 * double(p.y) + double(p.z)) / 32;
    if (srgb && channel < 3)
        value = value <= 0.04045 ? value / 12.92
                                 : std::pow((value + 0.055) / 1.055, 2.4);
    return value;
}



std::string
texture_shader_source(string_view selection, bool paired, bool interactive)
{
    std::string source = "shader arnold_colorspace(string space=\"raw\"";
    if (interactive)
        source += " [[int interactive=1]]";
    source += R"OSL(,
        output color result[6] = {0,0,0,0,0,0},
        output color dx[6] = {0,0,0,0,0,0},
        output color dy[6] = {0,0,0,0,0,0},
        output float alpha[6] = {0,0,0,0,0,0},
        output float alpha_dx[6] = {0,0,0,0,0,0},
        output float alpha_dy[6] = {0,0,0,0,0,0},
        output string messages[6] = {"","","","","",""})
        {
    )OSL";
    const char* operations[] = { "texture", "texture3d", "environment" };
    for (int i = 0; i < (paired ? 6 : 3); ++i) {
        const std::string index = std::to_string(i);
        source += "result[" + index + "] = " + operations[i % 3]
                  + "(\"arnold_colorspace_fixture\", "
                  + (i % 3 == 0   ? "u, v"
                     : i % 3 == 1 ? "P"
                                  : "vector(P)");
        if (i >= 3)
            source += ", \"colorspace\", \"raw\"";
        else if (!selection.empty())
            source += ", \"colorspace\", " + std::string(selection);
        source += ", \"alpha\", alpha[" + index
                  + "], \"errormessage\", messages[" + index + "]);\n";
        source += "dx[" + index + "] = Dx(result[" + index + "]);\n" + "dy["
                  + index + "] = Dy(result[" + index + "]);\n" + "alpha_dx["
                  + index + "] = Dx(alpha[" + index + "]);\n" + "alpha_dy["
                  + index + "] = Dy(alpha[" + index + "]);\n";
    }
    return source + "}";
}



void
check_texture_outputs(ShadingSystem& ss, ShadingContext& ctx,
                      const ShaderGlobals& sg, cspan<TextureRequest> requests)
{
    const char* names[] = { "result", "dx",       "dy",
                            "alpha",  "alpha_dx", "alpha_dy" };
    for (int output = 0; output < 6; ++output) {
        TypeDesc type;
        const auto* values = static_cast<const float*>(
            ss.get_symbol(ctx, ustring(names[output]), type));
        const int channels = output < 3 ? 3 : 1;
        OIIO_CHECK_ASSERT(values && type.basetype == TypeDesc::FLOAT
                          && type.aggregate == channels && type.arraylen == 6);
        if (!values)
            continue;
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto request = requests[i];
            const bool srgb    = request.space == ustringhash("sRGB");
            const bool invalid = request.space == ustringhash("unknown-space");
            const Vec3 p = request.operation == 0 ? Vec3(sg.u, sg.v, 0) : sg.P;
            const Vec3 dpdx = request.operation == 0 ? Vec3(sg.dudx, sg.dvdx, 0)
                                                     : sg.dPdx;
            const Vec3 dpdy = request.operation == 0 ? Vec3(sg.dudy, sg.dvdy, 0)
                                                     : sg.dPdy;
            for (int c = 0; c < channels; ++c) {
                const int channel = output < 3 ? c : 3;
                double expected   = 0;
                if (!invalid) {
                    if (output % 3 == 0)
                        expected = reference_texture(channel, p, srgb);
                    else if (request.operation != 2) {
                        // Check analytical callback derivatives by independent
                        // central differences, not the callback's slope formula.
                        const Vec3 step = (output % 3 == 1 ? dpdx : dpdy)
                                          * 0.001f;
                        expected = (reference_texture(channel, p + step, srgb)
                                    - reference_texture(channel, p - step, srgb))
                                   / 0.002;
                    }
                    // OSL intentionally zeros environment lookup derivatives.
                }
                OIIO_CHECK_EQUAL_THRESH(values[i * channels + c], expected,
                                        0.00003);
            }
        }
    }
    TypeDesc type;
    const auto* messages = static_cast<const ustringhash*>(
        ss.get_symbol(ctx, ustring("messages"), type));
    OIIO_CHECK_ASSERT(messages && type == TypeDesc(TypeDesc::STRING, 6));
    if (messages) {
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i].space == ustringhash("unknown-space")) {
                const auto text = ustring_from(messages[i]).string();
                OIIO_CHECK_ASSERT(text.find("unknown-space")
                                  != std::string::npos);
                OIIO_CHECK_ASSERT(text.find("colorspace") != std::string::npos);
            } else {
                OIIO_CHECK_EQUAL(messages[i], ustringhash());
            }
        }
    }
}



void
check_cpp_texture_calls(ShadingSystem& ss, ShaderGroup& group,
                        cspan<TextureRequest> requests, bool dynamic_space)
{
    int group_id = 0;
    OIIO_CHECK_ASSERT(ss.getattribute(&group, "group_id", group_id));
    const std::string filename = "group-cpp-arnold_colorspace_"
                                 + std::to_string(group_id) + ".cpp";
    std::string source;
    OIIO_CHECK_ASSERT(OIIO::Filesystem::read_text_file(filename, source));
    OIIO::Filesystem::remove(filename);
    // The color hash must be the third argument, before the resource handle.
    const std::regex call(
        R"(osl_(texture|texture3d|environment)\(\(void\*\)sg,\s*([^,]+),\s*([^,]+),\s*nullptr,\s*\(void\*\)&_tex_opt,)");
    const char* operations[] = { "texture", "texture3d", "environment" };
    size_t received          = 0;
    for (std::sregex_iterator it(source.begin(), source.end(), call), end;
         it != end; ++it, ++received) {
        OIIO_CHECK_ASSERT(received < requests.size());
        if (received >= requests.size())
            continue;
        const auto request = requests[received];
        OIIO_CHECK_EQUAL((*it)[1].str(), operations[request.operation]);
        auto check_hash = [&](std::string expression, ustringhash expected,
                              bool must_be_dynamic) {
            const auto first = expression.find_first_not_of(" \t\r\n");
            const auto last  = expression.find_last_not_of(" \t\r\n");
            OIIO_CHECK_ASSERT(first != std::string::npos);
            if (first == std::string::npos)
                return;
            expression            = expression.substr(first, last - first + 1);
            const bool is_dynamic = expression.size() >= 7
                                    && expression.compare(expression.size() - 7,
                                                          7, ".hash()")
                                           == 0;
            if (must_be_dynamic)
                OIIO_CHECK_ASSERT(is_dynamic);
            if (is_dynamic)
                return;
            if (expression == "0") {
                OIIO_CHECK_EQUAL(expected, ustringhash());
            } else {
                const std::string declaration
                    = "static const uint64_t " + expression
                      + " = OSL::ustring(\"" + ustring_from(expected).string()
                      + "\").hash();";
                OIIO_CHECK_ASSERT(source.find(declaration)
                                  != std::string::npos);
            }
        };
        check_hash((*it)[2].str(), ustringhash("arnold_colorspace_fixture"),
                   false);
        check_hash((*it)[3].str(), request.space, dynamic_space);
    }
    OIIO_CHECK_EQUAL(received, requests.size());
}



void
test_texture_colorspaces(string_view stdosl, int optimize, int llvm_optimize)
{
    print(stderr,
          "  texture colorspace callbacks and fixed-transform fixture\n");
    struct TestCase {
        const char* name;
        const char* selection;
        const char* bound;
        bool paired      = false;
        bool runtime     = false;
        bool interactive = false;
    };
    const TestCase cases[] = {
        { "omitted", "", "" },
        { "empty", "\"\"", "" },
        { "literal", "\"sRGB\"", "sRGB", true },
        { "bound", "space", "sRGB" },
        { "custom", "space", "renderer-linear" },
        { "runtime", "u < 0.5 ? space : \"raw\"", "sRGB", false, true },
        { "interactive", "space", "sRGB", false, false, true },
        { "invalid", "space", "unknown-space" },
    };
    for (const auto& test : cases) {
        print(stderr, "    {}\n", test.name);
        ColorspaceRenderer renderer;
        TextureDiagnostics diagnostics;
        ShadingSystem ss(&renderer, nullptr, &diagnostics);
        OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", llvm_optimize));
        // Source-only ABI coverage does not require an external C++ compiler.
        OIIO_CHECK_ASSERT(ss.attribute("debug_output_cpp", 1));
        OIIO_CHECK_ASSERT(ss.attribute("cpp_output_dir", "."));
        const char* outputs[] = { "result",   "dx",       "dy",      "alpha",
                                  "alpha_dx", "alpha_dy", "messages" };
        OIIO_CHECK_ASSERT(ss.attribute("renderer_outputs",
                                       TypeDesc(TypeDesc::STRING, 7), outputs));
        const auto bytecode
            = compile_shader(texture_shader_source(test.selection, test.paired,
                                                   test.interactive),
                             stdosl);
        if (bytecode.empty())
            continue;
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("arnold_colorspace", bytecode));
        auto group = ss.ShaderGroupBegin("arnold_colorspace");
        OIIO_CHECK_ASSERT(group);
        if (!group)
            continue;
        OIIO_CHECK_ASSERT(ss.Parameter(*group, "space", ustring(test.bound),
                                       test.interactive
                                           ? ParamHints::interactive
                                           : ParamHints::none));
        OIIO_CHECK_ASSERT(ss.Shader("surface", "arnold_colorspace", "surface"));
        OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        auto* thread = ss.create_thread_info();
        auto* ctx    = ss.get_context(thread);
        OIIO_CHECK_ASSERT(ctx);
        if (ctx) {
            const int repeats = test.runtime || test.interactive ? 3 : 1;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                const char* selected = test.bound;
                if ((test.runtime || test.interactive) && repeat == 1)
                    selected = "raw";
                if (test.interactive && repeat > 0)
                    OIIO_CHECK_ASSERT(ss.ReParameter(*group, "surface", "space",
                                                     ustring(selected)));
                renderer.received = 0;
                renderer.expected.clear();
                for (int i = 0; i < (test.paired ? 6 : 3); ++i)
                    renderer.expected.push_back(
                        { i % 3, ustringhash(i < 3 ? selected : "raw") });
                ShaderGlobals sg {};
                sg.u    = test.runtime && repeat == 1 ? 0.8f : 0.2f;
                sg.v    = 0.3f;
                sg.dudx = 0.4f;
                sg.dudy = -0.2f;
                sg.dvdx = 0.1f;
                sg.dvdy = 0.3f;
                sg.P    = Vec3(0.2f, 0.3f, 0.4f);
                sg.dPdx = Vec3(0.4f, 0.1f, 0.2f);
                sg.dPdy = Vec3(-0.2f, 0.3f, -0.1f);
                OIIO_CHECK_ASSERT(
                    ss.execute(*ctx, *group, 0, repeat, sg, nullptr, nullptr));
                if (repeat == 0)
                    check_cpp_texture_calls(ss, *group, renderer.expected,
                                            test.runtime || test.interactive);
                OIIO_CHECK_EQUAL(renderer.received, renderer.expected.size());
                check_texture_outputs(ss, *ctx, sg, renderer.expected);
                for (int count : renderer.calls)
                    OIIO_CHECK_EQUAL(count, 0);
                OIIO_CHECK_ASSERT(diagnostics.errors.empty());
            }
            ss.release_context(ctx);
        }
        ss.destroy_thread_info(thread);
    }
}

#endif

#if OSL_USE_OPTIX

class OptixCodegenRenderer final : public RendererServices {
public:
#    if OSL_ARNOLD_COMPAT
    bool texture_colorspaces = false;
#    endif

    int supports(string_view feature) const override
    {
#    if OSL_ARNOLD_COMPAT
        if (feature == "TextureColorSpaces")
            return texture_colorspaces;
#    endif
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



#    if OSL_ARNOLD_COMPAT
void
check_ptx_texture_abi(const std::string& ptx)
{
    const uint64_t hash             = ustringhash("sRGB").hash();
    const std::string unsigned_hash = std::to_string(hash);
    const std::string signed_hash = std::to_string(static_cast<int64_t>(hash));
    const char* functions[]       = { "osl_texture", "osl_texture3d",
                                      "osl_environment" };
    const int counts[]            = { 19, 17, 16 };
    for (int operation = 0; operation < 3; ++operation) {
        const std::string name = functions[operation];
        const std::regex declaration("\\.func\\s+\\([^)]*\\)\\s+" + name
                                     + "\\s*\\(([^)]*)\\)");
        std::smatch match;
        const bool found = std::regex_search(ptx, match, declaration);
        OIIO_CHECK_ASSERT(found);
        if (!found)
            continue;
        const std::string parameters = match[1].str();
        const std::regex parameter("\\.param\\s+(\\.[A-Za-z0-9_]+)");
        int index = 0;
        for (std::sregex_iterator
                 p(parameters.begin(), parameters.end(), parameter),
             end;
             p != end; ++p, ++index) {
            const bool small = operation == 0   ? index >= 5 && index <= 11
                               : operation == 1 ? index == 9
                                                : index == 8;
            OIIO_CHECK_EQUAL((*p)[1].str(), small ? ".b32" : ".b64");
        }
        OIIO_CHECK_EQUAL(index, counts[operation]);
        const std::regex call("call(?:\\.uni)?\\s+\\([^)]*\\),\\s*" + name
                              + ",\\s*\\(([^)]*)\\)");
        int calls                      = 0;
        bool literal_at_third_argument = false;
        for (std::sregex_iterator c(ptx.begin(), ptx.end(), call), end;
             c != end; ++c, ++calls) {
            const std::string arguments = (*c)[1].str();
            const std::regex identifier("[A-Za-z_][A-Za-z0-9_]*");
            std::vector<std::string> args;
            for (std::sregex_iterator
                     a(arguments.begin(), arguments.end(), identifier),
                 aend;
                 a != aend; ++a)
                args.push_back(a->str());
            OIIO_CHECK_EQUAL(args.size(), size_t(counts[operation]));
            if (args.size() < 3)
                continue;
            const std::string prefix = ptx.substr(0, c->position());
            const size_t block       = prefix.rfind('{');
            OIIO_CHECK_ASSERT(block != std::string::npos);
            if (block == std::string::npos)
                continue;
            const std::string setup = prefix.substr(block);
            const std::regex store("st\\.param\\.b64\\s+\\[\\s*" + args[2]
                                   + "(?:\\+0)?\\s*\\],\\s*([^;\\s]+)\\s*;");
            std::smatch value;
            const bool stored = std::regex_search(setup, value, store);
            OIIO_CHECK_ASSERT(stored);
            if (!stored)
                continue;
            const std::string operand = value[1].str();
            const std::regex constant("mov\\.(?:u64|b64)\\s+" + operand
                                      + ",\\s*(?:" + unsigned_hash + "|"
                                      + signed_hash + ")\\s*;");
            literal_at_third_argument |= operand == unsigned_hash
                                         || operand == signed_hash
                                         || std::regex_search(prefix, constant);
        }
        // Inlining into both exported entry points may duplicate these calls.
        OIIO_CHECK_ASSERT(calls >= 2);
        OIIO_CHECK_ASSERT(literal_at_third_argument);
    }
}



void
test_optix_texture_codegen(string_view stdosl, int optimize, int llvm_optimize)
{
    print(stderr, "  OptiX texture colorspace PTX ABI (no device/context)\n");
    const auto bytecode = compile_shader(R"OSL(
        shader arnold_optix_texture(string space = "raw",
                                   output color result = 0)
        {
            result = texture("arnold_colorspace_fixture", u, v,
                             "colorspace", "sRGB")
                   + texture3d("arnold_colorspace_fixture", P,
                               "colorspace", "sRGB")
                   + environment("arnold_colorspace_fixture", vector(P),
                                 "colorspace", "sRGB")
                   + texture("arnold_colorspace_fixture", u, v,
                             "colorspace", space)
                   + texture3d("arnold_colorspace_fixture", P,
                               "colorspace", space)
                   + environment("arnold_colorspace_fixture", vector(P),
                                 "colorspace", space);
        }
    )OSL",
                                         stdosl);
    if (bytecode.empty())
        return;
    for (bool capable : { false, true }) {
        OptixCodegenRenderer renderer;
        renderer.texture_colorspaces = capable;
        OptixDiagnostics diagnostics;
        ShadingSystem ss(&renderer, nullptr, &diagnostics);
        OIIO_CHECK_ASSERT(ss.attribute("optimize", optimize));
        OIIO_CHECK_ASSERT(ss.attribute("llvm_optimize", llvm_optimize));
        const char* outputs[] = { "result" };
        OIIO_CHECK_ASSERT(ss.attribute("renderer_outputs",
                                       TypeDesc(TypeDesc::STRING, 1), outputs));
        OIIO_CHECK_ASSERT(
            ss.LoadMemoryCompiledShader("arnold_optix_texture", bytecode));
        auto group = ss.ShaderGroupBegin("arnold_optix_texture");
        OIIO_CHECK_ASSERT(group);
        if (!group)
            continue;
        OIIO_CHECK_ASSERT(ss.Parameter(*group, "space", ustring("raw"),
                                       ParamHints::interpolated));
        OIIO_CHECK_ASSERT(
            ss.Shader("surface", "arnold_optix_texture", "surface"));
        OIIO_CHECK_ASSERT(ss.ShaderGroupEnd());
        ss.optimize_group(group.get(), nullptr);
        std::string ptx = "must be cleared on failure";
        OIIO_CHECK_ASSERT(ss.getattribute(group.get(), "ptx_compiled_version",
                                          TypeDesc::PTR, &ptx));
        if (capable) {
            OIIO_CHECK_ASSERT(!ptx.empty());
            OIIO_CHECK_ASSERT(diagnostics.errors.empty());
            check_ptx_texture_abi(ptx);
        } else {
            OIIO_CHECK_ASSERT(ptx.empty());
            OIIO_CHECK_ASSERT(diagnostics.errors.find("OptiX")
                              != std::string::npos);
            OIIO_CHECK_ASSERT(diagnostics.errors.find("TextureColorSpaces")
                              != std::string::npos);
        }
    }
}
#    endif



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
#if OSL_ARNOLD_COMPAT
    test_texture_defaults();
#endif
    for (int optimize : { 0, 2 }) {
        for (int llvm_optimize : { 0, 2 }) {
            print(stderr, "Arnold acceptance: OSL O{}, LLVM O{}\n", optimize,
                  llvm_optimize);
            test_closures(argv[1], optimize, llvm_optimize);
            test_metadata(argv[1], optimize, llvm_optimize);
            test_shaderglobals(argv[1], optimize, llvm_optimize);
#if OSL_ARNOLD_COMPAT
            test_closure_userdata(argv[1], optimize, llvm_optimize);
            test_texture_colorspaces(argv[1], optimize, llvm_optimize);
#endif
#if OSL_USE_OPTIX
            test_optix_codegen(argv[1], optimize, llvm_optimize);
#    if OSL_ARNOLD_COMPAT
            test_optix_texture_codegen(argv[1], optimize, llvm_optimize);
#    endif
#endif
        }
    }
    return unit_test_failures ? 1 : 0;
}
