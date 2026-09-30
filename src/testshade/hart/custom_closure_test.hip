// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_device.h>
#include <hip/hip_runtime.h>

#include <OSL/oslclosure.h>
#include <OSL/rs_free_function.h>

#include <new>

#include "../hartcustomclosureparams.h"

using namespace OSL;
using namespace custom_closure_test;

extern "C" __constant__ Params osl_hart_render_params;

extern "C" __device__ void*
osl_hart_allocate_closure_component(void* ec, int id, int size, int alignment,
                                    const Color3* weight);

namespace {

struct State {
    ShaderGlobals* expected_sg;
    Result* result;
    unsigned char* arena;
    unsigned used, capacity, index;
};



OSL_HOSTDEVICE bool
initialized_sg(const ShaderGlobals& sg, const State& state, bool check_ci)
{
    const float legacy    = OSL_ARNOLD_COMPAT ? 0.0f : 77.0f;
    const void* transform = OSL_ARNOLD_COMPAT ? nullptr : &state;
    bool valid
        = sg.dtime == legacy && sg.dPdtime == Vec3(legacy)
          && sg.Ps == Vec3(legacy) && sg.dPsdx == Vec3(legacy)
          && sg.dPsdy == Vec3(legacy) && sg.object2common == transform
          && sg.shader2common == transform && sg.flipHandedness == int(legacy)
          && sg.shade_index == int(state.index)
          && sg.context == reinterpret_cast<const ShadingContext*>(&state)
          && sg.renderstate == &state && sg.tracedata == state.result
          && sg.objdata == &state && sg.P == Vec3(float(state.index), 2, 3)
          && sg.N == Vec3(0, 0, 1) && sg.Ng == Vec3(0, 0, 1)
          && sg.dPdx == Vec3(1, 0, 0) && sg.dPdy == Vec3(0, 1, 0)
          && sg.dPdz == Vec3(0, 0, 1) && sg.I == Vec3(0, 0, -1)
          && sg.dIdx == Vec3(2, 0, 0) && sg.dIdy == Vec3(0, 2, 0)
          && sg.dPdu == Vec3(3, 0, 0) && sg.dPdv == Vec3(0, 3, 0)
          && sg.u == float(state.index) && sg.time == 0.25f
          && sg.surfacearea == 1.5f && sg.raytype == 123 && sg.backfacing == 1;
#if OSL_ARNOLD_COMPAT
    valid = valid && (!check_ci || !sg.Ci) && !sg.renderer
            && !sg.shadingStateUniform && sg.thread_index == 0;
#else
    (void)check_ci;
#endif
    return valid;
}



OSL_HOSTDEVICE State*
checked_state(OpaqueExecContextPtr ec)
{
    auto* sg    = static_cast<ShaderGlobals*>(ec);
    auto* state = get_rs<State>(ec);
    if (!state)
        return nullptr;
    const bool first = state->result->component_calls == 0
                       && state->result->node_calls == 0;
    if (state->expected_sg != sg || !initialized_sg(*sg, *state, first)) {
        state->result->errors |= bad_state;
        return nullptr;
    }
    ++state->result->sg_allocator_checks;
    return state;
}



OSL_HOSTDEVICE void*
allocate(State& state, size_t bytes, size_t alignment, size_t prefix = 0)
{
    const uintptr_t base    = reinterpret_cast<uintptr_t>(state.arena);
    const uintptr_t payload = (base + state.used + prefix + alignment - 1)
                              & ~(alignment - 1);
    const size_t end        = payload + bytes - base;
    if (end > state.capacity) {
        state.result->errors |= exhausted;
        return nullptr;
    }
    state.used = unsigned(end);
    return reinterpret_cast<void*>(payload);
}



template<class PayloadT>
OSL_HOSTDEVICE void
capture(const ClosureComponent& component, const Color3& weight, Leaf& leaf)
{
    const auto* payload      = component.as<PayloadT>();
    leaf.id                  = component.id;
    leaf.guard               = payload->guard;
    leaf.end_guard           = payload->end_guard;
    leaf.payload_alignment   = alignof(PayloadT);
    leaf.payload_remainder   = reinterpret_cast<uintptr_t>(payload)
                               % alignof(PayloadT);
    leaf.component_remainder = reinterpret_cast<uintptr_t>(&component)
                               % alignof(ClosureComponent);
    leaf.header_bytes        = reinterpret_cast<const char*>(payload)
                               - reinterpret_cast<const char*>(&component);
    leaf.token               = payload->token;
    leaf.gain                = payload->gain;
    leaf.keyword             = payload->keyword;
    for (int c = 0; c < 3; ++c) {
        leaf.tint[c]   = payload->tint[c];
        leaf.weight[c] = weight[c] * component.w[c];
    }
}



__device__ void
inspect(const ClosureColor* root, Result& result)
{
    const ClosureColor* nodes[16];
    Color3 weights[16];
    unsigned count = 1, visited = 0;
    nodes[0]   = root;
    weights[0] = Color3(1);
    while (count) {
        const auto* node    = nodes[--count];
        const Color3 weight = weights[count];
        if (!node)
            continue;
        if (++visited > 16 || count > 13) {
            result.errors |= bad_tree;
            return;
        }
        if (node->id == ClosureColor::ADD) {
            ++result.add_nodes;
            nodes[count]     = node->as_add()->closureB;
            weights[count++] = weight;
            nodes[count]     = node->as_add()->closureA;
            weights[count++] = weight;
        } else if (node->id == ClosureColor::MUL) {
            ++result.mul_nodes;
            nodes[count]     = node->as_mul()->closure;
            weights[count++] = weight * node->as_mul()->weight;
        } else if (result.leaves < 4 && node->id == id32) {
            capture<Payload32>(*node->as_comp(), weight,
                               result.leaf[result.leaves++]);
        } else if (result.leaves < 4 && node->id == id64) {
            capture<Payload64>(*node->as_comp(), weight,
                               result.leaf[result.leaves++]);
        } else {
            result.errors |= bad_tree;
            return;
        }
    }
}

}  // namespace



OSL_RSOP OSL_HOSTDEVICE void*
rs_allocate_closure(OpaqueExecContextPtr ec, size_t size, size_t alignment)
{
    auto* state = checked_state(ec);
    if (!state)
        return nullptr;
    ++state->result->node_calls;
    return allocate(*state, size, alignment);
}



OSL_RSOP OSL_HOSTDEVICE void*
rs_hart_allocate_closure_component(OpaqueExecContextPtr ec, int id,
                                   int payload_size, int payload_alignment,
                                   const Color3* weight)
{
    auto* state = checked_state(ec);
    if (!state)
        return nullptr;
    auto& result = *state->result;
    ++result.component_calls;
    if (weight)
        ++result.weighted_calls;
    else
        ++result.unweighted_calls;
    const bool small = id == id32;
    if ((!small && id != id64)
        || payload_size != (small ? sizeof(Payload32) : sizeof(Payload64))
        || payload_alignment
               != (small ? alignof(Payload32) : alignof(Payload64))
        || (weight && *weight == Color3(0))) {
        result.errors |= bad_layout;
        return nullptr;
    }
    if (state->index == 4) {
        result.errors |= allocation_failed;
        return nullptr;
    }
    void* payload = allocate(*state, payload_size, payload_alignment,
                             sizeof(ClosureComponent));
    if (!payload)
        return nullptr;
    auto* component = new (static_cast<char*>(payload)
                           - sizeof(ClosureComponent)) ClosureComponent;
    component->id   = id;
    component->w    = weight ? *weight : Color3(1);
    if (small)
        new (payload) Payload32;
    else
        new (payload) Payload64;
    return component;
}



extern "C" __global__ void
__raygen__custom_closures()
{
    const unsigned index = hartGetLaunchIndex().x;
    Result result {};
    alignas(64) unsigned char arena[2048];
    ShaderGlobals sg {};
    State state { &sg, &result, arena, 0, sizeof(arena), index };
    sg.P    = Vec3(float(index), 2, 3);
    sg.dPdx = sg.dPdy = sg.dPdz = sg.I = sg.dIdx = sg.dIdy = Vec3(0);
    sg.N = sg.Ng = Vec3(0, 0, 1);
    sg.dPdu = sg.dPdv = sg.dPdtime = sg.Ps = sg.dPsdx = sg.dPsdy = Vec3(0);
    sg.u                                                         = float(index);
    sg.shade_index                                               = int(index);
    sg.renderstate = sg.objdata = &state;
    sg.context                  = reinterpret_cast<ShadingContext*>(&state);
    sg.tracedata                = &result;
    // Current preserves these renderer values; Arnold initializes omissions.
    sg.dtime   = 77;
    sg.dPdtime = sg.Ps = sg.dPsdx = sg.dPsdy = Vec3(77);
    sg.object2common = sg.shader2common = &state;
    sg.flipHandedness                   = 77;
#if OSL_ARNOLD_COMPAT
    sg.Ci                  = reinterpret_cast<ClosureColor*>(&state);
    sg.renderer            = reinterpret_cast<RendererServices*>(&state);
    sg.shadingStateUniform = &state;
    sg.thread_index = sg.shade_index = -77;
#endif
    sg.dPdx        = Vec3(1, 0, 0);
    sg.dPdy        = Vec3(0, 1, 0);
    sg.dPdz        = Vec3(0, 0, 1);
    sg.I           = Vec3(0, 0, -1);
    sg.dIdx        = Vec3(2, 0, 0);
    sg.dIdy        = Vec3(0, 2, 0);
    sg.dPdu        = Vec3(3, 0, 0);
    sg.dPdv        = Vec3(0, 3, 0);
    sg.time        = 0.25f;
    sg.surfacearea = 1.5f;
    sg.raytype     = 123;
    sg.backfacing  = 1;
    void* group    = osl_hart_render_params.groupdata
                     + index * osl_hart_render_params.group_stride;
    hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
        0u, &sg, group, nullptr, nullptr, int(index), nullptr);
    result.sg_init_checked = initialized_sg(sg, state, true);
    hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
        1u, &sg, group, nullptr, nullptr, int(index), nullptr);
    inspect(sg.Ci, result);
    // Exercise the production bridge with a runtime zero weight even when
    // OSL keeps a varying closure multiplication separate from allocation.
    const Color3 zero(osl_hart_render_params.zero_weight[0],
                      osl_hart_render_params.zero_weight[1],
                      osl_hart_render_params.zero_weight[2]);
    const unsigned before = result.component_calls;
    result.zero_probe_null
        = osl_hart_allocate_closure_component(&sg, id32, sizeof(Payload32),
                                              alignof(Payload32), &zero)
          == nullptr;
    result.zero_probe_calls               = result.component_calls - before;
    result.done                           = completed;
    osl_hart_render_params.results[index] = result;
}



// HartContext's pipeline also binds these entries; this fixture traces no rays.
extern "C" __global__ void
__miss__osl_hart()
{
}



extern "C" __global__ void
__closesthit__osl_hart()
{
}
