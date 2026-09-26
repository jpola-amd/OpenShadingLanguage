// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <amd/hart/hart_optix_compat.h>

#include "closure.h"
#include "texture.h"
#include "transform.h"

#include <OSL/oslclosure.h>
#include <OSL/oslconfig.h>
#include <OSL/shaderglobals.h>

#include "hartclosureparams.h"
#include "hartgeneratedparams.h"

extern "C" __constant__ testshade::HartGeneratedParams testshade_hart_params;

template<bool Fused>
__device__ const OSL::ClosureColor*
testshade_generated(testshade::HartClosurePool* closure_pool = nullptr)
{
    const uint3 index         = optixGetLaunchIndex();
    const uint3 size          = optixGetLaunchDimensions();
    const uint64_t shadeindex = uint64_t(index.y) * size.x + index.x;
    const auto& params        = testshade_hart_params;
    if (shadeindex >= params.point_count || shadeindex > 0x7fffffff
        || (!Fused && !params.group_stride)
        || (params.group_stride
            && params.point_count > params.scratch_bytes / params.group_stride))
        return nullptr;

    OSL::ShaderGlobals sg { };
    sg.u    = size.x == 1 ? 0.5f : float(index.x) / float(size.x - 1);
    sg.v    = size.y == 1 ? 0.5f : float(index.y) / float(size.y - 1);
    sg.dudx = 1.0f / float(size.x == 1 ? 1 : size.x - 1);
    sg.dvdy = 1.0f / float(size.y == 1 ? 1 : size.y - 1);
    sg.P    = OSL::Vec3(sg.u, sg.v, 1.0f);
    sg.dPdx = OSL::Vec3(sg.dudx, 0.0f, 0.0f);
    sg.dPdy = OSL::Vec3(0.0f, sg.dvdy, 0.0f);
    sg.dPdu = OSL::Vec3(1.0f, 0.0f, 0.0f);
    sg.dPdv = OSL::Vec3(0.0f, 1.0f, 0.0f);
    // Imath's default vector constructor does not initialize its components.
    sg.I = sg.dIdx = sg.dIdy = OSL::Vec3(0.0f);
    sg.time                  = 0.0f;
    sg.N = sg.Ng   = OSL::Vec3(0.0f, 0.0f, 1.0f);
    sg.surfacearea = 1.0f;
    sg.raytype     = params.raytype;
    testshade::HartRenderState state { params.textures, closure_pool };
    sg.renderstate   = &state;
    sg.object2common = params.transforms;
    sg.shader2common = params.transforms + 2;
    void* group      = params.group_stride
                           ? params.scratch + shadeindex * params.group_stride
                           : nullptr;
    optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int, void*>(
        0u, &sg, group, nullptr, params.output, int(shadeindex), nullptr);
    if (!Fused)
        optixDirectCall<void, OSL::ShaderGlobals*, void*, void*, void*, int,
                        void*>(1u, &sg, group, nullptr, params.output,
                               int(shadeindex), nullptr);
    return sg.Ci;
}



extern "C" __global__ void
__raygen__testshade_generated()
{ testshade_generated<false>(); }



extern "C" __global__ void
__raygen__testshade_generated_fused()
{ testshade_generated<true>(); }



__device__ bool
closure_contains(const testshade::HartClosurePool& pool, const void* ptr,
                 size_t bytes)
{
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t base    = reinterpret_cast<uintptr_t>(pool.data);
    return address >= base && address - base <= pool.used
           && bytes <= pool.used - (address - base);
}



template<bool Fused>
__device__ void
inspect_closures()
{
    const auto& params = testshade_hart_params;
    alignas(16) unsigned char storage[testshade::HartClosureCapacity];
    if (params.closure_capacity > sizeof(storage)) {
        atomicOr(params.textures->errors,
                 testshade::HartClosureAllocationFailed);
        return;
    }
    testshade::HartClosurePool pool { storage, params.closure_capacity };
    const OSL::ClosureColor* root = testshade_generated<Fused>(&pool);
    if (pool.failed)
        return;

    testshade::HartClosureSummary result { };
    struct Pending {
        const OSL::ClosureColor* closure;
        OSL::Color3 weight;
    };
    Pending pending[64];
    unsigned int count = 0, visited = 0;
    bool valid = params.closure_capacity <= sizeof(storage);
    if (root)
        pending[count++] = { root, OSL::Color3(1) };
    while (valid && count) {
        const Pending current = pending[--count];
        const auto* closure   = current.closure;
        if (!closure)
            continue;
        valid = ++visited <= 128
                && closure_contains(pool, closure, sizeof(OSL::ClosureColor));
        if (!valid)
            break;
        if (closure->id == OSL::ClosureColor::ADD) {
            valid = count + 2 <= 64
                    && closure_contains(pool, closure, sizeof(OSL::ClosureAdd));
            if (valid) {
                const auto* add  = closure->as_add();
                pending[count++] = { add->closureA, current.weight };
                pending[count++] = { add->closureB, current.weight };
            }
        } else if (closure->id == OSL::ClosureColor::MUL) {
            valid = count < 64
                    && closure_contains(pool, closure, sizeof(OSL::ClosureMul));
            if (valid) {
                const auto* mul  = closure->as_mul();
                pending[count++] = { mul->closure,
                                     current.weight * mul->weight };
            }
        } else {
            // IDs and N's offset are those of testshade's existing registry.
            valid = (closure->id == 1 || closure->id == 3)
                    && closure_contains(pool, closure,
                                        sizeof(OSL::ClosureComponent)
                                            + (closure->id == 3
                                                   ? sizeof(OSL::Vec3)
                                                   : 0));
            if (valid) {
                const OSL::ClosureComponent* comp = closure->as_comp();
                const OSL::Color3 weight          = current.weight * comp->w;
                for (int c = 0; c < 3; ++c) {
                    if (comp->id == 3) {
                        result.diffuse[c] += weight[c];
                        result.normal[c] += (*comp->as<OSL::Vec3>())[c]
                                            * weight.x;
                    } else {
                        result.emission[c] += weight[c];
                    }
                }
                if (comp->id == 3)
                    ++result.diffuse_count;
                else
                    ++result.emission_count;
            }
        }
    }
    if (!valid) {
        atomicOr(params.textures->errors, testshade::HartClosureInvalidTree);
        return;
    }
    result.used          = uint32_t(pool.used);
    const uint3 index    = optixGetLaunchIndex();
    const uint3 size     = optixGetLaunchDimensions();
    const uint64_t point = uint64_t(index.y) * size.x + index.x;
    reinterpret_cast<testshade::HartClosureSummary*>(
        params.output)[point] = result;
}



extern "C" __global__ void
__raygen__testshade_closures()
{ inspect_closures<false>(); }



extern "C" __global__ void
__raygen__testshade_closures_fused()
{ inspect_closures<true>(); }



extern "C" __global__ void
__raygen__testshade_closure_pool()
{
    alignas(16) unsigned char storage[128];
    testshade::HartClosurePool pool { storage, sizeof(storage) };
    testshade::HartClosureSummary result { };
    result.diffuse[0]  = pool.allocate(1, 1) == storage;
    result.diffuse[1]  = pool.allocate(16, 16) == storage + 16;
    result.diffuse[2]  = pool.allocate(96, 16) == storage + 32
                         && pool.used == sizeof(storage);
    result.emission[0] = !pool.allocate(1, 1) && pool.failed
                         && pool.used == sizeof(storage);
    result.emission[1] = !pool.allocate(1, 1);
    pool.reset();
    result.emission[2] = !pool.failed && pool.used == 0
                         && pool.allocate(128, 16) == storage;
    pool.reset();
    result.normal[0] = !pool.allocate(size_t(-1), 16) && pool.failed
                       && pool.used == 0;
    pool.reset();
    result.normal[1] = !pool.allocate(1, 3) && pool.failed;
    pool.reset();
    result.normal[2] = !pool.allocate(1, 0) && pool.failed;
    pool.reset();
    result.diffuse_count = pool.allocate(16, 16) == storage;
    pool.reset();
    result.emission_count = pool.allocate(16, 16) == storage;
    result.used           = uint32_t(pool.used);
    const uint3 index     = optixGetLaunchIndex();
    const uint3 size      = optixGetLaunchDimensions();
    const uint64_t point  = uint64_t(index.y) * size.x + index.x;
    reinterpret_cast<testshade::HartClosureSummary*>(
        testshade_hart_params.output)[point] = result;
}
