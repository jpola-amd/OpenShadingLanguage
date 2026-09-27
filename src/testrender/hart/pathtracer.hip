// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "../../testshade/hart/array.h"
#include "../../testshade/hart/closure.h"
#include "../../testshade/hart/color.h"
#include "../../testshade/hart/texture.h"
#include "../hartpathparams.h"
#include "../sampling.h"
#include "trace.h"

#include <OSL/oslclosure.h>

#include <cfloat>

extern "C" __constant__ OSL::HartPathParams osl_hart_render_params;

using namespace OSL;

namespace {

struct DiffuseLobe {
    Color3 weight;
    Vec3 normal;
    float probability;
};



__device__ bool
contains(const testshade::HartClosurePool& pool, const void* ptr, size_t bytes)
{
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    const auto base    = reinterpret_cast<uintptr_t>(pool.data);
    return address >= base && address - base <= pool.used
           && bytes <= pool.used - (address - base);
}



__device__ bool
collect(const ClosureColor* root, const testshade::HartClosurePool& pool,
        Color3& emission, DiffuseLobe* lobes, unsigned& num_lobes)
{
    struct Pending {
        const ClosureColor* closure;
        Color3 weight;
    };
    Pending pending[64];
    unsigned count = 0, visited = 0;
    if (root)
        pending[count++] = { root, Color3(1) };
    num_lobes = 0;
    emission  = Color3(0);
    while (count) {
        const auto current = pending[--count];
        const auto* node   = current.closure;
        if (!node)
            continue;
        if (++visited > 128 || !contains(pool, node, sizeof(ClosureColor)))
            return false;
        if (node->id == ClosureColor::ADD) {
            if (count + 2 > 64 || !contains(pool, node, sizeof(ClosureAdd)))
                return false;
            pending[count++] = { node->as_add()->closureB, current.weight };
            pending[count++] = { node->as_add()->closureA, current.weight };
        } else if (node->id == ClosureColor::MUL) {
            if (count == 64 || !contains(pool, node, sizeof(ClosureMul)))
                return false;
            const auto* mul  = node->as_mul();
            pending[count++] = { mul->closure, current.weight * mul->weight };
        } else {
            if ((node->id != 1 && node->id != 3)
                || !contains(pool, node,
                             sizeof(ClosureComponent)
                                 + (node->id == 3 ? sizeof(Vec3) : 0)))
                return false;
            const auto* comp = node->as_comp();
            Color3 weight    = current.weight * comp->w;
            if (!isfinite(weight.x) || !isfinite(weight.y)
                || !isfinite(weight.z))
                return false;
            if (comp->id == 1) {
                emission += weight;
            } else {
                float probability = fabsf(weight.x) + fabsf(weight.y)
                                    + fabsf(weight.z);
                if (!probability)
                    continue;
                Vec3 n = *comp->as<Vec3>();
                if (num_lobes == 32 || !isfinite(n.length2())
                    || n.length2() <= 0 || !isfinite(probability))
                    return false;
                lobes[num_lobes++] = { weight, n.normalized(), probability };
            }
        }
    }
    return true;
}



__device__ void
globals_from_hit(ShaderGlobals& sg, const Scene& scene, const Ray& ray,
                 const HartTraceHit& hit)
{
    const auto direction = ray.dual_direction();
    sg.I                 = direction.val();
    sg.dIdx              = direction.dx();
    sg.dIdy              = direction.dy();
    auto p               = ray.point(Dual2<float>(hit.t));
    sg.P                 = p.val();
    sg.N                 = scene.normal(p, sg.Ng, hit.primitive, hit.u, hit.v);
    scene.project(p, sg.N, sg.I);
    sg.dPdx = p.dx();
    sg.dPdy = p.dy();
    sg.dPdu = sg.dPdv = Vec3(0);
    const auto uv = scene.uv(p, sg.N, sg.dPdu, sg.dPdv, hit.primitive, hit.u,
                             hit.v);
    sg.u          = uv.val().x;
    sg.v          = uv.val().y;
    sg.dudx       = uv.dx().x;
    sg.dudy       = uv.dy().x;
    sg.dvdx       = uv.dx().y;
    sg.dvdy       = uv.dy().y;
    sg.dPdz = sg.dPdtime = sg.Ps = sg.dPsdx = sg.dPsdy = Vec3(0);
    sg.surfacearea = osl_hart_render_params.surfaceareas[hit.primitive];
    sg.backfacing  = sg.Ng.dot(sg.I) > 0;
    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }
    sg.raytype        = ray.raytype;
    sg.flipHandedness = sg.dPdx.cross(sg.dPdy).dot(sg.N) < 0;
}



__device__ Color3
radiance(const Scene& scene, Ray ray, Sampler& sampler, uint64_t pixel)
{
    const auto& params = osl_hart_render_params;
    Color3 result(0), throughput(1);
    unsigned previous = ~0u;
    alignas(16) unsigned char storage[1024];
    testshade::HartClosurePool pool { storage, sizeof(storage) };
    for (unsigned bounce = 0; bounce <= params.max_bounces; ++bounce) {
        auto hit = hart_trace(params.traversable, ray.origin, ray.direction, 0,
                              1.0e30f);
        if (hit.primitive != ~0u && hit.primitive == previous)
            hit = hart_trace(params.traversable, ray.origin, ray.direction,
                             nextafterf(hit.t, INFINITY), 1.0e30f);
        if (hit.primitive == ~0u)
            break;
        if (hit.primitive == previous
            || hit.material >= params.material_count) {
            atomicOr(params.textures->errors, testshade::HartInvalidRayHit);
            return Color3(0);
        }
        ShaderGlobals sg { };
        globals_from_hit(sg, scene, ray, hit);
        pool.reset();
        testshade::HartRenderState state { params.textures, &pool };
        sg.renderstate       = &state;
        const auto& material = params.materials[hit.material];
        void* group = material.local || !params.scratch
                          ? nullptr
                          : params.scratch + pixel * params.scratch_stride;
        hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
            material.callable, &sg, group, nullptr, nullptr, int(pixel),
            nullptr);
        if (!params.fused)
            hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int,
                           void*>(material.callable + 1, &sg, group, nullptr,
                                  nullptr, int(pixel), nullptr);
        DiffuseLobe lobes[32];
        unsigned num_lobes = 0;
        Color3 emission(0);
        if (pool.failed)
            return Color3(0);
        if (!collect(sg.Ci, pool, emission, lobes, num_lobes)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        result += throughput * emission;
        if (bounce == params.max_bounces || !num_lobes)
            break;
        float sum = 0;
        for (unsigned i = 0; i < num_lobes; ++i)
            sum += lobes[i].probability;
        if (!isfinite(sum)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        const Vec3 random  = sampler.get();
        const float target = random.x * sum;
        unsigned selected  = 0;
        float cumulative   = 0;
        while (selected + 1 < num_lobes
               && target >= cumulative + lobes[selected].probability)
            cumulative += lobes[selected++].probability;
        const float sample_x = fminf((target - cumulative)
                                         / lobes[selected].probability,
                                     0.99999994f);
        Vec3 direction;
        float pdf;
        Sampling::sample_cosine_hemisphere(lobes[selected].normal, sample_x,
                                           random.y, direction, pdf);
        Color3 eval(0);
        pdf = 0;
        for (unsigned i = 0; i < num_lobes; ++i) {
            const float cosine = fmaxf(0.0f, lobes[i].normal.dot(direction));
            eval += lobes[i].weight * cosine;
            pdf += (lobes[i].probability / sum) * cosine;
        }
        if (pdf <= 0)
            break;
        throughput *= eval / pdf;
        ray.radius += ray.spread * hit.t;
        ray.spread    = fmaxf(ray.spread, 1.0f);
        ray.roughness = 1;
        ray.raytype   = Ray::DIFFUSE;
        // Reconstruct on the triangle and move beyond its floating-point
        // uncertainty, including shared edges with a different primitive ID.
        const auto triangle = scene.triangles[hit.primitive];
        const Vec3 a        = scene.verts[triangle.a];
        const Vec3 b        = scene.verts[triangle.b];
        const Vec3 c        = scene.verts[triangle.c];
        const Vec3 point    = a + hit.u * (b - a) + hit.v * (c - a);
        float scale         = 0;
        for (int axis = 0; axis < 3; ++axis)
            scale = fmaxf(scale, fmaxf(fabsf(a[axis]),
                                       fmaxf(fabsf(b[axis]), fabsf(c[axis]))));
        const Vec3 normal = sg.Ng.dot(direction) >= 0 ? sg.Ng : -sg.Ng;
        ray.origin        = point + normal * (32 * FLT_EPSILON * scale);
        for (int axis = 0; axis < 3; ++axis)
            if (normal[axis] != 0)
                ray.origin[axis] = nextafterf(ray.origin[axis],
                                              copysignf(INFINITY, normal[axis]));
        ray.direction = direction;
        previous      = hit.primitive;
    }
    return result;
}

}  // namespace



extern "C" __global__ void
__raygen__osl_hart_path()
{
    const auto& params   = osl_hart_render_params;
    const auto index     = hartGetLaunchIndex();
    const auto size      = hartGetLaunchDimensions();
    const uint64_t pixel = uint64_t(index.y) * size.x + index.x;
    Scene scene { };
    scene.verts        = params.vertices;
    scene.normals      = params.normals;
    scene.uvs          = params.uvs;
    scene.triangles    = params.triangles;
    scene.n_triangles  = params.normal_indices;
    scene.uv_triangles = params.uv_indices;
    Color3 result(0);
    for (unsigned sample = 0; sample < params.aa * params.aa; ++sample) {
        Sampler sampler(index.x, index.y, sample);
        Vec3 jitter = params.no_jitter ? Vec3(0.5f) : sampler.get();
        jitter.x *= 2;
        jitter.y *= 2;
        jitter.x = jitter.x < 1 ? sqrtf(jitter.x) - 1 : 1 - sqrtf(2 - jitter.x);
        jitter.y = jitter.y < 1 ? sqrtf(jitter.y) - 1 : 1 - sqrtf(2 - jitter.y);
        const auto ray   = params.camera.get(index.x + 0.5f + jitter.x,
                                             index.y + 0.5f + jitter.y);
        const auto value = radiance(scene, ray, sampler, pixel);
        result           = OIIO::lerp(result, value, 1.0f / (sample + 1));
    }
    params.output[pixel] = result;
}
