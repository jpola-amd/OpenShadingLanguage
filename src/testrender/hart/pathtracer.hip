// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "../../testshade/hart/array.h"
#include "../../testshade/hart/closure.h"
#include "../../testshade/hart/color.h"
#include "../../testshade/hart/diagnostic.h"
#include "../../testshade/hart/noise.h"
#include "../../testshade/hart/texture.h"
#include "../hartpathparams.h"
#include "../sampling.h"
#include "material.h"
#include "trace.h"

#include <OSL/oslclosure.h>

#include <cfloat>

extern "C" __constant__ OSL::HartPathParams osl_hart_render_params;

using namespace OSL;

namespace {

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
    MediumStack medium_stack;
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
            material.interactive);
        if (!params.fused)
            hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int,
                           void*>(material.callable + 1, &sg, group, nullptr,
                                  nullptr, int(pixel), material.interactive);
        if (pool.failed)
            return Color3(0);
        ShadingResult shading;
        if (!hart_valid_closure(sg.Ci, pool)
            || !process_closure(sg, ray.roughness, shading, medium_stack, sg.Ci,
                                bounce == params.max_bounces)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        result += throughput * shading.Le;
        if (bounce == params.max_bounces)
            break;
        if (!shading.bsdf.prepare(-sg.I, throughput, false)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        const Vec3 random = sampler.get();
        const auto sample = shading.bsdf.sample(-sg.I, random.x, random.y,
                                                random.z);
        if (isnan(sample.pdf) || sample.pdf < 0) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        if (sample.pdf == 0)
            break;
        if (!isfinite(sample.weight.x) || !isfinite(sample.weight.y)
            || !isfinite(sample.weight.z) || !isfinite(sample.roughness)
            || !isfinite(sample.wi.x) || !isfinite(sample.wi.y)
            || !isfinite(sample.wi.z)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        throughput *= sample.weight;
        if (!(throughput.x > 0) && !(throughput.y > 0) && !(throughput.z > 0))
            break;
        const Vec3 direction = sample.wi;
        if (sg.Ng.dot(direction) < 0) {
            if (!sg.backfacing) {
                if (!medium_stack.add_medium(shading.medium_data)) {
                    atomicOr(params.textures->errors,
                             testshade::HartClosureInvalidTree);
                    return Color3(0);
                }
            } else {
                medium_stack.pop_medium();
            }
        }
        ray.radius += ray.spread * hit.t;
        ray.spread    = fmaxf(ray.spread, sample.roughness);
        ray.roughness = sample.roughness;
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
