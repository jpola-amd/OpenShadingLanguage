// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "../../testshade/hart/array.h"
#include "../../testshade/hart/closure.h"
#include "../../testshade/hart/color.h"
#include "../../testshade/hart/diagnostic.h"
#include "../../testshade/hart/noise.h"
#include "../../testshade/hart/texture.h"
#include "../background.h"
#include "../hartpathparams.h"
#include "../sampling.h"
#include "material.h"
#include "trace.h"

#include <OSL/oslclosure.h>

#include <cfloat>
#include <limits>

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



__device__ bool
hart_error(unsigned bits)
{
    atomicOr(osl_hart_render_params.textures->errors, bits);
    return false;
}



__device__ bool
finite_vec(const Vec3& value)
{ return isfinite(value.x) && isfinite(value.y) && isfinite(value.z); }



__device__ bool
valid_sample(const BSDF::Sample& sample)
{
    return (!isnan(sample.pdf) && sample.pdf >= 0 && finite_vec(sample.weight)
            && finite_vec(sample.wi) && isfinite(sample.roughness))
           || hart_error(testshade::HartClosureInvalidTree);
}



__device__ bool
shade_material(unsigned id, ShaderGlobals& sg, uint64_t pixel, int shade_index,
               testshade::HartClosurePool& pool)
{
    const auto& params = osl_hart_render_params;
    if (id >= params.material_count)
        return hart_error(testshade::HartInvalidRayHit);
    pool.reset();
    testshade::HartRenderState state { params.textures, &pool };
    sg.renderstate       = &state;
    const auto& material = params.materials[id];
    void* group          = material.local || !params.scratch
                               ? nullptr
                               : params.scratch + pixel * params.scratch_stride;
    hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
        material.callable, &sg, group, nullptr, nullptr, shade_index,
        material.interactive);
    if (!params.fused)
        hartDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
            material.callable + 1, &sg, group, nullptr, nullptr, shade_index,
            material.interactive);
    sg.renderstate = nullptr;
    return !pool.failed;
}



__device__ bool
eval_background(const Dual2<Vec3>& direction, int bounce, uint64_t pixel,
                int shade_index, testshade::HartClosurePool& pool, Vec3& value)
{
    ShaderGlobals sg { };
    sg.P = sg.dPdx = sg.dPdy = sg.dPdz = sg.N = sg.Ng = sg.dPdu = sg.dPdv
        = sg.dPdtime = sg.Ps = sg.dPsdx = sg.dPsdy = Vec3(0);
    sg.I                                           = direction.val();
    sg.dIdx                                        = direction.dx();
    sg.dIdy                                        = direction.dy();
    if (bounce >= 0)
        sg.raytype = bounce ? Ray::DIFFUSE : Ray::CAMERA;
    if (!shade_material(unsigned(osl_hart_render_params.background_material),
                        sg, pixel, shade_index, pool))
        return false;
    if (!hart_valid_closure(sg.Ci, pool, true))
        return hart_error(testshade::HartClosureInvalidTree);
    value = process_background_closure(sg.Ci);
    return finite_vec(value) || hart_error(testshade::HartClosureInvalidTree);
}



__device__ bool
trace_scene(const Vec3& origin, const Vec3& direction, float distance,
            unsigned skip, HartTraceHit& hit)
{
    const auto& params = osl_hart_render_params;
    if (!finite_vec(origin) || !finite_vec(direction)
        || !isfinite(direction.length2()) || direction.length2() <= 0
        || !isfinite(distance) || distance <= 0)
        return hart_error(testshade::HartInvalidRayHit);
    hit = hart_trace(params.traversable, origin, direction, 0, distance);
    if (hit.primitive != ~0u && hit.primitive == skip)
        hit = hart_trace(params.traversable, origin, direction,
                         nextafterf(hit.t, INFINITY), distance);
    if (hit.primitive == ~0u)
        return true;
    if (hit.primitive == skip || hit.primitive >= params.triangle_count
        || hit.material >= params.material_count || !isfinite(hit.t)
        || hit.t < 0 || !isfinite(hit.u) || !isfinite(hit.v))
        return hart_error(testshade::HartInvalidRayHit);
    return true;
}



__device__ Vec3
offset_origin(const Scene& scene, const HartTraceHit& hit, const Vec3& normal,
              const Vec3& direction)
{
    const auto triangle = scene.triangles[hit.primitive];
    const Vec3 a        = scene.verts[triangle.a];
    const Vec3 b        = scene.verts[triangle.b];
    const Vec3 c        = scene.verts[triangle.c];
    const Vec3 point    = a + hit.u * (b - a) + hit.v * (c - a);
    return scene.offset_ray_origin(hit.primitive, point, normal, direction);
}



__device__ Color3
radiance(const Scene& scene, const Background& background, Ray ray,
         Sampler& sampler, uint64_t pixel)
{
    const auto& params = osl_hart_render_params;
    Color3 result(0), throughput(1);
    float bsdf_pdf      = std::numeric_limits<float>::infinity();
    Vec3 previous_point = ray.origin;
    unsigned previous   = ~0u;
    alignas(16) unsigned char storage[1024];
    alignas(16) unsigned char light_storage[1024];
    testshade::HartClosurePool pool { storage, sizeof(storage) };
    testshade::HartClosurePool light_pool { light_storage,
                                            sizeof(light_storage) };
    MediumStack medium_stack;
    for (unsigned bounce = 0; bounce <= params.max_bounces; ++bounce) {
        HartTraceHit hit;
        if (!trace_scene(ray.origin, ray.direction, 1.0e30f, previous, hit))
            return Color3(0);
        if (hit.primitive == ~0u) {
            if (params.background_material >= 0) {
                Vec3 value;
                if (bounce && params.background_resolution) {
                    float pdf = 0;
                    value     = background.eval(ray.direction, pdf);
                    if (isnan(pdf) || pdf < 0 || !finite_vec(value)) {
                        hart_error(testshade::HartClosureInvalidTree);
                        return Color3(0);
                    }
                    result
                        += throughput * value
                           * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bsdf_pdf,
                                                                      pdf);
                } else {
                    // As on the CPU, direct misses have no direction gradients;
                    // the importance prepass supplies the map's gradients instead.
                    if (!eval_background(Dual2<Vec3>(ray.direction),
                                         int(bounce), pixel, int(pixel), pool,
                                         value))
                        return Color3(0);
                    result += throughput * value;
                }
            }
            break;
        }
        const auto event = medium_stack.integrate(ray, sampler, hit.t,
                                                  throughput, bsdf_pdf);
        if (event == MediumStack::Event::Error) {
            hart_error(testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        if (event == MediumStack::Event::Absorbed)
            break;
        if (event == MediumStack::Event::Scatter) {
            // A volume event moved the origin away from the last surface.
            previous       = ~0u;
            previous_point = ray.origin;
            continue;
        }
        ShaderGlobals sg { };
        globals_from_hit(sg, scene, ray, hit);
        if (!shade_material(hit.material, sg, pixel, int(pixel), pool))
            return Color3(0);
        ShadingResult shading;
        if (!hart_valid_closure(sg.Ci, pool)
            || !process_closure(sg, ray.roughness, shading, medium_stack, sg.Ci,
                                bounce == params.max_bounces)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        float emission_weight = 1;
        if (params.light_count && params.material_is_light[hit.material]) {
            const float pdf = scene.shapepdf(int(hit.primitive), previous_point,
                                             sg.P)
                              / params.light_count;
            if (isnan(pdf) || pdf < 0) {
                hart_error(testshade::HartInvalidRayHit);
                return Color3(0);
            }
            emission_weight = MIS::power_heuristic<MIS::WEIGHT_EVAL>(bsdf_pdf,
                                                                     pdf);
        }
        result += throughput * emission_weight * shading.Le;
        if (bounce == params.max_bounces)
            break;
        if (!shading.bsdf.prepare(-sg.I, throughput,
                                  int(bounce) >= params.rr_depth)) {
            atomicOr(params.textures->errors,
                     testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        const Vec3 random  = sampler.get();
        const float radius = ray.radius + ray.spread * hit.t;
        if (params.background_resolution) {
            Dual2<Vec3> direction;
            float pdf        = 0;
            const Vec3 value = background.sample(random.x, random.y, direction,
                                                 pdf);
            if (isnan(pdf) || pdf < 0 || !finite_vec(value)
                || !finite_vec(direction.val())) {
                hart_error(testshade::HartClosureInvalidTree);
                return Color3(0);
            }
            const auto bsdf = shading.bsdf.eval(-sg.I, direction.val());
            if (!valid_sample(bsdf))
                return Color3(0);
            const Color3 contribution
                = throughput * bsdf.weight * value
                  * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(pdf, bsdf.pdf);
            if (!finite_vec(contribution)) {
                hart_error(testshade::HartClosureInvalidTree);
                return Color3(0);
            }
            if (contribution.x + contribution.y + contribution.z > 0) {
                HartTraceHit shadow;
                if (!trace_scene(offset_origin(scene, hit, sg.Ng,
                                               direction.val()),
                                 direction.val(), 1.0e30f, hit.primitive,
                                 shadow))
                    return Color3(0);
                if (shadow.primitive == ~0u)
                    result += contribution;
            }
        }
        if (params.light_count) {
            const float pick_pdf    = 1.0f / params.light_count;
            const float scaled      = random.x * params.light_count;
            const unsigned selected = unsigned(floorf(scaled));
            if (selected >= params.light_count) {
                hart_error(testshade::HartInvalidRayHit);
                return Color3(0);
            }
            const unsigned primitive = params.light_primitives[selected];
            if (primitive >= params.triangle_count) {
                hart_error(testshade::HartInvalidRayHit);
                return Color3(0);
            }
            if (primitive != hit.primitive) {
                const auto light = scene.sample(int(primitive), sg.P,
                                                scaled - selected, random.y);
                if (!finite_vec(light.dir) || !isfinite(light.dist)
                    || light.dist <= 0 || isnan(light.pdf) || light.pdf < 0) {
                    hart_error(testshade::HartInvalidRayHit);
                    return Color3(0);
                }
                const auto bsdf = shading.bsdf.eval(-sg.I, light.dir);
                if (!valid_sample(bsdf))
                    return Color3(0);
                const Color3 contribution
                    = throughput * bsdf.weight
                      * MIS::power_heuristic<MIS::EVAL_WEIGHT>(pick_pdf
                                                                   * light.pdf,
                                                               bsdf.pdf);
                if (!finite_vec(contribution)) {
                    hart_error(testshade::HartClosureInvalidTree);
                    return Color3(0);
                }
                if (contribution.x + contribution.y + contribution.z > 0) {
                    const HartTraceHit light_hit {
                        primitive, unsigned(scene.shaderid(int(primitive))),
                        light.dist, light.u, light.v
                    };
                    ShaderGlobals light_sg { };
                    const Ray light_ray(sg.P, light.dir, radius, 0, 0,
                                        Ray::SHADOW);
                    globals_from_hit(light_sg, scene, light_ray, light_hit);
                    const Vec3 origin = offset_origin(scene, hit, sg.Ng,
                                                      light.dir);
                    const Vec3 end    = offset_origin(scene, light_hit,
                                                      light_sg.Ng, -light.dir);
                    Vec3 direction    = end - origin;
                    const float distance = direction.length();
                    if (!isfinite(distance) || distance <= 0) {
                        hart_error(testshade::HartInvalidRayHit);
                        return Color3(0);
                    }
                    direction /= distance;
                    HartTraceHit shadow;
                    if (!trace_scene(origin, direction, distance, hit.primitive,
                                     shadow))
                        return Color3(0);
                    if (shadow.primitive == ~0u
                        || shadow.primitive == primitive) {
                        // Light execution must not overwrite active surface closures.
                        if (!shade_material(light_hit.material, light_sg, pixel,
                                            int(pixel), light_pool))
                            return Color3(0);
                        ShadingResult light_result;
                        if (!hart_valid_closure(light_sg.Ci, light_pool)
                            || !process_closure(light_sg, ray.roughness,
                                                light_result, medium_stack,
                                                light_sg.Ci, true)) {
                            hart_error(testshade::HartClosureInvalidTree);
                            return Color3(0);
                        }
                        result += contribution * light_result.Le;
                    }
                }
            }
        }
        const auto sample = shading.bsdf.sample(-sg.I, random.x, random.y,
                                                random.z);
        if (!valid_sample(sample))
            return Color3(0);
        if (sample.pdf == 0)
            break;
        throughput *= sample.weight;
        if (!finite_vec(throughput)) {
            hart_error(testshade::HartClosureInvalidTree);
            return Color3(0);
        }
        bsdf_pdf = sample.pdf;
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
            } else if (!medium_stack.pop_medium()) {
                hart_error(testshade::HartClosureInvalidTree);
                return Color3(0);
            }
        }
        ray.radius     = radius;
        ray.spread     = fmaxf(ray.spread, sample.roughness);
        ray.roughness  = sample.roughness;
        ray.raytype    = Ray::DIFFUSE;
        previous_point = sg.P;
        ray.origin     = offset_origin(scene, hit, sg.Ng, direction);
        ray.direction  = direction;
        previous       = hit.primitive;
    }
    return result;
}

}  // namespace



extern "C" __global__ void
__raygen__osl_hart_background()
{
    const auto& params   = osl_hart_render_params;
    const auto index     = hartGetLaunchIndex();
    const auto size      = hartGetLaunchDimensions();
    const uint64_t pixel = uint64_t(index.y) * size.x + index.x;
    const uint64_t texel = params.background_offset + pixel;
    if (!params.background_resolution
        || texel >= uint64_t(params.background_resolution)
                        * params.background_resolution) {
        hart_error(testshade::HartInvalidRayHit);
        return;
    }
    Background background;
    background.set_variables(params.background_values, params.background_rows,
                             params.background_cols,
                             int(params.background_resolution));
    alignas(16) unsigned char storage[1024];
    testshade::HartClosurePool pool { storage, sizeof(storage) };
    Vec3 value;
    if (eval_background(
            background.map(float(texel % params.background_resolution) + 0.5f,
                           float(texel / params.background_resolution) + 0.5f),
            -1, pixel, int(texel), pool, value))
        params.output[pixel] = Color3(value.x, value.y, value.z);
}



extern "C" __global__ void
__raygen__osl_hart_path()
{
    const auto& params   = osl_hart_render_params;
    const auto index     = hartGetLaunchIndex();
    const auto size      = hartGetLaunchDimensions();
    const uint64_t pixel = uint64_t(index.y) * size.x + index.x;
    Background background;
    if (params.background_resolution)
        background.set_variables(params.background_values,
                                 params.background_rows, params.background_cols,
                                 int(params.background_resolution));
    Scene scene { };
    scene.verts        = params.vertices;
    scene.normals      = params.normals;
    scene.uvs          = params.uvs;
    scene.triangles    = params.triangles;
    scene.n_triangles  = params.normal_indices;
    scene.uv_triangles = params.uv_indices;
    scene.shaderids    = params.shader_ids;
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
        const auto value = radiance(scene, background, ray, sampler, pixel);
        result           = OIIO::lerp(result, value, 1.0f / (sample + 1));
    }
    params.output[pixel] = result;
}
