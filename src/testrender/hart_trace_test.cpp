// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>

#include <OpenImageIO/unittest.h>

#include "hart_raytracer_bitcode.h"
#include "hartcontext.h"
#include "hartparams.h"
#include "raytracer.h"

using namespace OSL;

namespace {

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

    int errors = 0;
    std::string last_error;
};



template<class T>
cspan<unsigned char>
bytes(cspan<T> data)
{
    return { reinterpret_cast<const unsigned char*>(data.data()),
             data.size() * sizeof(T) };
}



template<class T, size_t N>
T*
upload(HartContext& context, const std::array<T, N>& data)
{
    auto* device = static_cast<T*>(context.alloc(sizeof(data)));
    if (!device || !context.upload(device, bytes<T>(data)))
        return nullptr;
    return device;
}



void
check_hit(const HartProbeHit& hit, unsigned primitive, unsigned material,
          float t, float u, float v, const Vec3& normal)
{
    OIIO_CHECK_EQUAL(hit.primitive, primitive);
    OIIO_CHECK_EQUAL(hit.material, material);
    OIIO_CHECK_ASSERT(std::isfinite(hit.t));
    OIIO_CHECK_ASSERT(std::isfinite(hit.u));
    OIIO_CHECK_ASSERT(std::isfinite(hit.v));
    OIIO_CHECK_EQUAL_THRESH(hit.t, t, 2e-5f);
    OIIO_CHECK_EQUAL_THRESH(hit.u, u, 2e-5f);
    OIIO_CHECK_EQUAL_THRESH(hit.v, v, 2e-5f);
    for (int c = 0; c < 3; ++c) {
        OIIO_CHECK_ASSERT(std::isfinite(hit.normal[c]));
        OIIO_CHECK_EQUAL_THRESH(hit.normal[c], normal[c], 2e-5f);
    }
}



cspan<unsigned char>
module_for(string_view arch, Diagnostics& diagnostics)
{
    for (const auto& module : hart_raytracer_modules)
        if (arch == module.arch && *module.size > 0)
            return { module.data, size_t(*module.size) };
    diagnostics.errorfmt("No embedded HART raytracer module for '{}'", arch);
    return { };
}



bool
probe_scene(Diagnostics& diagnostics, unsigned primitive_count)
{
    const bool empty = primitive_count == 0;
    HartContext context(diagnostics);
    std::string arch;
    if (!context.init(0, arch))
        return false;
    const auto module = module_for(arch, diagnostics);
    if (module.empty())
        return false;

    // The middle triangle is a translated and sheared
    // copy of the first. Its normal must not be transformed like a position.
    const std::array<Vec3, 9> vertices { Vec3(-1, -1, 2), Vec3(1, -1, 2),
                                         Vec3(-1, 1, 2),  Vec3(3, -1, 3),
                                         Vec3(5, -1, 4),  Vec3(3, 1, 3),
                                         Vec3(-1, -1, 4), Vec3(1, -1, 4),
                                         Vec3(-1, 1, 4) };
    const std::array<HartTriangle, 3> triangles { HartTriangle { 0, 1, 2 },
                                                  HartTriangle { 3, 4, 5 },
                                                  HartTriangle { 6, 7, 8 } };
    const std::array<unsigned, 3> materials { 2, 0, 1 };
    Scene cpu_scene;
    if (!empty) {
        cpu_scene.verts.assign(vertices.begin(), vertices.end());
        for (unsigned i = 0; i < primitive_count; ++i) {
            const auto& tri = triangles[i];
            cpu_scene.triangles.push_back(
                { int(tri.a), int(tri.b), int(tri.c) });
            cpu_scene.n_triangles.push_back({ -1, -1, -1 });
            cpu_scene.uv_triangles.push_back({ -1, -1, -1 });
        }
        cpu_scene.prepare(diagnostics);
    }
    if (!context.build_accel(empty ? cspan<Vec3>() : cspan<Vec3>(vertices),
                             { triangles.data(), primitive_count },
                             { materials.data(), primitive_count }, 3)
        || !context.create_pipeline(module, "__raygen__osl_hart_probe", 3))
        return false;
    OIIO_CHECK_EQUAL(context.traversable() == 0, empty);

    std::array<HartProbeRay, 9> rays {
        HartProbeRay { Vec3(-0.5f, -0.5f, 0), Vec3(0, 0, 1), 0, 100 },
        HartProbeRay { Vec3(3.5f, -0.5f, 0), Vec3(0, 0, 1), 0, 100 },
        HartProbeRay { Vec3(6, 6, 0), Vec3(0, 0, 1), 0, 100 },
        HartProbeRay { Vec3(-0.5f, -0.5f, 0), Vec3(0, 0, 1), 2.1f, 100 },
        HartProbeRay { Vec3(-0.5f, -0.5f, 0), Vec3(0, 0, 1), 0, 1.99f },
        HartProbeRay { Vec3(-0.5f, -0.5f, 5), Vec3(0, 0, -1), 0, 100 },
        HartProbeRay { Vec3(-0.5f, -0.5f, 0), Vec3(0, 0, 2), 0, 100 },
        HartProbeRay { Vec3(-0.5f, -0.5f, 0), Vec3(0, 0, 1), 4.1f, 100 },
        HartProbeRay { Vec3(-0.8f, 0.2f, 0), Vec3(0, 0, 1), 0, 100 }
    };
    HartProbeParams params { };
    params.traversable = context.traversable();
    auto* device_rays  = upload(context, rays);
    params.rays        = device_rays;
    params.vertices    = upload(context, vertices);
    params.triangles   = upload(context, triangles);
    params.ray_count   = rays.size();
    std::array<HartProbeHit, 9> hits;
    params.hits = static_cast<HartProbeHit*>(context.alloc(sizeof(hits)));
    if (!params.rays || !params.vertices || !params.triangles || !params.hits)
        return false;

    for (int pass = 0; pass < 3; ++pass) {
        // A-B-A launch data rebinding: the middle pass swaps hit and miss rays.
        if (pass)
            std::swap(rays[0], rays[2]);
        if (!context.upload(device_rays, bytes<HartProbeRay>(rays))
            || !context.launch(&params, sizeof(params), 3, 3)
            || !context.download({ reinterpret_cast<unsigned char*>(hits.data()),
                                   sizeof(hits) },
                                 params.hits))
            return false;

        if (empty) {
            for (size_t i = 0; i < hits.size(); ++i)
                check_hit(hits[i], ~0u, ~0u, rays[i].tmax, 0, 0, Vec3(0));
        } else {
            for (size_t i = 0; i < rays.size(); ++i) {
                const auto& r = rays[i];
                Ray cpu_ray(r.origin + r.tmin * r.direction, r.direction, 0, 0,
                            0, Ray::CAMERA);
                const auto hit = cpu_scene.intersect(cpu_ray, r.tmax - r.tmin,
                                                     ~0u);
                if (hit.t < r.tmax - r.tmin) {
                    Vec3 ng;
                    Vec3 normal
                        = cpu_scene.normal(Dual2<Vec3>(cpu_ray.point(hit.t)),
                                           ng, hit.id, hit.u, hit.v);
                    check_hit(hits[i], hit.id, materials[hit.id],
                              hit.t + r.tmin, hit.u, hit.v, normal);
                } else {
                    check_hit(hits[i], ~0u, ~0u, r.tmax, 0, 0, Vec3(0));
                }
            }
            const Vec3 n(0, 0, 1);
            check_hit(hits[pass == 1 ? 2 : 0], 0, 2, 2, 0.25f, 0.25f, n);
            if (primitive_count == 3) {
                check_hit(hits[1], 1, 0, 3.25f, 0.25f, 0.25f,
                          Vec3(-1, 0, 2).normalized());
                check_hit(hits[3], 2, 1, 4, 0.25f, 0.25f, n);
                check_hit(hits[5], 2, 1, 1, 0.25f, 0.25f, n);
            } else {
                check_hit(hits[1], ~0u, ~0u, 100, 0, 0, Vec3(0));
                check_hit(hits[3], ~0u, ~0u, 100, 0, 0, Vec3(0));
                check_hit(hits[5], 0, 2, 3, 0.25f, 0.25f, n);
            }
            check_hit(hits[pass == 1 ? 0 : 2], ~0u, ~0u, 100, 0, 0, Vec3(0));
            check_hit(hits[4], ~0u, ~0u, 1.99f, 0, 0, Vec3(0));
            check_hit(hits[6], 0, 2, 1, 0.25f, 0.25f, n);
            check_hit(hits[7], ~0u, ~0u, 100, 0, 0, Vec3(0));
            check_hit(hits[8], 0, 2, 2, 0.1f, 0.6f, n);
        }
    }
    print("HART scene with {} triangles: 27 numerical ray checks on {}\n",
          primitive_count, arch);
    return context.clear();
}



void
reject_invalid_geometry(Diagnostics& diagnostics)
{
    HartContext context(diagnostics);
    std::string arch;
    if (!context.init(0, arch)) {
        OIIO_CHECK_ASSERT(false);
        return;
    }
    std::array<Vec3, 3> vertices { Vec3(0), Vec3(1, 0, 0), Vec3(0, 1, 0) };
    std::array<HartTriangle, 1> triangles { HartTriangle { 0, 1, 3 } };
    std::array<unsigned, 1> materials { 0 };
    auto reject = [&](const char* expected) {
        const int before = diagnostics.errors;
        OIIO_CHECK_ASSERT(
            !context.build_accel(vertices, triangles, materials, 1));
        OIIO_CHECK_ASSERT(diagnostics.errors > before);
        OIIO_CHECK_ASSERT(diagnostics.last_error.find(expected)
                          != std::string::npos);
        OIIO_CHECK_EQUAL(context.traversable(), uint64_t(0));
    };
    reject("index");
    triangles[0].c = 2;
    materials[0]   = 1;
    reject("material");
    materials[0]  = 0;
    vertices[0].x = std::numeric_limits<float>::infinity();
    reject("finite");
    OIIO_CHECK_ASSERT(context.clear());
}



void
check_resources(const HartContext::ResourceUsage& actual,
                const HartContext::ResourceUsage& expected = { })
{
    OIIO_CHECK_EQUAL(actual.allocations, expected.allocations);
    OIIO_CHECK_EQUAL(actual.bytes, expected.bytes);
    OIIO_CHECK_EQUAL(actual.modules, expected.modules);
    OIIO_CHECK_EQUAL(actual.program_groups, expected.program_groups);
    OIIO_CHECK_EQUAL(actual.context, expected.context);
    OIIO_CHECK_EQUAL(actual.stream, expected.stream);
    OIIO_CHECK_EQUAL(actual.pipeline, expected.pipeline);
}



bool
prepare_probe(HartContext& context, cspan<unsigned char> module, float distance,
              unsigned material, HartProbeParams& params)
{
    const std::array<Vec3, 3> vertices { Vec3(-1, -1, distance),
                                         Vec3(1, -1, distance),
                                         Vec3(-1, 1, distance) };
    const std::array<HartTriangle, 1> triangles { HartTriangle { 0, 1, 2 } };
    const std::array<unsigned, 1> materials { material };
    const std::array<HartProbeRay, 2> rays {
        HartProbeRay { Vec3(-0.5f, -0.5f, 0), Vec3(0, 0, 1), 0, 100 },
        HartProbeRay { Vec3(4, 4, 0), Vec3(0, 0, 1), 0, 100 }
    };
    if (!context.build_accel(vertices, triangles, materials, 3)
        || !context.create_pipeline(module, "__raygen__osl_hart_probe", 3))
        return false;
    params             = { };
    params.traversable = context.traversable();
    params.rays        = upload(context, rays);
    params.vertices    = upload(context, vertices);
    params.triangles   = upload(context, triangles);
    params.ray_count   = rays.size();
    params.hits        = static_cast<HartProbeHit*>(
        context.alloc(rays.size() * sizeof(HartProbeHit)));
    return params.rays && params.vertices && params.triangles && params.hits;
}



bool
launch_probe(HartContext& context, const HartProbeParams& params,
             float distance, unsigned material)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::array<HartProbeHit, 2> hits {
        HartProbeHit { 99, 99, nan, nan, nan, Vec3(nan) },
        HartProbeHit { 99, 99, nan, nan, nan, Vec3(nan) }
    };
    if (!context.upload(params.hits, bytes<HartProbeHit>(hits))
        || !context.launch(&params, sizeof(params), 2, 1)
        || !context.download({ reinterpret_cast<unsigned char*>(hits.data()),
                               sizeof(hits) },
                             params.hits))
        return false;
    check_hit(hits[0], 0, material, distance, 0.25f, 0.25f, Vec3(0, 0, 1));
    check_hit(hits[1], ~0u, ~0u, 100, 0, 0, Vec3(0));
    return true;
}



bool
check_lifecycle(Diagnostics& diagnostics)
{
    HartContext resident(diagnostics), reused(diagnostics);
    check_resources(resident.resource_usage());
    check_resources(reused.resource_usage());
    std::string arch;
    if (!resident.init(0, arch))
        return false;
    const auto module = module_for(arch, diagnostics);
    HartProbeParams resident_params { };
    if (module.empty()
        || !prepare_probe(resident, module, 7, 2, resident_params)
        || !launch_probe(resident, resident_params, 7, 2))
        return false;
    const auto resident_usage = resident.resource_usage();
    OIIO_CHECK_ASSERT(resident_usage.allocations > 0);
    OIIO_CHECK_ASSERT(resident_usage.bytes > 0);
    OIIO_CHECK_EQUAL(resident_usage.modules, size_t(1));
    OIIO_CHECK_EQUAL(resident_usage.program_groups, size_t(3));
    OIIO_CHECK_ASSERT(resident_usage.context && resident_usage.stream
                      && resident_usage.pipeline);
    OIIO_CHECK_EQUAL(resident.statistics().launches, size_t(0));
    OIIO_CHECK_EQUAL(resident.statistics().pipeline_seconds, 0.0);
    int expected_errors = 0;
    auto reject         = [&](auto operation, string_view expected) {
        OIIO_CHECK_EQUAL(diagnostics.errors, expected_errors);
        const int before = diagnostics.errors;
        OIIO_CHECK_ASSERT(!operation());
        OIIO_CHECK_ASSERT(diagnostics.errors > before);
        OIIO_CHECK_ASSERT(diagnostics.last_error.find(std::string(expected))
                          != std::string::npos);
        expected_errors = diagnostics.errors;
    };
    reject([&]() { return reused.init(-1, arch); }, "nonnegative");
    check_resources(reused.resource_usage());
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
        if (!reused.init(0, arch, cycle != 0, true))
            return false;
        HartProbeParams params { };
        const float distance = 2.0f + cycle;
        if (!prepare_probe(reused, module, distance, cycle, params)
            || !launch_probe(reused, params, distance, cycle))
            return false;
        const auto usage       = reused.resource_usage();
        const auto preparation = reused.statistics();
        OIIO_CHECK_ASSERT(std::isfinite(preparation.pipeline_seconds)
                          && preparation.pipeline_seconds >= 0);
        OIIO_CHECK_ASSERT(std::isfinite(preparation.launch_seconds)
                          && preparation.launch_seconds >= 0);
        OIIO_CHECK_EQUAL(preparation.launches, size_t(1));
        reused.reset_launch_statistics();
        OIIO_CHECK_EQUAL(reused.statistics().launches, size_t(0));
        OIIO_CHECK_EQUAL(reused.statistics().launch_seconds, 0.0);
        OIIO_CHECK_EQUAL(reused.statistics().pipeline_seconds,
                         preparation.pipeline_seconds);
        OIIO_CHECK_EQUAL(reused.statistics().state_stack,
                         preparation.state_stack);
        for (int repeat = 0; repeat < 3; ++repeat) {
            if (!launch_probe(resident, resident_params, 7, 2)
                || !launch_probe(reused, params, distance, cycle))
                return false;
            check_resources(resident.resource_usage(), resident_usage);
            check_resources(reused.resource_usage(), usage);
            OIIO_CHECK_EQUAL(reused.statistics().launches, size_t(repeat + 1));
            OIIO_CHECK_ASSERT(std::isfinite(reused.statistics().launch_seconds)
                              && reused.statistics().launch_seconds >= 0);
        }
        const std::array<unsigned char, 1> byte { 0 };
        reject([&]() { return reused.upload(resident_params.hits, byte); },
               "owned device allocation");
        reject([&]() { return reused.alloc(0) != nullptr; }, "zero-byte");
        reject([&]() { return reused.init(0, arch); }, "before initializing");
        reject([&]() { return reused.launch(&params, sizeof(params), 0, 1); },
               "launch parameters");
        reject([&]() { return reused.launch(&params, sizeof(params), 2, 1, 1); },
               "raygen index");
        check_resources(reused.resource_usage(), usage);
        OIIO_CHECK_EQUAL(reused.statistics().launches, size_t(3));
        if (!launch_probe(resident, resident_params, 7, 2)
            || !launch_probe(reused, params, distance, cycle))
            return false;
        OIIO_CHECK_EQUAL(reused.statistics().launches, size_t(4));
        if (!reused.clear())
            return false;
        const auto cleared = reused.statistics();
        OIIO_CHECK_EQUAL(cleared.pipeline_seconds, 0.0);
        OIIO_CHECK_EQUAL(cleared.launch_seconds, 0.0);
        OIIO_CHECK_EQUAL(cleared.launches, size_t(0));
        OIIO_CHECK_EQUAL(cleared.traversal_stack, 0u);
        OIIO_CHECK_EQUAL(cleared.state_stack, 0u);
        OIIO_CHECK_EQUAL(cleared.continuation_stack, 0u);
        OIIO_CHECK_EQUAL(reused.traversable(), uint64_t(0));
        check_resources(reused.resource_usage());
        OIIO_CHECK_ASSERT(reused.clear());
        check_resources(reused.resource_usage());
        reject([&]() { return reused.launch(&params, sizeof(params), 2, 1); },
               "not initialized");
        if (!launch_probe(resident, resident_params, 7, 2))
            return false;
        check_resources(resident.resource_usage(), resident_usage);
    }
    if (!reused.init(0, arch) || !reused.build_accel({ }, { }, { }, 3))
        return false;
    // SDKs may reject the missing export at group creation or defer linking
    // until pipeline creation/the stack query. Each leaves real owned objects.
    reject(
        [&]() {
            return reused.create_pipeline(module, "__raygen__osl_hart_probe", 3,
                                          { }, "__raygen__missing_lifecycle");
        },
        "failed:");
    const auto partial          = reused.resource_usage();
    const bool group_failure    = diagnostics.last_error.find(
                                      "hartProgramGroupCreate")
                                  != std::string::npos;
    const bool pipeline_failure = diagnostics.last_error.find(
                                      "hartPipelineCreate")
                                  != std::string::npos;
    const bool stack_failure    = diagnostics.last_error.find(
                                      "hartUtilAccumulateStackSizes")
                                  != std::string::npos;
    OIIO_CHECK_ASSERT(group_failure || pipeline_failure || stack_failure);
    OIIO_CHECK_EQUAL(partial.modules, size_t(1));
    OIIO_CHECK_EQUAL(partial.program_groups, size_t(group_failure ? 3 : 4));
    OIIO_CHECK_EQUAL(partial.pipeline, stack_failure);
    reject(
        [&]() {
            return reused.launch(&resident_params, sizeof(resident_params), 2,
                                 1);
        },
        "pipeline and built scene");
    if (!reused.clear())
        return false;
    check_resources(reused.resource_usage());
    if (!reused.init(0, arch))
        return false;
    HartProbeParams recovered { };
    if (!prepare_probe(reused, module, 5, 1, recovered)
        || !launch_probe(reused, recovered, 5, 1) || !reused.clear())
        return false;
    check_resources(reused.resource_usage());
    if (!launch_probe(resident, resident_params, 7, 2) || !resident.clear())
        return false;
    check_resources(resident.resource_usage());
    OIIO_CHECK_EQUAL(diagnostics.errors, expected_errors);
    if (!unit_test_failures)
        print("HART native context lifecycle: repeated reuse, isolated "
              "ownership, bounded rejections and partial-pipeline recovery "
              "on {}\n",
              arch);
    return true;
}

}  // namespace



int
main(int argc, const char* argv[])
{
    Diagnostics diagnostics;
    if (argc == 2 && string_view(argv[1]) == "--lifecycle") {
        OIIO_CHECK_ASSERT(check_lifecycle(diagnostics));
        return unit_test_failures ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    if (argc != 1) {
        print(stderr, "Usage: hart_trace_test [--lifecycle]\n");
        return EXIT_FAILURE;
    }
    OIIO_CHECK_ASSERT(probe_scene(diagnostics, 3));
    OIIO_CHECK_ASSERT(probe_scene(diagnostics, 1));
    OIIO_CHECK_ASSERT(probe_scene(diagnostics, 0));
    OIIO_CHECK_EQUAL(diagnostics.errors, 0);
    reject_invalid_geometry(diagnostics);
    OIIO_CHECK_EQUAL(diagnostics.errors, 3);
    return unit_test_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
