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
    OIIO_CHECK_EQUAL_THRESH(hit.t, t, 2e-5f);
    OIIO_CHECK_EQUAL_THRESH(hit.u, u, 2e-5f);
    OIIO_CHECK_EQUAL_THRESH(hit.v, v, 2e-5f);
    for (int c = 0; c < 3; ++c)
        OIIO_CHECK_EQUAL_THRESH(hit.normal[c], normal[c], 2e-5f);
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

}  // namespace



int
main()
{
    Diagnostics diagnostics;
    OIIO_CHECK_ASSERT(probe_scene(diagnostics, 3));
    OIIO_CHECK_ASSERT(probe_scene(diagnostics, 1));
    OIIO_CHECK_ASSERT(probe_scene(diagnostics, 0));
    OIIO_CHECK_EQUAL(diagnostics.errors, 0);
    reject_invalid_geometry(diagnostics);
    OIIO_CHECK_EQUAL(diagnostics.errors, 3);
    return unit_test_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
