// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

OSL_NAMESPACE_BEGIN

struct HartTriangle;

/// Host-owned native HART resources. Clear before replacing the pipeline.
/// All device allocations, including acceleration storage, live until clear().
class HartContext {
public:
    explicit HartContext(ErrorHandler& err);
    ~HartContext();
    HartContext(const HartContext&)            = delete;
    HartContext& operator=(const HartContext&) = delete;

    bool init(int device, std::string& arch);
    void* alloc(size_t bytes);
    bool upload(void* destination, cspan<unsigned char> source);
    bool download(span<unsigned char> destination, const void* source);
    bool build_accel(cspan<Vec3> vertices, cspan<HartTriangle> triangles,
                     cspan<unsigned> material_ids, unsigned material_count);
    uint64_t traversable() const;
    bool create_pipeline(cspan<unsigned char> bitcode, string_view raygen_entry,
                         unsigned material_count);
    /// Copy host parameters, launch, and wait for completion.
    bool launch(const void* params, size_t param_bytes, unsigned width,
                unsigned height);
    /// Synchronize and release resources; failed releases remain owned for retry.
    bool clear();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

OSL_NAMESPACE_END
