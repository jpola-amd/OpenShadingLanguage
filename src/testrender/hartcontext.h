// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

OSL_NAMESPACE_BEGIN

struct HartTriangle;

/// Borrowed bitcode and its ordered direct-callable exports.
struct HartCallable {
    cspan<unsigned char> bitcode;
    std::vector<std::string> entries;
};



/// Host-owned native HART resources. Clear before replacing the pipeline.
/// All device allocations, including acceleration storage, live until clear().
class HartContext {
public:
    explicit HartContext(ErrorHandler& err);
    ~HartContext();
    HartContext(const HartContext&)            = delete;
    HartContext& operator=(const HartContext&) = delete;

    bool init(int device, std::string& arch, bool cache_enabled = true,
              bool statistics = false);
    bool make_current();
    void* alloc(size_t bytes);
    bool upload(void* destination, cspan<unsigned char> source);
    bool download(span<unsigned char> destination, const void* source);
    bool build_accel(cspan<Vec3> vertices, cspan<HartTriangle> triangles,
                     cspan<unsigned> material_ids, unsigned material_count);
    uint64_t traversable() const;
    /// Callable SBT indices follow module order, then each module's entries.
    bool create_pipeline(cspan<unsigned char> bitcode, string_view raygen_entry,
                         unsigned material_count,
                         cspan<HartCallable> callables      = { },
                         string_view secondary_raygen_entry = { });
    /// Copy host parameters, launch raygen 0 (primary) or 1, and wait.
    bool launch(const void* params, size_t param_bytes, unsigned width,
                unsigned height, unsigned raygen_index = 0);
    /// Synchronize and release resources; failed releases remain owned for retry.
    bool clear();

    /// Host ownership accounting, excluding opaque SDK/driver cache storage.
    /// Query only between synchronous operations on this context.
    struct ResourceUsage {
        size_t allocations = 0, bytes = 0, modules = 0, program_groups = 0;
        bool context = false, stream = false, pipeline = false;
    };
    ResourceUsage resource_usage() const;

    /// Host-synchronized timings, not device-event or physical stack usage.
    struct Statistics {
        double pipeline_seconds = 0, launch_seconds = 0;
        size_t launches          = 0;
        unsigned traversal_stack = 0, state_stack = 0, continuation_stack = 0;
    };
    Statistics statistics() const;
    void reset_launch_statistics();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

OSL_NAMESPACE_END
