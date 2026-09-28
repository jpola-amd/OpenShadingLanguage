// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "hartcontext.h"
#include "simpleraytracer.h"

#include <cstddef>
#include <memory>

OSL_NAMESPACE_BEGIN

/// Native HART renderer. Initialize before constructing its ShadingSystem.
/// Check failed() after the inherited void lifecycle operations.
/// Background values are shaded on HART in bounded batches each render; only
/// their importance CDFs are prepared on the host. Surface and light calls
/// retain independent closure pools. Displacement is unsupported.
/// Errors are terminal for this renderer; clear() releases resources and
/// invalidates pixels, but does not reset the inherited error history.
class HartRaytracer final : public SimpleRaytracer {
public:
    HartRaytracer();
    ~HartRaytracer() override;

    bool initialize(int device, bool fused, size_t local_budget,
                    bool cache_enabled = true, bool statistics = false);
    bool failed() const;
    /// Native context ownership only; excludes textures and group allocations.
    HartContext::ResourceUsage resource_usage() const;
    void reset_launch_statistics();
    void print_statistics() const;

    int supports(string_view feature) const override;
    TextureHandle* get_texture_handle(ustring filename, ShadingContext* context,
                                      const TextureOpt* options) override;
    bool good(TextureHandle* handle) override;
    void* device_alloc(size_t size) override;
    void device_free(void* ptr) override;
    void* copy_to_device(void* dst, const void* src, size_t size) override;

    void prepare_render() override;
    void render(int xres, int yres) override;
    void warmup() override;
    void finalize_pixel_buffer() override;
    void clear() override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

OSL_NAMESPACE_END
