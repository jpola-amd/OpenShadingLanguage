// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "simpleraytracer.h"

#include <cstddef>
#include <memory>

OSL_NAMESPACE_BEGIN

/// Native HART renderer. Initialize before constructing its ShadingSystem.
/// Check failed() after the inherited void lifecycle operations.
class HartRaytracer final : public SimpleRaytracer {
public:
    HartRaytracer();
    ~HartRaytracer() override;

    bool initialize(int device, bool fused, size_t local_budget);
    bool failed() const;

    int supports(string_view feature) const override;
    TextureHandle* get_texture_handle(ustring filename, ShadingContext* context,
                                      const TextureOpt* options) override;
    bool good(TextureHandle* handle) override;

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
