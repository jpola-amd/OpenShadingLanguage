// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <memory>
#include <string>

#include "harttextureparams.h"

OSL_NAMESPACE_BEGIN

class ShadingSystem;

struct HartUserdataBinding {
    std::string name;
    TypeDesc type;
    bool derivatives = false;
    size_t stride    = 0;
    cspan<std::byte> data;
    cspan<uint8_t> present;
};

struct HartTransformBinding {
    ustringhash name;
    Matrix44 forward { 1 }, inverse { 1 };
    bool has_forward = false, has_inverse = false;
};

// Test renderer resources, used only between synchronous HART launches.
// Files are read as numeric data: first subimage, no color conversion,
// existing mips followed by box-filtered levels down to 1x1. Missing channels
// are zero, including alpha. Full, shifted windows are supported, crops are not.
class HartTextureStore {
public:
    explicit HartTextureStore(OIIO::ErrorHandler& err);
    ~HartTextureStore();
    HartTextureStore(const HartTextureStore&)            = delete;
    HartTextureStore& operator=(const HartTextureStore&) = delete;

    // Exact filenames are cached. IDs remain stable until clear(); zero fails.
    uint64_t load(OIIO::ustring filename);
    bool prepare();
    // Refresh color-system data before each render, even for cached groups.
    bool prepare(ShadingSystem& shadingsys);
    bool prepare_userdata(cspan<HartUserdataBinding> bindings, size_t points,
                          bool grid_defaults);
    bool prepare_attributes(const RenderContext& context);
    bool prepare_transforms(cspan<HartTransformBinding> bindings,
                            ustringhash commonspace, bool unknown_error);
    bool reset_errors();
    bool check_errors();
    bool clear();

    // RendererServices hooks; allocations belong to ShaderGroup, not this store.
    void* device_alloc(int device, size_t size);
    void device_free(int device, void* ptr);
    void* copy_to_device(int device, void* dst, const void* src, size_t size);

    // Reacquire after prepare(): adding files may replace the device table.
    const testshade::HartTextureState* device_state() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

OSL_NAMESPACE_END
