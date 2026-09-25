// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <memory>

#include "harttextureparams.h"

OSL_NAMESPACE_BEGIN

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
    bool reset_errors();
    bool check_errors();
    bool clear();

    // Reacquire after prepare(): adding files may replace the device table.
    const testshade::HartTextureState* device_state() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

OSL_NAMESPACE_END
