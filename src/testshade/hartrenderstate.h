// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include "harttextureparams.h"

namespace testshade {

// Storage belongs to the caller, not a shader's callable-local Groupdata.
struct HartClosurePool {
    unsigned char* data;
    size_t capacity;
    size_t used = 0;
    bool failed = false;

    OSL_HOSTDEVICE void reset()
    {
        used   = 0;
        failed = false;
    }

    OSL_HOSTDEVICE void* allocate(size_t size, size_t alignment)
    {
        if (failed || !data || !size || !alignment
            || (alignment & (alignment - 1)) || used > capacity) {
            failed = true;
            return nullptr;
        }
        const uintptr_t address = reinterpret_cast<uintptr_t>(data + used);
        const size_t padding    = (0 - address) & (alignment - 1);
        const size_t remaining  = capacity - used;
        if (padding > remaining || size > remaining - padding) {
            failed = true;
            return nullptr;
        }
        used += padding;
        void* result = data + used;
        used += size;
        return result;
    }
};

struct HartRenderState {
    const HartTextureState* textures;
    HartClosurePool* closure_pool;
};

}  // namespace testshade
