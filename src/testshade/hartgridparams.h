// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

namespace testshade {

// External raygen modules declare this as the constant launch-parameter
// symbol "testshade_hart_params". Each pixel writes three consecutive floats.
struct HartGridParams {
    float* output;
};

static_assert(sizeof(void*) == 8 && sizeof(HartGridParams) == 8,
              "The HART grid ABI requires 64-bit pointers");

}  // namespace testshade
