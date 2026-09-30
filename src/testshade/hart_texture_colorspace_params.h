// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cstdint>

namespace texture_colorspace_test {

enum Slot : unsigned {
    omitted,
    empty,
    raw,
    srgb,
    bound,
    varying,
    interactive,
    invalid,
    alpha_only,
    slots
};

constexpr unsigned points = 2, values_per_slot = 12;
constexpr unsigned value_count = slots * values_per_slot;
constexpr unsigned completed = 0x48415254, guard = 0x13579bdf;
constexpr unsigned invalid_source = 1, invalid_call = 2;
constexpr uint64_t texture_id = 1;

struct Output {
    unsigned head;
    float values[value_count];
    unsigned tail;
};

struct Result {
    unsigned done, errors;
    unsigned calls[slots], gradients[slots];
    uint64_t colorspaces[slots], filenames[slots], handles[slots];
    uint64_t diagnostics[slots];
};

struct Params {
    Output* outputs;
    Result* results;
    unsigned char* groupdata;
    void* interactive;
    uint64_t texture;
    unsigned group_stride;
};

}  // namespace texture_colorspace_test
