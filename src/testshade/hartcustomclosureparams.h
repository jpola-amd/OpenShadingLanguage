// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <cstdint>

namespace custom_closure_test {

constexpr int id32 = 1009, id64 = 1013;
constexpr unsigned cases    = 9;
constexpr unsigned sentinel = 0x13579bdf, tail = 0x2468ace0;
constexpr unsigned completed         = 0x48415254;
constexpr unsigned allocation_failed = 1, bad_state = 2, bad_layout = 4,
                   exhausted = 8, bad_tree = 16;

template<unsigned Alignment> struct alignas(Alignment) Payload {
    unsigned guard;
    int token;
    float gain;
    OSL::Color3 tint;
    float keyword;
    unsigned end_guard;

    OSL_HOSTDEVICE Payload()
        : guard(sentinel)
        , token(-1)
        , gain(-2)
        , tint(-3)
        , keyword(17.5f)
        , end_guard(tail)
    {
    }
};

using Payload32 = Payload<32>;
using Payload64 = Payload<64>;
static_assert(sizeof(Payload32) == 32 && alignof(Payload32) == 32);
static_assert(sizeof(Payload64) == 64 && alignof(Payload64) == 64);

struct Leaf {
    int id;
    unsigned guard, end_guard, payload_alignment, payload_remainder,
        component_remainder, header_bytes;
    int token;
    float gain, tint[3], keyword, weight[3];
};

struct Result {
    unsigned done, errors, component_calls, unweighted_calls, weighted_calls,
        node_calls, add_nodes, mul_nodes, leaves, zero_probe_null,
        zero_probe_calls, sg_init_checked, sg_allocator_checks;
    Leaf leaf[4];
};

struct Params {
    Result* results;
    unsigned char* groupdata;
    unsigned group_stride;
    float zero_weight[3];
};

}  // namespace custom_closure_test
