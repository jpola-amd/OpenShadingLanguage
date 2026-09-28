// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/encodedtypes.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

OSL_NAMESPACE_BEGIN

// Experimental caller-owned HART diagnostic storage, reset between launches.
constexpr uint32_t HartDiagnosticCapacity   = 256;
constexpr uint32_t HartDiagnosticMaxArgs    = 256;
constexpr uint32_t HartDiagnosticMaxValues  = 2048;
constexpr uint32_t HartDiagnosticMaxFormat  = 4096;
constexpr uint32_t HartDiagnosticMaxMessage = 4096;
constexpr uint32_t HartDiagnosticMaxField   = 1024;

enum class HartDiagnosticSeverity : int { Print = 0, Warning = 1, Error = 2 };

struct HartDiagnosticRecord {
    uint64_t format, shader, source;
    int32_t line, severity;
    uint64_t shade_index;
    uint32_t arg_count, arg_bytes;
    uint8_t arg_types[HartDiagnosticMaxArgs];
    uint8_t arg_values[HartDiagnosticMaxValues];
};

struct HartDiagnosticBuffer {
    uint32_t count;
    uint32_t reserved;
    HartDiagnosticRecord records[HartDiagnosticCapacity];
};

static_assert(sizeof(HartDiagnosticRecord) == 2352
                  && alignof(HartDiagnosticRecord) == 8
                  && offsetof(HartDiagnosticRecord, arg_types) == 48
                  && offsetof(HartDiagnosticRecord, arg_values) == 304
                  && offsetof(HartDiagnosticBuffer, records) == 8
                  && std::is_trivial<HartDiagnosticBuffer>::value,
              "Unexpected HART diagnostic storage layout");

OSL_NAMESPACE_END
