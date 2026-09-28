// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <hip/hip_runtime.h>

#include <OSL/hart_diagnostics.h>
#include <OSL/shaderglobals.h>

#include "../hartrenderstate.h"

extern "C" __device__ void
rs_hart_diagnostic(void* ec, OSL::ustringhash_pod format, int count,
                   const unsigned char* types, int bytes,
                   const unsigned char* values, int severity,
                   OSL::ustringhash_pod shader, OSL::ustringhash_pod source,
                   int line, int shade_index)
{
    const auto* sg    = static_cast<const OSL::ShaderGlobals*>(ec);
    const auto* state = static_cast<const testshade::HartRenderState*>(
                            sg->renderstate)
                            ->textures;
    auto* buffer      = state->diagnostics;
    if (!buffer || count < 0 || unsigned(count) > OSL::HartDiagnosticMaxArgs
        || bytes < 0 || unsigned(bytes) > OSL::HartDiagnosticMaxValues
        || (count && (!types || !values)) || severity < 0 || severity > 2
        || shade_index < 0) {
        atomicOr(state->errors, testshade::HartInvalidDiagnostic);
        return;
    }
    unsigned expected = 0;
    for (int i = 0; i < count; ++i) {
        switch (OSL::EncodedType(types[i])) {
        case OSL::EncodedType::kUstringHash: expected += 8; break;
        case OSL::EncodedType::kInt32:
        case OSL::EncodedType::kUInt32:
        case OSL::EncodedType::kFloat: expected += 4; break;
        default:
            atomicOr(state->errors, testshade::HartInvalidDiagnostic);
            return;
        }
    }
    if (expected != unsigned(bytes)) {
        atomicOr(state->errors, testshade::HartInvalidDiagnostic);
        return;
    }
    if (severity == int(OSL::HartDiagnosticSeverity::Error))
        atomicOr(state->errors, testshade::HartShaderError);
    unsigned slot = atomicAdd(&buffer->count, 0u);
    while (slot < OSL::HartDiagnosticCapacity) {
        const unsigned previous = atomicCAS(&buffer->count, slot, slot + 1);
        if (previous == slot) {
            auto& record       = buffer->records[slot];
            record.format      = format;
            record.shader      = shader;
            record.source      = source;
            record.line        = line;
            record.severity    = severity;
            record.shade_index = unsigned(shade_index);
            record.arg_count   = unsigned(count);
            record.arg_bytes   = unsigned(bytes);
            for (int i = 0; i < count; ++i)
                record.arg_types[i] = types[i];
            for (int i = 0; i < bytes; ++i)
                record.arg_values[i] = values[i];
            return;
        }
        slot = previous;
    }
    atomicOr(state->errors, testshade::HartDiagnosticOverflow);
}
