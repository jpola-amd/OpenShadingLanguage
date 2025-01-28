// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

namespace testshade {

struct RenderParams {
    float invw;
    float invh;
    hipDeviceptr_t output_buffer;
    bool flipv;
    int fused_callable;
    uint64_t osl_printf_buffer_start;
    uint64_t osl_printf_buffer_end;
    hipDeviceptr_t color_system;

    // for transforms
    hipDeviceptr_t object2common;
    hipDeviceptr_t shader2common;
    uint64_t num_named_xforms;
    hipDeviceptr_t xform_name_buffer;
    hipDeviceptr_t xform_buffer;

    // for used-data tests
    uint64_t test_str_1;
    uint64_t test_str_2;
};



struct GenericData {
    // For shader/material callables, data points to the interactive parameter
    // data arena for that material.
    void* data;
};

struct GenericRecord {
    // __align__(
    //     OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // // What follows should duplicate GenericData
    void* data;
};

}  // namespace testshade
