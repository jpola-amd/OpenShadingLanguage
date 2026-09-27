// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "../testshade/harttextureparams.h"
#include "../testshade/render_state.h"
#include <OSL/hart_diagnostics.h>
#include "opcolor.h"

// Inspection-only bitcode: external objects keep the genuine target record
// types alive when the HIP frontend lowers ordinary member accesses to byte GEPs.
// This module is never linked into shadeops or a renderer pipeline.
extern "C" {
extern OSL_DEVICE OSL::pvt::ColorSystem osl_hart_color_abi_probe;
extern OSL_DEVICE OSL::HartDiagnosticBuffer osl_hart_diagnostic_abi_probe;
extern OSL_DEVICE testshade::HartTextureState osl_hart_texture_abi_probe;
extern OSL_DEVICE testshade::HartUserdataState osl_hart_userdata_abi_probe;
extern OSL_DEVICE testshade::HartUserdataDesc osl_hart_userdata_desc_abi_probe;
extern OSL_DEVICE RenderContext osl_hart_attributes_abi_probe;
extern OSL_DEVICE testshade::HartTransformState osl_hart_transform_abi_probe;
extern OSL_DEVICE testshade::HartTransformDesc osl_hart_transform_desc_abi_probe;

OSL_DEVICE const void*
osl_hart_color_abi_address()
{
    return &osl_hart_color_abi_probe;
}



OSL_DEVICE const void*
osl_hart_diagnostic_abi_address()
{
    return &osl_hart_diagnostic_abi_probe;
}



OSL_DEVICE void
osl_hart_userdata_abi_addresses(const void** addresses)
{
    addresses[0] = &osl_hart_texture_abi_probe;
    addresses[1] = &osl_hart_userdata_abi_probe;
    addresses[2] = &osl_hart_userdata_desc_abi_probe;
}



OSL_DEVICE const void*
osl_hart_attributes_abi_address()
{
    return &osl_hart_attributes_abi_probe;
}



OSL_DEVICE void
osl_hart_transform_abi_addresses(const void** addresses)
{
    addresses[0] = &osl_hart_transform_abi_probe;
    addresses[1] = &osl_hart_transform_desc_abi_probe;
}
}
