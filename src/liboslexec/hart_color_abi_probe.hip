// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/hart_diagnostics.h>
#include "opcolor.h"

// Inspection-only bitcode: external objects keep the genuine target record
// types alive when the HIP frontend lowers ordinary member accesses to byte GEPs.
// This module is never linked into shadeops or a renderer pipeline.
extern "C" {
extern OSL_DEVICE OSL::pvt::ColorSystem osl_hart_color_abi_probe;
extern OSL_DEVICE OSL::HartDiagnosticBuffer osl_hart_diagnostic_abi_probe;

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
}
