// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "opcolor.h"

// Inspection-only bitcode: the external object keeps the genuine target record
// type alive when the HIP frontend lowers ordinary member accesses to byte GEPs.
// This module is never linked into shadeops or a renderer pipeline.
extern "C" {
extern OSL_DEVICE OSL::pvt::ColorSystem osl_hart_color_abi_probe;

OSL_DEVICE const void*
osl_hart_color_abi_address()
{ return &osl_hart_color_abi_probe; }
}
