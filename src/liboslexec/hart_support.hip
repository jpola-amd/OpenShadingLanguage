// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Carry HART's device-storage ABI provenance into generated shader groups.
#include <amd/hart/hart_device.h>

#include <OSL/oslconfig.h>

extern "C" __device__ bool
rs_hart_get_userdata(void* ec, int shade_index, OSL::ustringhash_pod name,
                     long long type, bool derivatives, void* data);

extern "C" __device__ bool
osl_hart_get_userdata(void* ec, int shade_index, OSL::ustringhash_pod name,
                      long long type, bool derivatives, void* data)
{ return rs_hart_get_userdata(ec, shade_index, name, type, derivatives, data); }



extern "C" __device__ bool
rs_hart_get_attribute(void* ec, int shade_index, OSL::ustringhash_pod object,
                      OSL::ustringhash_pod name, long long type,
                      bool derivatives, int index, void* result);

extern "C" __device__ bool
osl_hart_get_attribute(void* ec, int shade_index, OSL::ustringhash_pod object,
                       OSL::ustringhash_pod name, long long type,
                       bool derivatives, int index, void* result)
{
    return rs_hart_get_attribute(ec, shade_index, object, name, type,
                                 derivatives, index, result);
}



extern "C" __device__ void
rs_hart_diagnostic(void* ec, OSL::ustringhash_pod format, int count,
                   const unsigned char* types, int bytes,
                   const unsigned char* values, int severity,
                   OSL::ustringhash_pod shader, OSL::ustringhash_pod source,
                   int line, int shade_index);

extern "C" __device__ void
osl_hart_diagnostic(void* ec, OSL::ustringhash_pod format, int count,
                    const unsigned char* types, int bytes,
                    const unsigned char* values, int severity,
                    OSL::ustringhash_pod shader, OSL::ustringhash_pod source,
                    int line, int shade_index)
{
    rs_hart_diagnostic(ec, format, count, types, bytes, values, severity,
                       shader, source, line, shade_index);
}
