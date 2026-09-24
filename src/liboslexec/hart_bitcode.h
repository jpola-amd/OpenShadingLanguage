// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

OSL_NAMESPACE_BEGIN
namespace pvt {

// The returned bytes live as long as liboslexec. Unavailable architectures
// report an error and return an empty span.
OSLEXECPUBLIC cspan<unsigned char>
hart_shadeops_bitcode(string_view arch, ErrorHandler& errhandler);

}  // namespace pvt
OSL_NAMESPACE_END
