// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

namespace {
namespace Hashes {
__device__ const unsigned long long end = 42;
}

// A host reference makes Clang externalize this translation-unit-local
// device constant. Separate compilation must keep its device name unique.
const unsigned long long*
host_address()
{ return &Hashes::end; }
}  // namespace
