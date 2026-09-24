// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "hashes.h"

extern "C" __device__ const unsigned long long*
second_hash()
{ return &Hashes::end; }
