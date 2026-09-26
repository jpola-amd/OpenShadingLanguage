// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

struct HartLeaf {
    float weight;
    color shade;
};

struct HartPacket {
    HartLeaf leaves[2];
    float coefficients[3];
    vector axis;
};
