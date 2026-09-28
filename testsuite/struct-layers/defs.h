// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


struct coords {
    float s, t;
};

struct coords_packet {
    coords values[2];
};

struct shading_result {
   closure color Cout;
   color Copac;
};
