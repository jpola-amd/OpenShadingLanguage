#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade('--param:type=string[1] values "" test')
command += testshade('--param:type=string[1] values different test')
outputs = ["out.txt"]
