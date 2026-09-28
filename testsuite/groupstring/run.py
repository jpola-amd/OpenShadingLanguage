#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

command += testshade('-v --oslquery -group "' +
                         'shader a alayer, ' +
                         'shader b blayer, ' +
                         'connect alayer.f_out blayer.f_in, ' +
                         'connect alayer.c_out blayer.c_in"')

if int(os.environ.get("TESTSHADE_HART") or 0):
    # SDK cache/timing information is not part of the group/query contract.
    filter_re = r"^(?!INFO: )"
