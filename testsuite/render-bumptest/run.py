#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

failthresh = 0.01
failpercent = 0.5
hardfail = 0.035

# The boundary-win reference uses robust transmitted-ray origin offsets.
outputs = [ "out.exr" ]
# Keep the fixture's explicit LLVM level in HART variants too.
if int(os.environ.get('TESTSHADE_HART') or 0) :
    os.environ.pop("TESTSHADE_LLVM_OPT", None)
command = testrender("-r 128 128 -aa 4 --llvm_opt 13 bumptest.xml out.exr")

# Note: we pick this test arbitrarily as the one to verify llvm_opt=13 works
