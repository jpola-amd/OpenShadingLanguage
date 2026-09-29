# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import subprocess
import sys


def run(args):
    result = subprocess.run(args, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=300)
    print(result.stdout, end="")
    return result


binary, stdosl, mode = sys.argv[1:]
positive = run([binary, stdosl, mode])
if (positive.returncode
        or "HART closure inspection verified: 45 points" not in positive.stdout
        or "all 24 ramp components" not in positive.stdout):
    raise SystemExit("HART closure inspection did not complete")

negative = run([binary, stdosl, mode, "exhaust"])
if (negative.returncode != 1
        or "Launching HART grid" not in negative.stdout
        or "closure pool allocation failed" not in negative.stdout
        or "HART closure inspection verified" in negative.stdout):
    raise SystemExit("Expected explicit post-launch closure pool exhaustion")
