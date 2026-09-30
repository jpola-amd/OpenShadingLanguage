# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument("testshade", type=Path)
parser.add_argument("--oslc", type=Path, required=True)
parser.add_argument("--stdosl", type=Path, required=True)
args = parser.parse_args()
testshade, oslc = args.testshade.resolve(), args.oslc.resolve()
source = Path(__file__).with_name("smoke.osl").resolve()
env = os.environ.copy()
for key in ("TESTSHADE_HART", "TESTSHADE_OPTIX", "TESTSHADE_FUSED",
            "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[key] = "0"
for key in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT", "OSL_OPTIONS"):
    env.pop(key, None)


def run(command, root):
    result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                            text=True, timeout=180)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{command}\n{output}"
    return output


with tempfile.TemporaryDirectory(prefix="osl-oiio-smoke-") as temporary:
    root = Path(temporary)
    run([str(oslc), "-I" + str(args.stdosl.resolve().parent),
         "-o", str(root / "smoke.oso"), str(source)], root)
    for optimize in ("-O0", "-O2"):
        for backend in ("CPU", "HART"):
            flags = (["--hart", "--hart-no-cache"] if backend == "HART"
                     else ["-t", "1"])
            output = run([str(testshade), *flags, optimize, "-v", "-g", "3", "2",
                          "--print", "-o", "Cout", "null", "smoke"], root)
            assert output.count("Launching HART grid") == (backend == "HART"), output
            if backend == "HART":
                assert "HART pipeline cache disabled" in output, output
            pixels = re.findall(
                r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
                output,
            )
            assert len(pixels) == 6, output
            max_error = 0.0
            for index, row in enumerate(pixels):
                x, y = index % 3, index // 3
                assert tuple(map(int, row[:2])) == (x, y), output
                u, v = x / 2, float(y)
                # testshade's named "myspace" scales Y by two.
                expected = (u, 2 * v, 2) if u < v else (u, v, 1)
                for actual, reference in zip(map(float, row[2:]), expected):
                    assert math.isfinite(actual) and math.isclose(
                        actual, reference, rel_tol=0, abs_tol=2e-6
                    ), (backend, optimize, x, y, actual, reference, output)
                    max_error = max(max_error, abs(actual - reference))
            print(f"{backend} {optimize}: 6 pixels, max error {max_error:g}")
