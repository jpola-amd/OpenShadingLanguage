# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("testshade")
parser.add_argument("--oslc", required=True)
args = parser.parse_args()
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = {**os.environ, "TESTSHADE_OPTIX": "0", "TESTSHADE_BATCHED": "0",
       "TESTSHADE_RS_BITCODE": "0", "TESTSHADE_HART": "0", "TESTSHADE_FUSED": "0"}


def run(executable, arguments, error=None):
    result = subprocess.run([executable] + arguments, cwd=root, env=env,
                            capture_output=True, text=True, timeout=300)
    text = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, text
    else:
        assert result.returncode != 0 and error in text, text
        assert "Launching HART grid" not in text, text
    return text


def check(text, supplied):
    pixels = re.findall(r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s*([^\r\n]+)", text)
    assert len(pixels) == 25, text
    for n, (px, py, value) in enumerate(pixels):
        x, y = int(px), int(py)
        assert (x, y) == (n % 5, n // 5)
        u, v = x/4, y/4
        red = u if u > .5 else (9 if supplied else 2)
        green = v if u < .5 else 3
        blue = 1-u if int(12*v) % 2 == 0 else 4
        weight = (7 if u < .5 else 11) if supplied else (2 if u < .5 else 3)
        q = u + 2*v + red + 3*green + 5*blue
        q += (5 if supplied else 2)*weight + (3 if supplied else 1)
        q += (101+103+3) if supplied else (3+4)
        q += int(4*u)
        dx = .25 * (1 + (u > .5) - 5*(int(12*v) % 2 == 0))
        dy = .25 * (2 + 3*(u < .5))
        values = list(map(float, value.split()))
        assert len(values) == 3, (x, y, values, text)
        for actual, expected in zip(values, (q, dx, dy)):
            assert math.isclose(actual, expected, abs_tol=2e-5), (x,y,actual,expected,text)


with tempfile.TemporaryDirectory(prefix="osl-hart-userdata-") as temporary:
    root = Path(temporary)
    source = root / "hart_userdata_cli.osl"
    source.write_text("""
        shader hart_userdata_cli(
            int face_idx=-1 [[int interpolated=1]],
            float s=.25 [[int interpolated=1]],
            float t=.75 [[int interpolated=1]],
            float red=2 [[int interpolated=1]],
            float green=3 [[int interpolated=1]],
            float blue=4 [[int interpolated=1]],
            float gain=2 [[int interpolated=1]],
            float weights[2]={2,3} [[int interpolated=1]],
            matrix M=1 [[int interpolated=1]],
            int ids[2]={3,4} [[int interpolated=1]],
            string label="missing" [[int interpolated=1]],
            string tags[2]={"no","no"} [[int interpolated=1]],
            output color Cout=0) {
            float q=s+2*t+red+3*green+5*blue+gain*weights[int(u+.5)]
                    +M[0][0]+ids[0]+ids[1]+(label=="yes")+2*(tags[1]=="")+face_idx;
            Cout=color(q,Dx(q),Dy(q));
        }""", encoding="ascii")
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    run(oslc, ["-I"+str(includes), str(source)])
    supplied = ["--userdata", "red", "9.0", "--userdata", "gain", "5.0",
                "--userdata:type=float[2]", "weights", "7,11",
                "--userdata:type=matrix", "M", "3",
                "--userdata:type=int[2]", "ids", "101,103",
                "--userdata:type=string", "label", "yes",
                "--userdata:type=string[2]", "tags", "yes,"]
    base = ["-g", "5", "5", "--print", "hart_userdata_cli"]
    mismatched = ["--userdata:type=int", "gain", "5",
                  "--userdata:type=float", "ids", "9.0",
                  "--userdata:type=float", "label", "1.0"]
    cases = (([], False), (supplied, True), (mismatched, False))
    for options, supplied_values in cases:
        check(run(testshade, ["-t", "1"]+base+options), supplied_values)
    for name, flags in (
        ("split", []), ("fused", ["--hart-fused"]),
        ("fused-local", ["--hart-fused", "--hart-local-groupdata", "1048576"]),
        ("unoptimized", ["-O0", "--llvm_opt", "10"]),
    ):
        for options, supplied_values in cases:
            text = run(testshade, ["--hart", "-v"]+flags+base+options
                       + ["--warmup", "--iters", "2"])
            check(text, supplied_values)
            assert text.count("Launching HART grid") == 3, text
        print(f"HART userdata {name} defaults, typed arrays, strings and gradients passed")
