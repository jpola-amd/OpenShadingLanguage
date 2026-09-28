# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Check live division guards and immutable forward/inverse spline selectors."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("testshade")
parser.add_argument("--oslc", required=True)
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()
if not args.gpu:
    parser.error("Spline/division runtime checks require --gpu")
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = os.environ.copy()
for name in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
             "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[name] = "0"
for name in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(name, None)
root = Path.cwd() / ("hart-spline-division-" + uuid.uuid4().hex)
root.mkdir()
modes = [
    ["-O0", "--llvm_opt", "10"],
    ["-O2", "--llvm_opt", "3"],
    ["-O2", "--llvm_opt", "3", "--hart-fused"],
    ["-O2", "--llvm_opt", "3", "--hart-fused",
     "--hart-local-groupdata", "4096"],
]


def run(executable, arguments):
    return subprocess.run([executable] + arguments, cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)


def compile_source(name, source):
    (root / (name + ".osl")).write_text(source, encoding="ascii")
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    result = run(oslc, ["-I" + str(includes), name + ".osl"])
    assert result.returncode == 0, result.stdout + result.stderr


def shade(mode, arguments, gpu, error=None):
    flags = ["--hart", "-v"] + mode if gpu else mode[:3]
    result = run(testshade, flags + arguments)
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == (error is None), output
    if error:
        assert error in output, output
    assert output.count("Launching HART grid") == int(gpu and not error), output
    if gpu and not error:
        expected = "fused" if "--hart-fused" in mode else "split"
        assert "HART callable mode: " + expected in output, output
        storage = re.findall(
            r"HART group storage: (\d+) bytes, alignment (\d+), local (\d+) bytes, scratch (\d+) bytes",
            output,
        )
        assert len(storage) == 1, output
        size, alignment, local, scratch = map(int, storage[0])
        assert size > 0 and alignment > 0, storage
        assert (local == size and scratch == 0 if "--hart-local-groupdata" in mode
                else local == 0 and scratch > 0), storage
    return output


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def color_rows(output, width, height, expected):
    rows = re.findall(r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
                      output)
    assert [(int(x), int(y)) for x, y, *_ in rows] == [
        (x, y) for y in range(height) for x in range(width)], output
    for x, y, *values in rows:
        wanted = expected(int(x) / (width - 1), int(y) / (height - 1))
        assert all(math.isfinite(a) and abs(a - b) <= 6e-6
                   for a, b in zip(map(float, values), wanted)), (values, wanted)


success = False
try:
    pairs = [(2, 1.5), (7, 3), (-2, 1.5), (0, 2), (-0.0, 2),
             (2, 0), (-2, 0), (0, 0), (2, -0.0),
             (math.inf, 2), (-math.inf, 2), (math.nan, 2),
             (2, math.nan), (2, math.inf), (2, -math.inf),
             (math.inf, math.inf), (3e38, 1e-30)]
    wanted = []
    for a, b in pairs:
        a, b = float32(a), float32(b)
        quotient = a / b if b != 0 else math.nan
        wanted.append(float32(quotient) if math.isfinite(quotient)
                      and abs(quotient) <= float32(3.4028234663852886e38) else 0.0)
    count = len(pairs)
    compile_source("division_values",
        f"shader division_values(float a[{count}]={{0}},float b[{count}]={{1}},"
        f"float expected[{count}]={{0}},output float result=0,output int exact=0) {{"
        f"int i=int({count-1}*u+0.5); result=a[i]/b[i];"
        "exact=(result==expected[i]);}")
    compile_source("division_derivs",
        "shader division_derivs(output color Cout=0) {"
        "float q=(2+u)/(1.5+v); Cout=color(q,Dx(q),Dy(q));}")
    compile_source("spline_selectors",
        'shader spline_selectors(string basis="linear",output color Cout=0) {'
        'string selected=basis; float k[10]={0,1,2,3,4,5,6,7,8,9};'
        'float f=spline(selected,u,k); float inv=splineinverse(selected,f,k);'
        'Cout=color(f,inv,Dx(f));}')
    parameters = []
    for name, values in (("a", [p[0] for p in pairs]),
                         ("b", [p[1] for p in pairs]), ("expected", wanted)):
        parameters += [f"--param:type=float[{count}]", name,
                       ",".join(map(str, values))]
    for mode in modes:
        for gpu in (False, True):
            output = shade(mode, ["--print", "-g", str(count), "1",
                                  "-o", "result", "null", "-o", "exact", "null"]
                           + parameters + ["division_values"], gpu)
            rows = re.findall(
                r"Pixel \((\d+), (\d+)\):\s+result\s*[:=]\s+(\S+)\s+exact\s*[:=]\s+(\S+)",
                output,
            )
            assert [(int(x), int(y)) for x, y, *_ in rows] == [
                (x, 0) for x in range(count)], output
            for (_, _, value, exact), target in zip(rows, wanted):
                value = float(value)
                assert int(exact) == 1 and math.isfinite(value), output
                assert abs(value - target) <= 6e-6, (value, target)
                if target == 0:
                    assert math.copysign(1, value) == math.copysign(1, target), output
                if gpu:
                    assert float32(value) == target, (value, target)
            grid = ["--print", "-g", "5", "3", "-o", "Cout", "null"]
            output = shade(mode, grid + ["division_derivs"], gpu)
            color_rows(output, 5, 3, lambda u, v:
                       ((2+u)/(1.5+v), 0.25/(1.5+v),
                        -0.5*(2+u)/(1.5+v)**2))
            for basis, override in (("linear", []),
                                    ("bezier", ["--param:type=string", "basis", "bezier"])):
                output = shade(mode, grid + override + ["spline_selectors"], gpu)
                scale, offset = (7, 1) if basis == "linear" else (9, 0)
                color_rows(output, 5, 3, lambda u, v:
                           (offset+scale*u, u, scale/4))
        for rejected_parameters, error in (
                (["--param:type=string", "basis", "unknown"], "unsupported spline basis"),
                (["--param:type=string:interactive=1", "basis", "linear"], "immutable"),
                (["--param:type=string:interpolated=1", "basis", "linear"], "immutable")):
            shade(mode, ["-o", "Cout", "rejected.exr"] + rejected_parameters
                  + ["spline_selectors"], True, error)
            assert not (root / "rejected.exr").exists()
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Spline/division failure artifacts retained in", root, flush=True)

print("Live division values/guards/derivatives and spline selectors passed in four modes")
