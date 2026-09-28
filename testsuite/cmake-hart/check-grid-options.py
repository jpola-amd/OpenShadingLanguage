# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Check centered grid values/derivatives and the legacy output-format alias."""

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("testshade")
parser.add_argument("--oslc", required=True)
parser.add_argument("--oiiotool", required=True)
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()
if not args.gpu:
    parser.error("Grid-option runtime checks require --gpu")

testshade, oslc = (
    str(Path(value).resolve()) for value in
    (args.testshade, args.oslc)
)
oiiotool = args.oiiotool
env = os.environ.copy()
for key in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
            "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[key] = "0"
for key in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(key, None)
root = Path.cwd() / ("hart-grid-options-" + uuid.uuid4().hex)
root.mkdir()
names = ("Cout", "uvx", "uvy", "dpx", "dpy")
modes = [
    ["-O0", "--llvm_opt", "10"],
    ["-O2", "--llvm_opt", "3"],
    ["-O2", "--llvm_opt", "3", "--hart-fused"],
    ["-O2", "--llvm_opt", "3", "--hart-fused",
     "--hart-local-groupdata", "4096"],
]


def run(executable, arguments, error=None):
    result = subprocess.run(
        [executable] + arguments, cwd=root, env=env,
        capture_output=True, text=True, timeout=300,
    )
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
        assert error.lower() in output.lower(), output
    return output


def shade(arguments, mode=None, error=None):
    flags = ["-O2", "--llvm_opt", "3"] if mode is None else ["--hart", "-v"] + mode
    output = run(testshade, flags + arguments, error)
    assert output.count("Launching HART grid") == int(
        mode is not None and error is None), output
    if mode is not None and error is None:
        expected_mode = "fused" if "--hart-fused" in mode else "split"
        assert "HART callable mode: " + expected_mode in output, output
        storage = re.findall(
            r"HART group storage: (\d+) bytes, alignment (\d+), local (\d+) bytes, scratch (\d+) bytes",
            output,
        )
        assert len(storage) == 1, output
        size, alignment, local, scratch = map(int, storage[0])
        assert size > 0 and alignment > 0, storage
        if "--hart-local-groupdata" in mode:
            assert local == size and scratch == 0, storage
        else:
            assert local == 0 and scratch > 0, storage
    return output


def check_grid(output, width, height, centered):
    blocks = re.findall(
        r"Pixel \((\d+), (\d+)\):\s*\n(.*?)(?=Pixel \(|\Z)",
        output, re.DOTALL,
    )
    assert [(int(x), int(y)) for x, y, _ in blocks] == [
        (x, y) for y in range(height) for x in range(width)], output
    dx = 1 / (width if centered else max(1, width - 1))
    dy = 1 / (height if centered else max(1, height - 1))
    grid_dx, grid_dy = 1 / max(1, width - 1), 1 / max(1, height - 1)
    for sx, sy, block in blocks:
        x, y = int(sx), int(sy)
        u = (x + .5) / width if centered else (.5 if width == 1 else x * dx)
        v = (y + .5) / height if centered else (.5 if height == 1 else y * dy)
        wanted = {
            "Cout": (u, v, 1), "uvx": (u, dx, 0), "uvy": (v, 0, dy),
            "dpx": (grid_dx, 0, 0), "dpy": (0, grid_dy, 0),
        }
        rows = re.findall(r"^\s*(\w+)\s*[:=]\s*([^\r\n]+)", block, re.MULTILINE)
        assert [name for name, _ in rows] == list(names), block
        for name, text in rows:
            actual = list(map(float, text.split()))
            assert len(actual) == 3, (name, actual)
            assert all(math.isfinite(a) and abs(a - b) <= 2e-6
                       for a, b in zip(actual, wanted[name])), (name, actual, wanted[name])


success = False
try:
    (root / "grid_options.osl").write_text(
        "shader grid_options(output color Cout=0, output vector uvx=0, "
        "output vector uvy=0, output vector dpx=0, output vector dpy=0) {\n"
        "Cout = color(P); uvx = vector(u,Dx(u),Dy(u)); "
        "uvy = vector(v,Dx(v),Dy(v)); dpx=Dx(P); dpy=Dy(P);\n}\n",
        encoding="ascii",
    )
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    run(oslc, ["-I" + str(includes), "grid_options.osl"])
    outputs = [argument for name in names for argument in ("-o", name, "null")]
    for width, height in ((1, 1), (2, 2), (7, 3)):
        for centered in (False, True):
            arguments = (["-g", str(width), str(height), "--print"]
                         + (["--center"] if centered else [])
                         + outputs + ["grid_options"])
            check_grid(shade(arguments), width, height, centered)
            for mode in modes:
                check_grid(shade(arguments, mode), width, height, centered)

    for option in ("-d", "-od"):
        for datatype in ("float", "uint8"):
            for gpu in (False, True):
                image = root / f"{option[1:]}-{datatype}-{int(gpu)}.tif"
                shade(["-g", "2", "2", "--center", option, datatype,
                       "-o", "Cout", str(image), "grid_options"],
                      modes[1] if gpu else None)
                info = run(oiiotool, ["--info", str(image)])
                assert re.search(r",\s*3 channel,\s*" + datatype + r"\s+tiff\b",
                                 info), info
                dump = run(oiiotool, ["--dumpdata", str(image)])
                rows = re.findall(r"Pixel \((\d+),\s*(\d+)\):\s*([^\r\n]+)", dump)
                assert [(int(x), int(y)) for x, y, _ in rows] == [
                    (0, 0), (1, 0), (0, 1), (1, 1)], dump
                for x, y, text in rows:
                    actual = list(map(float, re.sub(r"\([^)]*\)", "", text).split()))
                    wanted = ((64 + 127 * int(x), 64 + 127 * int(y), 255)
                              if datatype == "uint8" else
                              (.25 + .5 * int(x), .25 + .5 * int(y), 1))
                    assert actual == list(wanted), (actual, wanted)

    invalid = root / "invalid.tif"
    shade(["-od", "invalid", "-o", "Cout", str(invalid), "grid_options"],
          modes[1], error="format")
    assert not invalid.exists()
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Grid-option failure artifacts retained in", root, flush=True)

print("Centered/default grids and derivatives passed in four HART modes; "
      "-d/-od float and uint8 files match CPU and independent values")
