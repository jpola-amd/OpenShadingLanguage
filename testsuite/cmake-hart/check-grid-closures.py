# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Check caller-owned closures in ordinary grids without exporting pointers."""

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
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()
if not args.gpu:
    parser.error("Grid-closure runtime checks require --gpu")
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = os.environ.copy()
for key in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
            "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[key] = "0"
for key in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(key, None)
root = Path.cwd() / ("hart-grid-closures-" + uuid.uuid4().hex)
root.mkdir()
modes = [
    ["-O0", "--llvm_opt", "10"],
    ["-O2", "--llvm_opt", "3"],
    ["-O2", "--llvm_opt", "3", "--hart-fused"],
    ["-O2", "--llvm_opt", "3", "--hart-fused",
     "--hart-local-groupdata", "4096"],
]
grid = ["-g", "7", "5", "--print", "-o", "Cout", "null"]


def run(executable, arguments, error=None):
    result = subprocess.run(
        [executable] + arguments, cwd=root, env=env,
        capture_output=True, text=True, timeout=300,
    )
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0 and error in output, output
    return output


def shade(arguments, mode=None, pool=1024, launches=1, error=None):
    flags = ["-O2", "--llvm_opt", "3"] if mode is None else ["--hart", "-v"] + mode
    output = run(testshade, flags + arguments, error)
    assert output.count("Launching HART grid") == (
        launches if mode is not None else 0), output
    if mode is not None and launches:
        expected = "fused" if "--hart-fused" in mode else "split"
        assert "HART callable mode: " + expected in output, output
        assert f"HART closure pool: {pool} bytes per point" in output, output
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


def check_pixels(output, expected):
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
        output,
    )
    assert [(int(x), int(y)) for x, y, *_ in rows] == [
        (x, y) for y in range(5) for x in range(7)], output
    for x, y, *text in rows:
        wanted = expected(int(x) / 6, int(y) / 4)
        assert all(math.isfinite(a) and abs(a - b) <= 2e-6
                   for a, b in zip(map(float, text), wanted)), (text, wanted)


success = False
try:
    sources = {
        "plain": "shader plain(output color Cout=0) { Cout=color(u,v,.5); }",
        "trees": "shader trees(output color Cout=0) {\n"
                 "closure color c[2]; c[0]=(.25+u)*diffuse(N); "
                 "c[1]=color(v,0,1)*emission(); Ci=c[0]+c[1];\n"
                 "Cout=color(0,u,v); if(Ci) Cout[0]=1;\n}",
        "producer": "shader producer(output closure color c=0) {\n"
                    "c=(.25+u)*diffuse(N);\n}",
        "consumer": "shader consumer(closure color c=0, output color Cout=0) {\n"
                    "Ci=c+emission(); Cout=color(0,0,v); "
                    "if(c) Cout[0]=1; if(Ci) Cout[1]=1;\n}",
        "exhaust": "shader exhaust(output color Cout=0) {\n"
                   "closure color sum=0; for(int i=0;i<int(64+u);++i) "
                   "sum+=(u+.25)*diffuse(N); Ci=sum; Cout=1;\n}",
        "bad_output": "shader bad_output(output closure color Cout=0) "
                      "{ Cout=emission(); }",
    }
    for name, source in sources.items():
        (root / (name + ".osl")).write_text(source + "\n", encoding="ascii")
        includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
        run(oslc, ["-I" + str(includes), name + ".osl"])
    connected = ["--shader", "producer", "a", "--shader", "consumer", "b",
                 "--connect", "a", "c", "b", "c"]
    for arguments, expected, pool in (
        (["plain"], lambda u, v: (u, v, .5), 0),
        (["trees"], lambda u, v: (1, u, v), 1024),
        (connected, lambda u, v: (1, 1, v), 1024),
        (connected + ["--entry", "a", "--entry", "b"],
         lambda u, v: (1, 1, v), 1024),
    ):
        check_pixels(shade(grid + arguments), expected)
        for mode in modes:
            output = shade(grid + ["--warmup", "--iters", "3"] + arguments,
                           mode, pool=pool, launches=4)
            check_pixels(output, expected)

    for mode in modes:
        image = root / "exhaust.tif"
        shade(["-g", "7", "5", "-o", "Cout", str(image), "exhaust"], mode,
              error="closure pool allocation failed")
        assert not image.exists()
        shade(["bad_output"], mode, launches=0, error="RGB color")
        shade(["-o", "Cout", "closure.tif", "bad_output"], mode,
              launches=0, error="must be numeric")
        assert not (root / "closure.tif").exists()
        # A now-supported, unused closure producer must not reject the group.
        check_pixels(shade(grid + ["--shader", "bad_output", "unused",
                                   "--shader", "trees", "last"], mode),
                     lambda u, v: (1, u, v))
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Grid-closure failure artifacts retained in", root, flush=True)

print("Ordinary closure grids passed: arrays, connected/explicit entries, "
      "repeated launches, pool-free numeric grids and no-image errors")
