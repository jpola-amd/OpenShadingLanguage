# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Check serialized group execution, pre-optimization queries and errors."""

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
    parser.error("Serialized-group runtime checks require --gpu")
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = os.environ.copy()
for key in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
            "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[key] = "0"
for key in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(key, None)
root = Path.cwd() / ("hart-serialized-groups-" + uuid.uuid4().hex)
root.mkdir()
modes = [
    ["-O0", "--llvm_opt", "10"],
    ["-O2", "--llvm_opt", "3"],
    ["-O2", "--llvm_opt", "3", "--hart-fused"],
    ["-O2", "--llvm_opt", "3", "--hart-fused",
     "--hart-local-groupdata", "4096"],
]
grid = ["-g", "3", "2", "--print", "-o", "Cout", "null",
        "--groupname", "serialized"]
spec = ("param float gain 3; shader source a; "
        "param float[2] offset 0.25 0.5; shader sink b; "
        "connect a.value b.input;")


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


def shade(arguments, mode=None, error=None, verbose=True):
    flags = ["-O2", "--llvm_opt", "3"] if mode is None else ["--hart"] + mode
    if verbose:
        flags += ["-v"]
    output = run(testshade, flags + arguments, error)
    assert output.count("Launching HART grid") == int(
        mode is not None and verbose and error is None), output
    if mode is not None and verbose and error is None:
        expected = "fused" if "--hart-fused" in mode else "split"
        assert "HART callable mode: " + expected in output, output
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


def check_pixels(output):
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
        output,
    )
    assert [(int(x), int(y)) for x, y, *_ in rows] == [
        (x, y) for y in range(2) for x in range(3)], output
    for x, y, *text in rows:
        wanted = (1.5*int(x)+.25, 3*int(y)+.5, .75)
        assert all(math.isfinite(a) and abs(a - b) <= 2e-6
                   for a, b in zip(map(float, text), wanted)), (text, wanted)


def query(output):
    pickles = re.findall(r"Shader group:\n---\n(.*?)\n---\n", output, re.DOTALL)
    layers = re.findall(
        r'Shader group "serialized" layers are:\n(.*?)\n\n',
        output, re.DOTALL,
    )
    assert len(pickles) == len(layers) == 1, output
    assert layers[0] == (
        "    a\n\tfloat gain\n\toutput color value\n"
        "    b\n\tcolor input\n\tfloat[2] offset\n\toutput color Cout"), layers
    assert "shader source a" in pickles[0] and "shader sink b" in pickles[0]
    assert "connect a.value b.input" in pickles[0]
    return pickles[0], layers[0]


success = False
try:
    for name, source in (
        ("source", "shader source(float gain=1, output color value=0) {\n"
                   "value=color(gain*u,gain*v,.25);\n}\n"),
        ("sink", "shader sink(color input=0, float offset[2]={0,0}, "
                 "output color Cout=0) {\n"
                 "Cout=input+color(offset[0],offset[1],.5);\n}\n"),
    ):
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
        run(oslc, ["-I" + str(includes), name + ".osl"])
    (root / "group.oslgroup").write_text(spec, encoding="ascii")
    for option, value in (("--group", spec), ("-group", "group.oslgroup")):
        arguments = grid + ["--oslquery", option, value]
        cpu = shade(arguments)
        check_pixels(cpu)
        expected_query = query(cpu)
        for mode in modes:
            output = shade(arguments, mode)
            check_pixels(output)
            assert query(output) == expected_query

    quiet = shade(grid + ["--oslquery", "--group", spec], modes[1], verbose=False)
    check_pixels(quiet)
    pickle, _ = query(quiet)
    roundtrip = shade(grid + ["--oslquery", "--group", pickle], modes[1])
    check_pixels(roundtrip)
    assert query(roundtrip) == query(quiet)
    verbose = shade(grid + ["--group", spec], modes[1])
    check_pixels(verbose)
    assert "Shader group:\n---\n" in verbose and "    a\n    b\n" in verbose
    assert "\tfloat gain\n" not in verbose

    for invalid in ("not_a_group", "missing.oslgroup",
                    spec + "connect a.missing b.input;"):
        for mode in (None, modes[1]):
            image = root / "invalid.tif"
            shade(["--group", invalid, "--shader", "sink", "later",
                   "-o", "Cout", str(image)], mode, error="Invalid shader group")
            assert not image.exists()
    (root / "directory").mkdir()
    for mode in (None, modes[1]):
        shade(["--group", "directory", "-o", "Cout", "unreadable.tif"],
              mode, error="Could not read shader group")
        assert not (root / "unreadable.tif").exists()
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Serialized-group failure artifacts retained in", root, flush=True)

print("Serialized inline/file groups, exact CPU queries, roundtrip, "
      "four HART modes and prelaunch errors passed")
