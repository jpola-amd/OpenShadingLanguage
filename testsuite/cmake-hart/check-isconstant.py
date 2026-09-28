# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Compare compile-time classification without confusing uniformity with constancy."""

import argparse
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
    parser.error("Constant-query runtime checks require --gpu")
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = os.environ.copy()
for name in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
             "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[name] = "0"
for name in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(name, None)
root = Path.cwd() / ("hart-isconstant-" + uuid.uuid4().hex)
root.mkdir()
modes = [
    ["-O0", "--llvm_opt", "10"],
    ["-O2", "--llvm_opt", "3"],
    ["-O2", "--llvm_opt", "3", "--hart-fused"],
    ["-O2", "--llvm_opt", "3", "--hart-fused",
     "--hart-local-groupdata", "4096"],
]


def run(executable, arguments):
    result = subprocess.run([executable] + arguments, cwd=root, env=env,
                            capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def shade(mode, parameters, gpu):
    flags = ["--hart", "-v"] + mode if gpu else mode[:3]
    output = run(testshade, flags + ["-g", "5", "3", "--print", "-o", "Cout", "null"]
                 + parameters + ["constants"])
    assert output.count("Launching HART grid") == int(gpu), output
    if gpu:
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
    rows = re.findall(r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
                      output)
    assert [(int(x), int(y)) for x, y, *_ in rows] == [
        (x, y) for y in range(3) for x in range(5)], output
    values = [tuple(map(float, row[2:])) for row in rows]
    for literal, varying, folded in values:
        assert literal == 7 and varying == 0, values
        assert folded in range(8), values
        interpolated = any("interpolated=1" in arg for arg in parameters)
        if mode[0] == "-O2":
            assert folded == (4 if interpolated else 7), values
        elif interpolated:
            assert int(folded) & 3 == 0, values
    return values


success = False
try:
    (root / "constants.osl").write_text(
        'shader constants(float A=1, string label="literal", output color Cout=0) {\n'
        'float twice=2*A; string selected_label=u>v ? "left" : "right";\n'
        'Cout=color(isconstant(3)+2*isconstant(2.0)+4*isconstant("literal"),\n'
        'isconstant(u)+2*isconstant(P)+4*isconstant(selected_label),\n'
        'isconstant(A)+2*isconstant(twice)+4*isconstant(label));\n}\n',
        encoding="ascii",
    )
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    run(oslc, ["-I" + str(includes), "constants.osl"])
    for mode in modes:
        for parameters in ([], ["--param:type=float", "A", "5"],
                           ["--param:type=float:interpolated=1", "A", "5"]):
            cpu = shade(mode, parameters, False)
            gpu = shade(mode, parameters, True)
            assert gpu == cpu, (mode, parameters, cpu, gpu)
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Constant-query failure artifacts retained in", root, flush=True)

print("Literal/varying/parameter constant classification passed in four HART modes")
