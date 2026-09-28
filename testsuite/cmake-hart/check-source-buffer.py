# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Verify genuine memory compilation, not a fallback to disk shader loading."""

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
    parser.error("Source-buffer runtime checks require --gpu")
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
testsuite = Path(__file__).resolve().parent.parent
env = os.environ.copy()
env["OSLHOME"] = str(testsuite.parent / "src")
for key in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
            "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[key] = "0"
for key in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(key, None)
root = Path.cwd() / ("hart-source-buffer-" + uuid.uuid4().hex)
root.mkdir()
modes = [
    ["-O0", "--llvm_opt", "10"],
    ["-O2", "--llvm_opt", "3"],
    ["-O2", "--llvm_opt", "3", "--hart-fused"],
    ["-O2", "--llvm_opt", "3", "--hart-fused",
     "--hart-local-groupdata", "4096"],
]
grid = ["-g", "3", "2", "--print", "-o", "Cout", "null"]


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


def shade(arguments, mode=None, error=None):
    flags = ["-O2", "--llvm_opt", "3"] if mode is None else ["--hart", "-v"] + mode
    output = run(testshade, flags + arguments, error)
    assert output.count("Launching HART grid") == int(
        mode is not None and error is None), output
    if mode is not None and error is None:
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


def check_pixels(output, expected):
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
        output,
    )
    assert [(int(x), int(y)) for x, y, *_ in rows] == [
        (x, y) for y in range(2) for x in range(3)], output
    for x, y, *text in rows:
        actual = list(map(float, text))
        wanted = expected(int(x) / 2, int(y))
        assert all(math.isfinite(a) and abs(a - b) <= 2e-5
                   for a, b in zip(actual, wanted)), (actual, wanted)


def stale_shader(name):
    (root / "disk.osl").write_text(
        f"shader {name}(output color Cout=0) {{ Cout=color(99); }}\n",
        encoding="ascii",
    )
    run(oslc, ["-I" + str(testsuite.parent / "src" / "shaders"),
               "-o", name + ".oso", "disk.osl"])


success = False
try:
    shutil.copyfile(testsuite / "compile-buffer" / "test.osl", root / "test.osl")
    distance = lambda u, v: (math.sqrt(u*u + v*v + 1),) * 3
    for mode in [None] + modes:
        output = shade(grid + ["--inbuffer", "test"], mode)
        check_pixels(output, distance)
        if mode is None:
            assert output.count("Hello, world!") == 6, output
        else:
            records = re.findall(
                r"HART shader 'test' \(<buffer>:11, point (\d+)\): Hello, world!",
                output,
            )
            assert list(map(int, records)) == list(range(6)), output
        assert not list(root.glob("*.oso"))

    (root / "producer.osl").write_text(
        "shader producer(float gain=1, output color value=0) {\n"
        "value=gain*color(u,v,distance(P,point(0)));\n}\n", encoding="ascii",
    )
    (root / "consumer.osl").write_text(
        "shader consumer(color input=0, output color Cout=0) {\n"
        "Cout=input+color(.25);\n}\n", encoding="ascii",
    )
    connected = [
        "--inbuffer", "--param:type=float", "gain", "2",
        "--shader", "producer", "a", "--shader", "consumer", "b",
        "--connect", "a", "value", "b", "input",
    ]
    for mode in [None] + modes:
        check_pixels(shade(grid + connected, mode),
                     lambda u, v: (2*u+.25, 2*v+.25, 2*distance(u, v)[0]+.25))
        assert not list(root.glob("*.oso"))

    stale_shader("test")
    before = (root / "test.oso").read_bytes()
    check_pixels(shade(grid + ["test"]), lambda u, v: (99, 99, 99))
    for mode in [None] + modes:
        check_pixels(shade(grid + ["--inbuffer", "test"], mode), distance)
        assert (root / "test.oso").read_bytes() == before

    stale_shader("missing")
    stale_shader("broken")
    (root / "broken.osl").write_text(
        "shader broken(output color Cout=0) { Cout = ; }\n", encoding="ascii",
    )
    for name, error in (("missing", 'Could not open "missing.osl"'),
                        ("broken", 'Could not compile "broken"')):
        for mode in (None, modes[1]):
            image = root / (name + ".tif")
            shade(["--inbuffer", "-o", "Cout", str(image), name],
                  mode, error=error)
            assert not image.exists()
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Source-buffer failure artifacts retained in", root, flush=True)

print("Source-buffer CPU/four-mode HART checks passed: original source, "
      "connected layers, stale disk shader precedence and prelaunch errors")
