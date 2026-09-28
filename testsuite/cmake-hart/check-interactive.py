# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
from collections import Counter
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
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0 and error in output, output
        assert "Pixel (" not in output, output
    return output


def check(output, warmup):
    records = Counter(re.findall(r"INTERACTIVE (\d+) (\d+) ([\d.]+) (\w+) (\w+)",
                                output))
    expected = Counter()
    for y in range(2):
        for x in range(3):
            expected[(str(x), str(y), "2.0", "red", "empty")] = 2 if warmup else 1
            expected[(str(x), str(y), "4.0", "blue", "green")] = 1
    assert records == expected, (records, expected, output)
    pixels = re.findall(r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s*([^\r\n]+)",
                        output)
    assert len(pixels) == 6, output
    for index, (x, y, values) in enumerate(pixels):
        assert (int(x), int(y)) == (index % 3, index // 3), output
        assert list(map(float, values.split())) == [2 * int(x), 0, 0], output


with tempfile.TemporaryDirectory(prefix="osl-hart-interactive-") as temporary:
    root = Path(temporary)
    source = root / "hart_interactive_cli.osl"
    source.write_text(r"""
        shader hart_interactive_cli(float gain = 2,
            string labels[2] = {"red",""} [[int interactive=1]],
            output color Cout = 0) {
            Cout = color(gain*u, labels[0]=="red", labels[1]=="");
            printf("INTERACTIVE %d %d %.1f %s %s\n", int(2*u), int(v), gain,
                   labels[0], labels[1]=="" ? "empty" : labels[1]);
        }""", encoding="ascii")
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    run(oslc, ["-I" + str(includes), str(source)])
    graph = ["--layer", "active", "--param:interactive=1", "gain", "2.0",
             "hart_interactive_cli"]
    updates = ["--reparam", "active", "gain", "4.0",
               "--reparam:type=string[2]", "active", "labels", "blue,green"]
    common = ["-g", "3", "2", "--iters", "2", "--print"] + graph
    check(run(testshade, ["-t", "1"] + common + updates), False)
    for mode, flags in (
        ("split", []),
        ("fused", ["--hart-fused"]),
        ("fused-local", ["--hart-fused", "--hart-local-groupdata", "1048576"]),
        ("unoptimized", ["-O0", "--llvm_opt", "10"]),
    ):
        base = ["--hart", "-v"] + flags + common
        output = run(testshade, base + ["--warmup"] + updates)
        check(output, True)
        assert output.count("Launching HART grid") == 3, output
        failed = root / ("failed-" + mode + ".exr")
        for update, error in (
            (["--reparam:type=string", "active", "gain", "bad"], "type mismatch"),
            (["--reparam", "missing", "gain", "4"], "unknown layer"),
            (["--reparam:type=string[3]", "active", "labels", "a,b,c"],
             "type mismatch"),
        ):
            output = run(testshade, base + ["-o", "Cout", str(failed)] + update,
                         error)
            assert output.count("Launching HART grid") == 1, output
            assert not failed.exists(), output
        print(f"HART interactive {mode} CLI updates, warmup, values and errors passed")
