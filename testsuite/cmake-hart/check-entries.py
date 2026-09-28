# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
from collections import Counter
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
args = parser.parse_args()
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = {**os.environ, "TESTSHADE_OPTIX": "0", "TESTSHADE_BATCHED": "0",
       "TESTSHADE_RS_BITCODE": "0", "TESTSHADE_HART": "0", "TESTSHADE_FUSED": "0"}
root = Path.cwd() / ("hart-entries-" + uuid.uuid4().hex)
root.mkdir()
sources = {
    "hart_entry_p": """
        shader hart_entry_p(output float value = 0) {
            value = 10 + u + 2*v;
            printf("RUN p %d %d\\n", int(2*u), int(v));
        }""",
    "hart_entry_a": """
        shader hart_entry_a(float value = 0, output float result = 0) {
            result = value + 1;
            printf("RUN a %d %d\\n", int(2*u), int(v));
        }""",
    "hart_entry_b": """
        shader hart_entry_b(float value = 0, output float result = 0) {
            result = 2*value;
            printf("RUN b %d %d\\n", int(2*u), int(v));
        }""",
    "hart_entry_unused": """
        shader hart_entry_unused(output color Cout = 0) {
            error("UNSELECTED ENTRY EXECUTED");
        }""",
}


def command(executable, arguments, error=None, launched=False):
    result = subprocess.run([executable] + arguments, cwd=root, env=env,
                            capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
        assert error in output, output
        assert "Pixel (" not in output, output
    if executable == testshade and "--hart" in arguments:
        assert ("Launching HART grid" in output) == launched, output
    return output


def declarations(names):
    return [value for name in names for value in ("--entry", name)]


def entry_outputs(names):
    return [value for name in names for value in ("--entryoutput", name + ".result")]


def check(output, selected, order, launches=1, last=""):
    wanted_order = ["p"]
    for name in order:
        if name == last or name not in wanted_order:
            wanted_order.append(name)
    messages = re.findall(r"RUN ([pab]) (\d+) (\d+)", output)
    runs = Counter(messages)
    expected = Counter({(name, str(x), str(y)): count * launches
                        for name, count in Counter(wanted_order).items()
                        for y in range(2) for x in range(3)})
    assert runs == expected, (runs, expected, output)
    for y in range(2):
        for x in range(3):
            actual_order = [name for name, px, py in messages
                            if (px, py) == (str(x), str(y))]
            assert actual_order == wanted_order * launches, (actual_order, wanted_order)
    pixels = list(re.finditer(r"Pixel \((\d+), (\d+)\):", output))
    assert len(pixels) == 6, output
    for i, pixel in enumerate(pixels):
        assert tuple(map(int, pixel.groups())) == (i % 3, i // 3), output
        end = pixels[i+1].start() if i+1 < len(pixels) else len(output)
        values = re.findall(r"^\s*[\w.]+\s*[:=]\s*([^\r\n]+)",
                            output[pixel.end():end], re.M)
        assert len(values) == 2, output
        value = 10 + (i % 3)/2 + 2*(i // 3)
        want = [value + 1 if "a" in selected else 0,
                2*value if "b" in selected else 0]
        for actual, expected_value in zip(values, want):
            assert math.isclose(float(actual), expected_value, abs_tol=2e-6), (
                actual, expected_value, output)


try:
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    for name, source in sources.items():
        path = root / (name + ".osl")
        path.write_text(source, encoding="ascii")
        command(oslc, ["-I" + str(includes), "-o", name + ".oso", str(path)])
    connections = ["--connect", "p", "value", "a", "value",
                   "--connect", "p", "value", "b", "value"]
    graph = [arg for name in ("p", "a", "b", "unused")
             for arg in ("--shader", "hart_entry_" + name, name)] + connections
    short_graph = [arg for name in ("p", "a", "b")
                   for arg in ("--shader", "hart_entry_" + name, name)] + connections
    outputs = ["-o", "a.result", "null", "-o", "b.result", "null"]
    cases = [
        (["a", "b", "a"], [], {"a", "b"}),
        (["b", "a", "b"], [], {"a", "b"}),
        (["a", "b", "unused"], ["b", "b"], {"b"}),
        (["a", "b", "unused"], ["a"], {"a"}),
    ]
    # CPU --print reads Groupdata even for unexecuted layers. Explicitly clear
    # it so those values can be compared with HART's zeroed output arena.
    cpu = ["-t", "1", "--options", "clearmemory=1", "-g", "3", "2", "--print"]
    for declared, chosen, selected in cases:
        options = declarations(declared) + entry_outputs(chosen)
        check(command(testshade, cpu + options + outputs + graph),
              selected, chosen or declared)
    repeated_last = declarations(["b", "b", "a"])
    check(command(testshade, cpu + repeated_last + outputs + short_graph),
          {"a", "b"}, ["b", "b", "a"], last="b")
    for mode, flags in (
        ("split", []),
        ("fused", ["--hart-fused"]),
        ("fused-local", ["--hart-fused", "--hart-local-groupdata", "1048576"]),
        ("unoptimized", ["-O0", "--llvm_opt", "10"]),
    ):
        base = ["--hart", "-v", "-g", "3", "2"] + flags
        for declared, chosen, selected in cases:
            options = declarations(declared) + entry_outputs(chosen)
            output = command(testshade, base + ["--print", "--warmup", "--iters", "2"]
                             + options + outputs + graph, launched=True)
            check(output, selected, chosen or declared, launches=3)
        check(command(testshade, base + ["--print"] + repeated_last + outputs
                      + short_graph, launched=True),
              {"a", "b"}, ["b", "b", "a"], last="b")
        failed = root / ("unselected-" + mode + ".tif")
        command(testshade, base + declarations(["a", "b", "unused"])
                + ["-o", "a.result", str(failed)] + graph,
                error="UNSELECTED ENTRY EXECUTED", launched=True)
        assert not failed.exists()
        print("Passed entry sequence, init/lazy flags and errors:", mode)
    for options, error in (
        (declarations(["missing"]), "Unknown HART entry layer"),
        (declarations(["a"]) + entry_outputs(["missing"]), "Unknown HART entry output"),
        (declarations(["a"]) + ["--entryoutput", "p.value"], "not in a declared entry layer"),
        (entry_outputs(["a"]), "requires declared --entry layers"),
    ):
        command(testshade, ["--hart", "-v", "--print"] + options + outputs + graph,
                error=error)
    print("Passed explicit entries and prelaunch selection validation")
finally:
    shutil.rmtree(root)
