# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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
args = parser.parse_args()
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
oiiotool = shutil.which(args.oiiotool)
assert oiiotool, "The output-file test requires OpenImageIO's oiiotool"
env = {**os.environ, "TESTSHADE_OPTIX": "0", "TESTSHADE_BATCHED": "0",
       "TESTSHADE_RS_BITCODE": "0", "TESTSHADE_HART": "0", "TESTSHADE_FUSED": "0"}
root = Path.cwd() / ("hart-outputs-" + uuid.uuid4().hex)
root.mkdir()

source = """
struct HartOutputFields {
    float weight;
    int codes[2];
    vector tangent;
};
shader hart_outputs(
    output int index = 0,
    output float value = 0,
    output point position = 0,
    output matrix transform = 1,
    output float samples[3] = {0,0,0},
    output color colors[2] = {0,0},
    output HartOutputFields fields = {0,{0,0},0},
    output string text = "",
    output color Cout = 0)
{
    int x = int(2*u);
    index = 16777217 + x;
    value = u + 2*v;
    position = point(u,v,u-v);
    transform = matrix(1+u,2+u,3+u,4+u,5+u,6+u,7+u,8+u,
                       9+u,10+u,11+u,12+u,13+u,14+u,15+u,16+u);
    samples[0] = value; samples[1] = u-v; samples[2] = u*v;
    colors[0] = color(u,v,1); colors[1] = color(v,u,2);
    fields.weight = value + 0.5;
    fields.codes[0] = -2147483647 + x;
    fields.codes[1] = 2147483647 - x;
    fields.tangent = vector(2*u,3*v,4);
    text = "not an image";
    Cout = color(u,v,u+v);
}
"""
tail = """
shader hart_output_tail(output color Cout = 0)
{
    Cout = color(3+u,4+v,5);
}
"""
array_source = """
shader hart_output_array(output float values[4] = {1,2,3,4})
{
    for (int i = 0; i < arraylength(values); ++i)
        values[i] += u + v;
}
"""


def command(executable, arguments, error=None, launch=False):
    result = subprocess.run([executable] + arguments, cwd=root, env=env,
                            text=True, capture_output=True, timeout=300)
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
        assert error.lower() in output.lower(), output
        assert "Launching HART grid" not in output, output
        assert "Pixel (" not in output, output
    if launch:
        assert "Launching HART grid" in output, output
    return output


def compile_shader(name, text):
    path = root / (name + ".osl")
    path.write_text(text, encoding="ascii")
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    command(oslc, ["-I" + str(includes), "-o", name + ".oso", str(path)])


def selection(names, filenames=None):
    result = []
    for i, name in enumerate(names):
        result += ["-o", name, filenames[i] if filenames else "null"]
    return result


def records(output, names):
    pixels = list(re.finditer(r"Pixel \((\d+), (\d+)\):", output))
    assert len(pixels) == 6, output
    result = []
    for i, pixel in enumerate(pixels):
        assert tuple(map(int, pixel.groups())) == (i % 3, i // 3), output
        end = pixels[i + 1].start() if i + 1 < len(pixels) else len(output)
        fields = re.findall(r"^\s*[\w.]+\s*[:=]\s*([^\r\n]+)",
                            output[pixel.end():end], re.MULTILINE)
        assert len(fields) == len(names), (names, fields, output)
        result.append([values.split() for values in fields])
    return result


def expected(name, x, y):
    u, v = x / 2, y
    leaf = name.removeprefix("out.")
    return {
        "index": [16777217 + x],
        "value": [u + 2*v],
        "position": [u, v, u-v],
        "transform": [n + u for n in range(1, 17)],
        "samples": [u + 2*v, u-v, u*v],
        "colors": [u, v, 1, v, u, 2],
        "fields.weight": [u + 2*v + .5],
        "fields.codes": [-2147483647 + x, 2147483647 - x],
        "fields.tangent": [2*u, 3*v, 4],
        "Cout": [u, v, u+v],
        "tail.Cout": [3+u, 4+v, 5],
        "values": [n + u + v for n in range(1, 5)],
    }[leaf]


def check(output, names):
    data = records(output, names)
    for i, fields in enumerate(data):
        for name, values in zip(names, fields):
            reference = expected(name, i % 3, i // 3)
            assert len(values) == len(reference), (name, values, reference)
            for actual, want in zip(values, reference):
                if name.endswith(("index", "fields.codes")):
                    assert re.fullmatch(r"-?\d+", actual), (name, actual)
                    assert int(actual) == want, (name, actual, want)
                else:
                    assert math.isclose(float(actual), want, abs_tol=2e-6,
                                        rel_tol=2e-6), (name, actual, want)
    return data


try:
    compile_shader("hart_outputs", source)
    compile_shader("hart_output_tail", tail)
    compile_shader("hart_output_array", array_source)
    names = ["out.index", "value", "out.position", "out.transform",
             "out.samples", "out.colors", "out.fields.weight",
             "out.fields.codes", "out.fields.tangent"]
    graph = ["--shader", "hart_outputs", "out"]
    grid = ["-g", "3", "2"]
    check(command(testshade, ["-t", "1", "--print"] + grid
                  + selection(names) + graph), names)
    for mode, flags in (
        ("split", []),
        ("fused", ["--hart-fused"]),
        ("fused-local", ["--hart-fused", "--hart-local-groupdata", "1048576"]),
        ("unoptimized", ["-O0", "--llvm_opt", "10"]),
    ):
        base = ["--hart", "-v", "--warmup", "--iters", "2"] + flags + grid
        check(command(testshade, base + ["--print"] + selection(names) + graph,
                      launch=True), names)
        aliases = ["index", "out.index", "out.samples", "samples", "fields.weight"]
        check(command(testshade, base + ["--print"] + selection(aliases) + graph,
                      launch=True), aliases)
        connected_names = ["out.value", "out.fields.codes", "tail.Cout"]
        two_layers = graph + ["--shader", "hart_output_tail", "tail"]
        check(command(testshade, base + ["--print"] + selection(connected_names)
                      + two_layers, launch=True), connected_names)
        # Unqualified selection prefers the final layer, qualified selection
        # still keeps and executes the disconnected producer.
        last = command(testshade, base + ["--print"] + selection(["Cout"])
                       + two_layers, launch=True)
        check(last.replace("Cout =", "tail.Cout ="), ["tail.Cout"])
        check(command(testshade, base + ["--print", "hart_output_array"]
                      + selection(["values"]), launch=True), ["values"])
        print("Passed typed outputs:", mode)

    base = ["--hart", "-v"] + grid
    for name, error in (("missing", "Unknown HART output"),
                        ("out.missing", "Unknown HART output"),
                        ("text", "must be numeric"),
                        ("fields", "must be numeric")):
        command(testshade, base + selection([name], ["absent.exr"]) + graph,
                error=error)
        assert not (root / "absent.exr").exists()
    command(testshade, base + ["hart_output_array"], launch=True)
    command(testshade, base + ["hart_output_array"] + selection(["Cout"]),
            error="Unknown HART output")
    # Existing Cout-only printing still works with additional unselected outputs.
    check(command(testshade, base + ["--print"] + graph, launch=True), ["Cout"])
    filenames = ["index.tif", "value.tif", "matrix.tif", "codes.tif", "index-copy.tif"]
    file_names = ["out.index", "value", "out.transform", "out.fields.codes", "index"]
    command(testshade, base + selection(file_names, filenames) + graph, launch=True)
    for name, filename in zip(file_names, filenames):
        dump = command(oiiotool, ["--dumpdata", filename])
        rows = re.findall(r"Pixel \((\d+),\s*(\d+)\):\s*([^\r\n]+)", dump)
        assert len(rows) == 6, dump
        for index, (x, y, row) in enumerate(rows):
            assert (int(x), int(y)) == (index % 3, index // 3), dump
            # OIIO appends normalized floating representations to integer dumps.
            values = re.sub(r"\([^)]*\)", "", row).split()
            want = expected(name, int(x), int(y))
            assert len(values) == len(want), dump
            assert all(float(a) == b for a, b in zip(values, want)), (dump, want)
    suppressed = ["suppressed1.tif", "suppressed2.tif"]
    check(command(testshade, base + ["--print"]
                  + selection(["value", "out.samples"], suppressed) + graph,
                  launch=True), ["value", "out.samples"])
    assert all(not (root / name).exists() for name in suppressed)
    print("Passed output files, integer precision, aliases and rejections")
finally:
    shutil.rmtree(root)
