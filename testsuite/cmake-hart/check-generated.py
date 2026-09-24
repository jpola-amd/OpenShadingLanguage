# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
fixtures = Path(__file__).resolve().parent
env = os.environ.copy()
env["TESTSHADE_OPTIX"] = "0"
env["TESTSHADE_BATCHED"] = "0"
env["TESTSHADE_RS_BITCODE"] = "0"
root = Path.cwd() / ("hart-generated-check-" + uuid.uuid4().hex)
root.mkdir()


def run(arguments, error=None, extra_env=None):
    result = subprocess.run(
        [testshade] + arguments, cwd=root, env={**env, **(extra_env or {})},
        capture_output=True, text=True, timeout=300,
    )
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, "Unexpected success:\n" + output
        assert error.lower() in output.lower(), output
        assert "Launching HART grid" not in output, output
    return output


def pixels(output, width, height):
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
        output,
    )
    assert len(rows) == width * height, output
    result = []
    for index, row in enumerate(rows):
        assert tuple(map(int, row[:2])) == (index % width, index // width), row
        result.extend(map(float, row[2:]))
    return result


def reference(width, height, sine):
    result = []
    for y in range(height):
        v = 0.5 if height == 1 else y / (height - 1)
        for x in range(width):
            u = 0.5 if width == 1 else x / (width - 1)
            result.extend((u, v, math.sin(u + v) if sine else u + v))
    return result


def compare(actual, expected, tolerance):
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert math.isclose(a, b, abs_tol=tolerance, rel_tol=1e-6), (
            index, a, b, tolerance
        )


def image_pixels(path, width, height):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"PF"
        assert list(map(int, stream.readline().split())) == [width, height]
        scale = float(stream.readline())
        data = struct.unpack(
            ("<" if scale < 0 else ">") + str(width * height * 3) + "f",
            stream.read(),
        )
    # PFM scanlines are stored bottom to top.
    return [value for y in reversed(range(height))
            for value in data[y * width * 3:(y + 1) * width * 3]]


try:
    for source in fixtures.glob("hart_*.osl"):
        result = subprocess.run(
            [oslc, "-I" + str(fixtures.parents[1] / "src" / "shaders"),
             "-o", str(root / (source.stem + ".oso")), str(source)],
            cwd=root, env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    base = ["--hart", "hart_first"]
    for option in (
        ["--batched"], ["--center"], ["--entry", "layer"],
        ["--entryoutput", "Cout"], ["--connect", "a", "Cout", "b", "Cout"],
        ["--userdata", "value", "1"], ["--use_rs_bitcode"],
        ["--no-output-placement"], ["--shadeimage"], ["--raytype", "shadow"],
        ["--scaleuv", "2", "2"], ["--offsetuv", "1", "1"],
        ["--options", "optimize=0"], ["--saveptx"],
        ["--reparam", "layer", "value", "2"],
    ):
        run(base + option, "unsupported option")
    for option in (["--hart-entry", "__raygen__other"],
                   ["--hart-callable-module", "other.bc"]):
        run(base + option, "cannot be mixed")
    run(base + ["--hart-module", "other.bc"], "not OSL shaders")
    run(base + ["-o", "other", "null"], "one RGB output")
    run(base + ["-o", "Cout", "null", "-o", "Cout", "null"], "one RGB output")
    run(base + ["-g", "0", "1"], "must be positive")
    run(base + ["-g", "46341", "46341"], "int shade-index range")
    run(base + ["--iters", "0"], "must be positive")
    run(base + ["--hart-device", "-1"], "must be nonnegative")
    run(base + ["--hart-device", "not-an-integer"], "error")
    run(base + ["--param:interpolated=1", "value", "2"], "interpolated")
    run(base + ["--param:interactive=1", "value", "2"], "interactive")
    run(base + ["-d", "invalid"], "output format")
    run(["--hart", "--param", "scale", "2"], "requires an OSL shader")
    for option in ("TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
        run(base, "does not support " + option, {option: "1"})

    if args.gpu:
        for shader, error in (
            ("hart_wrong_output", "RGB color"),
            ("hart_missing_output", "RGB color"),
            ("hart_extra_output", "RGB color"),
            ("hart_closure", "does not support parameter"),
            ("hart_string", "does not support parameter"),
            ("hart_printf", "HART"),
            ("hart_texture", "HART"),
            ("hart_userdata", "HART"),
        ):
            run(["--hart", "-v", shader], error)
        run(["--hart", "--shader", "hart_first", "first",
             "--shader", "hart_sine", "second", "-v"], "one shader layer")

        for shader in ("hart_first", "hart_sine"):
            sine = shader == "hart_sine"
            for width, height in ((1, 1), (3, 2), (37, 5)):
                grid = ["-g", str(width), str(height)]
                expected = reference(width, height, sine)
                cpu = pixels(run(["-t", "1"] + grid + ["--print", shader]),
                             width, height)
                # CPU testshade prints six significant digits.
                compare(cpu, expected, 5e-6)
                for cache_options in (["--hart-no-cache"], []):
                    flags = ["--hart", "-v", "--warmup", "--iters", "3"]
                    flags += cache_options + grid
                    output = run(flags + ["--print", shader])
                    if cache_options:
                        assert "HART pipeline cache disabled" in output, output
                    assert output.count("Launching HART grid") == 4, output
                    gpu = pixels(output, width, height)
                    compare(gpu, expected, 2e-6)
                    compare(gpu, cpu, 5e-6)

                    cpu_image, gpu_image = root / "cpu.pfm", root / "gpu.pfm"
                    for image in (cpu_image, gpu_image):
                        if image.exists():
                            image.unlink()
                    run(["-t", "1"] + grid
                        + ["-o", "Cout", str(cpu_image), shader])
                    run(flags + ["-o", "Cout", str(gpu_image), shader])
                    host_pixels = image_pixels(cpu_image, width, height)
                    device_pixels = image_pixels(gpu_image, width, height)
                    compare(host_pixels, expected, 2e-6)
                    compare(device_pixels, expected, 2e-6)
                    compare(device_pixels, host_pixels, 2e-6)
        # Exercise the ordinary named-layer setup, not just positional shaders.
        named = run(["--hart", "--groupname", "generated",
                     "--shader", "hart_first", "surface", "--print"])
        compare(pixels(named, 1, 1), reference(1, 1, False), 2e-6)
        # A bare "2" is inferred as int; the shader parameter is float.
        parameter_args = ["--param:type=float", "scale", "2",
                          "--shader", "hart_parameter", "surface", "--print"]
        cpu_parameter = pixels(run(parameter_args), 1, 1)
        gpu_parameter = pixels(run(["--hart"] + parameter_args), 1, 1)
        compare(cpu_parameter, [0.5, 0.5, 2.0], 2e-6)
        compare(gpu_parameter, cpu_parameter, 2e-6)
        suppressed = root / "suppressed.pfm"
        run(["--hart", "--print", "-o", "Cout", str(suppressed), "hart_first"])
        assert not suppressed.exists(), "--print must suppress image writing"
        run(["--hart", "-o", "Cout", str(root / "missing" / "out.pfm"),
             "hart_first"], "Cannot write HART output")
finally:
    shutil.rmtree(root)

print("Generated HART CLI checks passed"
      + ("; CPU/GPU numeric, image, cold-cache and repeated-launch checks passed"
         if args.gpu else ""))
