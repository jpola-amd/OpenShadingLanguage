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
suites = parser.add_mutually_exclusive_group()
suites.add_argument("--loops", action="store_true",
                    help="Run loop runtime cases instead of the basic runtime cases")
suites.add_argument("--derivatives", action="store_true",
                    help="Run derivative runtime cases instead of the basic runtime cases")
args = parser.parse_args()
if (args.loops or args.derivatives) and not args.gpu:
    parser.error("--loops and --derivatives require --gpu")
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


def reference(width, height, evaluate):
    result = []
    for y in range(height):
        v = 0.5 if height == 1 else y / (height - 1)
        for x in range(width):
            u = 0.5 if width == 1 else x / (width - 1)
            result.extend(evaluate(u, v))
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


def check_render(shader_args, width, height, expected):
    grid = ["-g", str(width), str(height)]
    cpu = pixels(run(["-t", "1"] + grid + ["--print"] + shader_args),
                 width, height)
    # CPU testshade prints six significant digits.
    compare(cpu, expected, 5e-6)
    for cache_options in (["--hart-no-cache"], []):
        flags = ["--hart", "-v", "--warmup", "--iters", "3"]
        flags += cache_options + grid
        output = run(flags + ["--print"] + shader_args)
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
            + ["-o", "Cout", str(cpu_image)] + shader_args)
        run(flags + ["-o", "Cout", str(gpu_image)] + shader_args)
        host_pixels = image_pixels(cpu_image, width, height)
        device_pixels = image_pixels(gpu_image, width, height)
        compare(host_pixels, expected, 2e-6)
        compare(device_pixels, expected, 2e-6)
        compare(device_pixels, host_pixels, 2e-6)


def connected_group(consumer, parameters=None, producer="hart_group_producer"):
    return (["--shader", producer, "producer"]
            + (parameters or [])
            + ["--shader", consumer, "consumer",
               "--connect", "producer", "value", "consumer", "value"])


def comparison_result(a, b):
    return (int(a < b) + 2 * int(a <= b),
            int(a > b) + 2 * int(a >= b),
            int(a == b) + 2 * int(a != b))


def loop_result(u, v, count=-1, value=None, reuse=False):
    if count < 0:
        count = 4 if u > v else (0 if u < v else 1)
    if value is None:
        value = u + v
    total = sum(math.sin(value + i) for i in range(count))
    return (value if reuse else u, v, total)


def derivative_result(u, v, width, height, count=1):
    value = sum(math.sin(u * v + i) for i in range(count))
    gradient = sum(math.cos(u * v + i) for i in range(count))
    return (value, gradient * v / max(1, width - 1),
            gradient * u / max(1, height - 1))


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
        ["--entryoutput", "Cout"],
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

    if args.loops:
        for operation in ("break", "continue", "dowhile"):
            shader = "hart_loop_" + operation
            error = "unsupported operation '" + operation + "'"
            run(["--hart", "-v", shader], error)
            run(["--hart", "-v", "--shader", shader, "producer",
                 "--shader", "hart_sine", "consumer"], error)
        for optimize in ("10", "3"):
            for shader in ("hart_for", "hart_loop"):
                loop_args = ["--llvm_opt", optimize] + connected_group(shader)
                for width, height in ((1, 1), (3, 2), (37, 5)):
                    check_render(loop_args, width, height,
                                 reference(width, height, loop_result))
                cases = [
                    (connected_group(shader, ["--param", "count", str(count)]),
                     lambda u, v, count=count: loop_result(u, v, count=count))
                    for count in (0, 1, 4)
                ]
                # Reusing the input after a zero-iteration loop must still
                # execute its producer, while nonempty loops reuse its result.
                cases.extend([
                    (connected_group(shader, ["--param", "reuse", "1"]),
                     lambda u, v: loop_result(u, v, reuse=True)),
                    (["--param:type=float", "value", "1",
                      "--param", "count", "4", shader],
                     lambda u, v: loop_result(u, v, count=4, value=1)),
                ])
                for loop_args, evaluate in cases:
                    flags = ["--llvm_opt", optimize, "-O0", "-g", "3", "3", "--print"]
                    expected = reference(3, 3, evaluate)
                    cpu = pixels(run(flags + loop_args), 3, 3)
                    gpu = pixels(run(["--hart", "--hart-no-cache", "--warmup",
                                      "--iters", "3"] + flags + loop_args), 3, 3)
                    compare(cpu, expected, 5e-6)
                    compare(gpu, expected, 2e-6)
                    compare(gpu, cpu, 5e-6)

    if args.derivatives:
        connected = connected_group("hart_deriv_consumer",
                                    producer="hart_deriv_producer")
        flow = connected_group("hart_deriv_consumer", producer="hart_deriv_flow")
        for optimize in ("10", "3"):
            flags = ["--llvm_opt", optimize]
            for shader_args, varying in ((["hart_deriv"], False),
                                         (connected, False), (flow, True)):
                for width, height in ((1, 1), (3, 2), (37, 5)):
                    expected = reference(
                        width, height,
                        lambda u, v: derivative_result(
                            u, v, width, height,
                            (3 if u > v else (0 if u < v else 1))
                            if varying else 1),
                    )
                    check_render(flags + shader_args, width, height, expected)
            for shader, evaluate in (
                ("hart_deriv_arithmetic",
                 lambda u, v: (v / (v + 1) - u, -0.5, 1 / (v + 1)**2)),
                ("hart_deriv_color",
                 lambda u, v: (math.cos(u * v) * v * 0.5,
                               math.cos(u + v), -0.5 * math.cos(u - v))),
            ):
                check_render(flags + [shader], 3, 2, reference(3, 2, evaluate))
            for specialize in ("-O0", "-O2"):
                for shader_args, value in (
                    (["--param:type=float", "value", "3",
                      "hart_deriv_consumer"], 3),
                    (["--param:type=float", "scale", "0"] + connected, 0),
                ):
                    constant_flags = flags + [specialize, "-g", "3", "2", "--print"]
                    expected = reference(3, 2, lambda u, v: (value, 0, 0))
                    cpu = pixels(run(constant_flags + shader_args), 3, 2)
                    gpu = pixels(run(["--hart", "--hart-no-cache", "--warmup",
                                      "--iters", "3"] + constant_flags + shader_args),
                                 3, 2)
                    compare(cpu, expected, 5e-6)
                    compare(gpu, expected, 2e-6)
                    compare(gpu, cpu, 5e-6)

    if args.gpu and not (args.loops or args.derivatives):
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
             "--shader", "hart_sine", "second",
             "--shader", "hart_first", "third", "-v"], "one or two shader layers")
        # Even an unused producer must be validated before optimization.
        for shader in ("hart_closure", "hart_string", "hart_printf",
                       "hart_texture", "hart_userdata"):
            run(["--hart", "--shader", shader, "producer",
                 "--shader", "hart_sine", "consumer", "-v"], "HART")

        connected = connected_group("hart_group_consumer")
        arithmetic = lambda u, v: (u, v, u + v)
        sine = lambda u, v: (u, v, math.sin(u + v))
        branch = lambda u, v: (u, v, math.sin(u + v) if u > v else 0)
        for shader_args, evaluate in (
            (["hart_first"], arithmetic), (["hart_sine"], sine),
            (["--llvm_opt", "10"] + connected, sine),
            (["--llvm_opt", "3"] + connected, sine),
            (["--llvm_opt", "10"] + connected_group("hart_branch"), branch),
            (["--llvm_opt", "3"] + connected_group("hart_branch"), branch),
        ):
            for width, height in ((1, 1), (3, 2), (37, 5)):
                check_render(shader_args, width, height,
                             reference(width, height, evaluate))
        for optimize in ("10", "3"):
            # Uniform outcomes, integer conditions/comparisons, and a second
            # use after reconvergence complement the divergent image tests.
            cases = [
                (connected_group("hart_branch",
                                 ["--param:type=float", "bias", "-2"]), sine),
                (connected_group("hart_branch",
                                 ["--param:type=float", "bias", "2"]),
                 lambda u, v: (u, v, 0)),
                (connected_group("hart_branch_reuse"),
                 lambda u, v: (math.sin(u + v) if u > v else 0, v, u + v)),
                (["--param:type=float", "value", "1", "hart_branch"],
                 lambda u, v: (u, v, math.sin(1) if u > v else 0)),
                (["hart_compare"], comparison_result),
                (["--param", "integer_inputs", "-1", "hart_compare"],
                 lambda u, v: comparison_result(int(u > 0.5) - 1,
                                                int(v > 0.5) - 1)),
            ]
            for shader_args, evaluate in cases:
                flags = ["--llvm_opt", optimize, "-O0", "-g", "3", "3", "--print"]
                expected = reference(3, 3, evaluate)
                cpu = pixels(run(flags + shader_args), 3, 3)
                gpu = pixels(run(["--hart", "--hart-no-cache", "--warmup",
                                  "--iters", "3"] + flags + shader_args), 3, 3)
                compare(cpu, expected, 5e-6)
                compare(gpu, expected, 2e-6)
                compare(gpu, cpu, 5e-6)
            parameter_group = ["--llvm_opt", optimize, "-O0",
                               "--param:type=float", "scale", "2"] + connected
            expected = [0.5, 0.5, math.sin(2)]
            compare(pixels(run(["--print"] + parameter_group), 1, 1),
                    expected, 5e-6)
            compare(pixels(run(["--hart", "--print", "--hart-no-cache",
                                "--warmup", "--iters", "3"] + parameter_group),
                           1, 1), expected, 2e-6)
        # Exercise the ordinary named-layer setup, not just positional shaders.
        named = run(["--hart", "--groupname", "generated",
                     "--shader", "hart_first", "surface", "--print"])
        compare(pixels(named, 1, 1), reference(1, 1, arithmetic), 2e-6)
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

suite = "derivative" if args.derivatives else ("loop" if args.loops else "CLI")
print("Generated HART " + suite + " checks passed"
      + ("; CPU/GPU numeric, image, cold-cache and repeated-launch checks passed"
         if args.gpu else ""))
