# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
import colorsys
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import time
import uuid


parser = argparse.ArgumentParser()
parser.add_argument("testshade")
parser.add_argument("--oslc", required=True)
parser.add_argument("--gpu", action="store_true")
suites = parser.add_mutually_exclusive_group()
suites.add_argument("--loops", action="store_true",
                    help="Run loop runtime cases instead of the basic runtime cases")
suites.add_argument("--control-flow", action="store_true",
                    help="Run integer operators, loop exits and function/shader returns")
suites.add_argument("--aggregates", action="store_true",
                    help="Run array/struct layout, derivatives and bounds cases")
suites.add_argument("--strings", action="store_true",
                    help="Run hashed string parameters, copies, comparisons and connections")
suites.add_argument("--selectors", action="store_true",
                    help="Run integer/string hashes and checked dynamic noise selection")
suites.add_argument("--diagnostics", action="store_true",
                    help="Run bounded print/warning/error payloads and launch resets")
suites.add_argument("--interactive-userdata", action="store_true",
                    help="Run numeric interactive defaults beneath userdata lookup")
suites.add_argument("--derivatives", action="store_true",
                    help="Run derivative runtime cases instead of the basic runtime cases")
suites.add_argument("--surface", action="store_true",
                    help="Run surface globals and vector math runtime cases")
suites.add_argument("--filterwidth", action="store_true",
                    help="Run scalar and triple filterwidth runtime cases")
suites.add_argument("--noise", action="store_true",
                    help="Run numeric Perlin noise runtime cases")
suites.add_argument("--noise-families", action="store_true",
                    help="Run periodic, cell, hash and named noise runtime cases")
suites.add_argument("--gabor", action="store_true",
                    help="Run literal aliases, Gabor options, filtering and input guards")
suites.add_argument("--math", action="store_true",
                    help="Run scalar and triple math runtime cases")
suites.add_argument("--numeric-math", action="store_true",
                    help="Run transcendental, geometric and IEEE classification cases")
suites.add_argument("--splines", action="store_true",
                    help="Run spline derivatives, knot bounds and nonfinite guards")
suites.add_argument("--colors", action="store_true",
                    help="Run built-in color conversions and color-system shadeops")
suites.add_argument("--procedural", action="store_true",
                    help="Run connected procedural material runtime cases")
suites.add_argument("--textures", action="store_true",
                    help="Run explicit HART texture sampler runtime cases")
suites.add_argument("--texture-alpha", action="store_true",
                    help="Run HART texture alpha value and derivative cases")
suites.add_argument("--texture-channels", action="store_true",
                    help="Run literal HART texture channel-offset cases")
suites.add_argument("--texture-materials", action="store_true",
                    help="Run channel-selecting alpha-blended material cases")
suites.add_argument("--matrices", action="store_true",
                    help="Run numeric matrix and matrix-transform runtime cases")
suites.add_argument("--spaces", action="store_true",
                    help="Run literal common/object/shader space runtime cases")
suites.add_argument("--geometry", action="store_true",
                    help="Run geometry globals and composed material runtime cases")
suites.add_argument("--groups", action="store_true",
                    help="Run numeric multilayer chain runtime cases")
suites.add_argument("--topology", action="store_true",
                    help="Run diamond, join and lazy dependency runtime cases")
suites.add_argument("--materials", action="store_true",
                    help="Run composed multilayer material runtime cases")
suites.add_argument("--fused", action="store_true",
                    help="Compare split and fused generated HART callables")
suites.add_argument("--fused-local", action="store_true",
                    help="Compare scratch and callable-local HART group storage")
suites.add_argument("--fused-benchmark", action="store_true",
                    help="Benchmark numeric, textured and closure-heavy grids: "
                         "cold no-cache, separate cache priming and three rotated "
                         "split/fused-scratch/fused-local trials")
args = parser.parse_args()
if (args.loops or args.control_flow or args.aggregates or args.strings or args.selectors or args.diagnostics or args.interactive_userdata or args.derivatives or args.surface or args.filterwidth
        or args.noise or args.noise_families or args.gabor or args.math or args.numeric_math or args.splines or args.colors
        or args.procedural or args.textures or args.texture_alpha
        or args.texture_channels or args.texture_materials or args.matrices
        or args.spaces or args.geometry or args.groups
        or args.topology or args.materials or args.fused
        or args.fused_local or args.fused_benchmark) and not args.gpu:
    parser.error("Runtime suites require --gpu")
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
fixtures = Path(__file__).resolve().parent
env = os.environ.copy()
env["TESTSHADE_OPTIX"] = "0"
env["TESTSHADE_HART"] = "0"
env["TESTSHADE_FUSED"] = "0"
env["TESTSHADE_BATCHED"] = "0"
env["TESTSHADE_RS_BITCODE"] = "0"
if args.fused_benchmark:
    for key in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT", "TESTSHADE_LLVM_JIT_FMA",
                "OSL_OPTIONS", "OSL_LLVM_DEBUG", "OSL_DEBUG_OUTPUT_CPP"):
        env.pop(key, None)
root = Path.cwd() / ("hart-generated-check-" + uuid.uuid4().hex)
root.mkdir()


def run(arguments, error=None, extra_env=None, error_after_launch=False,
        wall_times=None):
    start = time.perf_counter()
    result = subprocess.run(
        [testshade] + arguments, cwd=root, env={**env, **(extra_env or {})},
        capture_output=True, text=True, timeout=300,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    if wall_times is not None:
        assert math.isfinite(wall_ms) and wall_ms >= 0
        wall_times.append(wall_ms)
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, "Unexpected success:\n" + output
        assert error.lower() in output.lower(), output
        assert ("Launching HART grid" in output) == error_after_launch, output
    return output


def compile_fixture(source, name=None, defines=(), options=()):
    start = time.perf_counter()
    result = subprocess.run(
        [oslc] + list(options)
        + ["-I" + str(fixtures.parents[1] / "src" / "shaders")]
        + ["-D" + define for define in defines]
        + ["-o", str(root / ((name or source.stem) + ".oso")), str(source)],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    wall_ms = (time.perf_counter() - start) * 1000
    assert result.returncode == 0, result.stdout + result.stderr
    assert math.isfinite(wall_ms) and wall_ms >= 0
    return wall_ms


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


def compare(actual, expected, tolerance=2e-6, relative_tolerance=1e-6):
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert math.isclose(a, b, abs_tol=tolerance, rel_tol=relative_tolerance), (
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


def check_render(shader_args, width, height, expected, compare_device=compare):
    grid = ["-g", str(width), str(height)]
    cpu = pixels(run(["-t", "1"] + grid + ["--print"] + shader_args),
                 width, height)
    # CPU text rounds to six significant digits; allow a small float-computation
    # margin beyond rounding. Full-precision images still use 2e-6 below.
    compare(cpu, expected, 6e-6)
    for cache_options in (["--hart-no-cache"], []):
        flags = ["--hart", "-v", "--warmup", "--iters", "3"]
        flags += cache_options + grid
        output = run(flags + ["--print"] + shader_args)
        if cache_options:
            assert "HART pipeline cache disabled" in output, output
        assert output.count("Launching HART grid") == 4, output
        gpu = pixels(output, width, height)
        compare_device(gpu, expected)
        compare(gpu, cpu, 6e-6)

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
        compare_device(device_pixels, expected)
        compare_device(device_pixels, host_pixels)


def connected_group(consumer, parameters=None, producer="hart_group_producer"):
    return (["--shader", producer, "producer"]
            + (parameters or [])
            + ["--shader", consumer, "consumer",
               "--connect", "producer", "value", "consumer", "value"])


def comparison_result(a, b):
    return (int(a < b) + 2 * int(a <= b),
            int(a > b) + 2 * int(a >= b),
            int(a == b) + 2 * int(a != b))


def control_ops_result(u, v):
    s = min(31, int(32*u))
    a, b = -123456789+s, 0x01347bdf+int(16*v)
    if v < 0.125:
        return (a & b) & 65535, ((a | b) >> 16) & 65535, (a ^ b) & 65535
    if v < 0.375:
        return (~a) & 65535, ((a << s) >> 16) & 65535, (a >> s) & 65535
    if v < 0.625:
        x, y = s-16, s%5-2
        remainder = x-int(x/y)*y if y else 0
        return (int(bool(x) and bool(y)) + 2*int(bool(x) or bool(y))
                + 4*int(not x) + 8*int(not y),
                (x<y) + 2*(x<=y) + 4*(x>y) + 8*(x>=y)
                + 16*(x==y) + 32*(x!=y), remainder)
    if v < 0.875:
        k = s-16
        return 10*k+7, k, 1
    return 7, 7, 10


def control_flow_result(u, v, width, height):
    count = int(4*u)
    total = sum(20*i+2 for i in range(1, max(1, count)+1) if i != 2)
    if v > 0.5:
        return total, 0, -1
    h = u+v + (2 if count >= 3 else -1)
    return (total + sum(j for j in (1, 3, 4) if j <= count),
            h + 2/max(1, width-1) + 4/max(1, height-1),
            min(count, 3)+10)


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


def filterwidth_scalar_result(u, v, width, height, scale=1):
    gradient = abs(scale * math.cos(scale * u * v))
    footprint = math.hypot(v / max(1, width - 1), u / max(1, height - 1))
    return (gradient * footprint, 0, 0)


def surface_globals(u, v, width, height, field=-1):
    if field < 0:
        field = int(u > 0.25) + 2 * int(v > 0.25) + 4 * int(u > v)
    return ((u, v, 1), (1 / max(1, width - 1), 0, 0),
            (0, 1 / max(1, height - 1), 0), (0, 0, 1), (0, 0, 1),
            (1, 0, 0), (0, 1, 0), (0, 0, 0))[field]


def surface_result(u, v, width, height, operation=-1, planar=False, value=None):
    if operation < 0:
        operation = int(u > 0.5) + 2 * int(v > 0.5)
    du, dv = 1 / max(1, width - 1), 1 / max(1, height - 1)
    if value is None:
        value = (u, 2 * v, 0 if planar else 1 + u * v)
        dx = (du, 0, 0 if planar else v * du)
        dy = (0, 2 * dv, 0 if planar else u * dv)
    else:
        dx = dy = (0, 0, 0)
    square = sum(x * x for x in value)
    length = math.sqrt(square)
    projections = [sum(x * d for x, d in zip(value, deriv)) for deriv in (dx, dy)]
    if operation in (0, 1) and length == 0:
        # OSL's zero-length normalize/length return zero derivatives.
        return (0, 0, 0)
    if operation == 0:
        weighted = sum((i + 1) * x for i, x in enumerate(value))
        derivatives = [
            sum((i + 1) * d for i, d in enumerate(deriv)) / length
            - weighted * projection / length**3
            for deriv, projection in zip((dx, dy), projections)
        ]
        return (weighted / length, *derivatives)
    if operation == 1:
        return (length, *(p / length for p in projections))
    if operation == 2:
        return (square, *(2 * p for p in projections))
    return (value[2], dx[2], dy[2])


def noise_arguments(optimize, report=0, dimension=0, kind=-1, signed_noise=1,
                    constant_inputs=0, arithmetic=0, offset_u=0, offset_v=0,
                    specialize="-O2"):
    offsets = ["--param:type=float", "offset_u", str(offset_u),
               "--param:type=float", "offset_v", str(offset_v)]
    producer = ["--llvm_opt", optimize, specialize,
                "--param", "dimension", str(dimension),
                "--param", "kind", str(kind),
                "--param", "signed_noise", str(signed_noise),
                "--param", "constant_inputs", str(constant_inputs)] + offsets
    if report == 0 and not arithmetic:
        return producer + ["hart_noise"]
    return (producer + ["--shader", "hart_noise", "producer"]
            + ["--param", "report", str(report),
               "--param", "arithmetic", str(arithmetic)] + offsets
            + ["--shader", "hart_noise_consumer", "consumer",
               "--connect", "producer", "Cout", "consumer", "value"])


def noise_component_indices(width, height):
    return [3 * (y * width + x)
            + int(11.99 * (0.5 if width == 1 else x / (width - 1))) % 3
            for y in range(height) for x in range(width)]


def noise_cpu_image(shader_args, width, height):
    image = root / "noise-cpu.pfm"
    if image.exists():
        image.unlink()
    run(["-t", "1", "-g", str(width), str(height),
         "-o", "Cout", str(image)] + shader_args)
    return image_pixels(image, width, height)


def noise_reference(width, height, **options):
    values = noise_cpu_image(noise_arguments(**options), width, height)
    derivatives = noise_cpu_image(noise_arguments(report=1, **options),
                                  width, height)
    indices = noise_component_indices(width, height)
    compare(derivatives[0::3], [values[i] for i in indices], 2e-6)
    if options.get("constant_inputs"):
        compare(derivatives[1::3], [0] * (width * height), 0)
        compare(derivatives[2::3], [0] * (width * height), 0)
        return values, derivatives, ([0] * len(values), [0] * len(values))

    # Binary-exact h avoids decimal step error. Central differences have
    # O(h^2) truncation and O(float_epsilon/h) cancellation error. A separate
    # 5e-4 absolute bound on dF/du and dF/dv covers both for these unit-scale
    # coordinates; it is NOT the direct CPU/GPU rounding tolerance.
    step = 1 / 512
    gradients = []
    for axis, extent, packed in (("offset_u", width, 1), ("offset_v", height, 2)):
        plus = noise_cpu_image(noise_arguments(**{**options, axis: step}),
                               width, height)
        minus = noise_cpu_image(noise_arguments(**{**options, axis: -step}),
                                width, height)
        finite_difference = [(p - m) / (2 * step) for p, m in zip(plus, minus)]
        spacing = 1 / max(1, extent - 1)
        compare([d / spacing for d in derivatives[packed::3]],
                [finite_difference[i] for i in indices], 5e-4)
        gradients.append([d * spacing for d in finite_difference])
    return values, derivatives, gradients


def compare_noise(actual, expected, report):
    # CPU SIMD and scalar HIP interpolate Perlin's corners in different orders.
    # Only derivatives/footprints need the extra rounding margin.
    if report == 1:
        compare(actual[0::3], expected[0::3], 2e-6)
        compare(actual[1::3], expected[1::3], 4e-6)
        compare(actual[2::3], expected[2::3], 4e-6)
    else:
        compare(actual, expected, 4e-6 if report in (2, 3) else 2e-6)


def check_noise_render(shader_args, width, height, expected, report=0, full=False,
                       compare_device=None):
    if compare_device is None:
        compare_device = lambda actual, reference: compare_noise(actual, reference, report)
    if full:
        check_render(shader_args, width, height, expected,
                     compare_device=compare_device)
        return
    # The dimension/type matrix needs one GPU image per case, not the full
    # cache/image/repeat cross product used for the connected representatives.
    grid = ["-g", str(width), str(height)]
    compare(pixels(run(["-t", "1", "--print"] + grid + shader_args),
                   width, height), expected, 6e-6)
    image = root / "noise-gpu.pfm"
    if image.exists():
        image.unlink()
    output = run(["--hart", "--hart-no-cache", "-v"] + grid
                 + ["-o", "Cout", str(image)] + shader_args)
    assert "HART pipeline cache disabled" in output, output
    assert output.count("Launching HART grid") == 1, output
    actual = image_pixels(image, width, height)
    compare_device(actual, expected)
    return actual


def check_noise_relationship(signed, unsigned, packed=False):
    compare(unsigned, [0.5 * (value + (not packed or i % 3 == 0))
                       for i, value in enumerate(signed)], 2e-6)


def check_noise_suite():
    for optimize in ("10", "3"):
        print("Checking HART noise at LLVM level " + optimize, flush=True)
        cpu, gpu = {}, {}
        # Four dimensions, three return types, and all three triple components
        # fit in 36 samples. Two launches per sign check value-only RGB and
        # connected (value, Dx, Dy), including all components without reduction.
        for signed in (1, 0):
            options = dict(optimize=optimize, signed_noise=signed)
            cpu[signed] = noise_reference(12, 3, **options)
            gpu[signed] = [
                check_noise_render(noise_arguments(report=report, **options),
                                   12, 3, cpu[signed][report], report=report)
                for report in (0, 1)
            ]
        for report in (0, 1):
            check_noise_relationship(cpu[1][report], cpu[0][report], bool(report))
            check_noise_relationship(gpu[1][report], gpu[0][report], bool(report))
        assert max(abs(d) for d in cpu[1][1][1::3]) > 0.01
        assert max(abs(d) for d in cpu[1][1][2::3]) > 0.01

        # Compose scalar and triple filterwidth with all noise dimensions/types.
        # Selected components also have an exact independent hypot(Dx,Dy) check;
        # the remaining RGB components have the finite-difference cross-check.
        indices = noise_component_indices(12, 3)
        footprint = [math.hypot(x, y) for x, y in zip(*cpu[1][2])]
        selected = [math.hypot(x, y)
                    for x, y in zip(cpu[1][1][1::3], cpu[1][1][2::3])]
        for report in (2, 3):
            shader_args = noise_arguments(optimize, report=report)
            expected = noise_cpu_image(shader_args, 12, 3)
            if report == 2:
                compare(expected, footprint, 5e-4)
                compare([expected[i] for i in indices], selected, 2e-6)
            else:
                compare(expected, [v for w in selected for v in (w, 0, 0)], 2e-6)
            check_noise_render(shader_args, 12, 3, expected, report=report)

        # Keep expensive cold/cache-enabled, image and repeated-launch coverage
        # on a connected 4D vector case with arithmetic on both sides of noise.
        for width, height in ((1, 1), (3, 2), (37, 5)):
            options = dict(optimize=optimize, dimension=4, kind=2, arithmetic=1)
            _, derivatives, _ = noise_reference(width, height, **options)
            check_noise_render(noise_arguments(report=1, **options),
                               width, height, derivatives, report=1, full=True)

        for specialize in ("-O0", "-O2"):
            for signed in (1, 0):
                options = dict(optimize=optimize, specialize=specialize,
                               signed_noise=signed, constant_inputs=1)
                _, derivatives, _ = noise_reference(12, 3, **options)
                actual = check_noise_render(noise_arguments(report=1, **options),
                                            12, 3, derivatives, report=1)
                compare(actual[1::3], [0] * 36, 0)
                compare(actual[2::3], [0] * 36, 0)
            actual = check_noise_render(
                noise_arguments(optimize, specialize=specialize,
                                report=2, constant_inputs=1),
                12, 3, [0] * 108, report=2,
            )
            compare(actual, [0] * 108, 0)


def noise_family_arguments(optimize, report=0, periodic=0, named=0, shift=0,
                           canonical_periods=0, offset_u=0, offset_v=0):
    producer = ["--llvm_opt", optimize,
                "--param", "periodic", str(periodic),
                "--param", "named", str(named),
                "--param", "shift", str(shift),
                "--param", "canonical_periods", str(canonical_periods),
                "--param:type=float", "offset_u", str(offset_u),
                "--param:type=float", "offset_v", str(offset_v)]
    if report == 0:
        return producer + ["hart_noise_families"]
    return (producer + ["--shader", "hart_noise_families", "producer",
                        "--param", "report", str(report),
                        "--shader", "hart_noise_consumer", "consumer",
                        "--connect", "producer", "Cout", "consumer", "value"])


def compare_noise_families(actual, expected, report, periodic):
    compare(actual[0::3] if report else actual,
            expected[0::3] if report else expected, 2e-6)
    if not report:
        return
    for row in range(33):
        family = (row % 11) // (3 if periodic else 2)
        for channel in (1, 2):
            values = actual[3 * row * 65 + channel:3 * (row + 1) * 65:3]
            reference = expected[3 * row * 65 + channel:3 * (row + 1) * 65:3]
            if family in (2, 3):
                compare(values, [0] * 65, 0)
                compare(reference, [0] * 65, 0)
            else:
                # Retain Perlin's measured derivative margin; simplex uses
                # the ordinary image tolerance until measured otherwise.
                compare(values, reference, 4e-6 if family < 2 else 2e-6)


def noise_family_relationships(values, report, periodic):
    stride = 3 if periodic else 2
    pairs = [(0, 1)] if periodic else [(0, 1), (4, 5)]
    for kind in range(3):
        for signed, unsigned in pairs:
            rows = [(kind * 11 + family * stride) * 65 * 3
                    for family in (signed, unsigned)]
            check_noise_relationship(
                values[rows[0]:rows[0] + 65 * 3],
                values[rows[1]:rows[1] + 65 * 3], bool(report),
            )


def noise_family_reference(optimize, periodic, finite_difference=False, **options):
    arguments = dict(optimize=optimize, periodic=periodic, **options)
    images = [noise_cpu_image(noise_family_arguments(report=report, **arguments),
                              65, 33) for report in (0, 1)]
    indices = noise_component_indices(65, 33)
    compare(images[1][0::3], [images[0][i] for i in indices], 2e-6)
    for report, values in enumerate(images):
        compare_noise_families(values, values, report, periodic)
        noise_family_relationships(values, report, periodic)
    if finite_difference:
        # Cell/hash are intentionally discontinuous. Check only differentiable
        # families, including simplex probes offset from lattice boundaries.
        smooth = [i for i in range(65 * 33)
                  if ((i // 65) % 11) // (3 if periodic else 2) not in (2, 3)]
        step = 1 / 512
        for axis, extent, channel in (("offset_u", 65, 1), ("offset_v", 33, 2)):
            plus, minus = [
                noise_cpu_image(noise_family_arguments(
                    **{**arguments, axis: sign * step}), 65, 33)
                for sign in (1, -1)
            ]
            gradient = [(p - m) / (2 * step) for p, m in zip(plus, minus)]
            compare([images[1][3 * i + channel] * (extent - 1) for i in smooth],
                    [gradient[indices[i]] for i in smooth], 5e-4)
    return images


def check_noise_family_suite():
    for optimize in ("10", "3"):
        print("Checking HART noise families at LLVM level " + optimize, flush=True)
        for periodic in (0, 1):
            reference = noise_family_reference(optimize, periodic, finite_difference=True)
            # One matrix per value/derivative mode covers every dimension,
            # return type and family. Aliases and periodic shifts reuse it.
            variants = [({}, reference),
                        (dict(named=1, canonical_periods=periodic), None)]
            if periodic:
                variants.append((dict(shift=1), None))
            for options, cpu in variants:
                if cpu is None:
                    cpu = noise_family_reference(optimize, periodic, **options)
                for report in (0, 1):
                    def compare_family(actual, expected):
                        compare_noise_families(actual, expected, report, periodic)
                    compare_family(cpu[report], reference[report])
                    actual = check_noise_render(
                        noise_family_arguments(optimize, report=report,
                                               periodic=periodic, **options),
                        65, 33, cpu[report], report=report,
                        compare_device=compare_family,
                    )
                    compare_family(actual, reference[report])
                    noise_family_relationships(actual, report, periodic)


def math_arguments(optimize, report=0, constant_inputs=0, specialize="-O2"):
    producer = ["--llvm_opt", optimize, specialize,
                "--param", "constant_inputs", str(constant_inputs)]
    if report == 0:
        return producer + ["hart_math"]
    return (producer + ["--shader", "hart_math", "producer",
                        "--shader", "hart_math_consumer", "consumer",
                        "--connect", "producer", "Cout", "consumer", "value"])


def math_value_gradient(operation, probe, a):
    # These are OSL's chosen derivatives at discontinuities, not finite
    # differences across them. In particular min ties select the first operand,
    # max ties the second, and fmod ignores the divisor's derivatives.
    lo, hi, dlo, dhi = -1, 1, 0, 0
    if probe == 3:
        lo, hi, dlo, dhi = -1 + 0.125 * a, 1 + 0.25 * a, 0.125, 0.25
    elif probe == 4:
        lo = hi = 0
    if operation == 0:
        return abs(a), 1 if a >= 0 else -1
    if operation == 1:
        return (a, 1) if a <= 0.5 * a else (0.5 * a, 0.5)
    if operation == 2:
        return (a, 1) if a > 0.5 * a else (0.5 * a, 0.5)
    if operation == 3:
        # stdosl implements clamp as max(min(a, hi), lo).
        value, gradient = (a, 1) if a <= hi else (hi, dhi)
        return (value, gradient) if value > lo else (lo, dlo)
    if operation == 4:
        weight, gradient = ((0.25, 0) if probe == 4 else
                            (0.5 + 0.125 * a, 0.125))
        return a * (1 - 0.5 * weight), 1 - 0.5 * weight - 0.5 * a * gradient
    if operation == 5:
        return int(a >= 0.5 * a), 0
    if operation == 6:
        if a < lo:
            return 0, 0
        if a >= hi:
            return 1, 0
        t = (a - lo) / (hi - lo)
        dt = ((1 - dlo) - t * (dhi - dlo)) / (hi - lo)
        return (3 - 2 * t) * t * t, 6 * t * (1 - t) * dt
    if operation in (7, 8):
        return (math.floor(a) if operation == 7 else math.ceil(a)), 0
    if operation == 9:
        divisor = -1 if probe == 3 else (0 if probe == 4 else 1 + 0.25 * a)
        # Even a zero divisor returns the numerator's derivatives.
        return math.fmod(a, divisor) if divisor else 0, 1
    if operation == 10:
        return math.cos(a), -math.sin(a)
    if operation == 11:
        return (math.sqrt(a), 0.5 / math.sqrt(a)) if a > 0 else (0, 0)
    exponent = (2, 3, 1.5, 2 + 0.125 * a, 1.5)[probe]
    if a == 0 or (a < 0 and exponent != int(exponent)):
        return 0, 0
    value = a**exponent
    gradient = exponent * a**(exponent - 1)
    if probe == 3 and a > 0:
        gradient += 0.125 * math.log(a) * value
    return value, gradient


def math_reference(width, height, report=0, constant_inputs=0):
    def evaluate(u, v):
        column = int(64 * u)
        operation, probe = divmod(column, 5)
        kind = int(32 * v) // 11
        s = (8 * (u - 5 * operation / 64)
             + 16 * (v - 11 * kind / 32) - 2.75)
        if constant_inputs:
            s = -0.5
        if probe == 4 and operation < 4:
            a = int(2 * s)
            value = (abs(a), min(a, -a), max(a, -a), max(min(a, 2), -2))[operation]
            return (value, 0, 0) if report else (value, value, value)
        inputs = ((s, s, s) if kind == 0 else
                  (s + 0.25, -0.5 * (s + 0.125), 2 * s))
        # Bound pow's approximation error without widening the image tolerance.
        # OSL's fast_pow can differ from analytical pow by about 1e-5 relative.
        if operation == 12:
            inputs = tuple(0.03125 * a for a in inputs)
        results = [math_value_gradient(operation, probe, a) for a in inputs]
        if report == 0:
            return tuple(value for value, gradient in results)
        component = probe % 3
        value, gradient = results[component]
        scale = 1 if kind == 0 else (1, -0.5, 2)[component]
        if operation == 12:
            scale *= 0.03125
        if constant_inputs:
            gradient = 0
        return (value, gradient * scale * 8 / max(1, width - 1),
                gradient * scale * 16 / max(1, height - 1))
    return reference(width, height, evaluate)


def check_math_render(shader_args, expected, constant_inputs=False):
    # Each packed case needs one CPU image and one GPU image, not a
    # cache/warmup/text cross product. All comparisons retain the 2e-6 bound.
    images = [root / "math-cpu.pfm", root / "math-gpu.pfm"]
    results = []
    for flags, image in zip((["-t", "1"], ["--hart", "--hart-no-cache", "-v"]),
                            images):
        if image.exists():
            image.unlink()
        output = run(flags + ["-g", "65", "33", "-o", "Cout", str(image)]
                     + shader_args)
        if "--hart" in flags:
            assert "HART pipeline cache disabled" in output, output
            assert output.count("Launching HART grid") == 1, output
        values = image_pixels(image, 65, 33)
        compare(values, expected, 2e-6)
        if constant_inputs:
            compare(values[1::3], [0] * (65 * 33), 0)
            compare(values[2::3], [0] * (65 * 33), 0)
        results.append(values)
    compare(results[1], results[0], 2e-6)


def check_math_suite():
    for optimize in ("10", "3"):
        print("Checking HART math at LLVM level " + optimize, flush=True)
        for report in (0, 1):
            check_math_render(math_arguments(optimize, report=report),
                              math_reference(65, 33, report=report))
        for specialize in ("-O0", "-O2"):
            check_math_render(
                math_arguments(optimize, report=1, constant_inputs=1,
                               specialize=specialize),
                math_reference(65, 33, report=1, constant_inputs=1),
                constant_inputs=True,
            )
    # The center sample is a connected color smoothstep with nonzero Dx/Dy.
    # Keep repeated launch and cache coverage on this single representative.
    check_render(math_arguments("3", report=1), 1, 1,
                 math_reference(1, 1, report=1))


def procedural_arguments(optimize, recipe=0, octaves=3, filtered=1, report=0,
                         specialize="-O2"):
    return ["--llvm_opt", optimize, specialize,
            "--param", "recipe", str(recipe), "--param", "octaves", str(octaves),
            "--shader", "hart_procedural", "producer",
            "--param", "recipe", str(recipe), "--param", "filtered", str(filtered),
            "--param", "report", str(report),
            "--shader", "hart_procedural_consumer", "consumer",
            "--connect", "producer", "Cout", "consumer", "value"]


def check_procedural_edges(image, width, height, report):
    left, right = [], []
    for row in range(height):
        start, end = 3 * row * width, 3 * ((row + 1) * width - 1)
        left.extend(image[start:start + 3])
        right.extend(image[end:end + 3])
    compare_noise(left, right, report)
    compare_noise(image[:3 * width], image[-3 * width:], report)


def check_procedural_suite():
    cases = [
        ("fbm", dict(recipe=0), (0, 1)),
        ("single", dict(recipe=0, octaves=1), (0,)),
        ("pattern", dict(recipe=1), (0, 1)),
        ("filtered", dict(recipe=2), (0, 1)),
        ("sharp", dict(recipe=2, filtered=0), (0,)),
    ]
    for optimize in ("10", "3"):
        print("Checking HART procedural materials at LLVM level " + optimize,
              flush=True)
        cpu, gpu = {}, {}
        for name, options, reports in cases:
            for report in reports:
                shader_args = procedural_arguments(optimize, report=report, **options)
                expected = noise_cpu_image(shader_args, 17, 9)
                actual = check_noise_render(shader_args, 17, 9, expected, report=report)
                for image in (expected, actual):
                    check_procedural_edges(image, 17, 9, report)
                    if report:
                        for channel in (1, 2):
                            if name == "pattern":
                                compare(image[channel::3], [0] * (17 * 9), 0)
                            else:
                                assert max(abs(d) for d in image[channel::3]) > 1e-3
                if not report:
                    cpu[name], gpu[name] = expected, actual
        for images in (cpu, gpu):
            assert max(abs(a - b) for a, b in
                       zip(images["fbm"], images["single"])) > 1e-3
            # A partial ramp weight must differ visibly from the binary edge.
            weights = [(r - 0.125) / 0.75 for r in images["filtered"][0::3]]
            sharp = [(r - 0.125) / 0.75 for r in images["sharp"][0::3]]
            compare(sharp, [round(w) for w in sharp], 2e-6)
            assert any(0.1 < w < 0.9 and abs(w - s) > 0.1
                       for w, s in zip(weights, sharp))

    # Keep the loop/clamp wrapper at OSL O0 and exercise caching/repeated
    # launches only once. The non-square matrix above checks all recipes.
    shader_args = procedural_arguments("10", report=1, specialize="-O0")
    expected = noise_cpu_image(shader_args, 1, 1)
    check_render(shader_args, 1, 1, expected,
                 compare_device=lambda actual, reference: compare_noise(
                     actual, reference, 1))


def texture_mips(image):
    # Power-of-two fixture pyramid: exact box averages, including 2x1 -> 1x1.
    levels = [image]
    while len(image) > 1 or len(image[0]) > 1:
        height, width = len(image), len(image[0])
        sx, sy = min(2, width), min(2, height)
        image = [
            [tuple(sum(image[sy * y + j][sx * x + i][c]
                       for j in range(sy) for i in range(sx)) / (sx * sy)
                   for c in range(len(image[0][0])))
             for x in range(max(1, width // 2))]
            for y in range(max(1, height // 2))
        ]
        levels.append(image)
    return levels


def prepare_texture_images(scale=1):
    images = [
        [[((x + 1) / 8, (y + 1) / 4, 0.875 if (x + y) % 2 else 0.125)
          for x in range(8)] for y in range(4)],
        [[((x + 1) / 8,) for x in range(8)] for y in range(4)],
        [[(0.25, 0.5, 0.75) for x in range(8)] for y in range(4)],
    ]
    images = [[[tuple(scale * c for c in pixel) for pixel in row]
               for row in image] for image in images]
    for name, image in zip(("rgb", "mono", "constant"), images):
        channels = len(image[0][0])
        values = [c for row in reversed(image) for pixel in row for c in pixel]
        path = root / ("hart_texture_" + name + ".pfm")
        with path.open("wb") as stream:
            stream.write(b"PF\n8 4\n-1\n" if channels == 3 else b"Pf\n8 4\n-1\n")
            stream.write(struct.pack("<" + str(len(values)) + "f", *values))
    return [texture_mips(image) for image in images]


def texture_sample(levels, s, t, gradients, linear, wraps, nchannels=3,
                   firstchannel=0):
    height, width = len(levels[0]), len(levels[0][0])
    zeros = (0,) * nchannels
    dsdx, dtdx, dsdy, dtdy = gradients
    footprint = max(1, math.hypot(width * dsdx, height * dtdx),
                    math.hypot(width * dsdy, height * dtdy))
    lod = min(math.log2(footprint), len(levels) - 1)

    def level_sample(level):
        image = levels[level]
        h, w = len(image), len(image[0])

        def fetch(x, y):
            indices = []
            for coordinate, size, wrap in zip((x, y), (w, h), wraps):
                if wrap == "black" and not 0 <= coordinate < size:
                    return zeros
                indices.append(coordinate % size if wrap == "periodic"
                               else min(max(coordinate, 0), size - 1))
            pixel = image[indices[1]][indices[0]]
            return (pixel[firstchannel:] + zeros)[:nchannels]

        if not linear:
            return fetch(math.floor(s * w), math.floor(t * h)), zeros, zeros
        x, y = s * w - 0.5, t * h - 0.5
        ix, iy = math.floor(x), math.floor(y)
        a, b = x - ix, y - iy
        p00, p10 = fetch(ix, iy), fetch(ix + 1, iy)
        p01, p11 = fetch(ix, iy + 1), fetch(ix + 1, iy + 1)
        value = tuple((1 - b) * ((1 - a) * p00[c] + a * p10[c])
                      + b * ((1 - a) * p01[c] + a * p11[c])
                      for c in range(nchannels))
        ds = [w * ((1 - b) * (p10[c] - p00[c]) + b * (p11[c] - p01[c]))
              for c in range(nchannels)]
        dt = [h * ((1 - a) * (p01[c] - p00[c]) + a * (p11[c] - p10[c]))
              for c in range(nchannels)]
        return (value, tuple(ds[c] * dsdx + dt[c] * dtdx for c in range(nchannels)),
                tuple(ds[c] * dsdy + dt[c] * dtdy for c in range(nchannels)))

    if not linear:
        return level_sample(int(math.floor(lod + 0.5)))
    low = int(math.floor(lod))
    a, b = level_sample(low), level_sample(min(low + 1, len(levels) - 1))
    fraction = lod - low
    # The footprint/LOD stays fixed for the OSL output derivative chain rule.
    return tuple(tuple(x + fraction * (y - x) for x, y in zip(lhs, rhs))
                 for lhs, rhs in zip(a, b))


def texture_probe(column, row):
    band, probe = divmod(column, 8)
    coordinates = [(-0.125, 0.375), (0.3125, -0.25), (0, 0),
                   (0.0625, 0.125), (0.5, 0.5), (0.9375, 0.875),
                   (1, 1), (1.125, 1.25)]
    s, t = (0.3125, 0.375) if band == 8 else coordinates[probe]
    gradients = (1 / 64, 0, 0, 1 / 16)
    if band in (2, 8):
        gradients = (0, 0, 0, 0)
    elif band in (3, 7):
        gradients = (0.25, 0, 0, 0.25)
    elif band == 4:
        gradients = (0.375, 0, 0, 0.5)
    elif band == 5:
        gradients = (2, 0, 0, 2)
    elif band == 6:
        gradients = (0.125, 0.375, -0.125, 0.25)
    wraps = [("black", "black"), ("clamp", "clamp"),
             ("periodic", "periodic"), ("clamp", "periodic")][(row % 8) // 2]
    return s, t, gradients, bool(row % 2), wraps, row >= 8, probe % 3


def texture_reference(levels, report):
    result = []
    for row in range(17):
        for column in range(65):
            s, t, gradients, linear, wraps, scalar, component = texture_probe(column, row)
            sample = texture_sample(levels, s, t, gradients, linear, wraps)
            if scalar:
                sample = tuple((values[0],) * 3 for values in sample)
            result.extend(tuple(values[component] for values in sample)
                          if report else sample[0])
    return result


def check_texture_oracle(images):
    rgb = images[0]
    assert [(len(level[0]), len(level)) for level in rgb] == [(8, 4), (4, 2), (2, 1), (1, 1)]
    compare(rgb[-1][0][0], (0.5625, 0.625, 0.5), 0)
    sample = texture_sample(rgb, 0.375, 0.5, (1 / 64, 0, 0, 1 / 16),
                            True, ("clamp", "clamp"))
    compare(sample[0], (0.4375, 0.625, 0.5), 0)
    compare(sample[1], (1 / 64, 0, 0), 0)
    compare(sample[2], (0, 1 / 16, 0), 0)
    gradients = (0.125, 0.375, -0.125, 0.25)
    for wraps in (("black", "black"), ("clamp", "clamp"),
                  ("periodic", "periodic"), ("clamp", "periodic")):
        for s, t in ((0.37, 0.41), (-0.02, 0.37), (1.01, -0.04)):
            sampled = texture_sample(rgb, s, t, gradients, True, wraps)
            for axis in (0, 1):
                ds, dt = gradients[2 * axis:2 * axis + 2]
                h = 1 / 65536
                plus, minus = [texture_sample(
                    rgb, s + sign * h * ds, t + sign * h * dt,
                    gradients, True, wraps)[0] for sign in (1, -1)]
                compare(sampled[axis + 1], [(p - m) / (2 * h) for p, m in zip(plus, minus)])


def texture_arguments(optimize, image=0, report=0, connected=False):
    producer = (["--shader", "hart_texture_uv", "producer"] if connected else [])
    consumer = ["--param", "image", str(image), "--param", "report", str(report),
                "--param", "connected", str(int(connected))]
    if connected:
        consumer += ["--shader", "hart_texture_samples", "consumer",
                     "--connect", "producer", "Cout", "consumer", "value"]
    else:
        consumer += ["hart_texture_samples"]
    return ["--llvm_opt", optimize] + producer + consumer


def check_texture_cpu(shader_args, expected, image, report):
    actual = noise_cpu_image(shader_args, 65, 17)
    # OIIO's anisotropic minification is deliberately not the HART LOD rule.
    # Also exclude color reads of mono files: OIIO may promote gray to RGB,
    # whereas the HART resource contract zero-fills absent channels.
    # OIIO closest has different tie-breaking and does not reliably supply
    # zero derivatives. Only compare its values at unambiguous texel centers.
    indices = [3 * (row * 65 + col) + channel
               for row in range(17) for col in range(65) for channel in range(3)
               if col // 8 in (0, 1, 2, 8) and (image != 1 or row >= 8)
               and (row % 2 or ((col % 8 in (3, 5) or col == 64)
                               and (not report or channel == 0)))]
    compare([actual[i] for i in indices], [expected[i] for i in indices], 2e-6)


def check_texture_render(shader_args, width, height, expected, repeat=False,
                         cache_hit=False, compare_device=compare,
                         callable_mode=None):
    image = root / "texture-gpu.pfm"
    if image.exists():
        image.unlink()
    flags = (["--warmup", "--iters", "3"] if repeat else ["--hart-no-cache"])
    if callable_mode == "fused":
        flags += ["--hart-fused"]
    output = run(["--hart", "-v"] + flags + ["-g", str(width), str(height),
                 "-o", "Cout", str(image)] + shader_args)
    if callable_mode is not None:
        assert "HART callable mode: " + callable_mode in output, output
    assert output.count("Launching HART grid") == (4 if repeat else 1), output
    if not repeat:
        assert "HART pipeline cache disabled" in output, output
    if cache_hit:
        assert "cache hit for key" in output, output
    actual = image_pixels(image, width, height)
    compare_device(actual, expected)
    return actual


def texture_connected_reference(optimize, levels):
    # CPU-produced noise coordinates/gradients feed the independent sampler,
    # not a CPU minification oracle. This transform is entirely magnifying.
    coordinates = [noise_cpu_image(["--llvm_opt", optimize, "--param", "report",
                                    str(report), "hart_texture_uv"], 17, 9)
                   for report in (1, 2)]
    samples = []
    for i in range(17 * 9):
        s, dsdx, dsdy = coordinates[0][3 * i:3 * i + 3]
        t, dtdx, dtdy = coordinates[1][3 * i:3 * i + 3]
        assert max(math.hypot(8 * dsdx, 4 * dtdx),
                   math.hypot(8 * dsdy, 4 * dtdy)) < 1
        samples.append(texture_sample(levels, s, t, (dsdx, dtdx, dsdy, dtdy),
                                      True, ("clamp", "clamp")))
    images = []
    for report in (0, 1):
        expected = []
        for i, sample in enumerate(samples):
            component = int(8 * ((i % 17) / 16)) % 3
            expected.extend(tuple(values[component] for values in sample)
                            if report else sample[0])
        images.append(expected)
    return images


def check_texture_suite():
    images = prepare_texture_images()
    check_texture_oracle(images)
    for optimize in ("10", "3"):
        print("Checking HART textures at LLVM level " + optimize, flush=True)
        cases = [(0, (0, 1)), (1, (0, 1))]
        if optimize == "3":
            cases.append((2, (1,)))
        for image, reports in cases:
            for report in reports:
                shader_args = texture_arguments(optimize, image, report)
                expected = texture_reference(images[image], report)
                check_texture_cpu(shader_args, expected, image, report)
                actual = check_texture_render(shader_args, 65, 17, expected)
                if image == 1 and not report:
                    compare([actual[3 * i + c] for i in range(65 * 8)
                             for c in (1, 2)], [0] * (65 * 8 * 2), 0)
                if report:
                    for row in range(17):
                        for col in range(65):
                            _, _, _, linear, wraps, _, _ = texture_probe(col, row)
                            if (not linear or col // 8 in (2, 8)
                                    or (image == 2 and "black" not in wraps)):
                                i = 3 * (row * 65 + col)
                                compare(actual[i + 1:i + 3], (0, 0), 0)

        for report, expected in enumerate(texture_connected_reference(optimize, images[0])):
            shader_args = texture_arguments(optimize, report=report, connected=True)
            compare(noise_cpu_image(shader_args, 17, 9), expected, 2e-6)
            check_texture_render(shader_args, 17, 9, expected,
                                 repeat=optimize == "3" and report == 1)

    # All three handles coexist in one group. Change only file contents, not
    # shader code or filenames, and reuse the cached pipeline across processes.
    for index, scale in enumerate((1, 0.5, 1)):
        rebound = prepare_texture_images(scale)
        references = [texture_reference(levels, 1) for levels in rebound]
        expected = []
        for row in range(17):
            reference = references[int(2 * row / 16 + 0.5)]
            expected.extend(reference[3 * row * 65:3 * (row + 1) * 65])
        check_texture_render(texture_arguments("3", image=-1, report=1),
                             65, 17, expected, repeat=True, cache_hit=index > 0)

    run(["--hart", "-v", "--hart-no-cache", "-g", "65", "17",
         "--param:type=float", "coordinate_scale", "3.0e38"]
        + texture_arguments("3"), "nonfinite coordinates/gradients",
        error_after_launch=True)

    # The blur call is both untaken and in an unused layer. A varying filename
    # must not be bound to its parameter's initial value.
    errors = (
        "HART: texture requires explicit closest or linear interpolation",
        "HART: texture requires explicit wrap modes",
        "HART: texture requires a literal filename",
        "UDIM patterns are not supported",
        "HART: texture requires a literal filename",
        "HART: unsupported texture option 'blur'",
        "Cannot open HART texture 'hart_texture_missing.pfm'",
    )
    for case, error in enumerate(errors):
        name = "hart_texture_rejected_" + str(case)
        compile_fixture(fixtures / "hart_texture_rejected.osl", name,
                        ("TEXTURE_CASE=" + str(case),))
        shader_args = ([name] if case != 5 else
                       ["--shader", name, "unused", "--shader", "hart_first", "surface"])
        run(["--hart", "-v"] + shader_args, error)


def prepare_texture_alpha_images(alpha_scale=1):
    oiiotool = shutil.which("oiiotool", path=env.get("PATH"))
    assert oiiotool, "Texture alpha checks require oiiotool on PATH"

    def convert(arguments):
        result = subprocess.run(
            [oiiotool, "--no-autopremult"] + arguments, cwd=root, env=env,
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    rgba = [[((x + 1) / 8, 0.1875 + (x + 1) / 32 + (y + 1) / 16,
              0.875 if (x + y) % 2 else 0.125,
              alpha_scale * (0 if (x + y) % 4 == 0 else (1 + x + 2 * y) / 16))
             for x in range(8)] for y in range(4)]
    planes = []
    for channel in range(4):
        path = root / ("hart_texture_alpha_plane_" + str(channel) + ".pfm")
        values = [pixel[channel] for row in reversed(rgba) for pixel in row]
        with path.open("wb") as stream:
            stream.write(b"Pf\n8 4\n-1\n")
            stream.write(struct.pack("<32f", *values))
        planes.append(str(path))
    images = []
    readback = root / "hart_texture_alpha_readback.pfm"
    for channels in range(1, 5):
        path = root / ("hart_texture_alpha_" + str(channels) + ".exr")
        arguments = [planes[0]]
        for plane in planes[1:channels]:
            arguments += [plane, "--chappend"]
        convert(arguments + ["--chnames", ",".join(("R", "G", "B", "A")[:channels]),
                             "-d", "float", "-o", str(path)])
        # Read every channel back after EXR serialization, including nonzero
        # RGB at alpha zero. Channel naming/order must match the sampler oracle.
        for channel in range(channels):
            convert([str(path), "--ch", ",".join([str(channel)] * 3),
                     "-d", "float", "-o", str(readback)])
            expected = [pixel[channel] for row in rgba for pixel in row
                        for _ in range(3)]
            compare(image_pixels(readback, 8, 4), expected, 0)
        images.append(texture_mips([[pixel[:channels] for pixel in row]
                                    for row in rgba]))
    return images


def texture_alpha_reference(levels, report=1, firstchannel=0):
    result = []
    for row in range(17):
        for column in range(65):
            s, t, gradients, linear, wraps, scalar, component = texture_probe(column, row)
            sampled = texture_sample(levels, s, t, gradients, linear, wraps,
                                     4, firstchannel)
            if report == 2:
                result.extend(values[0 if scalar else component] for values in sampled)
            elif report:
                # OSL's alpha is the channel after the requested return type.
                result.extend(values[1 if scalar else 3] for values in sampled)
            else:
                result.extend((sampled[0][0],) * 3 if scalar else sampled[0][:3])
    return result


def check_texture_alpha_oracle(images):
    compare(images[3][-1][0][0], (0.5625, 0.484375, 0.5, 0.3515625), 0)
    gradients = (0.125, 0.375, -0.125, 0.25)
    for levels in images:
        for wraps in (("black", "black"), ("clamp", "clamp"),
                      ("periodic", "periodic"), ("clamp", "periodic")):
            for s, t in ((0.37, 0.41), (-0.02, 0.37), (1.01, -0.04)):
                sample = texture_sample(levels, s, t, gradients, True, wraps, 4)
                rgb = texture_sample(levels, s, t, gradients, True, wraps)
                for actual, expected in zip(rgb, sample):
                    compare(actual, expected[:3], 0)
                for axis in (0, 1):
                    ds, dt = gradients[2 * axis:2 * axis + 2]
                    h = 1 / 65536
                    plus, minus = [texture_sample(
                        levels, s + sign * h * ds, t + sign * h * dt,
                        gradients, True, wraps, 4)[0] for sign in (1, -1)]
                    compare(sample[axis + 1],
                            [(p - m) / (2 * h) for p, m in zip(plus, minus)])
    compare(texture_alpha_reference(images[0]), [0] * (65 * 17 * 3), 0)
    for levels in images[1:3]:
        compare(texture_alpha_reference(levels)[:65 * 8 * 3],
                [0] * (65 * 8 * 3), 0)
    rgba = images[3][0]
    assert rgba[0][0][3] == 0 and min(rgba[0][0][:3]) > 0


def texture_alpha_arguments(optimize, channels=4, report=1, connected=False,
                            specialize="-O2"):
    shader = "hart_texture_alpha_" + ("producer" if connected else str(channels))
    arguments = ["--llvm_opt", optimize, specialize, "--param", "report", str(report)]
    if connected:
        arguments += ["--shader", shader, "producer",
                      "--shader", "hart_deriv_consumer", "consumer",
                      "--connect", "producer", "value", "consumer", "value"]
    else:
        arguments += [shader]
    return arguments


def check_texture_alpha_cpu(shader_args, expected, report):
    actual = noise_cpu_image(shader_args, 65, 17)
    # Use only 3/4-channel inputs here: OIIO's gray-to-RGB promotion changes
    # missing-alpha behavior for 1/2-channel files. Its minification and closest
    # derivatives also differ, so compare mip-0 bilinear or closest-center values.
    indices = [3 * (row * 65 + col) + channel
               for row in range(17) for col in range(65) for channel in range(3)
               if col // 8 in (0, 1, 2, 8)
               and (row % 2 or ((col % 8 in (3, 5) or col == 64)
                               and (not report or channel == 0)))]
    compare([actual[i] for i in indices], [expected[i] for i in indices])


def check_texture_alpha_suite():
    images = prepare_texture_alpha_images()
    check_texture_alpha_oracle(images)
    for channels in range(1, 5):
        compile_fixture(fixtures / "hart_texture_alpha.osl",
                        "hart_texture_alpha_" + str(channels),
                        ("ALPHA_CHANNELS=" + str(channels),))
    compile_fixture(fixtures / "hart_texture_alpha.osl",
                    "hart_texture_alpha_producer", ("ALPHA_PRODUCER=1",))
    references = [texture_alpha_reference(levels) for levels in images]

    # Four channel counts, both return types and all sampler modes share each
    # packed grid. Fourteen GPU processes include representative fused/local use.
    for optimize in ("10", "3"):
        print("Checking HART texture alpha at LLVM level " + optimize, flush=True)
        for channels, expected in enumerate(references, 1):
            shader_args = texture_alpha_arguments(
                optimize, channels,
                specialize="-O0" if optimize == "10" and channels == 4 else "-O2",
            )
            if channels >= 3:
                check_texture_alpha_cpu(shader_args, expected, 1)
            actual = check_texture_render(shader_args, 65, 17, expected)
            if channels < 4:
                count = 65 * (17 if channels == 1 else 8) * 3
                compare(actual[:count], [0] * count, 0)
            for row in range(17):
                for col in range(65):
                    if row % 2 == 0 or col // 8 in (2, 8):
                        offset = 3 * (row * 65 + col)
                        compare(actual[offset + 1:offset + 3], (0, 0), 0)
        shader_args = texture_alpha_arguments(optimize, connected=True)
        check_texture_alpha_cpu(shader_args, references[3], 1)
        check_texture_render(shader_args, 65, 17, references[3])

    shader_args = texture_alpha_arguments("3", report=0)
    expected = texture_alpha_reference(images[3], report=0)
    check_texture_alpha_cpu(shader_args, expected, 0)
    actual = check_texture_render(shader_args, 65, 17, expected)
    compare(actual[9:12], images[3][0][0][0][:3], 0)
    for local in (False, True):
        shader_args = texture_alpha_arguments("3")
        if local:
            shader_args += ["--hart-local-groupdata", "2147483647"]
        check_texture_render(shader_args, 65, 17, references[3],
                             repeat=local, callable_mode="fused")
    shader_args = texture_alpha_arguments("3", connected=True)
    shader_args += ["--hart-local-groupdata", "2147483647"]
    check_texture_render(shader_args, 65, 17, references[3],
                         repeat=True, callable_mode="fused")


def check_texture_channel_oracle(images):
    gradients = (0.125, 0.375, -0.125, 0.25)
    for levels in images:
        for wraps in (("black", "black"), ("clamp", "clamp"),
                      ("periodic", "periodic"), ("clamp", "periodic")):
            for s, t in ((0.37, 0.41), (-0.02, 0.37), (1.01, -0.04)):
                base = texture_sample(levels, s, t, gradients, True, wraps, 4)
                for firstchannel in (0, 1, 2, 3, 4, 2147483647):
                    shifted = texture_sample(levels, s, t, gradients, True, wraps,
                                              4, firstchannel)
                    for actual, values in zip(shifted, base):
                        compare(actual, (values[firstchannel:] + (0,) * 4)[:4], 0)


def check_texture_channel_suite():
    images = prepare_texture_alpha_images()
    check_texture_alpha_oracle(images)
    check_texture_channel_oracle(images)
    # Two derivative-packed reports cover return channels and relative alpha.
    # Smaller images exercise last-present/first-absent boundaries separately.
    cases = [(4, firstchannel, report, False)
             for firstchannel in (0, 1, 2, 3, 4, 2147483647) for report in (2, 1)]
    for channels in (1, 2, 3):
        cases += [(channels, channels - 1, 2, False), (channels, channels, 1, False)]
    cases += [(4, 3, 2, True)]
    compiled = set()
    print("Checking HART literal texture channel offsets", flush=True)
    for channels, firstchannel, report, reset in cases:
        name = "hart_texture_alpha_channels_{}_{}{}".format(
            channels, firstchannel, "_reset" if reset else "",
        )
        if name not in compiled:
            defines = ["ALPHA_CHANNELS=" + str(channels),
                       "FIRSTCHANNEL=" + str(firstchannel)]
            if reset:
                defines += ["FIRSTCHANNEL_RESET=1"]
            compile_fixture(fixtures / "hart_texture_alpha.osl", name, defines)
            compiled.add(name)
        optimize = "10" if firstchannel in (1, 2147483647) and not reset else "3"
        specialize = ("-O0" if (channels, firstchannel, report, reset)
                      == (4, 1, 2, False) else "-O2")
        shader_args = ["--llvm_opt", optimize, specialize,
                       "--param", "report", str(report), name]
        effective = 0 if reset else firstchannel
        expected = texture_alpha_reference(images[channels - 1], report, effective)
        # Do not feed INT_MAX to OIIO; the independent oracle checks that case.
        if channels >= 3 and effective < channels:
            check_texture_alpha_cpu(shader_args, expected, report)
        if reset and report == 2:
            shader_args += ["--hart-local-groupdata", "2147483647"]
        actual = check_texture_render(
            shader_args, 65, 17, expected, repeat=reset and report == 2,
            callable_mode=("fused" if reset or (channels, firstchannel, report)
                           == (4, 2, 1) else None),
        )
        for row in range(17):
            for col in range(65):
                _, _, _, linear, _, scalar, component = texture_probe(col, row)
                selected = ((1 if scalar else 3) if report == 1
                            else (0 if scalar else component))
                offset = 3 * (row * 65 + col)
                if effective >= channels or selected >= channels - effective:
                    compare(actual[offset:offset + 3], (0, 0, 0), 0)
                elif not linear or col // 8 in (2, 8):
                    compare(actual[offset + 1:offset + 3], (0, 0), 0)

    for case in range(4):
        name = "hart_texture_alpha_rejected_" + str(case)
        compile_fixture(fixtures / "hart_texture_alpha_rejected.osl", name,
                        ("FIRSTCHANNEL_CASE=" + str(case),))
        shader_args = ([name] if case >= 2 else
                       (["--param", "enabled", "0", name] if case == 0 else
                        ["--shader", name, "unused",
                         "--shader", "hart_first", "surface"]))
        run(["--hart", "-v"] + shader_args, "firstchannel")


def matrix_identity(scale=1):
    return [[scale if i == j else 0 for j in range(4)] for i in range(4)]


def matrix_product(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4))
             for j in range(4)] for i in range(4)]


def matrix_inverse(matrix):
    rows = [list(row) + identity
            for row, identity in zip(matrix, matrix_identity())]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda i: abs(rows[i][col]))
        if rows[pivot][col] == 0:
            # Imath's non-throwing singular inverse returns identity.
            return matrix_identity()
        rows[col], rows[pivot] = rows[pivot], rows[col]
        factor = rows[col][col]
        rows[col] = [value / factor for value in rows[col]]
        for i in range(4):
            if i != col:
                factor = rows[i][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[col])]
    return [row[4:] for row in rows]


def matrix_determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1)**j * matrix[0][j] * matrix_determinant(
        [row[:j] + row[j + 1:] for row in matrix[1:]]) for j in range(len(matrix)))


def matrix_case(operation, u, v, specialize):
    scale = matrix_identity()
    scale[0][0], scale[1][1], scale[2][2] = 2, 0.5, 4
    translation = matrix_identity()
    translation[3][:3] = [0.5, -0.25, 0.75]
    shear = matrix_identity()
    shear[1][0], shear[2][1] = 0.5, 0.25
    if operation == 0:
        return matrix_identity()
    if operation == 1:
        return matrix_identity(1 + u)
    if operation == 2:
        return translation
    if operation in (3, 8, 10):
        return scale
    if operation == 4:
        return shear
    if operation == 5:
        return matrix_product(scale, translation)
    if operation == 6:
        return matrix_product(translation, scale)
    if operation == 7:
        return matrix_inverse(matrix_product(scale, translation))
    if operation == 9:
        return [list(row) for row in zip(*shear)]
    if operation == 11:
        result = matrix_identity()
        result[0][0], result[1][1] = 1 + u, 2 + v
        result[0][1], result[3][0] = 0.25, u - v
        return result
    if operation in (12, 13):
        result = matrix_identity()
        result[0][3], result[1][3], result[3][3] = (
            (0.5, 0.25, 1) if operation == 12 else (1, 0, 0))
        return result
    if operation == 14:
        return matrix_identity(0)
    if operation == 15:
        return matrix_identity()
    assert operation == 16
    # Preserve CPU behavior, including OSL O2's constant-zero division fold.
    if specialize == "-O2":
        return matrix_identity(0)
    return [[math.inf if i == j else math.nan for j in range(4)] for i in range(4)]


def matrix_reference(report=0, constant_input=0, specialize="-O2", supplied=None):
    result = []
    for row in range(9):
        v, kind = row / 8, row // 3
        for col in range(65):
            u = col / 64
            operation, probe = divmod(col, 4)
            matrix = (supplied if supplied is not None else
                      matrix_case(operation, u, v, specialize))
            if report == 2:
                value = (matrix_determinant(matrix) if kind == 0 else
                         matrix[3][0] if kind == 1 else matrix[1][1])
                result.extend((value, 0, 0))
                continue
            x, y = 16 * u - operation - 0.5, 2 * v - 1
            q = [x, y, 1 + 0.5 * x * y]
            dx, dy = [0.25, 0, 0.125 * y], [0, 0.25, 0.125 * x]
            if constant_input:
                q, dx, dy = [0.25, -0.5, 1], [0, 0, 0], [0, 0, 0]
            if operation == 16:
                values = [matrix[0][0], matrix[0][1], matrix[3][3]]
                derivatives = [[0, 0, 0], [0, 0, 0]]
            else:
                if kind == 2:
                    matrix = [list(row) for row in zip(*matrix_inverse(matrix))]
                # OSL uses row vectors. Matrix elements never contribute
                # derivatives, even when their values vary across the grid.
                values = [sum(q[i] * matrix[i][j] for i in range(3))
                          + (matrix[3][j] if kind == 0 else 0) for j in range(3)]
                derivatives = [[sum(d[i] * matrix[i][j] for i in range(3))
                                for j in range(3)] for d in (dx, dy)]
                if kind == 0:
                    w = sum(q[i] * matrix[i][3] for i in range(3)) + matrix[3][3]
                    if w == 0:
                        values, derivatives = [0, 0, 0], [[0, 0, 0], [0, 0, 0]]
                    else:
                        for axis, d in enumerate((dx, dy)):
                            dw = sum(d[i] * matrix[i][3] for i in range(3))
                            derivatives[axis] = [
                                (gradient * w - value * dw) / (w * w)
                                for gradient, value in zip(derivatives[axis], values)]
                        values = [value / w for value in values]
            component = probe % 3
            result.extend((values[component], derivatives[0][component],
                           derivatives[1][component]) if report else values)
    return result


def compare_matrices(actual, expected):
    assert len(actual) == len(expected)
    finite = []
    for i, (a, b) in enumerate(zip(actual, expected)):
        if math.isnan(b):
            assert math.isnan(a), (i, a, b)
        elif math.isinf(b):
            assert a == b, (i, a, b)
        else:
            finite.append(i)
    compare([actual[i] for i in finite], [expected[i] for i in finite])


def matrix_arguments(optimize, report=0, constant_input=0, specialize="-O2",
                     supplied=None):
    parameters = ["--param", "report", str(report),
                  "--param", "constant_input", str(constant_input)]
    if supplied is None:
        shader_args = connected_group("hart_matrix_consumer", parameters,
                                      producer="hart_matrix_producer")
    else:
        values = ",".join(str(value) for row in supplied for value in row)
        shader_args = parameters + ["--param:type=matrix", "value", values,
                                    "hart_matrix_consumer"]
    return ["--llvm_opt", optimize, specialize] + shader_args


def check_matrix_suite():
    supplied = [[1.5, 0.25, 0, 0], [0, 0.5, 0, 0],
                [0, 0, 2, 0], [0.25, -0.5, 0.75, 1]]
    cases = [dict(report=report) for report in (0, 1, 2)]
    cases += [dict(report=report, specialize="-O0") for report in (0, 1)]
    cases += [dict(report=1, constant_input=1), dict(report=1, supplied=supplied)]
    for optimize in ("10", "3"):
        print("Checking HART numeric matrices at LLVM level " + optimize,
              flush=True)
        for options in cases:
            shader_args = matrix_arguments(optimize, **options)
            expected = matrix_reference(**options)
            cpu = noise_cpu_image(shader_args, 65, 9)
            compare_matrices(cpu, expected)
            actual = check_texture_render(shader_args, 65, 9, expected,
                                          compare_device=compare_matrices)
            compare_matrices(actual, cpu)
            if options.get("constant_input") or options["report"] == 2:
                for image in (cpu, actual):
                    compare(image[1::3], [0] * (65 * 9), 0)
                    compare(image[2::3], [0] * (65 * 9), 0)


def space_to_common():
    c = math.sqrt(0.5)
    # setup_transformations preserves these translations while rotating the
    # basis: Imath's mutating rotate prepends the rotation to the translation.
    return [
        matrix_identity(),
        [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1]],
        [[c, c, 0, 0], [-c, c, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
    ]


def space_reference(report):
    spaces = space_to_common()
    inverse = [matrix_inverse(matrix) for matrix in spaces]
    result = []
    for row in range(33):
        v, kind = row / 32, row // 11
        operation = (row % 11) // 2
        for col in range(65):
            u = col / 64
            pair = min(int(9 * u), 8)
            from_space, to_space = divmod(pair, 3)
            gain = 0.5 + 0.25 * u - 0.125 * v
            q = [0.25 + u + 0.25 * gain, 2 * v - 0.5 - 0.5 * gain,
                 1 + u * v + 0.125 * gain]
            dx = [1.0625 / 64, -0.125 / 64, (v + 0.03125) / 64]
            dy = [-0.03125 / 32, 2.0625 / 32, (u - 0.015625) / 32]
            numeric = matrix_identity()
            if operation == 4:
                numeric = matrix_identity(1 + gain)
            elif operation == 5:
                numeric = [[1 + gain, 0.25, 0, 0], [0, 2, 0, 0],
                           [0, 0, 0.5, 0], [0.25, -0.125, 0.375, 1]]
            matrix = matrix_product(matrix_product(spaces[from_space], numeric),
                                    inverse[to_space])
            if report == 2:
                inspected = matrix_identity() if operation in (0, 3) else matrix
                result.extend((1, matrix_determinant(inspected), inspected[3][0]))
                continue
            if kind == 2:
                matrix = [list(row) for row in zip(*matrix_inverse(matrix))]
            # All these maps are affine, or homogeneous uniform scalings.
            # As in phase 1, matrix elements themselves have no derivatives.
            w = matrix[3][3] if kind == 0 else 1
            values = [(sum(q[i] * matrix[i][j] for i in range(3))
                       + (matrix[3][j] if kind == 0 else 0)) / w for j in range(3)]
            derivatives = [[sum(d[i] * matrix[i][j] for i in range(3)) / w
                            for j in range(3)] for d in (dx, dy)]
            component = col % 3
            result.extend((values[component], derivatives[0][component],
                           derivatives[1][component]) if report else values)
    return result


def space_arguments(optimize, report=0, specialize="-O2"):
    return (["--llvm_opt", optimize, specialize]
            + connected_group("hart_space_consumer", ["--param", "report", str(report)],
                              producer="hart_space_producer")
            + ["--connect", "producer", "gain", "consumer", "gain"])


def check_space_suite():
    for optimize in ("10", "3"):
        print("Checking HART literal spaces at LLVM level " + optimize, flush=True)
        cases = [(report, "-O2") for report in (0, 1, 2)] + [(1, "-O0")]
        for report, specialize in cases:
            shader_args = space_arguments(optimize, report, specialize)
            expected = space_reference(report)
            cpu = noise_cpu_image(shader_args, 65, 33)
            compare(cpu, expected)
            actual = check_texture_render(shader_args, 65, 33, expected)
            compare(actual, cpu)
            if report == 2:
                compare(actual[0::3], [1] * (65 * 33), 0)

    # This formerly rejected object-space constructor is now a positive case.
    # Keep repeated cached execution on just this small representative.
    expected = reference(17, 9, lambda u, v: (-v, u + 1, 1))
    shader_args = ["--llvm_opt", "3", "hart_surface_space"]
    compare(noise_cpu_image(shader_args, 17, 9), expected)
    check_texture_render(shader_args, 17, 9, expected, repeat=True)

    for case in range(3):
        name = "hart_space_rejected_" + str(case)
        compile_fixture(fixtures / "hart_space_rejected.osl", name,
                        ("SPACE_CASE=" + str(case),))
        evaluate = (lambda u, v: (u, 2*v, 1)) if case == 0 else (
            (lambda u, v: (u, v, 1)) if case == 1 else
            (lambda u, v: (-v, u+1, 1)))
        expected = reference(3, 2, evaluate)
        for specialize in ("-O0", "-O2"):
            shader_args = [specialize, "--llvm_opt", "3",
                           "--param", "enabled", "1", name]
            compare(noise_cpu_image(shader_args, 3, 2), expected)
            check_texture_render(shader_args, 3, 2, expected)


def geometry_globals_reference():
    def evaluate(u, v):
        field = min(int(12 * v), 11)
        du = dv = 1 / 16
        if field < 6:
            return (0, 0, 0)
        if field == 6:
            return (u, v, u * v)
        if field == 7:
            return (du, 0, v * du)
        if field == 8:
            return (0, dv, u * dv)
        if field == 9:
            return (du, dv, math.hypot(v * du, u * dv))
        if field == 10:
            return (u + v, du, dv)
        return (u * v, v * du, u * dv)
    return reference(17, 17, evaluate)


def geometry_material_reference(optimize, levels):
    coordinates = [noise_cpu_image(["--llvm_opt", optimize, "--param", "report",
                                    str(report), "hart_texture_uv"], 17, 9)
                   for report in (1, 2)]
    spaces = space_to_common()
    matrix = matrix_product(spaces[2], matrix_inverse(spaces[1]))
    images = [[], []]
    for i in range(17 * 9):
        s, dsdx, dsdy = coordinates[0][3 * i:3 * i + 3]
        t, dtdx, dtdy = coordinates[1][3 * i:3 * i + 3]
        # I, dIdx, dIdy and time contribute zero under CPU grid defaults.
        p = (s, t, 0.5)
        q = [sum(p[k] * matrix[k][j] for k in range(3)) + matrix[3][j]
             for j in range(2)]
        derivatives = [[0.375 * sum(d[k] * matrix[k][j] for k in range(3))
                        for j in range(2)]
                       for d in ((dsdx, dtdx, 0), (dsdy, dtdy, 0))]
        gradients = (*derivatives[0], *derivatives[1])
        assert max(math.hypot(8 * gradients[0], 4 * gradients[1]),
                   math.hypot(8 * gradients[2], 4 * gradients[3])) < 1
        sampled = texture_sample(levels, 0.875 + 0.375 * q[0],
                                 0.875 + 0.375 * q[1], gradients,
                                 True, ("clamp", "clamp"))
        component = int(8 * ((i % 17) / 16)) % 3
        images[0].extend(sampled[0])
        images[1].extend(values[component] for values in sampled)
    return images


def geometry_material_arguments(optimize, connected, report, specialize="-O2"):
    producer = ["--shader", "hart_texture_uv", "producer"] if connected else []
    consumer = ["--param", "connected", str(int(connected)),
                "--param", "report", str(report)]
    if connected:
        consumer += ["--shader", "hart_geometry_material", "consumer",
                     "--connect", "producer", "Cout", "consumer", "value"]
    else:
        consumer += ["hart_geometry_material"]
    return ["--llvm_opt", optimize, specialize] + producer + consumer


def check_geometry_suite():
    levels = prepare_texture_images()[0]
    for optimize in ("10", "3"):
        print("Checking HART geometry/materials at LLVM level " + optimize,
              flush=True)
        shader_args = ["--llvm_opt", optimize, "hart_geometry_globals"]
        expected = geometry_globals_reference()
        cpu = noise_cpu_image(shader_args, 17, 17)
        compare(cpu, expected)
        actual = check_texture_render(shader_args, 17, 17, expected)
        for image in (cpu, actual):
            # The first eight rows read only I/time, their derivatives or widths.
            compare(image[:8 * 17 * 3], [0] * (8 * 17 * 3), 0)

        expected = geometry_material_reference(optimize, levels)
        cases = [(False, 1, "-O0" if optimize == "10" else "-O2"),
                 (True, 1, "-O2"), (optimize == "3", 0, "-O2")]
        packed = []
        for connected, report, specialize in cases:
            shader_args = geometry_material_arguments(optimize, connected, report,
                                                      specialize)
            cpu = noise_cpu_image(shader_args, 17, 9)
            compare(cpu, expected[report])
            actual = check_texture_render(shader_args, 17, 9, expected[report])
            compare(actual, cpu)
            if report:
                assert max(abs(d) for d in actual[1::3]) > 1e-4
                assert max(abs(d) for d in actual[2::3]) > 1e-4
                packed.append(actual)
        compare(packed[0], packed[1])

    for case, global_name in enumerate(("I", "time")):
        name = "hart_geometry_rejected_" + str(case)
        compile_fixture(fixtures / "hart_geometry_rejected.osl", name,
                        ("GEOMETRY_WRITE_TIME=" + str(case),))
        if global_name == "I":
            check_render(["--param", "enabled", "1", name], 3, 2,
                         reference(3, 2, lambda u, v: (u, .5, 0)))
        else:
            run(["--hart", "-v", "--shader", name, "unused",
                 "--shader", "hart_first", "surface"],
                "HART: writing shader global '" + global_name + "'")


def group_arguments(depth, optimize, specialize="-O2"):
    names = ["chain" + str(i) for i in range(depth - 1)] + ["probe"]
    arguments = ["--llvm_opt", optimize, specialize]
    for name in names[:-1]:
        arguments += ["--shader", "hart_group_chain", name]
    arguments += ["--shader", "hart_group_probe", names[-1]]
    for source, destination in zip(names, names[1:]):
        for port in ("scalar", "tint", "position", "direction",
                     "orientation", "basis"):
            arguments += ["--connect", source, "next_" + port, destination, port]
    return arguments


def group_reference(depth, width, height):
    # Each of the depth-1 live chain layers applies x -> x/2 + seed.
    # Close the geometric series rather than executing the shader recurrence.
    gain = 1 - 0.5**(depth - 1)
    du, dv = 1 / max(1, width - 1), 1 / max(1, height - 1)

    def evaluate(u, v):
        values = (gain * (7 * u + 6.5 * v),
                  gain * (u + 10 * v),
                  2 + gain * (9.5 * u + v + u * v))
        # The matrix's varying diagonal/translation terms have zero derivatives.
        dx = (6.5 * gain * du, 0.5 * gain * du, gain * (8.5 + v) * du)
        dy = (7 * gain * dv, 9.5 * gain * dv, gain * u * dv)
        component = int(u >= 0.25) + int(u >= 0.75)
        return tuple(0.0625 * value[component] for value in (values, dx, dy))
    return reference(width, height, evaluate)


def check_group_suite():
    for optimize in ("10", "3"):
        print("Checking HART multilayer chains at LLVM level " + optimize,
              flush=True)
        for depth in (3, 5, 9):
            shader_args = group_arguments(depth, optimize)
            expected = group_reference(depth, 7, 5)
            cpu = noise_cpu_image(shader_args, 7, 5)
            compare(cpu, expected)
            actual = check_texture_render(shader_args, 7, 5, expected)
            compare(actual, cpu)
    # One singleton OSL O0 case covers both repeated cold and cached launches.
    # Keep values below one so the printed representative fits image tolerance.
    check_render(group_arguments(9, "10", "-O0"), 1, 1,
                 group_reference(9, 1, 1))


def topology_arguments(optimize, mode=-1, reuse_all=1, textures=False,
                       specialize="-O2"):
    arguments = ["--llvm_opt", optimize, specialize]
    if not textures:
        arguments += ["--shader", "hart_group_producer", "root"]
    for side, name in enumerate(("left", "right")):
        if side == 1:
            arguments += ["--shader", "hart_first", "unused"]
        arguments += ["--param", "side", str(side)]
        if textures:
            arguments += ["--param", "mode", str(mode)]
        arguments += ["--shader", "hart_topology_texture" if textures else
                       "hart_topology_branch", name]
    arguments += ["--param", "mode", str(mode),
                  "--param", "reuse_all", str(reuse_all),
                  "--shader", "hart_topology_join", "join"]
    for name in ("left", "right"):
        arguments += ["--connect", name, "value", "join", name]
        if not textures:
            arguments += ["--connect", "root", "value", name, "root"]
    if not textures:
        arguments += ["--connect", "root", "value", "join", "root"]
    return arguments


def topology_reference(width, height, mode=-1, reuse_all=1, textures=False):
    du, dv = 1 / max(1, width - 1), 1 / max(1, height - 1)

    def evaluate(u, v):
        selected_mode = min(int(4 * v), 3) if mode < 0 else mode
        take_left = (u > v, True, False, u == v)[selected_mode]
        if textures:
            # The red plane is a linear ramp: mip-0 samples equal s + 1/16.
            root = (0, 0, 0)
            left = (0.1875 + 0.5 * u, 0.5 * du, 0)
            right = (0.3125 + 0.5 * v, 0, 0.5 * dv)
        else:
            root = (u + v, du, dv)
            left = (0.5 * (u + v) + u * u, (0.5 + 2 * u) * du, 0.5 * dv)
            right = (-0.25 * (u + v) + v * v, -0.25 * du, (-0.25 + 2 * v) * dv)
        selected = left if take_left else right
        return tuple(1.5 * s + 0.25 * r + (0.125 * (a + b) if reuse_all else 0)
                     for s, r, a, b in zip(selected, root, left, right))
    return reference(width, height, evaluate)


def check_topology_suite():
    prepare_texture_images()
    for optimize in ("10", "3"):
        print("Checking HART dependency topology at LLVM level " + optimize,
              flush=True)
        # Mixed, all-left, all-right and equality regions share one diamond.
        # The unused middle node exercises remapped live-layer flag indexes.
        cases = [(9, 9, -1, 1, False, "-O2"),
                 (1, 1, 1 if optimize == "10" else 2, 0, False, "-O0"),
                 (9, 9, 0, 0, True, "-O2"), (9, 9, 3, 0, True, "-O2")]
        for width, height, mode, reuse_all, textures, specialize in cases:
            shader_args = topology_arguments(optimize, mode, reuse_all,
                                             textures, specialize)
            expected = topology_reference(width, height, mode, reuse_all, textures)
            cpu = noise_cpu_image(shader_args, width, height)
            compare(cpu, expected)
            actual = check_texture_render(
                shader_args, width, height, expected,
                repeat=width == 1 and optimize == "3",
            )
            compare(actual, cpu)
        # Forcing the other input after the join must execute its invalid lookup.
        run(["--hart", "--hart-no-cache", "-v", "-g", "9", "9"]
            + topology_arguments(optimize, mode=0, reuse_all=1, textures=True),
            "nonfinite coordinates/gradients", error_after_launch=True)


def material_arguments(optimize, report=0, strength=0.75, specialize="-O2",
                       noise_report=False, channels=False):
    arguments = [
        "--llvm_opt", optimize, specialize,
        "--shader", "hart_material_coords", "coords",
        "--param", "transform_only", "1",
        "--shader", "hart_material_coords", "space",
        "--param", "noise_report", str(int(noise_report)),
        "--shader", "hart_material_uv", "distort",
    ]
    connections = [("coords", "value", "space", "position"),
                   ("space", "value", "distort", "position")]
    if not noise_report:
        arguments += [
            "--shader", "hart_material_channels" if channels else
            "hart_material_texture", "texture",
            "--shader", "hart_material_mask", "mask",
            "--param:type=float", "strength", str(strength),
            "--param", "report", str(report),
            "--shader", "hart_material_mix", "surface",
        ]
        connections += [
            ("distort", "value", "texture", "position"),
            ("space", "value", "mask", "position"),
            ("coords", "value", "surface", "position"),
            ("texture", "value", "surface", "texture_value"),
            ("mask", "value", "surface", "mask_value"),
        ]
        if channels:
            connections += [("texture", "alpha", "surface", "texture_alpha")]
    for connection in connections:
        arguments += ["--connect", *connection]
    return arguments


def material_noise_signal(optimize):
    image = root / "material-noise.pfm"
    if image.exists():
        image.unlink()
    run(["-t", "1", "-g", "17", "9", "-o", "value", str(image)]
        + material_arguments(optimize, noise_report=True))
    return image_pixels(image, 17, 9)


def material_reference(levels, noise, strength, channels=False):
    spaces = space_to_common()
    matrix = matrix_product(spaces[2], matrix_inverse(spaces[1]))
    images = [[], [], []]
    alpha_weight_effect = [0, 0]
    for i in range(17 * 9):
        u, v = (i % 17) / 16, (i // 17) / 8
        coordinates = (0.25 + 0.375 * u, 0.125 + 0.5 * v, 0.5)
        position = [sum(coordinates[k] * matrix[k][j] for k in range(3))
                    + matrix[3][j] for j in range(3)]
        derivatives = [[sum(d[k] * matrix[k][j] for k in range(3))
                        for j in range(3)]
                       for d in ((0.375 / 16, 0, 0), (0, 0.5 / 8, 0))]
        n, ndx, ndy = noise[3 * i:3 * i + 3]
        s = 0.875 + 0.375 * position[0] + 0.03125 * n
        t = 0.875 + 0.375 * position[1] - 0.015625 * n
        gradients = tuple(value for d, dn in zip(derivatives, (ndx, ndy))
                          for value in (0.375 * d[0] + 0.03125 * dn,
                                        0.375 * d[1] - 0.015625 * dn))
        assert max(math.hypot(8 * gradients[0], 4 * gradients[1]),
                   math.hypot(8 * gradients[2], 4 * gradients[3])) < 1
        texture = texture_sample(levels, s, t, gradients, True, ("clamp", "clamp"),
                                 4 if channels else 3)
        if channels:
            alpha = tuple(values[3] for values in texture)
            # A color lookup starts at G, while a scalar lookup adds 1/8 B.
            texture = tuple(tuple(values[c] + 0.125 * values[2] for c in (1, 2, 3))
                            for values in texture)
        signal = 1 + 0.5 * position[0] + 0.25 * position[1]
        h = min(1, max(0, (signal - 0.375) / 0.25))
        weight = strength * h * h * (3 - 2 * h)
        dweight = [strength * 24 * h * (1 - h) * (0.5 * d[0] + 0.25 * d[1])
                   for d in derivatives]
        if channels:
            alpha_dweight = [weight * da for da in alpha[1:]]
            dweight = [dw * alpha[0] + da
                       for dw, da in zip(dweight, alpha_dweight)]
            weight *= alpha[0]
        background = (0.125 + 0.25 * coordinates[0],
                      0.25 + 0.125 * coordinates[1], 0.5)
        dbackground = ((0.25 * 0.375 / 16, 0, 0), (0, 0.125 * 0.5 / 8, 0))
        value = [(1 - weight) * a + weight * b for a, b in zip(background, texture[0])]
        derivs = [[(1 - weight) * da + weight * db + (b - a) * dw
                   for a, b, da, db in zip(background, texture[0], dbackground[axis],
                                           texture[axis + 1])]
                  for axis, dw in enumerate(dweight)]
        component = int(8 * u) % 3
        images[0].extend(value)
        images[1].extend((value[component], derivs[0][component], derivs[1][component]))
        images[2].extend(values[component] for values in texture)
        if channels:
            for axis, dw in enumerate(alpha_dweight):
                alpha_weight_effect[axis] = max(
                    alpha_weight_effect[axis],
                    abs((texture[0][component] - background[component]) * dw),
                )
    if channels:
        # Dropping the alpha-gradient product-rule term must visibly change the
        # derivative report, not disappear behind a zero mask or equal colors.
        assert min(alpha_weight_effect) > 1e-4, alpha_weight_effect
    return images


def check_material_suite():
    levels = prepare_texture_images()[0]
    for optimize in ("10", "3"):
        print("Checking HART multilayer materials at LLVM level " + optimize,
              flush=True)
        # CPU supplies only the Perlin signal; matrix, sampler, smoothstep and
        # mix derivatives are computed independently, with mip 0 verified above.
        noise = material_noise_signal(optimize)
        references = {strength: material_reference(levels, noise, strength)
                      for strength in (0.75, 0.375)}
        cases = [(0, 0.75, "-O2"), (1, 0.75, "-O2"),
                 (2, 0.75, "-O2"), (1, 0.375, "-O2")]
        if optimize == "10":
            cases.append((1, 0.75, "-O0"))
        packed = {}
        for report, strength, specialize in cases:
            shader_args = material_arguments(optimize, report, strength, specialize)
            expected = references[strength][report]
            cpu = noise_cpu_image(shader_args, 17, 9)
            compare(cpu, expected)
            # Repeat the same baseline artifact with caching enabled. A changed
            # strength can change bitcode, so do not infer a cache hit from it.
            actual = check_texture_render(
                shader_args, 17, 9, expected,
                repeat=optimize == "3" and report == 1 and strength == 0.75,
            )
            compare(actual, cpu)
            if report:
                assert max(abs(d) for d in actual[1::3]) > 1e-4
                assert max(abs(d) for d in actual[2::3]) > 1e-4
                if report == 1 and specialize == "-O2":
                    packed[strength] = actual
        assert max(abs(a - b) for a, b in zip(packed[0.75], packed[0.375])) > 1e-3


def check_texture_material_suite():
    levels = prepare_texture_alpha_images()[3]
    for optimize in ("10", "3"):
        print("Checking HART channel/alpha materials at LLVM level " + optimize,
              flush=True)
        noise = material_noise_signal(optimize)
        references = {strength: material_reference(levels, noise, strength, channels=True)
                      for strength in (0.75, 0.375)}
        cases = [(0, 0.75, "-O2"), (1, 0.75, "-O2"),
                 (2, 0.75, "-O2"), (1, 0.375, "-O2")]
        if optimize == "10":
            cases.append((1, 0.75, "-O0"))
        packed = {}
        for report, strength, specialize in cases:
            shader_args = material_arguments(optimize, report, strength, specialize,
                                              channels=True)
            expected = references[strength][report]
            cpu = noise_cpu_image(shader_args, 17, 9)
            compare(cpu, expected)
            actual = check_texture_render(shader_args, 17, 9, expected,
                                          callable_mode="split")
            compare(actual, cpu)
            if report:
                assert max(abs(d) for d in actual[1::3]) > 1e-4
                assert max(abs(d) for d in actual[2::3]) > 1e-4
                if report == 1 and specialize == "-O2":
                    packed[strength] = actual
        assert max(abs(a - b) for a, b in zip(packed[0.75], packed[0.375])) > 1e-3

    # Thirteen GPU processes: nine split cases, one fused-scratch case and
    # three cached fused-local runs with only the alpha pixels rebound.
    shader_args = material_arguments("3", report=1, channels=True)
    actual = check_texture_render(shader_args, 17, 9, references[0.75][1],
                                  callable_mode="fused")
    compare(actual, packed[0.75])
    rebound = []
    for index, scale in enumerate((1, 0.5, 1)):
        updated = prepare_texture_alpha_images(alpha_scale=scale)[3]
        for original_level, updated_level in zip(levels, updated):
            for original_row, updated_row in zip(original_level, updated_level):
                for original, current in zip(original_row, updated_row):
                    compare(current[:3], original[:3], 0)
                    assert current[3] == scale * original[3]
        expected = material_reference(updated, noise, 0.75, channels=True)[1]
        cpu = noise_cpu_image(shader_args, 17, 9)
        compare(cpu, expected)
        actual = check_texture_render(
            shader_args + ["--hart-local-groupdata", "2147483647"],
            17, 9, expected, repeat=True, cache_hit=index > 0, callable_mode="fused",
        )
        compare(actual, cpu)
        rebound.append(actual)
    assert max(abs(a - b) for a, b in zip(rebound[0], rebound[1])) > 1e-3
    compare(rebound[0], rebound[2])


def check_fused_suite():
    levels = prepare_texture_images()[0]

    def check_modes(shader_args, width, height, expected, repeat=False):
        cpu = noise_cpu_image(shader_args, width, height)
        compare(cpu, expected)
        images = []
        for mode in ("split", "fused"):
            actual = check_texture_render(
                shader_args, width, height, expected, repeat=repeat,
                callable_mode=mode,
            )
            compare(actual, cpu)
            images.append(actual)
        compare(images[0], images[1])

    # Six representative graphs, each run once per mode, plus four lazy-probe
    # processes keep this suite to 16 GPU processes rather than doubling suites.
    for optimize in ("10", "3"):
        print("Checking split/fused HART callables at LLVM level " + optimize,
              flush=True)
        check_modes(
            group_arguments(9, optimize, "-O0" if optimize == "10" else "-O2"),
            7, 5, group_reference(9, 7, 5), repeat=optimize == "3",
        )
        check_modes(topology_arguments(optimize), 9, 9,
                    topology_reference(9, 9))
        noise = material_noise_signal(optimize)
        expected = material_reference(levels, noise, 0.75)[1]
        check_modes(material_arguments(optimize, report=1), 17, 9, expected)

    # Divergent selection samples both safe branches while the unselected
    # producer would overflow. Repeated launches also reset live-layer flags.
    check_modes(
        topology_arguments("3", mode=0, reuse_all=0, textures=True),
        9, 9, topology_reference(9, 9, mode=0, reuse_all=0, textures=True),
        repeat=True,
    )
    for mode in ("split", "fused"):
        flags = ["--hart", "--hart-no-cache", "-v", "-g", "9", "9"]
        if mode == "fused":
            flags += ["--hart-fused"]
        output = run(
            flags + topology_arguments("3", mode=0, reuse_all=1, textures=True),
            "nonfinite coordinates/gradients", error_after_launch=True,
        )
        assert "HART callable mode: " + mode in output, output


def hart_group_storage(output):
    storage = re.findall(
        r"HART group storage: (\d+) bytes, alignment (\d+), "
        r"local (\d+) bytes, scratch (\d+) bytes", output,
    )
    assert len(storage) == 1, output
    size, alignment, local, scratch = map(int, storage[0])
    assert size > 0 and alignment > 0, storage
    assert alignment & (alignment - 1) == 0, storage
    return size, alignment, local, scratch


def check_hart_runstats(output, iterations):
    pipeline = re.findall(r"HART pipeline creation: (\S+) ms", output)
    launches = re.findall(
        r"HART synchronized launches: (\d+) iterations, "
        r"(\S+) ms total, (\S+) ms mean",
        output,
    )
    assert len(pipeline) == 1 and len(launches) == 1, output
    count, total, mean = launches[0]
    assert int(count) == iterations and iterations > 0, output
    pipeline_ms, total_ms, mean_ms = map(float, (pipeline[0], total, mean))
    assert all(math.isfinite(value) and value >= 0
               for value in (pipeline_ms, total_ms, mean_ms)), output
    # Total and mean are independently rounded to six decimal places.
    assert math.isclose(total_ms / iterations, mean_ms,
                        abs_tol=1e-6, rel_tol=0), output
    return pipeline_ms, mean_ms


def hart_stack_estimates(output):
    stacks = re.findall(
        r"^HART stack estimates: raygen ([0-9]+) bytes, "
        r"direct callable max ([0-9]+) bytes\r?$", output, re.MULTILINE,
    )
    assert len(stacks) == 1, output
    raygen, callable_max = map(int, stacks[0])
    # SDK estimates can include conservative floors, not register/spill counts
    # or the logical group-data allocation.
    return {"raygen": raygen, "direct_callable_max": callable_max}


def check_fused_local_suite():
    levels = prepare_texture_images()[0]

    def render(shader_args, width, height, mode="split", budget=None,
               repeat=False, error=None):
        image = root / "fused-local-gpu.pfm"
        if image.exists():
            image.unlink()
        flags = ["--hart", "--hart-no-cache", "--runstats", "-v",
                 "-g", str(width), str(height), "-o", "Cout", str(image)]
        if mode == "fused":
            flags += ["--hart-fused"]
        if budget is not None:
            flags += ["--hart-local-groupdata", str(budget)]
        if repeat:
            flags += ["--warmup", "--iters", "3"]
        output = run(flags + shader_args, error,
                     error_after_launch=error is not None)
        assert "HART callable mode: " + mode in output, output
        assert "HART pipeline cache disabled" in output, output
        assert output.count("Launching HART grid") == (4 if repeat else 1), output
        hart_stack_estimates(output)
        if error is None:
            check_hart_runstats(output, 3 if repeat else 1)
        storage = hart_group_storage(output)
        size, alignment, local, scratch = storage
        expected_local = (size if mode == "fused" and budget is not None
                          and budget >= size else 0)
        assert local == expected_local, (budget, storage)
        stride = ((size + alignment - 1) // alignment) * alignment
        assert scratch == (0 if local else width * height * stride), storage
        if error:
            assert not image.exists(), "Failed render wrote an output image"
            return None, (size, alignment)
        return image_pixels(image, width, height), (size, alignment)

    def check_case(shader_args, width, height, expected, thresholds=False,
                   scratch=True, local_budget=None):
        cpu = noise_cpu_image(shader_args, width, height)
        compare(cpu, expected)
        split, layout = render(shader_args, width, height)
        compare(split, expected)
        compare(split, cpu)
        size, _ = layout
        if thresholds:
            # The omitted option and explicit zero must both preserve scratch.
            budgets = [None, 0, size - 1, size, size + 1]
        else:
            budgets = ([0] if scratch else []) + [
                size if local_budget is None else local_budget
            ]
        for budget in budgets:
            actual, actual_layout = render(
                shader_args, width, height, "fused", budget,
                repeat=budget == size,
            )
            assert actual_layout == layout, (layout, actual_layout)
            compare(actual, expected)
            compare(actual, cpu)
            compare(actual, split)

    # Twenty GPU processes cover the thresholds once, both LLVM modes, and
    # local execution without multiplying every case in the split/fused suite.
    print("Checking HART callable-local group storage", flush=True)
    check_case(group_arguments(9, "10", "-O0"), 7, 5,
               group_reference(9, 7, 5), thresholds=True)
    check_case(group_arguments(9, "3"), 7, 5,
               group_reference(9, 7, 5), scratch=False)
    check_case(topology_arguments("3"), 9, 9, topology_reference(9, 9))
    noise = material_noise_signal("3")
    check_case(material_arguments("3", report=1), 17, 9,
               material_reference(levels, noise, 0.75)[1],
               local_budget=2147483647)
    check_case(
        topology_arguments("3", mode=0, reuse_all=0, textures=True),
        9, 9, topology_reference(9, 9, mode=0, reuse_all=0, textures=True),
    )

    # This graph can have a different optimized layout from the lazy success
    # case, so derive its exact threshold from its own failing split launch.
    shader_args = topology_arguments("3", mode=0, reuse_all=1, textures=True)
    error = "nonfinite coordinates/gradients"
    _, layout = render(shader_args, 9, 9, error=error)
    for budget in (0, layout[0]):
        _, actual_layout = render(shader_args, 9, 9, "fused", budget, error=error)
        assert actual_layout == layout, (layout, actual_layout)


def check_fused_benchmark():
    prepare_texture_images()
    width = height = 256
    iterations, trials = 100, 3
    modes = ("split", "fused-scratch", "fused-local")
    closure_source = root / "benchmark_closures.osl"
    closure_source.write_text("""shader benchmark_closures(output color Cout=0) {
    Ci=0;
    for (int i=0; i<8; ++i) {
        normal n=normalize(normal(u+0.0625*i, v-0.03125*i, 1));
        color weight=color(0.125+0.0625*i, 0.25+0.125*u, 0.125+0.125*v);
        Ci += weight*diffuse(n);
    }
    if (Ci) Cout=color(1, 0.25+u, 0.5+v);
}
""", encoding="ascii")
    cases = (
        ("chain9", "numeric", group_arguments(9, "3"),
         ("hart_group_chain", "hart_group_probe"), 0),
        ("material", "textured", material_arguments("3", report=1),
         ("hart_material_coords", "hart_material_uv", "hart_material_texture",
          "hart_material_mask", "hart_material_mix"), 0),
        ("closures8", "closure-heavy",
         ["-O2", "--llvm_opt", "3", "benchmark_closures"],
         ("benchmark_closures",), 1024),
    )

    def ranges(values):
        return {"median": statistics.median(values),
                "range": [min(values), max(values)], "trials": values}

    sources = {name: (closure_source if name == closure_source.stem
                      else fixtures / (name + ".osl"))
               for _, _, _, names, _ in cases for name in names}
    compile_ms = {name: [] for name in sources}
    for trial in range(trials):
        names = list(sources)
        for name in names[trial:] + names[:trial]:
            compile_ms[name].append(compile_fixture(sources[name], options=("-O1",)))

    image = root / "fused-benchmark.pfm"
    for graph, workload, shader_args, names, closure_pool in cases:
        print("Benchmarking HART " + graph, flush=True)
        samples = {mode: [] for mode in modes}
        storage_by_mode = {}
        stacks_by_mode = {}
        cpu = noise_cpu_image(shader_args, width, height)
        assert all(math.isfinite(value) for value in cpu) and max(cpu) > 0, graph
        if graph == "chain9":
            compare(cpu, group_reference(9, width, height))
        elif graph == "closures8":
            compare(cpu, reference(width, height, lambda u, v: (1, 0.25+u, 0.5+v)))
        baseline = None

        def sample(mode, count, warmup, no_cache):
            nonlocal baseline
            if image.exists():
                image.unlink()
            flags = ["--hart", "--runstats", "-v", "--iters", str(count),
                     "-g", str(width), str(height), "-o", "Cout", str(image)]
            if warmup:
                flags += ["--warmup"]
            if no_cache:
                flags += ["--hart-no-cache"]
            if mode != "split":
                flags += ["--hart-fused", "--hart-local-groupdata",
                          "2147483647" if mode == "fused-local" else "0"]
            wall = []
            output = run(flags + shader_args, wall_times=wall)
            callable_mode = "split" if mode == "split" else "fused"
            assert "HART callable mode: " + callable_mode in output, output
            assert ("HART pipeline cache disabled" in output) == no_cache, output
            assert output.count("Launching HART grid") == count + int(warmup), output
            pools = re.findall(r"HART closure pool: (\d+) bytes per point", output)
            assert pools == [str(closure_pool)], output
            pipeline_ms, launch_mean_ms = check_hart_runstats(output, count)
            preparation = re.findall(
                r"^HART OSL group preparation: (\S+) ms\r?$", output, re.MULTILINE)
            warmup_records = re.findall(
                r"^HART grid warmup: ([01]) launches, (\S+) ms total\r?$",
                output, re.MULTILINE)
            assert len(preparation) == 1 and len(warmup_records) == 1, output
            preparation_ms = float(preparation[0])
            warmup_count, warmup_time = warmup_records[0]
            warmup_count, warmup_ms = int(warmup_count), float(warmup_time)
            assert warmup_count == int(warmup), output
            assert all(math.isfinite(value) and value >= 0
                       for value in (preparation_ms, warmup_ms)), output
            if not warmup:
                assert warmup_ms == 0, output
            stacks = hart_stack_estimates(output)
            if mode in stacks_by_mode:
                assert stacks_by_mode[mode] == stacks, (graph, mode, stacks)
            stacks_by_mode[mode] = stacks
            storage = hart_group_storage(output)
            size, alignment, local, scratch = storage
            assert local == (size if mode == "fused-local" else 0), storage
            stride = ((size + alignment - 1) // alignment) * alignment
            assert scratch == (0 if local else width * height * stride), storage
            if mode in storage_by_mode:
                assert storage_by_mode[mode] == storage, storage
            storage_by_mode[mode] = storage
            assert storage[:2] == storage_by_mode["split"][:2], storage
            actual = image_pixels(image, width, height)
            assert all(math.isfinite(value) for value in actual) and max(actual) > 0
            compare(actual, cpu)
            if baseline is None:
                assert mode == "split"
                baseline = actual
            compare(actual, baseline)
            return {
                "osl_group_preparation_ms": preparation_ms,
                "pipeline_ms": pipeline_ms,
                "warmup_ms": warmup_ms,
                "launch_mean_ms": launch_mean_ms,
                "process_wall_ms": wall[0],
                "iterations": count, "warmup_launches": warmup_count,
                "cache_policy": "disabled" if no_cache else "enabled-default",
                "observed_cache_hit_keys": re.findall(
                    r"cache hit for key[ \t]+(\S+)", output),
            }

        # Cold means bypassing HART's pipeline cache, not flushing OS/driver
        # caches. Prime each mode separately, outside the repeated trials.
        cold = {mode: sample(mode, 1, False, True) for mode in modes}
        priming = {mode: sample(mode, 1, False, False) for mode in modes}
        for trial in range(trials):
            for mode in modes[trial:] + modes[:trial]:
                samples[mode].append(sample(mode, iterations, True, False))

        for mode in modes:
            size, alignment, local, scratch = storage_by_mode[mode]
            print(json.dumps({
                "benchmark": "hart-fused",
                "measurement": "host-synchronized-launch",
                "graph": graph, "workload": workload,
                "mode": mode, "grid": [width, height],
                "llvm_opt": 3, "osl_opt": 2,
                "warmup": 1, "iterations": iterations, "trials": trials,
                "trial_mode_order": [modes[i:] + modes[:i] for i in range(trials)],
                "oslc_process_wall_ms": {
                    "scope": "sum of source compiler subprocesses; includes startup, "
                             "not in-process OSL group compilation",
                    "sources": names, "oslc_opt": 1,
                    **ranges([sum(compile_ms[name][i] for name in names)
                              for i in range(trials)]),
                },
                "cold_no_cache": cold[mode],
                "cache_priming_outside_trials": priming[mode],
                "cache_policy": "enabled-default",
                "observed_cache_hit_keys_by_trial": [
                    record["observed_cache_hit_keys"] for record in samples[mode]],
                "launch_mean_ms": ranges([
                    record["launch_mean_ms"] for record in samples[mode]]),
                "osl_group_preparation_ms": ranges([
                    record["osl_group_preparation_ms"] for record in samples[mode]]),
                "pipeline_ms": ranges([
                    record["pipeline_ms"] for record in samples[mode]]),
                "warmup_ms": ranges([
                    record["warmup_ms"] for record in samples[mode]]),
                "process_wall_ms": ranges([
                    record["process_wall_ms"] for record in samples[mode]]),
                "timing_scope": "OSL group preparation includes optimize_group and "
                                "compiled artifact/metadata extraction, not source "
                                "parsing; pipeline covers full load(): modules, "
                                "program groups, pipeline, deferred stack-query "
                                "compilation and SBT setup; warmup includes its "
                                "launch, clears and error checks, not allocation "
                                "or readback; timed launch is synchronized host "
                                "latency excluding warmup, clears and error checks; "
                                "process includes setup, warmup, launches, image "
                                "IO and teardown",
                "pipeline_timing_comparison": "full-load timings are not directly "
                                              "comparable to the old T10 interval "
                                              "ending before the stack query",
                "cold_scope": "HART pipeline cache disabled; OS/driver caches untouched",
                "group_bytes": size, "alignment": alignment,
                "local_bytes": local, "scratch_bytes": scratch,
                "closure_pool_bytes_per_point": closure_pool,
                "closure_lobes": 8 if workload == "closure-heavy" else 0,
                "stack_estimate_bytes": stacks_by_mode[mode],
                "memory_scope": "logical group/pool storage and SDK stack estimates, "
                                "not physical VRAM, registers or spills",
            }, allow_nan=False), flush=True)


def check_image_render(shaders, flags, mode, width, height, expected,
                       tolerance=None, cpu_value_tolerance=None,
                       value_relative_tolerance=1e-6,
                       cpu_value_relative_tolerance=None, cpu_expected=None):
    assert len(expected) == width * height * 3
    assert cpu_expected is None or len(cpu_expected) == len(expected)
    host, device = root / "exact-cpu.pfm", root / "exact-gpu.pfm"
    for path in (host, device):
        if path.exists():
            path.unlink()
    common = flags + ["-g", str(width), str(height), "-d", "float"]
    run(["-t", "1"] + common + ["-o", "Cout", str(host)] + shaders)
    output = run(["--hart", "--hart-no-cache", "-v", "--warmup",
                  "--iters", "3"] + mode + common
                 + ["-o", "Cout", str(device)] + shaders)
    assert output.count("Launching HART grid") == 4, output
    size, alignment, local, scratch = hart_group_storage(output)
    assert local == (size if "--hart-local-groupdata" in mode else 0)
    assert (scratch == 0) == bool(local)
    # Packed integer results and binary-fraction grids remain exact by default.
    images = []
    for path in (host, device):
        actual = image_pixels(path, width, height)
        images.append(actual)
        target_image = cpu_expected if path == host and cpu_expected is not None else expected
        if tolerance is None:
            for i, (value, target) in enumerate(zip(actual, target_image)):
                assert value == target, (shaders, path.name, i, value, target)
        else:
            for channel in range(3):
                limit = (cpu_value_tolerance if path == host and channel == 0
                         and cpu_value_tolerance is not None else tolerance)
                relative = (cpu_value_relative_tolerance if path == host
                            and cpu_value_relative_tolerance is not None
                            else value_relative_tolerance) if channel == 0 else 1e-6
                compare(actual[channel::3], target_image[channel::3], limit, relative)
    if tolerance is not None and cpu_expected is None:
        for channel in range(3):
            limit = (cpu_value_tolerance if channel == 0
                     and cpu_value_tolerance is not None else tolerance)
            relative = max(value_relative_tolerance,
                           cpu_value_relative_tolerance or value_relative_tolerance)
            compare(images[1][channel::3], images[0][channel::3], limit,
                    relative if channel == 0 else 1e-6)
    return images


def check_control_flow_suite():
    prepare_texture_images()
    connected = connected_group("hart_control_flow",
                                producer="hart_control_source")
    configurations = [
        ("-O0", "10", []), ("-O2", "3", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"]),
    ]
    for osl_opt, llvm_opt, mode in configurations:
        print("Checking HART control flow", osl_opt, llvm_opt, mode, flush=True)
        flags = [osl_opt, "--llvm_opt", llvm_opt]
        cases = [
            (["hart_control_ops"], 33, 5, reference(33, 5, control_ops_result)),
            (["hart_control_short"], 5, 5, [0, 1, 0]*25),
        ]
        for width, height in ((1, 1), (5, 5)):
            expected = reference(
                width, height,
                lambda u, v: control_flow_result(u, v, width, height))
            cases.append((connected, width, height, expected))
        for shaders, width, height, expected in cases:
            check_image_render(shaders, flags, mode, width, height, expected)
        rejected = root / "control-rejected.pfm"
        run(["--hart", "--hart-no-cache", "-v"] + mode + flags
            + ["-g", "5", "5", "--param", "force", "1",
               "-o", "Cout", str(rejected), "hart_control_short"],
            "nonfinite coordinates/gradients", error_after_launch=True)
        assert not rejected.exists()


def check_aggregate_suite():
    for shader in ("hart_struct_source", "hart_struct_consumer"):
        original = (root / (shader + ".oso")).read_text()
        legacy = original.replace("%structfields{weight,shade}", "")
        assert legacy != original
        (root / (shader + "_legacy.oso")).write_text(legacy)
    run(["--hart", "-v"] + connected_group(
        "hart_struct_consumer_legacy", producer="hart_struct_source_legacy"),
        "missing struct-array field metadata")

    def array_value(u, v, write=False):
        if write:
            return u+v, 0.125, 0.25
        return ((u, 0.125, 0), (2*u+v, 0.25, 0.25),
                (3*u-v, 0.375, -0.25),
                (u*v, 0.125*v, 0.25*u))[int(3*u)]

    def structure_value(u, v):
        i, j = int(u >= 0.5), int(2*v)
        term, dx, dy = ((u, 1, 0), (v, 0, 1), (u*v, v, u))[j]
        return ((i+2)*u+2*v+i+u*v+term,
                (i+2+v+dx)*0.125, (2+u+dy)*0.25)

    def vector_value(u, v):
        q, dx, dy = ((u, 0.125, 0), (v, 0, 0.25),
                     (u*v, 0.125*v, 0.25*u))[int(2*u)]
        return q+u, dx+0.125, dy

    connected = connected_group("hart_struct_consumer",
                                producer="hart_struct_source")
    configurations = [
        ("-O0", "10", []), ("-O2", "3", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"]),
    ]
    for osl_opt, llvm_opt, mode in configurations:
        print("Checking HART aggregates", osl_opt, llvm_opt, mode, flush=True)
        flags = [osl_opt, "--llvm_opt", llvm_opt]
        cases = [
            (["hart_array"], reference(9, 5, array_value)),
            (["--param", "write", "1", "hart_array"],
             reference(9, 5, lambda u, v: array_value(u, v, True))),
            (connected, reference(9, 5, structure_value)),
            (["hart_aggregate_indices"], reference(9, 5, vector_value)),
            (["--param", "kind", "1", "hart_aggregate_indices"],
             reference(9, 5, lambda u, v: (2*u+v, 0, 0))),
            (["--param", "kind", "2", "hart_aggregate_indices"],
             reference(9, 5, lambda u, v: (3+2*int(2*u)+int(4*v), 0, 0))),
            (["--param", "kind", "3", "hart_aggregate_indices"],
             reference(9, 5, lambda u, v: (2*u+v, 0, 0))),
            (["--shader", "hart_struct_source", "producer",
              "--shader", "hart_array_param", "consumer",
              "--connect", "producer", "value.coefficients",
              "consumer", "values"],
             reference(9, 5, lambda u, v:
                       ((u+v+u*v)*(1+u),
                        ((1+v)*(1+u)+u+v+u*v)*0.125, 19))),
        ]
        for count in (1, 3, 9):
            params = (["--param:type=float[" + str(count) + "]", "values",
                       ",".join(str(i+1) for i in range(count))]
                      if count != 3 else [])
            total = count*(count+1)/2
            expected = reference(
                9, 5, lambda u, v, total=total, count=count:
                (total*(1+u), total*0.125, count+16))
            cases.append((params + ["hart_array_param"], expected))
        for shaders, expected in cases:
            check_image_render(shaders, flags, mode, 9, 5, expected)
        rejected = root / "aggregate-rejected.pfm"
        for shaders in (["hart_array"], ["--param", "write", "1", "hart_array"],
                        ["hart_aggregate_indices"],
                        ["--param", "kind", "1", "hart_aggregate_indices"],
                        ["--param", "kind", "2", "hart_aggregate_indices"],
                        ["--param", "kind", "3", "hart_aggregate_indices"]):
            for offset in ("-1", "4"):
                run(["--hart", "--hart-no-cache", "-v"] + mode + flags
                    + ["-g", "9", "5", "--param", "offset", offset,
                       "-o", "Cout", str(rejected)] + shaders,
                    "index out of range", error_after_launch=True)
                assert not rejected.exists()
        check_image_render(["hart_aggregate_bad"], flags, mode, 9, 5,
                           reference(9, 5, lambda u, v: (u, v, 0)))


def check_string_suite():
    configurations = [
        ("-O0", "10", []), ("-O2", "3", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"])]
    default_names = ["red", "green", "blue", "", "red"]
    cases = [("alpha", default_names), ("red", default_names), ("", default_names),
             ("ALPHA", [""]),
             ("a-long-distinct-string", ["red", "Red", "", "green", "blue",
                                       "alpha", "ALPHA", "red", "a-long-distinct-string"])]
    for osl_opt, llvm_opt, mode in configurations:
        flags = [osl_opt, "--llvm_opt", llvm_opt]
        print("Checking HART string values", flags, mode, flush=True)
        for text, names in cases:
            arguments = ["--param:type=string", "text", text,
                         f"--param:type=string[{len(names)}]", "names", ",".join(names)]

            def expected(u, v):
                i = (int(8*u+.5) + 9*int(4*v+.5)) % len(names)
                original = names[i]
                selected = text if v > .5 else original
                return ((original == "red") + 2*(selected == text)
                        + 4*(selected != "red") + 8*(original == "")
                        + 16*(original != ""), 18, 31)

            check_image_render(arguments + ["hart_string_values"], flags, mode,
                               9, 5, reference(9, 5, expected))
        for text in ("alpha", ""):
            shaders = ["--param:type=string", "text", text,
                       "--shader", "hart_string_source", "producer",
                       "--shader", "hart_string_consumer", "consumer",
                       "--connect", "producer", "value", "consumer", "value",
                       "--connect", "producer", "words", "consumer", "words"]
            expected = reference(9, 5, lambda u, v:
                                 ((text if u > .5 else "red") == "red",
                                  3, ("" if v > .5 else text) == ""))
            check_image_render(shaders, flags, mode, 9, 5, expected)

    source = root / "string_bounds.osl"
    source.write_text(
        'shader string_bounds(int write=0,int offset=0,output color Cout=0){'
        'string a[3]={"red","green","blue"};int i=int(2*u);'
        'if(write){a[i+offset]="changed";Cout=color(a[i]=="changed");}'
        'else Cout=color(a[i+offset]=="red");}', encoding="ascii")
    compile_fixture(source)
    for osl_opt, llvm_opt, mode in (configurations[0], configurations[3]):
        for write in (0, 1):
            for offset in (-1, 3):
                rejected = root / "string-rejected.pfm"
                run(["--hart", "--hart-no-cache", "-v", osl_opt, "--llvm_opt",
                     llvm_opt] + mode + ["-g", "9", "5", "--param", "write",
                     str(write), "--param", "offset", str(offset), "-o", "Cout",
                     str(rejected), "string_bounds"], "index out of range",
                    error_after_launch=True)
                assert not rejected.exists()

    for expression, operation in (('strlen(s)', "strlen"), ('getchar(s,0)', "getchar"),
                                   ('regex_match(s,"a")', "regex_match")):
        source = root / "string_operation.osl"
        source.write_text(
            f'shader string_operation(string s="alpha",output color Cout=0){{'
            f'if(u<0)Cout=color({expression});}}', encoding="ascii")
        compile_fixture(source)
        run(["--hart", "-v", "string_operation"], f"unsupported operation '{operation}'")


def check_diagnostic_suite():
    def shader(name, parameters, body):
        source = root / (name + ".osl")
        source.write_text(f"shader {name}({parameters}output color Cout=0) {{\n"
                          + body + "\nCout=color(u,v,.5);\n}\n", encoding="ascii")
        compile_fixture(source)

    shader("hart_diag_values", "", r'''
        string words[3]={"red","Red",""};
        int a[2]={-1,2};
        matrix m=matrix(1);
        printf("DIAG value=%d text=[%s] scalar=%.2f array=%d matrix=%.0f %% {ok}\n",
               int(2*u)-1,words[int(2*u)],.25+v,a,m);
        if (u==0 && v==0) {
            printf("");
            printf("DIAG plain\n");
            warning("DIAG warning %s %d\n","MiXeD",7);
        }
    ''')
    shader("hart_diag_source", 'output string text="",', '''
        text = u>.5 ? "Case" : "";
    ''')
    shader("hart_diag_connected", 'string text="missing",', r'''
        printf("DIAG connected [%s] %o %x %X %i\n",
               text,-1,-1,-1,-7);
    ''')
    shader("hart_diag_capacity", 'int reports=0,', r'''
        for(int i=0;i<reports;++i)
            printf("DIAG record %d\n",i);
    ''')
    shader("hart_diag_error", "", r'''
        if(u==0)
            error("DIAG failure [%s] %d\n","MiXeD",7);
    ''')
    shader("hart_diag_string", 'string text="",', 'printf("%s",text);')
    shader("hart_diag_message", "", 'printf("%1024s%1024s%1024s%1024s","a","b","c","d");')
    shader("hart_diag_long_message", "", 'printf("%1024s%1024s%1024s%1024s!","a","b","c","d");')
    names = ",".join('"x"' for _ in range(256))
    shader("hart_diag_payload", "", f'string a[256]={{{names}}}; printf("%s",a);')
    image = root / "diagnostic.pfm"
    expected_pixels = reference(3, 2, lambda u, v: (u, v, .5))
    configurations = [
        ("-O0", "10", []), ("-O2", "3", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"])]

    def records(output, name):
        return re.findall(r"HART shader '" + name
                          + r"' \(([^\r\n]*):(\d+), point (\d+)\): ([^\r\n]*)",
                          output)

    def render(shaders, flags, mode, width=1, height=1, repeat=False, error=None,
               launched=True):
        if image.exists():
            image.unlink()
        arguments = ["--hart", "--hart-no-cache", "-v", "-g", str(width),
                     str(height), "-o", "Cout", str(image)] + flags + mode
        if repeat:
            arguments += ["--warmup", "--iters", "3"]
        output = run(arguments + shaders, error,
                     error_after_launch=launched if error else False)
        assert image.exists() == (error is None), output
        return output

    for osl_opt, llvm_opt, mode in configurations:
        flags = [osl_opt, "--llvm_opt", llvm_opt]
        print("Checking HART diagnostics", flags, mode, flush=True)
        cpu = run(flags + ["-g", "3", "2", "hart_diag_values"])
        cpu_values = re.findall(r"DIAG value=[^\r\n]*", cpu)
        assert len(cpu_values) == 6, cpu
        output = render(["hart_diag_values"], flags, mode, 3, 2, repeat=True)
        reports = records(output, "hart_diag_values")
        assert len(reports) == 36, output
        assert all(Path(source).name == "hart_diag_values.osl" and int(line) > 0
                   for source, line, point, text in reports), reports
        for point in range(6):
            values = [text for _, _, index, text in reports
                      if int(index) == point and text.startswith("DIAG value=")]
            assert len(values) == 4 and values == [cpu_values[point]] * 4, values
        for message in ("", "DIAG plain", "DIAG warning MiXeD 7"):
            assert sum(text == message and point == "0"
                       for _, _, point, text in reports) == 4, reports
        compare(image_pixels(image, 3, 2), expected_pixels, 0, 0)
        connected = ["--shader", "hart_diag_source", "producer",
                     "--shader", "hart_diag_connected", "consumer",
                     "--connect", "producer", "text", "consumer", "text"]
        output = render(connected, flags, mode, 2, 1)
        messages = {int(point): text for _, _, point, text
                    in records(output, "hart_diag_connected")}
        for point, word in enumerate(("", "Case")):
            assert messages[point] == (f"DIAG connected [{word}] "
                                       "37777777777 ffffffff FFFFFFFF -7"), messages
        # Exact saturation boundary and resets across warmup + three launches.
        output = render(["--param", "reports", "256", "hart_diag_capacity"],
                        flags, mode, repeat=True)
        reports = records(output, "hart_diag_capacity")
        assert len(reports) == 1024, len(reports)
        for i in range(256):
            assert sum(text == f"DIAG record {i}" for _, _, _, text in reports) == 4
        output = render(["--param", "reports", "0", "hart_diag_capacity"], flags, mode)
        assert not records(output, "hart_diag_capacity"), output
        output = render(["--param", "reports", "257", "hart_diag_capacity"],
                        flags, mode, error="diagnostic buffer overflow")
        assert len(records(output, "hart_diag_capacity")) == 256, output
        output = render(["hart_diag_error"], flags, mode, 2, 1,
                        error="shader error")
        assert [text for _, _, _, text in records(output, "hart_diag_error")] == [
            "DIAG failure [MiXeD] 7"], output
        for text in ("", "x" * 1024):
            output = render(["--param:type=string", "text", text, "hart_diag_string"],
                            flags, mode)
            assert [s for _, _, _, s in records(output, "hart_diag_string")] == [text]
        render(["--param:type=string", "text", "x" * 1025, "hart_diag_string"],
               flags, mode, error="field exceeds 1024")
        output = render(["hart_diag_message"], flags, mode)
        assert [len(text) for _, _, _, text in records(output, "hart_diag_message")] == [4096]
        render(["hart_diag_long_message"], flags, mode, error="message exceeds 4096")
        output = render(["hart_diag_payload"], flags, mode)
        assert [text for _, _, _, text in records(output, "hart_diag_payload")] == [
            " ".join(["x"] * 256)], output

    output = render(["--param", "reports", "3", "hart_diag_capacity"],
                    ["-O2", "--llvm_opt", "3"], [], 86, 1,
                    error="diagnostic buffer overflow")
    assert len(records(output, "hart_diag_capacity")) == 256, output

    # Runtime argument errors and unsupported formats must not become printf success.
    for name, parameters, body, message in (
        ("dynamic", 'string format="%s",', 'printf(format,"x");', "must be literal"),
        ("width", "", 'printf("%1025d",7);', "width exceeds"),
        ("precision", "", 'printf("%.129f",u);', "precision exceeds"),
    ):
        name = "hart_diag_" + name
        shader(name, parameters, body)
        render([name], ["-O2", "--llvm_opt", "3"], [], error=message, launched=False)


def check_interactive_userdata_suite():
    producer = root / "hart_combined_runtime_producer.osl"
    producer.write_text("""
shader hart_combined_runtime_producer(
    float shared=2 [[int interpolated=1]], output float value=0) {
    value=shared;
}
""", encoding="ascii")
    consumer = root / "hart_combined_runtime_consumer.osl"
    consumer.write_text("""
shader hart_combined_runtime_consumer(
    float value=0, float shared=5 [[int interpolated=1]],
    float red=3 [[int interpolated=1]],
    int count=7 [[int interpolated=1]],
    color tint=color(.25,.5,.75) [[int interpolated=1]],
    float weights[2]={.25,.5} [[int interpolated=1]],
    matrix basis=1 [[int interpolated=1]], output color Cout=0) {
    float q=value+shared+red+count+tint[0]+2*tint[1]+3*tint[2]
            +weights[0]+2*weights[1]+basis[0][0];
    Cout=color(q,Dx(shared)+Dx(red)+Dx(tint[0])+Dx(weights[1]),
                 Dy(shared)+Dy(red)+Dy(tint[1])+Dy(weights[0]));
    printf("COMBINED %d %d %.9g %.9g %.9g %.9g %.9g\\n",
           int(2*u),int(v),value,shared,Cout[0],Cout[1],Cout[2]);
}
""", encoding="ascii")
    compile_fixture(producer)
    compile_fixture(consumer)
    hints_consumer = root / "hart_cli_hint_consumer.osl"
    hints_consumer.write_text("""
shader hart_cli_hint_consumer(float value=0, float shared=5,
                             output color Cout=0) {
    Cout=color(value+shared);
    printf("CLI_HINT %.9g\\n",Cout[0]);
}
""", encoding="ascii")
    compile_fixture(hints_consumer)
    fields = (("shared", "float"), ("red", "float"),
              ("count", "int"), ("tint", "color"),
              ("weights", "float[2]"), ("basis", "matrix"))
    states = (
        {"shared": "5", "red": "3", "count": "7", "tint": ".25,.5,.75",
         "weights": ".25,.5", "basis": "1"},
        {"shared": "13", "red": "6", "count": "11", "tint": "1,2,3",
         "weights": "2,4", "basis": "2"},
    )
    # Keep uniform bindings independent of the CPU's specialized builtin getters.
    supplied = ["--userdata:type=float", "shared", "9",
                "--userdata:type=int", "count", "17",
                "--userdata:type=color", "tint", "2,3,4",
                "--userdata:type=float[2]", "weights", "5,7",
                "--userdata:type=matrix", "basis", "3"]
    mismatched = ["--userdata:type=int", "shared", "9",
                  "--userdata:type=float", "count", "17",
                  "--userdata:type=float", "tint", "2",
                  "--userdata:type=float", "weights", "5",
                  "--userdata:type=float", "basis", "3"]
    cases = (("missing", [], False), ("found", supplied, True),
             ("mismatched", mismatched, False))

    def graph(state, interactive):
        hint = ":interactive=1" if interactive else ""
        result = ["--layer", "producer", "--param:type=float", "shared", "2",
                  producer.stem, "--layer", "consumer"]
        for name, datatype in fields:
            result += ["--param:type="+datatype+hint, name, state[name]]
        return result + [consumer.stem, "--connect", "producer", "value",
                         "consumer", "value", "--entry", "producer",
                         "--entry", "consumer"]

    def updates(state):
        result = []
        for name, datatype in fields:
            result += ["--reparam:type="+datatype, "consumer", name, state[name]]
        return result

    def expected(state, found):
        result = []
        for y in range(2):
            for x in range(3):
                u = x/2
                upstream = 9 if found else 2
                shared = 9 if found else float(state["shared"])
                red = u if u > .5 else float(state["red"])
                count = 17 if found else int(state["count"])
                tint = ((2, 3, 4) if found else
                        tuple(map(float, state["tint"].split(","))))
                weights = ((5, 7) if found else
                           tuple(map(float, state["weights"].split(","))))
                basis = 3 if found else float(state["basis"])
                q = (upstream+shared+red+count
                     +sum((i+1)*t for i, t in enumerate(tint))
                     +weights[0]+2*weights[1]+basis)
                result.append((x, y, upstream, shared, q,
                               .5 if u > .5 else 0, 0))
        return result

    def records(output):
        result = []
        for x, y, values in re.findall(r"COMBINED (\d+) (\d+) ([^\r\n]+)", output):
            values = tuple(map(float, values.split()))
            assert len(values) == 5, output
            result.append((int(x), int(y)) + values)
        return Counter(result)

    def check_cli_hints(flags, mode_env):
        output = run(flags + [
            "-g", "1", "1", "--iters", "2", "--print",
            "--userdata:type=float", "unused", "9",
            "--layer", "producer", "--param:type=float", "shared", "2",
            producer.stem,
            "--reparam:type=float", "consumer", "shared", "11",
            "--layer", "consumer",
            "--param:type=float:interactive=1", "shared", "5",
            hints_consumer.stem,
            "--connect", "producer", "value", "consumer", "value",
        ], extra_env=mode_env)
        assert re.findall(r"CLI_HINT ([^\r\n]+)", output) == ["7", "13"], output
        compare(pixels(output, 1, 1), [13, 13, 13])
        if mode_env.get("TESTSHADE_HART") == "1":
            kind = "fused" if mode_env["TESTSHADE_FUSED"] == "1" else "split"
            assert "HART callable mode: " + kind in output, output
            assert output.count("Launching HART grid") == 2, output

    grid = ["-g", "3", "2", "--print"]
    for backend, error in (
        ("TESTSHADE_OPTIX", "mutually exclusive"),
        ("TESTSHADE_BATCHED", "does not support TESTSHADE_BATCHED"),
        ("TESTSHADE_RS_BITCODE", "does not support TESTSHADE_RS_BITCODE"),
    ):
        run(["--print", hints_consumer.stem], error,
            {"TESTSHADE_HART": "1", backend: "1"})
    cpu_values = {}
    # The CPU's combined eager path bypasses userdata. Ordinary-interpolated
    # staged controls provide the intended lookup/default oracle instead.
    for optimize in (0, 2):
        mode_env = {"TESTSHADE_OPT": str(optimize),
                    "TESTSHADE_LLVM_OPT": "10" if optimize == 0 else "3"}
        check_cli_hints(["-t", "1"], mode_env)
        for name, userdata, found in cases:
            for stage, state in enumerate(states):
                output = run(["-t", "1"] + grid + graph(state, False) + userdata,
                             extra_env=mode_env)
                values = expected(state, found)
                assert records(output) == Counter(values), (
                    f"CPU O{optimize} {name} stage {stage}\n" + output)
                wanted_pixels = [v for row in values for v in row[4:]]
                actual_pixels = pixels(output, 3, 2)
                compare(actual_pixels, wanted_pixels)
                cpu_values[optimize, name, stage] = actual_pixels

    for mode, optimize, llvm, flags in (
        ("split", 2, 3, []),
        ("fused", 2, 3, ["--hart-fused"]),
        ("fused-local", 2, 3,
         ["--hart-fused", "--hart-local-groupdata", "1048576"]),
        ("unoptimized", 0, 10, []),
    ):
        mode_env = {"TESTSHADE_OPT": str(optimize),
                    "TESTSHADE_LLVM_OPT": str(llvm)}
        check_cli_hints(
            ["-v"] + [flag for flag in flags if flag != "--hart-fused"],
            {**mode_env, "TESTSHADE_HART": "1",
             "TESTSHADE_FUSED": str(int(mode in ("fused", "fused-local")))})
        for name, userdata, found in cases:
            # Both transitions execute in place on their compiled group.
            # Compiler/API coverage separately checks one group's A-B-A arena.
            for initial, final in ((0, 1), (1, 0)):
                output = run(["--hart", "-v", "--warmup", "--iters", "2"]
                             + flags + grid + graph(states[initial], True)
                             + userdata + updates(states[final]),
                             extra_env=mode_env)
                wanted = Counter(expected(states[initial], found))
                wanted.update(expected(states[initial], found))
                wanted.update(expected(states[final], found))
                assert records(output) == wanted, output
                assert output.count("Launching HART grid") == 3, output
                prefix = "HART shader 'hart_combined_runtime_consumer'"
                assert output.count(prefix) == 18, output
                compare(pixels(output, 3, 2),
                        cpu_values[optimize, name, final])
        output = run(["--hart", "-v", "--iters", "2"] + flags + grid
                     + graph(states[0], True)
                     + ["--reparam:type=int", "consumer", "red", "99"],
                     "type mismatch", mode_env, error_after_launch=True)
        assert output.count("Launching HART grid") == 1, output
        assert "Pixel (" not in output, output
        print("HART combined defaults " + mode
              + ": typed hits/misses, per-layer fallback, updates "
              "and gradients passed")


def check_selector_suite():
    configurations = [
        ("-O0", "10", []), ("-O2", "3", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"])]
    source = root / "selector_grid.osl"
    source.write_text(
        'shader selector_grid(int derivatives=0,output color Cout=0){'
        'Cout=derivatives?color(Dx(u),Dy(v),Dx(P[0])+Dy(P[1]))'
        ':color(u,v,int(6*u)+8*int(6*v));}', encoding="ascii")
    compile_fixture(source)
    def float32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]
    for width, height in ((7, 7), (1, 1)):
        for derivatives in (False, True):
            def expected_grid(u, v):
                if derivatives:
                    dx, dy = float32(1/max(1, width-1)), float32(1/max(1, height-1))
                    return dx, dy, float32(dx+dy)
                return (float32(u), float32(v),
                        int(float32(6*float32(u))) + 8*int(float32(6*float32(v))))
            for osl_opt, llvm_opt, mode in configurations:
                check_image_render(
                    ["--param", "derivatives", str(int(derivatives)), "selector_grid"],
                    [osl_opt, "--llvm_opt", llvm_opt], mode, width, height,
                    reference(width, height, expected_grid))
    source = root / "selector_hash.osl"
    source.write_text(
        'shader selector_hash(string value="",output color Cout=0){'
        'string names[5]={"","alpha","ALPHA","a-long-distinct-string","red"};'
        'float x=u<=.5?.25:.75,y=v<=.5?.25:.75;point p=point(x,y,1);'
        'int kind=int(5*u),h=0;'
        'if(kind==0)h=hash(u==0?value:names[int(4*v)]);'
        'else if(kind==1)h=hash(int(4*v)-2);'
        'else if(kind==2)h=hash(x);else if(kind==3)h=hash(x,y);'
        'else if(kind==4)h=hash(p);else h=hash(p,y);'
        'Cout=color(h&65535,(h>>16)&65535,h<0);}', encoding="ascii")
    compile_fixture(source)
    width, height = 11, 5
    # Published CPU hash regression vectors, packed without float precision loss.
    golden = {
        (2, .25, .25): -1518044388, (2, .25, .75): -1518044388,
        (3, .75, .25): -802388107, (3, .75, .75): 2050085294,
        (4, .75, .25): -1172708893, (4, .75, .75): 1398623619,
        (5, .75, .25): 1884197054, (5, .75, .75): -626565092}
    for shaders in (["selector_hash"],
                    ["--param:type=string", "text", "alpha",
                     "--shader", "hart_string_source", "producer",
                     "--shader", "selector_hash", "consumer",
                     "--connect", "producer", "value", "consumer", "value"]):
        host = root / "hash-reference.pfm"
        run(["-O0", "-t", "1", "-g", str(width), str(height), "-d", "float",
             "-o", "Cout", str(host)] + shaders)
        expected = image_pixels(host, width, height)
        assert expected[3:6] == [0, 0, 0]  # Empty string's hash.
        assert set(expected[2::3]) == {0, 1}
        for row in range(height):
            for column in range(width):
                key = (int(5*column/(width-1)),
                       .25 if column <= (width-1)/2 else .75,
                       .25 if row <= (height-1)/2 else .75)
                if key in golden:
                    h = golden[key]
                    i = 3*(row*width+column)
                    assert expected[i:i+3] == [h & 65535, (h >> 16) & 65535, int(h < 0)]
        for osl_opt, llvm_opt, mode in configurations:
            print("Checking HART hashes", shaders, osl_opt, mode, flush=True)
            check_image_render(shaders, [osl_opt, "--llvm_opt", llvm_opt],
                               mode, width, height, expected)

    names = ["perlin", "uperlin", "noise", "snoise", "cell", "hash", "gabor",
             "simplex", "usimplex"]
    position = "point(.3125+u*.125,.125+v*.25,.25+u*.0625)"
    source = root / "selector_source.osl"
    source.write_text(
        'shader selector_source(int periodic=0,output string name="",'
        'output point position=0){string names[9]={'
        + ",".join(f'"{name}"' for name in names)
        + '};name=names[int((periodic?6:8)*u)];position=' + position + ";}",
        encoding="ascii")
    compile_fixture(source)
    for periodic in (False, True):
        selected_names = names[:7] if periodic else names
        width, height = len(selected_names), 5
        op = "pnoise" if periodic else "noise"
        for dimension, coordinates in enumerate(
                ("position[0]", "position[0],position[1]", "position",
                 "position,.25+v*.125"), 1):
            periods = ("4", "4,5", "point(4,5,6)", "point(4,5,6),7")[dimension-1]
            operands = coordinates + ("," + periods if periodic else "")
            for triple in (False, True):
                kind = "color" if triple else "float"
                label = f"selector_{op}_{dimension}_{kind}"
                for literal in (False, True):
                    shader = label + ("_literal" if literal else "")
                    source = root / (shader + ".osl")
                    body = (f'{kind} value={op}(name,{operands});' if not literal
                            else f"{kind} value=0;" + "".join(
                                f'if(name=="{name}")value={op}("{name}",{operands});'
                                for name in selected_names))
                    source.write_text(
                        f'shader {shader}(string name="perlin",'
                        f'point position=0,int derivatives=1,output color Cout=0){{'
                        + body + ("float q=value[int(2*v)];" if triple else "float q=value;")
                        + "Cout=derivatives?color(q,Dx(q),Dy(q)):color(value);}",
                        encoding="ascii")
                    compile_fixture(source)
                def group(shader, derivatives=1):
                    return ["--param", "periodic", str(int(periodic)),
                            "--shader", "selector_source", "producer",
                            "--param", "derivatives", str(derivatives),
                            "--shader", shader, "consumer",
                            "--connect", "producer", "name", "consumer", "name",
                            "--connect", "producer", "position", "consumer", "position"]
                for derivatives in (1, 0):
                    host = root / "selector-reference.pfm"
                    run(["-O0", "-t", "1", "-g", str(width), str(height),
                         "-d", "float", "-o", "Cout", str(host)]
                        + group(label + "_literal", derivatives))
                    expected = image_pixels(host, width, height)
                    assert all(math.isfinite(x) for x in expected)
                    assert max(expected) - min(expected) > .01
                    modes = (configurations if dimension == 3 and derivatives
                             else (configurations[0], configurations[3]))
                    for osl_opt, llvm_opt, mode in modes:
                        print("Checking HART selectors", label, derivatives,
                              osl_opt, mode, flush=True)
                        check_image_render(group(label, derivatives),
                                           [osl_opt, "--llvm_opt", llvm_opt],
                                           mode, width, height, expected, tolerance=2e-6)
                    if derivatives:
                        check_image_render(group(label + "_literal"),
                                           ["-O2", "--llvm_opt", "3"], [],
                                           width, height, expected, tolerance=2e-6)

    for periodic, options in ((False, False), (True, False), (False, True), (True, True)):
        op = "pnoise" if periodic else "noise"
        operands = 'P' + (',point(4)' if periodic else '')
        operands += ',"bandwidth",1,"do_filter",0' if options else ''
        source = root / "selector_bad.osl"
        source.write_text(
            'shader selector_bad(string bad="unknown",output color Cout=0){'
            'string name=u>.5?bad:"gabor";'
            f'Cout={op}(name,{operands});}}', encoding="ascii")
        compile_fixture(source)
        invalid = (["perlin", "cell"] if options else
                   ["unknown", "", "cellnoise"] + (["simplex", "usimplex"] if periodic else []))
        for osl_opt, llvm_opt, mode in (configurations[0], configurations[3]):
            for name in invalid:
                rejected = root / "selector-rejected.pfm"
                run(["--hart", "--hart-no-cache", "-v", osl_opt, "--llvm_opt", llvm_opt]
                    + mode + ["-g", "3", "2", "--param:type=string", "bad", name,
                              "-o", "Cout", str(rejected), "selector_bad"],
                    "invalid noise arguments", error_after_launch=True)
                assert not rejected.exists()
        if options:
            host = root / "selector-gabor-reference.pfm"
            shaders = ["--param:type=string", "bad", "gabor", "selector_bad"]
            run(["-O0", "-t", "1", "-g", "3", "2", "-d", "float",
                 "-o", "Cout", str(host)] + shaders)
            for osl_opt, llvm_opt, mode in (configurations[0], configurations[3]):
                check_image_render(shaders, [osl_opt, "--llvm_opt", llvm_opt],
                                   mode, 3, 2, image_pixels(host, 3, 2), tolerance=2e-6)


def check_spline_suite():
    width, height = 17, 3
    cases = []
    bases = [("catmull-rom", 1, 7), ("bezier", 3, 7), ("bspline", 1, 7),
             ("hermite", 2, 8), ("linear", 1, 7), ("constant", 1, 7)]
    for basis, step, length in bases:
        for triple, dynamic in ((False, False), (True, False), (False, True)):
            shader = ("spline_" + basis.replace("-", "_")
                      + ("_triple" if triple else "_scalar")
                      + ("_dynamic" if dynamic else ""))
            kind = "color" if triple else "float"
            knots = []
            for i in range(length):
                tangent = basis == "hermite" and i % 2
                position = i/3 if basis == "bezier" else i//2 if basis == "hermite" else i
                value = "b" if tangent else f"(a+b*{position:.9g})"
                if triple:
                    value = (f"color({value},2*{value},-{value})" if tangent
                             else f"color({value},2*{value}+0.5,1-{value})")
                knots.append(value)
            count = f"4+{step}*int(v>0.5)," if dynamic else ""
            selected = "result[int(2*v)]" if triple else "result"
            source = root / (shader + ".osl")
            source.write_text(
                f"shader {shader}(output color Cout=0) {{"
                f"float a=0.25*v,b=1+0.5*v;{kind} knots[{length}]={{"
                + ",".join(knots) + "};"
                f'{kind} result=spline("{basis}",2*u-0.5,{count}knots);'
                f"float q={selected};Cout=color(q,Dx(q),Dy(q));}}",
                encoding="ascii")
            compile_fixture(source)

            def expected(u, v, basis=basis, step=step, length=length,
                         triple=triple, dynamic=dynamic):
                count = 4+step*int(v > 0.5) if dynamic else length
                segments = (count-4)//step+1
                x = max(0, min(1, 2*u-0.5))
                position = (1 if step == 1 else 0) + segments*x
                if basis == "constant":
                    position = 1+min(int(segments*x), segments-1)
                value = 0.25*v+(1+0.5*v)*position
                dx = (2*(1+0.5*v)*segments/(width-1)
                      if 0 <= 2*u-0.5 <= 1 else 0)
                dy = (0.25+0.5*position)/(height-1)
                if basis == "constant":
                    dx = dy = 0
                if triple:
                    scale, offset = ((1, 0), (2, 0.5), (-1, 1))[int(2*v)]
                    value, dx, dy = scale*value+offset, scale*dx, scale*dy
                return value, dx, dy

            representative = ((triple and basis in ("catmull-rom", "bezier"))
                              or (dynamic and basis == "linear"))
            cases.append(([shader], reference(width, height, expected), representative))

    curves = [
        ("catmull-rom", lambda t: (1+t)**2, lambda t: 2+2*t, 0.5, 4),
        ("bspline", lambda t: 4/3+2*t+t*t, lambda t: 2+2*t, 0.5, 4),
        ("bezier", lambda t: 3*t+6*t*t, lambda t: 3+12*t, -1, 11),
        ("hermite", lambda t: t+t*t+2*t**3, lambda t: 1+2*t+6*t*t, -0.5, 5),
        ("linear", lambda t: 1+3*t, lambda t: 3, 0.5, 4),
        ("constant", lambda t: 1, lambda t: 0, 0, 1),
    ]
    for basis, curve, derivative, low, span in curves:
        shader = "spline_curved_" + basis.replace("-", "_")
        source = root / (shader + ".osl")
        source.write_text(
            f"shader {shader}(output color Cout=0) {{"
            "float knots[4]={0,1,4,9};"
            f'float q=spline("{basis}",2*u-0.5,knots);'
            "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
        compile_fixture(source)

        def expected(u, v, curve=curve, derivative=derivative):
            x = 2*u-0.5
            t = max(0, min(1, x))
            return curve(t), (2*derivative(t)/(width-1) if 0 <= x <= 1 else 0), 0

        cases.append(([shader], reference(width, height, expected), False))
        if basis in ("linear", "constant"):
            continue
        inverse_knots = "0,1,4,9"
        if basis == "hermite":
            # Keep this numerical reference inside the existing bounded
            # solver's convergence range (a monotonic t+t*t curve).
            inverse_knots = "0,1,2,3"
            curve, derivative = lambda t: t+t*t, lambda t: 1+2*t
            low, span = -0.25, 2.5
        shader = "spline_curved_inverse_" + basis.replace("-", "_")
        source = root / (shader + ".osl")
        source.write_text(
            f"shader {shader}(output color Cout=0) {{"
            f"float knots[4]={{{inverse_knots}}};"
            f'float q=splineinverse("{basis}",{low}+{span}*u,knots);'
            "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
        compile_fixture(source)

        def expected(u, v, basis=basis, curve=curve, derivative=derivative,
                     low=low, span=span):
            y = low+span*u
            # The shared inverse clamps to knot values, not curve endpoints.
            lower = max(curve(0), 1 if basis in ("catmull-rom", "bspline") else 0)
            upper = min(curve(1), 4 if basis in ("catmull-rom", "bspline") else 9)
            if y <= lower:
                return 0, 0, 0
            if y >= upper:
                return 1, 0, 0
            left, right = 0, 1
            for _ in range(60):
                middle = (left+right)/2
                if curve(middle) < y:
                    left = middle
                else:
                    right = middle
            t = (left+right)/2
            return t, span/derivative(t)/(width-1), 0

        cases.append(([shader], reference(width, height, expected), False))

    for varying in (False, True):
        shader = "spline_inverse_" + ("varying" if varying else "fixed")
        source = root / (shader + ".osl")
        position = "2*u-0.5+0.03125" if varying else "2*u-0.5"
        source.write_text(
            f"shader {shader}(output color Cout=0) {{"
            + ("float a=0.25*v,b=1+0.5*v;" if varying else "float a=0,b=1;")
            + "float knots[7]={a,a+b,a+2*b,a+3*b,a+4*b,a+5*b,a+6*b};"
            + f'float q=splineinverse("linear",a+b*(1+4*({position})),knots);'
              "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
        compile_fixture(source)

        def expected(u, v, varying=varying):
            # Offset varying knots from solver branch boundaries, where float
            # rounding can select bisection (zero) or the ordinary derivative.
            x = 2*u-0.5+(0.03125 if varying else 0)
            if not 0 < x < 1:
                return max(0, min(1, x)), 0, 0
            if 4*x == int(4*x):
                # The existing inverse bisects at exact segment boundaries,
                # dropping derivatives rather than taking the one-sided slope.
                return x, 0, 0
            # The existing inverse helper deliberately ignores knot derivatives.
            dy = ((0.25+0.5*(1+4*x))/(4*(1+0.5*v))/(height-1)
                  if varying else 0)
            return x, 2/(width-1), dy

        cases.append(([shader], reference(width, height, expected), True))

    for shader, parameter in (("spline_parameters", "knots"),
                              ("spline_connected", "value")):
        source = root / (shader + ".osl")
        source.write_text(
            f"shader {shader}(float {parameter}[]={{0,1,2,3}},output color Cout=0) {{"
            f'float q=spline("linear",2*u-0.5,{parameter});'
            "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
        compile_fixture(source)
    for length in (4, 7, 9):
        parameters = ["--param:type=float[" + str(length) + "]", "knots",
                      ",".join(str(i) for i in range(length)), "spline_parameters"]

        def expected(u, v, length=length):
            x = 2*u-0.5
            return (1+(length-3)*max(0, min(1, x)),
                    2*(length-3)/(width-1) if 0 <= x <= 1 else 0, 0)

        cases.append((parameters, reference(width, height, expected), length == 7))
    source = root / "spline_producer.osl"
    source.write_text(
        "shader spline_producer(output float value[7]={0,0,0,0,0,0,0}) {"
        "for(int i=0;i<7;++i)value[i]=i+0.25*v;}", encoding="ascii")
    compile_fixture(source)
    connected = connected_group("spline_connected", producer=source.stem)
    cases.append((connected, reference(width, height, lambda u, v:
                  (1+4*max(0, min(1, 2*u-0.5))+0.25*v,
                   8/(width-1) if 0 <= 2*u-0.5 <= 1 else 0,
                   0.25/(height-1))), True))

    configurations = [
        ("-O2", "3", []), ("-O0", "10", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"]),
    ]
    for shaders, expected, representative in cases:
        for osl_opt, llvm_opt, mode in configurations if representative else configurations[:1]:
            print("Checking spline", shaders, osl_opt, llvm_opt, mode, flush=True)
            check_image_render(shaders, [osl_opt, "--llvm_opt", llvm_opt],
                               mode, width, height, expected, tolerance=2e-6)

    for inverse in (False, True):
        shader = "spline_invalid_" + ("inverse" if inverse else "forward")
        operation = "splineinverse" if inverse else "spline"
        source = root / (shader + ".osl")
        source.write_text(
            f"shader {shader} [[int range_checking=0]] "
            "(int offset=0,float special=0,output color Cout=0) {"
            "float knots[7]={0,1,2,3,4,5,6};int count=4+3*int(u>0.5)+offset;"
            f'float q={operation}("bezier",u<0.5 ? special : u,count,knots);'
            "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
        compile_fixture(source)
        for osl_opt, llvm_opt, mode in (configurations[1], configurations[3]):
            for text in ("inf", "-inf"):
                def expected(u, v, inverse=inverse, text=text):
                    segments = 1+int(u > 0.5)
                    if u < 0.5:
                        return (1 if inverse else 3*segments) if text == "inf" else 0, 0, 0
                    scale = 1/(3*segments) if inverse else 3*segments
                    return scale*u, scale/4, 0

                check_image_render(
                    ["--param:type=float", "special", text, shader],
                    [osl_opt, "--llvm_opt", llvm_opt], mode, 5, 3,
                    reference(5, 3, expected), tolerance=2e-6)
        for osl_opt, llvm_opt, mode in configurations:
            rejected = root / "spline-rejected.pfm"
            errors = [["--param", "offset", "-1"],
                      ["--param", "offset", "1"],
                      ["--param:type=float", "special", "nan"]]
            if mode == configurations[3][2]:
                errors += [["--param", "offset", "-2147483648"],
                           ["--param", "offset", "2147483647"]]
            for parameters in errors:
                run(["--hart", "--hart-no-cache", "-v", osl_opt, "--llvm_opt",
                     llvm_opt, "-g", "5", "3", "-o", "Cout", str(rejected)]
                    + mode + parameters + [shader], "invalid spline arguments",
                    error_after_launch=True)
                assert not rejected.exists()
    for basis, length, count, error in [
            ("unknown", 4, 4, "spline basis"), ("linear", 3, 3, "spline knot"),
            ("linear", 4, 3, "spline knot"), ("bezier", 5, 5, "spline knot")]:
        source = root / "spline_bad.osl"
        source.write_text(
            "shader spline_bad(output color Cout=0) {"
            f"float k[{length}]={{" + ",".join(str(i) for i in range(length))
            + f'}};Cout=color(spline("{basis}",u,{count},k));}}', encoding="ascii")
        compile_fixture(source)
        rejected = root / "spline-rejected.pfm"
        run(["--hart", "-v", "-o", "Cout", str(rejected), "spline_bad"], error)
        assert not rejected.exists()
    run(["--hart", "-v", "--param:type=float[3]", "knots", "0,1,2",
         "-o", "Cout", str(rejected), "spline_parameters"], "spline knot")
    assert not rejected.exists()


def check_gabor_suite():
    width, height = 9, 5
    configurations = [
        ("-O2", "3", []), ("-O0", "10", []),
        ("-O2", "3", ["--hart-fused"]),
        ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"])]
    setup = ("float U=u+offset_u,V=v+offset_v;"
             "point p=point(.317+1.3*U,.127+.7*V,.233+.4*U+.2*V);")
    params = "float offset_u=0,float offset_v=0"
    report = "Cout=color(n,Dx(n),Dy(n));"

    def compile_source(name, body, parameters=params):
        source = root / (name + ".osl")
        source.write_text(f"shader {name}({parameters},output color Cout=0){{"
                          + body + "}", encoding="ascii")
        compile_fixture(source)

    def call(selector, dimension, periodic, options="", point="p"):
        coords = ((point+"[0]",), (point+"[0]", point+"[1]"),
                  (point,), (point, ".61+.3*U-.2*V"))[dimension-1]
        periods = (("3",), ("3", "4"), ("point(3,4,5)",),
                   ("point(3,4,5)", "7"))[dimension-1] if periodic else ()
        return (("pnoise" if periodic else "noise") + f'("{selector}",'
                + ",".join(coords+periods) + options + ")")

    def render(name, configuration=configurations[0], target=None, tolerance=2e-6,
               arguments=()):
        osl_opt, llvm_opt, mode = configuration
        flags = [osl_opt, "--llvm_opt", llvm_opt]
        shaders = list(arguments) + [name]
        if target is None:
            target = noise_cpu_image(flags + shaders, width, height)
            assert all(math.isfinite(x) for x in target), name
            assert max(abs(x) for x in target) > 1e-5, name
        print("Checking Gabor", name, configuration, arguments, flush=True)
        return check_image_render(shaders, flags, mode, width, height, target,
                                  tolerance=tolerance)

    aliases = []
    for periodic in (False, True):
        for alias, canonical in (("noise", "uperlin"), ("snoise", "perlin")):
            for dimension in range(1, 5):
                for triple in (False, True):
                    kind = "color" if triple else "float"
                    a, b = call(alias, dimension, periodic), call(canonical, dimension, periodic)
                    aliases.append(f"{{{kind} a={a},b={b};"
                                   + ("d=a[0]-b[0];}" if triple else "d=a-b;}"))
    body = setup + "int probe=(int(8*u+.5)+9*int(4*v+.5))%32;float d=0;"
    for index, expression in enumerate(aliases):
        body += f"if(probe=={index})" + expression
    compile_source("gabor_aliases", body + "Cout=color(d,Dx(d),Dy(d));")
    for config in (configurations[1], configurations[3]):
        render("gabor_aliases", config, [0]*(width*height*3), None)

    for periodic in (False, True):
        for dimension in range(1, 5):
            for triple in (False, True):
                for derivatives in (False, True):
                    name = f"gabor_{int(periodic)}_{dimension}_{int(triple)}_{int(derivatives)}"
                    kind = "color" if triple else "float"
                    body = setup + f"{kind} q={call('gabor', dimension, periodic)};"
                    if derivatives:
                        body += ("float n=q[int(7*u+3*v)%3];" if triple else "float n=q;")
                        body += report
                    else:
                        body += "Cout=color(q);"
                    compile_source(name, body)
                    images = render(name)
                    if derivatives:
                        assert max(abs(x) for x in images[1][1::3]) > 1e-5
                    if not periodic and dimension == 3 and triple and derivatives:
                        for config in configurations[1:]:
                            render(name, config)

    defaults = ',"anisotropic",0,"do_filter",1,"direction",vector(1,0,0),"bandwidth",1,"impulses",16'
    relationships = []
    for periodic in (False, True):
        for dimension in range(1, 5):
            scalar = call("gabor", dimension, periodic)
            # Every Gabor color uses the scalar seed in channel zero.
            relationships.append(f"{{float a={scalar};color b={scalar};d=a-b[0];}}")
        for dimension, embedded in ((1, "point(p[0],0,0)"), (2, "point(p[0],p[1],0)")):
            first = call("gabor", dimension, periodic)
            periods = ",point(3,1,1)" if dimension == 1 else ",point(3,4,1)"
            second = (("pnoise" if periodic else "noise") + f'("gabor",{embedded}'
                      + (periods if periodic else "") + ")")
            relationships.append(f"{{color a={first},b={second};d=a[0]-b[0];}}")
        first, second = call("gabor", 3, periodic), call("gabor", 4, periodic)
        relationships.append(f"{{color a={first},b={second};d=a[0]-b[0];}}")
        explicit = call("gabor", 3, periodic, defaults)
        relationships.append(f"{{color a={first},b={explicit};d=a[0]-b[0];}}")
    body = setup + "int probe=(int(8*u+.5)+9*int(4*v+.5))%16;float d=0;"
    for index, expression in enumerate(relationships):
        body += f"if(probe=={index})" + expression
    compile_source("gabor_relationships", body + "Cout=color(d,Dx(d),Dy(d));")
    for config in (configurations[1], configurations[3]):
        # All 16 cases must be visited, including the narrow embedding slices.
        render("gabor_relationships", config, [0]*(width*height*3), 2e-6)

    options = ('"anisotropic",anisotropic,"do_filter",filtered,'
               '"direction",direction,"bandwidth",bandwidth,"impulses",impulses')
    option_params = (params + ",int anisotropic=0,int filtered=1,"
                     "vector direction=vector(1,0,0),float bandwidth=1,float impulses=16")
    compile_source("gabor_options", setup + 'color q=noise("gabor",p,'
                   + options + ");float n=q[int(7*u+3*v)%3];" + report, option_params)
    variants = [
        ("default", []),
        ("unfiltered", ["--param", "filtered", "0"]),
        ("directional", ["--param", "anisotropic", "1", "--param", "direction", "4,2,1"]),
        ("directional-unfiltered", ["--param", "anisotropic", "1", "--param", "direction", "4,2,1",
                                   "--param", "filtered", "0"]),
        ("hybrid", ["--param", "anisotropic", "2", "--param", "direction", "2,1,.5"]),
        ("bandwidth-low", ["--param:type=float", "bandwidth", ".01"]),
        ("bandwidth-clamp-low", ["--param:type=float", "bandwidth", "-10"]),
        ("bandwidth-high", ["--param:type=float", "bandwidth", "100"]),
        ("bandwidth-clamp-high", ["--param:type=float", "bandwidth", "1e38"]),
        ("impulses-low", ["--param:type=float", "impulses", "1"]),
        ("impulses-clamp-low", ["--param:type=float", "impulses", "-2"]),
        ("impulses-high", ["--param:type=float", "impulses", "32"]),
        ("impulses-clamp-high", ["--param:type=float", "impulses", "1e38"])]
    outputs = {}
    # CPU/HIP radius constants and transcendental approximations differ.
    # High directional frequencies amplify phase rounding (measured 2.31e-5);
    # bandwidth .01 amplifies exp2 rounding (9.40e-6). Defaults stay at 2e-6.
    option_tolerances = {"directional": 3e-6, "directional-unfiltered": 3e-5,
                         "hybrid": 3e-6, "bandwidth-low": 1e-5,
                         "bandwidth-clamp-low": 1e-5}
    for label, arguments in variants:
        outputs[label] = render("gabor_options", arguments=arguments,
                                tolerance=option_tolerances.get(label, 2e-6))[1]
    for key in ("bandwidth", "impulses"):
        for end in ("low", "high"):
            compare(outputs[key+"-"+end], outputs[key+"-clamp-"+end], 0, 0)
    assert max(abs(a-b) for a, b in zip(outputs["directional"][::3],
                                       outputs["directional-unfiltered"][::3])) > .002
    assert max(abs(a-b) for a, b in zip(outputs["default"], outputs["hybrid"])) > .002

    compile_source("gabor_option_derivatives",
                   'float n=noise("gabor",point(.317,.127,.233),"bandwidth",1+u);' + report)
    for image in render("gabor_option_derivatives"):
        assert image[1::3] == image[2::3] == [0]*(width*height)

    # A small affine patch keeps central differences away from the truncated
    # kernel's support boundary. Check both steps, not a filtered subset.
    for periodic in (False, True):
        name = f"gabor_finite_difference_{int(periodic)}"
        patch = ("float U=u+offset_u,V=v+offset_v;"
                 "point p=point(.317+U/1024,.127+V/2048,.233+U/4096+V/2048);")
        kind = "color" if periodic else "float"
        expression = call("gabor", 3, periodic, ',"do_filter",0')
        body = patch + f"{kind} q={expression};"
        body += "float n=q[int(7*u+3*v)%3];" if periodic else "float n=q;"
        compile_source(name, body + report)
        images = render(name)
        for step in (1/32, 1/64):
            for axis, extent, channel in (("offset_u", width, 1),
                                          ("offset_v", height, 2)):
                plus, minus = [
                    noise_cpu_image(["-O2", "--llvm_opt", "3",
                                     "--param:type=float", axis, str(sign*step),
                                     name], width, height)[::3]
                    for sign in (1, -1)]
                gradient = [(p-m)/(2*step) for p, m in zip(plus, minus)]
                assert max(abs(x) for x in gradient) > 1e-5
                for image in images:
                    actual = [x*(extent-1) for x in image[channel::3]]
                    error = max(abs(a-b) for a, b in zip(actual, gradient))
                    print("Gabor finite difference", periodic, axis, step,
                          "maximum error", error, flush=True)
                    compare(actual, gradient, 5e-6)

    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    # Gabor repeats in cells, not world units. Use its HIP constant and
    # rounded-float radius; CPU's constant differs by one ULP.
    a = f32(2/3 * f32(2.12893403886245235863))
    log_truncate = f32(math.log(f32(.02)))
    radius = f32(f32(math.sqrt(f32(-log_truncate/f32(math.pi))))/a)
    compile_source("gabor_periodic",
                   setup + "vector delta=max(vector(1),floor(vector(periods)))"
                   "*(world_units ? 1 : radius);"
                   'color a=pnoise("gabor",p,periods,"do_filter",0);'
                   'color b=pnoise("gabor",p+delta,periods,"do_filter",0);'
                   "int c=int(7*u+3*v)%3;float n=a[c]-b[c];" + report,
                   params + f",float radius={radius},int world_units=0,"
                   "point periods=point(3.8,4.5,5.1)")
    for periods in ("3.8,4.5,5.1", "0,-2,.7"):
        # The radius and translated positions each incur float rounding.
        render("gabor_periodic", target=[0]*(width*height*3), tolerance=2e-5,
               arguments=["--param", "periods", periods])
    control = render("gabor_periodic", tolerance=2e-5,
                     arguments=["--param", "world_units", "1"])
    assert max(abs(x) for x in control[1][::3]) > .01

    compile_source("gabor_input", setup + "value=p;", params + ",output point value=0")
    compile_source("gabor_connected", 'float n=noise("gabor",value);' + report,
                   "point value=0")
    connected = connected_group("gabor_connected", producer="gabor_input")
    target = noise_cpu_image(["-O2", "--llvm_opt", "3"] + connected, width, height)
    check_image_render(connected, ["-O2", "--llvm_opt", "3"], configurations[3][2],
                       width, height, target, tolerance=2e-6)

    compile_source("gabor_invalid", setup
                   + 'if(fault==0)p[0]=special;'
                     'if(fault==1)p=point(u*special,v*special,.233);'
                     'vector d=fault==2 ? vector(special,0,0) : vector(1,0,0);'
                     'float b=fault==3 ? special : 1;'
                     'float i=fault==4 ? special : 16;'
                     'point period=fault==5 ? point(special) : point(3,4,5);'
                     'if(fault==5)Cout=pnoise("gabor",p,period,"direction",d,"bandwidth",b,"impulses",i);'
                     'else Cout=noise("gabor",p,"direction",d,"bandwidth",b,"impulses",i);',
                   params + ",int fault=0,float special=0")
    for fault in range(6):
        special_values = ["nan", "inf", "-inf"]
        if fault in (0, 1, 2, 5):
            special_values.append("1e12" if fault == 5 else "1e38")
        for special in special_values:
            image = root / "gabor-invalid.pfm"
            if image.exists():
                image.unlink()
            output = run(["--hart", "-v", "--hart-no-cache", "-O0",
                          "--llvm_opt", "10", "-g", str(width), str(height),
                          "--param", "fault", str(fault), "--param:type=float",
                          "special", special, "-o", "Cout", str(image),
                          "gabor_invalid"],
                         "invalid noise arguments", error_after_launch=True)
            assert "error bits 1024" in output and not image.exists(), output

    for index, expression in enumerate((
            'noise("gabor",p,"unknown",1)',
            'noise("gabor",p,"bandwidth",vector(1))',
            'noise("gabor",p,"do_filter",.5)',
            'noise("perlin",p,"bandwidth",1)',
            'noise("unknown",p)')):
        name = f"gabor_rejected_{index}"
        compile_source(name, setup + f"if(u<0)Cout={expression};")
        for shaders in ([name], ["--shader", name, "unused", "--shader", "hart_first", "last"]):
            run(["--hart", "-v"] + shaders, "HART:")


def check_color_suite():
    # Double-precision solution from source primaries and its D65 y=.3291.
    to_xyz = ((.412135323427, .357675002654, .180356796374),
              (.212507276142, .715350005308, .072142718550),
              (.019318843286, .119225000885, .949879127571))
    from_xyz = ((3.242978965321, -1.538336175857, -.498919840819),
                (-.968997952917, 1.875491982259, .041544524053),
                (.055668324368, -.204117189350, 1.057698162996))

    def matrix(m, q):
        return tuple(sum(a*b for a, b in zip(row, q)) for row in m)

    def convert(space, inverse, q):
        if space in ("rgb", "RGB", "linear", "Rec709"):
            return q
        if space == "XYZ":
            return matrix(from_xyz if inverse else to_xyz, q)
        if space == "YIQ":
            return matrix(((1, .9557, .6199), (1, -.2716, -.6469),
                           (1, -1.1082, 1.7051)) if inverse
                          else ((.299, .587, .114), (.596, -.275, -.321),
                                (.212, -.523, .311)), q)
        if space == "xyY":
            if inverse:
                x, y, luminance = q
                return matrix(from_xyz, (x*luminance/y, luminance,
                                         (1-x-y)*luminance/y))
            x, y, z = matrix(to_xyz, q)
            return x/(x+y+z), y/(x+y+z), y
        if space == "hsv":
            return (colorsys.hsv_to_rgb if inverse else colorsys.rgb_to_hsv)(*q)
        if space == "hsl":
            if inverse:
                return colorsys.hls_to_rgb(q[0], q[2], q[1])
            h, l, s = colorsys.rgb_to_hls(*q)
            return h, s, l
        assert space == "sRGB", space
        if inverse:
            return tuple(x/12.92 if x <= .04045 else ((x+.055)/1.055)**2.4
                         for x in q)
        return tuple(12.92*x if x <= .0031308 else 1.055*x**(1/2.4)-.055
                     for x in q)

    width, height = 9, 5
    configurations = [("-O0", "10", []),
                      ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"])]
    inputs = "color(.05+.05*u,.25+.1*v,.7+.05*u+.1*v)"

    def coordinates(u, v):
        return (.05+.05*u, .25+.1*v, .7+.05*u+.1*v)

    gradients = ((.05/(width-1), 0, .05/(width-1)),
                 (0, .1/(height-1), .1/(height-1)))

    def sampled(evaluate, u, v, zero_derivatives=False):
        q = coordinates(u, v)
        component = int(7*u+3*v) % 3
        answer = [evaluate(q)[component]]
        if zero_derivatives:
            return answer + [0, 0]
        for gradient in gradients:
            h = 1e-4
            plus = evaluate(tuple(x+h*d for x, d in zip(q, gradient)))
            minus = evaluate(tuple(x-h*d for x, d in zip(q, gradient)))
            answer.append((plus[component]-minus[component])/(2*h))
        return answer

    report = ("int c=int(7*u+3*v)%3;float component_value=result[c];"
              "Cout=color(component_value,Dx(component_value),Dy(component_value));")
    cases = []
    for space in ("XYZ", "xyY", "YIQ", "hsv", "hsl", "sRGB", "rgb", "RGB", "linear", "Rec709"):
        for inverse in ((False, True) if space not in ("rgb", "RGB", "linear", "Rec709")
                        else (True,)):
            expression = (f'transformc("{space}", "rgb", q)' if inverse
                          else f'transformc("rgb", "{space}", q)')
            cases.append((expression, lambda q, s=space, inv=inverse: convert(s, inv, q), False))
    cases.append(("color(luminance(q))",
                  lambda q: (sum(a*b for a, b in zip(to_xyz[1], q)),)*3, False))
    for space in ("XYZ", "xyY", "YIQ", "hsv", "hsl", "rgb", "RGB", "Rec709"):
        cases.append((f'color("{space}",q[0],q[1],q[2])',
                      lambda q, s=space: convert(s, True, q), True))
    for index, (expression, evaluate, zero_derivatives) in enumerate(cases):
        name = f"color_math_{index}"
        source = root / (name + ".osl")
        source.write_text(
            f"shader {name}(output color Cout=0) {{color q={inputs};"
            f"color result={expression};" + report + "}",
            encoding="ascii")
        compile_fixture(source)
        target = reference(width, height, lambda u, v:
                           sampled(evaluate, u, v, zero_derivatives))
        for osl_opt, llvm_opt, mode in configurations:
            print("Checking color", expression, osl_opt, mode, flush=True)
            check_image_render([name], [osl_opt, "--llvm_opt", llvm_opt],
                               mode, width, height, target, tolerance=2e-6)

    for name, source in (
        ("color_inputs", f"shader color_inputs(output color value=0) {{value={inputs};}}"),
        ("color_connected", "shader color_connected(color value=0, output color Cout=0) {"
         'color result=transformc("rgb","XYZ",value);' + report + "}"),
    ):
        path = root / (name + ".osl")
        path.write_text(source, encoding="ascii")
        compile_fixture(path)
    target = reference(width, height,
                       lambda u, v: sampled(lambda q: matrix(to_xyz, q), u, v))
    for osl_opt, llvm_opt, mode in configurations + [
            ("-O2", "3", []), ("-O2", "3", ["--hart-fused"])]:
        check_image_render(connected_group("color_connected", producer="color_inputs"),
                           [osl_opt, "--llvm_opt", llvm_opt], mode,
                           width, height, target, tolerance=2e-6)

    for index, expression in enumerate((
            'transformc("not-a-built-in","rgb",color(u,v,1))',
            'transformc("rgb","not-a-built-in",color(u,v,1))',
            'transformc("not-a-built-in","not-a-built-in",color(u,v,1))',
            'color("not-a-built-in",u,v,1)')):
        name = f"color_rejected_{index}"
        source = root / (name + ".osl")
        source.write_text(f"shader {name}(output color Cout=0) {{"
                          f"if(u<0) Cout={expression};}}", encoding="ascii")
        compile_fixture(source)
        image = root / (name + ".pfm")
        for shaders in ([name], ["--shader", name, "unused", "--shader", "hart_first", "last"]):
            run(["--hart", "-v", "-o", "Cout", str(image)] + shaders,
                "not-a-built-in")
            assert not image.exists()

    for inverse in (False, True):
        name = "color_transfer_" + str(int(inverse))
        expression = ('transformc("sRGB","rgb",q)' if inverse
                      else 'transformc("rgb","sRGB",q)')
        source = root / (name + ".osl")
        source.write_text(f"shader {name}(output color Cout=0) {{"
                          "color q=color(-.01+.08*u,.001+.006*v,.3);"
                          f"color result={expression};" + report + "}",
                          encoding="ascii")
        compile_fixture(source)

        def expected_transfer(u, v):
            q = (-.01+.08*u, .001+.006*v, .3)
            c = int(7*u+3*v) % 3
            x = q[c]
            if inverse:
                slope = (1/12.92 if x <= .04045
                         else 2.4/1.055*((x+.055)/1.055)**1.4)
            else:
                slope = (12.92 if x <= .0031308
                         else 1.055/2.4*x**(1/2.4-1))
            return (convert("sRGB", inverse, q)[c],
                    slope*.08/(width-1) if c == 0 else 0,
                    slope*.006/(height-1) if c == 1 else 0)

        target = reference(width, height, expected_transfer)
        for osl_opt, llvm_opt, mode in configurations:
            check_image_render([name], [osl_opt, "--llvm_opt", llvm_opt],
                               mode, width, height, target, tolerance=2e-6)

    cie = {0: (.0014, 0, .0065), 1: (.0022, .0001, .0105),
           2: (.0042, .0001, .0201), 23: (.0147, .2586, .3533),
           24: (.0049, .3230, .2720), 25: (.0024, .4073, .2123),
           26: (.0093, .5030, .1582), 78: (.0001, 0, 0),
           79: (.0001, 0, 0), 80: (0, 0, 0)}

    def wavelength(nm):
        if not 375 < nm < 780:
            return (0, 0, 0)
        position = (nm-380)/5
        index = int(position)
        t = position-index
        xyz = tuple(a+(b-a)*t for a, b in zip(cie[index], cie[index+1]))
        return tuple(max(0, c/2.52) for c in matrix(from_xyz, xyz))

    for index, (expression, function) in enumerate((
            ("375+10*u", lambda u, v: 375+10*u),
            ("498+10*u", lambda u, v: 498+10*u),
            ("770+20*u", lambda u, v: 770+20*u))):
        name = f"color_wavelength_{index}"
        source = root / (name + ".osl")
        source.write_text(f"shader {name}(output color Cout=0) {{"
                          f"color result=wavelength_color({expression});"
                          + report + "}", encoding="ascii")
        compile_fixture(source)
        target = reference(width, height, lambda u, v:
                           (wavelength(function(u, v))[int(7*u+3*v) % 3], 0, 0))
        for osl_opt, llvm_opt, mode in configurations:
            check_image_render([name], [osl_opt, "--llvm_opt", llvm_opt],
                               mode, width, height, target, tolerance=2e-6)

    # Share only the CIE sampling data, not the shader's LUT or computation.
    table_source = (fixtures.parents[1] / "src" / "liboslexec" / "opcolor_impl.h").read_text()
    table_match = re.search(r"^\{\s*([\d.,\s]+)\};", table_source, re.M)
    assert table_match, "Missing CIE color-matching data"
    cie_samples = [float(x) for x in table_match.group(1).replace("\n", "").split(",")]
    assert len(cie_samples) == 81*3

    def blackbody(t):
        if t < 800:
            return (struct.unpack("<f", struct.pack("<f", 1e-12))[0], 0, 0)
        xyz = [0.0]*3
        for i in range(81):
            wavelength = (380+5*i)*1e-9
            energy = 3.74183e-16/(wavelength**5
                                 * math.expm1(.014388/(wavelength*t)))*5e-9
            for c in range(3):
                xyz[c] += energy*cie_samples[3*i+c]
        return tuple(max(0, c)/1e6 for c in matrix(from_xyz, xyz))

    for index, (expression, temperature) in enumerate((
            ("-1000+1000*u", lambda u, v: -1000+1000*u),
            ("790+40*u", lambda u, v: 790+40*u),
            ("3000+4000*u+500*v", lambda u, v: 3000+4000*u+500*v),
            ("11990+30*u", lambda u, v: 11990+30*u),
            ("16000+4000*u", lambda u, v: 16000+4000*u))):
        name = f"color_blackbody_{index}"
        source = root / (name + ".osl")
        source.write_text(f"shader {name}(output color Cout=0) {{"
                          f"color result=blackbody({expression})/1000000;"
                          + report + "}", encoding="ascii")
        compile_fixture(source)
        target = reference(width, height, lambda u, v:
                           (blackbody(temperature(u, v))[int(7*u+3*v) % 3], 0, 0))
        for osl_opt, llvm_opt, mode in configurations:
            print("Checking blackbody", expression, osl_opt, mode, flush=True)
            flags = [osl_opt, "--llvm_opt", llvm_opt]
            # LUT interpolation is approximate; above the LUT, retain a tight
            # GPU integration bound and account separately for CPU fast_expm1.
            images = check_image_render(
                [name], flags, mode, width, height, target,
                tolerance=None if index == 0 else (2e-12 if index == 1 else 2e-6),
                value_relative_tolerance=2e-6 if index == 4 else 4e-5,
                cpu_value_relative_tolerance=1.3e-5 if index == 4 else 4e-5)
            for image in images:
                assert image[1::3] == image[2::3] == [0]*(width*height)

    source = root / "color_special.osl"
    source.write_text(
        "shader color_special(float special=0, int kind=0, output color Cout=0) {"
        "float x=special+u; color result=kind ? blackbody(x) : wavelength_color(x);"
        "Cout=color(isnan(result[0]),isinf(result[0]),isfinite(result[0]));}",
        encoding="ascii")
    compile_fixture(source)
    for kind in (0, 1):
        for special in ("nan", "inf", "-inf", "1e38", "-1e38"):
            shaders = ["--param:type=float", "special", special,
                       "--param", "kind", str(kind), "color_special"]
            flags = ["-O0", "--llvm_opt", "10"]
            gpu_nan = kind == 1 and special in ("nan", "inf", "1e38")
            cpu_nan = kind == 1 and special in ("inf", "1e38")
            target = [int(gpu_nan), 0, int(not gpu_nan)]*(width*height)
            cpu_target = [int(cpu_nan), 0, int(not cpu_nan)]*(width*height)
            check_image_render(shaders, flags, [], width, height, target,
                               cpu_expected=cpu_target)


def check_numeric_math_suite():
    width, height = 17, 3
    unary = [
        ("tan", math.tan, lambda x: 1/math.cos(x)**2, "signed"),
        ("asin", math.asin, lambda x: 1/math.sqrt(1-x*x), "signed"),
        ("acos", math.acos, lambda x: -1/math.sqrt(1-x*x), "signed"),
        ("atan", math.atan, lambda x: 1/(1+x*x), "signed"),
        ("sinh", math.sinh, math.cosh, "signed"),
        ("cosh", math.cosh, math.sinh, "signed"),
        ("tanh", math.tanh, lambda x: 1/math.cosh(x)**2, "signed"),
        ("log", math.log, lambda x: 1/x, "positive"),
        ("log2", math.log2, lambda x: 1/(math.log(2)*x), "positive"),
        ("log10", math.log10, lambda x: 1/(math.log(10)*x), "positive"),
        ("exp", math.exp, math.exp, "signed"),
        ("exp2", lambda x: 2**x, lambda x: math.log(2)*2**x, "signed"),
        ("expm1", math.expm1, math.exp, "signed"),
        ("erf", math.erf, lambda x: 2/math.sqrt(math.pi)*math.exp(-x*x),
         "signed"),
        ("erfc", math.erfc, lambda x: -2/math.sqrt(math.pi)*math.exp(-x*x),
         "signed"),
        ("cbrt", lambda x: math.copysign(abs(x)**(1/3), x),
         lambda x: 1/(3*abs(x)**(2/3)), "negative"),
        ("inversesqrt", lambda x: 1/math.sqrt(x),
         lambda x: -0.5/x**1.5, "positive"),
        ("logb", lambda x: math.floor(math.log2(abs(x))), lambda x: 0,
         "positive"),
        ("round", lambda x: math.copysign(math.floor(abs(x)+0.5), x),
         lambda x: 0, "signed"),
        ("trunc", math.trunc, lambda x: 0, "signed"),
        ("sign", lambda x: (x > 0)-(x < 0), lambda x: 0, "signed"),
        ("fabs", abs, lambda x: -1 if x < 0 else 1, "signed"),
        # Preserve OSL's existing atan2 duals' reversed derivative signs.
        ("atan2", lambda x: math.atan2(x, 1.25),
         lambda x: -1.25/(x*x+1.25**2), "signed"),
        ("sincos", lambda x: math.sin(x)+math.cos(x),
         lambda x: math.cos(x)-math.sin(x), "signed"),
        ("degrees", math.degrees, lambda x: 180/math.pi, "signed"),
        ("radians", math.radians, lambda x: math.pi/180, "signed"),
    ]
    cases = []
    # CPU inverse-trig/tanh limits use OIIO's documented bounds; exp-family
    # limits cover measured errors on this grid. HIP and derivatives stay 2e-6.
    cpu_value_tolerances = {
        "asin": 4.6e-5, "acos": 4.6e-5, "atan": 1e-5, "atan2": 1e-5,
        "cosh": 5e-6, "tanh": 3.2e-6, "exp": 1.2e-5, "exp2": 1.1e-5,
        "expm1": 1.2e-5,
    }
    for name, value, derivative, domain in unary:
        for triple in ((False,) if name in ("erf", "erfc") else (False, True)):
            shader = "numeric_" + name + ("_triple" if triple else "_scalar")
            base = ("u-0.5+0.25*v" if domain == "signed" else
                    "0.5+u+0.25*v" if domain == "positive" else
                    "-0.5-u-0.25*v")
            kind = "color" if triple else "float"
            argument = "color(base-0.125,base,base+0.125)" if triple else "base"
            expression = (f"atan2(x,{kind}(1.25))" if name == "atan2" else
                          f"{name}(x)")
            operation = (f"{kind} c=0; sincos(x,x,c); {kind} result=x+c;"
                         if name == "sincos" else
                         f"{kind} result={expression};")
            selected = "result[int(2*v)]" if triple else "result"
            source = root / (shader + ".osl")
            source.write_text(
                f"shader {shader}(output color Cout=0) {{"
                f"float base={base}; {kind} x={argument}; {operation}"
                f"float q={selected}; Cout=color(q,Dx(q),Dy(q));}}",
                encoding="ascii")
            compile_fixture(source)

            def expected(u, v, triple=triple, domain=domain, value=value,
                         derivative=derivative):
                sign = -1 if domain == "negative" else 1
                x = (u-0.5+0.25*v if domain == "signed"
                     else sign*(0.5+u+0.25*v))
                if triple:
                    x += (int(2*v)-1)*0.125
                d = derivative(x)
                return value(x), d*sign/(width-1), d*sign*0.25/(height-1)

            cases.append(([shader], reference(width, height, expected),
                          name in ("asin", "log", "sincos") and triple,
                          cpu_value_tolerances.get(name, 2e-6)))

    for name in ("asin", "acos", "inversesqrt", "cbrt"):
        for triple in (False, True):
            shader = "numeric_edges_" + name + ("_triple" if triple else "_scalar")
            kind = "color" if triple else "float"
            argument = "color(x-0.25,x,x+0.25)" if triple else "x"
            selected = "result[int(2*v)]" if triple else "result"
            source = root / (shader + ".osl")
            source.write_text(
                f"shader {shader}(output color Cout=0) {{float x=4*u-2;"
                f"{kind} result={name}({argument});float q={selected};"
                "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
            compile_fixture(source)

            def expected(u, v, name=name, triple=triple):
                x = 4*u-2 + ((int(2*v)-1)*0.25 if triple else 0)
                if name == "inversesqrt":
                    return ((1/math.sqrt(x), -2/x**1.5/(width-1), 0)
                            if x > 0 else (0, 0, 0))
                if name == "cbrt":
                    return ((math.copysign(abs(x)**(1/3), x),
                             4/(3*abs(x)**(2/3))/(width-1), 0)
                            if x != 0 else (0, 0, 0))
                value = (math.asin if name == "asin" else math.acos)(
                    max(-1, min(1, x)))
                derivative = ((4 if name == "asin" else -4)
                              / math.sqrt(1-x*x)/(width-1) if abs(x) < 1 else 0)
                return value, derivative, 0

            cases.append(([shader], reference(width, height, expected), triple,
                          cpu_value_tolerances.get(name, 2e-6)))

    geometry = [
        ("cross", "vector result=cross(vector(u,v,u*v),vector(1,2,3));",
         lambda u, v: ((3*v-2*u*v, u*v-3*u, 2*u-v),
                       (-2*v, v-3, 2), (3-2*u, u, -1))),
        ("distance", "float q=distance(point(u,v,0),point(1+u,2-v,0));",
         lambda u, v: (math.sqrt(1+(2*v-2)**2), 0,
                       (4*v-4)/math.sqrt(1+(2*v-2)**2))),
        ("reflect", "vector result=reflect(vector(u,v,-1),vector(0,0,1));",
         lambda u, v: ((u,v,1), (1,0,0), (0,1,0))),
        ("refract", "vector result=refract(vector(u,v,-1),vector(0,0,1),0.5);",
         lambda u, v: ((0.5*u,0.5*v,-1), (0.5,0,0), (0,0.5,0))),
        ("refract_tir",
         "vector result=refract(vector(0.6,0,-0.8),vector(0,0,1),0.5+1.5*u);",
         lambda u, v: (((0.6*(0.5+1.5*u), 0,
                         -math.sqrt(1-0.36*(0.5+1.5*u)**2)),
                        (0.9, 0, 0.54*(0.5+1.5*u)
                         / math.sqrt(1-0.36*(0.5+1.5*u)**2)), (0,0,0))
                       if 1-0.36*(0.5+1.5*u)**2 > 0
                       else ((0,0,0), (0,0,0), (0,0,0)))),
        ("faceforward",
         "vector result=faceforward(vector(u,v,1),vector(0,0,u-0.5),"
         "vector(0,0,1));",
         lambda u, v: tuple(tuple(a*(-1 if u > 0.5 else 1) for a in vec)
                            for vec in ((u,v,1), (1,0,0), (0,1,0)))),
        ("rotate", "point result=rotate(point(u,v,0.25),1.5707963267948966,"
         "point(0),point(0,0,1));",
         lambda u, v: ((-v,u,0.25), (0,1,0), (-1,0,0))),
        ("calculatenormal",
         "normal result=calculatenormal(point(2*u,3*v,u*v));",
         lambda u, v: ((-3*v/32,-2*u/32,6/32), (0,0,0), (0,0,0))),
        ("area", "float q=area(point(2*u,3*v,u*v));",
         lambda u, v: (math.sqrt(9*v*v+4*u*u+36)/32, 0, 0)),
    ]
    for name, operation, evaluate in geometry:
        shader = "numeric_" + name
        scalar = name in ("area", "distance")
        source = root / (shader + ".osl")
        source.write_text(
            f"shader {shader}(output color Cout=0) {{{operation}"
            + ("" if scalar else "float q=result[int(2*v)];")
            + "Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
        compile_fixture(source)

        def expected(u, v, scalar=scalar, evaluate=evaluate):
            values = evaluate(u, v)
            if not scalar:
                values = [part[int(2*v)] for part in values]
            return values[0], values[1]/(width-1), values[2]/(height-1)

        cases.append(([shader], reference(width, height, expected), True, 2e-6))

    source = root / "numeric_sincos_alias.osl"
    source.write_text(
        "shader numeric_sincos_alias(output color Cout=0) {"
        "color x=color(u-0.5,u+0.25*v,u-v),c=x,s=0; sincos(c,s,c);"
        "color shared=0; sincos(x,shared,shared);"
        "color all=x; sincos(all,all,all);"
        "color us=0,uc=0; sincos(color(time+0.25),us,uc);"
        "color result=s+2*c+3*shared+5*all+us+uc;"
        "float q=result[int(2*v)];Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
    compile_fixture(source)

    def alias_expected(u, v):
        x, dx, dy = ((u-0.5, 1, 0), (u+0.25*v, 1, 0.25), (u-v, 1, -1))[int(2*v)]
        derivative = math.cos(x)-10*math.sin(x)
        return (math.sin(x)+10*math.cos(x)+math.sin(0.25)+math.cos(0.25),
                derivative*dx/(width-1), derivative*dy/(height-1))

    cases.append(([source.stem], reference(width, height, alias_expected),
                  True, 2e-6))

    producer = root / "numeric_connected_source.osl"
    producer.write_text(
        "shader numeric_connected_source(output color value=0) {"
        "color x=color(0.5+u,0.75+v,1+u*v);value=log(x)+sqrt(x);}",
        encoding="ascii")
    consumer = root / "numeric_connected_sink.osl"
    consumer.write_text(
        "shader numeric_connected_sink(color value=0,output color Cout=0) {"
        "float q=value[int(2*v)];Cout=color(q,Dx(q),Dy(q));}", encoding="ascii")
    compile_fixture(producer)
    compile_fixture(consumer)

    def connected_expected(u, v):
        x, dx, dy = ((0.5+u, 1, 0), (0.75+v, 0, 1), (1+u*v, v, u))[int(2*v)]
        derivative = 1/x+0.5/math.sqrt(x)
        return (math.log(x)+math.sqrt(x), derivative*dx/(width-1),
                derivative*dy/(height-1))

    connected = connected_group(consumer.stem, producer=producer.stem)
    cases.append((connected, reference(width, height, connected_expected),
                  True, 2e-6))

    for shaders, expected, representative, cpu_value_tolerance in cases:
        configurations = [("-O2", "3", [])]
        if representative:
            configurations += [
                ("-O0", "10", []),
                ("-O2", "3", ["--hart-fused"]),
                ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"]),
            ]
        for osl_opt, llvm_opt, mode in configurations:
            print("Checking numeric math", shaders, osl_opt, llvm_opt, mode,
                  flush=True)
            check_image_render(shaders, [osl_opt, "--llvm_opt", llvm_opt],
                               mode, width, height, expected, tolerance=2e-6,
                               cpu_value_tolerance=cpu_value_tolerance)

    source = root / "numeric_classify.osl"
    source.write_text(
        "shader numeric_classify(float special=0, output color Cout=0) {"
        "float x=u<0.5 ? special : 2*u-1;"
        "Cout=color(isnan(x),isinf(x),isfinite(x));}", encoding="ascii")
    compile_fixture(source)
    for osl_opt, llvm_opt, mode in [
            ("-O0", "10", []),
            ("-O2", "3", ["--hart-fused", "--hart-local-groupdata", "4096"])]:
        for text, special in (("0", 0), ("-0", -0.0), ("inf", math.inf),
                              ("-inf", -math.inf), ("nan", math.nan)):
            def expected(u, v, special=special):
                x = special if u < 0.5 else 2*u-1
                return int(math.isnan(x)), int(math.isinf(x)), int(math.isfinite(x))
            check_image_render(
                ["--param:type=float", "special", text, "numeric_classify"],
                [osl_opt, "--llvm_opt", llvm_opt], mode, 5, 3,
                reference(5, 3, expected))


try:
    if args.fused_benchmark:
        check_fused_benchmark()
        print("Generated HART benchmark correctness and timing records passed; "
              "no performance thresholds")
        raise SystemExit(0)

    for source in fixtures.glob("hart_*.osl"):
        compile_fixture(source)

    base = ["--hart", "hart_first"]
    for option in (
        ["--batched"],
        ["--use_rs_bitcode"],
        ["--no-output-placement"], ["--shadeimage"],
        ["--scaleuv", "2", "2"], ["--offsetuv", "1", "1"],
        ["--options", "optimize=0"], ["--saveptx"],
    ):
        run(base + option, "unsupported option")
    for option in (["--hart-entry", "__raygen__other"],
                   ["--hart-callable-module", "other.bc"]):
        run(base + option, "cannot be mixed")
    run(base + ["--hart-module", "other.bc"], "not OSL shaders")
    run(["--hart-fused", "-v", "hart_first"], "require --hart")
    run(["--hart", "--hart-fused", "-v", "--hart-module", "other.bc"],
        "generated")
    run(["--hart-local-groupdata", "0", "-v", "hart_first"], "require --hart")
    run(base + ["--hart-local-groupdata", "0", "-v"], "--hart-fused")
    run(["--hart", "--hart-local-groupdata", "0", "-v",
         "--hart-module", "other.bc"], "generated")
    run(["--hart", "--hart-fused", "--hart-local-groupdata", "0", "-v",
         "--hart-module", "other.bc"], "generated")
    for budget in ("-1", "1x", "2147483648", "+1", "1.0", "0x1", "", " 1", "1 "):
        for fused in ([], ["--hart-fused"]):
            run(["--hart", "-v"] + fused
                + ["--hart-local-groupdata", budget, "hart_first"],
                "Invalid --hart-local-groupdata" if fused else
                "requires --hart-fused")
    run(base + ["-g", "0", "1"], "must be positive")
    run(base + ["-g", "46341", "46341"], "int shade-index range")
    run(base + ["--iters", "0"], "must be positive")
    run(base + ["--hart-device", "-1"], "must be nonnegative")
    run(base + ["--hart-device", "not-an-integer"], "error")
    run(base + ["-d", "invalid"], "output format")
    run(["--hart", "--param", "scale", "2"], "requires an OSL shader")
    for option in ("TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
        run(base, "does not support " + option, {option: "1"})

    if args.loops:
        for operation in ("break", "continue", "dowhile"):
            shader = "hart_loop_" + operation
            evaluate = lambda u, v: tuple(
                c * (2 if operation == "dowhile" else (0 if u > v else 4))
                for c in (u, v, 0))
            for optimize in ("10", "3"):
                check_render(["--llvm_opt", optimize, "-O0", shader], 5, 3,
                             reference(5, 3, evaluate))
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
                    compare(cpu, expected, 6e-6)
                    compare(gpu, expected, 2e-6)
                    compare(gpu, cpu, 6e-6)

    if args.control_flow:
        check_control_flow_suite()

    if args.aggregates:
        check_aggregate_suite()

    if args.strings:
        check_string_suite()

    if args.selectors:
        check_selector_suite()

    if args.diagnostics:
        check_diagnostic_suite()

    if args.interactive_userdata:
        check_interactive_userdata_suite()

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
                    compare(cpu, expected, 6e-6)
                    compare(gpu, expected, 2e-6)
                    compare(gpu, cpu, 6e-6)

    if args.surface:
        for shader, error in (
            ("hart_surface_incident", "unsupported shader global 'Ps'"),
        ):
            run(["--hart", "-v", shader], error)
            run(["--hart", "-v", "--shader", shader, "producer",
                 "--shader", "hart_first", "consumer"], error)
        check_render(["--param", "enabled", "1", "hart_space_rejected"], 3, 2,
                     reference(3, 2, lambda u, v: (u, 2*v, 1)))
        connected = connected_group("hart_surface_consumer",
                                    producer="hart_surface_producer")
        for optimize in ("10", "3"):
            flags = ["--llvm_opt", optimize]
            check_render(flags + ["--param", "enable", "1", "hart_surface_write"],
                         3, 2, reference(3, 2, lambda u, v: (u, v, 1)))
            for width, height in ((1, 1), (3, 2), (37, 5)):
                check_render(flags + connected, width, height,
                             reference(width, height,
                                       lambda u, v: surface_result(u, v, width, height)))
            check_render(flags + ["hart_surface_globals"], 3, 2,
                         reference(3, 2, lambda u, v: surface_globals(u, v, 3, 2)))
            for planar in (0, 1):
                def values(u, v):
                    length = math.sqrt(u * u + v * v + (1 - planar))
                    return (1 - planar, length, u / length if length else 0)
                check_render(flags + ["--param", "planar", str(planar),
                                      "hart_surface_values"], 3, 2, reference(3, 2, values))
            for operation in (0, 1):
                planar_group = ["--param", "planar", "1"] + connected_group(
                    "hart_surface_consumer", ["--param", "operation", str(operation)],
                    producer="hart_surface_producer")
                check_render(flags + planar_group, 3, 2, reference(
                    3, 2, lambda u, v: surface_result(u, v, 3, 2, operation, planar=True)))
            cases = [
                (["--param", "field", str(field), "hart_surface_globals"],
                 lambda u, v, field=field: surface_globals(u, v, 3, 2, field))
                for field in range(8)
            ]
            for operation in range(5):
                parameters = ["--param", "operation", str(operation)]
                cases.append((
                    connected_group("hart_surface_consumer", parameters,
                                    producer="hart_surface_producer"),
                    lambda u, v, operation=operation: surface_result(u, v, 3, 2, operation),
                ))
                cases.append((
                    ["--param:type=vector", "value", "1,2,3"] + parameters
                    + ["hart_surface_consumer"],
                    lambda u, v, operation=operation: surface_result(
                        u, v, 3, 2, operation, value=(1, 2, 3)),
                ))
            for shader_args, evaluate in cases:
                text_flags = flags + ["-O0", "-g", "3", "2", "--print"]
                expected = reference(3, 2, evaluate)
                cpu = pixels(run(text_flags + shader_args), 3, 2)
                gpu = pixels(run(["--hart", "--hart-no-cache", "--warmup",
                                  "--iters", "3"] + text_flags + shader_args), 3, 2)
                compare(cpu, expected, 6e-6)
                compare(gpu, expected, 2e-6)
                compare(gpu, cpu, 6e-6)

    if args.filterwidth:
        scalar = connected_group("hart_filterwidth_consumer",
                                 producer="hart_deriv_producer")
        vector = connected_group("hart_filterwidth_vector_consumer",
                                 producer="hart_surface_producer")
        for optimize in ("10", "3"):
            flags = ["--llvm_opt", optimize]
            for width, height in ((1, 1), (3, 2), (37, 5)):
                expected = reference(
                    width, height,
                    lambda u, v: filterwidth_scalar_result(u, v, width, height))
                for shader_args in (["hart_filterwidth_scalar"], scalar):
                    check_render(flags + shader_args, width, height, expected)
                check_render(flags + vector, width, height, reference(
                    width, height,
                    lambda u, v: (1 / max(1, width - 1),
                                  2 / max(1, height - 1),
                                  math.hypot(v / max(1, width - 1),
                                             u / max(1, height - 1)))))
            cases = [
                (["-O0", "--param", "kind", str(kind), "hart_filterwidth_triple"],
                 lambda u, v, kind=kind: (
                     (0, 0, 0) if kind == 4 else
                     (0.5, 1, 0) if kind == 5 else
                     (math.hypot(0.5, 1), math.hypot(0.5, 1), math.hypot(v * 0.5, u))))
                for kind in range(6)
            ]
            cases.append((
                ["-O0", "--param:type=float", "scale", "-3"] + scalar,
                lambda u, v: filterwidth_scalar_result(u, v, 3, 2, scale=-3),
            ))
            for derivative in (1, 2):
                cases.append((
                    ["-O0"] + connected_group(
                        "hart_filterwidth_vector_consumer",
                        ["--param", "derivative", str(derivative)],
                        producer="hart_surface_producer"),
                    lambda u, v: (0, 0, 0),
                ))
            for specialize in ("-O0", "-O2"):
                for shader_args in (
                    ["--param:type=float", "value", "3", "hart_filterwidth_consumer"],
                    ["--param:type=vector", "value", "1,2,3",
                     "hart_filterwidth_vector_consumer"],
                    ["--param:type=float", "scale", "0"] + scalar,
                ):
                    cases.append(([specialize] + shader_args, lambda u, v: (0, 0, 0)))
            for shader_args, evaluate in cases:
                text_flags = flags + ["-g", "3", "2", "--print"]
                expected = reference(3, 2, evaluate)
                cpu = pixels(run(text_flags + shader_args), 3, 2)
                gpu = pixels(run(["--hart", "--hart-no-cache", "--warmup",
                                  "--iters", "3"] + text_flags + shader_args), 3, 2)
                compare(cpu, expected, 6e-6)
                compare(gpu, expected, 2e-6)
                compare(gpu, cpu, 6e-6)

    if args.noise or args.noise_families:
        for shader, error in (
            ("hart_noise_named", "unsupported noise type 'unknown'"),
            ("hart_noise_options", "noise options require gabor"),
            ("hart_noise_periodic", "unsupported noise type 'simplex'"),
        ):
            run(["--hart", "-v", shader], error)
            run(["--hart", "-v", "--shader", shader, "producer",
                 "--shader", "hart_first", "consumer"], error)
        if args.noise:
            check_noise_suite()
        else:
            for shader, error in (
                ("hart_noise_unknown", "unsupported noise type 'unknown'"),
                ("hart_noise_empty", "unsupported noise type ''"),
            ):
                run(["--hart", "-v", shader], error)
                run(["--hart", "-v", "--shader", shader, "producer",
                     "--shader", "hart_first", "consumer"], error)
            check_noise_family_suite()
            host = root / "dynamic-noise-reference.pfm"
            run(["-O0", "-t", "1", "-g", "9", "5", "-d", "float",
                 "-o", "Cout", str(host), "hart_noise_dynamic"])
            check_image_render(["hart_noise_dynamic"], ["-O0", "--llvm_opt", "10"],
                               [], 9, 5, image_pixels(host, 9, 5), tolerance=2e-6)

    if args.gabor:
        check_gabor_suite()

    if args.math:
        check_math_suite()

    if args.numeric_math:
        check_numeric_math_suite()
    if args.splines:
        check_spline_suite()
    if args.colors:
        check_color_suite()

    if args.procedural:
        check_procedural_suite()

    if args.textures:
        check_texture_suite()

    if args.texture_alpha:
        check_texture_alpha_suite()

    if args.texture_channels:
        check_texture_channel_suite()

    if args.texture_materials:
        check_texture_material_suite()

    if args.matrices:
        check_matrix_suite()

    if args.spaces:
        check_space_suite()

    if args.geometry:
        check_geometry_suite()

    if args.groups:
        check_group_suite()

    if args.topology:
        check_topology_suite()

    if args.materials:
        check_material_suite()

    if args.fused:
        check_fused_suite()

    if args.fused_local:
        check_fused_local_suite()

    if args.gpu and not (args.loops or args.control_flow or args.aggregates or args.strings or args.selectors or args.diagnostics or args.interactive_userdata or args.derivatives or args.surface or args.filterwidth
                         or args.noise or args.noise_families or args.gabor or args.math or args.numeric_math or args.splines or args.colors
                         or args.procedural or args.textures or args.texture_alpha
                         or args.texture_channels or args.texture_materials
                         or args.matrices
                         or args.spaces or args.geometry or args.groups
                         or args.topology or args.materials or args.fused
                         or args.fused_local or args.fused_benchmark):
        for shader, error in (
            ("hart_wrong_output", "RGB color"),
            ("hart_closure", "RGB color"),
            ("hart_string", "unsupported operation 'strlen'"),
            ("hart_texture", "HART: texture requires explicit closest or linear interpolation"),
        ):
            run(["--hart", "-v", shader], error)
        output = run(["--hart", "-v", "hart_missing_output"])
        assert "Launching HART grid" in output, output
        assert "HART output arena: 0 bytes" in output, output
        run(["--hart", "-v", "-o", "Cout", "null", "hart_missing_output"],
            "Unknown HART output")
        run(["--hart", "--shader", "hart_surface_incident", "unused",
             "--shader", "hart_first", "middle",
             "--shader", "hart_sine", "surface", "-v"],
            "HART: unsupported shader global 'Ps'")
        # Even an unused producer must be validated before optimization.
        for shader in ("hart_string", "hart_texture"):
            run(["--hart", "--shader", shader, "producer",
                 "--shader", "hart_sine", "consumer", "-v"],
                "HART: texture requires explicit closest or linear interpolation"
                if shader == "hart_texture" else "HART")

        connected = connected_group("hart_group_consumer")
        arithmetic = lambda u, v: (u, v, u + v)
        sine = lambda u, v: (u, v, math.sin(u + v))
        branch = lambda u, v: (u, v, math.sin(u + v) if u > v else 0)
        for shader_args, evaluate in (
            (["hart_first"], arithmetic), (["hart_sine"], sine),
            (["hart_userdata"], lambda u, v: (1, u, v)),
            (["--userdata", "value", "2.0", "hart_userdata"],
             lambda u, v: (2, u, v)),
            (["--param:interpolated=1", "value", "3.0",
              "--userdata", "value", "2.0", "hart_userdata"],
             lambda u, v: (2, u, v)),
            (["--shader", "hart_userdata", "unused",
              "--shader", "hart_sine", "surface"], sine),
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
                compare(cpu, expected, 6e-6)
                compare(gpu, expected, 2e-6)
                compare(gpu, cpu, 6e-6)
            parameter_group = ["--llvm_opt", optimize, "-O0",
                               "--param:type=float", "scale", "2"] + connected
            expected = [0.5, 0.5, math.sin(2)]
            compare(pixels(run(["--print"] + parameter_group), 1, 1),
                    expected, 6e-6)
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

if args.gabor:
    suite = "Gabor noise"
elif args.colors:
    suite = "color systems"
elif args.splines:
    suite = "splines"
elif args.numeric_math:
    suite = "numeric math"
elif args.selectors:
    suite = "hashes and dynamic noise selectors"
elif args.diagnostics:
    suite = "bounded diagnostics"
elif args.interactive_userdata:
    suite = "interpolated interactive defaults"
elif args.strings:
    suite = "string values"
elif args.aggregates:
    suite = "aggregates"
elif args.control_flow:
    suite = "control flow"
elif args.texture_materials:
    suite = "texture materials"
elif args.texture_channels:
    suite = "texture channels"
elif args.texture_alpha:
    suite = "texture alpha"
elif args.fused_benchmark:
    suite = "fused benchmark"
elif args.fused_local:
    suite = "callable-local group storage"
elif args.fused:
    suite = "split/fused callable"
elif args.noise:
    suite = "noise"
elif args.filterwidth:
    suite = "filterwidth"
else:
    suite = ("surface" if args.surface else
             ("derivative" if args.derivatives else ("loop" if args.loops else "CLI")))
print("Generated HART " + suite + " checks passed"
      + ("; staged CPU controls and in-place GPU updates/derivatives passed"
         if args.interactive_userdata else
         ("; GPU image comparisons and launch statistics passed"
         if args.fused_benchmark else
         ("; CPU/GPU numeric, image, cold-cache and repeated-launch checks passed"
          if args.gpu else ""))))
