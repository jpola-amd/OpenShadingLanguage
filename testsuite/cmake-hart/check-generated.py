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
suites.add_argument("--surface", action="store_true",
                    help="Run surface globals and vector math runtime cases")
suites.add_argument("--filterwidth", action="store_true",
                    help="Run scalar and triple filterwidth runtime cases")
suites.add_argument("--noise", action="store_true",
                    help="Run numeric Perlin noise runtime cases")
suites.add_argument("--noise-families", action="store_true",
                    help="Run periodic, cell, hash and named noise runtime cases")
suites.add_argument("--math", action="store_true",
                    help="Run scalar and triple math runtime cases")
suites.add_argument("--procedural", action="store_true",
                    help="Run connected procedural material runtime cases")
args = parser.parse_args()
if (args.loops or args.derivatives or args.surface or args.filterwidth
        or args.noise or args.noise_families or args.math
        or args.procedural) and not args.gpu:
    parser.error("Runtime suites require --gpu")
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


def compare(actual, expected, tolerance=2e-6):
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
                    compare(cpu, expected, 6e-6)
                    compare(gpu, expected, 2e-6)
                    compare(gpu, cpu, 6e-6)

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
            ("hart_surface_incident", "unsupported shader global 'I'"),
            ("hart_surface_write", "writing shader global 'N'"),
            ("hart_surface_space", "unsupported type 'string'"),
        ):
            run(["--hart", "-v", shader], error)
            run(["--hart", "-v", "--shader", shader, "producer",
                 "--shader", "hart_first", "consumer"], error)
        connected = connected_group("hart_surface_consumer",
                                    producer="hart_surface_producer")
        for optimize in ("10", "3"):
            flags = ["--llvm_opt", optimize]
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
            ("hart_noise_named", "unsupported noise type 'gabor'"),
            ("hart_noise_options", "unsupported type 'string'"),
            ("hart_noise_periodic", "unsupported noise type 'simplex'"),
        ):
            run(["--hart", "-v", shader], error)
            run(["--hart", "-v", "--shader", shader, "producer",
                 "--shader", "hart_first", "consumer"], error)
        if args.noise:
            check_noise_suite()
        else:
            for shader, error in (
                ("hart_noise_dynamic", "unsupported type 'string'"),
                ("hart_noise_unknown", "unsupported noise type 'unknown'"),
                ("hart_noise_empty", "unsupported noise type ''"),
            ):
                run(["--hart", "-v", shader], error)
                run(["--hart", "-v", "--shader", shader, "producer",
                     "--shader", "hart_first", "consumer"], error)
            check_noise_family_suite()

    if args.math:
        check_math_suite()

    if args.procedural:
        check_procedural_suite()

    if args.gpu and not (args.loops or args.derivatives or args.surface or args.filterwidth
                         or args.noise or args.noise_families or args.math
                         or args.procedural):
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

if args.noise:
    suite = "noise"
elif args.filterwidth:
    suite = "filterwidth"
else:
    suite = ("surface" if args.surface else
             ("derivative" if args.derivatives else ("loop" if args.loops else "CLI")))
print("Generated HART " + suite + " checks passed"
      + ("; CPU/GPU numeric, image, cold-cache and repeated-launch checks passed"
         if args.gpu else ""))
