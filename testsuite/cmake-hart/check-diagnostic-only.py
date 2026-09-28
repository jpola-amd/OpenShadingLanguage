# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Exercise real generated HART launches without a requested output arena."""

import argparse
from collections import Counter
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
if not args.gpu:
    parser.error("Diagnostic-only runtime checks require --gpu")

testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
testsuite = Path(__file__).resolve().parent.parent
env = os.environ.copy()
for name in ("TESTSHADE_HART", "TESTSHADE_FUSED", "TESTSHADE_OPTIX",
             "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
    env[name] = "0"
for name in ("TESTSHADE_OPT", "TESTSHADE_LLVM_OPT"):
    env.pop(name, None)
root = Path.cwd() / ("hart-diagnostic-only-" + uuid.uuid4().hex)
root.mkdir()

configurations = [
    ("-O0", "10", "split"),
    ("-O2", "3", "split"),
    ("-O2", "3", "fused"),
    ("-O2", "3", "local"),
]
grid = ["-g", "3", "2"]
values = [x * 0.5 + 2 * y for y in range(2) for x in range(3)]
rgb = [channel for y in range(2) for x in range(3)
       for channel in (x * 0.5, float(y), 0.5)]


def run(arguments, config=None, error=None, launches=0, arena=None):
    child_env = env.copy()
    flags = []
    if config is not None:
        osl, llvm, mode = config
        child_env["TESTSHADE_HART"] = "1"
        child_env["TESTSHADE_FUSED"] = "0" if mode == "split" else "1"
        flags = ["-v", "--runstats", osl, "--llvm_opt", llvm]
        if mode == "local":
            flags += ["--hart-local-groupdata", "4096"]
    result = subprocess.run(
        [testshade] + flags + arguments, cwd=root, env=child_env,
        capture_output=True, text=True, timeout=300,
    )
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, "Unexpected success:\n" + output
        assert error.lower() in output.lower(), output
    assert output.count("Launching HART grid") == launches, output
    if config is not None and launches:
        mode = "split" if config[2] == "split" else "fused"
        assert output.count("HART callable mode: " + mode) == 1, output
        if error is None:
            assert "HART synchronized launches:" in output, output
        if arena is not None:
            assert re.findall(r"HART output arena: (\d+) bytes", output) == [
                str(arena)], output
    return output


def compile_source(source, name):
    result = subprocess.run(
        [oslc, "-I" + str(testsuite.parent / "src" / "shaders"),
         "-o", str(root / (name + ".oso")), str(source.relative_to(root))],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def shader(name, parameters, body, declarations=""):
    source = root / (name + ".osl")
    source.write_text(
        declarations + f"shader {name}({parameters}) {{\n" + body + "\n}\n",
        encoding="ascii",
    )
    compile_source(source, name)


def records(output):
    # Compare complete context and payload; do not strip metadata from output.
    return [(name, Path(source).name, int(line), int(point), text)
            for name, source, line, point, text in re.findall(
                r"HART shader '([^']+)' \(([^\r\n]*):(\d+), point (\d+)\): ([^\r\n]*)",
                output)]


def no_images():
    images = [p.name for p in root.iterdir()
              if p.suffix.lower() in (".pfm", ".exr", ".tif", ".png", ".jpg")]
    assert not images, images


def rgb_pixels(output):
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)",
        output,
    )
    assert [(int(x), int(y)) for x, y, *_ in rows] == [
        (x, y) for y in range(2) for x in range(3)], output
    return [float(value) for _, _, *channels in rows for value in channels]


def pfm_pixels(path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"PF"
        assert list(map(int, stream.readline().split())) == [3, 2]
        scale = float(stream.readline())
        assert abs(scale) == 1
        data = struct.unpack(("<" if scale < 0 else ">") + "18f", stream.read())
    return list(data[9:]) + list(data[:9])


success = False
try:
    shader("hart_diagnostic_only", "int tag=7 [[int interactive=1]]", r'''
#line 100 "diagnostic_only_payload.osl"
    printf("ONLY point=%d value=%.2f tag=%d\n", int(2*u)+3*int(v), u+2*v, tag);
    warning("ONLY warning point=%d tag=%d\n", int(2*u)+3*int(v), tag);
''')
    shader("hart_diagnostic_producer", "output float signal=0", r'''
#line 100 "diagnostic_only_producer.osl"
    signal = u + 2*v;
    printf("ONLY producer %.2f\n", signal);
''')
    shader("hart_diagnostic_consumer", "float signal=0", r'''
#line 100 "diagnostic_only_consumer.osl"
    printf("ONLY consumer %.2f\n", signal);
''')
    shader("hart_diagnostic_empty", "", "")
    shader("hart_diagnostic_error", "", r'''
#line 200 "diagnostic_only_error.osl"
    printf("ONLY before-error point=%d\n", int(u));
    if (u==0) error("ONLY intentional error\n");
''')
    shader("hart_diagnostic_capacity", "int reports=0", r'''
#line 300 "diagnostic_only_capacity.osl"
    for (int i=0; i<reports; ++i) printf("ONLY record %d\n", i);
''')
    shader("hart_diagnostic_typed",
           'float Cout=4, output float scalar=0, output int number=0, '
           'output string text="not numeric", output OnlyRecord record={0}',
           r'''
#line 400 "diagnostic_only_typed.osl"
    scalar = u + 2*v;
    number = int(2*u) + 3*int(v);
    record.value = scalar + 1;
    printf("ONLY typed %.2f %d\n", scalar, number);
''', declarations="struct OnlyRecord { float value; };\n")
    shader("hart_diagnostic_rgb", "int fail=0, output color Cout=0", r'''
#line 500 "diagnostic_only_rgb.osl"
    Cout = color(u,v,.5);
    if (fail) error("ONLY output failure\n");
''')
    shader("hart_diagnostic_bad_default", "output float Cout=1", "")

    connected = [
        "--shader", "hart_diagnostic_producer", "producer",
        "--shader", "hart_diagnostic_consumer", "consumer",
        "--connect", "producer", "signal", "consumer", "signal",
    ]
    for osl, llvm in (("-O0", "10"), ("-O2", "3")):
        flags = [osl, "--llvm_opt", llvm]
        cpu = run(flags + grid + ["hart_diagnostic_only"])
        expected = [f"ONLY point={point} value={value:.2f} tag=7"
                    for point, value in enumerate(values)]
        expected += [f"ONLY warning point={point} tag=7" for point in range(6)]
        assert Counter(re.findall(r"ONLY (?:point|warning point)=[^\r\n]*", cpu)) == Counter(expected), cpu
        cpu = run(flags + ["-g", "2", "1"] + connected)
        assert Counter(re.findall(r"ONLY (?:producer|consumer) [^\r\n]*", cpu)) == Counter(
            f"ONLY {stage} {value:.2f}"
            for stage in ("producer", "consumer") for value in (1, 2)), cpu

    for config in configurations:
        print("Checking diagnostic-only", config, flush=True)
        output = run(grid + ["--warmup", "--iters", "3", "--layer", "only",
                            "--reparam:type=int", "only", "tag", "9",
                            "hart_diagnostic_only"],
                     config, launches=4, arena=0)
        expected = []
        for tag in (7, 7, 9, 9):
            for point, value in enumerate(values):
                expected += [
                    ("hart_diagnostic_only", "diagnostic_only_payload.osl",
                     100, point, f"ONLY point={point} value={value:.2f} tag={tag}"),
                    ("hart_diagnostic_only", "diagnostic_only_payload.osl",
                     101, point, f"ONLY warning point={point} tag={tag}"),
                ]
        assert Counter(records(output)) == Counter(expected), output
        assert "Pixel (" not in output, output
        no_images()

        output = run(["-g", "2", "1", "--iters", "2"] + connected,
                     config, launches=2, arena=0)
        expected = [
            (f"hart_diagnostic_{stage}", f"diagnostic_only_{stage}.osl",
             101 if stage == "producer" else 100, point,
             f"ONLY {stage} {value:.2f}")
            for _ in range(2) for point, value in enumerate((1, 2))
            for stage in ("producer", "consumer")
        ]
        assert records(output) == expected, output
        storage = re.findall(
            r"HART group storage: (\d+) bytes, alignment (\d+), local (\d+) bytes, scratch (\d+) bytes",
            output,
        )
        assert len(storage) == 1, output
        size, alignment, local, scratch = map(int, storage[0])
        assert size > 0 and alignment > 0, storage
        if config[2] == "local":
            assert local == size and scratch == 0, storage
        else:
            assert local == 0 and scratch > 0, storage
        no_images()

        # Even an optimized do-nothing group must compile and actually launch.
        output = run(grid + ["--iters", "2", "--print", "hart_diagnostic_empty"],
                     config, launches=2, arena=0)
        assert records(output) == [], output
        assert re.findall(r"^Pixel \((\d+), (\d+)\):$", output, re.MULTILINE) == [
            (str(x), str(y)) for y in range(2) for x in range(3)], output
        assert "Cout" not in output, output
        no_images()

        output = run(["-g", "2", "1", "--iters", "3", "hart_diagnostic_error"],
                     config, error="shader error", launches=1, arena=0)
        assert Counter(records(output)) == Counter([
            ("hart_diagnostic_error", "diagnostic_only_error.osl", 200, 0,
             "ONLY before-error point=0"),
            ("hart_diagnostic_error", "diagnostic_only_error.osl", 200, 1,
             "ONLY before-error point=1"),
            ("hart_diagnostic_error", "diagnostic_only_error.osl", 201, 0,
             "ONLY intentional error"),
        ]), output
        no_images()

    config = configurations[1]
    output = run(["--param", "reports", "257", "hart_diagnostic_capacity"],
                 config, error="diagnostic buffer overflow", launches=1, arena=0)
    assert Counter(records(output)) == Counter(
        ("hart_diagnostic_capacity", "diagnostic_only_capacity.osl", 300, 0,
         f"ONLY record {i}") for i in range(256)), output
    no_images()

    # No implicit output is selected from an input named Cout or other outputs.
    output = run(grid + ["hart_diagnostic_typed"], config, launches=1, arena=0)
    assert Counter(records(output)) == Counter(
        ("hart_diagnostic_typed", "diagnostic_only_typed.osl", 403, point,
         f"ONLY typed {value:.2f} {point}")
        for point, value in enumerate(values)), output
    no_images()

    output = run(grid + ["--print", "-o", "scalar", "null",
                        "-o", "number", "null", "-o", "record.value", "null",
                        "hart_diagnostic_typed"],
                 configurations[3], launches=1, arena=72)
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\n  scalar : (\S+)\n  number : (\S+)\n  record.value : (\S+)",
        output,
    )
    assert [(int(x), int(y), float(s), int(n), float(r))
            for x, y, s, n, r in rows] == [
                (point % 3, point // 3, value, point, value + 1)
                for point, value in enumerate(values)], output
    no_images()

    cpu = run(grid + ["--print", "hart_diagnostic_rgb"])
    output = run(grid + ["--print", "hart_diagnostic_rgb"],
                 configurations[0], launches=1, arena=72)
    assert rgb_pixels(output) == rgb_pixels(cpu) == rgb
    image = root / "rgb.pfm"
    run(grid + ["-o", "Cout", str(image), "hart_diagnostic_rgb"],
        configurations[3], launches=1, arena=72)
    assert pfm_pixels(image) == rgb
    image.unlink()

    run(grid + ["--param", "fail", "1", "-o", "Cout", str(image),
                "hart_diagnostic_rgb"],
        configurations[3], error="shader error", launches=1, arena=72)
    no_images()

    for name, selection, message in (
        ("hart_diagnostic_only", "Cout", "Unknown HART output"),
        ("hart_diagnostic_only", "tag", "Unknown HART output"),
        ("hart_diagnostic_typed", "Cout", "Unknown HART output"),
        ("hart_diagnostic_typed", "missing", "Unknown HART output"),
        ("hart_diagnostic_typed", "text", "must be numeric"),
        ("hart_diagnostic_typed", "record", "must be numeric"),
    ):
        run(["-o", selection, str(image), name], config, error=message)
        no_images()
    run(["hart_diagnostic_bad_default"], config,
        error="requires an RGB color output")
    run(["--param", "unused", "1"], config, error="requires an OSL shader")
    no_images()

    originals = [
        ("function-simple", [
            ('printf ("%g*%g', "2*2 = 4"),
            ('printf ("x =', "x = 2, y = 4"),
        ]),
        ("bug-locallifetime", [('printf ("Ran', "Ran 2 iterations")]),
        ("exit", [('printf ("This should print', "This should print")]),
    ]
    for fixture, messages in originals:
        source = testsuite / fixture / "test.osl"
        text = source.read_text(encoding="utf8")
        name = "diagnostic_original_" + fixture.replace("-", "_")
        local_source = root / "test.osl"
        shutil.copyfile(source, local_source)
        compile_source(local_source, name)
        cpu = run([name])
        output = run([name], config, launches=1, arena=0)
        expected = []
        for fragment, payload in messages:
            assert cpu.splitlines().count(payload) == 1, cpu
            assert text.count(fragment) == 1, fragment
            line = text[:text.index(fragment)].count("\n") + 1
            expected.append(("test", "test.osl", line, 0, payload))
        assert records(output) == expected, output
        assert "This should NOT NOT NOT print" not in cpu + output
        no_images()
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Diagnostic-only failure artifacts retained in", root, flush=True)

print("Diagnostic-only HART checks passed: real empty/side-effecting launches, "
      "metadata, repeats, errors, strict outputs and unchanged numeric results")
