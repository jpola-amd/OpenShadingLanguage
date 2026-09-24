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
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument("testshade")
parser.add_argument("--module", help="Enable GPU execution using the smoke bitcode")
parser.add_argument("--llvm-opt", help="LLVM opt for constructing bitcode validation cases")
args = parser.parse_args()
env = os.environ.copy()
env["TESTSHADE_OPTIX"] = "0"


def run(arguments, error=None, forbidden=()):
    try:
        result = subprocess.run(
            [args.testshade] + arguments, capture_output=True, text=True,
            env=env, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"HART command timed out: {exc.cmd}\n"
            f"stdout: {exc.stdout!r}\nstderr: {exc.stderr!r}"
        ) from exc
    output = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, "Unexpected success:\n" + output
        expected = (error,) if isinstance(error, str) else error
        assert any(message in output for message in expected), (
            f"Expected {expected!r}:\n{output}"
        )
    for message in forbidden:
        assert message not in output, f"Unexpected {message!r}:\n{output}"
    return result.stdout


def assemble(ir, path):
    result = subprocess.run(
        [args.llvm_opt, "-passes=verify", "-o", str(path), "-"],
        input=ir, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


with tempfile.TemporaryDirectory(prefix="osl-hart-grid-") as directory:
    root = Path(directory)
    missing = str(root / "missing.bc")
    empty = root / "empty.bc"
    empty.write_bytes(b"")
    text = root / "text.ll"
    text.write_text('target triple = "amdgcn-amd-amdhsa"\n')
    base = ["--hart", "--hart-module", str(text)]
    run(["--hart"], "requires --hart-module")
    run(["--hart", "--hart-module", missing], "Cannot read HART bitcode")
    run(["--hart", "--hart-module", str(empty)], "Cannot read HART bitcode")
    run(base, "Assemble textual LLVM IR with llvm-as")
    run(base + ["-g", "0", "2"], "must be positive")
    run(base + ["-g", "2147483647", "2147483647"], "size limit")
    run(base + ["--iters", "0"], "must be positive")
    run(base + ["--hart-device", "-1"], "must be nonnegative")
    run(base + ["--hart-entry", ""], "nonempty entry")
    run(base + ["shader.oso"], "not OSL shaders")
    run(base + ["--param", "x", "1"], "HART mode:")
    run(base + ["--center"], "HART mode:")
    run(base + ["-o", "other", "null"], "one RGB output")
    run(base + ["-o", "Cout", "null", "-o", "Cout", "null"], "one RGB output")
    run(["--hart", "--optix"], "mutually exclusive")
    run(["--optix", "--hart"], "mutually exclusive")
    run(["--hart-module", missing], "require --hart")

    corrupt = root / "corrupt.bc"
    corrupt.write_bytes(b"BC\xc0\xde" + b"\x00" * 12)
    run(["--hart", "--hart-module", str(corrupt)], "Cannot parse HART bitcode")
    if args.llvm_opt:
        host = root / "host.bc"
        assemble('target triple = "x86_64-pc-windows-msvc"\n', host)
        run(["--hart", "--hart-module", str(host)],
            "must target amdgcn-amd-amdhsa", forbidden=("HART device",))
        mixed = root / "mixed.bc"
        assemble('''target triple = "amdgcn-amd-amdhsa"
define void @one() #0 { ret void }
define void @two() #1 { ret void }
attributes #0 = { "target-cpu"="gfx1100" }
attributes #1 = { "target-cpu"="gfx1201" }
''', mixed)
        run(["--hart", "--hart-module", str(mixed)],
            "mixes target-cpu", forbidden=("HART device",))

    if args.module:
        base = ["--hart", "--hart-module", args.module, "-v"]
        for width, height in [(1, 1), (3, 2), (37, 5)]:
            output = run(base + ["-g", str(width), str(height), "--print",
                                 "--warmup", "--iters", "2"])
            rows = re.findall(
                r"Pixel \((\d+), (\d+)\): Cout = (\S+) (\S+) (\S+)", output
            )
            assert len(rows) == width * height, output
            for index, row in enumerate(rows):
                x, y = index % width, index // width
                assert (int(row[0]), int(row[1])) == (x, y), row
                u = 0.5 if width == 1 else x / (width - 1)
                v = 0.5 if height == 1 else y / (height - 1)
                for actual, expected in zip(map(float, row[2:]), (u, v, u + v)):
                    assert math.isclose(actual, expected, abs_tol=1e-6), row

        if args.llvm_opt:
            ir = subprocess.check_output(
                [args.llvm_opt, "-S", "-passes=verify", args.module, "-o", "-"],
                text=True, timeout=30,
            )
            targets = set(re.findall(r'"target-cpu"="([^"]+)"', ir))
            assert len(targets) == 1, targets
            target = targets.pop()
            wrong_target = "gfx1100" if target != "gfx1100" else "gfx1201"
            # The filename claims the correct architecture; only metadata differs.
            mismatch = root / f"grid_{target}.bc"
            assemble(ir.replace(f'"target-cpu"="{target}"',
                                f'"target-cpu"="{wrong_target}"'), mismatch)
            run(["--hart", "--hart-module", str(mismatch), "-v"],
                f"targets '{wrong_target}', but HIP device",
                forbidden=("Initializing HART runtime", "Compiling HART pipeline",
                           "Launching HART grid"))
            # Conversely, renaming matching bitcode must not cause rejection.
            renamed = root / f"grid_{wrong_target}.bc"
            shutil.copyfile(args.module, renamed)
            output = run(["--hart", "--hart-module", str(renamed), "--print"])
            assert "Pixel (0, 0): Cout = 0.5 0.5 1" in output, output

        run(base + ["--hart-entry", "__raygen__missing"],
            ("hartProgramGroupCreate failed", "hartPipelineCreate failed"))
        run(base + ["--hart-device", "2147483647"], "hipSetDevice")
        image = root / "grid.pfm"
        run(base + ["-g", "2", "2", "-o", "Cout", str(image)])
        with image.open("rb") as stream:
            assert stream.readline().strip() == b"PF"
            assert stream.readline().split() == [b"2", b"2"]
            scale = float(stream.readline())
            pixels = struct.unpack(("<" if scale < 0 else ">") + "12f",
                                   stream.read())
        # PFM stores rows bottom-to-top.
        assert pixels == (0, 1, 1, 1, 1, 2, 0, 0, 0, 1, 0, 1), pixels

print("HART grid CLI checks passed"
      + ("; GPU grid/readback/image checks passed" if args.module else ""))
