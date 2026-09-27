# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("unit")
parser.add_argument("stdosl")
parser.add_argument("--mode", choices=("split", "fused", "fused-local", "unoptimized"),
                    default="split")
parser.add_argument("--kind", choices=("transform", "color"), default="transform")
args = parser.parse_args()
if args.kind == "transform" and args.mode == "unoptimized":
    parser.error("Unoptimized mode is only available for color")

try:
    result = subprocess.run(
        [args.unit, args.stdosl, args.mode] + (["color"] if args.kind == "color" else []),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=300,
    )
except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        f"HART transform unit timed out: {exc.cmd}\n{exc.stdout!r}"
    ) from exc
output = result.stdout
assert result.returncode == 0, f"Unit exited with {result.returncode}:\n{output}"
assert re.findall(rf"^HART {args.kind} pass (\d+)$", output, re.M) == ["0", "1", "2"], output
blocks = re.split(rf"^HART {args.kind} pass [012]$", output, flags=re.M)
keys = [sorted(re.findall(r"cache hit for key[ \t]+(\S+)",
                         block.split("HART color unsupported rebound")[0]))
        for block in blocks[2:]]
assert keys[0] and keys[1] and keys[0] == keys[1], f"Expected identical B/A cache hits:\n{output}"
if args.kind == "color":
    assert "HART device services failed (error bits 512)" in output, output
    assert "unsupported color transform" in output, output
    assert "HART color error recovery passed" in output, output
print(f"HART {args.kind} {args.mode} A/B/A values, derivatives, artifact and cache reuse passed")
