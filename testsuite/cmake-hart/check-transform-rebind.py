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
parser.add_argument("--kind", choices=("transform", "color", "outputs", "interactive", "userdata", "raytypes", "library", "attributes", "spaces"),
                    default="transform")
parser.add_argument("--library-dir")
args = parser.parse_args()
if args.kind == "transform" and args.mode == "unoptimized":
    parser.error("Unoptimized mode is not available for transform tests")
if bool(args.library_dir) != (args.kind == "library"):
    parser.error("--library-dir is required only for library tests")

try:
    result = subprocess.run(
        [args.unit, args.stdosl, args.mode] + ([args.kind] if args.kind != "transform" else [])
        + ([args.library_dir] if args.kind == "library" else []),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=300,
    )
except subprocess.TimeoutExpired as exc:
    raise AssertionError(
        f"HART transform unit timed out: {exc.cmd}\n{exc.stdout!r}"
    ) from exc
output = result.stdout
assert result.returncode == 0, f"Unit exited with {result.returncode}:\n{output}"
tag = "output rebind" if args.kind == "outputs" else args.kind
assert re.findall(rf"^HART {tag} pass (\d+)$", output, re.M) == ["0", "1", "2"], output
blocks = re.split(rf"^HART {tag} pass [012]$", output, flags=re.M)
keys = [sorted(re.findall(r"cache hit for key[ \t]+(\S+)",
                         block.split("HART color unsupported rebound")[0]
                              .split("HART spaces expected unknown")[0]))
        for block in blocks[2:]]
if args.kind == "library":
    assert keys[1], f"Expected cache hits when returning to library A:\n{output}"
else:
    assert keys[0] and keys[1] and keys[0] == keys[1], f"Expected identical B/A cache hits:\n{output}"
if args.kind == "color":
    assert "HART device services failed (error bits 512)" in output, output
    assert "unsupported color transform" in output, output
    assert "HART color error recovery passed" in output, output
if args.kind == "outputs":
    assert output.count("Launching HART grid") == 3, output
    errors = re.findall(r"^ERROR: (.+)$", output, re.M)
    assert len(errors) == 3, output
    assert all("does not match the existing output layout" in e for e in errors[:2]), output
    assert ("does not match the existing output layout" in errors[2]
            or errors[2] == "HART output 'out.Cout' has no resolved numeric storage"), output
if args.kind in ("interactive", "raytypes", "library", "attributes"):
    assert output.count("Launching HART grid") == 3, output
    assert "ERROR:" not in output, output
if args.kind == "spaces":
    assert output.count("Launching HART grid") == 7, output
    assert output.count("HART device services failed (error bits 8)") == 2, output
    assert len(re.findall(r"^ERROR:", output, re.M)) == 2, output
    assert "HART spaces error recovery passed" in output, output
if args.kind == "userdata":
    assert output.count("Launching HART grid") == 3, output
    errors = re.findall(r"^ERROR: (.+)$", output, re.M)
    expected = ["invalid data extent, stride, or presence count"] * 3
    expected += ["unsupported name, type, or derivatives"] * 2
    expected += ["presence values must be zero or one", "duplicate userdata name or hash"]
    assert errors == ["HART userdata: "+message for message in expected], output
checks = ("typed values and layout rejection" if args.kind == "outputs"
          else "ray masks and name precedence" if args.kind == "raytypes"
          else "values and derivatives")
print(f"HART {args.kind} {args.mode} A/B/A {checks}, artifact and cache reuse passed")
