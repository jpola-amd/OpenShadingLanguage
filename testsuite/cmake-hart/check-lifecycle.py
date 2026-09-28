# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("unit")
parser.add_argument("stdosl")
parser.add_argument("mode", choices=("split", "fused", "fused-local", "unoptimized"))
args = parser.parse_args()
result = subprocess.run(
    [args.unit, args.stdosl, args.mode, "lifecycle"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600,
)
output = result.stdout
assert result.returncode == 0, f"Unit exited with {result.returncode}:\n{output}"
for stage in ("cycle", "initial", "bounded failure", "independent",
              "update failure", "recovered", "destroyed"):
    assert re.findall(rf"^HART lifecycle {stage} (\d+)$", output, re.M) == [
        "0", "1", "2"
    ], output
assert output.count("Launching HART grid") == 15, output
errors = re.findall(r"^ERROR: (.+)$", output, re.M)
assert len(errors) == 6, output
assert errors.count("HART interactive parameter update failed") == 3, output
assert sum("HART device services failed" in error for error in errors) == 3, output
for cycle in range(3):
    failure = output.split(f"HART lifecycle bounded failure {cycle}\n")[1].split(
        f"HART lifecycle independent {cycle}\n")[0]
    recovery = output.split(f"HART lifecycle recovered {cycle}\n")[1].split(
        f"HART lifecycle destroyed {cycle}\n")[0]
    failure_keys = sorted(re.findall(r"cache hit for key[ \t]+(\S+)", failure))
    recovery_keys = sorted(re.findall(r"cache hit for key[ \t]+(\S+)", recovery))
    assert failure_keys and failure_keys == recovery_keys, output
print(f"HART {args.mode}: three two-renderer lifetimes, isolated groups, "
      "bounded errors, unpublished partial output, cache and binding recovery passed")
