# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Exercise the real runner's exclusive CPU/HART/OSL0/fused references."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

testsuite = Path(__file__).resolve().parent.parent
root = Path.cwd() / ("hart-references-" + uuid.uuid4().hex)
source = root / "source"
source.mkdir(parents=True)
(source / "ref").mkdir()
for name, text in (("out.txt", "cpu"), ("out-hart.txt", "gpu"),
                   ("out-noopt-hart.txt", "noopt"),
                   ("out-fused-hart.txt", "fused")):
    (source / "ref" / name).write_text(text + "\n", encoding="ascii")
emitter = root / "emit.py"
emitter.write_text("import os\nprint(os.environ['REFERENCE_PAYLOAD'])\n",
                   encoding="ascii")
command = f'"{sys.executable}" "{emitter}" >> out.txt 2>&1;'
(source / "run.py").write_text(
    "compile_osl_files = False\ncommand = " + repr(command) + "\n",
    encoding="utf-8",
)
env = os.environ.copy()
env.update(OSL_SOURCE_DIR=str(testsuite.parent),
           OSL_TESTSUITE_ROOT=str(testsuite), OSL_TESTSUITE_SRC=str(source),
           TESTSUITE_CLEANUP_ON_SUCCESS="0", TESTSHADE_OPTIX="0",
           TESTSHADE_FUSED="0")
env.pop("OSL_REGRESSION_TEST", None)
success = False
try:
    cases = [(hart, optimize, fused, payload) for hart in (0, 1)
             for optimize in ("0", "2", "") for fused in (0, 1)
             for payload in ("cpu", "gpu", "noopt", "fused")]
    for i, (hart, optimize, fused, payload) in enumerate(cases):
        work = root / str(i)
        work.mkdir()
        child_env = dict(env, TESTSHADE_HART=str(hart), TESTSHADE_OPT=optimize,
                         TESTSHADE_FUSED=str(fused),
                         REFERENCE_PAYLOAD=payload)
        result = subprocess.run(
            [sys.executable, str(testsuite / "runtest.py"), str(work)],
            cwd=root, env=child_env, capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        wanted = ("cpu" if not hart else "noopt" if optimize == "0"
                  else "fused" if fused else "gpu")
        assert result.returncode == (0 if payload == wanted else 1), output
        if payload == wanted:
            filename = {"cpu": "out.txt", "gpu": "out-hart.txt",
                        "noopt": "out-noopt-hart.txt",
                        "fused": "out-fused-hart.txt"}[wanted]
            assert "PASS: " in output and filename in output, output
        else:
            assert "NO MATCH for  out.txt" in output, output
    success = True
finally:
    if success:
        shutil.rmtree(root)
    else:
        print("Reference-selection failure artifacts retained in", root, flush=True)

print("48 actual runner cases passed: CPU/HART/OSL0/fused references stay exclusive")
