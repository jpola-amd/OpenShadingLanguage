# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import argparse
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("testshade")
parser.add_argument("--oslc", required=True)
args = parser.parse_args()
testshade = str(Path(args.testshade).resolve())
oslc = str(Path(args.oslc).resolve())
env = {**os.environ, "TESTSHADE_OPTIX": "0", "TESTSHADE_BATCHED": "0",
       "TESTSHADE_RS_BITCODE": "0", "TESTSHADE_HART": "0", "TESTSHADE_FUSED": "0"}
width, height = 5, 3


def run(executable, arguments, error=None):
    result = subprocess.run([executable] + arguments, cwd=root, env=env,
                            capture_output=True, text=True, timeout=300)
    text = result.stdout + result.stderr
    if error is None:
        assert result.returncode == 0, text
    else:
        assert result.returncode != 0 and error in text, text
        assert "Launching HART grid" not in text, text
    return text


def check(text, evaluate):
    pixels = re.findall(r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s*([^\r\n]+)", text)
    assert len(pixels) == width * height, text
    for i, (x, y, value) in enumerate(pixels):
        x, y = int(x), int(y)
        assert (x, y) == (i % width, i // width), text
        actual = list(map(float, value.split()))
        assert len(actual) == 3, text
        expected = evaluate(x / (width - 1), y / (height - 1))
        for a, b in zip(actual, expected):
            assert math.isclose(a, b, rel_tol=0, abs_tol=2e-6), (x, y, actual, expected, text)


with tempfile.TemporaryDirectory(prefix="osl-hart-geometry-state-") as temporary:
    root = Path(temporary)
    sources = {
        "geometry_rays": """
            shader geometry_rays(output color Cout=0) {
                string selected = u<.5 ? "camera" : v>.5 ? "glossy" : "diffuse";
                Cout=color(raytype("camera")+2*raytype("diffuse")
                           +4*raytype("glossy")+8*raytype(selected)
                           +128*raytype("not-configured")+256*raytype(""),
                           surfacearea()+10*backfacing(),
                           dtime+length(dPdtime));
            }""",
        "geometry_write": """
            shader geometry_write(output float stamp=0) {
                stamp=u;
                P=point(2*u,3*v,1);
                I=vector(u,v,2);
                N=normal(0,1,0);
                Ng=normal(1,0,0);
                dPdu=vector(2,0,1);
                dPdv=vector(0,3,2);
                u=u+.25;
                v=v*2;
            }""",
        "geometry_read": """
            shader geometry_read(float stamp=0, output color Cout=0) {
                if (stamp > -1000) {
                    float q=P[0]+P[1]+I[0]+I[1]+u+v+N[1]+Ng[0]+dPdu[2]+dPdv[2];
                    Cout=color(q,Dx(q),Dy(q));
                } else Cout=-1;
            }""",
    }
    for name, expression in (("time", "u"), ("dtime", "u"),
                             ("dPdtime", "vector(1)")):
        sources["geometry_bad_" + name] = (
            f"shader geometry_bad_{name}(output color Cout=0) "
            f"{{ {name}={expression}; Cout=color(u,v,0); }}")
    sources["geometry_bad_Ps"] = (
        "shader geometry_bad_Ps(output color Cout=0) { Cout=color(Ps); }")
    includes = Path(__file__).resolve().parents[2] / "src" / "shaders"
    for name, source in sources.items():
        path = root / (name + ".osl")
        path.write_text(source, encoding="ascii")
        run(oslc, ["-I" + str(includes), str(path)])
    base = ["-g", str(width), str(height), "--print"]
    # An aliased connection need not execute a lazy global-writing layer.
    writers = ["--shader", "geometry_write", "writer",
               "--shader", "geometry_read", "reader",
               "--connect", "writer", "stamp", "reader", "stamp",
               "--entry", "writer", "--entry", "reader"]
    write_values = lambda u, v: (4*u+6*v+5.25, 4/(width-1), 6/(height-1))
    for cpu_flags in (["-O0", "--llvm_opt", "10"], ["-O2"]):
        check(run(testshade, ["-t", "1"] + cpu_flags + base + writers), write_values)
    for mode, flags in (
        ("split", []), ("fused", ["--hart-fused"]),
        ("fused-local", ["--hart-fused", "--hart-local-groupdata", "1048576"]),
        ("unoptimized", ["-O0", "--llvm_opt", "10"]),
    ):
        for ray in ("camera", "diffuse", "glossy", "shadow"):
            def values(u, v):
                selected = "camera" if u < .5 else "glossy" if v > .5 else "diffuse"
                return ({"camera": 1, "diffuse": 2, "glossy": 4}.get(ray, 0)
                        + 8*(selected == ray), 1, 0)
            parameters = base + ["--raytype", ray, "geometry_rays"]
            if mode == "split":
                check(run(testshade, ["-t", "1"] + parameters), values)
            text = run(testshade, ["--hart", "-v"] + flags + parameters
                       + ["--warmup", "--iters", "2"])
            check(text, values)
            assert text.count("Launching HART grid") == 3, text
        text = run(testshade, ["--hart", "-v"] + flags + base + writers
                   + ["--warmup", "--iters", "2"])
        check(text, write_values)
        assert text.count("Launching HART grid") == 3, text
        print(f"HART geometry {mode} live ray masks, globals and gradients passed")
    for name in ("time", "dtime", "dPdtime", "Ps"):
        error = ("unsupported shader global" if name == "Ps"
                 else "writing shader global") + f" '{name}'"
        run(testshade, ["--hart", "-v", "--shader", "geometry_bad_" + name, "unused",
                        "--shader", "geometry_rays", "output"], error)
