# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


renderer, compiler, stdosl, mode = sys.argv[1:]
env = os.environ.copy()
for key in ("TESTSHADE_OPTIX", "TESTSHADE_OPT", "TESTSHADE_LLVM_OPT", "TESTRENDER_AA"):
    env.pop(key, None)
flags = {
    "split": ["--llvm_opt", "3"],
    "fused": ["--hart-fused", "--llvm_opt", "3"],
    "fused-local": ["--hart-fused", "--hart-local-groupdata", "4096",
                    "--llvm_opt", "3"],
    "unoptimized": ["-O0", "--llvm_opt", "10"],
}[mode]
width, height = 16, 12


def run(args, root, expected=0):
    result = subprocess.run(args, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=360)
    print(result.stdout, end="")
    if result.returncode != expected:
        raise RuntimeError((args, result.returncode, expected))
    return result.stdout


def pixels(path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"PF"
        assert list(map(int, stream.readline().split())) == [width, height]
        scale = float(stream.readline())
        data = struct.unpack(("<" if scale < 0 else ">")
                             + str(width * height * 3) + "f", stream.read())
    assert all(math.isfinite(v) for v in data), path
    return list(data)


def compare(actual, expected, tolerance=3e-5):
    assert len(actual) == len(expected)
    for i, (a, b) in enumerate(zip(actual, expected)):
        assert abs(a - b) <= tolerance, (i, a, b, tolerance)


with tempfile.TemporaryDirectory(prefix="osl-hart-path-") as temporary:
    root = Path(temporary)
    shaders = {
        "path_emit": """shader path_emit(
            color tint = 1 [[int interactive=1]], int globals = 0) {
            color w = tint;
            if (globals) {
                N = -N;
                w += color(0.1*u, 0.1*v,
                           0.01*(abs(dot(I,N)) + abs(Dx(u)) + abs(Dy(v))));
                w += .01*color(N)
                     + color(.001*surfacearea(), .01*raytype("camera"),
                             .01*backfacing());
            }
            Ci = w * emission();
        }""",
        "path_diffuse": """shader path_diffuse(color tint = 0.5) {
            closure color lobes[2] = {
                0.25*tint*diffuse(N), 0.75*tint*diffuse(N)};
            closure color copy[2];
            copy = lobes;
            int i = u > v;
            Ci = copy[i] + copy[1-i];
        }""",
        "path_producer": """shader path_producer(output color weight = 0,
                                                output string label = "") {
            string labels[2] = {"warm", "cool"};
            label = labels[int(u>.5)];
            float knots[4] = {0, 0.5, 1, 2};
            weight = color(0.2 + 0.1*u, 0.3 + 0.1*v, 0.4)
                   * texture("path_texture.pfm", u, v, "wrap", "clamp",
                             "interp", "linear")
                   * spline("bspline", u, knots);
            weight *= color(normalize(vector(blackbody(4000+2000*v))))
                    * transformc("hsv", "rgb", color(0.1+0.1*u,0.3,1))
                    * (0.5+luminance(wavelength_color(500+100*v)));
            weight *= 0.85+0.15*noise("gabor",point(2*u,2*v,.25+u+v),
                                     "anisotropic",1,"direction",vector(1,.5,.25));
        }""",
        "path_connected": """shader path_connected(color weight = 0,
                                                    string label = "missing") {
            Ci = weight * (label=="warm" ? .8 : label=="cool" ? 1.2 : 0) * emission();
        }""",
        "path_bad": """shader path_bad() { Ci = background(); }""",
        "path_spline_bad": """shader path_spline_bad() {
            float knots[4] = {0,1,2,3};
            Ci = spline("linear", u, 4+int(u>=0), knots) * emission();
        }""",
        "path_gabor_bad": """shader path_gabor_bad() {
            Ci = noise("gabor",point(1e20,u,v))*emission();
        }""",
        "path_diagnostic": """shader path_diagnostic() {
            printf("PATH diagnostic %s %d\\n","MiXeD",7);
            Ci=color(.25,.5,.75)*emission();
        }""",
        "path_diagnostic_error": """shader path_diagnostic_error() {
            error("PATH failure %s %d\\n","MiXeD",7);
            Ci=emission();
        }""",
        "path_diagnostic_overflow": """shader path_diagnostic_overflow() {
            for(int i=0;i<257;++i)
                printf("PATH record %d\\n",i);
            Ci=emission();
        }""",
        "path_overflow": """shader path_overflow() {
            Ci = 0;
            for (int i = 0; i < 48; ++i)
                Ci += (u+1)*diffuse(normalize(N + vector(0.01*i,0,0)));
        }""",
    }
    for name, source in shaders.items():
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        run([compiler, "-I" + str(Path(stdosl).parent), name + ".osl"], root)
    (root / "path_texture.pfm").write_bytes(
        b"PF\n4 4\n-1.0\n" + struct.pack("<48f", *([0.25, 0.5, 1.0] * 16)))

    camera = '<Camera eye="0,0,4" dir="0,0,-1" fov="90"/>'
    emission = f"""<World>{camera}
      <ShaderGroup>color tint 0.6 0.1 0.2 [[int interactive=1]];
        int globals 1; shader path_emit m;</ShaderGroup>
      <Quad corner="-2,-1.4,0" edge_x="1.8,0,0" edge_y="0,2.8,0"/>
      <ShaderGroup>shader path_producer p; shader path_connected c;
        connect p.weight c.weight; connect p.label c.label;</ShaderGroup>
      <Quad corner="0.2,-1.4,0" edge_x="1.8,0,0" edge_y="0,2.8,0"/>
      </World>"""
    offset_camera = camera.replace('eye="0,0,4"', 'eye="0.13,0.07,4"')
    furnace = f"""<World>{offset_camera}
      <ShaderGroup>color tint 0.2 0.4 0.6 [[int interactive=1]];
        shader path_diffuse d;</ShaderGroup>
      <Quad corner="-10,-10,0" edge_x="20,0,0" edge_y="0,20,0"/>
      <ShaderGroup>color tint 2 1 0.5 [[int interactive=1]]; shader path_emit e;</ShaderGroup>
      <Quad corner="-10,-10,6" edge_x="20,0,0" edge_y="0,20,0"/>
      <Quad corner="-10,-10,0" edge_x="20,0,0" edge_y="0,0,6"/>
      <Quad corner="-10,10,0" edge_x="20,0,0" edge_y="0,0,6"/>
      <Quad corner="-10,-10,0" edge_x="0,20,0" edge_y="0,0,6"/>
      <Quad corner="10,-10,0" edge_x="0,20,0" edge_y="0,0,6"/>
      </World>"""
    multiple = furnace.replace(
        '<ShaderGroup>color tint 2 1 0.5 [[int interactive=1]];',
        '<Quad corner="0.7,-1,2" edge_x="1.5,0,0" edge_y="0,2,0"/>'
        '<ShaderGroup>color tint 2 1 0.5 [[int interactive=1]];')
    scenes = {"emission": emission, "furnace": furnace, "multiple": multiple,
              "seams": furnace.replace(offset_camera, camera),
              "empty": f"""<World>{camera}
                <ShaderGroup>shader path_emit unused;</ShaderGroup></World>""",
              "no-shaders": f"<World>{camera}</World>",
              "bad": f"""<World>{camera}
                <ShaderGroup>shader path_bad b;</ShaderGroup>
                <Quad corner="-2,-2,0" edge_x="4,0,0" edge_y="0,4,0"/>
                </World>"""}
    scenes["overflow"] = scenes["bad"].replace("path_bad", "path_overflow")
    scenes["spline-error"] = scenes["bad"].replace("path_bad", "path_spline_bad").replace(
        'corner="-2,-2,0" edge_x="4,0,0" edge_y="0,4,0"',
        'corner="-10,-10,0" edge_x="20,0,0" edge_y="0,20,0"')
    scenes["gabor-error"] = scenes["spline-error"].replace("path_spline_bad", "path_gabor_bad")
    for name in ("diagnostic", "diagnostic_error", "diagnostic_overflow"):
        scenes[name] = scenes["spline-error"].replace("path_spline_bad", "path_" + name)
    for name, scale in (("tiny", 0.001), ("large", 1000.0)):
        world = ET.fromstring(scenes["seams"])
        for node in world:
            for key in ("corner", "edge_x", "edge_y", "eye"):
                if key in node.attrib:
                    node.set(key, ",".join(str(float(v) * scale)
                                          for v in node.get(key).split(",")))
        scenes[name] = ET.tostring(world, encoding="unicode")
    for name, scene in scenes.items():
        (root / (name + ".xml")).write_text(scene, encoding="ascii")

    common = ["--res", str(width), str(height), "--no-jitter", "-t", "1"]

    def render(name, gpu, bounces=1, aa=1, repeat=False):
        image = root / (name + ("-gpu.pfm" if gpu else "-cpu.pfm"))
        args = [renderer] + common + ["-aa", str(aa)]
        if gpu:
            args += ["--hart", "-v", "--hart-bounces", str(bounces)] + flags
            if repeat:
                args += ["--warmup", "--iters", "3"]
        else:
            args += ["--max-bounces", str(bounces), "--llvm_opt", "3"]
        output = run(args + [name + ".xml", str(image)], root)
        if gpu:
            assert "HART path tracer" in output, output
            storage = re.search(r"(\d+) caller Groupdata bytes per pixel",
                                output)
            assert storage and (int(storage[1]) == 0) == (mode == "fused-local")
        return pixels(image)

    cpu = render("emission", False)
    gpu = render("emission", True, repeat=True)
    compare(gpu, cpu)
    assert min(gpu) == 0 and max(gpu) > 0.4
    assert len(set(round(v, 4) for v in gpu)) > 20
    compare(render("furnace", True, bounces=0),
            [0.0] * (width * height * 3))
    gpu = render("furnace", True, bounces=1, aa=4)
    # testrender requests HALF output, including when writing float PFM files.
    furnace_rgb = [struct.unpack("e", struct.pack("e", v))[0]
                   for v in (0.4, 0.4, 0.3)]
    compare(gpu, render("furnace", False, bounces=1, aa=4))
    compare(gpu, furnace_rgb * (width * height))
    # The CPU reference can self-hit the adjacent triangle at exact shared edges.
    for name in ("seams", "tiny", "large"):
        compare(render(name, True, bounces=1, aa=4),
                furnace_rgb * (width * height))
    low = render("multiple", True, bounces=1, aa=4)
    high = render("multiple", True, bounces=3, aa=4)
    assert sum(high) > sum(low) + 0.1, (sum(low), sum(high))
    compare(high, render("multiple", False, bounces=3, aa=4), 2e-4)

    diagnostic_image = root / "diagnostic.pfm"
    output = run([renderer, "--hart", "-v", "--warmup", "--iters", "3"]
                 + common + flags + ["diagnostic.xml", str(diagnostic_image)], root)
    records = re.findall(r"HART shader 'path_diagnostic' "
                         r"\(([^\r\n]*):(\d+), point (\d+)\): PATH diagnostic MiXeD 7",
                         output)
    assert len(records) == width * height * 4, len(records)
    for point in range(width * height):
        assert sum(int(index) == point for _, _, index in records) == 4
    assert all(Path(source).name == "path_diagnostic.osl" and int(line) > 0
               for source, line, _ in records), records
    compare(pixels(diagnostic_image), [.25, .5, .75] * (width * height), 0)

    image = root / "rejected.pfm"
    for scene, message, bits in (
            ("diagnostic_error", "PATH failure MiXeD 7", 8192),
            ("diagnostic_overflow", "diagnostic buffer overflow", 2048)):
        out = run([renderer, "--hart", "-v"] + flags
                  + ["--res", "2", "2", scene + ".xml", str(image)], root, 1)
        assert message in out and f"error bits {bits}" in out, out
        assert "HART path tracer rendered" not in out and not image.exists(), out
    out = run([renderer, "--max-bounces", "-1", "emission.xml",
               str(image)], root, 1)
    assert "--max-bounces must be nonnegative" in out and not image.exists()
    out = run([renderer, "--hart", "no-shaders.xml", str(image)], root, 1)
    assert "No shaders in scene" in out and not image.exists()
    out = run([renderer, "--hart", "empty.xml", str(image)], root, 1)
    assert "No primitives in scene" in out and not image.exists()
    out = run([renderer, "--hart", "--hart-bounces", "65", "empty.xml",
               str(image)], root, 1)
    assert "Invalid HART options" in out and not image.exists()
    out = run([renderer, "--hart", "bad.xml", str(image)], root, 1)
    assert "HART" in out and "background" in out and not image.exists()
    out = run([renderer, "--hart", "-v"] + flags
              + ["--res", "2", "2", "overflow.xml", str(image)], root, 1)
    assert "closure pool allocation failed" in out and not image.exists()
    out = run([renderer, "--hart", "-v"] + flags
              + ["--res", "2", "2", "spline-error.xml", str(image)], root, 1)
    assert "HART device services failed (error bits 256)" in out
    assert "invalid spline arguments" in out and "HART path tracer rendered" not in out
    assert not image.exists()

    out = run([renderer, "--hart", "-v"] + flags
              + ["--res", "2", "2", "gabor-error.xml", str(image)], root, 1)
    assert "HART device services failed (error bits 1024)" in out
    assert "invalid noise arguments" in out and "HART path tracer rendered" not in out
    assert not image.exists()

print("HART path tracer verified: " + mode)
