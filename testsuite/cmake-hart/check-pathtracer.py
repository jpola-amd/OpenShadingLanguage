# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


if len(sys.argv) not in (5, 6) or (len(sys.argv) == 6
                                 and sys.argv[5] not in ("--materials", "--lighting",
                                                        "--volumes", "--textures",
                                                        "--benchmark")):
    raise SystemExit("Usage: check-pathtracer.py renderer compiler stdosl "
                     "{split|fused|fused-local|unoptimized} "
                     "[--materials|--lighting|--volumes|--textures|--benchmark]\n"
                     "--benchmark rotates split/fused/fused-local, starting "
                     "with the supplied optimized mode; includes no-cache "
                     "samples and separate cache-enabled priming")
renderer, compiler, stdosl, mode = sys.argv[1:5]
materials = len(sys.argv) == 6 and sys.argv[5] == "--materials"
lighting = len(sys.argv) == 6 and sys.argv[5] == "--lighting"
volumes = len(sys.argv) == 6 and sys.argv[5] == "--volumes"
textures = len(sys.argv) == 6 and sys.argv[5] == "--textures"
benchmark = len(sys.argv) == 6 and sys.argv[5] == "--benchmark"
if benchmark and mode not in ("split", "fused", "fused-local"):
    raise SystemExit("--benchmark requires split, fused or fused-local as its starting mode")
env = os.environ.copy()
for key in ("TESTSHADE_OPTIX", "TESTSHADE_HART", "TESTSHADE_FUSED",
            "TESTSHADE_OPT", "TESTSHADE_LLVM_OPT", "TESTRENDER_AA"):
    env.pop(key, None)
if benchmark:
    for key in ("OSL_OPTIONS", "OSL_LLVM_DEBUG", "OSL_DEBUG_OUTPUT_CPP",
                "TESTSHADE_LLVM_JIT_FMA"):
        env.pop(key, None)
mode_flags = {
    "split": ["--llvm_opt", "3"],
    "fused": ["--hart-fused", "--llvm_opt", "3"],
    "fused-local": ["--hart-fused", "--hart-local-groupdata", "4096",
                    "--llvm_opt", "3"],
    "unoptimized": ["-O0", "--llvm_opt", "10"],
}
flags = mode_flags[mode]
width, height = (128, 96) if benchmark else (16, 12)


def run(args, root, expected=0, extra_env=None, wall_times=None):
    start = time.perf_counter()
    result = subprocess.run(args, cwd=root, env={**env, **(extra_env or {})},
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=360)
    wall_ms = (time.perf_counter() - start) * 1000
    if wall_times is not None:
        assert math.isfinite(wall_ms) and wall_ms >= 0
        wall_times.append(wall_ms)
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


def compare(actual, expected, tolerance=3e-5, half_output=False):
    assert len(actual) == len(expected)
    for i, (a, b) in enumerate(zip(actual, expected)):
        if half_output and abs(a - b) > tolerance:
            # testrender quantizes to HALF even when writing float PFM.
            # Permit only adjacent representable values, not a blanket
            # relative tolerance; component probes compare unquantized floats.
            packed = struct.pack("<ee", a, b)
            bits = struct.unpack("<HH", packed)
            if (struct.unpack("<ee", packed) == (a, b)
                    and abs(bits[0] - bits[1]) == 1):
                continue
        assert abs(a - b) <= tolerance, (i, a, b, tolerance)


common = ["--res", str(width), str(height), "--no-jitter", "-t", "1"]


def render(name, gpu, bounces=1, aa=1, repeat=False, material_count=None,
           cpu_unoptimized=False):
    image = root / (name + ("-gpu.pfm" if gpu else "-cpu.pfm"))
    args = [renderer] + common + ["-aa", str(aa)]
    if gpu:
        args += ["--hart", "-v", "--hart-bounces", str(bounces)] + flags
        if repeat:
            args += ["--warmup", "--iters", "3"]
    else:
        args += ["--max-bounces", str(bounces), "--llvm_opt", "3"]
        if cpu_unoptimized:
            args += ["-O0"]
    output = run(args + [name + ".xml", str(image)], root)
    if gpu:
        assert "HART path tracer" in output, output
        storage = re.search(r"(\d+) caller Groupdata bytes per pixel", output)
        assert storage and (int(storage[1]) == 0) == (mode == "fused-local")
        if material_count is not None:
            assert (f"HART path tracer rendered {width}x{height} with "
                    f"{aa * aa} samples per pixel") in output, output
            compiled = re.search(r"HART compiled (\d+) materials, "
                                 r"(\d+) callables", output)
            entries = 1 if mode in ("fused", "fused-local") else 2
            assert compiled and int(compiled[1]) == material_count, output
            assert int(compiled[2]) == material_count * entries, output
    if material_count is not None and not lighting:
        assert "triangles to be treated as lights" not in output, output
    return pixels(image)


def check_textures():
    for name, reverse in (("texture_a.pfm", False), ("texture_b.pfm", True)):
        data = [v for y in range(4) for x in range(4)
                for v in (0.125+0.25*(3-x if reverse else x),
                          0.125+0.25*(3-y if reverse else y),
                          0.25 if reverse else 0.75)]
        (root / name).write_bytes(b"PF\n4 4\n-1.0\n" + struct.pack("<48f", *data))

    # The CPU oracle explicitly selects the same magnifying sampler. OIIO's
    # omitted smart-bicubic defaults are not the reference GPU's defaults.
    options = {
        "defaults": ("", '"wrap","periodic","interp","linear"'),
        "wrap": ('"wrap","clamp"', '"wrap","clamp","interp","linear"'),
        "interp": ('"interp","closest"', '"wrap","periodic","interp","closest"'),
        "swrap": ('"swrap","black"', '"swrap","black","twrap","periodic","interp","linear"'),
        "twrap": ('"twrap","clamp"', '"swrap","periodic","twrap","clamp","interp","linear"'),
        "explicit": ('"wrap","clamp","interp","closest"',
                     '"wrap","clamp","interp","closest"'),
        "reset": ('"wrap","clamp","swrap","periodic","twrap","periodic",'
                  '"interp","closest","interp","linear"',
                  '"wrap","periodic","interp","linear"'),
    }
    shaders = {}
    for name, pair in options.items():
        for suffix, tokens in zip(("gpu", "ref"), pair):
            shader = "path_texture_" + name + "_" + suffix
            shaders[shader] = f"""shader {shader}(
                string filename="texture_a.pfm") {{
                color value=texture(filename, 5*u-2, 5*v-2
                                    {"," + tokens if tokens else ""});
                Ci=uniform_edf(value);
            }}"""
    shaders["path_texture_empty"] = """shader path_texture_empty(
        string filename="") {
        color value=texture(filename, 5*u-2, 5*v-2);
        Ci=uniform_edf(value);
    }"""
    shaders["path_texture_pair"] = """shader path_texture_pair(
        string first="texture_a.pfm", string second="texture_b.pfm") {
        color a=texture(first, 5*u-2, 5*v-2);
        color b=texture(second, 5*u-2, 5*v-2);
        Ci=uniform_edf(0.25*a + 0.75*b);
    }"""
    shaders["path_texture_pair_ref"] = """shader path_texture_pair_ref(
        string first="texture_a.pfm", string second="texture_b.pfm") {
        color a=texture(first, 5*u-2, 5*v-2,
                        "wrap","periodic","interp","linear");
        color b=texture(second, 5*u-2, 5*v-2,
                        "wrap","periodic","interp","linear");
        Ci=uniform_edf(0.25*a + 0.75*b);
    }"""
    for name, source in shaders.items():
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        run([compiler, "-I" + str(Path(stdosl).parent), name + ".osl"], root)

    def scene(name, shader, params=""):
        (root / (name + ".xml")).write_text(f"""<World>
          <Camera eye="0,0,4" dir="0,0,-1" fov="90"/>
          <ShaderGroup name="surface" is_light="0">
            {params} shader {shader} m;</ShaderGroup>
          <Quad corner="-10,-10,0" edge_x="20,0,0" edge_y="0,20,0"/>
          </World>""", encoding="ascii")

    results = {}
    for name in options:
        print("Checking HART texture options: " + name + " (" + mode + ")", flush=True)
        scene(name + "_ref", "path_texture_" + name + "_ref")
        scene(name, "path_texture_" + name + "_gpu")
        cpu = render(name + "_ref", False, bounces=0, material_count=1)
        results[name] = render(name, True, bounces=0, material_count=1)
        compare(results[name], cpu, half_output=True)
    compare(results["reset"], results["defaults"], 0)
    for name in ("wrap", "interp", "swrap", "twrap", "explicit"):
        assert max(abs(a-b) for a, b in zip(results[name], results["defaults"])) > 0.01

    override = 'param string filename "texture_b.pfm";'
    scene("override_ref", "path_texture_defaults_ref", override)
    expected_b = render("override_ref", False, bounces=0, material_count=1)
    for name, shader in (("override", "path_texture_defaults_gpu"),
                         ("empty_override", "path_texture_empty")):
        scene(name, shader, override)
        compare(render(name, True, bounces=0, material_count=1), expected_b,
                half_output=True)
    assert max(abs(a-b) for a, b in zip(expected_b, results["defaults"])) > 0.1
    # Both IDs coexist, including a swapped assignment of the same filenames.
    for name, params in (
            ("pair", ""),
            ("pair_swapped", 'param string first "texture_b.pfm";'
             'param string second "texture_a.pfm";')):
        scene(name, "path_texture_pair", params)
        scene(name + "_ref", "path_texture_pair_ref", params)
        expected = render(name + "_ref", False, bounces=0, material_count=1)
        compare(render(name, True, bounces=0, material_count=1), expected,
                half_output=True)
    compare(render("defaults", True, bounces=0, repeat=True, material_count=1),
            results["defaults"], 0)

    image = root / "texture-rejected.pfm"
    for name, shader, params, error in (
            ("empty", "path_texture_empty", "", "texture requires a literal filename"),
            ("empty_override_bad", "path_texture_defaults_gpu",
             'param string filename "";', "texture requires a literal filename"),
            ("missing", "path_texture_empty", 'param string filename "missing.pfm";',
             "HART: cannot prepare texture")):
        scene(name, shader, params)
        out = run([renderer, "--hart", "-v"] + flags + common
                  + [name + ".xml", str(image)], root, 1)
        assert error in out, out
        assert "HART path tracer rendered" not in out and not image.exists(), out


def check_materials():
    shaders = {
        "path_material_inputs": """shader path_material_inputs(
            output color weight=0, output string distribution="",
            output string label="") {
            weight = color(0.2+0.3*u, 0.25+0.25*v, 0.3+0.1*u)
                   * texture("material_texture.pfm", u, v, "wrap", "clamp",
                             "interp", "linear");
            distribution = u>v ? "ggx" : "beckmann";
            label = u>v ? "top" : "base";
        }""",
        "path_material_light": """shader path_material_light(
            output closure color value=0) {
            value = uniform_edf(color(2,1,0.5), "label", "enclosure");
            Ci = value;
        }""",
    }
    closures = {
        "oren": "oren_nayar(n, 0.35)",
        "phong": "phong(n, 12)",
        "ward": "ward(n, t, 0.2, 0.35)",
        "ggx_reflect": 'microfacet("ggx", n, t, 0.2, 0.35, 1.5, 0)',
        "beckmann_reflect": 'microfacet("beckmann", n, t, 0.2, 0.35, 1.5, 0)',
        "ggx_transmit": 'microfacet("ggx", n, t, 0.2, 0.35, 1.5, 1)',
        "beckmann_transmit": 'microfacet("beckmann", n, t, 0.2, 0.35, 1.5, 1)',
        "microfacet_dynamic": "microfacet(distribution, n, t, 0.2, 0.35, 1.5, 2)",
        "reflection": "reflection(n)",
        "fresnel_reflection": "reflection(n, 1.5)",
        "refraction": "refraction(n, 1.5)",
        "transparent": "transparent()",
        "mx_burley": ('burley_diffuse_bsdf(n, color(0.5,0.7,0.4), 0.35, '
                      '"label", label)'),
        "mx_oren": ('oren_nayar_diffuse_bsdf(n, color(0.5,0.7,0.4), 0.35, '
                    '"energy_compensation", 1, "label", label)'),
        "mx_sheen0": ('sheen_bsdf(n, color(0.5,0.7,0.4), 0.35, '
                      '"mode", 0, "label", label)'),
        "mx_sheen1": ('sheen_bsdf(n, color(0.5,0.7,0.4), 0.35, '
                      '"mode", 1, "label", label)'),
        "mx_conductor": ('conductor_bsdf(n, t, 0.2, 0.35, color(0.2,0.9,1.1), '
                         'color(3,2,1), "ggx", "thinfilm_thickness", 100.0, '
                         '"thinfilm_ior", 1.4)'),
        "mx_dielectric": ('dielectric_bsdf(n, t, color(0.8), color(0.6), '
                          '0.2, 0.35, 1.5, "ggx", "thinfilm_thickness", 100.0, '
                          '"thinfilm_ior", 1.4, "absorption", color(0.1,0.2,0.3))'),
        "mx_schlick": ('generalized_schlick_bsdf(n, t, color(0.8), color(0.6), '
                       '0.2, 0.35, color(0.04,0.09,0.16), color(0.95), '
                       '4.0, "ggx")'),
        "mx_layer": ('layer(sheen_bsdf(n, color(0.2,0.35,0.5), 0.35, '
                     '"mode", 1, "label", label), '
                     'burley_diffuse_bsdf(n, color(0.4,0.5,0.7), 0.25, '
                     '"label", "undercoat"))'),
        "thinlayer": ("thinlayer(n, t, 1.5, 0.25, 0.3, 0.1, "
                      "color(0.8), color(0.7), color(0.1,0.2,0.3))"),
    }
    # The registered thinlayer prototype follows render-spi-thinlayer.
    thinlayer = """closure color thinlayer(normal N, vector U, float IOR,
        float roughness, float anisotropy, float thickness, color refl_tint,
        color refr_tint, color sigma_t) [[int builtin=1]];"""
    constant = {"reflection", "transparent"}
    expressions = dict(closures)
    expressions["bad_distribution"] = (
        'microfacet(u>v ? "invalid_ggx" : "invalid_beckmann", '
        'n, t, 0.2, 0.35, 1.5, 0)')
    for name, expression in expressions.items():
        tint = "color(0.25,0.5,0.75)" if name in constant else "weight"
        prefix = thinlayer if name == "thinlayer" else ""
        shaders["path_material_" + name] = prefix + f"""
            shader path_material_{name}(color weight=0,
                string distribution="missing", string label="missing",
                output closure color value=0) {{
                normal n = normalize(N + vector(0.06+0.04*u, -0.05+0.04*v, 0));
                N = n;
                vector t = normalize(vector(1,0,0) - n[0]*vector(n));
                value = {expression};
                Ci = {tint} * value;
            }}"""
    for name, source in shaders.items():
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        run([compiler, "-I" + str(Path(stdosl).parent), name + ".osl"], root)
    texels = [v for y in range(4) for x in range(4)
              for v in (0.5+0.125*x, 0.5+0.125*y, 0.75)]
    (root / "material_texture.pfm").write_bytes(
        b"PF\n4 4\n-1.0\n" + struct.pack("<48f", *texels))

    # No light primitives or Background: CPU and HART both sample only the BSDF.
    # The lower emitter catches transmission; the offset camera avoids seams.
    for name in expressions:
        scene = f"""<World>
          <Camera eye="0.13,0.07,4" dir="0,0,-1" fov="90"/>
          <ShaderGroup name="surface" is_light="0">
            shader path_material_inputs p; shader path_material_{name} m;
            connect p.weight m.weight; connect p.distribution m.distribution;
            connect p.label m.label;</ShaderGroup>
          <Quad corner="-10,-10,0" edge_x="20,0,0" edge_y="0,20,0"/>
          <ShaderGroup name="enclosure" is_light="0">
            shader path_material_light e;</ShaderGroup>
          <Quad corner="-10,-10,6" edge_x="20,0,0" edge_y="0,20,0"/>
          <Quad corner="-10,-10,-6" edge_x="20,0,0" edge_y="0,20,0"/>
          <Quad corner="-10,-10,-6" edge_x="20,0,0" edge_y="0,0,12"/>
          <Quad corner="-10,10,-6" edge_x="20,0,0" edge_y="0,0,12"/>
          <Quad corner="-10,-10,-6" edge_x="0,20,0" edge_y="0,0,12"/>
          <Quad corner="10,-10,-6" edge_x="0,20,0" edge_y="0,0,12"/>
          </World>"""
        (root / ("material_" + name + ".xml")).write_text(scene, encoding="ascii")

    compare(render("material_reflection", True, bounces=0, material_count=2),
            [0.0] * (width * height * 3), 0)
    for name in closures:
        print("Checking HART material: " + name + " (" + mode + ")", flush=True)
        cpu = render("material_" + name, False, aa=4, material_count=2)
        if name == "mx_layer":
            compare(render("material_" + name, False, aa=4, material_count=2,
                           cpu_unoptimized=True), cpu, half_output=True)
        gpu = render("material_" + name, True, aa=4, material_count=2,
                     repeat=name == "mx_layer")
        compare(gpu, cpu, half_output=True)
        for values in (cpu, gpu):
            assert min(values) >= 0 and max(values) > 0, name
            if name in constant:
                compare(values, [0.5, 0.5, 0.375] * (width * height), 0)
            else:
                colors = {tuple(round(v, 4) for v in values[i:i+3])
                          for i in range(0, len(values), 3)}
                assert len(colors) > 8, (name, len(colors))

    image = root / "material-rejected.pfm"
    out = run([renderer, "--hart", "-v", "--hart-bounces", "1"] + flags
              + ["--res", "2", "2", "--no-jitter", "-t", "1", "-aa", "1",
                 "material_bad_distribution.xml", str(image)], root, 1)
    assert "HART compiled 2 materials" in out, out
    assert "HART device services failed (error bits 32)" in out, out
    assert "invalid closure tree" in out, out
    assert "HART path tracer rendered" not in out and not image.exists(), out


def check_lighting():
    shaders = {
        "path_lighting_surface": """shader path_lighting_surface(
            int glossy=0) {
            normal n=normalize(N+vector(0.03,-0.02,0));
            vector t=normalize(vector(1,0,0)-n[0]*vector(n));
            Ci=color(0.25+0.1*u,0.3+0.1*v,0.4)*diffuse(n);
            if (glossy)
                Ci += color(0.15)*microfacet("ggx",n,t,0.25,0.25,1.5,0);
        }""",
        "path_lighting_emitter": """shader path_lighting_emitter(
            color power=color(8,4,2)) {
            Ci=power*(0.8+0.2*u)*emission();
        }""",
        "path_lighting_background": """shader path_lighting_background(
            color radiance=color(0.5,1,2), int gradient=0, int dark_cap=0,
            int textured=0) {
            color value=radiance;
            if (gradient)
                value *= color(0.4+0.2*abs(I[0]),0.3+0.1*abs(I[1]),
                               0.2+0.05*abs(I[2]));
            if (dark_cap && min(I[0],min(I[1],I[2])) < -0.75)
                value=0;
            if (textured)
                value *= texture("lighting_background.pfm",
                                 0.5+0.25*I[0],0.5+0.25*I[1],
                                 "wrap","clamp","interp","linear");
            Ci=value*background();
        }""",
        "path_lighting_black": "shader path_lighting_black() { Ci=0; }",
        "path_lighting_reflect": """shader path_lighting_reflect() {
            Ci=color(0.25,0.5,0.75)*reflection(N);
        }""",
        "path_lighting_transmit": """shader path_lighting_transmit() {
            Ci=color(0.25,0.5,0.75)*transparent();
        }""",
        "path_lighting_bad": "shader path_lighting_bad() { Ci=diffuse(N); }",
    }
    for name, source in shaders.items():
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        run([compiler, "-I" + str(Path(stdosl).parent), name + ".osl"], root)
    texels = [v for y in range(4) for x in range(4)
              for v in (0.5+0.125*x, 0.5+0.125*y, 0.75)]
    (root / "lighting_background.pfm").write_bytes(
        b"PF\n4 4\n-1.0\n" + struct.pack("<48f", *texels))

    camera = '<Camera eye="0.13,0.07,4" dir="0,0,-1" fov="90"/>'
    floor = """<ShaderGroup name="floor">
        shader path_lighting_surface m;</ShaderGroup>
        <Quad corner="-10,-10,0" edge_x="20,0,0" edge_y="0,20,0"/>"""
    area = """<ShaderGroup name="area" is_light="yes">
        shader path_lighting_emitter e;</ShaderGroup>
        <Quad corner="4,-0.5,3" edge_x="0,1,0" edge_y="1,0,0"/>"""
    second_area = """<ShaderGroup name="second_area" is_light="yes">
        param color power 2 4 8; shader path_lighting_emitter e;</ShaderGroup>
        <Quad corner="-4,-0.5,2.75" edge_x="0,1,0" edge_y="1,0,0"/>"""
    occluder = """<ShaderGroup name="occluder">
        shader path_lighting_black m;</ShaderGroup>
        <Quad corner="3.5,-1,2.5" edge_x="1.5,0,0" edge_y="0,2,0"/>"""
    behind_camera = """<ShaderGroup name="behind_camera">
        shader path_lighting_black m;</ShaderGroup>
        <Quad corner="-1,-1,6" edge_x="2,0,0" edge_y="0,2,0"/>"""
    walls = """
        <Quad corner="-6,-6,0" edge_x="0,12,0" edge_y="0,0,6"/>
        <Quad corner="6,-6,0" edge_x="0,0,6" edge_y="0,12,0"/>
        <Quad corner="-6,-6,0" edge_x="0,0,6" edge_y="12,0,0"/>
        <Quad corner="-6,6,0" edge_x="12,0,0" edge_y="0,0,6"/>"""

    def background(varying=0, resolution=32, radiance="0.5 1 2", dark_cap=0,
                   textured=0):
        return f"""<ShaderGroup name="background">
            param color radiance {radiance}; param int gradient {varying};
            param int dark_cap {dark_cap};
            param int textured {textured};
            shader path_lighting_background b;</ShaderGroup>
            <Background resolution="{resolution}"/>"""

    scenes = {
        "lighting_background": background() + behind_camera,
        "lighting_background_varying": background(varying=1) + behind_camera,
        "lighting_background_unsampled": background(varying=1, resolution=0)
                                         + behind_camera,
        "lighting_background_black": background(radiance="0 0 0") + floor,
        "lighting_background_partial": background(dark_cap=1) + floor,
        "lighting_background_texture": background(varying=1, resolution=257,
                                                   textured=1) + floor,
        "lighting_diffuse_background": background() + floor,
        "lighting_reflect_background": background() + floor.replace(
            "path_lighting_surface", "path_lighting_reflect"),
        "lighting_transmit_background": background() + floor.replace(
            "path_lighting_surface", "path_lighting_transmit"),
        "lighting_area": floor + area,
        "lighting_blocked": floor + area + occluder,
        "lighting_two_lights": floor + area + second_area,
        "lighting_mixed": background(varying=1) + floor + area,
        "lighting_bounces": floor.replace("shader path_lighting_surface",
                                          "param int glossy 1; shader path_lighting_surface")
                            + walls + area,
        "lighting_bad_background": """<ShaderGroup name="background">
            shader path_lighting_bad b;</ShaderGroup><Background resolution="8"/>"""
                                   + behind_camera,
    }
    scenes["lighting_roulette"] = '<Option rr_depth="int 0"/>' + scenes["lighting_bounces"]
    for name, scene in scenes.items():
        (root / (name + ".xml")).write_text(
            "<World>" + camera + scene + "</World>", encoding="ascii")

    def paired(name, count, bounces=1, aa=4, repeat=False):
        print("Checking HART lighting: " + name + " (" + mode + ")", flush=True)
        cpu = render(name, False, bounces=bounces, aa=aa, material_count=count)
        gpu = render(name, True, bounces=bounces, aa=aa, repeat=repeat,
                     material_count=count)
        compare(gpu, cpu, half_output=True)
        assert min(cpu) >= 0 and min(gpu) >= 0, name
        return gpu

    compare(paired("lighting_background", 2, bounces=0, aa=1),
            [0.5, 1.0, 2.0] * (width * height), 0)
    compare(paired("lighting_background_black", 2, bounces=2),
            [0.0] * (width * height * 3), 0)
    for name in ("lighting_background_varying", "lighting_background_unsampled"):
        values = paired(name, 2, bounces=0, aa=1)
        colors = {tuple(round(v, 4) for v in values[i:i+3])
                  for i in range(0, len(values), 3)}
        assert len(colors) > 8 and min(values) > 0, (name, len(colors))
    for name in ("lighting_reflect_background", "lighting_transmit_background"):
        compare(paired(name, 2), [0.125, 0.5, 1.5] * (width * height), 0)

    compare(paired("lighting_area", 2, bounces=0, aa=1),
            [0.0] * (width * height * 3), 0)
    plain = paired("lighting_area", 2)
    assert min(plain) > 0, "Direct sampling must illuminate every unoccluded pixel"
    blocked = paired("lighting_blocked", 3)
    assert sum(blocked) < sum(plain) - 0.1, (sum(blocked), sum(plain))
    two_lights = paired("lighting_two_lights", 3)
    assert sum(two_lights) > sum(plain) + 0.1, (sum(two_lights), sum(plain))
    for name, count in (("lighting_diffuse_background", 2),
                        ("lighting_background_partial", 2),
                        ("lighting_background_texture", 2), ("lighting_mixed", 3)):
        values = paired(name, count, repeat=name == "lighting_mixed")
        assert min(values) > 0, name
    low = paired("lighting_bounces", 2)
    high = paired("lighting_bounces", 2, bounces=3)
    assert sum(high) > sum(low) + 0.1, (sum(high), sum(low))
    assert max(paired("lighting_roulette", 2, bounces=7)) > 0

    image = root / "lighting-rejected.pfm"
    out = run([renderer, "--hart", "-v"] + flags + common
              + ["lighting_bad_background.xml", str(image)], root, 1)
    assert "error bits 32" in out and "invalid closure tree" in out, out
    assert "HART path tracer rendered" not in out and not image.exists(), out


def check_volumes():
    shaders = {
        "path_volume_boundary": """volume path_volume_boundary(
            int kind=0, color albedo=0, color extinction=color(0.2,0.4,0.6),
            float anisotropy=0, float depth=1, color transmission=color(0.8,0.6,0.4),
            float ior=1, int priority=0, float weight=1, int layered=0, int glass=0) {
            closure color fog=0;
            if (kind)
                fog=medium_vdf(albedo,depth,transmission,anisotropy,ior,priority);
            else
                fog=anisotropic_vdf(albedo,extinction,anisotropy);
            vector tangent=normalize(cross(N,dPdv));
            closure color boundary=transparent();
            if (glass)
                boundary=dielectric_bsdf(
                    N,tangent,color(1),color(1),0.1,0.1,ior,"ggx");
            if (layered)
                Ci=weight*layer(1.0e-6*reflection(N,1.5),boundary+fog);
            else
                Ci=weight*(boundary+fog);
        }""",
        "path_volume_emitter": "shader path_volume_emitter() { Ci=emission(); }",
    }
    for name, source in shaders.items():
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        run([compiler, "-I" + str(Path(stdosl).parent), name + ".osl"], root)

    camera = '<Camera eye="0.13,0.07,4" dir="0,0,-1" fov="90"/>'

    def box(front, back):
        height = front-back
        return f"""
            <Quad corner="-10,-10,{front}" edge_x="20,0,0" edge_y="0,20,0"/>
            <Quad corner="-10,-10,{back}" edge_x="0,20,0" edge_y="20,0,0"/>
            <Quad corner="-10,-10,{back}" edge_x="0,0,{height}" edge_y="0,20,0"/>
            <Quad corner="10,-10,{back}" edge_x="0,20,0" edge_y="0,0,{height}"/>
            <Quad corner="-10,-10,{back}" edge_x="20,0,0" edge_y="0,0,{height}"/>
            <Quad corner="-10,10,{back}" edge_x="0,0,{height}" edge_y="20,0,0"/>"""

    def boundary(name, params, front=1, back=0):
        return (f'<ShaderGroup name="{name}" is_light="0">{params}'
                ' shader path_volume_boundary m;</ShaderGroup>' + box(front, back))

    enclosure = """<ShaderGroup name="enclosure" is_light="0">
        shader path_volume_emitter e;</ShaderGroup>
        <Quad corner="-12,-12,6" edge_x="24,0,0" edge_y="0,24,0"/>
        <Quad corner="-12,-12,-3" edge_x="24,0,0" edge_y="0,24,0"/>
        <Quad corner="-12,-12,-3" edge_x="24,0,0" edge_y="0,0,9"/>
        <Quad corner="-12,12,-3" edge_x="24,0,0" edge_y="0,0,9"/>
        <Quad corner="-12,-12,-3" edge_x="0,24,0" edge_y="0,0,9"/>
        <Quad corner="12,-12,-3" edge_x="0,24,0" edge_y="0,0,9"/>"""
    cases = {
        "volume_absorb": ("", 2),
        "volume_zero_channel": ("param color extinction 0 0.4 0.6;", 2),
        "volume_vacuum": ("param color extinction 0 0 0;", 2),
        "volume_medium": ("param int kind 1;", 2),
        "volume_medium_clear": ("param int kind 1; param color transmission 1 0.6 0.4;", 2),
        "volume_weighted": ("param float weight 0.5;", 2),
        "volume_layered": ("param float weight 0.5; param int layered 1;", 2),
        "volume_isotropic": ("param color albedo 0.5 0.6 0.7;", 8),
        "volume_forward": ("param color albedo 0.5 0.6 0.7; param float anisotropy 0.4;", 8),
        "volume_backward": ("param color albedo 0.5 0.6 0.7; param float anisotropy -0.4;", 8),
        "volume_medium_scatter": ("param int kind 1; param color albedo 0.4 0.5 0.6;", 8),
        "volume_glass": ("param int kind 1; param int glass 1; param float ior 1.3; "
                         "param color transmission 0.8 0.8 0.8;", 8),
    }
    scenes = {name: boundary(name, params) for name, (params, _) in cases.items()}
    outer = "param int kind 1; param int priority 1;"
    inner = "param int kind 1; param color transmission 0.9 0.7 0.5;"
    for name, priority in (("volume_nested_low", 0), ("volume_nested_equal", 1),
                           ("volume_nested_high", 2)):
        scenes[name] = (boundary("outer", outer, 1.5, 0)
                        + boundary("inner", inner + f"param int priority {priority};",
                                   1, 0.5))
    scenes["volume_bad_extinction"] = boundary("bad", "param color extinction -1 0.4 0.6;")
    scenes["volume_bad_anisotropy"] = boundary("bad", "param float anisotropy 2;")
    scenes["volume_capacity"] = "".join(
        boundary("nested" + str(i), "param color extinction 0 0 0;",
                 2.8-0.1*i, -1.8+0.1*i) for i in range(9))
    for name, scene in scenes.items():
        (root / (name + ".xml")).write_text(
            "<World>" + camera + scene + enclosure + "</World>", encoding="ascii")

    def paired(name, count=2, bounces=2, repeat=False):
        print("Checking HART volume: " + name + " (" + mode + ")", flush=True)
        cpu = render(name, False, bounces=bounces, aa=4, material_count=count)
        gpu = render(name, True, bounces=bounces, aa=4, repeat=repeat,
                     material_count=count)
        compare(gpu, cpu, half_output=True)
        assert min(cpu) >= 0 and min(gpu) >= 0 and max(gpu) > 0, name
        return gpu

    def beer(sigma, scale=1):
        expected = []
        for y in range(height):
            for x in range(width):
                distance = math.sqrt(1 + ((x+0.5-width/2)/height)**2
                                     + (0.5-(y+0.5)/height)**2)
                for value in sigma:
                    expected.append(struct.unpack("<e", struct.pack(
                        "<e", scale*math.exp(-value*distance)))[0])
        return expected

    results = {}
    for name, (_, bounces) in cases.items():
        results[name] = paired(name, bounces=bounces, repeat=name == "volume_layered")
    compare(results["volume_absorb"], beer((0.2, 0.4, 0.6)), half_output=True)
    compare(results["volume_zero_channel"], beer((0, 0.4, 0.6)), half_output=True)
    compare(results["volume_vacuum"], [1.0] * (width * height * 3), 0)
    compare(results["volume_medium"], beer(tuple(-math.log(v) for v in (0.8, 0.6, 0.4))),
            half_output=True)
    compare(results["volume_medium_clear"],
            beer(tuple(-math.log(v) for v in (1, 0.6, 0.4))), half_output=True)
    compare(results["volume_weighted"], beer((0.1, 0.2, 0.3), 0.25), half_output=True)
    compare(results["volume_layered"], results["volume_weighted"], half_output=True)
    compare(results["volume_layered"],
            render("volume_layered", False, bounces=2, aa=4, material_count=2,
                   cpu_unoptimized=True), half_output=True)
    outer_sigma = tuple(-math.log(v) for v in (0.8, 0.6, 0.4))
    inner_sigma = tuple(-math.log(v) for v in (0.9, 0.7, 0.5))
    for name, outer_width, inner_width in (
            ("volume_nested_low", 1.5, 0), ("volume_nested_equal", 1.5, 0.5),
            ("volume_nested_high", 1, 0.5)):
        result = paired(name, count=3, bounces=4)
        compare(result, beer(tuple(outer_width*a+inner_width*b
                                   for a, b in zip(outer_sigma, inner_sigma))),
                half_output=True)

    image = root / "volume-rejected.pfm"
    for name in ("volume_bad_extinction", "volume_bad_anisotropy", "volume_capacity"):
        out = run([renderer, "--hart", "-v", "--hart-bounces", "12"] + flags
                  + common + [name + ".xml", str(image)], root, 1)
        assert "error bits 32" in out and "invalid closure tree" in out, out
        assert "HART path tracer rendered" not in out and not image.exists(), out
        out = run([renderer, "--max-bounces", "12"] + common
                  + [name + ".xml", str(image)], root, 1)
        operation = "medium entry" if name == "volume_capacity" else "surface closure"
        assert "SEVERE ERROR: Invalid " + operation in out, out
        assert not image.exists(), out


def check_native_benchmark(compile_ms):
    iterations, trials, aa, bounces = 20, 3, 4, 1
    modes = ("split", "fused", "fused-local")
    first = modes.index(mode)
    modes = modes[first:] + modes[:first]

    def ranges(values):
        return {"median": statistics.median(values),
                "range": [min(values), max(values)], "trials": values}

    def statistics_for(output, trial_mode, count, warmup, no_cache):
        def one(pattern):
            rows = re.findall("^" + pattern + r"\r?$", output, re.MULTILINE)
            assert len(rows) == 1, output
            return rows[0]

        osl_ms, pipeline_ms = map(float, one(
            r"HART native preparation: (\S+) ms OSL groups, (\S+) ms pipeline"))
        cache_policy = one(
            r"HART native pipeline cache: (disabled|enabled) "
            r"\(SDK policy; hit status unavailable\)")
        assert cache_policy == ("disabled" if no_cache else "enabled"), output
        launch_count, launch_total = one(
            r"HART native synchronized launches: (\d+) launches, (\S+) ms total")
        launch_count, launch_total = int(launch_count), float(launch_total)
        owned, stride, budget = map(int, one(
            r"HART native memory: (\d+) context-owned bytes, "
            r"(\d+) groupdata stride bytes, (\d+) local budget bytes"))
        traversal, state, continuation = map(int, one(
            r"HART native stack estimates: traversal (\d+) bytes, "
            r"state (\d+) bytes, continuation (\d+) bytes"))
        frame_count, total, mean, warmup_ms = one(
            r"HART native frames: (\d+) iterations, (\S+) ms total, "
            r"(\S+) ms mean, (\S+) ms warmup")
        total, mean, warmup_ms = map(float, (total, mean, warmup_ms))
        assert int(frame_count) == count and count > 0, output
        # This furnace has no background prepass: one launch per timed frame.
        assert launch_count == count, output
        assert all(math.isfinite(value) and value >= 0 for value in
                   (osl_ms, pipeline_ms, launch_total, total, mean, warmup_ms)), output
        assert math.isclose(total / count, mean, abs_tol=1e-6, rel_tol=0), output
        if not warmup:
            assert warmup_ms == 0, output
        assert owned > 0 and budget == (4096 if trial_mode == "fused-local" else 0)
        assert (stride == 0) == (trial_mode == "fused-local"), output
        storage = re.findall(r"(\d+) caller Groupdata bytes per pixel", output)
        assert storage == [str(stride)], output
        compiled = re.findall(r"HART compiled (\d+) materials, (\d+) callables", output)
        assert compiled == [("2", "4" if trial_mode == "split" else "2")], output
        marker = (f"HART path tracer rendered {width}x{height} with "
                  f"{aa * aa} samples per pixel")
        assert output.count(marker) == count + int(warmup), output
        return {
            "osl_groups_ms": osl_ms, "pipeline_ms": pipeline_ms,
            "synchronized_launch_total_ms": launch_total,
            "synchronized_launch_mean_ms": launch_total / launch_count,
            "frames_total_ms": total, "frame_mean_ms": mean, "warmup_ms": warmup_ms,
            "iterations": count, "warmup_frames": int(warmup),
            "cache_policy": cache_policy, "cache_hit_status": "unavailable",
            "timed_launches": launch_count, "context_owned_bytes": owned,
            "groupdata_stride_bytes": stride, "local_budget_bytes": budget,
            "stack_estimate_bytes": {"traversal": traversal, "state": state,
                                     "continuation": continuation},
        }

    cpu_image = root / "benchmark-cpu.pfm"
    run([renderer] + common + ["-O2", "--llvm_opt", "3", "-aa", str(aa),
                              "--max-bounces", str(bounces), "furnace.xml",
                              str(cpu_image)], root)
    cpu = pixels(cpu_image)
    expected = [struct.unpack("e", struct.pack("e", value))[0]
                for value in (0.4, 0.4, 0.3)] * (width * height)
    compare(cpu, expected)
    assert max(cpu) > 0
    samples = {trial_mode: [] for trial_mode in modes}
    storage_by_mode = {}
    stacks_by_mode = {}
    scratch_stride = None
    baseline = None
    image = root / "benchmark-gpu.pfm"

    def sample(trial_mode, count, warmup, no_cache):
        nonlocal baseline, scratch_stride
        if image.exists():
            image.unlink()
        arguments = ([renderer] + common
                     + ["--hart", "-v", "-O2", "--runstats", "--iters", str(count),
                        "-aa", str(aa), "--hart-bounces", str(bounces)]
                     + mode_flags[trial_mode])
        if warmup:
            arguments += ["--warmup"]
        if no_cache:
            arguments += ["--hart-no-cache"]
        wall = []
        output = run(arguments + ["furnace.xml", str(image)], root, wall_times=wall)
        record = statistics_for(output, trial_mode, count, warmup, no_cache)
        record["process_wall_ms"] = wall[0]
        storage = (record["context_owned_bytes"], record["groupdata_stride_bytes"],
                   record["local_budget_bytes"])
        stacks = record["stack_estimate_bytes"]
        if trial_mode in storage_by_mode:
            assert storage == storage_by_mode[trial_mode], (trial_mode, storage)
            assert stacks == stacks_by_mode[trial_mode], (trial_mode, stacks)
        storage_by_mode[trial_mode] = storage
        stacks_by_mode[trial_mode] = stacks
        if trial_mode != "fused-local":
            if scratch_stride is not None:
                assert record["groupdata_stride_bytes"] == scratch_stride
            scratch_stride = record["groupdata_stride_bytes"]
        actual = pixels(image)
        assert max(actual) > 0
        # Isolated stochastic image differences are diagnostic, not a pixel-
        # equality gate. Preserve the furnace's energy and radiance bounds.
        mean_rgb = [statistics.mean(actual[channel::3]) for channel in range(3)]
        compare(mean_rgb, expected[:3])
        assert all(0 <= value <= expected[i % 3] + 3e-5
                   for i, value in enumerate(actual)), trial_mode
        if baseline is None:
            baseline = actual
        compare(mean_rgb, [statistics.mean(baseline[channel::3])
                           for channel in range(3)])
        record["image_diagnostics"] = {
            "mean_rgb": mean_rgb,
            "different_cpu_pixels": sum(
                actual[i:i+3] != cpu[i:i+3] for i in range(0, len(actual), 3)),
            "cpu_max_abs_error": max(abs(a - b) for a, b in zip(actual, cpu)),
            "cpu_relative_rmse": math.sqrt(
                sum((a - b) ** 2 for a, b in zip(actual, cpu))
                / sum(value ** 2 for value in cpu)),
            "baseline_max_abs_error": max(abs(a - b)
                                         for a, b in zip(actual, baseline)),
        }
        return record

    # Cache policy is observable, but the native API exposes no hit status.
    # Neither this sample nor priming flushes OS or driver caches.
    cold = {trial_mode: sample(trial_mode, 1, False, True) for trial_mode in modes}
    priming = {trial_mode: sample(trial_mode, 1, False, False) for trial_mode in modes}
    for trial in range(trials):
        for trial_mode in modes[trial:] + modes[:trial]:
            samples[trial_mode].append(sample(trial_mode, iterations, True, False))

    for trial_mode in modes:
        owned, stride, budget = storage_by_mode[trial_mode]
        print(json.dumps({
            "benchmark": "hart-native", "scene": "furnace",
            "mode": "fused-scratch" if trial_mode == "fused" else trial_mode,
            "cli_mode": trial_mode, "resolution": [width, height],
            "samples_per_pixel": aa * aa, "max_bounces": bounces,
            "osl_opt": 2, "llvm_opt": 3, "warmup_frames": 1,
            "iterations": iterations, "trials": trials,
            "timed_launches_per_trial": iterations,
            "trial_mode_order": [modes[i:] + modes[:i] for i in range(trials)],
            "cold_no_cache": cold[trial_mode],
            "cache_priming_outside_trials": priming[trial_mode],
            "cache_policy": "enabled", "cache_hit_status": "unavailable",
            "cold_scope": "HART SDK pipeline cache disabled; OS/driver caches untouched",
            "oslc_process_wall_ms": {
                "scope": "sum of source compiler subprocesses; includes startup, "
                         "separate from in-process OSL group preparation",
                "sources": list(compile_ms), "oslc_opt": 1,
                **ranges([sum(values[i] for values in compile_ms.values())
                          for i in range(trials)]),
            },
            "timings_ms": {
                key: ranges([record[key] for record in samples[trial_mode]])
                for key in ("osl_groups_ms", "pipeline_ms",
                            "synchronized_launch_total_ms", "synchronized_launch_mean_ms",
                            "frames_total_ms", "frame_mean_ms", "warmup_ms",
                            "process_wall_ms")
            },
            "timing_scope": "OSL groups include compile_materials and binding preparation, "
                            "not source parsing; pipeline includes modules, groups, "
                            "link, deferred stack-query compilation and SBT setup, "
                            "not context initialization or acceleration build; "
                            "synchronized launches include hartLaunch plus stream wait, "
                            "not uploads, clears, readbacks or warmup; frames include "
                            "full render/allocation/bindings/background/readback/"
                            "publication, not warmup or final disk write; process "
                            "includes all setup, warmup, frames, disk IO and teardown",
            "context_owned_bytes": owned, "groupdata_stride_bytes": stride,
            "local_budget_bytes": budget,
            "stack_estimate_bytes": stacks_by_mode[trial_mode],
            "image_diagnostics_by_trial": [
                record["image_diagnostics"] for record in samples[trial_mode]],
            "image_acceptance": "finite bounded furnace radiance and analytic mean "
                                "energy; per-pixel differences are diagnostics, "
                                "visually qualified with identical display conversion",
            "memory_scope": "native context ownership excludes textures, group-owned "
                            "interactive buffers and SDK/driver allocations; "
                            "logical groupdata and SDK stack estimates are not "
                            "physical VRAM, register or spill measurements",
        }, allow_nan=False), flush=True)


with tempfile.TemporaryDirectory(prefix="osl-hart-path-") as temporary:
    root = Path(temporary)
    if materials:
        check_materials()
        print("HART material path tracer verified: " + mode)
        sys.exit(0)
    if lighting:
        check_lighting()
        print("HART lighting path tracer verified: " + mode)
        sys.exit(0)
    if volumes:
        check_volumes()
        print("HART volume path tracer verified: " + mode)
        sys.exit(0)
    if textures:
        check_textures()
        print("HART texture path tracer verified: " + mode)
        sys.exit(0)

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
    compile_ms = {}
    for name, source in shaders.items():
        if benchmark and name not in ("path_emit", "path_diffuse"):
            continue
        (root / (name + ".osl")).write_text(source, encoding="ascii")
        if benchmark:
            compile_ms[name] = []
        else:
            run([compiler, "-I" + str(Path(stdosl).parent), name + ".osl"], root)
    if benchmark:
        for trial in range(3):
            names = list(compile_ms)
            for name in names[trial:] + names[:trial]:
                run([compiler, "-O1", "-I" + str(Path(stdosl).parent), name + ".osl"],
                    root, wall_times=compile_ms[name])
    else:
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
    if benchmark:
        (root / "furnace.xml").write_text(furnace, encoding="ascii")
        check_native_benchmark(compile_ms)
        print("HART native benchmark correctness and timing records passed; "
              "no performance thresholds")
        sys.exit(0)
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

    cpu = render("emission", False)
    gpu = render("emission", True, repeat=True)
    compare(gpu, cpu)
    environment_image = root / "emission-environment.pfm"
    environment = {"TESTSHADE_HART": "1",
                   "TESTSHADE_FUSED": str(int(mode in ("fused", "fused-local")))}
    output = run([renderer, "-v"] + common
                 + [flag for flag in flags if flag != "--hart-fused"]
                 + ["--hart-bounces", "1", "emission.xml", str(environment_image)],
                 root, extra_env=environment)
    compare(pixels(environment_image), cpu)
    compiled = re.search(r"HART compiled (\d+) materials, (\d+) callables", output)
    entries = 1 if mode in ("fused", "fused-local") else 2
    assert compiled and int(compiled[2]) == int(compiled[1]) * entries, output
    assert "HART path tracer rendered" in output, output
    conflict_image = root / "environment-conflict.pfm"
    output = run([renderer, "emission.xml", str(conflict_image)], root, 1,
                 {**environment, "TESTSHADE_OPTIX": "1"})
    assert "Invalid HART options" in output and not conflict_image.exists(), output
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
    assert "HART device services failed (error bits 32)" in out, out
    assert "invalid closure tree" in out, out
    assert "HART path tracer rendered" not in out and not image.exists(), out
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
