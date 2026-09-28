# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

"""Exercise installed tools, optionally HART, without source/build runtime paths.

Build install-consumer separately with ordinary CMake, then pass its executable
with --consumer to exercise the installed C++ compile/query/CPU execution APIs.
No inherited SDK/loader paths or OSL/OIIO options are used. Only explicit OIIO
and CUDA dependency runtime directories may supplement the installation.
GPU checks require an explicit HIP SDK root for cold compiler resources;
its bin directory is not added to the loader path.
This relocation check requires prefix/share/OSL/shaders/stdosl.h; custom
external shader directories are not fallback candidates.
"""

import argparse
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def within(path, directory):
    return path == directory or directory in path.parents


def installed_file(path, prefix):
    resolved = path.resolve(strict=True)
    require(resolved.is_file() and within(resolved, prefix),
            f"Not a file inside the installation: {path} -> {resolved}")
    return resolved


def runtime_role(name):
    match = re.fullmatch(r"(.+?)(?:\.dll|\.dylib|\.so(?:\.\d+)*)", name.lower())
    if not match:
        return None
    stem = re.sub(r"[._-]", "", match[1])
    if stem.startswith("lib"):
        stem = stem[3:]
    if "shadercompiler" in stem:
        return "shader compiler"
    if "shaderstackruntime" in stem:
        return "shader stack runtime"
    if stem.startswith("amdhart"):
        return "HART"
    if stem.startswith("amdhip"):
        return "HIP"
    if stem.startswith("hiprtcbuiltins"):
        return "HIPRTC builtins"
    if stem.startswith("hiprtc"):
        return "HIPRTC"
    if stem.startswith("amdcomgr"):
        return "COMGR"
    if stem.startswith("rocmkpack"):
        return "ROCm kpack"
    return None


def pe_imports(path):
    """Read normal/delay import names without dumpbin, an SDK, or new packages."""
    with path.open("rb") as stream:
        def read(offset, size):
            require(offset >= 0, f"Invalid PE offset in {path}")
            stream.seek(offset)
            data = stream.read(size)
            require(len(data) == size, f"Truncated PE file: {path}")
            return data

        def unpack(fmt, offset):
            return struct.unpack(fmt, read(offset, struct.calcsize(fmt)))

        require(read(0, 2) == b"MZ", f"Not a Windows executable: {path}")
        pe = unpack("<I", 0x3c)[0]
        require(read(pe, 4) == b"PE\0\0", f"Invalid PE signature: {path}")
        sections = unpack("<H", pe + 6)[0]
        optional_size = unpack("<H", pe + 20)[0]
        optional = pe + 24
        magic = unpack("<H", optional)[0]
        require(magic in (0x10b, 0x20b), f"Unsupported PE format: {path}")
        directories = 112 if magic == 0x20b else 96
        require(optional_size >= directories and 0 < sections <= 96,
                f"Invalid PE headers: {path}")
        image_base = unpack("<Q" if magic == 0x20b else "<I",
                            optional + (24 if magic == 0x20b else 28))[0]
        count = unpack("<I", optional + directories - 4)[0]
        require(directories + min(count, 14) * 8 <= optional_size,
                f"Truncated PE data directories: {path}")
        ranges = [unpack("<IIII", optional + optional_size + i * 40 + 8)
                  for i in range(sections)]

        def offset(rva, size):
            for _, address, raw_size, raw_offset in ranges:
                if address <= rva and rva - address + size <= raw_size:
                    return raw_offset + rva - address
            raise RuntimeError(f"Unmapped PE import RVA {rva} in {path}")

        names = set()
        for directory, stride in ((1, 20), (13, 32)):
            if count <= directory:
                continue
            rva, size = unpack("<II", optional + directories + directory * 8)
            if not rva:
                continue
            require(stride <= size <= 1024 * 1024,
                    f"Invalid PE import directory size in {path}")
            terminated = False
            for index in range(size // stride):
                entry = unpack("<" + "I" * (stride // 4),
                               offset(rva + index * stride, stride))
                if not any(entry):
                    terminated = True
                    break
                if directory == 13:
                    require(entry[0] in (0, 1),
                            f"Unsupported delay-import attributes in {path}")
                    name_rva = entry[1] - (0 if entry[0] else image_base)
                else:
                    name_rva = entry[3]
                name = bytearray()
                for char in range(256):
                    value = read(offset(name_rva + char, 1), 1)
                    if value == b"\0":
                        break
                    name.extend(value)
                else:
                    raise RuntimeError(f"Unterminated PE import name in {path}")
                require(name and b"/" not in name and b"\\" not in name,
                        f"Invalid PE import name in {path}: {name!r}")
                names.add(name.decode("ascii").lower())
            require(terminated, f"Unterminated PE import directory in {path}")
        return names


def check_windows_runtime(prefix, tools):
    bindir = prefix / "bin"
    local = {p.name.lower(): p for p in bindir.iterdir() if p.is_file()}
    roles = ("HART", "shader compiler", "shader stack runtime", "HIP",
             "HIPRTC", "HIPRTC builtins", "COMGR")
    pending = [tools["testshade"], tools["testrender"]]
    for role in roles:
        matches = [p for name, p in local.items() if runtime_role(name) == role]
        require(matches, f"Missing app-local {role} DLL in {bindir}")
        pending.extend(matches)
    worker = bindir / "hart-native-codegen.exe"
    require(worker.is_file(), f"Missing app-local HART compiler worker: {worker}")
    pending.append(worker)
    visited = set()
    while pending:
        path = installed_file(pending.pop(), prefix)
        require(path.parent == bindir,
                f"Windows GPU runtime must be app-local, not redirected: {path}")
        if path in visited:
            continue
        visited.add(path)
        for name in pe_imports(path):
            # Follow exact imported version names, including optional kpack
            # when this deployment's HIP runtime imports it.
            if runtime_role(name):
                require(name in local,
                        f"{path.name} imports missing app-local runtime {name}")
            if name in local:
                pending.append(local[name])
    print(f"PASS: Windows app-local runtime/worker files and imports "
          f"({len(visited)} PE files checked)", flush=True)


def runtime_environment(prefix, dependency_dirs, hip_root=None):
    keep = {"SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT",
            "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
            "TEMP", "TMP", "TMPDIR"}
    env = {key: value for key, value in os.environ.items() if key.upper() in keep}
    dependencies = []
    for directory in dependency_dirs:
        path = directory.resolve(strict=True)
        require(path.is_dir(), f"Not a dependency runtime directory: {path}")
        for file in path.iterdir():
            name = file.name.lower()
            require(not runtime_role(name) and name != "hart-native-codegen.exe"
                    and not re.match(r"(?:lib)?osl(?:comp|exec|query)", name)
                    and name not in ("oslc", "oslc.exe", "oslinfo", "oslinfo.exe",
                                     "testshade", "testshade.exe", "testrender",
                                     "testrender.exe"),
                    f"Dependency directory contains OSL/HART/HIP files: {file}")
        dependencies.append(str(path))
    if os.name == "nt":
        windows = Path(os.environ["SystemRoot"])
        path = [str(prefix / "bin"), *dependencies,
                str(windows / "System32"), str(windows)]
    else:
        path = [str(prefix / "bin"), *os.defpath.split(os.pathsep)]
        if dependencies:
            loader = ("DYLD_LIBRARY_PATH" if sys.platform == "darwin"
                      else "LD_LIBRARY_PATH")
            env[loader] = os.pathsep.join(dependencies)
    env["PATH"] = os.pathsep.join(path)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    if hip_root is not None:
        hip_root = hip_root.resolve(strict=True)
        require((hip_root / "include" / "hip" / "hip_runtime.h").is_file(),
                f"HIP SDK root lacks include/hip/hip_runtime.h: {hip_root}")
        env["HIP_PATH"] = str(hip_root)
    for name in ("TESTSHADE_OPTIX", "TESTSHADE_HART", "TESTSHADE_FUSED",
                 "TESTSHADE_BATCHED", "TESTSHADE_RS_BITCODE"):
        env[name] = "0"
    return env


def check_working_directory(root, prefix):
    require(not within(root, prefix),
            f"Temporary cwd is inside the install: {root}")
    for parent in (root, *root.parents):
        require(not (parent / ".git").exists()
                and not (parent / "CMakeCache.txt").exists()
                and not (parent / "src" / "include" / "OSL" / "oslexec.h").exists(),
                f"Temporary cwd is inside a source/build tree: {root}")


def run(command, root, env, timeout=300):
    command = [str(arg) for arg in command]
    print("Running: " + repr(command), flush=True)
    try:
        result = subprocess.run(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if error.stdout:
            output = error.stdout
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            print(output, end="", flush=True)
        raise
    print(result.stdout, end="", flush=True)
    require(result.returncode == 0,
            f"Installed command failed ({result.returncode}): {command}")
    require(not re.search(r"^(?:ERROR|SEVERE)(?::|\s)", result.stdout, re.M),
            f"Installed command reported an error: {command}")
    return result.stdout


def check_grid(output):
    rows = re.findall(
        r"Pixel \((\d+), (\d+)\):\s+Cout\s*[:=]\s+(\S+) (\S+) (\S+)", output)
    require(len(rows) == 9, f"Expected nine RGB pixel records, found {len(rows)}")
    for index, row in enumerate(rows):
        x, y = index % 3, index // 3
        require(tuple(map(int, row[:2])) == (x, y),
                f"Unexpected pixel order: {row}")
        actual = tuple(map(float, row[2:]))
        expected = (x + 0.125, y + 0.25, 2.5)
        require(all(math.isfinite(v) for v in actual) and actual == expected,
                f"Pixel ({x}, {y}): {actual}, expected exactly {expected}")


def check_emission_image(path):
    with path.open("rb") as stream:
        require(stream.readline().strip() == b"PF", f"Not an RGB PFM: {path}")
        require(stream.readline().split() == [b"8", b"4"],
                f"Expected an 8x4 PFM: {path}")
        scale = float(stream.readline())
        require(math.isfinite(scale) and abs(scale) == 1,
                f"Unexpected PFM scale {scale}: {path}")
        data = stream.read()
    require(len(data) == 8 * 4 * 3 * 4, f"Wrong PFM payload size: {path}")
    values = struct.unpack(("<" if scale < 0 else ">") + "96f", data)
    # Constant emission is also exact after testrender's HALF conversion;
    # scanline orientation cannot change this independent per-pixel oracle.
    for index, value in enumerate(values):
        expected = (0.125, 0.25, 0.5)[index % 3]
        require(math.isfinite(value) and value == expected,
                f"PFM component {index}: {value}, expected exactly {expected}")


def check_tools(tools, stdosl, root, env, gpu, consumer):
    shader = root / "installed_probe.osl"
    shader.write_text("""shader installed_probe(
    float gain = 2, output color Cout = 0) {
    Cout = color(gain*u + 0.125, gain*v + 0.25, gain + 0.5);
}
""", encoding="ascii")
    run([tools["oslc"], "-I" + str(stdosl.parent), "-o", "installed_probe.oso",
         shader], root, env)
    require((root / "installed_probe.oso").is_file(), "oslc produced no bytecode")
    info = run([tools["oslinfo"], "installed_probe.oso"], root, env)
    lines = [" ".join(line.split()) for line in info.splitlines() if line.strip()]
    require(lines == ['shader "installed_probe"', "float gain 2",
                      "output color Cout [ 0 0 0 ]"],
            f"Unexpected installed oslinfo metadata: {lines}")
    grid = ["-g", "3", "3", "--print", "-o", "Cout", "null", "installed_probe"]
    cpu = run([tools["testshade"], "-t", "1", "-O2", "--llvm_opt", "3", *grid],
              root, env)
    require("Launching HART" not in cpu and "HART callable mode" not in cpu,
            "The CPU control selected HART")
    check_grid(cpu)
    print("PASS: installed oslc, oslinfo and CPU testshade exact values", flush=True)
    if consumer:
        output = run([consumer, stdosl], root, env)
        require(output.splitlines().count(
                    "Installed OSL consumer: result=1.375") == 1,
                "Consumer did not confirm compile/query/CPU execution")
        print("PASS: installed C++ consumer", flush=True)
    else:
        print("NOT RUN: C++ consumer (supply --consumer after its separate build)",
              flush=True)
    if not gpu:
        print("NOT RUN: GPU validation (--gpu was not requested)", flush=True)
        return
    for mode in ("split", "fused"):
        flags = ["--hart-fused"] if mode == "fused" else []
        output = run([tools["testshade"], "--hart", "--hart-no-cache", "-v",
                      "-O2", "--llvm_opt", "3", *flags, *grid], root, env, 600)
        require(output.count("Launching HART grid 3 x 3") == 1
                and f"HART callable mode: {mode}" in output
                and "HART pipeline cache disabled" in output,
                f"Missing actual generated HART {mode} launch evidence")
        check_grid(output)
        print(f"PASS: installed generated HART {mode} exact values", flush=True)
    (root / "installed_emit.osl").write_text(
        "shader installed_emit() { Ci=color(0.125,0.25,0.5)*emission(); }\n",
        encoding="ascii")
    run([tools["oslc"], "-I" + str(stdosl.parent), "-o", "installed_emit.oso",
         "installed_emit.osl"], root, env)
    (root / "installed_emit.xml").write_text("""<World>
  <Camera eye="0,0,4" dir="0,0,-1" fov="90"/>
  <ShaderGroup name="emitter" is_light="0">
    shader installed_emit m;
  </ShaderGroup>
  <Quad corner="-10,-10,0" edge_x="20,0,0" edge_y="0,20,0"/>
</World>
""", encoding="ascii")
    image = root / "installed_emit.pfm"
    output = run([tools["testrender"], "--hart", "-v", "-O2", "--llvm_opt", "3",
                  "--res", "8", "4", "-t", "1", "-aa", "1", "--no-jitter",
                  "--hart-bounces", "0", "installed_emit.xml", image],
                 root, env, 600)
    require("HART compiled 1 materials, 2 callables" in output
            and "HART path tracer rendered 8x4 with 1 samples per pixel" in output,
            "Missing actual native HART render evidence")
    check_emission_image(image)
    print("PASS: installed native HART emission, exact finite PFM values",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, help="OSL installation prefix")
    parser.add_argument("--gpu", action="store_true",
                        help="Require actual HART split/fused grids and native rendering")
    parser.add_argument("--oiio-runtime-dir", action="append", type=Path, default=[],
                        help="Explicit OIIO/dependency DLL or shared-library directory")
    parser.add_argument("--cuda-runtime-dir", action="append", type=Path, default=[],
                        help="Explicit CUDA runtime directory for mixed builds")
    parser.add_argument("--hip-root", type=Path,
                        help="HIP SDK root for cold compilation (required with --gpu); "
                             "does not add SDK binaries to PATH")
    parser.add_argument("--consumer", type=Path,
                        help="Separately built install-consumer executable; never built here")
    parser.add_argument("--temp-root", type=Path,
                        help="Temporary parent directory outside source/build/install trees")
    args = parser.parse_args()
    require(not args.gpu or args.hip_root is not None,
            "GPU validation requires --hip-root for cold HART compiler resources")
    prefix = args.prefix.resolve(strict=True)
    require(prefix.is_dir(), f"Not an installation directory: {prefix}")
    suffix = ".exe" if os.name == "nt" else ""
    names = ["oslc", "oslinfo", "testshade"] + (["testrender"] if args.gpu else [])
    tools = {name: installed_file(prefix / "bin" / (name + suffix), prefix)
             for name in names}
    stdosl = installed_file(
        prefix / "share" / "OSL" / "shaders" / "stdosl.h", prefix)
    consumer = args.consumer.resolve(strict=True) if args.consumer else None
    if consumer:
        require(consumer.is_file(), f"Not a consumer executable: {consumer}")
    env = runtime_environment(prefix, args.oiio_runtime_dir + args.cuda_runtime_dir,
                              args.hip_root)
    if args.gpu and os.name == "nt":
        check_windows_runtime(prefix, tools)
    with tempfile.TemporaryDirectory(prefix="osl-installed-",
                                     dir=args.temp_root) as temp:
        root = Path(temp).resolve()
        check_working_directory(root, prefix)
        for key in ("TEMP", "TMP", "TMPDIR"):
            env[key] = str(root)
        print(f"Install: {prefix}\nStandard library: {stdosl}\nTemporary cwd: {root}",
              flush=True)
        check_tools(tools, stdosl, root, env, args.gpu, consumer)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: installed-tree validation: {error}", file=sys.stderr)
        sys.exit(1)
