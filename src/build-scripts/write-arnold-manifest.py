#!/usr/bin/env python3
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

"""Record the configured Arnold profile and actual CMake transitive link lines."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def header_define(path, name):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^#\s*define\s+" + name + r"\s+(\S+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {name} in {path}")
    return match[1].strip('"')


def binary_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compiler_version(executable):
    path = Path(executable).resolve(strict=True)
    return {
        "path": str(path),
        "version": subprocess.check_output([str(path), "--version"], text=True).strip(),
    }


def deployed_runtime(build, filename_pattern):
    # CMake has resolved imported target locations in these deployment rules.
    # Do not infer the runtime's identity from a nearby source checkout.
    paths = set()
    for script in build.glob("src/*/cmake_install.cmake"):
        for name in re.findall(r'"([^"\n]+\.dll)"',
                               script.read_text(encoding="utf-8")):
            if re.fullmatch(filename_pattern, Path(name).name, re.IGNORECASE):
                paths.add(Path(name).resolve(strict=True))
    if len(paths) != 1:
        raise ValueError(f"Expected one deployed {filename_pattern}, found {paths}")
    path = paths.pop()
    return {"path": str(path), "sha256": binary_sha256(path)}


def write_manifest(build):
    cache = {}
    for line in (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^/#][^:]*):[^=]*=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    llvm_root = Path(cache["LLVM_DIRECTORY"]).resolve()
    for key, value in cache.items():
        if key.startswith("_CLANG_") and key.endswith("_LIBRARY"):
            if not value.endswith("-NOTFOUND") and llvm_root not in Path(value).resolve().parents:
                raise ValueError(f"{key} is outside the selected OSL LLVM: {value}")
    llvm_version = subprocess.check_output(
        [cache["LLVM_CONFIG"], "--version"], text=True
    ).strip()
    if not llvm_version.startswith("23."):
        raise ValueError(f"The Arnold profile requires OSL LLVM 23, not {llvm_version}")

    reply = build / ".cmake" / "api" / "v1" / "reply"
    index = json.loads(max(reply.glob("index-*.json"), key=lambda p: p.stat().st_mtime)
                       .read_text(encoding="utf-8"))
    model = json.loads((reply / index["reply"]["codemodel-v2"]["jsonFile"])
                       .read_text(encoding="utf-8"))
    targets = {}
    for config in model["configurations"]:
        if config["name"] != "Release":
            continue
        for entry in config["targets"]:
            target = json.loads((reply / entry["jsonFile"]).read_text(encoding="utf-8"))
            if target["type"] in ("EXECUTABLE", "SHARED_LIBRARY", "MODULE_LIBRARY"):
                targets[target["name"]] = {
                    "type": target["type"],
                    "build_directory": str(build / target["paths"]["build"]),
                    "link": target.get("link", {}).get("commandFragments", []),
                }
                for fragment in targets[target["name"]]["link"]:
                    library = fragment["fragment"].strip('"').replace("\\", "/")
                    if (fragment["role"] == "libraries"
                            and re.match(r"(LLVM|clang).*\.lib$", library.split("/")[-1])
                            and llvm_root not in Path(library).resolve().parents):
                        raise ValueError(f"Mixed LLVM in {target['name']}: {library}")
    selected = {
        key: value for key, value in cache.items()
        if key.startswith(("LLVM_", "_CLANG_", "HART_", "ROCM_", "CUDA_",
                           "OPTIX_", "OSL_USE_", "JPEG_", "OpenImageIO_",
                           "Imath_", "CMAKE_MSVC_", "CMAKE_CXX_COMPILER",
                           "OSL_ARNOLD_"))
        or key.endswith("_DIR")
    }
    hip_header = Path(cache["ROCM_ROOT"]) / "include" / "hip" / "hip_version.h"
    jpeg_header = Path(cache["JPEG_INCLUDE_DIR"]) / "jconfig.h"
    cuda = json.loads((Path(cache["CUDA_TOOLKIT_ROOT_DIR"]) / "version.json")
                      .read_text(encoding="utf-8"))
    hart_version = (Path(cache["amd.hart_DIR"]) / "amd.hart-config-version.cmake")
    hart_version = re.search(r'set\(PACKAGE_VERSION "([^"]+)"',
                             hart_version.read_text(encoding="utf-8"))[1]
    manifest = {
        "profile": "Arnold static /MD; HART and OptiX in separate ShadingSystems",
        "llvm_version": llvm_version,
        "compiler_builds": {
            "osl_clang": compiler_version(cache["LLVM_BC_GENERATOR"]),
            "rocm_clang": compiler_version(cache["ROCM_CLANG_EXECUTABLE"]),
        },
        "runtime_binaries": {
            "selection": "Installed SDK files selected by CMake's generated "
                         "Release deployment rules; not source checkout identities.",
            "hart": deployed_runtime(build, r"amd\.hart\.dll"),
            "hip": deployed_runtime(build, r"amdhip64(?:_\d+)?\.dll"),
        },
        "sdk_versions": {
            "cuda": cuda,
            "optix": header_define(Path(cache["OPTIX_INCLUDE_DIR"]) / "optix.h",
                                   "OPTIX_VERSION"),
            "hart": hart_version,
            "hip": ".".join(header_define(hip_header, "HIP_VERSION_" + part)
                            for part in ("MAJOR", "MINOR", "PATCH")),
            "hip_git_hash": header_define(hip_header, "HIP_VERSION_GITHASH"),
            "jpeg": header_define(jpeg_header, "LIBJPEG_TURBO_VERSION"),
            "jpeg_abi": header_define(jpeg_header, "JPEG_LIB_VERSION"),
            "jpeg_sha256": binary_sha256(Path(cache["JPEG_LIBRARY_RELEASE"])),
        },
        "configuration": selected,
        "targets": targets,
        "note": "Link fragments are from CMake's generated Release codemodel, "
                "including the complete ordered transitive library lists. "
                "SDK paths refer to this machine; dependencies are not bundled.",
    }
    destination = build / "arnold-dependencies.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded actual target link requirements in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    write_manifest(parser.parse_args().build.resolve(strict=True))
