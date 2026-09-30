#!/usr/bin/env python3
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage Arnold's binary OIIO/Imath packages with HIP-only header backports."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil


PACKAGES = {
    "openimageio": Path("openimageio") / "autodesk-arnold-2.6.3.2-1-3",
    "imath": Path("imath") / "3.1.10-9",
}
HEADER_HASHES = {
    "openimageio/include/OpenImageIO/platform.h":
        "56f38173c9eb4713df3262b80d1f5154c82fcd826a9dfb2bc786f563c7416163",
    "openimageio/include/OpenImageIO/simd.h":
        "27fb3a786f58f46414c8a4b992243995d6c3c933ea2bfc888038842afae52b1f",
    "openimageio/include/OpenImageIO/fmath.h":
        "cc7a640857964cfec6cf5d73812b9fce0095cc0467bbc95b9a75117bcde232bd",
    "openimageio/include/OpenImageIO/ustring.h":
        "c5aaf93cf0027be7decbefbb5c8105cbd534d9ca286491fee5b16516fd84dc0e",
    "openimageio/include/OpenImageIO/bit.h":
        "aa42da78365813d3bd89896347950392d5a16f54d07f6a672836ad8a007383c6",
    "openimageio/include/OpenImageIO/color.h":
        "21eefdc022b4cc578f54e55c6014fe2a1590d0064b532ef6b349844cdaa0a55a",
    "openimageio/include/OpenImageIO/detail/farmhash.h":
        "da255e9c2c8e3ecb4d43331b4e4dab430f4ab87f92fd33cf5c4eaf6437288af6",
    "openimageio/include/OpenImageIO/detail/fmt.h":
        "942939932c31a6891282c5b1e886003aba4981262a635de0b79efe163d0979ce",
    "imath/include/Imath/ImathConfig.h":
        "8650356a0ec689f7b07edeff33196729fd5078a8221ca3bc67c3cb76b46c5197",
    "imath/include/Imath/half.h":
        "4e54aaddc57efa14bc45f72e1e094536f674422c64df94dc44541fb92356f158",
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch anchor: {old!r}")
    return text.replace(old, new)


def device_guards(text):
    # Parentheses preserve negation and precedence in compound conditions.
    text = re.sub(r"#ifdef __CUDA_ARCH__", "#if defined(__CUDA_ARCH__)", text)
    text = re.sub(r"#ifndef __CUDA_ARCH__", "#if !defined(__CUDA_ARCH__)", text)
    return text.replace(
        "defined(__CUDA_ARCH__)",
        "(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))",
    )


def patch_header(name, data):
    text = data.decode("utf-8").replace("\r\n", "\n")
    if name.endswith("platform.h"):
        text = replace_once(
            text, "#ifdef __CUDACC__",
            "#if defined(__CUDACC__) || defined(__HIP__)",
        )
        text = device_guards(text)
    elif name.endswith("bit.h"):
        text = text.replace(
            "!defined(__CUDACC__)",
            "!defined(__CUDACC__) && !defined(__HIP__)",
        )
        text = replace_once(
            text, "    memcpy((void*)&result, &from, sizeof(From));",
            "#if defined(__HIP_DEVICE_COMPILE__)\n"
            "    __builtin_memcpy((void*)&result, &from, sizeof(From));\n"
            "#else\n    memcpy((void*)&result, &from, sizeof(From));\n#endif",
        )
    elif name.endswith(("simd.h", "fmath.h", "half.h", "ustring.h",
                        "color.h", "farmhash.h", "fmt.h")):
        text = device_guards(text)
        if name.endswith("fmath.h"):
            # HIP uses libm, not CUDA's approximate intrinsics. The CPU and
            # CUDA branches remain unchanged, including their domain handling.
            for function, argument in (
                ("sin", "x"), ("cos", "x"), ("tan", "x"), ("log2", "xval"),
                ("log", "x"), ("log10", "x"), ("exp", "x"), ("exp10", "x"),
            ):
                original = f"return __{function}f({argument});"
                text = replace_once(
                    text, original,
                    f"#if defined(__HIP_DEVICE_COMPILE__)\n"
                    f"    return {function}f({argument});\n"
                    f"#else\n    {original}\n#endif",
                )
            text = replace_once(
                text, "__sincosf(x, sine, cosine);",
                "#if defined(__HIP_DEVICE_COMPILE__)\n"
                "    sincosf(x, sine, cosine);\n"
                "#else\n    __sincosf(x, sine, cosine);\n#endif",
            )
    elif name.endswith("ImathConfig.h"):
        text = replace_once(
            text, "#ifdef __CUDACC__",
            "#if defined(__CUDACC__) || defined(__HIP__)",
        )
    else:
        raise ValueError(f"No patch defined for {name}")
    return text.encode("utf-8")


def prepare(source, destination, arnold_root):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if (source == destination or source in destination.parents
            or destination in source.parents):
        raise ValueError("The staging directory must be outside the source packages")
    arnold_library = (arnold_root / "lib" / "ai.lib").resolve(strict=True)
    arnold_runtime = (arnold_root / "bin" / "ai.dll").resolve(strict=True)

    # Validate every input before writing anything. Do not silently patch a
    # different Arnold build or replace its headers with OIIO 3 headers.
    patches = {}
    for name, expected in HEADER_HASHES.items():
        package, relative = name.split("/", 1)
        header = source / PACKAGES[package] / relative
        data = header.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unsupported dependency header (SHA256 mismatch): {header}")
        patches[Path(name)] = patch_header(name, data)

    config = Path("lib") / "cmake" / "OpenImageIO" / "OpenImageIOConfig.cmake"
    original = (source / PACKAGES["openimageio"] / config).read_bytes()
    # Complete the static package's dependency export for standalone OSL and
    # its consumers. AiMalloc/AiFree must come from the real Arnold runtime.
    metadata = f"""
# OSL's isolated Arnold compatibility profile.
find_dependency(OpenEXR CONFIG)
find_dependency(Freetype MODULE)
if (NOT TARGET Arnold::ai)
    add_library(Arnold::ai SHARED IMPORTED)
    set_target_properties(Arnold::ai PROPERTIES
        IMPORTED_IMPLIB "{arnold_library.as_posix()}"
        IMPORTED_LOCATION "{arnold_runtime.as_posix()}")
endif ()
set_property(TARGET OpenImageIO::OpenImageIO APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES Arnold::ai)
"""
    patches[Path("openimageio") / config] = original + metadata.encode("utf-8")

    for package, relative in PACKAGES.items():
        package_source = source / relative
        for src in package_source.rglob("*"):
            if not src.is_file():
                continue
            name = Path(package) / src.relative_to(package_source)
            dst = destination / name
            data = patches.get(name)
            if data is None:
                if dst.exists() and src.read_bytes() == dst.read_bytes():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif not dst.exists() or dst.read_bytes() != data:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
    print(f"Staged Arnold OIIO 2.6.3.2 / Imath 3.1.10 in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Arnold Windows dependencies directory")
    parser.add_argument("destination", type=Path, help="Isolated staging directory")
    parser.add_argument("--arnold-root", type=Path, required=True,
                        help="Arnold distribution containing lib/ai.lib and bin/ai.dll")
    args = parser.parse_args()
    prepare(args.source, args.destination, args.arnold_root)
