#!/usr/bin/env python3
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_arnold", ROOT / "src" / "build-scripts" / "prepare-arnold-deps.py"
)
prepare_arnold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_arnold)


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.destination = root / "stage"
        self.runtime = root / "arnold"
        self.header_name = "openimageio/include/OpenImageIO/platform.h"
        self.header = self.source / self.header_name
        self.header.parent.mkdir(parents=True)
        self.original = b"#ifdef __CUDACC__\n#define OIIO_HOSTDEVICE __device__\n#endif\n"
        self.header.write_bytes(self.original)
        config = self.source / "openimageio/lib/cmake/OpenImageIO/OpenImageIOConfig.cmake"
        config.parent.mkdir(parents=True)
        config.write_text("# original config\n", encoding="utf-8")
        (self.source / "imath").mkdir()
        for name in ("lib/ai.lib", "bin/ai.dll"):
            path = self.runtime / name
            path.parent.mkdir(parents=True)
            path.write_bytes(b"runtime fixture")
        self.packages = patch.dict(
            prepare_arnold.PACKAGES,
            {"openimageio": Path("openimageio"), "imath": Path("imath")}, clear=True
        )
        self.hashes = patch.dict(
            prepare_arnold.HEADER_HASHES,
            {self.header_name: hashlib.sha256(self.original).hexdigest()}, clear=True
        )
        self.packages.start()
        self.hashes.start()
        self.addCleanup(self.packages.stop)
        self.addCleanup(self.hashes.stop)

    def stage(self):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_arnold.prepare(self.source, self.destination, self.runtime)

    def test_source_untouched_and_staging_idempotent(self):
        self.stage()
        self.assertEqual(self.header.read_bytes(), self.original)
        staged = self.destination / self.header_name
        self.assertIn(b"defined(__HIP__)", staged.read_bytes())
        files = [p for p in self.destination.rglob("*") if p.is_file()]
        for path in files:
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        self.stage()
        for path in files:
            self.assertEqual(path.stat().st_mtime_ns, 1_000_000_000)

    def test_unknown_header_fails_before_writing(self):
        self.header.write_bytes(self.original + b"// changed\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_missing_runtime_fails_before_writing(self):
        (self.runtime / "bin/ai.dll").unlink()
        with self.assertRaises(FileNotFoundError):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_overlapping_source_is_rejected(self):
        for destination in (self.source, self.source / "nested", self.source.parent):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "outside the source"):
                    prepare_arnold.prepare(self.source, destination, self.runtime)

    def test_static_export_uses_real_runtime(self):
        self.stage()
        config = (self.destination
                  / "openimageio/lib/cmake/OpenImageIO/OpenImageIOConfig.cmake")
        text = config.read_text(encoding="utf-8")
        self.assertIn("find_dependency(OpenEXR CONFIG)", text)
        self.assertIn("find_dependency(Freetype MODULE)", text)
        self.assertIn("INTERFACE_LINK_LIBRARIES Arnold::ai", text)
        self.assertIn((self.runtime / "lib/ai.lib").as_posix(), text)

    def test_device_guards_preserve_negation(self):
        result = prepare_arnold.device_guards(
            "#ifndef __CUDA_ARCH__\n#endif\n"
            "#if defined(__x86_64__) && !defined(__CUDA_ARCH__)\n#endif\n"
        )
        self.assertIn(
            "#if !(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))", result
        )
        self.assertIn(
            "defined(__x86_64__) && "
            "!(defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))", result
        )

    def test_ambiguous_patch_anchor_is_rejected(self):
        for text in ("missing", "anchor anchor"):
            with self.assertRaisesRegex(ValueError, "exactly one patch anchor"):
                prepare_arnold.replace_once(text, "anchor", "replacement")


if __name__ == "__main__":
    unittest.main()
