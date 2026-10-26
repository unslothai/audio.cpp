#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Windows ARM64 architecture and release-contract tests; stdlib, any host."""
from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_windows_arch as arch
import package_bundle as pb

TAG = "v0.9.2-drama3-base-small-unsloth.2"
COMMIT = "a" * 40
ESPEAK = "b" * 40


def pe(machine: int) -> bytes:
    data = bytearray(256)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 128)
    data[128:132] = b"PE\0\0"
    struct.pack_into("<H", data, 132, machine)
    return bytes(data)


class WindowsArm64PrebuiltTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.src = self.root / "src"
        self.dist = self.root / "dist"
        (self.bin / "licenses/espeak-ng").mkdir(parents=True)
        (self.src / "tools/cmake").mkdir(parents=True)
        (self.bin / "audiocpp_server.exe").write_bytes(pe(0xAA64))
        (self.bin / "espeak-ng-data.bin").write_bytes(b"data")
        (self.bin / "licenses/espeak-ng/COPYING").write_text("GPL-3.0-or-later")
        (self.src / "LICENSE").write_text("Apache-2.0")
        (self.src / "tools/cmake/espeak_static.cmake").write_text(
            "URL https://codeload.github.com/espeak-ng/espeak-ng/tar.gz/refs/tags/1.52.0\n"
            f"URL_HASH SHA256={'c' * 64}\n")

    def quiet(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pb.main(args), 0)

    def package(self, accel):
        args = ["--bin-dir", str(self.bin), "--src-dir", str(self.src), "--out-dir", str(self.dist),
                "--tag", TAG, "--upstream-tag", "v0.9.2-drama3-base-small", "--commit", COMMIT,
                "--os", "windows", "--arch", "arm64", "--accel", accel, "--min-os", "windows-11",
                "--espeak-commit", ESPEAK, "--no-run"]
        if accel == "cuda13":
            args += ["--runtime-line", "cuda13", "--cuda-toolkit", "13.4.1", "--sms", "121-real;120-real;100-virtual"]
        self.quiet(args)
        return self.dist / f"audio-{TAG}-bin-windows-arm64-{accel}.zip"

    def test_native_arm64_passes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            arch.check(self.bin, "arm64")

    def test_x64_binary_is_not_an_arm64_bundle(self):
        (self.bin / "audiocpp_server.exe").write_bytes(pe(0x8664))
        with self.assertRaisesRegex(ValueError, "expected arm64"):
            arch.check(self.bin, "arm64")

    def test_nested_x64_dll_is_rejected(self):
        nested = self.bin / "plugins"
        nested.mkdir()
        (nested / "ggml-cuda.dll").write_bytes(pe(0x8664))
        with self.assertRaisesRegex(ValueError, "ggml-cuda.dll"):
            arch.check(self.bin, "arm64")

    def test_empty_directory_is_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "no PE binaries"):
            arch.check(empty, "arm64")

    def test_malformed_headers_are_rejected(self):
        for data in (b"MZ", b"not a PE", b"MZ" + b"\xff" * 254):
            with self.subTest(data=data[:8]):
                (self.bin / "audiocpp_server.exe").write_bytes(data)
                with self.assertRaises(ValueError):
                    arch.check(self.bin, "arm64")

    def test_cpu_bundle_contract(self):
        bundle = self.package("cpu")
        with zipfile.ZipFile(bundle) as z:
            info = json.loads(z.read(pb.INFO_NAME))
            self.assertEqual(info["arch"], "arm64")
            self.assertEqual(info["os"], "windows")
            self.assertEqual(info["backend"], "cpu")
            self.assertEqual(info["min_os"], "windows-11")
            self.assertTrue(info["espeak"])
            self.assertEqual(arch.MACHINES["arm64"], struct.unpack_from("<H", z.read("audiocpp_server.exe"), 132)[0])
            self.assertIn("espeak-ng-data.bin", z.namelist())

    def test_cuda_runtime_and_manifest_keep_arm64_identity(self):
        self.package("cpu")
        bundle = self.package("cuda13")
        runtime = self.root / "runtime"
        runtime.mkdir()
        names = ["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "cufft64_12.dll"]
        for name in names:
            (runtime / name).write_bytes(pe(0xAA64))
        self.quiet(["--cudart-dir", str(runtime), "--out-dir", str(self.dist),
                    "--tag", TAG, "--arch", "arm64", "--accel", "cuda13"])
        self.quiet(["--emit-manifest", "--emit-sha256", "--dist", str(self.dist),
                    "--tag", TAG, "--upstream-tag", "v0.9.2-drama3-base-small", "--commit", COMMIT])
        with zipfile.ZipFile(bundle) as z:
            info = json.loads(z.read(pb.INFO_NAME))
            self.assertEqual(info["cuda_toolkit"], "13.4.1")
            self.assertEqual(info["runtime_line"], "cuda13")
            self.assertIn("121-real", info["cuda_architectures"])
            self.assertFalse(set(names) & set(z.namelist()), "runtime stays in its companion ZIP")
        manifest = json.loads((self.dist / pb.MANIFEST_NAME).read_text())
        self.assertEqual(len(manifest["artifacts"]), 3)
        self.assertTrue(all(a["arch"] == "arm64" for a in manifest["artifacts"]))
        runtimes = [a for a in manifest["artifacts"] if a["kind"] == "cuda-runtime"]
        self.assertEqual(runtimes[0]["asset"], f"audio-{TAG}-cudart-windows-arm64-cuda13.zip")
        index = json.loads((self.dist / pb.SHA256_NAME).read_text())["artifacts"]
        self.assertEqual(index[runtimes[0]["asset"]]["sha256"], runtimes[0]["sha256"])


if __name__ == "__main__":
    unittest.main()
