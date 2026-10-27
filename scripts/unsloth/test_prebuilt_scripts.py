#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Self-test for package_bundle.py / validate_bundle.py (stdlib, Linux, seconds).

    python3 scripts/unsloth/test_prebuilt_scripts.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import package_bundle as pb  # noqa: E402
import validate_bundle as vb  # noqa: E402

COMMIT = "ac16661d144f00f84ea0483f3574c374c9868e2d"
ESPEAK = "4870adfa25b1a32b4361592f1be8a40337c58d6c"
TAG = "v0.8.2-audio8-perf-hotfix-unsloth.1"
UT = "v0.8.2-audio8-perf-hotfix"


def _quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a), buf.getvalue()


class PrebuiltScriptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="acpp-selftest-"))
        self.bin = self.tmp / "bin"
        self.src = self.tmp / "src"
        (self.bin / "licenses" / "espeak-ng").mkdir(parents=True)
        (self.src / "tools" / "cmake").mkdir(parents=True)
        # Any runnable ELF stands in for the server; its closure is the host's libc.
        shutil.copy2(shutil.which("true"), self.bin / "audiocpp_server")
        (self.bin / "espeak-ng-data.bin").write_bytes(b"\0" * 64)
        (self.bin / "audiocpp_espeak_pack").write_bytes(b"tool, never shipped")
        (self.bin / "licenses" / "espeak-ng" / "COPYING").write_text("GPL-3.0-or-later")
        (self.src / "LICENSE").write_text("Apache-2.0")
        (self.src / "tools" / "cmake" / "espeak_static.cmake").write_text(
            "ExternalProject_Add(x\n"
            "    URL https://codeload.github.com/espeak-ng/espeak-ng/tar.gz/refs/tags/1.52.0\n"
            "    URL_HASH SHA256=" + "ab" * 32 + "\n")
        self.dist = self.tmp / "dist"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def package(self, accel="cpu", os_name="linux", extra=()):
        argv = ["--bin-dir", str(self.bin), "--src-dir", str(self.src), "--out-dir", str(self.dist),
                "--tag", TAG, "--upstream-tag", UT, "--commit", COMMIT, "--os", os_name, "--arch", "x64",
                "--accel", accel, "--espeak-commit", ESPEAK, "--no-run", *extra]
        return _quiet(pb.main, argv)[0]

    def test_package_layout_and_info(self):
        self.assertEqual(self.package(), 0)
        asset = self.dist / f"audio-{TAG}-bin-ubuntu-x64-cpu.tar.gz"
        with tarfile.open(asset) as t:
            names = set(t.getnames())
            info = json.loads(t.extractfile(pb.INFO_NAME).read())
            first = t.extractfile("BUILD_INFO.txt").read().decode().splitlines()[0]
            source = t.extractfile("licenses/espeak-ng/SOURCE.txt").read().decode()
        self.assertIn("audiocpp_server", names)
        self.assertIn("espeak-ng-data.bin", names)
        self.assertNotIn("audiocpp_espeak_pack", names)
        self.assertEqual(first, pb.FINGERPRINT)
        self.assertIn(ESPEAK, source)
        # The keys Studio's install record uses, with its values.
        self.assertEqual((info["component"], info["backend"], info["os"], info["espeak"],
                          info["server_relpath"], info["asset"]),
                         ("audio.cpp", "cpu", "linux", True, "audiocpp_server", asset.name))

    def test_backend_mapping_matches_installer(self):
        self.assertEqual(pb.backend_for("macos", "x64", "metal"), "cpu")
        self.assertEqual(pb.backend_for("macos", "arm64", "metal"), "metal")
        self.assertEqual(pb.backend_for("linux", "x64", "cuda13"), "cuda")
        self.assertEqual(pb.backend_for("linux", "x64", "cpu-portable"), "cpu")

    def test_cuda_needs_runtime_line(self):
        with self.assertRaises(SystemExit):
            self.package(accel="cuda12")
        self.assertEqual(self.package(accel="cuda12", extra=["--runtime-line", "cuda12"]), 0)

    def test_forbidden_runtime_refused(self):
        # A CUDA runtime the build left beside the server and the server NEEDs
        # must still not ship: the lean contract wins over the closure walk.
        shutil.copy2(shutil.which("true"), self.bin / "libcudart.so.12")
        real = pb._elf_needed
        pb._elf_needed = lambda path: ["libcudart.so.12"] if path.name == "audiocpp_server" else []
        try:
            with self.assertRaises(SystemExit):
                pb.curate("linux", self.bin, Path(tempfile.mkdtemp(dir=self.tmp)))
        finally:
            pb._elf_needed = real
        self.assertTrue(pb.FORBIDDEN_IN_BUNDLE.match("libnccl.so.2"))
        self.assertTrue(pb.FORBIDDEN_IN_BUNDLE.match("cublas64_12.dll"))
        self.assertFalse(pb.FORBIDDEN_IN_BUNDLE.match("libggml-cuda.so"))

    def test_cudart_zip_and_aggregate(self):
        self.assertEqual(self.package(), 0)
        rt = self.tmp / "cudart"
        rt.mkdir()
        for n in ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll", "cufft64_12.dll"):
            (rt / n).write_bytes(b"MZ")
        self.assertEqual(_quiet(pb.main, ["--cudart-dir", str(rt), "--out-dir", str(self.dist), "--tag", TAG,
                                          "--arch", "x64", "--accel", "cuda13"])[0], 0)
        (self.dist / f"audio.cpp-source-{TAG}.tar.gz").write_bytes(b"src")
        self.assertEqual(_quiet(pb.main, ["--emit-manifest", "--emit-sha256", "--dist", str(self.dist),
                                          "--tag", TAG, "--upstream-tag", UT, "--commit", COMMIT])[0], 0)
        man = json.loads((self.dist / pb.MANIFEST_NAME).read_text())
        sha = json.loads((self.dist / pb.SHA256_NAME).read_text())
        kinds = sorted(a["kind"] for a in man["artifacts"])
        self.assertEqual(kinds, ["bundle", "cuda-runtime"])
        self.assertIn(pb.MANIFEST_NAME, sha["artifacts"])
        self.assertIn(f"audio.cpp-source-{TAG}.tar.gz", sha["artifacts"])
        # A stray DLL in the runtime dir is refused.
        (rt / "nvrtc64_130_0.dll").write_bytes(b"MZ")
        with self.assertRaises(SystemExit):
            _quiet(pb.main, ["--cudart-dir", str(rt), "--out-dir", str(self.dist), "--tag", TAG,
                             "--arch", "x64", "--accel", "cuda13"])

    def test_aggregate_refuses_foreign_tag(self):
        self.assertEqual(self.package(), 0)
        with self.assertRaises(SystemExit):
            _quiet(pb.main, ["--emit-manifest", "--dist", str(self.dist), "--tag", "v9-unsloth.1",
                             "--upstream-tag", UT, "--commit", COMMIT])

    def test_validate_static_pass_and_branding_fail(self):
        self.assertEqual(self.package(), 0)
        asset = self.dist / f"audio-{TAG}-bin-ubuntu-x64-cpu.tar.gz"
        vb.failures.clear()
        rc, out = _quiet(vb.main, ["--bundle", str(asset), "--no-run"])
        self.assertEqual(rc, 0, out)
        # Unbranded copy: the gate must fail.
        work = self.tmp / "unbranded"
        with tarfile.open(asset) as t:
            t.extractall(work, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(work)
        (work / "BUILD_INFO.txt").write_text("not branded\n")
        vb.failures.clear()
        rc, out = _quiet(vb.main, ["--bundle", str(work), "--no-run"])
        self.assertEqual(rc, 1)
        self.assertIn("fingerprint", out)
        vb.failures.clear()

    def test_pe_import_parser(self):
        # Minimal PE32+ with one import descriptor naming FOO.dll.
        data = bytearray(0x400)
        data[:2] = b"MZ"
        data[0x3C:0x40] = (0x80).to_bytes(4, "little")
        data[0x80:0x84] = b"PE\0\0"
        data[0x86:0x88] = (1).to_bytes(2, "little")          # one section
        data[0x94:0x96] = (0xF0).to_bytes(2, "little")       # optional header size
        opt = 0x98
        data[opt:opt + 2] = (0x20B).to_bytes(2, "little")    # PE32+
        dd = opt + 112
        data[dd + 8:dd + 16] = (0x1000).to_bytes(4, "little") + (40).to_bytes(4, "little")
        sec = opt + 0xF0
        data[sec + 8:sec + 24] = b"".join(x.to_bytes(4, "little") for x in (0x200, 0x1000, 0x200, 0x200))
        data[0x200 + 12:0x200 + 16] = (0x1100).to_bytes(4, "little")
        data[0x300:0x308] = b"FOO.dll\0"
        p = self.tmp / "x.dll"
        p.write_bytes(bytes(data))
        self.assertEqual(vb._pe_imports(p), ["FOO.dll"])
        self.assertTrue(vb.WIN_HOST_IMPORTS.match("cudart64_12.dll"))
        self.assertTrue(vb.WIN_HOST_IMPORTS.match("vulkan-1.dll"))
        self.assertFalse(vb.WIN_HOST_IMPORTS.match("ggml-cuda.dll"))

    def test_zip_keeps_exec_bit(self):
        stage = self.tmp / "stage"
        stage.mkdir()
        (stage / "audiocpp_server.exe").write_bytes(b"MZ")
        out = self.tmp / "a.zip"
        pb._zip(stage, out)
        with zipfile.ZipFile(out) as z:
            self.assertEqual(z.getinfo("audiocpp_server.exe").external_attr >> 16, 0o755)

    def test_stamp_script(self):
        cm = self.tmp / "stamp"
        cm.mkdir()
        (cm / "CMakeLists.txt").write_text(
            'set(AUDIOCPP_GIT_SHA "unknown")\nset(AUDIOCPP_GIT_DATE "unknown")\nif (GIT_FOUND)\nendif()\n')
        patch = cm / "tools/cmake/espeak_patch.cmake"
        patch.parent.mkdir(parents=True)
        patch.write_text(
            '# upstream behavior stays\n# BEGIN UNSLOTH_WINDOWS_COMPAT\n# obsolete\n# END UNSLOTH_WINDOWS_COMPAT\n')
        r = subprocess.run(["bash", str(HERE / "stamp_source.sh"), str(cm), COMMIT, "2026-09-24"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (cm / "CMakeLists.txt").read_text()
        self.assertIn('set(AUDIOCPP_GIT_SHA "ac16661")', text)
        self.assertIn('EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git"', text)
        stamped_patch = patch.read_text()
        self.assertIn("# upstream behavior stays", stamped_patch)
        self.assertNotIn("# obsolete", stamped_patch)
        self.assertIn("#include <io.h>", stamped_patch)
        self.assertEqual(stamped_patch.count("# BEGIN UNSLOTH_WINDOWS_COMPAT"), 1)
        (cm / "CMakeLists.txt").write_text("project(x)\n")
        r = subprocess.run(["bash", str(HERE / "stamp_source.sh"), str(cm), COMMIT, "2026-09-24"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
