#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Windows ARM64 architecture and release-contract tests; stdlib, any host."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import struct
import subprocess
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

    def test_cuda_manifest_uses_native_host_tools_and_target_libraries(self):
        manifest = json.loads((Path(__file__).parent / "cuda_windows_arm64_redist.json").read_text())
        kinds = set()
        for component in manifest["components"]:
            kind, platform, filename = component["relative_path"].split("/")
            kinds.add(kind)
            self.assertEqual(platform, "windows-arm64")
            self.assertIn("windows-arm64", filename)
            self.assertRegex(component["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue({"cuda_nvcc", "libnvvm", "cuda_crt", "cccl", "cuda_cudart", "libcublas", "libcufft"}.issubset(kinds))

    def test_vc_runtime_directory_can_contain_an_x64_only_exception_runtime(self):
        redist = self.root / "redist"
        redist.mkdir()
        for name in ("vcruntime140.dll", "msvcp140.dll", "msvcp140_1.dll"):
            (redist / name).write_bytes(pe(0xAA64))
        (redist / "vcruntime140_1.dll").write_bytes(pe(0x8664))
        with contextlib.redirect_stdout(io.StringIO()):
            arch.copy_vc_runtime(redist, self.bin, "arm64")
            arch.check(self.bin, "arm64")
        self.assertTrue((self.bin / "msvcp140_1.dll").exists())
        self.assertFalse((self.bin / "vcruntime140_1.dll").exists())

    def test_vc_runtime_requires_native_core_dlls_before_copying(self):
        redist = self.root / "redist"
        redist.mkdir()
        (redist / "vcruntime140.dll").write_bytes(pe(0x8664))
        (redist / "msvcp140.dll").write_bytes(pe(0xAA64))
        with self.assertRaisesRegex(ValueError, "missing arm64"):
            arch.copy_vc_runtime(redist, self.bin, "arm64")
        self.assertFalse((self.bin / "msvcp140.dll").exists())

    @unittest.skipUnless(shutil.which("cmake"), "CMake is needed to exercise the eSpeak source patch")
    def test_espeak_windows_data_generator_gets_the_crt_declaration(self):
        patch = Path(__file__).resolve().parents[2] / "tools/cmake/espeak_patch.cmake"
        for windows in (False, True):
            with self.subTest(windows=windows):
                source = self.root / str(windows)
                (source / "cmake").mkdir(parents=True)
                (source / "cmake/deps.cmake").write_text("else()\n  FetchContent_Declare(sonic-git\n")
                (source / "src").mkdir()
                (source / "src/CMakeLists.txt").write_text("")
                (source / "src/espeak-ng.c").write_text("#include <fcntl.h>\n")
                (source / "src/compat").mkdir()
                (source / "src/compat/getopt.c").write_text("getopt_internal(nargc, __UNCONST(nargv), options);\n")
                for component in ("libespeak-ng", "speechPlayer"):
                    folder = source / "src" / component
                    folder.mkdir()
                    (folder / "CMakeLists.txt").write_text('target_compile_options(test PRIVATE "-fPIC" "-fvisibility=hidden")\n')
                wrapper = source / "run-patch.cmake"
                wrapper.write_text(
                    f"set(WIN32 {'ON' if windows else 'OFF'})\n"
                    f'include("{patch.as_posix()}")\n'
                )
                subprocess.run([
                    shutil.which("cmake"), f"-DSOURCE_DIR={source}", "-P", str(wrapper),
                ], check=True, capture_output=True)
                for component in ("libespeak-ng", "speechPlayer"):
                    result = (source / "src" / component / "CMakeLists.txt").read_text()
                    self.assertIn('"-fPIC"', result)
                    self.assertIn('"-fvisibility=hidden"', result)
                self.assertEqual(
                    "#include <io.h>" in (source / "src/espeak-ng.c").read_text(), windows,
                )
                self.assertEqual(
                    "(char **)(nargv)" in (source / "src/compat/getopt.c").read_text(), windows,
                )

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
