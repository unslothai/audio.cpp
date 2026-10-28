#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Exercise the exported capability probe without CUDA hardware or its driver."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CXX = shutil.which("c++") or shutil.which("g++")
RELATIVE_SOURCE = Path("src/framework/core/attention_fallback.cpp")

HARNESS = r'''
#include "engine/framework/core/attention_fallback.h"
#include <cstdint>
#include <stdexcept>

static int major_cc = 12, minor_cc = 1, failed_attribute = 0, calls = 0;
static const char * device_name = "CUDA0";
static enum ggml_backend_dev_type device_type = GGML_BACKEND_DEVICE_TYPE_GPU;
static auto device = reinterpret_cast<ggml_backend_dev_t>(std::uintptr_t{1});
static auto backend = reinterpret_cast<ggml_backend_t>(std::uintptr_t{1});

extern "C" ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t) { return device; }
extern "C" const char * ggml_backend_dev_name(ggml_backend_dev_t) { return device_name; }
extern "C" const char * ggml_backend_dev_description(ggml_backend_dev_t) { return "test device"; }
extern "C" enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t) { return device_type; }
// Deliberately provide only the runtime API: a driver dependency fails to link.
extern "C" int cudaDeviceGetAttribute(int * value, int attribute, int ordinal) {
    ++calls;
    if (ordinal != 0 || (attribute != 75 && attribute != 76)) {
        throw std::runtime_error("wrong runtime query");
    }
    if (attribute == failed_attribute) return 1;
    *value = attribute == 75 ? major_cc : minor_cc;
    return 0;
}

static void check(bool condition) {
    if (!condition) throw std::runtime_error("capability-gating regression");
}

int main() {
    using engine::core::AttentionPreference;
    using engine::core::resolve_flash_attention;
    for (auto cc : {61, 70, 75, 80, 121}) {
        major_cc = cc / 10;
        minor_cc = cc % 10;
        check(resolve_flash_attention(backend, 128, AttentionPreference::Auto) == (cc < 70 || cc >= 80));
    }
    for (int attribute : {75, 76}) {
        failed_attribute = attribute;
        check(resolve_flash_attention(backend, 128, AttentionPreference::Auto));
    }
    failed_attribute = 0;
    calls = 0;
    check(resolve_flash_attention(backend, 128, AttentionPreference::Flash));
    check(!resolve_flash_attention(backend, 128, AttentionPreference::Eager));
    check(resolve_flash_attention(nullptr, 128, AttentionPreference::Auto));
    device_type = GGML_BACKEND_DEVICE_TYPE_CPU;
    check(resolve_flash_attention(backend, 128, AttentionPreference::Auto));
    device_type = GGML_BACKEND_DEVICE_TYPE_GPU;
    device_name = "Vulkan0";
    check(resolve_flash_attention(backend, 128, AttentionPreference::Auto));
    check(calls == 0);
}
'''


@unittest.skipUnless(CXX and shutil.which("git") and shutil.which("bash"), "C++/git/bash required")
class ExportedCudaProbeTest(unittest.TestCase):
    def test_export_preserves_capability_gating_without_a_driver_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            exported = source / RELATIVE_SOURCE
            exported.parent.mkdir(parents=True)
            exported.write_text((ROOT / RELATIVE_SOURCE).read_text(), newline="\n")
            (source / "CMakeLists.txt").write_text(
                'set(AUDIOCPP_GIT_SHA "unknown")\nset(AUDIOCPP_GIT_DATE "unknown")\n'
                'if (GIT_FOUND)\nendif()\n', newline="\n")
            # Model the immutable upstream tree, before the tooling overlay.
            tooling = root / "tooling"
            (tooling / "patches").mkdir(parents=True)
            patch = tooling / "patches/cuda-runtime-capability-probe.patch"
            patch.write_text((ROOT / "scripts/unsloth/patches" / patch.name).read_text(), newline="\n")
            stamp = tooling / "stamp_source.sh"
            stamp.write_text((ROOT / "scripts/unsloth/stamp_source.sh").read_text(), newline="\n")
            self.run_command(["git", "-C", str(source), "apply", "--reverse", str(patch)])
            for _ in range(2):  # The source export must also be safe to restamp.
                self.run_command(["bash", str(stamp), str(source), "a" * 40, "2026-10-26"])
            harness = root / "probe.cpp"
            harness.write_text(HARNESS, newline="\n")
            executable = root / "probe"
            self.run_command([CXX, "-std=c++17", "-DGGML_USE_CUDA", "-I", str(ROOT / "include"),
                              "-I", str(ROOT / "external/ggml/include"), str(exported),
                              str(harness), "-o", str(executable)])
            self.run_command([str(executable)])

    def run_command(self, command):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
