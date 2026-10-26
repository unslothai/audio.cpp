#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Reject mislabeled or mixed-architecture Windows bundles/runtime directories."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

MACHINES = {"x64": 0x8664, "arm64": 0xAA64}


def pe_machine(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError(f"{path.name}: invalid DOS header")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if offset + 24 > len(data) or data[offset:offset + 4] != b"PE\0\0":
        raise ValueError(f"{path.name}: invalid PE header")
    return struct.unpack_from("<H", data, offset + 4)[0]


def check(root: Path, arch: str) -> None:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".exe", ".dll"))
    if not files:
        raise ValueError(f"no PE binaries in {root}")
    for path in files:
        actual = pe_machine(path)
        if actual != MACHINES[arch]:
            raise ValueError(f"{path.name}: machine 0x{actual:04x}, expected {arch} (0x{MACHINES[arch]:04x})")
    print(f"ok: {len(files)} Windows binaries are {arch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--arch", required=True, choices=MACHINES)
    args = parser.parse_args()
    try:
        check(args.dir, args.arch)
    except ValueError as exc:
        parser.exit(1, f"FAIL: {exc}\n")


if __name__ == "__main__":
    main()
