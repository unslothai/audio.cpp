#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Package + manifest tool for the Unsloth audio.cpp prebuilts (stdlib only).

Modes:

  package (default)   Curate one audiocpp_server bundle from a build's bin dir:
                      the server, the local shared-library closure (dlopen'd
                      ggml CPU variants included), espeak-ng-data.bin, LICENSE,
                      licenses/ (eSpeak-ng GPL text + source offer, vendored
                      notices), BUILD_INFO.txt (Unsloth fingerprint) and an
                      embedded UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json whose keys
                      match the record Studio's installer writes, so an
                      unpacked bundle reads as the same backend. Archive to
                      audio-<tag>-bin-<os>-<arch>-<accel>.{tar.gz,zip}.

  --cudart-dir DIR    Windows only: zip the CUDA runtime DLLs of one CUDA line
                      into audio-<tag>-cudart-windows-<arch>-<accel>.zip, the
                      companion Studio downloads unless torch ships the line.

  --emit-espeak-source  Fetch the pinned eSpeak-ng tarball into --dist as
                      espeak-ng-source-<tag>.tar.gz, so one release asset
                      serves every bundle's source offer.

  --emit-manifest /   Aggregate every bundle in --dist into
  --emit-sha256       audio-cpp-prebuilt-manifest.json and
                      audio-cpp-prebuilt-sha256.json, read from each bundle's
                      embedded info so the manifest cannot disagree with it.

Lean CUDA contract: Linux CUDA bundles link cudart / cuBLAS / cuFFT
dynamically and ship none of them (Studio loads them from torch's nvidia-*
wheels); Windows CUDA bundles carry them only in the separate cudart zip. No
bundle may carry or need NCCL. Curation refuses to break either rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

COMPONENT = "audio.cpp"
UPSTREAM_REPO = "0xShug0/audio.cpp"
INFO_NAME = "UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json"
MANIFEST_NAME = "audio-cpp-prebuilt-manifest.json"
SHA256_NAME = "audio-cpp-prebuilt-sha256.json"
FINGERPRINT = "Compiled by the Unsloth team"
ESPEAK_DATA = "espeak-ng-data.bin"

# What Studio's audio.cpp engine speaks to the server
# (studio/backend/core/inference/audio_cpp_server.py and its callers). Bump
# `version` on any breaking change to the config or the endpoints.
STUDIO_PROTOCOL = {
    "version": 1,
    "argv": ["--config", "<server.json>", "--no-ui", "--threads", "<n>"],
    "probe_argv": ["--help", "--version"],
    "config_keys": ["host", "port", "backend", "lazy_load", "max_loaded_models",
                    "busy_timeout_ms", "models"],
    "model_entry_keys": ["id", "family", "path", "task", "mode"],
    "ready": {"GET /health": {"status": "ok"}, "GET /v1/models": "data[].id lists the model id"},
    "endpoints": {
        "POST /v1/audio/speech": "application/json {model, input, voice?, options?, seed?} -> audio/wav",
        "POST /v1/audio/transcriptions": "multipart/form-data {model, file, language?} -> json {text}",
        "POST /v1/tasks/run": "application/json {model, request} (music generation)",
    },
}

BUNDLE_RE = re.compile(
    r"^audio-(?P<tag>.+)-bin-(?P<os>ubuntu|windows|macos)-(?P<arch>x64|arm64)-"
    r"(?P<accel>[A-Za-z0-9._-]+)\.(?P<ext>tar\.gz|zip)$"
)
CUDART_RE = re.compile(r"^audio-(?P<tag>.+)-cudart-windows-(?P<arch>x64|arm64)-(?P<accel>cuda[0-9.]+)\.zip$")

# Never in a bundle: the lean CUDA runtime is the host's (torch) or the
# companion zip's, and NCCL must not be linked at all.
FORBIDDEN_IN_BUNDLE = re.compile(
    r"^(lib)?(cudart|cublas|cublasLt|cufft|nccl|nvrtc|nvJitLink)(64)?([._-].*)?\.(so(\.\d+)*|dll|dylib)$",
    re.I,
)
CUDART_DLL_RE = re.compile(r"^(cudart64|cublas64|cublasLt64|cufft64)_[0-9]+\.dll$", re.I)

_C_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C"}
_FIXED_MTIME = 1704067200  # 2024-01-01T00:00:00Z: byte-stable archives

# The asset <os> token the Studio installer parses, and the os value its
# install record stores (platform.system().lower()).
OS_TOKEN = {"linux": "ubuntu", "windows": "windows", "macos": "macos"}
RECORD_OS = {"linux": "linux", "windows": "windows", "macos": "darwin"}
MIN_OS = {"linux": "glibc-2.35", "windows": "windows-10", "macos": "macos-13.3"}


def die(msg: str) -> "None":
    sys.exit(f"ERROR: {msg}")


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=_C_ENV)
    except FileNotFoundError:
        die(f"required tool '{cmd[0]}' not found")
    if r.returncode != 0:
        die(f"{' '.join(cmd)} failed (rc={r.returncode}): {r.stderr.strip()}")
    return r.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backend_for(os_name: str, arch: str, accel: str) -> str:
    """The backend Studio launches the bundle with; mirrors the installer's _backend_of."""
    if os_name == "macos" and arch == "x64":
        return "cpu"  # the Intel slice keeps the -metal name but is built with Metal off
    for b in ("metal", "cuda", "vulkan"):
        if accel.startswith(b):
            return b
    return "cpu"


# --------------------------------------------------------------------------- #
# Curation
# --------------------------------------------------------------------------- #
def _server_name(os_name: str) -> str:
    return "audiocpp_server.exe" if os_name == "windows" else "audiocpp_server"


def _elf_needed(path: Path) -> list[str]:
    return re.findall(r"\(NEEDED\)[^\[]*\[([^\]]+)\]", _run(["readelf", "-d", str(path)]))


def _macho_local(path: Path, bin_dir: Path) -> list[str]:
    deps = []
    for line in _run(["otool", "-L", str(path)]).splitlines()[1:]:
        m = re.match(r"\s+(\S+)\s+\(", line)
        if m and not m.group(1).startswith(("/usr/lib/", "/System/")):
            base = os.path.basename(m.group(1))
            if (bin_dir / base).exists():
                deps.append(base)
    return deps


def _copy(bin_dir: Path, stage: Path, name: str, symlinks: bool) -> None:
    src, dst = bin_dir / name, stage / name
    if dst.exists() or dst.is_symlink():
        return
    if symlinks and src.is_symlink():
        target = os.readlink(src)
        os.symlink(target, dst)
        _copy(bin_dir, stage, os.path.basename(target), symlinks)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)


def curate(os_name: str, bin_dir: Path, stage: Path) -> list[str]:
    """Copy the server, its local library closure and the eSpeak data into stage."""
    server = _server_name(os_name)
    if not (bin_dir / server).is_file():
        die(f"missing {bin_dir / server}")
    if not (bin_dir / ESPEAK_DATA).is_file() or (bin_dir / ESPEAK_DATA).stat().st_size == 0:
        die(f"missing {bin_dir / ESPEAK_DATA}: every Unsloth bundle links eSpeak-ng statically")
    shutil.copy2(bin_dir / server, stage / server)
    shutil.copy2(bin_dir / ESPEAK_DATA, stage / ESPEAK_DATA)

    if os_name == "linux":
        # Dynamically linked ggml (cpu-portable) plus its dlopen'd CPU variant
        # modules, which no NEEDED walk can see.
        roots = [stage / server]
        for p in sorted(bin_dir.glob("lib*.so*")):
            if re.match(r"^lib(ggml|engine|audiocpp)", p.name):
                _copy(bin_dir, stage, p.name, symlinks=True)
                roots.append(stage / p.name)
        seen: set[str] = set()
        while roots:
            for need in _elf_needed(roots.pop()):
                if need in seen:
                    continue
                seen.add(need)
                if (bin_dir / need).exists():
                    _copy(bin_dir, stage, need, symlinks=True)
                    roots.append(stage / need)
    elif os_name == "macos":
        roots = [stage / server]
        for pat in ("libggml*.dylib", "libggml*.so"):
            for p in sorted(bin_dir.glob(pat)):
                _copy(bin_dir, stage, p.name, symlinks=True)
                roots.append(stage / p.name)
        while roots:
            for need in _macho_local(roots.pop(), bin_dir):
                if not (stage / need).exists():
                    _copy(bin_dir, stage, need, symlinks=True)
                    roots.append(stage / need)
    else:
        # No portable import walker on the runner; ship the project DLLs and
        # the app-local MSVC / OpenMP runtime the workflow copied in.
        keep = re.compile(r"^(ggml|engine|audiocpp|vcruntime140|msvcp140|vcomp140|libomp)", re.I)
        for p in sorted(bin_dir.glob("*.dll")):
            if keep.match(p.name):
                shutil.copy2(p, stage / p.name)

    offenders = sorted(p.name for p in stage.iterdir() if FORBIDDEN_IN_BUNDLE.match(p.name))
    if offenders:
        die("bundle must not carry the CUDA runtime or NCCL, found: " + ", ".join(offenders))
    return sorted(p.name for p in stage.iterdir())


def _espeak_meta(src: Path) -> dict:
    """eSpeak-ng version, tarball URL and sha256 as the build fetched them."""
    cm = src / "tools" / "cmake" / "espeak_static.cmake"
    if not cm.is_file():
        die(f"{cm} not found; cannot describe the eSpeak-ng the bundle links")
    return _parse_espeak_meta(cm.read_text(encoding="utf-8"), cm)


def _parse_espeak_meta(text: str, cm: object) -> dict:
    url = re.search(r"^\s*URL\s+(\S+)", text, re.M)
    digest = re.search(r"URL_HASH\s+SHA256=([0-9a-fA-F]{64})", text)
    version = re.search(r"refs/tags/([^/\s]+)$", url.group(1)) if url else None
    if not (url and digest and version):
        die(f"could not parse URL / URL_HASH / tag from {cm}")
    return {"version": version.group(1), "url": url.group(1), "sha256": digest.group(1).lower()}


def espeak_source_asset(tag: str) -> str:
    return f"espeak-ng-source-{tag}.tar.gz"


def emit_espeak_source(dist: Path, tag: str) -> int:
    archive = dist / f"audio.cpp-source-{tag}.tar.gz"
    with tarfile.open(archive) as t:
        member = next((m for m in t.getmembers() if m.name.endswith("/tools/cmake/espeak_static.cmake")), None)
        if member is None:
            die(f"{archive.name} has no tools/cmake/espeak_static.cmake")
        espeak = _parse_espeak_meta(t.extractfile(member).read().decode("utf-8"), member.name)
    with urllib.request.urlopen(espeak["url"], timeout=120) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != espeak["sha256"]:
        die(f"eSpeak-ng tarball sha256 {actual} does not match the pinned {espeak['sha256']}")
    (dist / espeak_source_asset(tag)).write_bytes(data)
    return 0


def write_licenses(stage: Path, cfg: dict, espeak: dict) -> None:
    src = Path(cfg["src_dir"])
    bin_dir = Path(cfg["bin_dir"])
    lic = stage / "licenses"
    if (bin_dir / "licenses").is_dir():
        shutil.copytree(bin_dir / "licenses", lic, dirs_exist_ok=True)
    for tarball in (lic / "espeak-ng").glob("espeak-ng-*.tar.gz"):
        tarball.unlink()
    (lic / "espeak-ng").mkdir(parents=True, exist_ok=True)
    if not (lic / "espeak-ng" / "COPYING").is_file():
        die("the build did not stage licenses/espeak-ng/COPYING (GPL-3.0-or-later text)")
    if not (src / "LICENSE").is_file():
        die(f"{src}/LICENSE missing")
    shutil.copy2(src / "LICENSE", stage / "LICENSE")
    for name, rel in (("ggml", "external/ggml/LICENSE"), ("sentencepiece", "external/sentencepiece/LICENSE"),
                      ("cJSON", "external/cJSON/LICENSE"), ("libyaml", "external/libyaml/License"),
                      ("cpp-httplib", "external/cpp-httplib/LICENSE")):
        if (src / rel).is_file():
            (lic / name).mkdir(exist_ok=True)
            shutil.copy2(src / rel, lic / name / "LICENSE")

    commit = cfg["espeak_commit"]
    (lic / "espeak-ng" / "SOURCE.txt").write_text(
        "eSpeak-ng corresponding source (GPL-3.0-or-later)\n"
        "\n"
        "audiocpp_server in this bundle statically links eSpeak-ng, and\n"
        f"{ESPEAK_DATA} is compiled from eSpeak-ng's data files.\n"
        "\n"
        f"eSpeak-ng version: {espeak['version']}\n"
        f"eSpeak-ng commit:  {commit}\n"
        f"repository:        https://github.com/espeak-ng/espeak-ng/tree/{commit}\n"
        f"source tarball:    {espeak['url']}\n"
        f"tarball sha256:    {espeak['sha256']}\n"
        "\n"
        "The exact tarball the build used is published beside this bundle as the\n"
        f"release asset {espeak_source_asset(cfg['tag'])}. espeak_patch.cmake, the\n"
        "build-time patch applied to it, and espeak_static.cmake, the recipe, are\n"
        "included beside this file.\n"
        "\n"
        "Written offer: for three years from the date of this release, the Unsloth\n"
        "team will provide the complete corresponding source code of this binary\n"
        "(audio.cpp, eSpeak-ng and every other component) to anyone who asks,\n"
        "via https://github.com/unslothai/audio.cpp/issues. The audio.cpp source is\n"
        f"also published as the release asset audio.cpp-source-{cfg['tag']}.tar.gz\n"
        f"(upstream {UPSTREAM_REPO} commit {cfg['commit']}).\n",
        encoding="utf-8",
    )
    (stage / "COPYING-NOTICE.txt").write_text(
        f"{FINGERPRINT}\n"
        "\n"
        "audio.cpp itself is licensed under the Apache License 2.0 (LICENSE).\n"
        "This bundle statically links eSpeak-ng, which is GPL-3.0-or-later, so\n"
        "audiocpp_server as distributed here is a combined work under\n"
        "GPL-3.0-or-later (licenses/espeak-ng/COPYING). Its corresponding source\n"
        "and the written offer are described in licenses/espeak-ng/SOURCE.txt.\n"
        "Other bundled components keep their own notices under licenses/.\n",
        encoding="utf-8",
    )


def write_metadata(stage: Path, cfg: dict, asset: str, files: list[str], espeak: dict) -> dict:
    os_name, arch, accel = cfg["os"], cfg["arch"], cfg["accel"]
    backend = backend_for(os_name, arch, accel)
    server = _server_name(os_name)
    sms = [s for s in re.split(r"[ ;,]+", cfg.get("sms") or "") if s]
    info = {
        # Keys shared with the install record Studio writes (install_audio_cpp_prebuilt.py).
        "schema_version": 1,
        "component": COMPONENT,
        "published_repo": cfg["source_repo"],
        "release_tag": cfg["tag"],
        "asset": asset,
        "backend": backend,
        "espeak": True,
        "server_relpath": server,
        "server_sha256": sha256_file(stage / server),
        "server_version": cfg.get("server_version") or None,
        "os": RECORD_OS[os_name],
        "arch": arch,
        # Build provenance.
        "accel": accel,
        "asset_os": OS_TOKEN[os_name],
        "upstream_repo": UPSTREAM_REPO,
        "upstream_tag": cfg["upstream_tag"],
        "source_repo": cfg["source_repo"],
        "source_commit": cfg["commit"],
        "runtime_line": cfg.get("runtime_line") or None,
        "cuda_toolkit": cfg.get("cuda_toolkit") or None,
        "cuda_architectures": sms or None,
        "cuda_runtime_bundled": False if backend == "cuda" else None,
        "min_os": cfg.get("min_os") or MIN_OS[os_name],
        "espeak_ng": {"version": espeak["version"], "commit": cfg["espeak_commit"],
                      "sha256": espeak["sha256"], "license": "GPL-3.0-or-later"},
        "license": "GPL-3.0-or-later (combined work: audio.cpp Apache-2.0 + static eSpeak-ng)",
        "files": files,
        "studio_protocol": STUDIO_PROTOCOL,
    }
    (stage / INFO_NAME).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    lines = [
        FINGERPRINT,
        "",
        f"audio.cpp upstream: {UPSTREAM_REPO} {cfg['upstream_tag']} @ {cfg['commit']}",
        f"packaging tag: {cfg['tag']}",
        f"asset: {asset}",
        f"os: {os_name}",
        f"arch: {arch}",
        f"accel: {accel}",
        f"backend: {backend}",
        f"runtime line: {cfg.get('runtime_line') or ''}",
        f"cuda toolkit: {cfg.get('cuda_toolkit') or ''}",
        f"cuda architectures: {','.join(sms)}",
        f"min os: {info['min_os']}",
        f"eSpeak-ng: {espeak['version']} @ {cfg['espeak_commit']} (static, GPL-3.0-or-later)",
        f"source repo: {cfg['source_repo']}",
        f"built at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
    ]
    if cfg.get("server_version"):
        lines += ["", "audiocpp_server --version:", cfg["server_version"]]
    (stage / "BUILD_INFO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info


# --------------------------------------------------------------------------- #
# Archives
# --------------------------------------------------------------------------- #
def _tar(stage: Path, out: Path) -> None:
    def reset(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = _FIXED_MTIME
        return ti

    with tarfile.open(out, "w:gz", format=tarfile.GNU_FORMAT) as tar:
        for p in sorted(stage.rglob("*"), key=lambda x: x.relative_to(stage).as_posix()):
            tar.add(p, arcname=p.relative_to(stage).as_posix(), recursive=False, filter=reset)


def _zip(stage: Path, out: Path) -> None:
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*"), key=lambda x: x.relative_to(stage).as_posix()):
            if p.is_file():
                zi = zipfile.ZipInfo(p.relative_to(stage).as_posix(), date_time=(2024, 1, 1, 0, 0, 0))
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (0o755 if p.suffix.lower() in (".exe", ".dll") else 0o644) << 16
                z.writestr(zi, p.read_bytes())


def _server_version(stage: Path, os_name: str) -> str | None:
    """--version text when this host can run the bundle (not a CUDA build on a GPU-less runner)."""
    exe = stage / _server_name(os_name)
    env = dict(os.environ)
    var = {"windows": "PATH", "macos": "DYLD_LIBRARY_PATH"}.get(os_name, "LD_LIBRARY_PATH")
    env[var] = os.pathsep.join([str(stage), env.get(var, "")])
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60,
                           env=env, cwd=str(stage))
    except (OSError, subprocess.SubprocessError):
        return None
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else None


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def do_package(a: argparse.Namespace) -> int:
    cfg = {k: getattr(a, k) for k in ("bin_dir", "src_dir", "out_dir", "tag", "upstream_tag", "commit",
                                       "source_repo", "os", "arch", "accel", "runtime_line",
                                       "cuda_toolkit", "sms", "min_os", "espeak_commit")}
    missing = [k for k in ("bin_dir", "src_dir", "out_dir", "tag", "upstream_tag", "commit", "os",
                           "arch", "accel", "espeak_commit") if not cfg[k]]
    if missing:
        die("missing package inputs: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]):
        die(f"--commit must be a 40-hex commit, got {cfg['commit']!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["espeak_commit"]):
        die(f"--espeak-commit must be a 40-hex commit, got {cfg['espeak_commit']!r}")
    if backend_for(cfg["os"], cfg["arch"], cfg["accel"]) == "cuda" and not cfg["runtime_line"]:
        die("CUDA bundles need --runtime-line (cuda12 / cuda13)")
    bin_dir, out_dir = Path(cfg["bin_dir"]), Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".zip" if cfg["os"] == "windows" else ".tar.gz"
    asset = f"audio-{cfg['tag']}-bin-{OS_TOKEN[cfg['os']]}-{cfg['arch']}-{cfg['accel']}{ext}"
    if not BUNDLE_RE.match(asset):
        die(f"asset name {asset} breaks the naming contract")
    espeak = _espeak_meta(Path(cfg["src_dir"]))

    stage = Path(tempfile.mkdtemp(prefix="acpp-stage-"))
    try:
        files = curate(cfg["os"], bin_dir, stage)
        write_licenses(stage, cfg, espeak)
        if not a.no_run:
            cfg["server_version"] = _server_version(stage, cfg["os"])
        info = write_metadata(stage, cfg, asset, files, espeak)
        out = out_dir / asset
        (_zip if ext == ".zip" else _tar)(stage, out)
        print(f"wrote {out} ({out.stat().st_size / 1048576:.1f} MiB)")
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(stage).as_posix()}  {p.stat().st_size}")
        print(json.dumps({k: info[k] for k in ("asset", "backend", "runtime_line", "server_version")},
                         indent=2))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return 0


def do_cudart(a: argparse.Namespace) -> int:
    for k in ("tag", "arch", "accel", "out_dir"):
        if not getattr(a, k):
            die(f"--cudart-dir needs --{k.replace('_', '-')}")
    src = Path(a.cudart_dir)
    dlls = sorted(p for p in src.glob("*.dll") if CUDART_DLL_RE.match(p.name))
    if not any(p.name.lower().startswith("cudart64_") for p in dlls):
        die(f"no cudart64_*.dll in {src}")
    stray = sorted(p.name for p in src.glob("*.dll") if p not in dlls)
    if stray:
        die("unexpected DLLs in the CUDA runtime dir: " + ", ".join(stray))
    name = f"audio-{a.tag}-cudart-windows-{a.arch}-{a.accel}.zip"
    if not CUDART_RE.match(name):
        die(f"{name} breaks the cudart naming contract")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="acpp-cudart-"))
    try:
        for p in dlls:
            shutil.copy2(p, stage / p.name)
        _zip(stage, out / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"wrote {out / name}: " + ", ".join(p.name for p in dlls))
    return 0


def read_embedded_info(bundle: Path) -> dict:
    if bundle.name.endswith(".zip"):
        with zipfile.ZipFile(bundle) as zf:
            if INFO_NAME in zf.namelist():
                return json.loads(zf.read(INFO_NAME))
    else:
        with tarfile.open(bundle, "r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.lstrip("./") == INFO_NAME:
                    return json.loads(tar.extractfile(m).read())
    die(f"{bundle.name} has no top-level {INFO_NAME}")
    return {}


def do_aggregate(a: argparse.Namespace) -> int:
    dist = Path(a.dist)
    out = Path(a.out or a.dist)
    out.mkdir(parents=True, exist_ok=True)
    bundles = sorted(p for p in dist.iterdir() if BUNDLE_RE.match(p.name))
    runtimes = sorted(p for p in dist.iterdir() if CUDART_RE.match(p.name))
    if not bundles:
        die(f"no audio-*-bin-* bundles in {dist}")
    common = {
        "schema_version": 1,
        "component": COMPONENT,
        "source_repo": a.source_repo,
        "source_commit": a.commit,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_tag": a.upstream_tag,
        "packaging_tag": a.tag,
        "studio_protocol": STUDIO_PROTOCOL,
    }
    artifacts, sha_index = [], {}
    for b in bundles:
        m = BUNDLE_RE.match(b.name)
        if m.group("tag") != a.tag:
            die(f"{b.name} is not tagged {a.tag}")
        info = read_embedded_info(b)
        if info.get("asset") != b.name:
            die(f"{b.name} embeds info for {info.get('asset')!r}")
        digest = sha256_file(b)
        artifacts.append({
            "asset": b.name, "kind": "bundle",
            "os": m.group("os"), "arch": info["arch"], "accel": info["accel"], "backend": info["backend"],
            "runtime_line": info.get("runtime_line"), "cuda_toolkit": info.get("cuda_toolkit"),
            "cuda_architectures": info.get("cuda_architectures"),
            "cuda_runtime_bundled": info.get("cuda_runtime_bundled"),
            "min_os": info.get("min_os"), "espeak": info.get("espeak"),
            "espeak_ng": info.get("espeak_ng"), "server_sha256": info.get("server_sha256"),
            "server_version": info.get("server_version"), "sha256": digest,
            "size": b.stat().st_size,
        })
        sha_index[b.name] = {"kind": f"{m.group('os')}-{info['arch']}-{info['accel']}-bundle",
                             "repo": a.source_repo, "sha256": digest, "source_commit": a.commit,
                             "upstream_tag": a.upstream_tag}
    for r in runtimes:
        m = CUDART_RE.match(r.name)
        digest = sha256_file(r)
        artifacts.append({"asset": r.name, "kind": "cuda-runtime", "os": "windows", "arch": m.group("arch"),
                          "accel": m.group("accel"), "backend": "cuda", "runtime_line": m.group("accel"),
                          "sha256": digest, "size": r.stat().st_size})
        sha_index[r.name] = {"kind": f"windows-{m.group('arch')}-{m.group('accel')}-cuda-runtime",
                             "repo": a.source_repo, "sha256": digest, "source_commit": a.commit,
                             "upstream_tag": a.upstream_tag}
    for p in sorted(dist.glob("audio.cpp-source-*.tar.gz")):
        sha_index[p.name] = {"kind": "source", "repo": a.source_repo, "sha256": sha256_file(p),
                             "source_commit": a.commit, "upstream_tag": a.upstream_tag}
    if a.emit_manifest:
        mp = out / MANIFEST_NAME
        mp.write_text(json.dumps({**common,
                                  "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")
        sha_index[MANIFEST_NAME] = {"kind": "published-manifest", "repo": a.source_repo,
                                    "sha256": sha256_file(mp), "source_commit": a.commit,
                                    "upstream_tag": a.upstream_tag}
        print(f"wrote {mp} ({len(artifacts)} artifacts)")
    if a.emit_sha256:
        sp = out / SHA256_NAME
        sp.write_text(json.dumps({**common, "release_tag": a.tag, "artifacts": sha_index}, indent=2) + "\n",
                      encoding="utf-8")
        print(f"wrote {sp} ({len(sha_index)} entries)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emit-manifest", action="store_true")
    ap.add_argument("--emit-sha256", action="store_true")
    ap.add_argument("--emit-espeak-source", action="store_true")
    ap.add_argument("--dist", help="aggregate: dir holding the built archives")
    ap.add_argument("--out", help="aggregate: dir for the sidecars (default --dist)")
    ap.add_argument("--cudart-dir", help="Windows: dir holding one CUDA line's runtime DLLs")
    ap.add_argument("--bin-dir")
    ap.add_argument("--src-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--tag", help="packaging tag, e.g. v0.8.2-audio8-perf-hotfix-unsloth.1")
    ap.add_argument("--upstream-tag")
    ap.add_argument("--commit", help="upstream source commit (40 hex)")
    ap.add_argument("--source-repo", default="unslothai/audio.cpp")
    ap.add_argument("--os", choices=sorted(OS_TOKEN))
    ap.add_argument("--arch", choices=["x64", "arm64"])
    ap.add_argument("--accel", help="cpu | cpu-portable | vulkan | cuda12 | cuda13 | metal")
    ap.add_argument("--runtime-line", help="cuda12 | cuda13 for CUDA bundles")
    ap.add_argument("--cuda-toolkit", help="toolkit version built with, e.g. 12.8.1")
    ap.add_argument("--sms", help="CMAKE_CUDA_ARCHITECTURES as built")
    ap.add_argument("--min-os")
    ap.add_argument("--espeak-commit", help="eSpeak-ng commit of the tag the build fetched (40 hex)")
    ap.add_argument("--no-run", action="store_true", help="do not run --version (GPU-less runner)")
    a = ap.parse_args(argv)
    if a.emit_espeak_source:
        return emit_espeak_source(Path(a.dist), a.tag)
    if a.emit_manifest or a.emit_sha256:
        for req in ("dist", "tag", "upstream_tag", "commit"):
            if not getattr(a, req):
                die(f"aggregate mode needs --{req.replace('_', '-')}")
        return do_aggregate(a)
    if a.cudart_dir:
        return do_cudart(a)
    return do_package(a)


if __name__ == "__main__":
    raise SystemExit(main())
