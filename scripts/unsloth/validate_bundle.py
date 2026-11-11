#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CI gate for one Unsloth audio.cpp prebuilt bundle (stdlib only).

Checks, in order (any failure exits 1):

  1. layout     server, espeak-ng-data.bin, LICENSE, licenses/espeak-ng/{COPYING,
                SOURCE.txt}, BUILD_INFO.txt opening with the Unsloth fingerprint,
                the embedded info JSON naming this asset; no CUDA runtime / NCCL
                library inside the bundle.
  2. closure    Linux: every NEEDED of every ELF (readelf) resolves via ldd,
                except libcuda.so.1 and, for --lean-cuda N, the line-N cudart /
                cuBLAS / cuBLASLt / cuFFT the host supplies; nothing NEEDs
                libnccl; newest GLIBC_ symbol <= --max-glibc; no dynamic
                libstdc++. macOS: every non-system dylib reference resolves in
                the bundle. Windows: every imported DLL is in the bundle, in
                System32, or on the CUDA allowlist; the MSVC runtime only in the bundle.
  3. run        --help exits 0 and --version lists --expect-backends
                (skipped with --no-run, e.g. a CUDA build on a GPU-less runner).
  4. smoke      (--smoke) start the server exactly as Studio does (one model,
                --config/--no-ui), wait for /health + /v1/models, synthesize with
                Kokoro (the eSpeak-ng path) and assert a non-silent WAV, then
                transcribe that speech with Moonshine tiny and assert a
                non-empty transcript.
"""
from __future__ import annotations

import argparse
import array
import http.client
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid
import wave
import zipfile
from pathlib import Path

FINGERPRINT = "Compiled by the Unsloth team"
INFO_NAME = "UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json"
HF_REPO = "audio-cpp/audio.cpp-gguf"
HF_REVISION = "83c5d96c03023ff5a7712570d057ce26c8769f98"
KOKORO = {"file": "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf", "family": "kokoro_tts", "task": "tts"}
MOONSHINE = {"file": "Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf",
             "family": "moonshine_asr", "task": "asr"}
SMOKE_TEXT = "Hello from Unsloth. The quick brown fox jumps over the lazy dog."
FORBIDDEN_IN_BUNDLE = re.compile(
    r"^(lib)?(cudart|cublas|cublasLt|cufft|nccl|nvrtc|nvJitLink)(64)?([._-].*)?\.(so(\.\d+)*|dll|dylib)$", re.I)
# Host-provided on Windows: the NVIDIA driver (nvcuda), the cudart companion zip
# or torch (CUDA runtime), and any GPU driver (the Vulkan loader).
WIN_HOST_IMPORTS = re.compile(
    r"^(nvcuda|cudart64_\d+|cublas64_\d+|cublasLt64_\d+|cufft64_\d+|vulkan-1)\.dll$", re.I)
# Shipped app-local: must resolve in the bundle, since the runner's System32 would hide a gap.
WIN_APP_LOCAL_IMPORTS = re.compile(r"^(vcruntime|msvcp|vcomp|concrt)140(_\w+)?\.dll$", re.I)

failures: list[str] = []


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", flush=True)
    failures.append(msg)


def ok(msg: str) -> None:
    print(f"ok:   {msg}", flush=True)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          env={**kw.pop("env", os.environ), "LC_ALL": "C"}, **kw)


# --------------------------------------------------------------------------- #
# 1. layout
# --------------------------------------------------------------------------- #
def unpack(bundle: Path, work: Path) -> Path:
    if bundle.is_dir():
        return bundle
    dest = work / "bundle"
    dest.mkdir()
    if bundle.name.endswith(".zip"):
        with zipfile.ZipFile(bundle) as z:
            z.extractall(dest)
            for zi in z.infolist():  # zip loses the exec bit; restore it
                mode = zi.external_attr >> 16
                if mode:
                    os.chmod(dest / zi.filename, mode)
    else:
        with tarfile.open(bundle, "r:gz") as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(dest, filter="data")
            else:
                t.extractall(dest)
    return dest


def check_layout(root: Path, server: Path, asset: str | None) -> dict:
    info: dict = {}
    if not server.is_file():
        fail(f"no {server.name} at the bundle root")
        return info
    ok(f"{server.name} present ({server.stat().st_size} bytes)")
    data = root / "espeak-ng-data.bin"
    if data.is_file() and data.stat().st_size > 0:
        ok(f"espeak-ng-data.bin beside the server ({data.stat().st_size} bytes)")
    else:
        fail("espeak-ng-data.bin missing beside the server (Studio's eSpeak check fails)")
    for rel in ("LICENSE", "COPYING-NOTICE.txt", "licenses/espeak-ng/COPYING", "licenses/espeak-ng/SOURCE.txt"):
        (ok if (root / rel).is_file() else fail)(f"{rel} {'present' if (root / rel).is_file() else 'missing'}")
    bi = root / "BUILD_INFO.txt"
    first = bi.read_text(encoding="utf-8").splitlines()[0] if bi.is_file() else ""
    (ok if first == FINGERPRINT else fail)(f"BUILD_INFO.txt fingerprint line: {first!r}")
    try:
        info = json.loads((root / INFO_NAME).read_text(encoding="utf-8"))
        ok(f"{INFO_NAME}: backend={info.get('backend')} accel={info.get('accel')}")
        if asset and info.get("asset") != asset:
            fail(f"{INFO_NAME} names {info.get('asset')!r}, archive is {asset!r}")
        if info.get("espeak") is not True:
            fail(f"{INFO_NAME} does not record espeak=true")
    except (OSError, ValueError) as exc:
        fail(f"{INFO_NAME} unreadable: {exc}")
    bad = sorted(p.name for p in root.rglob("*") if p.is_file() and FORBIDDEN_IN_BUNDLE.match(p.name))
    (fail if bad else ok)(f"no CUDA runtime / NCCL inside the bundle{': ' + ', '.join(bad) if bad else ''}")
    return info


# --------------------------------------------------------------------------- #
# 2. closure
# --------------------------------------------------------------------------- #
def _is_elf(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def closure_linux(root: Path, lean_cuda: int | None, max_glibc: str | None, lib_path: list[str]) -> None:
    allowed = [re.compile(r"^libcuda\.so\.1$")]
    if lean_cuda is not None:
        n = lean_cuda
        allowed += [re.compile(rf"^libcudart\.so\.{n}$"), re.compile(rf"^libcublas\.so\.{n}$"),
                    re.compile(rf"^libcublasLt\.so\.{n}$"), re.compile(r"^libcufft\.so\.\d+$")]
    elves = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink() and _is_elf(p)]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(root), *lib_path, env.get("LD_LIBRARY_PATH", "")])
    needed_all: dict[str, set[str]] = {}
    newest_glibc = (0, 0)
    for elf in elves:
        rel = elf.relative_to(root).as_posix()
        needed = re.findall(r"\(NEEDED\)[^\[]*\[([^\]]+)\]", _run(["readelf", "-d", str(elf)]).stdout)
        for n in needed:
            needed_all.setdefault(n, set()).add(rel)
        for v in re.findall(r"GLIBC_(\d+)\.(\d+)", _run(["objdump", "-T", str(elf)]).stdout):
            newest_glibc = max(newest_glibc, (int(v[0]), int(v[1])))
        if any(n.startswith("libstdc++") for n in needed):
            fail(f"{rel} links libstdc++ dynamically (the C++ runtime must be static)")
        ldd = _run(["ldd", str(elf)], env=env).stdout
        for lib, where in re.findall(r"^\s*(lib(?:cuda|cudart|cublas|cublasLt|cufft)\.so\S*) => (\S+)", ldd, re.M):
            print(f"note: {rel}: {lib} -> {where}")
        for lib in re.findall(r"^\s*(\S+) => not found", ldd, re.M):
            if any(a.match(lib) for a in allowed):
                continue
            fail(f"{rel}: {lib} does not resolve")
    print("NEEDED (union): " + ", ".join(sorted(needed_all)))
    nccl = sorted(n for n in needed_all if n.startswith("libnccl"))
    (fail if nccl else ok)(f"nothing NEEDs libnccl{': ' + ', '.join(nccl) if nccl else ''}")
    if lean_cuda is not None:
        want = {f"libcudart.so.{lean_cuda}", f"libcublas.so.{lean_cuda}"}
        missing = want - set(needed_all)
        (fail if missing else ok)(f"lean CUDA {lean_cuda}: links {', '.join(sorted(want))} dynamically"
                                  + (f" (missing {', '.join(sorted(missing))})" if missing else ""))
    glibc = f"{newest_glibc[0]}.{newest_glibc[1]}"
    if max_glibc:
        cap = tuple(int(x) for x in max_glibc.split("."))
        (fail if newest_glibc > cap else ok)(f"newest GLIBC_ symbol {glibc} (floor {max_glibc})")
    else:
        print(f"note: newest GLIBC_ symbol {glibc} (no --max-glibc given)")
    ok(f"closure checked over {len(elves)} ELF file(s)")


def closure_macos(root: Path) -> None:
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in ("", ".dylib", ".so") or p.name.endswith(".bin"):
            continue
        out = _run(["otool", "-L", str(p)])
        if out.returncode != 0 or "is not an object file" in out.stdout:
            continue
        for line in out.stdout.splitlines()[1:]:
            m = re.match(r"\s+(\S+)\s+\(", line)
            if not m:
                continue
            ref = m.group(1)
            if ref.startswith(("/usr/lib/", "/System/")):
                continue
            base = os.path.basename(ref)
            if ref.startswith("@") and (p.parent / base).exists():
                continue
            if base == p.name:
                continue
            fail(f"{p.relative_to(root)}: {ref} is neither a system library nor in the bundle")
    ok("macOS dylib references resolve")


def _pe_imports(path: Path) -> list[str]:
    """DLL names from the import and delay-import directories of a PE32+ file."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return []
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return []
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_size = struct.unpack_from("<H", data, pe + 20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    sections = []
    for i in range(nsec):
        s = opt + opt_size + 40 * i
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
        sections.append((va, max(vsize, rsize), raw))

    def off(rva: int) -> int | None:
        for va, size, raw in sections:
            if va <= rva < va + size:
                return rva - va + raw
        return None

    def cstr(rva: int) -> str:
        o = off(rva)
        return data[o:data.index(b"\0", o)].decode("ascii", "replace") if o is not None else ""

    names = []
    for idx, stride, name_at in ((1, 20, 12), (13, 32, 4)):  # import, delay import
        rva, size = struct.unpack_from("<II", data, dd + 8 * idx)
        o = off(rva) if rva else None
        while o is not None and size:
            name_rva = struct.unpack_from("<I", data, o + name_at)[0]
            if not name_rva:
                break
            names.append(cstr(name_rva))
            o += stride
    return names


def closure_windows(root: Path) -> None:
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    present = {p.name.lower() for p in root.iterdir() if p.is_file()}
    for p in sorted(root.glob("*")):
        if p.suffix.lower() not in (".exe", ".dll"):
            continue
        for dll in _pe_imports(p):
            low = dll.lower()
            if low in present:
                continue
            if WIN_APP_LOCAL_IMPORTS.match(dll):
                fail(f"{p.name} imports {dll}: the MSVC runtime must ship in the bundle")
                continue
            if low.startswith(("api-ms-win-", "ext-ms-")) or (system32 / dll).exists():
                continue
            if WIN_HOST_IMPORTS.match(dll):
                print(f"note: {p.name} imports host-provided {dll}")
                continue
            fail(f"{p.name} imports {dll}: not in the bundle, System32, or the host allowlist")
    ok("Windows DLL imports resolve")


# --------------------------------------------------------------------------- #
# 3. run
# --------------------------------------------------------------------------- #
def loader_env(root: Path, lib_path: list[str], home: Path) -> dict:
    env = dict(os.environ)
    var = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    env[var] = os.pathsep.join([str(root), *lib_path, env.get(var, "")])
    # Studio runs the server with a scratch HOME: eSpeak data is extracted under it.
    for k in ("HOME", "USERPROFILE", "XDG_CACHE_HOME"):
        env.pop(k, None)
    env["HOME"] = env["USERPROFILE"] = str(home)
    return env


def check_run(server: Path, env: dict, expect: list[str]) -> str:
    r = _run([str(server), "--help"], env=env, cwd=str(server.parent), timeout=60)
    if r.returncode != 0:
        fail(f"--help exited {r.returncode}: {(r.stderr or r.stdout)[-800:]}")
        return ""
    ok("--help exits 0")
    r = _run([str(server), "--version"], env=env, cwd=str(server.parent), timeout=60)
    text = (r.stdout or "").strip()
    print(text)
    if r.returncode != 0 or not text:
        fail(f"--version exited {r.returncode} with no text")
        return text
    m = re.search(r"^backends:\s*(.+)$", text, re.M)
    have = {b.strip() for b in m.group(1).split(",")} if m else set()
    missing = [b for b in expect if b not in have]
    (fail if missing else ok)(f"--version backends {sorted(have)} (expected {expect})")
    return text


# --------------------------------------------------------------------------- #
# 4. smoke
# --------------------------------------------------------------------------- #
def fetch_model(models_dir: Path, rel: str) -> Path:
    dest = models_dir / rel
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{rel}"
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "unsloth-audio-cpp-ci"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:  # noqa: S310
                shutil.copyfileobj(resp, f, 1 << 20)
            tmp.replace(dest)
            return dest
        except OSError as exc:
            print(f"download attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"FAIL: could not download {url}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, server: Path, model: dict, model_path: Path, backend: str, env: dict,
                 threads: int, work: Path):
        self.model_id = f"studio-{uuid.uuid4().hex[:12]}"
        self.port = _free_port()
        self.dir = Path(tempfile.mkdtemp(prefix="acpp-srv-", dir=work))
        cfg = {"host": "127.0.0.1", "port": self.port, "backend": backend, "lazy_load": False,
               "max_loaded_models": 1, "busy_timeout_ms": 0,
               "models": [{"id": self.model_id, "family": model["family"], "path": str(model_path),
                           "task": model["task"], "mode": "offline"}]}
        (self.dir / "server.json").write_text(json.dumps(cfg), encoding="utf-8")
        self.log_path = self.dir / "server.log"
        self.log = open(self.log_path, "wb")
        cmd = [str(server), "--config", str(self.dir / "server.json"), "--no-ui"]
        if threads:
            cmd += ["--threads", str(threads)]
        print("starting: " + " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=env, cwd=str(self.dir))

    def tail(self, n: int = 3000) -> str:
        try:
            return self.log_path.read_bytes()[-n:].decode("utf-8", "replace")
        except OSError:
            return ""

    def get(self, path: str):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("GET", path)
            r = c.getresponse()
            return json.loads(r.read(1 << 20).decode("utf-8"))
        except Exception:  # noqa: BLE001 - not up yet
            return None
        finally:
            c.close()

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                fail(f"server exited {self.proc.returncode} during startup:\n{self.tail()}")
                return False
            h = self.get("/health")
            if isinstance(h, dict) and h.get("status") == "ok":
                m = self.get("/v1/models")
                ids = {str(x.get("id")) for x in (m or {}).get("data") or [] if isinstance(x, dict)}
                if self.model_id in ids:
                    return True
            time.sleep(0.5)
        fail(f"server not ready in {timeout:.0f}s:\n{self.tail()}")
        return False

    def post(self, path: str, body: bytes, ctype: str, timeout: float) -> tuple[int, str, bytes]:
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            c.request("POST", path, body=body, headers={"Content-Type": ctype})
            r = c.getresponse()
            return r.status, r.getheader("Content-Type") or "", r.read()
        finally:
            c.close()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        self.log.close()


def _pcm16_mono(wav_bytes: bytes) -> tuple[array.array, int]:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        s = array.array("h", raw)
    elif width == 4:  # float32 or int32 WAV: normalise to int16
        try:
            f = array.array("f", raw)
            s = array.array("h", (max(-32768, min(32767, int(x * 32767))) for x in f))
        except ValueError:
            s = array.array("h", (x >> 16 for x in array.array("i", raw)))
    else:
        raise ValueError(f"unsupported sample width {width}")
    if sys.byteorder == "big":
        s.byteswap()
    if ch > 1:
        s = array.array("h", s[::ch])
    return s, rate


def _resample16k(s: array.array, rate: int) -> bytes:
    """Linear resample to 16 kHz mono PCM16 WAV: what Studio's dictation path posts."""
    if rate != 16000 and len(s) > 1:
        n = int(len(s) * 16000 / rate)
        step = (len(s) - 1) / max(1, n - 1)
        s = array.array("h", (int(s[int(i * step)] + (s[min(len(s) - 1, int(i * step) + 1)] - s[int(i * step)])
                                  * (i * step - int(i * step))) for i in range(n)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(s.tobytes())
    return buf.getvalue()


def smoke(server: Path, env: dict, models_dir: Path, backend: str, threads: int, work: Path,
          timeout: float, out_dir: Path | None, min_words: int) -> None:
    kokoro = fetch_model(models_dir, KOKORO["file"])
    moonshine = fetch_model(models_dir, MOONSHINE["file"])

    s = Server(server, KOKORO, kokoro, backend, env, threads, work)
    wav = b""
    try:
        if s.wait_ready(timeout):
            ok(f"Kokoro server ready on backend={backend}")
            t0 = time.monotonic()
            status, ctype, wav = s.post("/v1/audio/speech",
                                        json.dumps({"model": s.model_id, "input": SMOKE_TEXT}).encode(),
                                        "application/json", timeout)
            dt = time.monotonic() - t0
            if status != 200 or not wav.startswith(b"RIFF"):
                fail(f"/v1/audio/speech HTTP {status} {ctype}: {wav[:300]!r}\n{s.tail()}")
                wav = b""
    finally:
        s.stop()
    if wav:
        pcm, rate = _pcm16_mono(wav)
        secs = len(pcm) / rate if rate else 0
        rms = (sum(x * x for x in pcm) / max(1, len(pcm))) ** 0.5
        peak = max((abs(x) for x in pcm), default=0)
        print(f"speech: {len(wav)} bytes, {secs:.2f}s @ {rate} Hz, rms={rms:.0f}, peak={peak}, took {dt:.1f}s")
        (ok if secs >= 1.0 and rms >= 100 and peak >= 1000 else fail)(
            f"Kokoro (eSpeak-ng) WAV is non-silent speech ({secs:.2f}s, rms {rms:.0f})")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "smoke_kokoro.wav").write_bytes(wav)
    else:
        return

    s = Server(server, MOONSHINE, moonshine, backend, env, threads, work)
    try:
        if not s.wait_ready(timeout):
            return
        ok(f"Moonshine server ready on backend={backend}")
        boundary = uuid.uuid4().hex
        audio = _resample16k(pcm, rate)
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{s.model_id}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="dictation.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n").encode() + audio + f"\r\n--{boundary}--\r\n".encode()
        status, ctype, data = s.post("/v1/audio/transcriptions", body,
                                     f"multipart/form-data; boundary={boundary}", timeout)
    finally:
        s.stop()
    text = ""
    try:
        text = str(json.loads(data.decode("utf-8")).get("text") or "").strip()
    except (ValueError, AttributeError):
        text = data.decode("utf-8", "replace").strip() if status == 200 else ""
    print(f"transcript (HTTP {status}): {text!r}")
    (ok if status == 200 and text else fail)("Moonshine transcript is non-empty")
    words = {w for w in re.findall(r"[a-z]+", text.lower())}
    hits = words & {"hello", "quick", "brown", "fox", "jumps", "lazy", "dog"}
    (ok if len(hits) >= min_words else fail)(
        f"transcript shares {len(hits)} of 7 prompt words (need {min_words}): {sorted(hits)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate one Unsloth audio.cpp prebuilt bundle")
    ap.add_argument("--bundle", required=True, help="bundle archive or an already-extracted dir")
    ap.add_argument("--expect-backends", default="cpu", help="comma list --version must report")
    ap.add_argument("--lean-cuda", type=int, help="CUDA major whose runtime the host supplies (Linux)")
    ap.add_argument("--max-glibc", help="newest GLIBC_ symbol version allowed, e.g. 2.35 (Linux)")
    ap.add_argument("--lib-path", action="append", default=[],
                    help="extra loader dir, e.g. torch's nvidia/*/lib (repeatable)")
    ap.add_argument("--no-run", action="store_true", help="static checks only (no device for this build)")
    ap.add_argument("--smoke", action="store_true", help="Kokoro TTS + Moonshine STT through the HTTP server")
    ap.add_argument("--backend", default="cpu", help="server config backend for --smoke")
    ap.add_argument("--models-dir", default=os.environ.get("AUDIOCPP_SMOKE_MODELS", "smoke-models"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--min-words", type=int, default=3,
                    help="prompt words the Moonshine transcript of the Kokoro speech must contain")
    ap.add_argument("--out-dir", help="keep the synthesized WAV here")
    a = ap.parse_args(argv)

    bundle = Path(a.bundle).resolve()
    work = Path(tempfile.mkdtemp(prefix="acpp-validate-"))
    try:
        root = unpack(bundle, work)
        name = "audiocpp_server.exe" if os.name == "nt" else "audiocpp_server"
        server = root / name
        print(f"== {bundle.name}")
        info = check_layout(root, server, None if bundle.is_dir() else bundle.name)
        if not server.is_file():
            return 1
        if sys.platform.startswith("linux"):
            closure_linux(root, a.lean_cuda, a.max_glibc, a.lib_path)
        elif sys.platform == "darwin":
            closure_macos(root)
        else:
            closure_windows(root)
        home = work / "home"
        home.mkdir()
        env = loader_env(root, a.lib_path, home)
        if a.no_run:
            print("note: --no-run: --help/--version/smoke skipped (no device for this build here)")
        else:
            expect = [b for b in a.expect_backends.split(",") if b]
            check_run(server, env, expect)
            if a.smoke and not failures:
                smoke(server, env, Path(a.models_dir).resolve(), a.backend, a.threads, work, a.timeout,
                      Path(a.out_dir).resolve() if a.out_dir else None, a.min_words)
        size = bundle.stat().st_size if bundle.is_file() else sum(
            p.stat().st_size for p in root.rglob("*") if p.is_file())
        print(f"bundle size: {size / 1048576:.1f} MiB; backend={info.get('backend')}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if failures:
        print(f"\nVALIDATION FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f.splitlines()[0]}")
        return 1
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
