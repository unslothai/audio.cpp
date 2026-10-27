# Unsloth audio.cpp prebuilt release CI

This repository is the Unsloth fork of
[0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp). Beyond mirroring
upstream, its job is to build and publish **prebuilt `audiocpp_server`
bundles** so Unsloth Studio can run local text-to-speech, speech-to-text and
music models without a C/C++ toolchain. It follows the
[`unslothai/whisper.cpp`](https://github.com/unslothai/whisper.cpp) and
[`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) prebuilt
pipelines.

The Studio installer (`studio/install_audio_cpp_prebuilt.py` in
unslothai/unsloth) downloads a release asset, verifies its sha256 against the
in-repo pins, starts the staged server once, and swaps it into the managed
`audio.cpp` directory.

Unlike whisper.cpp, audio.cpp vendors its own patched ggml, so it cannot reuse
the llama.cpp install's ggml: every bundle is self-contained.

## What lives here

```
.github/workflows/
  unsloth-prebuilt.yml         orchestrator: resolve tag -> build -> gates -> publish -> reclaim -> alert
  unsloth-prebuilt-build.yml   reusable child: every bundle, packaged + validated on its runner
  unsloth-prebuilt-retry.yml   reruns an automatic run lost to a cancel or a dead runner, once
  unsloth-prebuilt-deadman.yml daily: alerts when the newest aged upstream release is unpublished
.github/actions/sign-windows/  Azure Trusted Signing of Windows PEs (same action as whisper.cpp)
.github/actions/prebuilt-alert/ deduplicated tracking issue for pipeline health (same as whisper.cpp)
.github/scripts/               sign-windows-tree.ps1, assert-windows-bundle-signed.ps1
scripts/unsloth/
  stamp_source.sh              bake the resolved commit into --version for a .git-less tree
  build_unix.sh                configure + build one Linux/macOS flavour (CI and local runs)
  package_bundle.py            curate one bundle / the cudart zip / the manifest + sha256 index
  validate_bundle.py           CI gate: layout, closure, --help/--version, TTS + STT smoke
  test_prebuilt_scripts.py     self-test of the above (runs in the resolve job)
  build_windows_arm64.ps1      native ARM64 CPU / CUDA build, including static eSpeak
  windows_arm64_native.cmake   native LLVM and ARM64 CUDA target configuration
  setup_cuda_windows_arm64.ps1 assemble the pinned CUDA 13.4.1 components (SHA256 verified)
  check_windows_arch.py       verify all shipped PEs have the declared architecture
  test_windows_arm64_prebuilts.py architecture + ARM64 release-contract tests
```

Upstream's own workflows (`release.yml`, `linux-build.yml`, ...) are left
untouched. Releases created by this pipeline use the built-in `GITHUB_TOKEN`,
whose tag creation does not trigger upstream's `v*` tag workflow.

## Triggering a release

Releases are normally automatic (see "Scheduling" below). To build or publish
by hand, run **Unsloth audio.cpp prebuilt (full release)** (`workflow_dispatch`):

| input | default | meaning |
| --- | --- | --- |
| `upstream_tag` | `latest` | 0xShug0/audio.cpp tag to build, or `latest` = newest aged upstream release |
| `packaging_suffix` | (blank) | release tag = `<upstream_tag>-<suffix>`, e.g. `v0.8.2-audio8-perf-hotfix-unsloth.1`; blank = newest published `unsloth.N` of that tag, else `unsloth.1` |
| `min_age_hours` | (blank = 6) | for `latest`: skip upstream releases public for less than this |
| `publish` | `false` | `false` builds and validates only; `true` publishes a GitHub Release |
| `sign_windows` | `true` | Authenticode-sign Windows binaries; publishing refuses `false` |
| `keep_artifacts` | `false` | keep the run's artifacts even if nothing was published |

An explicit tag still works exactly as before, e.g. `upstream_tag=v0.8.2-audio8-perf-hotfix`,
`packaging_suffix=unsloth.1`. A manual dispatch of an already published tag
builds for artifacts only and never republishes.

`resolve` dereferences `upstream_tag` (annotated or lightweight) to an
**immutable commit SHA**, fetches exactly that commit, stamps it
(`stamp_source.sh`: `--version` reports the real short SHA and date instead of
`unknown`, and the git probe can no longer read the enclosing tooling checkout),
records the eSpeak-ng commit the build will fetch, and uploads the source
tarball **once**. Every build job extracts that tarball and checks its
`UNSLOTH_SOURCE_COMMIT`, so a tag re-point upstream cannot change what a
packaging tag shipped. A published packaging tag is never rebuilt: bump
`packaging_suffix` (`unsloth.2`, ...). Upstream `main` moves; the pipeline only
ever builds the tag it is given.

The shared source export also carries the marked Windows eSpeak compatibility
patch from the tooling checkout. It adds the CRT declaration and correct argv
casts needed by native `clang-cl`; other platforms keep their upstream behavior.
The bundle includes that exact patch with the eSpeak source and license notices.

Publication is **atomic**: the release is created as a draft (invisible to the
anonymous API the installer reads), every asset is uploaded, then it is flipped
to `draft=false`. A leftover draft from a failed run counts as "not published"
and is replaced on the next run. The orchestrator's default token is read-only;
only `assemble` gets `contents: write` and only `reclaim` gets `actions: write`.
Manual runs of one tag + suffix are serialised by a concurrency group.

## Scheduling, retry and dead-man

The orchestrator wakes up three ways, all running the same `resolve`:

- `schedule`: daily at 22:41 UTC (same slot as unslothai/whisper.cpp).
- `repository_dispatch` of type `llama-published`, sent by unslothai/llama.cpp's
  `unsloth-prebuilt.yml` ("Notify audio.cpp" step) after it publishes. It is
  purely a wake-up: `client_payload` is never read. audio.cpp ships its own
  patched ggml, so a llama release changes nothing here; it only shortens the
  wait for a new upstream tag from up to a day to hours. llama.cpp needs the
  `AUDIO_CPP_DISPATCH_TOKEN` secret for it (a fine-grained PAT with access to
  unslothai/audio.cpp and **Contents: read and write**, the permission
  `POST /repos/{owner}/{repo}/dispatches` requires); without it the step warns
  and the daily schedule still publishes.
- `workflow_dispatch` (above).

An automatic run resolves `latest`: the newest non-draft, non-prerelease
0xShug0/audio.cpp release whose tag looks like `v<version>[-label]` and that has
been public for at least 6 hours. If that upstream tag already has a published
release here at **any** `unsloth.<N>`, `exists=true` and the run is a cheap
no-op (build, assemble and their artifacts are skipped). Otherwise it builds,
gates and publishes `<upstream_tag>-unsloth.1`. Automatic runs always publish
and always sign the Windows binaries; they cannot publish unsigned. There is
no llama.cpp pairing and no automatic `unsloth.N` roll: rebuilding a published
upstream tag is a deliberate manual dispatch with the next suffix.

Automatic runs share one concurrency group without `cancel-in-progress`
(whisper.cpp cancels because a newer llama ggml can change its build; here every
wake-up resolves the same upstream release). A wake-up arriving during a ~6h build
waits as the single pending run, then finds the release published and exits.
Manual runs never cancel each other.

`unsloth-prebuilt-retry.yml` reruns the failed jobs of an automatic run once
(`run_attempt == 1`) when it was cancelled with nothing failing, or failed only
because a hosted runner was lost. Real build failures, hand-cancelled manual
runs, and runs superseded by a newer automatic run are left alone.

Failures are reported by the orchestrator's `alert` job and
`unsloth-prebuilt-deadman.yml` (daily, 09:37 UTC) through
`.github/actions/prebuilt-alert`: one deduplicated issue per key
(`audio-cpp-prebuilt-pipeline`, `audio-cpp-prebuilt-stale`, label
`prebuilt-failure`), commented on while failing and closed on recovery. The
dead-man alerts when the newest aged upstream release has had no published
`<tag>-unsloth.<N>` here for more than 36 hours after it aged (the next daily
run plus a multi-hour build plus one retry). Issues are disabled on this
repository today, so alerts land in the run summary with an `::error::`
annotation until issues are enabled.

## Asset contract

```
audio-<tag>-bin-<os>-<arch>-<accel>.<ext>      <os> = ubuntu | windows | macos
audio-<tag>-cudart-windows-<arch>-<accel>.zip     Windows CUDA runtime companion
```

This is upstream's naming, which the Studio installer's resolver also reads.

| asset | runner | toolkit / notes | required |
| --- | --- | --- | --- |
| `audio-<tag>-bin-ubuntu-x64-cpu.tar.gz` | `ubuntu-22.04` | gcc-13, AVX2 baseline, static | yes |
| `audio-<tag>-bin-ubuntu-x64-cpu-portable.tar.gz` | `ubuntu-22.04` | every ggml CPU variant, picked at runtime | yes |
| `audio-<tag>-bin-ubuntu-x64-vulkan.tar.gz` | `ubuntu-22.04` | LunarG `vulkan-sdk` (jammy) | yes |
| `audio-<tag>-bin-ubuntu-x64-cuda12.tar.gz` | `ubuntu-22.04` + `nvidia/cuda:12.8.1-devel-ubuntu22.04` | lean, see below | yes |
| `audio-<tag>-bin-ubuntu-x64-cuda13.tar.gz` | `ubuntu-22.04` + `nvidia/cuda:13.0.2-devel-ubuntu22.04` | lean, see below | yes |
| `audio-<tag>-bin-ubuntu-arm64-cpu.tar.gz` | `ubuntu-22.04-arm` | gcc-13 | best-effort |
| `audio-<tag>-bin-windows-x64-cpu.zip` | `windows-2022` | upstream `build_windows.ps1`, `-CpuArch avx2` | yes |
| `audio-<tag>-bin-windows-x64-cpu-portable.zip` | `windows-2022` | `-CpuArch baseline`, llamafile off | yes |
| `audio-<tag>-bin-windows-x64-vulkan.zip` | `windows-2022` | Vulkan SDK 1.4.357.0 | yes |
| `audio-<tag>-bin-windows-x64-cuda12.zip` | `windows-2022` | CUDA 12.8.1 (Jimver), `ggml-cuda.dll` | yes |
| `audio-<tag>-bin-windows-x64-cuda13.zip` | `windows-2022` | CUDA 13.0.2 (Jimver), `ggml-cuda.dll` | yes |
| `audio-<tag>-cudart-windows-x64-cuda12.zip` | `windows-2022` | cudart64 / cublas64 / cublasLt64 / cufft64 of 12.8.1 | yes |
| `audio-<tag>-cudart-windows-x64-cuda13.zip` | `windows-2022` | the same DLLs of 13.0.2 | yes |
| `audio-<tag>-bin-windows-arm64-cpu.zip` | `windows-11-arm` | native LLVM, single baseline ARM CPU backend, static eSpeak | yes |
| `audio-<tag>-bin-windows-arm64-cuda13.zip` | `windows-11-arm` | CUDA 13.4.1, sm_121a / sm_120a + compute_100 PTX | yes |
| `audio-<tag>-cudart-windows-arm64-cuda13.zip` | `windows-11-arm` | ARM64 cudart / cuBLAS / cuBLASLt / cuFFT | yes |
| `audio-<tag>-bin-macos-arm64-metal.tar.gz` | `macos-15` | Metal on, deployment target 13.3 | yes |
| `audio-<tag>-bin-macos-x64-metal.tar.gz` | `macos-15-intel` | Metal off (name kept; Studio runs it on CPU) | yes |

Each release also carries `audio-cpp-prebuilt-manifest.json`,
`audio-cpp-prebuilt-sha256.json`, and the stamped source as
`audio.cpp-source-<tag>.tar.gz` and `audio.cpp-source-commit-<sha>.tar.gz`.

### Every bundle

- `audiocpp_server[.exe]` at the archive root (no top-level directory).
- eSpeak-ng linked statically, with `espeak-ng-data.bin` beside the server
  (Studio's eSpeak check looks for it there). Kokoro, KittenTTS, Piper and
  Inflect need it.
- Model specs compiled in (`AUDIOCPP_DEPLOYMENT_BUILD=ON`); the native model
  manager is off (`AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF`): Studio downloads
  models itself, so the server carries no TLS stack and needs no host
  `libssl`.
- Never `-march=native`: a runner with AVX-512 would ship a binary that faults
  on older CPUs. Linux links libstdc++/libgcc statically and keeps a glibc 2.35
  floor (checked); it still needs the host's `libgomp.so.1` (OpenMP, symbol
  versions up to `GOMP_4.0`).
- `LICENSE` (audio.cpp, Apache-2.0), `COPYING-NOTICE.txt`, and `licenses/`:
  `espeak-ng/COPYING` (GPL-3.0-or-later), the exact eSpeak-ng source tarball
  and build patch the build used, `espeak-ng/SOURCE.txt` (eSpeak-ng version,
  **commit**, tarball URL + sha256, and a written source offer), and the notices
  of ggml, sentencepiece, cJSON, libyaml and cpp-httplib.
- **Licensing:** because eSpeak-ng is statically linked, the phonemizer-enabled
  `audiocpp_server` in these bundles is a combined work distributed under
  **GPL-3.0-or-later**, even though audio.cpp itself is Apache-2.0.
- `BUILD_INFO.txt`, whose first line is `Compiled by the Unsloth team`, then the
  upstream tag + commit, accel, CUDA toolkit and architectures, eSpeak-ng
  commit, and the `--version` output.
- `UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json`: same file name and keys as the record
  Studio's installer writes (`schema_version`, `component`, `published_repo`,
  `release_tag`, `asset`, `backend`, `espeak`, `server_relpath`,
  `server_sha256`, `server_version`, `os`, `arch`), plus provenance
  (`upstream_tag`, `source_commit`, `runtime_line`, `cuda_toolkit`,
  `cuda_architectures`, `espeak_ng`, `files`, `studio_protocol`). The installer
  overwrites it with its own record on install; a bundle unpacked by hand and
  pointed at with `UNSLOTH_AUDIO_CPP_PATH` therefore still launches on the right
  backend.
- `audiocpp_server --version` prints `audio.cpp <tag>`, `git: <short sha> <date>`,
  the compiler and platform, and `backends: cpu[,cuda|,vulkan|,metal]`.

### Lean Linux CUDA bundles

Like unslothai/llama.cpp's Linux CUDA bundles, `ubuntu-x64-cuda12` /
`ubuntu-x64-cuda13` link cudart, cuBLAS (and through it cuBLASLt) and cuFFT
**dynamically and ship none of them**: Studio supplies them from PyTorch's
`nvidia-*` wheels on the loader path. NCCL is off (`GGML_CUDA_NCCL=OFF`); the
earlier Etherll bundle NEEDed `libnccl.so.2` and could not start without torch.
NEEDED of `audiocpp_server`: `libcudart.so.<N>`, `libcublas.so.<N>`,
`libcufft.so.<11|12>`, `libcuda.so.1` (driver), `libgomp.so.1`, libc. Packaging
refuses to ship any `libcudart*`, `libcublas*`, `libcufft*`, `libnccl*`,
`libnvrtc*` or `libnvJitLink*`, and the build fails if anything NEEDs NCCL.

CUDA architectures (both lines; CUDA 13 dropped sm_50-72):
`75-real;80-real;86-real;89-real;90-real;100-real;120-real;100-virtual`.
audio.cpp rewrites `120` to `120a` (Blackwell arch-specific SASS), and
arch-specific PTX cannot JIT anywhere else, so the forward-compatible PTX is
`compute_100` (sm_103 runs the sm_100 SASS; sm_110/121 and future GPUs JIT from
the PTX). Fatbins use ggml's `-compress-mode=size`. Expect roughly 200 MB per
Linux CUDA bundle (vs about 1 GB for a bundle carrying the CUDA libraries).

Windows x64 CUDA bundles use `GGML_BACKEND_DL` (kernels once, in `ggml-cuda.dll`)
and get the runtime DLLs from the separate `-cudart-` zip, which Studio skips
when the running torch already ships that CUDA line.

### Windows ARM64

Both ARM64 legs run natively on `windows-11-arm`, allowing static eSpeak's data
generators and the finished Kokoro/Moonshine CPU smoke tests to execute. Native
LLVM is required: ggml rejects MSVC ARM CPU kernels. Use the baseline single CPU
backend with `ENGINE_ENABLE_CPU_ALL_VARIANTS=OFF`; ggml's Windows ARM all-variants
branch is unsupported. These bundles require Windows 11.

The CUDA leg compiles the entire server, including audio.cpp's CUDA ISTFT and
sampling sources, rather than overlaying just a `ggml-cuda.dll`. NVIDIA's native
ARM64 nvcc and libnvvm compile against ARM64 target libraries; the shipped runtime
DLLs are also ARM64. CUDA 13.4.1 components and SHA256 values are pinned in
`scripts/unsloth/cuda_windows_arm64_redist.json`. CUDA 12 is not offered for
Windows ARM64. The static CUDA leg disables ggml VMM to avoid a direct
`nvcuda.dll` import on the GPU-less runner, so its CPU fallback can be tested with
the companion runtime ZIP. GPU execution still requires an NVIDIA GPU and a
compatible driver; hosted-runner smoke tests do not test GPU execution.

Every executable/DLL in each finished ARM64 bundle and CUDA runtime ZIP is
checked for ARM64 PE machine type. Existing signing, signature verification,
layout/import checks, manifests and checksums apply. All three ARM64 assets are
required by the atomic publication gate. To add them to an already released
upstream version, manually dispatch with a **new** `packaging_suffix` (for
example `unsloth.2`): the automatic resolver skips already published upstream
tags. Studio's pinned release/digests must subsequently be updated to adopt the
new assets.

## Manifest schema (`audio-cpp-prebuilt-manifest.json`)

```json
{
  "schema_version": 1,
  "component": "audio.cpp",
  "source_repo": "unslothai/audio.cpp",
  "source_commit": "<40-hex upstream commit>",
  "upstream_repo": "0xShug0/audio.cpp",
  "upstream_tag": "v0.8.2-audio8-perf-hotfix",
  "packaging_tag": "v0.8.2-audio8-perf-hotfix-unsloth.1",
  "studio_protocol": { "version": 1, "argv": ["--config", "<server.json>", "--no-ui", "..."], "...": "..." },
  "generated_at_utc": "2026-...Z",
  "artifacts": [
    {
      "asset": "audio-v0.8.2-audio8-perf-hotfix-unsloth.1-bin-ubuntu-x64-cuda13.tar.gz",
      "kind": "bundle", "os": "ubuntu", "arch": "x64", "accel": "cuda13", "backend": "cuda",
      "runtime_line": "cuda13", "cuda_toolkit": "13.0.2",
      "cuda_architectures": ["75-real", "...", "100-virtual"],
      "cuda_runtime_bundled": false, "min_os": "glibc-2.35", "espeak": true,
      "espeak_ng": { "version": "1.52.0", "commit": "<40-hex>", "sha256": "<hex>", "license": "GPL-3.0-or-later" },
      "server_sha256": "<hex>", "server_version": "audio.cpp ...", "sha256": "<hex>", "size": 0
    },
    {
      "asset": "audio-v0.8.2-audio8-perf-hotfix-unsloth.1-cudart-windows-x64-cuda13.zip",
      "kind": "cuda-runtime", "os": "windows", "arch": "x64", "accel": "cuda13",
      "backend": "cuda", "runtime_line": "cuda13", "sha256": "<hex>", "size": 0
    }
  ]
}
```

Bundle entries are generated from each bundle's **embedded**
`UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json`, so the manifest cannot disagree with
what was compiled. `server_version` is null where the build runner could not
execute the binary at packaging time. `audio-cpp-prebuilt-sha256.json` is a flat
`name -> {kind, repo, sha256, source_commit, upstream_tag}` index over every
bundle, both cudart zips, both source archives and the manifest.
`studio_protocol` pins the server contract Studio speaks (argv, config keys,
readiness probes, `/v1/audio/speech`, `/v1/audio/transcriptions`,
`/v1/tasks/run`); bump its `version` in `package_bundle.py` on a breaking
change.

## CI validation gate

`scripts/unsloth/validate_bundle.py` runs on the finished archive, on the
runner that built it:

1. **Layout + branding**: server, `espeak-ng-data.bin`, license files,
   `BUILD_INFO.txt` fingerprint, embedded info naming this asset; no CUDA
   runtime or NCCL library inside.
2. **Closure**: Linux: every NEEDED of every ELF resolves (`ldd`) except
   `libcuda.so.1` and, for the lean CUDA bundles, that line's cudart / cuBLAS /
   cuBLASLt / cuFFT; nothing NEEDs `libnccl`; newest `GLIBC_` symbol <= 2.35;
   no dynamic libstdc++. macOS: every non-system dylib reference resolves
   inside the bundle. Windows: every imported DLL is in the bundle, in
   System32, or host-provided (`nvcuda`, the CUDA runtime DLLs, `vulkan-1`).
3. **Run**: `--help` exits 0; `--version` lists the expected backends.
4. **Smoke**: the server is started exactly as Studio starts it (one model,
   `--config server.json --no-ui`, scratch HOME), `/health` and `/v1/models`
   must list the model id, then Kokoro 82M (`kokoro-82m-q8_0.gguf`, the eSpeak-ng
   path) synthesizes a sentence through `/v1/audio/speech` (asserted: >= 1 s of
   non-silent WAV) and Moonshine tiny (`moonshine-streaming-tiny-q8_0.gguf`)
   transcribes that speech, resampled to 16 kHz as Studio posts it, through
   `/v1/audio/transcriptions` (asserted: non-empty, >= 3 of the 7 prompt words).
   Models come from `audio-cpp/audio.cpp-gguf` at revision
   `83c5d96c03023ff5a7712570d057ce26c8769f98` (Studio's pin) and are cached with
   `actions/cache`.

The smoke runs on every x64/arm64 CPU, Vulkan and macOS leg on the CPU backend.
CUDA bundles get it too, on the CPU backend: on Linux against the toolkit's
`libcuda` stub (CUDA init fails cleanly without a GPU), on Windows because
`ggml-cuda.dll` is a dlopen'd module that just does not load without
`nvcuda.dll`. On macOS arm64 an extra, non-gating smoke runs on Metal. The
Windows Vulkan leg runs the smoke when a `vulkan-1.dll` is available on the
runner and falls back to the static checks otherwise (with a warning).

**Real GPU inference is not part of hosted CI.** It was verified by hand for
the lean Linux CUDA 13 bundle on an NVIDIA B200 with only PyTorch's cu13 wheel
libraries on the loader path; add a self-hosted GPU runner job calling
`validate_bundle.py --lean-cuda 13 --lib-path <torch nvidia libs> --smoke
--backend cuda` to gate it.

The assemble job then re-checks every bundle's fingerprint, refuses to publish
unless every required asset (table above) exists, and writes the two sidecars.

## Runners and secrets

- Only GitHub-hosted runners: `ubuntu-22.04`, `ubuntu-22.04-arm` (also
  `resolve`, `assemble` and `alert`, which run no built binary and queue far
  less than x64), `windows-2022`, `macos-15`, `macos-15-intel`. The Linux CUDA legs run inside
  `nvidia/cuda:<toolkit>-devel-ubuntu22.04` containers and can take several
  hours (8 CUDA targets on a 4-vCPU runner; timeout 350 minutes).
- **Windows signing**: the Windows jobs run in the `release-signing`
  environment and sign every PE with Azure Trusted Signing before packaging,
  then assert every PE in each zip is validly signed. They need
  `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`,
  `AZURE_TRUSTED_SIGNING_ACCOUNT_NAME` and `AZURE_CERTIFICATE_PROFILE_NAME`
  available to this repository's `release-signing` environment. On an
  `unslothai` repo a missing secret fails the job closed; dispatch with
  `sign_windows=false` for a build-only test run (publishing then refuses).
- macOS bundles are unsigned and unnotarized.
- `Settings -> Actions -> General`: Actions allowed; the workflow sets its own
  token permissions, so the repository default can stay read-only.
- Every third-party action is pinned to a commit SHA.

## Trust and reproducibility

- **Studio's in-repo `studio/audio_cpp_prebuilt_pins.json` is the trust
  anchor.** Publish, review the assets, then regenerate the pins
  (`python studio/install_audio_cpp_prebuilt.py --write-pins`) and point
  `DEFAULT_REPO` / `DEFAULT_TAG` at this release. The sha256 index here proves
  integrity against the same origin, not authenticity.
- Every bundle carries the `Compiled by the Unsloth team` fingerprint in
  `BUILD_INFO.txt`; the orchestrator refuses to publish an unbranded one.
- The stamped source tarballs are release assets, and each bundle's
  `licenses/espeak-ng/` carries the eSpeak-ng source it was built from, so the
  GPL corresponding source ships with the binaries.
