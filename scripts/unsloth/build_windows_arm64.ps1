# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
param(
    [Parameter(Mandatory)][string]$Source,
    [Parameter(Mandatory)][string]$Build,
    [Parameter(Mandatory)][string]$Version,
    [ValidateSet('cpu', 'cuda13')][string]$Accelerator = 'cpu'
)
$ErrorActionPreference = 'Stop'
if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'Arm64') {
    throw 'This recipe requires a native Windows ARM64 runner (static eSpeak data generators must run).'
}
# eSpeak's child CMake receives compiler paths, so clang itself must default to
# ARM64 too; a target flag only on the parent would accidentally build x64 data tools.
$target = (& clang-cl --version) -join "`n"
if ($LASTEXITCODE -ne 0 -or $target -notmatch '(?m)^Target: (aarch64|arm64).*-windows') {
    throw "Native ARM64 LLVM is required; clang target is $target"
}
$cuda = if ($Accelerator -eq 'cuda13') { 'ON' } else { 'OFF' }
if ($cuda -eq 'ON') {
    # nvcc --use-local-env compares the PATH compiler with -ccbin literally.
    # Resolve the directory so VCToolsInstallDir's trailing slash cannot leave
    # a doubled separator, and put the same host compiler as the toolchain first.
    $hostBin = Join-Path $env:VCToolsInstallDir 'bin/Hostarm64/arm64'
    if (-not (Test-Path -LiteralPath (Join-Path $hostBin 'cl.exe'))) {
        $hostBin = Join-Path $env:VCToolsInstallDir 'bin/Hostx64/arm64'
    }
    $hostBin = (Resolve-Path -LiteralPath $hostBin).Path
    $env:PATH = "$hostBin;$env:PATH"
}
$configureArgs = @('-S', $Source, '-B', $Build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_TOOLCHAIN_FILE=$PSScriptRoot/windows_arm64_native.cmake",
    "-DAUDIOCPP_VERSION=$Version", "-DENGINE_ENABLE_CUDA=$cuda",
    '-DENGINE_ENABLE_OPENMP=OFF', '-DGGML_OPENMP=OFF',
    '-DENGINE_ENABLE_VULKAN=OFF', '-DENGINE_ENABLE_METAL=OFF',
    '-DENGINE_ENABLE_LLAMAFILE=OFF', '-DENGINE_ENABLE_NATIVE_CPU=OFF',
    # ggml CPU_ALL_VARIANTS rejects Windows ARM. Use its static single CPU backend.
    '-DENGINE_ENABLE_CPU_ALL_VARIANTS=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a',
    '-DENGINE_BUILD_TESTS=OFF', '-DAUDIOCPP_STATIC_ESPEAK=ON',
    '-DAUDIOCPP_DEPLOYMENT_BUILD=ON', '-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF')
if ($cuda -eq 'ON') {
    $configureArgs += @('-DENGINE_ENABLE_CUDA_GRAPHS=ON', '-DGGML_CUDA_NCCL=OFF',
        # No direct nvcuda.dll import: the GPU-less native runner can load the
        # static CUDA build with the companion runtime ZIP and exercise CPU fallback.
        '-DGGML_CUDA_NO_VMM=ON', '-DCMAKE_CUDA_ARCHITECTURES=121-real;120-real;100-virtual')
}
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw 'ARM64 configure failed' }
& cmake --build $Build --config Release --target audiocpp_server -j 3
if ($LASTEXITCODE -ne 0) { throw 'ARM64 build failed' }
exit 0
