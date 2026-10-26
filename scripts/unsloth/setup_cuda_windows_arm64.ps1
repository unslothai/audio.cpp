# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
param([Parameter(Mandatory)][string]$Destination, [Parameter(Mandatory)][string]$Stage)
$ErrorActionPreference = 'Stop'
# CUDA's Windows ARM64 target uses the x64 host compiler (emulated on Windows ARM).
# Pin every component and digest from NVIDIA's redistrib_13.4.1.json, as well as
# the ARM64 cuFFT needed by audio.cpp's own CUDA ISTFT implementation.
$manifest = Get-Content -Raw (Join-Path $PSScriptRoot 'cuda_windows_arm64_redist.json') | ConvertFrom-Json
New-Item -ItemType Directory -Force -Path $Destination, $Stage | Out-Null
foreach ($component in $manifest.components) {
    $zipName = Split-Path $component.relative_path -Leaf
    $zip = Join-Path $Stage $zipName
    curl.exe -fsSL --retry 3 -o $zip "https://developer.download.nvidia.com/compute/cuda/redist/$($component.relative_path)"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $zipName" }
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $component.sha256) {
        throw "SHA256 mismatch: $zipName"
    }
    tar.exe -xf $zip -C $Stage
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed: $zipName" }
    $tree = Join-Path $Stage ([IO.Path]::GetFileNameWithoutExtension($zipName))
    robocopy $tree $Destination /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Toolkit copy failed: $zipName" }
}
foreach ($lib in 'cudart.lib', 'cudart_static.lib', 'cuda.lib', 'cublas.lib', 'cublasLt.lib', 'cufft.lib') {
    if (-not (Test-Path -LiteralPath (Join-Path $Destination "lib/arm64/$lib"))) { throw "Missing ARM64 $lib" }
}
foreach ($pattern in 'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll', 'cufft64_*.dll') {
    if (-not (Get-ChildItem -Path "$Destination/bin/arm64/$pattern")) { throw "Missing ARM64 $pattern" }
}
"CUDA_PATH=$Destination" | Add-Content $env:GITHUB_ENV
"$Destination/bin" | Add-Content $env:GITHUB_PATH
exit 0
