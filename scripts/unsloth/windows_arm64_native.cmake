# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Native Windows ARM64 only: do NOT set CMAKE_SYSTEM_NAME. Static eSpeak builds
# and executes data generators, which cannot be built as a cross-compile here.
set(CMAKE_SYSTEM_PROCESSOR ARM64)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER_TARGET aarch64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET aarch64-pc-windows-msvc)

if(ENGINE_ENABLE_CUDA)
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda)
    file(TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" _vc)
    if(NOT EXISTS "${_cuda}/bin/nvcc.exe")
        message(FATAL_ERROR "Install the CUDA Windows ARM64 redistributables first")
    endif()
    # nvcc's host front-end is x64; target code and all link libraries are ARM64.
    set(CMAKE_CUDA_COMPILER "${_cuda}/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
    if(EXISTS "${_vc}/bin/Hostarm64/arm64/cl.exe")
        set(CMAKE_CUDA_HOST_COMPILER "${_vc}/bin/Hostarm64/arm64/cl.exe" CACHE FILEPATH "" FORCE)
    else()
        set(CMAKE_CUDA_HOST_COMPILER "${_vc}/bin/Hostx64/arm64/cl.exe" CACHE FILEPATH "" FORCE)
    endif()
    set(CMAKE_CUDA_FLAGS_INIT "-target-dir=arm64")
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "" FORCE)
    set(CUDAToolkit_ROOT "${_cuda}" CACHE PATH "" FORCE)
    foreach(_lib cudart cudart_static cublas cublasLt cufft)
        set(CUDA_${_lib}_LIBRARY "${_cuda}/lib/arm64/${_lib}.lib" CACHE FILEPATH "" FORCE)
    endforeach()
    set(CUDA_cuda_driver_LIBRARY "${_cuda}/lib/arm64/cuda.lib" CACHE FILEPATH "" FORCE)
endif()
