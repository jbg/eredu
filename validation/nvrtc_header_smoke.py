#!/usr/bin/env python3
"""Compile the CUDA JIT header stack with NVRTC without requiring a GPU."""

import ctypes
import os
import pathlib


NVRTC_SUCCESS = 0


def check(result: int, operation: str) -> None:
    if result != NVRTC_SUCCESS:
        raise RuntimeError(f"{operation} failed with NVRTC status {result}")


def main() -> None:
    configured = os.environ.get("MLX_CUDA_JIT_INCLUDE_DIRS", "")
    include_dirs = [pathlib.Path(path) for path in configured.split(os.pathsep) if path]
    cuda_home = pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    include_dirs.append(cuda_home / "include")
    missing = [str(path) for path in include_dirs if not path.is_dir()]
    if missing:
        raise RuntimeError("missing NVRTC include directories: " + ", ".join(missing))

    nvrtc = ctypes.CDLL("libnvrtc.so.12")
    program = ctypes.c_void_p()
    source = b"""
#include <cuda_runtime.h>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/numeric_conversion.h>
extern \"C\" __global__ void safemlx_header_smoke() {}
"""
    check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), source, b"safemlx_header_smoke.cu", 0, None, None
        ),
        "nvrtcCreateProgram",
    )
    try:
        options = [
            b"--device-as-default-execution-space",
            b"--gpu-architecture=compute_80",
        ] + [
            f"--include-path={path}".encode() for path in include_dirs
        ]
        option_array = (ctypes.c_char_p * len(options))(*options)
        result = nvrtc.nvrtcCompileProgram(program, len(options), option_array)
        if result != NVRTC_SUCCESS:
            log_size = ctypes.c_size_t()
            check(nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)), "log size")
            log = ctypes.create_string_buffer(log_size.value)
            check(nvrtc.nvrtcGetProgramLog(program, log), "program log")
            raise RuntimeError(log.value.decode(errors="replace"))
    finally:
        check(nvrtc.nvrtcDestroyProgram(ctypes.byref(program)), "nvrtcDestroyProgram")

    print("NVRTC CUDA/CuTe/CUTLASS header smoke test passed")


if __name__ == "__main__":
    main()
