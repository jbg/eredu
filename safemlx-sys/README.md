# safemlx-sys

`safemlx-sys` provides low-level Rust bindings and native build integration for
the MLX C API used by `safemlx`. Most applications should use the safe
[`safemlx`](../safemlx/) crate instead.

The crate vendors MLX C, fetches the compatible MLX source during the native
build, and exposes checked-in bindings. Its native surface includes SafeMLX's
completion events, typed host-transfer storage, variable-count all-to-all, and
packed-quantization support.

## Backends

| Target | Backend selection |
| --- | --- |
| macOS and supported Apple device targets | Accelerate and Metal through the default features |
| x86-64 Linux | CPU by default; optional CUDA and NCCL |
| x86-64 Windows MSVC | CPU by default; optional CUDA |

Native compilers, CMake, platform libraries, CUDA/cuDNN setup, Apple deployment
targets, and `mlx.metallib` bundling are covered in [Platform
setup](../doc/platforms.md).

## Features

- `accelerate`: build the Accelerate backend on Apple platforms.
- `metal`: build the Metal backend on Apple platforms.
- `cuda`: build MLX with CUDA support on x86-64 Linux or Windows.
- `nccl`: enable MLX's optional NCCL distributed backend on Linux.

Applications normally select these features through `safemlx` rather than
depending on `safemlx-sys` directly.

## License

The Rust crate is MIT licensed. Vendored components retain their stated
licenses and attribution files.
