# safemlx-sys

`safemlx-sys` provides low-level Rust bindings and native build integration for
the MLX C API used by Eredu's `safemlx` implementation layer. Applications
should use [`eredu`](https://github.com/jbg/eredu/tree/main/eredu) rather than depending on this crate directly.

The crate vendors MLX C, the compatible MLX source archive, and its common CPU
and Metal build dependencies, and exposes checked-in bindings. CMake verifies
each vendored archive's SHA-256 before extracting and patching it in the build
directory. Source provenance is recorded in
[`vendor/SOURCES.md`](vendor/SOURCES.md). The native surface includes the
completion events, typed host-transfer storage, variable-count all-to-all, and
packed-quantization support required by Eredu's MLX backend.

## Backends

| Target | Backend selection |
| --- | --- |
| macOS and supported Apple device targets | Accelerate and Metal through the default features |
| x86-64 Linux | CPU by default; optional CUDA and NCCL |
| x86-64 Windows MSVC | CPU by default; optional CUDA |

Native compilers, CMake, platform libraries, CUDA/cuDNN setup, Apple deployment
targets, and `mlx.metallib` bundling are covered in [Platform
setup](https://github.com/jbg/eredu/blob/main/doc/platforms.md).

## Features

- `accelerate`: build the Accelerate backend on Apple platforms.
- `metal`: build the Metal backend on Apple platforms.
- `cuda`: build MLX with CUDA support on x86-64 Linux or Windows.
- `nccl`: enable MLX's optional NCCL distributed backend on Linux.

Applications normally select these features through `eredu` rather than
depending on `safemlx-sys` directly.

## License

The Rust crate is MIT licensed. Vendored components retain their stated
licenses and attribution files.
