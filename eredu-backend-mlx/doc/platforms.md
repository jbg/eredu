# MLX backend platform setup

Eredu's default MLX backend builds a native MLX C library through its
`safemlx-sys` implementation crate. The first build can take longer because
CMake configures and compiles the native dependency. The minimum supported
Rust version is 1.89.

## macOS on Apple silicon

The default `eredu` features enable the MLX backend with Accelerate and Metal.
Install Xcode or the Xcode Command Line Tools and CMake, then build normally:

```sh
xcode-select --install
brew install cmake
cargo build --release -p eredu
```

The default desktop target is macOS 14 or newer. MLX uses unified memory: CPU,
Metal shared, and GPU allocations compete for the same physical RAM even when
Eredu reports separate logical residency tiers.

## Apple device targets

Eredu's MLX backend can cross-build with Accelerate and Metal for Apple silicon
devices and simulators:

| Platform | Device target | Apple silicon simulator | Minimum OS |
| --- | --- | --- | --- |
| iOS / iPadOS | `aarch64-apple-ios` | `aarch64-apple-ios-sim` | 17.0 |
| tvOS | `aarch64-apple-tvos` | `aarch64-apple-tvos-sim` | 17.0 |
| visionOS | `aarch64-apple-visionos` | `aarch64-apple-visionos-sim` | 1.0 |

Install the target and build on a macOS host with Xcode:

```sh
rustup target add aarch64-apple-ios
cargo build -p eredu --release --target aarch64-apple-ios
```

Some Xcode installations provide the Metal toolchain as a separate component:

```sh
xcodebuild -downloadComponent MetalToolchain
```

The build writes `mlx.metallib` to
`target/<rust-target>/<profile>/safemlx-resources/mlx.metallib`. Add that file
to the application's Copy Bundle Resources phase without renaming it. To stage
it directly into an Xcode product, set `SAFEMLX_METALLIB_OUTPUT_DIR` before the
Cargo build.

The build honors `IPHONEOS_DEPLOYMENT_TARGET`, `TVOS_DEPLOYMENT_TARGET`, and
`XROS_DEPLOYMENT_TARGET`. Mac Catalyst and watchOS are not supported.

## Linux CPU

CPU-only x86-64 Linux builds require Git, a C++20 compiler, CMake 3.25 or newer,
and BLAS, LAPACK, and LAPACKE development packages. On Ubuntu or Debian:

```sh
sudo apt-get update
sudo apt-get install git cmake build-essential \
  libblas-dev liblapack-dev liblapacke-dev
cargo build --release -p eredu --no-default-features --features mlx
```

## Linux CUDA

The `cuda` feature requires an x86-64 NVIDIA GPU with compute capability 7.5 or
newer, a supported CUDA 12 or CUDA 13 toolkit, and cuDNN 9 development files.
Install the toolkit and cuDNN according to NVIDIA's packages for the host, then
build:

```sh
cargo build --release -p eredu --no-default-features --features mlx,cuda
```

MLX normally detects the local GPU architecture. On a build machine without an
accessible GPU, or when targeting another machine, set a semicolon-separated
CMake architecture list:

```sh
SAFEMLX_CUDA_ARCHITECTURES=80 \
  cargo build --release -p eredu --no-default-features --features mlx,cuda
```

The build also honors `CMAKE_CUDA_COMPILER`, `CUDAToolkit_ROOT`,
`CUDNN_INCLUDE_PATH`, and `CUDNN_LIBRARY_PATH`. CUDA and cuDNN shared libraries
must be visible to the dynamic loader when the application runs.

NCCL is opt-in because it adds native link requirements. Enable `nccl` and use
`NCCL_ROOT_DIR`, `NCCL_INCLUDE_DIR`, or `NCCL_LIB_DIR` when discovery needs
help:

```sh
cargo build --release -p eredu --no-default-features --features mlx,nccl
```

## Windows x86-64

Native Windows uses the Rust MSVC target. CPU builds require Visual Studio 2022
Build Tools with Desktop development with C++, Git, CMake 3.25 or newer, and
Ninja.

CUDA builds additionally require CUDA 12.9 or 13.0 and the matching cuDNN 9
development archive. From a Visual Studio x64 developer PowerShell, point the
build at the installations:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:CUDAToolkit_ROOT = $env:CUDA_PATH
$env:CMAKE_CUDA_COMPILER = "$env:CUDA_PATH\bin\nvcc.exe"
$env:CUDNN_INCLUDE_PATH = "C:\tools\cudnn\include"
$env:CUDNN_LIBRARY_PATH = "C:\tools\cudnn\lib\x64"
$env:SAFEMLX_CUDA_ARCHITECTURES = "75"
$env:CMAKE_GENERATOR = "Ninja"
$env:PATH = "$env:CUDA_PATH\bin;C:\tools\cudnn\bin\x64;$env:PATH"

cargo build --release -p eredu --no-default-features --features mlx,cuda
```

Eredu's MLX backend builds `mlx` and `mlxc` as DLLs and stages them, along with
the fetched OpenBLAS DLL, next to Cargo binaries and examples. CUDA and cuDNN
DLLs must remain discoverable through their configured directories or `PATH`.

Common CUDA configuration failures:

- If `nvcc.exe` is missing, make `CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT`
  refer to the same toolkit.
- If CUDA import libraries are missing, install the cuBLAS, cuFFT, NVRTC, and
  runtime development components.
- If cuDNN headers or libraries are missing, point `CUDNN_INCLUDE_PATH` at the
  directory containing `cudnn.h` and `CUDNN_LIBRARY_PATH` at `lib\x64`.
- If the executable cannot find DLLs, add the CUDA and cuDNN `bin` directories
  to `PATH`.
- If the GPU architecture is rejected, set `SAFEMLX_CUDA_ARCHITECTURES` to a
  supported compute capability of 75 or newer.

The native MLX NumPy and SafeTensors paths use the Windows narrow-character
runtime, so use ASCII paths for those two formats. Rust-backed GGUF I/O supports
native Unicode paths. Windows ARM CUDA and Windows NCCL are not supported.

## Backend feature selection

Apple features are ignored on non-Apple targets, but portable applications
should still select their intended backend explicitly.

The `eredu-cli` crate has its own `cuda` feature, which enables CUDA in Eredu
and its MLX implementation layer.

At the facade layer, `mlx` selects the MLX backend. The weak `metal`, `cuda`,
`image`, `audio`, and `nccl` features configure that backend but never select
it. Direct facade builds therefore combine `mlx` with the capabilities they
need. Selecting `mlx,cuda` with default features disabled does not enable Metal
or Accelerate. The CLI's `cuda` convenience feature selects the facade's `mlx`
and `cuda` features together.
