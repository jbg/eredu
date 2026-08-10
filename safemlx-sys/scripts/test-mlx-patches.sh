#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd -- "${script_dir}/../.." && pwd)"
build_dir="${1:-${repository_dir}/target/mlx-patch-tests}"

cmake \
  -S "${repository_dir}/safemlx-sys/src/mlx-c" \
  -B "${build_dir}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_ACCELERATE=OFF \
  -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_C_BUILD_EXAMPLES=OFF \
  -DMLX_C_BUILD_PATCH_TESTS=ON

cmake \
  --build "${build_dir}" \
  --target mlx-patch-tests \
  --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
