#!/usr/bin/env bash
# Fast native feedback, shared by local development and CI on Apple silicon.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo test --locked -p eredu-backend-mlx --features metal,accelerate --lib \
  composition::mlx_architecture_conformance::native_mlx_execution_is_available -- --exact
# Keep normal test-runner parallelism: contention exposed the session defects.
cargo test --locked -p eredu-backend-mlx --features metal,accelerate --lib
cargo test --locked -p eredu-backend-mlx --features metal,accelerate --lib _ring:: -- \
  --ignored --skip moshi_ring_tp2_native_model_parity \
  --skip moshi_ring_tp2_personaplex_model_parity --test-threads=1
