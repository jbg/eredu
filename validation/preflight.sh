#!/usr/bin/env bash
# Run the local CPU Ring smoke suite.
set -euo pipefail
cd "$(dirname "$0")/.."

export MLX_ENABLE_TF32="${MLX_ENABLE_TF32:-0}"
python3 validation/run_native_ring_tests.py --test-timeout 240 --suite-timeout 480 --fail-fast
