#!/usr/bin/env bash
# Fast native feedback, shared by local development and CI on Apple silicon.
set -euo pipefail
cd "$(dirname "$0")/.."

export MLX_ENABLE_TF32="${MLX_ENABLE_TF32:-0}"
python3 validation/run_native_unit_tests.py --fail-fast
python3 validation/run_native_unit_tests.py --suite ring-smoke --test-timeout 240 --suite-timeout 480 --fail-fast
