# Mixed-width Metal projection baseline

This is stage 1 of the proposed prefill optimization: characterize existing
arithmetic, native dispatch, allocation and timing before changing kernels.
No GEMM, GEMV, precision, conversion-retention or inference scheduling policy has
changed. The baseline records real arrays from ordinary prefill, then replays
representative projections in isolation.

## Provenance and reproduction

The checkpoint is official `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`, revision
`6767265158422fb8a19c62ceb45f16f05363615b`, file
`LFM2.5-1.2B-Instruct-BF16.gguf`, 2,343,326,528 bytes, SHA-256
`3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b`.
The runner hashes the complete file before inference. The source remains outside
the tracked tree. The tokenizer and mixed F32/BF16 tensors are unchanged.

Measurements ran on an Apple M3 Ultra with 256 GiB unified memory and macOS 26.6.2,
using a release build, Rust 1.98.0, the pinned MLX 0.32.0 source plus the repository's
native patches, and Metal. Allocator caching was disabled. Policies were disabled,
256 MiB (the managed-default allowance, selected explicitly), and unlimited.
No other agent builds or benchmarks ran during the measurements.

The [machine-readable evidence](../validation/results/projection-baseline-2026-09-25.json)
records binary, checkpoint, native source archive, trace patch, runner and working
source hashes; exact process commands; all ordinary requests; settled projection
layouts; output-bit fingerprints; first-replay samples and warm statistics; and
actual native pipeline selection keys. Large raw logs and per-process JSON remain
in `/tmp/eredu-projection-baseline-2026-09-25-final`. The measured executable SHA-256
is `f0fa0a0fd8456c6fcdd3e6d38692ddf3a319361d8610d6fedc1abf2deb8b2fa4`.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --release \
  --example projection_baseline --no-default-features \
  --features mlx,metal,projection-profiling --locked --offline
python3 validation/projection_baseline.py \
  --binary target/release/examples/projection_baseline \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --output /tmp/eredu-projection-baseline --samples 5
```

Native Metal access is required. The runner sanitizes MLX environment overrides,
verifies exit status, exact output comparisons and trace/timing class agreement,
and rejects missing native selection evidence. Both build and run commands work
from the repository root. The optional profiling feature is disabled in ordinary
builds; the portable facade does not acquire native dependencies.

## What is measured

Each policy/length pair uses three fresh processes:

1. **Ordinary inference:** a cold request and five reset/warm requests, each using
   repeated token ID 1 and eight greedy predictions. Loading is excluded. Reset
   preserves admitted conversions. These are the only ordinary latency/peak data.
2. **Capture and replay:** collect real mixed-width dense/readout invocations
   through the first prediction without adding evaluations in the collection hook.
   At an explicit finish boundary, settle arrays and group by dtype, shape, actual
   strides, projection site and observed dispatch path. Retain the first example
   and count for every class. Captured owners change graph retention, so this
   process's full-model latency and peak are explicitly invalid as ordinary metrics.
3. **Selection trace:** replay the same classes with
   `MLX_METAL_LOG_KERNEL_SELECTION=1`. Native MLX logs pipeline requests, including
   cache hits and function-constant specialization keys. Only requests inside
   synchronized phase markers are attributed. Logging is disabled in timing runs.

The replay phases are an explicit F32 cast, GEMM with an already evaluated F32
weight, native mixed matmul, and the current Eredu projection dispatch. The latter
inherits conversion ownership from capture and previous replays; the other three
phases isolate defined conversion states. The preconverted GEMM's resident F32
weight is charged in its baseline, not its incremental peak; its byte count is
recorded separately. Thus its smaller incremental peak is not free conversion
storage or a measurement of the proposed kernel.

Every replay reports one first sample, then five measured warm samples after an
additional unreported warmup. Capture/reference creation may already warm kernel
caches; first replay does not mean cold JIT. Wall time includes graph building,
submission and completion. GPU time uses native completion timing and excludes
eager work done while constructing the graph. Neither duration is a precise
allocation-only timer, and independently synchronized phase times are not additive
end-to-end attribution. Peaks are MLX active allocations above each settled sample's
baseline, not RSS, driver memory or allocator-cache high water.

Rows 2, 4, 8 and 16 slice leading real 128-position activations. They are
verification-sized projection workloads, not full speculative decoding. Original
128- and 2,000-row invocations retain their observed input shapes and layouts.
All rows of each non-cast replay output are compared as F32 bits: original-sized cases use
the captured output; sliced cases use explicit F32-weight native matmul. Each
reference also has a SHA-256 fingerprint. This validates baseline arithmetic; it
is not a fresh independent-reference validation of the entire language model.

## Observed projection workload

Both prompt lengths contain **93 mixed-width prefill projections**, across the
same six weight shapes. Weights are BF16, inputs/outputs F32; all observed weight
rows are contiguous. Classes split further when conversion admission differs.
The 256 MiB case has ten such classes; disabled/unlimited each have six.

| Weight shape `[N,K]` | Invocations | F32 cast per matrix (MiB) |
|---|---:|---:|
| `[2048,2048]` | 22 | 16 |
| `[512,2048]` | 12 | 4 |
| `[6144,2048]` | 10 | 48 |
| `[8192,2048]` | 32 | 64 |
| `[2048,8192]` | 16 | 64 |
| `[65536,2048]` | 1 | 512 |

These matrix payloads sum to 4,464 MiB, matching the unlimited retained conversion
payload. The managed-size policy admits 256 MiB. The observed vocabulary/readout
projection receives **all prompt rows** in this request path: its output alone is
32 MiB at 128 rows and 500 MiB at 2,000 rows. A prefill optimization must preserve
that baseline unless a separate semantic readout change is deliberately made.

Native selection evidence establishes two families of FP32 arithmetic to preserve:

- At 2–16 rows, `[2048,2048]`, `[512,2048]` and `[2048,8192]` use NT split-K GEMM
  with `BM16/BN32/BK16/WM2/WN2`, followed by the split-K accumulator. At 128 rows,
  those shapes use `BM32/BN32/BK16/WM2/WN2` split-K plus its accumulator.
- The other short cases use regular NT GEMM with `BM32/BN64/BK16/WM1/WN2`.
  At 128 rows, `[8192,2048]` and the vocabulary projection use
  `BM64/BN64/BK16/WM2/WN2`, while `[6144,2048]` retains the smaller tile.
- All 2,000-row cases use regular NT GEMM. `[512,2048]` uses the smaller tile;
  the other shapes use `BM64/BN64/BK16/WM2/WN2`. Native mixed matmul includes
  `vn_copybfloat16float32`; preconverted GEMM omits that kernel.

Full specialization keys, including aligned/tail flags, are retained in the JSON.
This selection inventory is measured on this M3 Ultra, not a universal Metal
selector. No F16 checkpoint, noncontiguous-weight case, K tail, M5/NAX, CUDA or
native distributed run is claimed by these measurements.

## Timing and allocation results

Ordinary request medians over five warm samples; active peaks are the largest
warm **prefill** peak. Numbers exclude loading and include existing inference work.

| Retention | 128 first token (ms) | 2,000 first token (ms) | 128 prefill active peak (MiB) | 2,000 prefill active peak (MiB) |
|---|---:|---:|---:|---:|
| Disabled | 97.43 | 717.39 | 7156.7 | 4538.8 |
| 256 MiB | 92.62 | 713.71 | 7156.7 | 4794.7 |
| Unlimited | 23.77 | 688.16 | 7176.7 | 8028.6 |

Unlimited retention accelerates short prefill substantially but retains the large
conversion payload. At 2,000 positions the difference is much smaller; computation
and other model mechanisms remain important. These prompt lengths also cross the existing attention tile-evaluation threshold,
so full-model peaks need not grow monotonically with length. This does not predict
the speed of a future mixed-storage GEMM.

Representative isolated cases from the 256 MiB run, using matrices whose ordinary
invocations took the native fallback. Durations are warm medians; peaks exclude
already resident operands. The explicit cast alone measured exactly the 64 or
512 MiB output payload shown above.

| Rows | Weight `[N,K]` | Cast wall / GPU (ms) | Preconverted GEMM wall / GPU (ms) | Native mixed wall / GPU (ms) | Native mixed incremental peak (MiB) |
|---:|---|---:|---:|---:|---:|
| 8 | `[8192,2048]` | 1.485 / 0.128 | 0.573 / 0.295 | 1.891 / 0.462 | 64.250 |
| 128 | `[8192,2048]` | 1.494 / 0.128 | 0.804 / 0.463 | 2.191 / 0.661 | 68.000 |
| 2,000 | `[8192,2048]` | 1.496 / 0.120 | 4.477 / 3.012 | 5.750 / 3.125 | 126.500 |
| 8 | `[2048,8192]` | 1.476 / 0.128 | 0.371 / 0.097 | 1.643 / 0.231 | 64.562 |
| 128 | `[2048,8192]` | 1.488 / 0.129 | 0.776 / 0.452 | 2.120 / 0.650 | 67.000 |
| 2,000 | `[2048,8192]` | 1.524 / 0.120 | 3.931 / 3.164 | 5.011 / 3.274 | 79.625 |
| 128 | `[65536,2048]` | 12.466 / 1.148 | 2.896 / 1.918 | 14.392 / 3.055 | 544.000 |
| 2,000 | `[65536,2048]` | 12.148 / 1.167 | 33.888 / 23.333 | 45.717 / 24.611 | 1012.000 |

The 2,000-row vocabulary case accounts for 512 MiB of conversion and 500 MiB of
output in its incremental peak. Split-K cases additionally expose partial-result
workspace; the new kernel must preserve the applicable reduction and account for
that storage. Wall time materially exceeds GPU copy time, so eliminating the
conversion should target allocation/graph/submission costs as well as bandwidth.
The baseline does not isolate each of those host costs further.

## Validation and next implementation target

All six policy/position pairs passed in 18 fresh processes. There are 132 replay
workloads and 528 phase profiles. Every replay's non-cast outputs matched the
specified reference bits. Ordinary reset replay, capture and native-selection
tracing produced the same eight tokens for each prompt length. Trace and timing
processes had identical class inventories. The capture bound/nesting/drop regression
and trace-attribution parser tests pass; strict backend/example Clippy passes.

The first kernel prototype should target the measured regular NT FP32 path for
`[8192,2048]` and the vocabulary matrix, widening BF16 only while loading tiles.
Short-batch coverage also requires the measured split-K path and its accumulator;
substituting the existing single-row GEMV reduction would change the arithmetic
contract. Enable each optimized case only after bitwise comparison with these
real-array baselines and new layout/tail fixtures, then repeat this matrix. Retain
fallbacks and conservative forecasts wherever the new mechanism is not established.
