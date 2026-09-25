# Mixed-storage FP32 GEMM prototype

Stage two adds a loader-only native operation for the pinned MLX 0.32.0 FP32
GEMM. It is available explicitly through
`safemlx::fast::try_mixed_storage_gemm` and the optional projection profiler.
Ordinary inference selection and memory forecasts remain at their existing
stage-one behavior. Shared projection integration is stage three; forecast
mechanism facts are stage four.

## Arithmetic and storage contract

The operation computes `F32[M,K] @ narrow[N,K].T -> F32[M,N]`. It accepts F16 or
BF16 weight storage. The native patch extends the existing `BlockLoader` with a
separate storage type. Narrow weights widen as they enter the same FP32 shared
memory tile; activations are never narrowed. The homogeneous loader keeps its
original vector-copy implementation.

The distinct inference primitive delegates to the pinned native Matmul dispatch.
Regular GEMM and split-K keep native tile geometry, shared-memory padding, loop
order (including the different regular/split-K tail handling), MMA, accumulator,
partition count/size and final reduction. Bias remains a separate ordinary add;
there is no fused epilogue substitution. Kernel cache keys include the weight
storage dtype. Homogeneous kernel keys and ordinary Matmul promotion are unchanged.

The native operation returns `None` without evaluation for unsupported inputs:
only settled row-contiguous rank-two operands, `2 <= M <= 2000`,
`2 <= N <= 65536`, `1 <= K <= 8192`, and the validated Apple M3 Ultra are admitted.
Contiguous views with offsets are supported. Padded/column-major layouts, lazy
operands, single-row GEMV, CPU, CUDA, other devices (including NAX), system MLX
and non-JIT builds are excluded. These are prototype coverage limits, not model
limitations. Native autodiff, vmap and compilation transformations are unsupported
by the distinct primitive.

The operation creates the F32 output and, when selected by native dispatch,
F32 split-K partials. It creates no full promoted weight. Input readiness is a
validation boundary in this prototype; stage-three integration must preserve lazy
execution rather than introducing per-projection evaluations to satisfy it.

## Reproduction

Use the same official BF16 GGUF revision and SHA-256 as the
[stage-one baseline](projection-baseline.md). Checkpoint sources remain outside
the tracked tree. Run native tests serially because allocator counters are global:

```sh
CARGO_INCREMENTAL=0 cargo test -p safemlx --release \
  --test mixed_storage_gemm --locked --offline \
  -- --include-ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 cargo build -p eredu --release \
  --example projection_baseline --no-default-features \
  --features mlx,metal,projection-profiling --locked --offline
python3 validation/projection_baseline.py \
  --binary target/release/examples/projection_baseline \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --output /tmp/eredu-mixed-storage-gemm --samples 5 \
  --mixed-storage-prototype \
  --reference-manifest validation/results/projection-baseline-2026-09-25.json
```

The two hardware-specific tests are ignored by default and fail if explicitly run
on an unsupported device. The CPU rejection test runs normally and requires no
GPU. The runner uses fresh processes for ordinary timing, real projection replay,
and kernel tracing, at 128 and 2,000 positions with disabled, 256 MiB and unlimited
conversion retention. Timing runs disable kernel tracing and allocator caching.
Short rows 2, 4, 8 and 16 slice actual 128-position activations. For the prototype,
the profiler settles a contiguous `[M,K]` view before timing and compares all
flattened output bits. Restoring the original leading dimensions belongs to the
shared projection integration.

Each eligible replay adds `mixed_storage_prototype` to the four baseline phases.
It must match every output bit of the captured projection (or the explicit-F32
reference for sliced inputs). Native selection traces must match preconverted
GEMM after removing only the weight-storage suffix; the split-K accumulator must
remain present and unchanged. Measured peak growth must equal preconverted GEMM's
output/partial workspace. Unsupported replays fail rather than benchmarking a
fallback under the prototype label. The reference manifest additionally gates all
projection fingerprints, class inventories and ordinary cached-generation tokens
against the original pre-patch stage-one evidence.

The preconverted comparison includes a separately reported resident F32 weight;
its incremental peak is not total storage. Replay timings are isolated operations,
not additive full-model attribution. Capturing retains array owners and invalidates
ordinary inference peak/latency measurement, as in stage one.

## Validation results, 2026-09-25

Validation ran on Apple M3 Ultra, 256 GiB, macOS 26.6.2, Rust 1.98.0,
with the pinned vendored MLX and a release build. The measured executable SHA-256
is `359a152ec81d465c246fee02a8eca3ee0cb0cc52590e277f9179b6266b00f7a8`.
The native mixed-storage patch SHA-256 is
`19c4a4308e04ff8b961991dd5f803f35946f27169fdeee38dbcd1a03102c724d`.

[Machine-readable evidence](../validation/results/mixed-storage-gemm-2026-09-25.json)
contains the exact commands, source/checkpoint/binary hashes, native selection
keys, output fingerprints, first samples, warm statistics and ordinary baseline
requests. Raw logs and per-process results are in
`/tmp/eredu-mixed-storage-gemm-2026-09-25`.

- **290 synthetic F16/BF16 cases** passed exact-bit output and separate-bias
  comparisons, including all six dominant weight shapes, regular tile boundaries,
  small K (including the degenerate NN layout at K=1), MN/K tails, the split-K BM16/BM32 boundary and residual partitions.
  Active-allocation peaks equal output plus native split-K partials, including
  the pinned allocator's 16 KiB page rounding for larger buffers.
- Layout tests passed for contiguous views with nonzero offsets and rejection of
  matching-shape column-major weights, padded weights, column-major activations,
  narrow activations, lazy inputs/weights, single-row inputs and CPU streams.
  Unsupported lazy operands remain unevaluated.
- **132 real projection workloads / 660 phase profiles**, in 18 isolated
  processes, passed exact-bit comparisons at 2/4/8/16/128/2,000 rows. All six
  policy/position pairs also match the original stage-one class inventories,
  output-bit fingerprints and eight cached-generation tokens.
- Every prototype phase selected the same native regular/split-K geometry and
  accumulator as preconverted GEMM, with only the weight-storage suffix changed.
  None selected a conversion kernel. Peak growth equaled the preconverted GEMM
  phase in every workload.
- Strict Clippy passed for safemlx, backend profiling and the example. CPU-only
  safemlx tests, portable facade tests (27 passed, one existing ignored test),
  five Python evidence-validator tests, formatting and diff checks passed.

Representative **256 MiB policy, observed fallback** cases, reporting medians
of five warm samples. Peaks are incremental active allocation above settled
operands. Durations are wall / GPU milliseconds:

| Rows | Weight `[N,K]` | Cast + GEMM (ms) | Prototype (ms) | Peak before → after (MiB) |
|---:|---|---:|---:|---:|
| 8 | `[8192,2048]` | 1.876 / 0.467 | 0.537 / 0.281 | 64.250 → 0.250 |
| 128 | `[8192,2048]` | 2.154 / 0.660 | 0.791 / 0.461 | 68.000 → 4.000 |
| 2,000 | `[8192,2048]` | 5.728 / 3.128 | 4.492 / 2.934 | 126.500 → 62.500 |
| 8 | `[2048,8192]` | 1.633 / 0.237 | 0.366 / 0.094 | 64.5625 → 0.5625 |
| 128 | `[2048,8192]` | 2.113 / 0.649 | 0.778 / 0.458 | 67.000 → 3.000 |
| 2,000 | `[2048,8192]` | 5.300 / 3.434 | 4.254 / 3.535 | 79.625 → 15.625 |
| 8 | `[65536,2048]` | 13.194 / 2.063 | 1.193 / 0.498 | 514.000 → 2.000 |
| 128 | `[65536,2048]` | 13.940 / 3.096 | 2.569 / 1.664 | 544.000 → 32.000 |
| 2,000 | `[65536,2048]` | 46.390 / 24.610 | 34.605 / 22.844 | 1012.000 → 500.000 |

The regular FFN and vocabulary cases eliminate precisely their 64 and 512 MiB
weight promotions. The short down-projection cases retain the native split-K
partials: for example, 8 rows require 0.0625 MiB of output plus 0.5 MiB of
partials. The 2,000-row vocabulary output still consumes 500 MiB.

With **unlimited retention**, warm ordinary dispatch already reuses F32 weights.
Its isolated wall medians versus the prototype were:

| Rows | Weight `[N,K]` | Retained ordinary dispatch (ms) | Prototype (ms) |
|---:|---|---:|---:|
| 128 | `[8192,2048]` | 0.796 | 0.795 |
| 2,000 | `[8192,2048]` | 4.456 | 4.402 |
| 128 | `[65536,2048]` | 2.864 | 2.504 |
| 2,000 | `[65536,2048]` | 34.129 | 33.607 |
| 2,000 | `[2048,8192]` | 4.057 | 4.186 |

The prototype is not uniformly faster than preconverted arithmetic. For that
last down-projection workload, isolated preconverted GEMM measured 3.971 ms wall /
3.169 ms GPU versus 4.186 / 3.417 ms for the prototype, about 5% / 8% slower.
The managed fallback still benefits from removing conversion. These data support
integration experiments, not a universal throughput claim or an end-to-end
prefill speedup estimate. Existing retained conversions remain alive in the
capture owner even during prototype replay; the experiment proves the operation
does not allocate or require a promoted weight, not that current sessions have
already released their retained copies.

## Remaining integration validation

The real vocabulary replay matches captured logits and their original stage-one
fingerprints. Ordinary cached generation also retains its original tokens, but
it still uses the existing selector. Full released-checkpoint forward/logit and
generation parity with the prototype selected throughout, including controlled
sessions and speculative verification, remains a stage-three requirement.
The small-row replays validate verification-sized projections, not the speculative
driver. Likewise, the ordinary timing/peak records in the JSON are unchanged-path
baselines; repeated optimized full-prefill measurements under managed and unlimited
retention belong with that integration. Other hardware and broader layouts need
explicit validation before their native dispatch cases can be admitted.
