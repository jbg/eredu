# Mixed-storage FP32 GEMM

Stage two added a loader-only native operation for the pinned MLX 0.32.0 FP32
GEMM. Stage three selects it through the MLX backend's shared projection selector
for both ordinary linear layers and tied-embedding readout. Ordinary, controlled
and speculative execution consume those same mechanisms. Forecast mechanism facts
remain stage four; cold and runtime conversion allowances are still conservative.

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
settled row-contiguous rank-two weights and rank-two-or-higher F32 activations,
`2 <= M <= 2000` (total logical input rows),
`2 <= N <= 65536`, `1 <= K <= 8192`, and native SIMD GEMM dispatch are admitted.
There is no GPU model-name allowlist. MLX retains its device-specific tile and
split-K selection. Contiguous views with offsets are supported.

NAX/TF32 dispatch is excluded because its separate weight loader has not been
extended; passing narrow weights into that path would read them as FP32. The
operation and ordinary Matmul share the same native NAX-selection predicate.
NAX-capable hardware is eligible when native FP32 dispatch uses SIMD, for example
with `MLX_ENABLE_TF32=0` set before native initialization. The operation never
changes that setting or substitutes a different arithmetic path. Padded/column-major
weights, lazy weights, single-row GEMV, CPU, CUDA, system MLX and non-JIT builds
remain excluded from this GEMM operation. The backend retains its existing
single-row mixed GEMV selector and all other native fallbacks. These are implementation coverage limits, not model limitations. Native autodiff, vmap and compilation transformations are unsupported
by the distinct primitive.

The operation creates the F32 output and, when selected by native dispatch,
F32 split-K partials. It creates no full promoted weight. Activations may be lazy,
transposed, padded or sliced. Rank normalization uses exactly the native public
matmul's flatten/unflatten graph operations; native Matmul resolves activation
strides and performs any necessary FP32 activation copy at evaluation. Selection
never forces evaluation. The separate bias add is unchanged.

Unsupported cases flow through each caller's existing parameter-conversion
retention and native matmul path. In particular, this change does not alter
narrow-activation dispatch or promote a noncontiguous/lazy weight to obtain
eligibility. Linear and tied readout call the same backend selector.

## Reproduction

Use the same official BF16 GGUF revision and SHA-256 as the
[stage-one baseline](projection-baseline.md). Checkpoint sources remain outside
the tracked tree. Run native tests serially because allocator counters are global:

```sh
MLX_ENABLE_TF32=0 CARGO_INCREMENTAL=0 cargo test -p safemlx --release \
  --test mixed_storage_gemm --locked --offline \
  -- --include-ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 cargo build -p eredu --release \
  --example projection_baseline --no-default-features \
  --features mlx,metal,projection-profiling --locked --offline
python3 validation/projection_baseline.py \
  --binary target/release/examples/projection_baseline \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --output /tmp/eredu-mixed-storage-gemm --samples 5 \
  --mixed-storage-prototype --reference-dispatch \
  --reference-manifest validation/results/projection-baseline-2026-09-25.json
```

On NAX-capable hardware, add `--disable-tf32` to the Python runner to select
native SIMD FP32 dispatch explicitly. The runner otherwise sanitizes MLX
configuration overrides for reproducibility. Omit `--reference-manifest` when
running on a different device: exactness is against that device's native
cast-plus-matmul path, not another device's reduction or historical fingerprints.

The hardware-specific tests are ignored by default and fail if explicitly run
without Metal SIMD GEMM. They measure output/partial workspace against native
preconverted GEMM on the device under test, without hard-coded split-K thresholds
or host page sizes. The CPU rejection test runs normally and requires no GPU. The runner uses fresh processes for ordinary timing, real projection replay,
and kernel tracing, at 128 and 2,000 positions with disabled, 256 MiB and unlimited
conversion retention. Timing runs disable kernel tracing and allocator caching.
Short rows 2, 4, 8 and 16 slice actual 128-position activations. For the prototype,
the profiler settles a contiguous `[M,K]` view before timing and compares all
flattened output bits. The integrated selector also restores the original leading dimensions lazily.

Each eligible replay adds `mixed_storage_prototype` to the four baseline phases.
It must match every output bit of the captured projection (or the explicit-F32
reference for sliced inputs). Native selection traces must match preconverted
GEMM after removing only the weight-storage suffix; the split-K accumulator must
remain present and unchanged. Measured peak growth must equal preconverted GEMM's
output/partial workspace. Unsupported replays fail rather than benchmarking a
fallback under the prototype label. With `--reference-dispatch`, the reference manifest additionally gates all
projection fingerprints, class inventories and ordinary cached-generation tokens
against the original pre-patch stage-one evidence.

The preconverted comparison includes a separately reported resident F32 weight;
its incremental peak is not total storage. Replay timings are isolated operations,
not additive full-model attribution. Capturing retains array owners and invalidates
ordinary inference peak/latency measurement, as in stage one.

## Stage-two results, 2026-09-25

The original stage-two measurements below used the initial model-name gate,
which has since been replaced by mechanism-based eligibility. Their binary and
patch hashes identify that historical measurement, not the later correction.

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

## Scope of stage-two evidence

The stage-two records above exercised the new operation only in isolated replay;
ordinary timing/peak records still used the previous selector. Those measurements
established loader arithmetic and allocation parity. Full integrated forward,
controlled/speculative and ordinary timing validation is recorded in the
stage-three section below. Broader weight layouts and the NAX loader still need
implementation and validation before admission.

## Device eligibility correction

The original model-name gate was unnecessarily restrictive and has been removed.
Admission now shares ordinary Matmul's actual SIMD/NAX selection predicate.
The loader contains no GPU-model specialization; native tile geometry and split-K
partitioning remain selected by MLX on the device in use. The separate NAX/TF32
loader remains an implementation gap, so that selection returns `None` rather
than using incorrect storage interpretation or changing arithmetic implicitly.

After this correction the 290-case F16/BF16 bit/bias/allocation matrix and layout
checks passed again on M3 Ultra with `MLX_ENABLE_TF32=0`. Allocation tests now
compare with measured preconverted-GEMM workspace on the same device rather than
reimplementing its split-K thresholds and allocator page rounding. CPU-only
rejection, strict safemlx Clippy, the Python evidence tests and formatting checks
also passed. No physical validation on another GPU is claimed.

## Stage-three integration validation

Build the two release diagnostic examples and run the integration matrix:

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --release \
  --example projection_baseline --example projection_parity \
  --no-default-features --features mlx,metal,projection-profiling --locked --offline
python3 validation/projection_integration.py \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --output /tmp/eredu-projection-integration --samples 5 --disable-tf32
```

The nondefault `projection-profiling` feature exposes `CastGemmReference`, a
thread-affine, nestable guard that disables only multi-row mixed GEMM while alive.
It is a validation control, absent from production builds and inference options.
Both comparison paths use the same executable, checkpoint and native settings;
single-row mixed GEMV remains enabled. No execution driver gains another selector.

The parity executable compares full 65,536-entry F32 logit bit arrays, not just
argmax or tolerances, through prefill and seven cached predictions. Ordinary and
controlled paths are compared with each other and with the cast-GEMM reference.
Self-drafting with two proposals exercises actual speculative verification;
ordinary and controlled speculative tokens match ordinary generation, and every
target/draft capture matches the same speculative path with cast GEMM. Different
GEMV/GEMM batch reductions need not match each other; the comparison preserves each
path's native arithmetic. Captured projection classes must prove that short
verification batches selected mixed GEMM.

The runner uses fresh serial processes for parity, ordinary timings, real
projection replay and kernel tracing. It tests 128 and 2,000 positions with
256 MiB managed and unlimited conversion retention. Each ordinary run includes a
first request and five warm reset requests, with allocator caching disabled.
Every integrated replay must preserve native kernel geometry/reduction and have
the same peak allocation as preconverted GEMM, including split-K partials.
Capture and trace runs are excluded from ordinary timing/memory results.

Stage-three results on Apple M3 Ultra / macOS 26.6.2, release build, pinned
checkpoint, native SIMD dispatch (`MLX_ENABLE_TF32=0`), 2026-09-25:

| Conversion retention | Positions | Warm first-token ms, reference → integrated | Peak active GiB, reference → integrated |
|---|---:|---:|---:|
| Managed, 256 MiB | 128 | 93.57 → 22.64 | 6.989 → 2.649 |
| Managed, 256 MiB | 2,000 | 715.70 → 685.55 | 4.683 → 3.481 |
| Unlimited | 128 | 23.60 → 22.67 | 7.009 → 2.649 |
| Unlimited | 2,000 | 693.65 → 686.62 | 7.840 → 3.481 |

Times are medians of five warm reset requests, through the first generated token;
loading is excluded. Peaks are total MLX active allocation during prefill, not
process RSS or a cold-lifecycle forecast. Small latency differences are descriptive
measurements on this host, not a cross-device throughput guarantee. First-request
latency and peaks are recorded separately in the evidence.

[Stage-three machine-readable evidence](../validation/results/projection-integration-2026-09-25.json)
records 20 isolated processes, exact checkpoint/binary/source hashes, all samples,
logit fingerprints and native pipeline selections. Raw output is retained in
`/tmp/eredu-projection-integration-2026-09-25`.

- All four policy/length pairs passed exact 65,536-entry logits for eight
  predictions across reference/integrated and ordinary/controlled execution.
  Ordinary and controlled speculative generation produced the same eight tokens.
  Each reference/integrated speculative pair matched all 13 target/draft logit
  captures across three actual verification blocks; short verification projections
  selected mixed GEMM.
- All 93 prefill projections (six weight-shape classes) selected mixed GEMM in
  every case. Conversion-retention reports remained zero for every integrated
  cold and warm request, including unlimited policy.
- All **72 real projection replays** at 2/4/8/16/128/2,000 rows passed exact bits.
  Integrated pipeline selection matched preconverted native GEMM after removing
  only the storage suffix, with no conversion pipeline. Every measured peak
  equaled native output/partial workspace, including split-K.
- Native tests passed all 290 existing F16/BF16 arithmetic/bias/allocation cases
  plus 64 transposed/padded/sliced/batched activation cases. Lazy selection stayed
  unevaluated; unsupported weights, devices and shapes retained fallback behavior.
  Four shared-selector tests passed, including linear bias, tied readout and
  retained-conversion admission beyond the 2,000-row bound.
- Portable backend conformance (112 tests), portable facade (27 passed, one
  existing ignored), non-Metal backend build, strict release Clippy for both
  examples, Python evidence-validator tests, formatting and diff checks passed.

Only M3 Ultra hardware was available for native validation. Eligibility is based
on native arithmetic dispatch, not device name; the exactness/allocation suite is
portable to other supported SIMD Metal devices, but those runs are still a
validation gap. NAX/TF32 retains its existing fallback. Stage four must expose
per-invocation coverage/workspace before forecasts remove conversion allowances.
