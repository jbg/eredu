# Mixed-storage FP32 GEMM

The MLX backend selects mixed-storage FP32 GEMM through the shared projection
mechanism for ordinary linear layers and tied-embedding readout. Ordinary,
controlled and speculative execution use the same selector. Loaded forecasts
consume proven projection coverage and native workspace facts; cold inspection
retains conservative conversion allowances where final layout or native dispatch
is unknown.

## Arithmetic and storage contract

The operation computes `F32[M,K] @ narrow[N,K].T -> F32[M,N]`. It accepts F16 or
BF16 weight storage. The native patch extends the existing `BlockLoader` with a
separate storage type. Narrow weights widen as they enter the same FP32 shared
memory tile; activations are never narrowed. The homogeneous loader keeps its
vector-copy implementation.

The distinct inference primitive delegates to the pinned native Matmul dispatch.
Regular GEMM and split-K keep native tile geometry, shared-memory padding, loop
order (including the different regular/split-K tail handling), MMA, accumulator,
partition count/size and final reduction. Bias remains a separate ordinary add;
there is no fused epilogue substitution. Kernel cache keys include the weight
storage dtype; homogeneous kernels use the ordinary native keys.

The native operation admits settled row-contiguous rank-two weights and
rank-two-or-higher F32 activations with
`2 <= M <= 2000` (total logical input rows),
`2 <= N <= 65536`, `1 <= K <= 8192`, and native SIMD GEMM dispatch.
Unsupported inputs return `None` without evaluation.
There is no GPU model-name allowlist. MLX retains its device-specific tile and
split-K selection. Contiguous views with offsets are supported.

NAX/TF32 dispatch is excluded because its separate weight loader does not support
narrow storage; passing narrow weights into that path would read them as FP32. The
operation and ordinary Matmul share the same native NAX-selection predicate.
NAX-capable hardware is eligible when native FP32 dispatch uses SIMD, for example
with `MLX_ENABLE_TF32=0` set before native initialization. The operation never
changes that setting or substitutes a different arithmetic path. Padded/column-major
weights, lazy weights, single-row GEMV, CPU, CUDA, system MLX and non-JIT builds
remain excluded from this GEMM operation. The backend retains its existing
single-row mixed GEMV selector and all other native fallbacks. These are
implementation coverage limits, not model limitations. Native autodiff, vmap and
compilation transformations are unsupported
by the distinct primitive.

The operation creates the F32 output and, when selected by native dispatch,
F32 split-K partials. It creates no full promoted weight. Activations may be lazy,
transposed, padded or sliced. Rank normalization uses exactly the native public
matmul's flatten/unflatten graph operations; native Matmul resolves activation
strides and performs any necessary FP32 activation copy at evaluation. Selection
never forces evaluation. Bias uses a separate ordinary add.

Unsupported cases use each caller's parameter-conversion retention and native
matmul path. Narrow activations use their ordinary dispatch; the selector does
not promote a noncontiguous/lazy weight to obtain eligibility. Linear and tied
readout call the same backend selector.

## Forecast contract

Loaded residency reports expose current immutable parameter binding facts: exact
matrix dimensions, narrow weight dtype, F32 activation/output dtype, covered row
intervals, native split-K partial bytes per row and a possible activation-copy
bound. The shape/device query uses the same split-K partition helper as native
GEMM; dispatch and the query therefore use identical partition decisions. It
creates no arrays and does not evaluate lazy weights to establish eligibility.
The backend also provides `describe_bound_projection` for an individual logical
invocation. Unbound descriptions retain promotion and opaque-workspace costs.

Runtime requires a separate activation dtype proof. LFM2 declares that its dense
mixer, FFN and tied-readout projection inputs preserve the dtype from their
ungrouped learned RMS normalizations. MLX reports which actual gain bindings
produce F32 output. This distinction matters here: the pinned checkpoint's state
storage is two bytes, but its F32 gains produce F32 projection inputs. Unknown
flows and narrow gains get no conversion credit. This declaration lives with the
architecture equations, not in a backend family-specific branch.

A full-weight conversion allowance is removed only for an exactly attributed
canonical parameter whose **every invocation** is covered. Retained conversion
credits are not subtracted twice. Regular output and separate bias allocations
stay in their existing owners. Native split-K partials and possible input-layout
copies are additional workspace, summed conservatively across invocations with
the concurrent-layer overlap calibration retained. A forecast for up to M rows
includes partial-buffer maxima at smaller row counts:
split-K storage is not monotone in M, and short verification/final chunks matter.
The report distinguishes actual-M partial payload from this prefix envelope.
Allocator rounding and GPU-private storage remain unknown; existing calibrated
allocator, attention, state and graph-retention envelopes remain in place.

Cold inspection, absent/lazy/unsupported bindings, conflicting aliases, active
parameter overlays, distributed bindings and bounded residency retain their
conservative allowance. Other architectures retain it until they supply a dtype-flow
declaration. Device eligibility follows the shared native selector; there is no
GPU model-name restriction. An unsupported 2,001-row invocation retains its
conversion allowance even when a one-row continuation is covered. The retention
subledger still describes potential admissions across future invocations (which
can exceed the current request's admitted rows); it is not an additional charge
on top of the request workspace.

## Validation and reproduction

The checkpoint revision, hash and projection inventory are in
[projection profiling](projection-baseline.md). Checkpoint sources stay outside
the tracked tree. Run native tests serially because allocator counters are global:

```sh
MLX_ENABLE_TF32=0 CARGO_INCREMENTAL=0 cargo test -p safemlx --release \
  --test mixed_storage_gemm --locked --offline \
  -- --include-ignored --nocapture --test-threads=1
CARGO_INCREMENTAL=0 cargo build -p eredu --release \
  --example projection_forecast --example projection_baseline \
  --example projection_parity --no-default-features \
  --features mlx,metal,projection-profiling --locked --offline
python3 validation/projection_integration.py \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --output /tmp/eredu-projection-validation --samples 5 --disable-tf32 --forecasts
```

`--disable-tf32` explicitly selects native SIMD FP32 dispatch on NAX-capable
hardware. The runner otherwise sanitizes MLX configuration overrides. The
optional `--reference-manifest` checks recorded fingerprints in addition to the
mandatory same-build comparisons. Use a manifest from the same device: native
reduction geometry can differ between GPUs.

The nondefault profiler's thread-affine `CastGemmReference` guard disables only
multi-row mixed GEMM, providing the cast-plus-matmul comparison in the same
executable. It is absent from production inference options. Ordinary timing,
projection capture and kernel tracing use separate processes. Capture retains
arrays and therefore cannot supply ordinary inference peaks or latency.

The native suite covers 290 F16/BF16 exact-bit, separate-bias and allocation cases,
including tile boundaries, tails, split-K partitions and all six dominant weight
shapes. Layout coverage includes lazy, strided and batched activations, contiguous
weight views with offsets, and rejection of unsupported operands without
execution. CPU rejection requires no GPU. Explicitly selected Metal tests fail
when SIMD GEMM is unavailable; other Metal hardware needs its own native run.

Every eligible real-array replay must match the reference output bits and native
pipeline selection after removing only the weight-storage suffix. The split-K
accumulator remains part of that comparison. Incremental active allocation must
match preconverted GEMM's output/partial workspace; the separately resident F32
reference weight is not free storage. Full-checkpoint checks cover full-vocabulary
logits, eight predictions (prefill plus seven cached steps), controlled sessions,
and two-proposal speculative verification.

Forecast checks cover cold, loaded, ordinary/controlled continuation and external
speculative target/drafter requests. They require unchanged native allocations,
conversion admission and token advancement during observation; bounds covering
measured prefill growth; and conservative prefill costs beyond 2,000-row coverage.

## Validation evidence

The [evidence manifest](../validation/results/projection-forecast-2026-09-25.json)
contains 24 successful serial release processes and 72 real projection replays on
Apple M3 Ultra with 256 GiB unified memory, macOS 26.6.2 and Rust 1.98.0. It records
exact commands, checkpoint/source/binary hashes, native pipeline selections,
forecasts, first-request samples and five warm measurements per timing case.
Raw results are in `/tmp/eredu-projection-forecast-2026-09-25`.

All 93 bound projections have F32 input proofs and covered storage facts. They
avoid 4,464 MiB of full-weight promotion; other conversion and state allowances
remain. Replays match preconverted native workspace and recorded fingerprints.
The evidence includes ordinary, controlled and speculative parity and the
unsupported 2,001-row forecast check for both retention policies.

For the pinned checkpoint, 8 output tokens, one full prefill chunk and allocator
cache disabled, forecast values are **GiB**, not measured peaks:

| Positions | Upper with binding facts omitted | Upper with binding facts | Measured prefill growth | Additional-generation upper with facts |
|---:|---:|---:|---:|---:|
| 128 | 8.494 | 3.317 | 0.469 | 1.137 |
| 2,000 | 15.295 | 12.779 | 1.301 | 10.599 |

Both retention policies produce these forecast bounds before any conversions are
retained. Cold overall upper bounds are **8.494 / 15.295 GiB** respectively.
The remaining gap is explicit: future activation strides are unknown, so the
forecast retains possible F32 input copies, smaller-row split-K maxima, concurrent
workspace calibration, attention/state and other native overhead. It does not
replace those unknowns with measured timings or peaks. For example, a `[2048,8192]`
weight at 8 rows needs 0.5 MiB of split-K partials; at 2,000 rows it needs none, but
the up-to-2,000-row envelope retains the 4 MiB maximum at 256 rows.

Ordinary prefill measurements compare two modes in the same executable: the
cast-plus-matmul reference and the mixed-storage selector. Latency is median
warm first-token time over five samples; peak is maximum warm MLX active allocation.
Loading is excluded, and capture/tracing run in separate processes:

| Retention | Positions | Reference / mixed latency (ms) | Reference / mixed active peak (GiB) |
|---|---:|---:|---:|
| 256 MiB | 128 | 94.04 / 22.60 | 6.989 / 2.649 |
| 256 MiB | 2,000 | 720.48 / 687.96 | 4.683 / 3.481 |
| Unlimited | 128 | 23.70 / 22.66 | 7.009 / 2.649 |
| Unlimited | 2,000 | 692.84 / 688.46 | 7.840 / 3.481 |

These measurements describe this checkpoint and device, not universal throughput
or peak-memory guarantees. Hardware validation is limited to M3 Ultra; eligibility
uses native SIMD dispatch rather than a GPU model-name allowlist. Broader weight
layouts and NAX narrow loading require implementation and numerical validation.
