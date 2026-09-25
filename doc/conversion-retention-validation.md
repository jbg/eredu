# Bounded conversion retention validation

Conversion-retention validation covers policy propagation, precision, ownership,
request-state preservation and forecasts. The consumer contract is in
[generation memory](generation-memory.md). Eligible Metal single-row GEMV and
multi-row GEMM read narrow weights directly and need no retained F32 copy.
Fallback invocations exercise conversion admission and temporary casts.

## Policy and memory contract

The managed default is 256 MiB per eligible loaded execution. Explicit policies
are disabled, a bounded payload allowance, and unlimited. Admission counts
retained plus reserved conversion payload, with no automatic eviction. A denied
or oversized conversion can still require a temporary full F32 cast. The retention
limit therefore cannot cap temporary workspace, native backing capacity or RSS.

Reset preserves admitted conversions. Trim releases optional claims at a safe
settled boundary while preserving request state; repeated trim is idempotent.
Eligible bindings can admit conversions again within the same budget. Native
graphs, snapshots, other owners and allocator caching can keep backing alive after
claim release. Neither trim nor model drop promises bytes returned to the OS.

Allocator caching is independent of conversion retention. The active-allocation
high water and sampled cached bytes are separate observations. MLX exposes no
allocator-cache high-water counter through this API: `sampled_cached_peak` is the
largest token-boundary sample, not a strict peak. A nominal allocator-cache setting
is not a total-memory limit. Retained conversion payload already belongs to
resident parameters and must not be added to a forecast twice.

## Released-checkpoint validation

The checkpoint is official `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`, revision
`6767265158422fb8a19c62ceb45f16f05363615b`, file
`LFM2.5-1.2B-Instruct-BF16.gguf`, 2,343,326,528 bytes, SHA-256
`3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b`.
Runners verify the complete file hash before inference. The snapshot filename
must retain its `.gguf` extension, including when it is a symlink to a blob.
Checkpoints and full logit artifacts stay outside the tracked source tree.

The [mixed-storage validation](mixed-storage-gemm.md#validation-evidence) compares
managed and unlimited retention in the same release executable. It covers
128- and 2,000-position prefill, full-vocabulary logits, seven cached decode steps,
controlled sessions, speculative verification and loaded/continuation forecasts.
All 93 eligible projections avoid full F32 weight copies and retain zero conversion
payload. A zero retained count is valid; this checkpoint does not require a
nonzero count when the mixed-storage selector covers its invocations.

The integration runner supplies a same-build cast-plus-matmul comparison without
requiring a separate full-logit reference file. Its
[evidence manifest](../validation/results/projection-forecast-2026-09-25.json)
records commands, source/checkpoint/binary hashes, timings and allocation checks.
See [projection profiling](projection-baseline.md) for isolated conversion and
GEMM measurements. Capture and tracing use separate processes from ordinary timing.

## Policy/cache matrix harness

`validation/conversion_retention_matrix.py` crosses retention policies and
allocator-cache settings in fresh memory and parity test processes. It requires a
reference JSON containing F32 bit patterns for every vocabulary logit at eight
predictions (prefill plus seven cached decode steps), at both 128 and 2,000 prompt
positions. Comparisons use zero tolerance. The runner hashes the supplied reference
and records it in the provenance manifest; the caller is responsible for selecting
a matching checkpoint and numerical reference.

Build and run from the repository root with native device access:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --release --features mlx,metal \
  --test native_execution_control --no-run --locked --offline
python3 validation/conversion_retention_matrix.py \
  --binary target/release/deps/native_execution_control-<hash> \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --reference /path/to/lfm2-reference.json \
  --output /tmp/eredu-retention-matrix \
  --policies disabled 268435456 1073741824 5368709120 unlimited \
  --allocator-caches 0 268435456 --repetitions 3
```

Use the executable path printed by Cargo. The runner rejects a missing or failing
test and writes logs, full estimates, policy/usage/trim observations and timings.
Its manifest includes checkpoint, binary, reference, runner and working-diff hashes,
compiler, hardware, exact commands and harness environment.

The memory sequence is 128, 128, 2,000, 2,000 and 128 input positions, with eight
greedy tokens per request. Reset preserves conversions; trim follows the second
and fourth requests. Forecast checks cover fresh/warm requests and a settled
cached-decode boundary. Model drop stages potentially blocking native destructors.
The harness invokes `MlxNeuralBackend::reclaim_retired_resources()`, including
nested retirement passes, and requires active allocation to return to the pre-load
level. This does not require synchronous destruction inside `Drop` or cache flushing.

The parity process compares raw, observed and controlled inference. Controlled
execution trims at each settled prediction boundary. Prefill latency is time to
first token, excluding loading. Cached-decode throughput times four synchronized
tokens. These short windows are not sustained serving throughput or cross-device
performance guarantees. Validate nonzero retention separately with fallback
fixtures; mixed-storage coverage can legitimately leave every budget unused.

## Native behavioral coverage

Nonzero conversion fixtures cross disabled, tiny, exact-fit, partial and unlimited
budgets with allocator caching enabled and disabled. They cover F16/BF16 weights,
exact logits and greedy tokens, narrow-input bypass, aliases, independent groups,
multiple owners, final-owner release, invalidation/restoration, failed native work
and admission cancellation. Tiny budgets exercise a conversion larger than the cap.

Embedded speculative fixtures share one budget between target and prediction
owners; independent target/drafter fixtures keep separate budgets. Lookahead on/off,
accepted and rejected proposals, rollback, snapshots, forks, restore and trim
exercise state preservation. Pending transactions reject trimming. These tests
compare replayed output and preserve tokens, RNG and snapshot accounting.

```sh
cargo test -p eredu-backend-mlx --features metal --lib parameter_conversion \
  --locked -- --nocapture --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib mixed_projection \
  --locked -- --test-threads=1
cargo test -p safemlx --features metal --lib availability_query_does_not_evaluate_lazy_layout \
  --locked -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib loaded_conversion_retention_policy_reaches_native_residency \
  --locked -- --test-threads=1
cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_embedded_continuation_forecasts_preserve_settled_state_and_bound_growth \
  --locked -- --ignored --nocapture --test-threads=1
cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_speculative_continuation_forecasts_cover_settled_state_without_advancement \
  --locked -- --ignored --nocapture --test-threads=1
```

Multi-rank retention is disabled until cross-process admission is supported.
Loopback CPU Ring tests cover TP=2, PP=2 and combined TP=2 × PP=2 inference with
retention disabled, comparing prefill and cached decode against a resident
single-rank reference with absolute logit tolerance `1e-4`. They do not establish
shared retention admission. Multi-host and additional accelerator validation
remain gaps. Run the backend test binary printed by Cargo with these filters:

```sh
<backend-binary> tests::distributed_pipeline_ring::ring_two_process_qwen2_tensor_parallel_resident_reference \
  --exact --ignored --nocapture --test-threads=1
<backend-binary> tests::distributed_pipeline_ring::ring_two_process_qwen2_pipeline_resident_reference \
  --exact --ignored --nocapture --test-threads=1
<backend-binary> tests::distributed_pipeline_ring::ring_four_process_qwen2_tensor_pipeline_resident_reference \
  --exact --ignored --nocapture --test-threads=1
```

## Portable and feature boundaries

Neutral conformance covers policy propagation, independent loads, scope exclusions,
read-only queries, pending-work rejection, speculative trimming, snapshot/fork
accounting and forecast-credit removal after trim. Portable crates retain no native
dependency. The feature-boundary commands in `AGENTS.md` complement these checks:

```sh
cargo test -p eredu-core -p eredu-runtime --locked
cargo test -p eredu-architectures --lib --locked
cargo test -p eredu --no-default-features --locked
cargo check -p eredu-backend-mlx --no-default-features
cargo clippy -p eredu-core -p eredu-runtime -p eredu-architectures \
  -p eredu-backend-mlx -p eredu --no-default-features --lib --locked -- -D warnings
cargo clippy -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control --locked -- -D warnings
cargo clippy -p eredu-backend-mlx --features metal --lib --tests --locked -- -D warnings
```
