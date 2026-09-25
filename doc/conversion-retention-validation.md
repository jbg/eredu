# Bounded conversion retention validation

Phase 7 of the bounded parameter-conversion retention plan validates policy,
precision, request-state preservation and the throughput/memory tradeoff. The
consumer contract and earlier calibration are in [generation-memory.md](generation-memory.md).
The historical retention cost below is addressed for eligible Metal single-row
decode by mixed-dtype kernels; see the before/after release measurements below.
Multi-row prefill keeps its existing native matrix path.

## Reproduction and provenance

The native matrix uses the official mixed-width
`LiquidAI/LFM2.5-1.2B-Instruct-GGUF` checkpoint at revision
`6767265158422fb8a19c62ceb45f16f05363615b`, file
`LFM2.5-1.2B-Instruct-BF16.gguf`, 2,343,326,528 bytes, SHA-256
`3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b`.
The runner verifies this hash before inference. The Hugging Face snapshot
filename must retain its `.gguf` extension even when it is a symlink to a blob.

The exact-logit reference is the retained pre-conversion-reuse artifact from
revision `a43a3171`, described in the earlier resident-conversion calibration.
Its SHA-256 is
`f4625d733ac518405c7dfc95de9753f8c5b5c1c3deba299bc5caa619709ba6df`.
It contains F32 bit patterns for all 65,536 vocabulary logits at eight predictions
(prefill plus seven cached decode steps), for 128 and 2,000 prompt positions.
This run reuses that artifact; it does not rebuild the historical revision.
To recreate it, use the earlier `native_lfm2_forecast_recalibration_preserves_logits_and_cached_generation`
harness at that revision with `EREDU_LFM2_PARITY_WRITE=/tmp/lfm2-reference.json`,
the pinned GGUF and `EREDU_LFM2_MEMORY_LENGTHS=128,2000`.
Regression comparisons use **zero tolerance**, preserving BF16 rounding exactly.

Build and run from the repository root:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --features mlx,metal \
  --test native_execution_control --no-run --locked --offline \
  --config profile.test.package.eredu-backend-mlx.debug=0 \
  --config profile.test.package.eredu.debug=0
python3 validation/conversion_retention_matrix.py \
  --binary target/debug/deps/native_execution_control-<hash> \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --reference /tmp/lfm2-reference.json \
  --output /tmp/eredu-retention-matrix
```

Use the executable path printed by Cargo. Native device access is required;
sandboxed Metal initialization failed on the validation host. Each policy/cache
pair gets fresh memory and parity test processes. The runner rejects a missing
test as well as a failing test, and writes full logs, full memory estimates,
policy/usage/trim observations, timings and a provenance manifest. The manifest
records checkpoint, binary, reference, runner and working-diff hashes, compiler,
hardware, exact commands and harness environment. Large checkpoints and logit
artifacts remain outside the tracked source tree.

The memory sequence is 128, 128, 2,000, 2,000 and 128 input positions, with eight
greedy tokens per request. Reset preserves conversions. Explicit trim follows
the second and fourth requests; the next request must admit conversions again
within the same scope. Forecasts are checked before fresh/warm requests and at a
settled cached-decode boundary. Model drop stages potentially blocking native destructors. The harness records
immediate post-drop usage, invokes the existing public ordinary-host
`MlxNeuralBackend::reclaim_retired_resources()` boundary (including nested
retirement passes), and requires allocator active bytes to return to the pre-load
level. This does not promise synchronous destruction inside `Drop`, cache
flushing, or bytes returned to the OS. The parity process compares raw tokens with observed
and controlled inference; controlled execution additionally trims at every
settled prediction boundary before advancing.

Prefill latency means time to first token, excluding checkpoint load. Cached
decode throughput uses the last four tokens and synchronization. Timings are
single cold/warm samples per prompt length and policy in a debug test build;
they establish a local tradeoff, not a statistical or cross-device benchmark.
MLX reports active allocation high water, but exposes no allocator-cache high
water through this API. `sampled_cached_peak` is therefore the largest sample at
token boundaries, not a claim about the true cache peak. Neither number bounds
process RSS or driver memory. The forecast includes its calibrated allocator and
graph allowance; retained-conversion payload is already part of resident weights.

## Checkpoint matrix results (2026-09-24)

All ten policy/cache pairs passed on an Apple M3 Ultra with 256 GiB unified
memory, macOS 26.6.2, using Rust 1.98.0 and MLX Metal. The harness was applied to
base revision `3d47816da1e53d82f4df8a1503bdebf3bfa54b64`; the measured executable
SHA-256 is `bef5f23848867a41f6154472ce57cb67ce9a15b5e74e6502c8deec20fe6638c5`.
The [compact machine-readable evidence](../validation/results/conversion-retention-2026-09-24.json)
retains every request's timings, tokens, retained/reserved payload, trim release,
measured peaks, forecast bounds and post-drop observations, plus provenance.
Full raw logs and estimates were written to `/tmp/eredu-phase7-retention-matrix`.

Fifty memory requests passed absolute active-peak, incremental startup and settled
continuation bounds. All ten policies/cache combinations matched the historical
reference exactly at both prompt lengths: 1,048,576 logit bit patterns per pair,
plus identical greedy tokens and ordinary/controlled outputs. Controlled trimming
at every prediction preserved those results. Bounded policies retained exactly
32, 256 and 1,024 MiB, disabled retained zero, and unlimited retained
4,680,843,264 bytes (4,464 MiB). All settled reservation counters were zero.
Repeated reset kept those payloads; trims released the full current claim payload,
were idempotent, and subsequent inference repopulated within the same cap.

In the timing table, `empty/warm` means an empty conversion cache followed by a
reset preserving conversions. The 2,000-position empty request follows explicit
trim; the last column repeats 128 positions after the second trim. Decode columns
show 128 / 2,000 positions. Cache and retention settings are MiB, latencies are ms.

| Allocator setting | Retention setting | 128 prefill empty/warm | 2,000 prefill empty/warm | Cached decode tokens/s | 128 prefill after trim |
|---:|---|---:|---:|---:|---:|
| 0 | disabled | 133.2 / 103.6 | 935.9 / 934.9 | 10.97 / 10.51 | 103.9 |
| 0 | 32 | 134.3 / 104.0 | 951.6 / 944.3 | 11.01 / 10.52 | 105.0 |
| 0 | 256 | 136.0 / 98.5 | 955.4 / 941.8 | 11.52 / 10.99 | 106.6 |
| 0 | 1024 | 143.4 / 85.0 | 971.4 / 936.9 | 13.52 / 12.81 | 113.6 |
| 0 | unlimited | 172.4 / 29.8 | 1027.3 / 913.0 | 73.89 / 51.17 | 154.7 |
| 256 | disabled | 129.9 / 98.2 | 901.5 / 903.5 | 11.32 / 10.91 | 98.2 |
| 256 | 32 | 132.5 / 97.4 | 903.2 / 901.9 | 11.39 / 10.88 | 100.3 |
| 256 | 256 | 134.4 / 94.0 | 908.5 / 900.4 | 11.92 / 11.47 | 101.9 |
| 256 | 1024 | 140.7 / 80.4 | 924.0 / 891.0 | 13.70 / 13.41 | 108.2 |
| 256 | unlimited | 169.7 / 30.4 | 982.6 / 866.8 | 75.20 / 52.24 | 155.3 |

Memory values below are MiB; paired values are for warm 128 / 2,000-position
requests. The modeled column is the **total generation peak upper bound**, which
already includes resident source weights and retained conversions. Cache samples
are the maximum across all five raw requests, while trim latencies follow the
warm requests. All 50 measured active peaks and continuation growths stayed within
their applicable bounds, including the requests immediately after trimming.

| Allocator setting | Retention setting | Retained payload | Active peak | Modeled total upper | Sampled cache maximum | Trim ms |
|---:|---|---:|---:|---:|---:|---:|
| 0 | disabled | 0 | 7156.7 / 7140.9 | 8698.2 / 20584.6 | 0.0 | 0.13 / 0.09 |
| 0 | 32 | 32 | 7156.7 / 7140.9 | 8690.2 / 20576.6 | 0.0 | 0.12 / 0.10 |
| 0 | 256 | 256 | 7156.7 / 7140.9 | 8634.2 / 20520.6 | 0.0 | 0.12 / 0.11 |
| 0 | 1024 | 1024 | 7156.7 / 7140.9 | 8442.2 / 20328.6 | 0.0 | 0.85 / 0.82 |
| 0 | unlimited | 4464 | 7176.7 / 8028.6 | 7582.2 / 19468.6 | 0.0 | 6.89 / 6.90 |
| 256 | disabled | 0 | 7156.9 / 7140.9 | 8954.2 / 20841.6 | 257.0 | 0.09 / 0.08 |
| 256 | 32 | 32 | 7156.9 / 7140.9 | 8946.2 / 20833.6 | 269.0 | 0.11 / 0.11 |
| 256 | 256 | 256 | 7156.9 / 7140.9 | 8890.2 / 20777.6 | 301.0 | 0.12 / 0.13 |
| 256 | 1024 | 1024 | 7156.9 / 7140.9 | 8698.2 / 20585.6 | 301.0 | 0.20 / 0.21 |
| 256 | unlimited | 4464 | 7176.8 / 8013.0 | 7838.4 / 19728.4 | 303.8 | 8.16 / 6.56 |

The finite policies substantially reduce idle resident conversion storage, but
these runs have essentially equal generation active peaks: rejected conversions
still need temporary F32 casts. A payload cap must not clamp that workspace in a
forecast. The largest output conversion is 512 MiB, so the 32 and 256 MiB policies
also exercise a real individual conversion larger than their entire allowance.

Allocator caching is independent. With allocator caching disabled, every sampled
cache value and final cached value was zero even with 4,464 MiB of live retained
conversions. With a nominal 256 MiB allocator setting, generation samples reached
303.8 MiB. After dropping the unlimited model and reclaiming its staged native
owners, active bytes were zero but allocator cached bytes were 800,357,802
(763.3 MiB). MLX's allocator setting is a retention policy, not a strict total
allocation/cache guarantee. Neither trim nor model drop flushes that separate
cache or promises OS reclamation. The harness records these observations instead
of treating the conversion payload bound as a native backing or process limit.

The **256 MiB managed default stays finite and unchanged**, with a substantial
throughput tradeoff made explicit. With allocator caching disabled, warm
128-position decode was 11.52 tokens/s versus 73.89 with unlimited retention:
about **84% lower throughput**, or 6.4 times as long per cached token, in exchange
for 4,208 MiB less idle conversion payload. At 2,000 positions the corresponding
rates were 10.99 versus 51.17 tokens/s. A 1 GiB allowance improved short-context
decode by about 17% over 256 MiB while retaining another 768 MiB. The allocator
256 MiB rows show the same tradeoff. This single-model debug calibration does not
justify raising the managed allowance for every loaded execution; callers can
explicitly select a larger finite budget or unlimited retention when that memory
tradeoff is appropriate.

Explicit trimming itself cost approximately 0.12 ms at the 256 MiB default and
6.89 ms with unlimited retention in the zero-allocator-cache warm 128 case.
Readmission also matters: the next 128-position prefill took 106.6 ms rather than
98.5 ms at 256 MiB, and 154.7 ms rather than 29.8 ms with unlimited retention.
These are individual local measurements; repeated statistical throughput runs,
other model shapes and other accelerators remain outside this calibration.

## Repeated release baseline before mixed-dtype kernels (2026-09-24)

A sequential release-build rerun confirms that the retention bottleneck persists
without debug-build overhead. There were no other agent builds or benchmarks
running during measurement. The host, pinned checkpoint, exact historical logit
reference, request sequence and allocator telemetry semantics are unchanged.
The baseline code revision is `6451728020f704c12617a9144fe8dd28e4b940f9`;
the only working-tree change before measurement extended the runner with repeated
processes and explicit build-command provenance. The release executable SHA-256
is `4c725b0214200511a57f98291953c22067a513aee979afa9efdd13387bbba41e`.

The [release baseline evidence](../validation/results/mixed-dtype-matrix-before-2026-09-24.json)
contains all 75 request measurements, 15 fresh-process invocations, five exact
parity checks and provenance. Full logs and estimates are in
`/tmp/eredu-mixed-dtype-before`. Policies cycle before the next repetition; each
process performs the existing cold/warm/trim sequence. Parity runs once per
policy; all five passed the 1,048,576 historical logit-bit comparisons plus
ordinary/controlled token and settled-trim checks. All memory bounds, retention
caps and post-drop reclamation assertions passed in all three repetitions.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --release --features mlx,metal \
  --test native_execution_control --no-run --locked --offline
python3 validation/conversion_retention_matrix.py \
  --binary target/release/deps/native_execution_control-<hash> \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct-BF16.gguf \
  --reference /tmp/eredu-cast-reuse-gguf-reference.json \
  --output /tmp/eredu-mixed-dtype-before \
  --policies disabled 268435456 1073741824 5368709120 unlimited \
  --allocator-caches 0 --repetitions 3 \
  --build-command 'CARGO_INCREMENTAL=0 cargo test -p eredu --release --features mlx,metal --test native_execution_control --no-run --locked --offline'
```

The table reports medians across three fresh processes. Parenthesized decode
values are the observed minimum–maximum, not confidence intervals. Each warm
request still times four synchronized cached tokens, so this remains a short,
single-checkpoint calibration rather than sustained serving throughput.

| Retention | Warm 128 decode tokens/s | Warm 2,000 decode tokens/s | Warm prefill ms, 128 / 2,000 | Active peak MiB, 128 / 2,000 |
|---|---:|---:|---:|---:|
| Disabled | 11.60 (11.55–11.61) | 11.01 (11.00–11.10) | 97.81 / 707.95 | 7156.7 / 7140.9 |
| 256 MiB | 12.18 (12.17–12.27) | 11.58 (11.54–11.61) | 92.51 / 711.06 | 7156.7 / 7140.9 |
| 1 GiB | 14.58 (14.56–14.72) | 13.80 (13.73–13.86) | 78.34 / 713.80 | 7156.7 / 7140.9 |
| 5 GiB | 106.99 (106.90–107.22) | 68.09 (68.06–68.18) | 23.63 / 690.70 | 7176.7 / 8028.6 |
| Unlimited | 106.72 (106.47–107.20) | 68.15 (66.99–68.31) | 23.72 / 692.21 | 7176.7 / 8028.6 |

The 5 GiB allowance admits the same 4,464 MiB payload as unlimited and has
comparable throughput. The 256 MiB policy is 88.6% slower than 5 GiB at 128
positions (about 8.8 times as long per token). The 1 GiB allowance improves this
rate by only 19.7%. None of the smaller policies materially reduces the observed
short-context active peak. These findings support removing full-weight
promotion from the decode computation rather than merely changing admission
order. They do not isolate cast time from allocation, graph construction and
matrix execution: projection-level profiling is still required to attribute
costs precisely. Before optimization, mixed F32-activation/narrow-weight dispatch fell
through the BF16-only row kernel to retained promotion or native mixed-dtype
matmul; the reusable mixed-dtype projection below targets that path while
preserving its F32 output and numerical contract.

## Mixed-dtype decode kernel results (2026-09-24)

The same release matrix was repeated after introducing the reusable Metal
F32-activation/F16-or-BF16-weight GEMV kernel. It widens weights during the dot
product and preserves the native F32 reduction partition/order; it does not
materialize a full F32 weight copy. Eligible single-row dense and tied-embedding
projections use it regardless of retention policy. Multi-row prefill and
speculative verification, unevaluated weights, column-major weights, CPU and CUDA
keep the existing native fallback. No precision policy or managed retention
allowance changed. The implementation and these fallback boundaries are
recorded in [backend-architecture.md](backend-architecture.md).

The [after evidence](../validation/results/mixed-dtype-matrix-after-2026-09-24.json)
contains the same 75 requests and five parity cases, plus hashes for the new
source files. The executable SHA-256 is
`ff8151be35a2f4f60c3cc380633d8edd42c4d6928022c6fd670e8c29a10000b5`.
Reproduce with the preceding release build/run commands, changing the output
directory to `/tmp/eredu-mixed-dtype-after`. Measurements ran sequentially with
all other agents and builds idle. All 15 cases passed retention caps, memory
bounds, settled reservations and reclamation. Every policy still matched the
unchanged historical reference **bit for bit**, including ordinary versus
controlled generation with trimming at every prediction boundary.

Warm cached-decode medians across three fresh processes:

| Retention | 128 tokens/s, before → after | Speedup | 2,000 tokens/s, before → after | Speedup |
|---|---:|---:|---:|---:|
| Disabled | 11.60 → 139.07 | 11.99× | 11.01 → 80.57 | 7.32× |
| 256 MiB | 12.18 → 139.35 | 11.44× | 11.58 → 80.22 | 6.93× |
| 1 GiB | 14.58 → 133.93 | 9.19× | 13.80 → 79.75 | 5.78× |
| 5 GiB | 106.99 → 147.23 | 1.38× | 68.09 → 80.60 | 1.18× |
| Unlimited | 106.72 → 145.85 | 1.37× | 68.15 → 80.32 | 1.18× |

For the default allowance, the three after samples ranged from 138.30–140.18
at 128 positions and 80.11–80.99 at 2,000 positions. These are four-token timing
windows, not sustained-load confidence intervals. The remaining small
policy-to-policy differences should not be treated as a reliable ranking.

Default warm-request memory, in MiB:

| Observation | 128 before → after | 2,000 before → after |
|---|---:|---:|
| Full generation active peak | 7156.7 → 7156.7 | 7140.9 → 4794.6 |
| Additional peak after settled decode boundary | 4227.7 → 19.7 | 4449.7 → 241.7 |
| Retained conversion payload | 256 → 256 | 256 → 256 |

The default no longer pays repeated full-weight promotion during eligible
decode. Its long-context full-generation peak falls 32.9%; its short-context
peak remains dominated by prefill. Default warm prefill was 94.99 / 713.81 ms
afterward versus 92.51 / 711.06 ms before, so this change does not establish a
prefill speedup. F32 conversions are still admitted during prefill, and forecasts
remain conservative because the native multi-row fallback still needs that
workspace. Afterward, disabled retention had full-generation peaks of
7156.7 / 4538.6 MiB; 5 GiB and unlimited remained at 7176.7 / 8028.6 MiB.

Four additional native release fixtures passed after the timing window:
embedded mixed-width continuation with lookahead, target/drafter continuation,
controlled snapshot replay, and independent target/drafter fork edits. The
embedded cases preserved the previous 15 rejected proposals with lookahead off,
and six accepted/seven rejected with lookahead on. Together with the matrix's
controlled settled-trim parity, these cover reuse of the new native projection
through controlled and speculative drivers. Projection-level cast/allocation
profiling was not added: these end-to-end results demonstrate the gain, but do
not attribute every millisecond to an individual native operation. The subsequent
[projection baseline](projection-baseline.md) records real prefill layouts, native
GEMM selection, separate conversion/GEMM allocations and timing, including short
verification-sized projection replays.

Focused regression checks also passed: four mixed-projection Metal tests,
11 existing conversion/residency tests, and the nonblocking availability test.
The projection fixtures compare output bits for F16/BF16 weights, reduction
boundaries, tails, strided vectors, padded weights, bias and tied embeddings;
they also check native fallback and retention before/after trim. Reproduce with
native device access:

```sh
cargo test -p eredu-backend-mlx --features metal --lib mixed_projection --locked -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib parameter_conversion --locked -- --test-threads=1
cargo test -p safemlx --features metal --lib availability_query_does_not_evaluate_lazy_layout --locked -- --test-threads=1
```

Strict Clippy passed for the backend and safemlx libraries/tests, along with the
backend's no-default-features check and workspace formatting.

## Native behavioral coverage

The nonzero conversion fixture crosses allocator cache limits 0 and 256 MiB
with disabled, one-byte, exact-fit, partial, 32/256/512 MiB and unlimited retention.
It checks mixed F16/BF16 weights, unchanged logits and greedy tokens, aliases,
narrow-input bypass, independent groups, multiple permanent units, final-owner
release, invalidation/restoration and cancellation on native failure. Its tiny
budgets explicitly exercise an individual conversion larger than the cap.
Seven selected native tests passed with native device access.

The embedded speculative fixture uses one 1,024-byte budget for target and two
prediction owners, with F32 and mixed BF16/F32 weights and lookahead off/on. It
trims while snapshots and a fork remain alive, checks idempotence and typed
pending-transaction rejection, preserves tokens/RNG/snapshot accounting, and
compares restored replay output. After trim, both weight variants reject 15
proposals with lookahead off; with lookahead on each accepts six and rejects seven.
Thus the checks exercise actual rejection and rollback as well as acceptance.
The external mixed-BF16 fixture independently loads target and drafter, each
with a distinct 1,024-byte budget, and checks both scope reports. Each lookahead
mode accepts 25 proposals. Its retained state and restored replay remain equal.

Measured post-trim additional peaks stayed below their forecast bounds:

| Fixture | Lookahead | Measured bytes | Forecast upper bytes |
|---|---|---:|---:|
| Embedded F32 | Off | 10,784 | 1,996,132 |
| Embedded F32 | On | 10,776 | 3,308,742 |
| Embedded BF16/F32 | Off | 17,992 | 2,106,424 |
| Embedded BF16/F32 | On | 17,984 | 3,519,838 |
| External BF16 | Off | 51,458 | 138,440,438 |
| External BF16 | On | 51,474 | 141,788,654 |

Reproduce focused coverage:

```sh
cargo test -p eredu-backend-mlx --features metal --lib parameter_conversion \
  --locked -- --nocapture --test-threads=1
# Use the built native executable with accessible Metal for each filter:
<binary> native_embedded_continuation_forecasts_preserve_settled_state_and_bound_growth \
  --ignored --nocapture --test-threads=1
<binary> native_speculative_continuation_forecasts_cover_settled_state_without_advancement \
  --ignored --nocapture --test-threads=1
```

Multi-accelerator and multi-host retention admission is unvalidated. Multi-rank
retention continues to report explicitly disabled/unsupported eligibility;
portable partitioned and scope-admission coverage remains in the core/runtime,
architecture and backend conformance suites. This is not evidence of shared
cross-process retention admission. Native loopback CPU Ring execution passed nonzero Qwen2 resident-reference
checks for TP=2 (4.26 s), PP=2 (6.95 s), and combined TP=2 × PP=2 across four
processes (5.10 s). These compare prefill and cached decode with the single-rank
reference using the fixture's absolute logit tolerance of `1e-4`. They exercise
available native partitioned execution, with conversion retention effectively
disabled; they do not establish distributed retained-copy admission. The native
load-policy/eligibility fixture also passed (1.30 s).

```sh
cargo test -p eredu-backend-mlx --features metal --lib --no-run --locked
# Use the backend test executable printed by Cargo:
<backend-binary> loaded_conversion_retention_policy_reaches_native_residency \
  --nocapture --test-threads=1
<backend-binary> tests::distributed_pipeline_ring::ring_two_process_qwen2_tensor_parallel_resident_reference \
  --exact --ignored --nocapture --test-threads=1
<backend-binary> tests::distributed_pipeline_ring::ring_two_process_qwen2_pipeline_resident_reference \
  --exact --ignored --nocapture --test-threads=1
<backend-binary> tests::distributed_pipeline_ring::ring_four_process_qwen2_tensor_pipeline_resident_reference \
  --exact --ignored --nocapture --test-threads=1
```

## Portable and feature-boundary checks

The full portable runs passed 272 core tests; 649 runtime unit tests and 79
runtime integration tests; 631 architecture tests; 188 facade unit tests (two ignored), 112 backend
conformance tests, 27 portable facade tests (one ignored), one realtime test and
three doctests (one ignored). The neutral conformance suite includes policy
propagation, independent loads, scope exclusions, read-only queries, pending
rejection, speculative trimming, snapshot/fork accounting and forecast credit
removal after trim.

All crate/feature boundaries listed in `AGENTS.md` passed, including the portable
facade without MLX, backend without default features, portable NN/codec with all
features and media with/without host features. Strict Clippy (`-D warnings`)
passed for core, runtime, architectures, backend and facade libraries, the native
facade integration-test target, and the Metal backend library/tests. Reproduce:

```sh
cargo test -p eredu-core -p eredu-runtime --locked
cargo test -p eredu-architectures --lib --locked
cargo test -p eredu --no-default-features --locked
cargo check -p eredu-gguf
cargo check -p eredu-checkpoint
cargo check -p eredu-core
cargo check -p eredu-text --no-default-features
cargo check -p eredu-runtime
cargo check -p eredu-nn --all-features
cargo check -p eredu-codec --all-features
cargo check -p eredu-media --no-default-features
cargo check -p eredu-media --all-features
cargo check -p eredu-architectures
cargo check -p eredu-backend-mlx --no-default-features
cargo clippy -p eredu-core -p eredu-runtime -p eredu-architectures \
  -p eredu-backend-mlx -p eredu --no-default-features --lib --locked -- -D warnings
cargo clippy -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control --locked -- -D warnings
cargo clippy -p eredu-backend-mlx --features metal --lib --tests --locked -- -D warnings
```
