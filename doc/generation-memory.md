# Generation memory forecasts

Memory forecasts help compare a specific request with an application budget and
current available capacity. They are planning estimates, not allocation limits or
a guarantee about total process memory. Ordinary generation does not depend on
estimation being available.

## CLI

Inspect a model before allocating weights, using the number of model input
positions (including any chat-template positions):

```sh
eredu --model /path/to/model --no-auto --estimate-memory-tokens 2000 \
  --max-tokens 32 --prefill-chunk-size 512 \
  --memory-budget-bytes 2147483648 > memory.json
```

Omitting `--max-tokens` in cold inspection produces an explicitly labeled
256-output-position forecast and reports persistent-state growth per position;
it does not claim a finite peak for unlimited generation. Cold inspection does
not tokenize or render the supplied prompt. Actual generation still resolves its
ordinary checkpoint/default output allowance.

Report the exact tokenized generation request:

```sh
eredu --model /path/to/model --raw --max-tokens 32 \
  --memory-report request-memory.json --memory-budget-bytes 2147483648 \
  --memory-policy refuse 'Explain memory locality.'
```

The request report runs after loading and tokenization, before generation. Use
cold inspection to assess loading risk. `warn` is the default policy and allows
execution; `refuse` stops a predicted shortfall. Incomplete information is reported
and does not disable valid execution. Context, tensor validity, residency and
other ordinary safety checks remain independent. A memory fit is not context
admission; a forecast horizon can extend beyond the configured context limit.

`--memory-budget-bytes` applies to total modeled memory in the execution capacity
pool; it differs from the existing parameter-cache device/host budgets. Observed
available capacity is compared with **additional** request memory after subtracting
the declared resident portion. `--memory-reserve-bytes` defaults to 256 MiB per
physical pool for other work and uncertainty. Shared library overhead uses an explicit
0–64 MiB graph/driver planning interval plus the configured allocator-cache
limit, separately from this reserve. Without `--mlx-cache-limit-bytes`, reports
query the allocator's actual current limit. The allowance uses the larger of
that limit and already retained cache, because lowering the limit may defer
eviction. An explicit cache limit during cold inspection describes the proposed
policy; reports after loading observe the policy in force. Query failures and
arithmetic overflow remain explicit unknowns. A large native default can produce
a pessimistic forecast; `--mlx-cache-limit-bytes 0` disables retention when that
is the desired runtime policy.

On macOS, `discover_local_hardware()` observes available host memory from Mach
host VM statistics: free pages (which already include speculative pages) plus
inactive pages, multiplied by the host page size. Apple Silicon shares this
observation between host and Metal in one unified capacity pool. This estimated,
point-in-time reclaimable capacity can support a fit verdict without
`--memory-budget-bytes` when the request's modeled upper bounds are known.
It is not a reservation or process limit; reclaiming inactive pages may require
work, and other processes can consume the capacity after observation. Failed
queries remain unavailable, while a successful zero-capacity observation is zero.
The calculation follows Apple's [VM statistics definitions](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h).

Reports show phase contributions, interval bounds, missing coverage, state growth,
effective prefill size, and the dominant contributor. Smaller chunk and output
candidates are recomputed; recommendations are shown only when their modeled
generation peak falls. A different quantization or residency policy requires a
fresh inspection because conversion and transfer costs can change the result.

## Chunked prefill

Ordinary text generation requests chunks of 512 tokens by default.
`--prefill-chunk-size 0` requests a single pass. The shared generation driver
settles intermediate chunks before proceeding and samples only after the final
chunk. Controlled generation uses that same driver. Chunk boundaries preserve
cache positions and cancellation is checked between completed chunks.

Requests whose capture, media, speculative or distributed execution contract
requires a full prompt retain their existing execution behavior. The memory
projection uses a full prompt for uncovered selected paths; it does not assume
that asking for a smaller chunk makes every mechanism chunkable. Specialized
workspace coverage is explicit rather than silently assigned zero.

Cold callers can query `inspection.selected().unwrap().preparation()
.prefill_chunking_support()` before loading. This reports ordinary unobserved
plain-text support from retained architecture and backend facts, including
prediction, composite and distributed restrictions. LFM2.5 currently reports
unsupported; the Llama path used by SmolLM2 supports it. This describes current
implementation coverage, not an inherent limitation of the LFM architecture.
Prepared media, capture/intervention and external speculation can still require
a full pass after selection.

`inspected_generation_memory_request` derives chunk eligibility from that fact;
`GenerationMemoryOptions::chunked_prefill_supported` is deprecated and ignored.
To forecast an explicitly unchunked request, set `prefill_chunk_tokens` to the
input's model-position count. Workspace coverage is independent: a chunk-capable
routed decoder may still have an unknown workspace bound. Smaller chunks never
claim savings for a selection that requires a full pass.

## JSON wire format

Forecast requests, estimates and CLI JSON reports use snake_case enum values.
Rust enum variant names are separate from these serialized values:

| Field | JSON values |
|---|---|
| `fit`, `generation_fit` | `"likely_fit"`, `"likely_shortfall"`, `"insufficient_information"` |
| `phase` | `"loading"`, `"prefill"`, `"decode"` |
| `cache_update` | `"in_place"`, `"copy_state"`, `"unknown"` |
| `logits` | `"final_position"`, `"every_position"` |

Domains always use a tagged object, preserving device identity as a separate
field without normalizing the identifier:

| Domain | JSON value |
|---|---|
| Unified | `{"kind":"unified"}` |
| Host | `{"kind":"host"}` |
| Device | `{"kind":"device","device":"cuda:0"}` |

Attention workspace also uses a tagged object: `{"kind":"materialized"}`,
`{"kind":"score_matrix_upper_bound"}`, `{"kind":"unknown"}`, or
`{"kind":"fused","scratch":{...}}`, where `scratch` is a `MemoryBytes` record.
Consumers can switch on `kind` for every variant; device identifiers and scratch
estimates are never discarded to make a payload-bearing enum into a string.

This replaces the initial PascalCase and externally tagged representation.
Previously saved reports require migration: for example, `"LikelyFit"` becomes
`"likely_fit"`, `"Unified"` becomes `{"kind":"unified"}`, and
`{"Device":"cuda:0"}` becomes `{"kind":"device","device":"cuda:0"}`.
Deserialization uses the new format as well.

## API and ownership

Prefer `LoadedModel::forecast_prepared_generation(&request, &options)` for a
prepared ordinary request. It derives encoded input positions, checkpoint-resolved
output limits, prefill eligibility, the actual logits projection contract, selected
state geometry, parameter residency, available capacity and allocator policy.
It borrows the request without executing callbacks, advancing state or reopening
checkpoint files:

```rust,ignore
use eredu::api::{GenerationForecastOptions, MemoryBudget};

let options = GenerationForecastOptions {
    budget: MemoryBudget {
        application_limit_bytes: Some(2 * 1024 * 1024 * 1024),
        reserve_bytes: 256 * 1024 * 1024,
        ..Default::default()
    },
    ..Default::default()
};
let forecast = model.forecast_prepared_generation(&request, &options)?;
println!("{:?}", forecast.estimate.fit);
// Pass the same request to ordinary generation or controlled startup.
```

`forecast_token_ids` supports already encoded requests with the same prepared
settings. `forecast_observed_generation` consumes the preparation shared by
observed ordinary and controlled execution: trace-only requests preserve ordinary
chunking, while capture/intervention requests report their full-pass reason and
all-row logits contract. Capture transformations and retained records leave the
upper bound unknown. `forecast_prepared_speculative_generation` similarly leaves
concurrent draft and verification storage explicitly unbounded.

`GenerationForecast` contains the estimate, descriptive request, full-pass reason
and logits contract. `with_prefill_chunk` recomputes a candidate only when the
request supports chunking; it preserves full-pass requirements. Loaded forecasts
exclude completed loading and deduct only declared resident parameter backing
from additional memory, never the process-global active allocator counter. Fresh
requests should start from reset state. The MLX adapter detects nonempty or
unavailable state and leaves its continuation peak unbounded rather than silently
forecasting an empty cache. Prepared media, bounded transfers and unprojected
families likewise retain explicit unknowns.

`ForecastCalibration::default()` is the single library-owned source for the
score-matrix fallback, copy-state cache update, one-set-per-layer plus 25 percent
scratch envelope, and 64 MiB graph/driver allowance. Loaded forecasts apply it
and sample the allocator's current limit automatically. Applications can override
individual calibration fields or `backend_overhead`; budget, reserve and
warning/refusal decisions remain application policy. The CLI uses these same
facade calls. `forecast_inspected_generation` applies the same calibration to
cold selected geometry; its prefill contract comes from the selected preparation,
before a native executor exists.

`eredu::api::estimate_inspected_generation_memory` consumes a retained cold
`ModelInspectionOutcome` and `GenerationMemoryOptions`.
`inspect_generation_memory` combines metadata inspection and estimation using
cold backend mechanism facts. Neither function creates a native device or tensor.
`inspected_generation_memory_request` exposes the descriptive request for
recomputing supported alternatives.

With the `mlx` feature, `GenerationMemoryOptions::for_local_backend(input,
placement)` samples the current cache limit and retained bytes and defaults
`backend_overhead` to their maximum plus the 64 MiB graph/driver planning allowance.
This is an estimated allowance, not a measured upper bound on all graph or driver
allocations. Callers can override `backend_overhead` with their own calibration;
the lower-level estimator requires explicit attention, cache-update and graph-overlap
coverage, while `forecast_inspected_generation` supplies the shared calibration.
The generic `GenerationMemoryOptions::new` remains backend-neutral and leaves
native overhead unknown. Local policy sampling can initialize the native allocator;
the subsequent cold inspection and estimation remain free of native queries.

`local_allocator_cache_limit()` reads the native value without changing policy or
evicting cache. `set_local_allocator_cache_limit(bytes)` returns the previous value
for restoration; the backend adapter's setter now also preserves that result:

```rust,ignore
let previous = eredu::api::set_local_allocator_cache_limit(32 * 1024 * 1024)?;
let options = eredu::api::GenerationMemoryOptions::for_local_backend(input, placement);
// Inspect or run the request, then restore policy when appropriate.
eredu::api::set_local_allocator_cache_limit(previous)?;
```

Both policy and observations are process-global snapshots. Coordinate mutations
with other users, restore policy on error paths as well, and recreate options
after policy changes. The native getter also observes changes made outside Rust;
there is no shadow configuration or mutation-based getter.

Architecture projections reuse normalized state schedules and supply decoder
workspace dimensions. Runtime planning combines those with selected residency,
materialization, attention, cache update, scalar-width and physical-placement
facts. Low-level `GenerationMemoryRequest` accepts multiple disjoint physical
pools and simultaneous rank-local executions. Supply exact local geometry and
count shared parameter backing once; replicas consume separate payloads.

State follows the existing architecture estimator, including sliding windows,
recurrent tensors and allocation granularity. Periodic remainder-shaped state
whose intermediate maximum is not modeled retains an unknown upper bound. Prefill evaluates both the last
full chunk and the final remainder. Decode evaluates the output frontier.
Each phase includes overlapping parameters, state, retained input, workspace,
staging and overhead; the lifecycle peak is the maximum of phases. Loading
includes source/destination/conversion overlap separately from generation.

Dense workspace currently models residuals, Q/K/V, gated feed-forward arrays,
logits/probabilities, selected attention scratch and cache-update overlap. Ordinary
shared dense readout projects only the last hidden row in each chunk, including
the full-prompt case; observed full-logit contracts retain all rows. The
shared calibration conservatively assumes float32 score/probability matrices even when MLX
selects a fused kernel, and a whole extra state payload during cache replacement.
Attention contributes an interval from zero to that score-matrix envelope, so
a budget crossed only by its upper end yields insufficient information. The
shared calibration linear-workspace upper envelope permits one activation set per local state
layer plus 25% scratch margin, calibrated against the small Metal matrix below.
The lower end models one set. This accounts coarsely for lazy graph retention;
it does not assert exact tensor lifetimes and is uncalibrated on other models
and hardware. The initial one-layer formula underpredicted long-prompt peaks
by 22–35%; measured calibration is essential to these planning assumptions. Recurrent, routed,
media, speculative and distributed paths retain calculated facts but report
uncovered workspace rather than claim comprehensive coverage. The public runtime
API supports explicit fused scratch estimates and local device geometries.

Unified memory describes shared capacity, not automatic identity of every host
and device buffer. Separate devices are compared independently. The static-report
adapter accepts declared shared backing or reports an overlap interval; global
allocator counters are never added on top of parameter tensors.

## Reproducing calibration

Build the current source first. Old executables do not validate these changes.

```sh
cargo build -p eredu-cli --locked
python3 validation/memory_estimation_matrix.py \
  --model /tmp/eredu-memory-validation/SmolLM-135M \
  --output /tmp/eredu-memory-validation/matrix
```

The script runs fresh processes with original and 4-bit weights, short and long
prompts, one-pass and 512-token prefill, and eight greedy decode positions.
It records forecasts, MLX allocator peaks, macOS process peak RSS and process
peak footprint separately.
MLX peak measures active allocations tracked by MLX; RSS includes other resident
process pages. Footprint is the operating system's separate process charge and
can differ substantially from both. Neither measures total system unified-memory pressure. It disables
the allocator cache for a reproducible comparison; production allocator caching
and concurrent work need their own allowance.

The calibration checkpoint is the official `HuggingFaceTB/SmolLM-135M` revision
`1d461723eec654e65efdc40cf49301c89c0c92f4`, fetched by immutable revision URL.
Its `model.safetensors` is 538,090,408 bytes, SHA-256
`c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093`.
The local validation manifest records every fetched file's byte count and digest.
This release has 30 layers, hidden width 576, 9 query heads, 3 KV heads,
49,152 vocabulary entries and a 2,048-position configured context. Although its
configuration names bfloat16, this stored checkpoint contains float32 weights;
estimation follows selected materialization/state metadata.

## Observed calibration (2026-09-22)

Fresh debug builds ran on macOS/aarch64, MLX Metal, 256 GiB unified physical
capacity. The backend could not observe current available capacity; application
budget comparisons below therefore make no claim about free system memory.
Allocator caching was disabled. All eight requested greedy output tokens matched
between full and chunked prefill, for both weight formats and prompt lengths.

Values below are MiB. Forecast upper ends exclude the separate 256 MiB safety
reserve. The error column is `(forecast upper / measured MLX peak - 1) × 100`.
This is calibration on these cases, not independent validation of universal bounds.

| Weights | Input positions | Chunk | MLX active peak | Forecast upper | Upper excess | Process peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 32 | full | 574.6 | 616.9 | 7.4% | 633.8 |
| float32 | 32 | 512 | 574.6 | 616.9 | 7.4% | 636.4 |
| float32 | 2000 | full | 2633.8 | 3310.2 | 25.7% | 637.9 |
| float32 | 2000 | 512 | 1302.0 | 1350.8 | 3.7% | 635.6 |
| affine 4-bit | 32 | full | 244.1 | 275.2 | 12.8% | 483.1 |
| affine 4-bit | 32 | 512 | 250.0 | 275.2 | 10.1% | 475.2 |
| affine 4-bit | 2000 | full | 2296.6 | 2968.5 | 29.3% | 479.1 |
| affine 4-bit | 2000 | 512 | 866.5 | 1009.0 | 16.4% | 479.0 |

Process footprint also demonstrates the estimate's coverage limit: it ranged
from 968–2,089 MiB for float32 and 802–1,815 MiB for four-bit runs. The four-bit
short-request footprint exceeded the modeled generation upper plus the default
reserve. Executable/shared-library pages and unrelated process allocations need
additional application headroom; this model/request forecast must not be used
as a process-footprint bound. Library consumers can supply their own overhead
interval and reserve after measuring their deployment.

For the long prompt, chunking reduced measured active-allocation peaks by 50.6%
(float32) and 62.3% (4-bit). Process RSS changed little: it is a different counter
and must not be substituted for the MLX peak or added to it as disjoint storage.

With a **2 GiB application budget**, float32 chunked cold inspection reported
`LikelyFit`: its loading-inclusive upper was 1,414.3 MiB; with the 256 MiB reserve
it remained below the budget. Full prefill reported `InsufficientInformation`
because its interval crossed that budget; its actual 2,633.8 MiB MLX peak exceeded
it. A **512 MiB** budget reported `LikelyShortfall`, and `--memory-policy refuse`
exited with status 1 before model loading. These are modest application budgets,
not the host's 256 GiB capacity. Cold inspection plus candidate recomputation
took 37–39 ms per fresh process on this checkpoint.

A separate run of cached official `LiquidAI/LFM2.5-1.2B-Instruct` revision
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`, with 128 input positions and eight
decode tokens, completed successfully while reporting an unavailable workspace
upper bound. Its recurrent workspace was not treated as zero and ordinary
generation remained available. This is coverage of the uncertainty path, not
calibration of hybrid-model peak estimates.

Native distributed placement, media workspace and other model/hardware
calibrations remain open. CUDA validation was explicitly deferred. The native
fixture checks and portable domain tests do not establish those native results.

## Focused verification

- Cold chunking follow-up: portable preparation tests cover supported and absent
  backend mechanisms, Llama, LFM2, routed, composite, prediction and partitioned
  selections. CPU/Metal fixture tests compare cold support with the loaded executor
  across resident, host, disk and affine-quantized paths. Rebuilt CLI cold reports
  on the pinned SmolLM and LFM2.5 checkpoints above used 2,000 input positions,
  eight output positions, a requested 512-token chunk and zero allocator cache.
  SmolLM retained 512 and recomputed smaller-chunk savings; LFM2.5 used all 2,000
  positions, reported the architecture's full-pass reason and suggested no chunk
  savings. The existing unknown workspace bound remained unknown.
- Prepared-request facade forecasts are covered with a neutral mock backend:
  exact request settings, resident-byte deduction independent of allocator activity,
  calibration overrides, chunk recomputation, trace-only controlled parity,
  capture uncertainty and foreign-session rejection. Native CPU/Metal tests compare
  declared final-row/full-logit contracts with actual output shapes and check that
  reset restores fresh-request forecast coverage.
- The loaded CLI smoke test on the pinned SmolLM checkpoint used `--no-auto --raw
  --max-tokens 4 --mlx-cache-limit-bytes 0 --memory-report report.json Hello`.
  It reported `LikelyFit`, resident parameters of 538,060,032 bytes, a generation
  upper of 607,219,820 bytes and additional upper of 69,159,788 bytes. The loaded
  lifecycle upper equaled the generation upper, excluding completed loading.

- Cache-policy follow-up: native getter/setter/restore tests passed on CPU and
  Metal, including a native C setter bypassing the Rust wrapper and repeated
  queries preserving populated cache. Local facade policy tests passed on both
  builds; portable facade/conformance tests passed (19 and 82, one preexisting
  ignored test). CUDA getter implementation is present but was not run locally.
- On the pinned SmolLM checkpoint above, `target/debug/eredu --model
  /tmp/eredu-memory-validation/SmolLM-135M --no-auto --estimate-memory-tokens 16
  --max-tokens 4` observed the native default cache limit of 261,134,011,596 bytes
  and produced a finite loading-inclusive upper of 262,616,979,148 bytes without
  a cache flag. Its interval crossed available memory and honestly remained
  `InsufficientInformation`; an explicit proposed zero-cache policy reported
  `LikelyFit`. Metal tests and the default-policy CLI observation ran outside
  the sandbox; a denied Metal query remained an explicit unknown.
- `cargo test -p eredu-runtime --lib memory_estimation`: 20 tests, covering
  overlap, placement, state growth, unknowns, overflow and recomputed candidates.
- `cargo test -p eredu-architectures --lib memory_estimation::tests`: 3 tests.
- `cargo test -p eredu-core`: 254 tests, including shared-driver parity and
  between-chunk cancellation; facade cancellation without a first token is
  covered separately.
- `cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance`:
  19 and 82 passed respectively; one preexisting ignored facade test.
- `cargo test -p eredu-backend-mlx --features metal --test chunked_prefill --locked -- --test-threads=1`:
  CPU and Metal passed, outside the sandbox. Nonzero dense/sliding GQA fixtures
  cover prompts of 1, 3 and 9 tokens, chunk sizes 1, 2, 4 and 32, three cached
  decode steps, resident/host/disk paths and affine four-bit weights. Full
  observed logits and chunked results agree within absolute tolerance `2e-4`;
  ordinary and controlled greedy token sequences match exactly.
- Portable core/runtime/architecture checks, the portable facade check, and
  `cargo check -p eredu-backend-mlx --no-default-features` passed. Package lists
  and offline `cargo package --no-verify --allow-dirty` for core, runtime,
  architectures and facade passed. This verifies archive inclusion/resolution,
  not the full release script's extracted-package/all-target test matrix.
