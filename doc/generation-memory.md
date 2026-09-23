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
arithmetic overflow remain explicit unknowns. Native model realization now applies
Eredu's 256 MiB cache ceiling to an untouched native default, preserving smaller
defaults. Explicit limits remain authoritative. Pure cold queries still observe
the current policy; they do not apply runtime configuration. To forecast with the
managed default before loading, call `configure_local_runtime(&Default::default())`
first. `--mlx-cache-limit-bytes 0` disables retention when desired.

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
all-row logits contract. Their admitted capture limits now supply a logical
instrumentation envelope, so a bounded top-k capture can receive `likely_fit`
without the application projecting capture storage itself:

- Native transforms and intervention execution/evidence use the same capture
  ledger. MLX completes them synchronously, so native storage is bounded by the
  smaller of the per-step and cumulative retained limits. Backends that do not
  declare completion before the next prediction use the cumulative retained limit.
- Host storage uses the cumulative host limit, plus the larger of the cumulative
  encoded limit and the entire trace byte limit. This allows retaining one run's
  records and one compact JSON trace; trace encoding already includes captures.
  It also covers a captured record that exists before trace delivery rejects it.
- The logical storage of admitted plans, including intervention payloads, is added
  on the host. Native storage belongs to execution pools; host storage belongs to
  the host pool. Unified memory counts both contributions once, as does CPU execution.

These intervals are planning estimates, not measured process peaks or physical
allocator guarantees. They exclude extra application copies, framing, snapshots,
branches, and private native workspace; allocator/graph allowances remain separate.
Limits can be conservative, and a large envelope can still prevent a fit verdict.
The forecast adds instrumentation to existing input costs and preserves unrelated
unknowns. It neither reserves nor consumes capture or trace budgets, and applies
equally to ordinary and controlled startup. Trace-only requests retain their
ordinary forecast. `forecast_prepared_speculative_generation` still leaves
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

`local_allocator_cache_policy()` additionally returns `limit_bytes` and a
snake_case provenance value: `native_default`, `managed_default`, `explicit`, or
`preserved`. Native allocator locks protect both fields. Every native setter marks
the policy explicit, even when its argument equals the existing default; callers
outside Rust are included. Restoring a saved numeric limit also counts as explicit.

The MLX adapter initializes an untouched policy at native target/model realization,
including ordinary, controlled and realtime loading. The automatic ceiling is
256 MiB; CPU's smaller native default stays smaller. Initialization is atomic with
setters and idempotent across sessions. It never overrides explicit settings or
an earlier initialization, and does not forcibly evict already retained cache.
Forecasts continue using `max(limit, retained)` plus their graph/driver allowance.
This is a process-wide performance/memory tradeoff, not a physical process cap.

Applications can retain the native policy before any model is realized:

```rust,ignore
use eredu::api::{configure_local_runtime, LocalAllocatorCachePolicy, LocalRuntimeConfiguration};
configure_local_runtime(&LocalRuntimeConfiguration::default()
    .with_allocator_cache_policy(LocalAllocatorCachePolicy::PreserveNative))?;
```

`with_allocator_cache_limit(bytes)` selects an explicit limit and can override
earlier initialization. Forecasts and policy getters never select a policy as a
side effect; before runtime initialization, a pure cold forecast can therefore
still report a large native allowance. Cold selection remains backend-neutral.

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

## Declaring already-resident memory

`already_resident_bytes` is the portion of the forecast's modeled costs that
already occupies its physical capacity pool when available memory is observed.
It is a baseline for additional demand, not a memory budget or another cost to
add. The estimator compares `max(peak - already_resident_bytes, 0) + reserve`
with observed available bytes. An application limit still compares with the
whole modeled peak plus reserve, including already-loaded weights.

The loaded-model facade and CLI derive this baseline from
`StaticMemoryReport::current_host_resident_bytes` and
`current_device_resident_bytes`, using `static_parameter_placement`. On unified
memory, neither the device counter alone nor an unconditional sum describes
all cases. Shared capacity does not mean host and device views share every
allocation:

| Known backing relationship | Resident parameter bytes to declare |
|---|---|
| Exact overlap of `S` bytes | `host + device - S` |
| Host and device allocations are known to be distinct | `host + device` |
| Overlap is unknown | `max(host, device)`, a conservative lower bound |
| Separate host/device capacity pools | Host count in the host domain; each device count in its own domain |

For example, 2 GiB host and 3 GiB device residency with unknown overlap gives a
3–5 GiB parameter interval. The loaded facade declares 3 GiB already resident,
so it may conservatively overestimate additional demand by up to 2 GiB. If 1 GiB
of shared backing is known, the union is 4 GiB. Low-level callers express known
overlap through `static_parameter_placement(&report, Some(shared_bytes))` and
use the resulting `lower_bytes` as their parameter baseline; `Some(0)` means
distinct allocations, while `None` means unknown overlap. The loaded facade
currently uses `None`. Missing observations retain unknown upper bounds.

Declare only bytes included in the same forecast and currently resident. Do not
use `logical_parameter_bytes` for a partially loaded model, planned disk bytes,
RSS, or process-global allocator counters: they can include unresident or
unrelated storage, and allocator activity can already include the parameter
tensors. The facade deducts parameter residency only; it does not deduct cache
or unrelated process allocations. Sample residency and available capacity close
together, and refresh both after loading or residency changes.

Cold inspection defaults to zero already-resident bytes. With separate memory
pools, `GenerationMemoryOptions::already_resident_bytes` applies only to the
execution device. Use the loaded facade or explicit `DomainMemoryPlan` entries
to account for independently resident host bytes too.

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
- Managed cache-default calibration (2026-09-23): pinned SmolLM-135M above,
  original weights, Metal on this 256 GiB Mac, 256/2,000 repetitions of ` hello`,
  512-token prefill chunks, 32 generated tokens, temperature zero. Each case ran
  in two fresh processes using `target/debug/eredu --model
  /tmp/eredu-memory-validation/SmolLM-135M --no-auto --raw --max-tokens 32
  --temperature 0 --prefill-chunk-size 512 --telemetry-json <output>` with
  `--mlx-cache-limit-bytes 33554432`, `268435456`, or the prior untouched native
  default (261,134,011,596 bytes on this host; supply that explicit byte limit
  to reproduce the native-policy comparison after this default-policy change).
  Checkpoint provenance is unchanged. Median observations:

  | Prompt positions | Cache policy | Decode tokens/s | Retained cache MiB |
  | --- | --- | --- | --- |
  | 256 | 32 MiB | 73.84 | 31.6 |
  | 256 | 256 MiB | 74.47 | 254.8 |
  | 256 | Native | 75.56 | 419.6 |
  | 2,000 | 32 MiB | 70.89 | 32.1 |
  | 2,000 | 256 MiB | 71.21 | 255.5 |
  | 2,000 | Native | 74.31 | 1,950.4 |

  The selected 256 MiB ceiling traded about 1.4–4.2% decode throughput for
  lower retention here. This short, single-model calibration does not establish
  an optimal ceiling for large, routed, distributed or realtime models; those
  applications can preserve native policy or choose an explicit limit. The
  native limit is a retention policy, not a strict total-allocation guarantee.

  After rebuilding with managed initialization, the loaded CLI smoke command
  above with no cache flag reported `likely_fit`, a 268,435,456-byte managed cache
  limit and a 875,655,276-byte generation upper bound. Native provenance tests
  run in fresh CPU/Metal processes and cover default configuration, preservation,
  unchanged-value direct native setters, setter/initializer races, and smaller
  native defaults. Facade tests also verify that forecasts leave policy untouched.
  CUDA carries the same native implementation but was not compiled or executed
  on this macOS validation host.

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
