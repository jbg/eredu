# Generation memory forecasts

Memory forecasts help compare a specific request with an application budget and
current available capacity. They are planning estimates, not allocation limits or
a guarantee about total process memory. Ordinary generation does not depend on
estimation being available.

## Neutral resource-description contract

`eredu_core::resources` describes resources from ordinary execution contracts.
Target workspace forecasts now compose reusable mechanism descriptions with
explicit evaluation lifetimes. Parameter/state producers and native mechanism
inventories remain descriptive contracts; their unknown allocation capacities
are not silently converted into exact bounds. Embedded prediction coverage is
still a later phase.

A `ResourceDescription` records the current context and requested horizon as named
logical extents, with fixed or evaluated context-dependent resource sizes. For
example, a 39-position cache might hold 1,248 payload bytes in a 2,048-byte
allocation; a 30-position continuation might reach 2,208 payload bytes and a
4,096-byte capacity after allocation rounding. A score matrix can separately carry
a query-by-key scratch bound. Horizon bounds cover the entire interval, including
the starting state and temporary interior peaks, rather than only the endpoint.
These are producer-supplied facts, not a built-in geometry calculator.

Payload means bytes in the stored representation, including quantization metadata.
Allocation capacity already includes payload and unused reservation space; the
two are never added. Backing identities are distinct from logical owner identities
and physical-pool identities. Tied parameters/views share backing, while replicas,
copies and cached conversions get distinct allocation identities. Host/device
aliases on unified memory share a pool identity; identical artifact names or pool
names on different hosts do not imply shared storage or capacity.

Incomplete resource-set coverage, unknown upper bounds and unknown placement
remain explicit. An empty list cannot imply completeness, and exact payload alone
cannot establish finite native capacity. Constructed or decoded descriptions must
pass `ResourceDescription::validate()` before use. Validation rejects malformed
intervals, repeated backing identities, guessed pools and horizons that exclude
the starting state; the producer remains responsible for its equations and scope.
Descriptions carry no allocation authority, resident credit, lifetime ordering or
fit verdict, and summing their entries is not a peak calculation. Producing one
must not advance or synchronize execution or consume admission budgets. Existing
`GenerationMemoryEstimate` reports retain their format; target workspace values
now reflect the generic composition described below.

`eredu_runtime::execution_resources::PreparedResourceQuery` gives an execution
instance namespace, batch extent, hypothetical persisted prefix, additional
positions, and an optional authoritative device-pool observation. Call
`describe_prepared_resources` on the selected realization, prepared construction
contract, or retained `ReplicatedTextSession`. These queries describe prepared
geometry at the supplied prefix; they do not observe an installed session or
replace the live continuation outlook API.

The retained-session path reads ordinary `PreparedParameterSlot` metadata: exact
recipe output bytes, materialization location and canonical binding backing.
Actual binding aliases share one entry with several logical uses. Independent
materialization batches, copies and quantization companions stay separate.
Checkpoint aliases and shared source keys do not prove sharing. Cold selection
without exact prepared output facts reports named missing parameters. Aggregate
bank-member catalogs and prediction modules remain explicit gaps, as do missing
slot metadata and ambiguous physical backing. Bounded weight residency does not
infer that all copies occupy one pool.

State comes from `SelectedStateRealization`, including local component shape,
selected storage dtype, conditional presence and persistent segment offsets.
The shared `StateComponentPolicy` element evaluator covers full prefix ranges,
including quotient/remainder pooling shapes whose interior maximum exceeds both
endpoints. For mixed nonlinear axes the upper end can conservatively combine
dimension maxima that occur at different positions. Sliding attention preserves
the interval from visible-window payload to full-prefix retention until its
storage mechanism supplies a tighter fact. Frame-local state needs a frame
horizon, and paged state needs block/tier decomposition; both remain named gaps.

Exact logical payload does not establish native allocation capacity. Capacity,
alignment, replacement copies, mechanism scratch, cached conversions and retained
outputs remain unknown; descriptions name missing execution-unit and static-module
contracts. They are always partial for the complete execution. No new fit verdict
or embedded prediction support is enabled by these producers. They add no
family-specific forecasting dependencies or calibration constants. Production
workspace forecasting uses ordinary invocation topology; the legacy aggregate
workspace input remains supported for existing manually constructed or serialized
requests.


Neutral tests derive descriptions from ordinary Llama, LFM2, routed Qwen and
sliding-window Mistral selections, including queries after checkpoint files are
removed. Session conformance checks preserve materialization, state-allocation and
execution counters and distinguish local stateless partitions. Reproduce with:

```sh
cargo test -p eredu-core -p eredu-runtime
cargo test -p eredu-architectures --lib resource_descriptions
```

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
decoder with an undescribed mechanism may still have an unknown workspace bound. Smaller chunks never
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
all-row logits contract. Capture forecasts use selection geometry when the backend
can bound the complete source and transform cost. MLX currently covers ordinary
`model.logits` transforms, including top-k candidates and selected-token scores.
The projection reuses admission's native transform estimator, resolves slices,
counts absolute prefill/decode schedules, and includes per-step diagnostic metadata
even when no value is selected. Top-k storage depends on vocabulary width and k;
retained host records depend on selected occurrences and the prediction horizon.
Increasing an already sufficient admission quota does not increase this projection.

- Native transforms and intervention execution/evidence share the capture ledger.
  MLX completes transforms synchronously, so the native bound is the largest
  projected step, capped by per-step and remaining cumulative retained limits.
  Backends allowing cross-step overlap use cumulative projected native storage.
- Host storage covers projected records plus the larger of projected encoded
  records and the entire trace byte allowance. Projections are capped by admitted
  cumulative limits. This permits one retained record history and one compact JSON
  trace, including a capture produced before trace delivery rejects it.
- Immutable admitted plans, including intervention payloads, are added on the
  host. Native costs belong to execution pools; host costs belong to the host pool.
  Unified memory counts both contributions once, as does CPU execution.
- Unknown geometry, deferred source creation without a complete cost, distributed
  capture, and intervention execution/evidence currently retain an explicitly
  labeled admitted-limit fallback. A missing native cost is never treated as zero.

`GenerationForecast.capture` retains serializable projection facts.
`with_max_output_tokens` recounts scheduled occurrences without accumulating the
previous forecast's costs. Shape costs conservatively cover the original admitted
range; extending beyond that coverage uses the admitted-limit fallback. Continuation
forecasts use the sampler's absolute prediction index and actual cumulative capture
reservations. Restoring a snapshot does not refund delivered record history, while
completed step-scoped native transforms do not remain live.

These intervals are planning estimates, not measured process peaks or physical
allocator guarantees. They exclude extra application copies, framing, snapshots,
branches, and private native workspace; allocator/graph allowances remain separate.
The independent trace allowance remains even for small captures. The forecast adds
instrumentation to existing input costs and preserves unrelated unknowns. It neither
reserves nor consumes capture or trace budgets and applies equally to ordinary and
controlled startup. Trace-only requests retain their ordinary forecast.

Native calibration (2026-09-23): SmolLM2-135M-Instruct at revision
`1d461723eec654e65efdc40cf49301c89c0c92f4`, Metal, top-k 8 for 32 predictions,
1 GiB retained/host/encoded capture ceilings, and a 64 MiB trace allowance produced
68,337,271 bytes of retained upper allowance (about 65.2 MiB). The projected
capture host total was 37,088 bytes and the native step peak was 1,184,768 bytes.
Increasing sufficient quotas did not change the result; increasing k raised it
and capturing every other prediction lowered it. Four executed predictions stayed
within the projected ledger usage; continuation forecasts preserved that history
and neither forecasts nor quota comparisons changed native active allocations.
Run against the pinned local checkpoint with:

```sh
EREDU_CONTINUATION_MODEL=/path/to/SmolLM2-135M-Instruct \
  cargo test -p eredu --features mlx,metal --offline --test native_execution_control \
  native_capture_forecasts -- --ignored --nocapture --test-threads=1
```

### Speculative requests

`forecast_prepared_speculative_generation` borrows the same request accepted by
uninterrupted generation and `with_controlled_text_speculative` /
`with_controlled_chat_speculative`. `forecast_speculative_token_ids` exposes the
same projection for consumers that already counted/tokenized the input. The MLX
adapter currently supplies complete speculative resource facts for **independent
autoregressive drafters** when both models have ordinary workspace coverage.
Both must already be loaded. Forecasting reads retained geometry, residency and
native policy; it does not allocate a cache, run either model or consume a request.

The runtime projects full-pass prefill, drafting, verification and commitment
phases. Each includes distinct target/draft parameters, canonical caches,
rollback checkpoints, proposal seed/restore/replacement copies, accepted-prefix
replay and retained sampling distributions. Verification and replay permit K+1
rows for K draft tokens. Configured lookahead permits one additional block; the
estimate does not assume acceptance, early termination or adaptive disabling.
Global verification/branch ceilings do not multiply one lane's footprint. Token
history copies belong to the host pool; separate physical pools are compared
independently. A shared allocator is counted once, and both models' declared
resident parameter contributions are deducted from additional demand.

The v1 transaction envelope budgets up to three additional target and five
additional draft state payloads beyond canonical state/cache-update workspace,
plus two target and three draft payloads for lookahead. Prefill uses one/two
additional payloads respectively. These are conservative simultaneous-copy
allowances, not claims that every copy is allocated. Sampling uses an MLX-owned
calibration of 32 float32/index rows per live distribution, with target, draft
and optional optimistic proposal rows. Kernel scratch, graph and allocator
allowances remain planning assumptions rather than physical guarantees.

`GenerationForecast.speculative` retains the exact plan for JSON round trips
and recomputation. `with_max_output_tokens` updates both models; chunk changes
preserve speculative full-pass prefill. The CLI uses these facade operations for
its report and shorter-output advice.

Embedded prediction heads and feature-conditioned assistants still need their
architecture-specific prediction-state, retained-feature and workspace
projections. MLX reports these as unknown; it does not substitute an ordinary
draft model or count embedded parameters twice. Cold forecasts without a
realized drafter, cross-device MLX execution outside unified memory, uncovered
ordinary family workspaces, media and transfer costs also remain incomplete.
Other backends opt in through `SpeculativeForecastBackend<D>`; its default report
is unavailable. The portable `SpeculativeMemoryPlan` can describe embedded
ownership and independent physical pools when a backend has the required facts.

This forecasts one fresh lane. Extra controlled snapshots/branches, retained
traces, captures/interventions enabled after startup and concurrent lanes need
their own accounting. Ordinary and controlled execution share the startup
forecast; the API does not claim to bound subsequent user-controlled retention.

### Speculative startup validation

Validation on 2026-09-23 used debug builds on macOS/aarch64 Metal with 256 GiB
unified memory and the pinned `HuggingFaceTB/SmolLM-135M` checkpoint documented
below (`1d461723eec654e65efdc40cf49301c89c0c92f4`). Two independently loaded copies
used the original weights. In a separate fixture directory, only
`tokenizer_config.json` was changed to add the literal template
`{% for message in messages %}{{ message['content'] }}{% endfor %}`; the base
checkpoint has no chat template. The original checkpoint remained unchanged.

| Execution | Input/output positions | Forecast upper, bytes | MLX peak active, bytes |
| --- | ---: | ---: | ---: |
| One Metal stream, greedy, lookahead disabled | 9/32 | 1,515,947,000 | 1,101,895,909 |
| Separate Metal streams, stochastic, lookahead enabled | 9/32 | 1,561,923,812 | 1,104,419,037 |
| Separate Metal streams, longer prompt | 289/32 | 2,530,177,084 | 1,580,703,442 |
| Metal target, CPU draft, unified pool | 9/16 | 1,549,377,956 | 1,099,604,507 |

All four forecasts returned `likely_fit`. The short split-stream run created six
optimistic blocks, reused two and discarded four; the longer run reused six.
The two identical models accepted all verified draft tokens; rejection/replay
and controlled/uninterrupted parity are covered by neutral conformance tests.
These observations validate a small model's planning envelope, not a universal
allocator or process-RSS bound. CUDA and separate physical device pools were not
validated natively.

To reproduce after building with `cargo build -p eredu-cli`, set `MODEL` to the
fixture directory and run:

```sh
target/debug/eredu --no-auto --model "$MODEL" --draft-model "$MODEL" \
  --speculative-draft-device gpu:0 --speculative-draft-tokens 4 \
  --disable-speculative-adaptive-lookahead --max-tokens 32 \
  --temperature 0.8 --seed 42 --memory-report /tmp/speculative.json --verbose \
  'The quick brown fox jumped over the fence and'
```

For the serial row, omit `--speculative-draft-device`, use `--temperature 0`,
and add `--disable-speculative-lookahead`. For the CPU row, select `cpu` and
16 output tokens. For the long row, pipe 32 repetitions of
`The quick brown fox jumped over the fence. ` as the prompt.

### Settled speculative continuation

Inside `with_controlled_text_speculative` or `with_controlled_chat_speculative`, use
`session.forecast_remaining_generation(additional_tokens, &options)` after prefill
or a canonical commit. The object-safe operation returns
`SpeculativeContinuationForecast`: target `request`, the existing `speculative`
plan (including draft request), horizon-specific `continuation` observations, and
an `estimate`. It performs no native execution, completion polling, synchronization,
cache copy or budget reservation. It does not change the configured output limit.

The first implementation covers independent autoregressive drafters. It observes
both installed cache frontiers separately, current backing-capacity allowances,
state growth through the horizon and speculative overshoot, and the retained
assistant seed. It refreshes both parameter residency reports (including cached
conversions), available capacity and allocator-cache policy. Only parameters receive
already-resident credit. State, seed and snapshot/branch reservations are conservative
upper allowances, not measurements of distinct physical backing.

The `continuation_start` phase has installed state and no new forward workspace.
Positive horizons add drafting, verification and commit phases using the existing
copy, distribution and lookahead envelopes with native capacity substituted for
logical cache payload. Loading and prefill are excluded, including their validation
arithmetic. A zero-token horizon still includes current state and retained user
snapshots/branches. Native `additional_input_tokens` includes configured speculative
overshoot; `continuation.additional_tokens` is the requested committed-token horizon.
Reobserve for another horizon instead of editing that descriptive record.

Canonical boundaries contain no pending proposal distribution or completion.
Retained sampler/RNG/history and semantic state are observed, with conservative
future fork/replay growth and one compact trace allowance. Custom samplers,
controllers or semantic parsers without a continuation storage contract keep the
upper end unknown. This combined allowance includes native RNG storage and is
conservatively charged in every physical pool until exact attribution is available.
Instrumented speculative retention also remains unknown
in this phase; native state/residency facts remain available. Returned records or
external application copies retained outside these declared allowances are excluded.

Before prefill, terminal/cancelled/failed owners, retained proposals, optimistic
transactions and in-flight verification are explicitly unsupported. The method
does not make those states eligible by advancing execution. Embedded and
feature-conditioned prediction remain unsupported. User snapshots and inactive
branches are included from live non-rewindable reservations; restore and exchange
change the state observed by the next call without refunding those reservations.

Validation on 2026-09-23 used the same pinned SmolLM-135M revision below on
macOS/aarch64 Metal with allocator caching disabled. Original checkpoint files were
unchanged; a local `chat_template.jinja` supplied the literal content-only template
shown in the startup validation. Two independent copies ran 32-token requests with
proposal width four. Outlooks were taken after prefill committed the first token,
with one live user snapshot, for 31 further tokens.

| Prompt positions | Lookahead | Additional forecast upper, bytes | Subsequent MLX peak growth, bytes |
| ---: | --- | ---: | ---: |
| 10 | Disabled, greedy | 388,148,363 | 23,401,674 |
| 10 | Enabled, stochastic | 530,300,871 | 26,023,130 |
| 289 | Disabled, greedy | 1,468,286,819 | 99,036,825 |
| 289 | Enabled, stochastic | 1,996,132,275 | 114,372,220 |

All four returned `likely_fit`. Forecasts preserved tokens, RNG state, epochs and
snapshot budgets, and did not increase active native allocation. Snapshot restore
and branch exchange preserved subsequent token output; controlled and uninterrupted
speculative output matched exactly. In-flight calls were rejected. The native
nonzero Qwen2 fixture passed the same checks; portable conformance exercises
acceptance, rejection/replay and lookahead. These conservative bounds include
current state/capacity and snapshot allowances, so they deliberately exceed just
new allocation growth. Native separate physical pools and embedded prediction
were not validated by this phase.

Reproduce with the pinned checkpoint files and literal `chat_template.jinja` in
`MODEL` (omit the environment variables to use the nonzero native fixture):

```sh
EREDU_SPECULATIVE_CONTINUATION_MODEL="$MODEL" \
EREDU_SPECULATIVE_CONTINUATION_REPEAT=1 \
cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_speculative_continuation_forecasts -- --ignored --nocapture --test-threads=1
```

Use `EREDU_SPECULATIVE_CONTINUATION_REPEAT=32` for the longer prompt. Each run
checks serial greedy and separate-stream stochastic lookahead, zero-token outlooks,
read-only observation, snapshot/fork/exchange/restore and token parity. Structured
semantic parsers currently retain an unknown staged-event bound; plain text has a
finite bound derived from token and decoded-input geometry.

### Shared request calibration

`GenerationForecast` contains the estimate, descriptive request, full-pass reason
and logits contract. `with_prefill_chunk` recomputes a candidate only when the
request supports chunking; it preserves full-pass requirements. Loaded forecasts
exclude completed loading and deduct only declared resident parameter backing
from additional memory, never the process-global active allocator counter. Fresh
requests should start from reset state. The MLX adapter detects nonempty or
unavailable state and leaves a **fresh-request** forecast unbounded rather than
silently forecasting an empty cache. Use the continuation API for an active run. Prepared media, bounded transfers and unprojected
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

With the `mlx` feature, use the selected plan device to construct cold options:

```rust,ignore
let mut options = GenerationMemoryOptions::for_local_device(input, plan.device())?;
options.max_output_tokens = Some(32);
options.budget.application_limit_bytes = Some(2 * 1024 * 1024 * 1024);
options.budget.reserve_bytes = 256 * 1024 * 1024;
options.host_budget.reserve_bytes = 256 * 1024 * 1024;
let forecast = forecast_inspected_generation(&inspection, &options, &Default::default())?;
```

`for_local_device` validates the selected backend/device, observes hardware once,
and derives placement and availability. CPU execution uses host capacity; unified
accelerator execution uses the shared host observation; separate accelerators use
the matching device observation plus an independent host budget. Unknown physical
relationships and missing availability remain unknown. Installed capacity, another
device's free memory and automatic-planner budget fallbacks are never substituted.
Invalid or undiscovered selections return `GenerationForecastError::Capability`
with `CapabilityError::InvalidConfiguration`. Application limits and reserves
default to `None` and zero; already-resident bytes default to zero for cold models.
The CLI uses this constructor and then applies its application budget/reserve policy.

The backend-neutral `GenerationMemoryOptions::for_hardware_device(input,
hardware, device, host_execution)` performs the same mapping from supplied
observations. `host_execution` is a backend fact; portable code does not interpret
opaque device IDs or family strings. This pure constructor leaves native overhead
unknown and can be used without the `mlx` feature.

Validation covers exact device matching, CPU/unified/separate pools, missing and
zero availability, unknown physical relationships and invalid selections in seven
portable tests. Native policy tests verify the constructor does not change the
allocator policy. Cold CLI smoke tests on the pinned SmolLM checkpoint below, with
64 input positions, 32 output tokens and `--mlx-cache-limit-bytes 0`, returned
`likely_fit` without an application budget on both CPU (host) and Metal (unified).

`for_local_device` and the existing explicit-placement constructor
`GenerationMemoryOptions::for_local_backend(input, placement)` both sample the
current cache limit and retained bytes and default
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

Architecture preparation retains normalized state schedules and ordinary module
construction topology. Runtime planning combines those with selected residency,
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

Production target workspace forecasts use `TextExecutionTopology`, retained by
ordinary preparation and derived from the same projection, convolution, grouped
expert and output specifications used to construct execution. Selected parameter
formats replace checkpoint defaults after materialization selection. Families
supply ordinary equations and invocation facts; runtime owns memory calibration.

The shared runtime evaluator describes projection outputs, normalization and
residual intermediates, attention, gated convolution, dense or packed routed
feed-forward work, output softcapping, logits, sampling, cache replacement and
pending parameter conversions. It composes these resources through explicit
layer evaluation lifetimes. The default retains one set per layer plus 25%
overlap, rounded upward; an explicit smaller overlap releases older layer sets.
The lower end is a proven individual output allocation, rather than an assertion
that all layer temporaries coexist. The upper end remains a labeled planning
allowance, including scratch and promotion, not a native allocator guarantee.

Shared decoder construction covers Llama, dense Qwen, Nanbeige and Gemma2 without
family memory equations. Packed routed Qwen and LFM2 use the same generic expert
mechanism; LFM2 mixes the reusable attention and convolution descriptions. Missing
module equations, native input-score tile facts, rank-local invocation topology,
nondevice state and prediction/residency exclusions remain explicit unknowns.
Embedded prediction is still excluded pending its resource-description phase.
Ordinary cold, loaded, settled continuation and external-drafter forecasts consume
the same target evaluator. Legacy `WorkspaceGeometry` requests remain readable;
when ordinary topology is present it is authoritative.

Ordinary shared dense readout projects the final row per chunk; capture contracts
can require every row. Fused attention retains the configurable float32 score
matrix fallback. Explicit input-score attention instead consumes its selected
native tile/retention facts directly, without adding a second whole-score-matrix
fallback. A whole extra state payload covers cache replacement by default.
Unknown scratch or retention remains unbounded. New mechanism coverage is shared
by all families using those ordinary construction contracts; measurements below
are validation cases, not dispatch criteria.

### Earlier LFM2/LFM2.5 hybrid calibration

The following records the legacy calibration and its measurement history. The
generic migration below replaces its aggregate family geometry with ordinary
module outputs and lifetimes, retaining the native tile facts and conservative
conversion allowance.

Dense LFM2 selections projected the normalized feed-forward width, attention
schedule, convolution channels and kernel width for SafeTensors and GGUF.
Persistent convolution history remains in the state estimate. With batch `B`,
query positions `Q`, channels `C`, kernel width `K` and floating element size `S`,
one additional convolution set allows:

```text
B * C * (6*Q + 2*(Q+K-1) + Q*K + K) * max(S, 4)
```

This covers the three-way projection, two gate products, convolution output,
padded/history input copies, kernel-width unfolded scratch and a contiguous
kernel. The shared linear upper also permits float32 promotion. For either
operator, `min(copies, operator_layers + max(copies - total_layers, 0))` bounds
live sets using the existing layer-overlap calibration. The convolution lower
contribution stays zero; these are conservative planning allowances.

Explicit input-score attention needs more than a score-matrix allowance.
For complete key rows of at most 8,192 positions, MLX now shares expanded K/V
and contiguous BF16 projection layouts across query tiles in an invocation.
Large tiled calls evaluate batches of at most 32 query-tile outputs and retain the
completed outputs for final concatenation, releasing each batch's score/softmax
graphs before building the next batch. Small calls remain lazy. The backend
supplies its actual tile thresholds (8,192 query-by-key elements, at most 32 query
rows). For these full-key calls, the forecast charges four query-width K/V
payloads per invocation, at most 32 live tiles of score scratch at 32 bytes per
element (including a possible sink column), and two whole-query output payloads
for completed tiles and concatenation. Mixed-width execution promotes K/V and
output allowances to at least four bytes per scalar. These facts are
retained during selection, including cold inspection, and participate in cached
selection validation. Runtime adds their layer-overlap envelope to the ordinary
attention fallback. Above 8,192 key positions, attention still uses the existing
two-pass blockwise accumulator, which evaluates each block's running state and
releases temporary layouts. Retaining all prepared blocks would undermine that
bounded residency strategy; its FP32 accumulation also differs from full-key
BF16 projection. Those rows retain the conservative per-tile whole-context K/V
allowance with 32 working bytes per score element. Legacy serialized facts without
`full_key_tiles` retain their declared per-tile copy and score allowances.
Missing native facts leave
an unknown upper end. Attention-free convolution schedules need no attention
scratch declaration.

This coverage applies to ordinary cold, loaded and settled continuation
forecasts. LFM2 still requires full-pass prefill: reducing a requested chunk size
does not reduce its estimate. The generic migration adds packed routed LFM2
coverage; distributed projections, nondevice state and existing
prediction/residency exclusions retain their explicit gaps.
Mixed-width selected parameters add a separate float32 cast allowance, including
promoted state and replacement storage above the nominal state estimate. The
architecture computes this from selected logical parameter shapes and dtypes;
no file-format name selects the allowance. It is especially relevant to the
published BF16 GGUF, whose small convolution/norm tensors include float32.

The new optional `WorkspaceGeometry` JSON fields `gated_convolution`,
`input_score_attention` and `mixed_precision_parameter_bytes` are omitted for
ordinary dense attention records; older records deserialize with them absent.
Rust struct literals must initialize them.

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

An earlier run of cached official `LiquidAI/LFM2.5-1.2B-Instruct` revision
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`, with 128 input positions and eight
decode tokens, completed successfully while reporting an unavailable workspace
upper bound. Its recurrent workspace was not treated as zero and ordinary
generation remained available. This is coverage of the uncertainty path, not
calibration of hybrid-model peak estimates. The new hybrid calibration below
supersedes that workspace gap.

Native distributed placement, media workspace and other model/hardware
calibrations remain open. CUDA validation was explicitly deferred. The native
fixture checks and portable domain tests do not establish those native results.

## Hybrid calibration (2026-09-23)

Native tests used macOS/aarch64, Metal and the same 256 GiB unified-memory host.
The cached official `LiquidAI/LFM2.5-1.2B-Instruct` revision was
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`. Its 2,340,697,936-byte
`model.safetensors` SHA-256 was verified as
`1ba63d9adb03ae43581db0e136e4416febe0441aff7296397bd455fb6017f73a`.
The model has 16 layers (10 convolution, six attention), hidden width 2,048,
normalized MLP width 8,192, 32 query heads, eight KV heads and kernel width three.
Selected state storage is BF16. A nonzero float32 fixture additionally covers
one-token, 17-position and 39-position prompts.

Each run disables allocator caching, requests eight greedy tokens and a
four-token prefill chunk (correctly normalized to the full prompt). After four
committed predictions it forecasts four more tokens at a synchronized boundary.
It compares cold/loaded geometry, ordinary/controlled forecasts and output tokens,
and verifies that forecasts allocate no active native bytes. The table records
additional active-memory growth above loaded parameters, not process RSS. The
256 GiB application budget makes these calibration cases observable. Before
recalibration, a separate 16 GiB-budget check rejected `LikelyFit` for the
2,000-position request; updated results appear below.

The following measurements precede shared K/V preparation; updated measurements
appear below. All values are MiB; upper ends are calibrated envelopes, not tight
projections.

| Source | Positions | Cold lifecycle upper | Loaded additional upper | Measured growth | Continuation upper | Continuation growth |
|---|---:|---:|---:|---:|---:|---:|
| SafeTensors BF16 | 128 | 5296.5 | 786.9 | 546.0 | 124.3 | 13.9 |
| SafeTensors BF16 | 2000 | 187728.0 | 185495.7 | 67375.7 | 865.1 | 159.2 |
| SafeTensors affine 4-bit / BF16 | 128 | 3876.2 | 786.9 | 572.5 | 124.3 | 13.1 |
| SafeTensors affine 4-bit / BF16 | 2000 | 186307.7 | 185495.7 | 67485.1 | 865.1 | 159.5 |
| GGUF BF16/mixed | 128 | 8686.2 | 6453.7 | 4960.2 | 5738.6 | 4483.7 |
| GGUF BF16/mixed | 2000 | 349607.0 | 347374.5 | 110485.0 | 6901.7 | 4705.7 |

The GGUF run used official `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` revision
`6767265158422fb8a19c62ceb45f16f05363615b`, file
`LFM2.5-1.2B-Instruct-BF16.gguf` (2,343,326,528 bytes), with verified SHA-256
`3d80914b903cd6f3cc041208cf20ec46a3224f840c732e5fd7698832b4743d1b`.
Pass that file as `EREDU_LFM2_MEMORY_MODEL` to reproduce. At this calibration stage,
its 128-position fresh/continuation forecasts reported `LikelyFit`, while the
2,000-position fresh forecast remained `InsufficientInformation` because its
finite interval crossed host/application capacity. That was interval uncertainty rather than
missing workspace coverage. Parameter casts explain much of the GGUF continuation
cost; the broader envelope intentionally covers both promoted storage and casts.

The first convolution-only extension underpredicted the SafeTensors cases. Before
shared K/V preparation, the long case retained about 66 GiB of extra active allocations through explicit
input-score attention. Query tiling does not imply bounded whole-graph retention.
The native-copy envelope covers that measured behavior conservatively.

Reproduce with a cached immutable snapshot (no downloads or cache edits occur):

```sh
EREDU_LFM2_MEMORY_MODEL=/path/to/LFM2.5-1.2B-Instruct/snapshot \
EREDU_LFM2_MEMORY_LENGTHS=128,2000 \
cargo test -p eredu --features mlx,metal --offline \
  --test native_execution_control native_lfm2_workspace_forecasts \
  -- --ignored --nocapture --test-threads=1
```

Without environment variables the same test uses the small float32 fixture.
The optional `EREDU_LFM2_MEMORY_QUANTIZED=1` requests affine 4-bit weights with
64-value groups. The pinned BF16 SafeTensors checkpoint now passes the same
128/2,000-position matrix, including continuation and controlled-session parity.
The earlier failure compared an admitted BF16 recipe with an unloaded Float32
source slot. Exact-task quantization now retains the recipe's precision and checks
only the source slot's shape and floating category. Generated affine scales and
biases remain BF16; quantization does not reduce activation or attention scratch
precision. A nonzero native CPU regression covers F16, BF16 and F32 sources,
compares packed weights/scales/biases with direct native quantization, and preserves
rejections for incompatible source shapes and integer slots.

### Shared K/V preparation validation (2026-09-23)

Repeating the same pinned models, commands and 128/2,000-position matrix after
sharing prepared K/V across query tiles produced these active-memory measurements
(MiB). At that stage, forecast upper ends remained unchanged from the preceding table.

| Source | Positions | Measured growth | Continuation growth |
|---|---:|---:|---:|
| SafeTensors BF16 | 128 | 525.0 | 15.5 |
| SafeTensors BF16 | 2000 | 20641.3 | 175.1 |
| SafeTensors affine 4-bit / BF16 | 128 | 552.5 | 15.5 |
| SafeTensors affine 4-bit / BF16 | 2000 | 20641.3 | 175.1 |
| GGUF BF16/mixed | 128 | 4924.2 | 4483.7 |
| GGUF BF16/mixed | 2000 | 16922.5 | 4705.7 |

Long-prompt SafeTensors growth fell from 65.8 GiB to 20.2 GiB (about 69%).
Continuation measurements did not improve: SafeTensors growth increased modestly,
while GGUF remained unchanged. All runs passed cold/loaded forecast checks,
continuation bounds and controlled-session output parity. These observations do
not establish a throughput improvement or bounded whole-graph retention.

An isolated Metal allocation regression compares shared preparation with separate
preparation for every query tile: 22,683,712 versus 69,869,632 peak bytes, with
bit-identical output. Numerical coverage also includes an independent PyTorch
fixture on CPU and Metal, F32/F16/BF16, grouped heads, multiple batches, unequal
K/V widths, boolean/additive masks, sinks, softcaps, partial query tiles and both
sides of the 8,192-key threshold. That change added no tile evaluation barriers;
the subsequent bounded-batch change is measured separately below.

### Bounded tile-graph validation (2026-09-23)

The follow-up bounds live full-key tile graphs by evaluating 32-tile batches and
keeping their completed output arrays for final concatenation. Shared prepared
K/V stays resident across batches. Calls that fit within one batch remain lazy; the
longer-than-8,192-key blockwise path is unchanged. This limits temporary graph
retention, not total request memory: inputs, K/V, completed outputs and other
model state still grow with request geometry. These measurements preceded the
forecast recalibration below; their envelopes were unchanged and conservative.

Measurements used an Apple M3 Ultra with 256 GiB unified memory, Metal, the default
Cargo test profile and allocator cache limit zero. The isolated BF16 benchmark
uses head width 64, four query heads per KV head, 2,000 key positions, a boolean
mask, sink logits and softcap. Each configuration has two warmups and five
synchronized samples; timing includes graph construction, evaluation and stream
completion. Peak growth is active allocator bytes above already evaluated inputs,
not process RSS. All batch sizes produce bit-identical outputs to fully lazy
shared-K/V execution.

| Queries / heads | Live tile batch | Peak growth (MiB) | Median time (ms) |
|---|---:|---:|---:|
| 513 / 16 | Fully lazy | 463.6 | 125.9 |
| 513 / 16 | 1 | 16.2 | 272.8 |
| 513 / 16 | 8 | 40.8 | 221.8 |
| 513 / 16 | 16 | 68.9 | 149.5 |
| 513 / 16 | 32 | 125.0 | 148.6 |
| 2000 / 32 | Fully lazy | 3507.9 | 591.7 |
| 2000 / 32 | 1 | 38.2 | 1742.8 |
| 2000 / 32 | 8 | 86.7 | 809.8 |
| 2000 / 32 | 16 | 142.2 | 731.6 |
| 2000 / 32 | 32 | 253.0 | 609.9 |

The selected 32-tile batch reduces peak growth by 92.8% with 3.1% more elapsed
time for the larger case. For the smaller case it reduces peak growth by 73.0%
with 18.0% more elapsed time. Smaller batches save more memory but incur more
synchronization overhead.

These numbers expose the synchronization cost rather than assuming a throughput
win. Timing depends on device, geometry, cache policy and host build profile.
There is no timing assertion in the regression; it checks exact output parity
and a broad reduction in peak active memory. Reproduce the sweep with:

```sh
cargo test -p eredu-backend-mlx --features metal --offline --lib \
  bounded_input_score_tile_graph_batches_release_temporaries \
  -- --ignored --nocapture --test-threads=1
```

The pinned official SafeTensors BF16 model above was measured against
`2776e4c5` (shared K/V, fully lazy tile graphs), using the same timing instrumentation
in both binaries. For each prompt length, one warmup preceded three measured runs.
Each run resets the model, supplies repeated token ID 1 and generates eight greedy
tokens. The timing excludes loading, forecasting and the separate controlled
parity run; total generation time includes synchronization after four and eight
tokens. The first-token timer ends when the iterator returns the materialized
prediction. All values below are warmed medians except peak active-memory growth.

| Positions | Tile evaluation | Peak growth (MiB) | First token (ms) | Eight tokens (ms) |
|---|---|---:|---:|---:|
| 128 | Fully lazy | 525.0 | 308.0 | 509.3 |
| 128 | Batches of 32 | 525.0 | 307.3 | 503.5 |
| 2000 | Fully lazy | 20641.3 | 7700.9 | 7951.7 |
| 2000 | Batches of 32 | 1516.7 | 7895.3 | 8149.6 |

At 2,000 positions, peak growth fell 92.7% (20.16 to 1.48 GiB), with 2.5% longer
median generation time. Effective prompt throughput through the first prediction
was 259.7 versus 253.3 tokens/s. The measured eight-token ranges were
7,947–8,035 ms before and 7,916–8,269 ms after; this short local experiment is not a
cross-device performance guarantee. The 128-position case stays below the batching
threshold, and its memory measurement is unchanged. Reproduce the full-model
timing matrix with the preceding calibration command and
`EREDU_LFM2_MEMORY_LENGTHS=128,2000,128,2000,128,2000,128,2000`.

The same 128/2,000-position validation passed for the pinned affine 4-bit
SafeTensors and mixed-precision BF16 GGUF paths. Their 2,000-position peak growth
was 1,516.7 and 4,908.4 MiB respectively, versus 20,641.3 and 16,922.5 MiB before
batching. Continuation growth remained 175.1 and 4,705.7 MiB; GGUF parameter casts
still dominate that continuation cost. Every source passed ordinary/controlled
output parity and cold, loaded and continuation forecast checks. CPU and Metal
F32/F16/BF16 numerical tests cross the 32-tile boundary with grouped heads,
multiple batches, unequal K/V widths, masks, sinks, softcaps and a partial final
group. The backend's Metal Clippy and portable feature check also pass.

### Forecast recalibration after tile retention improvements (2026-09-23)

The native retention facts now distinguish full-key reuse/batching from legacy
per-tile retention. For a batch size `B`, query width `W`, query-head count `H`,
key positions `K`, query positions `Q` and scalar width `S`, the full-key extra
allowance per live layer is:

```text
4 * B * W * K * S                         shared K/V and contiguous layouts
+ 32 * B * H * min(Q, tile_rows * 32) * (K + 1)  live score scratch, including a sink
+ 2 * B * W * Q * S                       completed outputs and concatenation
```

`tile_rows` follows the native 8,192-element/32-query-row policy. Mixed-width
execution uses at least four bytes for `S`. Existing layer overlap, score-matrix
fallback, convolution, parameter-cast and graph allowances remain in place. These
are upper planning envelopes, not expected allocations. Above 8,192 keys the
old per-tile whole-context envelope is retained. Old serialized mechanism records
without `full_key_tiles` keep their declared copy/score allowances; an absent
mechanism still leaves the workspace upper unknown.

The isolated softcap/mask/sink benchmark showed that 16 working bytes per score
element was insufficient on its own. The new 32-byte allowance covers that
measured per-invocation peak without depending on model-wide overlap to hide the
difference. The regression now checks the published bound against active allocator
telemetry. This calibration changes no attention arithmetic or BF16 rounding.

The same pinned checkpoints, Metal host, zero allocator cache and eight-token
128/2,000-position matrix produced the following results (MiB):

| Source | Positions | Cold lifecycle upper | Loaded additional upper | Measured growth | Continuation upper | Continuation growth |
|---|---:|---:|---:|---:|---:|---:|
| SafeTensors BF16 | 128 | 5296.5 | 818.2 | 525.0 | 125.0 | 15.5 |
| SafeTensors BF16 | 2000 | 14916.7 | 12684.5 | 1516.7 | 875.0 | 175.1 |
| SafeTensors affine 4-bit / BF16 | 128 | 3876.2 | 818.2 | 552.5 | 125.0 | 15.5 |
| SafeTensors affine 4-bit / BF16 | 2000 | 13496.5 | 12684.5 | 1516.7 | 875.0 | 175.1 |
| GGUF BF16/mixed | 128 | 8667.4 | 6434.9 | 4924.2 | 5739.4 | 4483.7 |
| GGUF BF16/mixed | 2000 | 21014.5 | 18782.0 | 4908.4 | 6911.7 | 4705.7 |

Every measured peak remains below its forecast upper. At 2,000 positions the
SafeTensors loaded additional upper falls from 181.15 GiB to 12.39 GiB; both
original and affine 4-bit runs now report `LikelyFit` with a 16 GiB application
budget. The GGUF run still reports `InsufficientInformation` at that budget: its
remaining conservative cast/activation envelope crosses capacity. Small or
single-query upper ends can rise slightly because the corrected score allowance
also applies there. Budgets below the lower bound still report `LikelyShortfall`.

The native allowance regression also covers F16 and F32 short prefill, 2,000-row
batched prefill and single-row cached decode. For the long case, measured peaks
were 256,512,512 and 147,587,584 bytes against allowances of 311,427,072 and
360,579,072 bytes respectively. These isolated checks include softcap, boolean
mask and sink logits. They validate the per-invocation facts independently of
the model's layer-overlap allowance.

Full-model numerical validation compares revision `1aa87493` with the recalibrated
implementation using the same native test harness and pinned BF16 checkpoint.
For both 128 and 2,000 input positions, every one of the 65,536 logits at each
of eight predictions matches bit-for-bit: prefill plus seven cached decode
steps. All eight greedy tokens match raw generation, and explicit controlled
stepping matches ordinary observed execution at every prediction. The comparison
stores float32 host-logit bit patterns rather than decimal float approximations.
The small F32 fixture additionally covers 17/513 positions. This is regression
parity against the previous implementation; independent PyTorch attention
fixtures continue to validate the BF16 rounding contract.

To reproduce the before/after comparison, build the previous revision with the
same `native_lfm2_forecast_recalibration_preserves_logits` test harness, then run
it with `EREDU_LFM2_PARITY_WRITE=/tmp/lfm2-parity.json`. Run the current revision
with `EREDU_LFM2_PARITY_REFERENCE=/tmp/lfm2-parity.json`:

```sh
EREDU_LFM2_MEMORY_MODEL=/path/to/pinned/LFM2.5-1.2B-Instruct/snapshot \
EREDU_LFM2_MEMORY_LENGTHS=128,2000 \
EREDU_LFM2_PARITY_REFERENCE=/tmp/lfm2-parity.json \
cargo test -p eredu --features mlx,metal --offline \
  --test native_execution_control native_lfm2_forecast_recalibration_preserves_logits \
  -- --ignored --nocapture --test-threads=1
```

Without a reference-file variable the test still checks raw/observed/controlled
token parity and exact prefill/cached logits across ordinary and controlled runs.
Native CUDA calibration remains unvalidated on this Metal host.

Portable verification passed 28 memory-estimation tests, cold-selection cache
invalidation, 96 backend-conformance tests and 27 portable-facade tests (one
preexisting ignored case). Strict Clippy passed for the portable contracts,
architectures, backend and native facade test harness.

### Resident parameter conversion reuse (2026-09-23)

Fully resident MLX dense projections now retain evaluated F32 conversions when
F32 activations meet F16/BF16 parameters. This covers the mixed-width GGUF case
without changing BF16 arithmetic or introducing casts for narrow activations.
Tied aliases share a conversion. Reset preserves it; parameter publication
revokes it, and dropping the resident owner releases it. Restoring a replaced
parameter uses temporary casts until a new resident materialization enables reuse.
Host-layerwise and disk-streamed policies keep temporary casts within their
existing admission policy. An explicit device-residency ceiling also disables
conversion retention, so derived storage cannot bypass that ceiling.

`StaticMemoryReport.current_device_parameter_conversion_bytes` is the exact
retained subset already included in `current_device_resident_bytes`. Do not add
it again. Logical parameter sizes and bounded-residency ledger/telemetry remain
the original parameter bytes. Resident session reports query their live owner,
so later forecasts observe conversions created since loading. Loaded and
continuation forecasts remove retained conversions from the pending cast allowance
while preserving promoted activation/state sizing. Cold forecasts still reserve
the full potential conversion payload; mixed-width detection now applies to all
covered dense geometries. F32 activation and projection-logit upper allowances
remain in force even when every cast is already resident; nominal lower bounds
are unchanged. Neither inspection nor forecasting populates the cache.

Using the same pinned BF16/mixed GGUF, Metal host, zero allocator cache and eight
greedy output tokens as above, the first 128-position request retained exactly
4,680,843,264 bytes (4,464.0 MiB). That count remained unchanged after reset and
repeated 128/2,000-position requests. Measured incremental peaks and forecast
upper bounds were (MiB):

| Request | Loaded additional upper | Measured growth | Continuation upper | Continuation growth |
|---|---:|---:|---:|---:|
| 128, cache initially empty | 6450.9 | 4944.2 | 159.6 | 19.7 |
| 128, conversions resident | 870.9 | 480.2 | 159.6 | 19.7 |
| 2000, conversions resident | 13452.0 | 1332.1 | 1331.8 | 241.7 |

The earlier 2,000-position continuation required 4,705.7 MiB of additional peak
memory. Reuse trades those repeated casts for resident storage; smaller additional
peaks do not imply lower total model memory. Cold lifecycle upper bounds are
8,683.4/21,264.5 MiB for 128/2,000 positions. The conservative 2,000-position
forecast still crosses a 16 GiB application budget. Warm eight-token timings
were 131.2 ms at 128 positions and 1,079–1,116 ms at 2,000; these are local
observations, not a cross-device throughput guarantee.

Reproduce the memory checks with the preceding native forecast command, the
pinned GGUF path, `EREDU_LFM2_MEMORY_LENGTHS=128,2000,128,2000` and
`EREDU_LFM2_MEMORY_EXPECT_PARAMETER_CONVERSIONS=1`. The latter asserts that the
public report actually contains the retained conversions and that warm requests
reuse them. Tests also check unchanged logical bytes, exact device residency
growth, reset retention, ordinary/controlled forecasts and measured peaks.

Against revision `a43a3171`, all 65,536 logits at each of eight predictions match
bit-for-bit for both prompt lengths, including prefill, seven cached decode
steps, raw tokens and ordinary/controlled execution. The same parity command
above accepts this GGUF path and a reference exported from that revision.
Focused native tests cover F16/BF16 promotions, alias reuse, narrow-input bypass,
read-only parameter inspection, publication/restoration invalidation and owner
release, and explicit device ceilings. The original BF16 SafeTensors checkpoint
also passed the 128/2,000-position numerical, token, controlled-session and memory
checks, retaining zero conversions and unchanged measured memory. Portable tests cover cold mixed-width Llama geometry, loaded forecast
deductions and old JSON reports without the new observation.

## Focused verification

- Cold chunking follow-up: portable preparation tests cover supported and absent
  backend mechanisms, Llama, LFM2, routed, composite, prediction and partitioned
  selections. CPU/Metal fixture tests compare cold support with the loaded executor
  across resident, host, disk and affine-quantized paths. Rebuilt CLI cold reports
  on the pinned SmolLM and LFM2.5 checkpoints above used 2,000 input positions,
  eight output positions, a requested 512-token chunk and zero allocator cache.
  SmolLM retained 512 and recomputed smaller-chunk savings; LFM2.5 used all 2,000
  positions, reported the architecture's full-pass reason and suggested no chunk
  savings. At that time the workspace upper was unknown; the hybrid calibration
  now supplies the missing coverage.
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


## Mid-session continuation forecasts

At a settled ordinary decode boundary, both `ControlledGenerationSession` and
`TextGeneration` expose:

```rust,ignore
let forecast = session.forecast_remaining_generation(128, &options)?;
```

A controlled `step` settles the prediction and delivers its captures before
returning. For an asynchronous token iterator, call `tokens.synchronize()?`
first to settle retained submissions without consuming the next token. Forecasting
itself never waits, submits, snapshots, advances randomness, charges capture or
transport budgets, or changes the configured generation limit. The horizon is a
what-if allowance and may exceed the remaining configured tokens. Request a new
forecast after advancing, restoring, exchanging a branch, or changing the horizon.

`ContinuationForecast` records the actual cached position and the additional
pending inputs, plus a reproducible `ContinuationMemoryPlan`. The last emitted
token is still pending decode: after prefill of P positions, N further predictions
reach P + N cached positions. Phases are `continuation_start` and, for N > 0,
`decode` with one query position. Completed loading and prefill are excluded.
Zero predicts retained state without another model invocation.

Continuation plans report a logical lower bound for installed state and for the
horizon peak. The bound uses required architecture-declared tensors at the actual
cache frontier, including sliding windows, per-layer prefix offsets and fixed-dtype
recurrent state. It excludes optional tensors and allocation rounding. The peak
floor includes the starting state even when a remainder-shaped tensor shrinks at
the endpoint; zero-token forecasts still report the installed payload. Layouts
marked only as conservative upper estimates do not establish a new lower bound.
Known payload floors survive an unavailable native upper bound. These logical
bytes are not measured distinct backing and grant no already-resident credit.

MLX combines installed native state storage with its existing continuation-growth
mechanism. That mechanism covers capacity rounding, sliding-cache retention and
interior peaks of remainder-shaped state. These are conservative logical storage
allowances, not measured distinct backing. The same bound covers possible old and
replacement cache overlap. Sampler/history/pending-input retention, admitted
capture geometry (or explicit capture/intervention limit fallbacks) and the controlled facade's decoder, constraints,
semantic records and trace are included. Live snapshots and branches contribute
their reserved retention upper allowance, including reserved future growth.
Unattributed host/native allowances are conservatively included in each separate
physical pool. Unknown grammar, native growth, workspace or other storage keeps
its upper end unknown.

Semantic history is bounded by its actual wire contract. Delivery charges a
complete JSON record before retaining its semantic event; the record's fixed
fields cover the event header, and JSON covers its string payload. Current
history is counted directly. Future logical history is bounded by
`trace_limit - emitted_bytes`, without multiplying bytes by an event-header size.
Restore does not refund this allowance. Branch reservations use the fresh child's
trace limit once; inherited history is already covered by the source reservation.
A separate encoded-trace allowance remains for one consumer-retained JSON copy.
With a 64 MiB trace budget, future semantic history therefore contributes at most
64 MiB, plus the separately modeled encoded trace and other retained resources.
These are logical storage bounds, excluding vector spare capacity and allocator
overhead; they do not assert a physical process-memory ceiling.

Only declared resident parameter backing is deducted when comparing additional
memory with current available capacity. State, snapshot and branch allowances are
not credited as observed allocations. This can overestimate incremental cost,
especially with many large snapshots, but does not turn unused reservations into
free capacity. Immutable shared tokenizer data, arbitrary consumer copies,
external decoder state for raw iterators, unrelated sessions and total process
memory remain outside the modeled request; reserve capacity for them.

Initial prefill, failed/cancelled/terminal controlled sessions and unsettled
submissions are rejected. Selected executables without an installed-state
projection return `GenerationForecastError::UnsupportedContinuation`. Mid-flight
speculative transactions are not ordinary continuations. Settled external
autoregressive lanes use the separate speculative continuation plan above; the
existing prepared speculative forecast still covers fresh speculative runs.


Initial continuation validation (2026-09-23, before tightening the semantic-history
allowance below): portable tests cover pending-token offsets,
zero and extended horizons, overflow/shortfall/unknown growth, interior remainder
peaks, unchanged output/copy/trace budgets, restore, branch exchange and failed
iterator settlement. Native Metal validation used the nonzero Qwen2 dense and
mixed sliding-window fixtures (`sliding_window = 2`), each with a 33-position
prefill, a saved continuation, and 16 further forced ordinary tokens. Additional
upper allowances were 339,639,607 and 339,627,319 bytes; measured incremental native
active peaks were 16,854 and 7,636 bytes. The Gemma2 fixture confirmed bounded
native state while preserving its independently unmodeled workspace.

The pinned original-weight SmolLM-135M checkpoint at revision
`1d461723eec654e65efdc40cf49301c89c0c92f4` also passed on Metal: four prompt
positions, 16 further forced ordinary tokens, restore, and an ordinary iterator
forecast after explicit settlement. The additional upper was 352,293,335 bytes;
measured incremental native active peak was 3,102,774 bytes. Default allocator and
graph allowances dominate these small runs; these observations do not calibrate
all models or make the logical envelopes allocation guarantees. The local
`/tmp/eredu-spec-validation/model` copy uses the same original weights and a
literal content-only chat template added to tokenizer metadata. Reproduce with:

```sh
cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_continuation_forecasts -- --ignored --nocapture --test-threads=1
EREDU_CONTINUATION_MODEL=/tmp/eredu-spec-validation/model \
  cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_continuation_forecasts -- --ignored --nocapture --test-threads=1
```

The same fixture test can run without Metal using `--no-default-features
--features mlx` and omitting `--ignored`; this change was validated natively on
Metal, not CUDA or a CPU-only MLX build.


Semantic-history regression validation uses a 64 MiB trace budget, checks the
logical-storage/charged-record relationship for every semantic event variant
(including empty, escaped, Unicode and large string payloads), and exercises
capture-enabled forecasts, restore and branch exchange. Doubling the budget adds
one budget's worth to future semantic history and one to the separately retained
encoded trace. Branch reservations grow only once with the fresh child budget.


The updated native test uses a 64 MiB trace budget. On the same pinned original
SmolLM-135M checkpoint, four prompt positions and 16 further forced tokens produced
a retained upper of 136,357,159 bytes (about 130 MiB) and an additional generation
upper of 482,707,361 bytes. The measured incremental native active peak remained
3,102,774 bytes. Forecasting changed neither active native allocation nor snapshot
usage. The earlier calibration above used a 64 KiB trace budget and the older
semantic-history multiplier; use the commands above to reproduce the current
64 MiB regression case.


## Reusable mechanism descriptions (phase 4)

Low-level consumers can obtain `eredu_nn::mechanism_memory::MechanismInvocation`
from ordinary projection/convolution/grouped-linear specs or actual attention and
recurrent inputs. Runtime sampling producers derive it from the resolved config
or existing sampler state. Backend `mechanism_memory` hooks then describe known
storage, and `eredu_runtime::describe_mechanism_resources` binds owner identities
and physical pools into the neutral resource format. Querying these records does
not allocate tensors, evaluate lazy graphs, advance caches or consume budgets.

Coverage includes dense/packed projection geometry; explicit/tiled attention;
causal convolution output/history; FP32 recurrent state; selected expert routes;
cache append/capacity/window facts; and standard/Mirostat sampling policy.
MLX's optional selected-device query additionally distinguishes direct GGUF Metal
kernels from host row-wise decoding, and FP8 activation/scales from CPU fallback
dequantization. Cache-instance queries validate installed geometry, preserve the
frontier, and describe actual reserved capacity and backing reuse. Parameter
conversions with unknown residency ownership do not become fresh allocations.

These descriptions are inputs for generic lifetime composition, not a new
bounded forecast. Logical tensor bytes and physical allocation capacity are
separate: views/no-op casts can alias, lazy intermediate records are not a live
peak, and query/key tiles can overlap until evaluation. Opaque MLX kernel scratch,
allocator capacity, unresolved owner identity and unsupported native selection
facts remain named gaps. Existing generation/continuation forecasts and wire
records are unchanged in this phase; no previously unsupported family receives a
verdict solely because its reusable mechanisms now expose descriptions.

## Generic lifetime composition (phase 5)

`eredu_runtime::resource_lifetimes::compose_resource_peaks` accepts a
`ResourceLifetimePlan` containing acquisitions and explicit native-completion,
evaluation and owner-release events. For example, two independent allocations
of 10 and 20 MiB peak at 20 MiB when the first is released before the second is
acquired, or 30 MiB when their lifetimes overlap. Two aliases of one allocation
count once, and that backing survives until every declared reference ends.
Separate physical pools receive separate peak reports; their maxima need not
occur simultaneously.

`describe_mechanism_lifetimes` binds an existing mechanism description to those
release boundaries. Native completion does not release a lazy graph's retained
buffers; its evaluation boundary must complete. Returned tensors and parameter
conversions use explicit owner lifetimes. The bridge does not assume that all
intermediates within a mechanism coexist. An explicit schedule can assert known
coexistence through `live_at_acquire`.

Plans must declare their coverage. Unknown coverage, placement or retention
prevents a finite complete peak. Unknown capacity remains separate from known
payload, and independent resources' horizon maxima do not become a simultaneous
lower bound. Composition reads descriptions only; it does not run inference,
consume budgets or issue a fit verdict. Existing forecast APIs remain unchanged
until their workspace projections are migrated in phase 6.


### Generic target-workspace migration (2026-09-23)

Phase 6 replaces production family workspace formulas with ordinary construction
specifications and reusable mechanism/lifetime composition. The previous CLI
baseline was revision `2bde9278`. Validation used the same pinned checkpoints,
Metal device, greedy eight-token requests and disabled allocator cache before
and after the migration. SmolLM-135M used HuggingFace revision
`1d461723eec654e65efdc40cf49301c89c0c92f4`; LFM2.5-1.2B-Instruct used revision
`0f604ada3f766f9f257460c4c9f0b5d6f69d431b`. Load-time 4-bit runs used the CLI's
affine transformation. Prompts were repetitions of `" hello"`; the table reports
the resolved model-position count.

The upper bounds below are total modeled generation memory in MiB; measured
peaks are MLX active-allocation high-water marks. They exclude allocator cache
here, and are not total process or unified-memory pressure. The 64 MiB graph
allowance remains part of each forecast. Source checkpoint loading is excluded
from this generation comparison.

| Model / weights | Positions | Chunk | Previous upper MiB | Generic upper MiB | Measured peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM F32 | 32 | 0 | 616.9 | 628.6 | 577.1 |
| SmolLM F32 | 32 | 512 | 616.9 | 628.6 | 577.9 |
| SmolLM F32 | 2000 | 0 | 3310.2 | 4042.6 | 2633.8 |
| SmolLM F32 | 2000 | 512 | 1350.8 | 1538.3 | 1283.2 |
| SmolLM 4-bit | 32 | 0 | 275.2 | 286.9 | 250.0 |
| SmolLM 4-bit | 32 | 512 | 275.2 | 286.9 | 246.2 |
| SmolLM 4-bit | 2000 | 0 | 2968.5 | 3700.9 | 2296.6 |
| SmolLM 4-bit | 2000 | 512 | 1009.0 | 1196.6 | 866.5 |
| LFM2.5 BF16 | 129 | 0 | 3057.5 | 3088.5 | 2768.3 |
| LFM2.5 BF16 | 2001 | 0 | 14923.5 | 14492.9 | 3750.3 |
| LFM2.5 4-bit | 129 | 0 | 1637.3 | 1668.3 | 1353.7 |
| LFM2.5 4-bit | 2001 | 0 | 13503.3 | 13072.7 | 2330.1 |

SmolLM's upper bounds increase because the generic description explicitly includes
normalization, rotary and output allocations instead of the earlier aggregate
layer formula. LFM2's long-prompt bound falls by replacing the extra full score
matrix fallback with the selected explicit-score tile envelope; its smaller
cases can rise. This is a migration of resource accounting, not a tighter-bound
guarantee. All twelve measured peaks remained within the new planning upper
bounds, and generated output matched byte-for-byte across the migration.

Reproduce the matrix after building each revision, retaining the old binary:

```sh
cargo build -p eredu-cli --no-default-features --features mlx,metal --locked
python3 validation/memory_estimation_matrix.py \
  --model /path/to/pinned/SmolLM-135M --binary /path/to/eredu \
  --output /tmp/smollm-matrix --prompt-lengths 32 2000 --chunks 0 512 \
  --quantizations original 4
python3 validation/memory_estimation_matrix.py \
  --model /path/to/pinned/LFM2.5-1.2B-Instruct --binary /path/to/eredu \
  --output /tmp/lfm-matrix --prompt-lengths 128 2000 --chunks 0 \
  --quantizations original 4
```

Native LFM2.5 checks at 128 and 2,000 positions validated cold, loaded and
settled continuation forecasts, ordinary/controlled forecast agreement and
unchanged allocator activity during forecast calls. The pinned BF16 GGUF above
was also exercised at 128, 2,000 and 128 positions in one loaded model: retained
F32 conversions were credited once, reused after reset, and every measured
startup/continuation growth stayed inside the forecast upper. The warm 128-row
run retained 4,680,843,264 conversion bytes and forecast 928,753,920 additional
bytes against 503,571,756 measured bytes.

Before/after logit files compare every vocabulary value as an F32 bit pattern at
eight predictions (prefill plus seven cached steps), for 17 and 513 input positions
in both BF16 and affine 4-bit. All logits and token IDs match exactly. Ordinary
observed and explicitly stepped controlled runs also match each other. Use the
parity command above with `EREDU_LFM2_MEMORY_LENGTHS=17,513`; set
`EREDU_LFM2_MEMORY_QUANTIZED=1` for the 4-bit comparison. The reference is the
pre-migration native harness, so this is regression parity, not an independent
model-quality validation.

`native_generic_workspace_covers_softcapped_and_routed_modules` exercises nonzero
Gemma2 and Qwen3-MoE fixtures through startup and settled continuation. Measured
startup growth was 668,709 and 233,113 bytes, respectively; continuation growth
was 51,394 and 11,734 bytes. Both stay inside their planning allowances, and
forecast calls preserve native allocation counters. These small fixtures include
the default 64 MiB overhead allowance and do not independently calibrate it.

Portable validation passed the runtime, architecture and facade library suites,
97 backend-conformance tests, 27 portable-facade tests, construction/topology
consistency and strict portable/native Clippy. Tests cover retained selected
quantization, tied ownership, custom-selector unknowns, malformed geometry,
checked arithmetic, explicit evaluation overlap, zero-query continuation starts,
speculative sampling vocabulary and legacy JSON round trips.

Native CUDA, distributed invocation topology, other checkpoint scales and arbitrary
mechanism combinations were not measured on this Metal host. Packed expert
coverage has synthetic native validation; it has no released large-MoE calibration
claim. Opaque native capacity facts remain unknown in the raw mechanism contract;
finite forecast allowances are explicitly estimated.
