# Generation memory forecasts

Memory forecasts help compare a specific request with an application budget and
current available capacity. They are planning estimates, not allocation limits or
a guarantee about total process memory. Ordinary generation does not depend on
estimation being available.

## Controlling parameter-conversion retention

Eligible fully resident MLX executions retain optional F32 weight conversions
under a **256 MiB managed default**. Use
`ExecutionPlan::with_parameter_conversion_retention` at load time to select
`Bounded { max_bytes }`, `Disabled`, or explicitly `Unlimited`; `None` selects
the managed default and a zero-byte bound normalizes to disabled. Admission is
first-admitted, with no automatic eviction. Eligible single-row Metal projections
read materialized row-major F16/BF16 weights directly with F32 activations,
accumulation and output, avoiding full-weight casts and conversion admission.
This covers dense linear and tied-embedding output projections. Eligible multi-row
prefill and speculative verification use [mixed-storage GEMM](mixed-storage-gemm.md).
Unsettled or unsupported weight layouts, geometry or native arithmetic, CPU and
CUDA use their ordinary fallback paths. Those paths use an admitted F32
conversion or a temporary cast when a weight does not fit, without changing
precision. The [measured policy matrix](conversion-retention-validation.md)
records the retention tradeoff and validation scope.

The bound covers retained conversion **payload plus outstanding reservations**
for one loaded execution, shared by permanent units and embedded prediction.
An external drafter has its own budget: two independently loaded 32 MiB models
can retain 64 MiB of claims. Aliases within a group charge once; independent groups
sharing physical backing each admit their own claim. Their summed usage is not a
physical-residency total. Host-layerwise, disk-streamed and explicit device-ceiling
executions disable optional retention. Native multi-rank admission is currently
unsupported and effectively disabled until cross-process reservation exists;
temporary conversion execution remains available.

Use `LoadedModel::parameter_conversion_retention()` for requested/effective
policy, eligibility, scope and live retained/reserved usage. `PlannedModel` adds
role-labelled target and external-drafter observations. Read effective policy
from these reports rather than keeping a separate static limit. Unsupported or
unavailable observations mean unknown, not an empty cache. Some models never
need promotions, so an observed zero is also a valid result.

`reset()` clears request state and **preserves admitted conversions**.
`trim_parameter_conversions()` settles ordinary work and releases the caller's
optional claims while preserving source weights and request state. Trimming is
idempotent; later inference may admit conversions again within the same cap.
Controlled sessions require a canonical completed, drained boundary and reject
pending work instead of advancing implicitly. See
[controlled trimming](execution-control.md#trimming-retained-parameter-conversions).
Planned target/drafter trims run sequentially, so a later participant's error can
follow a successful earlier release; re-query participant usage after an error.
Other owners, snapshots, graphs or allocator caching can keep backing alive.
Released claim payload is not bytes returned to the OS, and trim does not flush
the allocator cache. There is no live policy setter.

For a fresh next request, reset request state and call `forecast_token_ids` or
`forecast_prepared_generation`; for a paused request use its controlled
`forecast_remaining_generation`. Query again after trimming so forecasts remove
stale retention credit. They read current claims without allocating execution
resources, populating or trimming caches, settling work, or consuming budgets.
Currently retained conversions are already included in resident parameters;
potential new admissions are a subset of pending cast workspace. Do not add
either again. Temporary casts are not capped by retention policy, and missing
native workspace/capacity facts still produce unknown bounds.

The combined memory picture includes original parameters, retained conversions,
request state and temporary workspace, allocator cache, and graph/driver allowance.
The allocator's separate 256 MiB cache setting plus the conversion default is
**not a combined 256 MiB ceiling**. Neither limits total process memory, padding,
or RSS, and retained native backing capacity can exceed its payload.

The [conversion_retention facade example](../eredu/examples/conversion_retention.rs)
configures 32 MiB, generates tokens, prints policy and usage, trims safely, then
resets and forecasts a fresh request using public APIs:

```sh
cargo run -p eredu --example conversion_retention -- \
  /path/to/model-with-tokenizer "Explain gravity briefly."
```

A model directory or GGUF with tokenizer metadata is accepted. The pinned mixed-width BF16 GGUF in the
[validation report](conversion-retention-validation.md) can report zero retained
conversions because mixed-storage kernels cover its projections. Narrow-activation
models can also report zero; fallback fixtures exercise nonzero admission. The example uses the default local device and
prints full observations and forecasts, including unknown values.

## Neutral resource-description contract

`eredu_core::resources` describes resources from ordinary execution contracts.
Target workspace forecasts compose reusable mechanism descriptions with
explicit evaluation lifetimes. Parameter/state producers and native mechanism
inventories remain descriptive contracts; their unknown allocation capacities
are not silently converted into exact bounds. Embedded prediction resource
descriptions feed speculative startup composition when both target and prediction
mechanisms are covered. Embedded continuation forecasts use settled state and retained-feature observations.

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
reflect the generic composition described below.

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
contracts. They are partial for the complete execution and do not independently
establish a fit verdict or prediction coverage. They contain no
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
arithmetic overflow remain explicit unknowns. Native model realization applies
Eredu's 256 MiB cache ceiling to an untouched native default, preserving smaller
defaults. Explicit limits remain authoritative. Pure cold queries still observe
the current policy; they do not apply runtime configuration. To forecast with the
managed default before loading, call `configure_local_runtime(&Default::default())`
first. `--mlx-cache-limit-bytes 0` disables allocator-cache retention when desired.

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

On iOS/iPadOS, tvOS and visionOS, availability instead comes from
`os_proc_available_memory()`: the current app's remaining allocation allowance.
Cold, loaded and continuation forecasts compare additional demand with this
freshly sampled headroom. Installed RAM is reported separately via `hw.memsize`
and is never substituted for app headroom. A zero allowance remains zero (also
possible for non-app/simulator runners), so a known positive additional demand
reports a shortfall instead of insufficient information. Both observations are
advisory; app limits and memory usage can change immediately after a query.

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

Serialization and deserialization both require this snake_case tagged format.

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

Native capture tests check top-k and prediction-schedule scaling, projected ledger
usage and read-only continuation observations. Sufficient quotas do not change
logical capture geometry. Run against a compatible local checkpoint with:

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

Embedded prediction startup uses ordinary module, state and capture contracts,
with explicit unknowns for uncovered mechanisms. Its parameters are already in
target residency and are counted once. Feature-conditioned external assistants,
cross-device MLX execution outside unified memory, uncovered ordinary workspaces,
media and transfer costs remain incomplete. Cold embedded planning can use
`inspected_speculative_generation_memory_plan` with explicit sampling calibration;
cold external-drafter planning still needs retained draft geometry.
Other backends opt in through `SpeculativeForecastBackend<D>`; its default report
is unavailable. The portable `SpeculativeMemoryPlan` can describe embedded
ownership and independent physical pools when a backend has the required facts.

This forecasts one fresh lane. Extra controlled snapshots/branches, retained
traces, captures/interventions enabled after startup and concurrent lanes need
their own accounting. Ordinary and controlled execution share the startup
forecast; the API does not claim to bound subsequent user-controlled retention.

### Speculative startup validation

The CLI can compare a target and independently loaded drafter on the same or
separate streams. Set `MODEL` to a checkpoint directory with a chat template:

```sh
cargo build -p eredu-cli --locked
target/debug/eredu --no-auto --model "$MODEL" --draft-model "$MODEL" \
  --speculative-draft-device gpu:0 --speculative-draft-tokens 4 \
  --disable-speculative-adaptive-lookahead --max-tokens 32 \
  --temperature 0.8 --seed 42 --memory-report /tmp/speculative.json --verbose \
  'The quick brown fox jumped over the fence and'
```

Omit `--speculative-draft-device` for a shared stream. Use `--temperature 0` and
`--disable-speculative-lookahead` for serial greedy execution, or select `cpu`
for an independent CPU drafter. Neutral conformance covers rejection/replay and
controlled/uninterrupted parity. The [mixed-storage suite](mixed-storage-gemm.md)
provides pinned released-checkpoint parity and memory evidence on Metal.

### Settled speculative continuation

Inside `with_controlled_text_speculative` or `with_controlled_chat_speculative`, use
`session.forecast_remaining_generation(additional_tokens, &options)` after prefill
or a canonical commit. The object-safe operation returns
`SpeculativeContinuationForecast`: target `request`, the existing `speculative`
plan (including a draft request or embedded prediction plan), horizon-specific
`continuation` observations, and
an `estimate`. It performs no native execution, completion polling, synchronization,
cache copy or budget reservation. It does not change the configured output limit.

Independent autoregressive drafters expose
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
for these requests; native state/residency facts remain available. Returned records or
external application copies retained outside these declared allowances are excluded.

Before prefill, terminal/cancelled/failed owners, retained proposals, optimistic
transactions and in-flight verification are explicitly unsupported. The method
does not make those states eligible by advancing execution. Embedded prediction
uses the settled observations described in the embedded continuation section; feature-conditioned
external assistants remain unsupported. User snapshots and inactive
branches are included from live non-rewindable reservations; restore and exchange
change the state observed by the next call without refunding those reservations.

Native continuation tests cover zero-token outlooks, read-only observation,
serial and split-stream lookahead, snapshot/fork/exchange/restore and token parity.
CUDA and separate physical-pool validation remain gaps.

Set `MODEL` to a checkpoint directory with a compatible chat template, or omit
the environment variables to use the nonzero native fixture:

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

### Supplied checkpoint metadata

`inspect_model_metadata` accepts complete SafeTensors or GGUF headers and
configuration without local weight files. `MetadataInspectionOptions` specifies
an explicit backend ID and portable loading policy. The facade obtains that
backend's cold capability facts without creating a device; unknown or disabled
backends return typed errors. Its retained selection feeds
`forecast_inspected_generation` and the existing request builders. Metadata
compatibility does not authorize payload loading;
`is_compatible()` and `is_loadable()` express these distinct outcomes. Source
provenance, bundle requirements and examples are in
[Forecasting from supplied checkpoint metadata](metadata-forecasting.md).

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
allocator policy. Cold CLI smoke coverage includes CPU host and Metal unified placement.

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
for restoration; the backend adapter preserves that result:

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
Embedded startup composes the same evaluator with prediction invocations and
transaction retention; uncovered prediction mechanisms remain explicit unknowns.
Ordinary cold, loaded, settled continuation and external-drafter forecasts consume
the same target evaluator. Legacy `WorkspaceGeometry` requests remain readable
through the aggregate compatibility lowerer and common lifetime evaluator; when
ordinary topology is present it is authoritative.

The selected logits contract distinguishes final-row projection from every-row
projection, including capture and speculative verification requirements.
Fused attention retains the configurable float32 score
matrix fallback. Explicit input-score attention instead consumes its selected
native tile/retention facts directly, without adding a second whole-score-matrix
fallback. A whole extra state payload covers cache replacement by default.
Unknown scratch or retention remains unbounded. Mechanism coverage is shared
by all families using those ordinary construction contracts; linked measurements
are validation cases, not dispatch criteria.

### Dense LFM2/LFM2.5 hybrid workspace

For complete key rows of at most 8,192 positions, MLX shares expanded K/V and
contiguous BF16 projection layouts across query tiles in an invocation. Large
calls evaluate batches of at most 32 query-tile outputs and retain completed
outputs for final concatenation, releasing each batch's score/softmax graphs.
Small calls remain lazy. The backend supplies an 8,192 query-by-key element
threshold and at most 32 query rows per tile.

For full-key calls the forecast charges four query-width K/V payloads per
invocation, at most 32 live tiles of score scratch at 32 bytes per element
(including a possible sink column), and two whole-query output payloads for
completed tiles and concatenation. Mixed-width execution allows at least four
bytes per scalar for K/V and output. Selection retains these native facts for
cold inspection and cached-selection validation. Runtime composes them with
ordinary attention geometry instead of adding another whole-score-matrix fallback.

Above 8,192 key positions, the two-pass blockwise accumulator evaluates each
block's running state and releases temporary layouts. Its FP32 accumulation
differs from full-key BF16 projection. Those rows retain a conservative per-tile
whole-context K/V allowance with 32 working bytes per score element. Serialized
facts without `full_key_tiles` use their declared per-tile copy and score
allowances. Missing native facts leave an unknown upper end; attention-free
convolution schedules need no attention scratch declaration.

LFM2 requires full-pass prefill, so reducing its requested chunk size does not
reduce the forecast. Dense and packed routed invocations use reusable mechanism
topology. Distributed projections, nondevice state and uncovered prediction or
residency combinations retain explicit gaps. Persistent convolution history stays
in the state estimate; gated/unfolded convolution workspace uses the module's
ordinary geometry and evaluation lifetimes.

Mixed-width selected parameters carry a potential F32 promotion allowance and
promoted-state/replacement storage above nominal state bytes. Selected task shapes
and dtypes determine the allowance, not a file-format name. Loaded forecasts
remove matrix promotions only where bound mechanism facts and activation dtype
proofs cover every invocation; see [projection storage facts](#projection-storage-facts).

The aggregate `WorkspaceGeometry` JSON fields `gated_convolution`,
`input_score_attention` and `mixed_precision_parameter_bytes` are optional.
Records without them deserialize with absent values. Ordinary selected execution
uses `TextExecutionTopology`.

Unified memory describes shared capacity, not automatic identity of host/device
buffers. Separate devices are compared independently. Declared shared backing or
an overlap interval drives the static-report adapter; global allocator counters
are never added on top of parameter tensors.

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

Build the executable from the source under validation.

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

## Validation coverage

[Mixed-storage validation](mixed-storage-gemm.md#validation-evidence) provides
pinned GGUF release measurements, exact logits and cached-generation parity,
controlled/speculative coverage, native allocation evidence and cold/loaded
forecast comparisons. [Conversion retention](conversion-retention-validation.md)
documents ownership, trim, fallback and policy/cache matrix checks.

Portable tests cover physical pools, unknown bounds, checked arithmetic, cache
frontiers, resource lifetimes, serialization, budget verdicts and read-only
observations. Native tests validate allocation envelopes and unchanged request
state at settled boundaries. Hardware coverage is explicit in each evidence
manifest; it does not establish universal allocator or process-memory bounds.

```sh
cargo test -p eredu-core -p eredu-runtime --locked
cargo test -p eredu-architectures --lib --locked
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance --locked
cargo test -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control native_generic_workspace --locked \
  -- --ignored --nocapture --test-threads=1
```

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
allowances, not measured distinct backing. The same bound covers possible installed and
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
autoregressive and embedded lanes use the separate speculative continuation plan;
the prepared speculative forecast still covers fresh speculative runs.


Continuation conformance covers pending-token offsets, zero and extended horizons,
overflow/shortfall/unknown growth, interior state maxima, restore, branch exchange
and failed iterator settlement. Native fixtures check bounded state and unchanged
allocation, tokens and snapshot usage during forecasting.

Semantic-history tests use a 64 MiB trace budget and cover empty, escaped, Unicode
and large event payloads. Doubling the budget adds one budget to future semantic
history and one to separately retained encoded trace; branch reservations charge
only the fresh child's budget. Capture, restore and branch exchange preserve
cumulative accounting.

```sh
cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_continuation_forecasts -- --ignored --nocapture --test-threads=1
EREDU_CONTINUATION_MODEL=/path/to/model-with-chat-template \
  cargo test -p eredu --features mlx,metal --test native_execution_control \
  native_continuation_forecasts -- --ignored --nocapture --test-threads=1
```

The fixture can also run with `--no-default-features --features mlx`, omitting
`--ignored`. Metal validation does not establish CUDA or CPU-only MLX coverage.

## Reusable mechanism descriptions

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

These descriptions are inputs for generic lifetime composition, not standalone
bounded forecasts. Logical tensor bytes and physical allocation capacity are
separate: views/no-op casts can alias, lazy intermediate records are not a live
peak, and query/key tiles can overlap until evaluation. Opaque MLX kernel scratch,
allocator capacity, unresolved owner identity and unsupported native selection
facts remain named gaps. Finite forecasts require covered invocation topology,
native mechanism facts and resource lifetimes together.

## Generic lifetime composition

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
consume budgets or issue a fit verdict. Forecast APIs use the composed resource
peaks with their capacity and budget policy.


### Embedded prediction resource descriptions

Embedded prediction has an additional execution description built from the
ordinary selected strategy, architecture discovery, capture schema and actual
prepared modules. `SelectedPreparation::embedded_prediction_topology()` exposes
cold facts; retained `PreparedModelDiscovery` exposes the same description after
loading without reopening the checkpoint or materializing modules.

The description distinguishes sequential proposal depths from fused proposal
rows, including their different context-preparation lengths. Invocation scopes
keep a fused context builder separate from its proposal blocks and score heads.
Prediction state remains independent of parameter sharing and proposal depth;
its policies and processed-position offsets come from ordinary state contracts.
The actual prepared module traversal retains local parameter declarations,
physical module ordinals and auxiliary residency owners.

Canonical shared parameter groups are logical uses, not extra allocations and
not proof that native storage aliases. Physical resource descriptions use exact
prepared backing identities. Retained target feature shapes are checked against
the same capture schema that admits execution; several feature views can refer
to one producer allocation. Missing producer backing, capacity or lifetime facts
remain explicit gaps.

`ResidencyReport::device_parameter_conversions()` optionally reports individual
existing conversion allocations with the residency owners and bindings that
retain them. MLX uses process-local lifetime identities rather than exposing
native pointers. Shared aliases contribute one conversion payload; replacements
get distinct identities, and invalidation or eviction removes released copies.
The report observes existing caches without evaluating tensors or populating a
conversion cache. Unknown native capacity and physical-pool identity stay unknown.
An absent allocation list means the producer only supplies aggregate telemetry;
an observed empty list means there are no retained conversions.

The generic prediction resource helpers merge authoritative feature/conversion
backings and preserve all logical uses. They reject conflicting physical facts,
malformed ownership and arithmetic overflow. Target weights already represented
by the target resource description are not added again simply because several
prediction heads reference them.

Prediction descriptions feed startup and continuation composition with the
speculative transaction plan. A description alone does not establish finite
native workspace, settle pending work or consume observation budgets.

Validation uses the neutral runtime and architecture unit suites, portable facade
and backend conformance, and the reference numerical suite. Focused regressions
cover cold descriptions for DeepSeek-V3, DeepSeek-V4, DSpark, Inkling, Qwen hybrid
and Nemotron-H; loaded queries preserve payload/materialization/execution counters.
They also cover sliding-state ranges, state offsets, exact module-slot ownership,
shared feature views, arithmetic errors and atomic rejection of conflicting
conversion observations. Native MLX tests cover conversion aliases, distinct
owners, invalidation, replacement and final-owner eviction.

Reproducible commands:

```sh
cargo test -p eredu-runtime -p eredu-architectures -p eredu --no-default-features --lib --locked
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance --locked
cargo test -p eredu-architectures --test reference_numeric --locked
cargo test -p eredu-architectures --test reference_conformance speculative_production --locked
cargo clippy -p eredu-runtime -p eredu-architectures -p eredu --no-default-features --lib --tests --locked -- -D warnings
cargo clippy -p eredu-backend-mlx --features metal --lib --tests --locked -- -D warnings
cargo test -p eredu-backend-mlx --features metal --lib parameter_conversion --locked -- --nocapture --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control internal_activations_match_continuous_and_controlled_metal --locked -- --ignored --nocapture --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control public_pooling_prediction_snapshots_metal --locked -- --ignored --nocapture --test-threads=1
```

Native prediction fixture coverage does not substitute for released-checkpoint
calibration. CUDA and native distributed prediction validation remain gaps.

### Embedded startup forecasts

`forecast_prepared_speculative_generation` and `forecast_speculative_token_ids`
retain an optional `SpeculativeMemoryPlan.embedded` record. It composes ordinary target
and prediction invocations with prediction state, retained target features,
verification, rollback, replay and configured lookahead. Sequential prediction
prepares all but the last prompt row; fused prediction prepares the full prefix.
Proposal width, physical module count and repeated shared readouts remain distinct.
Full-pass prefill and every-row verification logits remain transaction contracts.

Prediction parameters stay in the target's parameter residency. Shared embedding,
fusion and readout uses do not add another parameter allocation. Existing cached
conversions remain resident once; their authoritative native backing/binding
observations credit only the corresponding selected future-conversion allowances.
Only complete matching binding payloads receive credit; partial or missing
attribution keeps the conservative allowance. Selected physical formats,
possible F32 promotion, state frontier offsets and allocation-granularity allowances
are retained for recomputation. The resource descriptions remain logical facts;
finite scratch, graph retention and transaction-copy bounds are labeled calibration.

A finite estimate requires coverage for both sides. Dense Qwen3.5 text configurations
whose target layers all use full attention provide an end-to-end covered case.
The ordinary constructors and resource topology share the attention, MLP, fusion
and static-module specifications. This does **not** establish full coverage for
released Qwen hybrid schedules: gated-delta recurrent layers and shared/routed
expert reductions still report their missing mechanisms. Other embedded families
retain explicit gaps for latent/pooling/relative attention, hyper-connections and
other uncovered invocations. Rank-local prediction and unfamiliar capture layouts
also remain unknown.

Cold callers use `inspected_speculative_generation_memory_plan` to obtain a target
request and speculative plan from the same retained selection, supplying proposal
width, scheduler ceilings and backend sampling calibration. Pass both records to
`estimate_speculative_memory`. The legacy ordinary request stays unbounded for
embedded selection so it cannot silently omit prediction costs. The convenience
cold estimate reports unknown sampling scratch until backend calibration is
supplied. Loaded MLX forecasts reuse the adapter's sampling calibration.

The optional embedded plan and ordinary topology fields have serialization
defaults when omitted. JSON round trips and output-horizon recomputation
retain prediction costs. These operations do not allocate model tensors, populate
conversion caches, advance lanes or consume capture/snapshot budgets. Startup
forecasts describe a fresh isolated speculative lane even if an unrelated ordinary
cache has advanced. They are shared by continuous and controlled execution;
advanced embedded continuation outlooks use settled observations.

Native Metal startup fixtures cover nonzero all-attention Qwen3.5 text with F32
and mixed BF16/F32 parameters, two prediction modules and lookahead off/on. They
check serialized recomputation, horizon growth, budget shortfalls, read-only
allocator observations and continuous/controlled token and capture parity.
V3 and DSpark fixtures preserve explicit unknown prediction mechanisms.

The native startup checks are reproducible with:

```sh
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control native_embedded_startup --locked -- --ignored --nocapture --test-threads=1
```

These are synthetic fixtures. CUDA and native distributed prediction validation
remain gaps. Portable and reference checks use the commands in the prediction
resource-description section.

Additional validation commands:

```sh
cargo test -p eredu-architectures --test reference_structural --locked
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control native_speculative_continuation_forecasts_cover_settled_state_without_advancement --locked -- --ignored --nocapture --test-threads=1
cargo clippy -p eredu --no-default-features --features mlx,metal --test native_execution_control --locked -- -D warnings
```

### Embedded continuation forecasts

`ControlledSpeculativeSession::forecast_remaining_generation` supports
embedded prediction at the same canonical settled boundaries as independent
autoregressive drafters. The operation observes the installed target frontier,
each prediction layer's frontier, current and horizon-specific native state
capacity, retained target features and the durable prediction seed. It reuses the
startup prediction topology and transaction envelopes for drafting, verification,
rollback/replay and configured lookahead. It does not extrapolate an installed
prediction frontier from the target's position or the cold context offsets.

`SpeculativeContinuationMemoryPlan.embedded` retains those additional observations;
`draft` is optional and is present only for an independently resident drafter.
Existing serialized external-drafter records retain their draft object and read
without an embedded field. Rust consumers should handle the optional draft.
Recomputation validates that exactly one mechanism has matching observations and
that its horizon includes the requested tokens plus configured speculative
overshoot. Request a new observation after advancement, restore, branch exchange,
or changing the requested horizon.

Current prediction state has a logical payload lower bound derived from its
observed layer frontiers. Native backing-capacity allowances determine its upper
bound. Retained capture views include a conservative full-prefix feature allowance:
a small visible tensor cannot erase storage pinned by its parent allocation.
The current seed allowance also covers observed prediction capacity. These costs
remain in the zero-token outlook, which has no forward workspace. Positive
horizons include current retention plus future feature/state copies. Completed
loading and prefill are excluded in both cases. Snapshot and inactive-branch
reservations remain live across restore and exchange, without resident credit.

Each query refreshes parameter residency, available capacity, allocator-cache
policy and cached conversion ownership. Target and prediction weights share one
residency owner. Only exact complete conversion bindings replace their matching
future conversion allowances; aggregate conversion bytes are not credited again.

Coverage still depends on the ordinary selected mechanisms. The all-attention
dense Qwen3.5 text fixture is covered; this does not establish coverage for released
hybrid schedules or other missing prediction workspaces. Missing native capacity,
unknown retained storage, or mechanism topology required by a forecast phase
preserves an unknown upper bound. Feature-conditioned external assistants and instrumented speculative
retention remain outside this coverage. Arbitrary in-flight outlooks are still
unsupported: forecasting never polls, synchronizes, submits, copies caches,
advances a lane or consumes observation/snapshot budgets to make a query eligible.

Native embedded continuation fixtures cover F32 and mixed BF16/F32 weights,
lookahead off/on, live snapshots, zero/short/long horizons, JSON recomputation,
shortfalls, pending-work rejection, restore/fork/exchange and final token parity.
They require unchanged tokens, sampling state, epochs, snapshot usage and native
active allocation during queries. V3 and DSpark preserve explicit unknowns;
sequential/pooling cache capacity needs a reusable backing-capacity contract.

Portable conformance runs the same read-only/restore/branch/parity loop for embedded
and external mock mechanisms, including acceptance/rejection and lookahead. Runtime
coverage includes unequal observed layer frontiers, native capacity, full-prefix
feature retention, zero horizons, mismatched observations, unknowns, overflow,
serialization and external continuation records without an embedded field. The architecture observation
test distinguishes canonical prediction state from the retained seed.

Reproducible checks:

```sh
cargo test -p eredu-core -p eredu-runtime -p eredu-architectures -p eredu --no-default-features --lib --locked
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance --locked
cargo test -p eredu-architectures --test reference_conformance speculative_production --locked
cargo clippy -p eredu-core -p eredu-runtime -p eredu-architectures -p eredu --no-default-features --lib --tests --locked -- -D warnings
cargo clippy -p eredu-backend-mlx --features metal --lib --tests --locked -- -D warnings
cargo clippy -p eredu --no-default-features --features mlx,metal --test native_execution_control --locked -- -D warnings
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control native_embedded_ --locked -- --ignored --nocapture --test-threads=1
cargo test -p eredu --no-default-features --features mlx,metal --test native_execution_control native_speculative_continuation_forecasts --locked -- --ignored --nocapture --test-threads=1
```

These are synthetic fixture checks, not released-checkpoint calibration.
CUDA and native distributed embedded continuation validation remain gaps.


### Unified evaluation and compatibility

Ordinary, speculative, startup and continuation workspace estimates use
`describe_text_workspace` and the resource-lifetime peak composer. The aggregate
wire adapter and ordinary topology share attention scratch, tiled retention,
convolution intermediates, parameter-conversion and cache-replacement calibration.

`WorkspaceGeometry` records carry an aggregate linear/logit/coexistence envelope
as one compatibility allocation. They cannot supply exact projection formats,
parameter aliases or invocation order. Production selection supplies ordinary
topology, which takes precedence when both representations are present. The
compatibility lowerer preserves declared finite bounds, unknowns and checked
arithmetic without inventing a module topology or selecting family formulas.

Frozen serialization fixtures check bounds, verdicts, overlap, cache copies,
convolution, input-score tiling and mixed-parameter promotion. Nonzero Gemma2 and
Qwen3-MoE native fixtures set graph-driver overhead to zero so that allowance
cannot hide an underestimate. They compare reset, uninterrupted and controlled
generation and require forecasts to leave native allocation unchanged.

Released LFM2 checks accept a pinned checkpoint via `EREDU_LFM2_MEMORY_MODEL` and
cover cold, loaded and settled continuation at selected lengths:

```sh
EREDU_LFM2_MEMORY_MODEL=/path/to/pinned/LFM2.5-1.2B-Instruct \
EREDU_LFM2_MEMORY_LENGTHS=128,2000 \
cargo test -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control native_lfm2_workspace_forecasts --locked \
  -- --ignored --nocapture --test-threads=1
```

Set `EREDU_LFM2_MEMORY_QUANTIZED=1` for affine 4-bit with group size 64. Numerical
fixture export/comparison uses `native_lfm2_forecast_recalibration` and
`EREDU_LFM2_PARITY_WRITE` / `EREDU_LFM2_PARITY_REFERENCE`. Select a matching
checkpoint, weight mode and reference. The [mixed-storage runner](mixed-storage-gemm.md)
provides same-build reference comparison for the pinned BF16 GGUF.

Missing native contracts preserve unknown bounds; arbitrary in-flight outlooks
remain unsupported. CUDA and native distributed workspace validation remain gaps.

### Bounded native conversion retention

Eligible resident MLX executions enforce the managed 256 MiB conversion
payload allowance, shared by all permanent units and embedded prediction owners.
Admission counts retained plus reserved F32 payload; rejected weights follow the
same temporary promotion path, preserving F16/BF16 rounding and logits. There is
no automatic eviction. Reset preserves admitted copies and parameter publication
revokes them. The allocator-cache limit remains independent, including at zero.
See [conversion-retention validation](conversion-retention-validation.md) for the
policy/cache harness and the mixed-storage selector's allocation evidence.

Host-layerwise, disk-streamed and explicit device ceilings disable retention.
Multi-rank native execution also reports typed unsupported eligibility and effective
disabled retention until shared cross-process admission exists. Native inference
still uses temporary conversions. Payload ceilings do not bound temporary casts,
allocator padding, graph storage or process RSS.

Focused nonzero native fixtures cover disabled, 1-byte, exact 16-byte, partial
24-byte, exact 32-byte and unlimited policies; mixed F16/BF16 promotion; narrow
activation bypass; exact logits and greedy-token parity; tied aliases; independent
group claims; owner drop; replacement/restoration; a native shape failure; and
invalidation during evaluation. The policy matrix runs with allocator caching zero.
Reproduce with:

```sh
cargo test -p eredu-backend-mlx --features metal --lib parameter_conversion --locked -- --test-threads=1
```

Native fixtures require device access. Multi-process retention is unsupported;
these fixtures do not establish full-checkpoint throughput or process-memory bounds.

### Load-scoped conversion configuration and telemetry

Conversion retention can be selected without MLX-specific configuration:

```rust
use eredu_core::{DevicePlan, ExecutionPlan};
use eredu_core::residency::ParameterConversionRetentionPolicy;

let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "gpu:0")?)
    .with_parameter_conversion_retention(Some(
        ParameterConversionRetentionPolicy::Bounded { max_bytes: 32 * 1024 * 1024 },
    ));
```

Pass this plan to `LoadedModel::load_execution_plan`. The low-level portable
`NormalizedLoadRequest` has the same builder. Omitting the setting (`None`) uses
256 MiB for each eligible loaded execution; `Disabled` opts out and `Unlimited`
selects unbounded reuse deliberately. A zero bound is explicitly requested but
effectively disabled. Performance depends on which invocations require conversion;
see [native validation](conversion-retention-validation.md).
There is no process-global retention setting or live limit mutation.

`LoadedModel::parameter_conversion_retention()` reads policy and current retained
and reserved payload. The same observation is available in `static_memory()` and
the residency telemetry document. `PlannedModel::parameter_conversion_retention()`
labels the target and optional external drafter separately. Each independently
loaded drafter has its own budget, while embedded prediction owners share the
target budget. A 32 MiB target plus a 32 MiB external drafter can therefore retain
64 MiB of claims. Aliases within one group charge once; independent groups can
claim shared physical storage, so group usage is not an additive physical counter.

Requested policy and effective exclusions are both reported. Host-layerwise,
disk-streamed, explicit device ceilings and unsupported multi-rank admission
remain effectively disabled, even for an explicit unlimited request. Unsupported
observation is distinct from zero usage, and serialized reports with
no retention facts remain unknown. Auxiliary MLX assistants currently expose an
unsupported observation rather than inventing an empty ledger.

Conversion payload is already included in current resident parameter memory. Do
not add reported retained bytes again. Reservations describe pending admission,
not guaranteed materialized storage; backing capacity can exceed payload. The
allocator cache has its own independent allowance. Neither setting bounds total
memory, transient casts, graph storage, allocator padding or process RSS. Queries
leave native execution resources, generation state, completion authority and
observation budgets unchanged. Reset preserves the policy and admitted copies.
Explicit trimming and the retention-aware forecast subledger are described below.


The residency telemetry document preserves `current_device_bytes` and
`peak_device_bytes` as original-parameter admission-ledger counters.
`total_current_device_parameter_bytes` reports the checked sum of current original
parameters and deduplicated retained conversion payload;
`current_device_parameter_conversion_bytes` is a named subset of that total.
Use the total for current parameter accounting, without adding the subset again.
The parameter admission peak does not include optional conversion peaks.
An overflowing total is unavailable rather than saturated. This total covers the
ordinary residency ledger; independently managed routed banks retain their
separate telemetry. Use `StaticMemoryReport` for whole-model parameter composition,
including those banks.

The native load-policy fixture checks the managed default, explicit disabled,
17-byte bounded and unlimited policies, independent model identities, and matching
live/static observations. Reproduce with:

```sh
cargo test -p eredu-backend-mlx --features metal --lib loaded_conversion_retention_policy_reaches_native_residency --locked -- --test-threads=1
```

These checks cover configuration and telemetry; the
[validation matrix](conversion-retention-validation.md) covers execution behavior.

### Settled-boundary conversion trimming

`LoadedModel::trim_parameter_conversions()` settles submitted work and releases
optional retained parameter conversions without unloading source weights or
resetting generation state. Ordinary reset still preserves admitted conversions.
Controlled ordinary and speculative sessions expose the same operation at their
canonical completed, drained boundary, with typed rejection of pending work and
no implicit advancement. Planned executions report target and external-drafter
results separately; embedded prediction uses the target group.

The result reports released group claims and payload plus remaining claims and
reservations. Repeated trimming releases zero once the caller has no claims.
Other owners, snapshots, branches and outstanding native graphs can retain
physical storage; unobserved backing reclamation remains unknown. Group release
is not a promise of lower RSS or bytes returned to the OS. Allocator-cache
flushing remains independent and released backing may enter that cache.

Trimming preserves token history, cache frontiers, pending input, RNG, epochs and
observation/snapshot/fork accounting. Eligible source bindings remain registered,
so later inference may retain conversions again within the same load-selected
cap. Parameter editing invalidates obsolete bindings separately. Live telemetry
reflects released claims; read-only queries never perform a trim.

Regression coverage checks atomic alias release, independent groups,
idempotence, re-admission, pending reservations, retired bindings and delayed
publication. Facade conformance compares trimmed continuations with ordinary
output and retains snapshots and branches while checking sampling, sequence,
epoch and budget preservation. Native fixtures check live graph references across
trim, shared backing, re-admission and the loaded-session hook. Reproduce with:

```sh
cargo test -p eredu-core -p eredu-runtime -p eredu --no-default-features --lib --test portable_facade --test backend_conformance --locked
cargo test -p eredu-backend-mlx --features metal --lib parameter_conversion --locked -- --test-threads=1
cargo test -p eredu-backend-mlx --features metal --lib loaded_conversion_retention_policy_reaches_native_residency --locked -- --test-threads=1
```

The [validation matrix](conversion-retention-validation.md) documents checkpoint
measurements and local distributed coverage, including multi-host/accelerator gaps.
These fixtures do not establish total-memory or RSS reclamation bounds.

### Conversion-retention forecast subledger

`GenerationMemoryRequest::parameter_conversion_retention` records the selected
budget scopes, effective policies, current claims and reservations, and possible
new admissions. Each group's `additional_admission_payload` is a **subset** of
pending conversion workspace, not another contribution to the total. Published
conversions already belong to resident parameters. The ordinary pending-cast
allowance covers coexistence of future retained copies and temporary conversions;
it is never clamped to the retention ceiling. A tensor larger than the remaining
allowance can still require its full temporary cast during execution.

Cold plans start with zero claims and reservations. Their scope is local to that
cold request, and does not identify native backing or imply sharing between
independently loaded executions. New admission payload is bounded by eligible
selected payload and unreserved capacity. Exact attributed tensors too large to
fit receive no potential retention, and first-admitted winners are not predicted.
Unattributed geometry remains conservative. Cold partitioned selections currently
lack a neutral cross-rank retention-authority observation, so their effective
policy stays unavailable; this does not imply that native partitioned retention
is supported. Loaded reports replace cold scope with actual execution identities.

Loaded ordinary and speculative requests credit only complete matching selected
bindings with a current claim in the observed budget scope. Binding names in an
independent model do not establish reuse. Target and embedded prediction owners
share their budget; external drafters keep their own request and budget reports.
When target and external drafter observations identify the same conversion
allocation in the same single execution pool, composed parameter residency and
already-resident credit deduplicate its payload while retaining both admission
claims. Unknown placement across multiple execution pools receives no sharing
credit.

Every continuation observation refreshes the conversion identities and bindings,
including external drafters. Trim or parameter invalidation therefore removes
credit on the next query, even if another owner still keeps that backing alive
or an equal-sized replacement is admitted. Serialized requests with no retention
facts use aggregate loaded-profile credit; deserialization leaves the missing
policy unavailable rather than supplying the managed default.

Reservations consume admission capacity but do not establish whether a native
conversion has been materialized. Current group reports lack allocation/binding
attribution for outstanding reservations. Forecasts therefore keep the observed
parameter floor and an unknown phase upper end while reservations remain or their
usage is unavailable,
including a zero-query continuation boundary. They neither add a reservation as
a second allocation nor settle it to make the forecast finite. Once publication
is observed, the backing moves into the ordinary resident subset and exact
binding credit can remove the matching pending cast. A finite payload policy
never fills missing temporary-workspace, kernel, alignment or allocator facts.
All these queries are read-only: they do not create tensors, reserve retention,
trim owners, submit or settle execution, or consume observation budgets.

Controlled trim conformance checks that uncached workspace returns without
changing the frontier, output, sampling state or snapshot budgets. Restoring a
snapshot does not restore a released conversion claim. Reproduce with:

```sh
cargo test -p eredu-core -p eredu-runtime --lib --locked
cargo test -p eredu --no-default-features --test portable_facade --test backend_conformance --locked
cargo test -p eredu --no-default-features --lib api::forecast --locked
cargo clippy -p eredu-core -p eredu-runtime -p eredu-architectures -p eredu-backend-mlx -p eredu --no-default-features --lib --locked -- -D warnings
```


### Attention evaluation-aware transient lifetimes

The generic estimator consumes the selected backend's explicit upstream
evaluation fact. MLX's full-key InputScores path evaluates every batch, including
the final partial batch, when more than 32 query tiles are needed and the key row
is at most 8,192 positions. Merely sharing K/V layouts does not establish this
boundary. Short invocations, decode, Fused softcap, and the uncovered blockwise-key
path use conservative retention bounds.

At those frontiers, completed feed-forward intermediate projections/products and
unfolded convolution scratch end their lifetimes instead of overlapping subsequent
layers. The estimator
releases the completed attention invocation's score/layout workspace separately
from its retained outputs. It deliberately keeps residual, normalization, mixer,
final feed-forward and completed attention outputs, plus convolution inputs and
padded backing potentially held by history views. The additional 25% layer-workspace calibration and global
parameter-conversion/cache allowances overlap all interior peaks. This remains a conservative planning
upper end, not an exact allocation schedule or process-memory prediction.

Serialized full-key facts without an evaluation declaration default to no upstream
evaluation knowledge. Aggregate-only forecasts use their declared aggregate bounds. Portable tests
cover batch thresholds, key limits, Fused arithmetic, unknown retention, omitted facts,
and conversion/cache allowances at an early dominant peak.

The native lifetime harness checks the same request with and without the
upstream evaluation fact, holding placement and calibration fixed. It covers cold,
loaded and continuation forecasts, eight predictions, reset, read-only observations
and controlled replay. Long-key, CUDA and native distributed validation remain gaps;
the uncovered blockwise-key path keeps conservative costs and portable boundary tests.

Reproduce with each pinned checkpoint path:

```sh
EREDU_LFM2_MEMORY_MODEL=/path/to/pinned/LFM2.5-1.2B-Instruct \
EREDU_LFM2_MEMORY_LENGTHS=128,2000 \
EREDU_LFM2_MEMORY_WRITE=/tmp/lfm2-lifetimes-bf16.json \
cargo test -p eredu --no-default-features --features mlx,metal \
  --test native_execution_control native_lfm2_workspace_forecasts --locked \
  -- --ignored --nocapture --test-threads=1
```

Repeat with `EREDU_LFM2_MEMORY_QUANTIZED=1` for affine 4-bit, or point
`EREDU_LFM2_MEMORY_MODEL` to the pinned `.gguf` file for mixed-width BF16.
Run outside a sandbox that blocks Metal device access. The harness logs the
same-request upper with the evaluation fact disabled as well as the selected estimate.

## Projection storage facts

Loaded fully resident, unmodified, nondistributed MLX executions attach bound
`ProjectionStorageFacts` to ordinary and speculative target/drafter geometry.
Facts require settled supported weight layout, exact shape/dtype and eligible
native dispatch. Runtime separately requires an architecture declaration proving
F32 projection inputs from the backend's observed normalization gain bindings;
state storage width alone does not establish activation dtype.

A parameter receives full-weight promotion credit only when all shared invocations
are covered and its selected payload is exactly attributed. Existing retained
conversion credit is not subtracted twice. Outputs and bias retain their ordinary
allocation owners. Native split-K partials and possible activation-layout copies
are explicit additional workspace, with the concurrent-layer calibration retained.
The prefix envelope includes shorter-row split-K maxima because partial-buffer
size is not monotone in row count.

Cold inspection, missing dtype-flow declarations, unsupported dispatch, uncertain
layouts, bounded residency, distributed bindings and active parameter overlays
retain conservative costs. These observations create no native arrays, evaluate
no tensors and reserve no conversion storage. Allocator rounding and GPU-private
storage remain outside exact payload facts.

See [mixed-storage GEMM](mixed-storage-gemm.md#forecast-contract) for eligibility,
workspace details and measured cold/loaded forecasts. For the pinned BF16 GGUF,
8 output tokens and zero allocator cache, the loaded 2,000-position upper bound is
12.779 GiB, while the cold upper is 15.295 GiB. The loaded bound retains possible
input copies, short-row partial maxima and other calibrated workspace; the
measured prefill growth in that evidence is 1.301 GiB above loaded allocations.
