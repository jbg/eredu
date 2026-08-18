# safemlx-lm

`safemlx-lm` is the public Rust facade and MLX backend for loading and
running supported language, multimodal, and realtime speech models. Portable
contracts and orchestration live in the MLX-independent `safemlx-lm-core`
crate. The facade provides architecture
inspection, checkpoint validation, tokenization, chat preparation, generation,
structured tool calls, quantization, bounded weight and cache residency, and
distributed execution.

Support is determined from checkpoint metadata and validated tensor structure,
not from a model's display name. Unsupported configurations fail before weight
payloads are materialized.

## Features

- SafeTensors model directories and sharded GGUF checkpoints.
- Text, image, video, audio, and realtime codec-token model inputs where the
  selected architecture supports them.
- Greedy and stochastic generation, prepared chats, reasoning events, and
  constrained native tool calls.
- External and checkpoint-embedded multi-token prediction.
- Load-time affine or MXFP4 quantization for eligible dense weights.
- Fully resident, host-layerwise, disk-streamed, and independent expert-cache
  weight policies.
- Device-resident or paged live caches and reusable prompt-cache persistence.
- Tensor, pipeline, and expert parallel execution for supported families.

See the [model support matrix](../doc/model-support.md) for the supported
families, checkpoint formats, modalities, and major restrictions.

## Installation

```toml
[dependencies]
safemlx-lm = "0.4"
```

The default feature set selects the MLX backend. Backend implementers and
applications that provide another `TextGenerationBackend` can use the same
facade without compiling or linking MLX:

```toml
[dependencies]
safemlx-lm = { version = "0.4", default-features = false }
```

That build exposes the canonical `LoadedModel<B>`, `ModelRuntime<B>`, portable
artifact and generation contracts, tokenizer loading, EOS and chat-template
discovery, semantic chat inspection, `LoadedModel::load`, chat preparation, and
ordinary constrained prepared-chat generation. The `mlx` feature adds architecture implementations, native model
loading, prepared MLX inputs, sampling, residency, distributed execution, and
runtime diagnostics. `cuda`, `nccl`, and media-processing features imply
`mlx`; they are capabilities of that backend rather than facade features.

Enable `image-processing` when the application wants the built-in image
preprocessing path. Enable `cuda` for an MLX CUDA build on a supported Linux or
Windows host. Tokenization always has the portable pure-Rust `fancy-regex`
engine available, including under `--no-default-features`; default builds also
enable Oniguruma and select it as the faster tokenizer regex implementation.

## Inspect before loading

Backend structural inspection performs the same architecture, tensor,
quantization, residency, and topology preflight used by loading without
creating a stream or loading weight payloads. Text inspection is a separate
facade step because tokenizer, chat-template, EOS, and semantic behavior do not
belong to an execution backend. `inspect_text_model`, `load_tokenizer`, and the
chat request types remain available with `default-features = false`; only the
structural report producer is selected by the application:

```rust,no_run
use safemlx_lm::{
    api::{inspect_text_model, TextInspectionOptions},
    backend::mlx::{inspect_model, MlxInspectionOptions},
};

let structural = inspect_model("/path/to/model", MlxInspectionOptions::default())?;
let report = inspect_text_model(structural, TextInspectionOptions::default());
if !report.is_loadable() {
    for issue in &report.issues {
        eprintln!("{:?}: {}", issue.code, issue.detail);
    }
}
# Ok::<(), safemlx_lm::backend::mlx::error::Error>(())
```

Use the high-level `api` module for prepared inputs, generation, cache
creation, and memory admission. `safemlx_lm::load_model` is the single generic
model-loading entry point for supported checkpoint formats, architectures, and
execution backends. The caller passes the selected backend plus its associated
load options. Core first builds a backend-neutral `ModelPreparationPlan` from
`safemlx-lm-core`: format detection, model-family resolution,
SafeTensors/GGUF header catalogs, and load-route validation happen without
MLX. The selected backend then consumes that plan. `MlxBackend` owns its
execution and weight-materialization streams and maps payloads into executable
MLX arrays/modules; streams are not separate loader arguments.
Portable inspection and capability report types live in `safemlx-lm-core` and
are reexported by the facade. Tensor structure and process-memory probes are
selected-backend operations. Text sidecar and behavioral inspection composes
above that result and is never called from the MLX adapter.
Backends implement `ModelCapabilityBackend` against their selected
`ModelRuntime`; generic callers then use the same `LoadedModel<B>` methods for
capability discovery, token or prepared-input counting, runtime-state
estimation, static-memory reporting, and admission. Prepared inputs remain the
backend's opaque prompt type, so these methods expose no MLX arrays or streams.
Replicated, tensor-, pipeline-, and expert-parallel prefill/decode execution
crosses the single stateful `MlxModelSession` implementation of the core
session contract. `load_model(&backend, artifact, options)` returns a
`PreparedModel<B::Model>` under `ModelLoadError<B::Error>`; portable artifact
inspection/planning failures and selected-backend materialization failures are
distinct. The preparation marker is required when creating a session and is
consumed by that operation. For MLX, the opaque model selects
complete-model, rank-local pipeline, or rank-local expert materialization
inside the adapter. The session
owns that model, the matching cache state, and its optional distributed
capability. Direct
submissions return their exact completion; callers either retain it or use
`Submission::wait`. Raw generation retains in-flight completions internally.
Realtime Moshi/PersonaPlex requests use the backend-neutral core scheduler;
only their tensor execution and exact MLX completion adapter remain here.
Distributed pipeline schedule, cancellation, and completion agreement use the
same core scheduler through the MLX collective and completion adapters; the
facade does not maintain a second lifecycle implementation.
The Cartesian shape, rank coordinates, subgroup membership, balanced
layer/expert ownership, and preflight report are canonical
`safemlx-lm-core` values. `MlxParallelContext` binds one validated core rank to
an explicit process-local MLX device; native communicator construction and
collective tensor execution remain inside the selected MLX session. MLX
currently fails closed for data-parallel model sessions.
`MlxBackend::with_distributed_world` selects the world before
`Backend::create_session` consumes and binds the opaque model, topology-derived TP/PP/EP
communicators, and the correct cache form into one lifecycle. There are no
public architecture-specific distributed loaders, rank-local model types, or
standalone communication-session constructors.

The tokenizer-aware entry point is generic too:

```rust,ignore
use safemlx_lm::{api::LoadedModel, backend::mlx::{MlxBackend, ModelLoadOptions}};

let backend = MlxBackend::new(execution_stream, weights_stream);
let model = LoadedModel::load(backend, artifact, ModelLoadOptions::default())?;
```

MLX-owned executable, cache, load-policy, and generation types are exposed only
from `backend::mlx`: `Model`, `ModelCache`, `ModelLoadOptions`, and
`MlxGeneration`. Architecture tensor implementations and MLX neural-network
layers live under `backend::mlx::architectures` and `backend::mlx::nn`. MLX
cache, checkpoint, distributed, execution, sampling, media, and residency
infrastructure is grouped under `backend::mlx::runtime`.
The backend-neutral `api`, crate root, and facade `runtime` namespaces do not
alias these implementation types.

`LoadedModel<B>` has no default backend type. The same call accepts another
`ModelLoadingBackend + TextGenerationBackend`; only backend construction and
its associated options change. It shares one artifact inspection between
portable tokenizer/chat/EOS assembly and backend materialization. Metadata
assembly returns `TextMetadataError`, while tokenizer-aware operations
on an assembled model return `TextModelError`; neither type contains an MLX
exception. Portable decoded media requests are prepared by the capability of
the already-selected session rather than by injecting an MLX processor after
loading.
Cache policy selection, prompt-cache save/load, and embedded MTP generation
likewise stay on that session; there is no parallel stage-cache or distributed
request-scheduler API beside it, and no complete-model cache extraction or
replacement API.
Weight-residency plans, atomic admission, ownership leases, protected windows,
eviction decisions, exact transfer generations, and accounting likewise come
directly from `safemlx-lm-core`. The MLX facade mirrors those transitions with
concrete arrays and host buffers and owns native materialization, event/source
retention, physical-capacity queries, and allocator sampling.
Bounded disk-to-host prefetch ordering, admission, duplicate coalescing,
cancellation fencing, rollback, failure recovery, demand handoff, and telemetry
also come from core. The MLX worker channel only wakes the concrete
SafeTensors-to-host materializer; it does not own another request queue or
lifecycle table.
Aggregate live-cache admission is also canonical core state: the MLX pager
registers each cache session with `CacheResidencyPool` and holds core RAII
reservations while device/host transfers or disk writes own resources.
Stable cache block identities, logical tiers, layer geometry, and
fixed-state policies are also canonical core contracts. Core additionally owns
block registration, exact leases, access ordering, protected prefixes,
deterministic eviction choice, transactional truncation, and mutable tails.
The generic core `CacheBlockStorage` state machine also owns legal device,
host, writing, backed, disk, and reading transitions. Its opaque resource slots
hold MLX arrays, host buffers, disk locations, and native completion tickets;
the adapter cannot define a parallel phase enum.
Core `CacheIoExecutionState` also owns exact-key joining, bounded disk-worker
admission, queued/in-flight cancellation, and completion disposition. The MLX
cache adapter retains only task payloads, condition variables, physical worker
execution, SafeTensors I/O, mappings, and cleanup.

The same is true of prompt-cache model/prefix identity, topology, versioned
manifests, fingerprinting, compatibility, and catalog validation. MLX cache
code imports them directly and supplies only resource implementations, transfer
workers, exact native completion observation, safetensors payload verification,
and filesystem publication.

## Automatic execution planning

Embedding applications can use the same versioned single-device planner as the
example CLI. The request, policy, selected plan, explanation, and runtime
telemetry all implement Serde serialization:

```rust,no_run
use safemlx_lm::{
    backend::mlx::automatic::plan_automatic_execution,
    AutomaticPlanRequest, DevicePlan, ExecutionTelemetry,
};

let prior_runs: Vec<ExecutionTelemetry> = load_prior_runs();
let request = AutomaticPlanRequest::new(
    "/path/to/model",
    DevicePlan::new("mlx", "metal:0")?,
)
.with_prior_telemetry(prior_runs);
let report = plan_automatic_execution(&request)?;

# fn load_prior_runs() -> Vec<ExecutionTelemetry> { Vec::new() }
# Ok::<(), Box<dyn std::error::Error>>(())
```

Matching prior runs are grouped by exact plan and ranked by median decode
throughput. Telemetry from a different model, artifact layout, device, stable
hardware profile, or schema is ignored. Historical plans are also rechecked
against current available memory and header-only loader admission. Use
`AutomaticPlanner::new` with an `AutomaticPlannerPolicy` for explicit policy
bounds. Its `plan` method accepts an `AutomaticPlanningBackend`, so the same
selection policy works with another backend's discovery and admission adapter.
MLX discovery, bounded-load probing, telemetry collection, and
`execution_plan_load_options` live under `backend::mlx::automatic`. Backend and
device identifiers are adapter-defined rather than a closed core enumeration;
device creation and plan realization remain owned by the selected backend.
Speculative generation uses the same core-owned committed-token and terminal
lifecycle as ordinary generation. Its prefill/proposal/verification/commit
executor contract is also core-owned; the MLX implementations supply opaque
model input, logits, assistant state, cache transactions, stream placement, and
exact event completion. Core also owns proposal sequencing, stochastic
accept/reject flow, replacement/bonus decisions, and bounded fair action
selection through a high-level opaque sampling contract. Each in-flight
verification, cache checkpoint, draft block, optimistic branch, and exact
completion is one core-owned transaction. Core waits, commits the MLX cache,
resolves branch promotion/discard and telemetry, and only then authorizes token
and semantic-event publication. Core owns the transactional semantic-state
wrapper, callback publisher, cancellation delivery, and output error; none use
MLX exceptions. The generic core `SpeculativeRequestTable` also owns the
production request collection, cancellation scan, fair action application,
resource-slot invariants, aggregate accounting, and ordered finalization. The
facade supplies portable prompt, constraint, and semantic setup; the MLX
backend supplies the executor and sampling adapter used to instantiate it.
Embedded prediction heads and external assistants enter the same
`MlxMtpScheduler`; architecture adapters do not define parallel generation
loops or forwarding wrappers. Prepared-chat MTP requests contain portable
sampling and scheduler settings; target and drafter execution placement is
fixed when those models are loaded, and MLX cache lanes are adapter-owned.
`SpeculativeGenerationBackend` makes the same capability, single-request, and
batch methods available on `LoadedModel<B>`; its associated drafter type
prevents mixing backend families. Before backend dispatch, the facade resolves
checkpoint sampling defaults, prepares the opaque backend prompt, and constructs
the canonical speculative grammar and transactional semantic parser. Backends
receive an execution-ready request and cannot reimplement chat-plan parsing.
`PreparedChatMtpError<B::Error>` keeps portable setup failures distinct from the
selected backend's execution error. Ordinary prepared-chat
generation similarly returns `PreparedChatError<B::Error>` rather than the MLX
facade error. MLX retains the actual logits transforms,
probability/residual arithmetic, random arrays, cross-stream transfers,
component timing probes, and tensor execution.

## Inputs and generation

`LoadedModel<B>` owns a selected backend `ModelRuntime`, tokenizer, and
chat-template metadata. The runtime owns the executable, cache, optional
backend processor, and backend queue as one session; applications do not
construct or pair raw caches with models. Plain text clients can be generic over
`B: TextGenerationBackend`: `encode`, `generate_tokens`, token-id observation,
and `decode` use the same API for MLX and a future backend. The MLX adapter keeps
arrays, sampling math, PRNG state, streams, and exact events behind that
contract. `generate_prepared_chat` is generic over the same backend: core
passes portable vocabulary filters to backend sampling while the facade owns
grammar commitment, cancellation, stop/EOS precedence, and protocol-neutral
semantic events. `MultimodalRequest` owns ordered text/token segments and
decoded `RgbImage`, `Audio`, or `Video` values. A
`B: MultimodalPreparationBackend` client calls
`LoadedModel::prepare_multimodal_input` (or the exact-placeholder chat helper)
and receives `B::Prompt`; the same caller code works for MLX and another
backend. MLX keeps resizing, normalization, feature extraction, framing,
tensor construction, and placement behind that capability and returns
`MlxModelInput`. Raw MLX processor inputs and processor handles are not public
alternatives. Backend-native controls do not appear as specialized methods on
`LoadedModel`; applications reach intentional native session controls through
`model.runtime()` only. This keeps ordinary caller code identical across
backends while making native operations visibly backend-specific.

Raw generation remains available for completion workloads. It deliberately
bypasses chat-template and native-tool guarantees.

Realtime speech-token clients use `RealtimeModel<B>` and
`RealtimeScheduler<B>`. Request/session scheduling, portable sampling
configuration, cancellation, and exact completion are backend-generic. The MLX
adapter owns Moshi/PersonaPlex arrays, caches, samplers, random state, streams,
and events; another backend implements the same whole-step contract without
changing the scheduler client.

## Memory and execution policies

Weights and runtime state have separate policies and budgets:

- Weight residency controls where static parameters live and when they are
  materialized.
- Cache residency controls KV, compressed-latent, convolution, and recurrent
  request state.
- Load-time quantization changes eligible static weights but never silently
  transcodes an already packed checkpoint format.

Read [Model loading, quantization, and memory](../doc/model-loading.md) before
using bounded residency or prompt-cache persistence. Distributed users should
also read [Parallel execution](../doc/parallel-execution.md).

## More documentation

- [Backend architecture](../doc/backend-architecture.md)
- [Native tool calling](../doc/tool-calling.md)
- [Speculative decoding and MTP](../doc/speculative-decoding.md)
- [Cancellation and bounded execution](../doc/cancellation.md)
- [CLI example](../examples/safemlx-lm-cli/README.md)
- [Platform setup](../doc/platforms.md)

The minimum supported Rust version is 1.89.

## License

Licensed under either Apache-2.0 or MIT.
