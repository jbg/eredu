# Language-model backend architecture

`safemlx-lm-core` is the backend-neutral foundation of the language-model
runtime. `safemlx-lm` is the public facade and contains an optional MLX
implementation. Its default `mlx` feature preserves MLX as the standard
backend; `default-features = false` provides the same canonical
`LoadedModel<B>` and core contracts without compiling or linking MLX.

The dependency direction is one way:

```text
safemlx-lm-core
      ↑
  safemlx-lm portable facade
      ↑ optional `mlx` feature
  MLX adapter and model implementations
      ↑
applications and examples
```

Neither `safemlx-lm-core` nor the feature-disabled `safemlx-lm` facade has a
dependency on `safemlx`, `safemlx-sys`, Metal, CUDA, or another tensor runtime.
Separate dependency-graph tests enforce both properties. Native architecture,
array, stream, cache, sampler, residency, and distributed modules exist only
when `mlx` is enabled; they are not placeholder types in the portable API.

## Module ownership

`safemlx_lm::backend` is always present so backend selection has one stable
namespace in portable builds. Concrete implementations are feature-gated
inside it: the default `mlx` feature exposes `backend::mlx`, while a build with
`default-features = false` exposes no MLX implementation or native dependency.

The facade-level `runtime` namespace contains only backend-independent chat
preparation and committed semantic streaming. MLX model-family tensor
implementations and reusable neural-network layers live under
`backend::mlx::architectures` and `backend::mlx::nn`. MLX arrays, checkpoint
stores, KV-cache storage, native distributed transport, model execution,
sampling, media tensor preparation, residency workers, and the MLX-aware error
type live under `backend::mlx::runtime` and `backend::mlx::error`. Common MLX
load-policy types are reexported within `backend::mlx`, never at the facade
crate root. The crate root deliberately has no architecture or neural-network
namespace.
There are no forwarding modules at the former `runtime::{cache, checkpoint,
distributed, execution, media, residency}` or `runtime::generation::sampler`
paths. Backend-neutral attention types are canonical core exports rather than
a `runtime::attention` alias.

The generic `api::loaded` module owns tokenizer-aware loading, ordinary text
generation, prepared-chat construction, and backend-generic speculative
request assembly. It does not name MLX or architecture tensor types. The
feature-gated `backend::mlx::prepared_speculative` implementation binds those
prepared requests to MLX samplers, exact streams, model caches, and embedded or
external MTP executors. The backend depends only on core contracts; it does not
import facade API or runtime orchestration.

The same ownership rule applies to tests. Portable tokenizer, chat, EOS,
metadata, and request-orchestration tests compile in `api::tests` with no
backend feature. Native arrays, MLX model execution, quantization, MTP, and
architecture materialization tests live in the feature-gated
`api::tests::mlx` module.

## Core and backend responsibilities

Core owns concepts whose meaning does not depend on tensor representation:

- artifact/model identity, SafeTensors/GGUF header inspection, model-family
  resolution, neutral tensor catalogs, and materialization-route selection;
- validated attention schedules;
- generation configuration validation, committed-token sequencing, terminal
  precedence, cancellation, finish reasons, semantic output events, and
  speculative proposal generation, target acceptance/replacement, verification
  commit plans, fair speculative action selection, exact request transactions,
  optimistic continuation promotion/discard, and speculative telemetry;
- queue fairness, request/work lifecycle, transactional branch commit/discard,
  exact-completion observation, cancellation, abandonment, and capacity;
- backend/device descriptors and fail-closed capability discovery;
- validated decoded RGB image, mono PCM audio, and RGB video descriptions;
  ordered text/token/media request composition, exact rendered-chat placeholder
  binding, and portable video-sampling policy;
- model capability, input-position, cache-layout, runtime-state, static-memory,
  admission request/result, and artifact-inspection schemas; checked state
  estimation and context/memory admission policy;
- validated parallel shapes and rank snapshots, axes, coordinates, subgroup
  membership, balanced ownership/preflight planning, and portable placement
  descriptions;
- weight-residency policy, atomic ownership/capacity transitions, protected
  windows, deterministic eviction, exact transfer generations, accounting, and
  serialized reports; and bounded prefetch FIFO admission, duplicate
  coalescing, cancellation generations, exact operation disposition, failure
  recovery, demand handoff, and worker telemetry.
- process-wide live-cache membership, multi-resource admission, exact
  reservation ownership, occupancy reports, and high-water accounting;
- stable cache block/rank identity, logical tier vocabulary,
  per-layer attention-state geometry, fixed-state roles, symbolic shapes, dtype
  families, and pure state-residency validation;
- reusable prompt-cache model/prefix identity, rank-local topology, versioned
  manifests, stable fingerprints, catalog geometry and coverage, compatibility,
  and safe relative-path validation.

A backend owns executable models, tensor values, streams/queues, cache storage,
sampling math, transfers, collectives, kernels, native errors, and the concrete
completion primitive. Core does not define a primitive tensor algebra and does
not accept `Any`, raw pointers, or operation names as an execution escape hatch.
Backends also own loaded-model capability derivation, prepared-input accounting,
process allocator/availability observations, and structural checkpoint
admission. Those operations populate canonical core reports rather than
defining backend-specific copies of the schemas.

## Model, session, and completion lifecycle

`Backend::prepare_model` selects one backend and produces an opaque
`PreparedModel`. `Backend::create_session` consumes that value and creates a
backend-owned object implementing `BackendSession` directly. That object owns
the executable, persistent decode/cache state, and optional communication as
one unit. Prefill and decode therefore cannot be submitted with a different
model, nor can one mutable executable be driven through two sessions.
There is one backend session contract: the backend-selected path does not
provide an architecture-specific or borrowing execution escape hatch.

`ModelRuntime<B>` is the canonical client owner of the selected backend and
that session. Its `prefill` and `decode` methods are the production submission
path for both the core mock backend and MLX. Generic downstream coordinators
can be written once over `B: Backend`; associated input, output, and completion
types keep backend values opaque without runtime type erasure. A runtime may
expose its session for explicit optional capabilities, but it never exposes or
separates the executable from its cache.

`TextGenerationBackend` is the portable text-generation extension. It asks the
selected backend for whole prefill-and-sample and decode-and-sample submissions,
not logits or tensor primitives. The backend owns its token handle, sampler and
randomness state, cache, exact completion, and opaque prepared prompt.
`TextGeneration<B>` retains
unfinished completions, feeds the opaque token directly into the next decode,
and exposes only the canonical `u32` vocabulary id through `TokenOutput` when a
client actually needs it. The resolved token budget is enforced in core.
`api::LoadedModel<B>` combines this runtime with backend-independent tokenizer,
EOS, checkpoint-generation, and chat-template state. Code written over
`B: TextGenerationBackend` can therefore encode, generate, and decode without
naming MLX. `LoadedModel::load(backend, artifact, options)` is the sole
tokenizer-aware artifact entry point and has no default backend type. It shares
one inspection between portable tokenizer/chat/EOS assembly and
`prepare_inspected_model`; a second backend uses the same call rather than a
backend-specific facade constructor. `LoadedModel::from_runtime` remains the
constructor for an already-prepared runtime assembled by an application.

`MultimodalPreparationBackend` is the optional preparation capability for that
same selected session. `LoadedModel::prepare_multimodal_input` tokenizes caller
text, then passes ordered token IDs and decoded media to the backend and returns
the backend's existing opaque `Prompt`. Processors may ask the facade tokenizer
to encode checkpoint-defined framing text or timestamps through a typed
callback; tokenizer failures do not become backend errors. The chat helper
first replaces exact rendered placeholders with media in core, so placeholder
count and ordering semantics are shared by every backend.

Tokenizer reconstruction is also outside the MLX boundary. The portable
facade loads Hugging Face tokenizer sidecars, EOS metadata, chat templates, and
semantic protocol plans without an execution feature. `safemlx-lm-utils` owns
the canonical GGUF-tokenizer and Kimi tiktoken conversions; the facade owns
artifact-relative discovery and inspection policy. GPT-OSS Harmony and LFM2
wire protocols live with the other portable chat dialects rather than under
their MLX tensor architectures. Consequently `inspect_text_model` can enrich a
structural report produced by any backend in a `default-features = false`
build. MLX keeps only payload materialization, tensor execution, sampling, and
native completion in this path.

Constrained generation uses the same contract rather than a parallel MLX
loop. The core `TokenFilterController` contract lets a facade-owned grammar and
activation state return a validated portable `TokenFilter` before each submission.
`ControlledTextGeneration<B, C>` passes that filter to the selected backend,
observes the committed vocabulary id once, advances the controller, and retains
the exact backend completion. It never exposes logits. The ordinary
`PreparedChatGenerationRequest<B, F>` uses this path for tokenization, grammar
masking, cancellation, semantic parsing, stop/EOS precedence, and callback
publication. Its sampling settings contain a portable seed rather than an MLX
random array. This complete load, prepare, and generate path is compiled and
tested with a mock backend under `safemlx-lm --no-default-features`; it is not
an MLX-gated facade extension.

Every submission returns an opaque output and an exact `Completion`. Completion
can be polled or waited without draining unrelated work. A scheduler-owned
transition retains its branch, output, completion, arrays, and leases until
that exact completion succeeds. Only then can it publish the branch. Cancelled
submitted work becomes abandoned: it cannot publish, but its resources remain
owned until the same exact completion makes release safe.

The backend is selected for the entire prepared model/session. This keeps cache
and executable ownership coherent and avoids hidden cross-runtime transfers or
per-operation dispatch costs. Transfers and collectives are optional explicit
capabilities and therefore fail closed when absent.

Realtime token models use the parallel `RealtimeBackend` contract. A
`RealtimeModel<B>` binds one loaded temporal/depth model to its backend, and
`RealtimeScheduler<B>` owns fair request registration, frame queueing, batch
stability, session handoff, cancellation, and exact completion. Encoded frames,
generated text/audio frames, cache state, and completions are opaque associated
types. Sampling temperatures and the deterministic root seed are portable;
backend sampler objects, random state, and execution queues are not public
request fields. The production Moshi/PersonaPlex path uses this generic
scheduler rather than a separate MLX coordinator.

Realtime loading follows the same backend-selection rule.
`load_realtime_model[_with_options]` accepts a complete selected backend and
calls its `RealtimeModelLoadingBackend` implementation. MLX callers construct
`MlxRealtimeBackend` with execution and weight-materialization streams; neither
stream appears in the generic loading signature. The former MLX-only
`api::realtime` loader namespace was deleted rather than retained as a wrapper.

`DistributedBackend` is the optional extension for model sessions which
communicate across ranks. It exposes the `DistributedSession` attached to the
selected model session; communicator construction is not a second independent
core lifecycle. `DistributedSession` exposes opaque values, typed world/axis
scopes, sum and gather collectives, variable-count all-to-all exchange,
point-to-point transfer, portable scheduler-word consensus, and exact
completion. Core contains no communicator, stream, native event, or tensor
type. Capability discovery is operation-specific and defaults to false.

`ParallelTopology` is the canonical validated four-axis shape, and
`ParallelRankTopology` is its canonical rank-local snapshot. Rank/coordinate
mapping, TP/PP/EP/data subgroup membership, pipeline neighbors, balanced layer
and expert ownership, and preflight reports are pure core operations. World
size is derived from the axis sizes; there is no independent caller-supplied
world size that can disagree with the shape. These values contain no device or
communicator.

## MLX mapping

The core `load_model` entry point operates on any selected
`ModelLoadingBackend`. It calls `inspect_artifact` and
`plan_model_preparation`, then asks that backend to bind the neutral plan to
its associated load options. The resulting `ModelPreparationPlan` owns
the resolved `ModelKind`, raw portable configuration, neutral tensor catalog,
validated load-policy route, and—when applicable—the already-opened
`safemlx_gguf::Checkpoint`. Only then is the plan passed to the selected
backend's `prepare_model` implementation.

Nested `config.json` model identity and lossless GGUF integer metadata are
resolved once in core. The MLX `family` module validates the resulting raw
configuration against concrete MLX architecture implementations; architecture
and runtime modules do not reach back through the public `api` namespace.

The MLX adapter's `loading` module is the sole materializer. It consumes the
plan, performs exact architecture/module binding, creates MLX arrays and
modules, and applies MLX quantization, residency, stream, mapping, and transfer
semantics. It does not call back into the public facade loader. GGUF headers and
metadata are not reopened: `GgufCheckpoint::from_portable` wraps the core-owned
checkpoint for payload conversion. Combined model/tokenizer loading uses the
same inspection and core plan; tokenizer and chat sidecars remain facade
concerns. Architecture-derived GGUF EOS metadata is read portably before the
plan is consumed. The facade never calls an MLX GGUF materializer directly.

`MlxBackend` owns both its execution stream and its weight-materialization
stream. Callers select MLX by constructing that backend and then use the same
`load_model(&backend, artifact, options)` function used by another backend.
There is no MLX-only loader that accepts loose streams alongside the artifact.
The MLX materializer also prepares any supported media processor and stores it
inside the opaque `MlxModel`; session creation moves it beside the executable
and cache. There is no post-construction processor injection path.

`MlxBackend` maps session creation to `MlxModelSession`. The opaque `MlxModel`
contains exactly one private MLX executable form: complete, pipeline stage, or
expert partition. `MlxModelSession` allocates and owns the matching
`backend::mlx::ModelCache`, `PipelineCache`, or `ExpertParallelCache` plus the
optional `MlxDistributedSession`. It is the sole architecture-erased prefill/decode
implementation for replicated, TP, PP, EP, and Cartesian combinations.
Complete and expert submissions are wrapped by `async_eval_with_event`;
pipeline submissions preserve the stage completion that retains transfers and
stage outputs. Both implement the same core exact-completion contract and
synchronize on early drop where required. Native exceptions are converted at
the facade boundary and never appear in core trait signatures.

`MlxParallelContext` is the thin MLX binding around one
`ParallelRankTopology`. It adds only the process-local `DeviceAssignment` and
MLX stream/device validation. `ParallelCommunicators` consumes the core
subgroup plans to construct native MLX groups; it does not recalculate
coordinates, membership, ownership, or preflight policy. Data-parallel
membership is already modeled in core and currently fails closed as an MLX
collective capability.

`MlxModelInput` owns cloned MLX array handles for every typed input part and its
metadata. Backend submission therefore owns text, image, audio, video, and
embedding inputs without placing MLX arrays or modality-specific tensor layouts
in core.

`MlxBackend` implements `TextGenerationBackend` by turning portable prompt ids
into `MlxModelInput`, submitting through `MlxModelSession`, applying the
portable allow-mask followed by the existing MLX `GenerationSampler`, and
returning `MlxTextToken`. Unconstrained decode consumes that token's array
directly without a host round trip. Controlled generation observes its token id
once because grammar state must commit the selected vocabulary id.
`MlxTextCompletion` retains both the model submission and the sampled-token
submission and implements the same exact-completion contract.
The GGUF generation example puts its encode/generate/decode loop in a function
generic over the backend; only model loading and device selection are MLX
specific.

`TextGenerationBackend::Prompt` is also the multimodal boundary. Generic
clients construct `MultimodalRequest` from decoded `RgbImage`, `Audio`, or
`Video` values and call the same `LoadedModel` method for every backend. The
portable decoded-media values, validation, ordering, and placeholder binding
are always compiled and require no media or execution feature. The
MLX `MultimodalPreparationBackend` adapter locates the processor owned by the
selected `MlxModelSession`, performs architecture-specific resizing,
normalization, feature extraction, framing, tensor construction, and placement,
then returns an opaque `MlxModelInput`. It fails closed when the selected model
has no processor or a required MLX materialization feature is disabled.
`mlx-media` selects the base adapter; `mlx-image` and `mlx-audio` select its
image and audio preprocessing dependencies. The resulting prompt enters
`PreparedChatInput::PreparedBackendInput` and the ordinary generic generation
loop. Raw `MediaInput`, `ProcessorInput`, processor access, and the former
duplicate MLX chat-placeholder composer are private implementation details
rather than a second caller API.

`MlxRealtimeBackend` maps the realtime contract to `MlxRealtimeModel`, MLX
codec-token arrays, Moshi temporal/depth cache state, `DefaultSampler`, MLX
random state, one selected stream, and an exact event retaining every output
array. A released core `RealtimeSession<MlxRealtimeBackend>` contains the
opaque MLX session and can be resumed only against the same artifact and
normalized execution identity. Another backend can implement the same model,
session, input, output, and completion operations without reproducing scheduler
state transitions or exposing its queue type to callers.

Speculative model execution uses the core `SpeculativeExecutor` contract. Its
input, target and assistant state, cache checkpoint, verification output,
logits, execution context, completion, telemetry, and error are opaque
associated types. The contract submits high-level prefill, proposal,
verification, and exact commit operations; it does not expose tensor
primitives. MLX implementations use `MlxModelInput`, `MtpExecutionStreams`,
architecture-owned cache/state values, and `MlxSpeculativeCompletion`.
Verification accepts portable token ids, materializes the MLX token array in
the adapter, and returns a `Submission` whose exact event remains owned by the
scheduler until resolution. Dropping an unresolved MLX completion synchronizes
it, including cancellation and failure paths.

The corresponding `SpeculativeSampling` contract is deliberately high-level.
Core asks a backend to process opaque logits, sample an opaque distribution,
decide one proposal, and commit an accepted token; it never asks for softmax,
indexing, random tensors, or residual arithmetic. `propose_block` and
`resolve_round` own proposal sequencing, stochastic accept/reject flow,
replacement and bonus bookkeeping, grammar and terminal precedence, and the
transactional state fork. `SpeculativeSchedule` owns bounded fair action
selection across requests. `SpeculativeRequestTable` owns the production
request collection, stable insertion identities, per-request resource slots,
cancellation scan, action application, fairness/peak accounting, prefill,
terminal finalization, and output ordering. Its generic request records hold
opaque backend caches, target/draft state, random state, and publishers; no
MLX type appears in the table contract. `PendingSpeculativeVerification` retains the exact
completion, verification output, cache checkpoint, canonical draft block, and
optional optimistic branch as one transaction. `resolve_commit_and_publish`
waits for that exact completion, resolves portable sampling decisions, commits
the backend cache, promotes or discards optimistic state, and only then exposes
committed tokens. `SpeculativeOutputRuntime` is the sole canonical sampler,
constraint, sequence, cancellation, and output-publication state. Its
`SpeculativePublisher` adapter is a semantic sink rather than a scheduler: it
cannot commit caches or choose which branch becomes canonical.

`MlxSpeculativeSampling` maps sampling operations to MLX arrays,
position-stable PRNG keys, target/draft streams, cross-device distribution
transfer, probability ratios, and residual sampling. Core's
`SpeculativeSemanticState`, `SpeculativeSemanticConstraint`, and
`SpeculativeCallbackPublisher` own transactional semantic forks, cancellation,
and callback draining after exact commit authorization. Their structured
`SpeculativeOutputError` is independent of the executor error; prepared-chat
parsing no longer passes through `safemlx::Exception`. MLX component timings
are returned as opaque executor telemetry and folded into the core-owned
`MtpStats`. The facade contains no parallel request table, action dispatcher,
in-flight transaction, optimistic promotion, cache-commit, callback ordering,
or telemetry state machine.

Prepared-chat speculative setup is facade orchestration, not backend work.
`LoadedModel<B>` resolves checkpoint sampling defaults, tokenizes a rendered
prompt through `B::prepare_text_prompt` (or accepts an already prepared opaque
prompt), validates semantic support, and constructs both
`PreparedChatSpeculativeConstraint` and the transactional
`SpeculativeSemanticState`. `SpeculativeGenerationBackend` receives only the
resulting execution-ready single request or batch lanes. A second backend can
apply the portable constraint's filters around its own logits math and pass the
semantic state to the core output runtime without reproducing dialect, stop,
parser-fork, or replay semantics. MLX converts the resolved sampling policy to
its array sampler and owns only execution, PRNG arrays, streams, and caches.

The MLX whole-session capability lives in
`backend/mlx/prepared_speculative.rs` and has no dependency on the facade.
It consumes the core execution-ready request, validates target/drafter
vocabulary identity, creates backend-owned sampling and cache state, and
dispatches one scheduler implementation in
`backend/mlx/speculative/scheduler.rs`. The scheduler's `generate_tokens` and
`generate_semantic` entry points accept every MLX `SpeculativeExecutor`.
Checkpoint-embedded heads are adapted by
`backend/mlx/speculative/embedded.rs`; external Gemma and Muse assistants and
Qwen's model-specific head math all use that same embedded executor. There are
no embedded-only or architecture-specific generation loops, forwarding
`generate_*` wrappers, or duplicate Qwen executor. The remaining executor
implementations differ only in model math, cache checkpoints, verification
materialization, and exact completion ownership.

The public core `load_model` route performs format, architecture, catalog, and
policy planning before calling the selected backend's `prepare_model`. Its
return type is `PreparedModel<B::Model>`; `Backend::create_session` consumes
that marker and the opaque model it proves was prepared by the chosen backend.
`backend::mlx::ModelLoadOptions::with_parallel` selects TP, PP, EP, or a
supported Cartesian materialization through this same entry point.
Architecture-specific distributed loaders and rank-local model types are
private materializers, not public alternatives. `MlxBackend::with_distributed_world` supplies the world;
`Backend::create_session` derives the topology-scoped communication and binds
it to the selected model and cache state. There is no standalone public
communication-session constructor.
Every backend-contract prefill and decode goes through `MlxModelSession`, whose
model, cache, and communicator cannot be extracted or replaced. The former
complete-cache accessors, per-architecture `ModelGenerate` enum, Llama-only
executor, raw facade cache pairing, and borrowing backend-session trait were
deleted. `LoadedModel` owns a `ModelRuntime<MlxBackend>` rather than a raw model.
Ordinary generation, prepared-chat generation, speculative single-request
generation, cache-policy selection, and prompt-cache persistence all use the
session-owned cache and selected execution placement. Prepared-chat MTP requests
contain only portable sampling settings, scheduler policy, cancellation, and
semantic callbacks. `SpeculativeGenerationBackend` makes capability discovery,
single-request execution, and fair batch execution operations on generic
`LoadedModel<B>`; its associated drafter type prevents cross-backend pairing.
The facade turns those caller requests into execution-ready requests with an
opaque backend prompt, resolved generation settings, a canonical grammar state,
and a transactional semantic parser before invoking the backend. The MLX
implementation is the production path. The target and `MlxDrafter`
retain the MLX streams selected when each was loaded; batch cache lanes are
allocated inside the MLX adapter. The caller supplies the drafter tokenizer to
the MLX loader, so vocabulary compatibility is exact without making the
backend call into facade metadata loading.
MLX speculative lane caches are adapter-owned and are no longer part of the
facade API. Starting an unrelated sequence is an explicit
`MlxModelSession::reset` transition; loading a prompt cache deliberately
replaces the same session state so the next prefill/decode continues that
prefix.

`LoadedModel<B>` has no public MLX-only inherent methods. Portable text and
prepared-chat operations remain on the generic facade. Native cache policy,
prompt-cache persistence, residency telemetry, media processor access,
activation observation, and allocator diagnostics live on
`backend::mlx::MlxModelSession` and are reached through `ModelRuntime` only
when an application intentionally selects MLX. Raw prefill/decode callers use
the generic `ModelRuntime` submission contract with `MlxModelInput`; the
facade does not unwrap outputs into MLX arrays or accept caller streams.

## Tensor and cache ownership

MLX arrays, streams, devices, events, executable layer modules, and concrete KV
cache tensors remain wholly in `safemlx-lm`. Core sees only associated opaque
types. The MLX pager consumes core `CacheBlockId` and `LayerCachePolicy`
contracts directly; it owns only the arrays, transfer completions, and physical
storage that realize them. `CacheBlockLifecycle` owns block registration, exact
leases, access history, protected prefixes, deterministic victim selection,
truncation replacement, and mutable tails. The generic core
`CacheBlockStorage` owns the physical phase protocol and holds opaque backend
device, host, backing, host-demotion, and I/O values. Its exact I/O keys bind a
direction and generation to one block; private resource slots make conflicting
phase/resource combinations unrepresentable. Neutral checkpoint descriptors
describe names, shapes, dtypes, and byte locations. Core validates SafeTensors
headers/index paths and GGUF shard catalogs without reading tensor payloads.
MLX weight stores remain responsible for mapped payload access, exact
architecture binding, and array materialization. `OffloadPlan`
is the only weight-residency plan:
budgets, tier assignments, eviction policy, transfer accounting, prefetch and
eviction telemetry, process observations, and allocator observations all live
in core. Its deserializer re-runs the same validation as programmatic
construction. `ResidencyLedger` is the canonical ownership state machine. It
admits complete batches before mutation, distinguishes reserved from published
copies, owns lease counts and named protected windows, selects deterministic
victims, and keeps failed exact transfers from remaining logically resident.
Every removal returns a typed copy descriptor that the backend must mirror by
releasing its storage.

The MLX residency manager embeds this ledger beside a storage-only map of
`Array` and immutable host-transfer-buffer owners. It computes physical MLX
allocation requirements, asks the ledger to reserve capacity, materializes and
publishes concrete storage, and releases exactly the copies returned by ledger
eviction or failed-completion transitions. MLX `Event` objects and retained
transfer sources remain in the adapter; their stable generations are allocated
and resolved by core. Allocator samples obtained by `backend::mlx::residency`
are recorded into the neutral telemetry schema.

Dense disk-to-host prefetching uses `PrefetchExecutionState` as its only
logical coordinator. Core owns the bounded FIFO and decides admission,
coalescing, ordering, cancellation, rollback, exact completion publication,
retry supersession, and demand-visible failure delivery. The MLX adapter owns
one worker thread and sends it only work-available or shutdown notifications;
the channel is not a second queue or capacity authority. For each item selected
by core, the worker invokes `ResidencyManager` to map SafeTensors payloads and
construct host storage, then returns the structured outcome to the same exact
core work ticket. A different backend can reuse every transition and supply a
different materializer, I/O queue, and completion mechanism.

Paged-cache backing I/O uses the separate core `CacheIoExecutionState`.
Exact read/write keys are prepared once, duplicate callers join that owner,
bounded admission is decided before the adapter channel is notified, and core
tracks queued versus executing cancellation through exact completion. The MLX
adapter retains task payloads, completion condition variables, filesystem
operations, SafeTensors serialization, mapped buffers, and cleanup. Its
channel transports already-admitted opaque tasks but no longer decides queue
capacity, coalescing, cancellation phase, or result publication.

## Coupling still present

`safemlx-lm-core::execution::ExecutionPlan` is the canonical execution-plan
document used by the automatic planner and CLI. It selects an extensible
backend/device identifier for the whole session, embeds the canonical core
topology, and contains only portable residency, transformation, drafting, and
required-capability policy. `safemlx-lm-core::automatic::AutomaticPlanner`
owns policy validation, budgeting, candidate selection, feedback matching,
explanations, and the planning/telemetry schemas. Its high-level
`AutomaticPlanningBackend` receives whole candidate plans and supplies only
hardware/resource observations, admission, bounded-window requirements, and
embedded-drafting metadata. The MLX implementation lives in
`backend::mlx::automatic`; it translates CPU, Metal, and CUDA devices into
`mlx` identifiers, validates candidates with MLX checkpoint inspection, probes
bounded loads, and realizes a selected plan as `ModelLoadOptions`.
Process-global MLX allocator cache configuration is deliberately not part of
the neutral plan.

Capability and inspection follow the same direction. Core owns `Observed`,
`ModelCapabilities`, `StateLayout`, `RuntimeStateEstimate`, admission policy,
and `ModelInspectionReport`. Core's `ModelCapabilityBackend` extension binds
capability discovery, prepared-input counting, runtime-state estimation, and
static-memory observation to the selected `ModelRuntime<B>`. Its signatures
contain only portable reports and the backend's existing opaque prompt type;
native tensors, streams, allocators, and caches never enter the facade API.
Consequently the same `LoadedModel<B>` methods perform token counting,
estimation, memory reporting, and admission for every implementing backend.

`backend::mlx::capability` maps validated MLX model configuration and prepared
MLX media shapes into the portable values, fixes the current MLX cache scalar
width, and supplies native allocator/memory observations from the selected MLX
session. Distributed session shapes for which MLX cannot yet derive a complete
capability document fail closed rather than manufacturing partial facts.
`backend::mlx::inspection` performs MLX loader binding, tensor-catalog
validation, quantization/residency admission, and MLX media-companion checks.
`MlxInspectionOptions` and structural `inspect_model` therefore live under the
selected backend. The facade's `inspect_text_model` step enriches the same core
report with tokenizer, chat-template, EOS, semantic-streaming, and native-tool
readiness. A second backend reuses that text step and the core reports and
policies while providing its own structural derivation and admission
implementation.

Telemetry follows the same layering: MLX converts concrete residency and
expert-cache snapshots into core telemetry documents, while the caller or
facade obtains those snapshots from its loaded session. The adapter no longer
accepts a facade model merely to reach backend state.

Automatic planning documents now use schema version 2. This deliberately
breaks the former facade-owned schema: resource profiles use the canonical
core `ArtifactFormat`, and backend allocator fields are described generically.
No version-1 compatibility decoder or duplicate facade schema is retained.

The current boundary leaves these components MLX-coupled:

- architecture tensor execution, including multimodal preprocessing and
  realtime model math (portable decoded media, ordering, placeholder binding,
  and the session preparation capability are core-owned);
- MLX exact-completion objects, retained output arrays, and event-backend
  telemetry adapters;
- MLX device assignment, concrete communicator construction, collective tensor
  math, and movement inside `MlxDistributedSession`; the portable session
  capability, topology/rank descriptor, scopes, operation availability, and
  exact-completion contract are core-owned. Replicated, tensor-, pipeline-, and
  expert-parallel loading, prefill, decode, cache ownership, and sampling
  synchronization all cross the architecture-erased model session. The
  architecture executors still manipulate MLX stage values internally, but
  they are backend implementation details rather than alternate model/session
  lifecycles. The former MLX pipeline request scheduler and partial-stage
  execution, sampling, observer, and cache-inspection surfaces were deleted;
- per-block MLX cache resources, native completion observation, physical
  transfer/disk worker execution, prompt-cache filesystem publication, safetensors payload
  mapping and materialization (container/header validation, the transition
  protocol, block registration, exact leases, access ordering, protected
  prefixes, victim selection, mutable tails, prompt identity, topology,
  manifests, catalogs, fingerprints, compatibility, block identity, tier
  vocabulary, layer geometry, fixed-state policy, and pure validation are
  already core-owned);
- weight array/host-buffer materialization, native transfer events, retained
  source mappings, physical-capacity queries, and concrete disk-to-host worker
  execution (the corresponding ownership, bounded FIFO admission, coalescing,
  cancellation, failure recovery, eviction, window, lease, and generation
  state is now in core);
- architecture-specific checkpoint binding/weight recipes, mapped payload
  stores, GGUF decoding, and MLX array materialization (artifact detection,
  model-family resolution, neutral catalogs, and route planning are core-owned);
- logits filtering, probability and residual math, random tensor generation,
  architecture tensor execution inside the portable speculative-executor
  operations, assistant and MTP model math, concrete cache mutation, activation
  observation, MLX allocator sampling, and Metal/CUDA kernels. Speculative
  proposal sequencing, stochastic acceptance flow, replacement/bonus
  bookkeeping, optimistic-prefix reuse, fair action selection, cache-retention
  counts, cache-commit/publication ordering, request telemetry, token budgets,
  and terminal precedence are core-owned.
- tokenizer/chat-template rendering and checkpoint generation metadata remain
  in the facade, but are owned by generic `LoadedModel<B>` and do not name MLX.
  Plain text and prepared-chat tokenization, greedy/stochastic sampling,
  portable grammar filtering, prefill, decode, semantic events, cancellation,
  stop/EOS precedence, and decoding now have one backend-generic client
  surface. Multimodal request orchestration is generic through the opaque
  backend prompt. Decoded media descriptions and validation are portable,
  while tensor preprocessing and materialization remain backend
  implementations. Realtime loading and request/session orchestration are
  generic, while codec-token arrays and Moshi/PersonaPlex model math remain MLX
  adapter types. Prepared-chat speculative requests no longer expose samplers, random
  arrays, streams, or cache objects, and their drafter field is generic. The
  public methods live on `LoadedModel<B>` behind the production-used
  `SpeculativeGenerationBackend` capability. `MtpCapability` and
  `MtpCheckpointKind` are canonical core schemas. Raw tensor-oriented MTP APIs
  remain explicitly MLX adapter APIs; the remaining coupling is the concrete
  logits/sampler/executor implementation beneath the capability.

The former facade `runtime::residency::policy` module was deleted. It was not
retained as a forwarding namespace; MLX residency execution now lives under
`backend::mlx::runtime::residency`. The earlier placeholder core
`ResidencyPlan`, `ResourceSpec`, and `ResidencyReport` schemas were also deleted;
the validated `OffloadPlan`, `OffloadUnitSpec`, and `OffloadReport` types used by
production are now canonical and are reexported at the facade root.

The former logical-copy records, byte counters, recency/frequency counters,
window maps, blockers, eviction selection, lease accounting, and transfer
generation state were also deleted from the MLX residency manager. They were
not retained as forwarding wrappers or shadow state. `ResidencyLedger`,
`ResidencyLedgerError`, `ResidencyBlocker`, and `UnitResidencyReport` in core
are the sole definitions used by production.

The aggregate cache pool is canonical core state as well. The former facade
pool limits, resource axes, usage map, reservation map, manager membership,
admission checks, and high-water accounting were deleted. The MLX pager now
registers a manager through the core pool, publishes its concrete occupancy,
and holds core reservation tokens across asynchronous host transfers and live
disk operations. Pool errors cross the adapter as a transparent structured
source.

The per-session `CacheBlockLifecycle` is canonical core state too. The MLX
storage record no longer contains lease counts, recency/frequency counters,
protected-prefix flags, or mutable tails. Every production seal, import,
attention lease, discard, clear, truncation, report, and eviction decision uses
the core catalog. Truncation consumes its exact crossing-block lease as part of
the atomic replacement instead of relying on a later saturating release.

The former MLX physical-state, host-persistence, disk-read, and disk-operation
identity types were deleted. `CacheBlockStorage` is instantiated with MLX
arrays, typed host buffers, disk locations, and adapter completion tickets.
Every production promotion, demotion, read, write, rollback, cancellation, and
direct backed release calls its checked transition methods. The adapter retains
only native completion observation, resource accounting, worker scheduling,
and filesystem operations.

The former facade definitions of cache block identity, logical tiers, layer
cache geometry, and fixed-state policy were deleted as well.
Architecture modules import the canonical core contract directly. MLX manifest
and array validation call core policy validation, symbolic shape resolution,
and dtype-family matching; `backend::mlx::runtime::cache::residency` binds
them to MLX storage without redefining or forwarding the core types.

Prompt-cache identity, topology, descriptors, options, versioned manifests,
catalog entries, fingerprints, and structural/compatibility errors likewise
have a single definition in core. Architecture implementations construct those
types directly. The MLX adapter binds the portable `ParallelRankTopology` to a
process-local MLX device and owns only arrays, native groups, mapped shard storage, payload hashes,
safetensors I/O, and atomic filesystem publication. It invokes core validation
before accepting or publishing a catalog; there is no facade copy of the
catalog validator or its error taxonomy.

The neutral scheduler in core now owns both production single-rank realtime and
distributed pipeline request lifecycles. Queueing, fairness, deadlines,
branching, commit/discard, cancellation, abandonment, capacity accounting,
poisoning, and telemetry execute in `safemlx-lm-core`. The facade supplies only
stable work descriptors, opaque session branches, MLX submission closures,
exact MLX completion observation, and retained arrays. There is no second
facade scheduler implementation.

Distributed scheduler consensus is also backend-neutral. Core owns protocol
framing and fail-closed validation for schedule descriptors, cancellation and
deadline dispositions, work identities, and exact completion observations.
`ConsensusTransport` exposes only a participant count and topology-scoped
all-gather of portable `u32` words. It contains no tensor, stream, device,
group, or MLX error type.

Distributed scheduler adapters receive one `MlxDistributedSession` from the
selected `MlxBackend`. The session owns the world and topology-derived
TP/PP/EP communicators and implements `ConsensusTransport` directly. It
materializes portable word frames as MLX arrays, runs `all_gather`, waits for
the exact completion, and returns rank-major words to core. The same selected
session is the only public route for pipeline/expert execution, sampling,
point-to-point transfer, and scheduler cancellation consensus; the former
`CartesianExecution`, direct-world pipeline methods, and standalone
`MlxConsensusTransport` were deleted. Core decides whether work is globally
incomplete, complete, failed but still executing on a peer, or failed and safe
to release. Backend errors are
converted at the adapter boundary and consensus mismatches poison the canonical
core scheduler before any new model-session submission. Prepared branches are
explicitly discarded during poisoning; submitted MLX resources remain retained
until their exact completions resolve.

Cache policy changes and prompt-cache publication/restoration operate on
`MlxModelSession`. The MLX implementation of generic prepared-chat speculative
generation also uses that session's model, placement, and backend-owned cache.
Callers cannot extract any complete, pipeline, or expert cache, replace it with
a foreign cache, or combine a model with a different communicator. Architecture
cache types and rank-local stage models remain private MLX implementation
details.

The former facade-only `FairScheduler`, `CompletedWork`, and `FailedWork` APIs
were removed instead of retained as wrappers. `Scheduler`, `SchedulerProgress`,
and `SchedulerError` are the only lifecycle types and are reexported by the
facade.

Ordinary and speculative text generation share one core-owned
`GenerationSequence`. It is the sole committed token history and applies stop
sequence, grammar, EOS, token-budget, and cancellation precedence. The MLX
streaming adapter decodes and publishes each committed token but no longer owns
a parallel token vector or finish-reason state.

The facade preserves backend identity in its public failures as well as its
values. `PreparedChatError<B::Error>` carries submission, exact-completion, and
token-extraction failures as the selected backend's concrete error type;
constraint, tokenizer, semantic-streaming, and lifecycle failures are separate
portable variants. The committed-token driver likewise keeps source and decoder
errors typed instead of formatting them into strings.
`PreparedChatMtpError<B::Error>` similarly separates facade tokenizer,
configuration, constraint, and semantic setup from the selected backend's
target/drafter execution failure. Tokenizer, chat-template, and
sidecar loading use `TextMetadataError`, so generic `LoadedModel<B>` assembly
does not mention the MLX facade error. `LoadedModelLoadError<B::Error>` also
separates portable artifact inspection from backend materialization instead of
requiring backend errors to absorb artifact failures. MLX converts portable
constraint failures to native exceptions only where its tensor sampler
contract requires one.

For speculative decoding, `SpeculativeRound` records proposal acceptance, a
replacement or bonus tail, and whether the round terminated. Its commit plan
determines both the tokens that become visible and the exact number of target
verification inputs retained in the MLX cache. MLX still computes logits,
samples target and draft distributions, mutates cache tensors, and owns exact
completion. The scheduler publishes staged semantic events and tokens only
after that cache transaction commits. Optimistic lookahead uses the pure core
prefix/reuse decision before promoting or discarding MLX-owned draft state.
The encompassing model-execution lifecycle is also backend-neutral:
`SpeculativeExecutor` owns the prefill/propose/submit/observe/commit boundary,
while its MLX implementations retain concrete arrays, streams, assistant state,
cache transactions, and events. `SpeculativeSampling`, `propose_block`, and
`resolve_round` make the sampling algorithm backend-neutral while keeping
opaque distribution math in the adapter. `SpeculativeSchedule` selects the
same bounded fair actions for any backend. The core request coordinator owns
the request table, opaque per-request resources, action application, retained
verification resources, optimistic branch promotion/discard,
cache-commit-before-publication ordering, cancellation at an exact safe
boundary, stable output ordering, and speculative request/scheduler telemetry.
The former facade scheduled-request types, request vector, acceptance loop,
fair selector, action methods, probability helpers, request transaction,
optimistic transition, callback coordinator, telemetry definitions, and closed
executor-specific prefill and commit records were deleted rather than retained
as wrappers.

Checkpoint sampling recommendations, request overrides, resolved sampler
settings, MTP configuration, scheduler limits, request identity, and request
phases likewise have one canonical definition in core. The MLX adapter maps a
validated resolved configuration to its concrete `GenerationSampler`; it does
not repeat validation. `MtpRequestLifecycle` validates every prefill, draft,
submission, resolution, completion, and cancellation edge; cancellation stays
pending while an MLX verification retains resources and becomes terminal only
after that transaction reaches its exact safe boundary.

## Adding IREE or native Slang later

1. Add a backend crate or facade module depending on `safemlx-lm-core`, never on
   the MLX adapter.
2. Define opaque executable/model, session/cache, input/output, and completion
   types. Map native failures into structured backend errors.
3. Implement capability and device discovery with every unsupported capability
   defaulting to false.
4. Implement whole-model preparation and whole-session creation.
5. Implement prefill, decode, and exact completion; retain submitted resources
   until completion even after cancellation.
   Implement `SpeculativeExecutor` as well when the backend supports assistant
   or embedded-head decoding; choose backend-owned input, cache, checkpoint,
   verification, context, and completion types and materialize portable token
   ids only inside that adapter. Implement `SpeculativeSampling` with opaque
   logits, distributions, and random state so the core proposal and resolution
   drivers can be reused unchanged. Implement `SpeculativePublisher` as a thin
   output sink; do not repeat transaction, promotion, cache-commit, or callback
   sequencing outside core. Instantiate `SpeculativeRequestTable` with those
   opaque types instead of creating a backend-local request collection or
   action dispatcher.
6. Implement `DistributedBackend` only when the backend can bind communication
   to the complete selected session. Report `DistributedCapabilities`
   fail-closed, implement typed world/axis collectives and transfers with exact
   completion retention, and implement rank-ordered scheduler-word gathering.
   Do not expose native communicators as an application-level dispatch path.
7. Pair concrete weight storage with `ResidencyLedger`: reserve full batches
   before materialization, publish only complete copies, release every returned
   eviction descriptor, and resolve the exact generation attached to native
   transfer completion.
8. Register every live cache with `CacheResidencyPool`, publish concrete
   occupancy, and retain each admission token through the exact native
   transfer or persistence transition that owns it.
9. Reuse the generic client probes in `tests/backend_conformance.rs` for model
   loading, generation, capabilities and admission, multimodal preparation,
   speculative execution, and realtime scheduling. Run that suite without
   default features, then add backend-specific model load,
   architecture-erased prefill, multi-step decode, cancellation, checkpoint,
   and parity tests.
10. Add backend selection at the facade/application boundary. Do not dispatch
   individual tensor operations between MLX and the new runtime.
