# Language-model backend architecture

Attention arithmetic is a neutral request policy. An architecture can require
rounding QK products, scaled/masked scores and normalized probabilities to the
query dtype. Native operators implement those boundaries for contiguous,
sliding and paged storage. The paged implementation retains only global row
maxima/denominators between two bounded scans and accumulates rounded-probability
value products in FP32 before the output cast. Cache telemetry counts both
scans and their transfers; page size does not introduce extra output rounding.
Resident attention tiles query rows while complete key rows fit its bounded
score workspace, preserving the ordinary softmax and value-product reductions.
Larger key rows retain the two-scan bounded mechanism. Explicit rotary input
products retain inverse frequencies directly so positions multiply the same
rounded coefficients instead of dividing by reconstructed denominators.
The CPU realization uses complete FP32 matrix accumulations, cascade row sums
and a fixed softmax reduction with a pinned exponential polynomial. These are
generic native numerical operations, independent of checkpoint and family policy.
Metal realizes complete BF16 row and batched products, cascade normalization
and FP32 softmax/sigmoid with the same accumulation and rounding boundaries.
These operators select by device, dtype and geometry. Router cutoff ties use
value-only CPU partitioning of the small score/index tensors; expert parameters
remain on their selected device. Independent PyTorch fixtures verify the native
operations without importing architecture equations.
Grouped linear and gated-product specifications also declare their reduction
policy. Sequential group-order reduction sorts each token's selected global
group IDs and rounds every weighted addition to the output dtype. Compact-bank
and executable-format transformations retain this equation policy. CPU grouped
projection selects an expert with a unit slice, retaining its matrix strides and
avoiding an advanced-indexing copy that changes reduced-precision accumulation.
Native block-FP8 linear and grouped projections quantize activation blocks on
both CPU and GPU. The CPU realization independently expands the same E4M3 values
and scales before multiplication; checkpoint parsing remains portable.

Chat templates use Transformers-compatible Jinja whitespace and dictionary
construction. Rendered bytes, including leading newlines, remain artifact
policy. IFM reasoning and JSON/XML/typed-XML tools are facade declarative
protocols recognized from rendered probes and tokenizer properties. Separate
argument value tags, compact type annotations and literal value framing are
shared parser/constraint features. Partial annotation state is included in
snapshot storage estimates.

Tool-schema admission and completed-argument validation belong to the facade.
Application JSON Schemas are retained unchanged for rendering and checked with
the portable `jsonschema` validator without remote or filesystem retrieval.
Dialect grammars enforce the constraints they can represent and retain protocol
syntax, function names and call limits for other schemas. The shared semantic
event sink validates the complete argument object before emitting `ToolCallEnd`;
immutable validators are shared across forks while partial arguments remain
private to each parser. Backends continue to consume neutral token filters and
do not interpret schemas or decide which tool calls applications execute.

Tokenizer ID membership and model output width are separate domains. The facade
builds a closed validity mask from actual canonical, round-tripping tokenizer IDs,
including added and special tokens; holes and inconsistent mappings are excluded.
Raw token generation passes it to core's ordinary text machine via
`TextGeneration::with_token_filter`. Prepared text, semantic, observed, controlled and speculative
requests intersect the same immutable mask with their grammar controller. Forced
choices further restrict that intersection. The grammar and output parsers retain
their existing activation, EOS and termination behavior.

The facade exposes explicit `start_controlled_text` alongside strict
`start_controlled_chat`. Both consume the same admitted observed/intervened prompt
and share one ordinary execution driver. Text mode retains tokenizer validity,
incremental decoding, EOS and caller stops without a semantic grammar or profile.
Tool/thinking admission stays in facade preparation; backends receive the same
neutral filters and native execution requests. Text/semantic mode participates in
snapshot compatibility, with decoder and stop state covered by existing portable
copying and storage accounting. See [LM Inspector execution control](lm-inspector-execution-control.md).

`generate_prepared_text`, `generate_prepared_text_speculative`, and its batch
variant accept the same prepared-chat request types while selecting literal text
output explicitly. They share ordinary commitment/timing and the neutral
speculative scheduler with semantic chat; unknown templates need no protocol
parser to use TTFT or supported embedded/external drafting. The facade reuses
controlled text admission and decoder/stop state: tool declarations and required
calls are rejected, and explicit thinking requires `allow_unparsed_reasoning`.
Emulated tools supplied as ordinary prompt text remain application policy.
`generate_prepared_chat` and its speculative variants still require executable
semantic support. The CLI retains unrecognized prepared prompts and selects the
text methods, preserving EOS, caller stops, cancellation, and committed-token TTFT.

`eredu-core::TokenFilter` owns closed-set intersection and projection to an actual
output width: missing mask entries are forbidden, a shorter output uses the mask's
prefix, and an empty executable intersection is an error. Native samplers realize
these restrictions before their existing sampling policy; they neither reconstruct
tokenizers nor infer missing tokens. MLX uses negative infinity for rejected logits.
Low-level core callers that own no tokenizer may still explicitly use `All`.
Validity is shared immutably by controller snapshots and speculative forks; sparse
token IDs participate in the versioned vocabulary fingerprint used for compatibility.
A forced mapped ID outside a shorter model output fails when that output's width
is available, before sampling. No model-specific padding convention is assumed.

Architecture and capture discovery follows the same ownership boundaries.
`eredu-core::discovery` owns the versioned logical graph, physical decoder layer
groups, ordered execution passes, parameter-sharing declarations, symbolic axes,
declared observation catalog, and separate execution-support reports.
`ArtifactArchitecturePlan::architecture_descriptor` in `eredu-architectures`
projects admitted family configurations and existing component/parameter
declarations. Runtime traversal and routing collectors share typed path
declarations with discovery. Shared-stack projections retain distinct logical
execution nodes and capture paths, map them to physical layers and passes, and
declare exact weight aliases using the same architecture-owned mapping as
checkpoint lowering. These declarations do not imply shared mutable state or
native allocations. Architecture descriptor schema 4 is independent of the
unchanged observation catalog/support schema 1. `eredu-runtime` combines phase/conditional
requirements with selected session and collector facts; a backend reports only
native collection/conversion mechanisms. The facade exposes cold discovery and
enriches ordinary inspection reports. No discovery step requires native
resources. See [the consumer guide](architecture-discovery.md) for coverage,
limitations, and executable examples.

Bounded observation follows the same dependency direction. `eredu-core::capture`
owns immutable admitted plans, exact catalog selectors, symbolic request geometry,
budget comparison, native transform contracts, and versioned host records.
`eredu-runtime::capture` reserves before calling native transforms and owns the
one-step delivery ledger. Architecture preparation projects and retains catalog
semantics with the exact prepared source identity; backends consume that projection.
MLX implements eager slicing, previews and chunked reductions under its existing
submission/recovery owner. It queues no native capture handles. The facade composes
capture with `ControlledTextGeneration` and the ordinary committed-token semantic
driver: tokenizer, settings, EOS, decoding, cancellation and transport stay above
the backend. Physical allocator/private-workspace bounds remain explicitly
unsupported; logical capture storage is not a bound on total inference memory.
See [bounded capture](bounded-capture.md) for the accounting and delivery contract.

`TokenSamplingDecision` carries the exact sampling filter plus optional borrowed
tokenizer provenance across the neutral controller/backend boundary. Facade
constraint owners supply the intersection once; runtime forced-choice wrappers
retain its pre-override domain while narrowing only the sampling filter. MLX
candidate transforms consume that borrowed domain for Original and Effective
records. The speculative runtime exposes the same decision to the raw-logit
capture callback before applying its filter, so neither native path reconstructs
grammar policy or queries a second mask. Unknown provenance stays explicit in
the optional host summary; filters are not retained in capture or snapshot state.

`eredu-core::intervention` owns the versioned host-only operation schema, exact
activation payload validation, routing request validation, session/source-bound
immutable admission, and attributed outcomes. An intervention declaration is
separate from the observation catalog: observing a value does not authorize
mutation. Architecture declarations own targets, routing policy and score stages.
`eredu-nn::routing_intervention::execute_routing_intervention` owns the routing
action/stage sequence over `RoutingMechanism` primitives; it evaluates the router
projection once and validates effective decisions before provider dispatch.
`eredu-runtime::intervention::apply_activation` owns activation operation dispatch,
exact selected-region validation and update order over `InterventionBackend`
primitives. No native primitive receives the public intervention action enum.
Runtime mechanisms also own scheduling, ordered composition, shared capture
accounting, session admission traversal, immutable installation and observer
forwarding through `CaptureObserver`. Facade composition owns ordinary generation,
session association and experiment reset policy.
Pre-dispatch controls travel through the neutral selector and observed expert
provider contracts. Architecture construction and discovery consume the same
selection specifications. MLX implements projection, configured score transforms,
top-k and coefficient mechanisms, native predicates, exact payload realization and
native slicing/arithmetic/updates; it has no intervention recipe or family catalog.
Backend `InterventionEstimator` facts supply indexing constraints, evidence costs
and additional original-decision resource estimates to the shared cold traversal
and exact runtime reservation. MLX estimates include group-mask dimensions and
host partition lookup storage; runtime contains no MLX-specific cost formula.
Unknown estimates fail, and requests without original evidence neither estimate
nor construct an original decision. Native prompt inspection, devices, typed error
conversion, completion retention and recovery remain with the backend. Shared
observers borrow the existing run owner and do not own native completions.
See [intervention plans](interventions.md) for exact operation and state semantics.

Completed-token control follows the same dependency direction.
`eredu-core::execution_control` owns versioned support/status/resource contracts
and the thread-safe pause/cancellation request handle. `eredu-runtime::execution_control`
also owns canonical choice intersection and prospective sampling validation.
`TokenChoiceController` retains the ordinary grammar and commits through it once;
the native sampler consumes a one-candidate decision under its usual RNG/adaptive
policy. `TextSamplingControlBackend` supplies exact sampling facts and atomic
native temperature/key installation. It does not validate portable override policy
or reset history. The facade owns tokenizer membership, bounded override records,
and explicit field-audited semantic storage estimates; immutable tokenizer data is
shared across decoder forks. Unknown opaque grammar costs disable snapshots.
The same runtime module
owns validated lifecycle transitions and retained-state reservations; cumulative
copying accounting lives outside rewindable generation state. The existing shared
`CaptureSession` owns opaque checkpoint/restore/fork of its schedule and record
lifecycle. Child capture and intervention admissions pass through shared validation
with the child's actual discovery, session identity and estimator facts. Native
completion remains a separate prerequisite supplied by the existing backend owner.
Core's ordinary text machine stores its continuation separately from its runtime
borrow. `TextGenerationDriver` advances detached continuations through that same
machine, validates exclusive driver identity, and requires native completion and
record draining between steps. It does not switch model state or own semantic
termination; facade composition coordinates those with the native state slots.
Facade `CommittedGenerationCursor` is the ordinary one-token commitment driver;
the one-shot loop delegates to it. Exact incremental decoder/parser forks remain
in the facade and do not replay token history. See [execution control](execution-control.md)
for the boundary and current implementation status.
The shared replicated session additionally exposes `ReplicatedTextSnapshotMechanisms`
for side-effect-free native estimates and isolated state copying. Its control-state
slots retain exact executable identity, typed state geometry, prepared-input identity
and commit metadata. The portable session validates and exchanges those slots for
serial branches without rebuilding the executable or assigning a second native
completion owner. Backend realization and complete facade snapshot composition are
separate from this low-level state-slot mechanism.
`NativeTextStateBackend` provides the narrow outer backend contract for those
native slots. MLX implements it through the same typed replicated session and
architecture-erased adapter, without family dispatch. Copy operations settle under
the existing session authority and recovery owner. Exchange performs a validated
host move while that authority is idle, so no fallible native submission follows
the swap. Input-identity and commit metadata are included in portable storage
estimates. Full generation capabilities remain a facade composition concern.

`TextSnapshotBackend` supplies complete native sampler/RNG and pending-input
copying beside that model-state contract. `TextContinuationSnapshot` in runtime
composes those mechanisms with an explicit `SnapshotTokenController` independent
copy contract and the existing shared capture checkpoint. Every component supplies
known logical costs before reservation. Restore stages all fallible copies and
capture validation before native exchange and infallible host installation; no
sampling or prompt execution is involved. `ManagedTextContinuation` keeps the
child retention lease with the logical child when native slots are exchanged.
MLX copies RNG keys and input arrays through its existing submission/recovery
owner and preserves standard/Mirostat history and absolute next prediction.
Runtime owns child re-admission, inherited ledger policy and snapshot accounting.
Native growth facts use the saved typed state and an admitted future input span;
runtime combines them with sampling/pending-input facts. MLX prices declared
component geometry, actual cache capacity growth and potentially absent fixed
state without executing input. Unknown growth rejects runnable branch admission.

Facade `ControlledGenerationSession` borrows the loaded runtime exclusively and
uses the ordinary committed-token cursor, constraints and semantic pipeline for
step/pause/resume. Its bounded sequence-numbered delivery shares compact-JSON
transport accounting with observed generation. Backend
`text_execution_control_support` is an exact opt-in fact. Full facade snapshots
are reported separately after complete semantic/grammar storage estimates and
non-resettable resource limits are established. Built-in semantic parsers have
explicit logical storage estimates; opaque active llguidance state remains
unsupported until it supplies complete costs. Snapshot/restore pairs native state
with exact cursor/parser/decoder state and a versioned consumer output checkpoint.
The generic `LoadedModel<B>` exposes these controls and snapshots without
exposing native handles or duplicating their policy. Reusable host/native continuation
checks live in `eredu-evaluation::execution_control`, used only as validation
tooling, with native forward/reload probes and source-preserving copy failures.
Controlled speculative generation uses a scoped `SpeculativeGenerationVisitor`
over the same `SpeculativeScheduler` and core request transactions as uninterrupted
speculation. Backend resources remain borrowed inside the controlling worker
closure. Runtime exposes scheduler phases and host proposal/verification records;
it does not duplicate proposal acceptance, residual sampling, cache rollback or
semantic publication. Facade request preflight and terminal-output construction
are shared with uninterrupted generation. All inspected paths use runtime's
`TraceBudget` bounded JSON counter.

Core speculative snapshot contracts preserve exact request identity, canonical
sequence, constraint/parser, sampler, both RNG streams, lifecycle and adaptive
statistics. Runtime reserves known snapshot costs and owns opaque handles and
monotone restoration epochs. Architecture external-assistant adapters preserve
seed tensor topology and cache identity; MLX supplies isolated native copies and
completion. Transaction rollback markers alone never imply reusable snapshot
support. Capture owners are shared across tentative sampler clones and restores,
so budgets cannot be multiplied or refunded. MLX raw prediction capture uses the
same bounded collector as ordinary generation, with target/draft roles attached
by the sampling boundary. Other activations require explicit phase attribution.

Every new inference feature must support controlled advancement through these
shared owners, or expose and document a concrete capability limitation. Tests
compare controlled and uninterrupted behavior, not repository layout.

Facade branch construction derives finite decoder/parser/history growth and pairs
it with runtime's native growth facts before reserving retention. Its child stream
starts with explicit parent metadata and the delivered canonical/semantic prefix;
native and shared capture adapters never reconstruct that output. Branch exchange
moves lifecycle, cancellation, semantic buffers and transport accounting together.
Opaque core driver identity prevents a later run on the same loaded executable
from importing an old branch tree. Shared capture admission and native completion
ownership remain in the neutral control API for every backend.

This guide defines the boundary between the portable language-model runtime
and an execution backend. It is intended for backend authors and maintainers of
`eredu`.

## Dependency direction

```text
eredu-core / eredu-checkpoint / eredu-nn / eredu-media
                           |
                     eredu-runtime ---- eredu-codec
                           |
                   eredu-architectures
                           |
                   eredu-backend-mlx
                           |
                         eredu
```

The neutral crates contain no native accelerator dependency under any feature.
`safemlx` is the native-binding dependency of `eredu-backend-mlx`. It owns safe
wrappers for MLX arrays, operations, devices, streams, collectives, graph
transforms, serialization, host-resource observations, and accelerator/runtime
handles. It owns no
framework abstractions: neural-network modules, quantization policy, mutable
execution state, GGUF materialization, and composed backend operations belong
to `eredu-backend-mlx`. This includes native distributed groups and axis-aware
gathers composed over MLX's native collective primitives;
`safemlx::distributed` exposes only MLX-native groups and communication calls.
Logical rank placement, communication manifests, routes, validation, and
realization-wave proofs belong to `eredu-runtime`; the MLX backend realizes
only the prepared native groups and routes.
Model-family construction and equations remain in
`eredu-architectures`; MLX-specific realization remains in
`eredu-backend-mlx`.
`eredu-codec` owns backend-neutral neural audio codec architectures, exact
released checkpoint schemas, parameter topology, layout recipes, and typed
artifact construction. It uses neutral tensor and runtime parameter contracts.
Concrete backends implement general checkpoint materialization, recipe,
completion, and binding mechanisms without depending on a codec family.
Application composition combines a prepared codec artifact with those generic
mechanisms and receives the ordinary neutral codec type.
`eredu-media` owns deterministic host audio, image, and video validation and
processing. Its image and audio implementations are optional portable
features. Architecture-owned processor requests select the semantics; a
backend converts the resulting owned buffers to native tensors.
The workspace forbids unsafe Rust in every package except the native `safemlx`
wrapper, its raw `safemlx-sys` bindings, and the `eredu-ios` C-ABI example.
Unsafe MLX and operating-system calls remain encapsulated by safe `safemlx`
APIs; the iOS exception is limited to its foreign entry points, callbacks, and
pointer ownership boundary.
`eredu-gguf` is likewise a backend-neutral storage dependency. Backends that
execute nonlinear GGUF IQ blocks consume their canonical values through the
typed `IQuantCodebook` API; generated table modules remain private and cannot
serve as an undocumented cross-crate integration surface.
The `eredu` facade is also portable when built with
`default-features = false`. Concrete adapter and native-platform features stay
optional; capability features configure an enabled adapter rather than making
portable contracts depend on it. APIs for target-specific runtime
configuration are available only on their native targets, even when Cargo
features are enabled elsewhere for build-matrix validation. The facade exposes
a flat, application-facing local adapter under `eredu::api`; it does not
reproduce an implementation crate's backend or composition module tree. The
MLX feature mapping and native build requirements are documented with
the [MLX backend](../eredu-backend-mlx/doc/README.md).

Backend-internal
fixtures and composition tests are crate-private unit tests; published crates
expose no test-support feature or fixture namespace. The facade does not
directly depend on `eredu-nn`, `eredu-checkpoint`, or `eredu-media`. Portable
media algorithms are enabled through narrowly forwarded image or audio
features; a concrete backend feature adds only native conversion. Backend feature diagnostics
name the active public spelling; published Cargo features are selectable API,
not a privacy mechanism.

The facade root exposes the `api` and `runtime` namespaces. `LoadedModel<B>`,
`PlannedModel<B, D>`, prepared-chat requests, and controlled generation sessions,
snapshots, and branches are the canonical application API for every backend.
Applications select MLX by importing `eredu_backend_mlx::MlxBackendFactory` and
passing it to the generic loading methods; inference supplies the backend type.
Explicit model annotations use `LoadedModel<MlxBackend<'static>>`. The facade
provides no wrappers or aliases that fix these generic types to MLX. Application
errors do not carry backend error type parameters. Loading, media preparation,
ordinary/speculative/observed generation, control, snapshots and branches retain
portable operation error variants with `eredu_core::BackendFailure` for native
failures. Lifecycle and local diagnostics use that same common failure type.
Token iteration and token-ID observation cross the facade boundary through
`TextGeneration` and `GeneratedToken`, preserving asynchronous execution and
completion ownership while translating errors. Realtime generation transactions,
frame execution and sampling replacement also expose portable failure types.
Backend implementation traits may retain concrete errors internally; application
callers do not need to name or match them.

The implementation crate exports composition-owned adapter factories at its
flat root and makes its reusable `backend` module tree public for backend
authors. Family composition and architecture-erased dispatch remain
crate-private. Raw native binding APIs retain their canonical `safemlx` paths.
Native facade integration tests use the same generic model and control APIs
as portable conformance tests, which use neutral traits and mock backends.

The facade does not re-export dependency-owned types. Architecture identities
and preparation plans come from `eredu-architectures`; artifact, execution,
generation, media, realtime, scheduler, and prompt-cache contracts come from
`eredu-core`; processed host media buffers come from `eredu-media`; tokenizer
contracts come from `eredu-text`; and runtime
infrastructure comes from `eredu-runtime`. This gives every public type one
canonical import path and prevents a facade release from committing to aliases
for operations it does not own. In particular, the facade exposes no
prompt-cache types because it exposes no prompt-cache operation.

Realtime applications use `PreparedRealtimeModel<M>`,
`RealtimeSessionScheduler`, and `ReleasedRealtimeSession` directly. The MLX
`create_realtime_execution` helper realizes an architecture-owned preparation
and returns the native context and execution mechanism. The generic model
retains the selected session identity; runtime scheduling retains admission,
transaction, completion, and host-publication policy.
Realtime `resume` takes a mutable optional released-session slot. Failure leaves
the slot untouched and returns a portable `RealtimeSessionError`; success takes
the state. This preserves retry/disposal ownership without carrying native state
in the public error type.

Application targets depend on `eredu` for facade operations, their selected
backend for factories, and the neutral crates whose public values they
construct. Backend dependencies remain feature-gated. The MLX facade helpers
provide device-plan creation, process runtime configuration, allocator
telemetry, and diagnostic benchmarks. Rich allocator and residency telemetry
methods remain specialized on `LoadedModel<MlxBackend<'_>>`.
Session lifecycle is portable: every `TextGenerationBackend` implements explicit
`reset_session` and `synchronize_session` hooks, exposed by `ModelRuntime<B>` and
`LoadedModel<B>` as `reset()` and `synchronize()` with the same non-generic
`BackendFailure` on every backend. Its portable `BackendFailureKind` distinguishes
busy authority, invalid/poisoned sessions, identified resource exhaustion,
unsupported or invalid requests, I/O and other backend failures. The original
error is retained as a boxed `Send + Sync`
source, so applications can handle portable kinds and log the complete cause
chain without importing backend error types. Adapters classify from typed facts
and session state, never by parsing diagnostic strings; unknown failures remain
`BackendFailureKind::Other`. These kinds do not establish settlement or safe reuse. Reset first
synchronizes, then clears request state while preserving loaded parameters,
admission, placement and retained drafting target identity. Success establishes
fresh-session behavior for the next request. Synchronization preserves request
state and succeeds only when all session work has settled, submission authority
is idle, and the session remains healthy. MLX waits for both weight and execution
streams and checks its retained recovery/authority state; queue completion alone
does not rehabilitate a poisoned session. Neither operation uses an implicit
successful no-op default for backends.

Generic applications can drop a generation iterator after cancellation, call
`synchronize()` to observe settlement errors, and drop the model for eviction;
`reset()` establishes fresh state before reuse. Iterator destruction waits for
retained completions but cannot report errors. A synchronization/reset error is
not evidence of completion or safe reuse: native resources remain owned until
safe completion or teardown. Live detached submissions may be rejected as busy.
These operations do not promise an allocator-cache flush or immediate
process-wide memory reclamation, so native telemetry is not required for the
portable reuse/eviction protocol.
Backend adapters own native resources and outer type erasure, while shared
replicated-text execution and stateful lifecycle orchestration remain in the
neutral runtime.
`LocalLoadOptions` and `LocalInspectionOptions` contain only neutral
quantization, residency, and session-capability policy, while native
device-bound contexts are selected only by backend tooling. `BackendFailure`
retains operation context and the original typed source without exporting native
error variants in application signatures. Portable execution plans remain the application surface
for device and topology selection.

Backend-generic sampling policy lives in `eredu-runtime`. Concrete backends
implement `SamplingBackend` primitives and bind the runtime-owned `Sampler` and
`SpeculativeSampler` traits directly; they do not publish backend-local policy
traits or compatibility re-exports. Concrete cache storage, completion, and
model-input types stay in the selected backend. A replicated-text session is a
neutral generic session over those concrete mechanisms; its backend adapter
erases the completed session only at the outer boundary. Facade examples
construct local sessions through `eredu::api`, while backend-author probes that
manipulate native resources live with their implementation and depend downward
on neutral contracts.
All selected sessions expose completed outputs through
`BackendSession::observe_output`; backends retain native tensors internally and
materialize portable `ObservationSet` records only when explicitly requested.
Device discovery reports only device-scoped `DeviceCapabilities` (completion,
transfer, and collective support). Cache, output-observation, and activation-
inspection support belong to `SessionCapabilities` on the exact prepared
model/session. Model loading derives that report from header inspection,
residency policy, and topology and rejects unmet session requirements before
checkpoint payload materialization. The admitted report is carried through the
preparation marker and must equal the realized session report.
Portable execution-plan factories use the same ordering for native resources.
The factory first translates the plan and selects the exact backend preparation
against the inspected artifact without creating a device, stream, queue, or
communication resource. The neutral selector validates the normalized request
once and returns its unforgeable admission; core builds the preparation plan
directly from that retained admission without asking the backend to derive the
policy again. Only the retained selected target may then be passed back to
the factory for native realization. Materialization consumes that retained
selection instead of repeating route selection after native resources exist.
Core carries the neutral preparation plan and backend selection in one opaque
`SelectedModelPreparation`; callers cannot re-pair either half before
`ModelLoadingBackend` consumes it.
External drafting follows the same two-stage boundary: the facade supplies the
architecture-inspected assistant to the factory's cold drafting-selection hook
before target realization. That hook establishes architecture-owned
target/assistant compatibility and selects the neutral speculative realization
against the retained target inspection before target payload or native-resource
creation. The retained result contains every exact
physical source, executable format, lowering descriptor, and reader-cache
mechanism setting consumed later by assistant materialization. Target or draft
streams therefore cannot exist when assistant selection fails.
Backends may additionally implement `InspectableBackendSession` by binding the
named activation and routed-expert points already emitted by
`eredu-runtime::ActivationObserver`. These are general diagnostics contracts
used by telemetry, inspection, observability, and evaluation rather than an
evaluation-specific backend surface. `ActivationObserver` is also an
intervention contract: every named activation returned by instrumented
execution must include any replacement it supplies. Family adapters and
topology executors finalize logits through
`eredu_runtime::observe_model_logits`; pipeline execution does so on the
logits-owning rank before submitting its completion. Realtime applications
likewise exchange portable host token frames through the facade. Runtime ingress
validates their exact batch and token geometry before a backend conversion
mechanism can create native tensors. Runtime prepublication waits for the exact
backend completion, performs host observation, and commits the model, schedule,
history, samplers, randomness, and output as one transition. Realtime selection
validates the exact loaded route's fail-closed mechanism and observation
capabilities before checkpoint payload materialization. A realtime route must
not advertise a named observation or intervention unless its architecture
execution exposes the corresponding seam.
Distributed inspection is rank-local: every rank participates in the same
production collective and point-to-point execution, each rank returns only the
globally named units it owns, and only the logits-owning rank returns
`model.logits`. `eredu-core::MODEL_LOGITS_OBSERVATION_PATH` is the canonical
owner of that semantic path; family composition, distributed pipelines, and
architecture-erased sessions must use it rather than defining backend-local
names. Inspection never performs an implicit cross-rank host gather.

Backend-neutral parity, distribution metrics, timing summaries, evidence, and
evaluation drivers live in `eredu-evaluation`. Concrete backend examples only
select execution contexts and supply generic mechanisms for neutral model and
codec artifact construction; they do not own checkpoint interpretation,
comparison thresholds, or reference policy. Backend-specific validation
inputs and reference-fixture generators live with the backend that consumes
them, so every published backend package contains the tooling required by its
examples; they do not live under the facade.

## Ownership boundary

Portable crates split tensor-independent ownership by responsibility:

- `eredu-checkpoint` owns canonical SafeTensors index parsing, exact
  index-to-shard-header validation, shard-path admission, and stable
  filesystem-backed content fingerprinting. Inspection and
  conversion tooling consume strict discovery, while neutral stores consume
  the same parsed and admitted catalog but validate each opened payload header
  exactly and lazily, so selective loads detect every mismatch in a local shard
  without reading remote-only shards. None reinterpret `weight_map`.
  `PreparedCheckpointSource` binds exact catalog, provenance, admitted shard
  identity snapshots, selections, and lease geometry without retaining a file
  descriptor per shard. After complete recipe and destination preflight, the
  requested shard is reopened and validated, and matching reads admit only the
  exact selected ranges; immutable cached bytes back every published lease;
- the canonical in-memory content identity value and logical-member combining
  algorithm, header inspection, the
  model-configuration resolver contract, tensor catalogs, preparation plans,
  backend mechanism-capability admission, and reusable inspection-report
  assembly live in `eredu-core`;
- portable host audio, image, and video processing lives in `eredu-media`;
- validated attention schedules, parallel topologies, model capabilities,
  resource requirements, execution plans, and telemetry schemas;
- generation budgets, committed-token order, finish reasons, cancellation,
  semantic output events, and constrained-token filters;
- speculative proposal, verification, acceptance, rollback, and publication
  state;
- scheduler fairness, work lifecycle, capacity accounting, and transactional
  commit or discard;
- weight and cache residency policy, accounting, eviction, leases, transfer
  generations, and I/O coordination;
- prompt-cache identity, manifests, compatibility, and catalog validation;
- decoded image, audio, and video values plus ordered multimodal requests; and
- distributed scopes, topology membership, consensus messages, and operation
  capability descriptions; and
- canonical parameter-binding plans, generic module binding, logical tensor
  placement, communication-manifest validation, task partitioning, report
  aggregation, and reusable session construction live in `eredu-runtime`.

The facade owns text-tokenizer reconstruction, chat-template interpretation,
and generation-termination policy, including EOS metadata from model sidecars
and GGUF headers. Backend weight inspection and materialization do not parse
tokenizer or EOS policy metadata: malformed or unsupported facade-owned policy
cannot reject an otherwise valid weight artifact.
Architecture admission derives vocabulary geometry exclusively from
family-owned structural fields and the admitted tensor catalog; tokenizer token
lists never override that geometry. When GGUF media protocols identify framing
or placeholder tokens by spelling, inspection retains an unresolved typed media
plan. After reconstructing the tokenizer, the facade resolves those spellings
to numeric IDs and binds them before backend preparation. Missing or malformed
tokenizer metadata can therefore fail tokenizer-aware facade loading, but not
structural weight admission.
Public chat preparation contracts live under `eredu::runtime::chat`.
Committed-token streaming and constrained-decoding machinery remain private to
the facade, and constrained-decoding failures are exposed only through the
application-facing `eredu::api::ConstraintError` path.

For an external speculative assistant, the facade reconstructs both target and
assistant tokenizers and establishes the neutral `TokenizerCompatibilityProof`
before assistant materialization. Equality of their token-id vocabulary
fingerprints is portable compatibility policy. Architecture composition owns
target/assistant compatibility and selects assistant placement before native
construction. Backend factories receive those proven contracts, materialize
only the selected assistant tasks, and bind the selected native placement; they
decide neither tokenizer nor architecture compatibility.

`eredu-runtime` owns statically dispatched resident and bounded execution,
the singular declarative parameter-binding and logical placement plans,
generic module binding, residency, mutable state, exact completion, neutral
communication preparation, reusable session construction, and
generation-facing causal-model contracts. Backends retain native recipe
lowering, tensors, storage, streams, completions, groups, transfers, and
collective scheduling.

`eredu-architectures` owns model-family configuration, checkpoint contracts,
parameter topology, module construction, state geometry, parallel semantic
plans, and the complete embedding/layer/output lifecycle. Architecture code is
generic over `NeuralBackend` and passes backend-native tensor handles through
unchanged.

Portable preparation admission keeps architecture requirements, admitted
artifact facts, caller policy, and backend mechanism capabilities as separate
inputs. `eredu-core` intersects them once into an immutable retained admission
or a structured rejection before payload access and native resource creation.
Inspection and realized sessions assemble capability and readiness reports from
that same retained admission; a backend may discover support facts but does not
recompute the portable decision after resources exist.

Replicated text composition has one checked cross-crate construction flow and
one backend-neutral execution/session implementation. `eredu-architectures`
derives `ReplicatedTextRequirements` from the normalized architecture and the
exact admitted artifact after selecting the replicated execution class. The
requirements contain the execution graph and unit layout, group transport,
complete state layout and typed access profile, canonical parameter topology,
admitted physical sources and encodings, aliases and derivations, logical and
physical shapes, ownership, presence, architecture-native executable formats,
and exact transform constraints. Transform constraints include the packed
axis, linear input extent, affine group size and bit width, or the physical
block geometry needed by the requested format.

Caller-selected topology, weight residency, mutable-state residency, load-time
transformation, cache, observation, persistence, and completion facilities
live in a separate `ReplicatedTextSelectionRequest`. A concrete backend reports
only reusable neural, lowering, residency, state-storage, session, and
completion mechanisms through `BackendMechanismCapabilities`. Mutable-state
capabilities enumerate exact layer components with their declared semantic
role, shape, dtype, and supported device or paged placement, plus checkpoint,
rollback, reset, persistence, and observation facilities. They contain no
architecture-family or execution-class identity. The neutral selector resolves
requirements, request, and capabilities into one
`SelectedReplicatedTextRealization` before architecture modules or weight
payloads are constructed. Its opaque
`SelectedStateRealization` records the accepted layout, typed access profile,
per-component placement, residency policy, and lifecycle facilities.

Each architecture-derived parameter requirement also retains exact physical
provenance: the admitted catalog key, canonical shard, physical tensor
identity, selected logical output, source encoding, and encoded byte extent.
Recipe output is an explicit derived-source encoding rather than a fabricated
SafeTensors container encoding. This distinguishes multiple outputs converted
from one physical tensor and prevents a materializer from rediscovering shard
membership or reinterpreting the source after selection. Validated parallel topology fields are private;
callers construct them through checked constructors and consume named
accessors. Preparation is consumed as an opaque plan through named policy,
route, inspection, and artifact operations rather than a positional tuple.

Architecture preparation pairs that selected realization with a concrete
architecture and validates the graph, unit and transport layout, parameter
owners and shapes, state layout, lifecycle facilities, and cache-identity
inputs before a backend mechanism can allocate state or open a payload. The
prepared value contains an opaque neutral contract carrying the validated
selection and exact tasks. That contract is the proof-bearing handoff to
construction; later code does not repeat those checks from family
configuration or caller options.

The runtime-owned `NormalizedLoadRequest` is the single portable cold-load
policy. `NormalizedLoadRequest::from_execution_plan` derives transformation,
ordinary and independent-bank residency, exact reader-cache limits, session
requirements, and drafting intent from one portable `ExecutionPlan`. Invalid
plan structure and policy produce structured portable errors before backend
capability selection or native work. Distributed plans supply one complete
parallel request whose topology must equal the plan. Diagnostics choices
request backend/process memory observations without invoking those mechanisms
during normalization. Reader-cache limits remain exact even for resident loads.
Its checked parallel subrequest atomically binds rank topology, wire,
positive invocation limits, and bounded completion; drafting capacity is also
positive by construction. A concrete adapter may pair that request with a
native resource token, but cannot add a second policy representation. The MLX
adapter validates its `DeviceAssignment` pairing and consumes the paired
request before neutral selection completes. Total cold branch selection lives
in `eredu-architectures::preparation_selection`; MLX supplies only a
side-effect-free `PreparationMechanismProvider` and consumes the resulting
opaque `SelectedExecution` through its typed materialization dispatcher.

`eredu-architectures::select_preparation` is the singular total cold selector.
It accepts that normalized request and a side-effect-free mechanism provider,
then performs admission, grouped-operation checks, processor and prediction
selection, ordinary or partitioned execution dispatch, communication
admission, and architecture production-route validation as one operation. Its
opaque `SelectedPreparation` covers replicated, routed, composite,
partitioned-dense, partitioned-routed, and partitioned-composite execution.
Concrete backends consume the selected branch through a typed static
dispatcher; they do not mirror the sum or rebuild any request.

Architecture inspection retains validation results with the immutable architecture
and the exact core admission token. Clones share the target projection and the
replicated, routed, or composite requirements, including expert topology, recipe
geometry, and source ownership proofs. A new admission or semantic enrichment
uses a separate set of results. Preparation retains both successful selections and
rejections for each normalized request and the actual neutral mechanism facts
consulted by that selection; changed backend facts require a new selection. These
results are local to the admitted artifact's lifetime, not a filesystem or global
model cache. Retained composite source snapshots do not own their enclosing
validation storage, avoiding ownership cycles.

Automatic planning retains each exact candidate's admission and bounded geometry
through feedback and final selection. Loading reuses the architecture's selected
proofs. Prepared expert providers consume the validated catalog and route contract;
their native bank binding checks remain a separate validation of newly materialized
resources. Public constructors accepting raw plan/catalog parts still establish
those proofs themselves. Expert identities, parameter targets, companion names,
and layout discriminators use indexes, and selecting all members of a stacked
expert recipe traverses the stack once. Checkpoint resolution consumes the layout
selected by catalog validation instead of repeating discriminator validation.
Runtime-owned `AddressableBankTask` retains an immutable validated materialization
task and its source index. Expert members share that proof and validate only their
own recipe sources and geometry; they do not clone or validate a whole bank task
per member. Architecture projection indexes selected tasks once per projection.
Source-store binding checks provenance once per shared task for that handoff;
unchanged member recipes reuse their inferred metadata, while transformed recipes
require new inference.

SafeTensors admission resolves each distinct shard member once and retains one
checked header, exact index membership, file identity, and tensor catalog per
physical shard. Cloned admissions, core inspection, prepared stores, and bounded
payload-cache eviction share those results, including failed header admission.
Core projects tensor descriptors from this catalog instead of parsing headers.
Prepared metadata and provenance queries are in-memory snapshot lookups; new
leases and encoded batches must match that snapshot. File-change checks occur
when bytes are read or a cached payload is acquired, rather than on metadata
queries. A file changed after header admission is rejected at that read boundary.

Each immutable recipe catalog retains inference for complete recipes and shared
subrecipes. SafeTensors catalog/store clones share the cache; restricted views
have their own authorization scope. Mutable custom catalogs do not opt into this
contract. A source that supplies a cache promises stable metadata/provenance and
read batches/leases checked against that catalog. The same neutral storage can
retain success or failure text for a caller-defined, fixed metadata validator;
it never retains validator closures or native resources. MLX owns its constant
representation checks and reuses these results across preflight calls. Dynamic
device capabilities and validation of newly materialized resources remain
separate. Inference and validator results are initialized once under concurrency.
Binding preflight returns its validated owner/alias partition; selected binding
plans own that partition through materialization without reconstructing it.

Exact checkpoint content identity is demand-driven. `eredu-checkpoint` records
file identity, size and change metadata without retaining per-shard file handles;
when requested it hashes each admitted file in one sequential pass, checking
metadata before and after reading. `eredu-core::DeferredArtifactIdentity` shares
and caches that result or failure. Model, external-assistant and Moshi source
preparation retain the pending identity without reading payloads for hashing.
Architecture-owned discovery declarations carry it through native construction;
explicit capture/intervention discovery resolves it on demand. Ordinary generation
and trace-only requests do not resolve it. Backend adapters retain these neutral
declarations and never reconstruct source membership or fingerprint files.
Tensor loading reads selected ranges in execution order, so it is independent of
the optional whole-file hashing pass, which also covers headers and unused bytes.

The workspace optimizes the `sha2` dependency in development builds. Its
compression kernels otherwise spend substantial time in unoptimized code even
when CPU hash instructions are available. Eredu code retains ordinary debug
settings; the SHA-256 identity format, full-content reads and change detection
are unchanged. Downstream workspaces control their own Cargo profiles and can
apply the same `[profile.dev.package.sha2] opt-level = 3` override.
`cargo run -p eredu-core --example artifact_fingerprint -- FILE [ITERATIONS]`
measures fresh deferred resolutions and memoized lookups. With a warm filesystem
cache on the M3 Ultra, a 2,156,600,968-byte checkpoint took a median 8.71 seconds
with unoptimized SHA-2 and 1.03 seconds with this override (three runs each).
Release took 0.92 seconds and `sha256sum` took 1.01 seconds on the same file.
The artifact identity was identical in every run.

Each uncached SafeTensors range is read once into an owned buffer, with file
identity and change metadata checked before and after the read. The shard cache
weakly shares full-tensor buffers between live leases; it does not retain payload
copies after their final lease retires. Native materialization retains its source
leases until completion, then releases those host buffers independently of the
resident native weights.

For byte-preserving leading-axis joins, checkpoint recipes can instead compile
an exact encoded read batch. The batch retains admitted metadata and file
identities and groups reads by shard and offset. Adjacent file ranges use
vectored reads into disjoint destination slices, including reordered expert
ranges and separate parameter allocations. Physical reads are bounded to
64 MiB and 1024 buffers; consecutive chunks continue without another seek.
Short reads and interruptions advance within those slices without staging. The
batch owns no payload buffer or shard-cache lease. Prepared and
restricted source views preserve catalog and authorization checks for this path;
file identity is checked once before and after each shard's batch of reads.
Recipe inference reuses the immutable source's retained result, or uses batch
metadata for custom sources without an immutable catalog. Unsupported transforms,
partial selections, and source encodings retain the ordinary materializer.

The runtime owns a portable allocation-group budget of 1 GiB and 64 parameters;
an oversized single parameter stands alone. MLX module binding and resident-unit
loading retain each recipe's direct/fallback preparation decision and combine
compatible reads within those groups, opening and checking each participating
shard once per group. Aliases reuse their owner without another read.

On CPU and Metal, MLX materialization allocates the group's final contiguous
arrays first and supplies their writable bytes to the neutral read batches. The
safe native initializer owns exclusive, zero-initialized allocations until all
reads succeed, then publishes immutable arrays together. Error or unwind drops
the whole unfinished group. Native completion recovery retains initialized
inputs before any stream copy or submission. Apple unified memory lets the GPU
use that same allocation. Expert stacking and gate/up concatenation require no
per-expert native arrays or checkpoint-sized staging copy on this path. The
optimization follows each selected binding's recipe, so it neither changes
residency selection nor eagerly reads otherwise nonresident banks. Native
allocation and pointer access remain inside `safemlx`/`safemlx-sys`; checkpoint
code only receives bounded, disjoint mutable byte slices. CUDA retains its
existing materialization and transfer path.

After selection, `PreparedModelSources` is the sole architecture-aware
SafeTensors/GGUF source factory for model loading. It opens every admitted
physical source once, retains exact resolutions and metadata, composes typed
companions in semantic-role order, and publishes explicit primary, complete,
target, and prediction-extension views. Restricted views share the same source
cache and fail closed for unauthorized keys. MLX begins only with these source
roles and performs native lease conversion, transforms, binding, and tensor
materialization.

Architecture-aware model inspection uses the same normalized request,
mechanism provider, and total selector. The neutral outcome owns report state
transitions and can retain the exact selection used to establish readiness.
Backend adapters add only their name, static hardware context, and native
diagnostics; inspection itself opens no payload and creates no native resource.

Moshi realtime selection likewise derives shared policy from
`NormalizedLoadRequest`. Runtime's `synthesize_realtime_capabilities` enumerates
every execution alternative, stably deduplicates exact lowering capabilities,
and shares state-component traversal with replicated-text synthesis. MLX and the
independent realtime backend report only collection-independent facility facts
and per-descriptor lowering and placement support; neither capability provider
walks architecture requirement collections. Its neutral prepared-source handoff contains the exact
source, content identity under the architecture-selected domain, retained
resolution and metadata validation, selected topology and architecture, and a
consistent lowering summary. MLX receives that completed handoff and adds only
groups, streams, caches, arrays, random state, completion, and final erasure.
`MlxPreparedRealtimeExecution` privately pairs that handoff with the exact
native rank/device token validated during cold selection. Realtime
materialization consumes this pair without accepting replacement load options;
it validates the retained device against the execution stream before payload
binding or construction. `MlxLoadRequest` holds one `NormalizedLoadRequest`
and optional native device assignment, not an independently mutable portable
mirror. Backend-author callers build portable policy directly on the normalized
request and wrap it once; the MLX adapter exposes only native pairing operations.
Prepared MLX models retain their actual native device and, for distributed
execution, the selected world handle. Session publication and every
backend-taking session operation validate this retained target before input
conversion, mutation, or submission. Device labels and equal rank/size values
are not identity proofs; another stream on the same device remains valid.
Realtime state creation and frame submission apply the same retained-context
check, and direct execution rejects a stream on a different device. Realtime
selection rejects explicitly requested prompt-cache persistence or drafting
when the realtime architecture cannot honor them; explicit observation
capabilities require matching state-retention support even without named
observation requests.
Core retains exact capability admission throughout a `ModelRuntime` lifetime
and rechecks it before generic submission, observation, or inspection. Exposing
an unrestricted mutable backend session invalidates its execution-plan target
proof, since callers can replace that session. Drafting selected for the old
target cannot attach after such access. Native MLX C event and transfer producers
reserve output wrappers before submission, so publishing a successful producer
result needs no new wrapper allocation. Native submission recovery separately
records accepted CPU tasks and GPU command-buffer work before publication can
fail. A preallocated owner retains graph, primitive, temporary-buffer, stream,
and callback state until every accepted frontier is terminal. An error is not
terminal evidence, and evaluation is never retried as a cleanup operation.
Backend submission scopes retain application resources and session authority
across returned errors, discarded errors, and unwinding. Recovery neither
terminates the process nor waits for native work in destructors. Unobservable
work stays owned and its affected execution resources remain unavailable.
Terminal proof and reclamation are distinct: bounded native scope progress
publishes completion without destroying arbitrary primitive or group owners.
Ordinary owner-thread native operations reclaim terminal records outside the
registry lock; a departed owner thread cannot be replaced by a recycled thread
identifier. Rust recovery tries the runtime lock without waiting and retains
the node if it is busy. Events belong to the retained bundle, and terminal
handle destruction runs with housekeeping suppressed. Independent consumer
waits and host observations each have their own ticket; readiness requires all
such tickets to settle, not merely the original producer's completion.
Checkpoint materialization, quantization tiles, residency transfers, and bounded
execution units retain their own submission owners. Aborting a layerwise forward
does not drain events or whole streams. Terminal module, session, checkpoint,
and manager-lease owners are staged for ordinary unlocked reclamation, since
their destructors may acquire host locks or execute application code. Dense
forward cleanup is deferred from abort to an ordinary entry or owner reclamation.
Selected model materialization also reclaims an already-retired snapshot before
allocating the next model, so completed previous-session weights need not remain
alive until replacement session publication. Newly staged recursive owners wait
for a later ordinary entry; this is not a drain of unresolved native work.
Each completion poll keeps one nonblocking runtime guard across its native
observations. Arbitrary application resources retained by neural submissions
are staged separately after every native/consumer owner releases them: their
destructors do not run from polling, recovery Drop, or TLS teardown. The next
unlocked ordinary neural submission reclaims them, or callers explicitly use
`MlxNeuralBackend::reclaim_retired_resources`. This step may block in a caller's
destructor and is skipped under the runtime lock, during unwinding, and during
recursive reclamation. If no eligible reclamation occurs before owner-thread
exit, those resources remain retained.
If an application destructor panics during explicit reclamation, the remaining
detached resources are retained instead of invoking more destructors while
unwinding.
MLX prefetch, cache-I/O, and host-demotion worker handles request shutdown
without joining from recovery or teardown. Each worker owns its active inputs
until that operation returns. Prefetch and cache-I/O workers cancel queued work
and release its admission slots; the host-demotion worker drains queued inputs
after its sender disconnects. Portable worker APIs retain deterministic joining by default and
expose nonjoining shutdown as an explicit option. Final cache-state reclamation
removes its ephemeral files. Disk commits hold a weak state reference to avoid
ownership cycles and remove late ephemeral outputs when that state is gone.
Replicated routed text has its own architecture-owned requirements
and selected realization layered over the shared replicated-text contract.
Replicated composite text adds selected processor and ingress requirements to
that same contract. Partitioned, prediction, drafting, and realtime paths
retain their distinct selected construction values. Reusable MLX loading, residency, cache,
stream, transfer, and completion mechanisms receive only exact tasks, generic
limits, rank-local placement, or opaque group handles. The mechanism-only
`MlxRankContext` contains a world rank and local device assignment. Semantic
tensor, pipeline, and addressable axes remain in the checked neutral
`ParallelRankTopology` only until architecture selection projects them into an
opaque communication manifest. MLX combines the selected world rank with a
process-local `DeviceAssignment`; it never exposes a second backend parallel
plan or reconstructs axis groups. Prompt-cache manifests record the selected
placement as generic stage, state-shard, and addressable-group coordinates.

Selection compares the complete architecture transform constraint with an
exact backend lowering descriptor. A lowering descriptor identifies the source
encoding, target executable format, logical shape, and packed axis. A request
is rejected atomically when either side rejects the geometry; selection never
silently transforms only the convenient matrices. Rank-one normalization,
optional bias, tied, derived, and checkpoint-companion parameters remain in the
requirements even when no transform applies. The resulting per-parameter
realizations are projected into exact `ReplicatedTextMaterializationTask`
values. Each task retains its logical target, physical source and selected
output, recipe and companions, source encoding, executable format, and selected
lowering. These tasks are the only source, recipe, format, and lowering choices
consumed by module construction and materialization; no backend collapses them
into a model-wide transform or rediscovers policy from parameter names.
The exact-task binder applies retained logical placement before validating
rank-local destination geometry. Task-declared locally transformed outputs
are already placed and are not sharded again.
Packed lowering supports only the final logical matrix axis; any
other declared packed axis is rejected during selection. Companion bindings
are format-exact: affine formats retain scale and affine bias, while MXFP4
retains scale only and rejects an invented bias.

The same typed construction boundary is available to an additional
`ReplicatedTextArchitecture` through the generic extension visitor. Such an
architecture supplies its normalized configuration and parameter topology,
then uses the ordinary requirements, selection, exact materialization tasks,
neutral session constructor, and lifecycle. Adding it does not require a
backend family registry case or a complete-model wrapper.

Architecture-owned typed dispatch admits replicated Llama/Mistral, Nanbeige, dense
Qwen2/Qwen3, dense LFM2, dense Kimi Linear, target-only dense Nemotron-H, and
target-only text Qwen3-Next/Qwen3.5 configurations. Its additive routed class
admits replicated Qwen3 MoE, GPT-OSS, routed LFM2, Kimi Linear, Nemotron-H,
target-only text Qwen3-Next/Qwen3.5, and target-only DeepSeek V3/V4. It rejects
partitioned, prediction-bearing, and conditional-media graphs from replicated
routed construction. Gemma 4 remains composite even when its normalized
configuration omits media modules, so it does not enter this class. The
architecture registry selects both the replicated
execution class and the exact stateless, ordinary key/value, fixed-only,
attention-with-fixed, compressed-only, or compressed-with-fixed access profile.
Its replicated composite class admits Gemma 4, Inkling without an active
prediction group, Muse-Glimmer, Qwen3-VL and its sparse variant, and
conditional Qwen3.5 without an active prediction group.
It constructs modules with the selected executable formats, validates the
proof-bearing architecture value, and invokes the corresponding typed neutral
constructor adapter. Optional profiles are additive, so an ordinary backend
does not acquire heterogeneous-state bounds.

Nanbeige reuses the portable GQA/SwiGLU decoder blocks. `decoder::repeated`
lowers physical checkpoint blocks to logical invocations with exact source
aliases and an expanded attention schedule. The generic decoder's optional
post-block normalization expresses inter-pass RMSNorm. Each invocation owns
independent cache and residency state, so the ordinary dense drivers provide
per-block offloading, tensor sharding, pipeline cuts, combined TP/PP and captures.
Exact full-source aliases retain the physical tensor's encoding and geometry,
including native GGUF blocks, instead of treating encoded bytes as dense recipe
outputs. Architecture policy owns repetition, checkpoint geometry and cache
identity. Resident materialization currently creates a block for each logical
invocation; estimates count these replicas and all invocation caches, while
checkpoint storage describes the physical weights. Backends consume the usual
typed replicated or partitioned architecture and add no family dispatch.

`eredu-runtime::construct_replicated_text_session` and its execution-strategy
form are the single production construction family for these profiles. The
typed architecture adapter consumes the prepared handoff and passes its
concrete architecture, opaque validated
contract, cache identity, and `ReplicatedTextSessionMechanisms` to that
constructor. It chooses resident or bounded `LayerwiseRuntime` traversal,
realizes exact selected state, and returns a `ReplicatedTextSession`. That
session owns direct and observed forward, prefill, decode, causal output
selection, state publication, reset, checkpoint and rollback, prompt-cache
identity validation and replacement, residency reports, and exact completion
sequencing. Prediction lanes request an independently advanceable state fork
through the same neutral mechanisms; paged backends copy the canonical block
catalog into an independent residency-session namespace instead of applying a
canonical-session checkpoint to unrelated cache ownership or weakening
cross-session checkpoint validation. A non-MLX backend supplies its own tensor, policy, state,
persistence, and completion mechanisms to this same constructor and executes
the same lifecycle. Routed sessions supply a `RoutedReplicatedTextExecution`
strategy to this constructor; they do not define a second prefill, decode,
state, prompt-cache, reporting, observation, or completion lifecycle.

`RoutedTextRequirements` retains the architecture-global grouped-operation
plan, exact `ExpertResidencyCatalog`, and configured routes-per-token
cardinality alongside the shared text contract.
Neutral selection combines those facts with caller bank residency and generic
backend capabilities for grouped operations, indexed movement, addressable
storage, each required disk/host/device storage tier, compact-bank limits,
leases, and exact completion. Admission computes the largest selected compact
working set for one token row by summing the largest configured number of
members from the selected, post-transform byte geometry. It rejects an
undersized scratch bound before construction. The selected cardinality remains
in the checked handoff and every planned provider rejects a route tensor whose
last dimension differs before it can acquire storage. Resident and independently addressable
strategies consume the same grouped specifications. The addressable strategy
maps selected global identities to `ParameterBankKey` values, partitions bulk
requests to the selected byte bounds, processes incremental requests one token
row at a time, acquires keys in deterministic order,
remaps compact indices, invokes the selected grouped operation, reduces the
partial outputs, and completes each acquisition. Gated-product and ReLU² are
operation profiles of this one neutral driver.

Before the backend adapter runs, architecture composition projects the
`ExpertResidencyCatalog` into generic `AddressableBankMember` records. Each
record contains its stable bank key, exact compact bindings, source and
selected byte geometry, and architecture-selected owner group, unit, parameter
path, distribution, and optional rank placement. The selected geometry includes any load-time
transform and is therefore the geometry used for capacity checks, acquisition,
and telemetry. A backend adapter consumes these generic records; it does not
receive the architecture catalog or interpret family recipe policy.

MLX implements the mechanism bundle with native operators, module stores,
resident and bounded policies, selected state allocation, prompt-cache bytes,
streams, indexed tensor operations, generic addressable member storage,
compact grouped-bank construction, leases, telemetry, and completion objects.
Neutral lifecycle code chooses the final causal sequence position; MLX only
applies that exact index. State allocation iterates the exact selected components and placements
without deriving policy from architecture roles. Its
adapter selects neither router policy nor a family-specific provider and owns
no peer replicated-text lifecycle. It invokes the neutral constructor and performs the
final backend-private erasure of the completed typed session. SafeTensors
aliases, derived recipes, GGUF translated outputs, and selected transformations
feed the same exact task interface. Erasure performs one dispatch per outer
session operation; tensor operations, state-component access, and
execution-unit traversal remain statically dispatched.
The erased public session delegates the addressable-bank report from the typed
session, so hit, miss, movement, tier, and byte accounting remain observable at
the facade boundary. Reports retain the complete deterministic list of selected
transformation formats; a mixed bank is never collapsed to an absent or
arbitrary bank-wide format. They also retain the exact selected placement for
every bank key, including owner group, unit, parameter path, distribution, and
rank. Production construction requires the selected-member handoff; raw entry
constructors are not a public route around architecture selection. Resident
routed execution reports no addressable bank.

The selected realization is also the sole construction authority for exact
weight-residency limits, mutable-state paging, topology realization, session
facilities, prompt-cache persistence, and completion ownership. Backend stream,
device, and native group handles are execution contexts rather than policy.
Once selection succeeds, neutral construction and backend mechanisms receive
no second copy of caller load options or family configuration from which they
could choose a conflicting policy.

Source storage, executable format, and native lowering are independent values.
SafeTensors handoff carries the admitted `SafetensorsShards`, including index
validation and canonical shard order, into the bounded store without directory
rediscovery. GGUF handoff carries the admitted translated output map and exact
physical encoding; backend lowering does not reconstruct canonical names from
GGUF strings. Native and transformed parameters may coexist in one selected
plan because materialization consumes each exact task independently. The
backend reports resulting storage separately from residency telemetry.

`ReplicatedTextArchitecture` adds only ordinary borrowed text-input formation
to the layered lifecycle and remains generic over its runtime state. Hybrid
component state uses the `RuntimeStateComponents` extension. Architectures own
router equations, top-k semantics, shared contributions, grouped geometry,
parameter topology, and stable group/unit identity. Neutral routed composition
translates the resulting identities through the architecture plan into grouped
operations and independently addressable parameter-bank keys. Concrete
backends receive exact generic keys, grouped specifications, bindings, access
classes, and completion requests; reusable backend modules receive no expert
plan, routed request, family configuration, or router policy.
Architecture/runtime semantic
plans translate tensor, pipeline, routed, data, or later axes into ordered
world-rank memberships, group-local ranks, generic point-to-point routes, and
required collective operations before invoking a backend. Backend sessions
materialize groups by opaque identity and never infer why a group exists.
Before creating any native subgroup, every rank exchanges its complete opaque
communication manifest over the unsplit world control plane. The neutral
runtime gathers fixed-size length metadata followed by equally padded payload
words, reconstructs every rank artifact, and validates descriptor and
invocation-limit compatibility. Backend-local capability and world/rank checks
run only after that collective proof, so a mismatched rank cannot return early
while peers block in subgroup construction.
Each MLX manifest realization retains its exact opaque group ID and operation
requirements on the mechanism handle, including singleton and full-world
groups that reuse the same native membership. Reusable collective entry points
check the selected operation, dtype, tensor geometry, peer-count and output
limits, and exact-completion requirement before native submission. The
unsplit world consensus handle is explicitly uncontracted and is never returned
as a manifest group handle.
Tensor-parallel grouped partials are a required additive mechanism separate
from ordinary grouped execution; selection rejects the missing extension
before construction. Routed architecture plans derive those requirements from
their normalized expert configuration and selected topology, including the
tensor-parallel partial, before any payload is opened or module constructor is
called. Architectures requiring mixed mutable-state components use the same
selected state authority through `ArchitectureStateFactory`; fixed recurrent
and convolutional components remain device-resident when append-only attention
components use paged storage.
Partitioning adds `PartitionedLayeredArchitecture`, boundary schemas, and
driver-owned boundary exchange through opaque collective contexts;
one `PipelinePartitionExecutor` retains the selected typed per-unit strategy,
so ordinary and provider-backed routed units share the same local-address,
boundary, scheduling, publication, and state lifecycle rather than selecting
a backend pipeline implementation. A prepared routed partition carries one
immutable architecture-owned execution handoff: local-versus-pipeline dispatch,
erased grouped-expert realization, provider route cardinality, active tensor
mechanism, boundary and output geometry, per-unit tensor-reduction order, and
the resulting collective waves. The backend supplies only generic provider,
movement, communication, allocation, and execution mechanisms; it does not
inspect semantic ranks or rebuild a routed strategy. Pipeline combinations with
an expert axis use the architecture-declared all-rank operation order; inactive
cohorts submit exact-shaped zero work at every declared TP/EP wave and
bounded-complete it before advancing. Missing route cardinality or an
incomplete expert-wave recipe fails while preparing this handoff, before a
provider or native communication mechanism is bound.
Partition admission also retains the complete exact physical task selection
before any checkpoint payload or backend resource exists. Architecture-owned
parameter topology later attaches its atomic output companions and projects
that immutable selection onto the admitted rank ownership; it cannot reselect
sources, encodings, executable formats, recipes, or lowering from a module
walk. The prepared partition carries the resulting rank-local tasks into the
generic materializer together with the exact placement and communication
authority.
Composite partition admission also derives the complete TP-local mutable-state
layout from the normalized architecture schedule and selected rank, then
projects the PP-owned range before architecture or state construction. The
family partition geometry produced during typed construction must equal this
cold selected state; it is a validation result and cannot fill in or replace a
missing admission-time state plan.
composite models use `PreparedModelInput` and architecture-owned execution
groups; embedded prediction uses separately identified prediction groups and
`DraftStateTransaction`; and realtime models opt into `RealtimeFrameTransition`,
which consumes `RealtimeInputFrame` on an unpublished branch and attaches the
exact completion before atomic publication.
Composite and routed requirements are additive selections over explicit media
and grouped-computation capabilities; they do not broaden `NeuralBackend`.
Partition, prediction, and frame requirements remain confined to their own
execution classes.

Grouped execution is exposed through the required `GroupedNeuralBackend`
extension rather than a permissive base-trait fallback. Distributed vocabulary
and sum operations similarly live on the required
`DistributedNeuralBackend` extension. The neutral vocabulary-range contract
validates this rank against its architecture-selected balanced partition and
publishes the exact ordered peer widths; a backend consumes those widths for
uneven gather instead of reconstructing sharding. Public execution-plan, topology, and
speculative handoff records keep their fields private, use checked or named
constructors and accessors, and mark open semantic enums non-exhaustive.
Move-only multi-value architecture handoffs are represented by named artifacts
instead of positional tuples, so later fields can be added without changing a
cross-crate destructuring contract.

`eredu-nn::GroupedUnitObserver` describes the sparse activated-unit boundary
inside gated-product and ReLU-squared banks. Its borrowed `GroupedUnitBatch`
contains only actually selected routes: local group IDs, sorted-to-original
selection indices, chunk-relative token indices, original coefficients and the
chunk's offset in the operator request. Architecture/provider composition owns
the mapping to global expert, invocation and sequence identities; a native
operator cannot infer it from local storage. Original observation, intervention
and effective observation precede the actual down projection. MLX validates
replacement shape and dtype before that projection and preserves observer
failures through the neutral error source. Effective units precede any selected
input quantization and must not be described as transformed multiplication inputs.
An absent observer follows the ordinary path without extra tensor evaluation,
retention or materialization. Borrowing a batch does not admit capture copies or
transport; those remain runtime responsibilities. Mechanisms without the hook
return `GroupedUnitError::Unavailable` when one is requested. This mechanism
alone does not advertise loaded public expert-neuron capture: bounded admission,
architecture discovery and committed distributed delivery remain required.

`eredu-runtime::RoutedUnitObserver` carries the provider-owned interpretation of
those rows. `RoutedUnitBatch` borrows the original route tensor before compact-bank
remapping, the retained source-to-global expert map, and the provider chunk offset.
Resident and addressable providers pass the same observer to gated/ReLU-squared
operators and their tensor-parallel variants. Compact native group IDs never
become global expert IDs. Expert exchange borrows existing receive counts and
original route tags to identify the sending peer, token row and top-k slot,
including idle peers and duplicate selections. These token rows are not prediction
ordinals. No expanded activation tensor or additional host route vector is needed.

`ActivationObserver::routed_unit_observer` lends one invocation's observer to the
canonical provider call. Shared decoder adapters use `ObservedExpertProvider`;
custom routed/shared equations attach it with `with_routed_unit_observer` while
retaining their architecture-owned combined-output event. Two observer owners
are rejected before provider execution. Callback failures retain their original
neutral cause across provider and neural-adapter error formatting. The observer
remains scoped to the actual chunk computation and bank lease; it does not extend
source residency or authorize retention. Selected-linear operations have no
activation-to-down seam and reject an attached grouped-unit observer. Public
sparse intervention uses the shared runtime driver described below; EP completion
and delivery policy remain separate from ordinary selected-unit receipts.

`ArchitectureDescriptor::routed_components` separately declares sparse expert/unit
topology. `RoutedComponentId` keeps logical invocation, checkpoint-global expert
and per-expert unit distinct; token rows and duplicate route slots describe
participation in that identity. Architecture discovery projects this topology
from the same gated/ReLU-squared bank specifications used for construction,
including packed leading expert axes, independently named banks, fused gate/value
row offsets, exact activation policy and ordinary affine biases. No checkpoint
name interpretation is delegated to the facade or backend. An absent declaration
or residual scale means missing descriptive coverage, not measured zero.

The neutral `RoutedComponentGroup` selection helpers join declared coordinates to
`ParameterDiscovery`, validate exact effective rank/shape, and return ordinary
unsqueezed `ParameterRegion`s together with actual loaded alias and capability
facts. The existing query/overlay admission owns identity, dtype, storage and copy
authority. A topology join never upgrades unavailable operations or authorizes
native work. Sparse groups are separate from dense `components`.

Sparse observation points declare `ObservationValueType::RoutedUnits` with exact
provider invocation, global bank geometry and original token/route/unit axes.
Their `RoutedUnits` capture transform returns route-bearing host records rather
than an expert-dense tensor. The core owns those records and exact receipt checks;
runtime borrows the existing provider hooks, reserves a complete invocation before
the first chunk, and accumulates records under the ordinary capture transaction.
Contiguous source-token coverage, unique original token/route slots, global expert
bounds and unit geometry must all agree before committed delivery. A native
collector maps pre-compaction source groups, copies bounded route metadata and
selected unit rows, and retains no capture tensor beyond its callback.
`ObservationMechanisms::routed_unit_tensors` reports this separate collector fact.
Public partitioned sparse delivery uses separate retained ownership, source votes
and global receipts through the same capture session, as described below.
Ordinary route receipts alone cannot authorize exchanged or partitioned evidence.

Sparse mutable points add `RoutedUnitInterventionPoint` with a virtual global
`[token, component]` axis (`component = expert * units_per_expert + unit`). Core
owns exact declarations, participation coordinates and fixed-size progress
receipts. Runtime validates and lowers Zero, Scale, Mask, MaskComponents, Replace
and Add onto current participating values, preserves list order, and reserves
full-invocation native/host work before collecting any route metadata. It gathers
only addressed elements, calls the ordinary typed activation driver and scatters
them into the actual value consumed by the down projection. Backend primitives
copy native route coordinates and perform indexed gather/scatter; they do not
interpret global component identities or dispatch public actions. No virtual
expert-dense activation is allocated. `InterventionMechanisms::routed_units` is
an independent native fact, combined with actual loaded hook/capture support.
A contiguous completed-token receipt is required for success; `Unmatched` means
no participating value was addressed. Original/effective sparse captures supply
attributed evidence; dense Preview/Summary operation evidence lacks participation
coordinates and is rejected at sparse targets.

`RoutedComponentCoordinateMap` keeps selected expert ownership and within-expert
scalar placement as separate checked axes. Architecture-owned
`derive_routed_component_coordinates` takes the selected `RoutedGroupedPlan` plus
retained write layouts: stored independent matrices do not confer execution
ownership. Packed expert axes and scalar TP columns are validated separately;
the existing semantic-unit proof expands packed columns without guessing an
encoding factor. Independent matrices must agree on their bank's scalar map.
These coordinates describe placement, not invocation admission or completion.
The expert map follows the selected realization's owner-local order; it is not
the physical row order of a packed parameter or the sorted order of routed values.
Actual routed rows carry their checkpoint-global expert identity independently.

Prepared routed banks retain per-invocation coordinates derived from the original
selected grouped specification, its localized construction specification and the
retained physical write layouts. This derivation checks local expert and scalar
extents against the construction that will actually execute. The shared resident
and addressable partition provider factories attach these maps before native
provider chunking; EP origin adapters preserve them across exchange. Backends
receive the prepared providers and do not reconstruct scalar placement. These
bank coordinates describe storage; pipeline invocation ownership remains with
the partition execution plan.

`with_partition_unit_observer` lends the scalar map through provider chunks and
expert exchange, preserving original source peer/token/slot tags and actual unit
values. The portable `lower_partition_routed_intervention` maps global edit plans
onto those local columns and checks owned experts and distinct source coordinates.
It does not establish global route coverage or authorize publication. Ordinary
capture/intervention owners reject partition-tagged batches. Committed sparse
producer admission, quota reservation, collective agreement and receipt delivery
still require integration into the existing distributed capture owner.

The existing runtime partition receipt protocol now admits sparse declarations
through `PartitionCaptureReceiptPlan::new_routed`. Core's
`RoutedUnitCaptureOwnership` carries separate expert/unit placement and the
authoritative logical source peer; it does not confer native execution authority.
Runtime validates complete, disjoint expert-by-selected-unit coverage, retains
local storage order in the receipt identity, and checks each received route against
that ownership. Tensor-parallel fragments must agree on expert and coefficient
bits, and every selected token/slot/unit must arrive exactly once. Native chunk
ranges remain in `RoutedUnitCaptureProvenance` on each partition contribution;
the assembled logical payload does not fabricate a global native chunk sequence.
Delivery reservations include route accumulation and provenance copies before
producer work. This reuses the existing bounded exchange and all-rank failure
agreement. Connecting live observer admission and completed native invocations to
these receipts remains necessary before advertising public partitioned support.

Core now owns the checked borrowed `capture::RoutedUnitOrigins` coordinate view.
Runtime's provider adapter preserves its existing neural error contract and lends
the same tags without copying them. `PartitionRoutedUnitCaptureSource` supplies
actual provider columns and exchange tags; `PartitionRoutedUnitCaptureRequest`
separately supplies retained ownership, global bank geometry, original source-token
extent and the selected global fragment. Neither object is native submission
authority. Runtime's `RoutedUnitBatch::partition_capture_source` joins existing
borrowed provider inputs without retaining arrays or reading their values.

The MLX collector checks actual columns, native route width, source extent and
exchange-tag coverage before copies. It maps sorted native routes back to original
peer/token/slot coordinates, validates owned experts, and gathers selected global
units from actual local columns. Other source peers contribute no exported rows.
Its estimator covers worst-case incoming routes from all peers and native chunk
provenance for the entire invocation, even when the selected token set is small.
Contiguous cold unit ranges are checked without expansion. This is a native
collection mechanism; live sparse admission, source completion and committed
delivery still belong to the existing partition observer/session integration.

`CaptureSession::prepare_partition_routed_capture` now compiles those sparse
declarations through the same private preparation driver as dense partition
captures. It retains the original plan and run identities, prepays every producer
and receiver, and spends local fragment credits before allocating accumulators.
`begin_partition_routed_capture` pins the actual invocation's native row count,
dtype, scalar map and exchange topology. Each `observe_partition_routed_units`
call advances a contiguous native chunk; `finish_partition_routed_capture`
rejects missing chunks or provider failure. Zero-row EP owners must explicitly
begin and finish. Completed fragments use ordinary receipt exchange and staged
transaction commit, and failed work never refunds the parent ledger.

The shared `RoutedUnitObserver` invocation lifecycle encloses actual local provider
work before reverse route exchange or downstream tensor reduction. Architecture
adapters enter this scope for resident, host-layerwise and disk-streamed banks;
the EP adapter also enters it for zero-row owners. Begin borrows the actual input
and available prepared coordinates. Finish runs on success, preparation failure
and provider failure. Nested bank adapters preserve the active scope instead of
repeating callbacks. Native composition forwards the same borrowed arrays, maps
and origin tags without evaluating or retaining component values. The first local
error remains the source when the final callback also rejects the invocation.
Nemotron-H's partition adapter now forwards its routed ReLU² unit observations
through this driver, including the TP execution path.

These callbacks describe local execution; they do not certify native completion
or committed capture delivery. The shared `PartitionCaptureObserver` now consumes
architecture-supplied `PartitionRoutedCapturePlacement`: distinct producers plus
all actual provider members, their expert/unit ownership, original source peers
and input widths. It validates invocation identity and original/effective timing
against the retained admission. Source preparation reserves the maximum received
input rows on every member, including idle owners and nonexporting replicas, in
the ordinary selection digest. Actual input rows, width, dtype, optional scalar
map and origin topology pass a prepaid group vote before native dependency work.
The existing backend source-completion method retains its completion guarantees.

Every selected original/effective record consumes its source and final votes in
plan order. An earlier error prevents later native source work. Remaining votes
are attempted while transport is active; a native source failure fences its owner
and requires native settlement or teardown instead of further submission.
Finish validates complete sparse chunks and settles the
prepaid invocation-group vote before reverse exchange or downstream reduction.
Receipt delivery remains provisional until the same whole-forward transaction
commits. Errors and abort do not refund the ledger; no second execution or
completion owner was introduced.

`ComponentPartitionLayouts` now retains sparse placement from the selected bank
and logical invocation. `ExpertRealizationPlan` retains the same ownership policy
used by bank construction; cold rank projection uses that policy without creating
local bank specifications. Scalar columns come from the actual physical write
layout. Routed child operations inherit their declared enclosing MoE invocation's
parameter ownership, independently of checkpoint sharing. Each point retains
exact global expert/route/unit geometry, input width, publication source peer and
all executing source members. Identical coordinate replicas select one producer;
idle received owners remain members. Original/effective points retain distinct
timing while using the same invocation coordinates. Projection checks schedule,
producer and fragment limits before runtime admission. Current balanced expert
construction requires at least one expert per EP rank; placement preserves that
existing rejection. Loaded capture additionally requires the actual sparse hook
fact and exact native subgroup support. Runtime still owns reservations, receipts,
completion and publication; groups needing world participation waves and sparse
intervention composition remain unfinished.

DeepSeek V3's layered entry points emit internal hooks while the traversal owns
the unit input/output observations. The standalone block entry point supplies
those outer observations itself. This keeps one intervention at each boundary
when the same equations run through an expert-parallel provider. The layerwise
runtime exposes an architecture failure as its error source, retaining nested
observer causes.

V3's resident and paged latent-attention paths now share the architecture-owned
post-aggregation channel boundary and existing projection-input observer. The
dense-only and routed-capable block adapters share normalized inputs, scalar
SwiGLU units, write vectors and residual sequencing through
`ComponentInstrumentation`; the existing `compressed_attention.output` identity
remains intact. Compressed replicated traversal forwards its observer through a
family callback, just as fixed-state traversal does. Native backends gain no V3
equations or policy. These hooks alone do not establish complete discovery or
parallel support: sparse tensor-parallel block hooks and prediction internals
remain required integration. The target-only mixed V3 partition adapter now emits
embedding observations only on the input owner, carries the effective embedding
in its typed boundary, and emits normalization/projection/readout observations only
on the output owner. Serial and vocabulary-parallel readout share the same driver;
All-dense TP unit hooks reduce attention and FFN writes through the ordinary
collectives before observing complete writes. Sparse TP unit coverage remains
unverified. Local public native capture,
parameter-edit and controlled snapshot tests cover direct/low-rank query variants
and all three weight-residency modes.

All-dense prediction-free V3 partitions now use architecture-owned direct
construction from the retained selection. The constructor derives the target unit
range, compressed state, static ownership, and source/target layouts for transformed
weights. It creates no expert bank and requests no grouped mechanism. V3 static
modules consume explicit selected embedding/head formats; published global FP8
layer defaults still leave those statics dense. V3/V4 format expansion uses the
shared runtime mechanism, retaining companion-to-primary metadata and independent
fused FP8 row blocks rather than duplicating packing arithmetic.

V3 query rotation now receives `[batch, heads, sequence, features]`, following the
neutral rotary operator's penultimate sequence-axis contract. This fixes a prior
equation error in both ordinary and partitioned execution. The V3 cache fingerprint
includes MLA equation revision 2 so earlier states cannot be silently reused.
Native CPU Ring verifies all-dense TP, PP and combined TP/PP across resident,
host-layerwise and disk-streamed SafeTensors and GGUF weights, including the GGUF
split KV-B layout, public capture, masks, queries, coordinated edits and rollback.
All nine topology/residency cases also pass SafeTensors load-time affine
4-bit/group-32 transforms, exercising source/target layouts and packed companion
bindings in actual native execution. No family equations moved into the backend.

Component schema 2 adds ordered effective input projections to a read, carried by
architecture descriptor schema 4. Core owns this neutral relationship; architecture
discovery supplies V3's query bottleneck and the normalized latent part of KV-A.
The shared rotary key is a separate direct dependency and does not acquire the
latent normalization. `eredu-nn::LowRankProjection` supplies a normalized-rank
callback while keeping one projection implementation; architecture instrumentation
uses it to observe and edit the value consumed by the second matrix. The current
KV latent value is observed before cache insertion. Descriptions do not promise
that historical cache rows were captured, nor replace measured normalization or
selected input-quantization facts with a fixed matrix product.

Retained component placement also covers intermediate read stages. The architecture
joins each logical invocation to the stage weight's output-row placement, intersects
its declared row interval, and registers original/effective observations on that
axis. Shared source aliases do not transfer pipeline invocation ownership. Packed
input columns do not change output-row identities; short FP8 row chunks and indexed
row permutations retain their actual coordinates. Column-sharded partial products
cannot be advertised as complete stage values without an explicit output contract.

Native compressed-latent snapshot copies belong to the backend cache mechanism.
Resident copies compact the logical latent and rotary arrays into independent
storage; continuation estimates include chunk growth separately. Paged copies
duplicate the tail and the model-wide catalog, then bind each copied layer to that
catalog. They preserve the ordinary neutral snapshot admission, completion and
cumulative copy accounting; no family-specific controlled driver is introduced.

The architecture-owned `RoutedPlusShared` equation exposes shared-expert gated
units and the actual down-projection input through its ordinary component
instrumentation. Both local and TP execution use the same shared product policy;
the TP caller retains responsibility for completing the write reduction.
Discovery joins V3 gate/value rows and write columns to the shared-expert child
node. The child's output is a constituent of the complete sparse residual write,
so a reconstruction includes the whole sparse write once or expands all of its
branches; it never adds that whole write and its shared constituent independently.

Execution-group transport is also architecture policy. The runtime defines the
neutral placement, semantic-kind, merge-destination, parallel-subgroup, and
request-optionality data types, but every `LayeredArchitecture` must declare the
transport for each of its groups. It also declares the stable primary-group
identity and the stable identities of ordered prediction groups. Composition
resolves those IDs against the canonical graph; semantic kinds classify
lifecycle behavior and are never group addresses. Multiple groups may therefore
share a decoder, encoder, projector, or prediction kind without making selection
depend on declaration order. Request optionality applies only to root media
encoders; the neutral pipeline lifecycle rejects it on structural, decoder, or
prediction groups. Shared decoder defaults live in `eredu-architectures`; the
runtime does not assign decoder roles or placement to an unspecified group.
Composite families may declare different policies per group. Moshi's realtime
extension is deliberately pure tensor parallel: its temporal and ordered depth
groups remain local traversal groups rather than pipeline stages.

Mutable-state partitioning is a separate architecture declaration over the
complete `StateLayout`. Every layered architecture publishes an
`ArchitectureStatePartitionPlan` whose rules either align an exact state range
with one execution group's unit indices or attach that range to the realized
output owner. The runtime validates complete, non-overlapping coverage,
group-length agreement, and the contiguity required by `PartitionState` before
a backend receives rank-local geometry. Backends must consume the resolved plan;
they must not extend a decoder range to the end of the state layout or infer
prediction-state ownership from pipeline rank position.
`ArchitecturePartition::from_architecture` resolves this plan and selects the
partition's architecture-declared parameter groups as one neutral lifecycle.
Backend pipeline composition supplies physical group placement and consumes
family-owned local geometry; it does not create an ownership probe or populate
state in a second pass.

The ordinary Qwen decoder, block, and layered lifecycle uses only the base
neural mechanisms. Qwen architectures that select sparse groups use a separate
architecture lifecycle and the grouped-computation extension traits. A backend
used for the ordinary lifecycle is therefore not required to provide grouped
selection or grouped projection mechanisms.

External RoPE maps and their family aliases are architecture configuration
input, not a backend contract. Architectures normalize them into the closed
`RotaryAlgorithm` carried by `RotarySpec`, including all defaults and validated
scalar geometry. A concrete backend only lowers that typed algorithm to its
native rotary operator; it must not parse configuration keys, choose defaults,
or decide which model-family scaling names are supported.

The architecture configuration registry is the sole owner of Hugging Face
family aliases, GGUF `general.architecture` spellings, nested-wrapper
normalization, and the exhaustive dispatch to family parsers. During
SafeTensors resolution, the registry parses and validates the complete family
geometry and derives its typed checkpoint plan before Core admits the
configuration. During GGUF inspection, core passes the complete portable
header catalog to that registry; the registry performs full family parsing,
canonical-name collision checks, and architecture checkpoint-schema validation
before admission. Concrete backends consume those results and do not repeat a
family admission match. Its
external assistant resolvers likewise own both SafeTensors `model_type` and GGUF
`general.architecture` identities and aliases. `eredu-core` accepts that
registry through `ModelConfigurationResolver` while inspecting both
SafeTensors and GGUF artifacts. Each resolution couples the open canonical
family string and neutral `LoadingProtocol` with opaque architecture-owned
typed state. Artifact enrichment carries that state forward while adding any
processor sidecars or GGUF companions; it does not reconstruct it from the
portable configuration. Core routes the protocol and resolves companion paths;
it neither recognizes family strings nor exposes an exhaustive family type.

Qwen hybrid SafeTensors catalog admission distinguishes the official namespace
from MLX-VLM's `language_model.model.*` and `vision_tower.*` layout. The
architecture owns exact aliases, convolution axis conversions, and the inverse
of MLX-VLM's normalization-weight offset. These are ordinary bounded parameter
recipes consumed by all execution mechanisms. Canonical parameter geometry
remains independent of the selected physical axis order. Converted artifacts
with no MTP tensors normalize their retained prediction depth to zero before
capability reporting and execution selection; partial drafts still fail strict
validation. Official SafeTensors retain their declared-MTP requirements, and
GGUF retains its separate canonical conversion recipes. Qwen embedding
construction consumes the selected per-parameter format, including quantization
companions, rather than relying on a checkpoint-wide quantization default.
The CLI chooses embedded drafting from the admitted architecture's prediction
extension and reuses that inspection for loading; it does not infer family
capabilities from raw JSON configuration fields.

SafeTensors assistant admission requires explicit, agreeing outer and nested
model identities; architecture normalization occurs only after both identities
have been validated and cannot supply or disregard identity evidence.
The typed `ModelKind` and `GgufArchitecture` identities, their aliases, the
family-to-protocol mapping, and family-specific GGUF structural admission live
in `eredu-architectures`; core does not publish a parallel serialized model or
artifact identity schema. The serialized form of every `ModelKind` is exactly
its architecture-owned canonical family name and round-trips through
`ModelKind::resolve_family`. Nested `text_config.model_type` normalization is
available only to outer wrapper identities explicitly admitted by that closed
registry; an unknown outer `model_type` is rejected even when its nested text
identity is known. A known outer wrapper and nested text identity must also
resolve to the same canonical family; the preliminary identity resolver rejects
the configuration before returning a contradictory family. External assistant
admission remains separate: its closed
resolver explicitly selects the ordinary family tokenizer contract carried by
the admitted assistant plan, without admitting assistant identities as ordinary
models. Before invoking that resolver, `eredu-core` applies only
container-generic catalog checks; it does not recognize family metadata keys or
tensor names. Facades and concrete backend adapters select that shared
registry. Backend composition converts the resolved
canonical family through the architecture registry and consumes architecture
parser outputs, never a second raw `model_type` or `general.architecture`
dispatch table. Unsupported family identities therefore remain architecture
resolver errors transported through the backend-neutral artifact error;
concrete backends do not declare a parallel model-type dispatch error.
Architecture composition failures are reported as architecture-model errors,
while artifact normalization failures retain the backend-neutral artifact error
and facade-owned workflow validation retains a facade-owned error. Concrete
backend errors do not use an unsupported-architecture string funnel for these
unrelated ownership domains.
Loaded facade models, backend models, and sessions preserve these as two distinct identities:
`model_family()` returns the canonical architecture-owned `ModelKind`, while
`effective_model_type()` returns the parsed implementation or nested text-model
type. Architecture-owned typed family configurations retain that nested value
after geometry parsing; cache identity, backend metadata, and complete or
partitioned model reporting derive it from the same retained field. The
portable `LoadedTextModelConfig` requires both identities, and the
facade carries the resolved family through artifact loading rather than exposing
the effective type under the ambiguous `model_type` name. Portable
capability records, execution telemetry, rank-local parallel summaries, and
layerwise residency metadata likewise expose this identity only as
`effective_model_type`; serialized telemetry and capability records use that
same unambiguous key. The architecture resolver and facade retain the already
resolved `ModelKind`; backend model materialization stores only the selected
architecture-erased executable and does not reconstruct or pair family identity
from the effective type. Architecture-owned typed partition dispatch
retains that same admitted `ModelKind` for every rank-local partition; the
partition does not derive another family identity from its parsed variant or
nested configuration. Both identities are invariant across replicated,
tensor-parallel, and pipeline-parallel placement.

Architecture checkpoint modules also own canonical name translation and the
complete derived-weight recipe catalogs for static modules, execution units,
and independently resident experts. Concrete backends consume those neutral
recipes for binding, materialization, sharding, and transfer; they do not
redeclare family-specific stacking, concatenation, reshaping, normalization,
or recurrent-weight transformations. This includes format-dependent recipes
such as fused projection assembly and recurrent transition conversion; backend
composition may inspect recipe outputs but does not construct their equations.
Checkpoint binding is always described against the global architecture. A
rank-local tensor- or pipeline-parallel architecture describes executable
geometry only and must not become a smaller checkpoint schema. Load-time
quantization therefore constructs distinct global source and target parameter
descriptions, validates each side of the transform against the corresponding
description, and only then filters target bindings through realized stage
ownership.
Architecture-owned SafeTensors conversion plans likewise enumerate every exact
dense source, packed-weight output, scale companion, optional affine-bias
companion, and the complete output model configuration. Concrete backends
execute those plans literally. They do not select tensors by suffix, rank,
dtype, size, or substring; derive companion identities; canonicalize alternate
names; or inject compatibility metadata into `config.json`. Conversion fails
closed when a declared source is absent or any declared output collides with
another checkpoint tensor. Conversion obtains payload paths from canonical
`eredu-checkpoint` strict SafeTensors discovery. Duplicate index keys, empty
mappings, absolute or traversing shard names, missing payloads, contradictory
index mappings, and symlinks outside the admitted checkpoint access root
therefore fail before a concrete backend can publish any tensor. Neutral stores
share its index parser and path admission but defer payload-header validation
until a tensor in that shard is requested, preserving remote-shard skipping.
Hugging Face snapshot symlinks remain confined to their repository, including
its sibling `blobs` directory.

Portable SafeTensors schemas use released checkpoint names directly. A private
module spelling that inserts `inner` into an architecture name, such as
rewriting `projection.weight` to `projection.inner.weight`, is neither a
portable alias nor an accepted alternative layout identity. The `inner` path
segment is not reserved, however: an architecture may itself declare
`projection.inner.weight`, which remains distinct from `projection.weight`.
Backend operators expose architecture parameter identities through their
neutral `Parameterized` topology, including when native storage uses private
module slots. Generic binding, residency, and distributed-planning utilities
accept that topology rather than a backend module's raw physical parameter
tree, consume its exact identities, and never normalize path segments. A
single `eredu-runtime` binding-plan vocabulary records the stable target,
expected shape, dtype and bytes, direct or derived source, explicit source
sharing, overlays, exclusions, and rank ownership. Its generic
`ParameterBackend` orchestration preflights every recipe and immutable/mutable
module traversal before payload or native work, completes all materialization,
then validates and publishes the whole binding atomically. Layerwise and
addressable-bank executors reuse the same declarations while retaining their
backend-owned leases, storage, movement, and scheduling. A
concrete backend must not expose its physical-slot traversal, parameter
wrappers, training/freezing state, or nested parameter maps as a second public
parameter framework. Composition-facing backend types implement only the
neutral topology, and checkpoint loading, mutation, and byte accounting all
traverse that topology with exact missing/unexpected-name validation. A
backend-only physical slot is excluded explicitly by the operator's topology
mapping; its name, shape, neighboring slots, or storage dtype never determine
whether it is checkpoint-backed. Private native topology must not widen or
rewrite the checkpoint contract. Backend loading traverses the declared
topology, so any canonical identity-to-private slot association is fixed when
the operator is constructed rather than inferred from a checkpoint name.
Load-time quantization consumes explicit
weight, scale, and optional affine-bias relationships from the same topology;
the strict loader neither derives companion names nor treats backend-only
placeholder slots as checkpoint-backed parameters. An architecture identity
containing an `inner` segment is therefore loaded exactly like any other name.

The underlying catalogs are model-wide and configuration-derived, but each
family's architecture checkpoint API selects the complete recipe group for a
static module set or canonical execution-group/unit address. Backend adapters
must bind that selected group as a whole and fail if any recipe remains
unconsumed; they do not intersect recipes with backend-native module topology.
This keeps source-name discovery, parameter ownership, and recipe-group
selection together in the architecture.
Architecture checkpoint APIs also derive rank-local independently resident
expert recipes, including segmented fused-projection selection; backend
composition binds those recipes without reconstructing the segment equation.
Family recipe APIs derive their own checkpoint roots; in particular, Gemma 4
and Qwen expert recipe callers provide only the catalog, normalized
configuration, and layer identity, never a backend-selected layer namespace.
Independent expert residency is exposed to composition as a validated neutral
architecture catalog. Each entry carries expert identity, its owning group,
unit index, parameter path, and placement. Architecture composition maps that
identity to a generic `ParameterBankKey` and projects exact atomic bindings,
source bytes, and selected executable bytes into `AddressableBankMember`
records. Backend cache policy and telemetry refer only to those generic keys,
access classes, storage tiers, compact banks, and bytes.
Each architecture entry also carries exact acquired-bank binding names, logical
parameter targets, and checkpoint-derived recipes. Each parameter declares
whether it must be preserved or is a load-time-quantizable projection;
quantizable projections carry exact local scale and affine-bias companion
binding names. Architecture composition validates catalog targets, owners, and
grouped-plan coherence before producing the neutral members. It compares every
member recipe to the exact one-expert selection of the admitted whole-bank
recipe, including inferred physical weight, scale, affine-bias, MXFP4, and
block-FP8 geometry. The union of routed plan addresses and catalog paths must
also exactly match the independently derived family layer schedule, so a
coordinated omission from both plan and catalog is invalid.

Qwen hybrid recipes resolve admitted checkpoint aliases and assemble weights and
FP8 scale companions through the same construction for whole banks and individual
experts. Checkpoint inverse scales become the architecture's declared grouped
scale parameters; derived scale companions retain floating-point storage rather
than inheriting the weight's FP8 format. MLX represents FP8 values and exponent
scales as byte arrays while preserving their source encoding and byte accounting.
Its grouped adapter supports floating-point and UE8M0 scales with 128-by-128 blocks,
and its capability report applies that geometry to both direct and derived FP8
weights. These representation rules remain backend mechanisms; checkpoint aliases
and companion identities remain architecture policy.

Expert identity is not a backend ownership address. The backend consumes the
resulting atomic declaration directly and never infers eligibility or companion
identity from binding spelling, dtype, or rank. Addressable exclusions in the
ordinary replicated binding plan expand each architecture-selected primary to
its exact generated output companions, so quantization scales and affine
biases cannot be loaded twice. Family code owns
sparse-layer selection, routed versus shared-bank scheduling, expert counts,
and cache-layer numbering. Physical checkpoint keys excluded from ordinary
layer residency while experts are independently resident are projected from
the complete architecture catalog; backend composition does not rediscover
them by walking family layer policies or rebuilding per-layer recipes.
This applies uniformly to Gemma 4, Muse-Glimmer, DeepSeek, GPT-OSS, LFM2,
Kimi Linear, and other sparse families: each architecture checkpoint module emits its complete
`ExpertResidencyCatalog`, including compact acquired-bank names and every
per-expert or rank-local selection recipe. Replicated routed architecture
composition consumes the catalog and publishes only generic bank members to
the backend adapter. Adapters for distributed, composite, or
prediction graphs may filter it to the exact realized partition but do not
calculate a parallel expert topology.
Resident rank-local expert banks follow the same rule: architecture checkpoint
APIs select canonical parameter outputs for the assigned global expert IDs and
push those selections through their derived recipes. Backend composition only
matches the resulting recipes to exact logical targets before applying generic
tensor-parallel placement; it does not choose an expert axis or reconstruct
selection geometry from the physical checkpoint layout.
Cached and resident expert banks use that same tensor-parallel layout as their
ordinary routed execution path. Generic backend transforms preserve every
leading bank dimension while sharding only the architecture-declared projection
axis, and their returned partial or reduced output follows the same neutral
tensor-parallel contract as a directly resident bank.
Distributed cached dispatch is expressed once per expert-bank operator shape,
such as gated-product or ReLU², rather than through family-named forwarding
wrappers. Family composition supplies the architecture-declared bank spec and
cache-unit identity to that operator-shaped backend path.
Hybrid target/MTP families additionally declare the execution group, physical
MTP unit, checkpoint root, and cache identity of every sparse unit in this
catalog; backend adapters filter catalog units against the realized partition
using that canonical group-local owner address, without recovering placement
from flattened cache identities, target-layer counts, or inferred MTP ownership.
Pipeline logits follow the realized partition's output flag. Persistent state,
including prediction segments, follows the architecture's resolved state
partition plan, while prediction modules follow their realized execution-group
or architecture-declared static-role ownership. A stage's raw layer-range
endpoint is not an ownership signal. Backend stage metadata exposes input and
output ownership from the realized partition; pipeline coordinates describe
transport adjacency, not boundary ownership.
Composition lowers architecture catalog entries into native storage requests;
reusable backends do not rebuild the schedule, receive expert identity,
instantiate a family block, or match parameter targets by substring or suffix.
They also normalize physical checkpoint format metadata onto canonical runtime
parameters, including fused expert projections, so each backend consumes the
same family-specific quantization identities. Load-time quantization derives
its complete validated target configuration and format map, including mixed
checkpoint-policy replacement, aligned vision projections, and target and
prediction-unit roots, from architecture checkpoint APIs. Resident, bounded,
and distributed backend paths consume those same derived configurations rather
than rewriting family quantization fields. GGUF adapters hand the architecture
only canonical per-weight formats; architecture-owned `with_checkpoint_formats`
derivations normalize family names, partition composite text/media formats,
preserve native expert encodings, and return the complete validated source
configuration. The same rule applies to external assistant configurations.
Pipeline telemetry likewise resolves
checkpoint ownership from architecture-declared execution-unit paths rather
than synthesizing family layer prefixes. Catalog policies declare any
admitted checkpoint-only names by exact key, prefix, or suffix, and composite
checkpoint schemas partition canonical projector formats by architecture-owned
component identity; backend composition does not repeat those naming rules.
Gemma 4 and GPT-OSS expose the same complete load-time target-configuration
derivation as Qwen, so replicated and pipeline composition consume identical
format replacement policy while GPT-OSS native expert formats remain intact.
Catalog-dependent configuration normalization likewise receives a neutral
physical tensor-name catalog: the architecture owns exact-name probes and the
semantic configuration choices they control, while concrete backends only
adapt their native catalogs to that contract.
Shared projector formats retain their architecture-selected execution mode in
the normalized configuration. In particular, Qwen3-VL DeepStack and Qwen3.5
window-scheduled position, attention, and merger semantics are selected while
parsing the projector and consumed unchanged by inspection, checkpoint
planning, and execution.
Qwen projector admission also produces its GGUF checkpoint plan from the
admitted family composite. Family eligibility, required vision mode, and
decoder/projector width compatibility are architecture policy; backend
structural validation and materialization consume that composite plan without
rebuilding those rules from independently parsed text and projector
configurations.
Packed recipes expose format semantics rather than a concrete accelerator's
storage units. In particular, canonical MXFP4 expert recipes describe logical
F4 values regardless of whether a source stores byte blocks or integer words;
a backend adapter lowers those terminal recipes to the storage geometry
required by its kernels before constructing runtime bindings.

Parameter-class selection follows the architecture's validated parameter
description. Backends select exact targets by semantic role, retain the
declared static or execution-unit owner, and obtain sharding prefixes from the
architecture's canonical unit path. Checkpoint-only physical companions or
source-layout projections are exposed explicitly by the architecture
checkpoint contract. Loading, residency, and quantization code must not infer
any of those facts from substrings in checkpoint names. External-expert
tensor-parallel loaders validate the description against the neutral
architecture and register its owned groups directly; they do not reconstruct
target/prediction unit counts or enumerate family layer groups themselves.
The description includes replicated pinned and execution-unit groups alongside
sharded groups, so backend lowering does not omit multimodal units or rebuild a
family-specific subset of the topology.

Every layered family implements the state-independent
`ArchitectureParameters` supercontract. It is the authoritative source for the
complete mutable `StateLayout`, owned `ArchitectureParameterDescription`, and
role-addressed pinned parameter modules. The parameter description selects the
roles owned by a partition, and the architecture resolves each role to its
parameterized module for binding construction or mutable population. A backend
may materialize and shard those visited modules, but must not map roles to
family-specific fields, residency identifiers, or checkpoint roots. Stable
parameter identities on the visited module are the checkpoint-binding
namespace. `LayeredArchitecture` extends this contract with execution against a
specific runtime-state realization; geometry and binding therefore remain
callable without choosing that state type. Compatibility identity remains the
single consumed value returned by `ArchitectureParameters::state_identity`;
the execution lifecycle does not publish a second, ambiguous identity getter.
Concrete backend adapters use one shared visitor rather than family binding
tables.
MLX's unloaded floating parameter handles admit F16, BF16, and F32 source
storage through the neutral binding target's explicit source-dtype allowance.
Materialization preserves the selected recipe's dtype, bytes, and values;
the placeholder's initial F32 dtype does not request a conversion. Packed and
integer parameter handles retain exact dtype matching. The neutral binder
continues to validate exact shapes, selected sources, and declared dtype
allowances without inferring native storage flexibility.
Partitioned materializers use that same visitor for both ordinary decoder
families and multimodal families, leaving tensor-parallel binding selection generic and
the role-to-module mapping entirely within the architecture. Distinct pinned
modules use distinct storage roles even when they share a broader semantic
parameter class; for example, DeepSeek V4 exposes its target hyper-connection
head as `hyper_head` and its optional DSpark static bundle as `mtp`.
Architecture implementations also provide any released-checkpoint rewrite
recipes for their static modules, so alias and fused-layout handling does not
reintroduce checkpoint roots into a backend.

`eredu-checkpoint::LinearFormat` is the single public description of a matrix's
physical checkpoint encoding; `eredu-nn` does not re-export that checkpoint-owned
type. `eredu-nn::LinearFormatSpec` combines the encoding with the exact scale and
affine-bias companion parameters required by neural construction.
Ordinary linear, embedding, group-selector, and grouped-projection construction specifications use
it directly, and architecture parallel plans return the same declaration for
encoded parameters. Neutral runtime code derives packed shapes and remaps
declared sharding geometry from that typed declaration. Architectures may
centralize a released checkpoint's standard naming convention when creating
the declaration, but runtime and reusable backends never decide that a
parameter is quantizable or construct companion identities from weight,
projection, scale, or bias name suffixes. Native backend slot names are local
implementation details mapped to the literal neutral identities.
Neutral parameter traversal preserves each companion's scale or affine-bias
role together with its primary weight identity. Load-time quantization uses
that semantic link to build bounded targets whose weight and companion output
names are all explicit; bounded materializers reject missing or colliding
identities and never manufacture them from the weight name.

Routed architectures retain and expose their architecture-owned construction
specification. Architecture/runtime composition translates that semantic
geometry into grouped projection specifications before invoking a backend.
Resident, cached, and distributed execution consume the same projection
encoding, bias layout, and activation policy. Each projection declares the
exact identities of its scale and affine-bias companions; reusable backends
bind those identities to native slots without imposing projection suffixes or
synthesizing checkpoint names. Architecture APIs derive localized grouped-bank
specifications for placement-resolved expert counts and projection widths while
preserving canonical parameter identities and physical formats.
Cache-backed distributed callbacks receive the specification from the resident
unit bank or directly from that unit's realization-plan entry, including for
ReLU-squared banks and appended prediction units; they never rebuild it from a
family configuration and layer index.
Ownership and localized construction are published together as an
`ExpertRealizationPlan`. The plan contains the checkpoint-global expert count,
the complete global-expert-to-owner map, the current rank's global expert IDs,
and the exact rank-local bank specification for every routed execution unit.
Each plan entry uses the same canonical execution-group and group-local unit
address as the architecture execution layout and expert-residency catalog;
group aliases are not interchangeable ownership addresses.
Architecture/runtime composition consumes the plan's global count and lowers the
declared owner map into group indices plus ordinary collective calls without
running another assignment policy. The plan's presence or absence is also the
only authority for whether the realized architecture has routed execution
units; composition must not inspect a family schedule or configuration count
to decide expert availability. Rank-local partition telemetry exposes local expert
identities only when the plan enumerates a routed unit owned by that stage, and
expert-cache admission likewise follows plan presence rather than a family
variant predicate. Backend composition must not pass family fields
or a separately derived tensor-parallel width into expert-bank construction;
the bank specification retained by this same plan is the only construction
input.
The neutral expert-exchange primitive derives a stable destination-major route
permutation from that same ownership plan, agrees exact source-major peer
counts, and uses the transposed counts for the return path. It transports
owner-local identities, route scores, coefficients, and opaque route-position
tags in row-aligned order. Local addressable grouped execution consumes every
received row once and applies its coefficient once; the source rank accepts
returned rows only when their tags reproduce the dispatched permutation, then
additively combines duplicate routes into architecture source-token order.
Zero-count peers remain explicit. Generic movement and exchange mechanisms own
native tensor storage, selected dtype/count/shape enforcement, exact
completion, and resource retention. Production routed architectures bind this
primitive through their prepared routed execution handoff. MLX provides generic
arbitrary-row gather, flattened route-value gather,
scatter-add, and addressable-provider adapters for this primitive. It
also constructs the neutral `PartitionExpertRouteExchange` directly from an
already-realized opaque `PartitionCommunication`; it does not infer an expert
axis or create another group. MLX variable all-to-all admits the floating route
rows and I32 identity/tag rows with the same exact count, shape, completion,
and resource-retention contract while other integer dtypes remain fail-closed.
Qwen, Qwen3-VL, Qwen hybrid (including conditional vision and embedded MTP),
GPT-OSS, LFM2, Kimi Linear, Nemotron-H, Muse-Glimmer, Inkling, Gemma 4, and
DeepSeek V3/V4 expose family-specific realization entry points over their
constructed neutral architectures. These entry points select routed units,
preserve canonical parameter formats, and apply planner-derived local widths
before grouped backend mechanisms are selected. Family composition accepts the
plan rather than the family configuration or parallel topology when creating
dispatch assignment. The adapter exposes no independent balanced, round-robin,
or explicit policy engine. DeepSeek pipeline unit factories install the plan before
constructing target or prediction units, and tensor-parallel expert-cache
selection derives
its local width from the same plan entries rather than from family arguments.
Distributed architecture callbacks also
carry whether the requested result is globally complete or a rank-local
tensor-parallel contribution, so EP recombination preserves the reducible and
post-reduction terms without inventing or repeating a TP collective.
Observed routed execution calls the neutral `RoutedLayeredArchitecture`
observation hook for each unit. Its default uses the architecture's semantic
module path and expert cardinality around provider execution. Families with
shared or post-routed work override the hook and report routed, shared, and
combined tensors at the actual combination point, then apply intervention to
the combined output before downstream execution. Composition only adapts
native tensors to the neutral observer and cannot invent family path segments
or claim that a provider-only output is the family result.
Activation observers likewise derive unit input, output, and nested operator
names from the architecture's canonical `unit_path`; concrete composition must
not reconstruct a family path from a group or layer index.
The layered architecture also names group-ingress and group-output seams. The
shared resident and bounded drivers apply intervention at those seams, which
exposes projected vision, projected audio, and decoder-width multimodal merge
values at their real consumption points. `PreparedProcessor` exposes its final
ordered payload and metadata tensors under the processor output path before it
rebuilds their coupled identity.

Architecture capability estimates likewise own native and effective context,
accepted modalities, cache ownership and sharing, attention windows,
and runtime-state memory metadata. Their `StateMemoryLayout` wraps the exact
ordered `LayerCachePolicy` schedule copied from the architecture's executable
`StateLayout`; it is not a separately summarized scalar geometry. Admission
derives attention, recurrent, convolution, pooling, and embedded-prediction
bytes from those policies. Generic `Floating` tensors use the backend's
physical floating-state width, while fixed-dtype tensors such as FP32 recurrent
matrices use the width declared by their architecture policy. Concrete backends
add live allocator, residency, and system-memory observations; they do not
reconstruct family state geometry or maintain an independent layer count.
The serialized `CacheStateStrategy` is a public semantic summary of that same
layout, not of attention equations or checkpoint labels. In particular,
DeepSeek V4 reports every target and embedded-prediction layer as bounded
key-only state and separately counts the layers that own append-only pooling
state; compressed-attention layers are not full-KV layers.
The same exact estimate declares whether speculative draft weights use a
separate checkpoint, use configured embedded prediction layers, or are absent.
A backend maps that declaration to executable or unsupported status according
to its implementation; it does not maintain a family-name MTP table. Typed
prediction-extension admission reports unsupported declarations explicitly;
only an absent architecture declaration reports unavailable.
Prediction-enabled execution is composed exclusively through the typed neutral
target extension below, rather than by attaching prediction modules or caches
to the ordinary pipeline shell.

Speculative verification also carries a positive, caller-selected completion
deadline and cancellation disposition in the neutral scheduler options before
any target submission. The scheduler resolves an incomplete verification when
that deadline expires instead of polling forever. A backend completion must
then either finish native cancellation or transfer the completion and every
live retained resource to a quarantine owner; only exact completion permits
observation, cache commit, or publication. MLX uses a thread-affine quarantine
because its events do not expose native cancellation. Terminal native evidence
permits reclamation; thread-runtime teardown does not wait and permanently
retains any owner for which that evidence is still unavailable.

An admitted artifact with embedded prediction is split before partitioned
construction into an ordinary target projection and a typed prediction
extension. The target retains authority for admission, communication,
ordinary prefill/decode, state transactions, cache control, and public output;
the extension retains only its architecture-selected units, private lane
state, and draft depth. Architecture preparation validates the extension
identity and depth, derives rank-local placement, constructs paired
checkpoint-global and rank-local neutral units, and fixes recipes plus
prediction-only state geometry in a consumed typed handoff. MLX receives no
family configuration or semantic rank topology at this seam; it only lowers
the prepared bindings, moves tensors, populates local modules, and wraps native
execution objects. The architecture layer also owns the exhaustive
post-materialization topology and mutable lane-state variants; MLX implements
one generic materializer for arbitrary prepared modules, pooling caches,
sequential caches, and complete state layouts instead of repeating a
family-shaped construction match. Its physical sources are the exact admitted sources
not claimed by the projected target schema, rather than a backend prefix
filter. MLX materializes those units against the target's selected local
geometry carried by the handoff and invokes them through the neutral prediction-target operation
hook. This does not make every prediction family or topology supported:
extensions without a dedicated typed composition fail before target payload
construction. An excluded prediction selection cannot fall back to a complete
or pipeline target shell.

Conditional Qwen hybrid prediction units include their routed expert banks in
the exact extension parameter tasks. The generic materializer populates those
resident banks, and prediction uses the neutral resident expert provider
independently of the target's expert cache. Fused QKV and gate/up placement
maps FP8 scale companions into block-row coordinates while retaining logical
weight-row coordinates for the weights themselves.

For SafeTensors and GGUF materialization, architecture preparation also
identifies the checkpoint parameter that establishes the generic floating-state
dtype and resolves its schema-declared physical aliases against the inspected
catalog.
Dense GGUF scalar types remain typed in that portable catalog, so a concrete
backend can preserve the width used by its loader; packed encodings instead map
to their known materialized activation dtype. Generic backend composition must
not recognize family checkpoint names or silently choose a width when the
declared source is missing.

Runtime state follows the same contract at materialization time. Architecture
`StateLayout` values are authoritative for per-layer attention windows,
append-only pooling streams, and named segment boundaries and lifetimes.
Concrete backends derive native cache objects directly from those policies;
cache advancement likewise comes from the constructed architecture graph and
its state layout. Advertised draft proposal capacity is a distinct
architecture-owned graph property: sequential MTP derives it from prediction
depth, while fused DSpark derives it from the validated block width rather than
the number of DSpark layers. Parsed model-family arguments are not a second
source of execution or state geometry. Architecture preparation obtains the
layout through `ArchitectureParameters::state_layout` on the realized
architecture before the proof-bearing handoff enters neutral composition.
When a family owns additional state outside the ordinary layered target, its
architecture publishes the target, prediction placement, and composite
persistence layout together; its dedicated execution contract consumes that
value without reassembling segments or recovering offsets from layer-count
fields. Ingress state is a distinct transient contract when it does not span
that composite persistence layout. Inkling publishes its realized target-only
ingress layout explicitly: ordinary execution receives global target geometry,
parallel execution receives rank-local target geometry, and neither receives
embedded-prediction state. Both neutral forward entry and concrete pipeline
allocation consume that same architecture-owned layout. Prompt-cache identity
for both replicated and pipeline execution is derived from
`ArchitectureParameters::state_identity` and a canonical `PartitionState`; a
backend lowers only native storage or parallel topology. Pipeline execution
uses its exact placed partition, while neutral replicated composition attaches
the architecture's complete realized state layout at global offset zero.
Target or embedded-prediction cache allocation uses that identity and does not
recreate a global layout, offsets, or family identity after placement.
Pipeline cache validation likewise consumes the placed partition's localized
`StateLayout`, including its architecture-global offset; it does not reread a
family attention schedule or interpret configuration fields as a second cache
policy.
Composite model layouts, such as a target decoder plus embedded prediction
state, are
assembled by the architecture before a backend consumes them. Architecture
capability construction consumes that same composite layout for admission
accounting, so prediction segments cannot be admitted with target-only byte
estimates. Architecture state segments also declare their processed-token
frontier offsets. Runtime
preserves the named segment ranges in prompt-cache model identity, descriptor,
and manifest data while expanding their frontier offsets per layer. Segment
selection validates the architecture-declared ID and rebases its exact range;
contiguous partition slices preserve and rebase the intersecting segment
metadata. When separately materialized prediction state follows target state,
the architecture's rank-local geometry publishes its `PartitionState`, including
the architecture-global offset; a backend does not recover that offset from a
family layer-count field.
The public session returns this identity as one value. Descriptor construction
copies all model-owned fields from that value atomically and accepts only the
checkpoint fingerprint, processed-prefix fingerprint, and batch size from the
caller; backend adapters do not expose a parallel set of field-level identity
getters for callers to reassemble.
Each pipeline descriptor covers exactly the state executed and owned by that
stage. Required fixed components are persisted only on their architecture-owned
global layer, even when their current value also travels in an auxiliary
boundary for downstream execution. Prediction segments appear only on the
partition that owns them; topology coordinates and family-wide layer counts do
not synthesize additional cache coverage. When target and prediction caches
share a backend residency manager, prompt-cache persistence selects blocks by
the descriptor's owned global-layer range; blocks for another segment remain
live in the manager but do not enter that descriptor's manifest.
Architecture identity functions declare family, fingerprint, composite global
layer count, and placement; backends must not reconstruct family identity,
target/prediction boundaries, DSpark behavior, or shifted-prediction offsets.
Architecture-erased backend executables own the correctly typed execution
value in the same exhaustive outer variant; the replicated variant owns the
complete neutral session that pairs its architecture and concrete state. Model
state is not exposed as a second extensible erased enum: prefill, decode,
inspection, prompt-cache, residency, parallel, and speculative hooks dispatch
on the executable, so a new execution variant makes every operation site
non-exhaustive until its behavior is defined.
Draft commit and pipeline prompt-cache persistence likewise select the
architecture's named prediction segment; family configuration layer counts are
not commit-range metadata.

Pre-materialization capabilities follow the same rule. The normalized
architecture derives whether independently addressable routed experts exist
from its exact target and prediction layer schedule; a completely all-dense
schedule reports neither expert residency nor expert-parallel support even when
the family also admits routed variants. The selected backend intersects that
fact with its expert-cache
materializers. Core selects the neutral expert-cache route but does not infer
support from a family name, because one family may contain both dense and MoE
variants. Nonresident SafeTensors load-time quantization follows the same
intersection: the architecture declares whether its normalized parameter
topology can be transformed before bounded materialization, and the backend
declares whether its family composition implements that route. Core preserves
the neutral quantization and residency request without maintaining a family
allowlist. The runtime-owned normalized load request carries
`QuantizationRequest`, whose variants describe load-time transforms only; the
MLX request is a thin native-device adapter over that request. Checkpoint
storage encodings such as native GGUF blocks remain internal to artifact
inspection and materialization.

Routed execution classifies prefill and decode in the neutral request or
layered-architecture driver. Provider-backed backend wrappers invoke pass-free
neutral entry points; they neither derive nor receive the semantic pass.
Before an independently addressable parameter bank is invoked, the neutral
driver projects that pass to the exact storage workload class (`Bulk` or
`Incremental`). Reusable backend bank and grouped-execution APIs accept only
that workload class; accepting a semantic expert pass through a generic
conversion would make the backend mechanism another interpreter of execution
policy.

The same architecture capability report carries a typed parallel plan for
tensor sharding, pipeline staging, and expert partitioning. Each axis is
declared from the parsed family variant rather than inferred from parameter
addressability or a broad family identity. Independent expert residency
remains a separate artifact capability because it does not imply an
expert-parallel execution plan. Distributed composition consumes these facts
from that exact normalized report instead of reconstructing support from
raw or wrapper `model_type` values. GGUF inspection, planning, and
materialization all validate requested preparation against the architecture
plan retained by portable admission. A backend must not reparse the raw GGUF
checkpoint to derive a second capability report. SafeTensors capability and
floating-state dtype derivation likewise accepts only the typed architecture
plan retained by admission; raw JSON is parsed once by the architecture
registry.
Architecture group helpers treat a semantic partition count as a preferred
domain and refine it to the largest count shared by every partitioned physical
member. This includes weights constructed directly in a packed format and
their scale or affine-bias companions, so one group-level range always maps to
integral physical ranges without format-specific backend planning.
Stage-local load-time transforms validate pre-transform bindings against the
source architecture description and post-transform bindings against the target
description. Static binding traversal is restricted to the roles owned by that
pipeline stage, so a packed overlay need not materialize companions for an
unowned stage merely to construct the local adapter.
Validated architecture parameter descriptions retain their canonical
execution-unit layout alongside owner-tagged parameter groups. Pipeline
composition consumes those declared group ranges and flat unit ordinals for
target and prediction placement; it does not rebuild target/MTP counts from
family configuration fields. Pipeline topology preflight and stage-range
selection likewise use the decoder group count from that description, before
constructing any rank-local units.
Placement retains the neutral `ArchitectureGroupKind` vocabulary directly;
backend composition does not rename architecture-authored semantic kinds into
an untyped execution-group category.
Rank-local materialization traverses the canonical units exposed by its
`ArchitecturePartition`, so composite vision, target, and prediction ordering
is never restated by a backend loader. Dense-stream storage indexes those
addresses directly and derives its primary execution offset from the first
architecture-declared decoder address rather than summing preceding group
counts. The generic rank-local binder consumes tensor-parallel groups from the
complete architecture parameter description; it does not enumerate static,
media, decoder, or prediction parameters independently. The same description
supplies the ownership used by pipeline-stage materialization, preventing
tensor and pipeline planning from drifting apart. Conditional Qwen pipeline
boundaries come from the constructed architecture, including hidden width and
DeepStack activation cardinality, rather than being re-derived from family
arguments in backend composition.

Pure EP, PP+EP, TP+EP, and TP+PP+EP must share the architecture's capability
validation, expert assignment, residency plan, and execution declarations. A
backend must not introduce a separate family-dispatched ownership policy for
one topology combination; the same neutral plan covers multimodal and
hybrid-state MoE families.

Artifact inspection also consumes architecture-derived input modalities from
that exact normalized configuration. Backends translate those neutral flags
into report and build-feature readiness, but do not infer image, audio, or
video support from a family name. Text-only and partially multimodal variants
therefore do not acquire processor or feature requirements they cannot use.
SafeTensors materialization retains the typed normalized family configuration,
its complete architecture-derived checkpoint plan, the selected physical
layout, and the validated tensor catalog from the preparation plan.
Catalog-dependent architecture admission is finalized against that neutral
catalog before a backend receives the plan. The
architecture registry validates every family against its exact checkpoint plan;
for Moshi it additionally proves canonical recipe publication and retains the
recipes in the admitted plan. Concrete backends consume this admission proof;
they do not repeat generic catalog validation, family-specific recipe
validation, physical-layout selection, or raw-JSON parser dispatch. Reopening a
checkpoint only verifies that its metadata still matches the admitted portable
catalog before the retained layout is applied.
Catalog strictness belongs to that architecture-owned checkpoint schema;
residency and materialization options cannot weaken it. Conversely, tensors
left unclaimed by an explicitly non-strict schema remain outside the resolved
store presented to backend parameter binding.
Family composition does not reopen `config.json`, rediscover checkpoint shards,
or select a second catalog after admission. Every admitted artifact plan
retains the normalized `ModelKind`; SafeTensors plans additionally retain typed
family geometry and the checkpoint schema, while GGUF plans retain the exact
`GgufArchitecture`, typed family geometry, and architecture-derived main
checkpoint schema. GGUF plans also retain the complete canonical mapping from
each physical tensor output to its logical parameter name. Translation and
collision detection therefore happen once during architecture admission; they
are not family-dispatched again during backend materialization. Portable GGUF
materialized groups likewise carry the logical output identities from their
validated container catalog; concrete backends consume those names and do not
reconstruct quantization companion suffixes from physical names. Core keeps the
corresponding `ValidatedGguf` proof intact inside `ModelArtifact` until the
selected backend consumes it; materializers do
not downgrade that proof to an unvalidated checkpoint handle and rerun the
architecture parser or regenerate the main checkpoint schema. Backend
composition may enrich a clone of retained geometry with native encoding
descriptors, while composite families retain their separately admitted companion
schema; neither operation replaces the retained main artifact plan. Portable
architecture admission derives that companion geometry,
validates the companion against its exact family-owned GGUF schema, and retains
the typed composite configuration, schema, and canonical tensor mapping in the
artifact architecture plan. A concrete backend pairs that proof with its native
payload wrapper and passes the pair through replicated and partitioned
composition. It does not repeat companion family parsing or regenerate the
companion checkpoint schema. For Gemma 4, Inkling, Muse-Glimmer, and
Qwen, inspection also parses and retains the family processor plan from the
admitted model, projector, and SafeTensors processor sidecars. Materialization
consumes that snapshot directly and neither resolves family identity again nor
rereads processor sidecars or GGUF metadata to reconstruct preprocessing
policy.
The catalog-bound store revalidates tensor metadata when leases are
acquired so later header changes cannot silently replace the admitted catalog.
For composite GGUF artifacts, the architecture additionally owns required
versus optional companion roles, filename matching scope, search depth, and
dense-only versus dense-preferred admission. Portable inspection resolves
those declarations once, classifies dense and quantized candidates from their
GGUF tensor catalogs rather than filename substrings, and retains the exact
path and checkpoint handle in `ValidatedGguf`. After resolution, the
architecture registry reconstructs the composite family from the already
admitted primary geometry and companion header, validates the companion's exact
tensor schema, and retains that typed companion plan alongside the primary plan.
The architecture also owns one preparation plan that declares whether a media
projector is inapplicable, optional, or required and maps a structurally
validated projector to resulting input modalities. Portable companion
resolution, backend inspection, and materialization consume that policy rather
than matching families independently, so admission, expected modalities,
multimodal readiness, and loader requirements describe one composition.
Muse-Glimmer's base GGUF is a complete text decoder; its optional official
image-only projector adds the vision graph, parameters, processor plan, and
image modality without changing text checkpoint admission.

Multimodal preprocessing starts with an architecture-owned processor plan.
That neutral plan parses family model and processor metadata, selects released
defaults, declares framing token IDs and text, chooses frame sampling and
grouping, and derives resize, normalization, patch, and signal-feature policy.
Audio feature plans specify the analysis window, exact padding and frame-count
convention, frequency bounds, mel scale and normalization, spectrum value, and
logarithm as well as their dimensions and floors.
`PreparedProcessor` interprets that plan over `ProcessorMechanisms`. The
architecture layer first executes every host transformation through
`eredu-media`, validating the complete input before any native conversion.
`ProcessorMechanisms` receives typed processed buffers and constructs native
tensors or inspects small native metadata values; architecture code owns frame
selection, patch packing, framing order, metadata, and the ordered
`PreparedModelInput`. A new backend therefore supplies tensor conversion and
inspection mechanisms without implementing preprocessing mathematics or a
family processor.

Replicated composite selection combines the exact replicated-text realization
with one `SelectedProcessorExecution`. It admits raw media, prepared tensors,
and projected embeddings per modality and carries that decision into the typed
architecture handoff. Other execution classes select the same processor value
before construction; partitioned workers select prepared tensors without raw
media and therefore cannot attach a raw processor from the artifact plan. The
backend lowers only the selected mechanisms and cannot choose a second input
route. Gemma 4, Inkling, Muse-Glimmer, Qwen3-VL and its sparse
variant, and conditional Qwen3.5 use this handoff whenever their replicated
configuration has no separately placed prediction group. Partitioned,
prediction, assistant, and realtime responsibilities remain distinct execution
classes.

`PreparedModelInput` retains the ordered payload and metadata tensors together
with its exact descriptor identity. Architecture admission derives one ordered
part plan from that value, and `PreparedCompositeInput` couples the native
tensors to that admission before the graph can execute. Architecture-owned
composite adapters build embeddings, placeholders, masks, positions, optional
encoder roots, projectors, merge inputs, and per-layer visual contributions,
then enter the ordinary layered driver. Direct and routed composite execution
therefore use the same replicated session for prefill, decode, transactions,
state reset, cache persistence, observation, reports, and exact completion. An
admitted composite never falls back to a family pipeline implementation:
unsupported prediction or artifact forms fail during architecture selection,
before communication realization or payload materialization.
Complete, tensor-parallel, and pipeline composition use the same family ingress:
the MLX layer converts handles and pairs admitted tensors, while architecture
code assembles placeholders, batching, positions, masks, and ordered parts.
Qwen pipeline prefill and embedded prediction token identity both consume the
same architecture-prepared value; neither path recreates placeholder spans,
patch grids, or packed pixels in MLX.
Prompt-cache use additionally couples that descriptor identity to a nonempty
semantic-content fingerprint. Qwen composite architecture identity includes
both decoder and vision parameter formats as well as media-token policy. Full
session checkpoints retain the committed
pair; state-only rollback, reset, and an unbound cache load clear it. The
input-aware save and open operations reject a descriptor or prepared input
that differs from the successfully published prompt without changing state.
Processor output and composite execution also share the neutral activation
observer. Observation paths cover processor output, vision and audio projector
outputs, the modality merge, layer outputs, and final logits; interventions are
applied before the next declared consumer executes. Layered and placed pipeline
drivers both obtain group paths from the architecture, observe projector values
after architecture completion, install intervened values into transported
ingress, and expose the resulting merge value.
An observed processor result is intentionally unbound from exact prompt-cache
identity because an intervention may have changed native tensor content.

Composite GGUF requirements enumerate the admitted primary and media-projector
checkpoints as one physical source set. Their exact translated outputs,
provenance, recipes, source encodings, and independently selected executable
formats flow into materialization tasks. Module construction receives the
matching per-parameter format maps, so packed decoder weights and dense media
weights may coexist without a model-wide format decision. The final MLX layer
opens those already admitted sources, binds generic native modules and media
mechanisms, and erases the completed typed session.

Realtime prompt protocols follow the same ownership rule. The neutral Moshi
architecture declares PersonaPlex's released silence, sine-conditioning, and
text-padding tokens, system-text framing, accepted prompt geometry, and
ordered forced-frame plan. Architecture code resolves that plan against
validated host token storage and produces portable forced frames in enqueue
order. The common runtime ingress later performs family-blind host-to-tensor
conversion; a concrete backend does not match PersonaPlex prompt sources or
redeclare its application policy. Realtime model identity is likewise
architecture-owned. The selected architecture wrapper retains the source and
execution configurations plus the authoritative neutral realization; concrete
mechanisms are bound beneath that wrapper and do not restate a family identity.
Family metadata such as `EffectiveModelType` remains in
`eredu_architectures::moshi`.

Realtime artifact loading also crosses the backend boundary as a neutral
architecture preparation. Moshi inspection parses optional native defaults or
released PersonaPlex configuration, resolves the confined `moshi_name` or
indexed checkpoint source, validates the strict SafeTensors contract, and
publishes canonical binding recipes in `RealtimePreparationPlan`. Concrete
backends consume that plan and load tensor payloads; checkpoint-layout identity
names the physical Moshi or PersonaPlex SafeTensors namespace and never a
concrete backend. Backends do not receive a raw artifact path or reinterpret
family configuration and filename policy.

Realtime inspection derives one exact architecture requirement from this
metadata-only preparation. It includes separate source and execution
identities, per-parameter lowerings, immutable and mutable residency, speech
schedule and state geometry, topology and rank, finite invocation bounds,
observation seams, completion policy, and every required generic mechanism.
Family-blind selection compares that requirement with an independently reported
backend capability set before payload access, module or state allocation, or
route-owned queue and group realization. The resulting selected realization is
paired immediately with exact materialization tasks. Each task retains its
architecture owner, recipe traversal, admitted catalog keys, physical tensor
and output identities, source encoding, and backing shard before a backend
store or native resource exists. That prepared contract is the sole input to
construction; visitors consume it without rebuilding source semantics or
reselecting a format, residency, topology, or state profile from caller options.
The selected local adapter also
completes this inspection and selection before it realizes an MLX device or
creates execution and weight-materialization streams.

Typed Moshi architecture dispatch builds static, temporal, and depth modules
and passes them to `eredu_runtime::construct_realtime_model`. Resident and
bounded policies share that constructor and its `RealtimeLayerwiseRuntime`;
pure tensor parallelism wraps the same selected traversal in the partitioned
runtime. The implementation backend binds generic tensors,
operators, stores, state, streams, and collectives, then performs only a final
opaque erasure of the already constructed executable. Its public native handle
is this mechanism bundle, not a prepared Moshi model or semantic realtime
state, session, or scheduler.

`eredu-runtime` owns the complete frame lifecycle. Validated portable ingress,
typed coordinate payload history, schedule advancement, delayed temporal
resolution, forcing, sequential text/depth decisions, output alignment, and
history pruning execute once in the shared coordinator. The coordinator
branches model/cache state, history, schedule, samplers, and randomness as one
transaction. Moshi architecture composition enables its proven fully-forced
depth-tail omission; the neutral decision driver disables that optimization
when diagnostics are requested or any remaining decision is not forced.
`RealtimeSessionScheduler` is the only fair request lifecycle for
single and concurrent sessions, including deadlines, cancellation,
release/resume, and idle-only sampling replacement. Architecture observation
hooks compose with layered traversal and sequential decisions, so replacements
at declared temporal, layer, decision-logit, and final-output seams affect the
actual downstream transition rather than a diagnostic copy.

The MLX completion mechanism retains native input, output, diagnostics, history,
state, checkpoint stores, streams, and collective ownership. Runtime
prepublication waits for that exact completion and validates native token
scopes before host observation; only then may the scheduler publish the branch
and its portable output. Delayed or failed returned completions remain
quarantined and cannot expose host output. A scheduling failure before event
publication carries a recovery completion when work is unresolved. An outer
scope covers input materialization, traversal, sampling, and frame assembly;
it retains the entire native execution payload if any accepted work remains.
An unresolved or failed execution cannot be reused. These ownership rules are
mechanism-specific; commit and visibility rules remain backend-neutral.

Distributed realtime turns use the same neutral scheduler transaction. Before
submission, bounded consensus compares the topology-wide selected model
identity—including selected state realization and observation policy—rank
ordering, work descriptor, request, and portable frame on every participant. A
second bounded consensus round agrees the exact submission count and any local
submission failure. Following native completion and host observation, an output consensus round
compares each local outcome and a digest of the complete portable output. A
descriptor or output disagreement, peer execution or completion failure, or
indeterminate consensus discards every unpublished branch; only topology-wide
completed output agreement permits any rank to publish. The MLX transport
supplies only bounded word gathering and its retained native completion.

The Moshi pure tensor-parallel realtime extension is selected by the
architecture as a rank-local handoff. Architecture code validates its narrow
topology, derives physical parameter placement and local transformer/state
geometry from a configuration-only canonical parameter description, and projects exact
all-reduce and uneven-gather bounds into an opaque communication manifest. It
also publishes the fully local temporal/depth traversal as a neutral
`PartitionedTextRuntime` plan. Selection and manifest consensus finish before
any MLX module is constructed; MLX materialization then realizes the opaque
group before opening tensor payloads and installs the local layered runtime in the
generic traversal partition executor; it does not plan Moshi sharding,
reconstruct local family geometry, retain a distributed session in the
realtime model, or select a separate parallel forward path. Frame scheduling,
audio handling, forcing, and realtime decisions remain the distinct realtime
extension around that one neutral model driver.

Prepared-media admission follows the same boundary after tensor construction.
Architecture media plans validate family payload shapes,
patch/window/pooling geometry, valid-position masks, and artifact-specific
modality policy, then report decoder positions and conservative scalar
workspace. Admission consumes `eredu-core` input descriptors and
`eredu-runtime` prepared-input tensor containers directly; architectures and
concrete backends must not redeclare modality or payload vocabularies. A narrow
backend inspector describes native tensors and evaluates only the small signed
integer or Boolean metadata arrays required by family equations. Core extents
remain host-sized through neutral planning and are narrowed with checked
conversions only where a concrete tensor API requires it. Concrete backends
apply physical scalar widths and account for the arrays' actual byte sizes.
Architecture input-part plans additionally classify payload representations
for every family, including explicit rejection plans for text-only models. The
same plan must drive prefill materialization and capability accounting, and
both paths must consume the same runtime container and inspector, so they
cannot disagree about an accepted modality/payload pair or its metadata;
backend admission has no rank-only fallback.

Backend types also declare the optional neural and tensor operations they
support. This includes every `Tensor` method whose default implementation fails
closed, alongside the explicitly admitted `NeuralBackend` methods.
Architecture constructors preflight family-owned operator requirement sets
before allocating modules or loading parameters, so a missing tensor
construction, recurrent scan, sparse-attention primitive, normalization,
activation, media operator, or collective rejects composition instead of
failing on the first affected forward pass. Concrete backends must keep that
declaration aligned with both their `NeuralBackend` and `Tensor`
implementations.

`eredu-nn` owns tensor-independent normalization and causal-mask geometry
validation. Backends validate final-axis dimensions, epsilon, grouped gates and
divisibility, and checked sequence/offset/lookback ranges before tensor work.
Causal masks interpret lookback as an inclusive maximum past distance;
sliding-window attention uses its separately specified token-count window.
MLX and scalar reference mechanisms consume the shared geometry while retaining
their own dtype, layout, reduction, and kernel implementations.

The shared dense decoder also admits architecture-owned RMS scale offsets,
post-attention and post-feed-forward normalization, embedding scaling and output
soft caps. These policies run identically in resident, bounded and partitioned
execution; post-sublayer normalization follows TP reduction. Gemma 2 supplies
these policies and its SafeTensors/GGUF contracts without native dependencies.

`AttentionRequest::softcap` is a neutral positive score cap applied after score
scaling and before masking and softmax. Backends advertise
`ATTENTION_SOFTCAP` only when contiguous, sliding and paged attention implement
it. Paged caches retain the request policy through every block recurrence.
`AttentionCache::uses_blockwise_attention` declares when returned update tensors
exclude retained history, so the portable decoder routes chunked sliding
prefill through cache-owned attention. Neither capability identifies a family.

A backend owns runtime-specific resources and computation:

- tensors, neural operators, queues or streams, random state, and sampling math;
- model and request cache storage;
- checkpoint payload buffering and tensor materialization;
- native device discovery, allocation measurements, transfers, and kernels;
- communicators and collective tensor operations; and
- exact native completion objects and runtime-specific errors.

Backend modules expose only the backend-specific implementations of these
facilities. Checkpoint recipes, checkpoint store metadata and selections,
sampling policies, and observation contracts remain public through their
neutral owning crates rather than compatibility re-exports under a concrete
backend namespace. Checkpoint catalog, selection, mapping, and I/O failures
likewise retain `eredu-checkpoint`'s `StoreError`; a concrete backend error
covers only failures introduced while converting or materializing a neutral
lease. In particular, GGUF metadata, descriptors, encoded spans, and selection
plans are imported from `eredu-gguf`; backend GGUF APIs expose only their owned
checkpoints, materializers, and materialized tensors. Concrete-backend shard
discovery, raw tensor visitation, metadata copying, strict assignment, and
format-lowering helpers are composition details, not public alternatives to
the neutral checkpoint contracts.

Neutral contracts use associated concrete types for these values. They do not
use `Any`, untyped pointers, string-dispatched operations, or erased calls in
per-layer and per-token paths.

## Loading and session creation

Artifact loading has four stages:

1. Portable inspection validates checkpoint metadata and tensor catalogs,
   asks the architecture registry for a canonical family, neutral loading
   protocol, and companion requirements, then resolves any sibling artifacts.
2. Architecture dispatch selects the semantic execution class and state-access
   profile. Neutral selection validates the exact requested topology and policy
   against normalized architecture facts and the selected backend's reported
   mechanisms.
3. Portable planning binds the artifact description to that exact topology,
   quantization, residency, state, and session policy. Materialization options
   must reproduce the selected values rather than supplying a merely
   equivalent class of request.
4. The selected backend adapter realizes native mechanisms from the plan and
   passes its exact prepared-source bundle to `construct_prepared_execution`.
   Architecture-owned sealed routes invoke the appropriate typed construction
   contract. The backend erases the completed typed session at the outer boundary.

`ReplicatedTextMechanismSupport` reports backend facts and answers support for
one exact lowering candidate or state component. Runtime capability synthesis
enumerates primary and auxiliary requirements, validates portable geometry,
deduplicates lowering candidates and selects exact state placements. Both
native and independent scalar adapters use this same synthesizer; no provider
needs to reconstruct requirement enumeration or policy from family identity.

The total construction driver consumes `PreparedModelSources` and owns ordinary
versus partitioned communication consistency, target/extension agreement,
prediction preparation, the floating-state source, and processor preparation.
Native communication is checked against the exact retained manifest before
typed construction. `PreparedExecutionRoutes` contains only supplied native
visitors and contexts; optional routes impose no bounds on absent capabilities.
`KeyValueRoute` supports ordinary attention using the base neural and cache
traits, without requiring blockwise, grouped, distributed or prediction traits.
Complete profile dispatch uses an explicit `ReplicatedTextStateProfiles` set
and `SharedReplicatedTextVisitor`, retaining profile-specific static bounds and
the original visitor's construction-start notification.

`PreparedExecutableAssembler` receives an executable only after typed
construction succeeds, together with the exact selected state policy,
capabilities, physical floating-state width, prepared processor, and retained
native communication. MLX wraps these values; it does not repeat processor
selection, extension pairing, or the six-class dispatch.

Concrete backend preflight reports mechanisms rather than family support or a
parallel execution-class table. Architecture dispatch is authoritative for
ordinary causal text. Replicated, routed, and composite classes enter exact
typed architecture visitors; partitioned requests enter the corresponding
neutral admission and rank-local construction first. A request that selects no
such contract is rejected during architecture selection. MLX therefore has no
unclassified execution class, family binding table, or complete-model family
loader to consult. Addressable parameter banks requested for a configuration
without that neutral capability are rejected explicitly rather than being
silently discarded. Complete-model GGUF load-time quantization is encoded in
the selected per-parameter tasks, so selection and materialization cannot
disagree about whether a format or lowering is accepted. Execution classes
handled by a distinct protocol, such as realtime Moshi, do not enter ordinary
causal-text dispatch.

For GGUF artifacts, `ArtifactInspection::validated_gguf` is the authoritative
handoff from stage 1. Core validates format-generic tensor-count, required
metadata, and embedding floors using the submitted metadata prefix. The
selected architecture registry resolves the family spelling, applies
family-specific structural admission, declares composite requirements, and
validates each resolved companion's family identity, shared geometry, translated
tensor names, and exact checkpoint schema. The handoff contains the primary plus
the exact resolved companion checkpoint handles and an architecture plan with
typed companion geometry, schema, and physical-to-logical mappings. Backends
may wrap those portable handles and lower their admitted logical names into
native encoding descriptors, operator checks, or device compatibility checks,
but do not rescan directories, select companions, repeat either portable admission layer,
or parse facade-owned tokenizer and EOS metadata.

GGUF reader failures remain typed through shard and portable artifact error
wrappers. Backend inspection classifies unsupported encodings from the nested
GGUF error variant and its numeric type code, never from rendered diagnostic
text.

`ModelPreparationPlan` is the one-shot authority for stage 4. The architecture
source factory consumes it with the exact retained selection and publishes one
bundle containing the originating inspection plus primary, companion, target,
and extension views. Materializers, including partitioned materializers,
consume that inseparable bundle and selected route directly. They
must not reopen an artifact to rediscover configuration, checkpoint metadata,
or sibling filename policy after planning; payload stores may still map weight
members during materialization, but those reads do not replace the plan's
configuration, checkpoint metadata, companion selection, or route.

`ModelLoadingBackend` implements backend capability reporting, native
materialization, and outer adaptation. `BackendProvider::create_session`
consumes a `PreparedModel`, so an executable cannot be paired with state or
session mechanisms created by another backend. Replicated prepared models
already contain the neutral lifecycle paired with their concrete mechanisms.

`ModelRuntime<B>` owns a backend and its sole session. All facade clients use
`eredu::api::LoadedModel<B>`, whose runtime remains private while portable model
operations stay generic. Loading and generation failures retain `B::Error` in
their generic error types.

Automatic planning produces a portable `ExecutionPlan`. An
`ExecutionPlanBackendFactory` realizes the complete plan into a target backend,
backend load options, and the selected disabled, embedded, or external drafting
configuration. `LoadedModel::load_execution_plan` and `LoadedModel::plan_and_load`
therefore infer the backend from its factory without requiring callers to
construct backend devices, queues, streams, assistant models, or sessions. Concrete device families are
validated against the backend features and runtime hardware during realization.
The backend reports the canonical identity derived from that realized device
binding rather than echoing the plan's requested identifier.
Automatic feedback cannot introduce an assistant identity from telemetry:
an external plan requires explicit facade-owned assistant inspection and
tokenizer proof, neither of which is part of an automatic planning request.
Core binds cold drafting selection to the exact complete execution plan and
target selection, and rejects either selection if later realization supplies a
different plan. Each selected target carries an opaque core-owned identity;
`ExecutionPlanTarget::into_runtime` moves that exact backend/prepared-model pair
into the runtime, and external drafting realization rejects a runtime produced
from any other target even when both targets implement the same plan.

The loading APIs' `*_with_text_options` variants accept facade-owned `TextModelOptions`,
as do retained-inspection and direct backend loading. A caller-supplied single
or named chat template supersedes checkpoint template selection before native
realization, while tokenizer variables, EOS ids, and generation defaults remain
checkpoint-derived. These options stay outside the core execution plan and
backend load policy. Pre-load text inspection borrows the `TextModelOptions`
value intended for loading. Both operations use one facade-owned template
resolver, with invalid selected metadata reported instead of falling back to
another source. `TextInspectionOptions` holds only the optional behavioral probe
request. `LoadedModel::set_chat_template` replaces the template used
by future preparation and loaded-model template inspection, clearing compiled
templates through the neutral `eredu-text` cache utility. Existing prepared
chats retain their prompt and protocol metadata. All supplied templates pass
through the ordinary facade-owned rendering and protocol-recognition path.
Without an override, templates come only from checkpoint metadata or sidecar
files. No model family, including Gemma 4, receives an implicit template.
A model with no template still loads for raw token generation,
but chat preparation returns `TextModelError::MissingChatTemplate`; the facade
does not invent a generic prompt format. Clearing the template with the setter
also produces that error without reloading metadata or invoking a fallback.

External assistants cross that factory boundary as an architecture-owned
`ExternalAssistantPreparationPlan`. Architecture inspection fixes the
assistant family, normalized configuration, checkpoint format, and strict
SafeTensors or GGUF layout before a backend is selected. SafeTensors plans
retain the admitted header catalog, architecture schema, and exact resolution.
A GGUF assistant plan retains the admitted portable checkpoint, resolution,
and canonical tensor mapping.
Before target device or stream realization, architecture-owned assistant
selection resolves the exact schema variant and retains one physical task per
materialized output, including shard/member/output provenance, aliases,
encoding, geometry, executable format, lowering, recipe, and atomic
companions. The concrete backend later reopens payload only through the
catalog-bound source, revalidates both the catalog and resolution, and consumes
those retained tasks with the generic exact binder. It rejects tensors changed
after selection before constructing a native module and does not walk the
assistant module to rediscover names or quantization policy. Concrete backends
do not receive an unrestricted assistant store, reopen configuration or
metadata for dispatch, bind against raw native topology, or duplicate
assistant admission.

`eredu::api::local_device_plan` maps the facade's CPU or accelerator choice to
the selected local backend and rejects accelerator choices when that
build contains no native accelerator family. `LocalRuntimeConfiguration`
applies any process-global allocator or embedded accelerator-library
configuration before the factory realizes that plan. Platform applications
import the concrete backend factory and use the generic facade to load and
operate the session.

Architecture inspection reports both embedded-draft depth and maximum proposal
capacity from the normalized, admitted artifact composition in
`ModelResourceProfile`. Capacity comes from the architecture's prediction
contract; fused predictors can propose more tokens than their physical layer
count. The neutral automatic planner caps its preferred draft width at that
capacity and leaves automatic drafting disabled when capacity is unknown.
Explicit draft widths are validated as requested. Concrete backends must not infer
family semantics by searching raw configuration documents. In particular, a
DeepSeek-V4 base GGUF has target blocks only: its `nextn_predict_layers`
metadata describes weights omitted into a companion artifact, so base admission
normalizes embedded depth to zero and capability reporting does not expose it.
Automatic bounded-residency sizing is likewise a cold operation. It performs
the same exact neutral selection used by loading and computes executable bytes
and the largest unit window solely from the selected materialization tasks. It
uses `eredu-runtime::selected_text_bounded_requirement`, including checked
byte/count arithmetic and sparse group-local window accounting. Neutral
residency telemetry is projected in runtime; speculative statistics and common
host observations are projected in core. Accelerator and allocator observations
remain backend mechanisms. The sizing path
does not realize a device or stream, open payload data, construct an
architecture, or provoke a failed native load to discover the requirement.
Automatic planning retains that single admitted artifact inspection through
every candidate admission, feedback choice, expert-cache mutation, and exact
bounded-residency probe. Applications may supply deterministic, idempotent plan
overrides, which are applied before candidate validation and sizing. The CLI
translates every explicit planning override through this path in quick and plan
mode; reports and cached plans describe the validated result. Residency probes
use the overridden plan's budgets. Feedback and cached plans are reusable only
when they already satisfy the current overrides, so their measurements cannot
silently replace an explicit choice. The final plan is admitted and probed again,
then loading consumes the same retained inspection; automatic loading cannot
reinspect a changed path or construct a route different from the report.

## Submission and completion

`BackendSession` provides high-level prefill and decode submissions. Associated
types keep prompts, tokens, outputs, session state, and completions opaque.
Concrete ordinary sessions hold one already paired architecture-erased neutral
session. Prefill, decode, state controls, reports, observation, prompt-cache
operations, speculation, and distributed control all delegate through that
same contract; there is no complete-model family or pipeline variant to select.
They do not expose model/cache parts or architecture-specific constructors to
callers. Operations unavailable for a valid session topology return typed
errors rather than relying on unreachable or panicking accessors.

Cold replicated-text requirements retain the dtype of the architecture-declared
floating-state source. Architecture code resolves its checkpoint aliases and
container metadata once; backends receive that exact dtype through
`ReplicatedTextMechanismSupport::floating_state_dtype`. The backend reports the
native `StateStorageDtype`, including any packed-embedding conversion. Runtime
synthesis resolves each component's floating or fixed dtype policy and passes the
exact physical dtype into `supports_state_component` for device and paged placement
checks. Unknown floating representations are rejected before native allocation.

State capability reports retain the source-to-storage mapping. Selection rejects a
report for another source dtype and records physical dtypes in
`SelectedStateComponentRealization`, preserving them through partitioning. F16
activation storage does not narrow fixed F32 or integer components. Construction
checks its native floating dtype against the admitted representation, including
same-width distinctions such as F16 versus BF16. MLX shares
one source-to-storage mapping between cold support and construction accounting;
backend providers do not repeat architecture checkpoint-name or family dispatch.

`Completion::resources_releasable` exposes native resource safety separately from
`is_complete` and `wait`, which report execution/output success or failure. A failed
submission can eventually become releasable without losing its error. Pending work,
unavailable evidence, and runtime contention report false. Cancellation requests,
timeouts, and observation errors never establish this proof. The default query
recognizes successful exact completion only; native backends provide independent
scope or fence evidence for failed submissions. This query does not make failed
model state reusable, publish output, or transfer ownership of retained resources.

MLX implements the query from its native submission scopes and outstanding consumer
and observation tickets, including sampled text, realtime, speculative, and
communication work. A settled producer cannot hide a pending child. Polling remains
nonblocking and does not execute arbitrary application destructors. Text completion
uses the neutral query before resolving model authority after a token failure.

Every submission returns an exact completion object. A completion observes
only the submitted work; it must not drain unrelated backend work. Schedulers
retain outputs, branches, cache transactions, and other resources until that
completion succeeds. Cancellation prevents publication but does not release
in-flight resources early.

MLX ordinary sessions include lazy token-domain reductions in the neutral
session's exact output-and-state completion boundary. An invalid token therefore
restores the session checkpoint before commit and cannot escape as a later
completion-only failure. The public session admits no newer state-changing
submission while its prior completion remains unresolved; generation-tagged
completion ownership prevents an older resolved handle from releasing a newer
submission's gate.

Native `Tensor::take_axis` also retains a bounds reduction in that submission
and substitutes safe indices until completion validates it. This protects
architecture-owned table lookups, including token-selected routing, when an
earlier embedding has recorded an invalid token but bounded layer execution
must settle before the error is published. Valid signed negative indices keep
their ordinary indexing meaning; invalid indices cannot reach a native gather.

V4 declares its implemented input, unit and readout hooks on the ordinary,
tensor-parallel and pipeline architecture contracts. Routed executors combine
those declarations with their provider hook facts; a pipeline-only declaration
does not establish input or readout support for a routed target session.
Normalized routed inputs derive their producer from the declared input node's
invocation owner, independently of whether the residual equation is a scalar
sum or a learned stream mixture. Prediction replicas retain the same declared
input boundary alongside their sparse expert-unit observations.

`SessionAuthority` and its move-only `SubmissionLease` own this exclusion in
`eredu-core`. Every state mutation checks idle authority before native work.
Native completions retain the lease and their exact resources; ready resolution,
terminal failure, or safe teardown releases only that ticket. Pending or
unresolved observation errors cannot release newer work. Atomic ticket matching
makes repeated and stale releases harmless, and ticket exhaustion is a typed
error. MLX retains one shared executable owner and an independent ticket for
each native scope; one completed child cannot release another unresolved child.
Failure remains a session poison even after its resources become reclaimable.
The neutral runtime supplies fresh successful state-restoration evidence for
recoverable errors. Only that evidence combined with an independently settled,
healthy native scope permits retry; unknown host errors and unwinds do not.
When restoration is proven but native cleanup is still pending, MLX retains the
executable and its submission lease in nonblocking recovery without poisoning
the session. The lease prevents reset or new execution until every scope safely
retires. Runtime-lock contention leaves this recovery pending; a later native
failure or unobservable scope permanently poisons the session even after its
resources retire. Error callers must allow retirement to complete before reuse.

`SessionAdmission` compares complete admitted and realized capability
reports before publication, rather than accepting a subset or assuming universal
backend facilities. Sampled-token completion and model validation both finish
before a text submission releases its authority.

Successful synchronous MLX operations retire their native owners under the same
runtime guard that confirms completion, then finish staged host destruction at
an ordinary unlocked boundary. This prevents another runtime user from delaying
lease release between a successful wait and the next session mutation or bounded
checkpoint acquisition. Polling, errors, and teardown remain nonblocking: when
the runtime is busy, failure still reaches the retained owner, but resource
release waits for both terminal evidence and safe runtime access.

The backend is selected for the entire model/session. Per-operation backend
selection would make cache ownership ambiguous and introduce implicit data
movement. Explicit transfer and collective capabilities cover the cases where
movement is part of the session design.

Speculative publication is one atomic semantic callback for a committed token
batch. A cancellation already pending at that boundary is staged before the
batch becomes visible; no later fallible cancellation callback can expose a
partial success. A callback failure restores the backend checkpoint and leaves
the canonical sampler, constraint, sequence, cache, and publication state
unchanged. Target-only and speculative commits both cross publication, while a
speculative verification additionally crosses exact completion, observation,
and cache persistence before publication.

## Text and multimodal generation

`TextGenerationBackend` submits complete prefill-and-sample and
decode-and-sample operations. The backend owns logits, sampler state,
randomness, and token handles. Portable generation code owns token budgets,
stop and EOS precedence, cancellation, grammar state, and semantic events.

`eredu-core` owns the common `GenerationOutput<S>` terminal contract: committed
token ids, finish reason, `GenerationTiming`, and mode-specific statistics.
`PreparedChatGenerationOutput` is an alias for `GenerationOutput<()>`, used by
ordinary and observed generation; `SpeculativeGenerationOutput` is an alias for
`GenerationOutput<SpeculativeStats>`. They share one implementation of output
accessors. All modes expose TTFT through `output.timing().time_to_first_token()`.
The facade starts this host clock
at the generation call, including request preparation; speculative batch lanes
share the batch-call origin and include time waiting for earlier lanes. Ordinary
generation records the first constraint-committed, completed token before record
or semantic delivery. The neutral speculative lifecycle records the first target
commit before publication, and runtime adds preparation and queueing time.
Structural tokens, buffered Unicode, stop tokens, and EOS count even without
visible text. Draft proposals never count, and cancellation with no committed
tokens returns `None`. This lightweight terminal metric needs no capture plan,
token journal, device timing, or extra native synchronization.
`SpeculativeStats::submission_to_first_token()` explicitly retains the
lane-submission origin. Runtime adds the facade preparation interval when it
constructs `GenerationTiming`; the output contains no speculative timing policy.
The CLI consumes this committed-token TTFT directly.

`MultimodalPreparationBackend` accepts portable ordered text, token, image,
audio, and video inputs and produces the backend's ordinary opaque prompt.
Image resizing, signal processing, feature extraction, tensor construction,
and placement remain backend operations, while architecture processor plans
declare the family-specific transforms, sampling, framing, and packing
geometry those operations implement.

Architecture-owned ingress plans also bridge prepared media into generic model
inputs. They derive family-specific placeholder spans, pooling geometry,
padding logits, and subsampling masks from portable extents. Concrete backends
materialize those declared values and may perform generic padding or dtype
conversion, but must not independently reconstruct family mask or geometry
policy. Gemma 4 exposes one input-part admission plan shared by resident
prefill, pipeline prefill, and capability accounting. That plan admits the
exact modality/payload pair, validates decoder-width projected embeddings,
selects the placeholder token, and returns its vision or audio ingress geometry
plus workspace shape; its part/batch plans continue to own padding and mask
values. Qwen vision ingress validates the prepared payload against
its patch grid with checked geometry, selects the image or video placeholder
token, and returns the exact placeholder span and validated grid; an accelerator
adapter only reads the small metadata tensor and materializes the returned token
array. Qwen3-VL also exposes one input-part admission plan used by both prefill
and capability preflight: text token IDs and projected text embeddings occupy
text positions, native image/video tensors use the vision plan, and projected
non-text embeddings are rejected. Conditional Qwen3.5 uses its own shared
input-part plan across resident prefill, pipeline prefill, and capability
preflight: projected text, image, and video embeddings must be batch-one at
decoder width, while native image/video tensors use the vision ingress plan.
Inkling execution and capability preflight consume one architecture input-part
plan covering text, decoder-width projected image/audio embeddings, exact hMLP
and dMel geometry, placeholder tokens, valid audio prefixes, and decoder spans.
Muse-Glimmer likewise uses one architecture input-part plan for text and native
image/video tensors; the plan owns the placeholder span, validated grid, and
checkpoint-convention video policy while rejecting projected embeddings and
audio. Backends do not reconstruct modality, spatial-merging, or artifact
policy.

Layered model execution topology follows the same ownership rule. Loaders and
materializers derive execution graphs and per-group unit counts through
`LayeredArchitecture` into one canonical `ExecutionUnitLayout`; quantization,
residency, parallel planning, parameter accounting, state allocation, and unit
construction consume that layout together with the architecture's required
state-independent parameter contract. They do not reconstruct a family's group
order, dependencies, layer counts, or flat-to-group mapping from configuration.
This keeps every backend realization aligned with architecture execution.
Backend composition traverses the layout's flat ordinals and resolves each
unit through its canonical group-local address; tensor-parallel accounting uses
the layout carried by `ArchitectureParameterDescription`, and load-time quantization
requires its source and target layouts to match before visiting any units.
Family output projections also own output-vocabulary policy, including removal
of checkpoint padding after serial or tensor-parallel projection. Backend
composition returns those architecture-produced logits without applying a
second family-specific slice.

The facade owns tokenizer and chat-template discovery. A backend may request
tokenization of checkpoint-defined framing text through typed callbacks, but it
does not implement a parallel tokenizer or chat lifecycle.

## Speculative generation

Core owns the speculative transaction:

1. prepare target and assistant state;
2. collect a bounded proposal block;
3. submit target verification;
4. wait for its exact completion;
5. resolve acceptance, replacement, and bonus tokens;
6. commit or roll back backend cache state; and
7. publish only committed tokens and semantic events.

`SpeculativeExecutor` is the portable protocol implemented by architecture-owned
embedded and external strategies. `SpeculativeSampling` exposes only generic
logits, probability-arithmetic, sampling, and random-state mechanisms. A
backend supplies native tensors and operators, state storage, cache checkpoint
mechanics, queue and transfer operations, and concrete completion values; it
does not own proposal, verification, replay, or commit semantics. Embedded
prediction heads and external assistant models use the same portable lifecycle.

For embedded prediction, `eredu-architectures` owns the cache envelope that
pairs target state, typed prediction state, the selected realization, prepared
input identity, capture frontier, and fork/commit/restore membership. It also
owns the materialized prediction strategy and the object-safe executor erasure;
the MLX adapter performs one final continuation over that already paired
executor. Backend composition cannot re-pair an erased target with a different
prediction cache or substitute a family-specific speculative driver.

For external assistants, `eredu-architectures` likewise owns the semantic cache
envelope, capture schema validation, observer ordering, checkpoint restoration,
proposal/verification sequencing, and replay. The family-neutral MLX mechanism
adapter retains only opaque native cache storage, tensor operations, transfers,
streams, and completion values. Its activation observers are installed through
the architecture executor, so interventions and failures participate in the
same transaction rather than a test-only side channel. MLX sessions retain the
installed embedded observer set across erased executor lends; the backend does
not recreate a no-op observer set at each scheduler turn.

`SpeculativeLifecycleObserver` is an explicit, production-carried coarse
instrumentation seam. Neutral selection observes admission and compatibility
before native construction; the shared request table observes input,
execution, completion, resolution, cache persistence, publication, and
cancellation before each corresponding boundary. Observer failure prevents
that boundary from starting, and failures after verification submission restore
the retained cache checkpoint before publication. This lifecycle seam does not
replace typed `ActivationObserver` intervention over backend-native tensors;
the two report complementary control-flow and value-flow evidence.

The prepared-generation capability does not expose whole-request execution
methods. A backend implements `with_speculative_execution` only to consume the
retained neutral realization, prepare typed lane resources, and lend them
through `SpeculativeGenerationVisitor`. Compatibility and placement are fixed
before assistant or extension construction. The facade supplies
`RunSpeculativeGeneration`; `eredu-runtime::SpeculativeScheduler` registers
lanes, selects fair actions, drives exact completions, validates terminal
reasons, and constructs public outputs. Single requests use that same path as
one-lane batches. Concrete backends must not substitute their own lifecycle or
fair-batch loop.

Architecture families also own external-assistant compatibility proofs. Those
proofs match target-state publishers, hidden and rotary geometry, target-layer
captures, and vocabulary requirements before neutral composition constructs
the paired executables. They and the neutral speculative realization are
selected from the retained target inspection during cold drafting selection,
before target payload or native-resource realization. The facade supplies the
distinct tokenizer proof.
Backend composition consumes both proven contracts and does not restate
family-specific compatibility rules.

Gemma 4 and Muse-Glimmer own their external-assistant execution strategies.
Gemma shared-attention capture truncation, target-to-draft state assembly,
proposal progression, and partial-verification replay live in
`eredu-architectures::gemma4::speculative`; ordered DFlash capture assembly,
sliding committed context, anchor-plus-mask proposal construction, and replay
live in `eredu-architectures::muse_glimmer::speculative`. The MLX adapter binds
both strategies through one family-neutral mechanism implementation for arrays,
streams, exact completions, target-cache checkpointing, and already materialized
neutral assistants. `ExternalAssistantArchitecture` selects and constructs the
typed executor before lending it to the shared runtime visitor; MLX neither
matches assistant families nor provides a second family-specific
`SpeculativeExecutor` lifecycle. Compatibility-proven preparation is the only
public path to assistant materialization, so an incompatible pairing cannot
open assistant weight payloads.

Before materialization, the compatible external preparation derives one
neutral selection contract. Architecture-owned target profiles supply the
stable target/geometry identity and maximum capture sequence; the admitted
assistant checkpoint supplies artifact and physical-format identities. The
facade supplies only its tokenizer proof, requested proposal capacity,
processor identity, and already selected rank topology. Concrete backends must
not synthesize family, artifact-format, or capture identities.

The selected capture schema fixes paths, order, ownership, observation seams,
fixed dimensions, and upper bounds for request-sized sequence dimensions.
Immediately before target execution, each lane binds its exact prepared-input
identity. Each committed target pass then closes the actual positive sequence
extents and target-cache generation, and the architecture executor validates
that capture against the retained realization before proposal state can
advance. A stale generation, reordered path, wrong fixed dimension, oversized
request dimension, or cache/input mismatch fails the transaction. External
assistant targets are replicated-only; partitioned target requests fail during
architecture contract selection before assistant payload or queue creation.

Prediction-enabled composition is additive to the ordinary target session.
Architecture admission projects the target and a typed prediction-extension
plan from one already-admitted artifact; it does not reopen or reclassify the
checkpoint. The ordinary neutral session remains the sole owner of target
prefill, decode, mutable state, prompt-cache control, completion, observation,
rollback, and publication. A target pass may retain an architecture-declared
hidden capture from that same transaction for an embedded predictor or
external assistant. Missing capture fails the transaction before publication.
Architecture composition owns prediction-local state geometry, identity, and
transaction semantics; a backend owns only its native storage and prediction
operator mechanisms. It must not construct a second target model or target
cache. Any extension whose unit execution lacks a typed additive handoff
remains fail-closed; loading a second complete model beside the neutral target
is not an admissible compatibility mechanism. DeepSeek-V4 DSpark uses its
configured, ordered intermediate-layer captures and exact fused block capacity
through that typed handoff. Ordinary final hidden state is never substituted
for the declared capture.

Controlled speculative forks reuse the core's exact canonical snapshot contract.
Runtime owns scope-local branch handles, logical run identity, bounded serial
exchange and non-rewindable accounting. Immutable source snapshots may be shared;
activation copies through the same backend snapshot/restore mechanism. Anonymous
retained branch copies share `SnapshotBudget` without allocating extra user-visible
snapshot or branch counts. No backend schedules child runs or replays the prompt.

Prospective token restrictions use runtime's ordinary `TokenChoiceController`,
including grammar checks and absolute speculative positions. Ordinary and speculative
samplers share temperature/RNG validation. Core snapshots retain configuration,
choice, both RNG streams and semantic state; the native adapter supplies only seed
construction and existing sampling primitives. Tensor-edit discovery is a distinct
speculative capability, re-admitted against the loaded source/session. MLX initially
exposes prediction-row logits for each model role and uses the shared `CaptureSession`
and intervention engine for transforms, evidence and cumulative budgets. Selecting
a role's immutable plan between drained rows never resets that ledger. Additional
layer/routing hooks require architecture-owned speculative attribution before they
can appear as supported in discovery.

## Scheduling and cancellation

The core state machines own queued, prepared, submitted, committed, failed,
cancelled, and abandoned states. Facade/runtime schedulers own action selection
and progress; backend adapters supply typed resources, submission, and exact
completion observation. A submitted request remains transactional until its
completion is observed, even when its client has cancelled or disconnected.

Realtime and distributed request scheduling use the same ownership rules.
Protocol framing and cross-rank schedule agreement are portable; transport and
native collective execution belong to the backend. Realtime backends expose
step submission and completion, while facade/runtime schedulers own offline
greedy loops, request registration, frame progression, and output collection.
Distributed cancellation uses a caller-selected bounded consensus transport.
Every turn first exchanges the exact padded active-request set so a deadline
observed by any rank becomes a common candidate. Explicit cancellation and
deadline expiry then run the same preparation and commit-authorization votes;
the scheduler does not remove the request, publish a terminal disposition, or
discard a provisional branch before both votes succeed. A transport deadline,
malformed status, or negative vote fences the scheduler and makes retry fail
before another transport call. After commit authorization, every submitted
branch is marked abandoned and retained until exact backend completion.
Backend adapters implement only a deferred equal-word all-gather returning a
bounded completion that owns its group, buffers, stream, and native work; they
must not synchronize or resolve host words while constructing the submission.
Only after exact completion may the adapter resolve its opaque gathered output
into rank-major words. The scheduler caller, not the backend, supplies the
selected wait policy. MLX implements this contract with a lazy native gather
whose completion retains the input, output, world handle, and stream. Timeout
uses the same communicator-fencing quarantine as partition communication; no
host word access occurs until the caller observes exact completion.
The retained public distributed-session view and the partition runtime share
one manifest-derived deadline and poison authority. Public collective
completions retain their inputs, outputs, count buffers, exact group, routes,
and streams; on timeout an explicit event-backed quarantine keeps those native
resources alive until completion while every later submission in that session
fails before reaching the transport.

## Cache and residency

Core separates logical policy from physical storage:

- `OffloadPlan` and the residency state machines validate budgets, placement,
  prefetch, eviction, leases, and transfer generations.
- Cache lifecycle and storage state machines validate block identity, geometry,
  mutable tails, tier transitions, and exact I/O ownership.
- Process-wide cache admission tracks device, host, transfer, and disk
  reservations atomically.
- Prompt-cache schemas validate identity, topology, fingerprints, coverage,
  and safe payload paths. Architecture fingerprints use a backend-neutral
  `eredu` domain; obsolete backend-branded domains are intentionally
  incompatible.
- Architecture-owned shared-decoder configurations must provide an
  authoritative fingerprint covering all construction, equation, state, and
  encoding policy; the shared decoder does not synthesize compatibility
  identities from a partial policy subset.
- Shared-decoder construction and execution enter through `LayeredModel`,
  which validates and retains the authoritative architecture configuration.
  Unit construction, cache policy, and forward execution reuse those retained
  arguments rather than accepting an independent configuration.

A backend supplies concrete tensors, host buffers, buffered payloads, files,
workers, native transfer objects, and allocation observations. It applies core
transition results and releases the physical resources selected by those
results.

## Distributed execution

Portable topology uses data, tensor, pipeline, and expert axes. Core validates
rank coordinates, subgroup membership, balanced ownership, and operation
scopes. Runtime owns the backend-neutral tensor-placement decisions consumed by
architecture planning and concrete backend realization.
Data parallelism is represented but unsupported for partitioned model
execution. A request with `data > 1` reaches neutral architecture
selection and fails there before payload or native communication realization;
concrete topology adapters do not erase or reject that axis independently.

`ArchitecturePartition` is publicly constructed only through
`from_architecture`, which derives its execution graph and unit layout from the
concrete neutral architecture. Backend placement code supplies rank-local
ranges and ownership, but cannot substitute a caller-built topology that only
resembles the architecture it will execute.
Balanced unit ranges are only a placement default. When an architecture
declares dependency-safe pipeline ranges—for example, to keep a shared-state
publisher and consumer together—composition installs those exact contiguous
ranges on the canonical group. A group's terminal owner also routes its result
to a distinct architecture-declared merge destination before dependent groups
or final output ownership are evaluated.

`LayeredPartitionDriver` executes that canonical partition contract. It
validates backend unit storage and mutable-state ranges, enforces ownership,
and sequences the selected group for both replicated and tensor-parallel
paths. `PartitionedLayeredArchitecture` owns typed entry, group entry and
completion, and typed exit: it converts tokens or an architecture-prepared or
upstream activation plus auxiliary state into a forward context, projects only
on the output owner, and returns either a typed transport boundary or a final
output with an optional embedded-predictor capture. Concrete backends supply
only unit residency, state storage, collectives, and encoding or decoding
through the partition's `ArchitectureBoundary`; backend composition does not
reconstruct a family's begin/finish lifecycle. Families do not expose alternate
public execution seams that accept caller-supplied embedding, reduction, or
projection operations, or piecemeal setters for partition forward state.
The primary decoder and embedded-prediction groups have explicit stable
identities, so multimodal ingress ends at the declared primary boundary and
prediction runs only in the architecture-declared identity order. Their
semantic kinds remain useful for lifecycle classification but do not select
either group. Composition dispatch guards consume those declared identities;
they do not reconstruct family prediction-group names from depth indices.

Pipeline ingress uses the same canonical execution graph through
`LayeredPipelineSchedule`. The neutral runtime consumes both architecture-authored
group kinds and request optionality, queries request work only for declared
optional encoder roots, propagates activity through merge dependencies, admits
compatible ready batches, and owns group completion transitions. Concrete
backends retain that resolved optionality with physical placement and realize
streams, residency, payload transport, and collectives; they do not infer
optionality from backend-local kinds or maintain a second graph lifecycle.
Compatible-batch selection is exposed only through this layered lifecycle;
the underlying ready-set bookkeeping remains an internal implementation detail.
Backend adapters address media storage and the primary boundary by the graph's
stable architecture-owned IDs, including when several groups have the same
semantic kind.

The partition also carries architecture-owned route-indexed boundary schemas.
Every selected opaque route retains its semantic source and destination group
and rank, plus the exact schema used by its manifest descriptor; composition
must not recover those endpoints from physical adjacency or apply one route's
schema to another. Each schema
declares the primary evolving activation and every auxiliary tensor, including
their stable roles, canonical order, symbolic shapes, logical dtypes, and
configuration-dependent cardinality. It also owns conversion of auxiliaries to
and from the family's typed boundary value. The architecture derives the whole
schema from its normalized configuration, including a primary transport width
that differs from the ordinary hidden width. `eredu-runtime` validates and
resolves batch and sequence dimensions. A concrete backend only maps the
logical activation or exact integer dtype to its native dtype, allocates the
declared receive buffers, validates produced tensors, and transports them. It
must not extract or override boundary geometry from model-family arguments.

The physical dtype of tensors declared with the logical `Activation` dtype is
selected by the separate backend-neutral `PipelineWireContract`. Distributed
load configuration must select that contract explicitly and provide the same
value to every stage. Backends normalize outgoing floating activations to the
contract during local group execution. After mechanical validation, the source
submits every typed boundary tensor as rank-local dependencies through the same
communication completion mechanism and caller-selected bound used by the
route. Only exact completion advances `BoundarySourceCompletion(route)` and
then `BoundarySourceReady(route)`; a deadline or cancellation retains the lazy
outputs and executor through safe teardown, prevents point-to-point submission,
and fences retry. Unsupported source conversions fail before that completion
phase; checkpoint parameter dtypes,
rank-local loading order, quantization encodings, and stage ownership never
select the wire dtype.
Distributed placement dependency routes consume that same schema directly. A
concrete backend must reject inactive dependency routes with tensors and active
routes whose exact cardinality, ordered shapes, or physical dtypes differ from
the resolved architecture boundary; backend-local string or optional-field
schemas are not an alternate wire contract.
Each logical point-to-point boundary tensor is one role-exact in-band frame.
The serialized route descriptor carries the architecture schema identity and
ordered role, logical dtype, symbolic invocation dimensions, fixed dimensions,
and bounds. The transmitted native U8 message concatenates a canonical header
containing route, schema, role ordinal and name, actual dtype and shape, and
payload byte length with the tensor bytes. A receiver validates header bytes
sliced from that received message after exact native completion and only then
reconstructs the typed tensor. A backend-synthesized return tag or a separate
independently reorderable control message is not role/schema metadata
provenance. `RoleExactV1` does not add a payload digest: same-length payload-bit
integrity remains the native transport's responsibility.
MLX copies a received byte payload into aligned output storage before viewing
it as a wider dtype when the header length is not divisible by that dtype's
width. This uses the admitted logical output extent, retains the frame and
output through the same completion, and adds no host transfer or wire padding.
An unaligned byte view is insufficient even when a host byte copy can read it:
typed native kernels must consume the exact received values.
The byte materialization uses U8 addition of zero because MLX's `Copy` primitive
shares the original allocation. The received frame remains retained, preventing
donation of its unaligned payload storage to the typed output.
`RoleExactV1` is an explicit communication capability, so admission fails
before native resources or payload work if it is unavailable. Batch and
sequence may vary within admitted maxima; architecture-fixed axes remain exact.

The neutral composite executor binds dense and routed prepared ingress,
request-optional roots, group-local traversal and merge, typed decoder
continuation, and the selected routed-expert provider. Cross-stage routed
execution retains the architecture-selected expert wave schedule, including
zero-work pipeline participants. Before any payload materialization or
mechanism binding, architecture preparation seals group transport kinds,
boundary-route schemas, publication
geometry, collective placement, resident routed-provider authority, and the
complete TP/PP/EP wave order into one opaque composite executor plan. The
backend supplies only its already-selected unit policy, parallel handle, tensor
allocator, and route-movement mechanism; it does not inspect or reconstruct
those semantics. Its typed prepared architecture, exact
partition, selected manifest, and materialization tasks are consumed by the same
`prepare_partitioned_session_runtime` handoff and ordinary
`ReplicatedTextSession` lifecycle as other neutral partitioned text execution;
the backend factory cannot replace the composite graph after admission.
A media encoder split whose continuation bundle depends on request-variable
part cardinality remains unsupported unless the architecture admits an exact
request-bounded continuation schema; no fixed role bundle is invented from
maximum batch or sequence limits.

MLX production consumes that same family-blind composite handoff for admitted
prediction-free Gemma 4, Muse-Glimmer, Inkling, Qwen-VL, and conditional Qwen
partitions. Indexed SafeTensors and architecture-admitted GGUF sources share the
selected checkpoint-source representation, and selected resident or bounded
weight policies retain the same rank-local materialization tasks. The backend
visitor binds only generic MLX tensors, operators, state, materialization,
expert exchange, and opaque communication; architecture-selected static tasks
remain the exact local payload-read set. Prediction-bearing graphs and artifact
or media-continuation forms without an admitted architecture contract are
rejected before native communication or payload work rather than reconstructed
by a backend family branch.

Pipeline admission also selects one opaque session group containing every
world rank in rank order. The group is the first projected communication
resource and carries exactly a bounded floating-tensor broadcast for final
output publication plus payload-free failure agreement. Architecture admission
retains that exact group ID, the publication owner, and any independent tensor
group ID and passes them to the partitioned execution plan; a backend realizes
the descriptors without interpreting tensor or pipeline coordinates.

For prediction-free dense Llama/Mistral artifacts backed by indexed or
unindexed SafeTensors, or by an admitted GGUF checkpoint, pure TP, pure PP, and
TP+PP use an architecture-owned partition dispatch and executor under selected
fully resident, host-layerwise, or dense disk-stream residency. The backend
binds one selected layerwise policy shared by execution and reporting; it does
not construct a second cache or residency-specific family executor.
Selected Llama/Mistral load-time transforms use this same route. Architecture
construction carries both the exact target partition and its source-format
partition with rank-local physical layouts and identical local unit/owner
addresses. The generic materializer applies each selected task recipe to the
source-local tensor before conversion and does not shard the resulting local
weight or companion a second time; untouched bindings retain their ordinary
single sharding step. No Llama-specific pipeline loader participates.
Dense Qwen2/Qwen3 select the same route for indexed or unindexed SafeTensors
and admitted GGUF, including architecture-selected transforms, under fully
resident, host-layerwise, and dense disk-stream residency. Prediction-free
dense LFM2, Kimi Linear, and
Nemotron-H select the same route for admitted SafeTensors or GGUF catalogs
under resident, layerwise-host, and dense-disk-stream residency, including
architecture-selected load-time transforms. Classification follows the
normalized unit schedule: unused expert metadata does not turn an otherwise
fully dense Kimi schedule into routed execution. One exhaustive architecture
dispatch selects the concrete normalized configuration, block or heterogeneous
operator policy, TP-local mutable-state geometry, and PP-local unit interval
before a single family-blind backend visitor runs. The executor
owns the exact partial static modules, local unit interval,
rank-local state geometry, tensor-parallel context, and typed boundary state;
the shared `PartitionedTextRuntime` and ordinary `ReplicatedTextSession` retain
rollback, observation, completion, and commit ownership. The architecture's
publication owner observes and may intervene on final logits before the
authoritative value is broadcast. Non-owner ranks receive that value for the
neutral lifecycle but do not expose it as public local logits, and sampling
uses the same selected owner expressed as a local rank in the already-realized
session group. Kimi partitions retain the exact three convolution histories
and Float32 recurrent tensor for KDA units plus compressed-latent/rotary MLA
state, sliced by their architecture-global PP unit offset. Nemotron-H retains
its TP-local Mamba convolution and Float32 recurrent state together with its
sliding/full KV policies, and transports architecture-typed `tokens` and
`embedded` auxiliaries in addition to hidden state. Prediction-free Qwen3-MoE
and GPT-OSS, DeepSeek V3/V4, routed LFM2 and Kimi Linear, and Nemotron-H use the
typed routed partition constructor for admitted SafeTensors and GGUF artifacts
under pure TP, pure PP, pure EP, and their admitted TP/PP/EP combinations.
Independent expert parameters reside in the addressable bank selected from the
architecture catalog; bounded ordinary non-expert parameters retain the same
generic layerwise policy where admitted. MLX owns only bank storage, movement,
materialization, and causal acquisition/eviction/reload telemetry.
Each addressable member retains its exact neutral materialization task together
with the member-local recipe, selected output, source and selected byte totals,
and any scale or affine-bias output companions. Rank-local projection preserves
that closure rather than reducing a bank to aggregate bytes or one bank-wide
quantization flag. The MLX mechanism lowers these tasks into per-binding
transformation descriptors, so direct and transformed bindings may coexist in
one bank. It validates source provenance, executable format, lowering,
companion ownership and dtype, recipe output, and the neutral selected byte
total before any checkpoint read or residency-manager construction.
Compound
placement first selects the exact packed expert-owner range and then applies
the independent tensor-parallel matrix placement. For a pipeline wave, the
architecture declares one global order containing ingress, per-unit attention,
expert exchange, routed-output, and final vocabulary collectives. Inactive PP
cohorts submit exact-shaped zero tensor work at those same positions; active
cohorts bounded-complete their local lazy dependencies before advancing. This
keeps overlapping TP and EP logical groups in one native-world order without
turning backend topology into model policy. Overlapping logical expert
subgroups participate in one consensus-proven native-world
variable-all-to-all wave with zero-padded nonmembers; admission rejects an
unproven or omitted wave before publication or commit. Qwen hybrid and
vision-language graphs, prediction-bearing configurations, and unsupported
family selections use their separately selected composite, extension, or
fail-closed paths; dense execution is not used as an implicit fallback.
Architecture selection classifies this production route exactly once, before
opening payloads, from the normalized artifact and consumed partition
admission. The resulting immutable neutral-partitioned or explicitly excluded
extension selection is carried into materialization; backend composition does
not re-test family, format, residency, or lowering eligibility and cannot
substitute a different route after communication selection.

Each opaque pipeline route is realized as its ordered two-member logical group
`[source, destination]`. Its backend handle retains Source/Destination role and
group-local peer rank; point-to-point execution never treats global descriptor
endpoints as native-group indices. This permits non-neighbor semantic routes on
a native ring without changing the neutral route graph.

Partitioned execution separately defines an opt-in payload-free
`FailureAgreement` operation. It is an all-member boolean conjunction, not a
barrier. Before agreement, each route endpoint prepares its exact resolved
schema and ordered values: the source retains its already wire-normalized
activation bundle, while the destination allocates its typed placeholders.
Both endpoints validate bundle cardinality, shapes, dtypes, route operation,
and payload limits without submitting native point-to-point work. Every rank
then agrees first that the source's lazy tensor dependencies completed under
the selected bound, and next that source execution and both endpoint
preparations succeeded. Only an agreed route consumes the retained prepared
bundle in send/receive; preparation is not recomputed after either vote. The
destination therefore does not enter native receive after a source conversion,
source-completion deadline, or either endpoint's preparation failure. The
shared session also agrees that every rank captured its transactional state
checkpoint before any rank enters forward execution or a model collective. A
checkpoint-capture failure therefore cannot leave a peer executing, but the
failing rank has no checkpoint to restore; checkpoint mechanisms must leave the
borrowed live state unchanged when capture returns an error. The same status
operation also follows rank-local graph execution, owner-only final
output observation, authoritative output publication, and exact mechanism
completion. Every rank submits whether that
phase succeeded and advances only when all statuses are successful. The shared
session performs this agreement even on a local error, rolls back before
returning on either a local or remote pre-decision failure, and attempts final
commit only after the completion phase agreed. Final commit uses a positive,
monotonic `DistributedCommitEpoch`. `Committed(epoch)` means this rank observed
the globally fixed all-success decision; it deliberately does not claim that
every peer observed that decision. A failure while submitting or completing
the final decision after this rank may have contributed is
`Indeterminate(epoch, phase)`: the session withholds its public result, does not
roll state back over a decision a peer may have committed, records the outcome
in its report and prompt-cache manifest, and rejects retries before execution.
An explicitly observed global false decision is `Aborted(epoch)` and retains
the ordinary rollback behavior. A backend selects the corresponding
`OpaqueFailureAgreement` policy only when the session-group manifest contains
the exact operation and the backend implements the narrow status-agreement
trait. Barrier-only policies retain local status and are rejected if paired
with a manifest that selects failure agreement.

Manual prompt-cache control on a partitioned session uses the same selected,
bounded failure-agreement group. Load first agrees exact descriptor, topology,
rank-local identity, and prepared-input preflight; each stateful rank then
loads and validates a detached provisional state while stateless pipeline ranks
still vote. Live state and commit metadata are replaced only after the second
all-rank success. Save requires the backend mechanism to expose a reversible
transaction: serialization remains unpublished through preparation, publication
retains any replaced destination, and a final all-rank vote either commits every
rank's candidate or restores every prior destination. Failure after any selected
control phase permanently fences that session before retry, execution, or new
cache I/O. Replicated sessions retain their ordinary local save/load behavior;
partitioned strategies without exact failure agreement cannot claim this atomic
control API.

MLX binds that contract directly to the neutral replicated-text session. Each
stateful rank derives one storage directory from the descriptor's stage, tensor,
and addressable coordinates; family configuration and native group structure do
not participate in the path. Preparation serializes the already validated MLX
state into a hidden reversible destination. A fresh publication becomes visible
by directory rename. Replacement instead installs an immutable generation and
atomically switches `CURRENT`, retaining the prior generation until the final
all-rank vote. Rollback restores that exact pointer and removes the candidate,
while commit discards only the transaction's retained staging metadata. The
public MLX session selects these distributed methods from an explicit control
capability owned by the neutral executable. A retained native communicator is
only a mechanism resource and does not own this protocol. Cleanup or restore
failure is fail-stop rather than allowing execution with a partially published
cache.

Manual state control follows the same rule. A partitioned checkpoint is returned
only after every stateful rank captured its local opaque checkpoint; stateless
ranks still vote. Reset and rollback construct or restore a detached provisional
state, validate its selected local geometry, and swap live state plus commit/input
metadata only after all-rank success. A failed control phase discards the
provisional value and fences every later control or execution attempt. The
ordinary replicated checkpoint, reset, and rollback methods remain local.

This neutral mechanism does not turn a failed transport into a reliable
failure detector. A point-to-point or collective implementation must still
complete with an error or implement the bounded completion policy selected by
the caller before architecture projection. That positive deadline and timeout
disposition are serialized in every non-empty manifest, compared during
cross-rank consensus, and capability-checked before native group, route, or
payload realization. Partition communication poisons the session after a
deadline, terminal completion error, or synchronous native submission error;
later operations fail before another backend call. Routed cross-stage expert
waves have one narrow recovery exception: after a rank reports local failure
or its subgroup operation poisons the authority, the runtime may submit the
canonical full-session `FailureAgreement` once under the same bound so peers
can roll back together. A healthy rank cannot invoke that bypass for ordinary
work; a false result fences every rank, and the bypass never clears or replaces
the original poison. Ordinary pipeline waves first agree execution before preparing
boundary payloads. A completed negative execution vote is retained as evidence
that all participants reached that boundary; this logical rejection rolls back
without inventing a failed transport. It does not enter the recovery bypass or
clear poison. Boundary source-completion and actual communication failures retain
their existing terminal fencing. The reusable MLX
communication backend realizes this payload-free contract with an internal
scalar integer all-sum. Its completion polls the exact event without taking a
blocking runtime lock, retains arrays, count storage, groups, routes, and
streams, and defers the host boolean read until that event completes. Because
upstream MLX exposes no event abort, timed-out work is retained in a
thread-affine quarantine and its native communicator cannot be realized again
until a later safe reap observes terminal native evidence. Failed submissions
also retain these resources when no public event was returned. If the
originating thread exits first, teardown does not block: unresolved arrays,
count storage, groups, routes, streams, and native execution owners remain
retained permanently. This bounded-teardown policy can retain memory after an
unrecoverable device failure; it never converts a missing completion proof
into process termination or unsafe destruction. Complete all-rank manifest consensus also precedes
every rank-local MLX capability, quarantine, and world-identity check, so a
corrupt projection produces the same shared setup failure before subgroup or
payload realization. Because the serialized completion policy is itself part
of that untrusted projection, the unsplit control-plane exchange uses a fixed
bounded backend setup policy; only after consensus does MLX validate and
install the agreed manifest-selected policy. MLX capability preflight
advertises this quarantine disposition separately from the operation.
Architecture selection requires it for the
resident PP and TP+PP production slice, and MLX instantiates
`OpaqueFailureAgreement` before payload construction. Pure TP reuses
its exact architecture-selected tensor group for owner publication and phase
agreement; it does not invent a second session identity. Resident sampling retains a clone of the
partition communication authority: token and stop-status broadcasts share its
selected deadline and poison state, and stop-status host resolution occurs only
after exact event completion. Unsupported distributed graphs fail during
architecture admission before manifest realization, payload access, or native
group construction; there is no backend pipeline or expert bridge fallback.

Neutral communication preparation first proves complete compatible manifests,
operation capabilities, logical reachability, and complete-world group and
route waves without creating native resources. Opaque MLX setup then acquires
the process runtime lock through the prepared manifest's absolute submission
deadline and rechecks it immediately before the first
array, frame, or collective graph mutation. Lock contention therefore fails
before native submission and poisons the one partition-communication authority.
Manifest-owned subgroup realization never enters upstream MLX's synchronous,
non-abortable native split: it retains an exact logical membership view of the
world handle. A logical view may use a world-wide operation only when manifest
consensus proves that the creation batch partitions every world rank into
same-requirement groups; non-neighbor route relays additionally require a
disjoint, contract-identical route batch covering the world. Partial,
overlapping, optional, unreachable, or contract-mismatched batches fail in
neutral preparation before any native callback. A variable all-to-all in such a proven wave expands each logical
peer-count vector into world-rank order with exact zero counts for nonmembers,
submits one native world operation on every rank, and restores the opaque
member order after exact event completion. All ranks must invoke creation-order
waves in the same order; the manifest proof does not make divergent runtime
schedules safe. The backend retains every logical membership, local index,
route, array, stream, and event through exact completion. Uncontracted control
groups use native-split behavior.

MLX exposes no way to interrupt a C++ graph-construction call
after that call has entered native code. The selected deadline can bound lock
admission and all event-backed work, but cannot preempt such an individual
native call; the backend therefore performs each native submission once and
never retries it. The durable commit epoch and explicit indeterminate outcome
make the remaining asymmetric-observation window honest: one rank may report
`Committed(epoch)` while a peer reports `Indeterminate(epoch, phase)`, but
neither may falsely report `Aborted` and the uncertain session is poisoned
until external recovery resolves that epoch. These limits are distinct from safe
timeout disposition: orphaned MLX work retains every native dependency and
fences the same native world. Quarantine registers an idempotent thread-local
runtime housekeeping callback, so any later same-thread MLX entry observes and
releases exactly completed work even when no communication operation performs
an explicit reap. Owner-thread teardown makes only a nonblocking retirement
attempt. If terminal evidence or runtime access is unavailable, its retained
resources remain owned permanently; thread-affine work is not transferred to a
reaper thread. An asynchronous error alone never authorizes reclamation.

`DistributedSession` is an optional capability of the selected model session.
It exposes high-level sum, gather, variable-count exchange, point-to-point, and
consensus submissions with exact completion. Unsupported operations report an
explicit absent capability. Communicator construction, sharding, movement, and
collective tensor math remain backend-specific.
The retained MLX public view and neutral partition runtime are two consumers of
the same manifest-derived communication authority, including its deadline and
poison state; neither constructs a detached authority. Public completions keep
their exact inputs, outputs, count buffers, opaque groups, routes, streams, and
shared authority context alive until resolution, even after the public session
handle is dropped. Singleton manifest groups remain contracted operation
handles rather than disappearing from an advertised capability.

## Concrete implementations

Concrete adapters document their public implementation surfaces, native
features, platform requirements, and realization details with the crate that
owns them. See the
[MLX backend architecture](../eredu-backend-mlx/doc/architecture.md) for one
implementation of these contracts.

Backend-neutral `TextGenerationConfig` also selects standard or Mirostat V2
sampling. The chosen backend owns the corresponding logits, random state, and
adaptive sampler state, so applications do not fall back to a native tensor
loop for alternate sampling policies.

The canonical facade exposes this choice as
`PreparedChatGenerationSettings::strategy` (also re-exporting `TextSamplingStrategy`
from `eredu::api`). Standard sampling remains the default. Mirostat validates finite,
positive tau, eta, and effective temperature after checkpoint defaults and request
overrides are resolved; `do_sample = false` therefore cannot select Mirostat. It
retains the resolved seed and penalties, replacing top-k, top-p, and min-p. The
facade's vocabulary and tool masks apply before sampling, so adaptive updates use
the resulting constrained distribution. Ordinary, observed, intervened,
controlled, and prepared speculative generation share this resolution. Speculative
batches may mix strategies and retain independent sampler state per lane. The MLX
adapter shares sampler construction with ordinary generation and delegates logits
processing and committed-token updates to the neutral runtime policies. Mirostat
updates from each committed constrained target distribution, including accepted
proposals, replacements, and bonus tokens. Its policy disables exact optimistic
promotion without disabling drafting or verification; standard lanes retain their
existing lookahead capability. All lane settings are validated before backend
prompt preparation or execution.

```rust,ignore
use eredu::api::{PreparedChatGenerationSettings, TextSamplingStrategy};
use eredu_core::GenerationConfigOverrides;

let settings = PreparedChatGenerationSettings {
    overrides: GenerationConfigOverrides {
        do_sample: Some(true),
        temperature: Some(0.8),
        ..Default::default()
    },
    strategy: TextSamplingStrategy::MirostatV2 { tau: 5.0, eta: 0.1 },
    seed: 42,
};
```

## Implementing another backend

A new backend should:

1. implement the `NeuralBackend` operators required by the architectures it
   intends to run;
2. implement parameter materialization, binding, transfer, and exact submission
   completion capabilities;
3. implement concrete runtime-state/cache storage when cached or paged
   execution is desired;
4. implement `ReplicatedTextSessionMechanisms` and pass them to the neutral
   replicated constructor rather than defining a text-session lifecycle;
5. implement collective operations when distributed execution is desired;
6. bind those capabilities to neutral architecture and runtime contracts inside
   the concrete backend adapter;
7. expose a narrow adapter for selection by the facade, where
   backend-independent application orchestration remains;
8. populate portable capability, resource, admission, and telemetry reports;
9. add optional multimodal, speculative, realtime, transfer, or distributed
   capabilities only when supported;
10. realize portable execution plans through an
   `ExecutionPlanBackendFactory`; and
11. run the reusable backend and architecture conformance suites.

Facade selection is an upward dependency on the backend's adapter, not
ownership of concrete composition: the facade composes generic model and request
APIs with the adapter, while the backend must not depend on facade APIs or
orchestration.
Concrete binding consumes architecture-owned declarations and does not take
ownership of model-family configuration, checkpoint naming, execution
equations, or state geometry.

Adding a backend never requires implementing Llama, mapping Llama checkpoint
names, or constructing a Llama-specific cache. Backend-specific compiler
artifacts, buffers, command queues, caches, and completion primitives remain
associated implementation types.

## Guarantees and verification

The repository mechanically verifies portable behavior and feature-gated
builds:

- the feature-disabled `portable_facade` and `backend_conformance` suites compile
  and exercise the public contracts through mock backends;
- the unified `reference_conformance` target runs one independent numeric
  backend through the production replicated, routed, composite, and
  partitioned constructors, including exact SafeTensors/GGUF payload access,
  direct and transformed formats, resident and addressable storage,
  TP/PP/EP Cartesian execution, state rollback, completion, and failure timing;
  the same target drives the distinct neutral speculative and realtime
  lifecycles, including Moshi and PersonaPlex; a synthetic typed architecture
  extension reaches those existing mechanisms without a backend family case;
  bounded independent adapters also consume canonical artifact/prepared-source,
  host-media, binding, admission/report, replicated-session, and logical
  placement facilities without implementing those algorithms;
- native facade integration tests compile the published facade target normally,
  while backend composition coverage remains in crate-private backend unit
  tests; and
- architecture, runtime, and backend conformance tests cover the relevant
  production contracts.

Dependency direction and semantic ownership are review rules recorded in
`AGENTS.md` and expressed by the crate manifests and public type boundaries:

In-memory preparation, session, topology, state-identity, and prompt-cache
handoffs expose validated constructors and read-only accessors rather than
public fields. Versioned persistence records are the deliberate exception:
prompt-cache manifests, blocks, and tensor records remain plain public-field
wire schemas for stable Serde encoding, and every decoded record is validated
at the persistence ingress before it can become runtime state.

- `eredu-architectures` contains model-family policy but no concrete backend
  imports;
- reusable modules in `eredu-backend-mlx` contain backend mechanics but no
  model-family configuration, checkpoint naming policy, state geometry, or
  layer equations;
- native dependencies remain behind facade features, preserving the
  `default-features = false` build; Linux CI also checks every weakly forwarded
  native and media feature without enabling `mlx`; and
- model families and their neutral execution remain available without enabling
  a concrete backend feature; feature gates belong on backend adapters, not on
  entire families; and
- backend implementations do not depend upward on `eredu::api` or portable
  facade runtime orchestration.

These rules are recorded in the repository-root
[architecture rules](../AGENTS.md). We intentionally do not enforce them by
inspecting the Cargo dependency graph, scanning Rust source for substrings, or
asserting a particular file layout: those checks couple architecture to
repository shape instead of semantic ownership.
Repeated violations should be made unrepresentable with a crate boundary or
visibility change.

## Independent projection banks

The shared decoder's `DecoderProjectionOperator` supplies both optional values
before attention and the feed-forward projection after attention. Q/K projection,
rotary position handling, normalization, masks, ordinary KV storage, output gates,
output projection and residuals remain in `eredu-architectures::decoder`.
Architecture block factories select these stages and their parameter topology;
native backends receive ordinary operator specifications.

`eredu-nn::GroupedLinearSpec` describes independently selected complete input
projections, a per-expert pointwise activation, and an owned contiguous output
range. Its output is the weighted sum of activated expert projections. Output
partitioning retains the entire input axis, so nonlinear activation occurs after
a complete projection. Unlike a row-parallel feed-forward down projection, this
owned output does not require a tensor-parallel sum. MLX realizes the contract
with generic indexed matrix multiplication and pointwise operations. CPU reduced
precision uses ordinary matmul over contiguous selection runs and views of the
selected matrices, without expanding weights per token or widening a complete
expert bank.

Every routed provider request and control callback carries a `RoutedBankId` in
addition to its logical layer. `RoutedObservationPoints` maps bank identity to
its canonical observation path and global cardinality. `RoutedBankProviders`
retains independent providers and observation identities. Their storage handles
share the selected residency pool, while acquisition and completion leases protect
individual entries. Heterogeneous prepared providers may be erased behind
the same neutral provider contract. A missing bank fails before dispatch.

Grouped RMS normalization retains one full-width checkpoint scale while reducing
independently within each group. The native realization promotes input and scale
to FP32 for reduction and multiplication before restoring the input dtype.
Rotary specifications retain an explicit amplitude independently of frequency
scaling. Architecture normalization resolves family conventions before passing
that amplitude to a backend. `RotaryArithmetic::InputProducts` additionally
specifies input-dtype rotary products, while `Native` allows the backend's fused
rotary arithmetic.

`eredu-checkpoint::fp8` normalizes block-FP8 and compressed-tensors metadata into
physical block geometry, dequantization-scale names and explicit module
exclusions. Architecture schemas assign that policy to actual Linear parameters
and retain canonical companion recipes; embeddings remain a distinct operator.
Shared decoder construction consumes complete `LinearFormat` values, including
FP8, rather than treating the absence of integer quantization as dense storage.
Encoded input-axis alignment participates in the same portable partition rules.

Routed preparation and selection retain a `BTreeMap<RoutedBankId, ...>` of exact
bank contracts. Each entry owns its grouped equation, expert ownership, catalog,
route counts and selected materialization tasks. `PreparedRoutedTextArchitecture`
and the shared routed-session constructor consume the complete collection;
there is no separate ReLU-squared prepared architecture. Native bank binding is
called once for the complete selected collection and has no family branch.
`ParameterBankKey` includes bank, execution unit and global member; the backend
uses this neutral identity directly. Scoped handles share one residency manager
and the configured host/device budgets, so switching banks can evict entries
from either bank. Runtime reports retain per-bank placements and counters, while
aggregate peak occupancy counts each shared physical pool once.

Routing specifications independently select projection, score and coefficient
precision. Scaled softplus is a neutral operation with explicit beta and an
input-dtype output; native implementations may widen intermediates before one
final cast. These policies belong to portable equations, including cache identity.

Expert-exchange waves carry an architecture-selected `ExpertRouteInvocation`
with bank identity, logical unit, access class and local output width. The neutral
transport validates input and result geometry independently; projection banks
can therefore return fewer columns than they consume. Owner-local execution IDs
remain distinct from global `(bank, unit, member)` storage keys throughout the
forward and reverse route permutations.

Partitioned preparation retains the same identified bank collection and builds
one provider collection for resident or addressable execution. `ExpertOutput`
parameter groups combine ownership of the expert axis with ownership of output
rows; their complete input axis is preserved. Collective schedules retain each
bank's input/output geometry and order within a block, and pipeline cuts occur
only after a completed block. Ordered expert reduction sorts returned routes by
global expert identity before scatter, preserving the selected accumulation
precision across expert exchange.

Ordinary KV and hybrid KV/fixed-state snapshots copy strided tails through
contiguous views and copy every sealed paging block into an independent session
namespace. These managers share the source process pool, so branches retain the
same finite host/device limits. Snapshot admission includes sealed blocks across
all tiers, catalog metadata, tail arrays and conservative growth through future
block boundaries. Completion/accounting ledgers remain outside the exchanged
model state. Native tests compare interleaved branches against an independent
attention calculation for full and sliding paging.

Selected grouped-linear materialization derives compact field identities and
storage dtypes from the declared projection format, including byte FP8/native
GGUF weights and integer MXFP4 scales. Architecture recipes preserve both
logical projection dimensions and encoded byte shapes. The neutral lowering
contract validates those distinct geometries before a backend accepts a derived
native-block source; source catalog ownership remains in `eredu-checkpoint`.

Ordinary distributed snapshot copies complete their native work and pass a
portable all-rank preparation agreement before publication. Branch exchange
validates every rank before moving state; the existing native communication
owner retains completion resources. Live commit epochs, communication/capture
budgets and copy ledgers are never part of a rewindable branch. Independently
addressable bank observations are added once to ordinary parameter-memory
reports; sharing a physical pool does not duplicate its owned entries.

Cold parameter resource totals are derived in `eredu-runtime` from the retained
executable tasks, including attached physical companions once. Architecture
selection supplies the independent-bank identities; shared experts remain in
ordinary block windows. Artifact profiles retain their unsharded scope, while
native session observations report actual rank-local replicas and bank occupancy.

Automatic schema version 7 also retains a separate selected-rank resource
profile. K2's tensor-free physical parameter description feeds the same TP/EP
placement primitive used by construction. Runtime sizing applies each output's
own physical geometry, including scale companions and replicated global routers;
PP ownership selects static consumers and repeated units before aggregation.
KV accounting uses only locally owned layers and complete KV heads. Source-recipe
workspace uses the existing bounded binding-placement rewrite for both ordinary
parameters and individual bank members. Compact-bank bounds preserve each bank's
independent identity and the selected finite assembly budget. Native recipe estimates include source copies, intermediate tensors, indices
and contiguous outputs. Separate conversion estimates count encoded outputs;
queue concurrency and per-bank compact bounds are explicit. These are conservative
tensor-buffer estimates, not bounds on opaque driver/allocator bookkeeping.

The vendored MLX CPU BF16 matrix kernel accumulates complete dot products in FP32
with explicit row-dot and strided-column reduction trees. It does not widen
or expand a bank's weights. Batch row count does not change accumulation
precision, and bank slicing preserves the projection's row-dot layout. This is a native matrix mechanism; it carries
no family configuration, checkpoint names, routing policy, or layer equations.


Independent external drafting uses architecture-owned inspection and preparation
of a second ordinary model. The source graph, lowering, tokenizer proof and
selected placement remain paired until native materialization. The shared runtime
owns private proposal branches, full-sequence target verification, accepted-prefix
replay and joint rollback. Ordinary sessions expose complete sequence logits
through their existing publication/completion transaction. Backend adapters bind
the selected ordinary executables and isolated state copies; the facade retains
one speculative sampling, termination, observation and controlled-session driver.

### Scalar component instrumentation

`eredu-core::component` holds compact, versioned component identities and semantic
parameter relationships. Shared decoder configuration also supplies parameter
aliases to the constructed neural slots, so repeated invocations share edit
authority while retaining separate execution units and mutable state.
`eredu-architectures` derives component relationships from the same decoder
configuration used to construct projections, including gate/value reads, GQA row
mapping, fused segments and shared invocations. They remain declarations; runtime
capture/intervention reports describe support for the selected execution.

`LayeredArchitecture::forward_unit_observed` lends the existing observer to the
architecture's internal boundaries. The shared decoder owns normalization,
attention aggregation, FFN activation and residual timing. It records originals,
applies interventions to values consumed downstream, and offers distinct bounded
`.effective` observations. The runtime owns traversal, ledger admission and
pre-work activation-edit reservations; MLX implements masks and reports native
storage/copy allowances. No family equation or analysis algorithm moves into the
backend. See [component analysis](component-analysis.md) for current coverage and
outstanding integration requirements.

Ordinary non-gated FFNs use the architecture-owned `decoder::unary` equation
(affine read, activation, affine write). Nemotron-H dense units reuse it for
ReLU², including input/output biases. Its fixed-state replicated adapter forwards
observers through the same physical attention and FFN boundaries as its full
architecture. The resident driver forwards internal unit observations as well as
input/readout hooks. Observation stays optional and does not introduce extra
component evaluation when disabled.

The native key/value profile includes an explicit stateless entry for physical
FFN invocations. It retains no arrays and rejects cache updates; snapshots,
transactions and paging preserve these entries without allocating cache blocks.
This realizes the neutral layout rather than inferring state from a family name.

The same layered traversal lends observers to `begin_forward_observed` and
`finish_forward_observed`. Architecture implementations own the scaled embedding,
final residual, normalization and affine-score boundaries. Traversal hooks compose
observation and intervention without adding another forward driver; disabled
hooks use the ordinary methods. `ComponentReadout` declares the output equation,
including nonlinear post-transforms and intervening block normalizations. Host
signed contribution calculations remain in examples and `eredu-evaluation`.

`eredu-core::parameters` defines loaded parameter facts, bounded rectangular
queries, logical-axis projections and immutable multi-parameter edit admission. The runtime owns cumulative
reservation and traversal through architecture-declared static modules and
selected residency units. `PreparedParameterSlot` retains the exact recipe output
metadata and architecture owner at construction, allowing loaded discovery without
building units or allocating tensors. The neutral parameter-operation loan passes
the semantic group/unit address to the selected policy; native completion retains
the unit and its transfer through query or edit evaluation. Random-access
inspection does not advance the ordinary disk-forward transfer window. `eredu-nn::ParameterSlotVisitor` carries stable parameter metadata;
it does not reopen checkpoint sources. The MLX realization performs native region
copies, contracts bounded host directions against effective weights natively,
and prepares affected-parameter replacements before publication. Packed matrices
are decoded using the retained materialization format and architecture-declared
companion roles; an admitted edit publishes an F32 copy of only its affected
matrix. The native projection/embedding selects that floating handle until
restoration. Full conversion and per-edit destination costs are reserved first.
Original parameters remain within the session's completion-retained payload.
Bounded transactions prepare each affected unit separately and retain only their
charged original/replacement handles. One publication updates the selected
populator's immutable override set plus pinned static slots; every subsequent
unit reload applies that set after its original exact binding. Removing it restores
pinned/resident handles and resumes the untouched prepared sources for bounded
units. The ordinary residency ledger accounts for unit materialization separately
from cumulative query and overlay work reservations.
Publication clears incompatible state and invalidates native snapshots; facade
session identities invalidate previously prepared requests. Parameter queries and
edit records carry the effective execution version. Packed-format verification and
additional residency/partition mechanisms remain explicit outstanding work in
the component-analysis support matrix.

Schema-admitted redundant sources remain distinct from executable parameter
sources. A tied parameter may retain exact redundant source keys in its neutral
requirement; strict native source-use validation consumes those keys. Removing a
redundant output-head materialization also removes its own companions, while the
executed embedding retains its original companion contract. Backends infer none
of this from family or checkpoint-name patterns.

The shared static-module readout owns optional residual, normalization and affine
score hooks. Both dense and heterogeneous replicated decoders call that helper;
Nemotron-H declares its embedding and final norm identities from family policy.
Draft prediction paths retain their distinct normalization timing.

Native GGUF parameter decoding preserves the retained byte order. Little-endian
optimized kernels remain available; big-endian dispatch uses endian-aware scalar
loads in the general kernels and canonical host decoding. Effective-parameter
verification covers byte order as well as packed codes and scale companions.

Component read declarations also retain architecture-owned Q/K head normalization
geometry and gain-sharing policy. The declarations mirror the normalization
specifications used by shared attention construction, including independent
per-head gains and shared head-width gains. This interpretation does not move
into parameter-query or backend code.

Query/key component dependencies use checked full-head row ranges, distinct from
scalar value-channel reads; the contract permits unequal read and value widths.
LFM2 owns its mixed attention/convolution component seams and the normalization,
residual and state-update order shared by full and replicated module execution.
Convolution writes remain separate undecomposed residual terms. Dense FFN hooks
share the existing gated equation, and routed providers retain their observations.

Readout declarations identify undecomposed residual writes by architecture node,
invocation, original/effective observation and residual scale. Consumers can account
for mixed convolution and routed-expert contributions without parsing path names.

LFM2 configuration admission accepts the published `block_ff_dim` width and
`full_attn_idxs` schedule, as well as their normalized counterparts. Architecture
normalization derives exact dense widths and state schedules and rejects conflicting
schedule declarations; a backend never interprets these source-format aliases.

Nemotron-H's shared physical-unit driver observes complete Mamba and routed-plus-shared
expert writes before residual addition. Architecture readout declarations join these
undecomposed terms to their operator nodes; existing routing captures retain their
original timing. Ordinary execution uses the same driver with instrumentation disabled.

Nemotron's always-executed shared expert also declares individual ReLU² units,
canonical up rows and down columns, optional biases, and the block's RMSNorm gain
and epsilon. Its sibling `shared.feed_forward` scope reuses the ordinary unary
component driver in replicated and tensor-parallel execution. Shared input edits
affect that branch; effective shared units feed the down projection, and the reduced
effective write enters the routed sum once. The shared group inherits its sparse
operator's parameter ownership. Existing neutral partition coordinates and native
parameter mechanisms consume these declarations without a backend family branch.

Independent expert banks retain architecture parameter declarations alongside
their exact lowered member bindings. Runtime joins these into bank-owned
`PreparedParameterSlot` records, preserving original member identities, companion
relationships and local physical geometry. Native parameter loans use the same
prepared cache manager as inference, copying selected members sequentially under
completion-safe transfer ownership. They do not reopen artifacts or retain every
source member in the residency cache at once.

An overlay publishes ordinary module handles and independent-bank replacement
handles together. Bank providers select those handles while constructing compact
groups; immutable checkpoint-backed cache entries remain available for restoration.
The bank reports updated compact-member byte costs to the neutral chunk planner,
including packed-to-floating promotion. The operation's parameter budget owns
retained original/replacement matrices, while bank residency and compact-scratch
limits continue to govern inference. Compact binders accept floating weights only
at the logical shape supplied by the neutral operator specification; packed
weights retain their encoding geometry and companions retain exact shapes. All
bindings validate before compact handles change. Native and neutral acceptance
for this path is tracked separately in the component-analysis guide.

Neutral parameter placement also represents shared logical chunks with a short
final physical chunk. Architectures choose the chunk width from the selected
projection format; runtime validates matching unit counts and remaps encoding
companions. The architecture planner lowers these declarations to ordinary
contiguous ranges or ordered segment indices. Native binders consume those exact
placements without inferring quantization boundaries. Dense FP8 FFNs use this
contract for partial intermediate-axis blocks, preserving the same ownership for
read rows, output columns and scale companions. Independently encoded fused
gate/up tails require their own explicit scale geometry; the chunk contract alone
does not establish support for that layout.

`eredu-nn::LinearRowLayout` declares independent row-block origins in a projection.
Architecture format specifications carry that declaration into parameter metadata;
the equal-partition form retains its meaning after matching gate/up slices change
each partition's local width. Runtime expands scale geometry and segment coordinates
from those origins. Native grouped projection kernels consume the declaration while
sharing the ordinary quantized input, and effective-parameter decoding and scratch
admission use the same scale rows. Backends do not infer this layout from checkpoint
names. The K2 checkpoint convention declares two independent row partitions for its
fused FP8 gate/up source recipes.

Shared decoder attention placement derives one GQA-head partition from the joint
alignment of query/key/value read rows, optional gate rows and output columns.
The architecture configuration declares the value projection's format even when
a routed bank supplies it. Split attention retains partial final head chunks;
ordinary projections, per-head parameters and separate value banks consume the
same ownership. Scale companions use primary geometry instead of ratios inferred
from rounded scale shapes. Cold state sizing and complete partition state retain
the actual per-layer local head counts, which can differ between dense and FP8
layers. Native binders consume these retained ranges without selecting head cuts.

Additional diagnostic tensors use a deferred observation factory. Architectures
supply the observation identity, native mechanisms describe exact prototype geometry
and conservative creation storage, and runtime invokes the factory only inside the
ordinary capture transform after reservation. Absent/skipped observations perform
no factory work; generated handles are released before returning to inference.

Neutral projection operators optionally expose the actual multiplication input.
The default is identity; input-transforming native mechanisms override it.
Architecture component seams select observation paths without selecting native
quantization. MLX reuses the activation codes and scales already needed by FP8
inference, offering their F32 reconstruction as a deferred diagnostic on GPU;
the CPU reference path already needs that reconstruction. Loaded parameter
facts retain the selected input transformation, including its change when an
admitted F32 overlay replaces a packed matrix. Research consumers account for
final-head input quantization separately from residual component contributions.


Capture source precision is a neutral optional `TensorDtype` fact, independent of
host payload conversion. `CaptureBackend::source_dtype` reads metadata only;
runtime records it through the shared capture/evidence driver. Deferred values
report their own dtype only after generation, never their prototype's dtype.
MLX reuses one native scalar mapping for capture and communication.

Native parameter publication retains the effective dtype upper bound alongside
the overlay identity. Loaded runtime-state estimates combine it with the original
state-width assumption, covering promotion by F32 matrix replacements without
family-specific dependency analysis in the backend. Removal clears both the edited
state and this bound. Snapshot/copy accounting continues to use actual tensor
sizes. Publication and removal return parameter facts for the resulting operators,
including their current projection-input transformation.


Component read mappings distinguish a scalar row from a complete query/key head.
An optional head stride and scalar `Blocked` mapping describe interleaved fused
head segments without exposing checkpoint-name parsing to applications. Absent
head stride retains contiguous geometry and old serialized records. Range
construction rejects zero widths, overlapping strides and arithmetic overflow.
Architecture declarations supply the mapping; backends still receive only exact
bounded parameter regions. Hybrid-family declarations and shared execution hooks
have neutral and native coverage in the component-analysis matrix.


Qwen hybrid target blocks now share one instrumented residual driver between
resident and rebuilt units. Their dense SwiGLU units and gated self-attention
channels reuse the ordinary decoder hooks; the gated-delta mixer and routed/shared
expert sum remain explicit whole residual writes. Query/gate reads describe the
actual per-head interleaving, and Q/K, sublayer and final RMS normalizations retain
the learned gain offset of one. Full and typed replicated text adapters forward
observers through the same native-independent equations and readout. Existing
routed provider ownership stays in runtime. Multimodal conditional propagation,
recurrent channels, routed units and partitioned execution use their declared
component boundaries; the component guide records their verified configurations.

The shared decoder's physical-format declaration permits an unaligned final FP8
block in a complete projection. Runtime retains such a group as one logical
partition unit and still expands exact ceiling-sized scale companions. Aligned
groups retain their existing finer units. Strict packed formats keep their
alignment checks. This permits ordinary resident and bounded loading without
allowing a rank slice to start inside an encoded block; uneven tail-aware TP
partitioning needs additional placement support.

Shared tensor-parallel decoder blocks expose the same rank-local channel and
FFN-unit boundaries before their existing row-projection reductions. Normalized
inputs and reduced residual writes retain their ordinary block equations. These
operator seams add no collectives; global component coordinates, capture delivery
and parameter transactions remain responsibilities of the partition driver.
The new seams alone do not establish public partitioned component support.
The parallel layered contract now has an internal-observation unit entry. The
ordinary parallel residency traversal supplies its shared observer to that entry,
retains forward resources for completion, and keeps group/unit boundaries in the
existing traversal hook. Shared full/local decoders and transparent composite
wrappers forward it, and the direct partition executor consumes that traversal.
Parallel layered traversal also invokes architecture-owned observed ingress and
readout when an observer is present. Shared decoders expose embedding, residual,
normalization, actual projection input and complete linear logits, with the
ordinary tied/untied vocabulary gather and output softcap. Neutral vocabulary
projection input observers delegate input transforms to the backend; MLX uses the
same deferred FP8 evidence and failure propagation as ordinary projections. An
absent observer does not construct component paths or request transformed input
copies. Global coordinate admission, specialized pipeline propagation and
capture transport remain unfinished integration work.

Direct pipeline and composite unit strategies now forward the supplied observer
to the selected architecture's serial or tensor-parallel internal-unit entry.
The partition executor retains its existing boundary, residency and completion
sequence. Local shared decoder blocks validate ownership and index local state
before using the shared observed equations. This change propagates unit internals;
the ordinary pipeline driver also forwards observed ingress and final readout.
Its existing optional-observer implementation shares state-layout validation,
group entry/exit and output ownership. Shared decoder ingress marks embeddings
only when it actually looks up tokens; upstream hidden boundaries are not
relabelled as embeddings. Final readout runs only on its output owner.
Specialized routed strategies, composite ingress and global capture transport
require further integration.

The default bounded serial and parallel traversal-hook entries also dispatch
architecture internal-unit observers when the hook requests activations. This
uses the same residency traversal and completion sequence. Explicit custom unit
executors keep authority over their own internal observations; the default unit
dispatch does not replace specialized provider calls. Unobserved hooks continue
through ordinary unit execution.

Global scalar-axis identity belongs to `eredu-core::component::ComponentCoordinateMap`.
Its validated range or ordered-index representation maps local storage back to
global component IDs and rejects duplicate or invalid global masks before local
filtering. `eredu-architectures` compiles these maps from the invocation's logical
write projection, semantic partition units and retained execution ownership.
Packed physical columns do not define scalar offsets, and shared source aliases
do not confer ownership of another invocation. `SelectedExecution` exposes that
descriptive layout without exposing its private family-selection branches.
Runtime lowers compact masks and explicit token slices through those coordinates,
preserving empty keep-only sets and global identity validation. This lowering
performs no native work and grants no capture or intervention authority. Global
budget admission, rank agreement, delivery and parameter transactions remain
separate requirements before public distributed capture can be advertised.

Core `CaptureSlicePartition` validates bounded native fragment slices and their
destinations within the original global selection. Contiguous partitions retain
the global stride origin; ordered index maps become strided fragments without
exporting unrelated local columns. Runtime fragment capture borrows the original
admitted plan and uses the same native reservation/transform path as ordinary
capture and intervention evidence. Global raw-tensor assembly validates identity,
source precision and complete nonoverlapping coverage and reserves its host copy
before allocating it. Missing fragments and non-value outcomes cannot become
measured zeros. These mechanisms do not establish peer completion, producer
authority, transport budgets or publication agreement; the distributed session
driver must supply those before public support changes.

Core fragment geometry also owns allocation-free intersection of finite strided
coordinate sets. Runtime uses it to check coverage for raw tensors, summaries
and fixed-edge histograms; reduction assembly allocates by producer/fragment
count and bin count, not by global tensor width. Runtime combines native finite
statistics in F64, preserving exact count/non-finite semantics and documenting
reduction-order rounding. Ordinary and partition-generated observations share
one lazy native adapter, so reservation precedes factory construction and
source precision comes from the generated value. Backends continue to supply
only ordinary native transforms and storage estimates. These additions do not
supply distributed expected-producer authority or native completion agreement.

Versioned partition capture host-record schemas live in `eredu-core`. They contain
bounded execution/run/overlay/submission context and ordinary local capture
records; deserialization grants no execution or placement authority. Runtime owns
expected-producer receipt admission, canonical geometry/context digests, bounded
encoding/decoding, exact sender/ordinal/shape checks, one-shot receipt collection
and empty-selection acknowledgments. It reuses ordinary raw/reduced assembly and
preserves source precision and typed non-value outcomes. Native collectors still
produce the same local records. The distributed session must bind the context to
retained execution, establish global native quotas, and agree forward/publication
outcomes; receipt bytes cannot substitute for those owners.

Runtime `PartitionCaptureExchange` owns capture-specific transport reservations,
readiness/length frames, bounded payload gathering, and the final all-rank
receipt decision. It shares core's portable-word transport and exact bounded
completion contract with scheduler protocols. Backend `PartitionCaptureTransport`
adapters provide retained rank/policy facts, side-effect-free logical storage
estimates and the existing shared failure owner; they contain no capture geometry,
family semantics, JSON decoding, or publication policy. MLX implements those facts
on `MlxDistributedSession`, reusing its native gathers and completion owners.
Transport failures and unagreed protocol mismatches fence that same authority.
Fully agreed local/peer rejection returns a typed cause for the enclosing forward
driver to apply skip/rollback/termination policy. Producer and decoder work retains
ordinary capture reservations; prepaid control credits remain available after a
value-budget miss. The exchange must be admitted on every rank before work and
called only at a common post-forward boundary. Public distributed capture remains
unverified until session admission, producer quotas and generation commit use it.
Per-selection receipts share the immutable global capture plan through `Arc`,
avoiding copies of all selections for each captured point.

The retained architecture selection also projects component layouts for any rank
in its admitted topology. `component_partition_layout_for_rank` reuses admission's
ownership compiler against the same retained global execution requirements; it
creates no state, device, communicator or execution selection. The local-rank
path consumes the already retained local ownership. `component_partition_layouts`
checks its rank bound before compiling one reusable map set; selection projection
then borrows those maps without repeatedly deriving full parameter layouts. `ComponentPartitionLayouts::capture_producers`
compiles one global selection into bounded rank/fragment declarations, choosing
the lowest rank for identical coordinate replicas of the same invocation while
retaining distinct shards with empty selected overlap. Backends do not reconstruct
pipeline ownership or derive it from shared checkpoint aliases. These layouts
remain descriptive: runtime receipt admission validates exact coverage and binds
them to a live run, native quotas and delivery policy.

The shared partition runtime constructor retains the complete parameter
description it already checks against the selected architecture and local
materialization tasks. `PreparedPartitionedSessionRuntime` transfers that
immutable declaration into `ReplicatedTextSession` through `Arc`; native
erasure forwards it for both text and composite sessions. Prepared discovery
retains the architecture descriptor, execution identity and cold partition
selection, then binds this checked description when the model is assembled.
`PreparedModelDiscovery::component_partition_layouts` therefore uses the actual
retained execution without revisiting artifacts or inferring global ownership
from local native slots. Compilation remains bounded by an explicit rank limit
and does not resolve the deferred artifact hash. This metadata does not by
itself enable distributed capture or interventions.

Transactional observation is part of the shared text-session lifecycle.
`ActivationObserver` supplies local preparation, coordinated admission, completed
delivery and an infallible final commit notification. Runtime checks uniform
participation before any observer collective, propagates local preparation
failure before model/state work, and invokes delivery after exact output/state
completion while retaining the ordinary rollback checkpoint. Delivery remains
provisional until the existing final commit succeeds. Early failure, rollback or
an uncertain commit discards provisional delivery through the shared guard.
Native observation adapters forward these callbacks; they do not implement a
second transaction driver.

Input preparation uses an earlier shared `InputPreparation` phase. The
`prefill_input_result_with_observer` and `decode_input_result_with_observer`
entrypoints accept a local preparation result so a rank cannot return from
prompt validation while its peers enter observer or model collectives. Ordinary
execution participates too. The fallible-input entrypoints require selected
bounded phase agreement on a partitioned session; final-commit support alone
cannot authorize them, including when a local input succeeds. An agreed rejection
consumes the step epoch, keeps
the installed state and prompt identity, and discards the observer transaction
before observer admission, checkpointing, or forward execution. Native adapters
carry token extraction, composite-input admission, generation parameter-version
checks, capture prompt alignment, and decode tensor framing through this result.
Framing stays under the existing native submission owner. Native recovery must
still establish settlement before reuse; state-preservation evidence is not a
completion signal. Successful ordinary generation prefill retains the core
runtime's exact admission check and execution-plan target proof.
`eredu_nn::Error::backend_source` preserves the typed original
preparation cause while keeping concrete error types out of portable signatures.

Architecture capture placement now retains a `PartitionedObservation` separately
from scalar component groups. Each observation declares its own axis and local
coordinates. Normalized sublayer inputs follow their logical invocation and use
the full hidden axis; embedding observations follow input ownership, while final
residual, normalization and gathered affine scores follow output ownership.
Tied parameter storage does not transfer any of these invocations. Equal retained
replica maps select one exporter while all executing ranks retain hook agreement.
The scalar group maps remain the source of component-mask lowering.

Partition capture also distinguishes source participation from fragment export.
Architecture placement declares the exact ordinary source shape on every hook
member, including replicas and shards with no local selection. Core's capture
backend contract supplies cold dependency-completion bounds and bounded source
preparation. Runtime reserves those costs, seals shapes and costs into common
coordination, and agrees geometry before submitting any source graph. A distinct
prepaid vote settles capture success afterward; preflight alone cannot certify
completion. Globally empty and skipped selections submit no source work, and a
nonexporting generated capture never invokes its factory.

MLX completes ordinary prototype dependencies on all active ranks without host
tensor export. Its exact event retains the source, stream and native world, and
the nested recovery scope remains under the enclosing model submission owner.
Deadlines quarantine work and fence the shared communication authority; polling
errors do not release model or source ownership. Source reservations are separate
from transform/factory credits and survive failure and restore. Their logical
bounds exclude ordinary inference storage and private native allocator workspace.

Final `model.logits` retains distinct export authority. Architecture rank admission
selects the publication owner from the final group's semantic merge destination
and cohort; direct, routed, pipeline and composite plan construction consume that
same retained owner. Observation placement distinguishes exporter eligibility from
ordinary source coordinates. All output invocations can prepare dependencies,
while only the publication owner supplies final-logit fragments.

Runtime offers nonpublishing output tensors through `ActivationObserver::observe_replica`.
Its default does no work; ordinary inspection and intervention still execute only
at the authoritative seam. Partition capture uses the replica callback for its
prepaid source and status work. Native adapters forward the callback and retain
errors without selecting ownership. Capture timing includes source preparation,
transforms, coordination and final receipt delivery, including failed attempts.

Runtime also admits bounded `TokenScores` and `TopCandidates` receipts from one
complete ordered final-logits producer. It checks exact local and destination
geometry before source or deferred-factory work; shard-local reductions cannot
authorize a global probability. Native collectors use their ordinary full-vocabulary
algorithms. Runtime validates bounded payload identities, score/rank/competitor
consistency and optional domain bounds, then moves the reduction into the global
record with producer provenance. Decoder and output reservations precede their
allocations; final-logits replicas still complete ordinary dependencies without
exporting. Structural receipt checks do not prove numerical model correctness.

Preview assembly uses the same exact global coverage proof. Each positive-stride
fragment has increasing flattened destinations, so its first N local selected
values contain all of its contributions to the first N global values. Runtime
validates local truncation counts and payload precision, reserves the global
prefix, and checks its complete emitted count. Global selected geometry remains
separate from the flat preview payload. Native backends supply ordinary bounded
prefix transforms without deciding global order.

Declared block residual normalizations use their logical gain/bias parameter's
retained invocation ownership for input/output observation placement. Aliasing
that parameter to a shared checkpoint gain does not transfer the observation to
another invocation or pipeline stage. Architecture placement exposes complete
hidden coordinates; native collectors retain their ordinary replica preparation
and bounded delivery policy.

Observation coordinates do not prove that a selected driver emits the value.
Runtime exposes separate input, unit, routed-unit, readout and publication hook facts.
Architectures declare coverage for ordinary, tensor-parallel and partition
methods separately; default methods declare none. A partition executor combines
those declarations with its actual per-unit strategy. The shared runtime adds
its final-publication seam independently. Native erasure forwards these facts
alongside the retained global parameter declaration, and architecture discovery
checks the observation's declared execution site before consulting native
collector support. Missing internal hooks report `Unverified`; a coarse session
inspection capability cannot enable them. This mechanism is not a substitute
for implementing the remaining specialized family/provider hooks.

Sparse provider coverage is independent of complete internal layer coverage.
`routed_sparse_observations` and `parallel_routed_sparse_observations` describe
the actual provider-aware calls; `ObservationHookSite::RoutedUnits` selects that
fact without enabling other unit observations. Pipeline strategies forward their
own sparse coverage. Loaded MLX discovery combines this fact with retained sparse
placement, native routed collection, and exact invocation-group status transport.
It admits `CaptureTransform::RoutedUnits` through the existing shared observer.
Sparse interventions additionally require the native gather/scatter and coordinate
mechanisms. Architecture-owned `routed_activation_members` projects retained
expert/unit ownership and read widths. Runtime extends the same
`SessionPartitionIntervention` with sparse chunk coverage, pre-forward reservation,
source/final votes and committed global affected counts. The backend maps actual
native rows through preserved origin tags and applies ordinary tensor operations;
it does not choose expert ownership or edit semantics. Every physical replica and
received source peer is edited, while retained publication ownership determines
which logical effects count once. Disconnected native Ring groups still require
coordinated relays and retain their explicit capability rejection.

Parallel communication projection also supports an explicitly selected group of
all tensor and expert ranks at a fixed pipeline/data coordinate. These groups
follow existing axis groups and preserve their IDs. Architectures request exact
failure agreement for a combined expert/pipeline stage; an axis group alone cannot
certify every physical replica of an observation. MLX realizes connected Ring
stage membership with a member-only reduction and broadcast along neighboring
ranks. Intermediate sends use the existing bounded communication owner, including
retention and quarantine of arrays, groups and streams. Model tensor collectives
keep their selected wave protocol. The same exact membership predicate controls
cold capture admission and native status execution; disconnected sets still need
coordinated relay support.

Final `model.logits` placement follows the catalog's vocabulary axis and retained
output/publication ownership even when component topology and readout equations
have not been declared. The optional component readout must reference that same
public final-output seam. Missing decomposition metadata cannot remove an
otherwise implemented final-output observation.

Whole residual-write ownership joins the architecture node's declared canonical
parameter namespaces to retained logical execution-unit owners. All matching
parameters must agree on one unit; static roles, missing namespaces and ambiguous
owners reject. Physical aliases and a node's display ordinal do not determine
placement. Whole-write values are complete hidden-axis contributions after the
operator's selected collective reduction, and still require actual driver hooks.
LFM2's ordinary, tensor-parallel and partition methods share block residual
ordering and readout instrumentation. Its dense feed-forward scalar hook precedes
the row-parallel projection; the convolution hook follows the complete mixer
output. Routed provider strategies retain their separate hook requirements.
The provider-aware architecture contracts declare internal unit coverage
independently for ordinary and tensor-parallel calls. Their observed defaults
preserve routing observations without claiming scalar component hooks. Runtime
shares its observed input/group/unit/readout traversal and forward-resource
retention with caller-selected provider execution. Direct and pipeline executors
forward the same admitted provider and parallel context; pipeline strategies
retain one provider-selection path for ordinary and observed calls. Complete
routed residual observations occur after the architecture's existing expert
reduction, while inactive pipeline collective waves retain their original
participation and completion policy.

Partitioned activation edits retain the original admitted operation rather than
rewriting its plan into rank-local identities. Core supplies a separate exact
intent digest for peer comparison while preserving each loaded session's original
admission identity. Digest construction streams canonical serialization; snapshot
and child-preparation estimates include both retained digest strings.

`PartitionActivationLayout` supplies all actual invocation members from retained
architecture coordinates, including nonexporting replicas and empty local shards.
Runtime projects payload positions, reserves host copies and native update work,
and issues move-only `ReservedPartitionActivation` work. It executes the ordinary
activation mechanism with the exact local shape and dtype. Compact masks remain
one local operation for permuted axes. A valid global logit mask can exclude an
entire local vocabulary shard; the original global admission must still leave a
candidate. These local contracts do not establish distributed session/epoch
authority, group agreement or global outcome publication. The shared capture
owner and forward protocol now compose those responsibilities through
`SessionPartitionIntervention`, as described below.

The shared decoder declares attention components independently of its feed-forward
operator. Routed Qwen and GPT-OSS therefore retain query/key/value mappings,
normalization, biases, channels and readout equations. An architecture-selected
routed unit declares its complete expert contribution separately; it does not
invent scalar expert units. `ComponentResidualWrite::input` identifies the actual
normalized whole-operator input and shares the invocation's hidden-axis placement.
`AttentionAttributes::sink_logits` names learned per-query-head sink logits used
in attention normalization; their value contribution is zero.

Generic decoder provider calls pass instrumentation through the same attention,
normalization, residual and cache engine as ordinary calls. Projection operators
select the completed residual boundary, preserving provider-owned routing events
without reusing their pre-reduction output as a complete contribution. Generic
partitioned input/readout hooks execute their actual shared boundaries. Qwen and
GPT-OSS explicitly report provider unit coverage; other custom projections retain
the conservative default until their implementation supplies these hooks.

K2 Horizon supplies direct and provider component hooks. Its ordinary and observed
feed-forward calls share one architecture driver: sparse units flow through the
existing routed-unit observer, shared SwiGLU uses a nested component scope, and
the completed routing event follows expert reduction and shared addition. Dense
prefix units retain the ordinary decoder hooks. Native backends implement the
same grouped and linear operators without selecting a K2 semantic branch.

K2 also derives routed and shared intermediate widths independently from retained
local parameter placement. Mixed encodings can require different TP boundaries:
384 routed FP8 units split into 256/128 complete-block slices, while a dense shared
FFN of the same global width splits into 192/192. The local architecture retains
the shared down-projection's actual input width for module construction; it never
reconstructs that width from the routed expert count or local routed width. This
keeps construction, materialization, component coordinates and overlays consistent
without moving family geometry into a backend.

`ComponentGroup::routed_reads` and `ComponentRoutedRead` are neutral relationships
for activated expert mixtures feeding an attention component. K2's architecture
projection declares its MoVA value bank, group-to-channel row mapping, SiLU,
selector/correction parameters and routing node. It does not declare a nonexistent
ordinary V matrix. Checked read/bias joins use actual loaded parameter geometry;
the resulting selections carry no query or overlay authority. The architecture
also owns shared-expert scalar groups, normalization and source identities.

Identical hidden coordinates still do not prove identical transformed
multiplication inputs. Such points require retained transform facts before global
capture can advertise them. Native code supplies collector facts and tensor
transformations without selecting architecture ownership.

Speculative facade entry points also agree whole-batch host and native-prompt
preparation. Controlled speculation agrees its initial capture validation before
entering the shared preparation driver. The facade exposes the same result and
cancellation agreement for caller-owned media/opaque-prompt preparation; local
inspection and immutable admission do not introduce hidden collectives. Core owns
local-result/peer-outcome resolution, reused by native adapters while an existing
submission lease is held. Embedded speculative execution uses retained session
transport to agree sampler/lane readiness before cache construction, then cache
results and initial lane delivery before invoking the ordinary or controlled
scheduler. Cache construction may itself enter architecture preparation
collectives, so a single vote after sampler and cache construction is too late. Native error wrapping retains its
original source. A rejected broad native operation remains fenced without a
whole-run restoration witness; completed agreement is not proof of safe reuse.

Detached media conversion uses the same native submission recovery as text
prompt construction. The generic adapter retains caller-owned tokenizer errors
without translating them to native failures; only native begin/status errors pass
through the supplied backend mapper. Processor observers execute inside that
scope, so abandoning failed preparation retains unresolved native work through
completion or teardown. This does not introduce session collectives into local
media preparation.

The facade's shared committed-token cursor also carries local host results and
cancellation through bounded delivery readiness before a prediction and after
its token/semantic callbacks. Ordinary and controlled sources use the same core
result policy and session ledger. A local decoder or record-budget error remains
typed; successful peers receive rejection before another prediction. A peer
cancellation preserves the already committed prefix and does not run another
forward. Empty capture draining establishes native completion for this boundary
without enabling component collection. Controlled lifecycle and boundary-record
publication use the same readiness policy after token commitment; peer cancellation
is retained before the next prediction. Completed ordinary publication agrees its
record result without rewriting an established termination reason. A best-effort
failure record preserves the original agreed error. Snapshot, branch and sampling
records share the controlled delivery helper; these collective operations must be
called in the same order by all participants. Speculative scheduling has additional
boundaries whose failure audit remains separate.

The neutral `SpeculativeExecutor::agree_text_preparation` contract also carries
visitor-owned setup through the retained session agreement. Architecture-owned
embedded executors and their erased adapters forward mechanism facts unchanged;
MLX supplies the same session transport and cumulative ledger. The shared runtime
scheduler agrees policy and completion support before registration or prefill.
Controlled setup agrees lane geometry and collector installation before creating
that scheduler. Local setup errors retain their typed cause; successful peers
receive `SpeculativeDriverError::Preparation` or the corresponding neutral control
error. This does not make failed native operations reusable or establish completion.

Speculative action selection now has a neutral host-fact contract separate from
native model execution. Runtime coordinates exact request identities/lifecycles,
per-request cancellation, completion, deadlines and optimistic eligibility using
confirmed fixed-size frames on the retained session transport. Completion and
eligibility require all participants; cancellation and expired deadlines propagate
from any participant. Core selects actions from those common facts and agrees
local preparation/action failures. Draft loops agree each sampling/prefix result
before another native forward; execution observers, proposal-state creation and
verification checkpoints agree before subsequent native work. Native adapters
supply transport and retain completion ownership; scheduler facts never release it. Runtime cancellation
signals a request for the next coordinated boundary instead of invoking a local
publisher immediately. Core agrees prefill, verification preparation, cache commit
and publication results before installing the new logical lifecycle. Publication
cancellation propagates before another scheduler action. Pending cancellation
settles the retained completion and commits only its safe prefix. Payloads remain
owned through rollback; successful completion remains retained through publication.
Runtime controlled sessions charge and agree each step record, and cancellation
drains that same action/record driver. Caller errors at an unfinished boundary or
a completed closure are agreed with peers before returning; already agreed failed
steps do not start another exit exchange. Neutral and facade tests cover these
boundaries; native distributed validation is recorded separately. Sampling uses
the shared native capture-error translator so portable budget causes remain
directly discoverable beneath the neutral public error.

Cold text-run preparation has a separate shared boundary before that forward.
Core declares request, prompt, sampling, instrumentation and initial delivery
stages, with ready/cancelled/failed dispositions and typed peer rejection.
The ordinary generation machine and detached controlled driver use the same
fallible preparation composition. Facade code supplies tokenizer, semantic policy,
trace delivery and cancellation decisions; native prompt and sampler construction
remain backend mechanisms. Exact token IDs can enter `PreparedChatInput::TokenIds`
without early native construction. Raw token iteration now returns the neutral
`BackendFailure`; capture installation separates typed `CaptureError` from
preparation/transport failure with `TextCaptureSetupError`.

Runtime's `TextPreparationCoordinator` uses the existing bounded portable-word
transport. Two fixed-size exchanges bind readiness to the retained setup identity,
monotone attempt and stage, then confirm validated decisions before continuation.
The native session shares that coordinator and its actual communication authority
across adapter clones; model snapshots and resets do not copy or rewind it.
Native gather estimates and runtime framing are reserved before submission, with
cumulative logical charges exposed separately from capture through
`text_preparation_usage`. These charges are not physical allocator measurements.
Unagreed failures fence the selected authority; native completion still retains
live resources through its safe disposition. Session synchronization checks that
authority before entering native queue waits, since a timed-out collective may
still own quarantined work. The existing forward input agreement remains
mandatory, including after successful cold preparation: a local failure during
final confirmation resolution need not be visible in another rank's completed
confirmation.

The retained execution distinguishes requested exact completion from its backend
availability. A transactional observer can require that available mechanism even
when ordinary execution is asynchronous; absent support is rejected before the
forward. Partitioned execution performs a participation status agreement even
when observers are absent, so mixed rank configuration cannot enter incompatible
observer collectives. No component tensor is created by this control agreement.
The shared capture owner blocks record draining while a transaction is pending,
keeps failed attempts outside snapshot boundaries, and preserves its consumed
epoch and ledger across restore. The partition observer below composes bounded
producer quotas, coordinated identities and receipt exchange into these callbacks;
public distributed composition is still incomplete.

`CaptureObserver::for_step` defers envelope admission to the shared local
preparation callback and obtains its phase from the actual forward. The capture
owner consumes the epoch before fallible reservation; failed admission cannot
complete or be restored while pending, even if it produced no record envelope.
Runtime reports `ReplicatedTextSessionError::BeforeStateMutation` only for the
shared preparation path before the driver receives mutable model state. Native
recovery consumes this evidence separately from successful rollback and from
native settlement; it never increments the successful-restoration counter for
admission rejection. Model-call error adaptation retains the runtime error and
its available cause chain rather than formatting away that object. The wider
neural/observer adapter source-conversion audit remains part of integration.

Core `CaptureReservation` exposes spending without ledger replacement or reset.
Its move-only `CaptureQuota` grants prepaid local work; unused or dropped credits
are never refunded. Runtime binds partition capture to the actual capture owner
and pending forward epoch, reserves every producer's work and every rank's receive
and assembly work, and coordinates a digest of the complete selection, layout and
quota set before source access. Core `CapturedStep.partitions` carries serializable
context, producer acknowledgments and selected-region evidence; these host records
cannot grant execution authority.

`PartitionCaptureObserver` owns this shared lifecycle integration. Architecture
`ComponentPartitionLayouts` implements the neutral `PartitionCaptureLayout`
contract, declaring both distinct producers and all invocation participants,
including replicas. A separate, prepaid `SessionPartitionHook` uses only those
participants for failure agreement before the next tensor collective. Inactive
pipeline ranks participate in common preparation and final receipt delivery.
The architecture manifest now selects exact failure agreement on tensor groups
even when pipeline parallelism is active. MLX supplies retained group lookup,
native status submission and bounded completion; it does not infer family ownership.

Native communicator ownership retains deduplicated opaque group facts from the
already validated setup-consensus manifest set, projected onto the local rank.
Inactive ranks can therefore validate and price a remote hook without realizing
its communicator. Native submission still requires the locally selected group.

Runtime `establish_communication_session` agrees a versioned setup transcript
containing each rank's manifest and fresh native instance nonce. Its common
`CommunicationSessionIdentity` distinguishes repeated setups with identical
manifests. MLX issues the nonce and retains the agreed identity alongside the
actual communicators; the existing two bounded setup gathers carry both facts.
Failed nonce issuance is exchanged before rejection. The runtime rejects metadata
exceeding 16 MiB per rank or 128 MiB in the padded gather before allocating padded
gather buffers or transferring payloads. Local proposal serialization precedes
this check; these serialized limits are not physical allocator guarantees.
The identity is descriptive provenance, never a communicator or submission lease.

`PartitionCaptureIdentity::for_session` binds this retained setup to the first
actual forward epoch claimed by its capture owner. Failed preparation consumes
that epoch; later steps and restores retain the original run identity. Production
loading creates a setup per model construction, while clones share that setup;
the shared model runtime owns the monotone forward sequence. Native composition
must keep those owners paired and supply actual artifact/execution/overlay facts.
This mechanism does not itself select the public distributed capture adapter.

MLX hook admission accepts the connected pipeline-stage and whole-session groups
selected by prepared component execution, including singleton and neighbor-pair
cases. Arbitrary disconnected low-level Ring memberships retain an explicit
rejection: independent status would require inactive peers to participate in a
coordinated world wave. This limitation does not exclude the prepared model
placements covered by the component-analysis matrix.

Loaded MLX capture now caches architecture-declared component layouts on discovery
demand and selects `PartitionCaptureObserver` through the same ordinary and
controlled prefill/decode entrypoints. Runtime
`observation_support_with_partition` applies placement/collector facts only after
the existing selection, phase and instrumentation gates. Architecture supplies
hook membership; MLX reports its exact selected independent-group mechanisms.
Unknown ownership stays unverified. This does not transfer family semantics into
the backend or enable parallel parameter edits.

Effective parameter partition geometry follows the same ownership direction.
`eredu-core::parameters::ParameterCoordinateMap` projects a selected global
rectangle into local rectangles and result destinations, including independent
axes and permutations. Its bounded edit and contraction projections retain the
original global selection. A contraction fragment also retains its source region,
so the receiver can distinguish additive partial contractions from duplicate
source coverage. Geometry is descriptive and grants no session, epoch, edit or
communication authority. Reservations precede metadata and payload copies; failed
or unused projections do not refund them.

Architecture `PreparedModelDiscovery::parameter_partition_layout_for_rank`
resolves the selected task, its declared aliases and emitted companions against the parameter topology
retained by construction. It reuses the production pipeline-ownership compiler and
lowers only the requested atomic parameter group through the production TP/EP
layout compiler. Selected packing-axis and encoding facts convert complete
physical quantization blocks to effective scalar coordinates. Fused segments,
ordered expert axes, absent pipeline owners and shared static consumers retain
their distinct meanings. FP8's uncompressed axes preserve uneven scalar ranges.
Neither source content resolution nor native parameter access occurs here.
Companion coordinates describe scalar scale/bias cells; they never inherit the
weight's packed-block expansion. Actual prepared slot metadata supplies sharing,
including intentional sharing between distinct materialization tasks.

Runtime `LayeredParameterOwner` lends the architecture and residency policy
already retained by ordinary, routed and partitioned executors. Static traversal,
exact unit loans and publication use that shared mechanism. Architecture-owned
direct, routed, pipeline and composite executors expose their actual retained
owners; native erasure forwards the operation without selecting a family or
reopening a source. The residency policy validates unit addresses and preserves
completion ownership, including rejected callbacks. Publication first asks the
policy to accept completed replacements; only then are pinned static handles
updated. The enclosing live transaction remains responsible for admission,
quiescence, budgets, peer agreement, caches and snapshots. A custom executor with
no retained parameter owner reports unavailable access.
MLX composition retains the architecture-supplied unit address separately from
the policy's local residency address. Parameter loans validate the former before
mapping to the latter, including nonzero pipeline offsets. Both resident and
bounded policies reject a foreign unit address before traversal or construction.

These methods are prerequisites for distributed parameter operations. The live
runtime now provides `PartitionParameterReadPlan` and `ParameterOperationCoordinator`.
The read plan assigns each selected source cell to one producer, including partial
replica overlaps and different permutation tilings. It rejects incomplete coverage
before native work and assembles exact payload counts; contractions sum disjoint
source contributions in F64 before one F32 result conversion.

Core retains local immutable edit authority and a separate common intent digest,
streaming bounded serialization rather than copying complete encoded edit plans.
Core also owns typed coordination outcomes and separate lifetime control accounting.
Runtime owns setup-bound attempts, ready-peer geometry agreement, source preparation,
payload transfer, completed delivery, publication and rollback decisions. Every
decision includes confirmation of its completed validation. Mandatory rejection
and rollback control is admitted from exact native setup facts and charged before
submission, separately from caller-limited parameter and payload work. Failed
local preparation needs no fabricated layout or payload ceiling; all ready peers
must agree those facts before source work. Failure, reset and snapshots refund
neither allowance. Abandonment, ambiguous transport, failed restoration or protocol
disagreement fence the retained owner without claiming physical native completion.
After an unwind, poisoned authority reports terminal failure while already charged
control usage remains readable; reading counters cannot reactivate the owner.

MLX retains that coordinator alongside its existing native communication authority
and supplies ordinary bounded word transport, exact rank/setup facts and native
storage estimates. Each constructed model receives a distinct runtime-issued
`ParameterModelIdentity` from the retained coordinator. Corresponding ranks pair
model constructions in the same order; operations compare the resulting model
identity as well as source, execution and parameter version. Reset, snapshot
restoration and destruction never recycle that identity. Registration performs
no native work or communication. Source reads and contractions share the same
native mechanisms
as ordinary public parameter operations. Architecture semantics and native tensor
handles do not enter the control protocol.

Public partition discovery exchanges bounded loaded metadata, without evaluating
parameter tensors. Runtime `PartitionParameterCatalog` validates actual sharing,
global/local geometry, per-operation capabilities and complete source coverage.
An empty shard cannot downgrade a nonempty owner's native mechanism. The model's
catalog-only binding permits source-resolution errors to reach shared admission;
it cannot authorize parameter reads or edits. Queries and projections use the
full retained model/source/execution/version binding and existing native loans.
`LoadedParameter::access()` distinguishes query, projection and replacement;
explicit flags override the legacy aggregate support field. Metadata exchange,
source preparation, conversion, native contraction, transport and host delivery
are charged before their work; counters survive rejected attempts.

Distributed overlay composition prepares all replacements and a fresh mutable
state before publishing any handle. Runtime supplies a local state exchange for
use under the parameter coordinator, without nesting ordinary control collectives.
The exchange retains the previous cache and prompt identity for rollback and does
not rewind commit epochs. After every peer confirms publication, native composition
invalidates old snapshots and advances the parameter version, including on ranks
that own no edited slot. A completed publication rejection restores both weights
and state; failed restoration fences the session. All actual aliases and replicas
participate. Packed affected parameters are promoted on every nonempty owner,
including shards outside the edit rectangle, so the selected input arithmetic
remains consistent. Distributed active provenance uses the common admitted edit
intent; rank-local admission authority remains distinct.

Affine and MXFP4 effective parameter access uses the same native dequantization
for matrices and grouped banks. Loaded support validates every leading group/row
extent and the exact packed feature width before native work. A floating
replacement passes through ordinary grouped multiplication even though the
declared packed format and original companions remain retained for restoration.
This dispatch depends on actual storage, without selecting a model family or
changing architecture-owned group equations.

Block-FP8 and native GGUF effective access also admits rank-three matrix banks.
The native adapter validates actual FP8 geometry or the exact GGUF byte extent,
retains singleton group axes after decoding, and reserves FP8 scale expansion
across every group and both padded matrix axes. Grouped GGUF host conversion is
charged for the complete affected bank. Floating replacements use ordinary
grouped multiplication; an FP8 bank therefore switches its input transform to
identity while active, matching the matrix overlay policy. Packed storage and
companions remain available for exact restoration. Gated, unary and selected
grouped operators share that storage dispatch. Quantized routing projections
likewise dispatch from the actual storage dtype, preserving their declared
input transformations and routing precision through edit and restoration.

Qwen GGUF admission retains each native matrix encoding before constructing
derived expert recipes and cold lowering requirements. Architecture-owned format
normalization maps separate gate/up and down checkpoint identities to their
canonical bank parameters, using the configured parameter root. Split gate/up
encodings must agree. A byte-producing bank recipe never inherits a metadata-only
dense default. The runtime continues to require exact agreement between retained
tasks and their lowering descriptors; diagnostics identify the differing facts.

`CaptureCheckpoint` retains a setup-scoped child identity seed and accounts for
its host storage. Fork installs that seed with a fresh capture owner and no first
epoch; shared preparation checks it against actual loaded artifact, execution,
setup and overlay facts. Restore keeps the parent's first epoch. Loaded observer
composition turns local setup failures into shared preparation rejection before
model work. Receipt byte allowances use actual fragment metadata and native
encoded-size estimates, bounded by the admitted ceiling; small selections do not
prepay an unrelated maximum-sized receipt. Deferred projection-input factory
credits use the same native bound as the actual FP8 implementation.

The partition collector reserves common coordination before optional capture work.
Runtime seals successful producer claims, skipped selections and consumed credits
into the same pre-forward agreement. Limit skips discard local producer authority
without refunds and publish typed records only after final commit. Native code
supplies the existing transport and resource facts; it does not choose rank-local
skip policy. This also applies to re-admitted controlled children.

Deferred tensor observations carry neural-owned `TensorElementType` and
`GeneratedTensorSource` facts. `eredu-nn` remains independent of `eredu-core`;
runtime translates those facts into core `GeneratedCaptureSource` metadata at
the architecture observation adapter. The prototype declares geometry only.
Native FP8 declares the actual F32 diagnostic output. Runtime validates declared
precision before transforming a generated value, and `CaptureBackend` can price
that source independently of the prototype's dtype. Empty selected outputs use
a shared reserved host constructor without creating a source tensor. Partition
receipts retain each empty producer's declared precision and reject disagreement.

Ordinary MLX observation composition carries the backend's structured `Error`
through array/tensor adapters. Its neural adapter uses `eredu_nn::Error::backend_source`
to retain the original cause across the portable execution driver. Capture policy
failures remain typed sources, and native exceptions retain their creation location;
bounded record diagnostics remain separate from this error chain. The public facade
continues to expose `eredu_core::BackendFailure`. Neither a capture error nor its
classification establishes safe completion: the existing shared rollback witness
and native submission/recovery owner still govern session reuse and lease release.

Routed provider, addressable-bank, indexed-movement and route-exchange contracts
require owned standard errors (`Error + Send + Sync + 'static`), as does
`LayeredArchitecture`. This lets portable adapters retain the cause of a failed
mechanism through resident, bounded and partitioned execution. Grouped operation
bridges return `eredu_nn::Error`; architecture-owned routed execution errors carry
a source-bearing variant alongside their typed plan and diagnostic variants.
Provider observation and family adapters preserve those sources instead of
formatting them into strings. Source-bearing errors do not implement equality:
callers match policy variants or inspect the source chain. Cold preparation and
submission-policy diagnostics have separate contracts; this does not turn a
provider error into evidence that native work completed or remote peers stopped.

Provider execution agreement is independent of component observation. Runtime
supplies source-preserving provider wrappers; architecture construction retains
the selected tensor group explicitly. Routed TP calls vote before their following
model collective. Exchanged expert work votes after received-route validation,
local bank execution and output validation, before reverse exchange: TP first,
then EP even when TP rejected the work. Shared pipeline waves additionally vote
on the exact session agreement group retained from the prepared execution plan.
That vote includes inactive stages before any participant advances to reverse
exchange or another tensor collective; stage-local votes alone cannot certify a
shared wave. Idle owners and inactive pipeline waves participate in the same
sequence. Expert-group manifests therefore require exact failure agreement as well as count gathering and variable exchange. Local failures
retain their original cause; healthy peers receive a typed rejection, carried by
`ReplicatedTextSessionError::Partition` through pipeline and session adapters.
These votes do not replace native completion ownership or establish recovery from a failed
communication operation.

Nemotron-H partition adapters now reuse the block's ordinary component hooks and
the shared readout instrumentation for local, tensor-parallel and pipeline
execution. Effective embedding values remain in the typed boundary payload used
by downstream target/prediction stages. Discovery joins the Mamba operator to its
canonical `mamba` parameter namespace while retaining the logical mixer observation
identity; a checkpoint alias does not define invocation ownership.
The routed partition adapter declares ordinary and TP unit hooks separately from
provider-unit hooks. Its TP block driver applies normalized-input, attention-channel,
ReLU²-unit and residual interventions to the values consumed by the existing
parallel operators. Routed observations retain provider completion and reduction
ordering. Disabled instrumentation uses the same block driver without extra tensors.

Mamba convolution parameters follow the same semantic segmentation as their input:
value channels, input-state channels and output-state channels. TP selects its group
interval within each segment for both convolution weights and bias. A contiguous
slice of the concatenated channel axis would pair kernels with different projected
channels. This placement belongs to Nemotron's neutral parameter plan and is consumed
unchanged by materialization, effective-parameter queries and overlays.

`CapturedStep.outcome` carries the shared model transaction result: committed,
aborted, or untracked for low-level/legacy records without transaction evidence.
Runtime supplies it when draining the ledger; backends do not infer it from error
text. Individual observations retain their original meaning, so an observation
completed before a later provider failure can remain present in an aborted batch.
Aborted batches retain charged usage and cannot establish a snapshot boundary or
safe native reuse. A committed forward also does not imply successful sampling.


Partition activation interventions now share `CaptureSession` and
`PartitionCaptureObserver` with ordinary bounded capture. Runtime owns move-only
operation authority, local admission/epoch checks, globally prepaid projection and
dependency work, exact coordinate/intent agreement, active-member failure votes,
and completed world outcome delivery. Only final shared commit publishes `Applied`
and per-operation evidence. Independent invocation groups exclude inactive pipeline
ranks while retaining nonexporting replicas and empty local selections.

Architecture layouts supply actual activation members and cold region bounds.
Their coordinate identities join common agreement; equal tensor shapes do not
establish equal placement. Native MLX composition selects retained support facts
and supplies ordinary activation arithmetic, source completion and transport.
No family equations or semantic dispatch move into the backend. Evidence uses
private derived global capture admissions keyed by operation and before/after
ordinal; it never rewrites the parent's capture plan or aliases its selection keys.
Projection host copies, dependency completion, group votes, receipt transport and
evidence delivery consume nonrefundable credits before work. These remain logical
storage bounds, including conservative decoder allowances, not allocator guarantees.

Scalar component groups declare `write_output` and `output` observation relationships
for the complete affine and normalized contributions. The architecture partition
layout derives their hidden-axis replica placement from the same logical write
parameter that owns the component invocation; shared source storage does not
transfer execution ownership. `ComponentWritePartition::Complete` observes after
projection reduction; unit/channel observations retain sharded scalar coordinates.
Component schema 3 also declares `TensorParallelSum`: every TP rank supplies an
additive full-hidden-width term before an ordinary, possibly fused reduction.
Architecture layouts retain every TP term, remove only true replicas, and require
complete TP invocation membership. Each replica sum designates TP rank zero for
Add offsets and replacements; masks and scaling affect all terms.

Partition receipt/evidence schema 3 binds `PartitionCaptureCombination` to the
retained authority and wire records. Runtime's `SumF64ToF32` collects selected raw
native F32 terms, sums in canonical world-rank order with compensated F64 arithmetic,
then rounds to F32 before summary/histogram transformation. Preview collects only
the selected prefix. It prepays raw export, host assembly, transport and delivery;
the native backend supplies ordinary collectors and completion. This host assembly
does not imply that a separate complete native tensor existed, nor bitwise equality
with a native reduction. The neutral mechanism and V3 shared-write declarations are
implemented. Mixed V3 TP execution hooks pass native F32 SafeTensors/GGUF and
load-time affine 4-bit/group-32 acceptance across TP/PP/EP and all three ordinary
weight residencies.

LFM2 selects the existing neutral arithmetic contracts for its published reduced-
precision equations: rotary products round in the input dtype, attention rounds
scores and probabilities, and sigmoid routing retains input precision through
coefficient normalization. F32 expert correction biases affect selection only.
Expert mixtures accumulate in ascending expert identity with a rounding boundary
at each addition. Both replicated and placement-resolved construction retain
these choices; reusable native operators implement them without family dispatch.

The shared native learned RMS operator preserves its input-precision rounding
boundary before multiplying the gain. On Metal, matching half-precision inputs
and gains use the existing float32 row reduction so reduction-order differences
near a rounding midpoint do not change downstream component experiments. The
parameterized operator, learned-offset form and direct weighted operation share
that mechanism; mixed-precision gains retain the native promotion behavior.


The V3 provider-aware TP traversal now invokes the additive shared-write hooks and
preserves its ordinary fused reduction through a shared helper. A nonzero neutral
test covers complete/partial provider results and literal post-bias across five
shared-branch experiments (15 cases), with exactly one reduction. All eleven V3
component numerical tests pass (2.45 seconds). Nine partition-coordinate tests also
pass, including TP2/EP2 term ownership and one offset owner per replica sum. Target-only mixed-TP traversal hooks are now enabled after neutral execution
conformance. Native F32 SafeTensors/GGUF and load-time affine public
capture/intervention acceptance passes all 63 TP/PP/EP and ordinary-residency cases.


DeepSeek expert geometry expands the retained uniform partition domain into scalar
FFN coordinates. Encoding expansion may group several neurons into one partition
unit so weights, packed input columns and scale/bias companions share legal cuts.
The architecture validates divisibility and exact ranges before deriving local
routed/shared ownership; native code receives those scalar ranges. A partition
unit is therefore not assumed to be one neuron. Nonuniform chunk layouts need
their own explicit mapping. Mixed affine V3 native acceptance passes all 21
TP/PP/EP and ordinary-residency cases.

Routed partition construction now retains an optional typed source architecture
and its exact local layout alongside the selected target. The runtime handoff
revalidates source parameter addresses and layout before forwarding both to the
existing materialization driver. V3 constructs this source from retained encoding
facts, including expert ownership, target-unit range and compressed-state geometry.
Native local and pipeline adapters forward that authority without choosing family
semantics or reopening artifacts. Mixed V3 load-time affine native acceptance
passes for this path. Other routed families provide their corresponding exact
source construction for load-time transforms; the component guide records
native family/format acceptance.

DeepSeek's shared router construction consumes the exact retained matrix format.
V3/V4 routers default to dense when no per-parameter format is declared; selected
load-time transforms or encoded sources supply an explicit override. The same
rule determines parallel weight and companion geometry. Backend group selectors
receive the ordinary neutral linear-format declaration.

Exact native transform materialization validates the admitted whole-source recipe
before applying retained rank placements. It then checks the selected local shape
and floating slot category against the source module. Unloaded source slots
use construction-default dtypes; the admitted recipe supplies F16/BF16/F32
precision for transformation and affine companions. Global derived-output metadata and
local executable geometry remain separate authorities; comparing the former to an
already sharded recipe would reject valid TP/EP materialization.

Independently cached routed parameters retain all selected output destinations,
including generated quantization companions absent from the original source
catalog. Architecture construction forwards those exact member-task identities
as addressable ownership. Ordinary module binding still rejects any uncovered
destination; it does not infer scale/bias names or silently load bank-owned copies.

Exact replicated-text affine transforms preserve the admitted floating source precision for
generated scales and biases, matching independent-bank materialization and neutral
selected-byte accounting. Unloaded native destination dtypes describe placeholders
and cannot silently request an F16 cast. Reusable prompt-cache schema 9 rejects
older persisted state because its metadata cannot distinguish these numerical
policies. Reconstruction uses exact token-ID replay.

V3 embedded prediction primitives share fusion and head execution between ordinary
and observed paths. Architecture-owned hooks surround normalized embedding/hidden
inputs, the fusion result and the prediction head; the existing V3 block supplies
MLA, routed and shared component hooks. Editing the prediction readout residual
precedes head normalization and changes both returned hidden state and draft logits.
The older `mtp.{depth}.output` point remains after head computation. Low-level
ordinary/provider/TP traversal forwards these hooks. Prediction scopes are declared
separately; public speculative admission still requires phase-aware integration.
Their loaded component capability remains unadvertised until that integration is complete.

The independently cached mixed V3 component matrix passes all 63 native CPU Ring
cases across F32 SafeTensors/GGUF and affine loading. Cache telemetry distinguishes
an EP owner with no routed work from a missing bank: exact ownership remains, idle
owners construct no compact banks, and every routed pipeline stage has an active
owner. The numerical/query/overlay checks remain unchanged.

Architecture descriptor schema 5 and component schema 4 describe separately
invoked component scopes. V3 prediction depths own their decoder groups, fused
residual base and `ComponentReadoutEquation`; primary target component lists retain
their existing meaning. `ComponentResidualBase::LinearFusion` declares ordered
normalized sources, canonical effective fusion parameters and actual projection
input evidence. Its supplied hidden input is invocation-local: prefill/replay may
supply target hidden state, while a proposal may supply the preceding prediction
output. It is not a second token embedding base. Prediction nodes, parameter joins
and scope membership are architecture policy. The common readout equation lives
in core; primary readout JSON keeps its flattened fields.

Runtime support additionally requires the selected call path's explicit
`prediction_inspection` fact before consulting partition placement. A declared
scope, native collector or ordinary target hook cannot establish that fact.
Ordinary target and cold declarations alone leave it unverified. Loaded
prediction construction supplies the complete selected-hook fact before public
speculative capture admission.

The typed prediction extension now accepts optional internal observers for V3
prefill, proposal and retained-input replay. Architecture-owned prediction
operations forward those observers through the target's ordinary embedding and
selected serial/TP equations, preserving outer unit timing. Absence calls the
existing unobserved methods and does not enable diagnostic generation. The
enclosing speculative driver continues to own lane checkpoints, failure recovery,
phase attribution and completion. An extension must report its complete selected hooks before accepting an
explicit internal observer. Public bounded collection additionally requires the
phase-aware admitted authority; the component-analysis guide records the
implemented family and encoding matrix.


The architecture-owned embedded strategy now accepts a separately installed
`SpeculativeActivationObserver`. The common driver passes it to target prefill,
prediction seed, depth-specific proposals, target verification, prediction replay
and target replay. Runtime brackets each attempted invocation with its actual
physical sequence width; this width is not a committed prediction index. Completion
stages records but does not commit tokens or replace native completion ownership.
An infallible finish callback also runs after failed admission and unwinding.
`with_activation_observer` lends one observer for the entire forward, allowing
move-only producer work and coordination to remain on the stack across callbacks.
The default borrows the existing owner without allocating. The operation receives
only `ActivationObserver`; admission and finalization remain outside it. Borrowed
work is dropped before completion or failure finalization, including unwinding.
A live five-rank fixture exercises this lifetime through actual partition edits,
receipt assembly and final commit. The speculative collector uses a partition provider and shared auxiliary
transactions, bound to the selected prediction layouts and retained native
transport.

Target passes reuse the existing observed replicated-session transaction, retaining
its exact target capture, output publication, completion agreement and rollback.
Prediction calls reuse the typed extension operation. Runtime's `ObserverErrorBridge`
forwards generated reservations, routing/unit hooks and transaction callbacks across
neural/native error types and returns the original observer failure. Absence still
uses ordinary methods. MLX only binds the complete observer set through its session;
it adds no phase or family dispatch. Complete hook selection is supplied by each supported materialized extension.
Public capture budgets, phase/frontier attribution, effective query/overlay
sources and distributed producer bindings are verified through the shared
collector; the component-analysis guide records family-specific acceptance.


Core now separates `CaptureInvocationShape` (exact batch, physical sequence and
optional actual context) from prediction schedule coordinates. Capture and
intervention plans can be admitted with `CaptureInvocationBounds`; their immutable
identity includes this mode and bounds, while ordinary admission digests retain
the existing encoding. Unknown context is never inferred from a generated position.

Runtime's `CaptureSession::begin_invocation` checks actual bounds and active slices,
then uses the existing capture/intervention ledger, generated-source driver, sparse
receipts and native transformation contracts. Explicit masks select applicability
within admitted entries; absent scopes receive `NotInvoked` capture outcomes and
inactive intervention outcomes. Every invocation reserves its geometry envelope;
repeated prediction coordinates spend cumulative resources. Checkpoint restoration
never refunds these charges, and child re-admission preserves the geometry mode.

Partition capture now accepts independently admitted invocation geometry. Core's
partition receipt/evidence schema 3 retains the optional exact batch, sequence and
context in `PartitionCaptureContext`. Runtime binds it to producer admission,
fragment validation, all-rank coordination and final assembly; a different context
is rejected even when it does not change the selected tensor's shape. Ordinary
plans require absent invocation geometry, while invocation plans require explicit
geometry within their original bounds. Deferred factories cannot run before this
validation.

Architecture-owned `ComponentPartitionLayouts` projects dense and routed placement
using that exact geometry. Runtime's `PartitionCaptureObserver` uses the active
capture/intervention applicability masks, applies edits to actual local values,
and retains normal source/hook votes, receipt completion and final commit.
Nonexporting replicas and empty local shards still participate. Invocation metadata
is reserved for every rank and restore does not refund cumulative work.

The shared speculative collector retains these partition observer transactions
across internal target/prediction callbacks and binds the selected prediction
producers and native transport. Public distributed admission requires those
loaded bindings; the neutral invocation protocol alone grants no authority.

The shared speculative scheduler now supplies `SpeculativeActivationOrigin` to
internal observers: request identity, committed frontier, logical prediction
coordinate, an exact-prefix digest and optimistic-work status. The operation
scope clears on errors and unwinding. Architecture executor erasure forwards this
scope and drains bounded records and original portable capture failures. Scope
applicability comes from declared node ancestry and prediction depths, not native
checkpoint-name parsing. Runtime `SpeculativeCaptureObserver` borrows ordinary
capture/intervention mechanisms through `CaptureBackendProvider`; it retains one
request's existing ledger and charges each queued invocation envelope before work.
A separate loaded speculative discovery and admission contract now binds this
collector. Ordinary prediction inspection stays disabled for ordinary generation.
Partition producer agreement uses the selected layout and retained transport.
Explicit CPU/Metal collector binding and distributed execution have native
conformance coverage.

Native session adapters now drain the architecture observer set's already charged
speculative records and neutral failures after the shared executor scope returns.
The collector's error propagation signal may cross a native error domain, but its
first original failure remains owned as `SpeculativeControlError::Capture` or a
neutral `BackendFailure` retaining the native source chain. Sparse callback errors
retain their shared original neural source rather than discarding it into text.


`SpeculativeActivationPlan` admits capture and optional intervention plans together
under independent invocation bounds. Core owns its immutable authority, exact node
bindings and source/execution/session identity. Architecture projection resolves
node ancestry and discharges only the prediction-execution condition from a sealed
materialized extension's complete selected-hook fact. Media and partition gates
remain separate. The loaded execution identity includes the active overlay.

The facade prepares and validates this authority; `ControlledSpeculativeOptions`
installs it before lane prefill. Architecture executor erasure forwards installation
and restores prior observer ownership when its borrowed scope ends. Reinstallation
after execution or within the same scope is rejected. The MLX adapter supplies an
owned stream and the existing native capture/intervention estimator and operations;
it makes no family or scope decisions. Failed/cancelled host evidence can be drained
one invocation at a time through `take_activation_evidence`, under the existing
monotone delivery sequence and cumulative JSON budget. This does not recover native
work or clear session failure.

Prepared discovery keeps the ordinary target catalog and the selected prediction
catalog separately. The latter is projected from the original admitted complete
architecture retained with the source graph, before its target-only projection.
Neither discovery nor the native binding reopens artifacts or reconstructs a family
configuration to recover prediction points.

Embedded speculative durable copies use the same core snapshot and runtime
reservation machinery as external drafting. Core snapshot/restore contracts return
`SpeculativeControlError`, preserving typed policy failures and original native
sources through executor erasure. The architecture driver enumerates the complete
target state, canonical prediction state, independent prediction seed, capture tensor
and host identities; transactional rollback clones do not establish isolation.
`PredictionExtensionMaterializer` supplies bounded native copies of the declared
state profiles, while architecture code owns their membership and replica count.
MLX settles source and destination copies and supplies conservative storage bounds.
The driver builds every replacement before replacing the live cache, leaving the
saved checkpoint reusable. Internal activation authority remains scoped to the
collector; restore and branch exchange never rewind its invocation counter or
capture ledger. Runtime owns opaque collector checkpoints and move-only prepared
authority restoration. Architecture checkpoints combine these with complete model
state, prepare authority first, copy every native replacement, then commit both
without a further fallible operation. Transaction rollback checkpoints omit this
durable authority component.

`ControlledSpeculativeSession::readmit_activation_interventions` validates against
the facade-retained loaded speculative discovery at a canonical boundary. Runtime
preflights replacement operations with the existing estimator and cumulative usage,
charges their host storage, and changes only prospective edits and their scopes.
Capture selections, invocation bounds and allowances remain fixed from run creation.
Saved collector authority includes exact admitted identity and scope bindings;
serial branch exchange restores it alongside model and sampler state. Backend
adapters forward this operation through typed executor erasure without interpreting
an architecture family or intervention target.

Prediction parameter access uses the same loaded-parameter contracts as ordinary
execution. `PredictionModuleVisitor` traverses each architecture-owned materialized
module in stable physical-owner order, distinct from proposal depth or lane-state
position. The architecture enumerates sequential, fused and grouped modules;
backends do not reconstruct their family membership. Shared target embeddings and
heads remain with the target owner and are borrowed by prediction execution.

MLX retains exact prepared binding metadata and materialization tasks beside each
prediction module. `PreparedParameterLocation::Prediction` routes bounded queries
to that actual retained module, including its selected encodings and companions.
The loaded catalog combines target, bank and prediction ownership without reopening
sources. Complete replacement tensors are prepared by the existing parameter
transaction. After target/bank publication succeeds, prediction slots receive
infallible handle replacements under the same exclusive session; removal follows
the same path. Architecture parameter identities and aliases govern these moves.

Pooling-attention durable copies include local keys, every pending value/gate
window, pooled outputs and both overlap buffers. Native copies compact strided
views while preserving frontiers and compression ratios. A complete target copies
its shared paging catalog once into an independent namespace in the same bounded
pool, then binds each copied local tail to that manager. Inconsistent manager
ownership is rejected before copying. Prediction-profile copies use the same
native mechanism and the architecture's existing complete replica enumeration.
Storage bounds include retained array views, sealed blocks, catalog growth and
the native one-channel key-only persistence sentinel. Completion and atomic live
state replacement stay with the existing controlled-session copy driver.
Pooling transaction checkpoints retain immutable array views with their original
strides. Array replacement on advancement preserves those checkpoint values.
Native state forks and resident rollback copies compact logical views on the
caller's stream; raw backing-memory copies cannot preserve an overlap slice's
logical element order. This is distinct from the independently budgeted durable
copy of every array and paging namespace.
DSpark's architecture-owned bidirectional proposal mask is broadcast over the
actual keys returned by pooling attention. Multi-token append may return bounded
past keys plus the whole submitted block; the mask cannot infer that extent by
clipping the absolute frontier to the single-token window. Ordinary causal masks
retain their existing timing and eligibility semantics.

Paged sliding-cache fetch preserves the complete submitted span independently of
post-append retention. It gathers the bounded old history before storage eviction,
returns that history plus every submitted key, and applies the existing final-window
retention to persistent state. Ordinary key/value and key-only storage share this
mechanism, normalization, validation and append rollback. A long prefill therefore
uses the same logical attention extent as resident storage even when its earliest
keys are no longer retained for the next call.

Blockwise paged attention likewise retains the current query span until native
attention completes, then discards history outside the final window. Both arithmetic
passes consume the retained blocks before that eviction. Failed attention keeps its
charged storage with the existing completion/teardown owner; an early error does not
permit discarding potentially live native inputs.

Observed auxiliary prediction phases use `ReplicatedTextSession::with_prediction_observation`
for the same observer preparation, target-state recovery, completion agreement,
receipt delivery and final commit as ordinary forwards. Architecture composition
wraps the entire prediction prefill, proposal or replay invocation; one prefill
may execute several prediction depths under that transaction. Each typed operation
also agrees checkpoint success before entering its own execution collectives.

`MaterializedPredictionExecutor::complete_state` enumerates exact retained lane
values and actual output dependencies. Sequential, pooling and model-state profiles
reuse neutral runtime retention contracts. `PredictionExtensionMaterializer`
settles that iterator using native completion and deferred validation resources;
it does not interpret a family or rebuild state geometry. Observed proposals
retain their actual logits and hidden output until completion. Prefill and replay
settle retained caches without evaluating discarded prediction readouts. An absent
observer bypasses the auxiliary observation transaction and dependency enumeration.

The native error handoff accepts `BackendFailure` with its owned source.
`safemlx::Exception::from_source` preserves that chain through the existing native
error domain; it introduces no new unsafe boundary. Diagnostic-only contract
messages continue to use the existing text adapter. Public distributed internal
activation admission still requires selected prediction producer layouts and a
bound partition collector/transport; this lifecycle alone does not enable it.

`CaptureBackendProvider` lends one ordinary observer for an entire speculative
phase. Its default implementation keeps the local path allocation-free;
`PartitionCaptureBackendProvider` instead retains the selected layout and transport
owners and lends the existing partition observer. It binds the communication
identity before invocation admission, so metadata reservations include every rank.
Native composition supplies primitives, estimates and source-preserving error
conversion. Architecture ownership remains in the supplied layout. Routed neural
callbacks retain typed admission errors and native sources through the same outer
speculative owner. The provider does not select prediction ownership; public
partitioned speculative admission still requires the prepared prediction layouts
and their native factory binding.

Prepared prediction modules retain their exact `LocalModelLayout` and global
parameter declaration through shared `Arc` owners. The total architecture
construction driver publishes those owners into the prepared source graph's
once-only discovery slot after successful materialization. Cold discovery has
no constructed placement. Backend materialization consumes the same local layout;
producer discovery projects peers through the same tensor-axis compiler.
Prediction units retain resident expert banks and replicas over non-tensor axes,
independently of the target's pipeline and expert ownership.

`speculative_component_partition_layouts` combines the target with each declared
prediction scope. Scalar coordinates reuse ordinary component proofs; fusion
inputs, residuals and prediction heads retain their own invocation semantics.
Prediction score heads are not target publication. Native composition lends a
`SpeculativePartitionBinding` through the existing execution context to the shared
collector factory. Layout compilation and content identity remain deferred until
discovery or nonempty capture admission; absent instrumentation evaluates no
additional component values. The ordinary session guard and shared speculative
control driver retain completion and state ownership.

A prepared resident prediction bank also retains its compiled scalar coordinates
in the architecture module. Its sparse provider scope wraps the actual local
input and every native chunk, and settles source and failure votes before the
following tensor reduction. This scope is separate from target expert-exchange
ownership; the target's existing providers retain their own invocation scopes.

Auxiliary predictors can declare target static parameters required on every
pipeline rank. `ReplicatedTextRequirements::replicated_static_roles` retains those
storage dependencies through selection. `PartitionOwnership` distinguishes them
from ordinary invocation roles: materialization, effective parameter operations
and residency sizing include the replicas, while target observation ownership
remains unchanged. V3 predictors declare their borrowed target embedding this
way. Disabling the predictor removes its auxiliary requirements before selection.

Loaded distributed parameter discovery joins ordinary target ownership with the
placement retained by successful prediction construction. A target parameter,
including a shared embedding replica, keeps its ordinary storage declaration.
A separately owned prediction parameter resolves through its retained task and
architecture declaration. Both paths use the same bounded alias/companion
resolution and single-group coordinate compiler. Prediction groups use their
actual tensor-only placement, retaining the global participant rank and replicas
over pipeline and expert axes. Metadata reservation precedes allocation; neither
path opens sources, selects execution again, or allocates native tensors. Native
catalog assembly, projections and atomic overlays consume those neutral maps.

Prepared prediction units can retain a separate floating source layout for
load-time transforms. The architecture applies selected auxiliary formats to the
executable module and compiles source and destination coordinates separately.
Native prediction materialization consumes those modules and exact tasks through
the ordinary bounded quantizer and local binding builder. Conversion selects this
rank's source geometry before reading tiles, and binding does not shard a locally
transformed result again. Completed auxiliary conversion reports join ordinary
session telemetry; selected resource authority is unchanged. The V3 affine
4-bit/group-32 and MXFP4 CPU Ring component/parameter matrices verify this
construction across all seven parallel-axis combinations and three target residency modes;
published FP8 and other supported prediction encodings have their own native
acceptance matrices in the component-analysis guide.

The prepared source graph preserves the selected drafting mode in its retained
target architecture. Disabled and external drafting remove the additive prediction
declaration before construction requirements are derived, matching cold selection.
The full admitted artifact remains available as provenance; this does not authorize
auxiliary materialization. Neutral coverage compares the requirements retained by
selection and source preparation across all four drafting modes.

Component schema five can declare an optional grouped linear stage before the
final write matrix. The neutral `ComponentGroup::write_column` joins one scalar
to exact regions of actual loaded factors, preserving loaded support and alias
facts. It allocates only the small region descriptions and grants no execution
authority; ordinary bounded parameter operations perform reads and contractions.
Architecture declarations own grouping, latent dimensions and the two stage
observations. A matrix product must not conceal a selected projection's input
rounding: the grouped output and final multiplication input are distinct evidence.
Existing direct groups omit this field. V4 discovery and its shared execution hooks consume this contract; a declaration
alone still grants no native observation authority.

Grouped component placement expands complete first-projection row groups into
scalar channel coordinates. Intermediate output and final-input observations use
the latent axis, checked against the final matrix's column ownership. Equal local
orders are accepted across range/index representations; split or reordered rows
inside a group are rejected. Repeated invocations retain separate paths while
sharing each factor's declared source identity. These projections are architecture
contracts and do not select a device or materialize a factor product.

Architecture descriptor schema six carries these grouped write semantics; readers
must check that wire version before interpreting final write columns. The first
factor's actual input retains `[batch, group, sequence, channels_per_group]`
geometry and partitions on complete groups. `GroupedNeuralBackend` supplies a
projection-input observer with the same deferred admission contract as ordinary
linear projections. MLX routes it through its existing selected linear mechanism,
including FP8 input rounding, before selecting the grouped diagonal outputs.
Architecture instrumentation forwards that evidence without an eager diagnostic
reshape. An absent observer creates no additional paths, views or evidence tensors.

V4 attention instrumentation now lives in its portable attention driver: the
mutable channel boundary follows inverse rotary aggregation, before either output
factor. Its ordinary path keeps the existing tensor operations when no observer
is installed. V4 block serial, provider and tensor-parallel entry points share one
portable driver; native collectives remain injected operations. It reports actual
normalization inputs and already-computed hyper-connection coefficients, preserving
hash-selected routing in observed resident TP execution. Discovery owns channel,
latent and stream geometry. Target and prediction hyper-stream readouts are architecture-owned, with
separate sequential and fused prediction equations and invocation scopes.


`ComponentStreamResidual` declares target residual broadcast, ordered stream
cycles, and final learned collapse. Architecture parameter ownership determines
cycle producers; retained input/output ownership determines the two outer
boundaries. Placement checks the full declared geometry and pairs original and
effective observations before capture admission. Complete stream values remain
replicated even when their sublayer writes required a native parallel sum.

`eredu-nn::TensorValueObserver` generalizes the existing projection-input observer
name without changing its deferred-generation contract. `HyperHeadOperator`
accepts an optional coefficient observer. MLX borrows its already-computed
coefficient tensor before the ordinary sum, preserving the original neutral
failure and source if admission rejects it. Family equations, coefficient timing,
intervention semantics and score reconstruction stay in the architecture and
consumer layers. Native code adds no V4-specific readout policy.

An architecture can declare that its observed unit executor owns both unit
boundaries, including interventions and effective evidence. Shared layered and
partitioned traversals then omit their outer delivery of those same boundaries.
V4 uses this contract because output intervention precedes its retained target
capture. Custom executors without internal observation retain the ordinary outer
hooks. The contract prevents duplicate records and double application of edits
without moving family timing into runtime or backend policy.

A separately invoked residual can declare `ComponentResidualBase::ProjectedSum`:
each term retains its own normalization, effective projection, actual multiplication
input and optional explicit axis broadcast before addition. This represents
separate projections without inventing a concatenated parameter matrix. The
architecture declares these terms and the runtime continues to own bounded
parameter authority; the declaration grants no materialization or editing rights.

Sequential V4 prediction discovery separates checkpoint parameter roots from
observation invocation paths. The predictor owns its outer input/output seams;
the nested decoder owns fused-stream seams. Prepared prediction construction
retains the selected executable formats, original source layout when conversion
is required, and routed-unit coordinates from the neutral partition plan.
Complete stream observations remain replicated across prediction execution
replicas; component columns retain their tensor partition. The shared prediction
driver supplies prefill, proposal and replay hooks to the existing controlled
speculative executor. DSpark uses its own context/fused phase declarations and
shared instrumented context/proposal equations, described below.

Prepared V4 prediction replicas consume the target embedding and vocabulary head.
The neutral selected requirements therefore replicate both static roles across
pipeline stages while retaining each tensor-parallel shard. This applies to both
sequential prediction and DSpark. Role replication drives ordinary materialization,
parameter ownership and residency accounting; prediction does not use an unbound
placeholder head on stages that do not publish target logits. Disabling the
prediction extension removes these extra consumers.

V4's architecture-owned checkpoint specifications also define which logical
matrices feed configurable projection operators. Load-time quantization consumes
that declaration. Rank-two integer routing tables, additive compressor positions,
dense hyper-connection coefficients and dense lookup embeddings retain their
native storage. Their rank alone does not grant a packed-linear transform, and
the backend does not infer their family semantics from parameter names.

Partitioned V4 transforms retain a separate source-format architecture and local
parameter layout. Neutral preparation checks source/target parameter space, unit
addresses, ownership and pooling-state geometry before handing both models to
ordinary backend materialization. The selected projection formats do not replace
the original source geometry used to read and convert each rank's parameters.


Fused component discovery keeps physical execution groups, pinned parameter roles
and invocation applicability separate. Core owns versioned direct residual-source
and dynamic score-write contracts; architectures declare and validate the actual
source expansion, score broadcasting and producer placement. Prepared prediction
layouts consume these declarations and retain complete context/proposal replicas
alongside tensor-sharded component columns. A context-cache invocation has no
invented FFN or score head. Runtime phase filtering uses the admitted context or
fused scope and preserves completion, delivery and cumulative accounting. The
backend neither chooses the semantic phase nor supplies a family-specific score
equation. DSpark declarations alone do not establish complete native hook support.


The materialized fused-prediction contract accepts a caller-owned temporary
lane for observed proposals. The shared speculative driver retains that lane
and actual logits through ordinary auxiliary completion and final capture
publication, then discards the proposal state. Accepted context is unchanged.
Context preparation and replay use their existing committed prediction lane;
architecture hooks observe and edit the inputs actually used to update it.
Separate module and observer borrows span each complete native invocation.

The prediction extension declares the physical sequence extent of cache prefill.
Sequential predictors consume shifted hidden/next-token pairs; DSpark consumes
the entire accepted target capture, including a single-token prefix. The shared
phase wrapper uses that neutral fact for admission and runtime shape checks.
Native collectors do not infer this distinction from a family or path name.

Transient native sliding-cache checkpoints retain the sealed history required by
the saved frontier. Their lightweight catalog guards prevent semantic discard
while allowing ordinary device/host/disk movement; retained blocks stay charged
to the existing manager and process pool. Restoring a frontier does not transfer
the guard into the resumed live cache, so releasing the saved checkpoint permits
normal sliding discard. Durable snapshots keep their independent namespaces.
The neutral pooling-cache checkpoint contract returns a typed result because
history-retention admission can fail before speculative work. Native visible-range
fetches also reject incomplete history before returning an attention tensor.


Component schema 7 and architecture descriptor schema 8 add a measured scalar
output gate to a component group. The neutral declaration identifies its affine
read, actual multiplication input, activation, and original/effective gate. Its
position is after the group's projected write and optional output normalization,
before the complete contribution hook. Architecture code owns the sigmoid and
multiplication order; backends continue to provide ordinary linear and elementwise
mechanisms. Qwen shared experts use this contract because their scalar gate follows
the down projection. Disabled observation retains the same order without extra
component tensors. This declaration does not itself establish loaded or partitioned
capture support; placement and execution acceptance remain separate.


Qwen hybrid component observation follows the architecture-owned serial and
parallel block drivers. The parallel shared-expert path retains the ordinary
complete/partial routed reduction order and emits its actual gated local write.
The text layered adapter forwards embedding, internal-unit, partition and readout
hooks; the shared static-module projection helper preserves tied/untied vocabulary
projection and observes the actual multiplication input. Native adapters still
supply mechanisms and collectors. Prediction-unit primitives do not by themselves
enable prepared prediction or conditional-media component capabilities.

Prediction-free Qwen hybrid text partitions now use the neutral dense/routed
prepared constructors for TP, PP and EP. Cold state selection derives heterogeneous
attention and recurrent geometry without a native context. Construction retains the
global parameter description separately from local modules, binds selected local
expert-bank specifications while retaining global router decisions, and maps target
unit addresses into the selected PP cache slice. Source and transformed parameter
layouts are checked before the ordinary family-independent materialization visitor
runs. These paths use the existing bounded residency and communication handoffs;
prepared conformance and native validation are recorded in the component guide.

Qwen recurrent convolution placement follows the concatenated Q/K/V projection:
each TP rank receives the corresponding slice of all three channel segments.
The ordinary linear-attention equations and native convolution mechanism consume
this architecture-declared layout; they do not reinterpret contiguous checkpoint
rows as local head ownership.

Conditional Qwen target observations use the same text blocks and final projection
as ordinary execution. Input evidence is taken from the assembled text/media value
at the first target block, after execution-group assembly. Each DeepStack addition
uses one shared serial/parallel driver and exposes the actual contribution after
placement at decoder token positions. Discovery links these contributions to their
target-layer invocation and includes them as whole terms in the readout equation.
The token embedding parameter describes the lookup branch; the observed assembled
input is the primary residual base. Numerical and native partition acceptance for these conditional hooks is
recorded in the component-analysis guide.


Composite partition strategies retain the observer through their selected direct
or routed call and through the final projection on its output owner. Text and
composite routed execution share provider agreement, expert transport and tensor
parallel dispatch, with an optional observer over the same driver. Capability
facts combine the retained architecture hooks with the actual selected strategy;
the composite input adapter forwards serial, parallel, routed and final-partition
hooks instead of inheriting ordinary fallback calls. Scalar composite conformance
uses the shared selected-residency preparation driver, so bounded cases acquire
units through the same portable policy contract as native binding.


Pinned parameters may declare exact unit consumers separately from their static
binding role. Runtime validates those consumers against the canonical execution
graph, selects one copy on each owning partition and includes that storage in its
ordinary selected-task accounting. They remain pinned modules, rather than unit
payloads. Shared Qwen vision declarations use this contract for the final merger
on the last vision unit and each DeepStack merger on its configured vision unit;
patch and position modules keep the ingress role. This fixes source-bound pipeline
execution without giving a backend ownership of vision placement or replicating
all static weights on every stage. Numerical and native acceptance is recorded
in the component-analysis guide.


Conditional Qwen construction retains the selected expert-bank specifications
alongside its text and vision geometry. Both ordinary composite and embedded
prediction target constructors pass those local bank specifications to the shared
block factory; the router keeps global expert identities. EP source binding then
uses the admitted local bank shape instead of attempting to fill a global bank
with a local payload. This follows the same architecture-owned construction used
by the text-only target path.

Controlled capture prefill consumes the ordinary prepared-input admission. The
architecture supplies the assembled decoder geometry (including media positions);
`eredu-core::capture::CaptureRequestShape::validate_prefill` performs the exact
comparison. Native erased adapters carry this optional request geometry into the
existing admission/result path, before any model forward, so peer rejection and
state-preservation follow the ordinary distributed transaction. They neither
reconstruct media-token policy nor admit all media based on collector support.

Completed composite MLX sessions expose the same retained parameter-slot, bounded
loan, replacement-publication and reset-state operations as completed text
sessions. Their adapters delegate to the shared neutral session and residency
policy, and include selected prediction modules and independently cached bank
slots. Native binding retains these handles and prepared materialization tasks;
it does not reopen source artifacts or infer component ownership from names.

Conditional observation support remains a discovery fact owned by the portable
runtime. Core capture and intervention admission accept those sites without
claiming that the input or invocation condition holds. Runtime capture emits
`Missing` for absent scheduled sites; scheduled interventions must finish before
commitment, so their absence fails through the existing transactional driver.
No backend parses condition descriptions or promotes them to unconditional
support. Unsupported and unverified phases remain inadmissible.

Prepared-input accounting now goes through the retained executable adapter.
Composite architectures project their ordinary whole-request admitted part plans
into neutral accounting plans; the backend adds native payload/metadata bytes to
those architecture-owned workspace estimates. Processor selection checks are
shared with ordinary preparation. Counting neither executes the model nor
reconstructs a family-specific plan, and does not compute a prompt-content
fingerprint.

Prepared composite routed execution retains scalar unit coordinates derived
from the original selected bank, its localized plan, and physical parameter
layout. The resident provider uses the same invocation and coordinate wrapper
as text partition banks. The scope encloses all chunks and closes on failure;
nested expert exchange borrows the scope. Disabled observations use the ordinary
provider path without creating tensors or coordinate payloads.

Each successful composite partition group now finishes its layerwise policy
with the actual completed output or transport boundary, just as text pipeline
execution does. Dropping the guard remains the failure path. This closes dense
streaming windows before later parameter operations and prevents a successful
forward from being misclassified as an aborted policy invocation.

Qwen hybrid prediction separates its checkpoint-shared fusion and three
normalizations from the independently streamable decoder block at each depth.
The hybrid decoder's generic pinned extension carries those canonical modules;
conditional composition preserves that extension when consuming the text
modules. Static unit-consumer ownership places one copy on each partition that
executes a prediction unit. Repeated logical depths therefore do not create
repeated canonical parameters or charge one fusion copy per depth on the same
partition. Prepared prediction construction materializes the shared module once
from its retained tasks and exposes it once to parameter control, followed by
independent block modules. Native binders consume these typed owners; they do
not infer sharing from checkpoint names.

Logical MLX subgroups that share a world collective complete each packed
collective before returning to its caller. This preserves the admitted
participation-wave order when an active pipeline stage builds independent
embedding and vision branches and inactive stages submit zero-work entries.
Leaving those world collectives lazy can match unrelated operations when graph
consumers evaluate branches in different orders. The rule applies to packed
sum, gather, and point-to-point fallbacks; it agrees with the existing completed
neighbor-exchange path. The backend owns this native ordering, without changing
the architecture's wave declarations or parameter ownership.

Intermediate world-wave completion uses the selected communication deadline and
retains its input, output, stream, and native group in the ordinary completion
owner. A timeout quarantines those resources and fences group reuse until safe
completion; ordering does not introduce an unbounded wait into a contracted
communication path.

Prepared Qwen text prediction now pairs with the ordinary replicated text target
as well as the conditional target. Both call one architecture-owned fusion and
prediction-state driver; their adapters supply only token embedding and shared
vocabulary projection. The generic replicated decoder reuses its existing
prediction capture and readout helpers. Source construction selects text or
conditional parameter descriptions before deriving the same text/MTP geometry.

Prepared Qwen hybrid prediction executors now borrow the ordinary target embedding
and output head and thread internal observers through the shared fusion/decoder
driver in prefill, proposal and accepted-token replay. Discovery places each
prediction decoder in its own score scope while retaining canonical shared fusion
and normalization identities. The target and extension continue to own separate
mutable state; backend binding supplies storage and invocation without interpreting
prediction equations or checkpoint names.

Qwen prediction targets also reuse the architecture-owned routed and direct
partition constructors. Typed pairing occurs after the retained target admission
and source-store identity checks; the existing mixed-state backend visitor supplies
binding. Prediction producer placement admits pinned shared parameters only for
their declared static role and consuming execution group. It retains both sides
of the normalization intervention boundary across execution replicas.

Prepared resident prediction banks use the runtime's shared invocation scope to
finish routed observation work before entering a tensor collective. Architecture
construction binds their retained global unit coordinates; Qwen and DeepSeek
reuse the same scope and partition mapping. Ordinary residency providers keep
their existing outer scope, and disabled observation inspects no tensors.

Qwen prepared prediction declares replicated target embedding and output roles
in the neutral load requirements, like other extensions that borrow the target
head. Every pipeline replica binds the selected tensor shard and residency
estimates count those copies. Ordinary target input/output hook ownership stays
with its declared pipeline stage; auxiliary storage grants no publication role.
Prediction fusion also registers read-only embedding input evidence under its
own logical invocation, separately from target embedding observations.

Composite prediction input obtains semantic token IDs through the architecture's
prepared-input contract. Qwen and Inkling share their ordinary ingress token-part
construction with prediction, including admitted media placeholders and projected
spans. Backends perform generic prepared-input admission and call that contract;
they do not infer placeholder IDs or discard media while extracting text tokens.
Token identity construction creates no media features, grids or tower output.

Nemotron prediction units share one architecture-owned fusion and final-normalization
driver across ordinary and observed serial/parallel paths. Observed scheduled
operators call the existing target component hooks; the effective normalized and
fused values feed the actual decoder. The first physical unit owns fusion and the
last owns readout normalization, preserving each depth's physical schedule.

Nemotron prediction discovery keeps each depth's physical attention/MoE schedule
separate from its target layers. Its component joins use the declared `mixer`
parameter roots, first-unit fusion and last-unit normalization. Prepared extension
construction binds resident expert coordinates from those declarations and retains
shared embedding/head replicas through neutral load requirements. Prediction
prefill, proposals and accepted-token replay share the same observed driver.

Nemotron-H prepared prediction groups use the same retained tensor-rank layout
as the other embedded prediction families. They replicate across target pipeline
and expert coordinates, with shared embedding/readout roles retained on each
participating rank. The architecture preparation driver owns this policy and the
physical prediction schedule; native binding consumes those prepared units.
For transformed prediction weights, preparation derives executable modules and
their companion geometry from the retained task formats. It separately retains
the source tensor-rank layout for bounded conversion before binding; this avoids
applying a global source shape to a local transform or reconstructing format
selection inside the backend.
Routed Nemotron target construction likewise retains a separate source
architecture when selected tasks transform weights. Preparation checks identical
local state geometry and unit ownership before handing both layouts to ordinary
materialization. Native quantization does not reconstruct that source model.
Packed MLX embedding lookups normalize selected decoded rows to F32, matching
the backend's cold activation/state-width fact. This includes MXFP4, whose native
dequantizer otherwise defaults to BF16. The conversion precedes vocabulary
reduction and uses only the requested rows; it does not widen an entire table.
The runtime's uniform partition-range expansion converts retained encoded blocks
to architecture-supplied scalar widths. DeepSeek, Nemotron-H and Kimi Linear use
that shared checked conversion for expert ownership; a block count is not a
neuron count. Missing ranges and explicit nonuniform chunk layouts remain errors.
Composite partition preparation also reports the exact admitted recipe sources
assigned to other expert owners. Native strict-loading validation consumes that
declaration, as it does for text partitions, so split expert checkpoint layouts
retain the same source-ownership policy as packed layouts. Unrelated keys remain
unexpected; this does not authorize ignoring a checkpoint-name prefix.

Inkling component instrumentation shares its ordinary decoder equations, including
relative attention and all four causal convolution histories. Unit/channel hooks
precede their actual down/output projections. Projection writes, the learned
dense-branch scalar, and the complete post-convolution residual contributions are
separate evidence. Routed and jointly weighted shared banks borrow the runtime's
unit-observation adapter on their ordinary provider requests. These equations and
semantic boundaries remain in `eredu-architectures`; providers retain residency,
placement and completion ownership.

Inkling's prediction primitive uses the same component driver as its ordinary
text blocks. It preserves the released sequence of two uses of the same hidden
normalization, independent embedding normalization, hidden-before-embedding
fusion, the ordinary decoder, and optional chain normalization. Captures identify
each normalization application and effective fusion input separately; shared
normalization parameters do not merge logical invocations or their mutable state.

The materialized Inkling prediction adapter forwards its admitted observer through
those same operations, including observed prefill and explicit depth advancement.
The target-owned vocabulary projection exposes its actual muP-scaled multiplication
input as read-only evidence and its original/effective scores after vocabulary
trimming. Disabled execution uses the same driver without constructing observation
paths. Target state and each prediction depth's causal histories keep their existing
owners; an observation failure does not imply that partially advanced state can be
reused without restoring the lane.

Component normalization contracts explicitly represent identity normalization for
heads that omit a data-dependent norm. Optional gains and biases remain affine;
there is no RMS denominator for this case. The evaluation layer consumes this
neutral declaration when constructing measured readout directions, including a
zero residual. Family declarations select the equation from normalized configuration.

Architecture descriptor schema 11 and component schema 10 retain
`component_transforms`, explicit identity normalization, and typed tensor-transform equations.
Transform records join actual input/output observations to their owning nodes
and effective parameter groups. Constant/learned scaling, normalization and causal
depthwise convolution remain architecture semantics. They are not extra additive
residual terms, and their presence does not imply loaded capture or edit support.
Inkling declares its four decoder convolutions and dense learned scalar from the
same normalized layer policy used by construction. Convolution kernels use the
canonical `[channel, 1, tap]` layout, ordered oldest to current, including the
identity bypass. Current-position evidence does not grant historical-cache access.
Ordinary bounded parameter operations continue to admit every read or edit.

The component activation contract also distinguishes gated delta attention from
softmax attention. It describes key-by-value recurrent updates, Q/K scales after
head normalization, per-head log transition rates, key-channel decay biases,
headwise update reads, and normalization and gating before the output-channel
boundary. Convolution transforms can declare their post-convolution activation.
Kimi supplies these declarations from the same normalized configuration as its
portable KDA driver. Its MLA declarations retain normalized query/KV bottlenecks
and the admitted fused or split KV-B layout. Shared SwiGLU declarations reuse the
ordinary gated-unit builder. The backend supplies the existing projections,
convolution, recurrence, parameters and completion resources.

Kimi's ordinary, routed, replicated and partitioned execution use the same
instrumented sublayer drivers. Partition readout uses the neutral
`PartitionStaticModules` driver; capture happens at the input/output owner and
state lookup retains the partition's local ordinal. Discovery alone does not
establish loaded collector support or native validation.

Component reads can name their actual affine `projection_output`. Tensor-transform
joins describe causal operations before subsequent head normalization. A shared
grouped projection record describes Inkling's relative-query features multiplied
by the learned `[relative_feature, relative_position]` table for each head. The
observation retains current-position profiles before their ordinary axis
permutation; it is not an attention-probability or historical-cache capture.
Readout `scaled` evidence separately records muP multiplication before the selected
projection's input arithmetic, so input rounding cannot be attributed to scaling.

Primary component readouts can declare embedding normalization after assembly and
scaling. The captured post-normalization embedding remains the residual base;
consumers must not substitute raw token rows for it. Inkling joins this declaration
to its shared embedding norm, final norm and untied output projection, and describes
its muP scaling through a tensor-transform record. Whole post-convolution writes
contain their scalar projection groups, so a score decomposition includes the whole
write once or expands its declared transforms without counting both.

Routed component declarations can specify `routes_per_token` independently of an
ancestor router's top-k. Inkling uses this for its separate selected and always-on
bank invocations; the shared branch is not relabeled as an independently routed
MoE. Both declarations consume the existing family expert specifications and
actual jointly normalized unit coefficients. This does not advertise standalone
routing events that the joint selector does not emit.

Inkling prediction preparation retains one typed module per physical depth and
one optional canonical chain-normalization module. The ordinary whole-chain API
and prepared executor share the same depth equation. Source constructors retain
checkpoint encodings; executable constructors consume the exact selected task
formats. Module visitation reserves ordinal zero for the optional shared norm and
uses depth plus one for decoder modules, so a shared parameter is visited once.
The native prediction materializer applies the retained residency policy to
these physical owners. Shared normalization overlaps the active sequential
module; bounded policies page depth modules through the common residency ledger.

Prediction component scopes describe the actual static parameter roles when the
parameter graph has no separate pageable execution groups. Inkling additionally
declares source embedding normalization before the per-depth embedding norm and
two applications of the same hidden norm before hidden-first fusion. The optional
chain norm uses the neutral identity normalization equation when disabled. These
semantics remain architecture-owned; native adapters consume ordinary prepared
modules, exact tasks and observation paths.

Prepared Inkling prediction now retains the parameter declaration and exact local
layout lent to materialization. Its depth weights remain replicated, while shared
embedding and output parameters retain their ordinary tensor-axis placement.
Prediction selection retains embedding, embedding-normalization and output roles
on every prediction replica, including pipeline stages that do not publish target
scores. A static sequential component scope proves ownership through these roles;
it does not invent an execution-unit owner. Discovery's physical layer inventory
still describes each real prediction block independently of residency ownership.
Released aliases retain canonical projection roles during recipe finalization, so
selected formats and their companions agree with the constructed operators.

Component placement follows declared affine reads and tensor transforms through
exact retained layouts. Affine read outputs keep their actual row coordinates;
convolution kernels prove channel ownership, and shared grouped projections map
complete input groups to output heads. A whole residual-write invocation can
establish both sides of its channel-preserving convolution, including a sum of
routed and shared writes. Missing dependencies and inconsistent coordinates are
rejected before capture admission.

Routed placement joins each component's exact write parameter to its retained
expert catalog, keeping logical execution ownership separate from provider cache
addresses. Per-invocation catalog distribution decides whether expert coordinates
are partitioned or replicated. Equal-sized routed and shared banks therefore
remain distinct. Prepared provider coordinates use the same catalog declaration;
shared invocations preserve their configured cardinality and global identities.
Native mechanisms consume these facts without family-specific placement branches.

The selected grouped realization retains per-invocation catalog distribution
through format selection and local construction. Replicated invocations execute
their complete local bank without expert route exchange; partitioned invocations
use the retained expert assignment. Both paths share the same ordered provider
success agreement across tensor, expert and optional pipeline-wave groups before
a caller can reduce or publish results. Inactive pipeline stages follow the
selected agreement-only wave for replicated providers. Catalog distribution is
execution policy, independent of the numerical equality of bank dimensions.

`AttentionCache::relative_attention` is the neutral cache-aware contract for
learned distance profiles. Resident caches consume their returned history;
blockwise caches consume their retained blocks. The default rejects a blockwise
cache without that mechanism so a caller cannot silently attend only to newly
appended keys. Inkling uses the same operation for target and prediction layers.
MLX reuses its leased block scan and online softmax accumulator with per-block
profile bias, shared with contiguous relative attention. Query-position scaling
and relative profile gathering remain native general mechanisms; the architecture
owns profile parameters and attention policy. Sliding history is discarded only
after every submitted query has consumed it, and scan telemetry includes the
additional bias/scaling workspace. No complete-history tensor is reconstructed.

Composite partition construction also retains an optional typed source architecture
and its exact local layout for selected load-time transforms. The optional source
is heap-owned across handoffs; completing the typed admission uses a separate
construction frame so unrelated family branches do not multiply stack storage. Inkling target and
prediction-target construction obtain this source from the retained original
configuration, preserve routed/shared ownership, and check execution-unit
addresses, state and vocabulary ranges against the selected target. The shared
composite session handoff passes both descriptions into the ordinary partitioned
materialization mechanism. Native adapters bind them without reconstructing
source configuration or reopening artifacts. Other composite families use the
same handoff when their source construction supplies it.

Inkling's joint routed/shared selector consumes a floating router matrix through
its neutral tensor operation. Requirements therefore classify that matrix as a
raw parameter, consistently with family format policy and parallel declarations.
Model-wide affine/MXFP4 conversion applies to the declared linear modules and
expert banks; it preserves this router, relative-profile tables and convolution
kernels in their original floating representation.

Bounded checkpoint selection pushdown retains leading-axis restrictions nearest
the physical source. It pushes each independent selection through the original
tree once and reapplies existing selections without recursively exchanging their
order again. This prevents rewrite cycles while allowing expert/member and row
ranges to form a contiguous source tile before column selection. The checkpoint
crate proves this without native tensors, including all six axis-selection
orders and matrix-row tiling with distinguishable source values.

Inkling prediction accepted-token replay uses the same observed prefill driver
as initial lane preparation. Its declared internal-observation support therefore
also applies when target verification rejects a tentative draft. Attention-input
interventions occur before the prediction cache update; later cached predictions
retain their effect. Observation alone preserves ordinary replay state exactly.

Addressable partition providers retain route cardinality per invocation. Expert
exchange lowers routed invocations to one route per exchanged row, while shared
invocations keep their complete replicated route set. Before acquisition, binding
checks exact member keys, catalog distribution, owning group and selected bytes.
Routed members use the retained rank-local global-ID map; shared members use their
own complete ID range. A shared bank may contain more members than the routed
bank, and a foreign unit cannot satisfy validation through aggregate entry counts.

Kimi Linear and LFM2 dense and routed partition construction retain the source model
when lowering includes a parameter transform. The architecture reconstructs its
source formats from exact admitted requirements, verifies target/source semantic
addresses and local state geometry, and passes its typed source layout into the
shared runtime factory. This includes the routed expert realization and shared
bank geometry; the backend only materializes the retained source and target.

Kimi attention and LFM2 attention/convolution companions inherit their primary projection's semantic head or
output-column sharding through neutral parameter metadata. Their own physical
shapes determine the scale/bias intervals; checkpoint-name suffixes do not make
companions replicated when their primary matrix is sharded.
LFM2 routed expert ownership expands encoded logical partition units through the
shared checked range contract before deriving component and bank coordinates.

Qwen hybrid's recurrent attention now declares scalar value-head output channels.
The same ordinary, provider and parallel block drivers observe fused projected
Q/K/V, their causal SiLU convolution, control projections and the consumed
normalized/gated channels. Original/effective channel hooks precede the actual
row projection. Its read-only input evidence comes through the shared projection
observer, after any selected dynamic input transformation. With instrumentation disabled, these seams neither create path
strings nor evaluate or retain extra tensors. Channel interventions occur after
the recurrence has updated state; sublayer-input interventions occur before it.

The neutral gated-delta equation distinguishes key and value head counts and
widths and per-head versus per-key-channel decay. Qwen uses additive-epsilon L2
normalization of Q/K, contiguous head repetition, per-head decay and a SiLU output
gate. Kimi uses its declared RMS/scaling convention and per-key-channel decay.
L2 is explicitly `x / sqrt(sum(x*x) + epsilon)`. Evaluation readout directions
honor this denominator without substituting RMS or a clamped norm. Recurrent
writes with scalar component declarations no longer appear again as opaque
readout terms, while routed combined FFN contributions remain explicit.

Composite partitions retain the same immutable partition-bank selection as
routed text: bank identities, per-invocation routes, local component coordinates,
exact member tasks and selected residency. Shared member ownership follows its
catalog distribution and actual execution-unit owner rather than the routed
expert range. The common architecture construction driver binds resident or
independently cached providers to this retained selection. Native composition
supplies the shared cache and transfer mechanisms, excludes independently
materialized targets from ordinary unit loading, and retains bank handles for
bounded parameter work, atomic overlays and telemetry. Composite pipeline waves
and media boundaries continue to use their original architecture-owned schedule.
Addressable member keys identify bank invocations independently of their owning
execution unit. Runtime validates every retained parameter task against that
owner, while architecture catalogs validate the invocation and member identities.
This permits routed and shared invocations to belong to the same decoder unit.
Gemma 4's ordinary and routed TP/provider drivers translate global layer
identities through the retained partition state offset before borrowing caches.
The layer identity still selects equations, per-layer inputs and shared-KV roles;
only mutable-state access uses the local ordinal.
Its expert modules also consume the retained localized grouped specification;
router identities and cardinality remain global. This keeps EP-local storage
geometry separate from routing policy and preserves the same block construction
for resident and independently cached providers.
Muse's centered normalization derives its effective gain from the current
authoritative parameter on each invocation. It does not retain a separate gain
tensor across parameter replacement or unloading, so overlays and residency
transitions cannot reuse a stale derived parameter.

Muse component execution uses a single architecture-owned block driver for
ordinary, provider-backed, tensor-parallel, and observed calls. Its hooks expose
the actual gated attention channels and FFN units before their selected output
projections, and both sides of the separate post-projection RMS normalizations.
The postnorm acts on the complete reduced write; sparse expert contributions
must be summed before that normalization. The same architecture-owned final
readout handles tied and untied heads and retains output multiplication after
the actual vocabulary projection. Backends continue to supply ordinary tensor,
projection, sparse-unit, and collective mechanisms.

Descriptor schema 12/component schema 11 adds an explicit normalized-head
`output_scale`, token-only embedding normalization before media assembly, and
`scaled_softcap` for `cap * tanh(affine_score * scale / cap)`. The captured
assembled embedding remains the residual base. Additive score contributions
refer to the unscaled affine score, with the actual projection input and its
rounding retained as evidence. These are neutral equation declarations, not
claims of loaded support or new backend family branches.

For a complete route-weighted residual write, neutral component placement may
establish a post-normalization input from its already established output owner.
This follows the normalization's preserved axes and invocation. Admission still
requires the complete normalization axis, exact input/output geometry, and
replicated gain/bias parameters owned by that invocation; a foreign pipeline
stage's gain cannot authorize the input capture.

Muse's expert checkpoint alternatives discriminate packed and split storage by
the gate/up identities. Their shared down projection remains required by either
selected layout, but its presence cannot select both alternatives. Header-only
admission validates the complete selected layout before native materialization.

Muse load-time media quantization derives eligible matrices from its checked
architecture-owned checkpoint geometry. It selects aligned patch, position,
attention, FFN and adapter matrices, retaining dense storage where an input
axis cannot hold the requested packed blocks. Exact checkpoint format admission
remains separate. The retained selected formats drive cold capabilities, memory
plans, materialization and cache identity; backends do not repeat this selection.

`ParameterTransformConstraint::LinearIfAligned` carries the architecture's
optional packing policy into neutral realization selection. Valid requests whose
blocks do not fit retain the exact native source format; invalid requests still
fail. Muse applies this declaration to media projections after canonical recipe
finalization, so source aliases and derived outputs retain the same policy. Text
projections retain their required transform semantics.

Muse GGUF requirements derive logical shapes and canonical identities from the
GGUF text plan and, when present, its separately admitted projector. GGUF expert
recipes consume the retained prepared source. A text-only GGUF never needs a
SafeTensors vision configuration merely to enumerate executable parameters.

Mixed Muse ingress defers vocabulary-parallel text lookup until the primary text
execution group. The preceding replicated vision group performs no text lookup
collectives. A shared neutral helper declares one exact reduction shape per text
segment, excluding media positions; inactive pipeline stages consume that same
ordered schedule. Inkling uses the helper for its existing segmented ingress.
These are architecture equations and schedules, not backend family branches.

Muse canonicalization is shared by source recipes and parameter requirements,
including the generated scale/bias identities for load-time quantization. The
released source prefix is provenance, not an executable companion name. Its
optional vision root also shares one cold/loaded transport declaration; absent
projectors contribute no static vision role to text-only partition ownership.

Muse selected partition transforms retain a separately constructed, typed source
architecture and physical source layout, as the existing composite transform
contract requires. The construction driver validates execution addresses, state,
vocabulary ownership and static roles against the selected target; it also binds
the source's local expert realization. Native mechanisms consume this retained
source and the admitted tasks. They never reconstruct source geometry from
packed target slots. GGUF's learned per-head Q/K norm gains appear as replicated
architecture parameters; the Hugging Face convention remains weightless there.

Qwen3-VL's text traversal uses the ordinary Qwen decoder's component driver,
including resident and independently supplied experts, rotary embeddings and
local mutable state. Its architecture wrapper adds the assembled embedding and
final readout seams and instruments prepared DeepStack additions at the point
where they enter the residual. Disabled hooks use the same ordinary addition.
DeepStack declarations require media input and apply during prefill; they appear
as separate residual terms, with their own original and effective captures.
Traversal continues to own unit boundaries, so unit output includes the vision
addition. Family equations and semantic branching remain in architectures.

Qwen3-VL's selected transforms now use the same typed source-partition handoff:
source encodings and local layout remain distinct from packed target slots,
while execution units, state, static ownership and DeepStack roles must agree.
Optional vision packing uses the neutral alignment constraint and retains dense
matrices when the requested blocks do not fit. The shared Qwen vision format
lookup resolves relative module names and canonical checkpoint names to the
same encoding and rejects conflicting aliases before construction. Cold
requirements and module construction consume that same lookup.
The Qwen vision position table remains an ordinary dense lookup and is excluded
from transformable projection geometry, in both Qwen3-VL and conditional Qwen.
Its rank-two storage alone does not authorize affine weight packing.

Component schema 12 / architecture descriptor 13 adds two neutral relationships
for shared-state and residual equations. A read can identify the attention
component group that actually publishes its projected state; its consumer does
not acquire a fictitious local projection or a capture of historical positions.
Readout equations can reference ordered transformations of the whole residual,
including learned scalar parameters, after each layer's writes. These transform
references affect the embedding and all prior writes. They grant no parameter
query or edit authority. Older serialized descriptions default to local reads
and no additional block transformations.

Gemma4 component integration uses its existing portable dense/routed block and
shared attention cache equations. Ordinary, controlled, provider, and partition
entry points share text preparation and the same instrumented projection and
readout drivers. Native backends continue to supply generic tensor, cache,
projection, collection, and parameter mechanisms. Family topology and the
publisher/consumer and residual-scaling semantics remain architecture-owned;
Gemma4 neutral and native acceptance is recorded in the component-analysis guide.

Declared whole-residual component transforms establish complete hidden-axis
capture placement at their architecture-owned block invocation. The portable
partition builder validates the referenced layer and containing block before
propagating the existing typed transform equation; learned parameter placement
remains independently checked. Gemma4 attention-bias checkpoint admission
follows its normalized projection schedule, including shared-KV consumers and
full-attention key reuse. Neither rule is reconstructed in a native adapter.

Gemma shared attention across pipeline cuts uses architecture-owned typed
current-position key/value roles. The receiving partition declares one additional
KV history for each remotely published attention policy, attached to its first
local consumer; subsequent local consumers borrow that publication. Cold state
selection and native construction use the same replica geometry, so residency,
snapshots, prompt-cache compatibility, and copies include those histories.
Pipeline transfer scales with the current sequence length rather than the full
cached prefix. Fixed wire roles preceding their publisher are inactive and never
consumed; the publisher replaces them before any shared read. Key-reusing
projections still transport and retain both normalized value and rotary-key
payloads, since these tensors differ after their respective transformations.

Gemma attention executes through the owning cache whenever that cache retains
history outside the returned K/V tensors. Shared consumers borrow the exact
publisher cache, or the first local receiver cache across a pipeline cut. The
architecture chooses this semantic owner; the backend implements ordinary
cache-aware attention without reconstructing the layer schedule. Published
tensors may contain only the current submission, so direct tensor attention
must never stand in for reading a paged history. The neutral numerical matrix
exercises that contract against a resident full-history oracle.

A partition's additional state must enter the neutral requirements before
mechanism capability synthesis and selection. `ReplicatedTextRequirements::with_state_layout`
validates the unchanged layer address space and traversal access, then lets the
ordinary selector admit, place, and account for every declared component. It
does not relax the selected-state comparison at binding time. Gemma uses this
for its receiver histories; the subsequent TP geometry projection preserves
the selected component roles and physical storage policy.

Composite architectures also own packing and restoring their intermediate
pipeline activations. Default continuation hooks preserve the supplied tensor;
Gemma flattens its padded media batch into the admitted sequence dimension and
restores the encoder batch from the retained request geometry. Native backends
perform ordinary tensor reshaping and transfer, without interpreting media parts.

Gemma's cold and constructed media transport use one architecture-owned phase
ownership declaration. Encoder input parameters belong to the first encoder
stage and output projectors to the last. The current atomic vision static group
is also retained on the last stage when learned standardization is enabled;
the atomic audio static group is retained on the first and last stages because
it contains both subsampling and output projection. Selected materialization
accounts for these replicas. Continuation stages prepare parameter-free media
masks/rotary state and shape placeholders; they do not execute unowned embeddings
or input encoders. Only the first decoder stage performs segmented vocabulary
lookups, and the declared inactive collective waves reflect that ownership.

Component schema 13 / architecture descriptor 14 adds optional routed-bank
`write_output` and `output` boundaries. They describe the complete bank write
after route weighting and distributed reduction, and its transformed output
before combination with other branches. Architecture declarations join these
to the existing tensor-transform equations. Portable partition derivation checks
input/output widths and exact invocation ownership; transform validation still
checks normalization axes and learned parameter placement. This exposes Gemma's
routed post-normalization without inferring a bank write from a shared FFN sum.

Gemma's key-reusing publishers declare two physical cache payloads in the base
state layout as well as pipeline receiver replicas. Reuse applies to the raw
projection only: value RMS normalization and key normalization/rotary transforms
produce distinct tensors. This is ordinary neutral KV storage; a key-only state
would lose the value history and undercount storage.

Composite pipeline boundary delivery retains the semantic destination alongside
its typed payload. A result from one independent media root can arrive while a
different root is executing; only the declared destination installs that result.
An intervening group cannot consume or reinterpret the pending boundary. This
policy lives in the portable executor and applies equally to all composite
architectures and native transports.

Gemma media format selection uses the projection's logical input width in both
cold requirements and module construction. Family-wide quantization requests
retain unaligned media matrices in floating storage under the existing family
alignment policy. Logical-shape recovery expands encoded schema dimensions
only; a floating schema tensor is already expressed in logical coordinates.
Media-to-text projectors consume the same exact media format as their checkpoint
schema, including under global text quantization.

Gemma partitioned load-time transforms retain a separately constructed typed
source architecture and its exact source layout, as other composite families do.
The architecture driver checks execution-unit addresses, selected state, media
ownership, vocabulary ranges, and per-layer input ownership before passing both
source and target to native materialization. Backends consume that prepared
handoff rather than reconstructing family configuration from target tensors.

Transform source modules use the selected target's partition ownership. Runtime's
`derive_transform_source_layout` maps those exact cuts into the source encoding
with checked axis scaling, preserving logical ranges and independent expert
placements. Architecture constructors use the resulting geometry for both
source modules and prepared-source validation, including composite families and
prediction extensions. A floating source is not independently rebalanced before
conversion: for example, a 96-column matrix converted in 32-column groups over
two ranks consumes source cuts of 64/32. Native materialization consumes those
retained cuts without interpreting family equations.

Qwen3-VL, conditional Qwen hybrid, and Muse partition static roles describe
execution consumers. The final
stage retains the `output` role even when readout shares the embedding matrix.
The architecture's parameter group declares its embedding/output consumers and
selects the required replicas. Geometry validation must not replace that
consumer role with a physical parameter name when weights are tied.


Prepared prediction modules now receive the complete execution's retained weight
residency policy from the total construction driver. Physical owners remain
separate from logical depths: shared normalization/fusion modules receive one
policy and unit identity regardless of their number of consumers. The typed
materializer's invocation contract carries an equation result together with
output and changed-state dependencies even when the equation fails. Unit and
shared owners overlap until those dependencies settle. DSpark's context and
proposal loops use the same equations through this scoped module mechanism.
Backends retain native completion and parameter leases; architectures enumerate
the actual state and output roots. MLX now keeps exact bindings and unloaded
module placeholders until invocation. Target construction registers the additional
physical owners and their exact stores before initializing one common residency
manager. Queries acquire one owner through the same loan; reversible edits live
in a per-module replacement set reapplied after population. Completion precedes
unloading and bounded device eviction, including equation failure. Native
behavioral matrices verify this combined path; cold sizing is checked against
constructed modules and actual rank geometry.

Qwen3-VL GGUF companion admission uses its composite projector name translation
for both token-pending and token-bound plans. The two temporal patch slices stay
distinct exact source keys until the architecture's stack recipe constructs the
single patch-projection parameter. A physical source slice never becomes an
independent executable topology target.


The neutral residency controller can validate each physical unit against its
retained exact catalog while sharing one plan and reservation ledger. The MLX
manager retains an immutable per-unit source registry for this contract; batch
reservation, initialization, alias-owner acquisition and eviction still use the
same ledger. Identical source keys in distinct transformed stores remain distinct
values. A residency report keeps each alternate source view's diagnostics
separate because views may share physical counters. Target and prediction
construction register their units together before initialization, preserving one
weight reservation ledger.


Prediction preparation also declares whether a physical module is shared while
other units execute. This residency overlap is distinct from tensor/pipeline
parameter placement. Shared fusion, normalization and DSpark statics must be
counted alongside the largest active unit; their shared checkpoint ownership
must not force all prediction units into one resident block. The generic native
module-transfer helper retains both changed-state and output roots with the
source transfer through existing submission recovery, including equation errors.
The prediction materializer now uses this helper for both execution and scoped
parameter queries. Runtime's `AuxiliaryWeightRequirements` computes the shared
plus largest sequential owner requirement for paged execution and the complete
physical total for resident execution. Native load admission and cold support
reports use those values alongside the ordinary target window.


Cold prediction residency is represented separately from parallel parameter
ownership. Runtime's `AuxiliaryModuleResidency` is retained on parameter
requirements and selected materialization tasks. Architecture declarations assign
physical units, including Nemotron's full physical pattern and DSpark's shared
input/readout modules. Prepared module construction checks that a declared group
matches the actual module's task set and shared/sequential role. Runtime sizes
shared owners plus the largest sequential owner for a bounded phase; resident
execution counts all owners. Ordinary bounded admission now includes auxiliary
tasks as well as the target tasks. Rank-local sizing uses the same tensor-only
prediction placement as construction, counting a complete prediction schedule
on every target pipeline and expert coordinate. The caller supplies the complete
architecture parameter description. Enabling prediction can also replicate
target statics, such as token embeddings on later pipeline stages; their selected
storage ownership contributes separately to pinned bytes. A neutral numerical
fixture compares cold totals with constructed module tensor geometry across
tensor, pipeline, expert and combined eight-rank placement. Shared-phase sizing
and constructed-module roles are also checked in runtime and native fixtures;
the complete family/format paging matrices are recorded in the component-analysis
guide.

Gemma 4 primary GGUF admission maps text tensors into the complete family's
`model.language_model` parameter tree. The standalone text/assistant mapper
remains available for those construction contracts. This keeps token embedding,
per-layer input and decoder-unit ownership identical across SafeTensors and GGUF;
projector tensors retain their separately declared media owners.
The GGUF layer multiplier and routing scales map to scalar parameter identities,
including the router's per-expert scale; they do not inherit a matrix weight's
suffix or projection owner.

The component transport audit distinguishes invocation membership from parameter
storage. Ordinary hooks use `ParameterGroupOwner::is_owned_by`; auxiliary copies
are accounted through `is_stored_by` and do not add ordinary callback peers.
Prepared target hooks use the complete tensor/expert stage, and replicated
prediction hooks use the session group. With the retained pipeline-major rank
order these are connected Ring memberships, already covered by bounded member
status agreement. An arbitrary disconnected low-level membership still receives
the existing typed rejection: a world-participation wave cannot run independently
of inactive pipeline peers. No such relay is required by the selected component
execution paths described here.

Detached prompt and sampler preparation seals and observes its native submission
scope before returning. An existing operation error remains the reported cause
even if the scope also reports failed or blocked work. That error does not grant
completion: unresolved roots stay in the same recovery owner until terminal
evidence permits retirement. A generic status error is used only when the
operation itself supplied no error.

GGUF admission retains each Qwen primary tensor's exact packed format before
constructing embedding, expert and linear parameter topology. Standalone and
Qwen3-VL contexts share that resolver, using the context's declared parameter
root. Qwen3-VL and Qwen3.5 companion admission also records vision-block and
DeepStack-merger formats before constructing their composite configuration.
Facade-owned media-token binding preserves those facts. Backends consume the
resulting typed bindings and recipes; they do not infer missing family formats
from packed byte widths during native construction.

Shared architecture requirement discovery converts packed GGUF schema byte widths
back to logical coordinates using exact block geometry, rejecting partial blocks
and overflow before capability selection. Encoded storage widths never stand in
for the logical dimensions of a linear operation.

Prepared composite matrix-format handoff includes executable embeddings as well
as linear projections, preserving quantized lookup and tied-readout storage.
Normalization vectors and fixed dense position tables keep their separate roles.

Muse GGUF projector recipes flatten the admitted spatial patch kernel into the
family-declared linear patch input order. Cold requirements retain that logical
geometry, and backend binding consumes the prepared recipe without family-specific
reshaping. Both raw and canonical source names refer to the same checked layout.

Native communication uses the stream selected by the actual MLX communicator,
independently of model computation. The safe native wrapper exposes that retained
stream; backend groups share its ownership with logical groups and completion
recovery. Ring therefore communicates on its CPU transport while model operations
can run on Metal. Compatible compute streams retain caller ordering and device
affinity; singleton collectives keep the compute stream without initializing a
transport device. Initial manifest exchange uses the same stream selection and
retains its exact group through completion, before rank-local contract checks.
Native tensor dependencies order the two streams. Packing and
model arithmetic keep their compute stream, and existing communication budgets
continue to cover their owned buffers; no physical allocator-workspace guarantee
is inferred from this stream selection.

Manifest and session consensus preserve transport causes in the neutral generic
protocol error. The MLX adapter retains that exact policy/transport error through
its backend failure conversion, so application diagnostics can reach the original
native source without exposing native error parameters in facade operations.

Rank-local runtime construction also retains architecture and mechanism errors
through the typed factory handoff and final session wrapper. Partition executor
and traversal-policy constructors retain their neutral error types as sources.
Boxed errors retain the leaf itself, so a policy or I/O error without a further
source remains discoverable through the public neutral failure chain.
Vocabulary and tied-embedding readout gathers also retain the original native
exception through the neutral neural error conversion.
The replicated and composite session adapters retain typed failures during
parameter loans, state reset, snapshots, prompt-cache operations and ordinary
prediction-lane execution instead of replacing them with display strings.
If a prediction operation and its state recovery both fail, the adapter retains
both errors and exposes the recovery cause. An earlier state-preserving rejection
cannot authorize retry after that recovery failure. This does not itself prove
native completion or release pending resources.
Setup failures do
not imply native completion or permission to release pending resources.

DeepSeek V4's published FP8 default applies to linear projections. Its text and
Markov embedding tables, and target/prediction hyper-head functions, retain their
dense schema and operator semantics. Explicit separately admitted embedding
formats remain distinct from that default. Vocabulary construction derives token
ownership from the architecture-declared weight rows; FP8 companion rows index
128-entry blocks and retain their own format-derived physical placement.

DeepSeek V4's family format resolver retains published dense embedding, router
and DSpark Markov-embedding defaults, including under block-FP8 metadata. Explicit
selected per-parameter formats take precedence for admitted transformations. Cold
lowering and executable construction consume these same family-owned decisions.
The same resolver distinguishes published expert encodings from the global
projection default: mixed FP8/MXFP4 experts keep full per-row scale geometry.
Schemas, routed construction and partition placement share this decision for
target and prediction layers.

Prepared partition discovery retains an immutable index of its exact selected
tasks, aliases, companion outputs and checked parameter members. Its size is
bounded by those existing declarations, and it is rebuilt when parameters are
rebound. Queries still compile ownership and reserve the same per-call metadata
and transport costs; indexing avoids repeated full-declaration scans during
all-layer experiments. It confers no loaded authority and retains no tensors.

DeepSeek encoded expert assembly retains scale and affine-bias companions under
the same source selection, expert ordering and gate/up concatenation as weights.
Target and prediction preparation consume the complete canonical recipe outputs;
the backend does not reconstruct family source names or infer missing scales.
Independent expert residency carries these same companions in each bounded unit.
Its optional intermediate selection respects source packing and scale blocks;
pre-encoded weights and their companions remain preserved, while ordinary sources
retain eligibility for the selected load-time conversion. Target and prediction
format decisions come from the architecture's existing routed-block policy.

Native catalog preparation indexes borrowed selected output names once per query
under an additional finite host-metadata reservation. It continues to validate
each loaded slot against the retained neutral placement before publishing facts.

Muse GGUF routed recipes enumerate decoder layers independently of the composite
vision-plus-text execution ordinals. Projector recipes retain their own semantic
source roles; adding vision layers never creates text-expert source requirements.

Generic native execution-layout and unit preparation retain typed architecture
failures through the backend error source, including resident-policy conversion.
The outer public failure remains backend-neutral; preparation does not flatten
these causes into diagnostic strings.

Muse vision pipeline edges use architecture-declared flattened patch geometry
within the tower and flattened projected-media geometry at decoder ingress.
Cold continuation bounds expand the selected decoder extent by the spatial merge
area; request admission supplies the exact grid and placeholder counts. The
portable composite driver transports those declared roles, without treating
image patches as text positions or adding backend family branches.

Partitioned target readout discovery registers the actual projection-input
observation with output-invocation ownership, alongside residual and normalized
values. It remains read-only and replicated across output TP ranks; parameter
storage replicas on other stages do not grant observation ownership. The same
contract already governs prediction-scope readout inputs.

A Muse rank that has already executed a vision partition resumes the incoming
decoder activation in its retained request context. The ordinary primary-only
boundary carries no additional learned context and must not trigger a second
text/media assembly.

Muse static vision parameters use exact unit-consumer ownership. Patch/position
inputs and the grouped static norms are available on each vision partition that
constructs request context; the merge adapter and final projection are owned by
the final vision unit. Cold source selection, parameter topology, materialization
and residency sizing consume the same family decision, including replicas.

Portable partition execution preserves typed architecture, policy, state and
communication failures through the neutral neural error source. This includes
active and inactive pipeline waves, dependency completion and boundary transport;
retaining a diagnostic source does not grant completion or retry authority.

Neutral partition communication retains owned submission and completion errors
through `BackendFailure` sources while preserving operation, phase, route and
poison-domain semantics. Diagnostic-only poison notifications remain available
for borrowed failures that callers already retain. Neither form establishes
resource release; native recovery and completion still own that decision.

Composite architectures report whether retained context still contains pending
decoder-ingress collectives. Muse keeps token lookups pending through vision
execution; Gemma and Inkling resolve them at context creation. Inactive pipeline
ranks consume this semantic state when mirroring the declared primary wave.
Native backends execute the same operation order without selecting family branches.

Final source adaptation and partition error wrappers preserve owned errors as
`Error::source`, including cold model/preflight failures and non-architecture
`LayerwiseRuntimeError` variants. This changes diagnostic retention only; the
architecture/runtime layers still own preparation and residency policy, and the
native backend still establishes completion before releasing native resources.

### Request memory forecasts and prefill chunks

Cold options can derive physical placement and available capacities through the
portable facade's `GenerationMemoryOptions::for_hardware_device`. It matches
backend/device identities exactly, takes a backend-supplied host-execution fact,
and maps hardware observations into execution and host budgets without substituting
installed capacity or application policy. `for_local_device` composes that helper
with local hardware discovery and allocator observations. MLX classifies canonical
device IDs using the same parser as realization, without creating a native device
or stream. The CLI supplies its selected plan device and retains ownership of
application limits and reserves. The explicit-placement constructor remains available.

Ordinary architecture construction retains `TextExecutionTopology` beside exact
parameter and state requirements. Shared decoder and gated-convolution builders
consume the same projection specifications used by this topology; no family
selects allocation multipliers or forecast equations. Runtime owns checked
mechanism/lifetime composition, physical-capacity comparisons and candidate
recomputation. The topology preserves each layer's selected attention arithmetic,
convolution kernel, packed expert routing, physical projection format, explicit
parameter owner and tied output head. `PreparationMechanismProvider` supplies
neutral input-score workspace facts without family inspection. MLX reports its
query tile thresholds and conservative temporary-copy allowances from the same
constants used by its native attention mechanism; selection retains these facts
and includes them in cached-selection validation. Within a full-key input-score
attention invocation (at most 8,192 key positions), MLX prepares expanded K/V and
contiguous BF16 RHS projection layouts once for all query tiles, preserving
reduction order. Large tiled calls synchronously evaluate groups of at most 32 tile
outputs before constructing the next group, keeping completed outputs and shared
K/V while releasing temporary graphs. Small calls remain lazy. Batching is a
native mechanism policy with no family condition. Longer rows retain the separate
two-pass blockwise realization and its evaluated per-block state, without retaining
every expanded block. The optional neutral `FullKeyAttentionTiles` contract
describes the covered key length, shared K/V layouts, maximum live tile count and
retained outputs. MLX reports four shared query-width K/V payloads, 32 live tiles,
two output payloads and 32 working bytes per score element. Runtime combines
these facts with architecture geometry for K/V projection buffers, score conversions,
convolution scratch and overlap. Missing native facts preserve an unknown bound.
Legacy records lacking the full-key contract retain per-tile allowances; longer
key rows keep that conservative fallback. Selection-cache identity includes the
new retention facts. No native device is needed to select or estimate them.
Persistent convolution history remains in the state layout. Mixed-width selected
parameter metadata adds a float32 cast allowance and promoted-state/replacement
storage above nominal state bytes; it is derived from task shapes and dtypes,
not checkpoint-format dispatch in a backend.
Fully resident MLX parameter materializations own lazy F16/BF16-to-F32 conversion
caches for dense projections, including tied output embeddings. A weak native
identity registry shares aliases without extending residency; evaluated conversions
are released with their owner. Bounded host/disk policies do not enable the cache.
Explicit device-residency ceilings also disable retention rather than allowing
derived storage to bypass original-parameter admission.

`eredu-core::residency` defines the portable parameter-conversion retention
contract: disabled, bounded payload bytes, and explicitly unlimited policy.
Pure normalization preserves the requested value and managed-default/explicit
provenance, canonicalizes a zero-byte bound to disabled, and reports effective
disabled policy for host-layerwise, disk-streamed, explicit device-ceiling or
unsupported mechanisms. The managed policy is 256 MiB and is enforced by
eligible MLX resident materializations. `ExecutionPlan::with_parameter_conversion_retention`
and `NormalizedLoadRequest::with_parameter_conversion_retention` select a
model-scoped optional override before preparation. `None` retains managed-default
provenance; explicit disabled, bounded (including zero), and unlimited requests
survive normalization and architecture selection. The retained neutral selection
supplies the policy to native residency construction before target or embedded
owners register. Loading another model cannot change a published budget.
Runtime owns shared admission and reservation policy; the backend owns native
conversion storage, evaluation, completion safety and observations. Allocator
caching remains an independent policy. A retained-payload ceiling covers retained
plus reserved conversion payload, not backing capacity, temporary casts, graph
storage, process RSS or total memory. Admission is first-admitted without automatic
eviction; denied retention leaves temporary conversion execution available.

`ParameterConversionRetentionGroup` wraps the existing scoped `ResourceIdentity`
in a distinct budget namespace. One selected loaded execution shares this group
across permanent units and embedded prediction owners. Native partitions must
share admission for that execution; current multi-process MLX construction
disables retention because it lacks cross-process reservation authority.
Separately loaded external drafters have independent groups. Existing conversion
allocation identities and parameter-binding owners remain authoritative for
physical storage. Aliases within a group charge once, while independent groups
sharing an allocation each admit their own retention claim. Group charges cannot
be summed as physical residency; allocation observations still deduplicate shared
backing. Core `OffloadReport` can carry the separate group reports without adding
their payload to its original-parameter ledger. Unsupported or unavailable
observations differ from an observed empty cache or group list. Legacy records
and unsupported telemetry default to unavailable, never today's managed policy.
Native backing capacity is separately observable and is not added to payload.

`StaticMemoryReport::parameter_conversion_retention` and
`ResidencyTelemetry::parameter_conversion_retention` preserve the load-selected
policy, eligibility, budget identity, retained claims and reservations. Retained
conversion payload remains a named subset of the memory report's current device
parameters; group admission charges must not be added to physical residency.
`LoadedModel::parameter_conversion_retention` reads these facts through the neutral
`ModelCapabilityBackend` hook. `PlannedModel` composes role-labelled `target` and
`external_drafter` observations through `ParameterConversionRetentionObserver`;
embedded prediction has only the shared target entry. A missing external entry
means no separate participant, while an unsupported entry means a participant
whose ledger cannot be observed. The auxiliary MLX assistants currently report
unsupported, while independently loaded autoregressive drafters expose their own
budget. Execution-plan policy applies separately to both loaded models.

The residency telemetry document preserves `current_device_bytes` and
`peak_device_bytes` as original-parameter admission-ledger counters.
`total_current_device_parameter_bytes` reports the checked sum of current original
parameters and deduplicated retained conversion payload;
`current_device_parameter_conversion_bytes` is a named subset of that total.
Use the total for current parameter accounting, without adding the subset again.
The historical peak ledger is not a historical peak of optional conversions.
An overflowing total is unavailable rather than saturated. This total covers the
ordinary residency ledger; independently managed routed banks retain their
separate telemetry. Use `StaticMemoryReport` for whole-model parameter composition,
including those banks.

Ordinary and controlled reporting uses the same native owner observations and
preserves backend errors as `BackendFailure` sources. These queries do not
allocate native execution resources, evaluate or settle work, populate conversions,
or consume state/observation budgets. This initial API has no live limit setter;
a request is immutable once loaded. Explicit settled trimming is described below;
[retention-aware forecasts](#retention-aware-forecast-ownership) consume the same
neutral scope and live binding observations. The
[consumer workflow](generation-memory.md#controlling-parameter-conversion-retention)
and [facade example](../eredu/examples/conversion_retention.rs) use these contracts
without managing another cache or importing native storage mechanisms.

`eredu-runtime::residency::conversion_retention` implements the shared portable
budget and weak registry. Composition may attach one `ConversionRetentionBudget`
to each execution `ResidencyController`; target units, embedded owners and
in-process partitions can receive clones of the same selected budget. MLX
multi-process partitions reject retention as described below. Named layer windows
do not create allowances. MLX residency constructors now install the managed budget
without creating conversions. Registry lookup joins live group identities only
when policy facts agree, and immutable source identities only when payload agrees.
Allocation descriptors preserve the actual backing
identity and reject conflicting live payload observations across groups.

Admission atomically counts retained plus reserved payload with checked arithmetic.
A move-only reservation permits one publisher per immutable entry while native
conversion and evaluation happen outside portable locks. Publication consumes the
reservation and returns a group claim keyed by immutable source and actual backing;
failed publication, abandoned tickets and unwinding cancel exactly once. Aliases
share claims, and multiple parameter bindings to the same allocation share its
charge within a group. Reusing another group's published allocation still requires
independent admission; rejection grants no permission to attach its native value.
Claim clones share ownership, and dropping the final allocation binding releases
only that group's charge. Registry maps contain weak references and cannot retain
parameters, models or native storage. Observations expose exact coherent payload
accounting while native capacity remains explicitly unavailable.

Portable lock order is registry, parameter entry, budget; claim release takes only
the budget lock. No native handles, callbacks or lock guards enter or leave these
operations. Backends pair claims with their own native storage and serialize
native attachment/publication with parameter replacement. An `is_active` query
alone is not a native fence. Invalidation permanently retires the old entry,
cancels its reservation and revokes its parameter-specific claims in every group;
it cannot release other parameter bindings to shared backing. Re-registration
creates a distinct entry, and stale ticket publication or destruction cannot
revive the retired entry or alter the replacement's reservations. Backends must
detach revoked native values at a completion-safe boundary; logical release does
not establish physical reclamation. This mechanism provides admission and
ownership accounting and settled-boundary claim release.

MLX retention owners pair each native conversion reference with their own admitted
portable claim. The global native registry holds weak entries and each entry holds
weak owner and backing references, so a denied group's source registration cannot
keep another group's conversion alive. Admission reserves checked F32 payload
before casting; native creation and evaluation occur outside native entry, registry
and portable controller locks. Publication takes the native entry lock, checks
retirement, and attaches the evaluated backing only to independently admitted
owners. Native failures and retirement during evaluation drop or cancel the ticket.
Source publication revokes all obsolete claims; restoration needs a new resident
registration. Native graphs may still retain storage after logical claim release.

One MLX residency manager shares its budget across permanent units and supplementary
embedded prediction owners. Independently loaded models and external drafters have
separate budgets. Multi-rank selected TP, PP, EP and DP topologies currently reject
retention with typed `Unsupported` eligibility and effective `Disabled`: the local
controller cannot coordinate reservations across native processes. Inference remains
available with temporary casts. This is a mechanism wiring gap, not an architectural
model limitation; no rank silently receives its own full managed allowance. Native
cross-process retention remains unvalidated and requires coordinated admission.

Host-layerwise, disk-streamed and explicit device-ceiling selections report their
typed exclusion even for explicit unlimited requests. Selection becomes immutable
before the first native owner registers. `OffloadReport` carries the effective policy
and live group ledger; physical conversion observations include only admitted owner
bindings and deduplicate shared allocations independently from group accounting.

`ParameterConversionRetentionTrimReport` specifies released group claims and
payload, remaining claims/reservations, and independently observed backing
reclamation. Releasing one group's claims neither releases another's nor proves
physical reclamation: snapshots, other owners or native graphs may retain storage,
and allocator caches may retain backing even after its final live reference drops.
No reported claim release means bytes returned to the OS. The trim operation
uses existing submission/completion authority at a settled boundary and preserves
source weights, request state, budgets and eligibility for future admission.
Controlled transactions need typed pending-work rejection unless their operation
contract explicitly permits settlement. Ordinary reset preserves conversions;
trim, parameter invalidation and allocator-cache flushing have separate semantics.
Reports remain read-only and cannot evaluate, settle, trim or populate conversions.

`ModelCapabilityBackend::trim_parameter_conversions` is the neutral settled-only
backend hook. `ModelRuntime` and the ordinary facade synchronize before calling
it; controlled ordinary continuations call it through `TextContinuationBoundary`
after canonical completion and record draining. Controlled speculative execution
uses its existing transaction boundary and shared driver, rejecting pending work
with `ParameterConversionTrimError::NotQuiescent`. Neither path changes request
state or run budgets. Backend errors cross the facade as `BackendFailure` with
the original source. Backends without the operation return typed `Unsupported`.
MLX owns native reference detachment and healthy idle-authority validation;
portable residency policy owns coherent claim accounting. Embedded prediction
uses its target group, while composed external drafting keeps separate scopes.
Ordinary composed trim visits the target and then external drafter; it is not an
atomic transaction across participants. If a later participant fails, callers
must re-query usage rather than assume earlier releases were rolled back.
Bindings remain eligible for admission after trimming; invalidation alone retires
them. Idempotent claim release and unknown physical reclamation are independent
of allocator-cache flushing.

`safemlx` exposes immutable graph identity through its existing native safety
boundary; no pointer or native dependency enters portable crates.
`Tensor::publish_parameter` is the neutral, infallible publication hook used by
parameter transaction drivers. MLX revokes derived storage before replacing a
parameter; read-only discovery preserves it. Restored values use ordinary casts
until a new resident materialization enables reuse.
Runtime reports conversion payload separately from its original-parameter ledger.
The backend includes it once in `StaticMemoryReport` device residency and exposes
the subset as `current_device_parameter_conversion_bytes`. Current forecast
composition credits only complete selected bindings with a matching live claim
and budget scope, preserving promoted activation/state sizing. Aggregate subset
subtraction remains only for historical records lacking the scoped policy facts.
Cold forecasts retain the full potential cast allowance and separately bound
possible persistent admission; neither forecasts nor reports populate conversions.
Runtime also owns `ForecastCalibration`, including the labeled attention fallback,
cache-copy overlap, layer workspace multiplier and graph allowance. Architecture
preparation retains selected state, ordinary invocation topology and native
input-score mechanism facts in `PreparedModelDiscovery`. Selected materialization
formats replace checkpoint defaults, including load-time quantization; cached
conversion credit reduces only the remaining selected promotion allowance.
The neutral `GenerationForecastBackend` supplies loaded residency, allocator and
capacity observations plus the actual prompt/instrumentation execution contract.
The MLX adapter derives final-row projection from architecture and execution-strategy
capabilities, shares prompt eligibility with the prefix executor, and leaves existing
unprojected mutable state unknown. It does not reconstruct family semantics.
The facade's prepared-request forecast combines these facts with resolved generation
settings, token counts and application budgets without reopening artifacts or
submitting inference. Observed ordinary and controlled preparations share the same
forecast path. The CLI consumes the facade and owns warning/refusal policy. Unknown workspace or
overhead affects advice, never execution authorization. No allocation ledger or
new funding protocol is involved. See [generation memory](generation-memory.md).

Runtime owns ordinary capture sizing shared by preflight and forecasts: shape/slice
resolution, absolute schedules, per-step diagnostic metadata and checked cumulative
usage. `GenerationForecastBackend::capture_memory_projection` supplies complete
native source/transform cost observations without submission or reservation. MLX
reuses its admission estimator for ordinary logits; deferred sources, partition
transport and intervention evidence without full projection coverage retain an
explicit admitted-limit fallback. Backend cost observations never migrate into
the portable runtime or facade.

Runtime's serializable `CaptureMemoryPlan` caps complete geometry projections by
admitted limits, adds immutable plan storage, and places host/native costs in
physical pools. A default-false backend fact declares whether native transforms
complete before the next prediction; MLX declares this synchronous lifetime, so
runtime uses a per-step peak rather than cumulative native retention. Host records
accumulate across selected occurrences with one compact trace allowance. The facade
composes plans and trace limits from shared ordinary/controlled preparation. Horizon
alternatives recount scheduled usage without accumulating prior forecasts;
continuations retain actual charged history and use the sampler's absolute index,
independent of cache position. Restoration does not refund capture history. Unknown
costs remain explicit and forecasts consume no admission authority.

`SpeculativeForecastBackend<D>` supplies immutable selected-drafter resource
facts. MLX derives independent autoregressive draft geometry from the retained
architecture discovery and reports its own residency, placement and sampling
calibration. The runtime owns single-lane speculative phase accounting: joint
canonical state, rollback/seed copies, K+1 verification and replay, configured
optimistic overlap, distributions, host histories and physical-pool comparisons.
The facade composes target and draft facts for the same prepared request used by
ordinary and controlled speculative drivers and retains a serializable plan for
recomputation. The CLI consumes it, including shorter-output alternatives.
Embedded/feature-conditioned mechanisms remain explicitly uncovered until their
architecture-owned auxiliary state and feature geometry is supplied; native
adapters must not infer those facts from family names or checkpoints.

The existing `safemlx::system` native observation boundary reads macOS Mach host
VM statistics and releases the host port after querying. Available physical memory
is estimated from free (including speculative) plus inactive pages, using the host
page size. The MLX adapter labels this as a point-in-time estimate and exposes the
same observation for host and unified Metal capacity. Runtime fit policy continues
to compare additional request memory with that observation; native queries and
unsafe calls remain in `safemlx`.

Allocator-cache policy is also observed at the native boundary. A vendored MLX
getter reads the CPU, Metal or CUDA allocator's limit under its allocator lock;
the C API and `safemlx` expose it without temporary mutations or cache eviction.
The MLX adapter and selected facade expose both the getter and a setter returning
the previous value. `GenerationMemoryOptions::for_local_backend` composes this
snapshot with retained cache and a labeled graph/driver allowance; the CLI uses
it when no proposed cold cache limit is supplied. This explicit local diagnostic
may initialize the native allocator. The generic options constructor, portable
estimator and cold architecture-selection driver do not perform native queries.

The native allocator also retains cache-policy provenance. Its locked getter
returns the limit and origin together; direct native setters mark explicit
ownership even for unchanged values. Conditional initialization under that same
lock caps only untouched defaults at the adapter's 256 MiB ceiling, or records
an explicit choice to preserve the native policy. The MLX adapter applies this
native cache policy at target/model realization, including controlled and realtime
paths. It never applies it during cold selection or forecasting. Selected facade
configuration exposes automatic, preserved-native and fixed policies; portable
`AllocatorCachePolicyReport` and `AllocatorCachePolicySource` describe observations.
Explicit settings and earlier initialization win across sessions. Reclamation is
still native; existing retention and graph allowances remain in the forecast.
No new unsafe-code boundary is introduced.

Cold prefix support is an explicit architecture declaration in
`ReplicatedTextRequirements`, combined with the backend's settled-prefix mechanism
fact and retained in the selected realization. `SelectedPreparation` applies
prediction, composite and distributed restrictions and exposes
`prefill_chunking_support()`. The shared causal decoder's cold declaration and
loaded trait use the same implementation constant; construction rejects a cold
positive declaration that disagrees with the architecture trait. Cold forecasts
consume this selection rather than a caller hint or workspace-coverage heuristic.
This keeps model semantics out of the native capability provider and allows
chunking support and estimator coverage to evolve independently.

The neutral ordinary/controlled text driver requests bounded prefill through an
optional backend prefix operation. Native adapters preserve compatible prompt
slicing, positions and completion; an intermediate prefix is settled before the
next is submitted. Requests with incompatible observation or specialized input
contracts keep a full prefill pass. Sampling and commitment remain in the shared
driver, and the prefill policy survives controlled continuation.


### Settled continuation memory observations

`eredu-runtime::memory_forecast::ContinuationForecastBackend` supplies neutral
installed-state/frontier facts and horizon-specific native storage envelopes.
Runtime owns decode-only phase composition, cache-copy overlap, physical-pool
comparisons and uncertainty. The MLX adapter validates idle session authority and
uses the existing typed state geometry/capacity growth mechanism through
`ReplicatedTextSession::estimate_installed_control_state_growth`; it never
snapshots or submits work to obtain a forecast. Architecture state geometry stays
in portable family declarations. Core's state estimator also supplies required
logical payload floors, excluding allocation rounding and optional tensors.
Runtime combines those floors with native upper envelopes for both the installed
frontier and the continuation horizon. The facade returns the normalized plan used
by phase estimation; backend snapshot reservations remain upper allowances.

Core's ordinary iterator lends read-only sampler/pending-input observations and
provides explicit completion settlement via `synchronize`. The facade composes
those facts with sampling, constraints, semantic state, admitted capture/trace
limits and live snapshot/branch reservation usage. Controlled forecasts borrow
the shared quiescent continuation boundary. Neither path changes execution limits
or refunds resource budgets. Native state/reservation estimates are upper
allowances, never credits against observed available memory. See
[generation memory](generation-memory.md#mid-session-continuation-forecasts).

### Settled external speculative memory observations

`eredu-core::SpeculativeExecutor::continuation_memory_observation` is an optional,
read-only contract for actual target/draft cache frontiers, retained-capacity and
horizon bounds, current parameter backing, available capacity and allocator policy.
Core checks the canonical lifecycle before calling it. The neutral autoregressive
executor composes two `AutoregressiveMechanisms::memory_observation` results and
its retained assistant seed. MLX reads native state metadata/capacity and residency;
it never creates checkpoints, settles completions or submits tensors for this call.
Embedded and feature-conditioned executors retain the unsupported default.

Runtime owns physical-pool request composition (`loaded_generation_request`),
`GenerationForecastOptions`, calibrated continuation phase accounting and the
object-safe controlled outlook. The facade reexports the options at the existing
API path and supplies selection profiles before lending execution resources.
Native observations refresh mutable residency/capacity on every outlook; geometry
continues to belong to architecture preparation. No family dispatch was added.
Sampler/controller/semantic continuation-storage contracts default to unknown;
the facade supplies tokenizer/parser-specific host facts. Live snapshot/branch
reservations are conservative allowances and never resident credits. Instrumented
retention without coverage leaves an explicit unknown contribution. Forecasting
has no admission, copy, randomness, observation-budget or execution authority.

`SpeculativeForecastBackend::speculative_target_memory_profile` supplies selection
geometry for the isolated lane. MLX retains that geometry even if an unrelated
installed ordinary cache has advanced; forecasting never resets either cache.

### Neutral execution resource descriptions

`eredu-core::resources` owns a descriptive contract below portable families and
mechanisms. It does not depend on runtime forecast types. A `ResourceDescription`
names its execution/module scope, current logical dimensions and requested
horizon. Entries describe parameter storage, mutable state, retained tensors or
temporary workspace. Each backing allocation has one scoped identity and a list
of logical uses: tied slots and views can share backing, while independently
loaded instances, replicas, copied snapshots and cached dtype conversions retain
distinct allocation identities. Equal checkpoint keys do not establish sharing.

Physical pools have separately scoped, authoritative identities. Host and device
access to unified backing uses one pool, while independent devices and different
hosts remain separate. Unknown placement stays explicitly unavailable; guessed
identities cannot establish either shared storage or a shared capacity pool.

`ResourceSize` supports fixed extents and evaluated context-dependent extents.
Actual stored payload (including encoding metadata) and allocation capacity are
separate, non-additive facts. Whole-horizon maxima include the starting boundary
and interior peaks, allowing nonlinear scratch geometry and rounded cache growth
without an equation language or a bytes-per-token assumption. The producer's
ordinary execution contract defines dimension names and units. Missing sizes,
placement and resource-set coverage remain explicit; a missing coverage field
defaults to unspecified, never complete. Public validation checks decoded records
before use but cannot prove producer equations or claims of completeness.
Dimensions are sizing provenance derived from typed invocation/state facts, not
family-specific configuration that applications must supply.

Producers must observe metadata without submitting, polling, synchronizing or
advancing execution, allocating execution resources, or consuming budgets.
Constructing the host description is allowed. Runtime `execution_resources`
producers read the retained ordinary selection, prepared parameter outputs and
local selected state. `describe_prepared_resources` is available on selected and
prepared text contracts and the generic retained text session. Its typed query
describes hypothetical prepared geometry, not an observation of current live
state. The existing continuation outlook is still the live-state operation.

`PreparedParameterSlot::backing` retains the canonical output of the actual
materialization binding plan after resolving binding aliases. Static roles share
the static batch, while execution units and prediction modules have separate
materialization owners. MLX retains this fact while inferring its ordinary binding
recipe metadata; describing resources never reruns inference or reads a payload.
Logical parameter aliases and checkpoint names cannot establish physical sharing.
An aggregate bank slot has no single backing identity and remains incomplete.
This ordinary materialization metadata is also suitable for neutral backends.

Core `StateComponentPolicy` owns checked element geometry at a prefix and across
an inclusive prefix interval. Runtime applies the selected native scalar width,
local state layout and segment offsets without redispatching on a family.
Conditional presence and nonlinear quotient/remainder dimensions retain interval
bounds; frame-local horizons and paged backing decomposition stay explicitly
missing. Sliding visibility does not prove physical history truncation. Resource
capacity is unknown until an implementation contract supplies allocation facts.

Selected execution-unit/static identities name missing workspace and retention
contracts. Reusable mechanisms supply per-invocation sizing facts through the
contracts below; runtime owns generic lifetime/overlap composition separately from fit policy. These producers do not migrate
forecasting, derive embedded prediction invocations, or calibrate kernels.
These producers add no family-specific forecasting dependencies or calibration
constants. Existing forecast records and legacy estimation paths are unchanged.


### Reusable mechanism memory contracts

`eredu-nn::mechanism_memory` owns tensor-independent invocation geometry for dense
and quantized projections, grouped-query attention, causal depthwise convolution,
gated-delta/selective state-space scans, expert dispatch and logical cache appends.
The ordinary `LinearSpec`, `AttentionRequest`, convolution/grouped-linear specs and
scan inputs expose `memory_invocation`; no family imports forecast types or chooses
calibration constants. Logical values describe shapes and nominal scalar types,
not backing allocations. Packed parameter payload/sharing remains owned by the
prepared-binding description rather than being charged as workspace.

`NeuralBackend::mechanism_memory`, `AttentionCache::update_memory_contract`, and
runtime `SamplingBackend::mechanism_memory` are additive read-only hooks whose
default is explicit unknown implementation coverage. Backends own scratch,
conversion buffers, allocation capacity and retention boundaries. MLX shares
attention tile selection/limits and native quantization device selection with
ordinary execution. Its `describe_for_device` accepts an already-selected device
kind without constructing a device or stream. A cache-instance hook reads its
retained frontier, actual shapes/dtypes and existing growth/window policy, rejects
mismatched requests, and distinguishes capacity reuse, new concatenations,
key-only sharing and sliding views. Paged/compressed append decomposition remains
an explicit missing implementation fact.

Native output dtype can differ from nominal portable geometry (mixed dense
parameter/activation dtypes, GGUF host fallback, gated-delta FP32 results). MLX
refines it when selected facts establish the representation; missing device,
bias or bound parameter metadata leaves output storage uncertain. Named lazy
intermediate tensors do not imply they all coexist. Views/no-op casts and
optional retained parameter conversions keep backing unknown until their actual
owner provides identity. MLX does not publish finite per-kernel scratch or
allocator-capacity bounds, so the contract explicitly retains those gaps.

Portable runtime sampling producers use actual `GenerationSampler` or
`MirostatV2Sampler` state, or an admitted `TextGenerationConfig`: rows, vocabulary,
provided history, active top-k/top-p/min-p/penalties and resolved strategy are
mechanism facts. Mirostat's adaptive scalar belongs to host sampler state, not a
fresh device allocation. Host penalty vectors are separate from execution-pool
filtering tensors; arbitrary token filters/custom samplers remain outside this
ordinary producer and require their own contracts.

Runtime `describe_mechanism_resources` converts only backend storage records with
resolved backing identities. It binds host/execution access to authoritative
physical pools, validates compatible facts for shared owners, and retains the
original contract and storage-name bindings. Unknown owner/alias identity,
placement, capacity and retention remain explicit. This bridge never sums a live
peak or treats logical input/output shapes as allocation proof. The lifetime
composer below consumes these descriptions. Production target forecasts now use
ordinary module topology and explicitly labeled planning envelopes; opaque native
scratch and allocator capacity in the raw mechanism contract remain unknown.

### Generic resource lifetime and peak composition

`eredu-runtime::resource_lifetimes` composes prepared `ResourceDescription` and
bound `MechanismResourceDescription` records without family or native dependencies.
A `ResourceLifetimePlan` is an explicit acquisition/completion event schedule.
Acquisitions before earlier work completes describe concurrency; completing work
before the next acquisition describes sequential execution. Native completion,
lazy evaluation and owner release are separate token namespaces. Evaluation
retains graph dependencies until its declared completed boundary, including after
native submission completes. Returned values, installed state and parameter
conversions survive until their owning reference is released. Tokens cannot be
retired twice or acquired again after retirement; owners may remain live at the
end of the requested horizon.

Each authoritative allocation identity is charged once while any reference is
live. Distinct copies retain distinct identities. Multiple views can contribute
different retention references to the same allocation, so releasing a parameter
owner cannot release a surviving lazy graph alias. Repeated descriptions of a
backing must agree on size and physical pool; context-dependent descriptions must
also agree on starting context and horizon. An updated query requires a new plan,
not replacing existing facts silently. Physical pools compose independently;
host/device access to one unified pool uses the same identity. Pool peaks need
not occur at the same event and are not presented as a simultaneous global sum.

`describe_mechanism_lifetimes` binds mechanism retention to schedule-supplied
completion/evaluation tokens and authoritative allocation owners. It checks the
public wrapper's storage bindings, backing semantics and numeric bounds before
composition. Named mechanism intermediates need not coexist: the bridge describes
an internal overlap envelope and does not assert simultaneous current minima.
An explicit schedule may set `live_at_acquire` when that coexistence is known.

Upper bounds sum potentially overlapping per-allocation horizon maxima. Lower
bounds take the larger of proven simultaneous current minima and individual
resource horizon minima, never the sum of independent horizon maxima. Dynamic
current bytes describe the acquisition boundary, not a minimum maintained over
a potentially shrinking lifetime. Fixed bytes with declared retention can
establish later simultaneous minima; unknown retention cannot. Payload and
capacity are independent bounds and are never added to each other. Capacity's
minimum includes its payload minimum. Equal numerical bounds do not upgrade
estimated or observed source bytes to exact metadata; source quality is preserved
across all descriptions of a shared backing.

A plan must explicitly declare resource-set completeness. Missing coverage,
placement, or retention leaves all observed pool upper bounds unknown, with
unplaced allocation identities and reasons retained. Unknown capacity in a known
pool does not erase known payload or bounds in other pools. Unknown lifetime
references remain conservatively live but never manufacture a minimum overlap.
Overflow, conflicting shared facts, malformed descriptions and invalid release
order are errors. Small synthetic schedules validate sequential/concurrent work,
shared backing, independent pools, explicit evaluation batches, dynamic growth
and shrinkage, and unknown facts. This is a read-only composer, not a native
scheduler or budget verdict. The target-workspace evaluator below uses this
composer while preserving the existing report format.


### Generic target workspace forecasts

`ReplicatedTextRequirements` retains optional `TextExecutionTopology` as an
ordinary construction contract. Shared decoder attention, gated MLP and static
output modules reuse their actual `LinearSpec`/`StaticModuleSpec` producers.
Gated convolution uses its existing `GatedShortConvolutionSpec`; packed routing
uses the same `TopKGroupSelectorSpec` and `GroupedGatedProductSpec` as execution.
The topology contains invocation geometry and parameter identities, never byte
calibration constants. Selected projection formats come from exact physical
realizations, preserving tied owners and load-time quantization.

Runtime composes target workspace by reusable mechanism kind and explicit lazy
retention boundaries. Llama, dense Qwen, Nanbeige and dense LFM2 migrate through
these contracts. Shared Gemma2 and packed Qwen/LFM routed experts inherit the
same mechanism composition. New shared-decoder families require no independent
workspace formula. The old `WorkspaceGeometry` wire contract remains a
compatibility input for serialized and manually constructed requests. It lowers its historical aggregate
envelope into the same resource-lifetime plan and peak evaluator, reusing the
ordinary mechanism calibrations. It does not reconstruct missing module identities
or invocation order. Ordinary selected production requests use topology directly.

Finite planning estimates remain estimates. Native scratch, aliasing and
allocator alignment are not proved by logical shapes: explicit calibrated
mechanism envelopes account for their planning allowance without rewriting the
raw phase-4 storage contract as exact. Unsupported equation providers, missing
rank-local topology and unsupported residency lifetimes retain named gaps.
An embedded selection keeps ordinary request-only forecasts explicitly unbounded;
startup forecasts retain target topology only alongside the prediction and
transaction plan. Advanced embedded continuations additionally use settled
observations of installed prediction state and retained target features.
Cold and loaded calls remain read-only and cannot materialize modules, populate
parameter conversion caches or consume observation budgets.

### Embedded prediction resource descriptions

`SelectedPreparation::embedded_prediction_topology` projects additional invocations
from the ordinary architecture descriptor and the admitted speculative strategy.
It retains canonical parameter groups, explicit per-node prediction/context/fused
scopes, target capture geometry and ordinary state policies. Data-flow edges do
not establish native allocation lifetimes. Sequential context preparation consumes
all but the last target row; fused context preparation consumes the full prefix.
The materialized executor and the descriptive contract use the same
`PredictionExecutionMode` operation for that rule.

Actual prediction preparation retains local parameter declarations, physical
module ordinals, auxiliary residency owners and local state policies alongside
`PreparedPredictionPlacement`. These are the modules and state supplied to native
materialization, not a second family forecasting implementation. Shared modules
remain distinct from sequential units and from target parameters borrowed by an
invocation. Cold logical sharing is never upgraded to physical aliasing.

`PreparedModelDiscovery::describe_embedded_prediction_resources` composes these
facts with the exact prepared parameter slots, explicitly bound target-feature
allocations and current residency observations. Prediction slots use their own
materialization batches; borrowed target weights keep the target batch identity.
Two retained feature views of one target allocation add uses, not extra bytes.
Native cached conversions are separate allocations identified by their actual
resident owners. Reusing a target readout conversion from several prediction
heads does not charge another copy. Reading these reports does not materialize
weights, populate conversion caches, evaluate tensors or advance execution.

Logical resource descriptions do not establish native capacity. Missing dtype,
local geometry, target-feature backing, mechanism scratch and evaluation retention
remain explicit. Startup forecasts now compose covered ordinary target and
prediction invocations with the speculative transaction envelopes; settled
embedded continuations substitute observed live state and feature retention.

Qwen hybrid attention, dense feed-forward, shared prediction fusion and static
readout construction expose their actual `LinearSpec`/`StaticModuleSpec` inputs
through ordinary helpers. Both allocation and cold topology consume these same
specifications. The runtime describes fusion projections, sigmoid attention gates
and repeated output invocations generically, without consulting a family name.
Prediction layers retain their actual parameter identities: each depth invokes
the shared fusion and target readout, while its decoder parameters remain distinct.
Selected physical task encodings replace source defaults before projection; any
mixed-precision prediction promotion allowance derives from selected auxiliary
tasks. These facts contain no calibration constants.

A dense Qwen target consisting entirely of full-attention layers and its embedded
predictor therefore use covered mechanisms. A gated-delta target or shared/routed
expert reduction still names the missing invocation mechanism. Other prediction
families retain concrete missing mechanism descriptions (latent/pooling or learned
relative attention, hyper-connections, patterned operators) rather than acquiring
a finite bound from the presence of an embedded head alone. Rank-local prediction
invocation geometry remains unknown until ordinary partition contracts describe
it.

### Settled embedded continuation observations

The speculative executor observes the installed target cache, prediction-layer
frontiers and retained target features through neutral contracts in `eredu-core`.
The architecture adapter forwards the ordinary prediction state's membership;
it contains no forecast calibration or family-specific memory equations. Native
mechanisms report state-capacity allowances and current parameter ownership.
`ReplicatedTextSession` exposes a read-only parameter observation because the
ordinary session is lent to the embedded executor during a controlled run.
The MLX adapter projects its existing residency report through this operation.

Resident parameter conversion allocation and binding records now live in
`eredu-core::residency`, with their existing runtime paths reexported. This lets the
neutral observation carry fresh native ownership after prefill without introducing
a dependency from core to runtime. The runtime retains validation and forecast
credit policy. Embedded conversions belong to the shared target parameter owner;
prediction invocations never acquire a second parameter-residency charge.

The portable runtime composes the phase-8 prediction/transaction plan with observed
current frontiers and horizon-specific capacity. Current feature and seed retention
survive a zero-token outlook. Loading and prefill do not reappear in continuation
phases. Missing native bounds or ordinary invocation mechanisms remain explicit.
Only canonical settled boundaries are eligible; observation cannot poll, submit,
settle, snapshot, advance execution or consume capture/transport/copy budgets.


### Unified workspace evaluation and legacy compatibility

All workspace estimates now flow through `describe_text_workspace` and
`compose_resource_peaks`. Ordinary topology remains authoritative when both old
and new fields are present. Attention score scratch, explicit input-score tile
retention, convolution intermediates, parameter conversion and cache replacement
share the same runtime calibration helpers. The old aggregate workspace evaluator
is removed from generation-memory estimation. No backend or architecture family
selects a compatibility equation.

Old `WorkspaceGeometry` records lack exact projection encodings, parameter
identities and invocation lifetimes. A compatibility lowerer preserves their
historical aggregate linear/logit and overlap contract as one conservative
allocation. This remaining arithmetic is a wire-compatibility obligation, not a
second ordinary topology or family forecast. Serialized report schemas remain
readable, and old finite bounds, unknowns and checked arithmetic are retained.
Frozen pre-migration requests and numeric bounds test this behavior; explanatory
assumption strings may describe the unified evaluator.

### Retention-aware forecast ownership

The neutral `ResidentParameterConversionBinding` carries optional budget-group
attribution. Backends observe that group alongside each current retaining binding
and actual allocation identity; missing historical attribution grants no modern
scoped credit. Runtime forecast composition matches complete selected payloads,
not aggregate conversion bytes or names from another budget. MLX observations are
refreshed for ordinary execution and each speculative participant, including
settled continuation; startup geometry can be reused but startup conversion
observations cannot. Shared native conversion identities deduplicate physical
payload across external participants without merging their admission budgets.

`eredu-runtime` owns the serialized conversion-retention forecast subledger and
potential-admission bounds. The facade composes cold selection facts and a
request-local cold scope without native allocation. Missing cross-rank authority
facts remain unavailable. Native budget scopes replace this local scope after
loading. Possible retained payload is already covered by the pending conversion
envelope, and is never added again or used to clamp temporary workspace. Group
reservations without allocation attribution preserve an unknown phase upper
bound until publication is observed. Historical records without the new optional
facts preserve historical accounting. Forecasting never reserves, evaluates,
settles, trims, or advances execution.
