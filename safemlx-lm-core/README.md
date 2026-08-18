# safemlx-lm-core

`safemlx-lm-core` contains backend-neutral language-model contracts,
transactional scheduling, validation, and portable planning schemas. It does
not depend on MLX, SafeMLX, Metal, CUDA, or another accelerator runtime.

Most applications should use `safemlx-lm`, which reexports canonical core
types at its facade root and supplies the default MLX backend. Backend authors
can depend on this crate to implement whole-model preparation and session-level
prefill/decode without implementing a primitive tensor algebra.

`ModelRuntime<B>` is the canonical downstream owner of a selected backend and
its sole model session. It prepares or accepts a `PreparedModel`, binds the
backend and session once, and submits prefill/decode without exposing an
executable or cache. The same generic client code is exercised by the core mock
backend and the production MLX adapter.

Capability and inspection documents are canonical here as well. `Observed<T>`
is the single fail-closed measurement vocabulary; `ModelCapabilities`,
`StateLayout`, runtime-state reports, memory reports, and admission outcomes do
not name an accelerator. Core performs checked state estimation and admission
policy. `ModelInspectionReport` carries portable readiness and structured
issues. A selected backend derives these values from its executable formats,
prepared inputs, allocator counters, and structural loader constraints.

`load_model(&backend, artifact, options)` is the canonical artifact entry
point. `ModelLoadingBackend` supplies only policy resolution and conversion of
the neutral `ModelPreparationPlan` into its associated model configuration.
The selected backend instance owns devices, queues, and communication state;
generic caller code never passes a native stream or chooses an architecture
loader. `prepare_inspected_model` is the lower-level form for tokenizer-aware
facades which must consume portable metadata from the same
`ArtifactInspection` before transferring it to backend planning.
`ModelLoadError<B::Error>` keeps portable inspection/planning failures separate
from backend policy and materialization failures; a backend error is not
required to absorb `ArtifactError`. `ModelRuntime::load` combines ordinary
loading with session creation under the same typed error boundary.

Multimodal request semantics are portable. `RgbImage`, `Audio`, and `Video`
describe validated decoded media; `MultimodalRequest` preserves exact ordered
text/token/media composition and rendered-chat placeholder binding.
`MultimodalPreparationBackend` lets the selected session preprocess those
values into its existing opaque text-generation prompt while requesting any
checkpoint-defined framing text from the facade tokenizer. Backends retain
resizing, feature extraction, tensors, placement, and processor state.

Core owns the production weight-residency policy and ownership state machine,
not a parallel summary schema. `OffloadPlan` validates stable unit identities,
tier assignments, budgets, prefetch depth, and eviction policy.
`ResidencyLedger` atomically admits batches, reserves and publishes copies,
tracks leases and protected windows, chooses deterministic eviction victims,
and resolves exact transfer generations. Its transitions explicitly return the
copies whose concrete storage a backend must release. `OffloadTelemetry` and
`OffloadReport` account for transfers, prefetch, eviction, process memory, and
backend allocator observations. Deserialization cannot bypass plan invariants.
Backends own only resource materialization, native transfer/completion objects,
and destruction of the storage selected by ledger transitions.
`PrefetchExecutionState` is the reusable bounded execution coordinator for
residency materialization. It owns FIFO admission, coalescing, cancellation
generations, exact work tickets, rollback, failure retention and retry,
demand handoff, and worker telemetry. A backend supplies only its concrete
worker, storage operation, I/O, and completion observation.

Generation orchestration is portable too. `GenerationSequence` owns committed
token order, token budgets, cancellation, and stop/grammar/EOS precedence.
`SpeculativeRound` owns acceptance, replacement/bonus tails, transactional
publication, and the exact verified-input retention count used by a backend
cache commit. Optimistic-prefix reuse and checkpoint/request sampler
configuration validation are pure core decisions. `SpeculativeSchedule` owns
bounded fair action selection, while `propose_block` and `resolve_round` own
proposal sequencing, stochastic accept/reject flow, replacement/bonus
bookkeeping, and transactional sampler/constraint state. A backend owns logits,
distribution and random-state representations, executable assistant heads,
tensor caches, and exact native completion; it cannot maintain a second logical
token or terminal state.
`MtpRequestLifecycle` rejects illegal phase edges and defers cancellation while
a backend verification transaction remains retained.
`SpeculativeExecutor` defines the production whole-session boundary for
prefill, assistant proposals, target verification submission, exact completion
observation, and transactional cache commit. Inputs, logits, target/draft
state, checkpoints, verification outputs, execution contexts, telemetry, and
completion objects are backend-owned associated types. The contract never
models primitive tensor operations or names a native runtime.
`SpeculativeSampling` similarly exposes only complete opaque operations such as
processing logits, sampling, proposal decisions, and committed-token updates;
core never models softmax, indexing, or random tensor primitives.
`PendingSpeculativeVerification` owns each submitted completion, verification
output, cache checkpoint, canonical draft block, and optional optimistic branch
until the exact safe boundary. `resolve_commit_and_publish` then performs
resolution, backend cache commit, branch promotion/discard, telemetry updates,
and output authorization as one transaction. `SpeculativeOutputRuntime` owns
the canonical sampler, token sequence, semantic constraint, cancellation, and
publication lifecycle. `SpeculativeSemanticState` and
`SpeculativeCallbackPublisher` provide the standard backend-independent
transactional event sink. Output errors are separate from executor and sampling
errors, and no publisher can run before cache commit.
`MtpStats`, `MtpSchedulerStats`, and `MtpBatchOutput` are canonical core types.
`SpeculativeRequestTable` is the production request coordinator. It owns the
stable request collection, lifecycle-bound resource slots, cancellation scan,
fair action application, aggregate accounting, and ordered finalization while
holding backend caches, states, randomness, completions, and publishers only as
opaque associated types. Backends do not implement another request table.

Model loading starts here as well. `inspect_artifact` parses `config.json` or a
portable `safemlx-gguf` checkpoint, validates SafeTensors/GGUF catalogs, and
resolves the canonical `ModelKind` without a tensor runtime.
`plan_model_preparation` combines those facts with a fail-closed quantization,
residency, and topology policy. A selected backend consumes the resulting
`ModelPreparationPlan`; core never constructs arrays or executable modules.

Automatic execution planning is backend-neutral. `AutomaticPlanner` owns
policy validation, memory budgeting, residency selection, speculative-drafting
selection, historical-feedback matching, explanations, and every serialized
planning/telemetry schema. `AutomaticPlanningBackend` is deliberately
high-level: an adapter reports hardware and model resources, admits complete
execution-plan candidates, and establishes a bounded residency window. It does
not expose tensors, devices, streams, or allocator objects. The MLX adapter is
one production implementation; core tests the same planner with a mock backend.

Core also owns the process-wide live-cache admission boundary.
`CacheResidencyPool` registers independent cache managers, publishes their
concrete occupancy, and issues exact RAII reservations for device, host,
in-flight transfer, and live-disk resources. Concurrent admission is atomic and
fails closed per resource axis. Backends retain the cache tensors and native
completion objects; dropping a reservation or manager membership releases only
that owner's accounting.

Cache semantics are core-owned as well. Stable block/rank identities, logical
tiers, per-layer attention-state geometry, fixed-state roles,
symbolic shapes, dtype families, and mutable-state residency classes have one
canonical definition in this crate. Constructors and deserialized policies use
the same validation. A backend maps these contracts to its arrays and storage;
it cannot redefine cache geometry or fixed-state ownership.

`CacheBlockLifecycle` is the canonical live-block ownership catalog. It owns
registration, exact lease counts, access clocks, protected-prefix state,
deterministic LRU/LFU selection, transactional truncation replacement, and
mutable-tail frontiers. `CacheBlockStorage` separately owns the legal physical
phase protocol for a block: device demotion, host promotion and rollback,
backing writes, backing reads, and direct release of backed resources. Device,
host, backing, and completion values are opaque type parameters. Exact operation
keys are core types and are bound to one block and generation. A backend owns
the values but cannot construct contradictory resource/phase combinations.
`CacheIoExecutionState` complements that per-block protocol with exact-key
join ownership, bounded worker admission, queued/in-flight cancellation, and
completion publication disposition. Backends retain only opaque task payloads,
physical scheduling, files, buffers, and completion primitives.

Reusable prompt-cache identity and catalogs are also canonical here. Core owns
model and prefix identity, rank-local topology, the versioned manifest,
architecture and token fingerprints, block/state descriptors, compatibility,
coverage, geometry, ordering, and safe relative-path validation. A backend may
publish those catalogs to its chosen storage and materialize their payloads,
but does not repeat or reinterpret their semantic validation.

The production Moshi/PersonaPlex realtime scheduler uses the core request state
machine. Its MLX adapter supplies opaque work, session branches, submissions,
and exact completions while core owns fairness and transactional publication.
Distributed pipeline requests use that same state machine, including poisoning
and exact failure retention. Core also owns protocol framing and fail-closed
schedule, disposition, and completion agreement. Backends implement only the
portable `ConsensusTransport` word collective.

Optional distributed execution is described by `DistributedBackend` and
`DistributedSession`. The contract uses opaque values, typed world/parallel-axis
scopes, explicit fail-closed capabilities, high-level sum/gather/all-to-all and
point-to-point submissions, and exact completion. It deliberately contains no
tensor algebra, communicator, stream, native device, or runtime exception.

See [the backend architecture guide](../doc/backend-architecture.md) for the
ownership boundary and backend implementation checklist.
