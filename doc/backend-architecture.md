# Language-model backend architecture

`safemlx-lm-core` is the backend-neutral foundation of the language-model
runtime. `safemlx-lm` remains the public facade and contains the MLX
implementation. Applications continue to depend on `safemlx-lm` unless they
are implementing or testing orchestration without an accelerator runtime.

The dependency direction is one way:

```text
safemlx-lm-core
      ↑
  safemlx-lm (MLX adapter and model implementations)
      ↑
applications and examples
```

`safemlx-lm-core` has no dependency on `safemlx`, `safemlx-sys`, Metal, CUDA,
or another tensor runtime. Its dependency-graph test enforces that property.

## Core and backend responsibilities

Core owns concepts whose meaning does not depend on tensor representation:

- artifact/model identity, SafeTensors/GGUF header inspection, model-family
  resolution, neutral tensor catalogs, and materialization-route selection;
- validated attention schedules;
- generation phases, cancellation, finish reasons, and semantic output events;
- queue fairness, request/work lifecycle, transactional branch commit/discard,
  exact-completion observation, cancellation, abandonment, and capacity;
- backend/device descriptors and fail-closed capability discovery;
- parallel axes, coordinates, membership, and portable placement descriptions;
- weight-residency policy, atomic ownership/capacity transitions, protected
  windows, deterministic eviction, exact transfer generations, accounting, and
  serialized reports.
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

## Model, session, and completion lifecycle

`Backend::prepare_model` selects one backend and produces an opaque
`PreparedModel`. `Backend::create_session` creates backend-owned persistent
decode/cache state. `BackendSession` then submits prefill and decode operations
against that model and state. There is one session contract: the facade does
not provide an architecture-specific or borrowing execution escape hatch.

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

## MLX mapping

`api::load_model_with_options` first calls the core `inspect_artifact` and
`plan_model_preparation` functions. The resulting `ModelPreparationPlan` owns
the resolved `ModelKind`, raw portable configuration, neutral tensor catalog,
validated load-policy route, and—when applicable—the already-opened
`safemlx_gguf::Checkpoint`. Only then is the plan passed to
`MlxBackend::prepare_model`.

The MLX adapter's `loading` module is the sole materializer. It consumes the
plan, performs exact architecture/module binding, creates MLX arrays and
modules, and applies MLX quantization, residency, stream, mapping, and transfer
semantics. It does not call back into the public facade loader. GGUF headers and
metadata are not reopened: `GgufCheckpoint::from_portable` wraps the core-owned
checkpoint for payload conversion. Combined model/tokenizer loading uses the
same core plan; tokenizer and chat sidecars remain facade concerns.

`MlxBackend` maps session creation to `ModelCache`.
`MlxModelSession` is the sole architecture-erased prefill/decode
implementation. It dispatches matching `Model` and `ModelCache` variants to
their MLX architecture implementation, then wraps the output with
`async_eval_with_event`. `MlxCompletion` owns that exact `Event`, retains the
output array, and synchronizes on early drop so resources cannot be released
before exact completion. Native exceptions are converted at the facade
boundary and never appear in core trait signatures.

`MlxModelInput` owns cloned MLX array handles for every typed input part and its
metadata. Backend submission therefore owns text, image, audio, video, and
embedding inputs without placing MLX arrays or modality-specific tensor layouts
in core.

The public `api::load_model_with_options` route performs format, architecture,
catalog, and policy planning in core before calling `Backend::prepare_model`.
`MlxGeneration` is one
architecture-erased iterator over `Model` and `ModelCache`; every supported
text and multimodal model submits prompt prefill and every cached decode through
`MlxModelSession`. The former per-architecture `ModelGenerate` enum, Llama-only
executor, and borrowing session trait were deleted. There is no public
architecture-specific loading or execution wrapper.

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

## Coupling still present

The current boundary leaves these components MLX-coupled:

- architecture tensor execution, including multimodal and realtime model math;
- MLX exact-completion objects, retained output arrays, and event-backend
  telemetry adapters;
- concrete topology device assignment, communicator construction, collectives,
  and tensor movement;
- per-block MLX cache resources, native completion observation, transfer/disk
  worker scheduling, prompt-cache filesystem publication, safetensors payload
  mapping and materialization (container/header validation, the transition
  protocol, block registration, exact leases, access ordering, protected
  prefixes, victim selection, mutable tails, prompt identity, topology,
  manifests, catalogs, fingerprints, compatibility, block identity, tier
  vocabulary, layer geometry, fixed-state policy, and pure validation are
  already core-owned);
- weight array/host-buffer materialization, native transfer events, retained
  source mappings, and physical-capacity queries (the corresponding ownership,
  admission, eviction, window, lease, and generation state is now in core);
- architecture-specific checkpoint binding/weight recipes, mapped payload
  stores, GGUF decoding, and MLX array materialization (artifact detection,
  model-family resolution, neutral catalogs, and route planning are core-owned);
- sampling, speculative decoding, activation observation, MLX allocator
  sampling, and Metal/CUDA kernels.

The former facade `runtime::residency::policy` module was deleted. It was not
retained as a forwarding namespace. The earlier placeholder core
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
and dtype-family matching; `runtime::cache::residency` does not forward these
types.

Prompt-cache identity, topology, descriptors, options, versioned manifests,
catalog entries, fingerprints, and structural/compatibility errors likewise
have a single definition in core. Architecture implementations construct those
types directly. The MLX adapter converts `ParallelTopology` to the portable
rank description and owns only arrays, mapped shard storage, payload hashes,
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

The production pipeline scheduler constructs `MlxConsensusTransport` for its
world group. That adapter materializes the portable word frame as an MLX array,
runs `all_gather`, waits for its exact completion, and returns rank-major words
to core. Core decides whether work is globally incomplete, complete, failed but
still executing on a peer, or failed and safe to release. Backend errors are
converted at the adapter boundary and consensus mismatches poison the canonical
core scheduler before any new pipeline submission. Prepared branches are
explicitly discarded during poisoning; submitted MLX resources remain retained
until their exact completions resolve. Since further collectives are unsafe,
`PipelineInferenceScheduler::poll_poisoned_completions` releases those resources
using local exact-completion observation only and never publishes state.

The former facade-only `FairScheduler`, `CompletedWork`, and `FailedWork` APIs
were removed instead of retained as wrappers. `Scheduler`, `SchedulerProgress`,
and `SchedulerError` are the only lifecycle types and are reexported by the
facade.

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
6. Add explicit transfer/collective support only when the backend implements
   exact ownership and synchronization semantics. For distributed scheduling,
   implement `ConsensusTransport` over a topology-wide, rank-ordered word
   all-gather and run the core mismatch/failure tests.
7. Pair concrete weight storage with `ResidencyLedger`: reserve full batches
   before materialization, publish only complete copies, release every returned
   eviction descriptor, and resolve the exact generation attached to native
   transfer completion.
8. Register every live cache with `CacheResidencyPool`, publish concrete
   occupancy, and retain each admission token through the exact native
   transfer or persistence transition that owns it.
9. Run the core mock conformance tests plus backend-specific model load,
   architecture-erased prefill, multi-step decode, cancellation, checkpoint,
   and parity tests.
10. Add backend selection at the facade/application boundary. Do not dispatch
   individual tensor operations between MLX and the new runtime.
