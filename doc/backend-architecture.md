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

- artifact/model identity and neutral tensor catalog descriptors;
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
decode/cache state. `BackendSession` then submits prefill and decode operations.
The borrowing `SessionExecutor` form supports existing public model owners
without forcing them into a new container.

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

`safemlx-lm::backend::mlx::MlxBackend` maps model preparation to the existing
architecture-neutral `api::load_model_with_options` dispatcher and session
creation to `ModelCache`.
`MlxLlamaExecutor` maps prefill/decode to the existing `LlamaModel` operations.
It submits the returned MLX array with `async_eval_with_event`; `MlxCompletion`
owns that exact `Event` and retains the output array. Native exceptions are
converted at the facade boundary and never appear in core trait signatures.

The public `api::load_model_with_options` route now calls
`Backend::prepare_model` before format and architecture dispatch. The existing
generic Llama generation iterator calls `MlxLlamaExecutor` for prompt prefill
and every cached decode step. Thus general loading and real inference cross the
new boundary; the contract is not a disconnected future-backend interface.
There is no public Llama-specific loading wrapper: architecture dispatch is
owned exclusively by the central loader.

## Tensor and cache ownership

MLX arrays, streams, devices, events, executable layer modules, and concrete KV
cache tensors remain wholly in `safemlx-lm`. Core sees only associated opaque
types. The MLX pager consumes core `CacheBlockId` and `LayerCachePolicy`
contracts directly; it owns only the arrays, transfer completions, and physical
storage that realize them. `CacheBlockLifecycle` owns block registration, exact
leases, access history, protected prefixes, deterministic victim selection,
truncation replacement, and mutable tails. Neutral checkpoint descriptors
describe names, shapes, dtypes, and byte locations; MLX weight stores remain
responsible for validating source formats and materializing arrays. `OffloadPlan`
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

- non-Llama tensor execution, including multimodal and realtime model math;
- MLX exact-completion objects, retained output arrays, and event-backend
  telemetry adapters;
- concrete topology device assignment, communicator construction, collectives,
  and tensor movement;
- per-block cache promotion/demotion and disk-read/write phase ownership,
  prompt-cache filesystem publication, safetensors payload validation and
  materialization, and concrete cache tensors (block registration, exact
  leases, access ordering, protected prefixes, victim selection, mutable tails,
  prompt identity, topology, manifests, catalogs, fingerprints, compatibility,
  block identity, tier vocabulary, layer geometry, fixed-state policy, and pure
  validation are already core-owned);
- weight array/host-buffer materialization, native transfer events, retained
  source mappings, and physical-capacity queries (the corresponding ownership,
  admission, eviction, window, lease, and generation state is now in core);
- checkpoint weight recipes/stores and MLX array materialization;
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
Concrete tier transitions remain in the MLX adapter until device/host/disk
resources and their exact native completions can be split from the transition
protocol as one complete state machine.

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
9. Run the core mock conformance tests plus backend-specific Llama load,
   prefill, multi-step decode, cancellation, checkpoint, and parity tests.
10. Add backend selection at the facade/application boundary. Do not dispatch
   individual tensor operations between MLX and the new runtime.
