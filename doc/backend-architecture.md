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
- residency policy, resource plans, accounting, and serialized reports.

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
types. Neutral checkpoint descriptors describe names, shapes, dtypes, and byte
locations; MLX weight stores remain responsible for validating source formats
and materializing arrays. Neutral residency plans account for logical resources;
MLX managers execute transfers and report concrete memory counters.

## Coupling left for the next milestone

The first vertical slice intentionally leaves these components MLX-coupled:

- non-Llama tensor execution, including multimodal and realtime model math;
- the facade's distributed pipeline lifecycle machine and MLX event-backend
  reporting;
- concrete topology device assignment, communicator construction, collectives,
  and tensor movement;
- cache-residency array storage, prompt-cache materialization, and transfer
  buffers;
- checkpoint weight recipes/stores and MLX array materialization;
- sampling, speculative decoding, activation observation, memory counters, and
  Metal/CUDA kernels.

The neutral scheduler in core now owns the production single-rank realtime
request lifecycle. Moshi and PersonaPlex queueing, fairness, deadlines,
branching, commit/discard, cancellation, abandonment, capacity accounting, and
telemetry execute in `safemlx-lm-core`. The facade adapter supplies only stable
work descriptors, MLX submission closures, exact MLX completion observation,
and retained arrays. The two Metal realtime lifecycle tests exercise this path.

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
converted at the adapter boundary and consensus mismatches poison the facade
scheduler before any new pipeline submission.

The remaining scheduler duplication is the distributed pipeline lifecycle
machine in the facade. A later milestone can make it delegate queueing and
transaction publication to core while injecting consensus at its prepare,
cancel, and completion boundaries. MLX events and retained arrays should remain
facade-owned.

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
7. Run the core mock conformance tests plus backend-specific Llama load,
   prefill, multi-step decode, cancellation, checkpoint, and parity tests.
8. Add backend selection at the facade/application boundary. Do not dispatch
   individual tensor operations between MLX and the new runtime.
