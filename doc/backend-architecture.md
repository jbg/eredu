# Language-model backend architecture

This guide defines the boundary between the portable language-model runtime
and an execution backend. It is intended for backend authors and maintainers of
`eredu`.

## Dependency direction

```text
eredu-core        eredu-checkpoint        eredu-nn
        \                |                /
                     eredu-runtime
                           |
                   eredu-architectures
                           |
             eredu composition + backends
```

The neutral crates contain no native accelerator dependency. The `eredu`
facade is also portable when built with `default-features = false`. Its default
`mlx` feature adds reusable MLX capabilities under `eredu::backend::mlx` and
connects them to neutral architectures under `eredu::composition`.

The facade root and `api` namespace expose portable application concepts.
Backend-native types remain in their backend namespace.

## Ownership boundary

Portable crates split tensor-independent ownership by responsibility:

- artifact identity, header inspection, model-family resolution, tensor
  catalogs, and preparation plans;
- validated attention schedules and parallel topologies;
- model capabilities, resource requirements, admission decisions, execution
  plans, and telemetry schemas;
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
- distributed scopes, topology membership, placement, consensus messages, and
  operation capability descriptions.

`eredu-runtime` owns statically dispatched resident and bounded execution,
parameter binding, residency, mutable state, exact completion, distributed-plan
realization, and generation-facing causal-model contracts.

`eredu-architectures` owns model-family configuration, checkpoint contracts,
parameter topology, module construction, state geometry, parallel semantic
plans, and the complete embedding/layer/output lifecycle. Architecture code is
generic over `NeuralBackend` and passes backend-native tensor handles through
unchanged.

A backend owns runtime-specific resources and computation:

- tensors, neural operators, queues or streams, random state, and sampling math;
- model and request cache storage;
- checkpoint payload mapping and tensor materialization;
- native device discovery, allocation measurements, transfers, and kernels;
- communicators and collective tensor operations; and
- exact native completion objects and runtime-specific errors.

Neutral contracts use associated concrete types for these values. They do not
use `Any`, untyped pointers, string-dispatched operations, or erased calls in
per-layer and per-token paths.

## Loading and session creation

Artifact loading has three stages:

1. Portable inspection validates checkpoint metadata and tensor catalogs and
   resolves a model kind.
2. Portable planning combines the artifact description with topology,
   quantization, and residency policy.
3. The selected backend materializes the plan into its executable model and
   creates a stateful session.

`ModelLoadingBackend` implements backend policy and materialization.
`Backend::create_session` consumes a `PreparedModel`, so an executable cannot
be paired with a cache or session created by another backend.

`ModelRuntime<B>` owns the selected backend and its sole session. Applications
normally use `eredu::api::LoadedModel<B>`, which combines that runtime
with tokenizer, EOS, generation-default, and chat-template metadata.

Automatic planning produces a portable `ExecutionPlan`. An
`ExecutionPlanBackendFactory` realizes the complete plan into a target backend,
backend load options, and the selected disabled, embedded, or external drafting
configuration. `LoadedModel::load_execution_plan` and
`LoadedModel::plan_and_load` therefore do not require callers to construct
backend devices, queues, streams, or assistant models.

## Submission and completion

`BackendSession` provides high-level prefill and decode submissions. Associated
types keep prompts, tokens, outputs, session state, and completions opaque.

Every submission returns an exact completion object. A completion observes
only the submitted work; it must not drain unrelated backend work. Schedulers
retain outputs, branches, cache transactions, and other resources until that
completion succeeds. Cancellation prevents publication but does not release
in-flight resources early.

The backend is selected for the entire model/session. Per-operation backend
selection would make cache ownership ambiguous and introduce implicit data
movement. Explicit transfer and collective capabilities cover the cases where
movement is part of the session design.

## Text and multimodal generation

`TextGenerationBackend` submits complete prefill-and-sample and
decode-and-sample operations. The backend owns logits, sampler state,
randomness, and token handles. Portable generation code owns token budgets,
stop and EOS precedence, cancellation, grammar state, and semantic events.

`MultimodalPreparationBackend` accepts portable ordered text, token, image,
audio, and video inputs and produces the backend's ordinary opaque prompt.
Image resizing, signal processing, feature extraction, tensor construction,
and placement remain backend operations.

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

`SpeculativeExecutor` and `SpeculativeSampling` expose high-level opaque
operations for these stages. A backend owns assistant execution, target logits,
probability arithmetic, random state, cache checkpoints, and concrete
completion values. Embedded prediction heads and external assistant models use
the same portable lifecycle.

## Scheduling and cancellation

The core scheduler owns queued, prepared, submitted, committed, failed,
cancelled, and abandoned states. Backend adapters supply submission and exact
completion observation. A submitted request remains transactional until its
completion is observed, even when its client has cancelled or disconnected.

Realtime and distributed request scheduling use the same ownership rules.
Protocol framing and cross-rank schedule agreement are portable; transport and
native collective execution belong to the backend.

## Cache and residency

Core separates logical policy from physical storage:

- `OffloadPlan` and the residency state machines validate budgets, placement,
  prefetch, eviction, leases, and transfer generations.
- Cache lifecycle and storage state machines validate block identity, geometry,
  mutable tails, tier transitions, and exact I/O ownership.
- Process-wide cache admission tracks device, host, transfer, and disk
  reservations atomically.
- Prompt-cache schemas validate identity, topology, fingerprints, coverage,
  and safe payload paths.

A backend supplies concrete tensors, host buffers, mapped payloads, files,
workers, native transfer objects, and allocation observations. It applies core
transition results and releases the physical resources selected by those
results.

## Distributed execution

Portable topology uses data, tensor, pipeline, and expert axes. Core validates
rank coordinates, subgroup membership, balanced ownership, placement, and
operation scopes.

`DistributedSession` is an optional capability of the selected model session.
It exposes high-level sum, gather, variable-count exchange, point-to-point, and
consensus submissions with exact completion. Unsupported operations report an
explicit absent capability. Communicator construction, sharding, movement, and
collective tensor math remain backend-specific.

## MLX implementation

The default MLX implementation is split between backend mechanics under
`eredu::backend::mlx` and the public model facade under
`eredu::composition::mlx`:

- `MlxBackend` owns execution and weight-materialization streams.
- `composition::mlx::MlxModelSession` owns the executable model, cache,
  processor state, and
  optional distributed context.
- neural-network modules implement reusable MLX tensor operations;
- runtime modules implement checkpoint materialization, sampling, caches,
  residency workers, media processing, and collectives; and
- MLX events provide exact completion while retaining arrays and source
  resources required by submitted work.

Model-family selection and MLX composition live under `eredu::composition`;
the backend module does not re-export that facade.
The MLX backend directory contains no Llama configuration, checkpoint names,
cache geometry, parallel head rules, or layer math.

The adapter translates native failures into structured backend errors and
populates portable capability, inspection, memory, admission, and telemetry
reports. MLX arrays, streams, devices, events, groups, and exceptions do not
appear in core or generic facade signatures.

## Implementing another backend

A new backend should:

1. implement the `NeuralBackend` operators required by the architectures it
   intends to run;
2. implement parameter materialization, binding, transfer, and exact submission
   completion capabilities;
3. implement concrete runtime-state/cache storage when cached or paged
   execution is desired;
4. implement collective operations when distributed execution is desired;
5. compose those capabilities with existing neutral architectures and runtime
   policies in the facade;
6. populate portable capability, resource, admission, and telemetry reports;
7. add optional multimodal, speculative, realtime, transfer, or distributed
   capabilities only when supported;
8. realize portable execution plans through an
   `ExecutionPlanBackendFactory`; and
9. run the reusable backend and architecture conformance suites.

Adding a backend never requires implementing Llama, mapping Llama checkpoint
names, or constructing a Llama-specific cache. Backend-specific compiler
artifacts, buffers, command queues, caches, and completion primitives remain
associated implementation types.

## Enforced guarantees

Tests verify that:

- the core dependency graph contains no `safemlx` or native accelerator runtime;
- architecture crates contain no backend imports and MLX backend sources contain
  no model-family knowledge;
- the neutral Llama hot path has no erased per-layer call, host tensor
  conversion, or explicit evaluation boundary;
- the feature-disabled facade compiles and exercises loading, generation,
  capabilities, multimodal preparation, speculative execution, realtime,
  distributed sessions, planning, and residency through a mock backend; and
- the MLX implementation uses the same production contracts for loading,
  prefill, decode, scheduling, and plan realization.
