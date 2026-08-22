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
                  eredu-backend-mlx
                           |
                         eredu
```

The neutral crates contain no native accelerator dependency under any feature.
The `eredu` facade is also portable when built with
`default-features = false`. Its default `mlx` feature selects the optional
`eredu-backend-mlx` crate. Compatibility re-exports keep MLX capabilities
available under `eredu::backend::mlx` and family adapters under
`eredu::composition`. Backend fixture APIs are activated only by the facade's
development dependency; the production `mlx` feature does not enable backend
`test-support`.

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

Architecture checkpoint modules also own canonical name translation and the
complete derived-weight recipe catalogs for static modules, execution units,
and independently resident experts. Concrete backends consume those neutral
recipes for binding, materialization, sharding, and transfer; they do not
redeclare family-specific stacking, concatenation, reshaping, normalization,
or recurrent-weight transformations.

Architecture capability estimates likewise own native and effective context,
accepted modalities, cache ownership and sharing, attention windows,
recurrent-state geometry, and backend-neutral scalar state layouts. Concrete
backends select the loaded architecture estimate, apply their physical state
scalar width, and add live allocator, residency, and system-memory
observations; they do not reconstruct family state geometry.

Backend types also declare the optional neural forward operators they support.
Architecture constructors preflight family-owned operator requirement sets
before allocating modules or loading parameters, so a missing recurrent scan,
sparse-attention primitive, normalization, activation, or collective rejects
composition instead of failing on the first forward pass. Concrete backends
must keep that declaration aligned with their `NeuralBackend` implementation.

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

`ArchitecturePartition` is publicly constructed only through
`from_architecture`, which derives its execution graph and unit layout from the
concrete neutral architecture. Backend placement code supplies rank-local
ranges and ownership, but cannot substitute a caller-built topology that only
resembles the architecture it will execute.

`DistributedSession` is an optional capability of the selected model session.
It exposes high-level sum, gather, variable-count exchange, point-to-point, and
consensus submissions with exact completion. Unsupported operations report an
explicit absent capability. Communicator construction, sharding, movement, and
collective tensor math remain backend-specific.

## MLX implementation

The complete concrete implementation lives in `eredu-backend-mlx`, split
internally between reusable backend mechanics and family/backend composition:

- `MlxBackend` owns execution and weight-materialization streams.
- `MlxTensor` is a transparent, zero-copy wrapper around `safemlx::Array` and
  is the sole MLX implementation of `eredu_nn::Tensor`;
- `composition::mlx::MlxModelSession` owns the executable model, cache,
  processor state, and
  optional distributed context.
- neural-network modules implement reusable MLX tensor operations;
- runtime modules implement checkpoint materialization, sampling, caches,
  residency workers, media processing, and collectives;
- generic layerwise policy construction derives its execution graph and unit
  layout directly from the concrete neutral architecture before binding
  checkpoint units; composition cannot supply or reconstruct that layout; and
- MLX events provide exact completion while retaining arrays and source
  resources required by submitted work.

Model-family definitions, equations, checkpoint schemas, and state geometry
remain in `eredu-architectures`. The backend crate owns only the MLX binding,
materialization, and execution adapters. `eredu` delegates through neutral
contracts and may re-export backend APIs for compatibility, but the backend
crate never depends upward on the facade.

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

## Guarantees and verification

The repository mechanically verifies stable dependency and behavior boundaries:

- `eredu-core`, `eredu-nn`, `eredu-runtime`, `eredu-architectures`, and
  `eredu-codec` dependency-graph tests reject upward or accelerator-runtime
  dependencies;
- `eredu-backend-mlx` dependency-graph tests reject a dependency on `eredu`;
- the feature-disabled `portable_facade` and `backend_conformance` suites compile
  and exercise the public contracts through mock backends; and
- architecture, runtime, and backend conformance tests cover the relevant
  production contracts.

The following ownership constraints are review rules rather than claims made by
source-layout tests:

- `eredu-architectures` contains model-family policy but no concrete backend
  imports;
- reusable modules in `eredu-backend-mlx` contain backend mechanics but no
  model-family configuration, checkpoint naming policy, state geometry, or
  layer equations;
- native dependencies remain behind facade features, preserving the
  `default-features = false` build; and
- model families and their neutral execution remain available without enabling
  a concrete backend feature; feature gates belong on backend adapters, not on
  entire families; and
- backend implementations do not depend upward on `eredu::api` or portable
  facade runtime orchestration.

These rules are recorded in the repository-root
[architecture rules](../AGENTS.md). We intentionally do not enforce them by
scanning Rust source for substrings or by asserting a particular file layout:
those checks couple architecture to names and migration artifacts instead of
semantic ownership. Repeated violations should be made unrepresentable with a
crate boundary, visibility change, or manifest-level dependency check.
