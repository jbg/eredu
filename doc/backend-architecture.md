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
`eredu-backend-mlx` crate. The facade exposes a flat, application-facing local
adapter under `eredu::api`; it does not reproduce the implementation crate's
backend or composition module tree. Backend fixture APIs are activated only by
the facade's development dependency; the production `mlx` feature does not
enable backend `test-support`.

The facade root and `api` namespace expose portable application concepts plus
the narrow selected-backend adapter. `eredu-backend-mlx` likewise exposes its
adapter types and operations as an explicit flat root API; its backend runtime,
family composition, and architecture-erased dispatch modules are crate-private.
The feature-gated `testing` namespace exists only for cross-crate integration
fixtures and is not enabled by the production `mlx` feature.

## Ownership boundary

Portable crates split tensor-independent ownership by responsibility:

- artifact identity, header inspection, the model-configuration resolver
  contract, tensor catalogs, and preparation plans;
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

The architecture configuration registry is the sole owner of Hugging Face
family aliases, GGUF `general.architecture` spellings, nested-wrapper
normalization, assistant-family identity, and the exhaustive dispatch to
family parsers. `eredu-core` accepts that registry through
`ModelConfigurationResolver` while inspecting both SafeTensors and GGUF
artifacts. The resolver returns an open canonical family string and the neutral
`LoadingProtocol` required to prepare the artifact. Core routes that protocol;
it neither recognizes family strings nor exposes an exhaustive family type.
The typed `ModelKind` and `GgufArchitecture` identities, their aliases, the
family-to-protocol mapping, and family-specific GGUF structural admission live
in `eredu-architectures`. Facades and concrete backend adapters select that
shared registry. Backend composition converts the resolved canonical family
through the architecture registry and consumes architecture parser outputs,
never a second raw `model_type` or `general.architecture` dispatch table.

Architecture checkpoint modules also own canonical name translation and the
complete derived-weight recipe catalogs for static modules, execution units,
and independently resident experts. Concrete backends consume those neutral
recipes for binding, materialization, sharding, and transfer; they do not
redeclare family-specific stacking, concatenation, reshaping, normalization,
or recurrent-weight transformations. This includes format-dependent recipes
such as fused projection assembly and recurrent transition conversion; backend
composition may inspect recipe outputs but does not construct their equations.
These catalogs are model-wide and configuration-derived. Backend adapters may
filter their outputs to the parameters present in a static module, execution
unit, or independently resident bank; that filtering does not transfer source
name discovery or recipe construction into backend composition.
Architecture checkpoint APIs also derive rank-local independently resident
expert recipes, including segmented fused-projection selection; backend
composition binds those recipes without reconstructing the segment equation.
They also normalize physical checkpoint format metadata onto canonical runtime
parameters, including fused expert projections, so each backend consumes the
same family-specific quantization identities.

Parameter-class selection follows the architecture's validated parameter
description. Backends select exact targets by semantic role, retain the
declared static or execution-unit owner, and obtain sharding prefixes from the
architecture's canonical unit path. Checkpoint-only physical companions or
source-layout projections are exposed explicitly by the architecture
checkpoint contract. Loading, residency, and quantization code must not infer
any of those facts from substrings in checkpoint names.

Pinned parameters additionally expose an architecture-owned bindable static
module interface. The parameter description selects the roles owned by a
partition, and the architecture resolves each role to its parameterized module
for binding construction or mutable population. A backend may materialize and
shard those visited modules, but must not map roles to family-specific fields,
residency identifiers, or checkpoint roots. Stable parameter identities on the
visited module are the checkpoint-binding namespace. Every pipeline-composed
family, including multimodal and hybrid graphs, implements this interface;
concrete backend adapters use one shared visitor rather than family binding tables.
Architecture implementations also provide any released-checkpoint rewrite
recipes for their static modules, so alias and fused-layout handling does not
reintroduce checkpoint roots into a backend.

Routed expert banks retain and expose their architecture-owned construction
specification. Resident, cached, distributed, and future backend execution
paths all consume that same geometry, projection encoding, bias layout, and
activation policy; backend residency adapters must not reconstruct a parallel
family descriptor from model arguments. Architecture APIs derive localized
expert-bank specifications for placement-resolved expert counts and projection
widths while preserving canonical parameter identities and physical formats;
backend composition only materializes those returned specifications.
Distributed expert callbacks also
carry whether the requested result is globally complete or a rank-local
tensor-parallel contribution, so EP recombination preserves the reducible and
post-reduction terms without inventing or repeating a TP collective.

Architecture capability estimates likewise own native and effective context,
accepted modalities, cache ownership and sharing, attention windows,
recurrent-state geometry, and backend-neutral scalar state layouts. Concrete
backends select the loaded architecture estimate, apply their physical state
scalar width, and add live allocator, residency, and system-memory
observations; they do not reconstruct family state geometry.

For SafeTensors materialization, architecture preparation also identifies the
checkpoint parameter that establishes runtime-state dtype and resolves its
schema-declared physical aliases against the inspected catalog. A concrete
backend maps that resolved dtype to its physical scalar width; generic backend
composition must not recognize family checkpoint names or silently choose a
width when the declared source is missing. Backend-native encoded formats may
instead report their known materialized activation dtype.

Runtime state follows the same contract at materialization time. Architecture
`StateLayout` values are authoritative for per-layer attention windows,
append-only pooling streams, and named segment boundaries and lifetimes.
Concrete backends derive native cache objects directly from those policies;
model-family arguments are not a second source of state geometry. Composite
model layouts, such as a target decoder plus embedded prediction state, are
assembled by the architecture before a backend consumes them. Architecture
state identity also declares the composite global layer count and each
segment's processed-token frontier offset; backends must not reconstruct
target/prediction boundaries or shifted-prediction offsets.

Pre-materialization capabilities follow the same rule. The normalized
architecture reports whether independently addressable routed experts exist;
the selected backend intersects that fact with its expert-cache
materializers. Core selects the neutral expert-cache route but does not infer
support from a family name, because one family may contain both dense and MoE
variants. Nonresident SafeTensors load-time quantization follows the same
intersection: the architecture declares whether its normalized parameter
topology can be transformed before bounded materialization, and the backend
declares whether its family composition implements that route. Core preserves
the neutral quantization and residency request without maintaining a family
allowlist.
The same architecture capability report carries a typed parallel plan for
tensor sharding, pipeline staging, and expert partitioning. Each axis is
declared from the parsed family variant rather than inferred from parameter
addressability or a broad family identity. Independent expert residency
remains a separate artifact capability because it does not imply an
expert-parallel execution plan. Distributed backend preflight consumes these
facts from that exact normalized report instead of reconstructing support from
raw or wrapper `model_type` values.

The MLX backend materializes every active expert-parallel axis through its
single distributed-stage loader. Pure EP, PP+EP, TP+EP, and TP+PP+EP therefore
share architecture capability validation, expert assignment, residency, and
execution; there is no separate family-dispatched EP model loader. The same
path covers every architecture whose neutral parallel plan declares EP,
including multimodal and hybrid-state MoE families.

Artifact inspection also consumes architecture-derived input modalities from
that exact normalized configuration. Backends translate those neutral flags
into report and build-feature readiness, but do not infer image, audio, or
video support from a family name. Text-only and partially multimodal variants
therefore do not acquire processor or feature requirements they cannot use.
For composite GGUF artifacts, the architecture additionally owns the plan that
maps a validated sibling media projector to the resulting input modalities.
Inspection applies that plan after projector discovery and structural
validation, so expected modalities and multimodal readiness describe the same
validated artifact composition.

Multimodal preprocessing starts with an architecture-owned processor plan.
That neutral plan parses family model and processor metadata, selects released
defaults, declares framing token IDs and text, chooses frame sampling and
grouping, and derives resize, normalization, patch, and signal-feature policy.
Audio feature plans specify the analysis window, exact padding and frame-count
convention, frequency bounds, mel scale and normalization, spectrum value, and
logarithm as well as their dimensions and floors.
Concrete backends execute those declarations: they resize pixels, extract
features, pack patches, and construct native tensors. A new backend therefore
consumes the same family protocol instead of reimplementing it.

Realtime prompt protocols follow the same ownership rule. The neutral Moshi
architecture declares PersonaPlex's released silence, sine-conditioning, and
text-padding tokens, system-text framing, accepted prompt geometry, and
ordered forced-frame plan. A concrete backend only slices and materializes
native arrays from that plan and enqueues the resulting frames; it does not
redeclare PersonaPlex application policy.

Prepared-media admission follows the same boundary after tensor construction.
Architecture media plans validate family payload shapes,
patch/window/pooling geometry, valid-position masks, and artifact-specific
modality policy, then report decoder positions and conservative scalar
workspace. Concrete backends only extract shapes and small metadata values
from native arrays, apply physical scalar widths, and account for the arrays'
actual byte sizes.

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

Artifact loading has four stages:

1. Portable inspection validates checkpoint metadata and tensor catalogs and
   asks the architecture registry for a canonical family and neutral loading
   protocol.
2. The selected backend validates the requested policy against normalized
   architecture facts and its own capabilities.
3. Portable planning combines the artifact description with topology,
   quantization, and residency policy.
4. The selected backend materializes the plan into its executable model and
   creates a stateful session.

For GGUF artifacts, `ArtifactInspection::validated_gguf` is the authoritative
handoff from stage 1. Core validates format-generic tensor-count, required
metadata, and embedding floors using the submitted metadata prefix. The
selected architecture registry resolves the family spelling and applies
family-specific structural admission. Backends may wrap the resulting portable
checkpoint handle and add architecture or device compatibility checks, but do
not repeat either portable admission layer.

`ModelLoadingBackend` implements backend policy, architecture/backend
capability intersection, and materialization.
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

Architecture inspection also reports embedded-draft depth from the normalized
family configuration in `ModelResourceProfile`. The neutral automatic planner
consumes that observation directly; concrete backends must not infer family
semantics by searching raw configuration documents.

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
and placement remain backend operations, while architecture processor plans
declare the family-specific transforms, sampling, framing, and packing
geometry those operations implement.

Architecture-owned ingress plans also bridge prepared media into generic model
inputs. They derive family-specific placeholder spans, pooling geometry,
padding logits, and subsampling masks from portable extents. Concrete backends
materialize those declared values and may perform generic padding or dtype
conversion, but must not independently reconstruct family mask or geometry
policy. Gemma 4 exposes this contract through its vision and audio ingress
part/batch plans.

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

Architecture families also own external-assistant compatibility proofs. Those
proofs match target-state publishers, hidden and rotary geometry, target-layer
captures, and vocabulary requirements before a concrete backend composes the
two executables. Backend composition may enforce the proof, but must not
restate family-specific compatibility rules.

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
  and safe payload paths. Architecture fingerprints use a backend-neutral
  `eredu` domain; obsolete backend-branded domains are intentionally
  incompatible.
- Architecture-owned shared-decoder configurations must provide an
  authoritative fingerprint covering all construction, equation, state, and
  encoding policy; the shared decoder does not synthesize compatibility
  identities from a partial policy subset.

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

The partition also carries an architecture-owned boundary schema. That schema
declares every auxiliary tensor's stable role, canonical order, symbolic shape,
logical dtype, and configuration-dependent cardinality, and it owns conversion
to and from the family's typed boundary value. `eredu-runtime` validates and
resolves batch and sequence dimensions. A concrete backend only maps the
logical activation or exact integer dtype to its native dtype, allocates the
declared receive buffers, validates produced tensors, and transports them. It
must not reconstruct boundary geometry from model-family arguments.
Distributed placement dependency routes consume that same schema directly. A
concrete backend must reject inactive dependency routes with tensors and active
routes whose exact cardinality, ordered shapes, or physical dtypes differ from
the resolved architecture boundary; backend-local string or optional-field
schemas are not an alternate wire contract.

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
  optional distributed context. The prepared `MlxModel` wrapper exposes
  neutral capabilities and telemetry while keeping its executable kind and
  architecture-specific payload private.
- neural-network modules implement reusable MLX tensor operations;
- runtime modules implement checkpoint materialization, sampling, caches,
  residency workers, media processing, and collectives. Their production
  checkpoint API consumes canonical parameter names and architecture-derived
  recipes; generic matching applies only explicitly configured prefix rewrites
  and backend-module parameter structure, and never invents family aliases or
  exposes parsers for physical family checkpoint names;
- GGUF family selection and portable family-specific structural admission are
  architecture-registry concerns. MLX composition resolves the already
  admitted spelling through the same registry, validates native checkpoint
  compatibility once, and passes an admitted source to resident,
  tensor-parallel, pipeline, or expert family loaders. Reusable backend runtime
  modules neither parse `general.architecture` nor invoke family composition;
- generic layerwise policy construction derives its execution graph and unit
  layout directly from the concrete neutral architecture before binding
  checkpoint units; composition cannot supply or reconstruct that layout.
  Unit-binding adapters receive the validated group-local address and the
  architecture's canonical `unit_path`, rather than a flattened residency
  ordinal, so heterogeneous group boundaries and parameter roots remain
  architecture-owned; and
- MLX events provide exact completion while retaining arrays and source
  resources required by submitted work.

Model-family definitions, equations, checkpoint schemas, and state geometry
remain in `eredu-architectures`. The backend crate owns only the MLX binding,
materialization, and execution adapters. `eredu` delegates through neutral
contracts and exposes only the narrow selected-backend types needed by
applications; the backend crate never depends upward on the facade.

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
  and exercise the public contracts through mock backends, with the facade's
  complete normal, build, and development dependency graph remaining free of
  MLX; MLX-specific facade tests are owned by the separate `eredu-mlx-tests`
  package; and
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
