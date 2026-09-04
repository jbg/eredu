# Language-model backend architecture

This guide defines the boundary between the portable language-model runtime
and an execution backend. It is intended for backend authors and maintainers of
`eredu`.

## Dependency direction

```text
eredu-core        eredu-checkpoint        eredu-nn
        \                |                /    \
                     eredu-runtime          eredu-codec
                           |                     :
                   eredu-architectures           : optional
                           |                     :
                           +-- eredu-backend-mlx -+
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
to `eredu-backend-mlx`. This includes logical distributed subgroups and
axis-aware gathers composed over MLX's native collective primitives;
`safemlx::distributed` exposes only MLX-native groups and communication calls.
Model-family construction and equations remain in
`eredu-architectures`; MLX-specific realization remains in
`eredu-backend-mlx`.
`eredu-codec` owns backend-neutral neural audio codec architectures and depends
on the neutral tensor contracts in `eredu-nn`. A concrete backend may depend on
`eredu-codec` for an optional codec binding; the codec crate does not acquire a
backend feature or dependency in the other direction.
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
directly depend on `eredu-nn` or `eredu-checkpoint`. Media processing is enabled
only through concrete image or audio capabilities. Backend feature diagnostics
name the active public spelling; published Cargo features are selectable API,
not a privacy mechanism.

The facade root exposes only the `api` and `runtime` namespaces. The `api`
namespace exposes facade-owned operations and the opaque selected-backend
adapter. `LocalBackendFactory`, `LocalModel`,
`LocalPlannedModel`, `LocalDrafting`, `LocalPrompt`, and local generation
iterators are facade-owned wrappers with private backend state. The selected
factory does not implement the public neutral factory traits, and the facade
does not name or alias the concrete backend, session, prompt, drafter, token,
completion, or error types in its public selected API. The implementation crate
exports composition-owned adapter factories at its flat root and deliberately
makes its reusable `backend` module tree public for backend authors. Family
composition and architecture-erased dispatch remain crate-private. Raw native
binding APIs retain their canonical `safemlx` paths rather than being
re-exported by the implementation crate. Native
facade integration tests realize execution plans through facade-owned methods;
tests that require backend facilities live in `eredu-backend-mlx`, while facade
sampling-policy tests use neutral sampling traits and mock backends. Direct
backend consumers import neutral contracts from their owning crates.

The facade does not re-export dependency-owned types. Architecture identities
and preparation plans come from `eredu-architectures`; artifact, execution,
generation, media, realtime, scheduler, and prompt-cache contracts come from
`eredu-core`; tokenizer contracts come from `eredu-text`; and runtime
infrastructure comes from `eredu-runtime`. This gives every public type one
canonical import path and prevents a facade release from committing to aliases
for operations it does not own. In particular, the facade exposes no
prompt-cache types because it exposes no prompt-cache operation.

The selected adapter exposes only facade-owned causal and realtime model,
planning, drafting, scheduler, completed-step, and error wrappers. The realtime
factory loads an architecture-owned preparation directly into the facade model.
The facade materializes portable inputs and observes portable outputs while
concrete associated types and handle-oriented constructors remain private.
Explicit native sessions and token handles remain backend-author concerns;
streams and distributed collective groups come directly from `safemlx`.

Application targets depend on `eredu` for facade operations and directly on
the neutral crates whose public values they construct. Infrastructure-aware
clients such as the CLI likewise import low-level policies from their owning
crates. The
selected-local-backend API owns device-plan creation, process runtime
configuration, allocator telemetry, and diagnostic benchmarks, while
`LocalModel::synchronize` is the sole application-facing synchronization entry
point. These APIs do not expose native tensors, streams, devices, or random
state. Direct native binding access remains an explicit backend-author concern
through `safemlx`; native session adapters remain in the implementation crate.
Those adapters own native resources and outer type erasure, while shared
replicated-text execution and stateful lifecycle orchestration remain in the
neutral runtime. Neither surface is an application dependency.
`LocalLoadOptions` and `LocalInspectionOptions` contain only neutral
quantization, residency, and session-capability policy, while native
device-bound contexts are selected only by backend tooling. `LocalBackendError`
records facade operation context and a diagnostic message without exporting
native error variants. Portable execution plans remain the application surface
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
likewise exchange portable host token frames and observations through
`RealtimeBackend::materialize_input` and `RealtimeBackend::observe_output`.
Realtime backends report the exact loaded route's fail-closed
`SessionCapabilities`; realtime loading validates requested capabilities before
checkpoint payload materialization just as ordinary model loading does. A
realtime route must not advertise named activation inspection unless it exposes
the corresponding inspection contract.
Distributed inspection is rank-local: every rank participates in the same
production collective and point-to-point execution, each rank returns only the
globally named units it owns, and only the logits-owning rank returns
`model.logits`. `eredu-core::MODEL_LOGITS_OBSERVATION_PATH` is the canonical
owner of that semantic path; family composition, distributed pipelines, and
architecture-erased sessions must use it rather than defining backend-local
names. Inspection never performs an implicit cross-rank host gather.

Backend-neutral parity, distribution metrics, timing summaries, evidence, and
evaluation drivers live in `eredu-evaluation`. Concrete backend examples only
select execution contexts and materialize model and codec artifacts; they do
not own comparison thresholds or reference policy. Backend-specific validation
inputs and reference-fixture generators live with the backend that consumes
them, so every published backend package contains the tooling required by its
examples; they do not live under the facade.

## Ownership boundary

Portable crates split tensor-independent ownership by responsibility:

- `eredu-checkpoint` owns canonical SafeTensors index parsing, exact
  index-to-shard-header validation, and shard-path admission. Inspection and
  conversion tooling consume strict discovery, while neutral stores consume
  the same parsed and admitted catalog but validate each opened payload header
  exactly and lazily, so selective loads detect every mismatch in a local shard
  without reading remote-only shards. None reinterpret
  `weight_map`;
- artifact identity, header inspection, the model-configuration resolver
  contract, tensor catalogs, and preparation plans live in `eredu-core`;
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
fingerprints is portable compatibility policy. Backend factories receive that
proof and its shared fingerprint; they own assistant placement, materialization,
and architecture compatibility, but do not decide tokenizer compatibility.

`eredu-runtime` owns statically dispatched resident and bounded execution,
parameter binding, residency, mutable state, exact completion, distributed-plan
realization, and generation-facing causal-model contracts.

`eredu-architectures` owns model-family configuration, checkpoint contracts,
parameter topology, module construction, state geometry, parallel semantic
plans, and the complete embedding/layer/output lifecycle. Architecture code is
generic over `NeuralBackend` and passes backend-native tensor handles through
unchanged.

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
provenance: the canonical admitted shard, physical tensor identity, and the
selected logical output. This distinguishes multiple outputs converted from a
single GGUF tensor and prevents a materializer from rediscovering shard
membership after selection. Validated parallel topology fields are private;
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

The MLX adapter consumes its load request before neutral selection completes.
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

Architecture-owned typed dispatch admits replicated Llama/Mistral, dense
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
sequencing. A non-MLX backend supplies its own tensor, policy, state,
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
record contains only its stable bank key, exact compact bindings, and source
and selected byte geometry. The selected geometry includes any load-time
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
the facade boundary. Resident routed execution reports no addressable bank.

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
same unambiguous key. Complete-model
materialization stores the already resolved `ModelKind`
alongside the concrete model implementation; it does not reconstruct family
identity from the effective type. Architecture-owned typed partition dispatch
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
execution objects. Its physical sources are the exact admitted sources
not claimed by the projected target schema, rather than a backend prefix
filter. MLX materializes those units against the target's selected local
geometry carried by the handoff and invokes them through the neutral prediction-target operation
hook. This does not make every prediction family or topology supported:
extensions without a dedicated typed composition fail before target payload
construction. An excluded prediction selection cannot fall back to a complete
or pipeline target shell.

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
allowlist. The composition-owned MLX load request carries `QuantizationRequest`, whose
variants describe load-time transforms only; checkpoint storage encodings such
as native GGUF blocks remain internal to artifact inspection and
materialization.

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
mechanisms normalize pixels, extract requested audio features, inspect small
metadata values, and construct native tensors; architecture code owns frame
selection, patch packing, framing order, metadata, and the ordered
`PreparedModelInput`. A new backend therefore supplies media and tensor
mechanisms without implementing a family processor.

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
ordered forced-frame plan. A concrete backend only slices and materializes
native arrays from that plan and enqueues the resulting frames; it does not
redeclare PersonaPlex application policy. Realtime model identity is likewise
architecture-owned: backend models return the Moshi architecture's
`EffectiveModelType` directly. Backend adapters must not re-export or wrap that
type; direct consumers import it from `eredu_architectures::moshi`.

Realtime artifact loading also crosses the backend boundary as a neutral
architecture preparation. Moshi inspection parses optional native defaults or
released PersonaPlex configuration, resolves the confined `moshi_name` or
indexed checkpoint source, validates the strict SafeTensors contract, and
publishes canonical binding recipes in `RealtimePreparationPlan`. Concrete
backends consume that plan and load tensor payloads; checkpoint-layout identity
names the physical Moshi or PersonaPlex SafeTensors namespace and never a
concrete backend. Backends do not receive a raw artifact path or reinterpret
family configuration and filename policy.

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
4. The selected backend adapter realizes native mechanisms from the plan. For
   replicated text it passes those mechanisms to the neutral session
   constructor and erases the completed typed session at the outer boundary;
   other execution classes retain their dedicated neutral contracts.

Concrete backend preflight reports mechanisms rather than family support or a
parallel execution-class table. Architecture dispatch is authoritative for
replicated text. For the generic replicated and routed classes, MLX performs
only loading-protocol validation before execution-class dispatch and then lets
neutral selection validate the concrete mechanisms. Excluded-family binding
tables apply only to the `Other` class. Addressable parameter banks requested
for dense replicated text are rejected explicitly rather than being silently
discarded. Complete-model GGUF load-time quantization is encoded in the
selected per-parameter tasks, so selection and materialization cannot disagree
about whether a format or lowering is accepted. Execution classes handled by a
distinct protocol, such as realtime Moshi, do not enter replicated dispatch.

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

`ModelPreparationPlan` is the one-shot authority for stage 4. Materializers,
including partitioned materializers, consume its inspected configuration,
primary and companion checkpoint handles, and selected route directly. They
must not reopen an artifact to rediscover configuration, checkpoint metadata,
or sibling filename policy after planning; payload stores may still map weight
members during materialization, but those reads do not replace the plan's
configuration, checkpoint metadata, companion selection, or route.

`ModelLoadingBackend` implements backend capability reporting, native
materialization, and outer adaptation. `BackendProvider::create_session`
consumes a `PreparedModel`, so an executable cannot be paired with state or
session mechanisms created by another backend. Replicated prepared models
already contain the neutral lifecycle paired with their concrete mechanisms.

`ModelRuntime<B>` owns a backend and its sole session. Backend-generic clients
use `eredu::api::LoadedModel<B>`, whose runtime remains private while portable
model operations stay generic. Applications using the selected local backend
use `eredu::api::LocalModel`, which also erases the backend parameter and maps
native loading and generation failures into `LocalBackendError`.

Automatic planning produces a portable `ExecutionPlan`. An
`ExecutionPlanBackendFactory` realizes the complete plan into a target backend,
backend load options, and the selected disabled, embedded, or external drafting
configuration. `LocalModel::load_execution_plan` and `LocalModel::plan_and_load`
therefore do not require callers to construct or name backend devices, queues,
streams, assistant models, sessions, or errors. Concrete device families are
validated against the backend features and runtime hardware during realization.
The backend reports the canonical identity derived from that realized device
binding rather than echoing the plan's requested identifier.

External assistants cross that factory boundary as an architecture-owned
`ExternalAssistantPreparationPlan`. Architecture inspection fixes the
assistant family, normalized configuration, checkpoint format, and strict
SafeTensors or GGUF layout before a backend is selected. SafeTensors plans
retain the admitted header catalog, architecture schema, and exact resolution.
A GGUF assistant plan retains the admitted portable checkpoint, resolution,
and canonical tensor mapping.
A concrete backend reopens the payload only through a catalog-bound source,
revalidates both the catalog and resolution, and rejects tensors changed after
preparation before constructing a native module. Concrete backends consume
that retained proof or the admitted portable GGUF checkpoint; they do not
receive an unrestricted assistant store, reopen configuration or metadata for
dispatch, bind against raw native topology, or duplicate assistant admission.

`eredu::api::local_device_plan` maps the facade's CPU or accelerator choice to
the selected local backend and rejects accelerator choices when that
build contains no native accelerator family. `LocalRuntimeConfiguration`
applies any process-global allocator or embedded accelerator-library
configuration before the factory realizes that plan. Platform applications
therefore do not need a concrete backend crate merely to create and complete a
model session.

Architecture inspection also reports embedded-draft depth from the normalized,
admitted artifact composition in `ModelResourceProfile`. The neutral automatic
planner consumes that observation directly; concrete backends must not infer
family semantics by searching raw configuration documents. In particular, a
DeepSeek-V4 base GGUF has target blocks only: its `nextn_predict_layers`
metadata describes weights omitted into a companion artifact, so base admission
normalizes embedded depth to zero and capability reporting does not expose it.

## Submission and completion

`BackendSession` provides high-level prefill and decode submissions. Associated
types keep prompts, tokens, outputs, session state, and completions opaque.
Concrete sessions dispatch complete-model and pipeline variants internally;
their replicated variant delegates prefill, decode, state controls, reports,
and observation to the neutral `ReplicatedTextSession`. They do not expose
model/cache parts or variant-specific constructors to callers. Operations
unavailable for a valid session topology return typed errors rather than
relying on unreachable or panicking accessors.

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

`SpeculativeExecutor` and `SpeculativeSampling` expose high-level opaque
operations for these stages. A backend owns assistant execution, target logits,
probability arithmetic, random state, cache checkpoints, and concrete
completion values. Embedded prediction heads and external assistant models use
the same portable lifecycle.

The prepared-generation capability does not expose whole-request execution
methods. A backend implements `with_speculative_execution` only to validate the
draft pairing, prepare typed lane resources, and lend them through
`SpeculativeGenerationVisitor`. The facade supplies
`RunSpeculativeGeneration`; `eredu-runtime::SpeculativeScheduler` registers
lanes, selects fair actions, drives exact completions, validates terminal
reasons, and constructs public outputs. Single requests use that same path as
one-lane batches. Concrete backends must not substitute their own lifecycle or
fair-batch loop.

Architecture families also own external-assistant compatibility proofs. Those
proofs match target-state publishers, hidden and rotary geometry, target-layer
captures, and vocabulary requirements before a concrete backend composes the
two executables. Backend composition may enforce the proof, but must not
restate family-specific compatibility rules.

Prediction-enabled composition is additive to the ordinary target session.
Architecture admission projects the target and a typed prediction-extension
plan from one already-admitted artifact; it does not reopen or reclassify the
checkpoint. The ordinary neutral session remains the sole owner of target
prefill, decode, mutable state, prompt-cache control, completion, observation,
rollback, and publication. A target pass may retain an architecture-declared
hidden capture from that same transaction for an embedded predictor or
external assistant. Missing capture fails the transaction before publication.
Backend composition may own prediction-local state and native prediction
mechanisms, but must not construct a second target model or target cache. Any
extension whose unit execution lacks a typed additive handoff remains
fail-closed; loading a second complete model beside the neutral target is not
an admissible compatibility mechanism.
DeepSeek-V4 DSpark remains in that fail-closed set: its target capture is a
configured set of intermediate-layer values rather than the sequential MTP
target hidden state, so projecting it as ordinary final hidden state would
change prediction semantics.

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
materialization, and causal acquisition/eviction/reload telemetry. Compound
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
the original poison. The reusable MLX
communication backend realizes this payload-free contract with an internal
scalar integer all-sum. Its completion polls the exact event without taking a
blocking runtime lock, retains arrays, count storage, groups, routes, and
streams, and defers the host boolean read until that event completes. Because
upstream MLX exposes no event abort, timed-out work is retained in a
thread-affine quarantine and its native communicator cannot be realized again
until a later safe reap observes completion. If the originating thread exits
first, that thread waits for exact completion (or a terminal asynchronous
error) and deterministically releases the retained arrays, count storage,
groups, routes, streams, and event; quarantine ownership is never leaked or
transferred across threads. Complete all-rank manifest consensus also precedes
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

Opaque MLX setup acquires the process runtime lock through the manifest's
absolute submission deadline and rechecks it immediately before the first
array, frame, or collective graph mutation. Lock contention therefore fails
before native submission and poisons the one partition-communication authority.
Manifest-owned subgroup realization never enters upstream MLX's synchronous,
non-abortable native split: it retains an exact logical membership view of the
world handle. A logical view may use a world-wide operation only when manifest
consensus proves that the creation batch partitions every world rank into
same-requirement groups; non-neighbor route relays additionally require a
disjoint, contract-identical route batch covering the world. Partial,
overlapping, optional, or contract-mismatched batches fail before native
submission. A variable all-to-all in such a proven wave expands each logical
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
an explicit reap. If the owning thread exits first, it synchronizes each live
event itself and releases the retained resources only after completion (or a
terminal asynchronous error); thread-affine work is neither leaked nor
transferred to a reaper thread.

`DistributedSession` is an optional capability of the selected model session.
It exposes high-level sum, gather, variable-count exchange, point-to-point, and
consensus submissions with exact completion. Unsupported operations report an
explicit absent capability. Communicator construction, sharding, movement, and
collective tensor math remain backend-specific.

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
ownership of concrete composition: the facade may select and opaquely wrap the
adapter, while the backend must not depend on facade APIs or orchestration.
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
