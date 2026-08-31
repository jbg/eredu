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
transforms, serialization, and accelerator/runtime handles. It owns no
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
`eredu-gguf` is likewise a backend-neutral storage dependency. Backends that
execute nonlinear GGUF IQ blocks consume their canonical values through the
typed `IQuantCodebook` API; generated table modules remain private and cannot
serve as an undocumented cross-crate integration surface.
The `eredu` facade is also portable when built with
`default-features = false`. Concrete adapter and native-platform features stay
optional; capability features configure an enabled adapter rather than making
portable contracts depend on it. The facade exposes a flat,
application-facing local adapter under `eredu::api`; it does not reproduce an
implementation crate's backend or composition module tree. The current MLX
feature mapping and native build requirements are documented with the
[MLX backend](../eredu-backend-mlx/doc/README.md).

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
composition and architecture-erased dispatch remain crate-private. Native
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
Explicit native sessions, streams, token handles, and distributed
collective groups remain backend-author concerns.

Application targets depend on `eredu` for facade operations and directly on
the neutral crates whose public values they construct. Infrastructure-aware
clients such as the CLI likewise import low-level policies from their owning
crates. The
selected-local-backend API owns device-plan creation, process runtime
configuration, allocator telemetry, and diagnostic benchmarks, while
`LocalModel::synchronize` is the sole application-facing synchronization entry
point. These APIs do not expose native tensors, streams, devices, or random
state. Direct native access remains an explicit backend-author escape hatch in
the implementation crate; it is not an application dependency.
`LocalLoadOptions` and `LocalInspectionOptions` contain only neutral
quantization, residency, and session-capability policy, while native
device-bound contexts are selected only by backend tooling. `LocalBackendError`
records facade operation context and a diagnostic message without exporting
native error variants. Portable execution plans remain the application surface
for device and topology selection.

Backend-generic sampling policy lives in `eredu-runtime`. Concrete backends
implement `SamplingBackend` primitives and bind the runtime-owned `Sampler` and
`SpeculativeSampler` traits directly; they do not publish backend-local policy
traits or compatibility re-exports. Concrete cache-state, session, completion,
and model-input types stay in the selected backend. Facade examples construct
local sessions through `eredu::api`, while backend-author probes that
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

Execution-group transport is also architecture policy. The runtime defines the
neutral placement, semantic-kind, merge-destination, and parallel-subgroup data
types, but every `LayeredArchitecture` must declare the transport for each of
its groups. Shared decoder defaults live in `eredu-architectures`; the runtime
does not assign decoder roles or placement to an unspecified group. Composite
families may declare different policies per group. In particular, Moshi places
its temporal decoder across the pipeline with the pinned embedding and output
modules, while its ordered depth predictions run on the output owner and own no
pinned static roles.

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

The ordinary Qwen decoder, block, and layered lifecycle are dense construction
surfaces and require only `NeuralBackend`. Concrete adapters that dynamically
admit both dense and Qwen MoE configurations use the separate routed Qwen
lifecycle, which requires `RoutedNeuralBackend`. A backend implementing dense
Qwen is therefore not required to provide a router or either expert-bank
implementation.

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
type. The portable `LoadedTextModelConfig` requires both identities, and the
facade carries the resolved family through artifact loading rather than exposing
the effective type under the ambiguous legacy `model_type` name. Portable
capability records, execution telemetry, rank-local parallel summaries, and
layerwise residency metadata likewise expose this identity only as
`effective_model_type`; serialized telemetry and capability records use that
same unambiguous key. Complete-model
materialization stores the already resolved `ModelKind`
alongside the concrete model implementation; it does not reconstruct family
identity from the effective type. Distributed composition carries that same
admitted `ModelKind` into every pipeline stage; a rank-local architecture
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
dtype, size, or substring; derive companion identities; canonicalize legacy
names; or inject compatibility metadata into `config.json`. Conversion fails
closed when a declared source is absent or any declared output collides with
another checkpoint tensor.

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
Independent expert residency is exposed as a validated neutral catalog. Each
catalog entry carries the router/cache identity, the owning architecture group,
unit index and parameter path, expert-parallel versus replicated placement,
exact acquired-bank binding names, exact logical parameter targets, and
checkpoint-derived recipes. Each parameter also declares whether it must be
preserved or is a load-time-quantizable projection; quantizable projections
carry the exact local scale and affine-bias companion binding names. Backends
filter neutral expert catalogs by the architecture's owning group and
group-local unit before lowering them to backend cache entries. Router/cache
identity is not an ownership address and must not be flattened back into one.
Backends consume the remaining declaration directly and never infer eligibility
or companion identity from binding spelling, dtype, or rank. Family code owns
sparse-layer selection, routed versus shared-bank scheduling, expert counts,
and cache-layer numbering. Physical checkpoint keys excluded from ordinary
layer residency while experts are independently resident are projected from
the complete architecture catalog; backend composition does not rediscover
them by walking family layer policies or rebuilding per-layer recipes.
This applies uniformly to Gemma 4, Muse-Glimmer, DeepSeek, GPT-OSS, LFM2,
Kimi Linear, and later
families: each architecture checkpoint module emits its complete
`ExpertResidencyCatalog`, including compact acquired-bank names and every
per-expert or rank-local selection recipe. Backend family adapters may request
and filter that catalog, but do not calculate a parallel expert topology.
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
Concrete backends lower catalog entries into native storage and may apply the
declared unit path to a rank-local placement; they do not rebuild the schedule,
instantiate a family block to discover expert parameters, or match parameter
targets by substring or suffix.
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
callable without choosing that state type. Concrete backend adapters use one
shared visitor rather than family binding tables.
Pipeline stage loaders use that same visitor for both ordinary decoder families
and multimodal families, leaving tensor-parallel binding selection generic and
the role-to-module mapping entirely within the architecture. Distinct pinned
modules use distinct storage roles even when they share a broader semantic
parameter class; for example, DeepSeek V4 exposes its target hyper-connection
head as `hyper_head` and its optional DSpark static bundle as `mtp`.
Architecture implementations also provide any released-checkpoint rewrite
recipes for their static modules, so alias and fused-layout handling does not
reintroduce checkpoint roots into a backend.

`eredu-nn::LinearFormatSpec` is the single neutral declaration of a matrix's
physical encoding and exact scale and affine-bias companion parameters.
Ordinary linear, embedding, router, and expert construction specifications use
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

Routed expert banks retain and expose their architecture-owned construction
specification. Resident, cached, distributed, and future backend execution
paths all consume that same geometry, projection encoding, bias layout, and
activation policy. Each expert projection also declares the exact identities
of its scale and affine-bias companions; reusable backends bind those identities
to native slots without imposing projection suffixes or synthesizing checkpoint
names. Backend residency adapters must not reconstruct a parallel family
descriptor from model arguments. Architecture APIs derive localized
expert-bank specifications for placement-resolved expert counts and projection
widths while preserving canonical parameter identities and physical formats;
backend composition only materializes those returned specifications.
Cache-backed distributed callbacks receive the specification from the resident
unit bank or directly from that unit's realization-plan entry, including for
ReLU-squared banks and appended prediction units; they never rebuild it from a
family configuration and layer index.
Ownership and localized construction are published together as an
`ExpertRealizationPlan`. The plan contains the checkpoint-global expert count,
the complete global-expert-to-owner map, the current rank's global expert IDs,
and the exact rank-local bank specification for every routed execution unit.
Distributed preflight consumes the plan's global count, and a concrete backend
lowers the declared owner map into its native dispatch representation without
running another assignment policy. The plan's presence or absence is also the
only authority for whether the realized architecture has routed execution
units; composition must not inspect a family schedule or configuration count
to decide expert availability. Backend composition must not pass family fields
or a separately derived tensor-parallel width into expert-bank construction;
the bank specification retained by this same plan is the only construction
input.
Qwen, Qwen3-VL, Qwen hybrid (including conditional vision and embedded MTP),
GPT-OSS, LFM2, Kimi Linear, Nemotron-H, Muse-Glimmer, Inkling, Gemma 4, and
DeepSeek V3/V4 expose family-specific realization entry points over their
constructed neutral architectures. These entry points select routed units,
preserve canonical parameter formats, and apply planner-derived local widths
before a backend sees the plan. Family adapters accept the plan rather than the
family configuration or parallel topology when creating their native dispatch
assignment. A backend assignment type is only a validated lowering of the
plan's owner map; it exposes no independent balanced, round-robin, or explicit
policy engine. DeepSeek pipeline unit factories install the plan before
constructing target or prediction units, and tensor-parallel expert-cache
selection derives
its local width from the same plan entries rather than from family arguments.
Distributed expert callbacks also
carry whether the requested result is globally complete or a rank-local
tensor-parallel contribution, so EP recombination preserves the reducible and
post-reduction terms without inventing or repeating a TP collective.
Observed routed execution asks the neutral `RoutedLayeredArchitecture` for each
unit's optional observation point. The architecture supplies both the semantic
module path and expert cardinality; backend composition only adapts native
tensors to the neutral observer and cannot invent family path segments.
Activation observers likewise derive unit input, output, and nested operator
names from the architecture's canonical `unit_path`; concrete composition must
not reconstruct a family path from a group or layer index.

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
The same exact estimate declares whether speculative draft weights use a
separate checkpoint, use configured embedded prediction layers, or are absent.
A backend maps that declaration to executable or unsupported status according
to its implementation; it does not maintain a family-name MTP table. Pipeline
sessions therefore report architecture-declared separate draft weights as
unsupported while external drafting remains unavailable for pipeline
execution; only an absent architecture declaration reports unavailable.

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
cache advancement and advertised embedded-draft limits likewise come from the
constructed architecture graph and its state layout. Parsed model-family
arguments are not a second source of execution or state geometry. Backend
composition obtains the layout through `ArchitectureParameters::state_layout`
on the realized architecture before transferring that architecture into its
runtime. When a family owns additional state outside the ordinary layered
target, its architecture publishes the target, prediction placement, and
composite persistence layout together; the backend consumes that value without
reassembling segments or recovering offsets from layer-count fields. Ingress
state is a distinct transient contract when it does not span that
composite persistence layout. Inkling publishes its realized target-only
ingress layout explicitly: ordinary execution receives global target geometry,
parallel execution receives rank-local target geometry, and neither receives
embedded-prediction state. Both neutral forward entry and concrete pipeline
allocation consume that same architecture-owned layout. Prompt-cache identity
for both replicated and pipeline execution is derived from
`ArchitectureParameters::state_identity` and a canonical `PartitionState`; a
backend lowers only its parallel topology. Pipeline execution uses its exact
placed partition, while replicated execution attaches the architecture's
complete realized state layout at global offset zero. Target or
embedded-prediction cache allocation uses that identity and does not recreate
a global layout, offsets, or family identity after placement. Composite
model layouts, such as a target decoder plus embedded prediction state, are
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
not synthesize additional cache coverage.
Architecture identity functions declare family, fingerprint, composite global
layer count, and placement; backends must not reconstruct family identity,
target/prediction boundaries, DSpark behavior, or shifted-prediction offsets.
Architecture-erased backend executables own their concrete model and correctly
typed mutable state in the same exhaustive variant. Model state is not exposed
as a second extensible erased enum: prefill, decode, inspection, prompt-cache,
residency, parallel, and speculative hooks dispatch on the executable, so a new
family makes every operation site non-exhaustive until its behavior is defined.
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
allowlist. Public backend load options carry `QuantizationRequest`, whose
variants describe load-time transforms only; checkpoint storage encodings such
as native GGUF blocks remain internal to artifact inspection and
materialization.
The same architecture capability report carries a typed parallel plan for
tensor sharding, pipeline staging, and expert partitioning. Each axis is
declared from the parsed family variant rather than inferred from parameter
addressability or a broad family identity. Independent expert residency
remains a separate artifact capability because it does not imply an
expert-parallel execution plan. Distributed backend preflight consumes these
facts from that exact normalized report instead of reconstructing support from
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
Rank-local materialization traverses the canonical units exposed by its
`ArchitecturePartition`, so composite vision, target, and prediction ordering
is never restated by a backend loader. Every distributed family loader
registers tensor-parallel groups from the complete architecture parameter
description; it does not enumerate static, media, decoder, or prediction
parameters independently. The same description supplies the ownership used by
pipeline-stage materialization, preventing tensor and pipeline planning from
drifting apart. Conditional Qwen pipeline boundaries come from the constructed
architecture, including hidden width and DeepStack activation cardinality,
rather than being re-derived from family arguments in backend composition.

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
are not family-dispatched again during backend materialization. Core keeps the
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
payload wrapper and
passes the pair through complete, tensor-parallel, and distributed-stage
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
Concrete backends execute those declarations: they resize pixels, extract
features, pack patches, and construct native tensors. A new backend therefore
consumes the same family protocol instead of reimplementing it.

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
- checkpoint payload mapping and tensor materialization;
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
checkpoints, materializers, and materialized tensors.

Neutral contracts use associated concrete types for these values. They do not
use `Any`, untyped pointers, string-dispatched operations, or erased calls in
per-layer and per-token paths.

## Loading and session creation

Artifact loading has four stages:

1. Portable inspection validates checkpoint metadata and tensor catalogs,
   asks the architecture registry for a canonical family, neutral loading
   protocol, and companion requirements, then resolves any sibling artifacts.
2. The selected backend validates the exact requested topology and remaining
   policy against normalized architecture facts and its own capabilities.
3. Portable planning binds the artifact description to that exact topology,
   quantization, and residency policy. Materialization options must reproduce
   the bound topology rather than supplying a merely distributed/non-distributed
   equivalent.
4. The selected backend materializes the plan into its executable model and
   creates a stateful session.

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
including distributed stage loaders, consume its inspected configuration,
primary and companion checkpoint handles, and selected route directly. They
must not reopen an artifact to rediscover configuration, checkpoint metadata,
or sibling filename policy after planning; payload stores may still map weight
members during materialization, but those reads do not replace the plan's
configuration, checkpoint metadata, companion selection, or route.

`ModelLoadingBackend` implements backend policy, architecture/backend
capability intersection, and materialization.
`BackendProvider::create_session` consumes a `PreparedModel`, so an executable
cannot be paired with a cache or session created by another backend.

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
the currently selected local backend and rejects accelerator choices when that
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
they do not expose model/cache parts or variant-specific constructors to
callers. Operations unavailable for a valid session topology return typed
errors rather than relying on unreachable or panicking accessors.

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

A backend supplies concrete tensors, host buffers, mapped payloads, files,
workers, native transfer objects, and allocation observations. It applies core
transition results and releases the physical resources selected by those
results.

## Distributed execution

Portable topology uses data, tensor, pipeline, and expert axes. Core validates
rank coordinates, subgroup membership, balanced ownership, and operation
scopes. Runtime owns the backend-neutral tensor-placement decisions consumed by
architecture planning and concrete backend realization.

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
Primary decoder and embedded
prediction groups have distinct semantic kinds, so multimodal ingress ends at
the decoder boundary and prediction runs only in its explicit phase.

The partition also carries an architecture-owned boundary schema. That schema
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
contract before validation and transport; checkpoint parameter dtypes,
rank-local loading order, quantization encodings, and stage ownership never
select the wire dtype.
Distributed placement dependency routes consume that same schema directly. A
concrete backend must reject inactive dependency routes with tensors and active
routes whose exact cardinality, ordered shapes, or physical dtypes differ from
the resolved architecture boundary; backend-local string or optional-field
schemas are not an alternate wire contract.
Point-to-point boundary arrays are exact-completion values. A backend must
complete each ordered receive before consuming or reusing its destination and
complete each ordered send before submitting another boundary array that could
alias the same lazy storage. Retaining a submitted handle without exact
completion is not sufficient ordering for a multi-array boundary.

`DistributedSession` is an optional capability of the selected model session.
It exposes high-level sum, gather, variable-count exchange, point-to-point, and
consensus submissions with exact completion. Unsupported operations report an
explicit absent capability. Communicator construction, sharding, movement, and
collective tensor math remain backend-specific.

## Concrete implementations

Concrete adapters document their public implementation surfaces, native
features, platform requirements, and realization details with the crate that
owns them. See the current
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
inspecting the Cargo dependency graph, scanning Rust source for substrings, or
asserting a particular file layout: those checks couple architecture to
repository shape and migration artifacts instead of semantic ownership.
Repeated violations should be made unrepresentable with a crate boundary or
visibility change.
