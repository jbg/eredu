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
`default-features = false`. A private `_mlx` feature selects the optional
`eredu-backend-mlx` adapter. The public `mlx` bundle adds the backend crate's
`accelerate` and `metal` features, while the sibling `cuda` bundle adds only
its `cuda` feature. The backend crate itself has no default features and
disables the `safemlx` defaults, so direct backend users select native
execution support explicitly. The facade exposes a flat, application-facing
local adapter under `eredu::api`; it does not reproduce the implementation
crate's backend or composition module tree. Backend-internal fixtures and
composition tests are crate-private unit tests; published crates expose no
test-support feature or fixture namespace. The facade does not directly depend
on `eredu-nn`. Direct MLX backend users also
enable `image` and `audio` as needed; the facade forwards its `mlx-image` and
`mlx-audio` features through the private adapter without selecting either
platform bundle. Backend feature diagnostics name both active public spellings.

The facade root and `api` namespace expose portable application concepts plus
the narrow selected-backend adapter. `eredu-backend-mlx` exposes the same
application-facing adapter as an explicit flat root API and deliberately makes
its reusable `backend` module tree public for backend authors. Family
composition and architecture-erased dispatch remain crate-private. Native
facade integration tests realize execution plans through
`eredu::api::LocalBackendFactory`; they do not construct native devices,
streams, tensors, samplers, or backend load options. Tests that require those
backend facilities live in `eredu-backend-mlx`, while facade sampling-policy
tests use neutral sampling traits and mock backends. The backend crate does not
alias neutral crates into its namespace; direct backend consumers import
neutral contracts from their owning crates. Composition-only load validation
helpers remain crate-private rather than becoming part of the backend API.

The facade likewise does not alias `eredu-core` or re-export `eredu-runtime`
infrastructure. Cache workers, checkpoint storage bindings, residency and
prefetch engines, generic schedulers and work-lifecycle contracts, parameter
plans, and other infrastructure contracts are imported from their owning
crates. Backend providers, sessions, prepared models, runtime handles,
completion and submission primitives, and distributed execution contracts are
likewise imported directly from `eredu-core`. The facade root retains only
deliberately application-facing types.

The selected adapter exposes the causal backend type and a facade-owned
realtime backend factory because the neutral execution contracts have
different model, input, output, session, and completion associated types. The
factory returns an opaque implementation of the neutral realtime loading and
execution traits, preventing the concrete backend's native associated types
and handle-oriented constructors from crossing `eredu::api`. An application
does not depend directly on the concrete backend crate merely to select local
realtime loading and execution. Explicit native streams and distributed
collective groups remain backend-author concerns.

Application-only targets and platform examples can depend solely on the
`eredu` facade. Infrastructure-aware clients such as the CLI also depend on
the neutral owning crates for the low-level policies they configure. The
selected-local-backend API owns device-plan creation,
process runtime configuration, synchronization, allocator telemetry, and
diagnostic benchmarks without exposing native tensors, streams, devices, or
random state. Direct native access remains an explicit backend-author escape
hatch under `eredu-backend-mlx::native`; it is not an application dependency.
Sampling functions and sampler traits whose signatures expose raw MLX arrays,
streams, or random state are exported only through that native namespace, not
through the flat application-facing adapter. Concrete realtime backend types,
inputs, outputs, sessions, completions, and prompt helpers follow the same
rule; the flat realtime factory returns an opaque neutral-trait implementation.
Concrete causal sessions, exact completion types, speculative drafters, and
owned model inputs also live under `eredu-backend-mlx::native`; callers of the
flat backend use their neutral trait interfaces and inferred associated types.
Raw completion submission remains crate-private. The flat adapter keeps native
borrowed model-input views private and returns model logits through the
backend-owned `MlxTensor` handle.
Facade examples follow the same rule and construct selected local sessions
through `eredu::api`. Backend-author probes that intentionally manipulate MLX
tensors, streams, distributed groups, caches, or checkpoint packing live as
`eredu-backend-mlx` examples and depend downward on neutral contracts.
All selected sessions expose completed outputs through
`BackendSession::observe_output`; backends retain native tensors internally and
materialize portable `ObservationSet` records only when explicitly requested.
Backends may additionally implement `InspectableBackendSession` by binding the
named activation and routed-expert points already emitted by
`eredu-runtime::ActivationObserver`. These are general diagnostics contracts
used by telemetry, inspection, observability, and evaluation rather than an
evaluation-specific backend surface. Realtime applications likewise exchange
portable host token frames and observations through
`RealtimeBackend::materialize_input` and `RealtimeBackend::observe_output`.

Backend-neutral parity, distribution metrics, timing summaries, evidence, and
evaluation drivers live in `eredu-evaluation`. Concrete backend examples only
select execution contexts and materialize model and codec artifacts; they do
not own comparison thresholds or reference policy.

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
The typed `ModelKind` and `GgufArchitecture` identities, their aliases, the
family-to-protocol mapping, and family-specific GGUF structural admission live
in `eredu-architectures`; core does not publish a parallel serialized model or
artifact identity schema. Nested `text_config.model_type` normalization is
available only to outer wrapper identities explicitly admitted by that closed
registry; an unknown outer `model_type` is rejected even when its nested text
identity is known. External assistant admission remains separate: its closed
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
Loaded backend models and sessions preserve these as two distinct identities:
`model_family()` returns the canonical architecture-owned `ModelKind`, while
`effective_model_type()` returns the parsed implementation or nested text-model
type. Complete-model materialization stores the already resolved `ModelKind`
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
Architecture-owned SafeTensors conversion plans likewise enumerate every exact
dense source, packed-weight output, scale companion, optional affine-bias
companion, and the complete output model configuration. Concrete backends
execute those plans literally. They do not select tensors by suffix, rank,
dtype, size, or substring; derive companion identities; canonicalize legacy
names; or inject compatibility metadata into `config.json`. Conversion fails
closed when a declared source is absent or any declared output collides with
another checkpoint tensor.

Portable SafeTensors schemas use released checkpoint names directly. In
particular, a matrix is named `*.weight`; the `*.inner.weight` spelling from an
old safemlx module serialization is neither a portable alias nor an accepted
architecture layout identity. Backend implementations may translate canonical
schema names to private module slots, but that translation must not widen the
checkpoint contract.

These catalogs are model-wide and configuration-derived. Backend adapters may
filter their outputs to the parameters present in a static module, execution
unit, or independently resident bank; that filtering does not transfer source
name discovery or recipe construction into backend composition.
Architecture checkpoint APIs also derive rank-local independently resident
expert recipes, including segmented fused-projection selection; backend
composition binds those recipes without reconstructing the segment equation.
Family recipe APIs derive their own checkpoint roots; in particular, Gemma 4
expert recipe callers provide only the catalog, normalized configuration, and
layer identity, never a backend-selected layer namespace.
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
Hybrid target/MTP families additionally declare the execution group, physical
MTP unit, checkpoint root, and cache identity of every sparse unit in this
catalog; backend adapters filter catalog units against the realized partition
using that canonical group-local owner address, without recovering placement
from flattened cache identities, target-layer counts, or inferred MTP ownership.
Pipeline logits and persistent prediction-state attachment likewise follow the
realized partition's output flag and architecture-declared static roles; a
stage's raw layer-range endpoint is not an ownership signal.
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
decoder/projector width compatibility are architecture policy; MLX structural
validation and materialization consume that composite plan without rebuilding
those rules from independently parsed text and projector configurations.
Packed recipes expose format semantics rather than a concrete accelerator's
storage units. In particular, canonical MXFP4 expert recipes describe logical
F4 values regardless of whether a source stores byte blocks or integer words;
the MLX adapter lowers those terminal recipes to the U32 geometry required by
its affine kernels before constructing runtime bindings.

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
recurrent-state geometry, and backend-neutral scalar state layouts. Concrete
backends select the loaded architecture estimate, apply their physical state
scalar width, and add live allocator, residency, and system-memory
observations; they do not reconstruct family state geometry.
The same exact estimate declares whether speculative draft weights use a
separate checkpoint, use configured embedded prediction layers, or are absent.
A backend maps that declaration to executable or unsupported status according
to its implementation; it does not maintain a family-name MTP table.

For SafeTensors and GGUF materialization, architecture preparation also
identifies the checkpoint parameter that establishes runtime-state dtype and
resolves its schema-declared physical aliases against the inspected catalog.
Dense GGUF scalar types remain typed in that portable catalog, so a concrete
backend can preserve the width used by its loader; packed encodings instead map
to their known materialized activation dtype. Generic backend composition must
not recognize family checkpoint names or silently choose a width when the
declared source is missing.

Runtime state follows the same contract at materialization time. Architecture
`StateLayout` values are authoritative for per-layer attention windows,
append-only pooling streams, and named segment boundaries and lifetimes.
Concrete backends derive native cache objects directly from those policies;
model-family arguments are not a second source of state geometry. Composite
model layouts, such as a target decoder plus embedded prediction state, are
assembled by the architecture before a backend consumes them. Architecture
state segments also declare their processed-token frontier offsets. Runtime
preserves the named segment ranges in prompt-cache model identity, descriptor,
and manifest data while expanding their frontier offsets per layer. Segment
selection validates the architecture-declared ID and rebases its exact range;
contiguous partition slices preserve and rebase the intersecting segment
metadata. When separately materialized prediction state follows target state,
the architecture's rank-local geometry publishes its `PartitionState`, including
the architecture-global offset; a backend does not recover that offset from a
family layer-count field.
Architecture identity functions declare family, fingerprint, composite global
layer count, and placement; backends must not reconstruct family identity,
target/prediction boundaries, DSpark behavior, or shifted-prediction offsets.
If a backend model and cache representation do not match, dispatch rejects that
pair before descriptor validation rather than synthesizing a model identity for
the invalid pairing.
Draft commit and pipeline prompt-cache persistence likewise select the
architecture's named prediction segment; family configuration layer counts are
not commit-range metadata.

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
raw or wrapper `model_type` values. GGUF inspection, planning, and
materialization all validate requested preparation against the architecture
plan retained by portable admission. A backend must not reparse the raw GGUF
checkpoint to derive a second capability report. SafeTensors capability and
runtime-state dtype derivation likewise accept only the typed architecture plan
retained by admission; raw JSON is parsed once by the architecture registry.
Validated architecture parameter descriptions retain their canonical
execution-unit layout alongside owner-tagged parameter groups. Pipeline
composition consumes those declared group ranges and flat unit ordinals for
target and prediction placement; it does not rebuild target/MTP counts from
family configuration fields.
Rank-local materialization traverses the canonical units exposed by its
`ArchitecturePartition`, so composite vision, target, and prediction ordering
is never restated by a backend loader. Every MLX distributed family loader
registers tensor-parallel groups from the complete architecture parameter
description; it does not enumerate static, media, decoder, or prediction
parameters independently. The same description supplies the ownership used by
pipeline-stage materialization, preventing tensor and pipeline planning from
drifting apart. Conditional Qwen pipeline boundaries come from the constructed
architecture, including hidden width and DeepStack activation cardinality,
rather than being re-derived from family arguments in MLX.

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
SafeTensors materialization retains the typed normalized family configuration,
its complete architecture-derived checkpoint plan, and the validated tensor
catalog from the preparation plan. Catalog-dependent architecture admission is
finalized against that neutral catalog before a backend receives the plan. For
Moshi this proves the strict physical schema and canonical recipe publication
as one architecture-owned operation and retains the recipes in the admitted
plan. MLX structural validation applies the exact checkpoint plan generically;
it has no family-specific recipe validation or second raw-JSON parser dispatch.
Catalog strictness belongs to that architecture-owned checkpoint schema;
residency and materialization options cannot weaken it. Conversely, tensors
left unclaimed by an explicitly non-strict schema remain outside the resolved
store presented to backend parameter binding.
Family composition does not reopen `config.json`, rediscover checkpoint shards,
or select a second catalog after admission. Every admitted artifact plan
retains the normalized `ModelKind`; SafeTensors plans additionally retain typed
family geometry and the checkpoint schema, while GGUF plans retain the exact
`GgufArchitecture`, typed family geometry, and architecture-derived main
checkpoint schema. Core keeps the corresponding `ValidatedGguf` proof intact
inside `ModelArtifact` until the selected backend consumes it; materializers do
not downgrade that proof to an unvalidated checkpoint handle and rerun the
architecture parser or regenerate the main checkpoint schema. Backend
composition may enrich a clone of retained geometry with native encoding
descriptors, while composite families retain their separately admitted companion
schema; neither operation replaces the retained main artifact plan. Portable
architecture admission derives that companion geometry,
validates the companion against its exact family-owned GGUF schema, and retains
both the typed composite configuration and schema in the artifact architecture
plan. A concrete backend pairs that proof with its native payload wrapper and
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
The architecture also owns the plan that maps a structurally validated sibling
media projector to resulting input modalities. Inspection applies that plan to
the same resolved artifact, so expected modalities and multimodal readiness
describe one composition.

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
backend namespace.

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
typed companion geometry and schema. Backends may wrap those portable handles
and add native encoding, operator, or device compatibility checks, but do not
rescan directories, select companions, repeat either portable admission layer,
or parse facade-owned tokenizer and EOS metadata.

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

`ModelRuntime<B>` owns the selected backend and its sole session. Applications
normally use `eredu::api::LoadedModel<B>`, which combines that runtime
with tokenizer, EOS, generation-default, and chat-template metadata.

Automatic planning produces a portable `ExecutionPlan`. An
`ExecutionPlanBackendFactory` realizes the complete plan into a target backend,
backend load options, and the selected disabled, embedded, or external drafting
configuration. `LoadedModel::load_execution_plan` and
`LoadedModel::plan_and_load` therefore do not require callers to construct
backend devices, queues, streams, or assistant models. Concrete device families
are validated against the backend features and runtime hardware during
realization. The backend reports the canonical identity derived from that
realized device binding rather than echoing the plan's requested identifier.

External assistants cross that factory boundary as an architecture-owned
`ExternalAssistantPreparationPlan`. Architecture inspection fixes the
assistant family, normalized configuration, checkpoint format, and strict GGUF
layout before a backend is selected. Concrete backends consume the retained
SafeTensors payload source or admitted portable GGUF checkpoint; they do not
receive a raw assistant path, reopen configuration or metadata for dispatch, or
duplicate assistant admission.

`eredu::api::local_device_plan` maps the facade's CPU or accelerator choice to
the currently selected local backend. `LocalRuntimeConfiguration` applies any
process-global allocator or embedded accelerator-library configuration before
the factory realizes that plan. Platform applications therefore do not need a
concrete backend crate merely to create and complete a model session.

Architecture inspection also reports embedded-draft depth from the normalized
family configuration in `ModelResourceProfile`. The neutral automatic planner
consumes that observation directly; concrete backends must not infer family
semantics by searching raw configuration documents.

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
audio. MLX does not reconstruct modality, spatial-merging, or artifact policy.

Layered model execution topology follows the same ownership rule. Loaders and
materializers derive execution graphs and per-group unit counts through
`LayeredArchitecture` into one canonical `ExecutionUnitLayout`; quantization,
residency, parallel planning, parameter accounting, state allocation, and unit
construction consume that layout together with the architecture's required
state-independent parameter contract. They do not reconstruct a family's group order,
dependencies, layer counts, or flat-to-group mapping from configuration. This
keeps every backend realization aligned with architecture execution.

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
reconstruct a family's begin/finish lifecycle. Primary decoder and embedded
prediction groups have distinct semantic kinds, so multimodal ingress ends at
the decoder boundary and prediction runs only in its explicit phase.

The partition also carries an architecture-owned boundary schema. That schema
declares every auxiliary tensor's stable role, canonical order, symbolic shape,
logical dtype, and configuration-dependent cardinality, and it owns conversion
to and from the family's typed boundary value. When the evolving activation's
transport width differs from the ordinary hidden width, that derived width is
also part of the family boundary plan. `eredu-runtime` validates and resolves
batch and sequence dimensions. A concrete backend only maps the logical
activation or exact integer dtype to its native dtype, allocates the declared
receive buffers, validates produced tensors, and transports them. It must not
reconstruct boundary geometry from model-family arguments.
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
between public reusable backend mechanics and private family/backend
composition:

- reusable public mechanics are rooted directly at
  `eredu_backend_mlx::backend::{nn, runtime, ...}`. The dedicated MLX crate
  does not repeat its backend name as another module layer;
- `MlxBackend` is the facade backend provider and privately owns execution and
  weight-materialization streams. Its production inherent API neither accepts
  nor returns native handles; callers that deliberately construct
  backend-native sessions do so through `eredu_backend_mlx::native`.
  `MlxNeuralBackend` is
  the distinct zero-sized selector that implements the neutral neural,
  parameter, submission, and transfer traits for architecture specialization.
- `MlxTensor` is a transparent, zero-copy wrapper around `safemlx::Array` and
  is the sole MLX implementation of `eredu_nn::Tensor`;
- `native::MlxModelSession` composition owns the executable model, cache,
  processor state, and
  optional distributed context. The prepared `MlxModel` wrapper exposes
  neutral capabilities and telemetry while keeping its executable kind and
  architecture-specific payload private.
- neural-network modules implement reusable MLX tensor operations;
- runtime modules implement checkpoint materialization, sampling, caches,
  residency workers, media processing, and collectives. Their production
  checkpoint API consumes exact local parameter identities and
  architecture-derived bindings and recipes. Strict module loading performs no
  prefix stripping, prefix rewriting, unused-prefix exemptions, or implicit
  parameter-name expansion. Selective partition loading likewise requires
  exact checkpoint keys in its placement plan; neither surface invents family
  aliases or exposes parsers for physical family checkpoint names;
- GGUF family selection and portable family-specific structural admission are
  architecture-registry concerns. MLX composition resolves the already
  admitted spelling through the same registry, trusts the retained portable
  schema proof, validates only native encoding and operator compatibility, and
  passes an admitted source to resident,
  tensor-parallel, pipeline, or expert family loaders. Reusable backend runtime
  modules neither parse `general.architecture` nor invoke family composition;
- generic layerwise policy construction derives its execution graph and unit
  layout directly from the concrete neutral architecture before binding
  checkpoint units; composition cannot supply or reconstruct that layout.
  Unit-binding adapters receive the validated architecture-flat ordinal,
  group-local address, and canonical `unit_path`. Composite-family recipe
  selection accepts that flat ordinal unchanged; architecture-owned recipe
  catalogs translate optional media and prediction units into checkpoint-local
  identities instead of backend binders reconstructing group order or layer
  counts. Group ownership and parameter roots remain authoritative in the
  address and path;
- Moshi binding enumerates stable `Parameterized` identities and consumes its
  architecture-owned canonical recipe publication; composition does not use
  native module reflection to rediscover parameter names or aliases; and
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
Collective capability is reported only by an MLX backend instance with an
attached world communicator. Replicated activation inspection dispatches every
family observer implemented by MLX composition through the same erased session
path; missing family/cache pairs fail before execution instead of falling back
to an uninstrumented pass.

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
