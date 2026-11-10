# Architecture and observation discovery

`eredu::api::inspect_architecture(path)` returns a serializable
`eredu_core::ArchitectureDescriptor` with no backend feature enabled. It reads
artifact headers and the admitted family plan. It does not create a device,
read tensor payloads, or initialize an accelerator.

For selected execution, use `eredu::api::inspect_local_model` or
`eredu_architectures::inspect_model` with the exact intended load request.
`ModelInspectionReport.architecture_descriptor` contains the same logical
graph, even when execution selection is rejected.
`ModelInspectionReport.observation_support` reports prefill and decode support
separately. Invalid/unrecognized artifacts can have no descriptor. Older
serialized reports deserialize with both discovery fields absent.

Architecture descriptors use schema version 15. Observation catalogs and support
reports retain their independent schema version 1.

The descriptor has a `schema_version`, typed nodes and attributes, containment
through `parent`, and directed data-flow edges. Residual edges identify bypass
inputs; routed and shared experts form distinct branches. Attention head
sharing, receptive field, mechanism, recurrence, head dimensions, and positional
encoding are independent attributes of each layer. Consumers traverse these
types; they do not match family names or parse checkpoint keys.

Neural construction specs expose logical parameter declarations with exact
geometry, neural roles and authoritative sharing metadata. The shared decoder's
cold placement and materialization projections consume those same specs and
retained encoding extensions. Logical and stored pinned projections preserve the
same role and sharing identities, including tied vocabulary tables. Repeated
invocations retain independent state and placement while between-pass
normalizations can reference an authoritative parameter in another group.
Partial group validation preserves that reference; complete preparation rejects
unresolved targets, alias chains and incompatible geometry.
Primary vocabulary tables and component-owned lookup tables retain distinct
ownership. Materialization projects per-layer, position and media tables as
`LookupTable` while discovery retains their neural embedding or position role.
Each table keeps its declared geometry and encoding; table rank or name does not
assign it to the vocabulary embedding or authorize a conversion.
Cold selection retains embedded prediction parameter declarations independently
of the ordinary target. Selected formats and physical companions validate against
the admitted complete declaration before allocation. Rank resource sizing uses the
prediction's retained tensor layout, including replicas across target pipeline and
expert coordinates. Prediction constructors consume the same selected catalog;
observation points still come from explicit prediction execution contracts.

Parameter declarations do not establish an observation or intervention point:
hooks retain explicit meanings, axes, phases and allowed operations, and loaded
support remains conditional on the retained selection and backend mechanisms.
Qwen3-MoE and GPT-OSS routed parameter groups project router and expert-bank
construction declarations, including biased experts and encoding companions.
Qwen3-MoE, GPT-OSS, LFM2 and Kimi sparse construction, admission and placement consume
the same neutral `RoutedGatedProductSpec` contract; discovery reads global selector
counts and normalization. Kimi's family scope retains its always-on shared branch
and validates the residual equation alongside complete parameter references. Rank-local banks retain independent geometry, while provider
invocation coordinates remain architecture-owned.
Logical discovery preserves those identities and roles before storage expansion;
selected placement uses the physical projection with explicit expert partition
semantics. Storage projection does not create execution or observation hooks.
Shared-decoder readout and component normalizations project their learned gains,
offsets, epsilon and reduction groups from the construction specs. Invocation
paths and head-row mappings remain explicit architecture semantics; reusing a
parameter does not merge invocation state or observation phases.
Vocabulary graph roots, logical hidden/vocabulary axes, output identities and tying
project the validated static construction used by materialization. The shared
decoder, Qwen hybrid, LFM2, Kimi, Nemotron, DeepSeek and Diffusion Gemma use the
same static graph projection. Qwen hybrid prediction scopes borrow that token
table and output projection, including authoritative shared-weight references;
their fusion and normalization slots retain separate declarations. Prediction
invocation identities, static consumer roles, equations and hook phases remain
explicit family semantics.
Qwen hybrid attention, recurrent and prediction normalizations use their shared
construction policy, including learned-offset gains. Recurrent coefficient roles
remain parameter metadata and do not create intervention or observation hooks.
Hybrid target and prediction storage groups project retained encoding references
from those construction declarations. Logical geometry and rank-local comparisons
retain the same identities and formats independently of physical expansion.
Hybrid attention reads and writes project `AttentionParameters`, including biases,
authoritative sharing identities and query/key normalization. Dense and shared-expert
component rows project their declared affines and scalar gate. Interleaved query/gate
row mappings and observation phases remain explicit family equations.

Architecture descriptor schema **2** adds `layer_groups` and parameter sharing.
Observation catalogs and support reports retain schema **1**; their capture
selectors and admission version are unchanged. Older descriptors deserialize
with an empty `layer_groups` and absent `shared_with` declarations. Consumers
must treat that as unknown structure, rather than infer a single pass from the
number of decoder nodes.

Schema **3** adds component read input-projection chains. Each stage identifies
effective parameter rows, an optional bias and normalization, and the effective
observation consumed downstream. An empty chain means a direct read from the
normalized sublayer input. Check the descriptor version before interpreting these
equations; ignoring a latent normalization changes the meaning of a read. The
[component guide](component-analysis.md#topology-and-actual-support) describes
bounded parameter queries and current-position capture semantics.
Retained partition layouts derive stage observations from the projection's output
rows and the invocation's execution owner. The stage's effective and original
paths share that placement; shared checkpoint aliases do not move the invocation.
Loaded support still checks the selected executor's hook facts separately.

Each `ArchitectureLayerGroup` declares a `physical_layer_count`, ordered
`passes`, and `weight_sharing`. Each pass has a zero-based `index` and ordered
`executions`; an execution identifies its `physical_layer_index` and exact
graph `node_id`. Physical layer indices are local to the group. A pass visits
each physical layer once. Ordinary decoder stacks have one pass and
`weight_sharing: "none"`. Repeated stacks declare `"shared_across_passes"`.
Groups describe decoder stacks; media encoders and other opaque components
remain represented by their existing nodes and completeness declarations.

For the released Nanbeige 4.2 configuration, the decoder group contains **22
physical layers, 2 passes, and 44 executions**. LM Inspector can render
**“22 layers × 2 passes · shared weights”** from these fields, then expand
`passes[].executions[]` and resolve each `node_id` with `descriptor.node(...)`.
For example, physical layer `0` appears as `decoder.layers.0` in pass `0`
and `decoder.layers.22` in pass `1`. Their capture paths are respectively
`model.layers.0.input` / `.output` and `model.layers.22.input` / `.output`.
Use those advertised paths to select individual executions, including with
bounded `CapturePlan` selections. Support must still be joined by exact path.
For display, add one to pass and physical layer indices; do not change paths.

`ArchitectureNode.layer_index` retains its existing **logical execution**
ordinal (0–43 for Nanbeige). Node IDs, logical parameter prefixes, graph edges,
and observation paths retain their identities. `ArchitectureParameterGroup.shared_with`
references another parameter-group ID when distinct logical groups reuse the
same weights. For example, `parameters:model.layers.22.self_attn` shares with
`parameters:model.layers.0.self_attn`. The inter-pass output normalization
shares with `parameters:model.norm`; it remains a distinct operation. Nodes
that already reference the same parameter group, such as tied embedding and
output weights, continue to do so. An absent `shared_with` means no additional
sharing declaration, not proof of independent weights.
Repeated-stack group sharing projects the typed invocation parameters' aliases.
Aggregate module groups preserve each member's relative parameter identity;
explicit child groups retain separately owned boundaries such as between-pass
normalization. Missing declarations, incompatible geometry, conflicting aggregate
owners and unresolved owner groups reject the projection. Pass identities and
physical indices describe invocation order independently of that sharing.

Weight sharing describes architecture parameters. Each Nanbeige execution has
independent mutable KV state, and resident execution can materialize replicas
of the shared checkpoint weights. The group is neither a residency unit nor a
claim about native allocation sharing. Controlled and uninterrupted capture
use the same per-execution identities.

Three namespaces have different purposes:

| Field | Meaning |
| --- | --- |
| `ArchitectureNode.id` | Stable logical identity within this descriptor/schema |
| `ArchitectureParameterGroup.canonical_prefix` | Architecture-declared logical checkpoint module prefix; physical aliases and encoding companions remain in checkpoint schemas |
| `ObservationPoint.path` | Exact selector accepted by the existing observation API |

An operation with no observation paths is not instrumented by this catalog.
In particular, attention nodes do not imply that attention probabilities can be
captured. `completeness` distinguishes complete coverage from partial and
unsupported descriptions, with reasons. `None` in an attribute, tensor shape,
or byte estimate means unknown, never zero.

All currently cataloged values are tensors. Activations usually have
`[batch, sequence, hidden]` axes. Expert IDs, selected scores, and coefficients
have `[token_rows, selected_expert]`, where token rows flatten batch and
sequence. The semantic dtype is independent of checkpoint storage. MLX declares
F32 conversion for floating host observations; integer signedness is retained in
the returned `TensorObservationData`. Captures retain complete tensors, and byte
estimates remain unknown when the request shape or native dtype is unknown.
These statements describe the legacy `ObservationRequest` API. The admitted
`CapturePlan` API adds native slicing, previews, summaries, histograms, and raw
candidate-score summaries with explicit budgets. Selected support reports now
include `capture` transformation capabilities and conditions. See
[bounded capture and observed generation](bounded-capture.md).

Block inputs/outputs and logits are observed **before an intervention at that
exact path**. Earlier interventions may already have affected them. Routing
events remain read-only observations after expert dispatch. Separately declared
intervention targets support control before dispatch where the loaded session
advertises it; see [intervention plans](interventions.md).

Nemotron-H attention, dense/shared expert, normalization, readout and prediction
fusion descriptors reference their typed construction parameters. Mamba mixer
attributes use the mixer declaration. These parameter facts do not establish
observation availability; the family retains explicit hook declarations and
component equations. Target and patterned-prediction parameter groups use the
same construction-owned encoding references for physical companions and selected
placement; Mamba segments and routed/shared ownership remain explicit family rules.
Prediction boundary descriptors use `PredictionUnitParameters` for the first-unit
embedding/hidden normalizations and fusion and the last-unit readout normalization.
Physical units retain independent invocation identities, mutable state and placement.

LFM2 attention, dense SwiGLU, normalization, readout and short-convolution
attributes project their typed construction specs. Routed components reference
the declared expert banks. Parameter geometry and sharing do not imply a hook or
an allowed intervention; those contracts remain explicit. LFM2 attention discovery
projects its complete `AttentionParameters` scope, including declared head geometry
and query/key normalization, through the same declarations used by construction.
Gemma 4, Inkling, Muse, DeepSeek V4, K2 Horizon and Nemotron affine reads share the
same construction-metadata projection. Published KV-state sources, normalization
stages, row mappings and observation paths remain family declarations.
Gemma 4, Muse and Inkling composite graph widths and vocabulary axes reference
the static construction declaration also consumed by their readout projection.
DiffusionGemma supplies its pinned declaration to the same Gemma readout projection.
Tied output authority and Inkling's unpadded score domain retain their explicit
semantics; these geometry references add no capture or intervention support.
Tensor-transform parameters use the same typed slot projection for identity and
authoritative sharing. This includes Muse compressed projections and learned or
stored residual coefficients, Inkling convolution kernels, relative projections
and dense branch scales, and Gemma residual scales. Their parameter groups,
equations and observation boundaries remain explicit.
Ordinary decoder components project attention and gated feed-forward reads,
writes, biases, sharing targets and widths from their construction specs, with
normalization supplied by the shared block declarations. `AttentionParameters`
supplies checked head axes and typed split/fused slots. Fused read offsets use its
retained component-major layout; grouped-query mappings retain their explicit
equation. Gate activation comes from the separate declared gate slot. Repeated Nanbeige invocations
retain the slots' declared aliases; K2 routed values have no ordinary affine
value read.
Kimi and DeepSeek V3 MLA components project checked head, latent, positional,
reconstruction and value axes from `LatentAttentionGeometry`. Their typed family
scopes validate direct or low-rank query connections and fused or split K/V
reconstruction. Component row strides use those declarations; rotary/NoPE,
normalization, cache ownership and hook support remain explicit family contracts.
Low-rank input stages in Kimi and DeepSeek use a shared projection of the exact
affine construction slot. Each stage retains its explicit row slice,
normalization and effective output path. Nemotron attention and dense reads
reference their declared projection slots directly, including optional biases.
Nemotron attention discovery consumes its complete `AttentionParameters`
declaration, shared with module construction, placement and checkpoint admission.
Its affine slots and checked head axes come from `SelfAttentionProjectionSpec`.
The declaration retains unpositioned attention arithmetic and global or local
head geometry; invocation schedules and observation paths remain explicit.
Qwen hybrid attention projects its interleaved query/gate affine and head axes
from `SelfAttentionProjectionSpec`. The typed query layout supplies per-head row
strides; source admission, placement and target/prediction construction use the
same declaration. Sigmoid activation, partial rotary and explicit channel hooks
remain family execution contracts. Parameter geometry adds no capture operations.
Nemotron dense and shared-expert components project `DenseParameters`, including
the unary activation, projection widths, biases and sharing metadata. Complete
sparse factories supply the selector, routed bank and shared branch to construction,
checkpoint admission, placement and discovery for target and appended prediction
identities. Discovery reads routing counts and normalization from the retained
selector and shared width from its unary projections. Local binding preserves
those identities while restricting routed and shared channels independently. A parameter or
activation declaration establishes no observation or intervention support.
Ordinary decoder components consume `GatedProductSpec` for resolved policy,
named split slots or fused gate/value rows, and output geometry. The same neural
scope supplies module construction, checkpoint projection, cold topology and
selected placement. Moshi temporal/depth declarations retain its fused form;
VoiceChat TTS aliases reference its split form. Fused row equations and hook
phases remain explicit discovery semantics.
LFM2 dense, Kimi dense/shared, DeepSeek V3 dense and V3/V4 shared, Qwen
hybrid dense/shared, Qwen4 shared and Muse dense/DSpark and direct/convolved DFlash
factories return the neutral
`SplitGatedProductSpec`.
Discovery, checkpoint admission, placement and construction consume the same
exact slots, widths and resolved policy. The common gated-policy projection
retains SiLU or approximate GELU, the SiLU multiplier, bounds and offset.
Sequential prediction and DSpark widths come from these shared declarations.
Qwen hybrid and Qwen4 shared/routed compositions retain this shared neural
scope alongside `RoutedGatedProductSpec` selectors and banks, separate scalar
gates and activations. Selector counts, bank geometry and encoding companions
come from the same routed scope in construction, admission, placement and discovery. Family code
supplies names, formats, component roles, groups, row mappings and invocation
boundaries; neural connectivity validation establishes no hooks or backend
support. Muse dense components project their declared SiLU policy through the
same visitor, while target and assistant storage aliases remain explicit.
LFM2 and Kimi physical placement groups consume the construction declarations'
retained encoding references, preserving primary/companion ownership and their
explicit attention, convolution, recurrent and expert partition domains.

Kimi KDA and MLA read identities, recurrent coefficients, normalization and
low-rank stages come from their construction declarations. Shared affine-read
projections retain declared bias and sharing metadata while family equations
supply roles, row mappings and input stages. KDA transform records use typed
references to their specific convolution and recurrent parameter slots. Checked
head geometry supplies KDA component axes, control and normalization widths, and
transition/decay shapes. Its declaration validates projection/convolution coupling
and projects the three histories and FP32 recurrence used by selected state sizing.
MLA's fused and split
latent reconstruction retain explicit row mappings. Hooks remain independent of
parameter declarations and selected storage encodings.

DeepSeek V3 MLA and prediction discovery reference their typed construction
parameters and normalized low-rank stages. Validated V3 policy retains immutable
static, decoder and prediction scopes; discovery borrows those scopes across
source-format, selected-format and rank-local projections. Policy changes use
explicit encoding, geometry or prediction-removal projections, preserving
identities, arithmetic and independent mutable state.
V3/V4 family declarations provide one
complete typed selector, routed-bank and shared-branch declaration for construction,
admission, placement and discovery. Selector/bank declarations compose the generic
neural routed contract; shared-branch coupling remains family-owned. Provider
invocation coordinates stay separate from parameter declarations. Shared SwiGLU component projection uses its
split-gated scope for identities, geometry, sharing, biases and resolved policy.
Execution equations, fused/split row mappings and hook support remain explicit.

DeepSeek V4 routed and shared-expert descriptors consume the declared selector,
expert bank and gated shared projections for identities, geometry and sharing.
Checkpoint expert-source extensions project split and independent layouts from
these same packed declarations; recipes and residency retain the declared targets.
Token-selected routing parameters do not declare forced-routing hooks or imply
backend intervention support; those contracts remain explicit.
V4 configuration normalization retains immutable typed scopes for static and
hyper-stream owners, decoder blocks, and sequential-MTP or DSpark extensions.
Discovery, admission, construction and placement borrow those scopes. Format and
local-geometry selection preserve operator policies and ownership; target-only
views remove prediction/DSpark parameters while keeping target-capture meaning.
Decoder invocation state, provider selection and explicit hook permissions remain
separate from parameter declarations.

DeepSeek V4 attention descriptors project normalized query reads, shared KV
reads and grouped writes from typed attention construction declarations. Head,
rank and bottleneck geometry comes from those specs. Main and indexer heads use
checked neural geometry; grouped rank comes from declared output-stage width.
Compressor state width comes from its declared normalization and projection slots.
The complete scope validates projection connections and local/compressed branch
presence. Row mappings, compressed history equations and hook availability remain
explicitly declared.
Layer placement and encoded companions consume the construction declarations'
retained format references. Logical ownership and invocation identities remain
independent of physical expansion. Pinned vocabulary and DSpark lookup tables
retain explicit selected encodings in the same construction specs used by cold
placement. Dense published-table defaults remain separate from model-wide linear
formats; source constraints and observation support remain explicit extensions.

DeepSeek V4 residual-stream coefficient descriptors reference the same typed
hyper-connection and hyper-head parameters used by construction, checkpoint
admission and placement. Target, sequential prediction and DSpark collapse
coefficients retain explicit equations, invocation identities and hook meanings.
Sequential fusion reads, DSpark context width and Markov score writes project
construction identities and geometry. Parameter roles and geometry do not
establish collector or intervention support.

Nemotron dense and shared ReLU² descriptors read affine identities, geometry,
biases and activation from the neural unary scope used by construction. Family
validation retains residual coupling and the down projection's exact-zero input
hint. Muse vision blocks use the same unary contract with exact GELU; their
expansion, activation and contraction phases remain explicit hook declarations.

Muse vision normalization slots, epsilon and moment arithmetic come from the
same typed declarations used by the generic neural `LayerNorm`. Family scopes
require gain and bias, and cold admission and construction project the cascaded
moment capability from the declared arithmetic. These parameters establish no
additional observation or intervention support.

Gemma 4 and DiffusionGemma vision parameter groups project pinned patch, position,
and optional pooled-standardization slots alongside independently owned block
matrices, head and residual gains, and optional scalar clipping bounds from their
typed construction declarations retained by immutable `VisionConfig`. The
`VisionOwner` root distinguishes Gemma4 and DiffusionGemma parameter ownership;
format projections preserve geometry and companion ownership. Vision feed-forward
affines and their fixed GELU-tanh policy come from one split-gated scope; optional clipping bounds attach
to its named projection slots. Vision and audio attention compose
`ClippedSelfAttentionProjectionSpec` for checked head axes, named affine slots and
learned clipping extensions. Family declarations retain normalization, rotary,
relative-key and per-dimension scale coupling. Physical companions retain their primary matrix
owner through the declarations' retained encoding references and the common
runtime projection. The media invocation structure and explicit capture declarations remain
independent of these parameter roles.

Gemma audio and VoiceChat Conformer feed-forward parameter groups consume typed
unary declarations with fixed SiLU. Family scopes retain normalization, residual
coupling, clipping extensions and residual coefficients. Those neural parameters
and equation policies add no observation or intervention declarations.

Gemma 4 audio parameter groups project pinned subsampling and projection owners,
encoder-block projections and gains, per-dimension query scales, clipping bounds,
and depthwise kernels from immutable construction declarations retained by
`AudioConfig`. Format projections retain source geometry and declared companion
owners. Optional output-bias placement follows the declared invocation policy;
its source slot remains present independently of selected construction. Source optionality and encoding
constraints remain storage extensions referencing the declared slots; admission
and placement borrow their encoding references. They do not establish capture support.

VoiceChat MoG parameter owners and selected intermediate/rank placement project
the typed residual blocks, prediction head, and low-rank component bank retained
in `MogDeclaration` by construction, preparation and source admission. Discovery
borrows the published profile scope. Residual blocks retain their affine slots and
fixed GELU-tanh policy in `SplitGatedProductSpec`; the surrounding gains, residual
width and selected local bound remain MoG constraints. Per-mixture mean segments
and expansion columns share the declared rank domain. MoG, RVQ and RNNT physical groups project the
same retained slots and encoding references through the shared runtime visitor.
Their acquisition owners and invocation graph remain explicit. Prediction and
selected-component invocations, guidance and supplied random draws keep independent
meanings; parameter roles
add no observation or intervention support.

VoiceChat's coupled graph attaches neural parameter groups from the perception,
RNNT, Nemotron, speech/character, conditioning, MoG, RVQ and EAR declarations.
Backbone identities consume the same canonical source namespace extension as
admission. The frozen speaker node's feature width comes from its declared prompt
table, independently of codec latent width. Host preprocessing geometry remains
explicit processor policy. The coupled graph remains partial, and its prefill/decode
observation catalog remains unsupported; parameter membership grants no capture
or intervention support.

VoiceChat's function readout projects its independent matrix owner, geometry,
and physical companions from `FunctionHeadDeclaration`, retained by construction,
preparation and source admission. Physical format selection projects companions
from its typed readout spec while discovery borrows the published profile scope. Selected vocabulary rows apply coherently to the matrix and companions.
The shared normalized input, separate agent/function invocations, and explicit
capture contracts retain their existing meanings; encoding adds no capture support.

VoiceChat's backbone source namespace projects the retained Nemotron-H
declaration for its target-only configuration, including static embedding, normalization and readout slots and typed
Mamba, attention and dense blocks. Discovery, source extensions, exact preparation
and module construction borrow that scope and its component ownership projection.
Embedding, readout and target-unit identities retain the ordinary ownership and
selected placement contracts. Published source aliases, encoding companions and SSM
storage transformations reference those slots; format metadata follows their geometry
and the validated schedule. Source identity translation establishes no additional
observation or intervention support.

VoiceChat causal speech and bidirectional character transformers project the
shared decoder construction specs for attention, feed-forward and normalization
owners from retained `TransformerDeclaration` scopes. Discovery borrows the same
published profile scopes used by checkpoint extensions and speech construction.
Physical formats and local widths project cloned slots; arithmetic selection stays
separate from canonical source membership. Final normalization remains an independent
residency unit. Selected heads,
intermediate channels and encoding companions follow the retained local layout;
mutable caches, character key visibility and invocation identities remain
separate contracts. Parameter ownership and source gain conversions establish no
new observation or intervention support.

Transformer encoding references resolve against that retained stack's parameter
identities. Non-dense formats require declared affine slots; dense entries may
describe its other owners. Preparation projects formats only for the selected
causal or character stack, and rejects unresolved references before allocation.
Explicit speech precision remains a separate arithmetic selection.

EAR neural encoder and decoder owners project their frozen convolution,
normalization and pointwise slots from shared construction declarations.
Canonical forward and transposed kernel axes remain distinct. Grouped encoder
outputs and decoder output columns use their selected domains; intermediate
channels project coherently across both MLP affines. Causal carry belongs to
decoder invocations, independently from parameter leases and selected local
storage. These declarations add no capture or intervention hooks.

Fixed-mean codec quantizers project their mean tables and retained variance
scalars from codec-owned construction declarations. Residual depths retain
independent owners and invocation identities; selected mean channels share a
coherent domain while scalar variances replicate. Encoder/decoder borrowing,
selected depth ranges and distance-scoring equations remain explicit execution
semantics. Parameter roles establish no additional capture or intervention hooks.

VoiceChat RVQ feedback projects the frozen stacked codebook bank and shared
feedback affine from the retained `RvqDeclaration` shared by source admission,
preparation and construction. The published profile also shares its conditioning
and codec scopes with discovery; parameter attachment borrows those scopes.
Selected latent channels cut the
bank's last axis and the affine's input columns coherently, with independent
leases for the two owners. Codebook roles describe parameter meaning; they add
no capture or intervention support. Refinement candidates, schedule boundaries
and operation-specific invocation activity remain explicit execution contracts.

Qwen4 lexical-injection geometry and parameter ownership project the same typed
feature affines, grouped normalization and causal convolution used by construction
and source admission. Feature axes and convolution width derive from those slots.
Provider-owned n-gram tables and exact hash controls remain separate from learned
parameter ownership; history, phases and observation permissions remain explicit.

Qwen4 residual-mixer parameter groups project normalization and coefficient slots
from the gated residual construction declaration, including component ownership
and sharing. Target placement retains independent decoder invocation and state
identities. Parameter roles do not establish hooks or supported interventions.

Gated-delta component axes and KDA recurrent axes project the checked
`GatedDeltaHeadGeometry` retained by their construction scopes. Fused Q/K/V row
ranges, head expansion, normalization widths and rank-local state dimensions use
that contract. Source permutations, recurrence equations, gate activation and
observation/intervention support remain explicit family declarations.

Qwen4 recurrent mixer descriptors obtain convolution geometry from the gated-delta
construction specification. Target parameter groups use the same declarations for
ordinary slots and routed/shared banks, while selected encodings supply physical
companions. Source permutations do not change logical parameter identities.

Qwen4 indexed attention obtains parameter groups, head geometry and channel widths
from `IndexedParameters`. Its main slots use the shared gated self-attention
specification; replicated selection heads use checked neural head geometry.
Family equations validate gate row order, key/value widths, normalization and
residual output coupling. Indexer selection budgets, cache state, rotary positions
and observation phases remain explicit invocation contracts. GGUF's separate index
query/key sources project one fused logical slot without introducing extra module
parameters. Tensor placement localizes attention heads and projections while the
indexer remains replicated.

Qwen4 shared/routed descriptors project routing counts, shared width, geometry and
parameter groups from the shared construction declaration. Invocation layer and
bank coordinates remain separate from those slots. Packed, split-bank and
individual-expert source layouts reference the declared logical bank; source
packing and quantization companions do not establish observations or interventions.

Qwen4 target and prediction vocabulary nodes reference `StaticParameters` and its
output projection. Tied output nodes share the token table's canonical parameter
group. Independent heads retain separate ownership. Residual-collapse groups and
vocabulary axes project declared slots; capture points, execution phases and allowed
operations remain explicit observation contracts.

Qwen4 target inspection projects complete unit declarations through the shared
runtime parameter-group visitor. Primary slots reference their retained linear
formats, so companion geometry and encoding alignment come from storage extensions.
Complete-scope validation rejects conflicting identities across invocations;
provider-owned row tables and execution state retain separate ownership.
Preparation identifies expert conversion targets through the declared bank slots;
rank does not establish a bank role or choose companion identities. Physical source
matrix eligibility uses retained encoding references. These storage projections do
not create execution, observation or intervention support.
SafeTensors configuration and header admission share an immutable Qwen4 family
declaration for target and embedded prediction owners. Discovery borrows retained
static, routed, recurrent, indexed and lexical scopes for both SafeTensors and
GGUF. Provider-owned table sources and invocation state remain separate contracts.

Target and prediction format callbacks borrow typed primary slots and their
retained format declarations. Storage selection preserves operator arithmetic and
exact-zero policy, and validates those policies alongside ownership and geometry.
Packed expert projections retain bank geometry while applying physical formats.

Qwen4 prediction-fusion parameter references project its shared construction
specification. `PredictionSharedSpec` owns fusion and collapse slots; complete
prediction and depth visitors validate those references across all parameter scopes.
Shared weights have a static owner and decoder weights retain depth-local owners.
Target, prediction and vision recipe requirements project identities, geometry,
roles, encodings and ownership from construction declarations and check source
recipes for matching stored shapes and completeness. Full-residual normalization
and per-stream projection retain their distinct geometry; target vocabulary parameters remain separately
owned. Prediction invocation state and observation permission do not follow from
these parameter roles.

Shared Qwen vision parameter groups project the construction declarations and
their retained encoding references through the common runtime visitor. A typed
block scope supplies checked multi-head axes, fused Q/K/V segments, residual
normalization and feed-forward slots. Local head/channel resolution and cold
memory invocations use these same identities and geometry. Affine
normalization uses `LayerNormalizationConstructionSpec` for dimensions, gain,
bias, epsilon and arithmetic across construction, admission, placement and cold
memory reporting. Family
placement rules retain QKV segments, output and MLP axes, and explicit ingress,
final-merger and DeepStack consumers. Checkpoint admission borrows those same
encoding references. Parameter storage and invocation state remain separate.

K2 Horizon's complete parameter declaration composes ordinary decoder slots with
separate routed-value, routed-SwiGLU and shared-expert ownership. The feed-forward
scope composes `RoutedGatedProductSpec` with its optional shared branch. Value banks
retain their independent input/output geometry and activated-linear equation.
Construction and expert planning project local banks and shared channels from the
same retained block declaration. Source format
projections retain typed references to logical slots, including individual expert
sources belonging to fused neural banks. Cold placement consumes the retained
router, routed-value, expert-bank and shared-affine encoding references directly.
Head ownership, intermediate partition domains and bounded residency remain
explicit. Affine transform eligibility uses declared geometry. Shared component
reads and writes reference the complete `SplitGatedProductSpec` used by construction, including its resolved
gated policy and sharing targets. Input/output channel extents come from its named
affine slots; explicit hooks retain their meaning, axes and allowed operations.

K2 Horizon's routing nodes and routed-value component reads reference the selector
declarations used by construction, source admission and replicated placement.
Selectable counts, normalization, score transforms and correction-bias identities
come from those declarations. Routing observation phases and allowed interventions
remain explicit contracts.

Nemotron-Labs-Diffusion text discovery borrows the complete decoder declaration
retained by family configuration. Static vocabulary ownership, head geometry,
gated connectivity and normalization slots come from the same scopes as source
admission and native construction. Common decoder discovery requires retained
scopes; parameter geometry does not establish observation permissions.

Nemotron-Labs-Diffusion's draft adapter nodes reference the same declared down/up
matrices used by source admission, local construction and parameter placement.
Admission retains complete draft scopes once and shares them with text and media
plans. Discovery borrows each retained layer's projection rather than rebuilding
geometry from configuration. The input affine follows the base attention head
selection; rank and output remain replicated. Adapter invocation scope, cache
writes and observation hooks remain separate execution contracts.

Pixtral attention reads its Q/K/V/output slots and head axes from the same neutral
`SelfAttentionProjectionSpec` used by construction and checkpoint admission.
Family declarations retain its per-image visibility and two-dimensional rotary
semantics, with explicit observation phases and allowed operations.
Nemotron-Labs-Diffusion projects Pixtral parameter groups and selected head/channel
placement from the same ingress, block and spatial-projector declarations used by
construction and checkpoint admission. Discovery borrows the source tower retained by validated Pixtral configuration
and reads its declared residual and head geometry. Selected local invocations
project retained slots without changing their source ownership. Spatial-projector
nodes borrow the family-retained normalization, merger and projection identities
and dimensions; their observation paths and permissions remain explicit.
Per-image visibility, rotary positions, residual equations and allowed
projector observations remain explicit execution contracts. Its
feed-forward slots and fixed SiLU policy belong to one `SplitGatedProductSpec`,
shared by construction, source admission and placement. The enclosing image layer
validates residual connectivity independently of parameter sharing and hooks.

Mimi's codec declaration supplies canonical parameter identities, exact geometry,
component groups and semantic roles to admission and construction. Codebook
initialization, EMA usage and vector sums remain distinct from code vectors.
The full released checkpoint catalog includes inactive codebooks; the retained
active declaration determines materialized parameters. These parameter facts
establish neither streaming invocation state nor capture support.

VoiceChat conditioning owners project character tables and affines, frozen speaker
features, the right-multiplied prompt matrix and gated fusion from construction
declarations. Character and output feature slices retain independent placement;
flag tables, controls and normalization remain replicated. Scalar residual scaling
and featurewise gate logits have distinct roles. Prompt phases, guidance lanes and
explicit capture contracts remain independent from parameter ownership.
Physical conditioning groups consume these retained slots and encoding references
through the common runtime visitor. The prompt matrix retains its declared storage
orientation; optional flags and speaker controls retain their declared replicated
acquisition owners.

VoiceChat Conformer owners project normalization, feed-forward, attention, head-bias,
and convolution parameters from their typed construction declarations. The ordinary
attention affines compose `SelfAttentionProjectionSpec`; relative-position projection
and bias roles remain family declarations, with bias shapes derived from its head axes.
Local attention heads and both feed-forward branches use the selected shared domains;
normalizations, convolution and causal state retain the global width. Content and
relative-position query biases have distinct parameter roles. Those roles and
source-storage transformations add no capture or intervention hooks.

VoiceChat perception boundary owners project the typed subsampling kernels,
flattened encoder affine, and perception-to-language affine retained with Conformer
slots in `PerceptionDeclaration`. Discovery, construction, preparation and complete/
split source extensions borrow that published scope. Selected precision remains an
explicit execution policy, transferred with the declaration to pipeline construction. Logical OIHW geometry remains distinct from native OHWI
views and split-container bias singleton axes. These boundary parameter groups
retain independent leases and explicit replication; they establish no additional
streaming state or capture support.
Physical perception groups consume these same declarations and retain explicit
complete-head and feed-forward partition domains. Raw convolution parameters and
affine gains remain replicated; parameter roles add no internal capture points.

VoiceChat RNNT parameter ownership and selected vocabulary sharding project the
typed embedding, LSTM gate affines, and joint projections retained by construction
and source admission in the published `RnntDeclaration`. Projection-only and predictor
providers select phases from that scope; discovery borrows its global slots and phase
membership. Embedding, recurrent-layer, and joint invocation identities
remain independent from parameter roles. Recurrent carry and explicit observation
contracts remain separate declarations; parameter geometry adds no capture support.

Gemma 4 and DiffusionGemma readout discovery borrows the immutable pinned text
scope retained by its family configuration. The embedding owns tied readout
identity and encoding; independent heads retain their own slots. Per-layer input
tables and projections retain their complete source geometry and static owners.
Selected vocabulary ranges remain placement projections over those declarations.

Gemma 4 and DiffusionGemma family configurations retain immutable media-to-text
projection scopes. Discovery borrows the same identities, geometry, normalization
and encodings as checkpoint admission, ownership and materialization. Selected
formats project onto those scopes; their parameters establish no hook support.

Gemma 4 vision/audio projector nodes and DiffusionGemma's image projector reference
their declared media-to-text matrix owners. The weightless input normalization adds
no learned gain slot. Parameter groups and replicated placement use the same
projection declaration; existing media capture phases and support remain explicit.

DiffusionGemma's family configuration retains feedback and independently owned
encoder residual-scale scopes. Discovery borrows their identities, geometry and
equations from the same objects used by source admission and native construction.
The refinement node references the feedback declaration's parameter
owner and hidden width. Its gain and complete `SplitGatedProductSpec` share the
construction, source-admission, placement and iterative-topology declaration. The
fixed GELU-tanh equation and normalization remain family constraints. Context and
refinement invocation structure and the explicit capture catalog remain separate
contracts.

Gemma 4 and DiffusionGemma readout descriptors project static token-table ownership,
tied or independent output weights, and the final gain from the static construction
declaration. A tied projection references its embedding owner and encoding rather
than introducing another parameter. Per-layer input table/projection groups use
the same typed static slots as admission and materialization; they establish no
additional internal hooks.

Gemma 4 and DiffusionGemma dense component geometry and read/write identities
come from their split-gated construction scope. The scope's resolved policy
selects exact or tanh-approximated GELU in discovery, matching native execution;
residual normalization and hook availability remain explicit family contracts.

Gemma 4 and DiffusionGemma attention reads, output writes, head geometry and gains,
and dense gated-MLP references project their ordinary construction specs. Checked
`AttentionHeadGeometry` supplies query, K/V and attended widths; the family validates
owned K/V and key-gain presence against the invocation's state policy. Shared
attention consumers reference the declared publisher's K/V matrices and key gain;
key-as-value equations reference the key matrix for the value read. Invocation,
state, hook phases, and allowed operations remain separate from parameter ownership.
Admission and physical groups consume those declarations' retained encoding
references. Head, intermediate, per-layer-input and fused-bank placement remain
explicit family rules; physical companions retain their declared primary owner.

Gemma 4 block normalization transforms project their gain identities and epsilon
from typed construction specs, including the per-layer input contribution gain.
Whole-residual scaling references the declared scalar slot. DiffusionGemma's
encoding and refinement scales remain separate parameter groups even though
both invocations reuse the block matrices. Parameter presence does not establish
capture, intervention, or backend support.

Gemma 4 and DiffusionGemma routed components borrow expert geometry, encoding
slots, and dense/sparse branch normalization references from their family-retained
sparse construction declarations. Source schemas, split/fused layout extensions,
recipes, placement, expert residency and native units consume those same owners.
Local bank axes project the source while selector axes remain global. Its neutral `RoutedGatedProductSpec` supplies selector
counts, scoring, normalization and bank parameters. Gemma 4 declares selected-only
softmax with no selected-sum normalization; DiffusionGemma declares full softmax
and selected-sum normalization. The parameter root identifies the actual shared
bank; encoding and refinement keep independent invocation and state identities. Hook
meanings, phases, and allowed interventions remain explicit contracts.

Muse `DecoderConfig` retains pinned vocabulary, attention, dense/routed and scalar
construction scopes alongside independent invocation policy. Its complete logical
visitor includes retained media scopes. Discovery borrows these source owners;
selected head and intermediate dimensions determine local bindings and state.
Muse block normalization descriptors read their gain identity, offset, epsilon,
and equation from the retained scalar construction specs. Residual cycles reference
those declarations' learned logits or independently stored coefficients. The
readout references the static-module declaration, including its actual embedding
owner when weights are tied. Hook phases and allowed operations remain explicit;
parameter declarations do not establish capture or intervention support.

Muse target attention component reads and writes reference the typed projection
slots used by construction, checkpoint admission, and placement. Target and DSpark
compose `SelfAttentionProjectionSpec` for the four attention affines and checked
head axes; the independent sigmoid gate and optional learned GGUF Q/K gains remain
family declarations with validated width coupling. Dense SwiGLU reads and writes
use the same typed expansion and contraction slots as target and DSpark
construction. Head counts and widths project from those declarations. Channel
hooks, normalization equations, phases, and interventions remain explicit and do
not follow from the presence of a parameter slot.

Muse routed bank groups and compressed expert component references project the
selector, packed bank, latent projections, and normalization construction specs
retained by immutable `DecoderConfig`. Format projection preserves their geometry,
equation and companion ownership; media-source selection shares the text owners.
The direct declaration projects `RoutedGatedProductSpec`; the compressed declaration
requires the complete decoder-to-latent-to-decoder path. Parameter discovery uses
the corresponding typed visitor and retains the global selector with local expert
geometry. Routing controls retain explicit scoring and selection order. Global route
identities remain distinct from local bank ownership, mutable invocation state,
and the explicitly declared observation phases and interventions.

Gemma and DiffusionGemma attention components borrow retained affine and head-gain
scopes. Their declared publisher references determine state reads, while shared-KV
consumers retain only query/output parameters. Global and selected cache geometry
uses the same scopes, including Gemma prefix state and Diffusion context-only
histories. Local heads and source aliases project retained geometry; permitted
unused GGUF K/V tensors remain source exclusions. These declarations establish no
additional observation hooks or backend capabilities.

Gemma and DiffusionGemma dense MLP components borrow their family-retained gate,
value and contraction declarations, including the resolved GELU equation. Source
schemas, aliases, placement and native materialization consume those same scopes;
local intermediate axes project the retained source geometry. Matrix formats and
parameter identities establish no additional capture hooks or execution support.

Gemma and DiffusionGemma decoder block gains, residual multipliers and optional
layer-input projections come from their immutable family scopes. Discovery borrows
the same declarations as source admission, placement and native binding. Namespace
aliases and local channel widths project those scopes; shared decoder parameter
ownership remains separate from Diffusion's independent encoder scale and invocation
state. These parameter facts establish no additional observation hooks.

Gemma assistant vocabulary, fusion and decoder parameter catalogs project its
retained construction scopes. Checkpoint requirements and source aliases consume
those same scopes, including shared-KV omissions and routed bank layouts. Tied,
independent-output and ordered vocabulary ownership remains distinct from unused
GGUF vocabulary source tables. Source extensions and parameter roles establish no
assistant capture hooks or backend support.

DSpark scoring groups project the same vocabulary, transition, confidence and
exact token-mapping declarations used by checkpoint admission and unit construction.
Acquisition boundaries and complete scoring replicas remain explicit; these
parameters do not establish assistant observation points.
DFlash2's candidate-table and hidden-projection declarations also supply released
checkpoint geometry and acquisition ownership. Their invocation graph and
conditional-score equations remain explicit; table declarations establish no
additional capture hooks.
Glimmer and DFlash2 body slots share typed context, attention, SwiGLU, gain and
dynamic-convolution declarations across construction, admission and acquisition
ownership. `SelfAttentionProjectionSpec` retains their attention affines and head
axes; family validation couples residual widths, rotary dimensions and Q/K gains.
The complete `SplitGatedProductSpec` retains their SwiGLU connectivity and policy.
Memory inspection reads the declared head and block-convolution geometry.
Muse assistant source schemas visit the retained complete definition before
selection; dtype rules, GGUF names and quantization companions remain explicit
format extensions. Admission, inspection and materialization share those scopes.
Parameter slots, selected formats and companions establish no capture hooks. Rotary,
residual and block-local convolution equations retain their explicit invocation
contracts; parameter roles establish no assistant capture points.

Target, DSpark and media physical groups consume construction-owned encoding
references through the common runtime visitor. Format companions retain the
same unit or pinned-resource owner as their primary declaration.

Muse waveform residual blocks and the vision adapter project ordinary affine
geometry from unary construction scopes. Exact GELU policies, the adapter's
second activation and patch/projection width coupling remain explicit declarations.
Storage aliases and independent static-resource consumers reference those same
slots. Readout and waveform capture phases remain separate hook contracts.

Muse's vision encoder and projector parameter groups project their typed patch,
position, affine-normalization, block, and adapter construction slots. Attention
uses `SelfAttentionProjectionSpec` with family-validated MHA, residual and rotary
geometry. Static replicas retain explicit consumer ownership in placement. Vision hook meanings,
execution phases, axes, and collector support remain separate declarations.

Muse's audio encoder and projector groups reference the waveform construction
specs used by checkpoint admission and bounded execution. The specs retain
per-patch and per-convolution ownership and state geometry; attaching parameter
groups does not declare internal waveform observations or intervention support.

Inkling routed components obtain global selection counts, correction-bias identity,
projection geometry and raw route-scale ownership from the neutral
`JointGroupSelectorSpec` used by construction and admission. Its combined global
axis has checked selectable/shared cardinality. Routed and always-on bank geometry
and encodings come from the same complete sparse construction declaration. Rank-local expert ownership and the existing jointly normalized coefficient
equation stay separate; these declarations add no standalone router events.
Pinned readout references and normalizations project the shared text declarations.
Decoder storage groups lower their retained encoding references directly, with
partition axes and fused gate/up segments supplied by family semantics.

Inkling's target and prediction attention components project head geometry,
projection identities, Q/K normalization policy, the relative-profile table and
causal convolution geometry from their common construction declarations. Inkling
normalization retains all target common scopes, and discovery borrows the exact
invocation rather than reconstructing it. Selected rank-local state projects the
same declared head and history channels. Inkling and LFM2 compose the neutral `SelfAttentionProjectionSpec` for Q/K/V/output
affines and checked head axes. LFM2 retains its rotary and Q/K normalization
coupling; Inkling retains its relative-profile and convolution equations. Execution
policy and explicit hook meaning, phase and allowed operations remain independent
of parameter ownership.

Inkling prediction scopes borrow fusion, normalization and readout identities from
the complete prediction and target declarations. Normalized `ModelArgs` retains
all prediction scopes; discovery borrows their common and dense block specs,
including local-attention geometry, without reconstructing depth overrides. Target
and prediction readout discovery borrow the same retained token table, untied head,
post-embedding gain and final normalization declaration. The two hidden-normalization
invocations reference one learned slot; hidden-first fusion column ranges come
from the declared geometry. Shared chain normalization remains separate from
per-depth state and hooks. Declared parameters do not create additional capture
points or change the component equations.

Inkling's dense target and prediction components project gate/up/down identities,
intermediate geometry and the learned branch scale from retained construction
declarations. Target dense/routed ownership follows the normalized schedule's typed
branch scope. Routing, shared-bank geometry and encoding companions project from
that same retained source, without rebuilding router or bank catalogs. The dense scope composes a complete `SplitGatedProductSpec` with
the learned branch scale; discovery projects its resolved gated policy and sharing.
Normalization, scalar observation phases and allowed operations remain explicit
component contracts.

Inkling's media groups borrow the dMel embedding/normalization and bounded hMLP
stage declarations retained by normalized media configuration. Source extensions retain released SafeTensors
and mmproj GGUF identities and operation classes; selected storage retains each
tower's own encoding policy. Media nodes attach logical parameter references and
output widths from these same declarations. They retain the existing capture
points and component graph completeness declarations; parameter geometry and
roles do not establish internal media hooks.

Moshi's temporal/depth graph borrows the complete construction scopes retained
by normalized source configuration. It projects parameter groups and PersonaPlex
depth-norm sharing from those declarations. Physical placement and matrix contracts
project the same retained encoding references and companions. Its graph remains
coarse, and ordinary
prefill/decode observation discovery does not establish realtime frame hooks.
Realtime invocation, state and channel contracts remain separate declarations.

The initial detailed projections cover Llama/Mistral, Qwen2/3, GPT-OSS, Qwen
hybrids, LFM2, Kimi Linear, Nemotron-H, and DeepSeek V3. Multimodal components
reuse existing Gemma, Inkling, and Muse component graphs; complex block
interiors, DeepSeek V4 hyper-connections, embedded prediction, and Moshi frame
execution have explicit partial descriptions. Their catalogs never fabricate
missing internal measurements. Conditional Qwen adapters currently expose block
and media boundaries, without normalized routing events. Media projector points
require media input during prefill and explicitly report decode as unsupported.
Partitioned support requires both retained invocation placement and callable hooks;
the logical graph remains independent of rank placement. All-dense V3 target
execution verifies both for TP, PP and combined TP/PP. Target-only sparse V3 TP now emits the same internal hooks, including additive
shared-write terms, and passes neutral TP/PP/residency conformance and native
TP/PP/EP acceptance for F32 SafeTensors/GGUF and load-time affine 4-bit/group-32.
Embedded prediction interiors remain unverified.

Architecture descriptor schema 4 includes component schema 3's `write_partition`:
`Complete` or `TensorParallelSum`. The latter declares full-hidden-width native
terms before an ordinary TP reduction; equal hidden coordinates across TP ranks
are distinct contributions. Placement requires complete TP groups and retains one
producer per TP coordinate across replicas. Partition evidence independently
specifies bounded host assembly arithmetic. The declaration alone does not enable
loaded hooks. Mixed V3 F32 SafeTensors/GGUF and load-time affine native TP/PP/EP
acceptance passes all 63 ordinary-residency cases.

Run the portable example:

```sh
cargo run -p eredu --no-default-features --example architecture_discovery -- /path/to/artifact
```

The example also contains a compiled generic `capture_advertised` helper using
an application's existing `ModelRuntime`. A complete native example inspects,
prints support, selects an advertised path, loads the model, and captures both
prefill and decode values:

```sh
cargo run -p eredu-backend-mlx --example capture_discovery -- /path/to/artifact 1 2 3
```

Provide valid token IDs for the artifact. This example explicitly chooses CPU
execution; native resources are created only after cold discovery.

LM Inspector should retain the inspected descriptor, render `nodes`/`edges`,
list `observations.points`, and join selected support by exact `path`. Offer
capture for `Supported` points, show conditional requirements, and display
unsupported/unverified reasons. Pass the selected path to
`ObservationSelector::Exact` and use the existing `inspect_prefill` /
`inspect_decode` calls. Do not use an empty selector list as “capture none”:
the existing observation API treats it as capture-all. Missing returned values
must remain missing, not become numerical zero. For ordinary instrumented text
generation, use `capture_discovery`, `prepare_observed_chat`, and
`generate_observed`; `CapturePlan::none()` and empty new-plan selections both
mean capture none. The complete [`observed_generate`](../eredu/examples/observed_generate.rs)
example includes prompt alignment, ordinary sampling, bounded capture and cancellation.

Mixed V3 independently cached expert variants pass all 63 CPU Ring TP/PP/EP and
residency cases for F32 SafeTensors/GGUF and affine 4-bit/group-32. Parameter
discovery reflects the effective source-preserving companion dtype and retained
generated destinations. An owned expert with no selected routes remains declared;
zero local cache activity does not mean its parameters or topology are absent.

Architecture descriptor schema 5 adds `component_scopes` (empty when absent in
older JSON), using component schema 4. Primary `components`, `routed_components`
and `component_readout` still describe target scores only. Each V3 prediction
depth has a separate scope joined to its execution group, decoder, routed/shared
components and its own readout head. Its residual starts with a linear projection
of separately normalized token embedding and supplied hidden state. Fusion input
column ranges and effective parameter names are declared; clients need no
checkpoint-name parsing. The supplied hidden source is the actual invocation
capture, not an assumption that every invocation consumes a preceding depth.
V3, sequential V4 and DSpark prediction descriptors project parameter prefixes,
sharing targets, optional biases and vocabulary geometry from their construction
specs. V4 prediction stream axes use the hyper-head declaration. Normalization
prefixes follow the declared neural role; parameter geometry adds no hook support.

All prediction-scope observations, including generated projection inputs and
routed-unit values, require `PredictionExecution`. Selected target-only call paths
report them unsupported even when their tensor collectors are available. A full
declaration therefore does not advertise public speculative component support.
The shared readout equation preserves existing primary JSON fields; Rust struct
construction now places head fields in `ComponentReadoutEquation`.


Internal V3 speculative calls now carry explicit invocation phase, proposal depth
and physical sequence width through the shared architecture strategy. This does
not change loaded capture support: a typed hook fact alone cannot supply public
phase/frontier admission, cumulative collection or record delivery. The existing
`prediction_inspection` gate remains false in ordinary and cold adapters.

Descriptor schema 7 extends the separately scored component domains with a fused
prediction scope spanning ordered `execution_groups` and separately declared
`static_parameter_roles`. Component schema 6 adds direct residual sources with
explicit expansion and dynamic `score_writes` in vocabulary space. These joins
preserve DSpark's multiple decoder blocks and anchor-dependent Markov addition
without treating either block count as proposal capacity or dynamic scores as a
static bias. `speculative_invocations` declares context-cache roots that execute
without a component score equation. Descendants inherit the nearest explicit
invocation binding; ambiguous or unknown roots fail validation. Loaded support
still requires the corresponding selected executor and native collector facts.


Descriptor schema 8 carries component schema 7's optional `output_gate`. This is a
per-row scalar multiplier after the affine write and any output normalization.
It declares the gate's effective input, actual projection input, exact read
parameters, activation and original/effective observations. The gate is measured
again for every trial. Qwen shared FFN groups retain ordinary gated-unit identities
and add this separate post-projection relationship. Older groups omit the field.


Schema **9** adds `component_transforms`, with an empty default for older
serialized descriptors. Component schema **8** declares identity normalization
and exact tensor transformations between observed boundaries: constant or learned
scaling, normalization, and causal depthwise convolution. Each transform names its
owning node, actual input, original output, optional effective output and canonical
parameter identities. Node ownership locates the transform in a target or
prediction invocation. These records are equations, not extra additive residual
contributions and not loaded-operation capabilities. For causal kernels, the last
tap consumes the current input; earlier taps may consume retained state. The
ordinary capture API admits only the requested current positions. Inkling's
expanded transform declarations currently supplement its partial component graph;
complete scalar groups, prediction scopes and native component support remain work
in progress.

Descriptor schema 11/component schema 10 describes gated-delta recurrence with
separate projected key-head and recurrent value-head geometry, exact per-head or
per-key-channel decay, and normalization/gating before scalar output channels.
Qwen recurrent channels use `.mixer.channels`; Kimi channels use
`.attention.channels`. Both join the actual pre-projection tensor and effective
write columns. Fused causal Q/K/V transforms preserve their physical segmentation.
The L2 normalization kind uses additive epsilon under the square root of the
sum of squares, while RMS uses the mean. Captured channels already include these
stateful calculations; discovery does not imply historical state observations.

Descriptor schema 12/component schema 11 adds `scaled_softcap`, projected-head
`output_scale` (default one), and `token_embedding_normalization` before media
assembly. The existing `embedding_normalization` describes normalization of the
assembled tensor; consumers must keep those stages distinct. Muse declares its
per-channel sigmoid gate and four separate pre/post RMS normalizations, with
HF centered gains and GGUF full gains. Sparse postnorm applies after summing
the complete route-weighted write. Loaded support still depends on the selected
executor and collector mechanisms.


Nemotron prediction boundaries borrow first/last unit scopes from the complete
family declaration. Boundary ownership remains distinct from the scheduled block,
invocation identity and mutable state. Complete prediction scopes include both sets
of parameters for validation and preparation; the boundary projection exposes only
the normalization/fusion slots used by boundary equations. Hooks retain their explicit
meaning, execution phase, axes and allowed operations.


Nemotron source plans retain typed scopes across immutable inspection snapshots.
Selected format and ordinary-target projections preserve parameter identities and
ownership while excluding prediction invocations from the target projection. Cold
placement validation compares the retained declaration with selected component
parameters and state, independently of backend allocation. Prediction source and
executable scopes retain their own explicit invocation and hook contracts.
Encoding selection visits the retained ReLU-squared bank slots, target matrices,
static vocabulary matrices and prediction fusion slots. Geometry, arithmetic and
exact-zero input policy stay with those slots while physical companions follow
the selected encoding. The bank visitor retains contiguous row origins and
separate up/down identities without reconstructing its group geometry.


Nemotron target and prediction discovery project static readouts, attention channels,
dense units, shared experts and routed banks from retained typed scopes. Normalization
references borrow the unit declaration; source parameter prefixes follow the declared
affine identities. Explicit invocation graphs and hook declarations retain their own
semantics and permissions. Expert bank ownership remains distinct from invocation
ordinals and selected local placement.


Nemotron local state and vocabulary placement project retained neural scopes and
selected parameter identities. State policies preserve target/prediction segment
boundaries independently of shared parameter ownership. Cold state forecasts do not
establish an encoded matrix's execution support or add hook permissions.
Nemotron capability and generation-memory projections borrow the retained state
layout and static width. Hybrid strategy counts cover target invocations, while
complete state estimates retain prediction segments. VoiceChat cache admission
borrows its prepared backbone and causal TTS declarations; the bidirectional
character encoder has no persistent causal cache.
VoiceChat transformer state geometry projects retained attention slots for both global
and selected local widths. Runtime cache borrows use the selected component definition,
including its separate normalization stage and stateless character invocations.


Cold decoder geometry and parameter-topology projections require retained
common declarations; state reporting resolves family-owned scopes for K2 and
Qwen and retained text scopes for composite diffusion models. Missing retained
slots reject before backend allocation.

Dense decoder component and memory-topology projections use `DecoderDeclaration`'s
static, attention, gated and normalization slots. Repeated invocations retain their
independent state while parameter ownership follows the declared sharing targets.
Logical geometry and stored encoding are separate projections of those slots;
neither projection grants observation permissions or backend execution support.
Llama-compatible, Gemma2, Nanbeige, ordinary dense Qwen2/Qwen3 and dense K2 generic inspection
retain the scopes admitted by their
SafeTensors or GGUF plan. Source-role joins, static ownership and boundary geometry
project those scopes. Discovery's attention axes, intermediate extents, terminal
normalizations and readout ownership use the retained slots. Execution names,
component equations, completeness and hook permissions remain explicit declarations.
Capability reports use retained state geometry alongside the family's context policy.
Nanbeige discovery projects shared body and inter-pass normalization references
from the same retained scopes used by physical checkpoint admission. Repeated
invocations retain independent cache state and observation identities.
Dense Qwen2/Qwen3 discovery uses its admitted attention, head-normalization and
feed-forward scopes. Window policy, component equations and observation permissions
remain explicit family declarations; parameter scopes do not grant hook support.
Dense K2 discovery reads retained grouped normalization, head-owned Q/K gains,
attention output gates and dense feed-forward slots. Routing and value-bank hooks
remain explicit family contracts for configurations that declare those invocations.
Format selection preserves identities and logical geometry;
the family rejects changes to non-format execution policy before preparing the
selected parameter catalog.
Dense local projection retains the declared identities, sharing targets, arithmetic
and normalization policy while selecting head and intermediate widths. Fused segment
layout and local state use those same projected slots. Inconsistent shared geometry
and dimensions outside the retained scope reject before native materialization;
vocabulary residency remains an explicit static placement contract.

LFM2 component parameter descriptions project the family declaration's typed mixer,
normalization and feed-forward scopes. Tied static ownership and selected physical
companions preserve canonical identities; arithmetic and grouped reduction policy
remain part of the retained neural specs. Parameter projection creates no additional
observation or intervention permissions.

Complete and partitioned LFM2 construction retain this same declaration. Rank-local
block scopes preserve parameter identities and arithmetic while projecting selected
head, convolution and expert dimensions. Ordinary global state and vocabulary geometry
project retained scopes; invocation placement and mutable state ownership remain explicit.
Local geometry validation and cold state forecasts use these scopes and the retained
component topology. Expert ownership projects declared banks at the selected EP and TP
coordinates, independently of pipeline invocation ownership.
SafeTensors and GGUF admission retain the declaration used for their schema projections;
source recipes and expert residency use those same scopes. Physical layout alternatives,
dtype constraints, aliases and encoding companions remain explicit checkpoint extensions.
LFM2 graph attributes, component relationships and readout normalization borrow the
retained neural scopes. Execution-memory topology and capability state estimates use
the same declaration. Routed hook sites remain explicit invocation contracts with
declared bank cardinality; component equations and allowed operations are specified
independently of parameter ownership.
LFM2's generic text-inspection value also retains this declaration. Parameter-role and
source joins, tied-readout ownership and state eligibility project retained scopes,
without a separate configuration-derived catalog.

Kimi Linear parameter descriptions project retained static, KDA/MLA, normalization and
dense/shared/routed scopes. KDA history and recurrent geometry and MLA latent state
project the same typed mixers. Encoding projection preserves logical identities and
operator policies; these parameter declarations add no observation or intervention
permissions.

Complete and partitioned Kimi construction retains the same declaration. Local KDA and
MLA scopes preserve identities and policies while projecting selected head dimensions;
dense, shared and routed intermediate widths follow their retained slots. Compact bank
overrides preserve source identities and encodings. Invocation placement and mutable
state ownership remain independent of shared parameter ownership.

Kimi local geometry validation and state forecasting use retained neural scopes and
component topology. EP bank authority and TP shared-expert intervals reference the same
declared parameters. KDA histories follow selected heads; MLA compressed state retains
its head-independent dimensions. These projections add no backend support claims or
hook permissions.

Kimi admission retains the declaration used for SafeTensors and GGUF schema projection.
Source recipes, ordinary shape joins and expert residency borrow those same scopes.
Published aliases, KDA storage transforms, GGUF transition-rate conversion and expert
layout alternatives remain explicit checkpoint extensions.

Kimi graph attributes, component read/write equations, normalization references and
readout ownership borrow retained scopes. Capability state estimates and generic text
inspection use the same declaration. Routed hook sites remain explicit invocation
contracts with declared bank cardinality. Parameter ownership adds no allowed operation
or backend collector support.

GPT-OSS discovery reads common attention/static/normalization scopes, selector
cardinality and packed expert slots from its admitted family declaration. Parameter ownership remains separate from logical
invocations, mutable cache state and explicit observation/intervention permissions.
Selected placement comes from retained execution selection. External assistant
encoding selection projects its retained Gemma or Muse construction scopes;
logical identities, geometry, sharing and acquisition owners remain fixed while
physical weights and companions follow the selected encoding. These declarations
do not establish observation or intervention permissions.

Qwen2/Qwen3 text discovery reads common scopes and routed selector/bank facts
from the admitted `qwen::Declaration`. Static vocabulary sharing and body geometry
come from the same typed slots used for checkpoint admission and construction.
Split expert format projection resolves canonical bank owners from those slots;
Qwen3-VL uses the same text projection under its composite namespace.
Explicit routing hook declarations retain their operation permissions and phases.

Qwen ordinary and hybrid decoders, LFM2, Kimi Linear and K2 Horizon encoding
projections derive packed gate/up
row origins from the bank's component layout. Logical ownership and geometry stay
fixed while selected storage and companion shapes follow that layout and encoding.
The shared projection retains operator arithmetic, exact-zero input policy and
bank reduction order. These parameter facts establish no hook permissions.
Common decoder attention, feed-forward and vocabulary encoding selection retains
the slots' arithmetic and exact-zero policy. GPT-OSS selector selection and Qwen
hybrid target/prediction storage use the same contract; native GPT-OSS expert
encoding and explicit observation declarations remain independently owned.
Ordinary Qwen text, GPT-OSS, LFM2, Kimi Linear and Nemotron-H preparation resolve
named encoding extensions against those owners or explicit primary checkpoint
source aliases. Tied redundant
output sources and native expert block sources retain their separate storage
contracts; companions and allowed artifact keys do not become logical owners.
Unknown or incompatible references reject before native allocation. Resolving a
source alias grants no execution capability or observation permission.
Nemotron-H target and prediction references resolve within the complete declaration;
their distinct router storage policies remain explicit. Independent expert source
storage does not add logical owners. Floating split-expert eligibility and raw
convolution or recurrent parameter roles remain family contracts.
DeepSeek V3/V4 selected encoding extensions resolve against retained parameter
slots and explicit primary source layouts. Split V3 GGUF reconstruction sources,
V4 native packed expert sources and independent prediction expert sources retain
their distinct storage contracts. Raw hyper-connection coefficients, routing
tables, position offsets and normalization gains do not acquire matrix encoding
permissions from their rank. Unknown or incompatible references reject during
preparation.
Llama/Mistral/Phi, Gemma 2 and Nanbeige resolve explicit format references when
their complete common decoder declarations are prepared or projected. Fused
projections retain fused identities; tied readout and repeated invocations refer
to their declared owners. Conflicting encoded sharing rejects without merging
invocation state or placement. Companion names and artifact exclusions do not
declare weights.
Gemma 4 and DiffusionGemma resolve storage references across complete retained
text, media and projector scopes. Split expert sources retain their explicit
encodings. Standalone towers defer only their declared external projector to the
family boundary; that boundary checks its geometry and encoding. Explicit
misaligned media requests reject without changing uniform quantization eligibility,
parameter ownership or observation permissions.
Inkling and Muse text/media directives resolve through a shared matrix-only view
of retained logical slots. Typed format membership supplies eligibility; tensor
rank and parameter spelling do not. Exact source aliases and independent expert
layouts remain family extensions, and parameter membership grants no hooks.
Muse split and independent expert storage references project typed packed-bank
owners and slice geometry. Normalization retains the source encodings while
inferring homogeneous executable bank formats; incomplete or conflicting requests
reject. Source identities add no parameter groups, execution invocations or mutable
state to discovery.
Gemma and Muse assistant preparation resolves format directives through complete
retained declarations and verifies selected encodings after projection. Family
aliases, ordered dense vocabularies and source-only parameters retain their
separate contracts; they do not create execution invocations or observation sites.
Shared Qwen vision encoding selection derives alignment eligibility from retained
affine geometry. Patch kernels and position tables retain their distinct roles.
Format extensions resolve relative or `model.visual` identities against declared
affine slots, including merger and DeepStack projections, before backend allocation.
Composite quantization retains those scopes and their arithmetic policies;
parameter storage eligibility adds no observation or execution support claim.

K2 source-format normalization retains the complete typed scopes used by discovery
and construction. Dense, routed/shared and routed-value owners share that authority
through parsing and storage selection. Projected physical sources reference typed logical
owners for packed-bank encoding agreement; physical aliases remain format extensions.
K2 discovery composes common slots, dense-prefix feed-forward stages, shared
experts and both routed banks from the admitted family declaration. Logical
invocations and explicit routing hooks retain separate identities and permissions;
selected rank placement remains an execution-selection projection.

Qwen3-VL discovery borrows the text declaration retained by composite source
admission, including GGUF projector admission before media token binding. Its
composite parameter report maps text body owners into the text execution group
and composes retained shared vision construction slots. Patch and position storage,
fused Q/K/V head domains, unary MLP channels and final/DeepStack merger consumers
project the same vision declaration used by admission and materialization. Selected
text state validates against the retained attention geometry; the composite adds the persisted mRoPE
position delta once. Parameter declarations do not grant observation permissions.

Conditional Qwen hybrid and Qwen4 configurations retain their shared vision
construction declarations. Their parameter descriptions and selected placement
project those scopes. Qwen4 header-only and bound vision roles carry the same
source declaration through processor and tokenizer binding; recipes and physical
source provenance remain separate from logical parameter geometry.

Hybrid target and prediction block parameter reports project complete typed
`BlockParameters` scopes. The recurrent/full-attention schedule and shared MTP
fusion remain explicit invocation semantics. Routed-bank placement references
those slots; selected local construction retains their parameter identities.

Hybrid discovery and parameter reports borrow the complete text scopes retained
by normalized source configuration. Format and target-only views preserve the
authoritative static and target identities. Local placement projects source
slots while state and observation invocation declarations remain independent.

Muse waveform discovery borrows the complete audio declaration retained by source
configuration. Patch, convolution and output owners project their parameter slots
and encoding companions from that declaration; convolution state uses the retained
channel and history geometry. Declaration membership grants no observation or
intervention permissions.

Muse vision discovery borrows the static and block scopes retained by normalized
source configuration. Format selection preserves parameter identity, semantic
roles and ownership while projecting physical companions. Explicit block/readout
observations, source position layout and image/video policy remain independent of
parameter membership.
