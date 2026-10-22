# Language-model backend architecture

Local artifact admission retains the exact primary checkpoint, shard admissions,
and explicitly selected or discovered GGUF companion roles in `ArtifactInspection`.
`ArtifactSource` and companion selection are neutral contracts in `eredu-core`;
architecture registries validate family geometry and composition. Explicit sources
bypass filename discovery and have no fallback. The portable facade exposes
canonical SafeTensors snapshot completeness from `eredu-checkpoint` independently
of architecture support and native backend availability.

The facade's `PreparedTextModel` retains resolved tokenizer, template, EOS policy,
and generation defaults once. Text probes borrow that state. Execution selection
retains the complete plan, target admission and its compatibility/resource report,
plus the selected drafting preparation and vocabulary proof. The MLX factory
obtains the report and native binding selection from one architecture inspection
outcome. `SelectedTextModel` borrows the selecting factory and consumes those
retained selections during realization; callers cannot replace its plan or text
options. Convenience path loaders use this same pipeline. Native source guards
remain authoritative for changed files and later payload leases. Metadata-only
inspection does not grant local text or payload loading authority.

Prepared generation, including retained external and embedded draft sources,
uses the request-local `DecodingMode` in shared generation settings. The
self-speculation variant owns its validated proposal and scheduler options.
Forecasts consume that same mode; ordinary diffusion retains full refinement
geometry and uses zero draft proposals.
Prepared target sources retain a declared shared-backbone draft automatically;
targets with ordinary refinement alone enter the causal/refinement executor.
Continuation forecasts retain complete adapter declarations and select workspace
from both the active and pending modes. Cold source forecasts compose refinement
and source phases while preserving loading peaks and counting shared parameters
once. Separately retained source state uses an explicit conservative upper allowance.

Decoding modes and their validated options are portable contracts in
`eredu-core::decoding`. A generation owns one command mailbox with immutable
realized capabilities; commands contain no native handles or checkpoint paths.
The shared physical driver attempts handoff through `settle_decoding_mode` only
after preceding completions settle. A deferred handoff retains the pending
command through the current block or verification transaction. Successful handoff
appends any committed suffix once, preserves next-target logits and clears
tentative state before `ModeChanged` delivery. Cancellation fences unacknowledged
commands. Native failures retain their existing completion and recovery owners.
Portable snapshots copy logical modes with cache, sampler and pending input;
branches retain independent command owners and charge their storage separately.

The core text machine advances architecture-declared physical operations through
`TextGenerationBackend::submit_text_operation`. Progress retains pending input
without querying or committing the token controller. A selecting operation alone
receives the current commitment decision. Native completion is retained before
publication, and the ordinary iterator and controlled facade use this same machine.
The shared facade streaming cursor parses only authoritative tokens; context and
refinement progress drain their own captures and remain pause boundaries.
Generation positions retain independent physical-step, block, iteration and
committed-token coordinates. Complete continuation snapshots save and restore
these logical coordinates while output epochs and resource consumption remain
monotone. Generation-event and execution-control schemas are version 2.

Tentative token IDs are opt-in request data. The facade owns draft revisions and
reconciliation identities; they never enter the tokenizer, constraint controller,
semantic parser or committed-token metrics. A draft clears before the block's
first authoritative token, cancellation, failure, restore or branch exchange.
Restoration advances the output epoch, and forks receive independent draft identities.
Physical capture bounds cover the complete context and canvas separately from the
committed prediction schedule; repeated refinement cannot refund capture budgets.

Seeded and fixed-position canvas contracts live in `eredu-core::refinement`.
The runtime validates them and preserves fixed tokens and one-hot feedback through
refinement; confidence averaging excludes fixed positions. Native backends supply
seed installation, masking and finite-candidate validation as general numerical
mechanisms. They do not compile question schemas or interpret alternatives.
Ordinary observed requests and controlled sessions share this state, including
snapshots and finalization. Explicit-position token-score capture retains the
requested coordinate in its admitted selection and reduces one complete vocabulary
row with bounded host output. Workspace envelopes cover masking and one-hot
feedback; snapshot reservations cover both immutable native pins and host layout.
The CLI owns question schemas, letter-template compilation, normalized option
probabilities and numeric level expectations. See [canvas controls](refinement-canvases.md).

`RefinementPolicy` distinguishes entropy feedback from masked-token unmasking.
The shared runtime owns both numerical schedules, immutable selected-token state,
score retention, exact-settlement publication and dependent-suffix invalidation.
Entropy feedback uses `EntropyRefinementOptions`; masked selection uses validated
`MaskedRefinementOptions` with an even transfer schedule or confidence threshold.
Native numerical implementations do not imply that a loaded architecture supports
a policy. Session admission checks the retained family contract and returns
`RefinementPolicyMismatch` before context execution for an incompatible policy.
DiffusionGemma selects entropy feedback. No loaded Nemotron-Labs-Diffusion family
is advertised by the architecture registry.

The portable `nemotron_labs_diffusion` modules declare one dense parameter catalog
for causal append and prefix-plus-block transient attention. Invocation values
carry visibility, cache-write policy, adapter scope and score semantics; shared
modules retain no mutable decoding-mode flag. Exact source-only recipes normalize
NVIDIA checkpoint names. Pixtral layers, patch embedding and the spatial projector
are separately addressable portable modules. The optional vision graph precedes
the canonical text group, and its ingress and projector have distinct first/last
pipeline-owner roles. Cold topology shards vision attention heads and MLP channels;
each branch reduces its residual write while preserving complete-width RMS norms.
Image admission proves patch payload geometry, grid metadata, host extent and the
complete delimiter/row-break sequence together. Context assembly replaces pad
embeddings with projected patches and retains ordinary embeddings for delimiters.
Image processing is owned by the family and floating cubic resizing by
`eredu-media`. Prepared composite input carries optional immutable attention
invocation authority without changing the admitted tensor identity. See
[released numerical evidence](nemotron-labs-diffusion-validation.md).

Nemotron text GGUF admission binds incomplete legacy metadata to an exact content
fingerprint produced by `eredu-checkpoint`. A generic `dream` or `mistral3` tag
does not identify this family. The family checks independent head width, YaRN,
reserved tokenizer IDs and every physical tensor before constructing one canonical
source mapping. Its explicit bounded fallback decodes selected quantized matrices
to F32. The largest 14B vocabulary projection requires 2,684,354,560 decoded bytes;
the selected reader allocation bound covers that matrix rather than the complete
model. Prepared binding and layer residency use the ordinary native mechanisms.

`eredu-nn::low_rank::LowRankProjection` retains its down/up parameters independently
of a base affine operator. Invocation owners borrow it explicitly. TP shards its
input matrix and reduces rank activations before applying the replicated output
matrix. The shared decoder composes this delta before the ordinary attention
residual and restores the base output precision after the sum. Family-owned cold
topology puts optional Nemotron draft matrices in their corresponding text-layer
residency units. Its pipeline boundary carries immutable invocation semantics and
the original canonical prefix position alongside the complete activation span.

The MLX masked numerical mechanism uses fixed-canvas host score copies and float64
noise arithmetic with an explicit 53-bit proposal stream. These temporary host
copies are separate from parameters and cache storage. Selected tokens and their
position-score rows remain fixed across later invocations. Finalization preserves
an admissible selected token; a constrained replacement remasks the dependent
suffix, preserves pinned positions and advances the existing cumulative invocation
count. The local transfer schedule restarts for that suffix without rewinding RNG
or outer capture, transport and commitment ledgers. Unresolved mask identities
cannot become finalized tokens, even when an output constraint permits only a mask.

Iterative token-block policy lives in `eredu-core::refinement` and
`eredu-runtime::generation::refinement`. The portable engine owns temperature,
entropy acceptance, confidence/stability stopping, feedback precision and RNG
adoption. Native settlement precedes publication of a new iteration. Entropy
feedback finalizes the last argmax prediction independently of its noisy canvas;
masked refinement finalizes retained unmasked selections. `BlockRefinementState`
retains the explicit random stream, optional feedback, stopping history,
transfer schedule, finalization cursor and authoritative
block. Context encoding consumes only finalized tokens. Position-wise constrained
finalization preserves the final argmax when allowed, otherwise selects the
highest-scoring currently admissible token from that position's final scores.
The generic MLX adapter realizes these operations from architecture-declared
geometry and inputs. It routes progress through the ordinary observation and
completion owners and queries output constraints separately at each finalization
position. Snapshot sampling state includes the active block and policy. Immutable
canvases and feedback may share storage; retained-byte reservations include every
value, while copy reservations account for mutable host state and native context.
Future context-growth reservations include an already finalized partial block
when a branch proceeds beyond it. `TextGenerationConfig::refinement_policy` and
the facade's `refinement` setting select a validated iterative schedule independently
of causal token-sampling temperature.
`BlockRefinementArchitecture` supplies context and feedback inputs to the shared
layered session. Cold requirements retain the positive canvas and vocabulary
geometry. `prepare_block_refinement_contract` checks it against the constructed
architecture and selects every sequence position. The session's observed context
and refinement operations reuse ordinary rollback, residency and publication.
Invocation output semantics suppress the model-score observation
seam for context-only outputs; refinement retains all position scores.

`ReadOnlyAttentionCache` borrows authoritative context history and attends with
transient key/value projections without appending them. MLX contiguous and paged
storage implement the same request; paged attention scans transient projections
alongside retained pages without installing pages or advancing their frontier.
Read-only scans clip physical page overhang to the logically retained suffix
and permanent prefix; the sliding history bound does not clip the canvas.
An explicit mask selects canvas visibility independently of the cache's append
policy. The model owns that visibility and the positions used by rotary operators.

`EmbeddingOperator::weighted_sum` consumes coefficients without sampling or
normalizing them. MLX casts coefficients to embedding arithmetic and bounds
packed-table decoding to 512 rows per block. Vocabulary-partitioned operators
consume their owned coefficient columns; the neutral distributed operation sums
the local contributions. DiffusionGemma declares shared text matrices, separate
encoder/decoder residual scales and decoder self-conditioning in
`eredu-architectures::diffusion_gemma`. Learned-normalization, GELU and attention
rounding are explicit portable construction choices. Its text lifecycle traverses
one shared unit per layer through the ordinary resident and bounded-runtime
contracts. Context invocations append authoritative KV; refinement invocations
borrow it. Shared Gemma parameter-placement functions accept architecture-owned
roots; DiffusionGemma separately owns feedback units and encoder scale placement.
Rank-local static construction includes only the retained ingress/readout roles.
Shared vision and modality-projector construction also accept architecture-owned
parameter roots; optional clipping remains an atomic projection policy.
Vision equations preserve the working precision across patch scaling and rotary
products. Learned normalization and spatial pooling use FP32 intermediates with
one output cast; magnitude scaling and standardization occur in FP32. Explicit
input-score attention arithmetic remains an architecture construction choice.
Shared routed banks retain one expert ownership plan for both invocation phases;
routed drivers pass the request's expert mechanism policy into the shared layers.
Resident, bounded and tensor-parallel unit calls use the same component instrument,
including intervention of the effective residual and sparse routing evidence.
Readout instrumentation covers every canvas position without causal slicing.
Its pipeline boundary carries the invocation phase and authoritative rotary
frontier as exact I32 values; receiving stages validate that frontier against
local context. The same shared layer equations and read-only cache request run
under tensor parallelism and combined tensor/pipeline partitions.

The image/text composition cuts the image tower into individual layer units.
Composite text boundaries additionally transport each query row's image interval.
Receiving stages reconstruct image-span visibility and intersect sliding attention
with its history lower bound while retaining visibility inside the image span.
Composite publication geometry follows the invocation: encoder hidden states
and refinement position scores have distinct final axes. Expert transport bounds
include the complete multi-rank fan-in and the return submission by its owner.
Text and image parameter ownership, local K/V geometry and transport schemas live
in the family; native adapters consume the retained construction and sources.

Checkpoint constraints distinguish executable tensors, format companions and
artifact metadata. Metadata tensors retain strict shape/encoding admission and
source provenance, but produce no executable materialization task. Published
rotary-frequency tables use this role when the equation derives frequencies from
the admitted configuration. GGUF tokenizer reconstruction selects explicit BPE
merges when present, with the declared whitespace and byte-fallback policy;
score-based Unigram vocabularies retain their own mechanism.
`eredu-checkpoint::expert` assembles selected expert banks and their encoded
companions together; stacking and concatenation preserve every scale and bias.

Cold invocation topology describes additive feed-forward branches and optional
whole-canvas feedback projections. Format selection visits every projection.
Forecasts include canvas scores, feedback and refinement scratch even when the
requested commitment ends before the next context append. Parallel invocation
and logical-state forecasts use a conservative unpartitioned envelope; parameter
residency uses selected rank-local tasks and replicas. Offload staging and native
collector limits remain explicit in the forecast's assumptions.

The shared gated-delta mixer and shared/routed feed-forward construction live in
`eredu-architectures::{gated_delta, shared_routed}`. Construction specs carry exact
parameter identities, linear formats, local geometry, normalization epsilon,
gate activation and intermediate-rounding policy. Qwen3-Next/Qwen3.5 construction
translates its configuration and checkpoint naming once into these specs, including
rank-local and prediction
construction. The reusable equations do not inspect family configuration.

`GatedDeltaStateGeometry` owns the exact convolution-history and FP32 recurrent
matrix policies. Existing Qwen state construction and Flash-Next recurrent specs
consume it, including rank-local heads, dilation and history-free one-tap kernels.
Flash-Next's recurrent and feed-forward sublayers compose those shared mechanisms
with learned gated residual ingress/injection. Parameter identities and retained
formats stay in the family; complete residual streams survive the sublayer boundary.
Padding masks recurrent inputs while retaining causal time and original residuals.
The feed-forward spec preserves FP32 scores, input-dtype routing coefficients,
ascending expert-ID accumulation, and the sigmoid-gated shared branch. A global
router may dispatch to a smaller rank-local expert bank through the existing provider.

`NormalizationScale::LearnedOffset` uses float32 normalization and evaluates the
learned scale plus offset in float32, then casts the scaled result to the input
dtype. MLX applies this contract to grouped and ungrouped normalization alike,
including low-precision QSA summaries retained by declared append streams.

`eredu-nn::NeuralBackend::output_gated_group_rms_norm` explicitly selects SiLU or
sigmoid gating after float32 normalization. `OutputGatedNormArithmetic` selects either FP32 intermediates
or a cast of normalized values to the input dtype before learned scaling. In the
latter case scaling uses ordinary input/weight promotion; gating uses FP32 and the
final result returns to the input dtype. Qwen's recurrent families select the
released rounded-normalization boundary. MLX implements both contracts without
model-family branches. The shared mixer also normalizes Q/K and forms decay in
FP32, then casts scan output back to the convolution activation dtype. Its shared
operator requirement set includes floating casts and feeds cold Qwen admission.
`CausalDepthwiseConvolutionSpec` carries dilation; retained history is exactly
`(kernel_size - 1) * dilation`, checked before allocation. Logical state and MLX
scratch descriptions use this receptive field. Portable convolution topology
retains dilation so cold workspace estimates also charge its padded history.
Other recurrent families select dilation one.

The MLX selective state-space scan completes its recurrent carry and newly emitted
outputs at each declared chunk boundary when a sequence exceeds `chunk_size`.
Completed output tensors remain available for concatenation, while their recurrence
graphs release chunk-local intermediates. Input/output storage and allocator caches
remain separate from that graph bound. Short scans retain lazy execution. This is
a general neural mechanism inside the existing submission scope; it neither changes
the scan equations nor publishes or commits a model transaction.
F32 single-step scans on Metal fuse stable timestep softplus/clamping,
transition discretization, recurrent update, output-vector products and direct
skip products into one native launch. Each threadgroup shares its timestep and
decay; output projection retains MLX's ordinary reduction order and final add.
The kernel writes independent F32 carry, preserves intermediate rounding without
multiply/add contraction, and accepts strided inputs through ordinary native
contiguous preparation. Complete input geometry is checked before indexing.
CPU, multi-token prefill and other precisions use the ordinary scan path.

`SamplingBackend::token_ids` groups a fixed number of scalar decision readbacks
without transferring logits or choosing sampling policy. Its portable default
reads IDs in order. MLX packs the integer scalars, evaluates their dependency
graphs together and transfers one contiguous ID vector. VoiceChat uses this
mechanism for its agent/function pair at selection and publication validation;
domain checks, owner/follower agreement, sampler/RNG adoption and completion
remain in the shared portable driver. Stochastic special-token preflight may
still require an earlier greedy-ID read before penalties and filtering.


`eredu-nn::residual_streams` owns checked stream expansion and flattening plus a
portable gated residual module assembled from ordinary normalization, projection,
activation and reduction primitives. DeepSeek expansion consumes the same stream
geometry. Gated low-rank mixing and Sinkhorn mixing remain separate algorithms;
sharing their value layout does not change the latter's equations. Architecture
code continues to own parameter names, learned-offset normalization choices,
module invocation order, discovery, and pipeline contracts. Family registration
and prepared construction remain architecture-owned.

Attention arithmetic is a neutral request policy. An architecture can require
rounding QK products, scaled/masked scores and normalized probabilities to the
query dtype. Native operators implement those boundaries for contiguous,
sliding and paged storage. The paged implementation retains only global row
maxima/denominators between two bounded scans and accumulates rounded-probability
value products in FP32 before the output cast. Cache telemetry counts both
scans and their transfers; page size does not introduce extra output rounding.
Resident attention tiles query rows while complete key rows fit its bounded
score workspace, preserving the ordinary softmax and value-product reductions.
Larger key rows retain the two-scan bounded mechanism. Explicit rotary input
products retain inverse frequencies directly so positions multiply the same
rounded coefficients instead of dividing by reconstructed denominators.
The CPU realization uses complete FP32 matrix accumulations, cascade row sums
and a fixed softmax reduction with a pinned exponential polynomial. These are
generic native numerical operations, independent of checkpoint and family policy.
Small BF16 batched products use sequential FP32 accumulation when
`rows * width * columns < 400`, excluding batch and head axes. The selection
uses the complete invocation geometry before query tiling. This preserves the
pinned reference's rounding at halfway BF16 boundaries. Larger score products
use an explicit 32-lane FP32 row reduction; value products retain their ordinary
GEMM reduction above the cutoff. The CPU path borrows bounded BF16 layouts and
converts products in registers, sharing physical KV heads. Its memory report
includes conditional layout copies and BF16 host results, with unresolved
allocator capacity, donation and contiguous sharing stated explicitly.
Metal realizes complete BF16 row and batched products, cascade normalization
and FP32 softmax/sigmoid with the same accumulation and rounding boundaries.
The Metal FP32 last-axis softmax uses one cooperative SIMD group per row.
Maximum selection is parallel; four lanes retain the ordered exponential sum
chains and their fixed final merge. A device-memory barrier separates writing
exponentials from parallel normalization in the output buffer. Row widths/counts
are runtime tensor inputs, so growing histories reuse the same shader instead of
creating a shape specialization. Input layout preparation and native graph
retention keep their ordinary ownership and completion contracts.
The shared FP32 sigmoid/SiLU kernel reshapes scalar inputs to a one-element
array for Metal argument binding, while retaining the original output shape.
Scalar, singleton and vector calls use the same exponential rounding policy.
Single-row F32 activations with materialized row-major F16/BF16 dense weights
use a mixed-dtype Metal GEMV for linear and tied-embedding output projections.
It converts weights in registers and preserves the pinned MLX F32 GEMV's lane,
coefficient and reduction order; it neither rounds activations down nor creates
a full F32 weight copy. A safe nonblocking availability query prevents using
provisional strides from lazy parameter graphs. Unsettled or column-major
weights, unsupported geometry or native arithmetic, CPU and CUDA use fallback
paths. Eligible multi-row prefill and speculative verification use mixed-storage
FP32 GEMM through the same selector. Conversion retention applies to fallback
invocations; forecasts remove promotion allowances only for proven coverage.
These operators select by device, dtype and geometry. Router cutoff ties use
value-only CPU partitioning of the small score/index tensors; expert parameters
remain on their selected device. Independent PyTorch fixtures verify the native
operations without importing architecture equations.
The native build locates an executable Metal compiler rather than accepting
Xcode's placeholder path. An explicit `TOOLCHAINS` selection takes precedence
over the generic Metal alias and participates in build invalidation.
Grouped linear and gated-product specifications also declare their reduction
policy. Sequential group-order reduction sorts each token's selected global
group IDs and rounds every weighted addition to the output dtype. Compact-bank
and executable-format transformations retain this equation policy. CPU grouped
projection selects an expert with a unit slice, retaining its matrix strides and
avoiding an advanced-indexing copy that changes reduced-precision accumulation.
Native block-FP8 linear and grouped projections quantize activation blocks on
both CPU and GPU. The CPU realization independently expands the same E4M3 values
and scales before multiplication; checkpoint parsing remains portable.

Chat templates use Transformers-compatible Jinja whitespace and dictionary
construction. Rendered bytes, including leading newlines, remain artifact
policy. IFM reasoning and JSON/XML/typed-XML tools are facade declarative
protocols recognized from rendered probes and tokenizer properties. Separate
argument value tags, compact type annotations and literal value framing are
shared parser/constraint features. Partial annotation state is included in
snapshot storage estimates.

Response framing and native-tool capabilities are admitted independently. The
facade recognizes delimited `<think>`/`<thinking>` reasoning, IFM effort wrappers,
Harmony analysis/final channels, Gemma channels, Inkling message frames and ATEM
recipients from tool-free rendered probes and the required response tokens.
Tool rendering and tool-only tokens are optional evidence for a separate native
capability. Their absence or failure cannot disable recognized reasoning.
Boolean and named-effort controls also use tool-free rendering evidence.

Semantic-only plans have no grammar, regardless of their format. They retain the
ordinary tokenizer validity controller and work when constraint compilation is
unavailable. The response plan attaches a grammar only for a requested native-tool
surface. Declarative response parsing activates tool envelopes only on that
surface; application tool emulation can consume visible text independently.
Unsupported native-tool requests cannot bypass admission through a response-only
plan. The same semantic plan supplies ordinary, controlled, speculative and
snapshot execution, including each profile's framing and stop policy.
Delimited parsers bind initial state to the actual generation suffix rather than
older messages. Atomic added delimiters survive decoding regardless of their
special-token flag; ordinary delimiters use bounded incremental lookbehind.
Chat preparation validates application tool declarations and request consistency
before selecting native capabilities. `TextModelError::InvalidRequest` identifies
invalid schemas, history, names and options; `UnsupportedCapability` identifies
valid requests the selected protocol cannot honor. `ConstraintPreparation`
identifies protocol/tokenizer inconsistencies and constraint compiler failures.
Template and tokenizer errors retain their separate error variants and sources.
Unavailable reasoning parsing and controls use distinct `ChatCapabilityRejection`
variants. `NativeToolSupport::Unsupported` carries the same `NativeToolRejection`
type used by native-tool capability errors. The facade reports these facts;
applications own emulation and retry policy. All execution modes consume this
shared preparation result.

Tool-schema admission and completed-argument validation belong to the facade.
Application JSON Schemas are retained unchanged for rendering and checked with
the portable `jsonschema` validator without remote or filesystem retrieval.
Dialect grammars enforce the constraints they can represent and retain protocol
syntax, function names and call limits for other schemas. The shared semantic
event sink parses the complete argument object once and validates it before
emitting `ToolCallCompleted` with the call index, ID, original application name
and parsed arguments. Incremental start and argument events remain provisional.
Snapshot sizing includes the parsed object; tool-enabled continuation and history
bounds include JSON container expansion. Immutable validators are shared across
forks while partial arguments remain
private to each parser. Backends continue to consume neutral token filters and
do not interpret schemas or decide which tool calls applications execute.

`eredu-text::tiktoken` owns rank-file reconstruction. Muse Spark's official
`l4_200k_base` uses the published Llama 4 split expression and complete BPE split
pairs, with direct vocabulary-piece matches before merging. Its declared added
IDs are complete and contiguous; missing, duplicate or incompatible declarations
fail before returning a tokenizer. The facade selects this importer only when
its selected Spark artifact has no converted tokenizer. Embedded GPT-4o GGUF
metadata selects the same split rule and direct-piece behavior. `UNUSED` rows
remain absent from the tokenizer vocabulary while model output geometry retains
the complete declared width. No backend owns these text-side policies.

Tokenizer ID membership and model output width are separate domains. The facade
builds a closed validity mask from actual canonical, round-tripping tokenizer IDs,
including added and special tokens; holes and inconsistent mappings are excluded.
Raw token generation passes it to core's ordinary text machine via
`TextGeneration::with_token_filter`. Prepared text, semantic, observed, controlled and speculative
requests intersect the same immutable mask with their grammar controller. Forced
choices further restrict that intersection. The grammar and output parsers retain
their existing activation, EOS and termination behavior.

The facade's `PlannedModel` owns the target, source and retained drafting policy.
One `GenerationRequest` selects `OutputMode::Semantic` or `OutputMode::Text`;
`generate`, `forecast_generation` and `with_controlled` share its input, sampling,
stops and cancellation. The owner dispatches disabled, embedded and external
sources through the existing ordinary driver or neutral speculative scheduler.
`NoDrafting` sessions use only ordinary backend contracts. Explicit borrowed
sources and factory-owned sources use the same session implementation.

Request defaults inherit the selected plan. Neutral `SpeculationLimits` constrain
proposal capacity, lookahead, transaction storage and completion deadlines.
The scheduler narrows its decoding-admission facts with those limits before
binding command handles. Snapshots and forks retain this authority. Backends
continue to supply native resource facts and typed executable visitors; they do
not resolve request policy or select semantic output interpretation.

`OutputMode::Text` retains tokenizer validity, incremental decoding, EOS and
caller stops without requiring a protocol parser. It rejects native tools and
required calls; explicit thinking requires `allow_unparsed_reasoning`.
`OutputMode::Semantic` requires executable semantic support. Text/semantic mode
participates in snapshot compatibility, including decoder and stop state under
existing copy and storage budgets. Applications own emulated tools and the
choice of output mode. The low-level admitted `start_controlled` and
`generate_observed` operations accept the same explicit output policy.

`eredu-core::TokenFilter` owns closed-set intersection and projection to an actual
output width: missing mask entries are forbidden, a shorter output uses the mask's
prefix, and an empty executable intersection is an error. Native samplers realize
these restrictions before their existing sampling policy; they neither reconstruct
tokenizers nor infer missing tokens. MLX uses negative infinity for rejected logits.
Low-level core callers that own no tokenizer may still explicitly use `All`.
Validity is shared immutably by controller snapshots and speculative forks; sparse
token IDs participate in the versioned vocabulary fingerprint used for compatibility.
A forced mapped ID outside a shorter model output fails when that output's width
is available, before sampling. No model-specific padding convention is assumed.

Physical artifact declarations separate one retained physical store from its
logical consumers. SafeTensors declarations contain neutral tensor metadata,
admitted shard identities and resolved schemas. Architecture-owned component
roles use restricted source views and may combine disjoint inputs from several
containers. Opening a role does not rediscover a directory or duplicate a store
shared with another role.

VoiceChat's portable admission module uses these declarations for perception,
backbone, function head, TTS, codec and RNNT parameters. It parses nested NeMo and
component-based inference configurations into one released equation contract and
normalizes the pinned upstream defaults; Python
`_target_` strings are inert. Exact auxiliary assets, canonical source recipes,
integer-buffer values and embedded GGUF tokenizer declarations are validated
before retention succeeds. The checkpoint crate derives ordinary affine matrix
formats from logical geometry, U32 packing and group companion shapes. VoiceChat
uses the same strict matrix schema for dense and packed SafeTensors input,
including atomic scales/biases admission and agreement with explicit configuration
declarations. Architecture preparation retains these formats for existing
Nemotron, function-head and speech construction; native bindings do not select a
checkpoint family or reconstruct quantization policy. Sensitive backbone vectors
use explicit F32 recipes, while packed projection storage remains encoded. All
materializations consume the ordinary prepared-checkpoint cache and residency
contracts. GGUF storage schemas distinguish exact integer types
from explicit allowed floating/block encodings. GGUF header admission accepts an explicit
quantized representation before deriving logical names, so quantized buffers and
rank-three convolutions do not require linear-weight naming. The generic GGUF
catalog keeps affine/MXFP4 `.weight` companion layouts and admits quantized buffers
outside that naming convention as single F32 outputs with bounded lazy decoding.
Architecture preparation can select an explicit representation for any quantized
parameter before construction. Canonical quantized source views defer bounded F32 conversion to leases and report encoded input,
converted output and conversion/recipe workspace separately. Storage admission
remains backend-neutral; shared realtime cold selection chooses the execution
mechanisms separately. See [VoiceChat validation](voicechat-validation.md) for the
measured scope.

The `eredu realtime` command calls `api::realtime::prepare_realtime_model`,
`prepare_mlx_realtime` and the consuming `PreparedMlxRealtime::load`, then drives
the returned `BoundVoiceChatSession`. Checkpoint naming, admission, neural equations
and conversation policy remain in their existing owners. The CLI owns configuration, recorded tool scripts, wall-clock pacing and
file/device delivery.
`eredu-media::audio::wav` supplies bounded, standard-library-only mono WAV processing
without optional host-audio dependencies. The facade selects cache realization
through MLX's cold `native::state_capabilities`; the CLI supplies policy and
budgets. Fallible `ExecutionContext::cpu` and `accelerator` constructors keep native
device creation in the backend.

`eredu-media::audio::system` gates CPAL capture/playback behind `system-audio`.
Bounded callback queues, scalar format conversion and host resampling depend on
no native inference runtime. The CLI drives file and system PCM through the same
controlled-session adapter; native completion, rollback and neural state remain
in their existing owners.

`eredu-media::audio::processing` owns optional host WebRTC echo cancellation and
noise suppression, independent of model families and accelerator runtimes. One
capture processor handles both filters, with or without a playback reference.
System-audio duplex construction
connects capture and actual native playback through bounded timestamped reference
queues; callbacks only convert/copy PCM and update counters. Resampling and echo
processing run on the capture reader. Capture permission follows source samples
through resampling and processing; filtering continues while delivery is muted.
`audio::admission` owns bounded level admission with pre-roll and hangover. It
returns rejected windows as silence and preserves source duration at EOF. The
CLI owns terminal PTT controls, selects filter/admission configuration and applies
the same admission mechanism to microphone or WAV input. It does not own filter
equations or playback-reference synchronization. The controlled speech-session
driver and model state advance normally during muted input.
The safe external host-library wrapper retains its own FFI boundary; repository
crates retain `unsafe_code = "forbid"`.

`eredu-checkpoint::quantization` owns bounded transformation targets, cold output
metadata and pending prepared-source publication. Architecture preparation selects
eligible matrices and execution precision before residency admission. The MLX
backend realizes the retained plan through its generic bounded quantizer and
prepared-checkpoint cache, then publishes exact encoded storage after completion.
Restricted views preserve that preparation authority and deny unrelated reads;
transformed reads fail until publication. No family branch selects quantization
inside the backend.

VoiceChat retains selected host integer payloads during their bounded hash-validation
pass and reports their canonical host storage separately from learned weights.
`PreparedHostConditioning` reconstructs the tokenizer through backend-neutral
`eredu-text`, resolves live control markers, and builds character mapping and
speech feedback from those retained values. Construction does not reread the
integer sources or reopen tokenizer assets. Saved character EOS flags remain
independent of live text-channel EOS policy.
`PreparedSpeechParameterUnit` projects one module's ordinary parameter traversal
onto canonical admitted recipes and further restricts its source view to that
unit's physical inputs. Shared runtime selection, completion-safe materialization
and atomic binding consume this projection. Individual transformer, perception,
MoG and codec blocks can be selected independently; host preprocessing and
integer buffers are excluded. This unit mechanism does not declare a full speech
graph residency schedule or distributed executable.

`SpeechConstructionPlan` validates normalized speech geometry without native
resources and projects each unit's canonical parameter catalog from the existing
family declarations by execution-group identity and global unit index. It constructs
unloaded perception, RNNT, function-head, character/TTS, MoG and codec units through
ordinary neural mechanisms. Transformer and convolution stacks expose individual
blocks; codec codebook depths are separate units.
`PreparedSpeechConstruction` restricts that catalog to retained component roles.
Its cold `prepare_unit` resolves recipes and validates declared geometry through
`PreparedComponent` before native construction. Module construction and consuming
assembly use those same exact bindings; parameter discovery needs no unloaded
native module. The low-level module projection remains available for caller-supplied
modules and validates their explicit aliases. Missing roles, invalid indices and
recipe geometry errors fail before module construction. Default perception, TTS
and codec preparation selects F32 execution, with input/output conversion overlap included
in materialization bounds. The portable speech precision policy selects eligible
narrow perception/TTS weights and arithmetic; RNNT and function-head precision
are governed by their component equations.

`SpeechConstructionPlan::assemble_with` consumes those units into
`LearnedSpeechComponents`, preserving the bound parameter objects. The prepared
`materialize` route checks every required speech source role and prepares every
unit's recipes before construction or payload reads, then selects and binds each unit through shared runtime materialization and exact
completion handling. Binding failure stops assembly and preserves the original
error through `BackendFailure`; no partial aggregate is returned. Materialization
is unitwise, but the returned aggregate retains all its bound speech weights.
It does not establish a bounded full-graph residency policy.

The aggregate borrows the existing coupled executor around a caller-owned Nemotron
runtime and prepared host conditioning. Fixed-Aria prompt encoding borrows the
decoder's quantizer, so the encoder owns no duplicate codebooks. Codec consuming
constructors validate unit geometry, order and codebook identities without
declaring replacement learned parameters; spectral constants remain codec-owned.
The shared realtime loader initializes the joint native Nemotron/TTS cache
transaction and binds the assembled components to the public session.

Selected perception sources also retain the saved analysis window and mel filters
as finite F32 host coefficients through bounded reads. `PreparedPreprocessing`
owns 133,184 payload bytes. `VoiceChatFrontend::from_retained` and
`InputState::from_retained` construct host processing without reopening artifacts
or regenerating coefficients. The shared media processor accepts explicit
coefficients and owns a padded window/filter table of 133,632 bytes, separate from
the retained coefficients, FFT workspace and streaming caches. These host tables
are excluded from learned-unit parameter binding.

`PreparedSilenceConditioning` retains the exact coefficients and one-second
16 kHz carrier policy without allocating an FFT or backend resource. Resident
construction retains this cold preparation alongside its host assets. Execution
uses the shared media frontend to extract all 101 centered mel rows, then runs
fresh causal perception and compacts only the first projected frame. Perception
uses the selected speech precision with F32 boundary features; the result is cast
to the selected backbone precision. RNNT and live session state are not supplied
to this preparation. The retained coefficient
copy contains 133,184 payload bytes; temporary PCM and mel payloads occupy 64,000
and 51,712 bytes, respectively, in addition to media tables and neural workspace.
MLX's `MlxVoiceChatExecution::prepare_silence` invokes this architecture method
under the existing submission/completion owner and waits before exposing the
tensor. Failed or unwinding preparation retains unresolved work and poisons its
execution owner. This is startup preparation, not another session engine.

`eredu-nn::recurrent::Lstm` composes ordinary parameterized affine projections,
sigmoid, tanh and tensor operations. It preserves independent input/recurrent
biases, explicit gate order and per-layer hidden/cell state. The caller supplies
checkpoint identities and storage formats. A step returns new compact state;
completion and publication remain with the execution owner. No backend owns
RNNT equations or recurrent checkpoint naming.

VoiceChat's RNNT component owns embedding, two-zero streaming initialization,
additive ReLU joint equations and the bounded label loop. It consumes the
unprojected perception frame, projects it once, and returns candidate state plus
first-decision blank evidence. Integer greedy decisions use existing tensor
operations. Hypothesis capacity is explicit; capacity failure preserves the
parent state. The architecture casts encoder input to the joint parameter's
selected floating precision before projection. Text rendering, punctuation bias,
turn counters and coupled audio scheduling remain outside this neural component.

`RnntExecution` shares that decoding policy across ordinary and provider-backed
execution. `RnntKernel` leases the embedding, each LSTM layer, and the three
joint projections independently. Phase-specific traversal acquires only the units
needed for prediction or joint evaluation and retains candidate hidden/cell values
through their unit completion. Exact prepared recipes normalize bounded execution
to F32 and charge conversion storage. The MLX adapter binds these architecture
units through the existing resident, host-layerwise and disk-streamed mechanisms.
Perception/RNNT composition accepts either provider without invoking perception
again or introducing another scheduler.

`RnntPredictionState` separates globally indexed local LSTM carry from the final
prediction readout. Final-layer owners reuse their last hidden tensor; other
stages retain the published row without retaining remote recurrent cells.
Ordinary execution owns the full range. Selected ownership is validated before
acquisition, and snapshot copying accounts for every retained carry/readout tensor.
The predictor phase's state layout describes local hidden/cell pairs only; its
completed-state byte report also charges the nonfinal stage's published row.

VoiceChat's causal subsampling and FastConformer layers use ordinary tensor,
convolution, affine and normalization mechanisms. Architecture equations own
asymmetric two-dimensional padding, relative-position shift and masking, GLU
convolution and half-weight feed-forward residuals. Canonical OIHW convolution
parameters become OHWI views for the neutral convolution contract. Attention
retains normalized inputs and convolution retains post-GLU values in compact,
bounded caches. `PerceptionKernel` declares ordered subsampling, Conformer
and projection groups. Subsampling, each Conformer block and the final projection
are independently materialized execution units; the graph has no learned static
parameters. Resident `Perception` and provider-backed `LayerwisePerception` use
these same equations through the ordinary runtime traversal. The pinned streaming
wrapper's F32 policy is the default. The portable speech precision policy also
supports eligible F16/BF16 projections and convolutions while retaining F32
normalization, residuals and streaming histories; component execution rejects
parameters or boundary activations that disagree with its selected policy.

`ResidentRuntime::from_units` adopts already-bound modules after checking graph
group and unit counts, preserving their parameters without constructing replacement
modules. `LayeredInvocation::invoke_with_context` returns the architecture's
request-local captures alongside the ordinary output. Perception captures the
unprojected encoder result at projection ingress, so RNNT and Nemotron receive
their respective feature branches from one encoder invocation. The capture and
new attention/convolution caches participate in the shared unit-completion roots.
Ordinary `PerceptionRnnt` accepts the `PerceptionExecution` contract; startup
silence also uses that path with fresh candidate caches. The coupled executor
consumes `InputExecution`, whose associated perception cache may be ordinary or
stage-local. PCM carry, paired input state, execution output and the enclosing
transaction retain that exact cache type through completion and publication.

Shared decoder forward context constructs block attention inputs for complete and
selected pipeline models. Ordinary, observed, parallel and routed equations borrow
the same mask, cache, sliding-prefill policy and explicit rotary embeddings;
selected models retain their local cache ordinal mapping. Hidden-state ingress
validates the selected state layout and derives causal visibility from the selected
cache offset through one shared function, including supplied masks and explicit
rotary embeddings.

Shared decoder block factories declare complete canonical parameter topology from
normalized configuration without a native context. Dense, routed Qwen, GPT-OSS and
K2 use the same neutral decoder projection for graph, state-independent parameter
ownership and ordinary attention geometry. Learned attention sinks retain their
per-head placement. Families supply router and packed-expert construction specs;
`eredu-runtime` derives their logical matrix/bias geometry, while family format
recipes expand exact encoded companions. Routed declarations omit unused dense
feed-forward geometry. Cold routed preparation consumes the factory declaration
directly, without constructing a model for parameter discovery. Native module
visitation remains available for low-level resource operations.

Complete and selected decoder construction retain a validated `ComponentDefinition`
before pinned native modules are materialized. Block factories declare canonical
parameter topology without constructing native blocks. Graph, unit paths, state
partition rules and parameter discovery project that retained definition.

`LayeredModel` and `PartitionedLayeredModel` are `ComponentAdapter` specializations
over the same `DecoderKernel`. `CompleteDecoderKernel::new` and `new_parallel`
construct complete adapters; `DecoderKernel::from_partition` constructs exact
selected ownership. `CompleteDecoderGeometry` and `PartitionLocalGeometry` supply
the geometry used by the shared equations. Complete geometry preserves replicated
state admission, explicit TP admission, chunked text prefill and final-position
text projection. Selected geometry maps global invocations into sliced local
caches. Routed banks retain global invocation indices independently of cache slots.
The adapter supplies ordinary, observed, parallel, provider-aware and pipeline
driver contracts, including exact local state and pinned-role selection.

Low-level decoder operations borrow `kernel()` or `kernel_mut()`. The generic kernel
exposes `prepare_partition` and `prepare_partition_parallel` with explicit state
geometry and caller-selected cache origin; the typed partition kernel capability
also supplies observed ingress and boundary/final egress. Ordinary driver calls
use the same embedded-input and block methods as direct low-level execution.
Block methods map global invocations through the selected local cache geometry,
including composition-supplied feed-forward equations. Driver-facing partition
calls require retained ownership selection. Cloning an adapter clones its typed
resources and geometry and retains the validated definition and ownership.
`DecoderStaticResources` borrows canonical embedding, normalization and output
roles from either complete `StaticModules` or selected `PartitionStaticModules`.
The kernel is generic over that physical aggregate. Parameter visitation, editing,
input lookup and ordinary normalized readout use those same borrowed roles;
resource views allocate no parameters and retain no second owner. Specialized
residual boundaries retain their own expansion and collapse equations.

Complete and pipeline-local decoder readout borrow their actual tied embedding
or untied head through the same vocabulary projection and component instrumentation.
The typed static modules retain final normalization and the decoder applies its
configured logit softcap after projection. Serial equations and resource traversal
require only `NeuralBackend` and ordinary configuration; vocabulary collectives
and partition construction require their distributed and selected-geometry
capabilities separately. Observation timing and vocabulary subgroups follow the
selected projection.

`eredu-runtime::ComponentDefinition` retains canonical parameter ownership and its
execution graph with logical unit paths, state references, group transport and
primary/prediction identities. Construction validates addresses, paths, state
ranges, partition rules and optional group semantics without native allocation.
`ComponentDefinition::select_partition` validates canonical bindings, selected unit
state references and exact state ownership before materialization. Its retained
`ComponentSelection` resolves only owned global invocation addresses and preserves
state gaps. The shared adapter consumes that same selection and additionally checks
actual pinned roles and compatible retained ownership.
`ParameterStorageId` and `StateSlotId` distinguish physical parameter ownership
from invocation addresses and mutable state. `ComponentAdapter` implements the
ordinary architecture and layered contracts for independently leased components;
its typed `ComponentKernel` supplies resource construction, operation activity,
request roots and family cache identity. `ComponentEquations` supplies equations
for each concrete state profile, including borrowed cache slices and owned stage
caches, while retaining the same resources and request context. Kernel construction
does not require a canonical mutable-state type. Parallel and partition capabilities
name their actual state profile explicitly; selected runtime construction retains
that same type. Ordinary, parallel and partition equations expose typed observation hooks through
the same adapter. Unit hooks receive validated invocation addresses; partition
hooks run after the same state and ownership checks as ordinary ingress and egress.
Internal hook coverage and ownership of unit boundary interventions are explicit
kernel reports, independent of the presence of default hook methods.
Provider-aware execution requires `RoutedComponentKernel`; tensor-parallel provider
execution additionally requires `ParallelRoutedComponentKernel`. The shared adapter
validates invocation ownership before passing a provider to either equation. Kernels
report component and sparse expert observation coverage explicitly. Pipeline bank
order and collective requirements are typed kernel hooks over validated invocations.
Text request construction uses the independent `TextComponentKernel` capability.
`ComponentEquations` exposes final-position text readout explicitly; its default
uses ordinary full-output readout and advertises no narrowing. Ordinary and observed
readout keep their full result geometry.
Parallel adaptation requires the separate
`ParallelComponentKernel` bound. Typed partition and pipeline adaptation require
`PartitionComponentKernel` and `PipelineComponentKernel`. Consuming partition
selection validates canonical parameter ownership and the declared state projection,
retains global invocation addresses, and maps declared state slots to local ordinals.
Ingress checks both the complete local state and its entry-group projection;
egress checks output ownership. Declared state consumers also select the ordinary
local-stage traversal without a family-authored state-group predicate. The
adapter uses `EmptyParameterModule` for wholly leased components. Kernels with
pinned storage expose `ComponentPinnedResources`; `KernelParameterModules` borrows
those same modules for binding, editing and completion roots without duplicating
ownership. Complete construction checks exact canonical roles. Selected construction
checks physically retained roles against the partition, including shared consumers
and replicas. Selection rejects a retained pinned module outside that ownership.
The adapter does not infer missing capabilities from equation defaults.

VoiceChat RVQ, MoG, character conditioning, the function head, RNNT, perception,
the TTS transformer, codec PRVQ and the EAR encoder/decoder
use this adapter with their typed kernels and runtime-owned empty pinned modules.
Components without mutable state use shared stateless traversal metadata. RVQ retains bank-only sum and refinement operations and the
bank-to-projection embedding operation. PRVQ retains selected depth ranges and
ordered residual subtraction. Their collective-bearing equations retain explicit
rank agreement and transient input/candidate roots through ordinary completion.
RNNT's complete and projection-only selections use one kernel and typed semantic
phases. Each selection derives its graph, paths, unit counts and canonical
parameter groups from those phases. Projection execution owns no recurrent
parameters. The resident RNNT consumes the same kernel as bounded execution;
caller-owned LSTM parent and candidate state remains separate from parameter
leases, including repeated cached predictor steps.
Perception declares subsampling, independently leased Conformer layers and projection
with cache references only on the Conformer invocations. Complete, tensor-parallel
and selected pipeline execution share `PerceptionKernel`. A parameter-only ingress
or projection stage owns no cache slots. Typed boundaries retain frame frontiers;
projection ingress captures the raw encoder result alongside its projected output.

`PreparedCheckpointSource::retain` restricts physical inputs and snapshots their
exact metadata and provenance before deferred payload access. Runtime's
`PreparedParameterRecipes` couples this authority with validated family-resolved
recipes for one ordinary, expert or pinned owner. Empty owners retain an empty
source. Consuming `map_recipes` applies family-authored transformations while
preserving canonical targets, validating outputs and narrowing authority to the
transformed recipes' physical inputs. It cannot introduce a neighboring owner's
sources. Consuming `project` selects existing canonical parameter identities,
keeps their recipes and exact source guards, and rejects unknown targets. Qwen4
target and prediction tensor partitions use the shared transformation traversal.
Runtime's `select_parameter_recipe` pushes declared physical selections through
canonical recipes and assembles ordered range segments on one axis, retaining
canonical parameter identities and typed recipe failures. It validates geometry
without acquiring payloads. Families supply packed coordinates and selected local
geometry; Qwen4 target and prediction ordinary and expert owners, including
GGUF low-level expert selection, consume this projection. Prediction vocabulary
derives from the retained target owner through identity projection. Runtime's
source-free `ParameterBankRecipes` validates canonical
bank recipes and their declared member counts, then projects every parameter to
bounded leading-axis member recipes with singleton axes. Its `map_recipes`
transforms complete banks before deriving members and retains the declared member
count. Qwen4 target, prediction and inspection consume this shared catalog.
GPT-OSS uses it to prepare one independently leased unit per global expert,
including packed weights, scales and biases. Muse uses the same member projection
while retaining its declared parameter order and format-dependent quantization
roles. Inkling projects routed and replicated shared-expert banks through the
same catalog while retaining distinct bank cache identities, distribution policies
and declared parameter order. Families retain bank namespaces, formats and expert
selection policy. The gated-product residency builder used by Gemma4 and diffusion
Gemma projects its selected weight, scale and bias outputs through the same bank
catalog while preserving per-expert leases. DeepSeek V3/R1 and V4 target and
prediction experts use this bank catalog for their selected weight, scale and
bias recipes. DeepSeek retains encoded contraction and row-block alignment,
packed column coordinates and separate gate/up intermediate ranges in its family
preparation. Both its low-level expert operation and encoded residency preparation
use `select_parameter_recipe` to project these ranges and assemble the gate/up
segments. Member projection retains bank-global IDs and allocates no parameter
payload or native resource.
`select_parameter_bank_recipes` validates nonempty, unique ordered global member
IDs before pushing a rank-local selection through canonical recipes. Families
supply the member axis and geometry; runtime retains selection order and typed
member or parameter errors. Bank selection uses the same typed recipe selector;
duplicate canonical destinations are rejected.
GPT-OSS rank-local expert preparation uses this shared projection. Shared resource
binding preparation uses the same checkpoint snapshot primitive while keeping
aliases local to each independently leased resource.
Runtime's borrowed `MaterializationTaskIndex` indexes primary tasks and exact
standalone companion tasks under their declared topology output identities.
Generated companions and aliases do not become independent acquisition units;
equal physical source inputs do not merge distinct outputs. Duplicate selected
outputs, including overlaps between primary and auxiliary tasks, return typed
errors naming both owners. Routed bank preparation and compact residency sizing
use this shared index. Families retain expert roles, route cardinality and
format policy; indexing acquires no payload or native resource.
Runtime's `resolve_parameter_member` resolves selected primary outputs, aliases
and companions to their canonical parameter groups and module slots, rejecting
ambiguous identities and invalid effective geometry. `ParameterMemberIndex`
retains immutable selected tasks and the validated parameter declaration for
repeated lookup; callers cannot supply replacement slices to that index.
Resolved records expose read-only accessors and are constructed only by runtime.
Partitioned discovery and assistant static admission consume this shared
resolution. Runtime's `derive_parameter_member_layout` projects an owned or
absent parameter through retained or ordinary placement and returns a
`ParameterPartitionLayout`. Its coordinate compiler converts complete physical
encoding blocks into effective scalar coordinates and prepays metadata budgets;
companions use their own scalar geometry. Architectures retain semantic placement
and pipeline ownership policy. Resolution inspects metadata and grants
no parameter value, editing, checkpoint or native resource authority.
Runtime's `PipelinePlacementPlan` validates complete canonical graph/layout
identities and declared transport coverage before projecting pipeline owners.
Sequential groups share one balanced invocation sequence; ordinary groups balance
independently and output-owner groups remain on the terminal stage. Route
projection uses this same owner table and the graph's terminal producers, including
cross-group dependencies and intra-group continuations. Merge destinations do not
relocate a producer. Ownership-only queries allocate no route metadata. Families
retain boundary schemas and collective semantics; native composition realizes the
declared endpoints. Invalid placement returns typed group and geometry errors.
Runtime's `PipelineRankProjection` applies declared first/last-owner static roles,
replicated roles, parameter consumers and state rules to one actual Cartesian
rank. It retains read-only `PipelineGroupOwnership` intervals, local state,
publication rank and optional logical parameter targets. Construction consumes
these shared parts; ownership-only inspection omits target names and state.
Families retain the declarations and selected mechanisms. State and ownership
errors preserve their neutral causes; active embedded prediction uses its
separate placement contract and is rejected explicitly by this base projection.
Qwen4 n-gram tables, integer hash controls and scalar scales use this same
retention primitive. The family checks admitted table headers and provenance
before bounded literal reads; controls and scales retain separate exact source
projections.

Runtime partitions destination bindings by retained
`ParameterAcquisitionUnit` output identities, preserving binding order within
independently leased resources. It rejects missing, repeated or foreign
canonical destinations and verifies each unit's non-alias payload against its
selected byte count before native materialization. Native module paths use the
binding's canonical logical target for this projection.

Runtime's `physical_binding_bytes` supplies exact executable payload accounting
for residency-controller and acquisition validation, and native resident,
bounded, assistant and loading reports. Logical aliases add no storage; independent
physical owners count separately even when they read the same checkpoint tensor. Checked overflow
identifies the canonical destination. Alias topology validation, source payload,
loading workspace and backend host-buffer capacity remain separate contracts.

MLX ordinary text, prediction and assistant construction apply runtime-selected
transform groups through one bounded materialization sequence. Each successful
group replaces the retained source with its completed overlay and contributes to
the combined report. A later failure retains earlier completed overlays; the
single-group mechanism owns native completion and failed-resource retirement.
Format grouping and task identities remain runtime contracts.

`PreparedComponent` validates family-resolved recipes against canonical global
shapes, groups bindings by parameter-storage owner, and restricts checkpoint
source authority to the recipes' physical inputs. Before recipe preparation,
`ArchitectureParameterDescription` checks ownership coverage and agreement with
the authoritative parameter declarations: global shapes, member sharding and
encoding metadata, logical group identity, role, partition domain and assignment.
Member ordering does not change this agreement. Conflicting declarations fail
with typed errors naming the parameter and group. Physical `ComponentRecipe`
declarations may include a `ComponentRecipeConversion`: the family selects the
required output dtype and exact source dtypes admitted for casting. Runtime keeps
recipes already producing that dtype unchanged, applies an explicitly allowed cast
once, and rejects other representations with a typed component/parameter error.
EAR encoder, decoder, PRVQ and RNNT declare their F32 execution requirement and
allow F16/BF16 conversion through this contract. Ordinary recipe inference supplies
the conversion workspace and binding-byte bounds. `parallel_layout` projects the
requested topology from the retained parameter declaration and checks any retained
selection. Speech and codec selectors use this projection rather than reconstruct
parameter catalogs from configuration. Families retain their topology limits and
additional local-geometry validation. Consuming placement selection retains the
exact local layout; identical reselection validates it without
transforming recipes again, and incompatible selection fails. Recipe policy
transforms preserve targets, shapes and source authority. Identity-based
projections retain transformed bindings and project local placement; sparse unit
selection preserves global invocation addresses while compacting local ordinals
and narrowing source access. Families own source admission, naming, aliases,
precision policy, special recipes and additional geometry constraints. VoiceChat's
shared admitted-member resolver verifies component membership, the declared global
shape and the complete inferred recipe output metadata before execution conversion.
Its ordinary module resolver uses the same component lookup while retaining host
exclusion and explicit alias checks. Component precision rules and original storage,
converter workspace and conversion-overlap accounting remain family-owned.

RVQ, PRVQ, RNNT, RNNT projection, perception, conditioning, MoG, the TTS transformer
and EAR encoder/decoder preparation use this shared container. Function-head
preparation retains its primary parameter accounting alongside the same container.
RNNT projection selects declared canonical targets and ownership. EAR, perception
and TTS transformer stages retain the selected `PreparedComponent` through the
native handoff. Inspection and consuming transfer use its shared addressed
binding views, restricted source and retained placement. Family stage structures
supply configuration, precision, namespaces, local state and boundary geometry.

`PreparedPerceptionBindings` validates symbolic parameter geometry against the
admitted recipes without reading payloads or allocating native resources. It
retains only the physical inputs for subsampling, Conformer blocks and projection,
excluding preprocessing and neighboring components. Execution casts and bounded
conversion costs use the shared speech recipe projection. MLX's
`materialize_voicechat_perception` consumes these exact bindings through the
existing host-layerwise or disk-streamed `MlxLayerwisePolicy`, or its
`MlxResidentPolicy` conversion for fully resident execution. The adapter supplies
streams and ordinary materialization mechanisms;
it performs no source discovery or family naming reconstruction. This component
residency path does not declare full speech-graph residency or distributed session
support.

Perception tensor placement uses the shared physical-layout planner. Conformer
attention groups partition complete heads across Q/K/V, relative-position
projection and both position biases; the output projection partitions its input
axis. Each feed-forward branch partitions its intermediate axis. Shared row
reductions combine these three contributions before their residual updates.
Ordinary and tensor-parallel execution share the causal masks, relative-position
shift and convolution equations. Subsampling, modality projection, normalization
and convolution parameters remain replicated and included in each rank's exact
binding bytes. Both streaming caches retain full hidden width because they store
normalized inputs and post-GLU convolution values rather than projected heads.
The architecture reports that complete cache geometry on every tensor rank.

`PreparedPerceptionBindings::select_parallel` applies the selected placements to
retained canonical recipes once, preserving source authority and bounded
conversion. `materialize_voicechat_perception_parallel` binds the communicator
to the same resident, host-layerwise or disk-streamed traversal. It returns both
encoder and projected features from one invocation; it does not allocate RNNT
state or select a coupled distributed VoiceChat session.

Perception stages also bind to the shared pipeline executor. The architecture
owns cuts after subsampling, within the Conformer stack and before modality
projection. Transport dimensions use the subsampled count after overlap removal;
raw mel-row counts remain part of ingress. Boundaries carry encoder-width F32
features and an exact I32-word representation of the start/end u64 frame frontier.
Only selected Conformer layers own caches. Subsampling and projection stages
retain their frame identity without allocating another stage's cache tensors.
Prepared stage sources expose only the canonical recipes used by those units.

The final perception stage publishes a packed encoder/projected tensor through
the existing shared output publisher. Architecture code unpacks the two feature
branches without repeating perception. `MlxComponentPipeline` supplies native
binding, transport, phase agreement, completion and commit for both perception
and TTS wrappers. Output preparation participates in all-rank failure agreement;
its exposed tensors and candidate caches remain retained through exact completion
before commitment. These component pipelines do not select the full coupled
VoiceChat graph or distribute its conversational policy.
Shared MLX policy preparation omits static residency units when their binding set
is empty. Resident conversion retains the actual static leases and reports their
count, including zero; it does not synthesize a pinned allocation for an empty role.
Disk-streamed traversal completes and releases each group's transfer window at
its last unit, after exact consumer completion. Successive groups therefore
share the declared maximum group-window budget without retaining obsolete window
protections; within-group lookahead keeps its ordinary lease lifetime.

`eredu-media::audio::streaming` owns bounded PCM framing and centered Slaney mel
windows, including symmetric Hann, pre-emphasis, finite-window reflection and
reusable interior FFT rows. FFT dependencies remain behind `audio`. Architecture
policy owns the VoiceChat 80 ms cadence, finite rolling-buffer length, initial
one-row mel phase and later 17-row windows with two subsampler outputs discarded.
PCM ingress has an explicit F32/F16/BF16 rounding policy, defaulting to the pinned
BF16 setting before float32 host extraction. The host and neural components return independent candidate states; they do not
select a backend, publish events or create another session scheduler.

Nemotron-H target input can carry already fused embeddings and select normalized
hidden output. Ordinary, bounded and partitioned execution use the same layer
traversal and cache ownership. The target pipeline boundary includes a validated
I32 readout control in its versioned schema; transport accounting includes that
control alongside token attribution and retained embeddings. Shared decoder static
modules project normalized values without repeating normalization or embedding
storage. Nemotron target RMS normalization uses one full-width float32 reduction
and float32 scale multiplication before returning to input precision.

VoiceChat channel fusion uses that input seam with a single shared token embedding
table. It casts perception features to the embedding precision before ordered
agent + audio + 2 × function addition. Prompt framing and repetition have an
explicit token bound. The function head has its own vocabulary parameter group
and can be materialized/projected separately from the existing agent head; both
consume the same normalized backbone result. Tensor-parallel lookup and vocabulary
projection use existing collectives. Greedy function decoding does not sample the
agent channel or advance conversational policy.

Guarded channel ingress agrees invocation kind, fusion weights, tensor geometry
and dtype, or exact prompt IDs and framing policy before vocabulary work. Prompt
tensor construction, each lookup result and final fusion have local-result
agreement boundaries. Ordinary and guarded paths share the same lookup and
cast/add equations. `BackboneInput::prepare_parallel_guarded` feeds this result
into the existing supplied-embedding path; `ParallelBackbone` accepts an optional
retained ingress callback. An ingress rejection precedes layered acquisition and
cache advancement. The callback does not guard operations inside subsequent
Nemotron blocks. Native low-level bindings can borrow
`MlxReplicatedCoordinator::parallel_group()` to retain the same bounded failure
and completion authority in vocabulary mechanisms and architecture agreements.

`FunctionHeadExecution` supplies the same readout to ordinary and controlled
coupled execution. Its shared dual-head method projects agent and function logits
from one retained normalized tensor, without a second Nemotron traversal.
`FunctionHeadKernel` declares one independently leased projection unit and
no static learned parameters. Shape, hidden width and floating dtype are checked
before provider acquisition; the shared traversal retains input and output through
exact unit completion. Resident, host-layerwise and disk-streamed function-head
providers execute the existing projection equation. Tensor-parallel providers use
the same traversal and vocabulary gather, including uneven row partitions. Cold
selection validates every physical format companion and retains the exact source
and row recipes. Invocation checks communicator rank and degree before acquisition;
the acquired unit must match the selected vocabulary range. Gathering completes
before releasing the head lease. The native adapter binds the selected communicator
and provider policy without owning sharding equations or conversational scheduling.

The native TP function-head constructor requires a retained
`MlxReplicatedCoordinator`. Architecture-owned cold validation checks bounded
failure agreement, F32 uneven-gather input/output capacity and vocabulary degree
before payload reads. Guarded projection agrees full hidden shape, dtype and
logical dimensions, then uses the existing layered traversal hook for parameter
acquisition and completion. Leased-unit identity is agreed before local projection;
the shared MLX vocabulary primitive agrees local multiplication success before
entering its gather. Resident and bounded providers use this same guarded path.
A local failure retains its original source, and acquired peer leases remain
owned by ordinary abort/completion handling. These checks protect the head's
collectives; they do not guard operations inside the Nemotron backbone.

Prepared function-head bindings restrict source authority to exact recipes and
retain the admitted dense or affine matrix format. Floating weights and affine
companions use F32, with bounded conversion storage included in admission.
Selected output storage uses local vocabulary rows; source conversion workspace
and encoded storage remain charged separately. A matching retained tensor
selection is reused without applying its row slice again; a different communicator
selection is rejected before payload reads. These component entry points do not
select a complete distributed VoiceChat session.
Resident speech construction and bounded providers consume the same retained
matrix format and companion bindings.
`PreparedSpeechProviders` retains exact plans for all eleven speech providers before
binding. Its architecture-owned visitor orders construction and consumes providers
through `SpeechConstructionPlan::assemble_providers`, also used by resident unit
assembly. Provider geometry and spectral mechanisms are validated; no second
execution engine or mutable conversation state belongs to construction. The MLX
visitor delegates to the existing resident, host-layerwise and disk-streamed
component loaders, including RNNT's independently leased units.

RNNT tensor-parallel preparation partitions embedding and output-vocabulary rows,
including output bias and the blank row, through shared vocabulary mechanisms.
Its LSTM and encoder/predictor joint affines remain explicit replicas. Exact
local layouts and communicator rank/degree are validated before acquisition;
source authority and conversion recipes remain retained. The architecture-owned
`rnnt_coordination` driver agrees explicit request, turn and invocation identities,
frame frontier, limits and complete hypothesis before neural work. It uses the
ordinary greedy loop, with one coordinator evaluating each symbol and exact I32
publication before feedback prediction. Recurrent candidates stay caller-owned.

`eredu-runtime::replicated_execution` composes existing partition failure
agreement, output publication and bounded completion over one retained authority.
It owns no family equations, random stream or session scheduler. Architecture
drivers provide phase descriptors and local success; MLX only realizes selected
groups and their integer-safe broadcasts. Pre/post neural-phase agreements fence
asymmetric failures, preventing another phase after a participant has failed.
Neural tensor collectives retain the coordinator's selected bounded group and
failure authority. Cold RNNT admission uses the shared requirement comparator to
check exact operations, dtypes, logical input/output extents and completion before
weight materialization. Broadcast I32 values retain integer storage, including
values outside the F32 exact range; native completion retains their dependencies.
The enclosing session retains the final state-commit and host-publication boundary.

The RNNT predictor declares only its recurrent layers through the shared component
adapter. Its `RnntPredictorKernel` supplies the same LSTM equations for complete
and selected pipeline execution, with independent local parent and candidate carry. The shared `Recurrent` group kind is mandatory even without
text or modality ingress. Exact prepared views restrict each stage to its owned
LSTM parameters; TP cohorts replicate these small recurrent layers explicitly.
`PreparedRnntPredictorBindings` projects canonical resource identities from the
retained RNNT `PreparedComponent` and selects stage addresses through the shared
sparse layout. Its compact local ordinals preserve global recurrent layer
indices. Projection excludes embedding, joint and neighboring-stage sources,
and retains any selected replica placement; repeated selection validates that
placement without applying recipes twice.
Typed boundaries carry the hidden row, zero-state prototype, global layer cut and
full-width explicit predictor step. Cold communication admission checks the
readout's complete F32 broadcast bounds as well as the boundary roles.
The native adapter supplies mechanisms through
`MlxComponentPipeline`, including transport, failure agreement, completion and
commit. Its caller supplies the invocation descriptor and commit epoch and adopts
the returned local candidate only after commitment. The adapter includes the full
predictor step in rank agreement before parameter reads. This predictor primitive does
not select a coupled speech session.

`RnntKernel::projections` selects embedding and the three joint affines in a compact
four-unit provider. It reuses ordinary RNNT equations and exact vocabulary
partitioning, while its seven-tensor source view excludes recurrent weights.
Recurrent pipeline stages retain only their selected LSTM layers; projection
parameters are independently resident, host-layerwise or disk-streamed, with
vocabulary shards inside each stage's tensor cohort and explicit replication
between stages.

`rnnt_pipeline::decode_frame` composes these providers through the existing
coordinated RNNT driver and bounded greedy loop. A caller-owned cursor reserves
the frame's worst-case predictor steps and commit epochs before learned work;
failure, replay and early blank decisions do not refund that reservation. Each
recurrent phase must return its exact committed epoch before its readout enters
the joint network. Its local carry, hypothesis and transcript remain unpublished
until the enclosing frame commits.

`perception_rnnt_pipeline::advance` composes the perception pipeline and this RNNT
driver into one unpublished input candidate. Ordinary and pipeline input paths
share geometry, capacity, frame-frontier checks and paired-state assembly. All
participants agree restored local cache ownership, precision and RNNT reservation
before the first perception lease. Exactly one perception traversal supplies the
raw RNNT features and projected Nemotron features. Its exact committed outcome is
required before recurrent execution; either component's failure leaves the paired
parent unchanged. Stage-owned perception caches implement the same retained-root
and bounded snapshot contracts as recurrent carry. The MLX adapter clones the
borrowed stage state and delegates execution to `MlxComponentPipeline`; it owns no
frame scheduling or RNNT decisions.

`PreparedPerceptionRnntPipelineBindings` consumes the three retained component
selections before native binding. It checks connected widths, world and rank,
publication participants, decision broadcast/agreement and the selected projection
cohort. Its exact per-role parameter and initialized cache census charges each
rank's replicas, while conversion scratch and pool allowances remain in component
residency reports. It preserves restricted source views and performs no payload
reads or source resolution.

`PipelineInput` implements the same PCM input contract using a borrowed paired
pipeline, decision coordinator and caller-owned RNNT cursor. It shares ordinary
frontend preparation, tensor materialization and candidate assembly. Ranks agree
readiness, frontend policy, sample attribution and a digest of the processed mel
values before allocating input tensors. Incomplete input returns without spending
the perception epoch or recurrent reservation; one ready-frame attempt consumes
the adapter's explicit epoch. Fresh adapters supply replay authority independently
of snapshotted PCM and neural state. Coupled live, perception-only and fresh-audio
tool invocations use this contract through their existing execution branches.

The MLX paired materializer consumes retained perception, predictor and projection
bindings and their declared coordinator owners. Coordinator selections share one
native group realization and retained communication authority. The resulting
component bundle exposes initial local state and independent residency reports;
it retains no sampling, PCM scheduling, epoch allocation or conversational policy.
Its three residency arguments describe separate parameter pools, while full-owner
budgeting remains the enclosing construction driver's responsibility.

Parallel layered traversal exposes local-result agreement hooks at initialization,
policy admission, group ingress, unit acquisition, execution, completion and
finish. Coordinated providers attach their retained cohort authority; acquired
leases remain protected during agreement and abort on peer failure. Vocabulary
mechanisms separately agree local setup before data collectives, using the same
bounded authority. The mechanisms own neither RNNT phase selection nor symbol
policy, and uncoordinated traversal retains its ordinary behavior.

Component pipeline traversal also fences local setup and parameter acquisition
inside the selected stage's tensor cohort before entering unit collectives. The
shared executor requires the neutral failure-agreement mechanism and uses the
retained communication authority and completion bounds. Cold admission validates
that cohort's exact agreement capability before native binding; inactive pipeline
stages do not join these local fences. A local read failure cannot leave another
tensor rank waiting in the unit's reduction, and the ordinary lease guard protects
acquired parameters during agreement and abort.
MLX also checks its existing independent-status transport fact before realizing
the selected group. Full-world and connected Ring cohorts can fence without
inactive stages; a disconnected logical Ring cohort lacks that independent route
and receives a native capability rejection before materialization.

`SpeechResidencyPolicy` requires an explicit selection for every provider and
finite aggregate host/device allowances. Independent pool capacities are summed,
including the selected RNNT policy; one total allowance is never reused for every pool.
Cold descriptors use the actual converted binding payloads, exclude aliases, and
reserve the largest unit's encoded input, recipe and converter overlap per
provider. Disk background slots add conservative concurrent loading reserves.
Admission precedes every binding callback. Native alignment and multi-unit window
checks remain in the shared materializer. These reports describe logical learned
parameter pools and loading reserves, excluding backbone weights, live state,
activation/spectral workspaces, source caches and allocator overhead.

The native coupled adapter consumes the selected speech-provider aggregate through
`VoiceChatComponents`. Its existing completion owner retains every provider and
the selected Nemotron layer runtime until native work is safe to retire. Submission,
prepublication, discard and conversational commit use the ordinary shared path.
`VoiceChatResidencyPolicy` admits resident, host-layerwise or disk-streamed
backbone blocks together with independent speech pools before any binding.
The backbone device allowance includes pinned static weights exactly once and
must fit the largest consecutive block window selected by the shared policy.
Host pools and materialization reservations are accounted for separately. The selected report remains with the components and is available after
adoption by `MlxVoiceChatExecution`. These logical parameter bounds exclude live
state, activation/spectral workspaces, source caches and allocator overhead.
Nemotron embedding, final normalization and agent-head matrices remain pinned.
The two official vocabulary matrices require 4,480 MiB of logical device storage.
Block offload does not remove that floor or establish native distributed parity.

`voicechat::backbone::PreparedBackboneParameters` retains the existing restricted
backbone source, projects names through Nemotron's common naming contract and
separates static and per-layer recipes. Canonical VoiceChat transformations are
preserved, including exactly one reconstruction of split-GGUF Mamba `A_log`.
No backend detects source layout or reopens a checkpoint to make this projection.
Its cold `prepare_bindings` projects the source-free Nemotron parameter declaration
into `PreparedComponent`, retaining canonical pinned storage and independently
leased blocks. Residency inspection consumes those same bindings. The shared
storage-preference pass checks invocation geometry and applies native immutable
layout choices without changing checkpoint recipes, source authority or charges.
Resident binding borrows preferences from its existing modules. Pinned and unit
source views expose only their actual physical inputs; selected unit views use
global addresses even when local storage ordinals are compact. The native adapter
supplies the shared `LayerwiseRuntime` and parameter policy for all three residency
selections. Parameter selection precedes materialization, and each binding receives only the physical
inputs named by its retained recipe. The shared realtime loader supplies the
complete speech-graph residency selection.
`voicechat::components::PreparedVoiceChatConstruction` admits the full retained
component set and host conditioning together. It constructs exact host input
carry before binding speech and the ordinary Nemotron runtime, then returns
those components with input state separate for joint transaction ownership.
The native `materialize_voicechat_components` adapter supplies MLX
parameter mechanisms and a submission recovery scope; it contains no checkpoint
dispatch or model equations. Its result borrows the existing coupled executor.
The constructor accepts independent backbone and speech-provider policies under
complete parameter allowances. Cache and distributed execution selection remain
separate from component construction.
The retained construction also owns `NeuralStateGeometry`. It derives
Nemotron state through the existing hybrid contract and TTS state through the
shared decoder contract, preserving every local/global attention window. Batch
geometry declares one conversation row and two classifier-free guidance rows.
Native cache initialization validates both exact selected layouts before
allocating either owner. The shared selector requires checkpoint and
rollback support. Initialization uses the ordinary
MLX selected-state realization driver for device and paged placements, then
adopts both caches into the joint transaction.

`voicechat::preparation::PreparedGeneration` validates the retained text controls,
explicit component capacities and mandatory Aria startup. It constructs the
existing shared generation state with PAD/PAD feedback, one agent sampler and
explicit retained random state. Its frame contract declares the sparse mixed
channels; prompt prefill remains an explicit transaction. Component preparation
derives this policy directly from retained configuration and host controls,
without artifact access or native allocation.

The portable facade's `VoiceChatTextPreparation` clones the exact retained host
tokenizer, disables padding and truncation, and admits bounded ASCII system and
tool-response text. Architecture prompt policy owns BOS/EOS framing and repetition.
The facade renders the pinned function-calling template with existing tool
definitions; `eredu-text` supplies a general Python-compatible JSON writer that
can stop at a bounded output writer. Template text and HTML escaping remain
facade policy. `VoiceChatToolChannel` collects committed function tokens and
normalizes their wire payload into the existing schema-validated semantic parser.
It owns request/turn tickets and commit-driven response injection, with no neural
execution, wall clock, scheduler or external tool execution. These helpers are
available without a native backend; the shared realtime loader composes them
with selected execution resources.

`VoiceChatToolWait` composes that committed channel with explicit recorded time
and bounded prepared speech phrases. It returns ordinary Reminder, Tool or
perception-only work for the enclosing shared driver. Reminder receipts retain
admission until commit or safe discard; separate playback receipts establish
presentation completion before gaps and timeout begin. Phrase rendering retains
generated feedback and waveform through PAD tails and EOS; architecture-owned
EOS input substitution remains independent. The facade owns the timeout response
and call policy. It owns no native completion objects, external executor or second
scheduler. Phrase selection is supplied explicitly so host randomness and clock
reads cannot change replay.

VoiceChat channel selection accepts a fixed override or a pure `DecisionResolver`.
The architecture supplies the current perception frame's first RNNT blank decision
and raw agent/function selections before adopting sampling RNG, effective history
or TTS input. Resolved tokens and Main/Tool roles are validated at that boundary;
native adapters forward the borrowed invocation without owning turn policy.
The facade's `VoiceChatTurnTaking` implements RNNT speech/silence evidence,
bounded BOS/EOS history and main-loop speech-tail gating. Resolution reads
canonical state without mutation. A captured admission retains invocation,
request/turn identity and function-channel activity; matching committed host
output adopts the deterministic successor and publishes turn events. Tool-result
arrival cannot alter the activity used by already admitted decisions.
The facade applies its sustained-speech interruption threshold during active
tool work, including fresh perception in silent waits. Architecture-owned
interruption feedback replaces the last agent/function tokens with EOS/PAD and
the sampler's last effective token with EOS, preserving cache, RNG and position
ownership. The publication transaction applies this host-only change to its
completed neural candidate before adopting the matching tool/turn successor;
backends perform no turn-policy decisions.

Perception-only VoiceChat transitions retain projected audio in an ordered,
architecture-owned queue bounded by the admitted source-frame capacity. On
interruption the facade selects ordinary `ReplayAudio` work until that queue is
empty. Each invocation runs the existing backbone's hidden-state execution once,
advances its cache position and PAD feedback history, and leaves perception,
RNNT, TTS, codec and RNG unchanged. The first replay supplies EOS/PAD feedback;
later replay positions supply PAD/PAD. No vocabulary head, transcription, speech
or tool event is produced by replay. Normal tool completion releases the queue
when live processing resumes. Queue payloads, replay state and host counters
participate in the same branch, completion, snapshot and fork accounting as the
other component state. Cold preparation reports the bounded projected payload
for an explicitly selected activation precision, separately from parameter pools,
native handles and independent state copies.

`VoiceChatConversation` groups these host policies under one owned admission.
Its private candidate attaches a one-shot projection to the
ordinary `PrepublicationRealtimeFrame` through the family-independent
`map_host_output` mechanism. Exact completion and architecture observation run
first; turn, tool-schema, reminder and publication-capacity checks then run on a
private candidate before the scheduler can commit neural state. Immutable text,
tool definitions and prepared phrases are shared; mutable history and cursors
remain bounded copies. Mapping errors retain their original sources.

That candidate also retains incremental agent and RNNT text decoders. Agent text
uses the facade's shared `TextDecoder` over effective agent feedback, with bounded
tokenizer lookbehind, complete UTF-8 chunks and explicit control filtering.
EOS and drain reject unfinished byte sequences; interruption discards private
lookbehind without retracting delivered text. Independently spoken reminders and
function payloads are not agent-text events. Architecture-owned
`voicechat::transcription` supplies the released RNNT vocabulary presentation
policy over retained assets: word-marker replacement and Python whitespace
trimming. Its immutable vocabulary is shared and its only mutable text carry is
bounded trailing whitespace. The facade advances both decoders on unpublished
observations and adopts them with the same neural commit. No history is reopened
or decoded from its beginning for each frame.

`VoiceChatPublicationState` wraps the ordinary neural semantic transaction and
owns bounded publication storage. Applications reserve move-only permits before
enqueueing work; exhausted slots or payload allowances return typed backpressure
before scheduler submission. The branch admits host policy with that permit and
stages observed output privately. Its semantic commit validates exact charges,
commits the neural branch, then adopts host policy and queues output without
fallible work or allocation. Discard delegates native retirement to the ordinary
transaction before releasing host admission and credit. No scheduler or backend
equations are duplicated, and dropping scheduler progress cannot lose host state.

The independent, move-only `VoiceChatPublicationReceiver` retains committed
output after cancellation or request removal and permits delivery during pending
execution. Delivery releases actual queued payload charges; unused reservation
allowance is released at commit. Limits cover publication slots, waveform
samples, RNNT IDs, decoded agent/transcription bytes, residual codes, diagnostic record counts and complete tool-event
strings. Diagnostic contents retain separate observation budgets. Applications
retain backpressured external results. Session loading and snapshots compose
this publication owner with model state, host policy and ingress.

`VoiceChatPublicationState::new_with_input` also binds an independent bounded PCM
sender. The facade depends directly on backend-neutral `eredu-media` without its
optional host-audio feature for queueing. The media queue can peek one frame and
consume its metadata without allocation; VoiceChat owns the 16 kHz/1,280-sample
cadence and selected-precision validation. Its architecture-owned `PcmInputState`
facet exposes only host input policy, frontier and candidate submission/finish,
and delegates through the ordinary realtime generation branch.

Input permits retain the head prefix while subsequent chunks may arrive. A fresh
audio invocation transfers only that frame into its unpublished frontend, whose
queue is empty between committed frames. Source-frame attribution is checked
before neural commit; successful commit then consumes the external prefix without
allocation. Failed work leaves the prefix and later arrivals intact. The accepted
PCM limit includes pending samples; each permit additionally holds at most one
cadence-sized copy, separate from the model's bounded preprocessing window.
Reminder and retained/silence tool invocations leave ingress untouched, while
perception-only tool-wait work uses the same fresh-frame path. Canonical owner
removal closes the sender without releasing a pending receipt before retirement.

`VoiceChatPublicationState::prepare_prefill` and `prepare_next` produce move-only
work for the existing scheduler. Preparation reserves output space and available
PCM, selects an invocation on a private conversation candidate, and retains the
existing exclusive host admission while queued. Tool results and policy mutations
receive `Busy` until that work commits or is discarded. Work admission transfers
those same permits and candidate into the ordinary unpublished neural branch;
it does not repeat selection or create another completion owner.

Selection distinguishes live input, function serialization, external tool waits,
response injection and reminders. The released asynchronous tool default uses
prepared silence for backbone conditioning, alternating tool work with fresh
perception/RNNT without dropping audio. An explicit real-audio override uses fresh
PCM or retained perception, falling back to prepared silence before the first
encoder frame. Reminder speech alternates
with queued perception and respects an explicit 80 ms presentation cadence.
Timeout response injection and reminder selection remain tentative until the
selected neural work commits. Blocked selection consumes no time, response token
or input. A stable scheduler descriptor retains intent, recorded time, allowances,
reminder identity and a digest of the exact source PCM after permits move into
the branch. This descriptor does not supply a distributed speech realization.

Finishing input declares a `PcmFinishPolicy`: incomplete-frame handling and a
finite number of trailing silence frames. The media queue retains only a counter
for that suffix and materializes one cadence frame at a time. Architecture
preprocessing consumes it through the same perception path as real PCM. Frame
provenance distinguishes source samples from synthetic continuation; both advance
the acoustic clock. The facade commits the frontend finish declaration separately
from completion of the suffix, preserving late finish during pending input work.
Subsequent advancement selects terminal codec drain after the declared suffix and
pending tool work complete. Silence length is explicit; checkpoint training delays
do not imply a finish duration or guarantee speech EOS within that duration. The
composition adapter retains responsibility for prepared prompt/Aria/silence assets.
The shared realtime loader supplies those assets; the session owns advancement
and snapshot/restore.

The facade exposes committed function-channel activity through
`VoiceChatSession::has_tool_work`. The WAV application uses that observation to
submit bounded silence after source EOF while tool work is active, then supplies
its configured live response tail before declaring input finished. Waiting and
response serialization therefore do not exhaust that response window. This
policy uses ordinary PCM admission and advancement; native execution does not
choose the source duration or read a wall clock.

`VoiceChatWork::submit` assembles the architecture invocation from frozen work and
borrowed `VoiceChatExecutionInputs`, consumes its admission, and attaches mandatory
host publication to the executor's existing prepublication gate. Prompt position
counts and required silence inputs are checked before consuming admission. Live
steps use the admitted turn resolver and prior-state tail policy; tool steps use
idle-only PAD gating and apply substitutions to waveform as well as feedback.
The facade prepares acknowledgement and initial-reminder speech through
`VoiceChatPhrasePolicy::{Disabled, Fixed, Streaming}` and the retained tokenizer
before startup. The released streaming pools contain 20 acknowledgements and ten
initial reminders. Each role uses an explicitly seeded, inline ChaCha8
`HostRandomState` from `eredu-runtime`, with unbiased `u64` uniform indices. The
facade selects both phrases once in the unpublished SOTC successor; committing
adopts their random states with the conversation. Snapshot/fork copies include
buffered draws and selected phrase state, with immutable pools charged to retained
owners. Native sampling keys remain independent.

During function-call collection, the facade supplies each acknowledgement token
through the architecture's independent spoken-text override, while agent feedback
remains PAD. BOS, content, trailing PADs and EOS preserve generated speech. The
cursor advances with committed Tool work; collection end or interruption clears
its remaining suffix. Perception and response injection consume no acknowledgement
tokens. The initial-reminder selection survives EOTC for `start_wait` to use.
`VoiceChatInitialReminder::{Configured, Disabled, Text}` chooses that retained
phrase, suppresses it or supplies an explicit per-call override. Initial speech
finishes through recorded playback completion even when a result arrives early.
Interruption or EOTR clears the per-call selection. Reminder receipts retain their
own supplied-token and waveform policy. No clock or process-global random stream
chooses phrases; explicit speech flags are separate from the TTS model's input-EOS
configuration and read no environment.

The returned `VoiceChatSubmittedWork` erases completion and observer types behind
one owned scheduler output. It retains the original gate and neither borrows the
executor nor changes native retirement. Submission and completion failures expose
`BackendFailure` with the original concrete source chain. The shared runtime
mapping error exposes an `Arc`-retained observer's inner error as its source, so
typed diagnostic downcasts reach the original failure. The feature-gated
`submit_mlx_voicechat` binding calls this same portable driver and the existing
`MlxVoiceChatExecution::submit`; family policy stays in portable layers and
execution uses the core scheduler.

`VoiceChatSession` composes already prepared publication state with one request
and one outstanding invocation in the existing core scheduler. Explicit
`advance` and recorded-tick `drive` use identical preparation, submission,
completion and publication. The independent input sender accepts PCM while work
is pending; the receiver retains committed output after cancellation or terminal
drain. Drain commitment finishes the scheduler request and closes ingress.
Facade transaction/output wrappers retain original failure sources before core
scheduler diagnostics format them. They do not change native completion authority.
The supplied clock origin maps monotonic recorded durations to scheduler instants;
`cancel_at` and failure abandonment use that same timeline for retirement metrics.
The shared realtime loader supplies its prepared artifacts and native session
resources.

`BoundVoiceChatSession` owns that session, its bounded PCM/output endpoints and a
`VoiceChatExecutor`. The executor holds immutable startup inputs and parameter
resources; all neural, random and conversational state stays in the transaction.
Both controlled and iterator advancement use the same submission fence and session.
A busy shared executor allows polling and PCM ingress but rejects new admission
with retryable `Busy`. The iterator can expose its endpoints for bounded delivery
between ticks. Fork binding is explicit; the MLX `fork_shared` method retains the
same model and startup tensors while the ordinary session copies accounted state.

The portable `api::realtime::prepare_realtime_model` accepts a
`RealtimeArtifactRequest` and returns an architecture-owned
`PreparedRealtimeArtifact`. The shared family registry recognizes VoiceChat's nested
NeMo configuration without a top-level `model_type`; Python target names are inert
data. Generic artifact and logical-architecture inspection admit the official
SafeTensors and complete `nemotron_voicechat` GGUF catalogs. Split GGUF preparation
retains explicit component roles and the complete companion that supplies RNNT and
codec-encoder weights. The prepared artifact retains exact auxiliary assets,
restricted component sources, canonical recipes and conversion bounds.

With the MLX feature, `prepare_mlx_realtime` applies `RealtimeLoadOptions` to this
retained artifact before native device or tensor creation. Architecture dispatch
selects the categorical-frame or PCM-speech protocol; a mismatch is a typed error.
For speech, cold selection uses `PreparedVoiceChatSession` to validate recorded
startup time, queue reservations, matching PCM precision, text/tool policy,
RNNT/turn configuration and prompt capacities. It also selects both neural cache
layouts and validates the complete parameter-residency allowances.
`PreparedMlxRealtime::load` consumes that selection and explicit execution and
weight contexts, binds components, prepares fixed-Aria and silence inputs, and
registers the existing session. Loading accepts no replacement sources or policy.
Prefill runs only on advancement. `LoadedMlxRealtime::Speech` exposes the same
bound controlled/iterator session. The facade consumes the neutral `eredu-nn` dtype
contract; native realization stays behind its MLX feature and no accelerator
dependency enters portable definitions.

Discovery identifies the realtime loading protocol; it does not advertise ordinary
text generation or native support in every backend. The logical descriptor names
the shared perception outputs, RNNT branch, dual language heads, TTS/codec components
and feedback dependencies. Its internal block/clock granularity is explicitly
partial, and text-prefill/decode capture discovery does not claim realtime frame
captures. Whole-session VoiceChat binding selects one device; component parallel
contracts do not imply a distributed session loader.

Shared MLX realtime completion retains a family-independent execution-resource
owner. The owner keeps selected stores, streams and other authorities alive and
poisons the associated execution on native submission or observation failure.
Moshi and VoiceChat supply this owner through final composition. The shared
submission scope begins before coupled equations can issue eager native work;
final output evaluation and token validation remain inside that same scope.
Errors and unwinds seal failed recovery without evaluating another graph.

`eredu_nn::RetainedTensorRoots` visits borrowed tensor roots without allocation,
evaluation or family inspection in the backend. Recurrent, convolution, spectral
and VoiceChat component owners expose independent carry as well as visible output.
The native VoiceChat adapter combines these roots with the joint neural caches,
invocation inputs and RNG. Its completion owner retains model weights, stream,
paging namespaces and the joint copy permit independently of candidate lifetime.
Native retirement controls release even when candidate disposal follows a failed
completion observation. Retained handles are not a deduplicated memory census.

Fixed-Aria prompt and silence preparation share a native completion scope with
retained model ownership. Architecture-declared roots cover every startup tensor;
errors and unwinds preserve pending native work and poison shared execution.
Successful preparation returns only after exact completion. Shared execution
handles retain one active/poison gate without copying caches or random state.

`MlxVoiceChatExecution` binds the portable coupled executor to the existing shared
output submission and prepublication gate. Architecture preflight runs before the
native scope, so missing PCM and malformed requests remain retryable without
poisoning execution. Once equations begin, errors or unwinds attach failed
completion and poison the execution. Callers consume and release completed native
frames before preparing another branch; retained completion handles keep joint
admission occupied. This selected adapter does not choose conversational policy
or distributed placement.
Compact coupled conformance fixtures live in `eredu-evaluation`, consumed only
by neutral and native validation code. The native test binds nonzero learned
components to the production coupled equations and shared submission mechanism;
synthetic geometry does not bypass released-artifact admission. Its explicit
branch and shared-scheduler runs compare output and retained state at the same
committed boundaries with identical native RNG seeds.

`eredu-codec::streaming` owns shared causal convolution carry and transposed
convolution overlap-add. Geometry declares stride, dilation, groups, padding and
tail policy. Carry is compact and bounded; finish is idempotent, later input is
rejected until reset, and cloned state supports independent continuation. Failed
synchronous operations preserve the previous carry. Parameter-free `Conv1dState`
accepts materialized kernels for each invocation; Mimi's streaming convolution
modules and EAR's independently visitable ConvNeXt blocks use this same carry.

`eredu-codec::ear_decoder` owns EAR's causal ConvNeXt equations, canonical
convolution parameters and decoder-order upsampling stages. `ear_spectral` owns
the released magnitude/phase conversion, periodic Hann window, constrained
inverse transform, overlap normalization and bounded spectral history. The
portable tensor contract supplies split-real forward/inverse FFT and sine/cosine;
MLX supplies generic native FFT and trigonometric operations. Complex native
storage does not cross the portable boundary. `eredu-codec::ear::AudioDecoderExecution`
connects selected residual-code lookup, neural decoding and spectral reconstruction
to speech execution as unpublished state transitions. Ordinary and pipeline
providers share codec-owned waveform validation, reconstruction and drain. Their
state retains selected convolution and spectral carry, not learned weights.
VoiceChat projects its normalized codec configuration into
these contracts, reversing the checkpoint's encoder-ordered stages. Speech
control-code replacement, timestamps and conversational policy remain outside
the codec.

`decoder_component_definition` declares independently leased units and per-unit
carry identities. `DecoderKernel` supplies the same streaming neural equations to
`ComponentAdapter` for complete and selected pipeline execution. The shared adapter
owns graph, parameter visitation and local state-root mapping. Upsampling
convolutions, ConvNeXt blocks and the output projection are independent units with no learned static parameters. Each unit
retains its candidate convolution carry through completion; a failed traversal
does not replace the supplied parent state. `DecoderExecution` connects these
providers to `AudioDecoder` and VoiceChat speech composition. Initial state and
neural finish depend only on geometry, so spectral drain loads no neural weights.
Codec-owned `PreparedDecoderBindings` validates exact recipes, restricts retained
source authority and accounts for F32 conversions. VoiceChat projects its admitted
decoder component into that contract; MLX supplies the ordinary resident,
host-layerwise and disk-streamed parameter policies.


EAR decoder tensor parallelism partitions upsampling outputs within each
convolution group, ConvNeXt intermediate channels and spectral output rows.
Upsampling restores each group's channel order before the next unit; MLP
contributions are reduced before adding the replicated output bias and residual.
Depthwise kernels, channel normalization and their causal carry remain replicated.
Their parameter and state geometry is explicit in the selected declarations.
Every unit still returns full-width activations and an unpublished candidate
cache through the ordinary completion path. Exact layout and communicator checks
precede acquisition; ordinary assembly rejects rank-local units. MLX retains the
supplied group and implements the same `DecoderExecution` contract without
owning codec equations or waveform publication policy.

Streaming decoder pipeline stages own contiguous learned-unit ranges and the
matching slices of causal convolution state. Codec-owned route declarations
derive channel width and frame expansion at each cut; request metadata carries
the completed cut, original batch and latent-frame count. A receiving stage
validates that metadata and its local state identity before parameter acquisition.
Only owned caches enter its candidate branch and completion roots. Tensor
parallelism replicates those local caches within the stage's tensor cohort.
Finishing the neural stage releases its carry without loading parameters or
producing artificial spectral frames; overlap-add drain belongs to the separate
spectral component. The portable waveform pipeline composes these stages with
codebook lookup and spectral reconstruction. It coordinates local success before
entering the neural pipeline and before returning the complete waveform candidate.
Participants agree on complete geometry, input code values and waveform frontiers;
each stage validates local cache ownership, batch and lifecycle before codebook
acquisition. Its request identity and commit epoch are explicit caller-owned authority. Native
bindings implement the decoder pipeline mechanism; codec equations and waveform
composition remain portable.

VoiceChat speech feedback, partial refinement and output state retain the selected
codec carry through the existing coupled transaction. Snapshot estimates and tensor
root traversal include that carry and spectral overlap. The shared executor uses
the same live, tool, reminder and drain branches for ordinary and selected waveform
providers; selecting one component does not select an entire distributed conversation.
The shared scheduler treats `AudioDecoder` groups as mandatory. This declaration
keeps codec state ownership distinct from text decoding and from projections
whose activity follows their upstream modality dependencies.

The shared component pipeline owns transport, bounded parameter leases,
completion and exact commit agreement for both encoder and decoder stages.
Common cold route validation checks contiguous ranges and equal unit ownership
within tensor cohorts. Codec preparation retains restricted source views and
exact local state layouts; the MLX adapters bind them without discovering cuts
or reopening checkpoint artifacts.

`ear_encoder` uses the same ConvNeXt equations with encoder-order strided
convolutions. `spectral::PeriodicHannStft` supplies zero-padded, unnormalized
forward transforms using the shared periodic Hann window. `ear::AudioEncoder`
concatenates real and imaginary features and borrows the decoder's residual
codebooks; it declares neither duplicate learned tables nor streaming encoder
state. Finite PCM must contain whole codec frames. Spectral gather indices,
framed samples, split spectra and native FFT scratch are separate temporary
costs from learned encoder units.

`ear_encoder_layered` leases the input projection, each ConvNeXt block and each
downsampler independently through the same traversal as the resident encoder.
It declares no learned static parameters or persistent encoder carry. The final
frame-major transpose runs inside the last unit before its completion, so a
failed invocation returns no partial latent or codes. `EncoderExecution` connects
these providers to finite waveform encoding and fixed-Aria preparation. The
encoder finishes before the shared quantizer provider runs; codebooks are not
duplicated. Codec-owned `PreparedEncoderBindings` validates per-unit recipes and
F32 conversions against a restricted source view. VoiceChat supplies the retained
encoder source role, and MLX realizes the selected parameter policy.

Runtime `prepare_resource_bindings` lowers family-admitted physical recipes and
logical aliases for independently owned resources. It restricts source authority
to their actual physical inputs before atomic recipe validation, preserving
canonical resource keys and separate materializations even when two resources
read the same checkpoint tensor. Aliases resolve within one resource.
`PreparedComponent`, low-level VoiceChat speech/backbone binding and backbone
residency inspection use this preparation mechanism. VoiceChat retains topology
admission, alias agreement, execution precision and native storage preferences;
preparation allocates no native values and reads no checkpoint payloads.

Finite encoder tensor parallelism partitions input-projection rows and ConvNeXt
intermediate channels. Downsampling partitions output rows within every
convolution group; segmented checkpoint reads preserve group order, and the
collective reconstructs full-width channels before the next unit. ConvNeXt
projection reduction precedes its replicated bias and residual addition.
Depthwise filters, normalization and projection biases are replicated, charging
354,816 F32 bytes per rank for the released geometry. There is no persistent
encoder carry. Exact retained layouts and the actual communicator are checked
before acquisition; ordinary construction rejects selected shards. Resident,
host-layerwise and disk-streamed providers use the same traversal, including the
final frame-major conversion before quantizer execution.

Finite encoder pipeline stages retain contiguous global unit ranges and exact
route-to-cut declarations in `eredu-codec`. Each route derives channel width and
frame reduction from the completed encoder units; its I32 metadata carries the
cut and original request dimensions. The receiving stage validates these values
before acquiring parameters. The last unit completes the frame-major conversion,
and the shared runtime publishes full latents with exact completion and commit
agreement. Restricted stage source views retain only owned recipes. Combined
TP/PP selection verifies that tensor participants own the same unit range.
The MLX adapter binds these retained contracts through `MlxComponentPipeline`.
Spectral preprocessing and borrowed quantization remain separate components.

`encoder_component_definition` retains the finite encoder's canonical unit paths,
parameter owners and stateless transport declaration. `EncoderKernel` supplies
geometry, equations and route-specific channel/cadence resolution through
`ComponentAdapter`; full and pipeline construction use that same kernel. Empty
pinned modules and stateless traversal metadata belong to the runtime. The shared
pipeline adapter consumes typed route resolution instead of assuming request and
intermediate frame counts are identical.

An empty architecture state partition plan is valid only when every complete
layout entry is `NoState`; it selects no local state on any stage. Layouts with
mutable state require explicit placement rules. The finite encoder uses this
contract and creates invocation-local progress without retaining convolution
carry or allocating a persistent pipeline cache.

`eredu-codec::prvq` owns fixed-mean probabilistic residual quantization and
canonical per-depth mean/variance slots. Its nearest-code operation explicitly
selects the full squared-distance equation or the equivalent equation without
the residual norm, preserving each caller's floating-point operation order.
`eredu-architectures` depends on this portable codec component for EAR-TTS
refinement; codebooks have no speech scheduling or special-token policy.

`prvq_layered` gives each codebook depth one independently leased unit, including
its canonical mean table and retained variance. Resident and bounded quantizers
use the same ordered equations; range requests skip unselected depths through
the shared execution graph. Quantization retains the original latent, residual
and selected codes through completion. Decode adds means in depth order, and
range encoding starts from the supplied residual without subtracting an implicit
prefix. `QuantizerExecution` connects either provider to waveform decoding and
the finite encoder used for fixed-Aria startup. Both borrow the same quantizer
owner. Cold `PreparedQuantizerBindings` restricts exact source recipes to codec
codebooks and accounts for F32 conversion; native binding consumes that plan
through the existing parameter policies.


PRVQ tensor parallelism partitions each mean table's channel axis with the same
rank range at every depth. Variance scalars remain replicated and charged.
Each nearest-code decision follows a full distance-score reduction; local
residual updates preserve the selected range and depth order. The final active
unit reconstructs full-width residual, quantized and decoded tensors before
completion. Codec-owned validation checks the complete physical layout and
actual communicator before parameter acquisition. MLX binds these plans through
the same `QuantizerExecution` interface and resident or bounded providers.
The native tensor-parallel binding consumes retained cohort authority with exact
reduction and failure-agreement capabilities. Shared traversal gates acquisition;
codec-owned gates agree on local score, lookup and channel-reconstruction results
before entering each collective. Waveform composition validates this coordination
contract before invoking a codebook provider.
Native reductions on guarded groups establish bounded completion before the next
control agreement, preventing lazy data reductions from interleaving with status
traffic on the same transport. Timeout handling retains native resources through
the existing completion quarantine.

`eredu-runtime` owns strict rank-local lowering of architecture-declared parameter
groups, including TP/EP ranges, segmented or chunked members and exact retained
layout validation. Codec and model families consume that mechanism directly;
family equations and parameter declarations remain with their portable owners.

VoiceChat's `tts_mog` owns the offset RMS-normalized residual MLPs, post-stack
classifier-free guidance, selected low-rank Gaussian mean and noise equations.
`MogKernel` declares one unit per residual block, a separate final-normalization
and logits head, and a selected-component bank containing residual/scale projections
and low-rank means. `MogExecution::predict` executes only the residual and prediction
groups; `selected` leases only the component bank and consumes the retained prediction
without rerunning those groups. Selected dense parameters use ordinary backend
mechanisms; only the selected mixture is multiplied. The shared resident and
layerwise traversals retain prediction, decisions, draws and all moment outputs
through unit completion.

`PreparedMogBindings` validates the symbolic parameter topology against exact retained
recipes without reading payloads or allocating native modules. Its restricted source
excludes neighboring TTS/RVQ parameters and declares bounded conversion to selected
speech projection precision, with F32 normalization and component banks.
`materialize_voicechat_mog` consumes this plan through existing MLX resident,
host-layerwise or disk-streamed policies. No learned weights belong to the static role.
The host policy can lease one unit; disk residency charges the streamer's actual
within-group transfer window, independently of other groups. These are component
providers, not a complete speech-graph residency or distributed execution plan.

`tts_rvq` owns PAD-to-zero feedback and
the eight-slot cubic schedule, including three slots with no draws and the final
22-codebook refinement. Shared sampling owns uniform/Gaussian draws and retains
the existing session RNG. These components accept supplied TTS backbone hidden
states. The coupled execution graph orders their invocations and retains
component resources through completion.

`RvqKernel` declares separate resources for the canonical stacked feedback bank
and its single latent-to-hidden projection. Ordered depth sums and residual quantization lease
only the bank; embedded feedback also executes the projection. `RvqExecution`
shares those operations and the refinement schedule across resident and bounded
providers. Prompt ingress and each refinement iteration use the same projection
identity. The bank remains one independently materialized checkpoint parameter:
its released `[31, 1024, 512]` F32 representation costs 62 MiB. Residency admission
charges that full bank and conversion workspace; it does not assume per-depth
streaming. Neither unit belongs to a permanently pinned static role.

`voicechat::tts_generator` composes character conditioning, prompt ingress,
gated fusion, shared Gemma execution and RVQ refinement. Prefill consumes prepared
prompt codes without sampling. A recurrent invocation advances caller-owned
unpublished attention caches once, retaining conditional/unconditional hidden
lanes in a parameter-free frame. Explicit refinement and ordinary frame
generation call the same sampled transition in `tts_rvq`. The generator, refinement,
speech sampling/rendering and coupled executor accept `TransformerExecution`,
`CausalTransformerExecution`, `MogExecution`, `RvqExecution` and
`ConditioningExecution` providers; selecting parameter residency does not create another
TTS generation loop. Native backends supply numerics, cache storage and random mechanisms.
`tts_feedback` owns released integer control substitutions with separate input
EOS and output-feedback policies. Idle/tool-wait decisions are explicit inputs;
completion, publication and conversational state belong to the enclosing
VoiceChat session transaction.

`voicechat::speech_sampling` connects generator and speech-component refinement
to the shared realtime branch's retained RNG. Each active slot draws uniforms
for nucleus-truncated Gumbel selection, then a Gaussian latent, including when
noise scale is zero. The architecture retains the MoG/RVQ equations and iteration
order. Controlled slot advancement and uninterrupted refinement use the same
transition; synchronous sampling or dependent neural failures discard the
requested operation's candidate RNG.

`SpeechSamplingCoordination` borrows retained replicated communication for one
recorded request, turn and invocation. The same refinement strategy gates exact
phase geometry, policy and local success around learned operations. Only the
selected owner draws mixture choices and Gaussian noise; followers receive those
tensors and need no random stream. Final agreement remains inside the ordinary
branch RNG transaction. Rendering also agrees output preparation before entering
the codec. Providers with internal collectives retain their own preparation and
failure agreements. Backends implement numerical draws, bounded agreement and
publication; architecture code owns the refinement phases and equations. This
authority supplies neither a session scheduler nor native completion.

`voicechat::agent_sampling` owns the released agent selection policy through the
shared `Sampler` and `SamplingBackend` contracts. Original-logit special winners
bypass stochastic processing; other stochastic decisions apply unique nonspecial
feedback penalties, temperature, then nucleus filtering. The pinned streaming
defaults are temperature zero, top-p one and repetition penalty one. The sampler
retains bounded effective feedback after conversational overrides, separately
from raw selection. The unpublished branch exposes mutable sampler state without
changing its cardinality: callers adopt the decision driver, accept effective
feedback, then use the same retained RNG for speech draws.

`voicechat::channel_decision` resolves both heads through that shared branch.
Agent selection consumes its ordinary draw before any override; function decoding
is greedy and consumes none. Validated effective feedback is distinct from raw
decisions and from the spoken TTS token. Tool substeps retain agent PAD while
optionally speaking an acknowledgment. The handoff validates head geometry,
override domains and history capacity before native work, and adopts sampling
state only after both decisions and effective feedback succeed. It does not
advance a schedule, publish an event or establish native completion.

Optional `ChannelDecisionCoordination` keeps that handoff on the same sequential
decision driver. `ReplicatedChannelDecisions` borrows retained communication for
one recorded request, turn and invocation. It agrees head geometry, vocabulary,
perception evidence and complete sampling policy/history before selection. Only
the owner samples both raw channels and evaluates the conversational resolver;
it publishes raw IDs, effective feedback and spoken text as a bounded I32 payload.
Followers validate this payload and resolve the agent position through the
existing forced-decision mechanism without consuming RNG. Final candidate
agreement precedes RNG/history adoption. Scalar-token portable backends can
continue using ordinary selection; tensor publication is a typed adapter, not a
requirement of `SamplingBackend`. This component authority does not bind the
full coupled graph or advance conversational policy independently on each rank.

`voicechat::execution::ExecutionCoordination` optionally binds those two authorities
to the same coupled component transition. It borrows retained replicated
communication with explicit request, turn and invocation identity. Admission
agrees exact schedule descriptors, predecessor and expected frontiers, feedback,
invocation kind, prompt IDs and masks, tensor geometry and output policy before
claiming the candidate. Metadata length agreement precedes variable word
exchanges. Component gates cover branch claim, prompt/backbone execution, speech
prefill, perception, feedback ingress, character/TTS preparation, drain and output
assembly. Only the selected owner evaluates the conversational resolver or draws
agent and speech randomness; followers receive decisions through the existing
channel and refinement drivers and may retain no RNG. The final output gate
precedes schedule adoption. Inner authority failure fences the enclosing
invocation without an additional recovery collective.

This coordination contract covers replicated ordinary component providers.
Providers containing TP or pipeline collectives must coordinate failures within
their own operations: an outer component gate cannot protect a peer already
waiting in an internal collective. The selected backbone and function-head
adapters do not acquire that protection from this authority. Native cold session
loading selects ordinary execution. The authority does not establish native
completion, commit model state or publish external events; failures after branch
claim require the enclosing owner to complete and discard the candidate.

`voicechat::schedule` owns checked per-stream invocation plans. It separates
live audio, perception, RNNT, Nemotron positions, both TTS guidance lanes, codec
frames and emitted PCM samples. Plans check explicit component capacities before
execution and accept candidate progress only when every observed counter matches
the declared transition. Their predecessor identity participates in the existing
`RealtimeScheduleState` validation before model-state publication. Neural caches,
feedback tensors and RNG remain in their existing transaction owners.
Sparse payload declarations retain same-step RVQ-to-waveform dependencies;
recurrent feedback is component state rather than a fixed publication-tick lag.
Prefill advances only the declared backbone prompt and Aria conditioning;
reminders advance TTS and codec without Nemotron; cached-audio tool substeps do
not advance perception. Separate perception-only work can process admitted speech
during tool waits. Codec drain accounts for the overlap tail without inventing
another audio or codec frame. These are invocation contracts, not a second
scheduler or a complete coupled neural executor.
The codec's cold spectral configuration supplies startup and drain sample counts
without constructing native spectral constants.

`voicechat::perception_rnnt` executes one perception invocation and sends its
unprojected output to bounded RNNT decoding, retaining the projected output for
Nemotron. Its candidate pairs both caches at the same audio frontier.
`voicechat::input` joins that transition with portable PCM framing and mel carry;
incomplete chunks produce no neural invocation, and failed decoding leaves the
parent PCM queue and caches unchanged. Tensor materialization follows cold
geometry and capacity checks.

`voicechat::speech` composes the existing TTS generator, feedback controls and
EAR waveform decoder. Prompt prefill retains the last supplied prompt code;
cached generation retains a partial RVQ frame, then applies the selected output
substitutions before codec execution. Speech state binds the exact generator,
feedback and codec policy. The enclosing transaction supplies unpublished
attention caches and random draws and establishes completion before committing
speech state or publishing audio. The shared realtime loader selects residency
and binds these components to the public session.
Speech composition accepts explicit spoken IDs and matching supplied conditioning,
or `begin_text_frame` derives conditioning from the retained character mapping
through the actual character encoder. Live control tokens retain true character
masks. Character-cell capacity and retained-cache validation precede that neural
invocation; shared schedule preflight owns component-position capacity admission.
`render_sampled_frame` encloses refinement, feedback substitution and codec
rendering in the same shared branch RNG transaction. A synchronous codec error
discards all draws made by that call, while the prior speech state remains
unchanged. Native submission completion remains the enclosing owner's duty.

`voicechat::output` packages component results into `CompletedRealtimeFrame`.
It validates the observed schedule receipt and exact sparse payload requirements,
then reshapes single-stream decisions, codes and PCM to the shared observer's
declared channel geometry. Agent and function decisions retain their actual
order; the RVQ code frame is derived output rather than another categorical
decision. The retained projection verifies host domains, finiteness, identities
and invocation coordinates before returning sample-attributed PCM and decision
metadata. Effective feedback and spoken text remain distinct from raw decisions.
The completion owner publishes the matching schedule and model state; assembly
does not establish completion, decode text or execute tools.

`voicechat::state` binds input carry, speech/codec carry, retained perception
features and effective channel feedback to one caller-owned neural transaction.
That transaction branches and publishes Nemotron and both TTS guidance cache
lanes together. The architecture does not clone mutable native cache handles or
compose two fallible cache commits. Component candidates are adopted only after
the joint neural commit succeeds; discard delegates to the same neural owner.
MLX's reusable `MlxJointState` owns a hybrid state and a key/value state under
one transaction identity. Identity and layout validation precede replacement of
either canonical state. Admission uses actual retained tensor extents, including
both rows of classifier-free guidance, and keeps cumulative copy usage outside
the candidate. Native recovery retains unresolved copy work. The final composition
adapter implements the architecture's borrowed cache access contract; the reusable
owner has no VoiceChat scheduling policy.

Joint frame candidates copy metadata and share immutable attention, fixed-state
and resident append-stream roots. Fixed-state replacement constructs unpublished
roots; discard leaves committed values intact. Full-attention device caches use
private capacity in 128-position chunks, publishing only logical prefix views.
Each suffix is reserved once before native construction, without refunds on
failure or discard. Committed append chains reuse unused slots; divergence,
truncation and restoration allocate independent backing before appending. Native
one-use append nodes reject graph tracing and retain ordinary stream dependencies
and completion resources. Published prefix elements remain immutable even while
private suffix writes are pending. Sliding caches use reserved suffixes before the
first retention crop, with capacity clipped to the declared retention limit. Once
cropping changes the prefix origin, they use immutable concatenation; returned
multi-token spans, absolute positions and attention windows remain unchanged.
Fully immutable candidates submit no copy graph and require no copy-completion
wait. Overwriting capacity caches, compressed storage and mutable paging namespaces
retain physical isolation and native recovery before execution. Mixed candidates
share immutable device/fixed/stream roots and sealed paging blocks while copying
mutable storage. Paging candidates own independent catalogs and identities within
the same finite process pool. Explicit snapshots and forks deep copy all state. Branch retention limits remain conservative; immutable candidates
consume zero native-copy allowance, and mutable isolation keeps the full
conservative charge without refunds. Exclusive admission and generation identity
remain held through native completion, including failed observations.

The backend's reusable cache mechanism consumes existing neutral cache policies;
no family selects a separate storage path. Append memory contracts and continuation
bounds include padded capacity; snapshot reservations additionally price private
spare bytes absent from visible tensor extents. Explicit snapshots copy only
logical elements and discard shared capacity ownership. `safemlx::PrefixAppendBuffer`
keeps capacity arrays inaccessible through its safe API and reserves suffixes
through thread-affine ownership. Borrowed host slices require row-contiguous
storage; owned reads and copies gather strided logical elements, and strided
views cannot exceed logical input extents. CPU and Metal perform suffix writes;
CUDA uses immutable copy construction because host access may transfer complete
physical allocations. This mechanism changes no portable family equations or
public inference capabilities.

`eredu-runtime::execution_control::IsolatedStateCopy` declares completed,
independent state copying and atomic restore without text- or family-specific
policy. `IsolatedStateSnapshot` reserves through the existing `SnapshotBudget`
before invoking those mechanisms, retains the original budget for later restores,
and translates mechanism failures through `BackendFailure`. Native recovery
retains a clone of the logical reservation with unresolved copied resources.

`MlxJointState` realizes this contract through its completed isolated-copy path.
An isolated owner has its own exclusive execution admission and shares its
origin's cumulative copy counter and limits. Saved caches therefore coexist with
continued inference; restoration adopts both cache families only after origin,
layout, idle-state and generation-capacity checks. Its target admission and
non-rewindable accounting survive restore. The enclosing session remains
responsible for complete architecture carry, conversation and external-delivery
state.

The shared `RealtimeGenerationState` composes isolated model copying with explicit
`SnapshotValue` copies of its schedule, contract identity, sampling controls,
sampler bank and optional RNG. These values declare complete logical storage and
independent copy behavior; `Clone` alone does not admit them. Restore validates
contract identity and cardinality, completes all copying, then restores model state
before moving the copied values into place. Publication revision advances from the
live boundary even when model and audio clocks rewind. The model copy mechanism
owns quiescent admission and native resource accounting.

VoiceChat schedule/policy values have inline storage; its agent sampler prices and
copies retained effective-token history. MLX's RNG adapter copies its two-word key
into independent host storage using an explicit CPU compaction stream, preserving
strided input keys without drawing randomness or selecting a model execution
device. Source, temporary, output and logical reservation remain protected by the
ordinary copy-recovery owner. The same general native array mechanism copies
VoiceChat component tensors; it compacts arbitrary strides, preserves scalar
representations, and uses independent host storage. Neural cache copies retain
their selected native storage policy. Estimates charge logical payloads,
compaction temporaries and descriptors, not allocator workspace or a backing-byte
census of borrowed source aliases.

`voicechat::state` implements isolated copying of its complete component state:
PCM queue/finish markers, rolling mel history and cached rows, perception caches,
RNNT LSTM/hypothesis, speech feedback, codec convolution/spectral carry, retained
audio features and effective channel feedback. The neural owner proves origin and
quiescent admission. Restore installs both neural cache families before infallibly
moving the completed component copies; no partial component restore is published.
The shared realtime wrapper adds schedule, sampler history and RNG under the same
reservation. Codec owners declare their own state copies and geometry-vector
estimates, using neutral `eredu-core` estimate/error types and runtime copy
contracts. The backend does not inspect family fields.

Host audio owners expose backend-independent mutable-carry copy estimates and
copies without depending on runtime. Immutable FFT/filter tables and speech policy
remain shared with the original processing/module resources; their storage is not
new snapshot allocation. Mutable vectors, cached mel rows, tensor payloads and
copy temporaries are charged. These model-generation snapshots exclude the facade's
external ingress queue, conversation policy and committed publication receiver.

The facade composes complete idle `VoiceChatSession` snapshots around the same
`IsolatedStateCopy`, `SnapshotValue` and `SnapshotBudget` contracts. Snapshot limits
are configured once per session; captures and restores share that accounting owner.
Its private saved host data is not another runnable publication owner and cannot close the
live sender when dropped. Capture copies accepted PCM, queued publications,
conversation/decoder state and recorded timing; restore validates owner identity
and unchanged external input, delivery and host-action frontiers before admitting
copy work. Live input and output are fenced while custom diagnostic and neural
copy mechanisms run. All fallible copies precede neural installation, after which
host state moves into the existing sender/receiver owners. Retired diagnostic
storage drops after complete host installation. The scheduler remains the sole
execution driver and keeps cumulative counters and submission identities.

Undelivered speech, text, tool-event strings and diagnostics have explicit bounded
copies. `eredu-runtime` supplies `SnapshotValue` for core realtime diagnostic
records, preserving observation accounting; architecture-owned speech and RNNT
host outputs compose those records without native types. Host policy prices its
mutable carry and retained response/reminder/RNNT vocabulary storage. Immutable
tokenizer/schema resources retain their shared preparation lifetime and are not
new per-snapshot allocations. Public restore errors preserve native sources in
`BackendFailure` and distinguish crossed external-effect boundaries.

`IsolatedStateFork` extends isolated copying with runnable-child retention. The
shared realtime and architecture wrappers use the same component copy driver for
snapshots and forks; the neural owner supplies the distinct lifetime behavior.
MLX retains the fork reservation in the child's independent joint admission, not
in the cumulative counter shared with its parent. Existing copy permits carry
that lease into frame retention and submission recovery before native work starts.
A provisional snapshot restore keeps the target admission and its fork lease,
without retaining the provisional restore reservation as a permanent allocation.

`ForkStateEstimate` composes completed neural-candidate storage through those same
layers. The facade checks it before semantic commit. `SnapshotRetainedGrowth`
admits positive growth against the shared reservation and retains conservative
high-water charges across restore and completion ownership. Host policy and output
queues supply separate regions, including diagnostic storage; ingress reserves
its maximum configured PCM carry. Retained growth spends no snapshot-copy credit.
Temporary execution workspace retains its ordinary execution admission.
Live host estimates include vector/string spare capacity. Core tensor observations
expose their owned storage size for neutral diagnostic accounting.

The facade forks the current idle publication owner into another ordinary
`VoiceChatSession`. Host policy owns tool/turn/reminder rekeying and pending receipt
mappings; queues own copying, ID-growth admission and recalculated payload charges.
Each child has independent ingress, output, conversation and restoration identity.
Its publication state, endpoints and pending branches retain the branch reservation,
while all descendants use the original budget. A separate bounded lineage ledger
prevents admitted request-ID reuse across cancellation, failed copying and restore.
The backend neither assigns conversational identities nor emits external events.

`voicechat::execution` runs the coupled equations on this shared generation
branch. It checks schedule capacity, input readiness, sampling admission and
predecessor frontiers before claiming one invocation. Prefill runs framed prompt
ingress without unused vocabulary readouts and fixed-Aria speech preparation.
Live and fresh-audio tool steps run perception/RNNT once, invoke the existing
Nemotron layered runtime once, select both channels and condition/render speech
using the retained RNG. Cached-audio tools, perception-only work, reminders and
codec drain preserve their declared independent clocks. Successful output carries
actual component receipts and resources for the shared completion mechanism.
Native submission scopes and prepublication observation belong to the enclosing
execution owner; the shared realtime loader binds that owner to the facade session.

`voicechat::observation::VoiceChatHostObserver` wraps a family-independent native
channel observer. After exact completion it validates the output projection,
retained schedule policy and fresh-input attribution, then returns owned PCM,
RVQ codes, raw/effective decisions, diagnostics and timestamped RNNT output.
Source sample counts exclude explicit final padding; tool and reminder steps
carry no new transcription. Conversion errors preserve their native source
through `BackendFailure`. `PrepublicationRealtimeFrame` caches these host values
and gates the shared scheduler's commit.

MLX exposes `MlxRealtimeChannelObserver` as an ordinary backend runtime mechanism.
It converts declared categorical/dense channels, aligned values and diagnostics
without applying family semantics. Moshi composition delegates to that mechanism
before its architecture-owned projection; VoiceChat's neutral observer accepts
the same mechanism. Submission scopes must still begin before neural execution
and retain or retire failed work before branch disposal. Wrapping a successful
result in `submit_realtime_output` does not discharge that failure-path duty.

`voicechat::backbone_execution::BackboneExecution` owns selected prompt/audio
fusion, normalized-hidden execution, agent projection and state-frontier validation.
The coupled executor holds this provider without requiring its state to implement
the ordinary layered cache interface. `OrdinaryBackbone` borrows the existing
resident or layerwise invocation; `ParallelBackbone` selects collective embedding,
the same Nemotron traversal and collective agent projection together. Shared
decode executes the hidden traversal once, then projects both heads from that
result. Prefill does not evaluate either head. `ParallelFunctionHead` separately
selects the function head's communicator and provider, with exact row-ownership
validation before projection. Runtime errors retain their original sources;
parameters and mutable caches keep their existing owners. Prepared supplied
ingress exposes the same hidden-state boundary to pipeline composition.
Nemotron target boundaries declare signed or unsigned token IDs explicitly;
partition construction retains the selected schema, and VoiceChat supplied
ingress validates its signed IDs before creating the boundary.

`voicechat::progress` reads actual input, selected Nemotron, TTS and codec
frontiers before completing a schedule receipt. The backbone provider validates
its state; ordinary and TP adapters skip stateless MLP slots and reject inconsistent
local positions. Both guidance lanes occupy batch rows
in the same TTS caches, so their reported positions describe the same causal
invocation. A provider for a stage without stateful backbone slots must retain an
explicit logical frontier; the layered adapters reject an absent local frontier.
Native coupled pipeline construction and conversational rank coordination remain
separate from these selected backbone adapters.
State reads may materialize selected local cache slots through the existing
storage policy. They do not advance inference or establish native completion.

`tts_prompt` owns fixed-Aria silent-carrier preparation. It validates explicit
carrier-sample and batch-position budgets before encoding, prepares source PAD
overrides and text/audio masks, and expands the single encoded carrier across
batch rows. The released geometry encodes 38 silent frames and retains 37 prefill
positions. Prepared inputs expose character packing and initial unshifted code
feedback without artifact access, sampling or conversational scheduling.

Shared decoder construction declares Q/K RMS scale offsets and float32 gain
arithmetic independently from residual normalization. Rotary specifications can
vary by layer. VoiceChat's `tts_transformer` uses these policies for Gemma3 causal
blocks and T5Gemma bidirectional character blocks, retaining ordinary affine,
attention, gated-MLP, parameter-group and tensor-parallel mechanisms. Gemma3 uses
local/global rotary bases and normalized Q/K; T5Gemma scales supplied embeddings,
omits Q/K normalization and combines key padding with bidirectional masks.
Both are embedded-input components with no vocabulary axis. Blocks and final
normalization construct separately; causal calls consume caller-owned unpublished
attention caches and validate frontiers/capacity before mutation.

`TransformerKernel` supplies equations for both stacks through `ComponentAdapter`,
with one independently leased unit per block and a separate final-normalization
unit. Resident, bounded and selected pipeline execution use this declaration.
Typed equation hooks accept either borrowed caches or owned candidate storage;
normalization-only stages own no attention slots. The shared adapter owns graph,
address and local state mapping and empty pinned parameter visitation. The shared decoder's
`embedded_parameter_description` supplies block and normalization topology without
inventing vocabulary parameters; speech maps those groups to its own execution
units. `PreparedTtsTransformerBindings` checks admitted shapes and recipes without
native allocation or payload reads, retains only the selected stack's physical
inputs, and selects bounded conversion to the admitted matrix format and portable
speech precision policy, with F32 normalization and state.
MLX consumes that plan through the existing resident, host-layerwise or
disk-streamed policy, without checkpoint naming or equation dispatch.

Tensor-parallel TTS construction derives local projections and KV geometry from
the same decoder parameter declarations. `PreparedTtsTransformerBindings::select_parallel`
uses the shared physical-layout planner and applies its placements to retained
recipes once. Rank-local construction preserves global parameter identities,
replicates normalization weights and invokes shared decoder reductions before
residual updates. Causal masks, rotary policy, character masks and embedding
scaling share ordinary execution code. The MLX component adapter derives the
selection from its communicator rank/size, retains that communicator and exposes
the architecture's local cache layout. It does not choose conversational policy
or random outputs. Full VoiceChat pipeline transport and coupled distributed
session selection are separate from this complete-stack TTS component entry point.

TTS partition construction also declares ownership of contiguous block ranges
and the final normalization unit. Architecture-owned boundaries retain the causal
position or character key visibility; character embedding scaling applies only
at ingress. A normalization-only partition has no cache slots. The shared
`LayeredPartitionDriver::begin_with_preparation` supplies optional state geometry
to architecture preparation, so stateless stages do not borrow another group's
cache layout. Stateful text partition entry still requires its selected layout.
The partition contract uses the same block equations and tensor reductions as
ordinary execution. The distributed component uses the existing graph scheduler,
transport and publication mechanisms through `PipelinePartitionArchitecture`;
its ingress accepts supplied embeddings and its output width describes hidden
features. Stateful text implementations retain their explicit state-layout
requirement under that same contract.
`LayerwisePartitionRuntime` executes a selected local dependency chain with
caller-owned state. It prepares ingress once, retains the architecture context
between groups and emits output after the last group. TTS stages can contain a
block range, a block suffix followed by normalization, or normalization alone.
Unit acquisition, retained-root completion and failure cleanup use the existing
`LayerwisePolicyForward` lifecycle.
Canonical layered partition validation requires the primary group's declared
pipeline placement; it does not impose decoder semantics on an encoder. Text
task requirements and prediction-group constraints retain their existing owners.

Selected execution layouts retain canonical group/unit addresses while compacting
local storage ordinals. Runtime's `ExecutionUnitAddressMap` validates the selected
logical addresses against each local policy slot before acquisition or parameter
inspection. It requires unique invocations and matching execution groups, retains
nonzero global indices, and supplies the same local slots for completion and abort.
Cloned native policy handles share this immutable mapping. Parameter-storage and
mutable-state ownership remain independent of this invocation address translation.
MLX materializes those addressed bindings through the
ordinary resident, host-layerwise and disk-streamed providers; unowned groups
have no residency or transfer declarations. Its TTS stage adapter consumes the
prepared partition and invokes the shared local-stage runtime, with optional
native tensor collectives.

`PreparedTtsTransformerPipelineBindings` retains exact component bindings,
partition ownership, the communication manifest and `PartitionedExecutionPlan`.
Cold validation checks canonical units, local state, tensor rank placement,
boundary roles and dimensions, stage ingress/egress routes, exact collective
requirements, all-rank publication and bounded completion. Restricted checkpoint
sources remain attached to the selected recipes. Native materialization consumes
this selection without reopening artifacts or deriving a family topology.

The MLX pipeline adapter realizes the retained manifest through
`MlxDistributedSession` and uses the existing `PipelinePartitionExecutor` and
`PartitionedTextExecution` strategy for cross-process PP and combined TP/PP.
The executor retains the architecture context across local groups, including a
block suffix followed by normalization, and emits a boundary or final output
only after the last local group. Typed boundaries carry causal position or
character visibility across both block cuts and the final-normalization cut.
Resident, host-layerwise and disk-streamed execution use the same selected policy;
residency reports borrow that policy's actual accounting.

`OwnedTransformerRuntimeState` holds a caller-owned candidate cache vector,
selected ownership and an explicit logical position. A normalization-only stage
retains its position even though it owns no attention slots. Mapping independently
copied caches into a replay branch preserves this metadata; it does not clone
native mutable cache handles or grant new execution authority. Recorded
invocation descriptors, phase failure agreement, output broadcast, bounded native
completion and commit epochs use the shared distributed contracts. The caller
adopts candidate state and publishes component output only after a committed
outcome. This component binding does not wire the full coupled VoiceChat graph,
its turn/tool policy, sampling authority or public session loader to distributed
execution.

`ResidentRuntime::forward_with_state` and `LayerwiseRuntime::forward_with_state`
borrow a compatible mutable state realization for the ordinary traversal. The
architecture implementations for the stored and borrowed state types declare the
same parameter graph and unit order. `LayerwiseRuntime::forward_parallel_with_state`
borrows local cache state through the existing parallel traversal and collective
hooks. Parameters, providers and executor caches remain owned by the runtime,
including during errors and unwinding. TTS wraps
the caller's unpublished cache slice in a transient state adapter; each layer
exposes its retained K/V through `RuntimeLayerState`, while masks remain in the
forward context. `TransformerExecution` describes transformer configuration and
character encoding; `CausalTransformerExecution` validates and advances the
provider-selected causal state. TTS generation, speech sampling/rendering and the
coupled transition use this same contract for complete cache slices and local
pipeline state. Character and causal providers can have different concrete types.
The joint neural transaction borrows its selected TTS state alongside Nemotron
state, and progress checks use the causal provider's validated frontier.
The architecture-owned causal pipeline adapter consumes an explicit invocation
identity and one-use commit epoch, using the existing coordinator and pipeline
executor. Its enclosing owner agrees conditioning failures before causal
submission; this component adapter does not supply full conversational rank
coordination. MoG prediction and selected
readout, TTS RVQ feedback and learned conditioning also accept resident or bounded
providers. The codec neural decoder and residual codebooks also accept those
policies, as does the finite codec encoder. These component providers do not
establish full speech-graph residency or coupled native TP/PP.

MoG tensor parallelism partitions the residual MLP intermediate dimension and
reduces each down projection before post-normalization and residual addition.
The large selected-component bank partitions the low-rank dimension coherently
across each mixture's mean rows and low-rank matrix columns. Selected contraction
contributions are summed before standard-deviation scaling, residual projection
and supplied Gaussian noise. Norms, mixture logits, latent residual and log-scale
projections remain explicitly charged replicas. Guided hidden values and outputs
remain full-width. Architecture-owned placement and exact recipes drive the same
resident, host-layerwise and disk-streamed providers; the native adapter binds
their retained coordinator. Invocation agrees the operation, geometry and scalar
policy before acquisition, and checks the complete retained layout against rank
and degree. Shared layered hooks agree parameter acquisition and completion;
architecture-owned gates agree local projection and selected-component preparation
before each reduction. These component calls do not introduce a sampling
authority, pipeline stage or conversational commit.

`eredu-nn::NeuralBackend::ordered_codebook_lookup_sum` describes a generic
F32 stack of codebooks, I32 per-depth indices and increasing-depth additions
starting from zero. The vocabulary-sized sentinel contributes zero; signed
indices retain ordinary gather semantics. The portable implementation uses
ordinary tensor operations. MLX fuses lookup, sentinel masking and the ordered
sum in one Metal kernel, with one lazy domain assertion owned by the enclosing
submission. Invalid candidates remain memory-safe and cannot publish successfully.
The native mechanism does not own speech-family geometry, refinement policy
or sampling.

RVQ banks retain squared norms `[31, cardinality]` of their bound F32 means.
Parameter mutation visits invalidate the norms. Bounded eviction releases the
cache with its bank; channel-sharded banks retain local norms and still reduce
scores before argmin. `eredu-codec` accepts caller-retained norms through its
shared score-reduction seam, including local preparation failure agreement.

TTS RVQ tensor parallelism partitions the stacked feedback bank's latent axis and
the shared feedback projection's input columns coherently. Codebook scoring stays
in `eredu-codec`: its reusable score-reduction seam combines channel contributions
before the existing first-minimum argmin and residual subtraction. The seam
requires an unchanged F32 score geometry. VoiceChat retains the released distance
metric, fresh residual per refinement slot and ordered depth schedule. Depth sums
restore full latent vectors; embedded feedback reduces the local projection
without first gathering its input. Code candidates remain replicated and leave
their parent untouched. Exact placement and retained recipes use ordinary
resident, host-layerwise and disk-streamed providers; native code supplies only
materialization, collectives and completion. Native TP construction requires the
retained coordinator. Invocation agreement covers the operation and refinement
frontier; local validation, codebook scoring, residual updates, padding and
projection results agree before subsequent collectives. The same layered hooks
propagate acquisition failures. Component execution introduces no
new random decisions or session commit boundary.

`validate_collective_coordination` owns common cold admission for these guarded
collectives and codec PRVQ: local membership, bounded completion, failure agreement
and exact requested tensor operations. Families declare required dtypes and
minimum payload geometry and retain their partition constraints. The selected
communication owner validates actual invocation extents. Guarded MLX reductions
complete through the existing bounded completion mechanism before control traffic
reuses the transport; a timeout retains native resources through quarantine.

`tts_conditioning` owns character packing, pooling, scattering, saved token flags
and gated fusion. Token mappings and integer controls remain retained host
preparation data; neural calls do not reopen tokenizer assets. `tts_ingress` owns
the shifted prompt-code equation, BOS boundaries, fixed Aria substitution and
conditional/unconditional lane order. Causal ingress and iterative refinement
share one `embed_code` parameter. The frozen speaker projection explicitly uses
the stored matrix orientation; learned speech equations remain outside MLX.

`ConditioningKernel` leases character parameters, ingress parameters and
gated fusion as three separate units, with no learned static leases. Character
lookup and pooling/flag projection are separate invocations around the selected
character transformer. Prompt/decode ingress consumes feedback only after the
selected RVQ provider completes. These boundaries avoid retaining a conditioning
lease while another provider runs. Empty character selections skip both character
phases and the transformer. Ingress retains both guidance outputs through unit
completion. Cold `PreparedConditioningBindings` preserves exact source views and
bounded F32 conversion recipes; MLX uses the ordinary resident, host-layerwise or
disk-streamed parameter policy. The generator consumes these providers through
one conditioning contract, including controlled advancement and sampled execution.


The generator caches at most 64 single-token character results with at most 64
packed columns. Its key includes the complete prepared batch, flags and masks,
plus the identities of both parameter providers. The neutral contracts default
to no immutable identity; mutable, bounded and collective providers bypass the
cache. Native resident providers certify their private immutable parameters only
on the bound execution stream. Provider replacement clears retained results.
The cache contains no causal state, random decisions or tentative codes and is
shared by ordinary and controlled advancement. Cold speech admission reserves
`execution_cache_device_bytes` separately from parameter pools: F32 RVQ norms
plus the maximum resident character-result payload. Both speech and combined
VoiceChat admission include this allowance in the device limit, including the
perception position tables and their projections.

`ConformerBlock` owns a single-frame relative-position table and its projected
values separately from causal conversation state. Larger chunks bypass this
fixed-size cache. The layer's position operator clears projected values before
mutable parameter traversal or precision reselection; cloned parameter owners
start cold, while snapshots retain only causal state. Context and input dtype
changes rebuild the constants. The neutral `Tensor::immutable_context_key`
contract defaults to no reuse; MLX certifies its device/stream identity using only
native metadata. The backend owns this identity and ordinary tensor operators;
position equations, cache policy and cold memory bounds remain architecture-owned.
The conservative cache allowance is two F32 position tensors per global layer;
local head projections do not exceed that bound. Unit eviction releases its
constants along with the parameter owner.

Conditioning tensor parallelism partitions character embedding columns, character
projection rows, speaker-projection output columns and both fusion projections'
output rows with their biases. Architecture-owned placement validates coherent
rank slices before any provider acquisition. Ordinary resident construction
rejects local modules in place of complete units. Full-width values are reconstructed
through shared collectives before character flags, speaker/control substitutions
and fusion's final RMS normalization. Flag tables, BOS/null, fixed Aria, gates,
normalization and residual scale remain explicit charged replicas. The same
three independently leased units serve ordinary and selected tensor execution;
MLX retains the supplied communicator and binds exact neutral recipes. This
component selection does not select a complete distributed speech session.

Architecture and capture discovery follows the same ownership boundaries.
`eredu-core::discovery` owns the versioned logical graph, physical decoder layer
groups, ordered execution passes, parameter-sharing declarations, symbolic axes,
declared observation catalog, and separate execution-support reports.
`ArtifactArchitecturePlan::architecture_descriptor` in `eredu-architectures`
projects admitted family configurations and existing component/parameter
declarations. Runtime traversal and routing collectors share typed path
declarations with discovery. Shared-stack projections retain distinct logical
execution nodes and capture paths, map them to physical layers and passes, and
declare exact weight aliases using the same architecture-owned mapping as
checkpoint lowering. These declarations do not imply shared mutable state or
native allocations. Architecture descriptor schema 14 and component schema 13
are independent of observation catalog/support schema 1. `eredu-runtime` combines phase/conditional
requirements with selected session and collector facts; a backend reports only
native collection/conversion mechanisms. The facade exposes cold discovery and
enriches ordinary inspection reports. No discovery step requires native
resources. See [the consumer guide](architecture-discovery.md) for coverage,
limitations, and executable examples.

Bounded observation follows the same dependency direction. `eredu-core::capture`
owns immutable admitted plans, exact catalog selectors, symbolic request geometry,
budget comparison, native transform contracts, and versioned host records.
`eredu-runtime::capture` reserves before calling native transforms and owns the
one-step delivery ledger. Architecture preparation projects and retains catalog
semantics with the exact prepared source identity; backends consume that projection.
MLX implements eager slicing, previews and chunked reductions under its existing
submission/recovery owner. It queues no native capture handles. The facade composes
capture with `ControlledTextGeneration` and the ordinary committed-token semantic
driver: tokenizer, settings, EOS, decoding, cancellation and transport stay above
the backend. Physical allocator/private-workspace bounds remain explicitly
unsupported; logical capture storage is not a bound on total inference memory.
See [bounded capture](bounded-capture.md) for the accounting and delivery contract.

`TokenSamplingDecision` carries the exact sampling filter plus optional borrowed
tokenizer provenance across the neutral controller/backend boundary. Facade
constraint owners supply the intersection once; runtime forced-choice wrappers
retain its pre-override domain while narrowing only the sampling filter. MLX
candidate transforms consume that borrowed domain for Original and Effective
records. The speculative runtime exposes the same decision to the raw-logit
capture callback before applying its filter, so neither native path reconstructs
grammar policy or queries a second mask. Unknown provenance stays explicit in
the optional host summary; filters are not retained in capture or snapshot state.

`eredu-core::intervention` owns the versioned host-only operation schema, exact
activation payload validation, routing request validation, session/source-bound
immutable admission, and attributed outcomes. An intervention declaration is
separate from the observation catalog: observing a value does not authorize
mutation. Architecture declarations own targets, routing policy and score stages.
`eredu-nn::routing_intervention::execute_routing_intervention` owns the routing
action/stage sequence over `RoutingMechanism` primitives; it evaluates the router
projection once and validates effective decisions before provider dispatch.
`eredu-runtime::intervention::apply_activation` owns activation operation dispatch,
exact selected-region validation and update order over `InterventionBackend`
primitives. No native primitive receives the public intervention action enum.
Runtime mechanisms also own scheduling, ordered composition, shared capture
accounting, session admission traversal, immutable installation and observer
forwarding through `CaptureObserver`. Facade composition owns ordinary generation,
session association and experiment reset policy.
Pre-dispatch controls travel through the neutral selector and observed expert
provider contracts. Architecture construction and discovery consume the same
selection specifications. MLX implements projection, configured score transforms,
top-k and coefficient mechanisms, native predicates, exact payload realization and
native slicing/arithmetic/updates; it has no intervention recipe or family catalog.
Backend `InterventionEstimator` facts supply indexing constraints, evidence costs
and additional original-decision resource estimates to the shared cold traversal
and exact runtime reservation. MLX estimates include group-mask dimensions and
host partition lookup storage; runtime contains no MLX-specific cost formula.
Unknown estimates fail, and requests without original evidence neither estimate
nor construct an original decision. Native prompt inspection, devices, typed error
conversion, completion retention and recovery remain with the backend. Shared
observers borrow the existing run owner and do not own native completions.
See [intervention plans](interventions.md) for exact operation and state semantics.

Completed-token control follows the same dependency direction.
`eredu-core::execution_control` owns versioned support/status/resource contracts
and the thread-safe pause/cancellation request handle. `eredu-runtime::execution_control`
also owns canonical choice intersection and prospective sampling validation.
`TokenChoiceController` retains the ordinary grammar and commits through it once;
the native sampler consumes a one-candidate decision under its usual RNG/adaptive
policy. `TextSamplingControlBackend` supplies exact sampling facts and atomic
native temperature/key installation. It does not validate portable override policy
or reset history. The facade owns tokenizer membership, bounded override records,
and explicit field-audited semantic storage estimates; immutable tokenizer data is
shared across decoder forks. Unknown opaque grammar costs disable snapshots.
The same runtime module
owns validated lifecycle transitions and retained-state reservations; cumulative
copying accounting lives outside rewindable generation state. The existing shared
`CaptureSession` owns opaque checkpoint/restore/fork of its schedule and record
lifecycle. Child capture and intervention admissions pass through shared validation
with the child's actual discovery, session identity and estimator facts. Native
completion remains a separate prerequisite supplied by the existing backend owner.
Core's ordinary text machine stores its continuation separately from its runtime
borrow. `TextGenerationDriver` advances detached continuations through that same
machine, validates exclusive driver identity, and requires native completion and
record draining between steps. It does not switch model state or own semantic
termination; facade composition coordinates those with the native state slots.
Facade `CommittedGenerationCursor` is the ordinary one-token commitment driver;
the one-shot loop delegates to it. Exact incremental decoder/parser forks remain
in the facade and do not replay token history. See [execution control](execution-control.md)
for the boundary and current implementation status.
The shared replicated session additionally exposes `ReplicatedTextSnapshotMechanisms`
for side-effect-free native estimates and isolated state copying. Its control-state
slots retain exact executable identity, typed state geometry, prepared-input identity
and commit metadata. The portable session validates and exchanges those slots for
serial branches without rebuilding the executable or assigning a second native
completion owner. Backend realization and complete facade snapshot composition are
separate from this low-level state-slot mechanism.
`NativeTextStateBackend` provides the narrow outer backend contract for those
native slots. MLX implements it through the same typed replicated session and
architecture-erased adapter, without family dispatch. Copy operations settle under
the existing session authority and recovery owner. Exchange performs a validated
host move while that authority is idle, so no fallible native submission follows
the swap. Input-identity and commit metadata are included in portable storage
estimates. Full generation capabilities remain a facade composition concern.

`TextSnapshotBackend` supplies complete native sampler/RNG and pending-input
copying beside that model-state contract. `TextContinuationSnapshot` in runtime
composes those mechanisms with an explicit `SnapshotTokenController` independent
copy contract and the existing shared capture checkpoint. Every component supplies
known logical costs before reservation. Restore stages all fallible copies and
capture validation before native exchange and infallible host installation; no
sampling or prompt execution is involved. `ManagedTextContinuation` keeps the
child retention lease with the logical child when native slots are exchanged.
MLX copies RNG keys and input arrays through its existing submission/recovery
owner and preserves standard/Mirostat history and absolute next prediction.
Runtime owns child re-admission, inherited ledger policy and snapshot accounting.
Native growth facts use the saved typed state and an admitted future input span;
runtime combines them with sampling/pending-input facts. MLX prices declared
component geometry, actual cache capacity growth and potentially absent fixed
state without executing input. Unknown growth rejects runnable branch admission.

Facade `ControlledGenerationSession` borrows the loaded runtime exclusively and
uses the ordinary committed-token cursor, constraints and semantic pipeline for
step/pause/resume. Its bounded sequence-numbered delivery shares compact-JSON
transport accounting with observed generation. Backend
`text_execution_control_support` is an exact opt-in fact. Full facade snapshots
are reported separately after complete semantic/grammar storage estimates and
non-resettable resource limits are established. Built-in semantic parsers have
explicit logical storage estimates; opaque active llguidance state remains
unsupported until it supplies complete costs. Snapshot/restore pairs native state
with exact cursor/parser/decoder state and a versioned consumer output checkpoint.
The generic `LoadedModel<B>` exposes these controls and snapshots without
exposing native handles or duplicating their policy. Reusable host/native continuation
checks live in `eredu-evaluation::execution_control`, used only as validation
tooling, with native forward/reload probes and source-preserving copy failures.
Controlled speculative generation uses a scoped `SpeculativeGenerationVisitor`
over the same `SpeculativeScheduler` and core request transactions as uninterrupted
speculation. Backend resources remain borrowed inside the controlling worker
closure. Runtime exposes scheduler phases and host proposal/verification records;
it does not duplicate proposal acceptance, residual sampling, cache rollback or
semantic publication. Facade request preflight and terminal-output construction
are shared with uninterrupted generation. All inspected paths use runtime's
`TraceBudget` bounded JSON counter.

Core speculative snapshot contracts preserve exact request identity, canonical
sequence, constraint/parser, sampler, both RNG streams, lifecycle and adaptive
statistics. Runtime reserves known snapshot costs and owns opaque handles and
monotone restoration epochs. Architecture external-assistant adapters preserve
seed tensor topology and cache identity; MLX supplies isolated native copies and
completion. Transaction rollback markers alone never imply reusable snapshot
support. Capture owners are shared across tentative sampler clones and restores,
so budgets cannot be multiplied or refunded. MLX raw prediction capture uses the
same bounded collector as ordinary generation, with target/draft roles attached
by the sampling boundary. Other activations require explicit phase attribution.

Every new inference feature must support controlled advancement through these
shared owners, or expose and document a concrete capability limitation. Tests
compare controlled and uninterrupted behavior, not repository layout.

Facade branch construction derives finite decoder/parser/history growth and pairs
it with runtime's native growth facts before reserving retention. Its child stream
starts with explicit parent metadata and the delivered canonical/semantic prefix;
native and shared capture adapters never reconstruct that output. Branch exchange
moves lifecycle, cancellation, semantic buffers and transport accounting together.
Opaque core driver identity prevents a later run on the same loaded executable
from importing an old branch tree. Shared capture admission and native completion
ownership remain in the neutral control API for every backend.

This guide defines the boundary between the portable language-model runtime
and an execution backend. It is intended for backend authors and maintainers of
`eredu`.

## Dependency direction

```text
eredu-core / eredu-checkpoint / eredu-nn / eredu-media
                           |
                     eredu-runtime ---- eredu-codec
                           |
                   eredu-architectures
                           |
                   eredu-backend-mlx
                           |
                         eredu
```

Backend-internal
fixtures and composition tests are crate-private unit tests; published crates
expose no test-support feature or fixture namespace. The facade does not
directly depend on `eredu-nn`, `eredu-checkpoint`, or `eredu-media`. Portable
media algorithms are enabled through narrowly forwarded image or audio
features; a concrete backend feature adds only native conversion. Backend feature diagnostics
name the active public spelling; published Cargo features are selectable API,
not a privacy mechanism.

The facade root exposes the `api` and `runtime` namespaces. `LoadedModel<B>`,
`PlannedModel<B, D>`, prepared-chat requests, and controlled generation sessions,
snapshots, and branches are the canonical application API for every backend.
Applications select MLX by importing `eredu_backend_mlx::MlxBackendFactory` and
passing it to the generic loading methods; inference supplies the backend type.
Explicit model annotations use `LoadedModel<MlxBackend<'static>>`. The facade
provides no wrappers or aliases that fix these generic types to MLX. Application
errors do not carry backend error type parameters. Loading, media preparation,
ordinary/speculative/observed generation, control, snapshots and branches retain
portable operation error variants with `eredu_core::BackendFailure` for native
failures. Lifecycle and local diagnostics use that same common failure type.
Token iteration and token-ID observation cross the facade boundary through
`TextGeneration` and `GeneratedToken`, preserving asynchronous execution and
completion ownership while translating errors. Realtime generation transactions,
frame execution and sampling replacement also expose portable failure types.
Backend implementation traits may retain concrete errors internally; application
callers do not need to name or match them.

The implementation crate exports composition-owned adapter factories at its
flat root and makes its reusable `backend` module tree public for backend
authors. Family composition and architecture-erased dispatch remain
crate-private. Raw native binding APIs retain their canonical `safemlx` paths.
Native facade integration tests use the same generic model and control APIs
as portable conformance tests, which use neutral traits and mock backends.

The facade does not re-export dependency-owned types. Architecture identities
and preparation plans come from `eredu-architectures`; artifact, execution,
generation, media, realtime, scheduler, and prompt-cache contracts come from
`eredu-core`; processed host media buffers come from `eredu-media`; tokenizer
contracts come from `eredu-text`; and runtime
infrastructure comes from `eredu-runtime`. This gives every public type one
canonical import path and prevents a facade release from committing to aliases
for operations it does not own. In particular, the facade exposes no
prompt-cache types because it exposes no prompt-cache operation.

Realtime applications prepare sources with `prepare_realtime_model` and select
MLX realization with `prepare_mlx_realtime`. Consuming `PreparedMlxRealtime::load`
returns categorical execution or a bound PCM speech session. Categorical clients
compose `PreparedRealtimeModel<M>`, `RealtimeSessionScheduler` and
`ReleasedRealtimeSession`; PCM clients use controlled advancement or the bound
session iterator. Runtime scheduling retains admission, transactions, completion
and host-publication policy for both protocols.
Realtime `resume` takes a mutable optional released-session slot. Failure leaves
the slot untouched and returns a portable `RealtimeSessionError`; success takes
the state. This preserves retry/disposal ownership without carrying native state
in the public error type.

Application targets depend on `eredu` for facade operations, their selected
backend for factories, and the neutral crates whose public values they
construct. Backend dependencies remain feature-gated. The MLX facade helpers
provide device-plan creation, process runtime configuration, allocator
telemetry, and diagnostic benchmarks. Rich allocator and residency telemetry
methods remain specialized on `LoadedModel<MlxBackend<'_>>`.
Session lifecycle is portable: every `TextGenerationBackend` implements explicit
`reset_session` and `synchronize_session` hooks, exposed by `ModelRuntime<B>` and
`LoadedModel<B>` as `reset()` and `synchronize()` with the same non-generic
`BackendFailure` on every backend. Its portable `BackendFailureKind` distinguishes
busy authority, invalid/poisoned sessions, identified resource exhaustion,
unsupported or invalid requests, I/O and other backend failures. The original
error is retained as a boxed `Send + Sync`
source, so applications can handle portable kinds and log the complete cause
chain without importing backend error types. Adapters classify from typed facts
and session state, never by parsing diagnostic strings; unknown failures remain
`BackendFailureKind::Other`. These kinds do not establish settlement or safe reuse. Reset first
synchronizes, then clears request state while preserving loaded parameters,
admission, placement and retained drafting target identity. Success establishes
fresh-session behavior for the next request. Synchronization preserves request
state and succeeds only when all session work has settled, submission authority
is idle, and the session remains healthy. MLX waits for both weight and execution
streams and checks its retained recovery/authority state; queue completion alone
does not rehabilitate a poisoned session. Neither operation uses an implicit
successful no-op default for backends.

Generic applications can drop a generation iterator after cancellation, call
`synchronize()` to observe settlement errors, and drop the model for eviction;
`reset()` establishes fresh state before reuse. Iterator destruction waits for
retained completions but cannot report errors. A synchronization/reset error is
not evidence of completion or safe reuse: native resources remain owned until
safe completion or teardown. Live detached submissions may be rejected as busy.
These operations do not promise an allocator-cache flush or immediate
process-wide memory reclamation, so native telemetry is not required for the
portable reuse/eviction protocol.
Backend adapters own native resources and outer type erasure, while shared
replicated-text execution and stateful lifecycle orchestration remain in the
neutral runtime.
`LocalLoadOptions` and `LocalInspectionOptions` contain only neutral
quantization, residency, and session-capability policy, while native
device-bound contexts are selected only by backend tooling. `BackendFailure`
retains operation context and the original typed source without exporting native
error variants in application signatures. Portable execution plans remain the application surface
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
Portable execution-plan factories use the same ordering for native resources.
The factory first translates the plan and selects the exact backend preparation
against the inspected artifact without creating a device, stream, queue, or
communication resource. The neutral selector validates the normalized request
once and returns its unforgeable admission; core builds the preparation plan
directly from that retained admission without asking the backend to derive the
policy again. Only the retained selected target may then be passed back to
the factory for native realization. Materialization consumes that retained
selection instead of repeating route selection after native resources exist.
Core carries the neutral preparation plan and backend selection in one opaque
`SelectedModelPreparation`; callers cannot re-pair either half before
`ModelLoadingBackend` consumes it.
Artifact preparation follows one immutable boundary in `eredu-architectures`:
format admission → normalized artifact declarations → family execution requirements
→ mechanism selection → exact source binding → typed construction.
`NormalizedArtifactPreparation` retains canonical parameter requirements and recipes,
metadata-only recipe catalogs, exact physical source provenance and resolutions,
and logical target/prediction/vision roles. Admission caches retain these declarations
without retaining their owning inspection cache. Projected prediction views share
physical artifact ownership and keep their exact source authorization.

`eredu-checkpoint::recipe::ArtifactCatalog` exposes enumerated headers and provenance
without payload acquisition. SafeTensors adapters resolve aliases, optional parameters
and quantization companions; GGUF adapters preserve native physical encodings and
canonical mappings. Family recipes remain in `eredu-architectures`. Cold inspection
uses these catalogs rather than constructing readable weight stores. Load limits,
residency policy, selected transformations and backend mechanisms belong to execution
preparation, outside the immutable artifact declarations.

Flash-Next's format-specific admission plans normalize into the same target declaration.
`TargetPreparationPlan` and `SelectedTargetPreparation` retain ordinary, routed, row and
prediction-state selection for both containers; conditional and partitioned composition
consume that common target. Compact table recipes retain either deferred bounded integer
reads or exact admitted GGUF integers. Scalar E4M3 table scaling remains independent of
block-scaled expert companions. Binding checks every required role's headers and provenance
before family literal acquisition and consumes retained parameter recipes.

A shared physical-artifact opener consumes admitted shard identities, resolutions and
mappings under the selected cache policy. Restricted role views share its source objects:
an embedded SafeTensors predictor and vision tower need no extra physical source, while
GGUF companions have independent artifact declarations. Backends receive the bound source
graph and retained selection without container parsing or alias rediscovery. Realtime Moshi
and codec construction use their separate preparation contracts.

External drafting follows the same two-stage boundary: the facade supplies the
architecture-inspected assistant to the factory's cold drafting-selection hook
before target realization. That hook establishes architecture-owned
target/assistant compatibility and selects the neutral speculative realization
against the retained target inspection before target payload or native-resource
creation. The retained result contains every exact
physical source, executable format, lowering descriptor, and reader-cache
mechanism setting consumed later by assistant materialization. Target or draft
streams therefore cannot exist when assistant selection fails.
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
likewise exchange portable host token frames through the facade. Runtime ingress
validates their exact batch and token geometry before a backend conversion
mechanism can create native tensors. Runtime prepublication waits for the exact
backend completion, performs host observation, and commits the model, schedule,
history, samplers, randomness, and output as one transition. Realtime selection
validates the exact loaded route's fail-closed mechanism and observation
capabilities before checkpoint payload materialization. A realtime route must
not advertise a named observation or intervention unless its architecture
execution exposes the corresponding seam.
Distributed inspection is rank-local: every rank participates in the same
production collective and point-to-point execution, each rank returns only the
globally named units it owns, and only the logits-owning rank returns
`model.logits`. `eredu-core::MODEL_LOGITS_OBSERVATION_PATH` is the canonical
owner of that semantic path; family composition, distributed pipelines, and
architecture-erased sessions must use it rather than defining backend-local
names. Inspection never performs an implicit cross-rank host gather.

Backend-neutral parity, distribution metrics, timing summaries, evidence, and
evaluation drivers live in `eredu-evaluation`. Concrete backend examples only
select execution contexts and supply generic mechanisms for neutral model and
codec artifact construction; they do not own checkpoint interpretation,
comparison thresholds, or reference policy. Backend-specific validation
inputs and reference-fixture generators live with the backend that consumes
them, so every published backend package contains the tooling required by its
examples; they do not live under the facade.

## Ownership boundary

Portable crates split tensor-independent ownership by responsibility:

- `eredu-checkpoint` owns canonical SafeTensors index parsing, exact
  index-to-shard-header validation, shard-path admission, and stable
  filesystem-backed content fingerprinting. Inspection and
  conversion tooling consume strict discovery, while neutral stores consume
  the same parsed and admitted catalog but validate each opened payload header
  exactly and lazily, so selective loads detect every mismatch in a local shard
  without reading remote-only shards. None reinterpret `weight_map`.
  `PreparedCheckpointSource` binds exact catalog, provenance, admitted shard
  identity snapshots, selections, and lease geometry without retaining a file
  descriptor per shard. After complete recipe and destination preflight, the
  requested shard is reopened and validated, and matching reads admit only the
  exact selected ranges; immutable cached bytes back every published lease;
- the canonical in-memory content identity value and logical-member combining
  algorithm, header inspection, the
  model-configuration resolver contract, tensor catalogs, preparation plans,
  backend mechanism-capability admission, and reusable inspection-report
  assembly live in `eredu-core`;
- portable host audio, image, and video processing lives in `eredu-media`;
- validated attention schedules, parallel topologies, model capabilities,
  resource requirements, execution plans, and telemetry schemas;
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
- distributed scopes, topology membership, consensus messages, and operation
  capability descriptions; and
- canonical parameter-binding plans, generic module binding, logical tensor
  placement, communication-manifest validation, task partitioning, report
  aggregation, and reusable session construction live in `eredu-runtime`.

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
fingerprints is portable compatibility policy. Architecture composition owns
target/assistant compatibility and selects assistant placement before native
construction. Backend factories receive those proven contracts, materialize
only the selected assistant tasks, and bind the selected native placement; they
decide neither tokenizer nor architecture compatibility.

`eredu-runtime` owns statically dispatched resident and bounded execution,
the singular declarative parameter-binding and logical placement plans,
generic module binding, residency, mutable state, exact completion, neutral
communication preparation, reusable session construction, and
generation-facing causal-model contracts. Backends retain native recipe
lowering, tensors, storage, streams, completions, groups, transfers, and
collective scheduling.

`eredu-architectures` owns model-family configuration, checkpoint contracts,
parameter topology, module construction, state geometry, parallel semantic
plans, and the complete embedding/layer/output lifecycle. Architecture code is
generic over `NeuralBackend` and passes backend-native tensor handles through
unchanged.

Portable preparation admission keeps architecture requirements, admitted
artifact facts, caller policy, and backend mechanism capabilities as separate
inputs. `eredu-core` intersects them once into an immutable retained admission
or a structured rejection before payload access and native resource creation.
Inspection and realized sessions assemble capability and readiness reports from
that same retained admission; a backend may discover support facts but does not
recompute the portable decision after resources exist.

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

Requirements also retain exact schema-admitted nonparameter source keys. The
architecture derives these from metadata and recipe-input roles, subtracting
every learned parameter and target or auxiliary recipe dependency. Requirement
builders reject overlapping declarations in either order. Native strict loading
consumes this retained set rather than recognizing checkpoint suffixes; every
other unconsumed source remains an error. Spark's ModelOpt calibration-only input
scales use this contract without allocating neural destinations or weakening
packed-weight and conversion-scale validation.

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

Parameter groups retain a typed `PartitionUnitAssignment`. `Distinct` balances
nonempty ranges, `Even` requires equal logical-unit counts, and
`Repeat { semantic_units }` repeats a complete semantic unit across adjacent ranks
when the tensor width exceeds that domain. Encoding expansion and chunk conversion
preserve this policy and the original semantic count; coarsened physical blocks
must still form equal whole-unit ranges. Companions follow their declared physical
placements, including replicated NVFP4 global scales. Muse declares even query-head
placement and repeated KV-head placement. Its cold state geometry and native
construction derive head counts from the same layout.

Each architecture-derived parameter requirement also retains exact physical
provenance: the admitted catalog key, canonical shard, physical tensor
identity, selected logical output, source encoding, and encoded byte extent.
Recipe output is an explicit derived-source encoding rather than a fabricated
SafeTensors container encoding. This distinguishes multiple outputs converted
from one physical tensor and prevents a materializer from rediscovering shard
membership or reinterpreting the source after selection. Validated parallel topology fields are private;
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

The runtime-owned `NormalizedLoadRequest` is the single portable cold-load
policy. `NormalizedLoadRequest::from_execution_plan` derives transformation,
ordinary and independent-bank residency, exact reader-cache limits, session
requirements, and drafting intent from one portable `ExecutionPlan`. Invalid
plan structure and policy produce structured portable errors before backend
capability selection or native work. Distributed plans supply one complete
parallel request whose topology must equal the plan. Diagnostics choices
request backend/process memory observations without invoking those mechanisms
during normalization. Reader-cache limits remain exact even for resident loads.
Its checked parallel subrequest atomically binds rank topology, wire,
positive invocation limits, and bounded completion; drafting capacity is also
positive by construction. A concrete adapter may pair that request with a
native resource token, but cannot add a second policy representation. The MLX
adapter validates its `DeviceAssignment` pairing and consumes the paired
request before neutral selection completes. Total cold branch selection lives
in `eredu-architectures::preparation_selection`; MLX supplies only a
side-effect-free `PreparationMechanismProvider` and consumes the resulting
opaque `SelectedExecution` through its typed materialization dispatcher.

`eredu-architectures::select_preparation` is the singular total cold selector.
It accepts that normalized request and a side-effect-free mechanism provider,
then performs admission, grouped-operation checks, processor and prediction
selection, ordinary or partitioned execution dispatch, communication
admission, and architecture production-route validation as one operation. Its
opaque `SelectedPreparation` covers replicated, routed, composite,
partitioned-dense, partitioned-routed, and partitioned-composite execution.
Concrete backends consume the selected branch through a typed static
dispatcher; they do not mirror the sum or rebuild any request.

Architecture inspection retains validation results with the immutable architecture
and the exact core admission token. Clones share the target projection and the
replicated, routed, or composite requirements, including expert topology, recipe
geometry, and source ownership proofs. A new admission or semantic enrichment
uses a separate set of results. Preparation retains both successful selections and
rejections for each normalized request and the actual neutral mechanism facts
consulted by that selection; changed backend facts require a new selection. These
results are local to the admitted artifact's lifetime, not a filesystem or global
model cache. A selected preparation retains the complete admitted artifact,
including any embedded predictor, separately from its target execution projection.
Prediction activation discovery uses that complete declaration; ordinary discovery
uses the target execution projection.
Target recipe construction sees only the admitted target source role; predictor
recipes use the complete architecture's separate catalog. Both views share the
same physical source ownership.
Retained architecture and composite source snapshots keep immutable declarations
without owning the enclosing selection cache, avoiding ownership cycles.

Automatic planning retains each exact candidate's admission and bounded geometry
through feedback and final selection. Loading reuses the architecture's selected
proofs. Prepared expert providers consume the validated catalog and route contract;
their native bank binding checks remain a separate validation of newly materialized
resources. Public constructors accepting raw plan/catalog parts still establish
those proofs themselves. Expert identities, parameter targets, companion names,
and layout discriminators use indexes, and selecting all members of a stacked
expert recipe traverses the stack once. Checkpoint resolution consumes the layout
selected by catalog validation instead of repeating discriminator validation.
Runtime-owned `AddressableBankTask` retains an immutable validated materialization
task and its source index. Expert members share that proof and validate only their
own recipe sources and geometry; they do not clone or validate a whole bank task
per member. Architecture projection indexes selected tasks once per projection.
Source-store binding checks provenance once per shared task for that handoff;
unchanged member recipes reuse their inferred metadata, while transformed recipes
require new inference.

SafeTensors shard-path admission belongs to `eredu-checkpoint`. Model-cache
snapshots admit payload symlinks into repository-local blobs and the enclosing
cache's shared `blobs` directory. The shared store does not authorize unrelated
cache files or other repositories; symlink targets must remain within an admitted
root after canonicalization, including links through the shared blob directory.
Index members remain relative paths without parent traversal.

SafeTensors admission resolves each distinct shard member once and retains one
checked header, exact index membership, file identity, and tensor catalog per
physical shard. Cloned admissions, core inspection, prepared stores, and bounded
payload-cache eviction share those results, including failed header admission.
Core projects tensor descriptors from this catalog instead of parsing headers.
Prepared metadata and provenance queries are in-memory snapshot lookups; new
leases and encoded batches must match that snapshot. File-change checks occur
when bytes are read or a cached payload is acquired, rather than on metadata
queries. A file changed after header admission is rejected at that read boundary.

Each immutable recipe catalog retains inference for complete recipes and shared
subrecipes. SafeTensors catalog/store clones share the cache; restricted views
have their own authorization scope. Mutable custom catalogs do not opt into this
contract. A source that supplies a cache promises stable metadata/provenance and
read batches/leases checked against that catalog. The same neutral storage can
retain success or failure text for a caller-defined, fixed metadata validator;
it never retains validator closures or native resources. MLX owns its constant
representation checks and reuses these results across preflight calls. Dynamic
device capabilities and validation of newly materialized resources remain
separate. Inference and validator results are initialized once under concurrency.
Binding preflight returns its validated owner/alias partition; selected binding
plans own that partition through materialization without reconstructing it.

Exact checkpoint content identity is demand-driven. `eredu-checkpoint` records
file identity, size and change metadata without retaining per-shard file handles;
when requested it hashes each admitted file in one sequential pass, checking
metadata before and after reading. Multi-member graphs of at least 8 MiB use up
to four host workers, bounded by available CPU parallelism, with one 1 MiB read
buffer and open file per worker. Logical member order and SHA-256 identity encoding
are independent of scheduling; failures cancel unscheduled work and all active
workers settle before the operation returns.
`eredu-core::DeferredArtifactIdentity` shares and caches that result or failure.
Model, external-assistant and Moshi source
preparation retain the pending identity without reading payloads for hashing.
Architecture-owned discovery declarations carry it through native construction;
`CaptureDeclarations` carries admission facts without content resolution.
`CaptureDiscovery` additionally resolves content-exact provenance on explicit
request. `CapturePlan::provenance` selects session-bound observations or content-
exact records; live score reads use session provenance. Facade requests retain
opaque session admission and active overlay identity in both modes. Intervention
admission and capture snapshot/fork validation independently require content-exact
identity. Ordinary generation and session-bound traces/captures leave it pending.
Backend adapters retain these neutral
declarations and never reconstruct source membership or fingerprint files.
Tensor loading reads selected ranges in execution order, so it is independent of
the optional whole-file hashing pass, which also covers headers and unused bytes.

The workspace optimizes the `sha2` dependency in development builds. Its
compression kernels otherwise spend substantial time in unoptimized code even
when CPU hash instructions are available. Eredu code retains ordinary debug
settings; the SHA-256 identity format, full-content reads and change detection
are unchanged. Downstream workspaces control their own Cargo profiles and can
apply the same `[profile.dev.package.sha2] opt-level = 3` override.
`cargo run -p eredu-core --example artifact_fingerprint -- FILE_OR_DIRECTORY [ITERATIONS]`
measures fresh deferred resolutions and memoized lookups. Compare profiles on the
same file and require identical artifact identities; throughput depends on build
configuration, hardware and filesystem caching.

For byte-preserving leading-axis joins, checkpoint recipes can instead compile
an exact encoded read batch. The batch retains admitted metadata and file
identities and groups reads by shard and offset. Adjacent file ranges use
vectored reads into disjoint destination slices, including reordered expert
ranges and separate parameter allocations. Physical reads are bounded to
64 MiB and 1024 buffers; consecutive chunks continue without another seek.
Short reads and interruptions advance within those slices without staging. The
batch owns no payload buffer or shard-cache lease. Prepared and
restricted source views preserve catalog and authorization checks for this path;
file identity is checked once before and after each shard's batch of reads.
Recipe inference reuses the immutable source's retained result, or uses batch
metadata for custom sources without an immutable catalog. Unsupported transforms,
partial selections, and source encodings retain the ordinary materializer.

The runtime owns a portable allocation-group budget of 1 GiB and 64 parameters;
an oversized single parameter stands alone. MLX module binding and resident-unit
loading retain each recipe's direct/fallback preparation decision and combine
compatible reads within those groups, opening and checking each participating
shard once per group. Aliases reuse their owner without another read.

On CPU and Metal, MLX materialization allocates the group's final contiguous
arrays first and supplies their writable bytes to the neutral read batches. The
safe native initializer owns exclusive, zero-initialized allocations until all
reads succeed, then publishes immutable arrays together. Error or unwind drops
the whole unfinished group. Native completion recovery retains initialized
inputs before any stream copy or submission. Apple unified memory lets the GPU
use that same allocation. Expert stacking and gate/up concatenation require no
per-expert native arrays or checkpoint-sized staging copy on this path. The
optimization follows each selected binding's recipe, so it neither changes
residency selection nor eagerly reads otherwise nonresident banks. Native
allocation and pointer access remain inside `safemlx`/`safemlx-sys`; checkpoint
code only receives bounded, disjoint mutable byte slices.

Checkpoint publication uses `Array::ensure_device_storage`, a lazy native
placement operation, rather than treating MLX's storage-sharing `copy` as a
host-to-device transfer. CPU and shared Metal backing remain shared. CUDA also
shares allocations already on the selected device, migratable managed memory,
and host-addressable storage on integrated or host-page-table-coherent devices.
Other host backing, including WSL's pinned system memory, is copied into the
selected GPU's device allocation. Placement requests exclude host and other-GPU
allocations from size-only cache reuse. Contiguous layouts retain their strides;
noncontiguous inputs are copied by logical indexing. Source leases, native
input arrays and destination arrays remain under the existing materialization
completion/recovery owner until submission is safely complete. Resident,
bounded-residency and controlled sessions consume the same placed parameters.

After selection, `PreparedModelSources` is the sole architecture-aware
SafeTensors/GGUF source factory for model loading. It opens every admitted
physical source once, retains exact resolutions and metadata, composes typed
companions in semantic-role order, and publishes explicit primary, complete,
target, and prediction-extension views. Restricted views share the same source
cache and fail closed for unauthorized keys. MLX begins only with these source
roles and performs native lease conversion, transforms, binding, and tensor
materialization.

Architecture-aware model inspection uses the same normalized request,
mechanism provider, and total selector. The neutral outcome owns report state
transitions and can retain the exact selection used to establish readiness.
Backend adapters add only their name, static hardware context, and native
diagnostics; inspection itself opens no payload and creates no native resource.

Moshi realtime selection likewise derives shared policy from
`NormalizedLoadRequest`. Runtime's `synthesize_realtime_capabilities` enumerates
every execution alternative, stably deduplicates exact lowering capabilities,
and shares state-component traversal with replicated-text synthesis. MLX and the
independent realtime backend report only collection-independent facility facts
and per-descriptor lowering and placement support; neither capability provider
walks architecture requirement collections. Its neutral prepared-source handoff contains the exact
source, content identity under the architecture-selected domain, retained
resolution and metadata validation, selected topology and architecture, and a
consistent lowering summary. MLX receives that completed handoff and adds only
groups, streams, caches, arrays, random state, completion, and final erasure.
`MlxPreparedRealtimeExecution` privately pairs that handoff with the exact
native rank/device token validated during cold selection. Realtime
materialization consumes this pair without accepting replacement load options;
it validates the retained device against the execution stream before payload
binding or construction. `MlxLoadRequest` holds one `NormalizedLoadRequest`
and optional native device assignment, not an independently mutable portable
mirror. Backend-author callers build portable policy directly on the normalized
request and wrap it once; the MLX adapter exposes only native pairing operations.
Prepared MLX models retain their actual native device and, for distributed
execution, the selected world handle. Session publication and every
backend-taking session operation validate this retained target before input
conversion, mutation, or submission. Device labels and equal rank/size values
are not identity proofs; another stream on the same device remains valid.
Realtime state creation and frame submission apply the same retained-context
check, and direct execution rejects a stream on a different device. Realtime
selection rejects explicitly requested prompt-cache persistence or drafting
when the realtime architecture cannot honor them; explicit observation
capabilities require matching state-retention support even without named
observation requests.
Core retains exact capability admission throughout a `ModelRuntime` lifetime
and rechecks it before generic submission, observation, or inspection. Exposing
an unrestricted mutable backend session invalidates its execution-plan target
proof, since callers can replace that session. Drafting selected for the old
target cannot attach after such access. Native MLX C event and transfer producers
reserve output wrappers before submission, so publishing a successful producer
result needs no new wrapper allocation. Native submission recovery separately
records accepted CPU tasks and GPU command-buffer work before publication can
fail. A preallocated owner retains graph, primitive, temporary-buffer, stream,
and callback state until every accepted frontier is terminal. An error is not
terminal evidence, and evaluation is never retried as a cleanup operation.
Backend submission scopes retain application resources and session authority
across returned errors, discarded errors, and unwinding. Recovery neither
terminates the process nor waits for native work in destructors. Unobservable
work stays owned and its affected execution resources remain unavailable.
Terminal proof and reclamation are distinct: bounded native scope progress
publishes completion without destroying arbitrary primitive or group owners.
Portable layered execution runs a strict dependency chain on the caller's
executor when its parameter policy explicitly permits deferred serial groups.
Resident policies retain every unit's parameter storage until the surrounding
forward completes, so these groups need no intermediate submit/event boundaries.
Unit acquisition, completion, abort and semantic group hooks remain in use.
Branched graphs, bounded policies and parallel execution retain independently
schedulable group executors and exact group submissions. The MLX realization
reports the generic resident policy fact; it contains no family equations or
graph-specific selection.
Ordinary owner-thread native operations reclaim terminal records outside the
registry lock; a departed owner thread cannot be replaced by a recycled thread
identifier. Rust recovery tries the runtime lock without waiting and retains
the node if it is busy. Events belong to the retained bundle, and terminal
handle destruction runs with housekeeping suppressed. Independent consumer
waits and host observations each have their own ticket; readiness requires all
such tickets to settle, not merely the original producer's completion.
Checkpoint materialization, quantization tiles, residency transfers, and bounded
execution units retain their own submission owners. Aborting a layerwise forward
does not drain events or whole streams. Terminal module, session, checkpoint,
and manager-lease owners are staged for ordinary unlocked reclamation, since
their destructors may acquire host locks or execute application code. Dense
forward cleanup is deferred from abort to an ordinary entry or owner reclamation.
Selected model materialization also reclaims an already-retired snapshot before
allocating the next model, so completed previous-session weights need not remain
alive until replacement session publication. Newly staged recursive owners wait
for a later ordinary entry; this is not a drain of unresolved native work.
Each completion poll keeps one nonblocking runtime guard across its native
observations. MLX realtime completion retains successfully validated readiness
across clones, so unrelated runtime-lock contention cannot turn completed work
back into pending work during semantic commit. New observation tickets invalidate
that proof; failure flags still prevent publication. Native resource-retirement
queries retain their nonblocking runtime guard even after publication is ready.
Arbitrary application resources retained by neural submissions
are staged separately after every native/consumer owner releases them: their
destructors do not run from polling, recovery Drop, or TLS teardown. The next
unlocked ordinary neural submission reclaims them, or callers explicitly use
`MlxNeuralBackend::reclaim_retired_resources`. This step may block in a caller's
destructor and is skipped under the runtime lock, during unwinding, and during
recursive reclamation. If no eligible reclamation occurs before owner-thread
exit, those resources remain retained.
If an application destructor panics during explicit reclamation, the remaining
detached resources are retained instead of invoking more destructors while
unwinding.
MLX prefetch, cache-I/O, and host-demotion worker handles request shutdown
without joining from recovery or teardown. Each worker owns its active inputs
until that operation returns. Prefetch and cache-I/O workers cancel queued work
and release its admission slots; the host-demotion worker drains queued inputs
after its sender disconnects. Host demotion reuses one native stream per device
for that worker's lifetime; MLX retains stream workers through process shutdown,
so allocating a stream per cache block would accumulate unbounded threads.
Portable worker APIs retain deterministic joining by default and
expose nonjoining shutdown as an explicit option. Final cache-state reclamation
removes its ephemeral files. Disk commits hold a weak state reference to avoid
ownership cycles and remove late ephemeral outputs when that state is gone.
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
The exact-task binder applies retained logical placement before validating
rank-local destination geometry. Task-declared locally transformed outputs
are already placed and are not sharded again.
Packed lowering supports only the final logical matrix axis; any
other declared packed axis is rejected during selection. Companion bindings
are format-exact: affine formats retain scale and affine bias, while MXFP4
retains scale only and rejects an invented bias.

The same typed construction boundary is available to an additional
`ReplicatedTextArchitecture` through the generic extension visitor. Such an
architecture supplies its normalized configuration and parameter topology,
then uses the ordinary requirements, selection, exact materialization tasks,
neutral session constructor, and lifecycle. Adding it does not require a
backend family registry case or a complete-model wrapper.

Architecture-owned typed dispatch admits replicated Llama/Mistral/Phi-4, Nanbeige, dense
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

Shared decoder parameter declarations retain the Q/K/V and gate/up component boundaries inside fused matrices. Tensor-parallel layouts partition each component independently, including its bias, and derive local head and intermediate dimensions from the retained semantic ranges. Native backends bind the ordinary linear, normalization, rotary and attention mechanisms.

The original Phi-4 uses the Llama-compatible execution class with retained `phi3` source identity and component-major fused QKV and gate/up projections. GGUF admission preserves the fused weights and their HF half-head rotary convention. Checkpoint schemas, tokenizer policy and discovery remain portable.

Nanbeige reuses the portable GQA/SwiGLU decoder blocks. `decoder::repeated`
lowers physical checkpoint blocks to logical invocations with exact source
aliases and an expanded attention schedule. The generic decoder's optional
post-block normalization expresses inter-pass RMSNorm. Each invocation owns
independent cache and residency state, so the ordinary dense drivers provide
per-block offloading, tensor sharding, pipeline cuts, combined TP/PP and captures.
Exact full-source aliases retain the physical tensor's encoding and geometry,
including native GGUF blocks, instead of treating encoded bytes as dense recipe
outputs. Architecture policy owns repetition, checkpoint geometry and cache
identity. Resident materialization currently creates a block for each logical
invocation; estimates count these replicas and all invocation caches, while
checkpoint storage describes the physical weights. Backends consume the usual
typed replicated or partitioned architecture and add no family dispatch.

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
sequencing. Prediction lanes request an independently advanceable state fork
through the same neutral mechanisms; paged backends copy the canonical block
catalog into an independent residency-session namespace instead of applying a
canonical-session checkpoint to unrelated cache ownership or weakening
cross-session checkpoint validation. A non-MLX backend supplies its own tensor, policy, state,
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
record contains its stable bank key, exact compact bindings, source and
selected byte geometry, and architecture-selected owner group, unit, parameter
path, distribution, and optional rank placement. The selected geometry includes any load-time
transform and is therefore the geometry used for capacity checks, acquisition,
and telemetry. A backend adapter consumes these generic records; it does not
receive the architecture catalog or interpret family recipe policy.

Runtime-owned `AddressableBankFootprint` supplies the same checked source and
selected payload totals to routed preflight sizing and retained member
construction. Every declared parameter contributes independently, including
parameters that read the same physical source. Overflow errors identify the bank
member. Architecture code admits transform roles and formats; runtime accounts
for the resulting payload without selecting policy, scratch or state geometry.

`ReplicatedTextMaterializationTask::transform_output_companions` projects generated
scale and affine-bias geometry from the selected logical shape and quantization
group size. Ordinary preparation and member-local bank preparation use this same
projection. Callers retain canonical output naming and select the logical subset;
runtime checks group alignment and derives parameter ownership. Complete output
role validation uses the task's ordinary companion attachment contract. Stored
companions retain their declared source and output descriptions.

Runtime-owned `AddressableBankTaskCache` retains one immutable exact task index
and shares validated cold tasks by canonical output, selected member shape and
local companion bindings. Architecture callbacks declare missing generated
outputs; they cannot substitute source authority, owner or lowering. Existing
output declarations remain authoritative, and failed preparation is not cached.
Sharing task metadata does not combine physical storage, parameter charges,
leases or mutable state across independently addressed bank members.

`place_addressable_bank_members` owns cold rank-local member projection. It applies
the retained non-member placements through the shared binding mechanism, infers
local recipe metadata and recomputes selected executable bytes. Global bank keys,
parameter task roots, companion bindings and rank ownership remain attached to
each independent member. The backend consumes these projected records and
preserves typed projection causes through its ordinary error boundary; it does
not reconstruct local parameter geometry.

Each `AddressableBankBindingPlan` retains canonical destinations for its local
bindings and generated companions, together with a checked source/executable
footprint. Source accounting uses the final prepared binding recipes, including
checkpoint-format lowering, rather than the earlier admitted member byte count.
Backend entry construction consumes these destinations and totals directly;
parameter editing uses the same retained logical destinations as materialization.

Native bank construction retains `AddressableBindingTransform` directly from
these plans. Transformation grouping compares the complete selected encoding and
companion precision, and materialization consumes those same values. Backend
code supplies the native converter without maintaining a second transformation
description or reselecting precision.

`addressable_projection_bytes` supplies the common packed payload arithmetic for
selected member admission, backend preflight and bounded transformation sizing.
It accounts for the declared companion precision, affine biases and MXFP4 scale
encoding. Selected tasks retain source-format, packed-axis and group-alignment
admission; the arithmetic helper does not imply native transformation support.

`validate_addressable_bank_selection` checks exact ownership-key coverage and
projects executable byte totals from retained bindings and transformation choices
before native transformation. Direct bindings retain their prepared byte extents;
transformed bindings use source metadata and shared packed geometry. Typed failures
identify missing ownership, missing extents, inconsistent totals and original
recipe or geometry causes. Native binding preflight and combined residency-budget
admission remain separate checks in construction.

`prepare_bank_parameter_members` joins final physical binding recipes to the
retained canonical destinations for parameter inspection and editing. It infers
geometry against the selected post-transformation source, preserves global member
identities and independent charges, and reports missing destinations or recipe
failures with the member and local binding. The backend supplies final binding
slices and source authority, then retains the neutral records alongside its
native cache. Metadata preparation neither acquires payloads nor grants editing
authority.

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
the facade boundary. Reports retain the complete deterministic list of selected
transformation formats; a mixed bank is never collapsed to an absent or
arbitrary bank-wide format. They also retain the exact selected placement for
every bank key, including owner group, unit, parameter path, distribution, and
rank. Production construction requires the selected-member handoff; raw entry
constructors are not a public route around architecture selection. Resident
routed execution reports no addressable bank.

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
`eredu-runtime::PipelinePartitionArchitecture` declares architecture-owned ingress,
optional local state and output geometry. The runtime owns the generic
`PipelinePartitionExecutor`, ordinary per-unit strategy, receive allocator contract,
and exact component-pipeline binding checks; codec and language-model families
implement these contracts without depending on each other. Routed provider
selection and family-specific unit strategies remain in `eredu-architectures`.
One `PipelinePartitionExecutor` retains the selected
typed per-unit strategy and context across contiguous local groups,
so ordinary and provider-backed routed units share the same local-address,
boundary, scheduling, publication, and state lifecycle rather than selecting
a backend pipeline implementation. Architecture hooks declare the typed schema
for each directed route, resolve its invocation-dependent extents, and declare
the final publication shape independently of the input sequence. This permits
different channel orders and rates at successive boundaries. The executor
checks resolved identity, roles and scalar kinds against the typed declaration;
cold component binding checks compare every route with its architecture-authored
wire schema. Receive allocation and final publication use these retained
contracts. Fixed-width transformer and perception implementations use the
ordinary batch/sequence defaults.

A prepared routed partition carries one
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
Partition admission also retains the complete exact physical task selection
before any checkpoint payload or backend resource exists. Architecture-owned
parameter topology later attaches its atomic output companions and projects
that immutable selection onto the admitted rank ownership; it cannot reselect
sources, encodings, executable formats, recipes, or lowering from a module
walk. The prepared partition carries the resulting rank-local tasks into the
generic materializer together with the exact placement and communication
authority.
Composite partition admission also derives the complete TP-local mutable-state
layout from the normalized architecture schedule and selected rank, then
projects the PP-owned range before architecture or state construction. The
family partition geometry produced during typed construction must equal this
cold selected state; it is a validation result and cannot fill in or replace a
missing admission-time state plan.
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

`eredu-nn::GroupedUnitObserver` describes the sparse activated-unit boundary
inside gated-product and ReLU-squared banks. Its borrowed `GroupedUnitBatch`
contains only actually selected routes: local group IDs, sorted-to-original
selection indices, chunk-relative token indices, original coefficients and the
chunk's offset in the operator request. Architecture/provider composition owns
the mapping to global expert, invocation and sequence identities; a native
operator cannot infer it from local storage. Original observation, intervention
and effective observation precede the actual down projection. MLX validates
replacement shape and dtype before that projection and preserves observer
failures through the neutral error source. Effective units precede any selected
input quantization and must not be described as transformed multiplication inputs.
An absent observer follows the ordinary path without extra tensor evaluation,
retention or materialization. Borrowing a batch does not admit capture copies or
transport; those remain runtime responsibilities. Mechanisms without the hook
return `GroupedUnitError::Unavailable` when one is requested. This mechanism
alone does not advertise loaded public expert-neuron capture: bounded admission,
architecture discovery and committed distributed delivery remain required.

`eredu-runtime::RoutedUnitObserver` carries the provider-owned interpretation of
those rows. `RoutedUnitBatch` borrows the original route tensor before compact-bank
remapping, the retained source-to-global expert map, and the provider chunk offset.
Resident and addressable providers pass the same observer to gated/ReLU-squared
operators and their tensor-parallel variants. Compact native group IDs never
become global expert IDs. Expert exchange borrows existing receive counts and
original route tags to identify the sending peer, token row and top-k slot,
including idle peers and duplicate selections. These token rows are not prediction
ordinals. No expanded activation tensor or additional host route vector is needed.

`ActivationObserver::routed_unit_observer` lends one invocation's observer to the
canonical provider call. Shared decoder adapters use `ObservedExpertProvider`;
custom routed/shared equations attach it with `with_routed_unit_observer` while
retaining their architecture-owned combined-output event. Two observer owners
are rejected before provider execution. Callback failures retain their original
neutral cause across provider and neural-adapter error formatting. The observer
remains scoped to the actual chunk computation and bank lease; it does not extend
source residency or authorize retention. Selected-linear operations have no
activation-to-down seam and reject an attached grouped-unit observer. Public
sparse intervention uses the shared runtime driver described below; EP completion
and delivery policy remain separate from ordinary selected-unit receipts.

`ArchitectureDescriptor::routed_components` separately declares sparse expert/unit
topology. `RoutedComponentId` keeps logical invocation, checkpoint-global expert
and per-expert unit distinct; token rows and duplicate route slots describe
participation in that identity. Architecture discovery projects this topology
from the same gated/ReLU-squared bank specifications used for construction,
including packed leading expert axes, independently named banks, fused gate/value
row offsets, exact activation policy and ordinary affine biases. No checkpoint
name interpretation is delegated to the facade or backend. An absent declaration
or residual scale means missing descriptive coverage, not measured zero.

The neutral `RoutedComponentGroup` selection helpers join declared coordinates to
`ParameterDiscovery`, validate exact effective rank/shape, and return ordinary
unsqueezed `ParameterRegion`s together with actual loaded alias and capability
facts. The existing query/overlay admission owns identity, dtype, storage and copy
authority. A topology join never upgrades unavailable operations or authorizes
native work. Sparse groups are separate from dense `components`.

Sparse observation points declare `ObservationValueType::RoutedUnits` with exact
provider invocation, global bank geometry and original token/route/unit axes.
Their `RoutedUnits` capture transform returns route-bearing host records rather
than an expert-dense tensor. The core owns those records and exact receipt checks;
runtime borrows the existing provider hooks, reserves a complete invocation before
the first chunk, and accumulates records under the ordinary capture transaction.
Contiguous source-token coverage, unique original token/route slots, global expert
bounds and unit geometry must all agree before committed delivery. A native
collector maps pre-compaction source groups, copies bounded route metadata and
selected unit rows, and retains no capture tensor beyond its callback.
`ObservationMechanisms::routed_unit_tensors` reports this separate collector fact.
Public partitioned sparse delivery uses separate retained ownership, source votes
and global receipts through the same capture session, as described below.
Ordinary route receipts alone cannot authorize exchanged or partitioned evidence.

Sparse mutable points add `RoutedUnitInterventionPoint` with a virtual global
`[token, component]` axis (`component = expert * units_per_expert + unit`). Core
owns exact declarations, participation coordinates and fixed-size progress
receipts. Runtime validates and lowers Zero, Scale, Mask, MaskComponents, Replace
and Add onto current participating values, preserves list order, and reserves
full-invocation native/host work before collecting any route metadata. It gathers
only addressed elements, calls the ordinary typed activation driver and scatters
them into the actual value consumed by the down projection. Backend primitives
copy native route coordinates and perform indexed gather/scatter; they do not
interpret global component identities or dispatch public actions. No virtual
expert-dense activation is allocated. `InterventionMechanisms::routed_units` is
an independent native fact, combined with actual loaded hook/capture support.
A contiguous completed-token receipt is required for success; `Unmatched` means
no participating value matches the address. Original/effective sparse captures supply
attributed evidence; dense Preview/Summary operation evidence lacks participation
coordinates and is rejected at sparse targets.

`RoutedComponentCoordinateMap` keeps selected expert ownership and within-expert
scalar placement as separate checked axes. Architecture-owned
`derive_routed_component_coordinates` takes the selected `RoutedGroupedPlan` plus
retained write layouts: stored independent matrices do not confer execution
ownership. Packed expert axes and scalar TP columns are validated separately;
the existing semantic-unit proof expands packed columns without guessing an
encoding factor. Independent matrices must agree on their bank's scalar map.
These coordinates describe placement, not invocation admission or completion.
The expert map follows the selected realization's owner-local order; it is not
the physical row order of a packed parameter or the sorted order of routed values.
Actual routed rows carry their checkpoint-global expert identity independently.

Prepared routed banks retain per-invocation coordinates derived from the original
selected grouped specification, its localized construction specification and the
retained physical write layouts. This derivation checks local expert and scalar
extents against the construction that will actually execute. The shared resident
and addressable partition provider factories attach these maps before native
provider chunking; EP origin adapters preserve them across exchange. Backends
receive the prepared providers and do not reconstruct scalar placement. These
bank coordinates describe storage; pipeline invocation ownership remains with
the partition execution plan.

`with_partition_unit_observer` lends the scalar map through provider chunks and
expert exchange, preserving original source peer/token/slot tags and actual unit
values. The portable `lower_partition_routed_intervention` maps global edit plans
onto those local columns and checks owned experts and distinct source coordinates.
It does not establish global route coverage or authorize publication. Ordinary
capture/intervention owners reject partition-tagged batches. Committed sparse
producer admission, quota reservation, collective agreement and receipt delivery
still require integration into the existing distributed capture owner.

The existing runtime partition receipt protocol admits sparse declarations
through `PartitionCaptureReceiptPlan::new_routed`. Core's
`RoutedUnitCaptureOwnership` carries separate expert/unit placement and the
authoritative logical source peer; it does not confer native execution authority.
Runtime validates complete, disjoint expert-by-selected-unit coverage, retains
local storage order in the receipt identity, and checks each received route against
that ownership. Tensor-parallel fragments must agree on expert and coefficient
bits, and every selected token/slot/unit must arrive exactly once. Native chunk
ranges remain in `RoutedUnitCaptureProvenance` on each partition contribution;
the assembled logical payload does not fabricate a global native chunk sequence.
Delivery reservations include route accumulation and provenance copies before
producer work. This reuses the existing bounded exchange and all-rank failure
agreement. Connecting live observer admission and completed native invocations to
these receipts remains necessary before advertising public partitioned support.

Core owns the checked borrowed `capture::RoutedUnitOrigins` coordinate view.
Runtime's provider adapter preserves its existing neural error contract and lends
the same tags without copying them. `PartitionRoutedUnitCaptureSource` supplies
actual provider columns and exchange tags; `PartitionRoutedUnitCaptureRequest`
separately supplies retained ownership, global bank geometry, original source-token
extent and the selected global fragment. Neither object is native submission
authority. Runtime's `RoutedUnitBatch::partition_capture_source` joins existing
borrowed provider inputs without retaining arrays or reading their values.

The MLX collector checks actual columns, native route width, source extent and
exchange-tag coverage before copies. It maps sorted native routes back to original
peer/token/slot coordinates, validates owned experts, and gathers selected global
units from actual local columns. Other source peers contribute no exported rows.
Its estimator covers worst-case incoming routes from all peers and native chunk
provenance for the entire invocation, even when the selected token set is small.
Contiguous cold unit ranges are checked without expansion. This is a native
collection mechanism; live sparse admission, source completion and committed
delivery still belong to the existing partition observer/session integration.

`CaptureSession::prepare_partition_routed_capture` compiles those sparse
declarations through the same private preparation driver as dense partition
captures. It retains the original plan and run identities, prepays every producer
and receiver, and spends local fragment credits before allocating accumulators.
`begin_partition_routed_capture` pins the actual invocation's native row count,
dtype, scalar map and exchange topology. Each `observe_partition_routed_units`
call advances a contiguous native chunk; `finish_partition_routed_capture`
rejects missing chunks or provider failure. Zero-row EP owners must explicitly
begin and finish. Completed fragments use ordinary receipt exchange and staged
transaction commit, and failed work never refunds the parent ledger.

The shared `RoutedUnitObserver` invocation lifecycle encloses actual local provider
work before reverse route exchange or downstream tensor reduction. Architecture
adapters enter this scope for resident, host-layerwise and disk-streamed banks;
the EP adapter also enters it for zero-row owners. Begin borrows the actual input
and available prepared coordinates. Finish runs on success, preparation failure
and provider failure. Nested bank adapters preserve the active scope instead of
repeating callbacks. Native composition forwards the same borrowed arrays, maps
and origin tags without evaluating or retaining component values. The first local
error remains the source when the final callback also rejects the invocation.
Nemotron-H's partition adapter forwards its routed ReLU² unit observations
through this driver, including the TP execution path.

These callbacks describe local execution; they do not certify native completion
or committed capture delivery. The shared `PartitionCaptureObserver` consumes
architecture-supplied `PartitionRoutedCapturePlacement`: distinct producers plus
all actual provider members, their expert/unit ownership, original source peers
and input widths. It validates invocation identity and original/effective timing
against the retained admission. Source preparation reserves the maximum received
input rows on every member, including idle owners and nonexporting replicas, in
the ordinary selection digest. Actual input rows, width, dtype, optional scalar
map and origin topology pass a prepaid group vote before native dependency work.
The existing backend source-completion method retains its completion guarantees.

Every selected original/effective record consumes its source and final votes in
plan order. An earlier error prevents later native source work. Remaining votes
are attempted while transport is active; a native source failure fences its owner
and requires native settlement or teardown instead of further submission.
Finish validates complete sparse chunks and settles the
prepaid invocation-group vote before reverse exchange or downstream reduction.
Receipt delivery remains provisional until the same whole-forward transaction
commits. Errors and abort do not refund the ledger; execution and completion
retain their existing owners.

`ComponentPartitionLayouts` retains sparse placement from the selected bank
and logical invocation. `ExpertRealizationPlan` retains the same ownership policy
used by bank construction; cold rank projection uses that policy without creating
local bank specifications. Scalar columns come from the actual physical write
layout. Routed child operations inherit their declared enclosing MoE invocation's
parameter ownership, independently of checkpoint sharing. Each point retains
exact global expert/route/unit geometry, input width, publication source peer and
all executing source members. Identical coordinate replicas select one producer;
idle received owners remain members. Original/effective points retain distinct
timing while using the same invocation coordinates. Projection checks schedule,
producer and fragment limits before runtime admission. Current balanced expert
construction requires at least one expert per EP rank; placement preserves that
existing rejection. Loaded capture additionally requires the actual sparse hook
fact and exact native subgroup support. Runtime still owns reservations, receipts,
completion and publication. Support for world participation waves and sparse
interventions is conditional on the selected mechanism facts.

DeepSeek V3's layered entry points emit internal hooks while the traversal owns
the unit input/output observations. The standalone block entry point supplies
those outer observations itself. This keeps one intervention at each boundary
when the same equations run through an expert-parallel provider. The layerwise
runtime exposes an architecture failure as its error source, retaining nested
observer causes.

V3's resident and paged latent-attention paths share the architecture-owned
post-aggregation channel boundary and existing projection-input observer. The
dense-only and routed-capable block adapters share normalized inputs, scalar
SwiGLU units, write vectors and residual sequencing through
`ComponentInstrumentation`; the existing `compressed_attention.output` identity
remains intact. Compressed replicated traversal forwards its observer through a
family callback, just as fixed-state traversal does. Native backends gain no V3
equations or policy. These hooks alone do not establish complete discovery or
parallel support: sparse tensor-parallel block hooks and prediction internals
remain required integration. The target-only mixed V3 partition adapter emits
embedding observations only on the input owner, carries the effective embedding
in its typed boundary, and emits normalization/projection/readout observations only
on the output owner. Serial and vocabulary-parallel readout share the same driver;
All-dense TP unit hooks reduce attention and FFN writes through the ordinary
collectives before observing complete writes. Sparse TP unit coverage remains
unverified. Local public native capture,
parameter-edit and controlled snapshot tests cover direct/low-rank query variants
and all three weight-residency modes.

All-dense prediction-free V3 partitions use architecture-owned direct
construction from the retained selection. The constructor derives the target unit
range, compressed state, static ownership, and source/target layouts for transformed
weights. It creates no expert bank and requests no grouped mechanism. V3 static
modules consume explicit selected embedding/head formats; published global FP8
layer defaults still leave those statics dense. V3/V4 format expansion uses the
shared runtime mechanism, retaining companion-to-primary metadata and independent
fused FP8 row blocks rather than duplicating packing arithmetic.

V3 query rotation receives `[batch, heads, sequence, features]`, following the
neutral rotary operator's penultimate sequence-axis contract. This fixes a prior
equation error in both ordinary and partitioned execution. The V3 cache fingerprint
includes MLA equation revision 2 so earlier states cannot be silently reused.
Native CPU Ring verifies all-dense TP, PP and combined TP/PP across resident,
host-layerwise and disk-streamed SafeTensors and GGUF weights, including the GGUF
split KV-B layout, public capture, masks, queries, coordinated edits and rollback.
All nine topology/residency cases also pass SafeTensors load-time affine
4-bit/group-32 transforms, exercising source/target layouts and packed companion
bindings in actual native execution. Family equations belong to the architecture layer.

Component descriptions declare ordered effective input projections to a read. Core owns this neutral relationship; architecture
discovery supplies V3's query bottleneck and the normalized latent part of KV-A.
The shared rotary key is a separate direct dependency and does not acquire the
latent normalization. `eredu-nn::LowRankProjection` supplies a normalized-rank
callback while keeping one projection implementation; architecture instrumentation
uses it to observe and edit the value consumed by the second matrix. The current
KV latent value is observed before cache insertion. Descriptions do not promise
capture of existing cache-history rows, nor replace measured normalization or
selected input-quantization facts with a fixed matrix product.

Retained component placement also covers intermediate read stages. The architecture
joins each logical invocation to the stage weight's output-row placement, intersects
its declared row interval, and registers original/effective observations on that
axis. Shared source aliases do not transfer pipeline invocation ownership. Packed
input columns do not change output-row identities; short FP8 row chunks and indexed
row permutations retain their actual coordinates. Column-sharded partial products
cannot be advertised as complete stage values without an explicit output contract.

Native compressed-latent snapshot copies belong to the backend cache mechanism.
Resident copies compact the logical latent and rotary arrays into independent
storage; continuation estimates include chunk growth separately. Paged copies
duplicate the tail and the model-wide catalog, then bind each copied layer to that
catalog. They preserve the ordinary neutral snapshot admission, completion and
cumulative copy accounting through the shared controlled driver.

The architecture-owned `RoutedPlusShared` equation exposes shared-expert gated
units and the actual down-projection input through its ordinary component
instrumentation. Both local and TP execution use the same shared product policy;
the TP caller retains responsibility for completing the write reduction.
Discovery joins V3 gate/value rows and write columns to the shared-expert child
node. The child's output is a constituent of the complete sparse residual write,
so a reconstruction includes the whole sparse write once or expands all of its
branches; it never adds that whole write and its shared constituent independently.

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

Qwen hybrid SafeTensors catalog admission distinguishes the official namespace
from MLX-VLM's `language_model.model.*` and `vision_tower.*` layout. The
architecture owns exact aliases, convolution axis conversions, and the inverse
of MLX-VLM's normalization-weight offset. These are ordinary bounded parameter
recipes consumed by all execution mechanisms. Canonical parameter geometry
remains independent of the selected physical axis order. Converted artifacts
with no MTP tensors normalize their retained prediction depth to zero before
capability reporting and execution selection; partial drafts still fail strict
validation. Official SafeTensors retain their declared-MTP requirements, and
GGUF retains its separate canonical conversion recipes. Qwen embedding
construction consumes the selected per-parameter format, including quantization
companions, rather than relying on a checkpoint-wide quantization default.
The CLI chooses embedded drafting from the admitted architecture's prediction
extension and reuses that inspection for loading; it does not infer family
capabilities from raw JSON configuration fields.

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
same unambiguous key. The architecture resolver and facade retain the already
resolved `ModelKind`; backend model materialization stores only the selected
architecture-erased executable and does not reconstruct or pair family identity
from the effective type. Architecture-owned typed partition dispatch
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
single `eredu-runtime` binding-plan vocabulary records the stable target,
expected shape, dtype and bytes, direct or derived source, explicit source
sharing, overlays, exclusions, and rank ownership. Its generic
`ParameterBackend` orchestration preflights every recipe and immutable/mutable
module traversal before payload or native work, completes all materialization,
then validates and publishes the whole binding atomically. Layerwise and
addressable-bank executors reuse the same declarations while retaining their
backend-owned leases, storage, movement, and scheduling. A
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

Qwen hybrid recipes resolve admitted checkpoint aliases and assemble weights and
FP8 scale companions through the same construction for whole banks and individual
experts. Checkpoint inverse scales become the architecture's declared grouped
scale parameters; derived scale companions retain floating-point storage rather
than inheriting the weight's FP8 format. MLX represents FP8 values and exponent
scales as byte arrays while preserving their source encoding and byte accounting.
Its grouped adapter supports floating-point and UE8M0 scales with 128-by-128 blocks,
and its capability report applies that geometry to both direct and derived FP8
weights. These representation rules remain backend mechanisms; checkpoint aliases
and companion identities remain architecture policy.

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
MLX's unloaded floating parameter handles admit F16, BF16, and F32 source
storage through the neutral binding target's explicit source-dtype allowance.
Materialization preserves the selected recipe's dtype, bytes, and values;
the placeholder's initial F32 dtype does not request a conversion. Packed and
integer parameter handles retain exact dtype matching. The neutral binder
continues to validate exact shapes, selected sources, and declared dtype
allowances without inferring native storage flexibility.
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
completion, and resource retention. Expert exchange admits the retained native
compute-precision envelope independently of pipeline-wire precision, plus exact
integer routing tags. Production routed architectures bind this
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

Speculative verification also carries a positive, caller-selected completion
deadline and cancellation disposition in the neutral scheduler options before
any target submission. The scheduler resolves an incomplete verification when
that deadline expires instead of polling forever. A backend completion must
then either finish native cancellation or transfer the completion and every
live retained resource to a quarantine owner; only exact completion permits
observation, cache commit, or publication. MLX uses a thread-affine quarantine
because its events do not expose native cancellation. Terminal native evidence
permits reclamation; thread-runtime teardown does not wait and permanently
retains any owner for which that evidence is still unavailable.

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
execution objects. The architecture layer also owns the exhaustive
post-materialization topology and mutable lane-state variants; MLX implements
one generic materializer for arbitrary prepared modules, pooling caches,
sequential caches, and complete state layouts instead of repeating a
family-shaped construction match. Its physical sources are the exact admitted sources
not claimed by the projected target schema, rather than a backend prefix
filter. MLX materializes those units against the target's selected local
geometry carried by the handoff and invokes them through the neutral prediction-target operation
hook. This does not make every prediction family or topology supported:
extensions without a dedicated typed composition fail before target payload
construction. An excluded prediction selection cannot fall back to a complete
or pipeline target shell.

Conditional Qwen hybrid prediction units include their routed expert banks in
the exact extension parameter tasks. The generic materializer populates those
resident banks, and prediction uses the neutral resident expert provider
independently of the target's expert cache. Fused QKV and gate/up placement
maps FP8 scale companions into block-row coordinates while retaining logical
weight-row coordinates for the weights themselves.

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
allowlist. The runtime-owned normalized load request carries
`QuantizationRequest`, whose variants describe load-time transforms only; the
MLX request is a thin native-device adapter over that request. Checkpoint
storage encodings such as native GGUF blocks remain internal to artifact
inspection and materialization.

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

The neutral `eredu-architectures::muse` module owns the shared Glimmer/Spark
configuration, checkpoint recipes, parameter topology, media preparation, state
geometry and construction. A source-free `muse::component_definition` validates
canonical parameter ownership, logical paths, independent state slots, transport
and partition rules before native allocation. Live models retain this definition;
graph, address, parameter, state and transport reports project it. Cold partition
preparation, inspection and assistant target placement use the same declaration.
Vision units retain no mutable state; decoder units reference their KV slots, and
waveform units reference separate histories after the decoder slots, including
stateless gaps required by the declared audio schedule.
Muse text arithmetic borrows the shared `decoder::PartitionStaticModules` resource
aggregate. Ordinary, parallel, assistant and observed readout use the same decoder
resource views for normalization, vocabulary projection and canonical visitation;
Muse supplies input normalization, stage rounding and logit transforms. Source-free
canonical bindings select optional text owners before native allocation through
`ComponentDefinition::select_complete_pinned_roles`. This runtime projection rejects
foreign, duplicate or incomplete module groups, resolves tied aliases, and leaves
sibling component resources independent. Selected untied projection requires
its declared head even when an input table is retained.
Muse vision groups declare semantic block owners and independently materialized
patch/position, static-normalization, adapter and projection resources. Full,
static and per-block parameter reports project these identities. Cold
`VisionStaticSelection` resolves canonical bindings through runtime
`select_pinned_groups`; each group includes its encoded companions. `VisionStatic::from_selection`
allocates only the retained modules, and ingress, adaptation and
readout validate their required resources before equations or observation hooks.
The current vision ownership retains patch/position and static norms on every
vision owner, and retains the language adapter and projection on the final owner.
Request-local permutation and rotary tensors keep the same completion lifetime.
Muse target and transform-source preparation construct the authoritative
`ArchitecturePartition` from the component definition before native construction.
They validate selected state ownership and dense/routed local geometry there.
Native construction consumes that partition's shared `ComponentSelection` and
canonical text/vision bindings; shared preparation retains the same authoritative
partition. Complete and selected Muse models use one native construction path.
Selected models allocate only owned pinned resources and acquire units through the
runtime selection's global addresses. Receiver ingress retains token roots and
request activity without unowned embedding lookups or vision projections; actual
hidden values enter through the declared transport boundary. Vision activity is
independent of locally retained vision equation state, and tied output storage
does not authorize token ingress on a decoder receiver. Pinned recipe destinations
follow the actually retained modules. Static and unit recipe groups resolve only
their participating expert banks, so restricted sources require no neighboring
layer or bank payloads. Decoder KV and waveform histories retain independent
global slots and explicit stateless gaps across pipeline cuts.
Distinct registered identities select typed attention,
normalization, residual, feed-forward and media policies. The backend binds the
retained declarations through ordinary native mechanisms and selects no family
semantics. Spark preserves its exact released attention schedule and FP32
compressed-expert/residual equations. An expert operation exposes complete
unweighted outputs so TP reduction precedes nonlinear per-expert normalization;
weighting and accumulation follow it. `muse::expert_realization_plan` consumes
configuration, validated optional local geometry and TP/EP coordinates. Replicated
requirements and partition preparation share its canonical bank owners and local
matrix dimensions; expert selection and transform-source inspection require no
native model or parameter allocation.

Vision, waveform and decoder units use the same resident, host-layerwise and
disk-streamed providers. The waveform pyramid retains convolution and incomplete
input carry. Encoder products are prepared once and retained across controlled
decoder prefixes. State plans explicitly map decoder KV and audio histories to
stage-local ownership. Media-to-decoder and decoder pipeline boundaries carry the
same retained geometry, positions and completion obligations as ordinary execution.

`MediaCompanionSources` selects independent vision/audio directories in normalized
load requests and execution plans. Muse admits each physical source once, validates
BF16 component schemas, decoder/token/processor geometry and provenance, and
restricts source authority to the selected roles. Indexed media sources need only
the complete members containing those roles. The CLI exposes both roles, retains
them through overrides/replay/benchmark candidates and requires matching companion
selection for a cached plan. Missing components reject media preparation with a
typed missing-companion error while text loading remains available.

Muse assistants retain one body/head parameter layout and `LayerwisePolicy`:
12 released DSpark units and 10 DFlash2 units. Context, blocks, final norm,
vocabulary and auxiliary tables have explicit acquisition owners. DFlash2 and
Glimmer share the normalized DFlash body; DSpark reuses Muse decoder blocks.
Prepared sources retain checkpoint identity, exact recipes, selected encodings,
tokenizer authority and target compatibility before native materialization.
Pure TP DSpark may shard its body and replicate scoring heads; other selected
placements retain explicitly accounted complete replicas. DFlash2's assistant
architecture marker takes precedence over its `qwen3` model type.

External target invocation admission proves capture, verification and static
operation geometry against the retained target selection and communication group.
It checks producer precision, ownership, sequence limits and exact completion
before assistant payload acquisition. Capture publication and proposal preparation
use the shared transaction: tentative values stay separate from committed state,
peer failures agree before publication, and aborted work publishes no output or
candidate state. Borrowed target groups retain exact native-world authority.

The common speculative scheduler owns sampling, verification, accepted-prefix
commitment, rejected-suffix rollback, cancellation and output. Muse supplies
architecture equations and typed state. Snapshots preserve projected assistant
context, pending target captures, positions, RNG and consumed copy/observation
budgets. Raw target embedding/readout operations remain distinct from ordinary
normalized target ingress/output; the selected assistant owns their finishing
policy and required transports.

`CheckpointSource` describes independently retained payload ownership and whether
leases expose direct selected encoded bytes. In-memory backing survives restricted
views; file weak caches retain no independent tensor payload. Shared loading
profiles combine exact unit recipe envelopes with enforced host/device reservations.
Conditioned forecasts compose architecture geometry with these observations and
retain the ordinary MLX contract's unknown native workspace and allocator capacity.
They do not promise finite total-process admission.

Family verification retains pinned layout/reference manifests, whole-model
acceptance runners and concise results. Shared execution conformance uses compact
Muse fixtures alongside other families. Component numerical diagnostics, tensor
dumps and historical experiment archives are not checked in. See
[model support](model-support.md#muse-spark-evidence-and-support-status) for current
released acceptance and explicit validation gaps.

Intervention schema 2 includes the router's ranked or unranked partition order in
its exact coefficient policy. Admission retains that order when constructing
neutral route controls. Muse's observed ordinary and tensor-parallel paths apply
these controls before provider dispatch and publish one completed routing event
after per-expert normalization, weighting and decoder projection. The event names
global expert IDs even when the selected bank is partitioned. Cold routing
declarations and loaded bank metadata use the same canonical routing boundary.
Pinned nonzero decoder fixtures cover observer and provider controls through
prefill and three cached steps, including reconstruction of complete TP2 expert
outputs before normalization. These component checks do not establish Spark
controlled-session or speculative-assistant acceptance.

The shared decoder consumes typed attention, block-normalization, residual and
routed-feed-forward policies. Normalized-head attention rounds weightless Q/K
normalization before query scaling and normalizes each attended head before
sigmoid gating. Weightless branch norms have no parameter slot. Learned residual
gates and compressed expert projections participate in ordinary parameter
traversal. Compressed routing reads decoder-width input before latent projection;
latent input normalization precedes expert execution. Each complete expert output
is input-scaled and normalized before coefficient multiplication, group-ordered
accumulation and the replicated latent-to-decoder projection. TP plans partition
expert intermediates and replicate routers, correction buffers, latent projections,
norms and residual gates. These policies are included in architecture fingerprints.
Top-k selection retains a neutral `GroupSelectionOrder` before selected-score
normalization. Ranked selection sorts by selection score; partition selection
retains value-only median-partition order. CPU selection uses bounded score and
index rows; the bounded Metal kernel shares the value-only partition mechanism
with cutoff-tie repair. Other devices copy bounded score rows through the host
operation. This ordering is independent of expert-output
accumulation order: FP32 denominator rounding can change a later BF16 weighted
expert value even when the selected set, raw outputs and normalization agree.
The compressed Hugging Face Muse equation requests partition order. Ordinary
ranked selectors retain their existing policy.
The neutral `GroupIndexSelection` memory invocation describes already transformed
score rows separately from projection and coefficient normalization. It retains
selection order and reports the index output, negated score buffer, reused host
index row and bounded output copy, or Metal thread-local indices. Native capacity,
cast aliasing, register spills and transfer overlap remain explicit unknowns.
Compressed expert TP sums have shape `[batch * sequence, selected_slots,
expert_width]`. The composite and partitioned Muse plans declare those exact
axes for inactive pipeline participants; their scalar representation follows the
selected activation dtype through `RoutedTensorReduction::Activation`.

The neutral Muse waveform pyramid composes biased patch projections, direct
FP32 RMS normalization, causal depthwise convolution, exact GELU MLPs and
residual addition. Patch and convolution modules have separate parameter
topologies; their canonical names follow `model.audio_tower`. The caller owns
bounded convolution histories. Forward evaluation borrows the previous carry
and returns a candidate, leaving commitment to the execution driver. State
geometry includes stage widths, patch sizes, kernel width and batch size.
Incomplete waveform frames and final zero-padding use the existing portable
`PcmFrameQueue`; encoder math accepts complete cadence-aligned chunks. These
encoder contracts are independent of public Spark source admission and sessions.
Explicit audio segments distinguish continuation of one stream from a new clip.
Patch reduction respects cadence-aligned boundaries, and every causal block
resets history at each new clip before publishing only the final clip's candidate
carry. Segment metadata participates in forward-context snapshot accounting.
The component and composite seams share audio ingress and unit-execution functions.
SafeTensors and GGUF audio schemas share one logical tensor geometry. GGUF
normalization retains explicit stage widths, patch cadence, flattened convolution
block identities and the waveform projector's direct RMS epsilon. Source mappings
use the declared stage depths to recover canonical convolution owners. Vision and
waveform quantization maps are selected independently from actual source encodings.
`muse::audio::execution::AudioArchitecture` declares each patch, each causal block
and the final perception projection as its own execution unit. It uses the ordinary
resident and layerwise drivers and `RuntimeStateComponents`; convolution histories
remain bounded, full-width, device-resident mutable state. Unit frontiers count
emitted audio tokens and must agree before another chunk begins, independently
of internal stage frame rates. The caller retains unpublished candidate
state until the invocation completes. The audio component's final adapter shares
its equations and parameter topology with `AudioPyramid`.
The parent Muse graph retains vision group zero and decoder group one, with an
independent waveform root at group two when the selected source contains audio.
That root uses the same small modules and unit-execution function as the audio
component. Prepared ingress packs waveforms separately from image patches and
keeps each attachment's convolution-reset boundary. Decoder assembly consumes
separate ordered vision and audio placeholder spans. Parent fixed-state frontiers
advance by the complete decoder invocation length, including text-only steps;
PCM cadence and internal stage frame rates remain separate quantities.
All 72 released audio parameters belong to 13 execution-unit recipes rather than
pinned modules. Local Muse construction retains the exact `PartitionState` proof,
including ownership gaps, instead of inferring ownership from a scalar offset.
Audio continuation declarations retain each producing unit's width and frame
reduction in both cold and request-specific transport geometry.

Prepared source declarations distinguish `ArtifactSourceRole::Vision` and
`ArtifactSourceRole::Audio`. Muse source spelling is normalized only to assign
ownership; each restricted view retains the exact admitted physical keys and
shares the already opened artifact with target execution. A combined Muse GGUF
projector supplies disjoint vision and waveform views. `media_source` exposes the
selected view without reopening any artifact. `CapabilityError::MissingCompanion`
identifies an absent required tower during media request preparation; text requests
remain independent of that tower. Architecture capability estimates admit raw
modalities against retained component coverage before processor lookup. Missing
encoder/projector coverage takes precedence over absent processor assets; a
present tower without a selected processor reports the processor assets instead.
The neutral `MultimodalPreparationFailure::Capability` and public
`MultimodalPreparationError::Capability` retain this typed admission error without
converting it into a native backend failure. Ordinary and observed preparation
share this preflight. Cold inspection consumes the same component-coverage policy
and records optional `MissingMediaCompanion` information without rejecting text
loading. These source views do not admit an unselected external SafeTensors
companion.

`ArchitectureStatePartitionPlan` normally requires a contiguous selected state
slice. Its explicit `StatePartitionPacking::PreserveGlobalOrdinals` policy also
represents disjoint component ownership: unowned gaps become `NoState` slots,
while owned KV and fixed buffers retain their original global cache keys.
`PartitionState` retains exact selected ranges separately from that represented
span and projects the complete layout through `selected_layout` for family
identity checks. Stateless gaps allocate no attention history or fixed tensors;
their ordinary host slot metadata remains part of the local layout. Existing
contiguous selections retain their contiguous packing policy.

`CompositeContinuationGeometry` distinguishes uniform encoder geometry from
shapes after individual units. Selected composite routes retain the producing
unit frontier along with their exact schema. Runtime transport resolves the
request's frame count and width at that frontier; cold boundary declarations
also retain per-unit sequence bounds. This represents patch pyramids whose frame
rate and width change between stages without moving model equations into a backend.

Muse processor assets retain a typed video timestamp policy. The released Spark
asset requests hundredths of a second; the Glimmer default remains tenths.
The Spark waveform policy declares mono 24 kHz PCM, right zero-padding to 1,920
samples per token and explicit soft-audio framing text. Portable
`pad_waveform_to_cadence` preserves samples and their valid count; it performs no
spectral extraction or backend allocation. Host preparation pads attachments
individually and emits explicit `InputExtent::AudioValidSamples` descriptors,
distinct from feature-frame extents. These descriptors retain the reference
attention mask's valid length without allocating a sample-sized native mask.
Architecture admission requires floating mono PCM, a positive valid-sample
extent, and padding confined to the final cadence frame. Audio host preparation
and source-tower admission remain distinct contracts.

`TopKGroupSelectorSpec` retains an explicit positive logit multiplier as well as
independent projection, score and coefficient arithmetic. The multiplier acts
on projected logits and ordinary bias before scoring; selection-only correction
does not contribute to mixture coefficients. Native ordinary selection,
caller-selected routes and routing interventions use the same projection path.
These mechanisms do not register Spark as an executable model family.

Grouped gated-product operators expose `forward_uncombined` and
`combine_outputs` around architecture-owned per-expert transformations. The
uncombined tensor keeps `[tokens, selected_slots, output_dimensions]` in exact
route-slot order and contains no routing coefficients. Its tensor-parallel
operation separates unweighted rank-local outputs from replicated down bias;
reconstruction and one bias addition precede nonlinear normalization. MLX uses
the same chunked projection and unit-observer driver for combined and
uncombined operations. Separate accumulation preserves the declared reduction
order and dtype boundaries.
Resident and addressable parameter providers expose the same uncombined boundary.
The addressable driver retains its selection validation, compact-bank acquisition,
chunking and completion path while preserving the selected-slot axis in its output.
Provider composition, bank dispatch, unit observation, type erasure and execution
agreement retain that operation. Unsupported neural mechanisms return the typed
`UncombinedGroupedOutputUnavailable` error without returning an accumulated tensor.
Expert-exchange invocations declare coefficient-weighted token accumulation or
unweighted per-selection return. Per-selection reverse exchange validates the
same route tags and scatters to unique flattened token/slot rows, preserving
duplicates without multiplying coefficients. TP partials and replicated bias
retain that slot geometry until the caller reconstructs each expert. This is a
neutral exchange contract; the registered families retain their selected
coefficient-weighted exchange policy.
Partitioned parameter providers select this policy through the same route
preparation, agreement, transport and completion driver. Owner-local uncombined
requests validate cardinality one independently of the source top-k cardinality,
preserve the original coefficient metadata for unit evidence, and flatten their
single-slot axis only for transport. Source-side reconstruction restores that axis
before architecture-owned expert normalization.

`eredu-checkpoint::nvfp4::NvFp4MatrixFormat` describes borrowed packed E2M1
weights, block-16 E4M3 scale geometry and contiguous output-row partitions with
independent F32 global scales. Its portable decoder expands only selected rows
under an explicit output-byte limit. Two row partitions preserve a fused
gate/up matrix's distinct scales. Input-quantizer scales are not weight
dequantization factors. `LinearFormat::NvFp4` and `LinearFormatSpec::nvfp4` retain
block and global scales as distinct physical companions. Parallel expansion
partitions packed words and block scales while replicating the small global-scale
tensor and retaining independent row origins. ModelOpt byte-packed values and
E4M3 scales use byte-preserving views into canonical U32/U8 sources; MLX split
banks stack their global scales along the component axis.

The MLX NVFP4 fallback converts at most 128 output rows from a selected expert
at once. Each tile completes its projection before conversion storage is released;
inference never expands an entire expert bank. Explicit effective-parameter
inspection can request a decoded parameter under its ordinary capture reservation.
Global scales participate in parameter traversal, retained dependencies and
conversion accounting. Dense-to-NVFP4 calibration/encoding is unavailable through
the existing load-time quantizer. These contracts do not register Spark or establish
full released-model inference parity.

The released NVFP4 text comparison exercises the ordinary disk-streamed layered
runtime with packed expert banks and bounded selected-expert conversion. Prefill
and three cached decode steps, including all 248 decoder-layer outputs, match the
pinned independent reference exactly on the local CPU. This is numerical evidence
for that selected artifact and invocation sequence; it does not register Spark or
establish multimodal, controlled, assistant or native distributed conformance.

Pinned Muse equation fixtures live in `eredu-evaluation::muse` for reuse by
neutral and native tests. Backend mechanisms have no production dependency on
that evaluation module.

Multimodal preprocessing starts with an architecture-owned processor plan.
That neutral plan parses family model and processor metadata, selects released
defaults, declares framing token IDs and text, chooses frame sampling and
grouping, and derives resize, normalization, patch, and signal-feature policy.
Audio feature plans specify the analysis window, exact padding and frame-count
convention, frequency bounds, mel scale and normalization, spectrum value, and
logarithm as well as their dimensions and floors.
`PreparedProcessor` interprets that plan over `ProcessorMechanisms`. The
architecture layer first executes every host transformation through
`eredu-media`, validating the complete input before any native conversion.
`ProcessorMechanisms` receives typed processed buffers and constructs native
tensors or inspects small native metadata values; architecture code owns frame
selection, patch packing, framing order, metadata, and the ordered
`PreparedModelInput`. A new backend therefore supplies tensor conversion and
inspection mechanisms without implementing preprocessing mathematics or a
family processor.

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
tensors to that admission before the graph can execute. Family kernels accept
native or admitted prepared input through the same
`ComponentAdapter`. `CompositeModelInput` and conditional input variants retain
these typed ingress choices; `prepared_input` and `prepared_view` preserve the
admission proof and request metadata when prefill or partition transport needs
the prepared view. The kernels build embeddings, placeholders, masks, positions,
optional encoder roots, projectors, merge inputs, and per-layer visual
contributions, then enter the ordinary layered driver. Composite preparation
adds no forwarding architecture or second ownership view. Chunked prefill,
prediction, parameter editing and capture use the adapter and its retained
component definition directly. Qwen4 prefill retains admitted encoder output and
original request roots across decoder chunks. Direct and routed composite
execution therefore use the same replicated session for prefill, decode, transactions,
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
ordered forced-frame plan. Architecture code resolves that plan against
validated host token storage and produces portable forced frames in enqueue
order. The common runtime ingress later performs family-blind host-to-tensor
conversion; a concrete backend does not match PersonaPlex prompt sources or
redeclare its application policy. Realtime model identity is likewise
architecture-owned. The selected architecture wrapper retains the source and
execution configurations plus the authoritative neutral realization; concrete
mechanisms are bound beneath that wrapper and do not restate a family identity.
Family metadata such as `EffectiveModelType` remains in
`eredu_architectures::moshi`.

Moshi execution retains a validated `ComponentDefinition` before native module
construction. `MoshiKernel` supplies temporal and depth equations to the shared
`ComponentAdapter`, which owns runtime addressing, parameter traversal, state
mapping and group transport. Temporal blocks retain independent persistent cache
slots. Each independently leased depth slice explicitly reuses the same
frame-local depth cache range; starting a frame resets that segment without
resetting temporal history. Pinned embeddings, normalization and text projection
retain their canonical roles. Serial and parallel execution use the same
declaration, with selected local geometry supplying the state layout. Realtime
construction identity is a typed kernel capability forwarded by the shared
adapter; frame decisions and sampling remain architecture policy.
`MoshiParameterContract` retains the source-free definition alongside exact
matrix formats. Parameter descriptions and replicated state requirements are
projections of that definition. Pure tensor-parallel selection projects its
group counts and state ownership, then retains a definition with validated local
geometry. Realtime construction consumes the retained source and selected
definitions directly before allocating pinned native resources.

Realtime artifact loading also crosses the backend boundary as a neutral
architecture preparation. Moshi inspection parses optional native defaults or
released PersonaPlex configuration, resolves the confined `moshi_name` or
indexed checkpoint source, validates the strict SafeTensors contract, and
publishes canonical binding recipes in `MoshiPreparationPlan`. The shared
`PreparedRealtimeArtifact` retains this categorical preparation or the coupled
speech preparation according to the discovered architecture. Concrete
backends consume that plan and load tensor payloads; checkpoint-layout identity
names the physical Moshi or PersonaPlex SafeTensors namespace and never a
concrete backend. Backends do not receive a raw artifact path or reinterpret
family configuration and filename policy.

Moshi categorical inspection derives one exact architecture requirement from its
metadata-only preparation. It includes separate source and execution
identities, per-parameter lowerings, immutable and mutable residency, speech
schedule and state geometry, topology and rank, finite invocation bounds,
observation seams, completion policy, and every required generic mechanism.
Family-blind selection compares that requirement with an independently reported
backend capability set before payload access, module or state allocation, or
route-owned queue and group realization. The resulting selected realization is
paired immediately with exact materialization tasks. Each task retains its
architecture owner, recipe traversal, admitted catalog keys, physical tensor
and output identities, source encoding, and backing shard before a backend
store or native resource exists. That prepared contract is the sole input to
construction; visitors consume it without rebuilding source semantics or
reselecting a format, residency, topology, or state profile from caller options.
The selected local adapter also
completes this inspection and selection before it realizes an MLX device or
creates execution and weight-materialization streams.

Typed Moshi architecture dispatch builds static, temporal, and depth modules
and passes them to `eredu_runtime::construct_realtime_model`. Resident and
bounded policies share that constructor and its `RealtimeLayerwiseRuntime`;
pure tensor parallelism wraps the same selected traversal in the partitioned
runtime. The implementation backend binds generic tensors,
operators, stores, state, streams, and collectives, then performs only a final
opaque erasure of the already constructed executable. Its public native handle
is this mechanism bundle, not a prepared Moshi model or semantic realtime
state, session, or scheduler.

`eredu-runtime` owns the complete frame lifecycle. Validated portable ingress,
typed coordinate payload history, schedule advancement, delayed temporal
resolution, forcing, sequential text/depth decisions, output alignment, and
history pruning execute once in the shared coordinator. The coordinator
branches model/cache state, history, schedule, samplers, and randomness as one
transaction. Moshi architecture composition enables its proven fully-forced
depth-tail omission; the neutral decision driver disables that optimization
when diagnostics are requested or any remaining decision is not forced.
`RealtimeSessionScheduler` is the only fair request lifecycle for
single and concurrent sessions, including deadlines, cancellation,
release/resume, and idle-only sampling replacement. Architecture observation
hooks compose with layered traversal and sequential decisions, so replacements
at declared temporal, layer, decision-logit, and final-output seams affect the
actual downstream transition rather than a diagnostic copy.

`RealtimeGenerationState` retains an architecture-owned `RealtimeScheduleState`
and sampling controls under an exact `RealtimeGenerationContract`. The contract
declares scheduling identity and categorical sampler count independently; it
does not derive decision cardinality from audio geometry. A transition declares
its own input type and may advance separate model, audio and tool clocks. The
ordinary transition helper and explicit branch execution share the same model,
sampler, RNG and completion ownership. Publication checks the contract, schedule
successor, sampler count and branch revision before model commit, then adopts
the remaining state without another fallible operation. Idle sampling replacement
also advances the revision so older branches cannot overwrite new controls or
randomness. Schedule validation failures preserve their portable semantic source.
Uniform and Gaussian draws use the same retained-RNG mechanism in decision
drivers and unpublished generation branches, including transitions with no
categorical samplers. A failed draw leaves the local RNG unchanged; branch
discard leaves canonical RNG unchanged.
`with_random_state_transaction` extends that rollback boundary across coupled
draws and dependent operations. It rejects branches with attached completion
and adopts the cloned random state only after the callback succeeds. Native
completion and canonical publication remain the enclosing transaction's
responsibility; the helper neither releases submission resources nor refunds
observation or transport budgets.
Moshi declares its text-plus-depth sampler bank in architecture composition.
Cold selection retains a `RealtimeFrameContract`: architecture policy identity,
versioned canonical policy descriptor, exact channel layout and explicit sampler
count. `RealtimeSessionScheduler` accepts architecture-owned input, schedule and
sampling types through the same core scheduler. Registration compares the
schedule descriptor and sampler bank before admission. Session payload contracts
derive their channels from that retained selection; callers cannot substitute a
different layout when requesting a payload contract. Batch admission checks
expanded channel geometry before invoking the executor.
Canonical frame descriptors encode channel geometry, cadence, delay and feedback
with explicit lengths for distributed consensus. Rank-local state identities
remain part of exact local selection and resumption; common consensus compares
the topology-wide architecture, frame policy and selected state realization.
Moshi binds this machinery to `RealtimeFrameScheduleState`, `RealtimeSampling`
and its categorical interpreter. Another family supplies its own transition
equations and selected component graph.

Retained coordinates use architecture-local `RealtimeChannelId` values.
`RealtimeChannelLayout` declares categorical domains or dense f32 frame geometry,
logical cadence, placement delays, and feedback lookbacks. Layout admission
rejects unknown dependencies, overflowing geometry, incompatible producer cadence,
and same-step feedback cycles; delayed recurrent feedback is valid. Exact payload
contracts retain the whole layout, batch, session owner, and history generation. Immutable layouts are
shared across envelopes, and pruning retains the declared delay and feedback
window plus the preceding completion's payloads. Moshi and PersonaPlex use these
contracts with their text/codebook declarations and existing coordinate policies.

`ValidatedRealtimeChannels` checks mixed host payloads before materialization.
`ValidatedRealtimeFrameInput` admits ordinary inputs and selected overrides
together, then creates `MaterializedRealtimeInput` with the exact layout, batch
and channel identities. Both groups retain their own payloads when they name the
same channel. The MLX frame mechanism copies categorical and dense arrays on the
selected stream; completion retention traverses both groups. Moshi admission
projects host matrices into channel columns, omitting unselected forcing values.
Its interpreter consumes the retained ingress contract and rejects missing or
misassigned channel roles before schedule or history mutation. Native column
extraction is not part of this input contract. Architecture policy determines
which channels a transition requires.
The text/depth frame interpreter remains a categorical execution policy; dense
ingress and retention alone do not declare a speech architecture executable.

`CompletedRealtimeFrame` carries categorical or dense channel values, explicit
ordered decision identities and optional diagnostics. Current and delay-aligned
values retain separate coordinates even when they name the same channel;
unattributed warmup padding is allowed only among current values. Completion
retention traverses these payloads without assembling family-specific matrices.
Native host observation checks tensor dimensions against each declaration and
converts values to portable categorical or dense storage. Shared validation
rejects invalid domains, extents and nonfinite values before publication.
Moshi's architecture-owned output projection collects validated host columns in
its public text/depth/codebook order, including PersonaPlex warmup behavior.
The native adapter supplies conversions and delegates that semantic projection.

`RealtimeCompletionMechanism<R>` consumes an explicitly typed resource bundle.
The categorical interpreter supplies `RealtimeFrameResources`; component
coordinators can supply their own bundles. `submit_realtime_output` attaches an
exact completion to the existing unpublished branch and packages arbitrary typed
output for `PrepublicationRealtimeFrame`. Submission failures distinguish work
that never escaped from work requiring a retained quarantine completion. This
generalization changes neither scheduler admission nor the commit boundary.
Host observers validate the selected output type before publication.

The MLX completion mechanism retains native input, output, diagnostics, history,
state, checkpoint stores, streams, and collective ownership. Runtime
prepublication waits for that exact completion and validates native token
scopes before host observation; only then may the scheduler publish the branch
and its portable output. Delayed or failed returned completions remain
quarantined and cannot expose host output. A scheduling failure before event
publication carries a recovery completion when work is unresolved. An outer
scope covers input materialization, traversal, sampling, and frame assembly;
it retains the entire native execution payload if any accepted work remains.
An unresolved or failed execution cannot be reused. These ownership rules are
mechanism-specific; commit and visibility rules remain backend-neutral.

Distributed realtime turns use the same neutral scheduler transaction. Before
submission, bounded consensus compares the topology-wide selected model
identity—including selected state realization and observation policy—rank
ordering, work descriptor, request, and portable frame on every participant. A
second bounded consensus round agrees the exact submission count and any local
submission failure. Following native completion and host observation, an output consensus round
compares each local outcome and a digest of the complete portable output. A
descriptor or output disagreement, peer execution or completion failure, or
indeterminate consensus discards every unpublished branch; only topology-wide
completed output agreement permits any rank to publish. The MLX transport
supplies only bounded word gathering and its retained native completion.

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

`eredu-nn` owns tensor-independent normalization and causal-mask geometry
validation. Backends validate final-axis dimensions, epsilon, grouped gates and
divisibility, and checked sequence/offset/lookback ranges before tensor work.
Causal masks interpret lookback as an inclusive maximum past distance;
sliding-window attention uses its separately specified token-count window.
MLX and scalar reference mechanisms consume the shared geometry while retaining
their own dtype, layout, reduction, and kernel implementations.

The shared dense decoder also admits architecture-owned RMS scale offsets,
post-attention and post-feed-forward normalization, embedding scaling and output
soft caps. These policies run identically in resident, bounded and partitioned
execution; post-sublayer normalization follows TP reduction. Gemma 2 supplies
these policies and its SafeTensors/GGUF contracts without native dependencies.

`AttentionRequest::softcap` is a neutral positive score cap applied after score
scaling and before masking and softmax. Backends advertise
`ATTENTION_SOFTCAP` only when contiguous, sliding and paged attention implement
it. Paged caches retain the request policy through every block recurrence.
`AttentionCache::uses_blockwise_attention` declares when returned update tensors
exclude retained history, so the portable decoder routes chunked sliding
prefill through cache-owned attention. Neither capability identifies a family.

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
4. The selected backend adapter realizes native mechanisms from the plan and
   passes its exact prepared-source bundle to `construct_prepared_execution`.
   Architecture-owned sealed routes invoke the appropriate typed construction
   contract. The backend erases the completed typed session at the outer boundary.

`ReplicatedTextMechanismSupport` reports backend facts and answers support for
one exact lowering candidate or state component. Runtime capability synthesis
enumerates primary and auxiliary requirements, validates portable geometry,
deduplicates lowering candidates and selects exact state placements. Both
native and independent scalar adapters use this same synthesizer; no provider
needs to reconstruct requirement enumeration or policy from family identity.

The total construction driver consumes `PreparedModelSources` and owns ordinary
versus partitioned communication consistency, target/extension agreement,
prediction preparation, the floating-state source, and processor preparation.
Native communication is checked against the exact retained manifest before
typed construction. `PreparedExecutionRoutes` contains only supplied native
visitors and contexts; optional routes impose no bounds on absent capabilities.
`KeyValueRoute` supports ordinary attention using the base neural and cache
traits, without requiring blockwise, grouped, distributed or prediction traits.
Complete profile dispatch uses an explicit `ReplicatedTextStateProfiles` set
and `SharedReplicatedTextVisitor`, retaining profile-specific static bounds and
the original visitor's construction-start notification.

`PreparedExecutableAssembler` receives an executable only after typed
construction succeeds, together with the exact selected state policy,
capabilities, physical floating-state width, prepared processor, and retained
native communication. MLX wraps these values; it does not repeat processor
selection, extension pairing, or the six-class dispatch.

Concrete backend preflight reports mechanisms rather than family support or a
parallel execution-class table. Architecture dispatch is authoritative for
ordinary causal text. Replicated, routed, and composite classes enter exact
typed architecture visitors; partitioned requests enter the corresponding
neutral admission and rank-local construction first. A request that selects no
such contract is rejected during architecture selection. MLX therefore has no
unclassified execution class, family binding table, or complete-model family
loader to consult. Addressable parameter banks requested for a configuration
without that neutral capability are rejected explicitly rather than being
silently discarded. Complete-model GGUF load-time quantization is encoded in
the selected per-parameter tasks, so selection and materialization cannot
disagree about whether a format or lowering is accepted. Execution classes
handled by a distinct protocol, such as realtime Moshi, do not enter ordinary
causal-text dispatch.

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

`ModelPreparationPlan` is the one-shot authority for stage 4. The architecture
source factory consumes it with the exact retained selection and publishes one
bundle containing the originating inspection plus primary, companion, target,
and extension views. Materializers, including partitioned materializers,
consume that inseparable bundle and selected route directly. They
must not reopen an artifact to rediscover configuration, checkpoint metadata,
or sibling filename policy after planning; payload stores may still map weight
members during materialization, but those reads do not replace the plan's
configuration, checkpoint metadata, companion selection, or route.

`ModelLoadingBackend` implements backend capability reporting, native
materialization, and outer adaptation. `BackendProvider::create_session`
consumes a `PreparedModel`, so an executable cannot be paired with state or
session mechanisms created by another backend. Replicated prepared models
already contain the neutral lifecycle paired with their concrete mechanisms.

`ModelRuntime<B>` owns a backend and its sole session. All facade clients use
`eredu::api::LoadedModel<B>`, whose runtime remains private while portable model
operations stay generic. Loading and generation failures retain `B::Error` in
their generic error types.

Automatic planning produces a portable `ExecutionPlan`. An
`ExecutionPlanBackendFactory` realizes the complete plan into a target backend,
backend load options, and the selected disabled, embedded, or external drafting
configuration. `LoadedModel::load_execution_plan` and `LoadedModel::plan_and_load`
therefore infer the backend from its factory without requiring callers to
construct backend devices, queues, streams, assistant models, or sessions. Concrete device families are
validated against the backend features and runtime hardware during realization.
The backend reports the canonical identity derived from that realized device
binding rather than echoing the plan's requested identifier.
Automatic feedback cannot introduce an assistant identity from telemetry:
an external plan requires explicit facade-owned assistant inspection and
tokenizer proof, neither of which is part of an automatic planning request.
Core binds cold drafting selection to the exact complete execution plan and
target selection, and rejects either selection if later realization supplies a
different plan. Each selected target carries an opaque core-owned identity;
`ExecutionPlanTarget::into_runtime` moves that exact backend/prepared-model pair
into the runtime, and external drafting realization rejects a runtime produced
from any other target even when both targets implement the same plan.

The loading APIs' `*_with_text_options` variants accept facade-owned `TextModelOptions`,
as do retained-inspection and direct backend loading. A caller-supplied single
or named chat template supersedes checkpoint template selection before native
realization, while tokenizer variables, EOS ids, and generation defaults remain
checkpoint-derived. These options stay outside the core execution plan and
backend load policy. Pre-load text inspection borrows the `TextModelOptions`
value intended for loading. Both operations use one facade-owned template
resolver, with invalid selected metadata reported instead of falling back to
another source. `TextInspectionOptions` holds only the optional behavioral probe
request. `LoadedModel::set_chat_template` replaces the template used
by future preparation and loaded-model template inspection, clearing compiled
templates through the neutral `eredu-text` cache utility. Existing prepared
chats retain their prompt and protocol metadata. All supplied templates pass
through the ordinary facade-owned rendering and protocol-recognition path.
Without an override, templates come only from checkpoint metadata or sidecar
files. No model family, including Gemma 4, receives an implicit template.
A model with no template still loads for raw token generation,
but chat preparation returns `TextModelError::MissingChatTemplate`; the facade
does not invent a generic prompt format. Clearing the template with the setter
also produces that error without reloading metadata or invoking a fallback.

External assistants cross that factory boundary as an architecture-owned
`ExternalAssistantPreparationPlan`. Architecture inspection fixes the
assistant family, normalized configuration, checkpoint format, and strict
SafeTensors or GGUF layout before a backend is selected. SafeTensors plans
retain the admitted header catalog, architecture schema, and exact resolution.
A GGUF assistant plan retains the admitted portable checkpoint, resolution,
and canonical tensor mapping.
Before target device or stream realization, architecture-owned assistant
selection resolves the exact schema variant and retains one physical task per
materialized output, including shard/member/output provenance, aliases,
encoding, geometry, executable format, lowering, recipe, and atomic
companions. The concrete backend later reopens payload only through the
catalog-bound source, revalidates both the catalog and resolution, and consumes
those retained tasks with the generic exact binder. It rejects tensors changed
after selection before constructing a native module and does not walk the
assistant module to rediscover names or quantization policy. Concrete backends
do not receive an unrestricted assistant store, reopen configuration or
metadata for dispatch, bind against raw native topology, or duplicate
assistant admission.

`eredu::api::local_device_plan` maps the facade's CPU or accelerator choice to
the selected local backend and rejects accelerator choices when that
build contains no native accelerator family. `LocalRuntimeConfiguration`
applies any process-global allocator or embedded accelerator-library
configuration before the factory realizes that plan. Platform applications
import the concrete backend factory and use the generic facade to load and
operate the session.

Architecture inspection reports both embedded-draft depth and maximum proposal
capacity from the normalized, admitted artifact composition in
`ModelResourceProfile`. Capacity comes from the architecture's prediction
contract; fused predictors can propose more tokens than their physical layer
count. The neutral automatic planner caps its preferred draft width at that
capacity and leaves automatic drafting disabled when capacity is unknown.
Explicit draft widths are validated as requested. Concrete backends must not infer
family semantics by searching raw configuration documents. In particular, a
DeepSeek-V4 base GGUF has target blocks only: its `nextn_predict_layers`
metadata describes weights omitted into a companion artifact, so base admission
normalizes embedded depth to zero and capability reporting does not expose it.
Automatic bounded-residency sizing is likewise a cold operation. It performs
the same exact neutral selection used by loading and computes executable bytes
and the largest unit window solely from the selected materialization tasks. It
uses `eredu-runtime::selected_text_bounded_requirement`, including checked
byte/count arithmetic and sparse group-local window accounting. Neutral
residency telemetry is projected in runtime; speculative statistics and common
host observations are projected in core. Accelerator and allocator observations
remain backend mechanisms. The sizing path
does not realize a device or stream, open payload data, construct an
architecture, or provoke a failed native load to discover the requirement.
Automatic planning retains that single admitted artifact inspection through
every candidate admission, feedback choice, expert-cache mutation, and exact
bounded-residency probe. Applications may supply deterministic, idempotent plan
overrides, which are applied before candidate validation and sizing. The CLI
translates every explicit planning override through this path in quick and plan
mode; reports and cached plans describe the validated result. Residency probes
use the overridden plan's budgets. Feedback and cached plans are reusable only
when they already satisfy the current overrides, so their measurements cannot
silently replace an explicit choice. The final plan is admitted and probed again,
then loading consumes the same retained inspection; automatic loading cannot
reinspect a changed path or construct a route different from the report.

## Submission and completion

`BackendSession` provides high-level prefill and decode submissions. Associated
types keep prompts, tokens, outputs, session state, and completions opaque.
Concrete ordinary sessions hold one already paired architecture-erased neutral
session. Prefill, decode, state controls, reports, observation, prompt-cache
operations, speculation, and distributed control all delegate through that
same contract; there is no complete-model family or pipeline variant to select.
They do not expose model/cache parts or architecture-specific constructors to
callers. Operations unavailable for a valid session topology return typed
errors rather than relying on unreachable or panicking accessors.

Cold replicated-text requirements retain the dtype of the architecture-declared
floating-state source. Architecture code resolves its checkpoint aliases and
container metadata once; backends receive that exact dtype through
`ReplicatedTextMechanismSupport::floating_state_dtype`. The backend reports the
native `StateStorageDtype`, including any packed-embedding conversion. Runtime
synthesis resolves each component's floating or fixed dtype policy and passes the
exact physical dtype into `supports_state_component` for device and paged placement
checks. Unknown floating representations are rejected before native allocation.

State capability reports retain the source-to-storage mapping. Selection rejects a
report for another source dtype and records physical dtypes in
`SelectedStateComponentRealization`, preserving them through partitioning. F16
activation storage does not narrow fixed F32 or integer components. Construction
checks its native floating dtype against the admitted representation, including
same-width distinctions such as F16 versus BF16. MLX shares
one source-to-storage mapping between cold support and construction accounting;
backend providers do not repeat architecture checkpoint-name or family dispatch.

`Completion::resources_releasable` exposes native resource safety separately from
`is_complete` and `wait`, which report execution/output success or failure. A failed
submission can eventually become releasable without losing its error. Pending work,
unavailable evidence, and runtime contention report false. Cancellation requests,
timeouts, and observation errors never establish this proof. The default query
recognizes successful exact completion only; native backends provide independent
scope or fence evidence for failed submissions. This query does not make failed
model state reusable, publish output, or transfer ownership of retained resources.

MLX implements the query from its native submission scopes and outstanding consumer
and observation tickets, including sampled text, realtime, speculative, and
communication work. A settled producer cannot hide a pending child. Polling remains
nonblocking and does not execute arbitrary application destructors. Text completion
uses the neutral query before resolving model authority after a token failure.

Every submission returns an exact completion object. A completion observes
only the submitted work; it must not drain unrelated backend work. Schedulers
retain outputs, branches, cache transactions, and other resources until that
completion succeeds. Cancellation prevents publication but does not release
in-flight resources early.

MLX ordinary sessions include lazy token-domain reductions in the neutral
session's exact output-and-state completion boundary. An invalid token therefore
restores the session checkpoint before commit and cannot escape as a later
completion-only failure. The public session admits no newer state-changing
submission while its prior completion remains unresolved; generation-tagged
completion ownership prevents an older resolved handle from releasing a newer
submission's gate.

Native `Tensor::take_axis` also retains a bounds reduction in that submission
and substitutes safe indices until completion validates it. This protects
architecture-owned table lookups, including token-selected routing, when an
earlier embedding has recorded an invalid token but bounded layer execution
must settle before the error is published. Valid signed negative indices keep
their ordinary indexing meaning; invalid indices cannot reach a native gather.

V4 declares its implemented input, unit and readout hooks on the ordinary,
tensor-parallel and pipeline architecture contracts. Routed executors combine
those declarations with their provider hook facts; a pipeline-only declaration
does not establish input or readout support for a routed target session.
Normalized routed inputs derive their producer from the declared input node's
invocation owner, independently of whether the residual equation is a scalar
sum or a learned stream mixture. Prediction replicas retain the same declared
input boundary alongside their sparse expert-unit observations.

`SessionAuthority` and its move-only `SubmissionLease` own this exclusion in
`eredu-core`. Every state mutation checks idle authority before native work.
Native completions retain the lease and their exact resources; ready resolution,
terminal failure, or safe teardown releases only that ticket. Pending or
unresolved observation errors cannot release newer work. Atomic ticket matching
makes repeated and stale releases harmless, and ticket exhaustion is a typed
error. MLX retains one shared executable owner and an independent ticket for
each native scope; one completed child cannot release another unresolved child.
Failure remains a session poison even after its resources become reclaimable.
The neutral runtime supplies fresh successful state-restoration evidence for
recoverable errors. Only that evidence combined with an independently settled,
healthy native scope permits retry; unknown host errors and unwinds do not.
When restoration is proven but native cleanup is still pending, MLX retains the
executable and its submission lease in nonblocking recovery without poisoning
the session. The lease prevents reset or new execution until every scope safely
retires. Runtime-lock contention leaves this recovery pending; a later native
failure or unobservable scope permanently poisons the session even after its
resources retire. Error callers must allow retirement to complete before reuse.

`SessionAdmission` compares complete admitted and realized capability
reports before publication, rather than accepting a subset or assuming universal
backend facilities. Sampled-token completion and model validation both finish
before a text submission releases its authority.

Successful synchronous MLX operations retire their native owners under the same
runtime guard that confirms completion, then finish staged host destruction at
an ordinary unlocked boundary. This prevents another runtime user from delaying
lease release between a successful wait and the next session mutation or bounded
checkpoint acquisition. Polling, errors, and teardown remain nonblocking: when
the runtime is busy, failure still reaches the retained owner, but resource
release waits for both terminal evidence and safe runtime access.

The backend is selected for the entire model/session. Per-operation backend
selection would make cache ownership ambiguous and introduce implicit data
movement. Explicit transfer and collective capabilities cover the cases where
movement is part of the session design.

Speculative publication is one atomic semantic callback for a committed token
batch. A cancellation already pending at that boundary is staged before the
batch becomes visible; no later fallible cancellation callback can expose a
partial success. A callback failure restores the backend checkpoint and leaves
the canonical sampler, constraint, sequence, cache, and publication state
unchanged. Target-only and speculative commits both cross publication, while a
speculative verification additionally crosses exact completion, observation,
and cache persistence before publication.

## Text and multimodal generation

The facade's `ChatInputRequest` accepts ordered typed `ChatMessage` values,
decoded text/image/audio/video `ChatPart` values, tool declarations, and parsed
historical `ChatToolCall` arguments. `ChatPreparationOptions` retains the existing
tool, thinking, reasoning-effort, generation-prompt and template-variable policy.
String content concatenates parts with an explicit separator (empty by default);
`ChatContentFormat::TextParts` supplies ordered text objects to templates that
expect arrays. These representations use internal media bindings rather than
checkpoint-specific image URL objects. Templates must preserve each binding
exactly once and in order. Bindings avoid caller/template spelling collisions,
are deterministic for unchanged inputs, and never enter tokenization.

`LoadedModel::prepare_typed_chat` performs portable lowering, format-profile
argument selection, template rendering and binding validation. Its
`PreparedTypedChat` permits capability inspection before native materialization.
`materialize_chat_input` consumes this retained conversation through ordinary
`MultimodalPreparationBackend` mechanisms; `prepare_chat_input` combines both
phases. Text-only backends use `prepare_chat_text_input` or
`materialize_chat_text_input`, without implementing media or capability traits.
`PreparedChatRequest` owns the rendered `PreparedChat` and an opaque payload
with `InputTokenCount`: context admission uses `model_positions`, while
`text_tokens` and `media_positions` retain their separate meanings. Splitting
the request and converting its payload with `into_input` produces the existing
generation input. Typed payloads retain their loaded-session and rendered-chat
identities through ordinary, speculative and forecast admission.

Typed tool arguments are JSON objects. `ChatToolCall::from_json_arguments`
parses serialized history once, rejecting malformed JSON and non-object values.
Automatic lowering prefers established mapping support, then established
serialized-string support, and uses mappings for unrecognized custom templates.
An explicit argument format overrides that representation; existing protocol
validation still applies. The raw JSON-valued `ChatTemplateRequest` entry point
keeps caller-provided argument representation unchanged. Application visibility,
emulation, fallback choice and encoded-media transport remain
caller policy; the typed API accepts decoded portable buffers without optional
host codec dependencies.

Both preparation APIs normalize application tool names before template rendering
and grammar construction. Dialects declare their additional naming restrictions;
normalization and wire validation use the same constraints. Shared wire names are
nonempty ASCII letters, digits, underscores or dashes within 64 bytes; LFM2 adds
Python identifier syntax and reserved-word exclusions. Accepted original names
are reserved first. Other names receive deterministic bounded aliases derived
from their complete original identities, with checked collision suffixes.
Declaration ordering does not change the mapping. Historical-only functions and
named tool results participate in the same namespace without adding tools to
the current executable schema registry. Only structural name fields are
rewritten; arguments, descriptions, schema properties, call IDs and text retain
their values. Missing/empty/non-string names and duplicate declarations fail
with `ToolNameNormalizationError`. An unrecognized tool protocol requiring
rewriting returns its typed unsupported-dialect error.

The immutable mapping belongs to the prepared semantic runtime plan and its
identity. The shared parser retains wire names for grammar and argument-schema
validation, and restores original application names in public `ToolCallStart`
events. Ordinary, speculative, observed and controlled execution use this same
sink. Parser forks share the immutable mapping while copying mutable call state.
Continuation event/storage bounds include expansion when a restored name is
larger than its alias. Raw tokens and literal text expose the actual model wire
representation; restoration does not replace text substrings. Backend and
architecture crates do not own alias allocation or application-name restoration.

`prepare_observed_chat_input` and `prepare_intervened_chat_input` admit the same
counted typed request for observed or controlled execution. Text retains its
encoded IDs; media uses `ModelCapabilityBackend::prepared_input_token_ids` for
complete decoder alignment, whose length must equal the counted model positions.
The MLX adapter delegates to architecture-owned semantic token construction,
including declared media placeholders and original projected-input IDs. Native
inspection reads those integer IDs rather than inferring IDs from embeddings.
Adapters without alignment observation return `BackendFailureKind::Unsupported`.
Public preparation errors retain native failures as `BackendFailure` sources.

`TextGenerationBackend` submits complete prefill-and-sample and
decode-and-sample operations. The backend owns logits, sampler state,
randomness, and token handles. Portable generation code owns token budgets,
stop and EOS precedence, cancellation, grammar state, and semantic events.

`eredu-core` owns the common `GenerationOutput<S>` terminal contract: committed
token ids, finish reason, `GenerationTiming`, and mode-specific statistics.
The facade's `GenerationOutput` uses `GenerationStats`, whose optional `drafting`
field carries `SpeculativeStats` from a scheduler-backed run. Ordinary execution,
controlled advancement and uninterrupted drafting have one public terminal
shape. Core's generic statistics parameter remains available to lower-level
execution mechanisms. All modes expose TTFT through `output.timing().time_to_first_token()`.
The facade starts this host clock
at the generation call, including request preparation; speculative batch lanes
share the batch-call origin and include time waiting for earlier lanes. Ordinary
generation records the first constraint-committed, completed token before record
or semantic delivery. The neutral speculative lifecycle records the first target
commit before publication, and runtime adds preparation and queueing time.
Structural tokens, buffered Unicode, stop tokens, and EOS count even without
visible text. Draft proposals never count, and cancellation with no committed
tokens returns `None`. This lightweight terminal metric needs no capture plan,
token journal, device timing, or extra native synchronization.
`SpeculativeStats::submission_to_first_token()` explicitly retains the
lane-submission origin. Runtime adds the facade preparation interval when it
constructs `GenerationTiming`; the output contains no speculative timing policy.
The CLI consumes this committed-token TTFT directly.

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

`SpeculativeExecutor` is the portable protocol implemented by architecture-owned
embedded and external strategies. `SpeculativeSampling` exposes only generic
logits, probability-arithmetic, sampling, and random-state mechanisms. A
backend supplies native tensors and operators, state storage, cache checkpoint
mechanics, queue and transfer operations, and concrete completion values; it
does not own proposal, verification, replay, or commit semantics. Embedded
prediction heads and external assistant models use the same portable lifecycle.

For embedded prediction, `eredu-architectures` owns the cache envelope that
pairs target state, typed prediction state, the selected realization, prepared
input identity, capture frontier, and fork/commit/restore membership. It also
owns the materialized prediction strategy and the object-safe executor erasure;
the MLX adapter performs one final continuation over that already paired
executor. Backend composition cannot re-pair an erased target with a different
prediction cache or substitute a family-specific speculative driver.

For external assistants, `eredu-architectures` likewise owns the semantic cache
envelope, capture schema validation, observer ordering, checkpoint restoration,
proposal/verification sequencing, and replay. The family-neutral MLX mechanism
adapter retains only opaque native cache storage, tensor operations, transfers,
streams, and completion values. Its activation observers are installed through
the architecture executor, so interventions and failures participate in the
same transaction rather than a test-only side channel. MLX sessions retain the
installed embedded observer set across erased executor lends; the backend does
not recreate a no-op observer set at each scheduler turn.

`SpeculativeLifecycleObserver` is an explicit, production-carried coarse
instrumentation seam. Neutral selection observes admission and compatibility
before native construction; the shared request table observes input,
execution, completion, resolution, cache persistence, publication, and
cancellation before each corresponding boundary. Observer failure prevents
that boundary from starting, and failures after verification submission restore
the retained cache checkpoint before publication. This lifecycle seam does not
replace typed `ActivationObserver` intervention over backend-native tensors;
the two report complementary control-flow and value-flow evidence.

The prepared-generation capability does not expose whole-request execution
methods. A backend implements `with_speculative_execution` only to consume the
retained neutral realization, prepare typed lane resources, and lend them
through `SpeculativeGenerationVisitor`. Compatibility and placement are fixed
before assistant or extension construction. The facade supplies
`RunSpeculativeGeneration`; `eredu-runtime::SpeculativeScheduler` registers
lanes, selects fair actions, drives exact completions, validates terminal
reasons, and constructs public outputs. Single requests use that same path as
one-lane batches. Concrete backends must not substitute their own lifecycle or
fair-batch loop.

Architecture families also own external-assistant compatibility proofs. Those
proofs match target-state publishers, hidden and rotary geometry, target-layer
captures, and vocabulary requirements before neutral composition constructs
the paired executables. They and the neutral speculative realization are
selected from the retained target inspection during cold drafting selection,
before target payload or native-resource realization. The facade supplies the
distinct tokenizer proof.
Backend composition consumes both proven contracts and does not restate
family-specific compatibility rules.

Core's drafting handoff proves the exact realized target before native assistant
materialization. The MLX drafter retains that target's immutable native session
identity beside its parameter owner. Each external generation request validates
the identity before queue binding, lane preparation or model work; equal artifact
bytes, dimensions, devices and world sizes do not confer session ownership.
This applies to conditioned assistants and independent autoregressive drafts.
External execution streams carry the actual target session's partition binding,
including its communicator and preparation-agreement owner. Request validation,
queue binding, sampling setup and cache preparation use the existing readiness
boundaries before participants enter the next stage. The backend does not
reconstruct architecture compatibility or initialize an assistant world.

External target and drafter mutation scopes participate in that retained readiness
agreement after acquisition and before executable access. A busy or retained
owner contributes its original failure instead of leaving peers to enter model
work. Sampler preparation and assistant acquisition finish while the acquired
target scope remains unused. A preparation refusal settles and releases healthy
unused scopes; it does not mark an unvisited executable as mutated. Failed or
unresolved native completion still fences its owner and retains its resources.
The same move-only drafter scope runs conditioned or independent execution once
both owners are ready; no separate generation driver or communicator is created.

Gemma 4 and Muse own their external-assistant execution strategies.
Gemma shared-attention capture truncation, target-to-draft state assembly,
proposal progression, and partial-verification replay live in
`eredu-architectures::gemma4::speculative`; ordered DFlash capture assembly,
sliding committed context, typed anchor-position proposal construction, and replay
live in `eredu-architectures::muse::speculative`. The MLX adapter binds
both strategies through one family-neutral mechanism implementation for arrays,
streams, exact completions, target-cache checkpointing, and already materialized
neutral assistants. `ExternalAssistantArchitecture` selects and constructs the
typed executor before lending it to the shared runtime visitor; MLX neither
matches assistant families nor provides a second family-specific
`SpeculativeExecutor` lifecycle. Compatibility-proven preparation is the only
public path to assistant materialization, so an incompatible pairing cannot
open assistant weight payloads.

The external preparation retains separate source and selected neural construction
configurations. Architecture normalization binds target-dependent geometry before
opening the source; the family-blind materializer constructs both encoding views
from those retained configurations. It does not derive geometry from a native
target or reopen an artifact.

Runtime's `ParameterAcquisitionUnit` groups admitted materialization tasks by
their declared independent storage owner. It validates unique primary/companion
outputs and companion ownership, computes selected global or rank-local payload
bytes through shared placement and executable-format accounting, and orders
execution-owned resources against the retained invocation layout. Source-sharing
resources remain separately charged; compact local ordering preserves nonzero
global unit indices. Architecture code declares owners and selected geometry;
native composition consumes these units without reconstructing their partitions.
`ArtifactError::Preparation` retains the original typed acquisition, placement or
accounting failure as its source without adding a runtime dependency to core.

Ordinary text, prediction and assistant materialization use runtime's exact
transform-group planner. Groups preserve first-observed format and task order,
exclude direct tasks, and reject missing packed formats. Assistant construction
validates these groups before creating native source/target modules. MLX applies
the declared transformations and retains original neutral task and assistant
neural-construction errors through its typed error boundary.

Spark's construction retains released identity and per-weight formats separately;
the same selected encoding drives exact source constraints and neural operators.
Matrix companion prefixes are explicit for transition codebooks without a
`.weight` suffix. Ordinary matrix and expert naming conventions retain their
existing declarations. Norms, convolution taps and I64 mapping buffers are not
converted into matrix encodings.
The shared `ExternalAssistantTargetProfile::Muse` retains the normalized target's
Glimmer or Spark identity. Ordered capture requests and values use the common
Muse contract; each assistant proves its released target geometry and capture
paths independently. Spark construction does not imply public session admission.

Assistant discovery precedes ordinary target discovery. An authoritative Spark
assistant marker is validated by its strict assistant schema, including DFlash2's
`qwen3` configuration; an invalid declaration retains its assistant-specific error.
The facade consumes `ExternalDraftTokenizerSource`: independent drafts reconstruct
their own tokenizer, while Spark assistants borrow the proven target's tokenizer.
The published assistant repositories contain no tokenizer assets. Borrowing uses
the actual target vocabulary fingerprint and retains the architecture compatibility
proof before materialization. Tokenizer ownership remains separate from the target
family registry and from native execution.

Target-owned assistant operations distinguish ordinary normalized embeddings and
logits from raw token-table lookup and raw vocabulary projection. Muse performs
vocabulary reduction or gathering through neutral distributed mechanisms. Raw
projection skips the final norm and output transform so a trained assistant can
rank complete global vocabulary scores before applying its own softcap.
`CompositeArchitecture` separates raw contributions from finishing equations.
Muse's complete operation composes these stages through the same interface:
embedding normalization follows combination, final readout normalization precedes
the local projection, and output scaling/capping follows combination. The raw
variants preserve the table or projection values. Without a parallel context,
a vocabulary-sharded module emits its local contribution; the existing parallel
context combines vocabulary shards before finishing. The finisher reads no
parameter table or projection. Static parameters retain their enclosing session's
completion owner; Muse selects no row-addressable providers for these tables.

Cold and bound capture preparation use one producer calculation from the
architecture descriptor, retained component coordinates and selected transport
precision. The prepared placement retains its session communication requirements.
Resolution checks one exact broadcast per ordered slot, including tensor rank,
input/output extent and dtype, and requires exact failure agreement. Slots publish
sequentially, so their count does not enlarge the per-operation tensor count.
Cold preflight resolves the complete selected decoder sequence envelope without
opening sources or realizing communication.
Session and tensor publication element bounds cover the larger of complete logits
and the declared primary decoder boundary, plus any additional prediction capture.
A vocabulary narrower than the decoder hidden state therefore does not truncate
capture admission. Media-only tensor-reduction envelopes remain separate.

Assistant body definitions retain exact K/V head count and head width alongside
context positions and unit layout. Context append and proposal traversal validate
every layer against `[1, kv_heads, retained_positions, head_width]` before beginning
the parameter policy. Equal-sized tensors with another head layout cannot be
substituted for committed context. The same ownership evidence covers the pinned
DFlash and DSpark numerical fixtures with malformed-context rejection followed by
valid resident/leased context eviction and full/partial proposals.

DSpark TP construction partitions complete query/KV heads and feed-forward
channels, preserves replicated context/norm/gate parameters, and reduces branch
writes before post-normalization and residual combination. Complete assistant
source placement retains exact recipes and original target invocation authority;
selected native groups cannot widen cold communication limits. DFlash2 uses
explicitly accounted per-participant replicas. These placements execute through
the same controlled and uninterrupted proposal/verification driver. See
[speculative decoding](speculative-decoding.md) for the application contract.

Retained target discovery exposes its declared tensor group paired with its
complete communication manifest. The MLX invocation adapter validates that
manifest against the target session's communication owner and borrows the
already realized group. Before lending it, the adapter checks the assistant's
operation, dtype, input/output extent, tensor count/rank and exact-completion
requirements against that group's retained admission. Backend capability does
not enlarge the selected limits. This check precedes assistant unit acquisition.
The adapter constructs no assistant communicator and accepts no caller-provided
substitute group. It does not select assistant source placement, enlarge target
communication requirements or grant public multi-rank speculative admission.

The cases cover TP2, PP2, PP3, EP2 and combined TP2/PP2/EP2 with resident,
host-layerwise and disk-streamed target policies. Pipeline cases execute declared
media-to-decoder return routes and decoder continuations. Only pure-TP DSpark
uses a sharded assistant body; the remaining cases hold explicitly reported
complete assistant replicas on each participating rank. These neutral fixtures
do not select public distributed assistant source placement or admit complete
assistant communication requirements, asymmetric failure recovery or native
released-checkpoint sessions.

The tensor contract exposes stable final-axis top-k selection and strict
final-axis scatter. Selection preserves score precision and breaks ties by
ascending original index. Scatter checks bounds in the original I32/I64
representation and rejects duplicate indices. MLX retains device validation in
the completion scope and sanitizes invalid indices before native indexing; a
deferred failure cannot publish a result. Spark neural score composition maps
DSpark's signed offsets into the target vocabulary before shared sampling and
verification. DFlash2 forms each conditional distribution from the actual sampled
predecessor and global candidate IDs. The neural helpers neither draw tokens nor
own a second proposal cursor.

Before materialization, the compatible external preparation derives one
neutral selection contract. Architecture-owned target profiles supply the
stable target/geometry identity and context ceiling; the admitted assistant
checkpoint supplies artifact and physical-format identities. The contract receives
an explicit positive decoder capture bound. Execution-plan preparation takes
that bound from the retained target partition's primary decoder boundary, or
uses the architecture ceiling for an unpartitioned target. The effective capture
extent is the smaller of the invocation bound and architecture ceiling; it also
enters speculative state identity. Media route extents do not replace decoder
invocation limits. The facade supplies its tokenizer proof, requested proposal
capacity and processor identity. Concrete backends must not synthesize family,
artifact-format, capture identities or invocation bounds.

Prepared target capture placement retains the same selected decoder batch and
sequence limits. Resolution rejects zero, unrepresentable or oversized dimensions
before creating a transport bundle, while preserving producer ownership and
complete hidden width. It cannot enlarge the retained invocation by substituting
a model's longer context length.

The selected capture schema fixes paths, order, ownership, observation seams,
fixed dimensions, and upper bounds for request-sized sequence dimensions.
Immediately before target execution, each lane binds its exact prepared-input
identity. Each committed target pass then closes the actual positive sequence
extents and target-cache generation, and the architecture executor validates
that capture against the retained realization before proposal state can
advance. A stale generation, reordered path, wrong fixed dimension, oversized
request dimension, or cache/input mismatch fails the transaction. External
assistant targets are replicated-only; partitioned target requests fail during
architecture contract selection before assistant payload or queue creation.

Prediction-enabled composition is additive to the ordinary target session.
Architecture admission projects the target and a typed prediction-extension
plan from one already-admitted artifact; it does not reopen or reclassify the
checkpoint. The ordinary neutral session remains the sole owner of target
prefill, decode, mutable state, prompt-cache control, completion, observation,
rollback, and publication. A target pass may retain an architecture-declared
hidden capture from that same transaction for an embedded predictor or
external assistant. Missing capture fails the transaction before publication.
Architecture composition owns prediction-local state geometry, identity, and
transaction semantics; a backend owns only its native storage and prediction
operator mechanisms. It must not construct a second target model or target
cache. Any extension whose unit execution lacks a typed additive handoff
remains fail-closed; loading a second complete model beside the neutral target
is not an admissible compatibility mechanism. DeepSeek-V4 DSpark uses its
configured, ordered intermediate-layer captures and exact fused block capacity
through that typed handoff. Ordinary final hidden state is never substituted
for the declared capture.

Controlled speculative forks reuse the core's exact canonical snapshot contract.
Runtime owns scope-local branch handles, logical run identity, bounded serial
exchange and non-rewindable accounting. Immutable source snapshots may be shared;
activation copies through the same backend snapshot/restore mechanism. Anonymous
retained branch copies share `SnapshotBudget` without allocating extra user-visible
snapshot or branch counts. No backend schedules child runs or replays the prompt.

Prospective token restrictions use runtime's ordinary `TokenChoiceController`,
including grammar checks and absolute speculative positions. Ordinary and speculative
samplers share temperature/RNG validation. Core snapshots retain configuration,
choice, both RNG streams and semantic state; the native adapter supplies only seed
construction and existing sampling primitives. Tensor-edit discovery is a distinct
speculative capability, re-admitted against the loaded source/session. MLX exposes prediction-row logits for each model role and uses the shared `CaptureSession`
and intervention engine for transforms, evidence and cumulative budgets. Selecting
a role's immutable plan between drained rows never resets that ledger. Additional
layer/routing hooks require architecture-owned speculative attribution before they
can appear as supported in discovery.

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
The retained public distributed-session view and the partition runtime share
one manifest-derived deadline and poison authority. Public collective
completions retain their inputs, outputs, count buffers, exact group, routes,
and streams; on timeout an explicit event-backed quarantine keeps those native
resources alive until completion while every later submission in that session
fails before reaching the transport.

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
- Shared-decoder construction enters through `CompleteDecoderKernel` and
  `DecoderKernel::from_partition`; execution uses `ComponentAdapter`,
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
MLX copies a received byte payload into aligned output storage before viewing
it as a wider dtype when the header length is not divisible by that dtype's
width. This uses the admitted logical output extent, retains the frame and
output through the same completion, and adds no host transfer or wire padding.
An unaligned byte view is insufficient even when a host byte copy can read it:
typed native kernels must consume the exact received values.
The byte materialization uses U8 addition of zero because MLX's `Copy` primitive
shares the original allocation. The received frame remains retained, preventing
donation of its unaligned payload storage to the typed output.
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

For prediction-free dense Llama/Mistral/Phi-4 artifacts backed by indexed or
unindexed SafeTensors, or by an admitted GGUF checkpoint, pure TP, pure PP, and
TP+PP use an architecture-owned partition dispatch and executor under selected
fully resident, host-layerwise, or dense disk-stream residency. The backend
binds one selected layerwise policy shared by execution and reporting; it does
not construct a second cache or residency-specific family executor.
Selected Llama/Mistral/Phi-4 load-time transforms use this same route. Architecture
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
materialization, and causal acquisition/eviction/reload telemetry.
Each addressable member retains its exact neutral materialization task together
with the member-local recipe, selected output, source and selected byte totals,
and any scale or affine-bias output companions. Rank-local projection preserves
that closure rather than reducing a bank to aggregate bytes or one bank-wide
quantization flag. The MLX mechanism lowers these tasks into per-binding
transformation descriptors, so direct and transformed bindings may coexist in
one bank. It validates source provenance, executable format, lowering,
companion ownership and dtype, recipe output, and the neutral selected byte
total before any checkpoint read or residency-manager construction.
Compound
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
the original poison. Ordinary pipeline waves first agree execution before preparing
boundary payloads. A completed negative execution vote is retained as evidence
that all participants reached that boundary; this logical rejection rolls back
without inventing a failed transport. It does not enter the recovery bypass or
clear poison. Boundary source-completion and actual communication failures retain
their existing terminal fencing. The reusable MLX
communication backend realizes this payload-free contract with an internal
scalar integer all-sum. Its completion polls the exact event without taking a
blocking runtime lock, retains arrays, count storage, groups, routes, and
streams, and defers the host boolean read until that event completes. Because
upstream MLX exposes no event abort, timed-out work is retained in a
thread-affine quarantine and its native communicator cannot be realized again
until a later safe reap observes terminal native evidence. Failed submissions
also retain these resources without a returned public event. If the
originating thread exits first, teardown does not block: unresolved arrays,
count storage, groups, routes, streams, and native execution owners remain
retained permanently. This bounded-teardown policy can retain memory after an
unrecoverable device failure; it never converts a missing completion proof
into process termination or unsafe destruction. Complete all-rank manifest consensus also precedes
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

Neutral communication preparation first proves complete compatible manifests,
operation capabilities, logical reachability, and complete-world group and
route waves without creating native resources. Opaque MLX setup then acquires
the process runtime lock through the prepared manifest's absolute submission
deadline and rechecks it immediately before the first
array, frame, or collective graph mutation. Lock contention therefore fails
before native submission and poisons the one partition-communication authority.
Manifest-owned subgroup realization never enters upstream MLX's synchronous,
non-abortable native split: it retains an exact logical membership view of the
world handle. A logical view may use a world-wide operation only when manifest
consensus proves that the creation batch partitions every world rank into
same-requirement groups; non-neighbor route relays additionally require a
disjoint, contract-identical route batch covering the world. Partial,
overlapping, optional, unreachable, or contract-mismatched batches fail in
neutral preparation before any native callback. A variable all-to-all in such a proven wave expands each logical
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
an explicit reap. Owner-thread teardown makes only a nonblocking retirement
attempt. If terminal evidence or runtime access is unavailable, its retained
resources remain owned permanently; thread-affine work is not transferred to a
reaper thread. An asynchronous error alone never authorizes reclamation.

`DistributedSession` is an optional capability of the selected model session.
It exposes high-level sum, gather, variable-count exchange, point-to-point, and
consensus submissions with exact completion. Unsupported operations report an
explicit absent capability. Communicator construction, sharding, movement, and
collective tensor math remain backend-specific.
The retained MLX public view and neutral partition runtime are two consumers of
the same manifest-derived communication authority, including its deadline and
poison state; neither constructs a detached authority. Public completions keep
their exact inputs, outputs, count buffers, opaque groups, routes, streams, and
shared authority context alive until resolution, even after the public session
handle is dropped. Singleton manifest groups remain contracted operation
handles rather than disappearing from an advertised capability.

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

The canonical facade exposes this choice as
`PreparedChatGenerationSettings::strategy` (also re-exporting `TextSamplingStrategy`
from `eredu::api`). Standard sampling remains the default. Mirostat validates finite,
positive tau, eta, and effective temperature after checkpoint defaults and request
overrides are resolved; `do_sample = false` therefore cannot select Mirostat. It
retains the resolved seed and penalties, replacing top-k, top-p, and min-p. The
facade's vocabulary and tool masks apply before sampling, so adaptive updates use
the resulting constrained distribution. Ordinary, observed, intervened,
controlled, and prepared speculative generation share this resolution. Speculative
batches may mix strategies and retain independent sampler state per lane. The MLX
adapter shares sampler construction with ordinary generation and delegates logits
processing and committed-token updates to the neutral runtime policies. Mirostat
updates from each committed constrained target distribution, including accepted
proposals, replacements, and bonus tokens. Its policy disables exact optimistic
promotion without disabling drafting or verification; standard lanes retain their
existing lookahead capability. All lane settings are validated before backend
prompt preparation or execution.

```rust,ignore
use eredu::api::{PreparedChatGenerationSettings, TextSamplingStrategy};
use eredu_core::GenerationConfigOverrides;

let settings = PreparedChatGenerationSettings {
    overrides: GenerationConfigOverrides {
        do_sample: Some(true),
        temperature: Some(0.8),
        ..Default::default()
    },
    strategy: TextSamplingStrategy::MirostatV2 { tau: 5.0, eta: 0.1 },
    seed: 42,
};
```

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
11. run the relevant shared execution conformance and independent family acceptance.

Facade selection is an upward dependency on the backend's adapter, not
ownership of concrete composition: the facade composes generic model and request
APIs with the adapter, while the backend must not depend on facade APIs or
orchestration.
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
- the runtime's `backend_independence` target and evaluation's shared execution
  drivers cover bounded/partitioned execution, completion, cancellation,
  speculative commitment, snapshot isolation and consumed budgets;
- native facade integration compiles the published facade target and exercises
  representative public lifecycle paths; shared native Ring drivers cover rank
  agreement and transport rather than repeating a matrix for each family; and
- complete-model numerical acceptance compares prefill score rows and cached
  continuations with independent pinned references. Quantized comparisons use
  the same encoded artifact; media and assistant inputs retain their exact
  histories. The [family acceptance index](family-acceptance.md) distinguishes
  recorded results, available routes and unvalidated scopes.

Checked-in family verification retains compact source/result manifests,
independent reference entry points and shared execution conformance. Family
unit tests, scalar reimplementations, component numerical fixtures and expanded
format/topology/residency matrices are temporary investigations outside the
tracked tree. They do not substitute for released-checkpoint numerical evidence.
Downloaded weights, captured tensor payloads and detailed traces remain outside
the repository. Compiler types, visibility, manifests and feature selections
establish ownership and API boundaries.

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

## Independent projection banks

The shared decoder's `DecoderProjectionOperator` supplies both optional values
before attention and the feed-forward projection after attention. Q/K projection,
rotary position handling, normalization, masks, ordinary KV storage, output gates,
output projection and residuals remain in `eredu-architectures::decoder`.
Architecture block factories select these stages and their parameter topology;
native backends receive ordinary operator specifications.

`eredu-nn::GroupedLinearSpec` describes independently selected complete input
projections, a per-expert pointwise activation, and an owned contiguous output
range. Its output is the weighted sum of activated expert projections. Output
partitioning retains the entire input axis, so nonlinear activation occurs after
a complete projection. Unlike a row-parallel feed-forward down projection, this
owned output does not require a tensor-parallel sum. MLX realizes the contract
with generic indexed matrix multiplication and pointwise operations. CPU reduced
precision uses ordinary matmul over contiguous selection runs and views of the
selected matrices, without expanding weights per token or widening a complete
expert bank.

Every routed provider request and control callback carries a `RoutedBankId` in
addition to its logical layer. `RoutedObservationPoints` maps bank identity to
its canonical observation path and global cardinality. `RoutedBankProviders`
retains independent providers and observation identities. Their storage handles
share the selected residency pool, while acquisition and completion leases protect
individual entries. Heterogeneous prepared providers may be erased behind
the same neutral provider contract. A missing bank fails before dispatch.

Grouped RMS normalization retains one full-width checkpoint scale while reducing
independently within each group. The native realization promotes input and scale
to FP32 for reduction and multiplication before restoring the input dtype.
Rotary specifications retain an explicit amplitude independently of frequency
scaling. Architecture normalization resolves family conventions before passing
that amplitude to a backend. `RotaryArithmetic::InputProducts` additionally
specifies input-dtype rotary products, while `Native` allows the backend's fused
rotary arithmetic.

`eredu-checkpoint::fp8` normalizes block-FP8 and compressed-tensors metadata into
physical block geometry, dequantization-scale names and explicit module
exclusions. Architecture schemas assign that policy to actual Linear parameters
and retain canonical companion recipes; embeddings remain a distinct operator.
Shared decoder construction consumes complete `LinearFormat` values, including
FP8, rather than treating the absence of integer quantization as dense storage.
Encoded input-axis alignment participates in the same portable partition rules.

Routed preparation and selection retain a `BTreeMap<RoutedBankId, ...>` of exact
bank contracts. Each entry owns its grouped equation, expert ownership, catalog,
route counts and selected materialization tasks. `PreparedRoutedTextArchitecture`
and the shared routed-session constructor consume the complete collection;
there is no separate ReLU-squared prepared architecture. Native bank binding is
called once for the complete selected collection and has no family branch.
`ParameterBankKey` includes bank, execution unit and global member; the backend
uses this neutral identity directly. Scoped handles share one residency manager
and the configured host/device budgets, so switching banks can evict entries
from either bank. Runtime reports retain per-bank placements and counters, while
aggregate peak occupancy counts each shared physical pool once.

Routing specifications independently select projection, score and coefficient
precision. Scaled softplus is a neutral operation with explicit beta and an
input-dtype output; native implementations may widen intermediates before one
final cast. These policies belong to portable equations, including cache identity.

Expert-exchange waves carry an architecture-selected `ExpertRouteInvocation`
with bank identity, logical unit, access class and local output width. The neutral
transport validates input and result geometry independently; projection banks
can therefore return fewer columns than they consume. Owner-local execution IDs
remain distinct from global `(bank, unit, member)` storage keys throughout the
forward and reverse route permutations.

Partitioned preparation retains the same identified bank collection and builds
one provider collection for resident or addressable execution. `ExpertOutput`
parameter groups combine ownership of the expert axis with ownership of output
rows; their complete input axis is preserved. Collective schedules retain each
bank's input/output geometry and order within a block, and pipeline cuts occur
only after a completed block. Ordered expert reduction sorts returned routes by
global expert identity before scatter, preserving the selected accumulation
precision across expert exchange.
Native logical subgroups share a physical world wave even when an inactive stage
uses F32 placeholders and an active cohort computes in F16/BF16. MLX transports
these floating values through a common F32 encoding, which represents all three
supported floating types exactly. Routed/direct sums restore each contribution's
logical dtype before addition; fallback world sums transport distinct per-rank
slots before logical reduction. Gather and variable exchange restore the original
logical dtype on return. Integer control values retain their integer encoding.
This transport encoding does not select model equations or change pipeline-wire
contracts; intermediate transport buffers remain native mechanism storage.

Ordinary KV and hybrid KV/fixed-state snapshots copy strided tails through
contiguous views and copy every sealed paging block into an independent session
namespace. These managers share the source process pool, so branches retain the
same finite host/device limits. Snapshot admission includes sealed blocks across
all tiers, catalog metadata, tail arrays and conservative growth through future
block boundaries. Completion/accounting ledgers remain outside the exchanged
model state. Native tests compare interleaved branches against an independent
attention calculation for full and sliding paging.

Selected grouped-linear materialization derives compact field identities and
storage dtypes from the declared projection format, including byte FP8/native
GGUF weights and integer MXFP4 scales. Architecture recipes preserve both
logical projection dimensions and encoded byte shapes. The neutral lowering
contract validates those distinct geometries before a backend accepts a derived
native-block source; source catalog ownership remains in `eredu-checkpoint`.

Ordinary distributed snapshot copies complete their native work and pass a
portable all-rank preparation agreement before publication. Branch exchange
validates every rank before moving state; the existing native communication
owner retains completion resources. Live commit epochs, communication/capture
budgets and copy ledgers are never part of a rewindable branch. Independently
addressable bank observations are added once to ordinary parameter-memory
reports; sharing a physical pool does not duplicate its owned entries.

Cold parameter resource totals are derived in `eredu-runtime` from the retained
executable tasks, including attached physical companions once. Architecture
selection supplies the independent-bank identities; shared experts remain in
ordinary block windows. Artifact profiles retain their unsharded scope, while
native session observations report actual rank-local replicas and bank occupancy.

Automatic schema version 7 also retains a separate selected-rank resource
profile. K2's tensor-free physical parameter description feeds the same TP/EP
placement primitive used by construction. Runtime sizing applies each output's
own physical geometry, including scale companions and replicated global routers;
PP ownership selects static consumers and repeated units before aggregation.
KV accounting uses only locally owned layers and complete KV heads. Source-recipe
workspace uses the existing bounded binding-placement rewrite for both ordinary
parameters and individual bank members. Compact-bank bounds preserve each bank's
independent identity and the selected finite assembly budget. Native recipe estimates include source copies, intermediate tensors, indices
and contiguous outputs. Separate conversion estimates count encoded outputs;
queue concurrency and per-bank compact bounds are explicit. These are conservative
tensor-buffer estimates, not bounds on opaque driver/allocator bookkeeping.

The vendored MLX CPU BF16 matrix kernel accumulates complete dot products in FP32
with explicit row-dot and strided-column reduction trees. It does not widen
or expand a bank's weights. Batch row count does not change accumulation
precision, and bank slicing preserves the projection's row-dot layout. This is a native matrix mechanism; it carries
no family configuration, checkpoint names, routing policy, or layer equations.


Independent external drafting uses architecture-owned inspection and preparation
of a second ordinary model. The source graph, lowering, tokenizer proof and
selected placement remain paired until native materialization. The shared runtime
owns private proposal branches, full-sequence target verification, accepted-prefix
replay and joint rollback. Ordinary sessions expose complete sequence logits
through their existing publication/completion transaction. Backend adapters bind
the selected ordinary executables and isolated state copies; the facade retains
one speculative sampling, termination, observation and controlled-session driver.

### Scalar component instrumentation

`eredu-core::component` holds compact, versioned component identities and semantic
parameter relationships. Shared decoder configuration also supplies parameter
aliases to the constructed neural slots, so repeated invocations share edit
authority while retaining separate execution units and mutable state.
`eredu-architectures` derives component relationships from the same decoder
configuration used to construct projections, including gate/value reads, GQA row
mapping, fused segments and shared invocations. They remain declarations; runtime
capture/intervention reports describe support for the selected execution.

`LayeredArchitecture::forward_unit_observed` lends the existing observer to the
architecture's internal boundaries. The shared decoder owns normalization,
attention aggregation, FFN activation and residual timing. It records originals,
applies interventions to values consumed downstream, and offers distinct bounded
`.effective` observations. The runtime owns traversal, ledger admission and
pre-work activation-edit reservations; MLX implements masks and reports native
storage/copy allowances. No family equation or analysis algorithm moves into the
backend. See [component analysis](component-analysis.md) for current coverage and
outstanding integration requirements.

Ordinary non-gated FFNs use the architecture-owned `decoder::unary` equation
(affine read, activation, affine write). Nemotron-H dense units reuse it for
ReLU², including input/output biases. Its fixed-state replicated adapter forwards
observers through the same physical attention and FFN boundaries as its full
architecture. The resident driver forwards internal unit observations as well as
input/readout hooks. Observation stays optional and does not introduce extra
component evaluation when disabled.

The native key/value profile includes an explicit stateless entry for physical
FFN invocations. It retains no arrays and rejects cache updates; snapshots,
transactions and paging preserve these entries without allocating cache blocks.
This realizes the neutral layout rather than inferring state from a family name.

The same layered traversal lends observers to `begin_forward_observed` and
`finish_forward_observed`. Architecture implementations own the scaled embedding,
final residual, normalization and affine-score boundaries. Traversal hooks compose
observation and intervention without adding another forward driver; disabled
hooks use the ordinary methods. `ComponentReadout` declares the output equation,
including nonlinear post-transforms and intervening block normalizations. Host
signed contribution calculations remain in examples and `eredu-evaluation`.

`eredu-core::parameters` defines loaded parameter facts, bounded rectangular
queries, logical-axis projections and immutable multi-parameter edit admission. The runtime owns cumulative
reservation and traversal through architecture-declared static modules and
selected residency units. `PreparedParameterSlot` retains the exact recipe output
metadata and architecture owner at construction, allowing loaded discovery without
building units or allocating tensors. The neutral parameter-operation loan passes
the semantic group/unit address to the selected policy; native completion retains
the unit and its transfer through query or edit evaluation. Random-access
inspection does not advance the ordinary disk-forward transfer window. `eredu-nn::ParameterSlotVisitor` carries stable parameter metadata;
it does not reopen checkpoint sources. The MLX realization performs native region
copies, contracts bounded host directions against effective weights natively,
and prepares affected-parameter replacements before publication. Packed matrices
are decoded using the retained materialization format and architecture-declared
companion roles; an admitted edit publishes an F32 copy of only its affected
matrix. The native projection/embedding selects that floating handle until
restoration. Full conversion and per-edit destination costs are reserved first.
Original parameters remain within the session's completion-retained payload.
Bounded transactions prepare each affected unit separately and retain only their
charged original/replacement handles. One publication updates the selected
populator's immutable override set plus pinned static slots; every subsequent
unit reload applies that set after its original exact binding. Removing it restores
pinned/resident handles and resumes the untouched prepared sources for bounded
units. The ordinary residency ledger accounts for unit materialization separately
from cumulative query and overlay work reservations.
Publication clears incompatible state and invalidates native snapshots; facade
session identities invalidate outstanding prepared requests. Parameter queries and
edit records carry the effective execution version. The component-analysis support matrix records packed-format verification and
residency/partition coverage.

Schema-admitted redundant sources remain distinct from executable parameter
sources. A tied parameter may retain exact redundant source keys in its neutral
requirement; strict native source-use validation consumes those keys. Removing a
redundant output-head materialization also removes its own companions, while the
executed embedding retains its original companion contract. Backends infer none
of this from family or checkpoint-name patterns.

The shared static-module readout owns optional residual, normalization and affine
score hooks. Both dense and heterogeneous replicated decoders call that helper;
Nemotron-H declares its embedding and final norm identities from family policy.
Draft prediction paths retain their distinct normalization timing.

Native GGUF parameter decoding preserves the retained byte order. Little-endian
optimized kernels remain available; big-endian dispatch uses endian-aware scalar
loads in the general kernels and canonical host decoding. Effective-parameter
verification covers byte order as well as packed codes and scale companions.

Component read declarations also retain architecture-owned Q/K head normalization
geometry and gain-sharing policy. The declarations mirror the normalization
specifications used by shared attention construction, including independent
per-head gains and shared head-width gains. This interpretation does not move
into parameter-query or backend code.

Query/key component dependencies use checked full-head row ranges, distinct from
scalar value-channel reads; the contract permits unequal read and value widths.
LFM2 owns its mixed attention/convolution component seams and the normalization,
residual and state-update order shared by full and replicated module execution.
Convolution writes remain separate undecomposed residual terms. Dense FFN hooks
share the existing gated equation, and routed providers retain their observations.

Readout declarations identify undecomposed residual writes by architecture node,
invocation, original/effective observation and residual scale. Consumers can account
for mixed convolution and routed-expert contributions without parsing path names.

LFM2 configuration admission accepts the published `block_ff_dim` width and
`full_attn_idxs` schedule, as well as their normalized counterparts. Architecture
normalization derives exact dense widths and state schedules and rejects conflicting
schedule declarations; a backend never interprets these source-format aliases.

Nemotron-H's shared physical-unit driver observes complete Mamba and routed-plus-shared
expert writes before residual addition. Architecture readout declarations join these
undecomposed terms to their operator nodes; existing routing captures retain their
original timing. Ordinary execution uses the same driver with instrumentation disabled.

Nemotron's always-executed shared expert also declares individual ReLU² units,
canonical up rows and down columns, optional biases, and the block's RMSNorm gain
and epsilon. Its sibling `shared.feed_forward` scope reuses the ordinary unary
component driver in replicated and tensor-parallel execution. Shared input edits
affect that branch; effective shared units feed the down projection, and the reduced
effective write enters the routed sum once. The shared group inherits its sparse
operator's parameter ownership. Existing neutral partition coordinates and native
parameter mechanisms consume these declarations without a backend family branch.

Independent expert banks retain architecture parameter declarations alongside
their exact lowered member bindings. Runtime joins these into bank-owned
`PreparedParameterSlot` records, preserving original member identities, companion
relationships and local physical geometry. Native parameter loans use the same
prepared cache manager as inference, copying selected members sequentially under
completion-safe transfer ownership. They do not reopen artifacts or retain every
source member in the residency cache at once.

An overlay publishes ordinary module handles and independent-bank replacement
handles together. Bank providers select those handles while constructing compact
groups; immutable checkpoint-backed cache entries remain available for restoration.
The bank reports updated compact-member byte costs to the neutral chunk planner,
including packed-to-floating promotion. The operation's parameter budget owns
retained original/replacement matrices, while bank residency and compact-scratch
limits continue to govern inference. Compact binders accept floating weights only
at the logical shape supplied by the neutral operator specification; packed
weights retain their encoding geometry and companions retain exact shapes. All
bindings validate before compact handles change. Native and neutral acceptance
for this path is tracked separately in the component-analysis guide.

Neutral parameter placement also represents shared logical chunks with a short
final physical chunk. Architectures choose the chunk width from the selected
projection format; runtime validates matching unit counts and remaps encoding
companions. The architecture planner lowers these declarations to ordinary
contiguous ranges or ordered segment indices. Native binders consume those exact
placements without inferring quantization boundaries. Dense FP8 FFNs use this
contract for partial intermediate-axis blocks, preserving the same ownership for
read rows, output columns and scale companions. Independently encoded fused
gate/up tails require their own explicit scale geometry; the chunk contract alone
does not establish support for that layout.

`eredu-runtime::validate_parameter_layout` validates supplied physical
layouts against typed family-selected group geometry. `ParameterGroupSelection`
selects a logical group interval or an explicit range of a declared balanced axis.
It uses the same member lowering as rank-based placement for uniform partitions,
fused segments, explicit chunks and balanced axes, including physical companions.
It rejects missing or extra parameters and mismatched shapes, group metadata or placements with typed errors.
`logical_parameter_group_range` extracts the declared semantic interval from a
partitioned member's retained metadata, including explicit chunks. EAR encoder
and decoder local geometry uses this extraction before complete layout validation.
RVQ, MoG, character/speaker conditioning and EAR encoder, decoder and PRVQ geometry
consume shared validation. Families retain coupled latent, feature and
convolution ranges and their state geometry; native construction consumes the
validated physical tensors. RNNT and the function head select complete vocabulary
axis ranges through this same validator. Their communicator validation remains
separate, since an uneven interval does not identify a rank or degree. Function-head
format admission retains its requirement that every physical companion preserves
the complete vocabulary axis. Packed expert ownership retains its rank-aware
placement contract.

`eredu-nn::LinearRowLayout` declares independent row-block origins in a projection.
Architecture format specifications carry that declaration into parameter metadata;
the equal-partition form retains its meaning after matching gate/up slices change
each partition's local width. Runtime expands scale geometry and segment coordinates
from those origins. Native grouped projection kernels consume the declaration while
sharing the ordinary quantized input, and effective-parameter decoding and scratch
admission use the same scale rows. Backends do not infer this layout from checkpoint
names. The K2 checkpoint convention declares two independent row partitions for its
fused FP8 gate/up source recipes.

Shared decoder attention placement derives one GQA-head partition from the joint
alignment of query/key/value read rows, optional gate rows and output columns.
The architecture configuration declares the value projection's format even when
a routed bank supplies it. Split attention retains partial final head chunks;
ordinary projections, per-head parameters and separate value banks consume the
same ownership. Scale companions use primary geometry instead of ratios inferred
from rounded scale shapes. Cold state sizing and complete partition state retain
the actual per-layer local head counts, which can differ between dense and FP8
layers. Native binders consume these retained ranges without selecting head cuts.

Additional diagnostic tensors use a deferred observation factory. Architectures
supply the observation identity, native mechanisms describe exact prototype geometry
and conservative creation storage, and runtime invokes the factory only inside the
ordinary capture transform after reservation. Absent/skipped observations perform
no factory work; generated handles are released before returning to inference.

Neutral projection operators optionally expose the actual multiplication input.
The default is identity; input-transforming native mechanisms override it.
Architecture component seams select observation paths without selecting native
quantization. MLX reuses the activation codes and scales already needed by FP8
inference, offering their F32 reconstruction as a deferred diagnostic on GPU;
the CPU reference path already needs that reconstruction. Loaded parameter
facts retain the selected input transformation, including its change when an
admitted F32 overlay replaces a packed matrix. Research consumers account for
final-head input quantization separately from residual component contributions.


Capture source precision is a neutral optional `TensorDtype` fact, independent of
host payload conversion. `CaptureBackend::source_dtype` reads metadata only;
runtime records it through the shared capture/evidence driver. Deferred values
report their own dtype only after generation, never their prototype's dtype.
MLX reuses one native scalar mapping for capture and communication.

Native parameter publication retains the effective dtype upper bound alongside
the overlay identity. Loaded runtime-state estimates combine it with the original
state-width assumption, covering promotion by F32 matrix replacements without
family-specific dependency analysis in the backend. Removal clears both the edited
state and this bound. Snapshot/copy accounting continues to use actual tensor
sizes. Publication and removal return parameter facts for the resulting operators,
including their current projection-input transformation.


Component read mappings distinguish a scalar row from a complete query/key head.
An optional head stride and scalar `Blocked` mapping describe interleaved fused
head segments without exposing checkpoint-name parsing to applications. Absent
head stride denotes contiguous geometry, including in serialized records. Range
construction rejects zero widths, overlapping strides and arithmetic overflow.
Architecture declarations supply the mapping; backends still receive only exact
bounded parameter regions. Hybrid-family declarations and shared execution hooks
have neutral and native coverage in the component-analysis matrix.


Qwen hybrid target blocks share one instrumented residual driver between
resident and rebuilt units. Their dense SwiGLU units and gated self-attention
channels reuse the ordinary decoder hooks; the gated-delta mixer and routed/shared
expert sum remain explicit whole residual writes. Query/gate reads describe the
actual per-head interleaving, and Q/K, sublayer and final RMS normalizations retain
the learned gain offset of one. Full and typed replicated text adapters forward
observers through the same native-independent equations and readout. Existing
routed provider ownership stays in runtime. Multimodal conditional propagation,
recurrent channels, routed units and partitioned execution use their declared
component boundaries; the component guide records their verified configurations.

The shared decoder's physical-format declaration permits an unaligned final FP8
block in a complete projection. Runtime retains such a group as one logical
partition unit and still expands exact ceiling-sized scale companions. Aligned
groups retain their existing finer units. Strict packed formats keep their
alignment checks. This permits ordinary resident and bounded loading without
allowing a rank slice to start inside an encoded block; uneven tail-aware TP
partitioning needs additional placement support.

Shared tensor-parallel decoder blocks expose the same rank-local channel and
FFN-unit boundaries before their existing row-projection reductions. Normalized
inputs and reduced residual writes retain their ordinary block equations. These
operator seams add no collectives; global component coordinates, capture delivery
and parameter transactions remain responsibilities of the partition driver.
Operator seams alone do not establish public partitioned component support.
The parallel layered contract has an internal-observation unit entry. The
ordinary parallel residency traversal supplies its shared observer to that entry,
retains forward resources for completion, and keeps group/unit boundaries in the
existing traversal hook. Shared full/local decoders and composite kernels use the same adapter entry,
and the direct partition executor consumes that traversal.
Parallel layered traversal also invokes architecture-owned observed ingress and
readout when an observer is present. Shared decoders expose embedding, residual,
normalization, actual projection input and complete linear logits, with the
ordinary tied/untied vocabulary gather and output softcap. Neutral vocabulary
projection input observers delegate input transforms to the backend; MLX uses the
same deferred FP8 evidence and failure propagation as ordinary projections. An
absent observer does not construct component paths or request transformed input
copies. Public capture requires selected global coordinate admission, pipeline
propagation and transport support in addition to the operator hooks.

Direct pipeline and composite unit strategies forward the supplied observer
to the selected architecture's serial or tensor-parallel internal-unit entry.
Composite kernel capabilities supply observed serial and tensor-parallel
ingress through the same adapter. Nemotron image ingress exposes the normalized patch embeddings before
the first vision layer through this contract.
The partition executor retains its existing boundary, residency and completion
sequence. Local shared decoder blocks validate ownership and index local state
before using the shared observed equations. These entries propagate unit internals;
the ordinary pipeline driver also forwards observed ingress and final readout.
Its existing optional-observer implementation shares state-layout validation,
group entry/exit and output ownership. Shared decoder ingress marks embeddings
only when it actually looks up tokens; upstream hidden boundaries are not
relabelled as embeddings. Final readout runs only on its output owner.
Specialized routed strategies and global capture transport
require further integration.

The default bounded serial and parallel traversal-hook entries also dispatch
architecture internal-unit observers when the hook requests activations. This
uses the same residency traversal and completion sequence. Explicit custom unit
executors keep authority over their own internal observations; the default unit
dispatch does not replace specialized provider calls. Unobserved hooks continue
through ordinary unit execution.

Global scalar-axis identity belongs to `eredu-core::component::ComponentCoordinateMap`.
Its validated range or ordered-index representation maps local storage back to
global component IDs and rejects duplicate or invalid global masks before local
filtering. `eredu-architectures` compiles these maps from the invocation's logical
write projection, semantic partition units and retained execution ownership.
Packed physical columns do not define scalar offsets, and shared source aliases
do not confer ownership of another invocation. `SelectedExecution` exposes that
descriptive layout without exposing its private family-selection branches.
Runtime lowers compact masks and explicit token slices through those coordinates,
preserving empty keep-only sets and global identity validation. This lowering
performs no native work and grants no capture or intervention authority. Global
budget admission, rank agreement, delivery and parameter transactions remain
separate requirements before public distributed capture can be advertised.

Core `CaptureSlicePartition` validates bounded native fragment slices and their
destinations within the original global selection. Contiguous partitions retain
the global stride origin; ordered index maps become strided fragments without
exporting unrelated local columns. Runtime fragment capture borrows the original
admitted plan and uses the same native reservation/transform path as ordinary
capture and intervention evidence. Global raw-tensor assembly validates identity,
source precision and complete nonoverlapping coverage and reserves its host copy
before allocating it. Missing fragments and non-value outcomes cannot become
measured zeros. These mechanisms do not establish peer completion, producer
authority, transport budgets or publication agreement; the distributed session
driver must supply those before public support changes.

Core fragment geometry also owns allocation-free intersection of finite strided
coordinate sets. Runtime uses it to check coverage for raw tensors, summaries
and fixed-edge histograms; reduction assembly allocates by producer/fragment
count and bin count, not by global tensor width. Runtime combines native finite
statistics in F64, preserving exact count/non-finite semantics and documenting
reduction-order rounding. Ordinary and partition-generated observations share
one lazy native adapter, so reservation precedes factory construction and
source precision comes from the generated value. Backends continue to supply
only ordinary native transforms and storage estimates. These additions do not
supply distributed expected-producer authority or native completion agreement.

Versioned partition capture host-record schemas live in `eredu-core`. They contain
bounded execution/run/overlay/submission context and ordinary local capture
records; deserialization grants no execution or placement authority. Runtime owns
expected-producer receipt admission, canonical geometry/context digests, bounded
encoding/decoding, exact sender/ordinal/shape checks, one-shot receipt collection
and empty-selection acknowledgments. It reuses ordinary raw/reduced assembly and
preserves source precision and typed non-value outcomes. Native collectors still
produce the same local records. The distributed session must bind the context to
retained execution, establish global native quotas, and agree forward/publication
outcomes; receipt bytes cannot substitute for those owners.

Runtime `PartitionCaptureExchange` owns capture-specific transport reservations,
readiness/length frames, bounded payload gathering, and the final all-rank
receipt decision. It shares core's portable-word transport and exact bounded
completion contract with scheduler protocols. Backend `PartitionCaptureTransport`
adapters provide retained rank/policy facts, side-effect-free logical storage
estimates and the existing shared failure owner; they contain no capture geometry,
family semantics, JSON decoding, or publication policy. MLX implements those facts
on `MlxDistributedSession`, reusing its native gathers and completion owners.
Transport failures and unagreed protocol mismatches fence that same authority.
Fully agreed local/peer rejection returns a typed cause for the enclosing forward
driver to apply skip/rollback/termination policy. Producer and decoder work retains
ordinary capture reservations; prepaid control credits remain available after a
value-budget miss. The exchange must be admitted on every rank before work and
called only at a common post-forward boundary. Public distributed capture remains
unverified until session admission, producer quotas and generation commit use it.
Per-selection receipts share the immutable global capture plan through `Arc`,
avoiding copies of all selections for each captured point.

Flash-Next retains its immutable cold family construction with `SelectedExecution` so
other-rank discovery can project exact physical parameter layouts on demand. The family
reuses its selected-format tensor selections and expert ownership; fused recurrent ranges,
K/V replicas and encoded companions keep the same geometry as materialization. Generic
partition projection accepts that authored layout and still applies shared pipeline
ownership. It does not infer family geometry from backend tensors or reopen a checkpoint,
and it does not build an eager catalog for every rank. Prediction provider layouts remain
separate from the target parameter description. Typed routed prediction construction carries
its shared discovery slot through the binding and publishes it only after successful state and
weight assembly. The retained predictor description includes its selected global specification
and exact local layout; peer queries apply prediction TP selections while preserving PP/EP
replicas. This uses the existing prediction placement contract for both component capture and
parameter coordinates, without transferring equations or checkpoint naming into a backend.
Declared target
input/output boundaries retain their real execution-group and unit ordinal, including separate
n-gram injection units. The shared ownership compiler places these replicated four-dimensional
residual observations on their owning pipeline stage. Layer numbers are not used as execution
unit indices. Internal prediction observation placement remains a separate integration step.

The retained architecture selection also projects component layouts for any rank
in its admitted topology. `component_partition_layout_for_rank` reuses admission's
ownership compiler against the same retained global execution requirements; it
creates no state, device, communicator or execution selection. The local-rank
path consumes the already retained local ownership. `component_partition_layouts`
checks its rank bound before compiling one reusable map set; selection projection
then borrows those maps without repeatedly deriving full parameter layouts. `ComponentPartitionLayouts::capture_producers`
compiles one global selection into bounded rank/fragment declarations, choosing
the lowest rank for identical coordinate replicas of the same invocation while
retaining distinct shards with empty selected overlap. Backends do not reconstruct
pipeline ownership or derive it from shared checkpoint aliases. These layouts
remain descriptive: runtime receipt admission validates exact coverage and binds
them to a live run, native quotas and delivery policy.

The shared partition runtime constructor retains the complete parameter
description it already checks against the selected architecture and local
materialization tasks. `PreparedPartitionedSessionRuntime` transfers that
immutable declaration into `ReplicatedTextSession` through `Arc`; native
erasure forwards it for both text and composite sessions. Prepared discovery
retains the architecture descriptor, execution identity and cold partition
selection, then binds this checked description when the model is assembled.
`PreparedModelDiscovery::component_partition_layouts` therefore uses the actual
retained execution without revisiting artifacts or inferring global ownership
from local native slots. Compilation remains bounded by an explicit rank limit
and does not resolve the deferred artifact hash. This metadata does not by
itself enable distributed capture or interventions.

Transactional observation is part of the shared text-session lifecycle.
`ActivationObserver` supplies local preparation, coordinated admission, completed
delivery and an infallible final commit notification. Runtime checks uniform
participation before any observer collective, propagates local preparation
failure before model/state work, and invokes delivery after exact output/state
completion while retaining the ordinary rollback checkpoint. Delivery remains
provisional until the existing final commit succeeds. Early failure, rollback or
an uncertain commit discards provisional delivery through the shared guard.
Native observation adapters forward these callbacks; they do not implement a
second transaction driver.

Input preparation uses an earlier shared `InputPreparation` phase. The
`prefill_input_result_with_observer` and `decode_input_result_with_observer`
entrypoints accept a local preparation result so a rank cannot return from
prompt validation while its peers enter observer or model collectives. Ordinary
execution participates too. The fallible-input entrypoints require selected
bounded phase agreement on a partitioned session; final-commit support alone
cannot authorize them, including when a local input succeeds. An agreed rejection
consumes the step epoch, keeps
the installed state and prompt identity, and discards the observer transaction
before observer admission, checkpointing, or forward execution. Native adapters
carry token extraction, composite-input admission, generation parameter-version
checks, capture prompt alignment, and decode tensor framing through this result.
Framing stays under the existing native submission owner. Native recovery must
still establish settlement before reuse; state-preservation evidence is not a
completion signal. Successful ordinary generation prefill retains the core
runtime's exact admission check and execution-plan target proof.
`eredu_nn::Error::backend_source` preserves the typed original
preparation cause while keeping concrete error types out of portable signatures.

Architecture capture placement retains a `PartitionedObservation` separately
from scalar component groups. Each observation declares its own axis and local
coordinates. Normalized sublayer inputs follow their logical invocation and use
the full hidden axis; embedding observations follow input ownership, while final
residual, normalization and gathered affine scores follow output ownership.
Tied parameter storage does not transfer any of these invocations. Equal retained
replica maps select one exporter while all executing ranks retain hook agreement.
The scalar group maps remain the source of component-mask lowering.

Partition capture also distinguishes source participation from fragment export.
Architecture placement declares the exact ordinary source shape on every hook
member, including replicas and shards with no local selection. Core's capture
backend contract supplies cold dependency-completion bounds and bounded source
preparation. Runtime reserves those costs, seals shapes and costs into common
coordination, and agrees geometry before submitting any source graph. A distinct
prepaid vote settles capture success afterward; preflight alone cannot certify
completion. Globally empty and skipped selections submit no source work, and a
nonexporting generated capture never invokes its factory.

MLX completes ordinary prototype dependencies on all active ranks without host
tensor export. Its exact event retains the source, stream and native world, and
the nested recovery scope remains under the enclosing model submission owner.
Deadlines quarantine work and fence the shared communication authority; polling
errors do not release model or source ownership. Source reservations are separate
from transform/factory credits and survive failure and restore. Their logical
bounds exclude ordinary inference storage and private native allocator workspace.

Final `model.logits` retains distinct export authority. Architecture rank admission
selects the publication owner from the final group's semantic merge destination
and cohort; direct, routed, pipeline and composite plan construction consume that
same retained owner. Observation placement distinguishes exporter eligibility from
ordinary source coordinates. All output invocations can prepare dependencies,
while only the publication owner supplies final-logit fragments.

Runtime offers nonpublishing output tensors through `ActivationObserver::observe_replica`.
Its default does no work; ordinary inspection and intervention still execute only
at the authoritative seam. Partition capture uses the replica callback for its
prepaid source and status work. Native adapters forward the callback and retain
errors without selecting ownership. Capture timing includes source preparation,
transforms, coordination and final receipt delivery, including failed attempts.

Runtime also admits bounded `TokenScores` and `TopCandidates` receipts from one
complete ordered final-logits producer. It checks exact local and destination
geometry before source or deferred-factory work; shard-local reductions cannot
authorize a global probability. Native collectors use their ordinary full-vocabulary
algorithms. Runtime validates bounded payload identities, score/rank/competitor
consistency and optional domain bounds, then moves the reduction into the global
record with producer provenance. Decoder and output reservations precede their
allocations; final-logits replicas still complete ordinary dependencies without
exporting. Structural receipt checks do not prove numerical model correctness.

Preview assembly uses the same exact global coverage proof. Each positive-stride
fragment has increasing flattened destinations, so its first N local selected
values contain all of its contributions to the first N global values. Runtime
validates local truncation counts and payload precision, reserves the global
prefix, and checks its complete emitted count. Global selected geometry remains
separate from the flat preview payload. Native backends supply ordinary bounded
prefix transforms without deciding global order.

Declared block residual normalizations use their logical gain/bias parameter's
retained invocation ownership for input/output observation placement. Aliasing
that parameter to a shared checkpoint gain does not transfer the observation to
another invocation or pipeline stage. Architecture placement exposes complete
hidden coordinates; native collectors retain their ordinary replica preparation
and bounded delivery policy.

Observation coordinates do not prove that a selected driver emits the value.
Runtime exposes separate input, unit, routed-unit, readout and publication hook facts.
Architectures declare coverage for ordinary, tensor-parallel and partition
methods separately; default methods declare none. A partition executor combines
those declarations with its actual per-unit strategy. The shared runtime adds
its final-publication seam independently. Native erasure forwards these facts
alongside the retained global parameter declaration, and architecture discovery
checks the observation's declared execution site before consulting native
collector support. Missing internal hooks report `Unverified`; a coarse session
inspection capability cannot enable them. This mechanism is not a substitute
for implementing the remaining specialized family/provider hooks.

Sparse provider coverage is independent of complete internal layer coverage.
`routed_sparse_observations` and `parallel_routed_sparse_observations` describe
the actual provider-aware calls; `ObservationHookSite::RoutedUnits` selects that
fact without enabling other unit observations. Pipeline strategies forward their
own sparse coverage. Loaded MLX discovery combines this fact with retained sparse
placement, native routed collection, and exact invocation-group status transport.
It admits `CaptureTransform::RoutedUnits` through the existing shared observer.
Sparse interventions additionally require the native gather/scatter and coordinate
mechanisms. Architecture-owned `routed_activation_members` projects retained
expert/unit ownership and read widths. Runtime extends the same
`SessionPartitionIntervention` with sparse chunk coverage, pre-forward reservation,
source/final votes and committed global affected counts. The backend maps actual
native rows through preserved origin tags and applies ordinary tensor operations;
it does not choose expert ownership or edit semantics. Every physical replica and
received source peer is edited, while retained publication ownership determines
which logical effects count once. Disconnected native Ring groups still require
coordinated relays and retain their explicit capability rejection.

Parallel communication projection also supports an explicitly selected group of
all tensor and expert ranks at a fixed pipeline/data coordinate. These groups
follow existing axis groups and preserve their IDs. Architectures request exact
failure agreement for a combined expert/pipeline stage; an axis group alone cannot
certify every physical replica of an observation. MLX realizes connected Ring
stage membership with a member-only reduction and broadcast along neighboring
ranks. Intermediate sends use the existing bounded communication owner, including
retention and quarantine of arrays, groups and streams. Model tensor collectives
keep their selected wave protocol. The same exact membership predicate controls
cold capture admission and native status execution; disconnected sets still need
coordinated relay support.

Final `model.logits` placement follows the catalog's vocabulary axis and retained
output/publication ownership even when component topology and readout equations
have not been declared. The optional component readout must reference that same
public final-output seam. Missing decomposition metadata cannot remove an
otherwise implemented final-output observation.

Whole residual-write ownership joins the architecture node's declared canonical
parameter namespaces to retained logical execution-unit owners. An exact logical
parameter group identifies its own invocation; a nested invocation keeps its
separate ownership. Other namespaces match an exact tensor or its dot-delimited
descendants. Matches must agree on one unit; static roles, missing namespaces and
ambiguous owners reject. Blocks with their own parameter groups use those owners;
structural blocks derive one owner from their parameter-bearing descendants.
Physical aliases and a node's display ordinal do not determine placement. Whole-write values are complete hidden-axis contributions after the
operator's selected collective reduction, and still require actual driver hooks.
A block boundary is registered only when its observation path is declared by that
node; changing the observation catalog alone cannot create a driver hook.
Complete and selected LFM2 execution use `ComponentAdapter` over the same
`Lfm2Kernel`. Typed complete and partition geometry retain their state admission,
global invocation ownership, local state indices, routed banks, chunked-prefill
policy and final-position readout policy. The kernel borrows canonical pinned
roles through `DecoderStaticResources`; complete and selected aggregates retain
their actual native modules. Graph, parameters, paths, state references and
transport project from the validated definition. Constructors validate that
definition before materializing pinned modules, and the shared adapter validates
selected ownership and retained state slots. Low-level construction, embedded
ingress and block operations use the same equations through `kernel()` and
`kernel_mut()`. Normalized readout and pinned parameter traversal use the shared
decoder resource facilities. A partition driver requires retained selection;
complete construction alone does not authorize an arbitrary partition view.

LFM2's ordinary, tensor-parallel and partition methods share block residual
ordering and readout instrumentation. Its dense feed-forward scalar hook precedes
the row-parallel projection; the convolution hook follows the complete mixer
output. Routed provider strategies retain their separate hook requirements.
The provider-aware architecture contracts declare internal unit coverage
independently for ordinary and tensor-parallel calls. Their observed defaults
preserve routing observations without claiming scalar component hooks. Runtime
shares its observed input/group/unit/readout traversal and forward-resource
retention with caller-selected provider execution. Direct and pipeline executors
forward the same admitted provider and parallel context; pipeline strategies
retain one provider-selection path for ordinary and observed calls. Complete
routed residual observations occur after the architecture's existing expert
reduction, while inactive pipeline collective waves retain their original
participation and completion policy.

Partitioned activation edits retain the original admitted operation rather than
rewriting its plan into rank-local identities. Core supplies a separate exact
intent digest for peer comparison while preserving each loaded session's original
admission identity. Digest construction streams canonical serialization; snapshot
and child-preparation estimates include both retained digest strings.

`PartitionActivationLayout` supplies all actual invocation members from retained
architecture coordinates, including nonexporting replicas and empty local shards.
Runtime projects payload positions, reserves host copies and native update work,
and issues move-only `ReservedPartitionActivation` work. It executes the ordinary
activation mechanism with the exact local shape and dtype. Compact masks remain
one local operation for permuted axes. A valid global logit mask can exclude an
entire local vocabulary shard; the original global admission must still leave a
candidate. These local contracts do not establish distributed session/epoch
authority, group agreement or global outcome publication. The shared capture
owner and forward protocol compose those responsibilities through
`SessionPartitionIntervention`, as described below.

The shared decoder declares attention components independently of its feed-forward
operator. Routed Qwen and GPT-OSS therefore retain query/key/value mappings,
normalization, biases, channels and readout equations. An architecture-selected
routed unit declares its complete expert contribution separately; it does not
invent scalar expert units. `ComponentResidualWrite::input` identifies the actual
normalized whole-operator input and shares the invocation's hidden-axis placement.
`AttentionAttributes::sink_logits` names learned per-query-head sink logits used
in attention normalization; their value contribution is zero.

Generic decoder provider calls pass instrumentation through the same attention,
normalization, residual and cache engine as ordinary calls. Projection operators
select the completed residual boundary, preserving provider-owned routing events
without reusing their pre-reduction output as a complete contribution. Generic
partitioned input/readout hooks execute their actual shared boundaries. Qwen and
GPT-OSS explicitly report provider unit coverage; other custom projections retain
the conservative default until their implementation supplies these hooks.

K2 Horizon supplies direct and provider component hooks. Its ordinary and observed
feed-forward calls share one architecture driver: sparse units flow through the
existing routed-unit observer, shared SwiGLU uses a nested component scope, and
the completed routing event follows expert reduction and shared addition. Dense
prefix units retain the ordinary decoder hooks. Native backends implement the
same grouped and linear operators without selecting a K2 semantic branch.

K2 also derives routed and shared intermediate widths independently from retained
local parameter placement. Mixed encodings can require different TP boundaries:
384 routed FP8 units split into 256/128 complete-block slices, while a dense shared
FFN of the same global width splits into 192/192. The local architecture retains
the shared down-projection's actual input width for module construction; it never
reconstructs that width from the routed expert count or local routed width. This
keeps construction, materialization, component coordinates and overlays consistent
without moving family geometry into a backend.

`ComponentGroup::routed_reads` and `ComponentRoutedRead` are neutral relationships
for activated expert mixtures feeding an attention component. K2's architecture
projection declares its MoVA value bank, group-to-channel row mapping, SiLU,
selector/correction parameters and routing node. It does not declare a nonexistent
ordinary V matrix. Checked read/bias joins use actual loaded parameter geometry;
the resulting selections carry no query or overlay authority. The architecture
also owns shared-expert scalar groups, normalization and source identities.

Identical hidden coordinates still do not prove identical transformed
multiplication inputs. Such points require retained transform facts before global
capture can advertise them. Native code supplies collector facts and tensor
transformations without selecting architecture ownership.

Speculative facade entry points also agree whole-batch host and native-prompt
preparation. Controlled speculation agrees its initial capture validation before
entering the shared preparation driver. The facade exposes the same result and
cancellation agreement for caller-owned media/opaque-prompt preparation; local
inspection and immutable admission do not introduce hidden collectives. Core owns
local-result/peer-outcome resolution, reused by native adapters while an existing
submission lease is held. Embedded speculative execution uses retained session
transport to agree sampler/lane readiness before cache construction, then cache
results and initial lane delivery before invoking the ordinary or controlled
scheduler. Cache construction may itself enter architecture preparation
collectives, so a single vote after sampler and cache construction is too late. Native error wrapping retains its
original source. A rejected broad native operation remains fenced without a
whole-run restoration witness; completed agreement is not proof of safe reuse.

Detached media conversion uses the same native submission recovery as text
prompt construction. The generic adapter retains caller-owned tokenizer errors
without translating them to native failures; only native begin/status errors pass
through the supplied backend mapper. Processor observers execute inside that
scope, so abandoning failed preparation retains unresolved native work through
completion or teardown. This does not introduce session collectives into local
media preparation.

The facade's shared committed-token cursor also carries local host results and
cancellation through bounded delivery readiness before a prediction and after
its token/semantic callbacks. Ordinary and controlled sources use the same core
result policy and session ledger. A local decoder or record-budget error remains
typed; successful peers receive rejection before another prediction. A peer
cancellation preserves the already committed prefix and does not run another
forward. Empty capture draining establishes native completion for this boundary
without enabling component collection. Controlled lifecycle and boundary-record
publication use the same readiness policy after token commitment; peer cancellation
is retained before the next prediction. Completed ordinary publication agrees its
record result without rewriting an established termination reason. A best-effort
failure record preserves the original agreed error. Snapshot, branch and sampling
records share the controlled delivery helper; these collective operations must be
called in the same order by all participants. Speculative scheduling has additional
boundaries whose failure audit remains separate.

The neutral `SpeculativeExecutor::agree_text_preparation` contract also carries
visitor-owned setup through the retained session agreement. Architecture-owned
embedded executors and their erased adapters forward mechanism facts unchanged;
MLX supplies the same session transport and cumulative ledger. The shared runtime
scheduler agrees policy and completion support before registration or prefill.
Controlled setup agrees lane geometry and collector installation before creating
that scheduler. Local setup errors retain their typed cause; successful peers
receive `SpeculativeDriverError::Preparation` or the corresponding neutral control
error. This does not make failed native operations reusable or establish completion.

Speculative action selection has a neutral host-fact contract separate from
native model execution. Runtime coordinates exact request identities/lifecycles,
per-request cancellation, completion, deadlines and optimistic eligibility using
confirmed fixed-size frames on the retained session transport. Completion and
eligibility require all participants; cancellation and expired deadlines propagate
from any participant. Core selects actions from those common facts and agrees
local preparation/action failures. Draft loops agree each sampling/prefix result
before another native forward; execution observers, proposal-state creation and
verification checkpoints agree before subsequent native work. Native adapters
supply transport and retain completion ownership; scheduler facts never release it. Runtime cancellation
signals a request for the next coordinated boundary instead of invoking a local
publisher immediately. Core agrees prefill, verification preparation, cache commit
and publication results before installing the new logical lifecycle. Publication
cancellation propagates before another scheduler action. Pending cancellation
settles the retained completion and commits only its safe prefix. Payloads remain
owned through rollback; successful completion remains retained through publication.
Runtime controlled sessions charge and agree each step record, and cancellation
drains that same action/record driver. Caller errors at an unfinished boundary or
a completed closure are agreed with peers before returning; already agreed failed
steps do not start another exit exchange. Neutral and facade tests cover these
boundaries; native distributed validation is recorded separately. Sampling uses
the shared native capture-error translator so portable budget causes remain
directly discoverable beneath the neutral public error.

Cold text-run preparation has a separate shared boundary before that forward.
Core declares request, prompt, sampling, instrumentation and initial delivery
stages, with ready/cancelled/failed dispositions and typed peer rejection.
The ordinary generation machine and detached controlled driver use the same
fallible preparation composition. Facade code supplies tokenizer, semantic policy,
trace delivery and cancellation decisions; native prompt and sampler construction
remain backend mechanisms. Exact token IDs can enter `PreparedChatInput::TokenIds`
without early native construction. Raw token iteration returns the neutral
`BackendFailure`; capture installation separates typed `CaptureError` from
preparation/transport failure with `TextCaptureSetupError`.

Runtime's `TextPreparationCoordinator` uses the existing bounded portable-word
transport. Two fixed-size exchanges bind readiness to the retained setup identity,
monotone attempt and stage, then confirm validated decisions before continuation.
The native session shares that coordinator and its actual communication authority
across adapter clones; model snapshots and resets do not copy or rewind it.
Native gather estimates and runtime framing are reserved before submission, with
cumulative logical charges exposed separately from capture through
`text_preparation_usage`. These charges are not physical allocator measurements.
Unagreed failures fence the selected authority; native completion still retains
live resources through its safe disposition. Session synchronization checks that
authority before entering native queue waits, since a timed-out collective may
still own quarantined work. The existing forward input agreement remains
mandatory, including after successful cold preparation: a local failure during
final confirmation resolution need not be visible in another rank's completed
confirmation.

The retained execution distinguishes requested exact completion from its backend
availability. A transactional observer can require that available mechanism even
when ordinary execution is asynchronous; absent support is rejected before the
forward. Partitioned execution performs a participation status agreement even
when observers are absent, so mixed rank configuration cannot enter incompatible
observer collectives. No component tensor is created by this control agreement.
The shared capture owner blocks record draining while a transaction is pending,
keeps failed attempts outside snapshot boundaries, and preserves its consumed
epoch and ledger across restore. The partition observer below composes bounded
producer quotas, coordinated identities and receipt exchange into these callbacks;
public distributed composition is still incomplete.

`CaptureObserver::for_step` defers envelope admission to the shared local
preparation callback and obtains its phase from the actual forward. The capture
owner consumes the epoch before fallible reservation; failed admission cannot
complete or be restored while pending, even if it produced no record envelope.
Runtime reports `ReplicatedTextSessionError::BeforeStateMutation` only for the
shared preparation path before the driver receives mutable model state. Native
recovery consumes this evidence separately from successful rollback and from
native settlement; it never increments the successful-restoration counter for
admission rejection. Model-call error adaptation retains the runtime error and
its available cause chain rather than formatting away that object. The wider
neural/observer adapter source-conversion audit remains part of integration.

Core `CaptureReservation` exposes spending without ledger replacement or reset.
Its move-only `CaptureQuota` grants prepaid local work; unused or dropped credits
are never refunded. Runtime binds partition capture to the actual capture owner
and pending forward epoch, reserves every producer's work and every rank's receive
and assembly work, and coordinates a digest of the complete selection, layout and
quota set before source access. Core `CapturedStep.partitions` carries serializable
context, producer acknowledgments and selected-region evidence; these host records
cannot grant execution authority.

`PartitionCaptureObserver` owns this shared lifecycle integration. Architecture
`ComponentPartitionLayouts` implements the neutral `PartitionCaptureLayout`
contract, declaring both distinct producers and all invocation participants,
including replicas. A separate, prepaid `SessionPartitionHook` uses only those
participants for failure agreement before the next tensor collective. Inactive
pipeline ranks participate in common preparation and final receipt delivery.
The architecture manifest selects exact failure agreement on tensor groups
even when pipeline parallelism is active. MLX supplies retained group lookup,
native status submission and bounded completion; it does not infer family ownership.

Native communicator ownership retains deduplicated opaque group facts from the
already validated setup-consensus manifest set, projected onto the local rank.
Inactive ranks can therefore validate and price a remote hook without realizing
its communicator. Native submission still requires the locally selected group.

Runtime `establish_communication_session` agrees a versioned setup transcript
containing each rank's manifest and fresh native instance nonce. Its common
`CommunicationSessionIdentity` distinguishes repeated setups with identical
manifests. MLX issues the nonce and retains the agreed identity alongside the
actual communicators; the existing two bounded setup gathers carry both facts.
Failed nonce issuance is exchanged before rejection. The runtime rejects metadata
exceeding 16 MiB per rank or 128 MiB in the padded gather before allocating padded
gather buffers or transferring payloads. Local proposal serialization precedes
this check; these serialized limits are not physical allocator guarantees.
The identity is descriptive provenance, never a communicator or submission lease.

`PartitionCaptureIdentity::for_session` binds this retained setup to the first
actual forward epoch claimed by its capture owner. Failed preparation consumes
that epoch; later steps and restores retain the original run identity. Production
loading creates a setup per model construction, while clones share that setup;
the shared model runtime owns the monotone forward sequence. Native composition
must keep those owners paired and supply actual artifact/execution/overlay facts.
This mechanism does not itself select the public distributed capture adapter.

MLX hook admission accepts the connected pipeline-stage and whole-session groups
selected by prepared component execution, including singleton and neighbor-pair
cases. Arbitrary disconnected low-level Ring memberships retain an explicit
rejection: independent status would require inactive peers to participate in a
coordinated world wave. This limitation does not exclude the prepared model
placements covered by the component-analysis matrix.

Loaded MLX capture caches architecture-declared component layouts on discovery
demand and selects `PartitionCaptureObserver` through the same ordinary and
controlled prefill/decode entrypoints. Runtime
`observation_support_with_partition` applies placement/collector facts only after
the existing selection, phase and instrumentation gates. Architecture supplies
hook membership; MLX reports its exact selected independent-group mechanisms.
Unknown ownership stays unverified. This does not transfer family semantics into
the backend or enable parallel parameter edits.

Effective parameter partition geometry follows the same ownership direction.
`eredu-core::parameters::ParameterCoordinateMap` projects a selected global
rectangle into local rectangles and result destinations, including independent
axes and permutations. Its bounded edit and contraction projections retain the
original global selection. A contraction fragment also retains its source region,
so the receiver can distinguish additive partial contractions from duplicate
source coverage. Geometry is descriptive and grants no session, epoch, edit or
communication authority. Reservations precede metadata and payload copies; failed
or unused projections do not refund them.

Architecture `PreparedModelDiscovery::parameter_partition_layout_for_rank`
resolves the selected task, its declared aliases and emitted companions against the parameter topology
retained by construction. It reuses the production pipeline-ownership compiler and
delegates atomic parameter-group lowering and effective coordinate projection to
runtime's `derive_parameter_member_layout`. The shared compiler consumes selected
packing-axis and encoding facts to convert complete physical quantization blocks
to effective scalar coordinates. Fused segments, ordered expert axes, absent
pipeline owners and shared static consumers retain their distinct meanings. FP8's uncompressed axes preserve uneven scalar ranges.
Neither source content resolution nor native parameter access occurs here.
Companion coordinates describe scalar scale/bias cells; they never inherit the
weight's packed-block expansion. Actual prepared slot metadata supplies sharing,
including intentional sharing between distinct materialization tasks.

Runtime `LayeredParameterOwner` lends the architecture and residency policy
already retained by ordinary, routed and partitioned executors. Static traversal,
exact unit loans and publication use that shared mechanism. Architecture-owned
direct, routed, pipeline and composite executors expose their actual retained
owners; native erasure forwards the operation without selecting a family or
reopening a source. The residency policy validates unit addresses and preserves
completion ownership, including rejected callbacks. Publication first asks the
policy to accept completed replacements; only then are pinned static handles
updated. The enclosing live transaction remains responsible for admission,
quiescence, budgets, peer agreement, caches and snapshots. A custom executor with
no retained parameter owner reports unavailable access.
MLX composition retains the architecture-supplied unit address separately from
the policy's local residency address. Parameter loans validate the former before
mapping to the latter, including nonzero pipeline offsets. Both resident and
bounded policies reject a foreign unit address before traversal or construction.

These methods are prerequisites for distributed parameter operations. The live
runtime provides `PartitionParameterReadPlan` and `ParameterOperationCoordinator`.
The read plan assigns each selected source cell to one producer, including partial
replica overlaps and different permutation tilings. It rejects incomplete coverage
before native work and assembles exact payload counts; contractions sum disjoint
source contributions in F64 before one F32 result conversion.

Core retains local immutable edit authority and a separate common intent digest,
streaming bounded serialization rather than copying complete encoded edit plans.
Core also owns typed coordination outcomes and separate lifetime control accounting.
Runtime owns setup-bound attempts, ready-peer geometry agreement, source preparation,
payload transfer, completed delivery, publication and rollback decisions. Every
decision includes confirmation of its completed validation. Mandatory rejection
and rollback control is admitted from exact native setup facts and charged before
submission, separately from caller-limited parameter and payload work. Failed
local preparation needs no fabricated layout or payload ceiling; all ready peers
must agree those facts before source work. Failure, reset and snapshots refund
neither allowance. Abandonment, ambiguous transport, failed restoration or protocol
disagreement fence the retained owner without claiming physical native completion.
After an unwind, poisoned authority reports terminal failure while already charged
control usage remains readable; reading counters cannot reactivate the owner.

MLX retains that coordinator alongside its existing native communication authority
and supplies ordinary bounded word transport, exact rank/setup facts and native
storage estimates. Each constructed model receives a distinct runtime-issued
`ParameterModelIdentity` from the retained coordinator. Corresponding ranks pair
model constructions in the same order; operations compare the resulting model
identity as well as source, execution and parameter version. Reset, snapshot
restoration and destruction never recycle that identity. Registration performs
no native work or communication. Source reads and contractions share the same
native mechanisms
as ordinary public parameter operations. Architecture semantics and native tensor
handles do not enter the control protocol.

Public partition discovery exchanges bounded loaded metadata, without evaluating
parameter tensors. Runtime `PartitionParameterCatalog` validates actual sharing,
global/local geometry, per-operation capabilities and complete source coverage.
An empty shard cannot downgrade a nonempty owner's native mechanism. The model's
catalog-only binding permits source-resolution errors to reach shared admission;
it cannot authorize parameter reads or edits. Queries and projections use the
full retained model/source/execution/version binding and existing native loans.
`LoadedParameter::access()` distinguishes query, projection and replacement;
explicit flags override the legacy aggregate support field. Metadata exchange,
source preparation, conversion, native contraction, transport and host delivery
are charged before their work; counters survive rejected attempts.

Distributed overlay composition prepares all replacements and a fresh mutable
state before publishing any handle. Runtime supplies a local state exchange for
use under the parameter coordinator, without nesting ordinary control collectives.
The exchange retains the previous cache and prompt identity for rollback and does
not rewind commit epochs. After every peer confirms publication, native composition
invalidates old snapshots and advances the parameter version, including on ranks
that own no edited slot. A completed publication rejection restores both weights
and state; failed restoration fences the session. All actual aliases and replicas
participate. Packed affected parameters are promoted on every nonempty owner,
including shards outside the edit rectangle, so the selected input arithmetic
remains consistent. Distributed active provenance uses the common admitted edit
intent; rank-local admission authority remains distinct.

Affine and MXFP4 effective parameter access uses the same native dequantization
for matrices and grouped banks. Loaded support validates every leading group/row
extent and the exact packed feature width before native work. A floating
replacement passes through ordinary grouped multiplication even though the
declared packed format and original companions remain retained for restoration.
This dispatch depends on actual storage, without selecting a model family or
changing architecture-owned group equations.

Block-FP8 and native GGUF effective access also admits rank-three matrix banks.
The native adapter validates actual FP8 geometry or the exact GGUF byte extent,
retains singleton group axes after decoding, and reserves FP8 scale expansion
across every group and both padded matrix axes. Grouped GGUF host conversion is
charged for the complete affected bank. Floating replacements use ordinary
grouped multiplication; an FP8 bank therefore switches its input transform to
identity while active, matching the matrix overlay policy. Packed storage and
companions remain available for exact restoration. Gated, unary and selected
grouped operators share that storage dispatch. Quantized routing projections
likewise dispatch from the actual storage dtype, preserving their declared
input transformations and routing precision through edit and restoration.

Qwen GGUF admission retains each native matrix encoding before constructing
derived expert recipes and cold lowering requirements. Architecture-owned format
normalization maps separate gate/up and down checkpoint identities to their
canonical bank parameters, using the configured parameter root. Split gate/up
encodings must agree. A byte-producing bank recipe never inherits a metadata-only
dense default. The runtime continues to require exact agreement between retained
tasks and their lowering descriptors; diagnostics identify the differing facts.

`CaptureCheckpoint` retains a setup-scoped child identity seed and accounts for
its host storage. Fork installs that seed with a fresh capture owner and no first
epoch; shared preparation checks it against actual loaded artifact, execution,
setup and overlay facts. Restore keeps the parent's first epoch. Loaded observer
composition turns local setup failures into shared preparation rejection before
model work. Receipt byte allowances use actual fragment metadata and native
encoded-size estimates, bounded by the admitted ceiling; small selections do not
prepay an unrelated maximum-sized receipt. Deferred projection-input factory
credits use the same native bound as the actual FP8 implementation.

The partition collector reserves common coordination before optional capture work.
Runtime seals successful producer claims, skipped selections and consumed credits
into the same pre-forward agreement. Limit skips discard local producer authority
without refunds and publish typed records only after final commit. Native code
supplies the existing transport and resource facts; it does not choose rank-local
skip policy. This also applies to re-admitted controlled children.

Deferred tensor observations carry neural-owned `TensorElementType` and
`GeneratedTensorSource` facts. `eredu-nn` remains independent of `eredu-core`;
runtime translates those facts into core `GeneratedCaptureSource` metadata at
the architecture observation adapter. The prototype declares geometry only.
Native FP8 declares the actual F32 diagnostic output. Runtime validates declared
precision before transforming a generated value, and `CaptureBackend` can price
that source independently of the prototype's dtype. Empty selected outputs use
a shared reserved host constructor without creating a source tensor. Partition
receipts retain each empty producer's declared precision and reject disagreement.

Ordinary MLX observation composition carries the backend's structured `Error`
through array/tensor adapters. Its neural adapter uses `eredu_nn::Error::backend_source`
to retain the original cause across the portable execution driver. Capture policy
failures remain typed sources, and native exceptions retain their creation location;
bounded record diagnostics remain separate from this error chain. The public facade
continues to expose `eredu_core::BackendFailure`. Neither a capture error nor its
classification establishes safe completion: the existing shared rollback witness
and native submission/recovery owner still govern session reuse and lease release.

Routed provider, addressable-bank, indexed-movement and route-exchange contracts
require owned standard errors (`Error + Send + Sync + 'static`), as does
`LayeredArchitecture`. This lets portable adapters retain the cause of a failed
mechanism through resident, bounded and partitioned execution. Grouped operation
bridges return `eredu_nn::Error`; architecture-owned routed execution errors carry
a source-bearing variant alongside their typed plan and diagnostic variants.
Provider observation and family adapters preserve those sources instead of
formatting them into strings. Source-bearing errors do not implement equality:
callers match policy variants or inspect the source chain. Cold preparation and
submission-policy diagnostics have separate contracts; this does not turn a
provider error into evidence that native work completed or remote peers stopped.

Provider execution agreement is independent of component observation. Runtime
supplies source-preserving provider wrappers; architecture construction retains
the selected tensor group explicitly. Routed TP calls vote before their following
model collective. Exchanged expert work votes after received-route validation,
local bank execution and output validation, before reverse exchange: TP first,
then EP even when TP rejected the work. Shared pipeline waves additionally vote
on the exact session agreement group retained from the prepared execution plan.
That vote includes inactive stages before any participant advances to reverse
exchange or another tensor collective; stage-local votes alone cannot certify a
shared wave. Idle owners and inactive pipeline waves participate in the same
sequence. Expert-group manifests therefore require exact failure agreement as well as count gathering and variable exchange. Local failures
retain their original cause; healthy peers receive a typed rejection, carried by
`ReplicatedTextSessionError::Partition` through pipeline and session adapters.
These votes do not replace native completion ownership or establish recovery from a failed
communication operation.

Nemotron-H partition adapters reuse the block's ordinary component hooks and
the shared readout instrumentation for local, tensor-parallel and pipeline
execution. Effective embedding values remain in the typed boundary payload used
by downstream target/prediction stages. Discovery joins the Mamba operator to its
canonical `mamba` parameter namespace while retaining the logical mixer observation
identity; a checkpoint alias does not define invocation ownership.
The routed partition adapter declares ordinary and TP unit hooks separately from
provider-unit hooks. Its TP block driver applies normalized-input, attention-channel,
ReLU²-unit and residual interventions to the values consumed by the existing
parallel operators. Routed observations retain provider completion and reduction
ordering. Disabled instrumentation uses the same block driver without extra tensors.

Mamba convolution parameters follow the same semantic segmentation as their input:
value channels, input-state channels and output-state channels. TP selects its group
interval within each segment for both convolution weights and bias. A contiguous
slice of the concatenated channel axis would pair kernels with different projected
channels. This placement belongs to Nemotron's neutral parameter plan and is consumed
unchanged by materialization, effective-parameter queries and overlays.

`CapturedStep.outcome` carries the shared model transaction result: committed,
aborted, or untracked for low-level/legacy records without transaction evidence.
Runtime supplies it when draining the ledger; backends do not infer it from error
text. Individual observations retain their original meaning, so an observation
completed before a later provider failure can remain present in an aborted batch.
Aborted batches retain charged usage and cannot establish a snapshot boundary or
safe native reuse. A committed forward also does not imply successful sampling.


Partition activation interventions share `CaptureSession` and
`PartitionCaptureObserver` with ordinary bounded capture. Runtime owns move-only
operation authority, local admission/epoch checks, globally prepaid projection and
dependency work, exact coordinate/intent agreement, active-member failure votes,
and completed world outcome delivery. Only final shared commit publishes `Applied`
and per-operation evidence. Independent invocation groups exclude inactive pipeline
ranks while retaining nonexporting replicas and empty local selections.

Architecture layouts supply actual activation members and cold region bounds.
Their coordinate identities join common agreement; equal tensor shapes do not
establish equal placement. Native MLX composition selects retained support facts
and supplies ordinary activation arithmetic, source completion and transport.
No family equations or semantic dispatch move into the backend. Evidence uses
private derived global capture admissions keyed by operation and before/after
ordinal; it never rewrites the parent's capture plan or aliases its selection keys.
Projection host copies, dependency completion, group votes, receipt transport and
evidence delivery consume nonrefundable credits before work. These remain logical
storage bounds, including conservative decoder allowances, not allocator guarantees.

Scalar component groups declare `write_output` and `output` observation relationships
for the complete affine and normalized contributions. The architecture partition
layout derives their hidden-axis replica placement from the same logical write
parameter that owns the component invocation; shared source storage does not
transfer execution ownership. `ComponentWritePartition::Complete` observes after
projection reduction; unit/channel observations retain sharded scalar coordinates.
Component descriptions declare `TensorParallelSum`: every TP rank supplies an
additive full-hidden-width term before an ordinary, possibly fused reduction.
Architecture layouts retain every TP term, remove only true replicas, and require
complete TP invocation membership. Each replica sum designates TP rank zero for
Add offsets and replacements; masks and scaling affect all terms.

Partition receipt/evidence schema 3 binds `PartitionCaptureCombination` to the
retained authority and wire records. Runtime's `SumF64ToF32` collects selected raw
native F32 terms, sums in canonical world-rank order with compensated F64 arithmetic,
then rounds to F32 before summary/histogram transformation. Preview collects only
the selected prefix. It prepays raw export, host assembly, transport and delivery;
the native backend supplies ordinary collectors and completion. This host assembly
does not imply that a separate complete native tensor existed, nor bitwise equality
with a native reduction. The neutral mechanism and V3 shared-write declarations are
implemented. Mixed V3 TP execution hooks pass native F32 SafeTensors/GGUF and
load-time affine 4-bit/group-32 acceptance across TP/PP/EP and all three ordinary
weight residencies.

LFM2 selects the existing neutral arithmetic contracts for its published reduced-
precision equations: rotary products round in the input dtype, attention rounds
scores and probabilities, and sigmoid routing retains input precision through
coefficient normalization. F32 expert correction biases affect selection only.
Expert mixtures accumulate in ascending expert identity with a rounding boundary
at each addition. Both replicated and placement-resolved construction retain
these choices; reusable native operators implement them without family dispatch.

The shared native learned RMS operator preserves its input-precision rounding
boundary before multiplying the gain. Half-precision inputs use float32
normalization followed by an explicit input-dtype cast on CPU and
Metal. Metal uses the existing float32 row reduction so reduction-order differences
near a rounding midpoint do not change downstream component experiments. The
parameterized operator, learned-offset form and direct weighted operation share
that mechanism. FP32 gains promote the rounded normalized values to FP32;
their precision does not remove the preceding input-dtype rounding boundary.

On Metal, F32 grouped and learned-offset RMS normalization combines the ordered four-level
reduction, normalization and optional gain in one native kernel. One lane reduces
each row; its threadgroup applies normalization and gain in parallel. Separate
native squaring preserves the F32 square boundary, and contraction is disabled
before the gain. Groups index the retained full-width gain. Weightless
normalization retains its existing mechanism. Strided inputs are made contiguous by the native kernel
interface. Other devices and dtypes retain their existing realization.

The native F32 realization of `RotaryArithmetic::InputProducts` combines position
construction, frequency multiplication, precise sine/cosine, amplitude scaling
and paired output arithmetic. Separate rounded products prevent contraction into
fused multiply-add. Position starts round to F32 before adding sequence indices;
traditional pairs, split halves and untouched tails retain their neutral
contracts. Sequence extents are runtime inputs and do not specialize shaders. Narrow dtypes and non-Metal execution retain the expanded realization.
These are generic backend mechanisms: architecture-owned equations, policies,
frequency preparation, state and completion ownership remain unchanged.


The V3 provider-aware TP traversal invokes the additive shared-write hooks and
preserves its ordinary fused reduction through a shared helper. A nonzero neutral
test covers complete/partial provider results and literal post-bias across five
shared-branch experiments (15 cases), with exactly one reduction. All eleven V3
component numerical tests pass (2.45 seconds). Nine partition-coordinate tests also
pass, including TP2/EP2 term ownership and one offset owner per replica sum. Target-only mixed-TP traversal hooks are enabled after neutral execution
conformance. Native F32 SafeTensors/GGUF and load-time affine public
capture/intervention acceptance passes all 63 TP/PP/EP and ordinary-residency cases.


DeepSeek V3 and V4 construct and retain a validated, source-free
`ComponentDefinition` before native modules. It owns canonical parameter bindings,
target and prediction graph identities, invocation paths, transport declarations,
and complete or TP-local mutable-state geometry. Public parameter catalogs project
that definition; source recipes and encoding policy remain family-owned. Static
resources declare their storage roles alongside their parameter groups, including
shared target/prediction embedding authority and V4 DSpark's pinned resources.

DeepSeek V3's public model is the shared `ComponentAdapter` over `V3Kernel`.
The kernel owns latent-attention, routed-expert, and MTP equations; typed hooks
supply ordinary, observed, parallel, provider-aware, and partition execution.
Serial and parallel ingress share request preparation and cache-mask construction.
The shared decoder resource aggregate holds selected embedding, normalization, and
vocabulary modules. Consuming partition construction retains only declared canonical
storage roles and derives the local state origin from selected ownership. Downstream
owners resume transported token IDs and effective embeddings without re-embedding;
MTP depths keep separate cache slots. Standalone prediction operations borrow the
kernel through the ordinary invocation and completion lifecycle.

DeepSeek V4 uses the same adapter over `V4Kernel`. Its hooks retain local and
compressed pooling attention, token-selected expert routing, hyper-connection
streams, sequential prediction, and fused DSpark equations. Target capture and
prediction-group completion share one equation across serial and parallel calls;
the backend supplies vocabulary projection and reduction operators. Selected pinned
resources contain the shared decoder aggregate plus optional hyper-head and DSpark
modules. Partition construction consumes canonical ownership and state origin;
transport carries token identity and explicit capture slots while the family owns
hyper-stream flattening and reconstruction. Prediction extensions invoke kernel
operations through the ordinary materialization and completion lifecycle.

DeepSeek execution metadata reads the retained definition. Group ingress uses its
validated dependencies, and unit construction validates global invocation addresses.
V3 latent/rotary caches, V4 local-key and pooling streams, and independent prediction
state slots keep their declared geometry and persistence. Target state follows its
pipeline unit cuts; appended prediction state belongs to the output owner. DSpark's
prediction frontier and target-capture policy retain their separate family semantics.
Catalog validation errors use the neutral neural error and retain their original
source.

DeepSeek expert geometry expands the retained uniform partition domain into scalar
FFN coordinates. Encoding expansion may group several neurons into one partition
unit so weights, packed input columns and scale/bias companions share legal cuts.
The architecture validates divisibility and exact ranges before deriving local
routed/shared ownership; native code receives those scalar ranges. A partition
unit is therefore not assumed to be one neuron. Nonuniform chunk layouts need
their own explicit mapping. Mixed affine V3 native acceptance passes all 21
TP/PP/EP and ordinary-residency cases.

Routed partition construction retains an optional typed source architecture
and its exact local layout alongside the selected target. The runtime handoff
revalidates source parameter addresses and layout before forwarding both to the
existing materialization driver. V3 constructs this source from retained encoding
facts, including expert ownership, target-unit range and compressed-state geometry.
Native local and pipeline adapters forward that authority without choosing family
semantics or reopening artifacts. Mixed V3 load-time affine native acceptance
passes for this path. Other routed families provide their corresponding exact
source construction for load-time transforms; the component guide records
native family/format acceptance.

DeepSeek's shared router construction consumes the exact retained matrix format.
V3/V4 routers default to dense when no per-parameter format is declared; selected
load-time transforms or encoded sources supply an explicit override. The same
rule determines parallel weight and companion geometry. Backend group selectors
receive the ordinary neutral linear-format declaration.

Exact native transform materialization validates the admitted whole-source recipe
before applying retained rank placements. It then checks the selected local shape
and floating slot category against the source module. Unloaded source slots
use construction-default dtypes; the admitted recipe supplies F16/BF16/F32
precision for transformation and affine companions. Global derived-output metadata and
local executable geometry remain separate authorities; comparing the former to an
already sharded recipe would reject valid TP/EP materialization.

Independently cached routed parameters retain all selected output destinations,
including generated quantization companions absent from the original source
catalog. Architecture construction forwards those exact member-task identities
as addressable ownership. Ordinary module binding still rejects any uncovered
destination; it does not infer scale/bias names or silently load bank-owned copies.
Joint target/prediction selection retains both roles' banks, while the target's
addressable parameter catalog excludes declared auxiliary parameters. Prediction
module construction validates those parameters against its separate selected tasks.

Exact replicated-text affine transforms preserve the admitted floating source precision for
generated scales and biases, matching independent-bank materialization and neutral
selected-byte accounting. Unloaded native destination dtypes describe placeholders
and cannot silently request an F16 cast. Reusable prompt-cache schema 10 rejects
incompatible persisted metadata that cannot distinguish these numerical
policies. Reconstruction uses exact token-ID replay.

V3 embedded prediction primitives share fusion and head execution between ordinary
and observed paths. Architecture-owned hooks surround normalized embedding/hidden
inputs, the fusion result and the prediction head; the existing V3 block supplies
MLA, routed and shared component hooks. Editing the prediction readout residual
precedes head normalization and changes both returned hidden state and draft logits.
The `mtp.{depth}.output` point follows head computation. Low-level
ordinary/provider/TP traversal forwards these hooks. Prediction scopes are declared
separately; public speculative component admission requires selected phase-aware
producer hooks and native collector support.

The independently cached mixed V3 component matrix passes all 63 native CPU Ring
cases across F32 SafeTensors/GGUF and affine loading. Cache telemetry distinguishes
an EP owner with no routed work from a missing bank: exact ownership remains, idle
owners construct no compact banks, and every routed pipeline stage has an active
owner. The numerical/query/overlay checks remain unchanged.

Architecture and component descriptions identify separately
invoked component scopes. V3 prediction depths own their decoder groups, fused
residual base and `ComponentReadoutEquation`; primary target component lists retain
their existing meaning. `ComponentResidualBase::LinearFusion` declares ordered
normalized sources, canonical effective fusion parameters and actual projection
input evidence. Its supplied hidden input is invocation-local: prefill/replay may
supply target hidden state, while a proposal may supply the preceding prediction
output. It is not a second token embedding base. Prediction nodes, parameter joins
and scope membership are architecture policy. The common readout equation lives
in core; primary readout JSON keeps its flattened fields.

Runtime support additionally requires the selected call path's explicit
`prediction_inspection` fact before consulting partition placement. A declared
scope, native collector or ordinary target hook cannot establish that fact.
Ordinary target and cold declarations alone leave it unverified. Loaded
prediction construction supplies the complete selected-hook fact before public
speculative capture admission.

The typed prediction extension accepts optional internal observers for V3
prefill, proposal and retained-input replay. Architecture-owned prediction
operations forward those observers through the target's ordinary embedding and
selected serial/TP equations, preserving outer unit timing. Absence calls the
existing unobserved methods and does not enable diagnostic generation. The
enclosing speculative driver continues to own lane checkpoints, failure recovery,
phase attribution and completion. An extension must report its complete selected hooks before accepting an
explicit internal observer. Public bounded collection additionally requires the
phase-aware admitted authority; the component-analysis guide records the
implemented family and encoding matrix.


The architecture-owned embedded strategy accepts a separately installed
`SpeculativeActivationObserver`. The common driver passes it to target prefill,
prediction seed, depth-specific proposals, target verification, prediction replay
and target replay. Runtime brackets each attempted invocation with its actual
physical sequence width; this width is not a committed prediction index. Completion
stages records but does not commit tokens or replace native completion ownership.
An infallible finish callback also runs after failed admission and unwinding.
`with_activation_observer` lends one observer for the entire forward, allowing
move-only producer work and coordination to remain on the stack across callbacks.
The default borrows the existing owner without allocating. The operation receives
only `ActivationObserver`; admission and finalization remain outside it. Borrowed
work is dropped before completion or failure finalization, including unwinding.
A live five-rank fixture exercises this lifetime through actual partition edits,
receipt assembly and final commit. The speculative collector uses a partition provider and shared auxiliary
transactions, bound to the selected prediction layouts and retained native
transport.

Target passes reuse the existing observed replicated-session transaction, retaining
its exact target capture, output publication, completion agreement and rollback.
Prediction calls reuse the typed extension operation. Runtime's `ObserverErrorBridge`
forwards generated reservations, routing/unit hooks and transaction callbacks across
neural/native error types and returns the original observer failure. Absence still
uses ordinary methods. MLX only binds the complete observer set through its session;
it adds no phase or family dispatch. Complete hook selection is supplied by each supported materialized extension.
Public capture budgets, phase/frontier attribution, effective query/overlay
sources and distributed producer bindings are verified through the shared
collector; the component-analysis guide records family-specific acceptance.


Core separates `CaptureInvocationShape` (exact batch, physical sequence and
optional actual context) from prediction schedule coordinates. Capture and
intervention plans can be admitted with `CaptureInvocationBounds`; their immutable
identity includes this mode and bounds, while ordinary admission digests retain
the existing encoding. Unknown context is never inferred from a generated position.

Runtime's `CaptureSession::begin_invocation` checks actual bounds and active slices,
then uses the existing capture/intervention ledger, generated-source driver, sparse
receipts and native transformation contracts. Explicit masks select applicability
within admitted entries; absent scopes receive `NotInvoked` capture outcomes and
inactive intervention outcomes. Every invocation reserves its geometry envelope;
repeated prediction coordinates spend cumulative resources. Checkpoint restoration
never refunds these charges, and child re-admission preserves the geometry mode.

Partition capture accepts independently admitted invocation geometry. Core's
partition receipt/evidence schema 3 retains the optional exact batch, sequence and
context in `PartitionCaptureContext`. Runtime binds it to producer admission,
fragment validation, all-rank coordination and final assembly; a different context
is rejected even when it does not change the selected tensor's shape. Ordinary
plans require absent invocation geometry, while invocation plans require explicit
geometry within their original bounds. Deferred factories cannot run before this
validation.

Architecture-owned `ComponentPartitionLayouts` projects dense and routed placement
using that exact geometry. Runtime's `PartitionCaptureObserver` uses the active
capture/intervention applicability masks, applies edits to actual local values,
and retains normal source/hook votes, receipt completion and final commit.
Nonexporting replicas and empty local shards still participate. Invocation metadata
is reserved for every rank and restore does not refund cumulative work.

The shared speculative collector retains these partition observer transactions
across internal target/prediction callbacks and binds the selected prediction
producers and native transport. Public distributed admission requires those
loaded bindings; the neutral invocation protocol alone grants no authority.

The shared speculative scheduler supplies `SpeculativeActivationOrigin` to
internal observers: request identity, committed frontier, logical prediction
coordinate, an exact-prefix digest and optimistic-work status. The operation
scope clears on errors and unwinding. Architecture executor erasure forwards this
scope and drains bounded records and original portable capture failures. Scope
applicability comes from declared node ancestry and prediction depths, not native
checkpoint-name parsing. Runtime `SpeculativeCaptureObserver` borrows ordinary
capture/intervention mechanisms through `CaptureBackendProvider`; it retains one
request's existing ledger and charges each queued invocation envelope before work.
A separate loaded speculative discovery and admission contract binds this
collector. Ordinary prediction inspection stays disabled for ordinary generation.
Partition producer agreement uses the selected layout and retained transport.
Explicit CPU/Metal collector binding and distributed execution have native
conformance coverage.

Native session adapters drain the architecture observer set's already charged
speculative records and neutral failures after the shared executor scope returns.
The collector's error propagation signal may cross a native error domain, but its
first original failure remains owned as `SpeculativeControlError::Capture` or a
neutral `BackendFailure` retaining the native source chain. Sparse callback errors
retain their shared original neural source rather than discarding it into text.


`SpeculativeActivationPlan` admits capture and optional intervention plans together
under independent invocation bounds. Core owns its immutable authority, exact node
bindings and source/execution/session identity. Architecture projection resolves
node ancestry and discharges only the prediction-execution condition from a sealed
materialized extension's complete selected-hook fact. Media and partition gates
remain separate. The loaded execution identity includes the active overlay.

The facade prepares and validates this authority; `GenerationControlOptions`
installs it before lane prefill. Architecture executor erasure forwards installation
and restores prior observer ownership when its borrowed scope ends. Reinstallation
after execution or within the same scope is rejected. The MLX adapter supplies an
owned stream and the existing native capture/intervention estimator and operations;
it makes no family or scope decisions. Failed/cancelled host evidence can be drained
one invocation at a time through `take_activation_evidence`, under the existing
monotone delivery sequence and cumulative JSON budget. This does not recover native
work or clear session failure.

Prepared discovery keeps the ordinary target catalog and the selected prediction
catalog separately. The latter is projected from the original admitted complete
architecture retained with the source graph, before its target-only projection.
Neither discovery nor the native binding reopens artifacts or reconstructs a family
configuration to recover prediction points.

Embedded speculative durable copies use the same core snapshot and runtime
reservation machinery as external drafting. Core snapshot/restore contracts return
`SpeculativeControlError`, preserving typed policy failures and original native
sources through executor erasure. The architecture driver enumerates the complete
target state, canonical prediction state, independent prediction seed, capture tensor
and host identities; transactional rollback clones do not establish isolation.
`PredictionExtensionMaterializer` supplies bounded native copies of the declared
state profiles, while architecture code owns their membership and replica count.
MLX settles source and destination copies and supplies conservative storage bounds.
The driver builds every replacement before replacing the live cache, leaving the
saved checkpoint reusable. Internal activation authority remains scoped to the
collector; restore and branch exchange never rewind its invocation counter or
capture ledger. Runtime owns opaque collector checkpoints and move-only prepared
authority restoration. Architecture checkpoints combine these with complete model
state, prepare authority first, copy every native replacement, then commit both
without a further fallible operation. Transaction rollback checkpoints omit this
durable authority component.

`ControlledSpeculativeSession::readmit_activation_interventions` validates against
the facade-retained loaded speculative discovery at a canonical boundary. Runtime
preflights replacement operations with the existing estimator and cumulative usage,
charges their host storage, and changes only prospective edits and their scopes.
Capture selections, invocation bounds and allowances remain fixed from run creation.
Saved collector authority includes exact admitted identity and scope bindings;
serial branch exchange restores it alongside model and sampler state. Backend
adapters forward this operation through typed executor erasure without interpreting
an architecture family or intervention target.

Prediction parameter access uses the same loaded-parameter contracts as ordinary
execution. `PredictionModuleVisitor` traverses each architecture-owned materialized
module in stable physical-owner order, distinct from proposal depth or lane-state
position. The architecture enumerates sequential, fused and grouped modules;
backends do not reconstruct their family membership. Shared target embeddings and
heads remain with the target owner and are borrowed by prediction execution.

MLX retains exact prepared binding metadata and materialization tasks beside each
prediction module. `PreparedParameterLocation::Prediction` routes bounded queries
to that actual retained module, including its selected encodings and companions.
The loaded catalog combines target, bank and prediction ownership without reopening
sources. Complete replacement tensors are prepared by the existing parameter
transaction. After target/bank publication succeeds, prediction slots receive
infallible handle replacements under the same exclusive session; removal follows
the same path. Architecture parameter identities and aliases govern these moves.

Pooling-attention durable copies include local keys, every pending value/gate
window, pooled outputs and both overlap buffers. Native copies compact strided
views while preserving frontiers and compression ratios. A complete target copies
its shared paging catalog once into an independent namespace in the same bounded
pool, then binds each copied local tail to that manager. Inconsistent manager
ownership is rejected before copying. Prediction-profile copies use the same
native mechanism and the architecture's existing complete replica enumeration.
Storage bounds include retained array views, sealed blocks, catalog growth and
the native one-channel key-only persistence sentinel. Completion and atomic live
state replacement stay with the existing controlled-session copy driver.
Pooling transaction checkpoints retain immutable array views with their original
strides. Array replacement on advancement preserves those checkpoint values.
Native state forks and resident rollback copies compact logical views on the
caller's stream; raw backing-memory copies cannot preserve an overlap slice's
logical element order. This is distinct from the independently budgeted durable
copy of every array and paging namespace.
DSpark's architecture-owned bidirectional proposal mask is broadcast over the
actual keys returned by pooling attention. Multi-token append may return bounded
past keys plus the whole submitted block; the mask cannot infer that extent by
clipping the absolute frontier to the single-token window. Ordinary causal masks
retain their existing timing and eligibility semantics.

Paged sliding-cache fetch preserves the complete submitted span independently of
post-append retention. It gathers the bounded old history before storage eviction,
returns that history plus every submitted key, and applies the existing final-window
retention to persistent state. Ordinary key/value and key-only storage share this
mechanism, normalization, validation and append rollback. A long prefill therefore
uses the same logical attention extent as resident storage even when its earliest
keys are absent from the retained state for the next call.

Blockwise paged attention likewise retains the current query span until native
attention completes, then discards history outside the final window. Both arithmetic
passes consume the retained blocks before that eviction. Failed attention keeps its
charged storage with the existing completion/teardown owner; an early error does not
permit discarding potentially live native inputs.

Observed auxiliary prediction phases use `ReplicatedTextSession::with_prediction_observation`
for the same observer preparation, target-state recovery, completion agreement,
receipt delivery and final commit as ordinary forwards. Architecture composition
wraps the entire prediction prefill, proposal or replay invocation; one prefill
may execute several prediction depths under that transaction. Each typed operation
also agrees checkpoint success before entering its own execution collectives.

`MaterializedPredictionExecutor::complete_state` enumerates exact retained lane
values and actual output dependencies. Sequential, pooling and model-state profiles
reuse neutral runtime retention contracts. `PredictionExtensionMaterializer`
settles that iterator using native completion and deferred validation resources;
it does not interpret a family or rebuild state geometry. Observed proposals
retain their actual logits and hidden output until completion. Prefill and replay
settle retained caches without evaluating discarded prediction readouts. An absent
observer bypasses the auxiliary observation transaction and dependency enumeration.

The native error handoff accepts `BackendFailure` with its owned source.
`safemlx::Exception::from_source` preserves that chain through the existing native
error domain; it introduces no new unsafe boundary. Diagnostic-only contract
messages continue to use the existing text adapter. Public distributed internal
activation admission requires selected prediction producer layouts and a
bound partition collector/transport; this lifecycle alone does not enable it.

`CaptureBackendProvider` lends one ordinary observer for an entire speculative
phase. Its default implementation keeps the local path allocation-free;
`PartitionCaptureBackendProvider` instead retains the selected layout and transport
owners and lends the existing partition observer. It binds the communication
identity before invocation admission, so metadata reservations include every rank.
Native composition supplies primitives, estimates and source-preserving error
conversion. Architecture ownership remains in the supplied layout. Routed neural
callbacks retain typed admission errors and native sources through the same outer
speculative owner. The provider does not select prediction ownership; public
partitioned speculative admission requires the prepared prediction layouts
and their native factory binding.

Prepared prediction modules retain their exact `LocalModelLayout` and global
parameter declaration through shared `Arc` owners. The total architecture
construction driver publishes those owners into the prepared source graph's
once-only discovery slot after successful materialization. Cold discovery has
no constructed placement. Backend materialization consumes the same local layout;
producer discovery projects peers through the same tensor-axis compiler.
Prediction units retain resident expert banks and replicas over non-tensor axes,
independently of the target's pipeline and expert ownership.

`speculative_component_partition_layouts` combines the target with each declared
prediction scope. Scalar coordinates reuse ordinary component proofs; fusion
inputs, residuals and prediction heads retain their own invocation semantics.
Prediction score heads are not target publication. Native composition lends a
`SpeculativePartitionBinding` through the existing execution context to the shared
collector factory. Layout compilation and content identity remain deferred until
discovery or nonempty capture admission; absent instrumentation evaluates no
additional component values. The ordinary session guard and shared speculative
control driver retain completion and state ownership.

A prepared resident prediction bank also retains its compiled scalar coordinates
in the architecture module. Its sparse provider scope wraps the actual local
input and every native chunk, and settles source and failure votes before the
following tensor reduction. This scope is separate from target expert-exchange
ownership; the target's existing providers retain their own invocation scopes.

Auxiliary predictors can declare target static parameters required on every
pipeline rank. `ReplicatedTextRequirements::replicated_static_roles` retains those
storage dependencies through selection. `PartitionOwnership` distinguishes them
from ordinary invocation roles: materialization, effective parameter operations
and residency sizing include the replicas, while target observation ownership
remains unchanged. V3 predictors declare their borrowed target embedding this
way. Disabling the predictor removes its auxiliary requirements before selection.

Loaded distributed parameter discovery joins ordinary target ownership with the
placement retained by successful prediction construction. A target parameter,
including a shared embedding replica, keeps its ordinary storage declaration.
A separately owned prediction parameter resolves through its retained task and
architecture declaration. Both paths use the same bounded alias/companion
resolution and single-group coordinate compiler. Prediction groups use their
actual tensor-only placement, retaining the global participant rank and replicas
over pipeline and expert axes. Metadata reservation precedes allocation; neither
path opens sources, selects execution again, or allocates native tensors. Native
catalog assembly, projections and atomic overlays consume those neutral maps.

Prepared prediction units can retain a separate floating source layout for
load-time transforms. The architecture applies selected auxiliary formats to the
executable module and compiles source and destination coordinates separately.
Native prediction materialization consumes those modules and exact tasks through
the ordinary bounded quantizer and local binding builder. Conversion selects this
rank's source geometry before reading tiles, and binding does not shard a locally
transformed result again. Completed auxiliary conversion reports join ordinary
session telemetry; selected resource authority is unchanged. The V3 affine
4-bit/group-32 and MXFP4 CPU Ring component/parameter matrices verify this
construction across all seven parallel-axis combinations and three target residency modes;
published FP8 and other supported prediction encodings have their own native
acceptance matrices in the component-analysis guide.

The prepared source graph preserves the selected drafting mode in its retained
target architecture. Disabled and external drafting remove the additive prediction
declaration before construction requirements are derived, matching cold selection.
The full admitted artifact remains available as provenance; this does not authorize
auxiliary materialization. Neutral coverage compares the requirements retained by
selection and source preparation across all four drafting modes.

Component descriptions can declare an optional grouped linear stage before the
final write matrix. The neutral `ComponentGroup::write_column` joins one scalar
to exact regions of actual loaded factors, preserving loaded support and alias
facts. It allocates only the small region descriptions and grants no execution
authority; ordinary bounded parameter operations perform reads and contractions.
Architecture declarations own grouping, latent dimensions and the two stage
observations. A matrix product must not conceal a selected projection's input
rounding: the grouped output and final multiplication input are distinct evidence.
Existing direct groups omit this field. V4 discovery and its shared execution hooks consume this contract; a declaration
alone still grants no native observation authority.

Grouped component placement expands complete first-projection row groups into
scalar channel coordinates. Intermediate output and final-input observations use
the latent axis, checked against the final matrix's column ownership. Equal local
orders are accepted across range/index representations; split or reordered rows
inside a group are rejected. Repeated invocations retain separate paths while
sharing each factor's declared source identity. These projections are architecture
contracts and do not select a device or materialize a factor product.

Architecture descriptors carry these grouped write semantics; readers
must check the descriptor wire version before interpreting final write columns. The first
factor's actual input retains `[batch, group, sequence, channels_per_group]`
geometry and partitions on complete groups. `GroupedNeuralBackend` supplies a
projection-input observer with the same deferred admission contract as ordinary
linear projections. MLX routes it through its existing selected linear mechanism,
including FP8 input rounding, before selecting the grouped diagonal outputs.
Architecture instrumentation forwards that evidence without an eager diagnostic
reshape. An absent observer creates no additional paths, views or evidence tensors.

V4 attention instrumentation lives in its portable attention driver: the
mutable channel boundary follows inverse rotary aggregation, before either output
factor. Its ordinary path keeps the existing tensor operations when no observer
is installed. V4 block serial, provider and tensor-parallel entry points share one
portable driver; native collectives remain injected operations. It reports actual
normalization inputs and already-computed hyper-connection coefficients, preserving
hash-selected routing in observed resident TP execution. Discovery owns channel,
latent and stream geometry. Target and prediction hyper-stream readouts are architecture-owned, with
separate sequential and fused prediction equations and invocation scopes.


`ComponentStreamResidual` declares target residual broadcast, ordered stream
cycles, and final learned collapse. Architecture parameter ownership determines
cycle producers; retained input/output ownership determines the two outer
boundaries. Placement checks the full declared geometry and pairs original and
effective observations before capture admission. Complete stream values remain
replicated even when their sublayer writes required a native parallel sum.

`eredu-nn::TensorValueObserver` generalizes the existing projection-input observer
name without changing its deferred-generation contract. `HyperHeadOperator`
accepts an optional coefficient observer. MLX borrows its already-computed
coefficient tensor before the ordinary sum, preserving the original neutral
failure and source if admission rejects it. Family equations, coefficient timing,
intervention semantics and score reconstruction stay in the architecture and
consumer layers. Native code adds no V4-specific readout policy.

An architecture can declare that its observed unit executor owns both unit
boundaries, including interventions and effective evidence. Shared layered and
partitioned traversals then omit their outer delivery of those same boundaries.
V4 uses this contract because output intervention precedes its retained target
capture. Custom executors without internal observation retain the ordinary outer
hooks. The contract prevents duplicate records and double application of edits
without moving family timing into runtime or backend policy.

A separately invoked residual can declare `ComponentResidualBase::ProjectedSum`:
each term retains its own normalization, effective projection, actual multiplication
input and optional explicit axis broadcast before addition. This represents
separate projections without inventing a concatenated parameter matrix. The
architecture declares these terms and the runtime continues to own bounded
parameter authority; the declaration grants no materialization or editing rights.

Sequential V4 prediction discovery separates checkpoint parameter roots from
observation invocation paths. The predictor owns its outer input/output seams;
the nested decoder owns fused-stream seams. Prepared prediction construction
retains the selected executable formats, original source layout when conversion
is required, and routed-unit coordinates from the neutral partition plan.
Complete stream observations remain replicated across prediction execution
replicas; component columns retain their tensor partition. The shared prediction
driver supplies prefill, proposal and replay hooks to the existing controlled
speculative executor. DSpark uses its own context/fused phase declarations and
shared instrumented context/proposal equations, described below.

Prepared V4 prediction replicas consume the target embedding and vocabulary head.
The neutral selected requirements therefore replicate both static roles across
pipeline stages while retaining each tensor-parallel shard. This applies to both
sequential prediction and DSpark. Role replication drives ordinary materialization,
parameter ownership and residency accounting; prediction does not use an unbound
placeholder head on stages that do not publish target logits. Disabling the
prediction extension removes these extra consumers.

V4's architecture-owned checkpoint specifications also define which logical
matrices feed configurable projection operators. Load-time quantization consumes
that declaration. Rank-two integer routing tables, additive compressor positions,
dense hyper-connection coefficients and dense lookup embeddings retain their
native storage. Their rank alone does not grant a packed-linear transform, and
the backend does not infer their family semantics from parameter names.

Partitioned V4 transforms retain a separate source-format architecture and local
parameter layout. Neutral preparation checks source/target parameter space, unit
addresses, ownership and pooling-state geometry before handing both models to
ordinary backend materialization. The selected projection formats do not replace
the original source geometry used to read and convert each rank's parameters.


Fused component discovery keeps physical execution groups, pinned parameter roles
and invocation applicability separate. Core owns versioned direct residual-source
and dynamic score-write contracts; architectures declare and validate the actual
source expansion, score broadcasting and producer placement. Prepared prediction
layouts consume these declarations and retain complete context/proposal replicas
alongside tensor-sharded component columns. A context-cache invocation has no
invented FFN or score head. Runtime phase filtering uses the admitted context or
fused scope and preserves completion, delivery and cumulative accounting. The
backend neither chooses the semantic phase nor supplies a family-specific score
equation. DSpark declarations alone do not establish complete native hook support.


The materialized fused-prediction contract accepts a caller-owned temporary
lane for observed proposals. The shared speculative driver retains that lane
and actual logits through ordinary auxiliary completion and final capture
publication, then discards the proposal state. Accepted context is unchanged.
Context preparation and replay use their existing committed prediction lane;
architecture hooks observe and edit the inputs actually used to update it.
Separate module and observer borrows span each complete native invocation.

The prediction extension declares the physical sequence extent of cache prefill.
Sequential predictors consume shifted hidden/next-token pairs; DSpark consumes
the entire accepted target capture, including a single-token prefix. The shared
phase wrapper uses that neutral fact for admission and runtime shape checks.
Native collectors do not infer this distinction from a family or path name.

Transient native sliding-cache checkpoints retain the sealed history required by
the saved frontier. Their lightweight catalog guards prevent semantic discard
while allowing ordinary device/host/disk movement; retained blocks stay charged
to the existing manager and process pool. Restoring a frontier does not transfer
the guard into the resumed live cache, so releasing the saved checkpoint permits
normal sliding discard. Durable snapshots keep their independent namespaces.
The neutral pooling-cache checkpoint contract returns a typed result because
history-retention admission can fail before speculative work. Native visible-range
fetches also reject incomplete history before returning an attention tensor.


Component and architecture descriptions include a measured scalar
output gate to a component group. The neutral declaration identifies its affine
read, actual multiplication input, activation, and original/effective gate. Its
position is after the group's projected write and optional output normalization,
before the complete contribution hook. Architecture code owns the sigmoid and
multiplication order; backends continue to provide ordinary linear and elementwise
mechanisms. Qwen shared experts use this contract because their scalar gate follows
the down projection. Disabled observation retains the same order without extra
component tensors. This declaration does not itself establish loaded or partitioned
capture support; placement and execution acceptance remain separate.


Qwen hybrid component observation follows the architecture-owned serial and
parallel block drivers. The parallel shared-expert path retains the ordinary
complete/partial routed reduction order and emits its actual gated local write.
The text layered adapter forwards embedding, internal-unit, partition and readout
hooks; the shared static-module projection helper preserves tied/untied vocabulary
projection and observes the actual multiplication input. Native adapters still
supply mechanisms and collectors. Prediction-unit primitives do not by themselves
enable prepared prediction or conditional-media component capabilities.

Prediction-free Qwen hybrid text partitions use the neutral dense/routed
prepared constructors for TP, PP and EP. Cold state selection derives heterogeneous
attention and recurrent geometry without a native context. Construction retains the
global parameter description separately from local modules, binds selected local
expert-bank specifications while retaining global router decisions, and maps target
unit addresses into the selected PP cache slice. Source and transformed parameter
layouts are checked before the ordinary family-independent materialization visitor
runs. These paths use the existing bounded residency and communication handoffs;
prepared conformance and native validation are recorded in the component guide.

Qwen recurrent convolution placement follows the concatenated Q/K/V projection:
each TP rank receives the corresponding slice of all three channel segments.
The ordinary linear-attention equations and native convolution mechanism consume
this architecture-declared layout; they do not reinterpret contiguous checkpoint
rows as local head ownership.

Conditional Qwen target observations use the same text blocks and final projection
as ordinary execution. Input evidence is taken from the assembled text/media value
at the first target block, after execution-group assembly. Each DeepStack addition
uses one shared serial/parallel driver and exposes the actual contribution after
placement at decoder token positions. Discovery links these contributions to their
target-layer invocation and includes them as whole terms in the readout equation.
The token embedding parameter describes the lookup branch; the observed assembled
input is the primary residual base. Numerical and native partition acceptance for these conditional hooks is
recorded in the component-analysis guide.


Composite partition strategies retain the observer through their selected direct
or routed call and through the final projection on its output owner. Text and
composite routed execution share provider agreement, expert transport and tensor
parallel dispatch, with an optional observer over the same driver. Capability
facts combine the retained architecture hooks with the actual selected strategy;
the composite input adapter forwards serial, parallel, routed and final-partition
hooks instead of inheriting ordinary fallback calls. Scalar composite conformance
uses the shared selected-residency preparation driver, so bounded cases acquire
units through the same portable policy contract as native binding.


Pinned parameters may declare exact unit consumers separately from their static
binding role. Runtime validates those consumers against the canonical execution
graph, selects one copy on each owning partition and includes that storage in its
ordinary selected-task accounting. They remain pinned modules, rather than unit
payloads. Shared Qwen vision declarations use this contract for the final merger
on the last vision unit and each DeepStack merger on its configured vision unit;
patch and position modules keep the ingress role. This fixes source-bound pipeline
execution without giving a backend ownership of vision placement or replicating
all static weights on every stage. Numerical and native acceptance is recorded
in the component-analysis guide.


Conditional Qwen construction retains the selected expert-bank specifications
alongside its text and vision geometry. Both ordinary composite and embedded
prediction target constructors pass those local bank specifications to the shared
block factory; the router keeps global expert identities. EP source binding then
uses the admitted local bank shape instead of attempting to fill a global bank
with a local payload. This follows the same architecture-owned construction used
by the text-only target path.

Controlled capture prefill consumes the ordinary prepared-input admission. The
architecture supplies the assembled decoder geometry (including media positions);
`eredu-core::capture::CaptureRequestShape::validate_prefill` performs the exact
comparison. Native erased adapters carry this optional request geometry into the
existing admission/result path, before any model forward, so peer rejection and
state-preservation follow the ordinary distributed transaction. They neither
reconstruct media-token policy nor admit all media based on collector support.

Completed composite MLX sessions expose the same retained parameter-slot, bounded
loan, replacement-publication and reset-state operations as completed text
sessions. Their adapters delegate to the shared neutral session and residency
policy, and include selected prediction modules and independently cached bank
slots. Native binding retains these handles and prepared materialization tasks;
it does not reopen source artifacts or infer component ownership from names.

Conditional observation support remains a discovery fact owned by the portable
runtime. Core capture and intervention admission accept those sites without
claiming that the input or invocation condition holds. Runtime capture emits
`Missing` for absent scheduled sites; scheduled interventions must finish before
commitment, so their absence fails through the existing transactional driver.
No backend parses condition descriptions or promotes them to unconditional
support. Unsupported and unverified phases remain inadmissible.

Prepared-input accounting goes through the retained executable adapter.
Composite architectures project their ordinary whole-request admitted part plans
into neutral accounting plans; the backend adds native payload/metadata bytes to
those architecture-owned workspace estimates. Processor selection checks are
shared with ordinary preparation. Counting neither executes the model nor
reconstructs a family-specific plan, and does not compute a prompt-content
fingerprint.

Prepared composite routed execution retains scalar unit coordinates derived
from the original selected bank, its localized plan, and physical parameter
layout. The resident provider uses the same invocation and coordinate wrapper
as text partition banks. The scope encloses all chunks and closes on failure;
nested expert exchange borrows the scope. Disabled observations use the ordinary
provider path without creating tensors or coordinate payloads.

Each successful composite partition group finishes its layerwise policy
with the actual completed output or transport boundary, just as text pipeline
execution does. Dropping the guard remains the failure path. This closes dense
streaming windows before later parameter operations and prevents a successful
forward from being misclassified as an aborted policy invocation.

Qwen hybrid prediction separates its checkpoint-shared fusion and three
normalizations from the independently streamable decoder block at each depth.
The hybrid decoder's generic pinned extension carries those canonical modules;
conditional composition preserves that extension when consuming the text
modules. Static unit-consumer ownership places one copy on each partition that
executes a prediction unit. Repeated logical depths therefore do not create
repeated canonical parameters or charge one fusion copy per depth on the same
partition. Prepared prediction construction materializes the shared module once
from its retained tasks and exposes it once to parameter control, followed by
independent block modules. Native binders consume these typed owners; they do
not infer sharing from checkpoint names.

Logical MLX subgroups that share a world collective complete each packed
collective before returning to its caller. This preserves the admitted
participation-wave order when an active pipeline stage builds independent
embedding and vision branches and inactive stages submit zero-work entries.
Leaving those world collectives lazy can match unrelated operations when graph
consumers evaluate branches in different orders. The rule applies to packed
sum, gather, and point-to-point fallbacks; it agrees with the existing completed
neighbor-exchange path. The backend owns this native ordering, without changing
the architecture's wave declarations or parameter ownership.

Intermediate world-wave completion uses the selected communication deadline and
retains its input, output, stream, and native group in the ordinary completion
owner. A timeout quarantines those resources and fences group reuse until safe
completion; ordering does not introduce an unbounded wait into a contracted
communication path.

Prepared Qwen text prediction pairs with the ordinary replicated text target
as well as the conditional target. Both call one architecture-owned fusion and
prediction-state driver; their adapters supply only token embedding and shared
vocabulary projection. The generic replicated decoder reuses its existing
prediction capture and readout helpers. Source construction selects text or
conditional parameter descriptions before deriving the same text/MTP geometry.

Prepared Qwen hybrid prediction executors borrow the ordinary target embedding
and output head and thread internal observers through the shared fusion/decoder
driver in prefill, proposal and accepted-token replay. Discovery places each
prediction decoder in its own score scope while retaining canonical shared fusion
and normalization identities. The target and extension continue to own separate
mutable state; backend binding supplies storage and invocation without interpreting
prediction equations or checkpoint names.

Qwen prediction targets also reuse the architecture-owned routed and direct
partition constructors. Typed pairing occurs after the retained target admission
and source-store identity checks; the existing mixed-state backend visitor supplies
binding. Prediction producer placement admits pinned shared parameters only for
their declared static role and consuming execution group. It retains both sides
of the normalization intervention boundary across execution replicas.

Prepared resident prediction banks use the runtime's shared invocation scope to
finish routed observation work before entering a tensor collective. Architecture
construction binds their retained global unit coordinates; Qwen and DeepSeek
reuse the same scope and partition mapping. Ordinary residency providers keep
their existing outer scope, and disabled observation inspects no tensors.

Qwen prepared prediction declares replicated target embedding and output roles
in the neutral load requirements, like other extensions that borrow the target
head. Every pipeline replica binds the selected tensor shard and residency
estimates count those copies. Ordinary target input/output hook ownership stays
with its declared pipeline stage; auxiliary storage grants no publication role.
Prediction fusion also registers read-only embedding input evidence under its
own logical invocation, separately from target embedding observations.

Composite prediction input obtains semantic token IDs through the architecture's
prepared-input contract. Qwen and Inkling share their ordinary ingress token-part
construction with prediction, including admitted media placeholders and projected
spans. Backends perform generic prepared-input admission and call that contract;
they do not infer placeholder IDs or discard media while extracting text tokens.
Token identity construction creates no media features, grids or tower output.

Nemotron prediction units share one architecture-owned fusion and final-normalization
driver across ordinary and observed serial/parallel paths. Observed scheduled
operators call the existing target component hooks; the effective normalized and
fused values feed the actual decoder. The first physical unit owns fusion and the
last owns readout normalization, preserving each depth's physical schedule.

Nemotron prediction discovery keeps each depth's physical attention/MoE schedule
separate from its target layers. Its component joins use the declared `mixer`
parameter roots, first-unit fusion and last-unit normalization. Prepared extension
construction binds resident expert coordinates from those declarations and retains
shared embedding/head replicas through neutral load requirements. Prediction
prefill, proposals and accepted-token replay share the same observed driver.

Nemotron-H prepared prediction groups use the same retained tensor-rank layout
as the other embedded prediction families. They replicate across target pipeline
and expert coordinates, with shared embedding/readout roles retained on each
participating rank. The architecture preparation driver owns this policy and the
physical prediction schedule; native binding consumes those prepared units.
For transformed prediction weights, preparation derives executable modules and
their companion geometry from the retained task formats. It separately retains
the source tensor-rank layout for bounded conversion before binding; this avoids
applying a global source shape to a local transform or reconstructing format
selection inside the backend.
Routed Nemotron target construction likewise retains a separate source
architecture when selected tasks transform weights. Preparation checks identical
local state geometry and unit ownership before handing both layouts to ordinary
materialization. Native quantization does not reconstruct that source model.
Packed MLX embedding lookups normalize selected decoded rows to F32, matching
the backend's cold activation/state-width fact. This includes MXFP4, whose native
dequantizer otherwise defaults to BF16. The conversion precedes vocabulary
reduction and uses only the requested rows; it does not widen an entire table.
The runtime's uniform partition-range expansion converts retained encoded blocks
to architecture-supplied scalar widths. DeepSeek, Nemotron-H and Kimi Linear use
that shared checked conversion for expert ownership; a block count is not a
neuron count. Missing ranges and explicit nonuniform chunk layouts remain errors.
Composite partition preparation also reports the exact admitted recipe sources
assigned to other expert owners. Native strict-loading validation consumes that
declaration, as it does for text partitions, so split expert checkpoint layouts
retain the same source-ownership policy as packed layouts. Unrelated keys remain
unexpected; this does not authorize ignoring a checkpoint-name prefix.

Inkling retains a source-free validated `ComponentDefinition` before native
module construction. Its parameter description, execution graph, unit counts
and paths project that definition. The declaration owns optional vision/audio
transport, independently leased vision and text units, target cache/convolution
slots and output-owned embedded-prediction state. Vision parameter geometry and
encoded companions derive from the hMLP fold specifications through shared
linear-format expansion. Parameter inspection and partition source planning do
not construct native modules to discover their storage. Rank-local construction
validates the same canonical parameters against selected state geometry.
`LayeredModel` is the shared `ComponentAdapter` over `InklingKernel` and its
typed pinned-resource aggregate. The runtime supplies parameter traversal,
address and dependency checks, retained state mappings and ordinary, parallel,
routed and partition driver contracts. The kernel supplies equations and
observation/intervention seams, explicit collective algorithms and partition
boundaries. Family composite admission uses that same adapter. Specialized MTP
operations borrow the kernel's target embeddings and readout through the
ordinary parameter and completion lifecycle; they retain independent prediction
state. Raw kernel operations remain available through typed kernel access.
Partition construction consumes the exact declared selection, derives its cache
offset from selected state and retains only canonical pinned roles owned there.
Source-format and execution-format partitions retain their own selected modules
and parameter authority. Unowned factories and pinned-resource operations reject
through typed errors; repeated selection retains the same geometry and ownership.

Inkling component instrumentation shares its ordinary decoder equations, including
relative attention and all four causal convolution histories. Unit/channel hooks
precede their actual down/output projections. Projection writes, the learned
dense-branch scalar, and the complete post-convolution residual contributions are
separate evidence. Routed and jointly weighted shared banks borrow the runtime's
unit-observation adapter on their ordinary provider requests. These equations and
semantic boundaries remain in `eredu-architectures`; providers retain residency,
placement and completion ownership.

Inkling's prediction primitive uses the same component driver as its ordinary
text blocks. It preserves the released sequence of two uses of the same hidden
normalization, independent embedding normalization, hidden-before-embedding
fusion, the ordinary decoder, and optional chain normalization. Captures identify
each normalization application and effective fusion input separately; shared
normalization parameters do not merge logical invocations or their mutable state.

The materialized Inkling prediction adapter forwards its admitted observer through
those same operations, including observed prefill and explicit depth advancement.
The target-owned vocabulary projection exposes its actual muP-scaled multiplication
input as read-only evidence and its original/effective scores after vocabulary
trimming. Disabled execution uses the same driver without constructing observation
paths. Target state and each prediction depth's causal histories keep their existing
owners; an observation failure does not imply that partially advanced state can be
reused without restoring the lane.

Component normalization contracts explicitly represent identity normalization for
heads that omit a data-dependent norm. Optional gains and biases remain affine;
there is no RMS denominator for this case. The evaluation layer consumes this
neutral declaration when constructing measured readout directions, including a
zero residual. Family declarations select the equation from normalized configuration.

Architecture and component descriptions retain
`component_transforms`, explicit identity normalization, and typed tensor-transform equations.
Transform records join actual input/output observations to their owning nodes
and effective parameter groups. Constant/learned scaling, normalization and causal
depthwise convolution remain architecture semantics. They are not extra additive
residual terms, and their presence does not imply loaded capture or edit support.
Inkling declares its four decoder convolutions and dense learned scalar from the
same normalized layer policy used by construction. Convolution kernels use the
canonical `[channel, 1, tap]` layout, ordered oldest to current, including the
identity bypass. Current-position evidence does not grant historical-cache access.
Ordinary bounded parameter operations continue to admit every read or edit.

The component activation contract also distinguishes gated delta attention from
softmax attention. It describes key-by-value recurrent updates, Q/K scales after
head normalization, per-head log transition rates, key-channel decay biases,
headwise update reads, and normalization and gating before the output-channel
boundary. Convolution transforms can declare their post-convolution activation.
Kimi supplies these declarations from the same normalized configuration as its
portable KDA driver. Its MLA declarations retain normalized query/KV bottlenecks
and the admitted fused or split KV-B layout. Shared SwiGLU declarations reuse the
ordinary gated-unit builder. The backend supplies the existing projections,
convolution, recurrence, parameters and completion resources.

Shared component equations close each validated group before its result enters
dependency traversal or partition publication. Parallel closure uses the same
ordinary hook unless a kernel supplies distinct collective semantics. Kernels can
retain an exact prediction-target capture and declare non-output placeholder
geometry independently of the state profile. These hooks do not grant prediction
execution support; explicit family preparation and target contracts admit that
operation. Captured roots remain in the ordinary forward-retention lifecycle.

Nemotron-H declares canonical target and patterned prediction parameter topology
from normalized configuration before native materialization. Mamba projections
retain their five input and three convolution segments, per-head scalars and
grouped normalization scale. Attention, dense and routed/shared expert declarations
retain their semantic partition units. Prediction depths own group-relative units
with flattened checkpoint paths, first-unit embedding/hidden norms and fusion,
and last-unit normalization. Static embedding ownership includes the prediction
role and tied output consumers. The shared runtime expands declared linear
formats and packed ReLU-squared bank geometry into exact physical companions;
family code retains checkpoint identities and format policy. Full metadata and
selected/source preparation consume the same cold declaration.
Materialized prediction-extension preparation also derives source and executable
metadata from this declaration. It selects exact formats and local placement
before constructing typed prediction units; embedding and readout continue to
borrow target resources.

Qwen hybrid expert ownership planning accepts normalized text or conditional
configuration, optional validated local geometry and neutral rank topology.
Replicated and partition planning use the same derivation for target and MTP bank
identities. Conditional planning validates the vision policy and its text-local
geometry without a backend model, then supplies the retained expert plan before
rank-local executable construction.

Qwen hybrid text parameter metadata is declared from normalized configuration
and the same recurrent, attention, routed, dense and static construction specs
used by its equations. Runtime encoding expansion supplies physical companions,
row layouts and placement units. Target and MTP groups retain distinct execution
owners; shared prediction fusion and norms declare their actual MTP consumers.
Full, selected and prediction-extension text preparation consume this
checkpoint-global declaration without constructing native modules. Conditional
Qwen composes its text and shared vision declarations into one cold parameter
description, preserving canonical group identities across the composite graph.
Conditional construction validates and retains one `ComponentDefinition` before
native materialization. Graph traversal, unit addresses, parameter metadata,
transport and state ownership project from that definition. Optional vision units
have empty mutable-state mappings; target and MTP slots retain text coordinates
and rank-local recurrent/cache geometry. The target pipeline owns its unit-aligned
state, and the output owner retains prediction state. Conditional partition and
prediction-extension preparation consume the cold parameter description without
constructing a discovery model. Low-level unit parameter descriptions
accept configuration and validated group coordinates without native modules.

Shared Qwen vision declarations and materialization consume the same patch,
position, normalization, attention, MLP and merger specifications. Runtime encoding
expansion supplies physical companions, row layouts and placement units. FP8 QKV
scale segments use block-row coordinates; weight and bias segments retain weight
rows. Ingress storage, the final merger and each DeepStack merger keep their
explicit static role or logical-unit consumers. Qwen3-VL and Qwen4 experimental
metadata consume these same vision declarations. Family code retains patch and
shuffle geometry, window schedules, DeepStack capture order and media equations.
`VisionStaticSelection` resolves complete ingress and merger resources from those
same canonical parameter groups before allocation. It rejects incomplete encoded
or normalization/projection resources, duplicates and incompatible declarations.
`VisionStatic` retains optional patch/position/final-merger modules and sparse
DeepStack banks in their original bank identities. Complete and selected resources
run the same equations; an operation rejects a missing required resource before
submitting its equation. Parameter-free continuation geometry remains available
when a rank retains no pinned vision parameters.
`VisionStatic::select` consumes an already constructed aggregate and retains a
validated subset without rebuilding parameters or changing local placement. It
rejects a different root or configuration and any resource absent from the retained
aggregate. Repeating the same selection preserves its modules. Explicit static
ownership may refine a resource's declared logical consumers while preserving its
exact canonical parameter groups.

Full and selected Qwen hybrid text execution uses `ComponentAdapter` with one
`QwenKernel`. The adapter and kernel retain the same `Arc<ComponentDefinition>`
before native materialization. The shared adapter supplies graph traversal,
validated addresses, parameter visitation, retained roots, partition ownership
and text/runtime contracts. Typed hooks retain recurrent and attention equations,
routed provider access, target captures, MTP fusion and collective semantics.
Observation capabilities retain the selected path's explicit target-only predicates.

Qwen pinned resources contain optional embedding, readout boundary, untied head
and shared MTP modules. Full construction supplies all declared roots; selected
target construction supplies only the canonical static owners present in its
partition bindings. Tied output partitions retain their actual embedding replica.
The same kernel and resource equations serve serial, parallel, observed and
partitioned execution. Local construction retains checkpoint-global parameter
geometry with planner-derived mutable state and validates exact unit/state
ownership. Low-level equations and consuming module extraction live on the kernel;
extraction rejects selected ownership that lacks required complete-model roots.
Conditional Qwen materializes its declared text and prediction roots
directly from the shared construction specs. Its serial, parallel, observed and
partition readout entry points share one equation over borrowed decoder resources;
target readout normalizes once and prediction readout consumes normalized hidden
values. Conditional Qwen uses one request-ingress equation for serial and
tensor-parallel target or prediction input. Strict token lookup selects the
retained vocabulary mechanism; media validation, masks, projected-input checks,
DeepStack placeholders and request roots share the same preparation. Pipeline
vision ingress consumes the same typed part preparation. Text state lookup uses
the declaration's invocation mapping and rejects slots outside selected pipeline
ownership. Low-level state-view restriction validates the exact declared slice
and global offset independently of retained parameters; identical selection is
valid and an incompatible retained selection fails.

Replicated dense LFM2, Kimi Linear, Nemotron-H, Qwen hybrid and DeepSeek V3
construction retains the same source-free family definitions used by their other
execution paths. Fixed-state and compressed-state kernels supply family equations
to the shared `ComponentAdapter`; addressing, graph and parameter descriptions,
pinned-role traversal, text projection and completion contracts use that adapter.
Typed profiles adapt attention-only, fixed-only, mixed, compressed and stateless
cache views without requiring unrelated backend state capabilities. Each invocation
keeps its own declared layer state and independent residency unit. The shared
decoder resource aggregate owns embedding and tied or untied readout. Its public
instrumented serial/parallel finalization and `final_hidden_position` helper also
preserve low-level readout operations independently of a complete model. Readout
preserves full projection, final-position text projection and observation timing;
Nemotron-H and Qwen targets retain their external prediction captures and ordinary
shared prediction resources. Family precision recipes and checkpoint admission
remain inputs to the existing preparation driver.

Conditional target and external-prediction partition construction consume a cold
`ArchitecturePartition` before native allocation. One materializer builds complete
or selected resources from the retained definition and canonical bindings. Selected
text roots contain only the physical embedding, normalization and output owners;
tied output keeps its declared embedding replica. Vision retains only selected
ingress and merger resources. Unit construction rejects unowned global addresses,
and selected local geometry, state, boundary and parameter ownership are validated
against the same definition.

Receiving conditional ranks retain declared request placeholders and parameter-free
vision continuation geometry without evaluating unowned token or patch parameters.
Text products track whether input-owner computation or transported values make them
available; assembly rejects an unresolved product. Completion roots retain every
prepared part's token and embedding tensors as well as assembled and transported
products. Vision edges carry compact DeepStack features. Decoder continuation
boundaries carry contributions expanded to the complete decoder sequence,
matching their wire schema and removing the
receiver's need for the original visual mask. Expansion and target-unit addition
use the same contribution equation and preserve observation attribution.
Conditional Qwen uses `ComponentAdapter<Arc<ComponentDefinition>, ConditionalKernel,
KernelParameterModules>` for full and selected vision/text execution. One retained
kernel supplies ordinary, parallel, routed, observed and partition equations;
the shared adapter supplies the runtime contracts, validated addresses, graph
projections, parameter traversal, state ownership and completion roots. Composite
hooks retain media admission, assembly and transport geometry. Composition consumes
the authoritative cold partition used for materialization, including its global
state offsets and selected local state.
Low-level media, prediction and restricted-state operations live on
`ConditionalKernel`; constructors return the shared adapter.

Complete and selected Nemotron-H target readout uses one `TargetReadout`
equation over borrowed decoder resources. It normalizes once, then returns hidden
values or projects through the actual tied embedding or untied head. Serial,
parallel and observed calls use the same equation and projection instrumentation.
Prediction readout projects the already-normalized draft activation and omits
target-only observation seams. Low-level target and prediction operations use
these same resource equations.

Full Nemotron-H execution uses `ComponentAdapter` with
`CompleteNemotronKernel`. The adapter and its low-level kernel share one retained
`Arc<ComponentDefinition>`; construction validates it before allocating pinned
modules. Graphs, parameter ownership, primary/prediction groups, transport,
invocation paths and state slots project from that definition. Target state
follows pipeline unit ownership; appended prediction state belongs to the output
owner. Rank-local construction retains exact local state geometry, while the
explicit ordinary state-layout operation returns global geometry. Group assembly
receives dependency activations after shared adapter validation.

Complete kernels retain target/draft selection, embedding ingress, patterned MTP
factories, provider geometry and draft readout without target normalization.
Resident, bounded, text and observed drivers use the shared adapter. Explicit
observation reports preserve the admitted coverage of complete and selected
kernels. Pinned roles borrow actual module owners, and family checkpoint rewrites
reach binding through the shared kernel recipe hook. Low-level partition and
prediction primitives remain on `kernel_mut()`; driver-facing partition calls
require retained selection. VoiceChat backbone and materialized prediction
composition consume these same adapter/resource contracts.

Nemotron-H derives expert ownership and rank-local ReLU-squared bank
specifications from normalized `ModelArgs`, optional validated `LocalGeometry`
and neutral rank topology. Target units and repeated prediction patterns retain
separate group-relative invocation addresses and flattened bank identities.
Replicated planning uses the same derivation. Selected and transform-source
preparation consume these plans without constructing a native model for expert
planning. The backend materializes the declared banks and supplies provider
resources.

Selected Nemotron-H targets use `ComponentAdapter` with `NemotronKernel`.
The validated definition retains graph, canonical parameters, global unit paths,
state references and transport class. The kernel retains exact Mamba/KV geometry,
provider equations, typed target boundary schema and readout policy. Group closure
captures the unnormalized target activation. Boundary transport carries original
tokens, effective input embeddings and exact readout control; supplied embeddings
can select normalized hidden output without vocabulary projection. The shared
adapter validates selected ownership and addresses, visits borrowed canonical
pinned roles and projects retained state slots. Low-level prediction embedding
and projection operations use the actual modules through `kernel_mut()`.
Existing materialized prediction extensions consume these target hooks through
the same typed composition contracts. Embedded prediction retains its typed
family equations, conditional group selection and prepared state.

Complete and selected Kimi Linear execution use `ComponentAdapter` over the same
`KimiKernel`. Typed complete and partition geometry retain global invocation
ownership, actual local state indices, specialized expert banks, ordinary global
state admission and tensor-local parallel admission. The kernel retains KDA and
compressed MLA equations and borrows canonical pinned roles through
`DecoderStaticResources`; actual complete or selected modules remain owned by
their aggregate. The validated definition owns graph, canonical parameters,
paths, state references and transport. Constructors validate that definition
before materializing pinned modules, and the shared adapter validates selection,
addresses and dependencies. Ordinary and parallel observations use the same
typed provider hooks with the resident provider. Low-level block construction,
embedded ingress and serial/parallel block operations remain on `kernel()` and
`kernel_mut()`. A partition driver requires retained selection; low-level kernel
partition hooks use explicit admitted state and context.

The shared pipeline kernel contract declares both unit-level routed reductions
and optional bank-specific reductions. A bank uses its unit's routed declaration
unless the kernel supplies a bank-specific order. Kimi declares one mixer
reduction before exchange and two feed-forward reductions for a sparse unit
(routed output and shared expert), or one for a dense unit. A mismatched routed
selection rejects before collective execution.

Kimi's ordinary, routed, replicated and partitioned execution use the same
instrumented sublayer drivers. Readout uses borrowed decoder resources over the
actual pinned modules; capture happens at the input/output owner and
state lookup retains the partition's local ordinal. Discovery alone does not
establish loaded collector support or native validation.

Component reads can name their actual affine `projection_output`. Tensor-transform
joins describe causal operations before subsequent head normalization. A shared
grouped projection record describes Inkling's relative-query features multiplied
by the learned `[relative_feature, relative_position]` table for each head. The
observation retains current-position profiles before their ordinary axis
permutation; it is not an attention-probability or historical-cache capture.
Readout `scaled` evidence separately records muP multiplication before the selected
projection's input arithmetic, so input rounding cannot be attributed to scaling.

Primary component readouts can declare embedding normalization after assembly and
scaling. The captured post-normalization embedding remains the residual base;
consumers must not substitute raw token rows for it. Inkling joins this declaration
to its shared embedding norm, final norm and untied output projection, and describes
its muP scaling through a tensor-transform record. Whole post-convolution writes
contain their scalar projection groups, so a score decomposition includes the whole
write once or expands its declared transforms without counting both.

Routed component declarations can specify `routes_per_token` independently of an
ancestor router's top-k. Inkling uses this for its separate selected and always-on
bank invocations; the shared branch is not relabeled as an independently routed
MoE. Both declarations consume the existing family expert specifications and
actual jointly normalized unit coefficients. This does not advertise standalone
routing events that the joint selector does not emit.

Inkling prediction preparation retains one typed module per physical depth and
one optional canonical chain-normalization module. The ordinary whole-chain API
and prepared executor share the same depth equation. Source constructors retain
checkpoint encodings; executable constructors consume the exact selected task
formats. Module visitation reserves ordinal zero for the optional shared norm and
uses depth plus one for decoder modules, so a shared parameter is visited once.
The native prediction materializer applies the retained residency policy to
these physical owners. Shared normalization overlaps the active sequential
module; bounded policies page depth modules through the common residency ledger.

Prediction component scopes describe the actual static parameter roles when the
parameter graph has no separate pageable execution groups. Inkling additionally
declares source embedding normalization before the per-depth embedding norm and
two applications of the same hidden norm before hidden-first fusion. The optional
chain norm uses the neutral identity normalization equation when disabled. These
semantics remain architecture-owned; native adapters consume ordinary prepared
modules, exact tasks and observation paths.

Prepared Inkling prediction retains the parameter declaration and exact local
layout lent to materialization. Its depth weights remain replicated, while shared
embedding and output parameters retain their ordinary tensor-axis placement.
Prediction selection retains embedding, embedding-normalization and output roles
on every prediction replica, including pipeline stages that do not publish target
scores. A static sequential component scope proves ownership through these roles;
it does not invent an execution-unit owner. Discovery's physical layer inventory
still describes each real prediction block independently of residency ownership.
Released aliases retain canonical projection roles during recipe finalization, so
selected formats and their companions agree with the constructed operators.

Component placement follows declared affine reads and tensor transforms through
exact retained layouts. Affine read outputs keep their actual row coordinates;
convolution kernels prove channel ownership, and shared grouped projections map
complete input groups to output heads. A whole residual-write invocation can
establish both sides of its channel-preserving convolution, including a sum of
routed and shared writes. Missing dependencies and inconsistent coordinates are
rejected before capture admission.

Routed placement joins each component's exact write parameter to its retained
expert catalog, keeping logical execution ownership separate from provider cache
addresses. Per-invocation catalog distribution decides whether expert coordinates
are partitioned or replicated. Equal-sized routed and shared banks therefore
remain distinct. Prepared provider coordinates use the same catalog declaration;
shared invocations preserve their configured cardinality and global identities.
Native mechanisms consume these facts without family-specific placement branches.

The selected grouped realization retains per-invocation catalog distribution
through format selection and local construction. Replicated invocations execute
their complete local bank without expert route exchange; partitioned invocations
use the retained expert assignment. Both paths share the same ordered provider
success agreement across tensor, expert and optional pipeline-wave groups before
a caller can reduce or publish results. Inactive pipeline stages follow the
selected agreement-only wave for replicated providers. Catalog distribution is
execution policy, independent of the numerical equality of bank dimensions.

`AttentionCache::relative_attention` is the neutral cache-aware contract for
learned distance profiles. Resident caches consume their returned history;
blockwise caches consume their retained blocks. The default rejects a blockwise
cache without that mechanism so a caller cannot silently attend only to newly
appended keys. Inkling uses the same operation for target and prediction layers.
MLX reuses its leased block scan and online softmax accumulator with per-block
profile bias, shared with contiguous relative attention. Query-position scaling
and relative profile gathering remain native general mechanisms; the architecture
owns profile parameters and attention policy. Sliding history is discarded only
after every submitted query has consumed it, and scan telemetry includes the
additional bias/scaling workspace. No complete-history tensor is reconstructed.

Composite partition construction also retains an optional typed source architecture
and its exact local layout for selected load-time transforms. The optional source
is heap-owned across handoffs; completing the typed admission uses a separate
construction frame so unrelated family branches do not multiply stack storage. Inkling target and
prediction-target construction obtain this source from the retained original
configuration, preserve routed/shared ownership, and check execution-unit
addresses, state and vocabulary ranges against the selected target. The shared
composite session handoff passes both descriptions into the ordinary partitioned
materialization mechanism. Native adapters bind them without reconstructing
source configuration or reopening artifacts. Other composite families use the
same handoff when their source construction supplies it.

Inkling's joint routed/shared selector consumes a floating router matrix through
its neutral tensor operation. Requirements therefore classify that matrix as a
raw parameter, consistently with family format policy and parallel declarations.
Model-wide affine/MXFP4 conversion applies to the declared linear modules and
expert banks; it preserves this router, relative-profile tables and convolution
kernels in their original floating representation.

Bounded checkpoint selection pushdown retains leading-axis restrictions nearest
the physical source. It pushes each independent selection through the original
tree once and reapplies existing selections without recursively exchanging their
order again. This prevents rewrite cycles while allowing expert/member and row
ranges to form a contiguous source tile before column selection. The checkpoint
crate proves this without native tensors, including all six axis-selection
orders and matrix-row tiling with distinguishable source values.

Inkling prediction accepted-token replay uses the same observed prefill driver
as initial lane preparation. Its declared internal-observation support therefore
also applies when target verification rejects a tentative draft. Attention-input
interventions occur before the prediction cache update; later cached predictions
retain their effect. Observation alone preserves ordinary replay state exactly.

Addressable partition providers retain route cardinality per invocation. Expert
exchange lowers routed invocations to one route per exchanged row, while shared
invocations keep their complete replicated route set. Before acquisition, binding
checks exact member keys, catalog distribution, owning group and selected bytes.
Routed members use the retained rank-local global-ID map; shared members use their
own complete ID range. A shared bank may contain more members than the routed
bank, and a foreign unit cannot satisfy validation through aggregate entry counts.

Kimi Linear and LFM2 dense and routed partition construction retain the source model
when lowering includes a parameter transform. The architecture reconstructs its
source formats from exact admitted requirements, verifies target/source semantic
addresses and local state geometry, and passes its typed source layout into the
shared runtime factory. This includes the routed expert realization and shared
bank geometry; the backend only materializes the retained source and target.
LFM2 declares complete dense and routed parameter topology from normalized
configuration without constructing native modules. Its embedding formats,
attention/convolution geometry, dense layers, selectors and packed expert banks
supply exact canonical members and ownership before placement. Runtime expands
selector resources and encoded projection companions through shared neutral
geometry helpers. Routed partition preparation uses these declarations for both
selected and transform-source layouts.
Full LFM2 construction validates and retains the same component declaration
before allocating pinned modules. Ordinary construction retains global state;
tensor-parallel construction retains admitted local state and creates its pinned
modules directly at that placement. Parameter discovery, execution addresses and
unit counts project from this retained declaration.
LFM2 derives expert ownership and local bank specifications directly from
`ModelArgs`, optional validated `LocalGeometry` and rank topology. The operation
has no backend type or native context. Replicated expert planning uses the same
algorithm, and partition preparation consumes these neutral plans for both source
and selected parameter layouts.
Kimi Linear uses the same ownership boundary: normalized `ModelArgs`, optional
validated `LocalGeometry` and rank topology determine expert plans without native
modules or a backend type. Replicated planning uses that algorithm with complete
ownership. Routed preparation consumes these neutral plans for both selected and
transform-source layouts.
Kimi's complete parameter topology is also cold: KDA projections, convolutions,
per-head/per-channel parameters, MLA direct or low-rank queries and
fused or split KV-B, dense/shared feed-forward projections and routed banks derive
from normalized policy. Exact linear-format expansion supplies encoding
companions and compatible partition units. Full metadata and dense/routed
partition source preparation consume this declaration without constructing
native modules for discovery.

Kimi attention and LFM2 attention/convolution companions inherit their primary projection's semantic head or
output-column sharding through neutral parameter metadata. Their own physical
shapes determine the scale/bias intervals; checkpoint-name suffixes do not make
companions replicated when their primary matrix is sharded.
LFM2 routed expert ownership expands encoded logical partition units through the
shared checked range contract before deriving component and bank coordinates.

Qwen hybrid's recurrent attention declares scalar value-head output channels.
The same ordinary, provider and parallel block drivers observe fused projected
Q/K/V, their causal SiLU convolution, control projections and the consumed
normalized/gated channels. Original/effective channel hooks precede the actual
row projection. Its read-only input evidence comes through the shared projection
observer, after any selected dynamic input transformation. With instrumentation disabled, these seams neither create path
strings nor evaluate or retain extra tensors. Channel interventions occur after
the recurrence has updated state; sublayer-input interventions occur before it.

The neutral gated-delta equation distinguishes key and value head counts and
widths and per-head versus per-key-channel decay. Qwen uses additive-epsilon L2
normalization of Q/K, contiguous head repetition, per-head decay and a SiLU output
gate. Kimi uses its declared RMS/scaling convention and per-key-channel decay.
L2 is explicitly `x / sqrt(sum(x*x) + epsilon)`. Evaluation readout directions
honor this denominator without substituting RMS or a clamped norm. Recurrent
writes with scalar component declarations do not also appear as opaque
readout terms, while routed combined FFN contributions remain explicit.

Composite partitions retain the same immutable partition-bank selection as
routed text: bank identities, per-invocation routes, local component coordinates,
exact member tasks and selected residency. Shared member ownership follows its
catalog distribution and actual execution-unit owner rather than the routed
expert range. The common architecture construction driver binds resident or
independently cached providers to this retained selection. Native composition
supplies the shared cache and transfer mechanisms, excludes independently
materialized targets from ordinary unit loading, and retains bank handles for
bounded parameter work, atomic overlays and telemetry. Composite pipeline waves
and media boundaries continue to use their original architecture-owned schedule.
Addressable member keys identify bank invocations independently of their owning
execution unit. Runtime validates every retained parameter task against that
owner, while architecture catalogs validate the invocation and member identities.
This permits routed and shared invocations to belong to the same decoder unit.
Gemma 4's ordinary and routed TP/provider drivers translate global layer
identities through the retained partition state offset before borrowing caches.
The layer identity still selects equations, per-layer inputs and shared-KV roles;
only mutable-state access uses the local ordinal.
Its expert modules also consume the retained localized grouped specification;
router identities and cardinality remain global. This keeps EP-local storage
geometry separate from routing policy and preserves the same block construction
for resident and independently cached providers.
Muse's centered normalization derives its effective gain from the current
authoritative parameter on each invocation. It does not retain a separate gain
tensor across parameter replacement or unloading, so overlays and residency
transitions cannot reuse a stale derived parameter.

Component kernels declare the meaning of an invocation's complete output.
Context activations and media features report no prediction-score kind, so the
shared session driver bypasses its model-score observation seam for those
outputs. Causal scores retain the existing default; refinement kernels declare
their own score kind from request-local invocation context.

Shared component adapters validate group identities and dependency cardinality
before serial or parallel group ingress. Parallel kernels can place token lookup
or another collective at the group's declared wave; the default uses the ordinary
group equation. Invalid ingress cannot advance kernel state or issue a collective.

The standalone Muse waveform tower uses `AudioKernel` with the same shared
component adapter. Its validated definition declares each patch operation,
convolution block, and output projection as an independently leased unit, with
one state slot per invocation and explicit stateless slots between histories.
The adapter supplies addressing, parameter traversal, state projections, and
serial/parallel runtime contracts; empty pinned resources use the shared empty
module. Audio equations retain cadence, clip reset/continuation boundaries,
request-local advancement, and independent convolution frontiers. Waveform
features report no prediction-score kind. Composite Muse execution borrows these
same audio equations and continuation geometry.

Muse construction returns the shared `ComponentAdapter` over `MuseKernel` and
its kernel-owned pinned resources. The adapter supplies graph, address, parameter,
state, provider, observation, and typed partition contracts from the retained
validated definition. Complete and selected partitions use the same equations;
selected ownership restricts unit acquisition and pinned traversal before execution.
Low-level Muse operations borrow the kernel through `kernel()` or `kernel_mut()`.
Muse retains media assembly, waveform clocks and histories, text cache geometry,
routed expert specifications, and assistant target operations. Parallel decoder
ingress resolves pending token lookups at its declared collective wave before
running the ordinary group assembly equation. Serial and parallel media calls
share the same vision and waveform equations; ordinary text calls use the same
provider-aware block driver with resident expert access.

Muse component execution uses a single architecture-owned block driver for
ordinary, provider-backed, tensor-parallel, and observed calls. Its hooks expose
the actual gated attention channels and FFN units before their selected output
projections, and both sides of the separate post-projection RMS normalizations.
The block postnorm acts on the complete reduced write; Glimmer's sparse expert
contributions are summed before that normalization. Compressed experts additionally
normalize each reconstructed expert output before the mixture. The same architecture-owned final
readout handles tied and untied heads and retains output multiplication after
the actual vocabulary projection. Backends continue to supply ordinary tensor,
projection, sparse-unit, and collective mechanisms.

Component descriptions include an explicit normalized-head
`output_scale`, token-only embedding normalization before media assembly, and
`scaled_softcap` for `cap * tanh(affine_score * scale / cap)`. The captured
assembled embedding remains the residual base. Additive score contributions
refer to the unscaled affine score, with the actual projection input and its
rounding retained as evidence. These are neutral equation declarations, not
claims of loaded support or new backend family branches.

For a complete route-weighted residual write, neutral component placement may
establish a post-normalization input from its already established output owner.
This follows the normalization's preserved axes and invocation. Admission still
requires the complete normalization axis, exact input/output geometry, and
replicated gain/bias parameters owned by that invocation; a foreign pipeline
stage's gain cannot authorize the input capture.

Muse's expert checkpoint alternatives discriminate packed and split storage by
the gate/up identities. Their shared down projection remains required by either
selected layout, but its presence cannot select both alternatives. Header-only
admission validates the complete selected layout before native materialization.

Muse load-time media quantization derives eligible matrices from its checked
architecture-owned checkpoint geometry. It selects aligned patch, position,
attention, FFN and adapter matrices, retaining dense storage where an input
axis cannot hold the requested packed blocks. Exact checkpoint format admission
remains separate. The retained selected formats drive cold capabilities, memory
plans, materialization and cache identity; backends do not repeat this selection.

`ParameterTransformConstraint::LinearIfAligned` carries the architecture's
optional packing policy into neutral realization selection. Valid requests whose
blocks do not fit retain the exact native source format; invalid requests still
fail. Muse applies this declaration to media projections after canonical recipe
finalization, so source aliases and derived outputs retain the same policy. Text
projections retain their required transform semantics.

Muse GGUF requirements derive logical shapes and canonical identities from the
GGUF text plan and, when present, its separately admitted projector. GGUF expert
recipes consume the retained prepared source. A text-only GGUF never needs a
SafeTensors vision configuration merely to enumerate executable parameters.

Mixed Muse ingress defers vocabulary-parallel text lookup until the primary text
execution group. The preceding replicated vision group performs no text lookup
collectives. A shared neutral helper declares one exact reduction shape per text
segment, excluding media positions; inactive pipeline stages consume that same
ordered schedule. Inkling uses the helper for its existing segmented ingress.
These are architecture equations and schedules, not backend family branches.

Muse canonicalization is shared by source recipes and parameter requirements,
including the generated scale/bias identities for load-time quantization. The
released source prefix is provenance, not an executable companion name. Its
optional vision root also shares one cold/loaded transport declaration; absent
projectors contribute no static vision role to text-only partition ownership.

Muse selected partition transforms retain a separately constructed, typed source
architecture and physical source layout, as the existing composite transform
contract requires. The construction driver validates execution addresses, state,
vocabulary ownership and static roles against the selected target; it also binds
the source's local expert realization. Native mechanisms consume this retained
source and the admitted tasks. They never reconstruct source geometry from
packed target slots. GGUF's learned per-head Q/K norm gains appear as replicated
architecture parameters; the Hugging Face convention remains weightless there.

Qwen3-VL declares parameters without native modules by composing the ordinary
Qwen text topology and shared vision specifications. Text execution owners map
to the composite decoder group; canonical targets, tied storage roles, encoded
companions and placement rules retain checkpoint-global geometry. Complete and
tensor-parallel construction validate and retain one `ComponentDefinition`
before allocating modules. Graphs, logical paths, transport, parameter catalogs
and complete state reports project that definition. Vision invocations declare
no decoder state slots; text slots include the persisted mRoPE position delta.
Target and transform-source partition preparation derive layout from the same
cold parameter declaration.

Complete and selected Qwen3-VL use one resource materializer. Selected construction
consumes a cold partition and validates its canonical parameters, state, boundary,
vision geometry and exact normalized local text policy before native allocation.
Text roots retain only physical embedding, normalization and readout owners;
tied output retains its embedding replica. Vision roots follow canonical ingress
and merger-bank consumers. Static visitation and derived recipes follow those
actual resources. Expert ownership is derived from cold configuration and local
geometry. Target and transform-source composition consume the same cold partition
used for materialization. Low-level unit/state views validate exact local policy
and retain pinned resource ownership independently; identical retained views are
accepted and incompatible reselection fails.

One ingress equation serves serial and tensor-parallel requests. Receiving ranks
retain explicit unavailable text products and parameter-free vision geometry;
assembly rejects unavailable products until transport installs them. Completion
roots retain original prepared products, rotary values, position delta and
DeepStack values. One readout equation over borrowed decoder resources serves
ordinary, parallel, observed and low-level output operations.

Full and selected Qwen3-VL execute through
`ComponentAdapter<Arc<ComponentDefinition>, VlKernel, KernelParameterModules>`.
The adapter and kernel retain the same definition. The adapter implements graph
traversal, validated logical addressing, selected state and parameter ownership,
pinned visitation, retained roots and the ordinary runtime contracts. Typed kernel
hooks supply ordinary, parallel, routed, observed and partition equations.
Ordinary units use the same provider-aware equation with the resident provider.
Architecture composite hooks retain media admission, assembly, collective schedules
and boundary geometry. Typed boundary resume validates the complete retained
local state independently of earlier stateless groups; pipeline ingress validates
the first selected group's optional state geometry. Both paths validate actual
state and output ownership against the same retained selection. Constructors on `VlKernel` return the shared adapter;
low-level pipeline and independent unit/state-view operations live on the kernel.

Qwen3-VL's text traversal uses the ordinary Qwen decoder's component driver,
including resident and independently supplied experts, rotary embeddings and
local mutable state. Its architecture wrapper adds the assembled embedding and
final readout seams and instruments prepared DeepStack additions at the point
where they enter the residual. Disabled hooks use the same ordinary addition.
DeepStack declarations require media input and apply during prefill; they appear
as separate residual terms, with their own original and effective captures.
Traversal continues to own unit boundaries, so unit output includes the vision
addition. Family equations and semantic branching remain in architectures.

Qwen3-VL's selected transforms use the same typed source-partition handoff:
source encodings and local layout remain distinct from packed target slots,
while execution units, state, static ownership and DeepStack roles must agree.
Optional vision packing uses the neutral alignment constraint and retains dense
matrices when the requested blocks do not fit. The shared Qwen vision format
lookup resolves relative module names and canonical checkpoint names to the
same encoding and rejects conflicting aliases before construction. Cold
requirements and module construction consume that same lookup.
The Qwen vision position table remains an ordinary dense lookup and is excluded
from transformable projection geometry, in both Qwen3-VL and conditional Qwen.
Its rank-two storage alone does not authorize affine weight packing.

Component and architecture descriptions include two neutral relationships
for shared-state and residual equations. A read can identify the attention
component group that actually publishes its projected state; its consumer does
not acquire a fictitious local projection or a capture of historical positions.
Readout equations can reference ordered transformations of the whole residual,
including learned scalar parameters, after each layer's writes. These transform
references affect the embedding and all prior writes. They grant no parameter
query or edit authority. Serialized descriptions with those fields omitted default to local reads
and no additional block transformations.

Gemma4 component integration uses its existing portable dense/routed block and
shared attention cache equations. Ordinary, controlled, provider, and partition
entry points share text preparation and the same instrumented projection and
readout drivers. Native backends continue to supply generic tensor, cache,
projection, collection, and parameter mechanisms. Family topology and the
publisher/consumer and residual-scaling semantics remain architecture-owned;
Gemma4 neutral and native acceptance is recorded in the component-analysis guide.

Declared whole-residual component transforms establish complete hidden-axis
capture placement at their architecture-owned block invocation. The portable
partition builder validates the referenced layer and containing block before
propagating the existing typed transform equation; learned parameter placement
remains independently checked. Gemma4 attention-bias checkpoint admission
follows its normalized projection schedule, including shared-KV consumers and
full-attention key reuse. Neither rule is reconstructed in a native adapter.

Gemma shared attention across pipeline cuts uses architecture-owned typed
current-position key/value roles. The receiving partition declares one additional
KV history for each remotely published attention policy, attached to its first
local consumer; subsequent local consumers borrow that publication. Cold state
selection and native construction use the same replica geometry, so residency,
snapshots, prompt-cache compatibility, and copies include those histories.
Pipeline transfer scales with the current sequence length rather than the full
cached prefix. Fixed wire roles preceding their publisher are inactive and never
consumed; the publisher replaces them before any shared read. Key-reusing
projections still transport and retain both normalized value and rotary-key
payloads, since these tensors differ after their respective transformations.

Gemma attention executes through the owning cache whenever that cache retains
history outside the returned K/V tensors. Shared consumers borrow the exact
publisher cache, or the first local receiver cache across a pipeline cut. The
architecture chooses this semantic owner; the backend implements ordinary
cache-aware attention without reconstructing the layer schedule. Published
tensors may contain only the current submission, so direct tensor attention
must never stand in for reading a paged history. The neutral numerical matrix
exercises that contract against a resident full-history oracle.

A partition's additional state must enter the neutral requirements before
mechanism capability synthesis and selection. `ReplicatedTextRequirements::with_state_layout`
validates the unchanged layer address space and traversal access, then lets the
ordinary selector admit, place, and account for every declared component. It
does not relax the selected-state comparison at binding time. Gemma uses this
for its receiver histories; the subsequent TP geometry projection preserves
the selected component roles and physical storage policy.

Composite architectures also own packing and restoring their intermediate
pipeline activations. Default continuation hooks preserve the supplied tensor;
Gemma flattens its padded media batch into the admitted sequence dimension and
restores the encoder batch from the retained request geometry. Native backends
perform ordinary tensor reshaping and transfer, without interpreting media parts.

Gemma 4 constructs a validated, source-free `ComponentDefinition` before native
modules. It retains the execution graph, canonical parameter ownership, invocation
paths, state references, and transport contracts for the complete or TP-local
geometry. Execution metadata projects this retained definition. Cold partition
and transform-source preparation read the same catalog without constructing
native media layers. Pinned text resources declare their storage roles alongside
their parameter groups, including the tied embedding/readout owner.

Gemma execution uses `ComponentAdapter` over `GemmaKernel` for serial, observed,
parallel, routed-provider, and partitioned calls. The kernel supplies the same block
equations, media assembly, external-assistant captures, and typed auxiliary boundary
values. Runtime owns address/dependency checks, canonical pinned-role traversal,
selected state mapping, and partition-contract validation. Low-level family operations
remain available on the kernel.

Partition construction consumes the exact ownership declaration and retains only
its canonical pinned resources. Shared-KV receiver histories form part of the
validated selected definition, with global invocation indices mapped to local state
rows independently of parameter ownership. Complete and transform-source construction
use the same kernel and adapter. Parameter-free media continuation prepares request
rotary/mask state from configuration; image layers and unstandardized pooling do not
require pinned patch/position storage. Learned standardization retains its declared
storage owner, and audio output retains the existing atomic subsampling/output group.

Gemma audio checkpoint admission and resource declarations share logical parameter
geometry. Resource catalogs retain only the configured output bias and expand each
matrix's configured native encoding through runtime's ordinary format projection.
Checkpoint admission retains its optional output-bias contract. Vision declarations
use the shared source-free vision topology; both media towers retain independently
leased encoder layers and their existing pinned resource boundaries. Shared KV
receiver state and per-layer input transport keep their family-defined geometry.

Gemma's cold and constructed media transport use one architecture-owned phase
ownership declaration. Encoder input parameters belong to the first encoder
stage and output projectors to the last. The current atomic vision static group
is also retained on the last stage when learned standardization is enabled;
the atomic audio static group is retained on the first and last stages because
it contains both subsampling and output projection. Selected materialization
accounts for these replicas. Continuation stages prepare parameter-free media
masks/rotary state and shape placeholders; they do not execute unowned embeddings
or input encoders. Only the first decoder stage performs segmented vocabulary
lookups, and the declared inactive collective waves reflect that ownership.

Component and architecture descriptions include optional routed-bank
`write_output` and `output` boundaries. They describe the complete bank write
after route weighting and distributed reduction, and its transformed output
before combination with other branches. Architecture declarations join these
to the existing tensor-transform equations. Portable partition derivation checks
input/output widths and exact invocation ownership; transform validation still
checks normalization axes and learned parameter placement. This exposes Gemma's
routed post-normalization without inferring a bank write from a shared FFN sum.

Gemma's key-reusing publishers declare two physical cache payloads in the base
state layout as well as pipeline receiver replicas. Reuse applies to the raw
projection only: value RMS normalization and key normalization/rotary transforms
produce distinct tensors. This is ordinary neutral KV storage; a key-only state
would lose the value history and undercount storage.

Composite pipeline boundary delivery retains the semantic destination alongside
its typed payload. A result from one independent media root can arrive while a
different root is executing; only the declared destination installs that result.
An intervening group cannot consume or reinterpret the pending boundary. This
policy lives in the portable executor and applies equally to all composite
architectures and native transports.

Gemma media format selection uses the projection's logical input width in both
cold requirements and module construction. Family-wide quantization requests
retain unaligned media matrices in floating storage under the existing family
alignment policy. Logical-shape recovery expands encoded schema dimensions
only; a floating schema tensor is already expressed in logical coordinates.
Media-to-text projectors consume the same exact media format as their checkpoint
schema, including under global text quantization.

Gemma partitioned load-time transforms retain a separately constructed typed
source architecture and its exact source layout, as other composite families do.
The architecture driver checks execution-unit addresses, selected state, media
ownership, vocabulary ranges, and per-layer input ownership before passing both
source and target to native materialization. Backends consume that prepared
handoff rather than reconstructing family configuration from target tensors.

Transform source modules use the selected target's partition ownership. Runtime's
`derive_transform_source_layout` maps those exact cuts into the source encoding
with checked axis scaling, preserving logical ranges and independent expert
placements. Architecture constructors use the resulting geometry for both
source modules and prepared-source validation, including composite families and
prediction extensions. A floating source is not independently rebalanced before
conversion: for example, a 96-column matrix converted in 32-column groups over
two ranks consumes source cuts of 64/32. Native materialization consumes those
retained cuts without interpreting family equations.

Qwen3-VL, conditional Qwen hybrid, and Muse partition static roles describe
execution consumers. The final
stage retains the `output` role even when readout shares the embedding matrix.
The architecture's parameter group declares its embedding/output consumers and
selects the required replicas. Geometry validation must not replace that
consumer role with a physical parameter name when weights are tied.


Prepared prediction modules receive the complete execution's retained weight
residency policy from the total construction driver. Physical owners remain
separate from logical depths: shared normalization/fusion modules receive one
policy and unit identity regardless of their number of consumers. The typed
materializer's invocation contract carries an equation result together with
output and changed-state dependencies even when the equation fails. Unit and
shared owners overlap until those dependencies settle. DSpark's context and
proposal loops use the same equations through this scoped module mechanism.
Backends retain native completion and parameter leases; architectures enumerate
the actual state and output roots. MLX keeps exact bindings and unloaded
module placeholders until invocation. Target construction registers the additional
physical owners and their exact stores before initializing one common residency
manager. Queries acquire one owner through the same loan; reversible edits live
in a per-module replacement set reapplied after population. Completion precedes
unloading and bounded device eviction, including equation failure. Native
behavioral matrices verify this combined path; cold sizing is checked against
constructed modules and actual rank geometry.

Qwen3-VL GGUF companion admission uses its composite projector name translation
for both token-pending and token-bound plans. The two temporal patch slices stay
distinct exact source keys until the architecture's stack recipe constructs the
single patch-projection parameter. A physical source slice never becomes an
independent executable topology target.


The neutral residency controller can validate each physical unit against its
retained exact catalog while sharing one plan and reservation ledger. The MLX
manager retains an immutable per-unit source registry for this contract; batch
reservation, initialization, alias-owner acquisition and eviction still use the
same ledger. Identical source keys in distinct transformed stores remain distinct
values. A residency report keeps each alternate source view's diagnostics
separate because views may share physical counters. Target and prediction
construction register their units together before initialization, preserving one
weight reservation ledger.


Prediction preparation also declares whether a physical module is shared while
other units execute. This residency overlap is distinct from tensor/pipeline
parameter placement. Shared fusion, normalization and DSpark statics must be
counted alongside the largest active unit; their shared checkpoint ownership
must not force all prediction units into one resident block. The generic native
module-transfer helper retains both changed-state and output roots with the
source transfer through existing submission recovery, including equation errors.
The prediction materializer uses this helper for both execution and scoped
parameter queries. Runtime's `AuxiliaryWeightRequirements` computes the shared
plus largest sequential owner requirement for paged execution and the complete
physical total for resident execution. Native load admission and cold support
reports use those values alongside the ordinary target window.


Cold prediction residency is represented separately from parallel parameter
ownership. Runtime's `AuxiliaryModuleResidency` is retained on parameter
requirements and selected materialization tasks. Architecture declarations assign
physical units, including Nemotron's full physical pattern and DSpark's shared
input/readout modules. Prepared module construction checks that a declared group
matches the actual module's task set and shared/sequential role. Runtime sizes
shared owners plus the largest sequential owner for a bounded phase; resident
execution counts all owners. Ordinary bounded admission includes auxiliary
tasks as well as the target tasks. Rank-local sizing uses the same tensor-only
prediction placement as construction, counting a complete prediction schedule
on every target pipeline and expert coordinate. The caller supplies the complete
architecture parameter description. Enabling prediction can also replicate
target statics, such as token embeddings on later pipeline stages; their selected
storage ownership contributes separately to pinned bytes. A neutral numerical
fixture compares cold totals with constructed module tensor geometry across
tensor, pipeline, expert and combined eight-rank placement. Shared-phase sizing
and constructed-module roles are also checked in runtime and native fixtures;
the complete family/format paging matrices are recorded in the component-analysis
guide.

Gemma 4 primary GGUF admission maps text tensors into the complete family's
`model.language_model` parameter tree. The standalone text/assistant mapper
remains available for those construction contracts. This keeps token embedding,
per-layer input and decoder-unit ownership identical across SafeTensors and GGUF;
projector tensors retain their separately declared media owners.
The GGUF layer multiplier and routing scales map to scalar parameter identities,
including the router's per-expert scale; they do not inherit a matrix weight's
suffix or projection owner.

The component transport audit distinguishes invocation membership from parameter
storage. Ordinary hooks use `ParameterGroupOwner::is_owned_by`; auxiliary copies
are accounted through `is_stored_by` and do not add ordinary callback peers.
Prepared target hooks use the complete tensor/expert stage, and replicated
prediction hooks use the session group. With the retained pipeline-major rank
order these are connected Ring memberships, already covered by bounded member
status agreement. An arbitrary disconnected low-level membership still receives
the existing typed rejection: a world-participation wave cannot run independently
of inactive pipeline peers. No such relay is required by the selected component
execution paths described here.

Detached prompt and sampler preparation seals and observes its native submission
scope before returning. An existing operation error remains the reported cause
even if the scope also reports failed or blocked work. That error does not grant
completion: unresolved roots stay in the same recovery owner until terminal
evidence permits retirement. A generic status error is used only when the
operation itself supplied no error.

GGUF admission retains each Qwen primary tensor's exact packed format before
constructing embedding, expert and linear parameter topology. Standalone and
Qwen3-VL contexts share that resolver, using the context's declared parameter
root. Qwen3-VL and Qwen3.5 companion admission also records vision-block and
DeepStack-merger formats before constructing their composite configuration.
Facade-owned media-token binding preserves those facts. Backends consume the
resulting typed bindings and recipes; they do not infer missing family formats
from packed byte widths during native construction.

Shared architecture requirement discovery converts packed GGUF schema byte widths
back to logical coordinates using exact block geometry, rejecting partial blocks
and overflow before capability selection. Encoded storage widths never stand in
for the logical dimensions of a linear operation.

Prepared composite matrix-format handoff includes executable embeddings as well
as linear projections, preserving quantized lookup and tied-readout storage.
Normalization vectors and fixed dense position tables keep their separate roles.

Muse GGUF projector recipes flatten the admitted spatial patch kernel into the
family-declared linear patch input order. Cold requirements retain that logical
geometry, and backend binding consumes the prepared recipe without family-specific
reshaping. Both raw and canonical source names refer to the same checked layout.

Native communication uses the stream selected by the actual MLX communicator,
independently of model computation. The safe native wrapper exposes that retained
stream; backend groups share its ownership with logical groups and completion
recovery. Ring therefore communicates on its CPU transport while model operations
can run on Metal. Compatible compute streams retain caller ordering and device
affinity; singleton collectives keep the compute stream without initializing a
transport device. Initial manifest exchange uses the same stream selection and
retains its exact group through completion, before rank-local contract checks.
Native tensor dependencies order the two streams. Packing and
model arithmetic keep their compute stream, and existing communication budgets
continue to cover their owned buffers; no physical allocator-workspace guarantee
is inferred from this stream selection.

Manifest and session consensus preserve transport causes in the neutral generic
protocol error. The MLX adapter retains that exact policy/transport error through
its backend failure conversion, so application diagnostics can reach the original
native source without exposing native error parameters in facade operations.

Rank-local runtime construction also retains architecture and mechanism errors
through the typed factory handoff and final session wrapper. Partition executor
and traversal-policy constructors retain their neutral error types as sources.
Boxed errors retain the leaf itself, so a policy or I/O error without a further
source remains discoverable through the public neutral failure chain.
Vocabulary and tied-embedding readout gathers also retain the original native
exception through the neutral neural error conversion.
The replicated and composite session adapters retain typed failures during
parameter loans, state reset, snapshots, prompt-cache operations and ordinary
prediction-lane execution instead of replacing them with display strings.
If a prediction operation and its state recovery both fail, the adapter retains
both errors and exposes the recovery cause. An earlier state-preserving rejection
cannot authorize retry after that recovery failure. This does not itself prove
native completion or release pending resources.
Setup failures do
not imply native completion or permission to release pending resources.

DeepSeek V4's published FP8 default applies to linear projections. Its text and
Markov embedding tables, and target/prediction hyper-head functions, retain their
dense schema and operator semantics. Explicit separately admitted embedding
formats remain distinct from that default. Vocabulary construction derives token
ownership from the architecture-declared weight rows; FP8 companion rows index
128-entry blocks and retain their own format-derived physical placement.

DeepSeek V4's family format resolver retains published dense embedding, router
and DSpark Markov-embedding defaults, including under block-FP8 metadata. Explicit
selected per-parameter formats take precedence for admitted transformations. Cold
lowering and executable construction consume these same family-owned decisions.
The same resolver distinguishes published expert encodings from the global
projection default: mixed FP8/MXFP4 experts keep full per-row scale geometry.
Schemas, routed construction and partition placement share this decision for
target and prediction layers.

Prepared partition discovery retains an immutable index of its exact selected
tasks, aliases, companion outputs and checked parameter members. Its size is
bounded by those existing declarations, and it is rebuilt when parameters are
rebound. Queries still compile ownership and reserve the same per-call metadata
and transport costs; indexing avoids repeated full-declaration scans during
all-layer experiments. It confers no loaded authority and retains no tensors.

DeepSeek encoded expert assembly retains scale and affine-bias companions under
the same source selection, expert ordering and gate/up concatenation as weights.
Target and prediction preparation consume the complete canonical recipe outputs;
the backend does not reconstruct family source names or infer missing scales.
Independent expert residency carries these same companions in each bounded unit.
Its optional intermediate selection respects source packing and scale blocks;
pre-encoded weights and their companions remain preserved, while ordinary sources
retain eligibility for the selected load-time conversion. Target and prediction
format decisions come from the architecture's existing routed-block policy.

Native catalog preparation indexes borrowed selected output names once per query
under an additional finite host-metadata reservation. It continues to validate
each loaded slot against the retained neutral placement before publishing facts.

Muse GGUF routed recipes enumerate decoder layers independently of the composite
vision-plus-text execution ordinals. Projector recipes retain their own semantic
source roles; adding vision layers never creates text-expert source requirements.

Generic native execution-layout and unit preparation retain typed architecture
failures through the backend error source, including resident-policy conversion.
The outer public failure remains backend-neutral; preparation does not flatten
these causes into diagnostic strings.

Muse vision pipeline edges use architecture-declared flattened patch geometry
within the tower and flattened projected-media geometry at decoder ingress.
Cold continuation bounds expand the selected decoder extent by the spatial merge
area; request admission supplies the exact grid and placeholder counts. The
portable composite driver transports those declared roles, without treating
image patches as text positions or adding backend family branches.

Partitioned target readout discovery registers the actual projection-input
observation with output-invocation ownership, alongside residual and normalized
values. It remains read-only and replicated across output TP ranks; parameter
storage replicas on other stages do not grant observation ownership. The same
contract already governs prediction-scope readout inputs.

A Muse rank that has already executed a vision partition resumes the incoming
decoder activation in its retained request context. The ordinary primary-only
boundary carries no additional learned context and must not trigger a second
text/media assembly.

Muse static vision parameters use exact unit-consumer ownership. Patch/position
inputs and the grouped static norms are available on each vision partition that
constructs request context; the merge adapter and final projection are owned by
the final vision unit. Cold source selection, parameter topology, materialization
and residency sizing consume the same family decision, including replicas.

Portable partition execution preserves typed architecture, policy, state and
communication failures through the neutral neural error source. This includes
active and inactive pipeline waves, dependency completion and boundary transport;
retaining a diagnostic source does not grant completion or retry authority.

Neutral partition communication retains owned submission and completion errors
through `BackendFailure` sources while preserving operation, phase, route and
poison-domain semantics. Diagnostic-only poison notifications remain available
for borrowed failures that callers already retain. Neither form establishes
resource release; native recovery and completion still own that decision.

Composite architectures report whether retained context still contains pending
decoder-ingress collectives. Muse keeps token lookups pending through vision
execution; Gemma and Inkling resolve them at context creation. Inactive pipeline
ranks consume this semantic state when mirroring the declared primary wave.
Native backends execute the same operation order without selecting family branches.

Final source adaptation and partition error wrappers preserve owned errors as
`Error::source`, including cold model/preflight failures and non-architecture
`LayerwiseRuntimeError` variants. The architecture/runtime layers own preparation and residency policy, and the
native backend still establishes completion before releasing native resources.

### Request memory forecasts and prefill chunks

Cold options can derive physical placement and available capacities through the
portable facade's `GenerationMemoryOptions::for_hardware_device`. It matches
backend/device identities exactly, takes a backend-supplied host-execution fact,
and maps hardware observations into execution and host budgets without substituting
installed capacity or application policy. `for_local_device` composes that helper
with local hardware discovery and allocator observations. MLX classifies canonical
device IDs using the same parser as realization, without creating a native device
or stream. The CLI supplies its selected plan device and retains ownership of
application limits and reserves. The explicit-placement constructor remains available.

Ordinary architecture construction retains `TextExecutionTopology` beside exact
parameter and state requirements. Shared decoder and gated-convolution builders
consume the same projection specifications used by this topology; no family
selects allocation multipliers or forecast equations. Runtime owns checked
mechanism/lifetime composition, physical-capacity comparisons and candidate
recomputation. The topology preserves each layer's selected attention arithmetic,
convolution kernel, packed expert routing, physical projection format, explicit
parameter owner and tied output head. `PreparationMechanismProvider` supplies
neutral input-score workspace facts without family inspection. MLX reports its
query tile thresholds and conservative temporary-copy allowances from the same
constants used by its native attention mechanism; selection retains these facts
and includes them in cached-selection validation. Within a full-key input-score
attention invocation (at most 8,192 key positions), MLX prepares expanded K/V and
contiguous BF16 RHS projection layouts once for all query tiles, preserving
reduction order. Large tiled calls synchronously evaluate groups of at most 32 tile
outputs before constructing the next group, keeping completed outputs and shared
K/V while releasing temporary graphs. Small calls remain lazy. Batching is a
native mechanism policy with no family condition. Longer rows retain the separate
two-pass blockwise realization and its evaluated per-block state, without retaining
every expanded block. The optional neutral `FullKeyAttentionTiles` contract
describes the covered key length, shared K/V layouts, maximum live tile count and
retained outputs. MLX reports four shared query-width K/V payloads, 32 live tiles,
two output payloads and 32 working bytes per score element. Runtime combines
these facts with architecture geometry for K/V projection buffers, score conversions,
convolution scratch and overlap. Missing native facts preserve an unknown bound.
Legacy records lacking the full-key contract retain per-tile allowances; longer
key rows keep that conservative fallback. Selection-cache identity includes the
retention facts. No native device is needed to select or estimate them.
Persistent convolution history remains in the state layout. Mixed-width selected
parameter metadata adds a float32 cast allowance and promoted-state/replacement
storage above nominal state bytes; it is derived from task shapes and dtypes,
not checkpoint-format dispatch in a backend.
Fully resident MLX parameter materializations own lazy F16/BF16-to-F32 conversion
caches for dense projections, including tied output embeddings. A weak native
identity registry shares aliases without extending residency; evaluated conversions
are released with their owner. Bounded host/disk policies do not enable the cache.
Explicit device-residency ceilings also disable retention rather than allowing
derived storage to bypass original-parameter admission.

`eredu-core::residency` defines the portable parameter-conversion retention
contract: disabled, bounded payload bytes, and explicitly unlimited policy.
Pure normalization preserves the requested value and managed-default/explicit
provenance, canonicalizes a zero-byte bound to disabled, and reports effective
disabled policy for host-layerwise, disk-streamed, explicit device-ceiling or
unsupported mechanisms. The managed policy is 256 MiB and is enforced by
eligible MLX resident materializations. `ExecutionPlan::with_parameter_conversion_retention`
and `NormalizedLoadRequest::with_parameter_conversion_retention` select a
model-scoped optional override before preparation. `None` retains managed-default
provenance; explicit disabled, bounded (including zero), and unlimited requests
survive normalization and architecture selection. The retained neutral selection
supplies the policy to native residency construction before target or embedded
owners register. Loading another model cannot change a published budget.
Runtime owns shared admission and reservation policy; the backend owns native
conversion storage, evaluation, completion safety and observations. Allocator
caching remains an independent policy. A retained-payload ceiling covers retained
plus reserved conversion payload, not backing capacity, temporary casts, graph
storage, process RSS or total memory. Admission is first-admitted without automatic
eviction; denied retention leaves temporary conversion execution available.

`ParameterConversionRetentionGroup` wraps the existing scoped `ResourceIdentity`
in a distinct budget namespace. One selected loaded execution shares this group
across permanent units and embedded prediction owners. Native partitions must
share admission for that execution; current multi-process MLX construction
disables retention because it lacks cross-process reservation authority.
Separately loaded external drafters have independent groups. Existing conversion
allocation identities and parameter-binding owners remain authoritative for
physical storage. Aliases within a group charge once, while independent groups
sharing an allocation each admit their own retention claim. Group charges cannot
be summed as physical residency; allocation observations still deduplicate shared
backing. Core `OffloadReport` can carry the separate group reports without adding
their payload to its original-parameter ledger. Unsupported or unavailable
observations differ from an observed empty cache or group list. Legacy records
and unsupported telemetry default to unavailable, not the managed policy.
Native backing capacity is separately observable and is not added to payload.

`StaticMemoryReport::parameter_conversion_retention` and
`ResidencyTelemetry::parameter_conversion_retention` preserve the load-selected
policy, eligibility, budget identity, retained claims and reservations. Retained
conversion payload remains a named subset of the memory report's current device
parameters; group admission charges must not be added to physical residency.
`LoadedModel::parameter_conversion_retention` reads these facts through the neutral
`ModelCapabilityBackend` hook. `PlannedModel` composes role-labelled `target` and
`external_drafter` observations through `ParameterConversionRetentionObserver`;
embedded prediction has only the shared target entry. A missing external entry
means no separate participant, while an unsupported entry means a participant
whose ledger cannot be observed. The auxiliary MLX assistants currently report
unsupported, while independently loaded autoregressive drafters expose their own
budget. Execution-plan policy applies separately to both loaded models.

The residency telemetry document preserves `current_device_bytes` and
`peak_device_bytes` as original-parameter admission-ledger counters.
`total_current_device_parameter_bytes` reports the checked sum of current original
parameters and deduplicated retained conversion payload;
`current_device_parameter_conversion_bytes` is a named subset of that total.
Use the total for current parameter accounting, without adding the subset again.
The parameter admission peak does not include optional conversion peaks.
An overflowing total is unavailable rather than saturated. This total covers the
ordinary residency ledger; independently managed routed banks retain their
separate telemetry. Use `StaticMemoryReport` for whole-model parameter composition,
including those banks.

Ordinary and controlled reporting uses the same native owner observations and
preserves backend errors as `BackendFailure` sources. These queries do not
allocate native execution resources, evaluate or settle work, populate conversions,
or consume state/observation budgets. This API has no live limit setter;
a request is immutable once loaded. Explicit settled trimming is described below;
[retention-aware forecasts](#retention-aware-forecast-ownership) consume the same
neutral scope and live binding observations. The
[consumer workflow](generation-memory.md#controlling-parameter-conversion-retention)
and [facade example](../eredu/examples/conversion_retention.rs) use these contracts
without managing another cache or importing native storage mechanisms.

`eredu-runtime::residency::conversion_retention` implements the shared portable
budget and weak registry. Composition may attach one `ConversionRetentionBudget`
to each execution `ResidencyController`; target units, embedded owners and
in-process partitions can receive clones of the same selected budget. MLX
multi-process partitions reject retention as described below. Named layer windows
do not create allowances. MLX residency constructors install the managed budget
without creating conversions. Registry lookup joins live group identities only
when policy facts agree, and immutable source identities only when payload agrees.
Allocation descriptors preserve the actual backing
identity and reject conflicting live payload observations across groups.

Admission atomically counts retained plus reserved payload with checked arithmetic.
A move-only reservation permits one publisher per immutable entry while native
conversion and evaluation happen outside portable locks. Publication consumes the
reservation and returns a group claim keyed by immutable source and actual backing;
failed publication, abandoned tickets and unwinding cancel exactly once. Aliases
share claims, and multiple parameter bindings to the same allocation share its
charge within a group. Reusing another group's published allocation still requires
independent admission; rejection grants no permission to attach its native value.
Claim clones share ownership, and dropping the final allocation binding releases
only that group's charge. Registry maps contain weak references and cannot retain
parameters, models or native storage. Observations expose exact coherent payload
accounting while native capacity remains explicitly unavailable.

Portable lock order is registry, parameter entry, budget; claim release takes only
the budget lock. No native handles, callbacks or lock guards enter or leave these
operations. Backends pair claims with their own native storage and serialize
native attachment/publication with parameter replacement. An `is_active` query
alone is not a native fence. Invalidation permanently retires the old entry,
cancels its reservation and revokes its parameter-specific claims in every group;
it cannot release other parameter bindings to shared backing. Re-registration
creates a distinct entry, and stale ticket publication or destruction cannot
revive the retired entry or alter the replacement's reservations. Backends must
detach revoked native values at a completion-safe boundary; logical release does
not establish physical reclamation. This mechanism provides admission and
ownership accounting and settled-boundary claim release.

MLX retention owners pair each native conversion reference with their own admitted
portable claim. The global native registry holds weak entries and each entry holds
weak owner and backing references, so a denied group's source registration cannot
keep another group's conversion alive. Admission reserves checked F32 payload
before casting; native creation and evaluation occur outside native entry, registry
and portable controller locks. Publication takes the native entry lock, checks
retirement, and attaches the evaluated backing only to independently admitted
owners. Native failures and retirement during evaluation drop or cancel the ticket.
Source publication revokes all obsolete claims; restoration needs a new resident
registration. Native graphs may still retain storage after logical claim release.

One MLX residency manager shares its budget across permanent units and supplementary
embedded prediction owners. Independently loaded models and external drafters have
separate budgets. Multi-rank selected TP, PP, EP and DP topologies currently reject
retention with typed `Unsupported` eligibility and effective `Disabled`: the local
controller cannot coordinate reservations across native processes. Inference remains
available with temporary casts. This is a mechanism wiring gap, not an architectural
model limitation; no rank silently receives its own full managed allowance. Native
cross-process retention remains unvalidated and requires coordinated admission.

Host-layerwise, disk-streamed and explicit device-ceiling selections report their
typed exclusion even for explicit unlimited requests. Selection becomes immutable
before the first native owner registers. `OffloadReport` carries the effective policy
and live group ledger; physical conversion observations include only admitted owner
bindings and deduplicate shared allocations independently from group accounting.

`ParameterConversionRetentionTrimReport` specifies released group claims and
payload, remaining claims/reservations, and independently observed backing
reclamation. Releasing one group's claims neither releases another's nor proves
physical reclamation: snapshots, other owners or native graphs may retain storage,
and allocator caches may retain backing even after its final live reference drops.
No reported claim release means bytes returned to the OS. The trim operation
uses existing submission/completion authority at a settled boundary and preserves
source weights, request state, budgets and eligibility for future admission.
Controlled transactions need typed pending-work rejection unless their operation
contract explicitly permits settlement. Ordinary reset preserves conversions;
trim, parameter invalidation and allocator-cache flushing have separate semantics.
Reports remain read-only and cannot evaluate, settle, trim or populate conversions.

`ModelCapabilityBackend::trim_parameter_conversions` is the neutral settled-only
backend hook. `ModelRuntime` and the ordinary facade synchronize before calling
it; controlled ordinary continuations call it through `TextContinuationBoundary`
after canonical completion and record draining. Controlled speculative execution
uses its existing transaction boundary and shared driver, rejecting pending work
with `ParameterConversionTrimError::NotQuiescent`. Neither path changes request
state or run budgets. Backend errors cross the facade as `BackendFailure` with
the original source. Backends without the operation return typed `Unsupported`.
MLX owns native reference detachment and healthy idle-authority validation;
portable residency policy owns coherent claim accounting. Embedded prediction
uses its target group, while composed external drafting keeps separate scopes.
Ordinary composed trim visits the target and then external drafter; it is not an
atomic transaction across participants. If a later participant fails, callers
must re-query usage rather than assume rollback of completed releases.
Bindings remain eligible for admission after trimming; invalidation alone retires
them. Idempotent claim release and unknown physical reclamation are independent
of allocator-cache flushing.

`safemlx` exposes immutable graph identity through its existing native safety
boundary; no pointer or native dependency enters portable crates.
`Tensor::publish_parameter` is the neutral, infallible publication hook used by
parameter transaction drivers. MLX revokes derived storage before replacing a
parameter; read-only discovery preserves it. Restored values use ordinary casts
until a new resident materialization enables reuse.
Runtime reports conversion payload separately from its original-parameter ledger.
The backend includes it once in `StaticMemoryReport` device residency and exposes
the subset as `current_device_parameter_conversion_bytes`. Current forecast
composition credits only complete selected bindings with a matching live claim
and budget scope, preserving promoted activation/state sizing. Aggregate subset
subtraction applies only to records lacking scoped policy facts.
Cold forecasts retain the full potential cast allowance and separately bound
possible persistent admission; neither forecasts nor reports populate conversions.
Runtime also owns `ForecastCalibration`, including the labeled attention fallback,
cache-copy overlap, layer workspace multiplier and graph allowance. Architecture
preparation retains selected state, ordinary invocation topology and native
input-score mechanism facts in `PreparedModelDiscovery`. Selected materialization
formats replace checkpoint defaults, including load-time quantization; cached
conversion credit reduces only the remaining selected promotion allowance.
The neutral `GenerationForecastBackend` supplies loaded residency, allocator and
capacity observations plus the actual prompt/instrumentation execution contract.
The MLX adapter derives final-row projection from architecture and execution-strategy
capabilities, shares prompt eligibility with the prefix executor, and leaves existing
unprojected mutable state unknown. It does not reconstruct family semantics.
The facade's prepared-request forecast combines these facts with resolved generation
settings, token counts and application budgets without reopening artifacts or
submitting inference. Observed ordinary and controlled preparations share the same
forecast path. The CLI consumes the facade and owns warning/refusal policy. Unknown workspace or
overhead affects advice, never execution authorization. No allocation ledger or
new funding protocol is involved. See [generation memory](generation-memory.md).

Runtime owns ordinary capture sizing shared by preflight and forecasts: shape/slice
resolution, absolute schedules, per-step diagnostic metadata and checked cumulative
usage. `GenerationForecastBackend::capture_memory_projection` supplies complete
native source/transform cost observations without submission or reservation. MLX
reuses its admission estimator for ordinary logits; deferred sources, partition
transport and intervention evidence without full projection coverage retain an
explicit admitted-limit fallback. Backend cost observations do not belong in
the portable runtime or facade.

Runtime's serializable `CaptureMemoryPlan` caps complete geometry projections by
admitted limits, adds immutable plan storage, and places host/native costs in
physical pools. A default-false backend fact declares whether native transforms
complete before the next prediction; MLX declares this synchronous lifetime, so
runtime uses a per-step peak rather than cumulative native retention. Host records
accumulate across selected occurrences with one compact trace allowance. The facade
composes plans and trace limits from shared ordinary/controlled preparation. Horizon
alternatives recount scheduled usage without accumulating prior forecasts;
continuations retain actual charged history and use the sampler's absolute index,
independent of cache position. Restoration does not refund capture history. Unknown
costs remain explicit and forecasts consume no admission authority.

`SpeculativeForecastBackend<D>` supplies immutable selected-drafter resource
facts. MLX derives independent autoregressive draft geometry from the retained
architecture discovery and reports its own residency, placement and sampling
calibration. The runtime owns single-lane speculative phase accounting: joint
canonical state, rollback/seed copies, K+1 verification and replay, configured
optimistic overlap, distributions, host histories and physical-pool comparisons.
The facade composes target and draft facts for the same prepared request used by
ordinary and controlled speculative drivers and retains a serializable plan for
recomputation. The CLI consumes it, including shorter-output alternatives.
Embedded/feature-conditioned mechanisms remain explicitly uncovered until their
architecture-owned auxiliary state and feature geometry is supplied; native
adapters must not infer those facts from family names or checkpoints.

The existing `safemlx::system` native observation boundary reads macOS Mach host
VM statistics and releases the host port after querying. Available physical memory
is estimated from free (including speculative) plus inactive pages, using the host
page size. The MLX adapter labels this as a point-in-time estimate and exposes the
same observation for host and unified Metal capacity. Runtime fit policy continues
to compare additional request memory with that observation; native queries and
unsafe calls remain in `safemlx`.

Allocator-cache policy is also observed at the native boundary. A vendored MLX
getter reads the CPU, Metal or CUDA allocator's limit under its allocator lock;
the C API and `safemlx` expose it without temporary mutations or cache eviction.
The MLX adapter and selected facade expose both the getter and a setter returning
the previous value. `GenerationMemoryOptions::for_local_backend` composes this
snapshot with retained cache and a labeled graph/driver allowance; the CLI uses
it when no proposed cold cache limit is supplied. This explicit local diagnostic
may initialize the native allocator. The generic options constructor, portable
estimator and cold architecture-selection driver do not perform native queries.

The native allocator also retains cache-policy provenance. Its locked getter
returns the limit and origin together; direct native setters mark explicit
ownership even for unchanged values. Conditional initialization under that same
lock caps only untouched defaults at a ceiling sized by portable runtime policy:
1/32 of currently available host memory, capped at 8 GiB, with a 256 MiB fallback
when the observation is unavailable. Smaller native defaults remain smaller;
there is no minimum for constrained processes. The adapter supplies the host
observation without selecting model-family behavior. Initialization can instead
record an explicit choice to preserve the native policy. The MLX adapter applies this
native cache policy at target/model realization, including controlled and realtime
paths. It never applies it during cold selection or forecasting. Selected facade
configuration exposes automatic, preserved-native and fixed policies; portable
`AllocatorCachePolicyReport` and `AllocatorCachePolicySource` describe observations.
Explicit settings and earlier initialization win across sessions. The adaptive
ceiling is selected once, not resized during active requests or in response to
later memory pressure. Reclamation is
still native; existing retention and graph allowances remain in the forecast.
The native wrapper owns the unsafe-code boundary.

VoiceChat component construction and independent speech-provider construction
use the same initialization contract. Construction boundaries trim unused native
buffers after binding and after completed fixed prompt/silence preparation;
bounded quantization also reaps completed tile owners
before trimming its one-use conversion buffers. Final text and VoiceChat
executable owners schedule cache trimming through ordinary retirement. Nested
resource retirement finishes before that trim runs, outside native locks and
without releasing unresolved completion roots. Inference frames retain reusable
buffers under the selected policy. `trim_local_allocator_cache` exposes explicit
ordinary-host trimming without changing the limit or its provenance.

`eredu-checkpoint::prepared_cache` owns content-addressed, file-backed prepared
weights. The producer identity includes the conversion mechanism version,
native platform, execution-device class, quantization policy, source recipes and companion
precision. Source identity hashes exact bounded leases, including scalar geometry
and source encoding; filenames and timestamps do not authorize reuse. The generic
MLX bounded quantizer writes each completed encoded tile directly to this neutral
sink, with no complete packed payload retained in host vectors. The default cache
uses the user cache directory under `eredu/prepared-checkpoints-v1` (with the
operating-system temporary directory as a fallback) and retains at most 32 GiB
of prepared files. `EREDU_PREPARED_CHECKPOINT_CACHE`
selects its directory; `BoundedQuantizationPlan::with_prepared_cache` accepts an
explicit neutral location and byte budget. External cache cleanup can remove
unused entries between processes; misses reconstruct them from the original
prepared source.

Exact-zero projection inputs are an optional architecture-declared hint in
`LinearFormatSpec`; the hint permits neither pruning nor approximate arithmetic.
MLX selects a reusable affine Q8 matrix-vector mechanism for supported contracting
F32 projections. Its 32-lane accumulation follows generic QMV arithmetic and skips
packed words whose four inputs are exactly zero. Nonfinite companions still
propagate NaNs; native fast-QMV widths, batched inputs, narrower arithmetic and
unsupported parameter layouts retain ordinary projection execution.

`ParameterStorageLayout` and `WeightBinding` carry optional immutable matrix
storage preferences without changing logical geometry, source authority or byte
charges. The neutral prepared cache provides byte-preserving matrix transposition
with bounded encoded reads and two host tiles within a 64 MiB materialization
working set. Prepared-source views retain their producer's explicit cache policy.
Capacity insufficient for this optional layout retains canonical preparation.
Both direct parameter materialization and residency-managed loading initialize
final MLX storage from the prepared transpose and expose a logical transpose view;
cross-stream copies preserve physical order. Parameters,
leases and sparse projection kernels share one native weight allocation rather
than retaining a separate packed-layout cache. File representations occupy the
same bounded prepared-cache budget; canonical prepared files can coexist with
transposed files until ordinary eviction. Parameter edits and host-restored
canonical arrays remain valid through ordinary projection fallback. Encoded source
leases cover direct reads; existing completion owners retain native initialization
storage until safe publication.

Quantized text, speech and independently addressable parameter banks publish
through the same `PreparedQuantizedSource` contract. Neutral plans declare exact
output metadata before native conversion; the MLX realization consumes that plan,
uses the bounded prepared cache and validates the encoded catalog at publication.
Immediate native preparation and retained cold sources share this realization.
Architecture-owned task selection and speech component eligibility remain distinct
from the generic conversion and publication mechanism.
Native quantized-text coverage compares two cached decode steps after prefill
across resident, host-layerwise and disk-streamed policies with a `1e-5` logits tolerance.
Affine and MXFP4 family fixtures exercise exact output admission; prepared-cache
coverage verifies canonical values and encoded-file reuse.

Runtime `execution_plan_layer_residency` normalizes resident, host-layerwise and
disk-streamed policies for both complete text plans and independent speech owners.
The CLI shares argument definitions, transformation normalization and allocator
configuration across text and realtime. Speech-specific asset, conversation and
per-provider budget policy remains above those mechanisms.

Cache admission serializes publication and eviction using a directory lock.
Only cache-owned content-addressed filenames participate in cleanup and eviction;
unrelated files in a configured directory remain untouched.
Published source owners hold shared per-entry file locks; eviction requires an
exclusive lock and skips leased entries. Capacity pinned by active sources is a
typed store failure, never permission to exceed the budget. Reuse refreshes
recency metadata separately from immutable tensor-file admission. Ordered bounded
writes must cover every declared tensor before publication. File checksums reject
corruption; incomplete files are unpublished and removed on failure or the next
admission. Prepared tensors use the existing SafeTensors source, bounded selection,
exact file-admission and weak payload-lease cache mechanisms. Repeated loads skip
native quantization, while still verifying source content and prepared-file
integrity. File-backed source pages remain subject to operating-system caching;
prepared storage does not claim zero physical-memory use.

Cold prefix support is an explicit architecture declaration in
`ReplicatedTextRequirements`, combined with the backend's settled-prefix mechanism
fact and retained in the selected realization. `SelectedPreparation` applies
prediction, composite and distributed restrictions and exposes
`prefill_chunking_support()`. The shared causal decoder's cold declaration and
loaded trait use the same implementation constant; construction rejects a cold
positive declaration that disagrees with the architecture trait. Cold forecasts
consume this selection rather than a caller hint or workspace-coverage heuristic.
This keeps model semantics out of the native capability provider and allows
chunking support and estimator coverage to evolve independently.

The neutral ordinary/controlled text driver requests bounded prefill through an
optional backend prefix operation. Native adapters preserve compatible prompt
slicing, positions and completion; an intermediate prefix is settled before the
next is submitted. Requests with incompatible observation or specialized input
contracts keep a full prefill pass. Sampling and commitment remain in the shared
driver, and the prefill policy survives controlled continuation.


### Settled continuation memory observations

`eredu-runtime::memory_forecast::ContinuationForecastBackend` supplies neutral
installed-state/frontier facts and horizon-specific native storage envelopes.
Runtime owns decode-only phase composition, cache-copy overlap, physical-pool
comparisons and uncertainty. The MLX adapter validates idle session authority and
uses the existing typed state geometry/capacity growth mechanism through
`ReplicatedTextSession::estimate_installed_control_state_growth`; it never
snapshots or submits work to obtain a forecast. Architecture state geometry stays
in portable family declarations. Core's state estimator also supplies required
logical payload floors, excluding allocation rounding and optional tensors.
Runtime combines those floors with native upper envelopes for both the installed
frontier and the continuation horizon. The facade returns the normalized plan used
by phase estimation; backend snapshot reservations remain upper allowances.

Core's ordinary iterator lends read-only sampler/pending-input observations and
provides explicit completion settlement via `synchronize`. The facade composes
those facts with sampling, constraints, semantic state, admitted capture/trace
limits and live snapshot/branch reservation usage. Controlled forecasts borrow
the shared quiescent continuation boundary. Neither path changes execution limits
or refunds resource budgets. Native state/reservation estimates are upper
allowances, never credits against observed available memory. See
[generation memory](generation-memory.md#mid-session-continuation-forecasts).

### Settled external speculative memory observations

`eredu-core::SpeculativeExecutor::continuation_memory_observation` is an optional,
read-only contract for actual target/draft cache frontiers, retained-capacity and
horizon bounds, current parameter backing, available capacity and allocator policy.
Core checks the canonical lifecycle before calling it. The neutral autoregressive
executor composes two `AutoregressiveMechanisms::memory_observation` results and
its retained assistant seed. MLX reads native state metadata/capacity and residency;
it never creates checkpoints, settles completions or submits tensors for this call.
Embedded executors supply installed prediction-layer observations. Conditioned
Muse executors compose the target cache observation with the separate assistant
parameter owner, projected-context backing and pending-capture seed. Core permits
these observations only at canonical boundaries without retained proposals;
missing mechanisms retain the unsupported default. Future native backing remains
unknown rather than extrapolating current allocator capacity.

Runtime owns physical-pool request composition (`loaded_generation_request`),
`GenerationForecastOptions`, calibrated continuation phase accounting and the
object-safe controlled outlook. The facade reexports the options at the existing
API path and supplies selection profiles before lending execution resources.
Native observations refresh mutable residency/capacity on every outlook; geometry
belongs to architecture preparation; the adapter contains no family dispatch.
For iOS/iPadOS, tvOS and visionOS, the native `safemlx::system` boundary samples
installed RAM with `hw.memsize` and app allocation headroom with
`os_proc_available_memory()`. The MLX adapter labels that headroom as estimated
available capacity in the unified pool. Runtime compares additional request
memory against it using the existing neutral budget contract. Zero headroom is
preserved, and installed RAM is never used as a substitute for an app allowance.
The argument-free libSystem query and read-only sysctl remain inside the existing
`safemlx` unsafe boundary; portable crates acquire no native dependencies.
Sampler/controller/semantic continuation-storage contracts default to unknown;
the facade supplies tokenizer/parser-specific host facts. Live snapshot/branch
reservations are conservative allowances and never resident credits. Instrumented
retention without coverage leaves an explicit unknown contribution. Forecasting
has no admission, copy, randomness, observation-budget or execution authority.

`SpeculativeForecastBackend::speculative_target_memory_profile` supplies selection
geometry for the isolated lane. MLX retains that geometry even if an unrelated
installed ordinary cache has advanced; forecasting never resets either cache.

### Neutral execution resource descriptions

`eredu-core::resources` owns a descriptive contract below portable families and
mechanisms. It does not depend on runtime forecast types. A `ResourceDescription`
names its execution/module scope, current logical dimensions and requested
horizon. Entries describe parameter storage, mutable state, retained tensors or
temporary workspace. Each backing allocation has one scoped identity and a list
of logical uses: tied slots and views can share backing, while independently
loaded instances, replicas, copied snapshots and cached dtype conversions retain
distinct allocation identities. Equal checkpoint keys do not establish sharing.

Physical pools have separately scoped, authoritative identities. Host and device
access to unified backing uses one pool, while independent devices and different
hosts remain separate. Unknown placement stays explicitly unavailable; guessed
identities cannot establish either shared storage or a shared capacity pool.

`ResourceSize` supports fixed extents and evaluated context-dependent extents.
Actual stored payload (including encoding metadata) and allocation capacity are
separate, non-additive facts. Whole-horizon maxima include the starting boundary
and interior peaks, allowing nonlinear scratch geometry and rounded cache growth
without an equation language or a bytes-per-token assumption. The producer's
ordinary execution contract defines dimension names and units. Missing sizes,
placement and resource-set coverage remain explicit; a missing coverage field
defaults to unspecified, never complete. Public validation checks decoded records
before use but cannot prove producer equations or claims of completeness.
Dimensions are sizing provenance derived from typed invocation/state facts, not
family-specific configuration that applications must supply.

Producers must observe metadata without submitting, polling, synchronizing or
advancing execution, allocating execution resources, or consuming budgets.
Constructing the host description is allowed. Runtime `execution_resources`
producers read the retained ordinary selection, prepared parameter outputs and
local selected state. `describe_prepared_resources` is available on selected and
prepared text contracts and the generic retained text session. Its typed query
describes hypothetical prepared geometry, not an observation of current live
state. The existing continuation outlook is still the live-state operation.

`PreparedParameterSlot::backing` retains the canonical output of the actual
materialization binding plan after resolving binding aliases. Static roles share
the static batch, while execution units and prediction modules have separate
materialization owners. MLX retains this fact while inferring its ordinary binding
recipe metadata; describing resources never reruns inference or reads a payload.
Logical parameter aliases and checkpoint names cannot establish physical sharing.
An aggregate bank slot has no single backing identity and remains incomplete.
This ordinary materialization metadata is also suitable for neutral backends.

Core `StateComponentPolicy` owns checked element geometry at a prefix and across
an inclusive prefix interval. Runtime applies the selected native scalar width,
local state layout and segment offsets without redispatching on a family.
Conditional presence and nonlinear quotient/remainder dimensions retain interval
bounds; frame-local horizons and paged backing decomposition stay explicitly
missing. Sliding visibility does not prove physical history truncation. Resource
capacity is unknown until an implementation contract supplies allocation facts.

Selected execution-unit/static identities name missing workspace and retention
contracts. Reusable mechanisms supply per-invocation sizing facts through the
contracts below; runtime owns generic lifetime/overlap composition separately from
fit policy. These producers neither derive embedded prediction invocations nor
calibrate kernels; they contain no family-specific forecasting dependencies or
calibration constants.


### Reusable mechanism memory contracts

`eredu-nn::mechanism_memory` owns tensor-independent invocation geometry for dense
and quantized projections, grouped-query attention, causal depthwise convolution,
gated-delta/selective state-space scans, expert dispatch and logical cache appends.
The ordinary `LinearSpec`, `AttentionRequest`, convolution/grouped-linear specs and
scan inputs expose `memory_invocation`; no family imports forecast types or chooses
calibration constants. Logical values describe shapes and nominal scalar types,
not backing allocations. Packed parameter payload/sharing remains owned by the
prepared-binding description rather than being charged as workspace.

`NeuralBackend::mechanism_memory`, `AttentionCache::update_memory_contract`, and
runtime `SamplingBackend::mechanism_memory` are additive read-only hooks whose
default is explicit unknown implementation coverage. Backends own scratch,
conversion buffers, allocation capacity and retention boundaries. MLX shares
attention tile selection/limits and native quantization device selection with
ordinary execution. Its `describe_for_device` accepts an already-selected device
kind without constructing a device or stream. A cache-instance hook reads its
retained frontier, actual shapes/dtypes and existing growth/window policy, rejects
mismatched requests, and distinguishes capacity reuse, new concatenations,
key-only sharing and sliding views. Paged/compressed append decomposition remains
an explicit missing implementation fact.

Native output dtype can differ from nominal portable geometry (mixed dense
parameter/activation dtypes, GGUF host fallback, gated-delta FP32 results). MLX
refines it when selected facts establish the representation; missing device,
bias or bound parameter metadata leaves output storage uncertain. Named lazy
intermediate tensors do not imply they all coexist. Views/no-op casts and
optional retained parameter conversions keep backing unknown until their actual
owner provides identity. MLX does not publish finite per-kernel scratch or
allocator-capacity bounds, so the contract explicitly retains those gaps.

Portable runtime sampling producers use actual `GenerationSampler` or
`MirostatV2Sampler` state, or an admitted `TextGenerationConfig`: rows, vocabulary,
provided history, active top-k/top-p/min-p/penalties and resolved strategy are
mechanism facts. Mirostat's adaptive scalar belongs to host sampler state, not a
fresh device allocation. Host penalty vectors are separate from execution-pool
filtering tensors; arbitrary token filters/custom samplers remain outside this
ordinary producer and require their own contracts.

Runtime `describe_mechanism_resources` converts only backend storage records with
resolved backing identities. It binds host/execution access to authoritative
physical pools, validates compatible facts for shared owners, and retains the
original contract and storage-name bindings. Unknown owner/alias identity,
placement, capacity and retention remain explicit. This bridge never sums a live
peak or treats logical input/output shapes as allocation proof. The lifetime
composer below consumes these descriptions. Production target forecasts use
ordinary module topology and explicitly labeled planning envelopes; opaque native
scratch and allocator capacity in the raw mechanism contract remain unknown.

### Generic resource lifetime and peak composition

`eredu-runtime::resource_lifetimes` composes prepared `ResourceDescription` and
bound `MechanismResourceDescription` records without family or native dependencies.
A `ResourceLifetimePlan` is an explicit acquisition/completion event schedule.
Acquisitions before earlier work completes describe concurrency; completing work
before the next acquisition describes sequential execution. Native completion,
lazy evaluation and owner release are separate token namespaces. Evaluation
retains graph dependencies until its declared completed boundary, including after
native submission completes. Returned values, installed state and parameter
conversions survive until their owning reference is released. Tokens cannot be
retired twice or acquired again after retirement; owners may remain live at the
end of the requested horizon.

Each authoritative allocation identity is charged once while any reference is
live. Distinct copies retain distinct identities. Multiple views can contribute
different retention references to the same allocation, so releasing a parameter
owner cannot release a surviving lazy graph alias. Repeated descriptions of a
backing must agree on size and physical pool; context-dependent descriptions must
also agree on starting context and horizon. An updated query requires a new plan,
not replacing existing facts silently. Physical pools compose independently;
host/device access to one unified pool uses the same identity. Pool peaks need
not occur at the same event and are not presented as a simultaneous global sum.

`describe_mechanism_lifetimes` binds mechanism retention to schedule-supplied
completion/evaluation tokens and authoritative allocation owners. It checks the
public wrapper's storage bindings, backing semantics and numeric bounds before
composition. Named mechanism intermediates need not coexist: the bridge describes
an internal overlap envelope and does not assert simultaneous current minima.
An explicit schedule may set `live_at_acquire` when that coexistence is known.

Upper bounds sum potentially overlapping per-allocation horizon maxima. Lower
bounds take the larger of proven simultaneous current minima and individual
resource horizon minima, never the sum of independent horizon maxima. Dynamic
current bytes describe the acquisition boundary, not a minimum maintained over
a potentially shrinking lifetime. Fixed bytes with declared retention can
establish later simultaneous minima; unknown retention cannot. Payload and
capacity are independent bounds and are never added to each other. Capacity's
minimum includes its payload minimum. Equal numerical bounds do not upgrade
estimated or observed source bytes to exact metadata; source quality is preserved
across all descriptions of a shared backing.

A plan must explicitly declare resource-set completeness. Missing coverage,
placement, or retention leaves all observed pool upper bounds unknown, with
unplaced allocation identities and reasons retained. Unknown capacity in a known
pool does not erase known payload or bounds in other pools. Unknown lifetime
references remain conservatively live but never manufacture a minimum overlap.
Overflow, conflicting shared facts, malformed descriptions and invalid release
order are errors. Small synthetic schedules validate sequential/concurrent work,
shared backing, independent pools, explicit evaluation batches, dynamic growth
and shrinkage, and unknown facts. This is a read-only composer, not a native
scheduler or budget verdict. The target-workspace evaluator below uses this
composer while preserving the existing report format.


### Generic target workspace forecasts

`ReplicatedTextRequirements` retains optional `TextExecutionTopology` as an
ordinary construction contract. Shared decoder attention, gated MLP and static
output modules reuse their actual `LinearSpec`/`StaticModuleSpec` producers.
Gated convolution uses its existing `GatedShortConvolutionSpec`; packed routing
uses the same `TopKGroupSelectorSpec` and `GroupedGatedProductSpec` as execution.
The topology contains invocation geometry and parameter identities, never byte
calibration constants. Selected projection formats come from exact physical
realizations, preserving tied owners and load-time quantization.

Runtime composes target workspace by reusable mechanism kind and explicit lazy
retention boundaries. Llama, dense Qwen, Nanbeige and dense LFM2 use
these contracts. Shared Gemma2 and packed Qwen/LFM routed experts inherit the
same mechanism composition. New shared-decoder families require no independent
workspace formula. The aggregate `WorkspaceGeometry` wire contract is a
compatibility input for serialized and manually constructed requests. It lowers its declared aggregate
envelope into the same resource-lifetime plan and peak evaluator, reusing the
ordinary mechanism calibrations. It does not reconstruct missing module identities
or invocation order. Ordinary selected production requests use topology directly.

Finite planning estimates remain estimates. Native scratch, aliasing and
allocator alignment are not proved by logical shapes: explicit calibrated
mechanism envelopes account for their planning allowance without rewriting the
raw mechanism storage contract as exact. Unsupported equation providers, missing
rank-local topology and unsupported residency lifetimes retain named gaps.
An embedded selection keeps ordinary request-only forecasts explicitly unbounded;
startup forecasts retain target topology only alongside the prediction and
transaction plan. Advanced embedded continuations additionally use settled
observations of installed prediction state and retained target features.
Cold and loaded calls remain read-only and cannot materialize modules, populate
parameter conversion caches or consume observation budgets.

### Embedded prediction resource descriptions

`SelectedPreparation::embedded_prediction_topology` projects additional invocations
from the ordinary architecture descriptor and the admitted speculative strategy.
It retains canonical parameter groups, explicit per-node prediction/context/fused
scopes, target capture geometry and ordinary state policies. Data-flow edges do
not establish native allocation lifetimes. Sequential context preparation consumes
all but the last target row; fused context preparation consumes the full prefix.
The materialized executor and the descriptive contract use the same
`PredictionExecutionMode` operation for that rule.

Actual prediction preparation retains local parameter declarations, physical
module ordinals, auxiliary residency owners and local state policies alongside
`PreparedPredictionPlacement`. These are the modules and state supplied to native
materialization, not a second family forecasting implementation. Shared modules
remain distinct from sequential units and from target parameters borrowed by an
invocation. Cold logical sharing is never upgraded to physical aliasing.

`PreparedModelDiscovery::describe_embedded_prediction_resources` composes these
facts with the exact prepared parameter slots, explicitly bound target-feature
allocations and current residency observations. Prediction slots use their own
materialization batches; borrowed target weights keep the target batch identity.
Two retained feature views of one target allocation add uses, not extra bytes.
Native cached conversions are separate allocations identified by their actual
resident owners. Reusing a target readout conversion from several prediction
heads does not charge another copy. Reading these reports does not materialize
weights, populate conversion caches, evaluate tensors or advance execution.

Logical resource descriptions do not establish native capacity. Missing dtype,
local geometry, target-feature backing, mechanism scratch and evaluation retention
remain explicit. Startup forecasts compose covered ordinary target and
prediction invocations with the speculative transaction envelopes; settled
embedded continuations substitute observed live state and feature retention.

Qwen hybrid attention, dense feed-forward, shared prediction fusion and static
readout construction expose their actual `LinearSpec`/`StaticModuleSpec` inputs
through ordinary helpers. Both allocation and cold topology consume these same
specifications. The runtime describes fusion projections, sigmoid attention gates
and repeated output invocations generically, without consulting a family name.
Prediction layers retain their actual parameter identities: each depth invokes
the shared fusion and target readout, while its decoder parameters remain distinct.
Selected physical task encodings replace source defaults before projection; any
mixed-precision prediction promotion allowance derives from selected auxiliary
tasks. These facts contain no calibration constants.

A dense Qwen target consisting entirely of full-attention layers and its embedded
predictor therefore use covered mechanisms. A gated-delta target or shared/routed
expert reduction still names the missing invocation mechanism. Other prediction
families retain concrete missing mechanism descriptions (latent/pooling or learned
relative attention, hyper-connections, patterned operators) rather than acquiring
a finite bound from the presence of an embedded head alone. Rank-local prediction
invocation geometry remains unknown until ordinary partition contracts describe
it.

### Settled embedded continuation observations

The speculative executor observes the installed target cache, prediction-layer
frontiers and retained target features through neutral contracts in `eredu-core`.
The architecture adapter forwards the ordinary prediction state's membership;
it contains no forecast calibration or family-specific memory equations. Native
mechanisms report state-capacity allowances and current parameter ownership.
`ReplicatedTextSession` exposes a read-only parameter observation because the
ordinary session is lent to the embedded executor during a controlled run.
The MLX adapter projects its existing residency report through this operation.

Resident parameter conversion allocation and binding records live in
`eredu-core::residency`, with their existing runtime paths reexported. This lets the
neutral observation carry fresh native ownership after prefill without introducing
a dependency from core to runtime. The runtime retains validation and forecast
credit policy. Embedded conversions belong to the shared target parameter owner;
prediction invocations never acquire a second parameter-residency charge.

The portable runtime composes the prediction/transaction plan with observed
current frontiers and horizon-specific capacity. Current feature and seed retention
survive a zero-token outlook. Loading and prefill do not reappear in continuation
phases. Missing native bounds or ordinary invocation mechanisms remain explicit.
Only canonical settled boundaries are eligible; observation cannot poll, submit,
settle, snapshot, advance execution or consume capture/transport/copy budgets.


### Unified workspace evaluation and legacy compatibility

All workspace estimates flow through `describe_text_workspace` and
`compose_resource_peaks`. Ordinary topology is authoritative when both topology
and aggregate fields are present. Attention score scratch, explicit input-score
tile retention, convolution intermediates, parameter conversion and cache
replacement share runtime calibration helpers. No backend or architecture family
selects a compatibility equation.

`WorkspaceGeometry` records lack exact projection encodings, parameter identities
and invocation lifetimes. A compatibility lowerer preserves their declared
aggregate linear/logit and overlap contract as one conservative allocation. This
wire contract supplies no second ordinary topology or family forecast. Frozen
serialization fixtures check finite bounds, unknowns and checked arithmetic.

### Retention-aware forecast ownership

The neutral `ResidentParameterConversionBinding` carries optional budget-group
attribution. Backends observe that group alongside each current retaining binding
and actual allocation identity; missing attribution grants no scoped credit. Runtime forecast composition matches complete selected payloads,
not aggregate conversion bytes or names from another budget. MLX observations are
refreshed for ordinary execution and each speculative participant, including
settled continuation; startup geometry can be reused but startup conversion
observations cannot. Shared native conversion identities deduplicate physical
payload across external participants without merging their admission budgets.

`eredu-runtime` owns the serialized conversion-retention forecast subledger and
potential-admission bounds. The facade composes cold selection facts and a
request-local cold scope without native allocation. Missing cross-rank authority
facts remain unavailable. Native budget scopes replace this local scope after
loading. Possible retained payload is already covered by the pending conversion
envelope, and is never added again or used to clamp temporary workspace. Group
reservations without allocation attribution preserve an unknown phase upper
bound until publication is observed. Records without optional scope facts use
aggregate accounting. Forecasting never reserves, evaluates,
settles, trims, or advances execution.

### Native attention evaluation frontiers

`FullKeyAttentionTiles::evaluates_input_dependencies` is an optional neutral
mechanism fact, defaulting to false when omitted from a record. MLX supplies it for
synchronous full-key query-tile batches. Runtime combines that fact with selected
ordinary topology and the invocation's query/key geometry; a short lazy invocation,
Fused arithmetic, or an uncovered blockwise-key path establishes no such frontier.
Cold inspection contains no family branch or native execution.

The generic lifetime plan separates gated feed-forward intermediate projections
and products and unfolded convolution scratch from retained module outputs.
A covered attention evaluation releases preceding transient dependencies and its
own score/layout workspace. It preserves completed tile outputs, final feed-forward
outputs, residual/normalization/mixer outputs, and gated/padded convolution backing
that cache/history views may own. Present-layer feed-forward acquisition follows
attention completion; its workspace cannot be released by that earlier boundary.

The existing additional overlap calibration, uncached parameter promotions,
promoted state/replacement, cache replacement and fallback attention scratch remain
conservative request-wide allowances and overlap every interior peak. Unknown
retention remains unknown. Aggregate-only forecasts use their declared
envelope because they do not describe invocation order. Loaded, cold and controlled
continuation forecasts consume the same neutral facts and shared lifetime composer.

### Opt-in projection baseline diagnostics

The backend's nondefault `projection-profiling` feature provides bounded,
thread-affine collection of native mixed-width dense and tied-embedding
projection inputs, weights and outputs. Capture hooks do not evaluate arrays or change
mechanism selection. At the explicit finish boundary, diagnostic tooling settles
samples and groups them by actual dtype, shape, strides and selected backend path.
One real representative and the invocation count are retained for each class.
Capture owns additional arrays, so its peak and latency are explicitly excluded
from ordinary inference measurements. The feature is forwarded by the facade
solely for native projection validation examples; portable defaults and neutral
contracts gain no native dependencies.

The example belongs to application/validation tooling. It performs separate
ordinary inference and isolated projection replay, using native allocator counters
and completion timing. Short verification-sized replays slice captured prefill
activations; they do not claim to execute a speculative inference driver.
`validation/projection_baseline.py` validates the pinned checkpoint and runs kernel
selection tracing in separate processes from timing measurements.

The native MLX patch `mlx-metal-kernel-selection-trace.patch` logs selected
pipeline names, including specialization keys, when
`MLX_METAL_LOG_KERNEL_SELECTION=1` is present before native initialization.
It records native pipeline requests during graph evaluation, including cache hits;
it is not a duration or an executed-GPU-instruction counter. Tracing is silent by
default and changes no dispatch choice, kernel equation, precision policy or
completion ownership. All native work continues through `safemlx`; no additional
unsafe-code exception or public inference error type is required.

Affine projection diagnostics use the same nondefault feature and native-only
ownership. Their hooks retain immutable operands without evaluating them or
changing dispatch. Explicit finish settles roots, groups by actual stream/layout
and quantization, and retains distinct parameters for fixed/rotating-weight
replay. Admission bounds invocation count and retained operand payload; overflow
is an error. Isolated native and zero-padded fast-Q8 replays account for padding
storage and report numerical differences. These evaluation intervals and payload
estimates are not per-shader duration, DRAM traffic or ordinary full-model metrics.

Application tooling selects capture advancement and replay/output policy. The
backend supplies generic native collection and replay; it owns no VoiceChat
configuration or family equations. The safe native wrapper owns process-wide
Metal export, nesting rejection and cleanup through MLX's existing capture API.
Diagnostics use the existing native wrapper safety boundary and neutral public
failure contract; portable crates contain no native diagnostic dependency. See
[projection diagnostics](projection-baseline.md#affine-projection-diagnostics).
The safe native completion timer reports command-buffer interval sum, union,
span, valid interval count and missing timestamps. Union removes overlap; span
includes gaps. Each still includes buffer waits and is not individual shader
compute time. Metal reserves timestamp slots before handler installation and
records completed buffers without callback allocation. Summary resolution first
establishes ordinary completion or reports its failure; it does not release
submission authority earlier. CPU/CUDA marker intervals report identical sum,
union and span. Only explicit timed evaluations retain interval records; diagnostic
tooling bounds their duration. Ranking uses wall time including construction and
completion, with ordinary conversation controls required for optimization claims.

The same diagnostic feature captures generic attention matrix products and
immutable cache concatenations with separate invocation/payload bounds. It retains
every real invocation, groups actual native layouts, and checks replay output bits
against captured outputs. Mutable capacity-backed appends are excluded rather
than replaying writes into retained storage. Hooks retain roots without evaluation
or mechanism changes; explicit finish settles them. Captured and independent
replay workloads can affect donation, allocation and scheduling, so their timings
are not additive full-model attribution. Generic native roles carry no family
geometry or equations; application tooling selects the capture advancement.

### Speech projection precision

`SpeechPrecision` is an architecture-owned storage and arithmetic policy. The
facade retains it in `VoiceChatStartOptions::speech_precision`; the default is F32.
Cold construction converts eligible perception matrices and speech attention/MLP
weights through exact retained recipes before resource admission. Norms, biases,
MoG component banks, RVQ tables and codec parameters retain F32. Narrow compute
requires matching F16 or BF16 weight storage. Perception convolution weights remain
F32 with F32 compute and become narrow only with the matching narrow compute policy.
Converted bindings carry their actual byte lengths and materialization recipe peaks.

Shared linear construction declares optional floating execution precision through
`LinearFormatSpec`. MLX casts projection inputs, performs the projection with the
selected stored weights, and restores the caller's floating dtype. Architecture
convolution calls likewise restore F32 before normalization, residual additions or
history publication. Perception attention and convolution histories, causal TTS KV
state, sampling reductions and recurrent transcription remain F32. TTS and perception
state identities retain the selected precision. The same prepared providers serve
explicit advancement and ordinary shared scheduling.

Native row-parallel projection casts its local result back to the original input
precision before reduction and bias addition. Cast/projection failures enter the
existing local agreement phase, and ordinary and observed paths use the same native
mechanism. Native operators own casts and kernels; they contain no VoiceChat policy,
parameter-name matching or source-format selection. The portable policy selects
eligible parameter roles and enforces boundary precision before neural execution.

### Native compilation safety

Native compilation forwards the complete scalar-specialization slice to MLX.
Scalar constants therefore distinguish cached graphs even when their array inputs
have identical geometry. The MLX 0.32.3 bridge records weak cache identities on
invocation, including each thread used by a moved wrapper. Cleanup erases through
those retained identities without accessing native TLS; expired caches are a
no-op. Caches retain their upstream thread-local lifetime. Native destruction
remains within the existing
`safemlx`/`safemlx-sys` unsafe boundary; portable crates gain no native dependency.
The native regressions exercise scalar specialization, lazy outputs after cache
erasure, cross-thread wrapper destruction, and wrapper teardown after the
populated C++ thread-local cache.

### Mixed-storage FP32 GEMM

`safemlx::fast::try_mixed_storage_gemm` is a native inference operation. Its distinct primitive delegates dispatch to the pinned
MLX FP32 SIMD Matmul implementation. The vendored patch adds a weight storage
type to the existing regular and split-K GEMM loaders; activations, shared-memory
tiles/padding, MMA, epilogues, output dtype, partition selection and split-K
accumulation remain FP32 with their existing geometry and order. Separate bias
addition stays with the caller.

The native operation accepts settled contiguous rank-two F16/BF16 `[N,K]`
weights and rank-two-or-higher F32 activations on Metal devices using native SIMD
GEMM, within the bounded validated geometry. Activations may be lazy or strided;
flatten/unflatten and layout handling match native Matmul. Device model names do
not control admission. The operation shares ordinary Matmul's native NAX-selection
predicate and rejects NAX/TF32 dispatch because its separate loader is not
implemented. NAX-capable hardware remains eligible when native FP32 dispatch
selects SIMD. It returns `None` without evaluation for unsupported cases, including
CPU, CUDA, system/unpatched MLX, non-JIT builds, lazy/noncontiguous weights and
single-row GEMV. This safemlx/C-ABI/native mechanism uses the native wrapper's
unsafe-code boundary and contains no family policy. Its separate primitive
rejects unsupported transformation APIs rather than inheriting homogeneous-Matmul
gradient or compiler assumptions.

Multi-row GEMM uses the backend projection selector shared by ordinary dense
linear layers and tied-embedding readout.
Single-row mixed GEMV and each caller's existing unsupported-case fallback remain
in place. Bias handling stays outside the selector. Ordinary, controlled and
speculative execution consume this shared mechanism without driver-specific
branches. Loaded forecasts consume proven invocation eligibility and workspace
facts; cold and unsupported invocations retain conservative allowances.

The nondefault projection profiler provides a thread-affine `CastGemmReference`
guard for same-build A/B validation; it disables only the multi-row optimization
until dropped. This control is absent from production builds and public inference
options. The validation examples compare exact logits and cached tokens through
ordinary, controlled and speculative paths, and separately measure allocations,
kernel selections and repeated release timings. See
[integration validation](mixed-storage-gemm.md).

### Projection storage facts and forecast refinement

The backend attaches `ProjectionStorageFacts` to residency reports for current
permanent bindings. It establishes settled weight dtype/layout and selected
native device arithmetic without evaluating tensors. Overlays, bounded residency,
missing bindings and conflicting aliases retain conservative coverage. The
portable records carry canonical identities, exact geometry, activation/output
representations and row intervals with native partial-result payloads; they carry
no native handles or hardware dispatch rules.

Loaded ordinary and continuation geometry and independent drafter geometry copy
these facts into the shared execution topology. Cold preparation initializes an
empty map; serialized reports without that field also default to no coverage. Runtime matches the
logical F32 invocation, shape, ordinary dense encoding and canonical task
attribution. Every shared use must be covered before its full conversion payload
is credited once. Retained-conversion credits remain separate and cannot be
subtracted twice. Native partials and possible activation-layout copies are
charged per invocation. Split-K partial storage can peak at an interior row count;
the envelope includes those smaller-row maxima for short verification/final
chunks. Output and bias storage remain with the existing topology producers.

State storage width does not establish projection activation dtype. Architecture
topology declares which ungrouped learned RMS normalizations feed each projection
through dtype-preserving equations. LFM2 supplies this declaration for its dense
mixer/FFN projections and final readout. The backend independently reports actual
F32 rank-one gain bindings: MLX ungrouped learned RMS promotes its output through
the gain. Runtime requires every declared source to have that guarantee. Missing
declarations, narrow gains, grouped normalization and conflicting aliases receive
no credit. This is a neutral dtype-flow contract, not a backend family branch.

The narrow safemlx workspace query shares the exact native SIMD split-K partition
helper with GEMM dispatch. It returns logical partial-buffer payload only and
never evaluates or creates arrays. Native output payload is separately known;
activation copying remains a zero-to-full-input bound until strides are known.
Allocator capacity/rounding and GPU-private registers/threadgroup storage are not
claimed exact. Existing calibrated graph, state, attention and allocator
allowances remain in the forecast. Cold mechanism descriptions retain promotion
and unknown native-workspace costs. No portable crate gains a native dependency
or unsafe-code exception.

## Supplied metadata and local payload admission

Cold recommendation clients supply `ArtifactMetadata` and
`MetadataInspectionOptions` through `eredu::api::inspect_model_metadata`.
The options select an explicit `BackendId` and portable loading policy. The
facade resolves the compiled adapter and obtains its cold capability provider;
the shared architecture driver performs validation and execution selection.
Unknown or disabled backends return typed errors, without fallback. Backend
adapters own capability facts; applications import the complete inspection and
forecasting surface through `eredu` without direct backend dependencies or
capability-provider implementations. Transport, caching and immutable-revision
provenance belong to the application.
`eredu-gguf` and `eredu-checkpoint` own supplied-header parsing and structural
validation, including full object lengths and complete shard membership.
`eredu-core` retains provenance and enforces the distinction between metadata
compatibility and local payload admission. Its model-preparation constructors
reject metadata-only inspections with `ArtifactError::MetadataOnly`.

`eredu-architectures` consumes explicit configuration, tensor catalogs, companion
roles and processor sidecar bytes. Its resolver declares applicable sidecar names;
local inspection reads those files, while supplied-metadata inspection uses only
the provided bytes. Both feed the same architecture finalization and cold
selection. SafeTensors recipe sizing uses catalog storage facts without requiring
local shard handles. `eredu-runtime` owns the shared memory calculations, and the
facade exposes their composition without native types. GGUF supplied-header
catalogs reject payload access even when a materializer is constructed directly.

See [Forecasting from supplied checkpoint metadata](metadata-forecasting.md) for
bundle requirements, recommendation semantics and validation commands.


### Completed causal prefill and compact retained tensors

`eredu-nn::Tensor::compact` is a required neutral storage contract: after
completion its backing storage is bounded by logical payload plus bounded backend
allocation overhead, independently of the size of a source view. It preserves
dtype and values and may remain lazy until the execution owner completes it.
The general causal depthwise convolution uses it for its retained history.
MLX realizes it with its bounded contiguous-materialization primitive; an MLX
`copy` alone shares storage and does not satisfy this contract. Shape-only and
owned numerical reference backends implement their ordinary owned representation.

The neutral replicated-session driver completes ordinary text-prefill output and
retained state before committing publication. This invocation-level requirement
uses the existing backend completion mechanism and rollback protocol; it does
not force asynchronous decode to become synchronous or impose an allocator policy.
Native submission owners continue to retain leases until safe completion, terminal
failure or teardown. Cancellation is checked by the shared core generation driver
between those completed prefix passes.

Architecture declarations enable dense and routed LFM2/LFM2.5 incremental prefill
for the existing resident/host/disk text execution strategies. Both cold selection
and typed executables consume the same family declaration. Shared heterogeneous
and dense decoder readout helpers select the final hidden position before ordinary
vocabulary projection. Observed, speculative and partitioned full-output entry
points retain their established contracts. No backend family branch or facade
scheduling policy is introduced.

Qwen hybrid target-only text execution also declares prefix continuity, including
text input through its composite adapter. The neutral composite contract distinguishes
plain-text prefix support from retained prepared-media cursors. The backend consumes
those declarations, refuses plain splitting for structured input/capture/distributed
execution, and admits each prefix's tensors without installing a suffix under the
whole request's prepared-input cache identity. Internal fragment ownership carries
that admitted whole-input identity separately; nonfinal completion clears committed
prompt identity, and successful final completion publishes it atomically. Snapshot
copies retain fragment ownership and failed completion follows ordinary rollback.
Ordinary composite readout delegates
final-position selection to the architecture; observed and prediction readout remains
full-output. Composite state-offset reporting preserves reset-state workspace topology
and invalidates it for unprojected continuation states.

The MLX gated-delta prefill mechanism evaluates both sequence and final recurrent
state before returning. This bounds retention of upstream lazy graphs across native
scan invocations; single-token decode retains its asynchronous contract. It adds no
family dispatch, cancellation policy, allocator cap, or hardware-fit claim. Runtime
workspace topology includes recurrent projections, convolution geometry and FP32 scan
output, while leaving uncalibrated scan scratch and state-version retention explicitly
unbounded. See [Qwen3.5 prefill validation](qwen35-prefill-validation.md).

Header-only SafeTensors inspection retains a `MetadataOnly` physical declaration with
its admitted alias resolution. It supports cold selection and forecasting without
inventing shard identities. Opening that declaration and preparing sources both reject
with `ArtifactError::MetadataOnly`.

### Selected history and addressable parameter banks

`eredu-nn::IndexedAttentionInput` declares grouped K/V heads, absolute selected
positions, validity, optional selected-score masks, bounded local sources and sinks.
Architecture equations own selection and causal eligibility. `AttentionCache::indexed_attention`
routes the request through resident or paged storage. MLX deduplicates positions and
completes one query's compact gathers before advancing. Paged reads acquire only
intersecting pages. Sparse attention retains full K/V history; its read counters are
separate from full-history scans. Failed native page copies retain source leases until
terminal completion or teardown, using the ordinary cache lifecycle.

`eredu-runtime::ParameterBank` owns operator-independent acquisition, completion
retention and telemetry. `AddressableGroupedBank` builds grouped operators from those
acquisitions; `RowLookupBank` decodes independently addressable rows. `ParameterProvider`
carries both operations through layered, partitioned, observation and prediction drivers.
`ParameterProviders` composes independently prepared row and grouped mechanisms.
Source names, formats, geometry and parallel ownership remain architecture declarations.

`PreparedRowSource` retains one restricted recipe over checkpoint shards. It deduplicates
requested rows, plans bounded selections and restores request order without a
vocabulary-sized catalog. `OffloadUnitRange` describes uniform source members compactly;
OffloadPlan schema 2 retains these ranges. The shared residency manager creates demanded
member definitions and retires them when no allocation, reservation or transfer remains.
Rows use the same pinning, host/device budgets and completion recovery as grouped weights.

`RowLookupDescriptor` and `RowLookupDescriptors` contain source-free geometry, encoding,
compact residency ranges, scalar recipe headers and finite request/workspace bounds.
`SelectedRowLookupPlans` selects generic storage, decoder and workspace facts once.
Binding compares exact descriptors and yields `SelectedRowLookups` with retained source
identity. Equal names and shapes do not authorize a replacement source. Native binding
rechecks mechanism facts but cannot reselect mechanisms or replace sources.

`PreparedRowLookup` pairs the source range with `RowLookupSpec`, an optional
`PreparedRowScale`, and finite limits. `PreparedRowLookups` checks table identities,
unit ownership, bank namespaces and range prefixes. Cold admission validates source
encoding, scalar identity, one-row access and maximum acquisition/planning/output bounds.
The largest permitted acquisition must fit the device cache. Missing mechanisms,
incompatible bindings and inadequate budgets produce typed errors before construction.

`MlxRowLookups::bind` consumes the selected parameter pool and retained row declarations.
It validates decoder availability, scratch, scalar budget and exact source ownership
before materializing companions. Binding reads no table rows. Native `MlxRowBank` supports
dense, shared-scalar E4M3 and GGML block rows; shared scalar FP8 uses its declared parameter
identity rather than expert block-scale equations. Lookup retains exact host integer IDs,
chunks native acquisitions and keeps source/residency leases through completion.

Selected grouped banks and compact row ranges share an `OffloadPlan` and residency
manager. Resident grouped execution can use a row-only pool. Pool reports count each
physical pool once, including row-only pools; grouped member statistics exclude table
rows. Row scalar allocations are reported separately. Auxiliary prediction providers
move from the complete selected provider collection into disjoint owners through
`RoutedBankProviders::split_off`, without cloning storage or creating another budget.

`TensorReadRequest` selects exact checkpoint regions. Bulk reads coalesce adjacent
physical ranges across separate output allocations, enforce admitted file identities
and use no payload staging. SafeTensors and byte-preserving GGUF outputs share this
mechanism; transformed GGUF outputs use bounded conversion. Inclusion views retain
only their explicit allow set and validate names through metadata lookup. Diagnostic
denied-key enumeration is computed on demand. Restricted views and leases retain the
original source authority.

### Attention, append streams and fixed state

`AttentionStreamState` requires ordinary attention, fixed components and
`RuntimeAppendStreams` together. Shared typed construction visitors enforce these
contracts. The combined layer layout declares K/V, named append streams and fixed
tensors; native storage consumes the selected geometry without family dispatch.

Named append streams declare record geometry, scalar policy and processed-token bounds.
`AppendStreamBinding` adds lanes, record/read/page limits and per-lane allowances.
`ReplicatedTextRequirements` and `SelectedStateRealization` retain exact bindings;
selection checks coverage, representation, resident capacity, scratch and page geometry.
State partitioning rebases owners and can narrow record width under architecture-authored
TP geometry without changing token cadence or increasing retained limits.

Resident append storage retains immutable chunks. Native paged append storage reuses
ordinary cache blocks, device/host/disk residency, completion leases and rollback.
`CacheStreamId` separates layer, representation, slot and sequence lane identities.
Reads compact only requested ranges and retain source leases through completion.
Record counts do not inflate attention-token counts; read/byte/scratch counters survive
rollback. Hybrid state preserves streams, fixed tensors and positions through checkpoint,
reset, fork, snapshot isolation and persistence.

Prompt-cache schema 10 records every declared slot/lane frontier, including empty lanes.
Validation checks exact dtype/shape, record bounds and contiguous page coverage separately
from attention-token coverage. Restore retains shard handles and bounded metadata;
`RetainedCacheShard` verifies the digest of bytes copied into backend transfer buffers.
Replacing or unlinking a path cannot redirect an admitted reader. Declared lanes are
capacity: saving a smaller active batch requires all omitted lanes to be empty.

`StateTensorRole::IntegerHistory` uses checked I32 storage. Token identities never come
from embeddings or floating proxies. `StateTensorRole::Auxiliary` identifies other
architecture-declared fixed buffers. `RuntimeStateComponents` validates replacement roles
and publishes the replacement set atomically; unknown or duplicate roles leave state
untouched. Prompt manifests match fixed tensors by owner and role rather than serialization
order. State-only admission uses `StateRealizationRequirements`, `select_state_realization`
and the same neutral storage/lifecycle capability predicates as ordinary text selection.

### Flash-Next architecture and artifact ownership

`ModelKind::Qwen4Exp` is a distinct portable family. SafeTensors admission accepts
`qwen4_exp` and `qwen4_exp_text`; GGUF admission accepts `qwen4exp`. Configuration,
checkpoint schemas, parameter recipes, equations, state geometry, processor policy and
parallel plans live in `eredu-architectures::qwen4_exp`. MLX binds generic operators,
state, sources and selected execution contracts. Current supported paths and verification
coverage are described in the [family validation report](qwen4-exp-validation.md).

`TargetSpec` declares target geometry. `TargetModel` is a `ComponentAdapter` over
`TargetKernel`, an `Arc<ComponentDefinition>` and `KernelParameterModules`. Runtime
implements architecture, layered, parallel, routed, text, typed-partition and
pipeline contracts from that declaration and the kernel's explicit capabilities. Execution alternates decoder blocks
with independently owned lexical units.
Each decoder unit contains a recurrent or indexed mixer and a shared/routed feed-forward
sublayer. Checkpoint layer numbers and execution ordinals remain distinct, including
expert/table owners. The target owns its pinned `StaticModules` directly and uses
shared resident, host-layerwise and disk-streamed execution machinery. Its retained
component definition supplies graph and dependency validation for those modules.

`TargetSpec::parameter_description` declares canonical parameter ownership and
physical shapes without a device or native module. It uses the exact selected
recurrent, indexed, lexical and feed-forward specifications, including independent
expert identities and tied vocabulary consumers. Runtime expands declared linear
encodings and their companions. Row-table payloads remain in their independently
admitted providers. Target inspection and global partition/transform-source
planning consume this declaration without constructing temporary global models.
Parameter-member declarations retain neutral encoding relationships through
`ParameterMemberSpec::with_parameter_metadata`.

Qwen4 target, prediction, tensor-partition and vision preparation expose runtime
`PreparedParameterRecipes` for ordinary parameters and bounded expert selections.
Family code retains naming, format admission, expert recipes and exceptional
partition transforms; checkpoint code retains exact source snapshots and runtime
owns recipe-resource preparation. Recipe failures preserve their typed causes
through the family preparation error.

`TargetSpec::definition` validates the target's graph, canonical storage, semantic
unit paths, one-to-one context slots and decoder transport before materialization.
The adapter retains this `ComponentDefinition` in an `Arc`. Parameter inspection,
serial and parallel address checks, group discovery, context projections and
selected ownership use the retained authority. Expert placement updates validate a new declaration before
publishing the changed specs; partition construction retains canonical global
parameter placement with the actual tensor-local context geometry. Each lexical
invocation keeps its own slot independently of the following decoder layer.

`TargetKernel` owns request preparation, original-ID and visibility provenance,
recurrent/QSA/lexical equations, provider access, observation seams, residual
readout and target capture. Serial, observed, parallel and routed hooks use the
same equations with the appropriate instrumentation and collective context.
`new` and `new_tensor_parallel` construct kernel resources; geometry selection
consumes the kernel through `with_expert_realization` and `with_partition_state`.
`into_model` assembles the standalone adapter, and selected construction retains
its exact partition ownership on that same adapter. Consuming geometry updates
keep a borrowed execution kernel aligned with its retained definition.

`ConditionalModel` uses the same `ComponentAdapter` with a `ConditionalKernel`
and `KernelParameterModules`. The kernel owns a raw `TargetKernel`, vision resources
and request assembly under the enclosing runtime lifecycle. Typed ordinary,
parallel, routed and partition hooks invoke those same target and vision equations.
The adapter supplies discovery, validated addresses, ownership and state projections,
parameter traversal and retained roots. Expert selection consumes the kernel before
adapter assembly. External prediction borrows target vocabulary
resources through the ordinary component adapter or the enclosing composite;
base-width lookup and prediction projection preserve their own equations.
`ComponentPinnedResources` exposes the conditional composition's target and vision
aggregate directly. Parameter traversal excludes kernel geometry and visits target
storage before vision storage. Selected executable construction validates partition
ownership against the retained definition and retains exactly its pinned resources;
empty vision aggregates retain continuation geometry and expose no parameter role.

`conditional::parameter_description` composes retained target storage with vision
parameter geometry without a native context. It projects from
`conditional::definition`, which retains target invocation paths and mutable-state
slots alongside stateless vision units, graph dependencies and transport contracts.
Complete and selected conditional construction validate and retain this definition
in an `Arc` before native allocation. Runtime metadata and address checks project
from that authority; selected definitions retain canonical parameter placement
with tensor-local target state geometry. Consuming expert selection validates the
updated definition before publishing the composition.
Cold joint requirements project their graph, unit layout, transport and state
from this same declaration before execution selection.
Tensor/expert placement uses global,
tensor-local and expert-local declarations; `TargetSpec::with_expert_realization`
consumes geometry and validates the exact admitted expert identities before
returning the local specification. Native target updates use the same selection
operation. Conditional target and transform-source planning derives placement
from these declarations. Selected conditional construction validates ingress and
placement, then constructs the tensor-local target and vision modules directly.

`DecoderBoundary<B>` owns embedding expansion and position-independent residual collapse.
Shared `StaticModules` owns vocabulary parameters, collectives and readout
instrumentation. `BoundaryModules` retains optional embedding, residual readout
and untied projection resources for selected owners. Selection uses canonical
static storage roles from parameter bindings, including tied vocabulary storage
and explicitly declared prediction replicas. Selection rejects a requested role
whose module is no longer retained. Parameter-free residual expansion
uses the declared geometry without retaining an unowned readout.
`ComponentPinnedResources` exposes the target's actual pinned
aggregate and canonical role traversal to binding and completion owners. Target,
media composition and external prediction borrow those same modules. Runtime's
`ComponentDefinition::validate_group_dependencies` checks invocation dependencies
for both shared adapters and declaration-backed family hooks. `NormalizedBoundary` supplies ordinary normalization. Flash-Next's
`ResidualBoundary` expands text/media embeddings and applies the released gated collapse.
Residual observation and prediction capture retain all streams before collapse; normalized
and projection-input observations consume the actual collapsed value. Final-token
narrowing preserves every non-sequence axis.

`IndexedSublayerSpec` declares QSA projections, normalization, rotary arithmetic and cache
geometry. `CachedQsaSelector` validates lane frontiers, stream geometry and rotary
provenance; complete summaries and original positions append once. Its incomplete visible
block survives arbitrary chunk boundaries. Selection scores bounded summary tiles and
passes selected original K/V positions to indexed attention. The indexer and residual
mixer are replicated independently of ordinary K/V head placement.

`QsaPartialStateSpec` packs the incomplete block into exact I32 controls/positions,
raw index keys and first-token media cosine/sine components. Packing and restoration
validate the complete tuple before publication. Summary construction and visible-token
selection are family equations; full K/V storage remains separate from summaries.
`QsaExecutionLimits` covers coordinator projections, selected-position output and partial
packing separately from backend operator workspace and retained state.

`LexicalInjectionSpec` consumes exact hash/table declarations and family parameter
identities. `LexicalSublayer` adds the n-gram injection to complete residual streams and
advances declared history/convolution state. Original IDs, visibility and absolute
prefix position are explicit inputs. Missing IDs or position drift reject before lookup.
EOS-separated n-gram reset policy belongs to the architecture and is separate from facade
generation termination. Shared transaction ownership restores state on failure without
refunding lookup, observation or transport activity.

The BF16/FP8 SafeTensors schema declares exact aliases, exclusions, matrix geometry,
expert layouts, scale companions, integer controls, shared table scale and vision/MTP
parameters. `SafetensorsTableSourcePlan` and `SafetensorsTargetPlan` consume metadata-only
`SafetensorsCatalog`. Binding checks the retained headers, physical source set and layout
before reading tiny I64 controls or a shared scalar. Table shards form one logical
row-addressable source; preparation never concatenates or decodes the complete table.
Static, decoder, lexical, expert, vision and prediction owners have restricted source views.

`NGramEmbeddingSpec` describes header geometry without literal hash constants.
`BoundTargetSpec` validates an exact per-injection control map before construction.
`geometry_fingerprint` identifies cold construction; `state_fingerprint` includes exact
integer controls and selected parameter transforms. Prompt-cache and speculative target
identity use the bound fingerprint. Equal headers and state shapes alone cannot authorize
restoring state computed with different controls.

Routed expert admission compares member recipes against the admitted catalog output
geometry. Direct GGUF outputs may contain packed words, native block bytes or scale
companions; their shapes differ from the physical GGUF tensor's decoded dimensions.
The physical source shape and exact bounded member recipe remain independently checked.
Ordinary source preparation retains architecture-selected GGUF representations,
including bounded F32 views for head permutations that cross quantization blocks;
binding opens those retained views with the same metadata used for cold selection.

Architecture-owned GGUF recipes map metadata and tensors, undo normalization offsets,
negative exponential recurrent coefficients, recurrent head permutations and convolution
squeezing, and join split indexer projections. The shared gated-delta head-layout recipe
transforms every encoded weight/scale/bias companion. Expert selection precedes decoding
or concatenation. Equal encodings retain packed storage; differently encoded concatenations
and permutations crossing physical blocks select bounded F32 views.

`GgufCatalog` is a metadata-only `RecipeCatalog` sharing the canonical output projection
with readable store construction. `GgufTextPlan` selects representations and mappings;
`GgufTargetPlan` declares geometry, compact row descriptors, requirements and capabilities.
Binding compares exact keys, metadata and provenance against an existing retained source.
GGUF table owners share the prepared store and reader cache with ordinary weights while
retaining distinct logical bank and unit identities. Exact GGUF integer metadata and
SafeTensors I64 controls use the same hash validation. GGUF padding matches its retained
physical row extent; SafeTensors uses the declared alignment and shard geometry.

`eredu-gguf` owns generic representation selection and bounded conversion. Encoded views
preserve bytes without payload reads during catalog construction; `DecodedF32` views
report expanded geometry while retaining physical encoding provenance. Per-allocation
limits apply to payload reads and expanded outputs, not metadata-only table extents.
Header validation checks arithmetic, blocks, ranges, alignment and overlap.
`BlockDecoder` writes Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2K/Q3K/Q4K/Q5K/Q6K/MXFP4 into caller-owned
F32 storage without heap scratch. MLX's generic fallback decodes only an admitted compact
row acquisition and retains its staging/output through completion.

Load-time transforms apply to aligned dense matrices, embeddings and expert input axes.
Unaligned, already encoded and nonmatrix parameters retain their declared format. Source
and executable layouts remain separate through bounded loading. Table rows, integer
controls and convolution kernels are not projection-transform targets. Encoded linear
weight requirements retain their final packed axis even when further transformations
are disallowed; capability checks still validate block alignment and exact shapes.

### Retained cold selection and construction

`RoutedTextRequirements` and `SelectedRoutedTextRealization` retain source-free row plans;
`PreparedRoutedTextArchitecture` owns their bound sources. The shared construction handoff
validates exact row descriptors without querying mechanisms again. Missing/unexpected
row sources reject. `PreparationMechanismProvider` supplies generic row storage, decoder
workspace and memory facts. `RecordingMechanisms` retains queried descriptors/responses;
changes to storage, support, workspace or retention invalidate cached agreement.

`TargetExecutionPlan` retains the source owner, spec, requirements and capability estimate.
Selection returns `SelectedTargetExecution`; typed visitors enforce attention, fixed-state
and append-stream access. The source-free container header plans use the same requirement
author over `RecipeCatalog`, physical source declarations and static/unit/expert recipes.
Parameter placement and workspace reporting require metadata catalogs, not readable
sources. Native estimates include ordinary source copies, intermediates and outputs even
when a later source permits direct encoded initialization.

`TargetLoadProjection` lowers `NormalizedLoadRequest` once in the architecture layer.
It validates invocation/history bounds, pages and summary capacity before row queries,
authors stream bindings and retains the exact selection request. The architecture supplies
defaults of one lane, up to 512 tokens per invocation, declared history, summary tiles up
to 256 records and a shared 64 MiB device/zero-host table cache. Explicit generic parameter
bank controls, page geometry and execution policies remain authoritative. Broad workspace
headroom does not allocate at the ceiling or promise complete memory accounting.

`SelectedPreparation` retains `SelectedQwen4Construction` target or conditional header
authority alongside the matching ordinary routed selection. Large immutable plans use
`Arc`; bound construction is boxed. Selection caches own no inspection/cache back-reference.
Source construction verifies origin-token/session/policy/route identity, opens retained
resolutions/mappings once and carries the bound authority through ordinary source graphs
and typed visitors. Backends receive generic routes and state contracts without family
configuration or artifact-name dispatch.

`StoreHandoffScope` distinguishes primary parameter authority from complete auxiliary
provider authority. `restrict_store_handoff` checks provenance, encoding, geometry and
recipes before retaining the exact declared view. Joint prediction construction retains
separate target and complete-provider views; auxiliary modules and row providers own their
sources. Strict unused-parameter validation consumes these views. A table or integer
buffer does not become an ordinary layer parameter because it shares a checkpoint file.

### Media ingress, processing and chunked control

`VisionPlan` and `GgufVisionPlan` own source-free metadata, recipes, processor policy and
layer-sized owners; `PreparedVision` retains bound source authority. Shared Qwen vision
modules provide full attention, bilinearly interpolated learned positions, LayerNorm and
exact merger GELU. Flash-Next declares no DeepStack captures. Temporal patch recipes are
shared across Qwen families, with explicit dense promotion and retained encoded companions.

GGUF projector admission does not invent tokenizer IDs. Facade tokenizer resolution binds
complete `QwenMediaTokenIds`; architecture validation checks vocabulary, uniqueness, target
aliases and image-reset semantics. `ConditionalHeaderExecutionPlan` joins target/vision
catalogs and selects weights, state, banks and processor before source access. Integrated
SafeTensors uses one source; GGUF retains target/projector sources in a joined view.
An explicit GGUF projector with default media intent selects prepared/projected images and
video plus supported raw processing. Disabled media rejects a supplied projector.

Core input metadata and wire identity retain `OriginalTokenIds`. Runtime's
`OriginalTokenIds` and `TokenVisibility` preserve exact host or I32/U32 identities;
floating proxies and out-of-range IDs reject. Shared `qwen::ingress` owns ordered
text/media assembly. Flash-Next requires original IDs even for embedding-based input.
Embedding replacement preserves those IDs and produces checked I32 lexical input.
Shared rotary preparation checks the complete grid and arithmetic before allocation.

`MediaAdmissionConfig` admits host handles through `PreparedInputInspector` without a
native tensor or source. `AdmittedMediaInput` retains checked IDs, grids, rotary axes and
decode delta. Native preparation consumes that proof and the retained request, without
reconstructing IDs. `MediaPrefillRequest` prepares the request once; `PreparedMediaInput::assemble`
embeds only the current target span and supports cuts inside raw or projected media.
Released video framing supplies one frame per media part with processor-owned timestamps.
Media ingress accepts initial batch-one, unpadded requests with the declared rotary policy.

Architecture-owned composite and retained-prefill policy borrows `ConditionalKernel`
through the shared adapter. The target is canonical group zero and vision group one; graph dependencies run
vision before target ingress. Text/projected continuation does not acquire encoder blocks.
Optional pass-through groups do not publish projector observations. Vision-unit, target
internal and ingress observations use the shared traversal.

`PositionDelta` is an exact I32 fixed component on the first target unit. Initial media
input commits its signed rotary delta; decode adds it to the causal prefix with checked
arithmetic. Continuation cannot change it. Reset clears it and fresh text commits zero.
`qwen4_exp.request.v2` transports the delta, original IDs, visibility, causal prefix,
complete residual streams and FP32 rotary products. Request validation precedes mutation;
resumption never recovers IDs from activations.

`InvocationLimits::invocation_tokens` and `TargetLimits::invocation_tokens` bound
batch-times-chunk work, not the whole prompt. Single-lane media uses `history_tokens` as
its complete-request ceiling; an explicit processor budget can lower it. Whole-request
IDs, rotary products, inputs and encoder output remain owned by request/state objects.
Target assembly and n-gram row requests scale with the chunk. Reused proofs validate
extent and retained chunk authority; the history ceiling includes later decode tokens.

`ChunkedPrefillRequest` supplies architecture-owned input spans and immutable continuation.
The shared replicated session owns advancement, output selection, rollback and commitment.
Retained encoder values complete before cursor publication, including rows unused by the
first span. Completion publishes whole-input cache identity only on the final chunk.
A later failure preserves earlier committed chunks. Cursor clones accompany matching
model-state snapshots and carry no observation, transport or copy budget.

`CompositeArchitecture` declares the typed request and chunk limit. The native prompt owns
an erased typed cursor. Ordinary and controlled generation use that same cursor; one
controlled nonfinal prefix step samples no token and consumes no controller decision.
Snapshots share immutable prepared/encoder roots and copy cursor progress with model state.
Initial raw-media snapshots complete and share input roots. Absolute capture selections
project onto each chunk, retaining request spans in records and using the cumulative
completion/commit ledger. Final-position scores come only from the final chunk.

The facade accepts prepared backend prompts with explicit corresponding original token
IDs for observed and controlled generation. Nonempty interventions on genuinely partial
spans produce a typed rejection; single-invocation interventions are supported. Direct
low-level composite calls using one full-request observer require caller-driven cursor
advancement for chunked capture. Embedded MTP uses the shared prediction driver and its
own speculative control boundaries, including streamed media prefill.

`MediaExecutionPolicy` declares required representations and optional coarse
`ProcessorRequestBudget` limits for decoded input, host buffers, output bytes, planning
items and decoder positions. Ordinary processing requires no accounting budget. The Qwen
planner checks supplied limits before expensive transforms; architecture host admission
checks IDs and complete token geometry before native lowering. Per-call limits cannot
relax retained ceilings. Backend input roots, including replaced or abandoned roots,
remain in the existing completion scope.

### Embedded prediction and shared state ownership

`PreparedPredictionSource::safetensors` admits an MTP-only role from a standalone artifact
or restricted complete checkpoint. It retains exact formats and bounded expert recipes
without reading target, table, vocabulary or vision payload. `with_prediction_source`
checks normalized vocabulary, schedule, residual/recurrent/expert/rotary and lexical-reset
compatibility. An attached role cannot be replaced. These checks establish architecture
compatibility, not authenticated training lineage; each source retains physical provenance.

Ordinary GGUF loading accepts a matching SafeTensors prediction companion through retained
`prediction_source` and explicit embedded drafting. GGUF owns embedding/output parameters;
the companion owns prediction parameters. `CompositeCheckpointSource` reports composite
backends and exact per-key provenance. Default drafting selects the ordinary target and
excludes unused prediction parameters from binding. Explicit SafeTensors text/media MTP
joins source-free requirements and independently selected prediction state before weight reads.

`PredictionSpec`, `PredictionShared` and `PredictionUnit` reuse shared mixer/feed-forward
construction while declaring independent state, bank identity and rotary policy. Fusion
normalizes flattened target residual streams before shared per-stream projection. Prediction
capture retains all residual axes; its gated collapse feeds target vocabulary projection
without applying the target collapse again. `MaterializedQwen4Prediction` uses ordinary
completion-aware materialization and injected parameter providers. Output/state roots
survive failed equations until completion ownership permits release.

`PreparedPredictionExecution` retains target authority, selected prediction weights/state,
source and bank identities. The generic `PreparedRoutedPrediction` visitor binds modules
and independent state, registers residency owners, then moves auxiliary providers into the
executor. Vocabulary storage remains shared with the target. `EmbeddedSpeculativeContract`
identifies residual capture, target/prediction invocation bounds, proposal depth, placement
and state. The existing speculative finalizer and scheduler consume these contracts.

Sequential prediction extensions declare a maximum causal prefill span. The shared driver
streams target chunks, retains the last residual to pair with the next original token and
seeds the predictor before advancing. The outer speculative checkpoint covers both lanes;
activation scopes retain actual invocation widths. Composite continuation retains original
IDs, positions and completed encoder products. Fused predictors use their declared complete
input contract unless they support sequential pairing.

Commitment restores the pre-proposal prediction checkpoint and replays the committed anchor
and accepted suffix with actual target captures. Tentative captures and rejected tokens do
not enter canonical state. The immutable anchor and mutable proposal capture remain distinct.
Providers and source owners remain outside rewindable lane state. Restore performs no weight
reads and does not refund lookup, observation, transport or copy activity. Copy admission
precedes isolation; failed isolation consumes copy allowance while releasing provisional retention.

### Tensor, pipeline and expert placement

`TargetTensorPartition` localizes recurrent heads and component-major Q/K/V convolution
channels, attention queries with GQA K/V replication, and shared/routed FFN intermediates.
Neutral range recipes retain exact encoded companion coordinates. Unrepresentable shards
reject rather than split encoding blocks. Vocabulary, residual mixers, lexical computation,
routers and indexers are replicated. Bound state identity distinguishes rank and hash controls.
Parallel execution uses the same equations, observations and lifecycle as ordinary execution.

`PreparedTensorTarget` retains global/local specs, rank plan and restricted static/unit/expert
sources. Checked `select_bounded` pushdown follows container-authored canonical transforms;
full bank recipes are partitioned before expert members. Column selection can read a complete
expert before slicing, so source I/O is not assumed to fall exactly by rank count. Repeated
reshape selections preserve source order and duplicates. `local_layout` projects exact
physical selections into `LocalModelLayout`; fused ranges use ordered indices and replicated
K/V heads retain the appropriate partial ranges on adjacent ranks. Placement applies once.

`TargetPartitionExecutionPlan` retains header authority, tensor geometry, stage-local state,
row descriptors and communication requirements. Source factories bind exact sources/controls
through typed visitors. `expert_realization` declares balanced EP ownership using TP-local
width; routers keep global IDs. `local_expert_layout` composes expert-axis and TP selections.
Existing provider exchange owns global-to-local translation and the shared equation separately
reduces its shared TP contribution.

`RoutedTensorReductions` declares inactive pipeline participation. Residual reductions use
selected wire representation; lexical lookup has an exact I32 status reduction and compact
`[batch * tokens * ngram_features, row_width]` data payload. Replicated vocabulary omits
unneeded embedding/readout collectives. `DistributedNeuralBackend::parallel_rank` supplies
the actual retained member order; construction rejects mismatched rank/size.

`PreparedRowLookups::bind_tensor_parallel` grants table acquisition to one owner and gives
peers compact declarations. `TensorParallelRowLookup` retains distinct status/data groups.
EP status includes the full participation group, including inactive pipeline stages; data
stays in the owning TP group. Every participant checks I32 failure status before row data,
so rejection cannot leave peers submitting a collective alone. Only the owner materializes
row decoders/scalars and reports pool occupancy. Stages without lexical units have no rows.

`ParallelCompositeArchitecture` separates parallel ingress from ordinary composite admission.
`CompositeTextRequirements` retains architecture-authored wire schemas, sequence bounds and
source-unit continuation geometry. Inactive media groups have no continuation
geometry; active encoders validate every source-unit shape before execution. The
shared composite executor consumes typed reductions and retained row sources.
Conditional partitioning composes target TP/PP/EP with shared Qwen vision sharding. Receiving vision stages construct geometry/rotary state without accessing
another stage's patch parameters, then consume transported activation. Encoder edges use
full-request extent; target edges use decoder chunks. Boundaries retain complete residual
streams, original IDs, visibility and rotary state.

Distributed chunk cursors agree request/step extents and commit epoch through selected
failure agreement before execution. Request-construction failures participate before peers
execute; cursor position/encoder continuation advance only after ordinary completion/commit.
An optional fixed integer descriptor extends the neutral agreement mechanism; status-only
calls retain their scalar implementation.

Prediction partitions use the same mixer/feed-forward placement rules, encoded companions
and independent state. Fusion, residual mixers, indexers and readout remain replicated.
Prediction units are TP-local and replicated across target PP/EP groups. Every prediction
participant retains embedding/output through static-role ownership. Supplemental banks use
TP-local widths and full expert cardinality; each bank's EP plan governs exchange independently
of target EP ownership. Provider layout is distinct from target module layout, so auxiliary
coordinates cannot widen target binding authority. Final target residual publication and the
tensor context feed the existing prediction-extension driver.

### Discovery, capture and parameter coordinates

Flash-Next discovery describes lexical execution units separately from physical decoder
layers, recurrent/QSA mixers, gated residuals, routed/shared branches, collapse and MTP fusion.
Logical parameter groups record vocabulary/fusion/readout sharing; mutable recurrent,
convolution, indexer and token-history owners have separate state nodes. Tables and expert
banks remain compact nodes. Target-to-prediction edges carry residuals before collapse.
Generic attention attributes do not fully describe QSA selection and retain explicit partial
coverage. Logical graphs are not allocator or residency estimates.

`PredictionDiscovery` joins the architecture descriptor and loaded activation discovery in
one node namespace. Explicit invocation roots establish target/prediction-depth scope;
sealed hook contracts validate it. Architecture specs supply shapes and mutable/read-only
status; native facts govern actual collector/editor support. Internal prediction discovery
retains target node identities and uses exact TP parameter layouts for channel placement.
Detailed routed feed-forward prediction points are outside this retained catalog.

Observation points include target unit boundaries, recurrent intermediates, attention and
shared-expert channels, routing events and selected QSA positions. Integer positions remain
read-only and preserve integer representation. Sparse expert evidence uses actual selected
rows and route order rather than an expert-dense tensor. Routing policy declares exclusions,
zero contribution, forced IDs and raw/transformed/ranking bias stages through the shared
intervention driver. Logit masking uses `LogitsBeforeSampling` and native support facts;
the canonical target publication is `model.logits`.

`SpeculativeCaptureObserver` and shared capture/intervention ledgers retain loaded authority
across restore/fork while budgets and invocation identity stay outside mutable snapshots.
Failed forwards retain bounded aborted evidence. Parameter/component coordinates consume
architecture-authored physical placement, including shared K/V and encoded chunks. Peer
queries use retained family authority in `SelectedExecution`; low-level queries without
that authority reject rather than infer balanced sharding.

Chat and tool behavior uses the shared portable tokenizer, template recognition and facade
parser/grammar drivers. Released reasoning controls and tagged-parameter syntax are template
policy. Opaque grammar snapshot rejection is explicit in capabilities; parser forks alone
do not establish full-session snapshot admission.

### Immutable projection parameters

Ordinary and row-parallel linear execution share a bias-free native projection helper.
Ordinary execution adds bias after projection; row-parallel execution adds it once after
the collective. Input observers borrow the retained parameter without temporarily removing
it. Observer errors and unwinding therefore preserve weight/bias identity for retries.
Encoding selection and input transforms belong to the shared helper; completion recovery
and distributed failure agreement retain their independent contracts.

For a group carrying retained failure authority, row-parallel projection agrees local
projection and observer results before reduction. Admission checks the selected reduction
limits and compares exact partial-output and bias metadata; local input widths may differ
across uneven shards. The original local error takes precedence over agreement failure,
including the neutral observer's retained source. Bias is applied once after the bounded
reduction completes, and its local result is agreed before returning. A reduction failure
returns to its completion owner without a recovery collective. Groups without retained
authority preserve their ordinary execution path. These gates cover returned construction
errors, not coordinated recovery from panics or failure in preceding family operations.

### Resource ranges, storage inspection and completion

Memory reporting distinguishes state payload, selected caches, workspace allowances and
unknown physical costs. Complete allocation accounting and whole-encoder physical-fit
proofs are not loading requirements. Context/chunk limits, checkpoint/expert/table residency,
large transfer/conversion bounds and safe completion ownership govern practical execution.

Named `PreparedStateResource` owners report target and independent prediction state without
duplicating shared parameter tasks. Estimation uses exact component geometry, dtype, placement,
segment offsets and active lanes. Append summaries can grow in discrete blocks; rounded
per-position rates are amortized. Resident stream payload and lane allowances are distinct
from paged cache occupancy. `TargetResourceReport` retains these facts and selected row
workspace/cache policy; encoded table source size is not resident memory. Queries describe
hypothetical horizons, not observed independent predictor frontiers.

Selected rows retain `MechanismMemoryContract` facts for one maximum acquisition. MLX
reports indices, compaction, conversion intermediates and independently owned output.
An identity gather supplies owned storage because `copy`/`contiguous` can share backing.
Scratch ends at completed decode; output belongs to its consumer. Generic GGML fallback
workspace includes host/native staging and copy. Unknown allocator capacity, source/cache
backing, private kernels and request composition stay explicit; local allowances are not a
complete process peak or concurrent aggregate reservation.

Native host-transfer storage carries allocation identities from the residency manager.
Aliases retain their owner; rematerialized buffers get new identities. Transfer lifetimes
remain distinct from cache ownership, so host eviction cannot release an in-flight source.
Read-only residency reporting uses retained metadata without reaping or admission failure
checks. Failed managers remain observable while new acquisition rejects. Process-local
identities need rank/session namespaces for cross-process composition; observations omit
host bookkeeping and do not turn measured capacities into future bounds.

Detached submission reporting uses neutral inventory contracts and pure native retained
metadata. The calling-thread report limits detached nodes to 64 and allocation records to
256. Truncation, undescribed nodes, borrowed queues and reentrant/in-progress reaping are
explicit. Reporting does not poll, evaluate, acquire or retire native work, establish global
coverage, or authorize refunds. Safe retirement still requires completion evidence or teardown.

`InputTransferResources` derives logical wire counts/payload and selected staging capacity
bounds before copies. Repeated aliases count once per wire occurrence; serial staging uses
maximum capacity while destination payloads sum. Native transfer recovery retains sources,
created destinations and staging through partial failures/unwinding. Successful copies complete
before publication and preserve structural/semantic cache identity. These are operation
ceilings rather than process/session memory reservations.

`NeuralBackend::mechanism_memory_in_context` supplies side-effect-free facts for an already
selected context; cold queries remain context-free. `segmented_attention_memory` describes
the actual segmented primitive. Shared Qwen vision declares compact invocation geometry,
projection formats and host setup/output payload. `FlattenedPatchSpec::memory_invocation`
uses complete channel/temporal/spatial geometry and unbound dense projection semantics.
Operator invocation counts do not imply simultaneous allocations. Cold ranges do not require
live callbacks during vision forwarding.

`MechanismInvocation::MultiAxisRotary` carries position shape, representation and the exact
portable rotary spec. Outputs preserve the prefix with F32 cosine/sine. MLX reports retained
implementation payloads with explicit unknown capacity, fusion, private scratch and retirement.
No backend reconstructs family rotary sections or guesses integer promotion. Geometry is
validated before host coordinate allocation.

`Tensor::inspect_storage` surveys established backing only. Default facts are unavailable;
MLX uses native wrapper equality and allocator capacity where known without exposing pointers,
evaluating, copying or initializing streams. Lazy/imported/custom backing stays unknown.
Records retain no lease and do not prevent donation. A survey's compact indices are local,
not persistent owner IDs, and external wrappers can overlap.

`TensorStorageSnapshot<Role>` combines caller-authored semantic roles, checked logical extents
and one backing survey over borrowed roots. Logical records are not deduplicated by role;
only the survey maps backing. Prepared-input traversal defines wire order; Qwen ingress and
vision define semantic roles. Media reports include original admitted inputs, prepared
products and encoder continuation because original pixels can outlive patch concatenation.
Current activations, target state, parameters, native scratch and host allocations are separate
owners. Fresh request construction retains the original admission proof without retaining
redundant chunk products/encoder continuations. Inspection promises neither summed physical
peak nor admission authority.

Reusable miniature artifacts and comparison fixtures belong to `eredu-evaluation` as development
dependencies. Codec and architecture tests import packaged fixture constants rather than
reading sibling workspace paths, so archive validation exercises the same references.
Nemotron Labs Diffusion's draft adapter configuration and Pixtral numerical reference
are packaged constants shared by architecture, backend and facade tests.
Production architectures, facade and backends do not depend on evaluation. Native
conformance uses ordinary selected constructors and erased executables. Numerical, distributed
and released-artifact verification scope is recorded in the family validation report.

### Automatic residency memory reserve

The automatic planner reserves 10% of observed available memory for runtime state,
temporary allocations and observation drift. Resident admission compares selected
materialized parameter bytes against the remaining device budget; installed memory
does not replace a current availability observation. This percentage is a tunable
heuristic, not a request-specific bound on context state or activation workspace.
The plan explanation includes parameter bytes and post-reserve host/device budgets.
CLI cached-plan admission uses the same default reserve as fresh planning.

### Native selection without intermediate host reads

The MLX routed selector retains cutoff-tie detection as a device reduction. Metal
repairs tied partitions for banks up to 1024 entries and selection widths up to 16
using the value-only partition convention of the CPU reference. Prefix ordering
uses native stable sorting. Larger geometries retain the general compatibility
path. The compatibility kernel derives from libc++ selection under
Apache-2.0 WITH LLVM-exception; its source records the attribution.

`Tensor::topk_rows` and `Tensor::sorted_unique_indices` are optional neutral row
ranking and validated index-compaction mechanisms. The MLX realization uses device
sorting/gathering and retains finite-score, range, ordering and uniqueness checks
in the submission validation batch. Completion checks these assertions before
publishing the submission; standalone synchronous calls validate before returning.
QSA scoring equations, causal bounds, stream geometry and padding remain in
`eredu-architectures`. The native path operates on one decode query per lane within one admitted summary
tile; prefill and longer histories retain completed-tile portable selection instead of retaining an
unbounded lazy graph. Workspace includes the bounded position tile.

Resident indexed attention queues a single decode query per batch directly from
device positions, preserving slot multiplicity, invalid-slot masking, source
origins, local sources and sinks. Range assertions join the same completion
batch. It does not materialize a host position list or complete an intermediate
attention result. Cache-owned paged readers and multi-query prefill retain the
completed-query bounded gather path.

Small Q4_K grouped projections (up to 16 routes) share the ordinary two-output-row
SIMD vector kernel, with independent route/input and physical-bank offsets.
Larger route batches retain the grouped prefill kernel. Encoded blocks and row
views remain unchanged; FP32 accumulation precedes the requested activation dtype.
The ignored `q4k_grouped_decode_benchmark` compares device timeline durations for
both kernels after warmup; its fixture is a `[32,640,2560]` bank with ten routes.

Scalar-scaled E4M3 weight-only conversion is a portable checkpoint recipe.
`DecodeTensorFp8` validates encoded weight/scalar scale metadata, preserves both
source dependencies and pushes bounded selections into the weight source. The
MLX recipe materializer owns E4M3 decoding, positive finite-scale validation,
completion retention and FP32 output allocation. Its workspace report counts
encoded inputs, decoded intermediates and the final output. This checkpoint
conversion does not select or imply an activation quantizer.

The speculative executor returns `SpeculativeProposalStep` through its native or
erased adapter. Causal sources expose logits for ordinary draft processing and
sampling. Masked sources expose the selected token with target-matching evidence;
the shared proposal driver preserves that token and does not create a causal
draft distribution or consume causal draft sampling draws. Both variants enter
the same verification, completion, cache-commit and publication transaction.

Pixtral optional-group transport declares its tensor-sharded subgroup. Each image
block contributes two residual projection sums over the complete unmerged patch
activation. The architecture supplies their exact shapes and per-stage order;
replicated patch ingress and projector contribute no sums. Stages without image
units have empty collective waves, and projected-only image input skips the tower.

The Nemotron VLM graph independently addresses the vision layers, one projection
unit, and the causal text layers. Only the first image owner retains patch ingress;
the projection owner retains the pre-merger norm, spatial merger and projection
parameters. Decoder embedding and readout belong to their selected first and last
text owners. Stateless image stages have no decoder cache layout. Image boundaries
carry their declared patch or projected-feature width; decoder continuations carry
attention invocation and causal frontier metadata validated against local ownership.

`eredu-runtime::speculative::shared_backbone` borrows one ordinary executable and
retains one canonical mutable cache. Masked drafting reads an immutable prefix
checkpoint, seeds the pending committed token, and returns selected proposals with
target-matching evidence. Verification appends causal inputs with adapters disabled;
partial commitment restores and replays only the accepted inputs transactionally.
The request seed retains proposal randomness independently of target sampling.
Conversion-retention and memory observations attribute the base only to the target.

Pinned GGUF admission retains its original artifact fingerprint source until the
lazy store captures bounded-read file guards. A payload-free identity check after
store construction closes that handoff. Nemotron's explicit Pixtral companion
validates its complete tensor/processor metadata and keeps the image-break vector
as artifact metadata. Q/K source coordinates receive architecture-owned bounded
reshape/transpose recipes into the same canonical vision parameter catalog.

`eredu-checkpoint::gguf_fingerprint` streams encoded tensor ranges through one
8 MiB host buffer to establish a shard-layout-independent catalog identity.
Canonical semantic metadata includes the complete tokenizer; sorted tensor names,
encodings, dimensions, endianness and exact payload bytes contribute. Split
ordinals/counts, alignment and offsets do not. Current headers must match retained
descriptors, secondary semantic metadata must agree with the first owner, and file
guards fence the complete pass. Family admission compares this identity with its
pinned conversions while retaining separate fingerprints of every actual member.

Shared-backbone execution declarations retain separate ordinary unmasking and
speculative draft policies. Family construction supplies seeding, mask identity,
complete canvas geometry and draft-only adapter availability. Native generation
consumes these declarations without inspecting family configuration. A settled
AR/diffusion handoff retains next-token logits separately from any committed token
awaiting cache append; the target sampler history and independent refinement RNG
survive the handoff. Pixtral preprocessing performs complete host-buffer admission
before floating cubic resize and emits patches/grid metadata; the architecture
owns image-marker insertion.

GGUF allocation ceilings are header-preserving checkpoint projections. They
select the logical conversion allocation bound without reopening source members
or changing validated encoded tensor ranges. Nemotron's content-backed admission
accepts an already-open checkpoint and retains its F32 conversion header catalog.

The shared physical driver supplies the remaining commitment budget to each
backend operation. A speculative invocation keeps its complete draft canvas while
shortening proposal capacity; a one-token budget uses causal target sampling.
Exact submission completion precedes controller commitment. The backend's
`commit_text_token` publication hook runs only after controller acceptance and
settles canonical cache state before returning the committed token. Full-attention
native state can transactionally retain a verified prefix without re-running its
equations; other state profiles retain checkpoint replay semantics.

The shared-backbone native queue context borrows a request-local target observer
only for its verification invocation. Its erased borrowing contract keeps queue
assignments covariant and does not install a mutable observer on shared model
modules. Causal sampling takes the last conditional row when instrumented prefill
retains the complete score matrix. Capture admission and commitment coordinates
are shared across causal and masked modes; a snapshot after an accepted verification
decision retains the correct next commitment coordinate. Native growth forecasts
include verification overshoot and the baseline, verified, restoration and rollback
cache allocations; those transient state copies do not duplicate base parameters.

Decoding commands use the retained text-preparation transport before physical
model work. `eredu-core` owns exact active/pending mode and committed-position
agreement. `eredu-runtime` frames scalar controls within a fixed 4 KiB per-rank
encoding bound, confirms each native exchange and the complete resolved command,
and charges topology-sized host buffers plus fixed native/host frame credits before
allocation. MLX supplies the existing admitted world transport; it does not select
family mode semantics. The session ledger survives model restore and fork. A
completed conflicting-command rejection permits a corrected request; unagreed
transport or protocol failure fences subsequent work.

Selected native residency maps every exact architecture-owned execution address
to its policy-local slot for both resident and bounded acquisition. A pipeline cut
can own global unit one at local ordinal zero. Parameter inspection and execution
use the same retained address map and reject another global owner even when its
group happens to match.

`ArchitectureGroupPlacement::PipelineSequence` partitions an ordered dependency
chain as one contiguous sequence of independently addressable units. The same
portable placement result determines route endpoints, first/last static roles,
local state slices and output ownership. Prepared session facts carry an optional
prompt-cache shard identity. Stateless stages have no shard identity and participate
in coordinated execution without claiming a decoder cache. The runtime factory
requests an explicit stateless mechanism realization; native hybrid storage retains
no layout, layers or cache manager for that owner. A stage may own only stateless vision or
projector units. Such a stage reports complete architecture state geometry while
its selected local state is empty; it does not constrain a decoder owner's causal
frontier. Ordinary per-group pipeline placement remains available for independent
roots. A sequence requires nonempty groups and at least one unit per selected stage.
Prepared media collective schedules consume the exact retained stage ranges.
Communication admission bounds reductions by native encoder sequence and width
as well as decoder activations, including spatial merge expansion.

Nemotron VLM places vision, projector and decoder in this ordered sequence. Its
projector follows vision request activity, so cached text continuation requires no
new media output. Composite dependency checks use architecture-owned invocation
activity for every producer. Nemotron construction dispatches before the larger
other-family constructor frame, keeping its debug construction within the default
Rust worker stack without changing tensor or parameter ownership.

Cartesian pipeline boundaries with more than two stages and tensor replicas select
explicit world-participating source broadcasts. The retained route still names
one producer and one consumer; every rank submits bounded, role-exact framing in
singleton route order. Idle ranks retain no transferred architecture state. This
transport requires an advertised world-boundary mechanism and all-rank failure
agreement before submission. MLX preserves the byte frame through a single-source
integer reduction on the retained world communicator, with the same exact
completion, header validation and quarantine ownership as endpoint transport.
Each byte frame reaches bounded exact completion before submitting the next role,
so independently lazy source and consumer graphs cannot reorder a multi-role route.

Dense composite decoder waves declare embedding reductions, per-layer residual
reductions and vocabulary gathering alongside media collectives. Inactive stages
submit matching bounded tensors using the declared peer counts. Wave ordinals map
to retained physical owners, so a decoder starting after vision/projector stages
uses its actual first owner's schedule. These invocation schedules contain no
parameter bindings and allocate no duplicate base weights.

The portable `DraftAdapterSource` names a separately admitted low-rank artifact
and its expected base repository. Architecture cold selection retains its strict
PEFT equations, exact tensor geometry, source aliases, shard admissions and
configuration bytes. The target role joins these parameters with the unchanged
base catalog; each physical artifact opens once. Mixed GGUF/SafeTensors catalogs
preserve encoding and provenance per source tensor. Prepared construction consumes
the retained architecture projection, including adapters, and rejects changed
adapter configuration before binding. Mode transitions consume this retained
selection without artifact discovery or merged base weights.

`SharedBackboneMemoryPlan` separates immutable canvas and additive-projection
geometry from target parameter residency. The ordinary generation forecast hook
and speculative source hook consume the same retained policy. Source-specific
continuation forecasts accept one shared target cache without inventing a causal
draft cache. Nemotron cold inspection reports selected rank parameters and local
KV bytes independently from the unsharded artifact resource profile; stateless
media owners report zero local KV payload.

`RefinementMemoryPlan` retains the admitted canvas, token domain, numerical
storage envelope and entropy-stability history. Fresh diffusion forecasts include
full-canvas transient block K/V and score storage, followed by causal append;
shortened output budgets preserve the trained invocation width. Continuation
forecasts combine active and pending policies over observed native cache capacity,
including pending block append and verification overshoot. Parameter residency
appears once in each phase; tentative state and snapshot retention have separate
contributions.

The native fused-attention adapter casts floating visibility masks to the promoted
query/key/value precision while preserving Boolean masks. Visibility representation therefore
keeps the same output/cache precision as the unmasked invocation. Mechanism
memory includes an optional mask-conversion envelope; its exact bound mask shape,
source dtype and aliasing remain runtime observations.

Nemotron's published text and Pixtral equations use input-precision rotary
products, scores and probabilities. Pixtral learned RMS normalization rounds
the normalized value to input precision before multiplying by the learned gain.
Its exact GELU projector restores input precision after FP32 pointwise arithmetic.
Image patches enter the patch projection in
its parameter precision, and image features enter text token positions in text
embedding precision. Caller-prepared text embeddings use that same embedding
precision. Scalar-scaled weight-only FP8 projections decode to F32;
their scalar scales remain mandatory provenance-guarded source companions
consumed by conversion recipes, outside the executable parameter catalog.
The neutral checkpoint schema marks such tensors as `RecipeInput`: exact
admission retains access for recipe dependency resolution, while prepared
parameter normalization excludes them from resident neural bindings.
The admitted F32 scale therefore establishes text cache storage precision even
when embedding and normalization parameters retain BF16 source storage.
Converted FP32 text queries select FP32 text rotary tables and products; the
vision tower retains its BF16 input-precision rotary arithmetic. Independent
mixed-precision comparisons use these selected arithmetic boundaries.

The portable `VerificationRound` accounts for causal target steps and proposal
verification. Zero proposals retains the pending committed input and samples its
single causal target row through the ordinary processed-target and publication
transaction. A zero-capacity draft-source state preserves alignment without
proposal execution or proposal RNG draws. Completion, cache rollback and semantic
commitment have the same ownership contract in both cases.

`GenerationRequestConfig` carries the initial `DecodingMode` and output sampling
policy. Its default is AR. `GenerationRequestTable` schedules causal steps with
zero proposals and speculative transactions with the mode's validated allowance.
Causal requests require no draft capacity and never enter optimistic drafting.
Ordinary fused samplers enter the same table as completed-decision sources.
They retain opaque sampler state, non-cloneable commitment controllers, native
output handles and exact completions. A numerical progress operation never
queries or advances a controller. Completed tokens enter the table's canonical
sequence after exact settlement and native commitment; no sampled token is
represented as logits or probability evidence. Score-backed sources retain
transactional causal verification and independent proposal/target randomness.
Scoped executor contexts borrow the loaded runtime without requiring a static
backend or cloning parameters. Owned lane caches may contain borrowed controller
state; exclusive borrowed caches retain a move-only dereference lease.
`GenerationCache` gives each lane exclusive borrowed or owned canonical state.
Prepared native batch requests move their fresh caches into the common driver;
cache and parameter owners are not cloned when entering controlled execution.
Borrowed caches retain their enclosing scope's final state through the same
request lifecycle and completion guards.
The shared-backbone executor also supplies complete ordinary diffusion through
`refine_block`. Initial diffusion retains causal prefill scores without sampling
or publishing an AR token. Complete admitted canvases remain separate from the
canonical causal cache; selected identities and position rows survive successive
left-to-right commitments. A replaced candidate invalidates the dependent suffix,
which is refined over the same immutable causal prefix with the corrected
committed canvas positions pinned. Ordinary diffusion disables draft adapters.
Target, proposal and ordinary refinement randomness have independent owners.

`RefinementProgress` separates settled numerical work from selected-row finalization.
A numerical advancement returns a pending complete canvas or its final selected
rows; finalization runs in a subsequent advancement. The shared executor retains
immutable numerical values and the refinement RNG in its request seed, charges
them in continuation and snapshot estimates, and restores them with the prefix.
Each table advancement commits at most one diffusion position. Partially finalized
canvases retain their selected rows, complete width and commitment cursor in
snapshots and forks. Mode commands wait for the active canvas to finish, then
append its complete finalized suffix through the common causal transaction before
acknowledging the effective commitment position. Cancellation is coordinated
before tentative selections become output, including a budget-terminal candidate.
Failed publication leaves the canonical sequence and cache unchanged and fences
further mode commands. Refinement counters record complete evaluated canvases and
numerical invocations separately from causal target and draft-token work.

`CausalExecutor` binds an ordinary model to the same request table and append
contract, with no assistant parameters, seed cache or proposal RNG.
Final generation-executor erasure belongs to `eredu-core`.
`GenerationExecutorTypes`, `GenerationExecutorCacheFactory` and
`DynGenerationExecutor` carry the complete constructed executor contract for
causal, refinement and speculative sources. They preserve source verification
policy, request-local mode configuration, full position-row refinement,
completion ownership and state/copy estimates. `eredu-architectures` retains
family pairing and typed cache construction; backends supply the fixed input,
score, stream, completion and telemetry types. Context-aware cache construction
uses the retained device/group binding. Erasure does not select a family or
materialize another model.
`CausalVerification` owns the exact causal inputs and immutable output for
causal-only, independent-model, shared-backbone, embedded-prediction and
feature-conditioned external executors. It retains either host token IDs or
native token tensors together with the source's complete output, including
auxiliary features. Ordinary-model executors use the same submission and
score-row mechanisms; family feature equations, proposal evidence and
accepted-prefix state remain separate. Native
shared-backbone mode settlement uses this causal executor for its finalized suffix.
Per-request lookahead and completion deadlines remain bounded by the shared
scheduler's resource ceilings. Released draft-source preparation retains the
same mode options used by the facade, including draft adapter and refinement
selection; it does not reconstruct a second proposal policy from a draft count.
The request and its durable snapshot retain the committed suffix awaiting
cache append explicitly. Causal settlement consumes the complete suffix once,
independently of proposal capacity; proposal transactions consume their pending
input once. Parser output is not used to reconstruct cache alignment.
Zero-width causal continuation forecasts omit draft adapter workspace and
optimistic overshoot while retaining the selected parameter owners and causal
verification/checkpoint allowances. Active or admitted pending diffusion retains
its full-canvas forecast independently of proposal width; partial canvas score
rows and their commitment cursor remain charged alongside snapshot reservations.
Settled mode changes reuse this transaction to append the complete committed suffix and
retain its causal next-token row without sampling. Ordinary shared-backbone
handoffs use this same portable append transaction; native composition only
publishes its settled cache and score row. Typed executor erasure
preserves mode facts, bounded command agreement and score-row storage estimates.
Request-local draft policy updates consume retained architecture declarations;
native adapters supply transport and storage mechanisms. The numerical masked
draft capability is distinct from complete diffusion commitment. Incremental
verification cursors live in `eredu-runtime` as
`SharedBackboneTransaction`: they retain completed evidence, enforce sampled-target
commitment, and select the accepted cache prefix. MLX binds its ordinary sampler,
observer and cache publication to that portable cursor without duplicating the
proposal-matching or rollback equations. Cursor snapshots isolate writable
progress while sharing immutable checkpoint evidence. Unsupported completion
dispositions reject before proposal execution. Failed submission, completion or
evidence retention restores the canonical checkpoint while retaining native
outputs through rollback; indeterminate native restoration propagates its failure.
The normalizer excludes declared GGUF artifact metadata and recipe-only inputs
from neural binding for both primary and companion sources. Admission retains
their exact source authority and schema obligations. A companion metadata entry
does not acquire a canonical parameter or a duplicate embedding allocation.
Independent branch command owners carry queued modes through serial exchange without
sharing writable mailbox state or reopening parameters. Closing command
ownership does not establish native completion.

Composite boundary encoding and acceptance receive the invocation's neutral
tensor context. Architectures use it to construct declared metadata and restore
activation precision after wire conversion. Nemotron text boundaries carry mode,
causal frontier and activation precision; vision and projector boundaries carry
activation precision. The receiver restores the declared floating element type
before local layers execute. Every metadata tensor belongs to the admitted wire
schema and its transport/copy accounting.

Partition collective requirements distinguish wire encoding from native
arithmetic. Reduction, readout gather and publication contracts admit the
selected floating parameter/recipe widths and F32 promotion. Boundary transfer
roles retain their exact wire dtype; the broader compute contract does not
expand boundary tensor geometry or communication limits.

Distributed output owners and destination placeholders prepare the same selected
wire dtype before publication. Local computation and cache storage preserve
their own precision. Native cast failures retain their original source through
the neutral `OutputPreparationFailed` error and enter ordinary execution-failure
agreement before publication.

The shared exact GELU primitive evaluates low-precision inputs in F32 and rounds
once to the input dtype at its output, matching the pointwise activation
contract. It preserves BF16/F16 through projector and other exact-GELU layers.

Shared-backbone activation discovery uses the ordinary target descriptor and
producer placement. It adds no prediction parameter owner or second component
layout. `SharedBackboneExecutor` owns request-scoped admitted observers; native
mechanisms bridge tensor and error domains while the portable collector owns
reservation, provenance and delivery. Activation schema version 3 distinguishes
ordinary `Refinement` iterations from draft-only `MaskedDraft` iterations. Each
complete-canvas forward has its own physical width and monotonically increasing
invocation identity. Prefill, verification and accepted-prefix replay retain
causal attribution. Internal edits apply within their actual forward, before
refinement selection or verification decisions.

Shared-executor snapshots retain admitted observation authority alongside native
cache and sampling state. Restore prepares authority replacement before native
state publication and commits it after a successful replacement. Authority
restoration never rewinds invocation identities or cumulative capture usage.
Observer checkpoint storage contributes to the complete snapshot estimate.

Residual capture placement follows the containing architecture execution unit.
A residual addition has a complete hidden value after its TP reduction and emits
that value on the executing unit's replicas. Parameter sharing and earlier
contribution producers do not move this invocation to another pipeline stage.
Neutral component layouts derive the unit through declared node ancestry and
prove its parameter ownership before exposing global capture. Prediction scopes
retain their separate invocation placement.

`eredu-core::generation::VerificationCursor` owns frozen target decisions and
accepted-prefix bookkeeping for both batched resolution and explicitly advanced
verification. Native evidence and sampling remain outside the cursor. Staging a
choice exposes the exact accepted cache boundary without publishing it; only a
successful enclosing commitment advances the cursor. A pending choice rejects
another score read or draw, and replacement, bonus or terminal acceptance closes
the transaction. Cloned cursors isolate writable progress. Their additional host
backing contributes to snapshot and continuation storage estimates.

Capture schema version 3 carries independent raw-patch and projected-media extents
in each physical invocation. Prepared architecture media plans supply these host
counts before observation transaction admission; decoder delimiter positions do
not enter either count. Pixtral tower normalization, reduced attention and MLP
writes, residuals, spatial-merger output, projector projections and GELU have
original and effective hooks. Their media axes resolve from the invocation and
complete values have replicated producer ownership on the containing execution
unit. Capture reservations bound the actual resolved tensors before retention,
transformation or receipt transport.

The independent causal executor composes `TargetRefinement` with its retained
assistant. The policy supplies complete target refinement state and actual
capabilities without changing proposal probabilities. `AutoregressiveInput`
moves the prepared prompt and independent refinement stream into the request;
its seed retains an immutable target prefix when refinement is realized, the
aligned assistant prefix and numerical continuation. Snapshot and continuation
estimates charge those seeds and numerical values separately from base weights.
Native preparation moves lane samplers, semantic state and callbacks through the
input handoff; it does not clone either model owner. Both canonical caches advance
through the same causal settlement transaction after diffusion commitment.

Ordinary causal checkpoints settle retained native values and use the cache's
immutable-prefix checkpoint mechanism. Published append-only KV roots remain
shared across causal, diffusion and verification seeds; mutable overwrite-capable
state receives the mechanism's isolated copy. Restoring a numerical invocation
creates independent host state without duplicating immutable KV backing. Durable
snapshot estimates remain conservative isolated-copy bounds, and paid snapshot
allowances remain cumulative.

`RefinementExecutionPolicy` retains ordinary numerical equations independently of
speculative drafting. Family declarations supply complete geometry, defaults,
causal-sampling support and masked seed policy. Backend adapters project these
facts through the selected execution rather than inferring entropy semantics from
the absence of a draft source. The ordinary single-model executor composes the
same `TargetRefinement` contract as independent drafting. Entropy feedback uses
full batch/canvas/vocabulary geometry and the context output's conditioning
precision; masked unmasking keeps its selected-token and pinned-prefix semantics.
All numerical continuation values settle before a paused advancement is exposed.

DiffusionGemma derives compact expert ownership from family configuration, optional
rank-local block geometry, and the admitted Cartesian topology. Cold text and
composite requirements, selected target construction, and transform-source
construction consume the same source-free declaration. Expert ownership never
requires native parameter modules; materialization retains the resulting plan
without reconstructing group identities or expert widths.

DiffusionGemma text construction retains a source-free validated component
definition before allocating pinned modules. Canonical parameters, graph,
invocation paths, and transport descriptions project that definition. Each
shared layer references one authoritative context slot; refinement borrows the
same slot through a read-only cache view. Local geometry projects owned slot
policies into the global ordinal schedule, and partition construction consumes
that schedule and the definition's state ownership rules. The shared adapter exposes the retained global declaration; the authoritative
partition and kernel local state view select its owned interval, including
nonzero global offsets. Unowned slots remain declaration metadata.

Shared component text contracts retain the kernel's explicit invocation admission,
iterative geometry, and output-row selection. Block refinement requires the
separate `RefinementComponentKernel` capability; a default text hook alone does
not advertise iterative execution. Composite parameter traversal borrows the same
kernel-owned pinned aggregate used by execution, without an additional resource
owner or parameter storage instance.

DiffusionGemma text execution uses `ComponentAdapter` over `TextKernel` and its
single pinned aggregate. Runtime implements ordinary, provider-backed, observed,
parallel, text/refinement, and typed pipeline contracts. Kernels retain equations,
feedback precision, read-only refinement views, and invocation phase/frontier
transport. Context output has no prediction-score kind; refinement publishes all
position scores. Low-level operations borrow the kernel through `kernel()` or
`kernel_mut()`; constructors live on `TextKernel` and return the shared adapter.

Selected standalone text construction consumes an authoritative partition before
allocation. Canonical binding identities select complete embedding, normalization,
head, and feedback modules while preserving retained local placement. A tied
readout replica does not imply ownership of token ingress. Receiver invocations
require transported hidden values and phase/frontier metadata. Target and
transform-source preparation use the same selected constructor. Composite family
equations borrow this same typed child kernel under the enclosing component's
resource and completion lifecycle; they retain their declared media assembly and
boundary semantics.

DiffusionGemma composite parameter catalogs are source-free. Shared Gemma vision
geometry declares patch and position storage, optional standardization, per-layer
clipping bounds and projection matrices under the family's physical roots.
Ordinary format expansion supplies exact packed weights and companion metadata.
Composite target and transform-source planning consume these declarations without
constructing image modules or traversing native placeholders. Text storage retains
its canonical shared owners when projected into the composite execution graph.

Complete and rank-local DiffusionGemma composite construction retains a validated
`ComponentDefinition` before materializing parameters. Its image units have no
persistent state; each text unit references its authoritative global context
slot. Retained local geometry agrees with the owned portion of the complete
state schedule. Graph, unit paths, canonical storage, transport and state
partition rules project the same definition for execution and inspection.

Gemma vision request state prepares rotary positions and visibility without pinned
modules and executes independently leased layers. DiffusionGemma image storage
selects its tower and projection roles separately; ranks with no such roles retain
only neutral image geometry and request-local tensors. Composite partitions
validate canonical bindings, unit ranges, state mappings and boundary geometry
before materialization. The same selected bindings restrict text pinned resources
and image roles for target and transform-source construction. Unit factories reject
unowned invocations; tied readout storage remains independent of ingress ownership.
A composite owner with no text invocations has no local context layout. Its
parameter-free image ingress validates a stateless runtime view, while text
continuations require their selected context slots. Global invocation ordinals
and context-slot identities remain unchanged on owners with text layers.

DiffusionGemma composite execution uses `ComponentAdapter` over `CompositeKernel`
and the same kernel-owned pinned aggregate for binding, execution and completion
roots. The shared runtime implements ordinary, provider-backed, observed,
parallel and typed partition contracts. Family hooks retain media assembly,
context/refinement equations, image-span visibility, collective semantics and
invocation phase/frontier transport. Composite request admission and media
continuation planning remain architecture-owned. Constructors live on
`CompositeKernel`; low-level equations and expert realization use `kernel()` and
`kernel_mut()` without an additional execution adapter or resource owner.

Nemotron Labs Diffusion retains source-free validated component definitions for
complete, tensor-local and partitioned text and Pixtral composite construction.
Causal, masked-refinement and draft invocations reference the same canonical text
parameters and authoritative context slots. Draft LoRA matrices belong to their
participating layer's resource owner; they do not duplicate base storage or
context state. Vision and projector invocations have no persistent state.
Graph, unit paths, transport, parameter catalogs and state partition rules project
the retained definition. Selected definitions preserve admitted physical placement
and validate canonical bindings, context geometry and typed boundary width before
native allocation. Complete state schedules retain global slot ordinals while
selected partitions project their actual local context ownership.

Complete Nemotron Labs Diffusion text execution uses `ComponentAdapter` over
`TextKernel` and one kernel-owned pinned aggregate. The kernel borrows the shared
decoder's physical construction and readout mechanisms directly. Runtime owns
parameter traversal, graph and address checks, retained state projections, and
ordinary, observed, parallel, text and refinement driver contracts. Family hooks
retain the causal/transient attention equation, optional draft delta, full-canvas
visibility and final-position projection. Serial and parallel unit execution use
one equation with an optional neutral parallel context and instrumentation.
Constructors live on `TextKernel` and return the shared adapter; low-level module
and equation access uses `kernel()` and `kernel_mut()`.

Selected Nemotron Labs Diffusion text uses the same `TextKernel`, specialized by
`DecoderStaticResources` and `DecoderGeometry`. Its retained selection owns only
its pinned roles and independently leased layers. Global unit addresses map to
local context slots through shared decoder geometry; the declaration preserves
global state identities. The shared adapter supplies typed partition and pipeline
contracts, while family hooks validate and transport invocation mode, canonical
prefix frontier and activation precision. Complete and selected text share mask
preparation, layer equations, observation seams and readout mechanisms. Only the
complete resource/geometry specialization implements the explicit block-refinement
kernel capability. Selected low-level ingress preparation remains available on the
kernel; `from_partition` validates authoritative ownership before materialization.

Complete Nemotron Pixtral composite execution uses `ComponentAdapter` over
`CompositeKernel` and its pinned text/image aggregate. The composite borrows the
same raw text kernel used by standalone text execution under the enclosing
resource and completion lifecycle. Runtime implements graph and address checks,
parameter traversal, state ownership projections, and ordinary, observed,
parallel and provider-backed driver contracts. Family hooks retain request-local
roots, prepared input
admission, image marker assembly, request-optional tower/projection activity,
collective schedules and image transport semantics. Serial and parallel ingress
assembly and unit execution share equations with an optional neutral parallel
context. Constructors live on `CompositeKernel`; direct module access uses
`kernel()` and `kernel_mut()`.

Complete and selected Nemotron Pixtral composites use one `CompositeKernel<B, R>`.
`CompositeResources` exposes the actual text storage/state geometry and optional
image ingress without creating absent modules. Image-only owners prepare request
geometry with no mutable context layout. Selected construction validates canonical
bindings, transport width and local ownership before materialization; target and
transform-source construction use that same path. Global text identities retain
their local cache mapping, and typed decoder resume validates its global origin.
Both views share media preparation, assembly, independently leased unit equations,
readout and retained request roots under the runtime adapter. The selected view
adds typed partition and parallel provider capabilities without a second driver.

Observation coverage remains specific to the invoked operation. Selected typed
partition readout emits internal readout hooks; its ordinary low-level parallel
readout leaves the observer unused. Complete ordinary parallel readout emits those
hooks. Composite observed ingress retains its explicit image-embedding seam.
