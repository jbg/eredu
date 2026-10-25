# Model and checkpoint support

`eredu` selects an architecture from checkpoint metadata, normalizes its
configuration, and validates the complete tensor catalog before loading
payloads. A compatible family name alone is not sufficient: unsupported layer
geometry, tensor layout, quantization, tokenizer, processor, or execution
policy produces a structured error.

## Inspect an artifact

Artifact admission and text preparation are portable, including builds with no
MLX feature. Execution selection uses the explicitly supplied backend factory:

```rust,no_run
use eredu::api::{
    inspect_artifact, ArtifactSource, CompanionSelection, PreparedTextModel,
    TextInspectionOptions, TextModelOptions,
};
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{DevicePlan, ExecutionPlan};

let source = ArtifactSource::new("/path/to/model.gguf")
    .with_projector(CompanionSelection::Explicit("/path/to/projector.gguf".into()));
let artifact = inspect_artifact(&source)?;
let prepared = PreparedTextModel::from_artifact(artifact, TextModelOptions::default())?;
let factory = MlxBackendFactory::default();
let plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "metal:0")?);
let mut selected = prepared.select_execution_plan(&factory, &plan)?;
let report = selected.inspect(TextInspectionOptions::default());
assert!(report.is_loadable());
let model = selected.load()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Admission reads configuration and bounded checkpoint headers and validates
processor metadata and media companions without materializing weights.
Selection validates the requested quantization, residency, topology, and drafting
policy without creating native resources. The selected report describes the exact
admission consumed by loading. Native device availability and payload integrity
remain realization-time concerns.

Configure a single or named override once in `TextModelOptions::chat_template`.
Preparation retains the resolved tokenizer, template, EOS ids, tokenizer variables,
and generation defaults for both inspection and loading. The override takes
precedence over embedded GGUF metadata and sidecars. Invalid selected metadata
returns a typed preparation error without falling back to another source.
`TextInspectionOptions::chat_request` supplies messages, tools, and template kwargs
to probe before loading; probe results apply to that request.

## Family matrix

The table describes high-level dispatch. “SafeTensors” means a supported
Hugging Face-style model directory with the required configuration, tokenizer,
and processor files. “GGUF” means the canonical architecture metadata and
tensor layout expected by Eredu; multimodal families may require a sibling
projector.

| Family | Inputs | SafeTensors | GGUF | Notable support |
| --- | --- | :---: | :---: | --- |
| Llama / Mistral | text | yes | yes | full or sliding attention, tied or untied head |
| Phi-4 (original 14B) | text | `phi3` | `phi3` | fused QKV and gate/up projections, full-head RoPE; [validation](phi4-validation.md) |
| Nanbeige 4.2 | text | yes | `nanbeige` | shared-weight repeated GQA/SwiGLU stack; independent KV caches for each pass |
| Qwen2 / Qwen2.5 | text | yes | `qwen2` | dense text models; exact full/sliding schedules |
| Qwen3 | text | yes | `qwen3`, `qwen3moe` | dense and MoE |
| Qwen3-VL | text, image, video | yes | `qwen3vl`, `qwen3vlmoe` + projector | dense and MoE text decoders, MRoPE and DeepStack |
| Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8 using `qwen3_5` | text; image/video where present | yes | `qwen3next`, `qwen35`, `qwen35moe` | shared Qwen3.5 hybrid recurrent/full-attention contract; dense and MoE, projector and embedded-MTP support; official Qwen3.6/3.8 chat and tagged-parameter tools |
| Qwen3.8-Flash-Next (`qwen4_exp`) | text, image, video | `qwen4_exp`, `qwen4_exp_text` | `qwen4exp` | distinct family route; resident/host/disk, image/video MTP, chunked media capture/control and snapshot fixtures; ordinary text and media TP/PP/EP native conformance; released-payload parity pending |
| DeepSeek-V3 / R1 | text | yes | `deepseek2` | compressed MLA, dense-to-MoE schedules, native block FP8 |
| DeepSeek-V4 | text | yes | `deepseek4` | mHC compressed sparse attention, routed/shared experts, native FP8 and mixed FP4; SafeTensors embedded MTP and fused DSpark; base GGUF target only |
| GPT-OSS | text | yes | `gpt-oss` | Harmony chat, scheduled attention, native MXFP4 experts |
| Gemma 2 | text | yes | `gemma2` | alternating sliding/full GQA, four RMSNorms, GELU and attention/output soft caps; [validation status](gemma2-validation.md) |
| Gemma 4 | text, image, video, audio | yes | `gemma4` + projector | dense and MoE variants, shared-KV and media towers |
| Gemma 4 assistant | text draft model | yes | `gemma4_assistant` | external multi-token prediction |
| DiffusionGemma | text, image, video | yes | `diffusion-gemma` (text only) | shared context/refinement stack, soft feedback, controlled block generation and constrained finalization; [validation](diffusion-gemma-validation.md) |
| K2 Horizon | text | yes | `k2-horizon` | dense, MoE and MoVA; independently addressable value/feed-forward experts, grouped RMSNorm, YaRN, IFM reasoning/tools and external drafting; [validation](k2-horizon-validation.md) |
| Kimi Linear | text | yes | `kimi-linear` | KDA/MLA hybrid layers, dense and routed/shared experts |
| LFM2 / LFM2.5 | text | yes | `lfm2`, `lfm2moe` | causal-convolution/attention hybrid, dense and MoE |
| Nemotron-H | text | yes | `nemotron_h`, `nemotron_h_moe` | Mamba, attention, dense MLP, and MoE layers |
| Inkling | text, image, audio | yes | `inkling` + projector | multimodal hMLP ingress and dense/MoE text execution |
| Muse-Glimmer | text, image, video | yes | `muse-glimmer`; optional image-only projector | vision tower, mixed attention, ATEM/DFlash paths |
| Muse Spark Open | text; image, audio, video where present | yes: official BF16/NVFP4, MLX 3/4-bit and oQ3e/oQ4e | `muse-spark` + selected media companions | compressed MoE, scheduled attention, ATEM, DSpark/DFlash2; [validation scope](#muse-spark-evidence-and-support-status) |
| Moshi | realtime codec tokens | yes | no | temporal/depth generation without a codec dependency |
| PersonaPlex | realtime speech tokens | yes | no | transformed model layout, voice and text conditioning |
| NemotronLabs VoiceChat | realtime PCM speech, text, transcription, tools | yes | `nemotron_voicechat`; explicit split component bundle and companion | single-device controlled/iterator session; [validation](voicechat-validation.md) |

Low-level Kimi Linear construction consumes a retained declaration through
`CompleteKimiKernel::from_definition`, `from_parallel_definition` or
`KimiKernel::from_partition`. `kimi_linear::project_parameters` preserves logical
consumers, observation seams and recurrent/compressed state ownership while
selecting physical formats.

Low-level LFM2 construction consumes a retained component declaration through
`CompleteLfm2Kernel::from_definition`, `from_parallel_definition` or
`Lfm2Kernel::from_partition`. `lfm2::project_parameters` selects physical formats
while preserving logical consumers, observation seams and mixed state ownership.

Low-level text Qwen hybrid construction takes a retained component declaration
through `QwenKernel::from_definition`, `from_parallel_definition` or `from_partition`.
`qwen::hybrid::project_parameters` selects physical formats while retaining
logical consumers, observation seams and independent target/MTP state slots.

Low-level conditional Qwen hybrid construction takes a retained component
declaration through `ConditionalKernel::from_definition`,
`from_parallel_definition` or `from_partition`.
`qwen::hybrid::project_conditional_parameters` selects physical formats while
retaining logical consumers, observation seams and target/MTP state ownership.

Low-level Qwen3-VL construction takes a retained component declaration through
`VlKernel::from_definition`, `from_parallel_definition` or `from_partition`.
`qwen::vl::project_parameters` selects physical formats while retaining canonical
consumers, observation seams and the persisted mRoPE position-delta state.
Complete and selected partitions share one materializer and family equations.

Low-level Gemma 4 construction consumes an explicit retained component declaration
through `GemmaKernel::from_definition` or `from_parallel_definition`.
`gemma4::model::project_parameters` selects physical formats while retaining
logical consumers and observation seams. Partition selection retains its declared
shared-KV receiver caches through explicit state-slot materialization. Optional
audio output-bias ownership follows admitted tensor presence.

Low-level Inkling construction consumes an explicit retained component declaration
through `InklingKernel::from_definition` or `from_parallel_definition`.
`inkling::model::project_parameters` selects executable physical formats while
retaining logical parameter consumers, observation seams and independent target
and embedded-prediction state identities. Complete and selected partition
preparation use the same constructors and family equations.

DiffusionGemma uses the ordinary loading and generation entry points. Its
`refinement` request setting controls the iterative schedule; `draft_events`
enables replaceable tentative blocks in observed delivery. Final token constraints
apply separately at each committed position. Causal sampling overrides and
likelihood-ratio speculative verification reject refinement position scores with
`RequiresCausalConditionalScores`. Resident, host-layerwise and disk-streamed
execution share the same layer units and authoritative context. The published
SafeTensors release includes vision; its published GGUF contains text parameters
and reconstructs the tokenizer from explicit BPE merges.

NemotronLabs VoiceChat uses the shared family registry and realtime loading
entry points for pinned official SafeTensors, complete GGUF and split GGUF artifacts.
`inspect_architecture` recognizes the nested NeMo configuration without requiring
an invented Hugging Face `model_type` and describes the coupled component graph.
`api::realtime::prepare_realtime_model` discovers the family and retains its exact
sources and auxiliary assets. `prepare_mlx_realtime` admits protocol, host policy,
parameter residency and cache realization before native binding;
`PreparedMlxRealtime::load` returns categorical execution or a bound speech session.
Speech requires `RealtimeLoadOptions::Speech`; categorical Moshi/PersonaPlex
execution uses `RealtimeLoadOptions::Categorical`. A mismatched input protocol is a
typed error. These realtime capabilities do not imply ordinary text generation,
full prefill/decode capture discovery, or a distributed VoiceChat session. Selected
component sources retain exact assets, canonical recipes and conversion bounds;
incomplete split components require an explicit companion. Host conditioning
uses retained tokenizer bytes and saved integer payloads. Learned module units
project exact canonical recipes into shared parameter selection and atomic
binding, with restricted source authority per unit. Its standalone RNNT, causal subsampling and FastConformer
components have neutral and MLX CPU numerical coverage. Bounded host PCM/mel
processing follows the pinned rolling-window schedule and explicit PCM precision.
Cold speech construction validates component geometry and builds individual
unloaded units with retained parameter bindings; perception, TTS and codec
preparation defaults to F32 execution recipes and conversion bounds. The portable
speech precision policy also selects eligible F16/BF16 weights and arithmetic.
Prepared speech materialization binds those units through shared completion
handling and consumes them into the perception/RNNT, TTS and codec components
used by the coupled executor. The resulting aggregate retains its bound speech
weights. Fixed-Aria encoding shares the decoder's codec codebooks, and assembly
preserves independently bound modules without constructing replacement weights.
Prepared input construction uses the checkpoint's retained window and mel filters
without reopening sources or regenerating coefficients. Shared realtime loading
consumes this retained construction under the selected speech-graph residency
policy.
Perception also executes through the shared resident, host-layerwise and
disk-streamed providers: subsampling, each Conformer block and the final projection
are separate units with no pinned learned parameters. Cold binding plans retain
only their exact source inputs and account for selected precision conversions. PCM,
RNNT and startup-silence composition accept this same provider-backed perception;
raw encoder features and the language projection come from one traversal.
Neutral and MLX CPU tests compare streaming outputs and caches across residency
policies, including retry after an aborted unit. Perception tensor parallelism
partitions Conformer attention and both feed-forward branches through shared
placement and reductions. Subsampling, convolution, normalization, modality
projection and full-width streaming caches remain replicated with per-rank
accounting. The native component supports the same three residency policies;
it does not select a full distributed session. Explicit perception pipeline
bindings support cuts after subsampling, within Conformer blocks and before
projection through the shared transport and completion driver, including combined
tensor/pipeline partitions. Variable encoded counts and exact frame frontiers
travel with the selected stage; both output feature branches share one encoder
invocation. These component entry points do not select a distributed speech graph or own
distributed conversational commit.
The causal TTS and bidirectional character transformers also use shared resident,
host-layerwise and disk-streamed traversal. Each decoder block and final
normalization is separately materialized. Causal execution borrows the enclosing
transaction's cache branch and retains its exact key/value roots through unit
completion; character execution is stateless. TTS generation, sampling, speech
rendering and coupled execution accept these selected transformer providers.
Cold bindings retain exact component sources and select bounded conversion to
weights in the admitted format and selected speech precision.
MoG also uses resident, host-layerwise and disk-streamed providers. Each residual
block, the final-normalization/logits head and the selected-component bank are
separate units. Prediction does not acquire the selected bank in bounded execution;
selected evaluation consumes retained guided hidden values and explicit Gaussian
draws without repeating the prediction stack. Cold bindings retain only exact MoG
source inputs and account for selected precision conversions. TTS generation, RVQ
refinement, speech sampling/rendering and coupled execution accept the same MoG
provider contract. Its tensor-parallel execution shards residual MLP projections
and the large selected-component bank's low-rank dimension through shared placement
and reductions. Small readouts and normalization remain accounted replicas;
sampling decisions remain explicit inputs. The same three residency policies
operate on exact selected source recipes. Native TP uses retained failure
authority, with invocation, acquisition and local projection checks before
reductions. This component route does not select
a complete distributed speech session. TTS RVQ feedback likewise uses resident, host-layerwise and
disk-streamed providers for its canonical stacked codebook bank and shared
feedback projection. Depth sums and quantization skip the projection; prompt
ingress and refinement use the same embedding parameter. The unpartitioned bank
is one 62 MiB F32 residency unit, with storage conversion charged separately.
Tensor-parallel providers partition its latent axis with the feedback projection's
input columns. Global distance reduction precedes code selection; full-width
feedback and replicated code candidates preserve the same refinement schedule.
Exact selected recipes reduce per-rank learned storage under all three residency
policies. Guarded TP execution agrees refinement position and preparation failures
before score, padding and feedback reductions; rejected candidates preserve the
parent codes. This does not select a complete distributed speech session.
Character embeddings/flags, prompt ingress and gated fusion likewise have separate
resident, host-layerwise and disk-streamed units. Character phases release their
leases around transformer execution, and ingress follows completed RVQ feedback;
empty character selections acquire neither character phase. Tensor-parallel
conditioning shards the large embedding/projection matrices and fusion biases,
reconstructing full-width values before control additions and final normalization.
Small control parameters are replicated and included in per-rank residency.
These component selections use the same provider contract. The EAR neural
decoder independently loads upsampling, ConvNeXt and output-projection units
under the same residency policies. Tensor-parallel decoding partitions grouped
upsampling outputs, ConvNeXt intermediate channels and spectral output rows;
small depthwise parameters, normalization and causal carry are explicit replicas.
Full-width unit outputs retain ordinary streaming and drain behavior.
Decoder PP and combined TP/PP retain only each stage's convolution caches and
parameter recipes under all three residency policies. Boundary geometry follows
upsampling; shared completion and commit protect each candidate cache branch.
The portable waveform pipeline combines codebook lookup, published spectra and
overlap-add reconstruction through the same waveform execution contract as
ordinary decoding. It retains stage-local convolution carry with spectral history,
coordinates component failures, and uses explicit request and epoch authority for
step, replay and drain. Speech and coupled execution retain this selected codec
state through their shared transaction paths. This component composition does
not select a complete distributed conversation.
Codec residual codebooks are independently
leased by depth, with partial encode/decode requests loading only their selected
range. Tensor-parallel quantizers shard channels at every depth, reduce distance
scores before code selection and return full-width latents through the same
provider interface. Their retained variance scalars remain explicit replicas.
Fixed-Aria encoding borrows that same quantizer provider. Its finite
encoder also uses independently leased input, ConvNeXt and downsampling units.
Tensor-parallel encoding partitions projection rows, intermediate channels and
per-group downsampling rows, with explicit small parameter replicas and no
persistent encoder cache. These component providers do not establish
a complete TTS or speech-graph residency schedule. The shared pipeline runtime
accepts architecture-declared changes in channel layout, cadence and publication
shape. Finite EAR encoder stages support PP and combined TP/PP through those
contracts under resident, host-layerwise and disk-streamed policies. Spectral
preprocessing and subsequent quantization remain separate; this does not select
an end-to-end distributed waveform or speech session. Full native distributed
execution and released-checkpoint parity are unvalidated.
The function head also has resident, host-layerwise and disk-streamed providers
with one independently leased projection and no static learned head in its graph.
Input shape, hidden width and floating dtype are checked before acquisition.
Generic coupled readout uses the same normalized Nemotron tensor for agent and
function logits, without repeating the backbone traversal. Prepared function-head
bindings retain the admitted dense or affine matrix format, use F32 floating
weights/companions and account for conversion storage. Ordinary unit construction
consumes the same formats and companion bindings.
Balanced vocabulary sharding executes through these same providers and shared
gathering, including uneven row ranges. Exact communicator preflight precedes
acquisition, and retained source recipes account for local output storage and
conversion overhead. Neutral and native CPU TP2 checks compare gathered logits
with the pinned readout fixture across residency policies. Native TP construction
requires retained bounded failure authority and exact F32 gather capacity.
Guarded input and leased-unit checks, shared provider lifecycle agreements and
backend projection checks prevent local failures from sending peers into the
head's gather. This component route does not select a complete distributed speech
session or guard the backbone's internal collectives.
A retained speech-provider constructor assembles all eleven selected providers through
one architecture-owned driver and the ordinary coupled-execution interface.
Its explicit budget sums independent pool capacities and conservative conversion
reserves before native binding. RNNT has separate embedding, recurrent-layer and
joint-projection units under resident, host-layerwise and disk-streamed policies.
All three use the same priming and greedy decoding policy and retain candidate
recurrent state separately from learned parameters.
Its tensor-parallel component path shards embedding and output vocabulary rows,
including uneven partitions and blank ownership, while charging recurrent and
joint-input replicas in full on each rank. An explicit coordinated-frame driver
uses one greedy decision owner, exact integer publication and bounded shared
agreement; all ranks retain unpublished recurrent candidates. These component
entry points do not select a complete distributed speech session.
The ordinary RNNT trait loop is rank-local; distributed decision authority
requires the explicit coordinated-frame entry point.
The recurrent predictor also exposes a pipeline component with stage-local
hidden/cell state, a published final readout, exact layer-cut/step transport and
explicit commit epochs. It uses the shared pipeline executor and all three
parameter residency policies; combined TP/PP replicates each stage's recurrent
parameters within its TP cohort. Embedding/joint phase scheduling and the full
greedy decoder use a separate architecture-owned frame driver over this predictor
entry point and a four-unit projection provider. The projection source excludes
LSTM tensors, shards vocabulary rows within each stage's TP cohort and accounts
for replicas across stages. The ordinary coordinated greedy loop owns decisions;
an explicit non-refundable cursor supplies predictor step and commit identities.
Neutral and MLX CPU Ring checks compare PP2 and TP2/PP2 cached carry/readout and
greedy frame decisions against the pinned NVIDIA component fixture, including
replay, snapshots and recovery from asymmetric preparation failures.
The paired input pipeline composes one perception traversal with that same greedy
driver. It returns raw and projected features together with stage-local perception
and RNNT caches at one acoustic frontier. Exact component outcomes gate the paired
candidate; a failure after perception preserves both parent states. This input
component uses caller-owned commit authority and does not select or publish a
complete distributed conversation session.
Neutral PP2 and MLX CPU Ring PP2 and combined TP2/PP2 checks compare this paired
transition with compatible pinned NVIDIA component fixtures, including replay and asymmetric
perception/RNNT acquisition failures under bounded residency. These component checks do not establish full distributed
conversation or released-checkpoint parity.
Ordinary and pipeline input providers implement one PCM execution contract. The
coupled driver retains the selected input-cache type through live, perception-only
and fresh-audio tool transactions. Distributed PCM checks cover arbitrary chunks,
readiness and feature disagreement, copied-state replay, final padding and silence
continuation. A retained MLX input bundle binds the three components and declared
coordinators once, preserving independent residency policies and reports. This
input selection does not select the remaining distributed conversation graph.
The full coupled constructor admits resident, host-layerwise or disk-streamed
backbone blocks and loading scratch alongside these independent speech allowances
before binding. The native submission adapter
consumes this selected aggregate using its shared completion and publication path,
and retains the checked report for inspection. These are logical learned-parameter
bounds, excluding live state, activation/spectral workspaces and source caches.
Backbone embedding, final normalization and agent-head matrices are pinned;
the two vocabulary matrices impose a 4,480 MiB logical device floor. Native
distributed parity of the full coupled graph is unvalidated. The retained backbone driver binds static
and per-layer recipes into the ordinary Nemotron layer runtime, with the same
invocation and cache semantics for all three parameter residency policies. MLX's joint cache owner branches hybrid
and TTS attention state together, with finite copy admission, isolated paging
namespaces and atomic publication. Native CPU tests cover both guidance rows,
discard, rejected publication and recovery from second-component pool exhaustion.
One retained component construction entry point joins selected speech, backbone and host
conditioning; its MLX adapter uses shared parameter and submission mechanisms.
It returns input carry separately for the enclosing joint transaction. The facade
retained-session startup composes this with selected cache storage and host policy.
The causal and character TTS stacks expose independently selected tensor-parallel
components through the shared layered driver, with local KV geometry and exact
rank-local weight recipes. Two-rank MLX CPU tests cover resident, host-layerwise
and disk-streamed TTS outputs and local caches against the pinned nonzero component
reference. This does not make the bound session loader distributed; complete
speech-graph pipeline transport and coupled rank coordination are not
implemented by that component entry point.
TTS architecture partition contracts describe block cuts and the separate final
normalization stage, including local cache ownership and typed auxiliary
boundaries. Native TTS stage execution supports block ranges, block suffixes plus
normalization, and standalone normalization through the shared partition runtime
and resident, host-layerwise or disk-streamed providers. An explicit prepared
component pipeline binds cross-process TTS transport through the shared pipeline
executor, publication, failure agreement and bounded native completion. It
retains owned candidate caches and accepts scheduler-issued invocation descriptors
and commit epochs. The shared TTS generator accepts this selected causal state
and a separately selected character encoder. Normalization-only stages retain an
explicit model position; prefill and cached frames use the same generator
equations and refinement path as ordinary execution. The causal adapter consumes
one explicit epoch and leaves preparation agreement, random decision authority
and conversational publication to the enclosing owner. This component binding
does not select a distributed coupled VoiceChat session.
Cold preparation declares the coupled cache geometry and shared generation
policy. The MLX initialization adapter consumes exact selected device or paged
cache realizations from the transaction-capable shared selector before creating
the joint owner. Generation starts at an explicit prefill boundary with retained RNG and
validated text controls.
The low-level resident MLX submission adapter runs coupled equations inside the
shared recovery scope and returns the shared prepublication gate. Completion
retains component carry, invocation inputs, RNG, weights and joint cache admission;
equation failures and unwinds cannot publish a candidate. Focused native lifecycle
coverage uses compact caches and injected work, not a complete learned graph.
Shared Nemotron execution accepts fused inputs and exposes one normalized result
for agent and function heads, with neutral TP/PP coverage. EAR-TTS components
include Gemma3/T5Gemma blocks, character conditioning, fixed-Aria silent-carrier
preparation and prompt ingress,
MoG prediction and eight-slot RVQ refinement through a composed cached TTS
transition with retained partial frames and explicit speech feedback policy.
The portable EAR codec composes
finite STFT/ConvNeXt encoding, shared residual quantization, causal ConvNeXt
decoding and streaming ISTFT with explicit
startup and drain behavior. These components reuse shared decoder, convolution
and sampling mechanisms. Portable input composition shares one perception result
with RNNT and retains matching PCM carry; speech composition connects retained
TTS refinement and feedback to actual codec waveform candidates. Their numerical
coverage includes arbitrary PCM chunking and independent NVIDIA codec output
under fixed generation decisions. Backbone composition uses one shared Nemotron
invocation for both heads through resident, layerwise and collective execution; prepared
pipeline ingress declares exact token storage. Schedule receipts can be derived
from actual retained component frontiers. Agent selection uses the released
greedy/special-token bypass and penalty/temperature/nucleus order, retaining
effective feedback and categorical/speech RNG in the shared transaction.
The coupled executor accepts ordinary or TP backbone providers and separately
selected function-head projection. Neutral TP2 coverage runs prompt prefill and
cached live/tool positions through the same full transition, comparing outputs,
clocks and shared retained state with ordinary execution. This provider-level
coverage does not establish native coupled TP/PP construction.
Optional retained ingress agreement rejects different prompt/audio operations,
prompt IDs or framing policy, and tensor geometry before collective lookup.
Agreement between the two lookups prevents a failed participant from advancing
its peer; final fusion is also agreed before layered execution. Selected backbone
cache and parameter-provider frontiers remain unchanged when ingress fails. This
protects channel preparation, independently of collective safety inside Nemotron
blocks.
The shared MLX row-parallel operator agrees returned local projection and observer
errors, reduction limits and output/bias metadata through retained group authority.
It permits unequal input shard widths and applies replicated bias once after the
bounded reduction. This protects the projection boundary; preceding Nemotron
normalization, recurrent and attention preparation have separate failure contracts.
The same coupled transition accepts optional `ExecutionCoordination`. It agrees
recorded invocation identity and exact schedule, prompt, geometry and output
metadata, then gates component preparation/results and output assembly. One owner
resolves conversational overrides and supplies agent and speech randomness;
followers receive decisions and retain synchronized feedback without an RNG or
resolver calls. Authority is single-use, and failed started candidates require
native completion and discard through the shared transaction owner.
This contract covers replicated ordinary providers. Outer component agreements
do not protect unguarded collectives inside TP providers, and native cold session
loading is single-device. It does not establish native distributed conversation
loading or completion.
Independent NVIDIA sampling fixtures cover the policy and native RNG progress.
Agent/function selection also accepts retained owner coordination. One rank
samples and resolves conversational overrides, publishes raw/effective/spoken IDs,
and adopts RNG only after all participants validate the candidate. Followers
retain effective history through the ordinary forced-decision driver without
requiring a random stream or evaluating conversational policy.
TTS refinement consumes that retained stream with atomic RNG rollback on
synchronous failures; neutral and MLX tests compare controlled and uninterrupted
refinement, including native agent-to-speech draw ordering.
Optional retained coordination runs those same transitions with owner-only
mixture/Gaussian draws, follower publication and phase agreement inside the RNG
transaction. Neutral failure/replay coverage and native two-rank CPU Ring checks
cover this component sampling path; they do not establish a complete distributed
conversation session.
Combined waveform rendering preserves RNG on synchronous codec failure. Output
assembly uses the shared completion envelope with separate invocation and PCM
sample coordinates and validates host values before publication.
The portable coupled component executor joins prefill, input, backbone decisions,
speech and drain on one shared branch. Compact neutral tests compare controlled
and ordinary execution, retained state, failure rollback and bounded Nemotron
traversal. Compact MLX CPU coverage also compares coupled equations through native
submission, explicit advancement and shared-scheduler publication, including
nonzero recurrent caches, partial input and drain. It uses shared synthetic and
independent component fixtures with identical native RNG seeds. Full-graph
residency, native distributed execution and released-weight coupled numerical
parity are not validated.
Owned speech and RNNT output pass through the shared scheduler's completion and
host-validation gate. MLX channel observation is shared with Moshi; neutral
coupled tests cover delayed completion, rejection and cancellation.
Channel decisions separate raw tokens, effective backbone feedback and spoken
tool acknowledgments. Spoken text can run through actual character encoding and
cached TTS; an independent NVIDIA fixture covers prefill and three cached steps
with fixed random draws, including EOS and PAD conditioning.
Portable facade helpers prepare bounded ASCII prompts and already formatted tool
responses from the retained tokenizer, render the pinned system/tool template,
and validate complete committed function calls through the shared semantic
parser. Response injection advances only on matching committed feedback;
request/turn tickets reject stale observations and results after interruption.
Recorded-time tool waiting supports bounded reminder speech, explicit playback
completion, optional generic-message gaps and timeout injection through the same
committed channel. Receipt identities protect retries and interruptions. Speech
phrases use the released BOS/text/PAD-tail/EOS framing. A fixed or seeded
acknowledgement phrase can also render during call serialization,
through an independent spoken-token override with PAD backbone feedback. Its
committed cursor survives snapshots and forks; response injection does not repeat
the phrase. `VoiceChatPhrasePolicy` supplies disabled, fixed and seeded streaming
choices for both acknowledgements and initial reminders. The released pools hold
20 and ten phrases respectively; independent retained host streams select both on
committed SOTC, separately from neural sampling. Initial selection survives EOTC;
`VoiceChatInitialReminder::{Configured, Disabled, Text}` chooses the retained
phrase or an explicit per-call policy. Early results wait until initial speech
finishes playback. Spoken-number conversion is not provided by these helpers.
A pure live decision resolver applies RNNT
end-of-utterance and barge-in rules before effective feedback and speech, with commit-checked host
state and independent audio/model clocks. Seventeen original NVIDIA host-method
fixtures cover the core policy. During active tool work, fresh perception applies
the facade's sustained-speech interruption policy, including silent waits. Its
commit invalidates pending tickets and resets agent/function feedback to EOS/PAD
without resetting caches or advancing model time. Explicit interruption also
invalidates stale work.
Perception-only frames retain their projected audio in a bounded queue. After
interruption, shared scheduler work feeds those frames into Nemotron in order
before fresh input resumes, with silent forced feedback and no additional RNNT,
TTS, codec or random advancement. These model-only positions consume the ordinary
model/history budgets and remain transactional and snapshot-visible.
`VoiceChatConversation` composes turn, function and reminder policy as one staged
host successor on the shared prepublication gate. It validates complete calls
and per-invocation publication allowances before neural commit, returns typed
backpressure while an admission is outstanding, and adopts host state through
`VoiceChatPublicationState` in the ordinary neural semantic transaction. Move-only
permits reserve bounded output storage before enqueueing work. An independent
receiver preserves committed speech and tool events after cancellation; dropping
scheduler progress does not lose them. External result retention belongs to
the enclosing driver. These are low-level composition APIs.
An optional external PCM sender accepts bounded 16 kHz chunks during pending work.
Input permits transfer one frame to the ordinary neural branch, preserving PCM
on failed transitions and source attribution on partial finish. Finish-only drain
and cancellation closure use the same transaction owner and shared audio
generation loop.
Committed publications include decoded effective agent text and RNNT transcript
deltas. Both decoders retain bounded private carry and roll back with rejected
neural transitions; output queues charge their UTF-8 bytes. Agent decoding reuses
the facade tokenizer stream and preserves split UTF-8, while RNNT vocabulary
presentation matches nine original NVIDIA host-method cases. Control markers,
function payloads and independent reminder speech are excluded from agent text.
Transactional work preparation selects live, tool, perception-only and reminder
invocations for the shared scheduler after explicit prompt/Aria prefill. Queued
work freezes its host policy and retains input/output capacity. Recorded timing
controls reminder cadence and timeout injection. Input finish declares final-frame
handling and an explicit finite silence suffix, produced lazily through ordinary
perception and speech advancement. Automatic codec drain follows that suffix and
resolved tool work; no duration is inferred from training delays or guarantees
response EOS. Synthetic frames retain distinct provenance and do not increase
accepted or committed source-sample counts. `VoiceChatSession`
provides explicit advancement, polling, cancellation and a recorded-tick iterator
over this shared scheduler path. Portable conformance compares their committed
streams, pending PCM, backpressure, drain and failure-source retention. Session
construction consumes already prepared publication state and execution bindings.
Shared realtime cold selection validates system/tool text, queues and startup
limits through `PreparedVoiceChatSession::from_retained`, without native work.
`PreparedMlxRealtime::load` binds the retained components and cache selections,
prepares Aria/silence, and returns `LoadedMlxRealtime::Speech` containing a
`BoundVoiceChatSession` with PCM ingress and committed output. Its recorded-tick iterator and controlled
advancement share the same session. Shared-model forks retain one parameter
owner and apply retryable backpressure while a sibling invocation is pending.
Prepared work can submit through a portable invocation bridge and an MLX binding
using already materialized execution. This bridge owns distinct main/tool/reminder
output policy and automatic host-gate attachment, preserves native error sources,
and returns an owned backend-neutral scheduler completion. Architecture-owned startup
silence preparation runs one second of zero PCM through fresh full perception
and retains its first projected frame at backbone precision; RNNT and live caches
remain independent. The native adapter completes that preparation through its
shared submission owner. Portable conformance
and MLX/audio compiler checks cover the bridge; native full-session and released
checkpoint parity are not established by that evidence.
The joint MLX neural cache owner supports budgeted isolated capture and restore
without blocking continued inference. Nonzero CPU tests cover both TTS guidance
rows and recurrent state, independent mutation and failure rollback. This cache
mechanism does not capture a complete conversation or replay external events.
Shared realtime snapshots compose model copies with schedule, sampler and RNG
copies; VoiceChat policy/history and MLX RNG have explicit value-copy adapters.
A native CPU composition test restores cache values, VoiceChat clocks and effective
history together and reproduces the next Gaussian draw. Architecture snapshots
also copy internal PCM/mel, perception/RNNT, TTS feedback and codec carry. The
compact coupled native test restores after live speech with partial PCM pending,
then reproduces perception, tool, reminder and terminal-drain outputs and state.
The facade composes these copies into idle `VoiceChatSession` snapshots containing
external PCM, conversation/tool-wait/text state and undelivered output. Restore
rejects crossing external input, delivery or host-action boundaries; it preserves
scheduler and copy accounting. Neutral tests cover partial PCM/speech replay,
complete tool-call retention, native-install failure rollback, callback fencing
and reminder timing. Idle session forks preserve independent PCM/output, rekey
tool and reminder identities, map outstanding external receipts and share bounded
copy accounting. Growth in committed state and output consumes shared retained
credit. Neutral tests cover continuation, tool-ID capacity, growth rejection and
failure or cancellation retention. Native checkpoint-backed facade snapshot/fork parity is
not established.
Released-checkpoint evidence covers single-device execution. Retained public
speech loading and the `eredu realtime` command complete a released-weight, resident
Metal conversation
using NVIDIA's pinned sample: prompt prefill, 396 audio frames, agent text, RNNT
transcription, speech publication and terminal codec drain. The output contains
698,552 finite samples at 22.05 kHz. The pinned tool example also completes its
709-token prompt, publishes `generate_random_number(min=1,max=50)`, accepts a
matching scripted response after 800 ms while perception continues, speaks the
result twenty and drains. A 12-second delayed result also succeeds after WAV EOF:
bounded silence keeps perception active, and the CLI preserves all 80 configured
live response frames after tool processing before drain. This run produces
414,548 finite speech samples and the answer “Your random number is twenty.”
A recorded acknowledgement phrase also produces nonzero speech during the tool
call's 37 serialization positions, followed by the same function arguments,
matching result submission and spoken-result agent text. Controlled/iterator and
snapshot tests cover the phrase cursor; that fixed-phrase recording does not
establish random selection or speech intelligibility.
A separate default-pool run completes two calls in one session, publishes
acknowledgement speech for both, accepts distinct results 20 and 37, and generates
the matching agent answers before drain. Its output contains 885,536 finite
samples. Portable tests cover seeded selection through rejected/discarded work,
snapshot/fork replay and controlled/iterator parity; Python random-sequence
parity and phrase intelligibility are not established.
With both automatic speech defaults enabled, the same released input publishes
26 initial-reminder frames while accepting a result after 800 ms. Response
injection waits until recorded playback finishes at 12,720 ms, then the session
answers “Your random number is twenty” and drains with 460,412 finite samples.
An overlapping recording with an explicit three-frame
interruption threshold cancels the pending tool wait and rejects its delayed
result through the public API. It replays 103 queued projected frames into the
backbone without advancing audio time and answers the new question with “The sky
is blue.” These single-device runs do not establish independent numerical parity,
speech quality, default-threshold or active-reminder interruption, or full-graph
native distributed execution.
See [the admission, example and validation contract](voicechat-validation.md).

Nanbeige support includes `Nanbeige/Nanbeige4.2-3B` and its dense base configuration.
The released 22 physical blocks execute twice, with 44 logical block invocations
and independent KV caches. Both passes use the same checkpoint parameters,
including the RMSNorm between passes. The ordinary
and controlled generation drivers support the same model, including snapshot
restore and fork. Checkpoint tokenizer, chat template, and EOS metadata use the
existing text utilities.

The family uses the standard dense execution paths: fully resident, per-block
host offloading, disk streaming, tensor parallelism, pipeline parallelism and
combined TP/PP, with selected weight transformations and logical-layer captures.
Publisher-layout GGUF checkpoints preserve their native encodings and Q/K row
ordering. Logical invocations reuse canonical physical bodies while retaining
independent KV caches. Between-pass normalization is a separate shared resource;
final normalization retains its accounted pinned replica. Bounded residency
leases each body and between-pass normalization independently. Pipeline owners
materialize the canonical resources needed by their selected invocations.

Validation includes independent numerical oracles, controlled/uninterrupted
generation comparisons, and the downloaded official SafeTensors checkpoint
against the publisher's Transformers implementation. See
[Nanbeige validation](nanbeige-validation.md) for revisions, commands, results,
and the distinction between native and neutral parallel testing. Experimental
n-gram, hyper-connection, depth-attention, and shared-KV equations are rejected
explicitly; they are not used by the released 4.2-3B checkpoint.

DeepSeek-V4 uses the shared execution infrastructure for fully resident,
host-layerwise, and dense disk-streamed weights; independent expert caches;
device-resident or paged request state; and reusable prompt caches, including
embedded MTP and fused DSpark draft state. Eligible dense SafeTensors weights
can be transformed to affine or MXFP4 layouts at load time. Checkpoint-native
FP8 and packed encodings remain native. Requests to produce a checkpoint-native
GGUF block encoding from dense weights fail explicitly; those encodings are
loadable storage formats, not dense quantization targets.

Qwen2 vision-language and Qwen2 MoE checkpoints are not accepted by the Qwen2
text adapter. A model that needs custom Python code or uses an unregistered
tensor layout is not implicitly compatible.

Qwen text checkpoints may declare an explicit `layer_types` schedule. Qwen3
dense and MoE require full attention in every layer; Qwen2 also supports explicit
full/sliding schedules. The schedule must match the decoder depth. Transformers 5
`rope_parameters` is normalized alongside legacy `rope_theta` and `rope_scaling`;
conflicting declarations and unsupported rotary policies are rejected.

Qwen3.5 also admits converted MLX-VLM checkpoints with
`language_model.model.*` and `vision_tower.*` tensors, including the released
`mlx-community/Qwen3.5-0.8B-8bit` artifact. Official SafeTensors and GGUF retain
their format-specific handling. See [MLX checkpoint validation](qwen35-mlx-validation.md)
for conversion semantics, pinned reference results and remaining validation gaps.

## Muse Spark evidence and support status

Spark is a distinct registered identity using the shared neutral `muse` decoder,
media, preparation, placement and execution implementation with Glimmer.
Supported paths include resident, host-layerwise and disk-streamed parameters;
TP, PP, EP and combined placement; resident or paged state; expert caching and
eligible load-time quantization; text/image/waveform/video preparation;
observations and interventions; prompt-cache identity; and ordinary/controlled
sessions with snapshots, restore, fork, cancellation and termination. Independent
external drafting and the published DSpark/DFlash2 assistants use shared proposal,
verification and commitment drivers. Spark accepts media and produces text.

Low-level Muse construction takes an explicit retained component declaration via
`MuseKernel::from_definition`, `from_parallel_definition` or `from_partition`.
`muse::model::project_parameters` selects physical formats while retaining logical
consumers, state ownership and observation seams. The same parent is consumed by
complete and partitioned preparation for both Spark and Glimmer.

The [pinned inventory](../validation/fixtures/muse-spark/inventory.json) records
complete revisions, publisher file sizes/hashes and reference-source identities.
It admits these checkpoint layouts:

| Publisher | Selected layouts | Component coverage |
| --- | --- | --- |
| Meta | official BF16 and ModelOpt NVFP4 SafeTensors | text, vision and waveform towers |
| ggml-org | BF16, Q2_K, Q4_K_S and Q8_0 GGUF | text plus selected published media companions |
| Unsloth | the eight pinned Dynamic GGUF variants, including IQ1/IQ2/IQ3 and Q2/Q3/Q4 | text plus selected published media companions |
| mlx-community | standard 3/4-bit and mixed oQ3e/oQ4e SafeTensors | standard releases are text-only; oQ releases include vision without audio |
| Inferact / RadixArk | Spark DSpark and DFlash2 assistants | target-conditioned text proposals; target supplies tokenizer and media context |

AMD Quark FP8/MXFP4 and Intel GPTQ admission are outside this selected scope.
MLX oQ checkpoints retain each weight's actual physical encoding; oQ4e includes
NVFP4 expert projections. GGUF admission preserves format-specific normalization,
rotary transforms, split members and projector identities. Complete catalogs and
companions are validated before native binding. Selected NVFP4 experts remain
packed until bounded conversion; loading does not expand a complete checkpoint.

Actual artifact coverage determines usable modalities. `MediaCompanionSources`
selects independently admitted BF16 vision and audio sources, including complete
indexed media members without unrelated decoder members. Geometry, token IDs,
processor policy and provenance are checked before loading. Missing components
produce a typed `MissingCompanion` error during request preparation; text-only
loading remains usable. The text CLI exposes `--vision-companion DIRECTORY` and
`--audio-companion DIRECTORY`. Plan overrides replace only the requested role;
replay and benchmark candidates preserve both roles, and cached selections must
match the request. See [model loading](model-loading.md#independent-media-sources).

Decoder policy retains the released 62-layer attention schedule, compressed MoE,
FP32 routing with selection-only correction bias, scale-before-RMS expert output
normalization, and learned FP32 residual combination. TP reconstructs each expert
output before nonlinear normalization, then applies route weights and accumulation.
Queries split evenly; complete KV heads split or repeat across adjacent TP ranks.
Routers, latent projections, gates and norms have explicit replica accounting.
Vision uses the selected adapter/projection policy. Mono 24 kHz audio uses the
four-stage causal pyramid, 1,920-sample cadence and retained convolution/input carry.

Spark chat uses the published ATEM tools, recipient and termination semantics.
`reasoning_effort` selects low/medium/high/xhigh strengths 8/32/256/512; the default
is xhigh. The template supplies BOS and distinguishes intermediate-message from
end-of-turn tokens. Official `l4_200k_base` reconstructs the tokenizer when
`tokenizer.json` is absent; GGUF uses its admitted tokenizer metadata and BOS policy.
DFlash2's `qwen3` configuration is recognized as an assistant before target-family
discovery. Pure TP DSpark can shard its body with replicated scoring heads;
DFlash2 and other selected placements use explicitly accounted replicas.

Implementation coverage and validation coverage are separate. The concise
[acceptance results](../validation/fixtures/muse-spark/acceptance-results.json)
retain measured results and source-report digests. Detailed tensors, checkpoint
payloads and run traces remain outside the repository.

| Recorded acceptance | Result | Scope |
| --- | --- | --- |
| Complete official NVFP4 target | prefill plus three cached calls; 808,192 scores agree exactly | pinned independent Transformers/ModelOpt reference, same encoded artifact, bounded CPU execution |
| NVFP4 mixed image/audio/video/text | prefill plus three cached calls; score error zero | prepared media tensors and pinned parent reference; excludes host preprocessing parity |
| Released DSpark and DFlash2 | four generated IDs `[131, 50133, 220, 43810]` match ordinary target generation | bounded public facade text generation |
| Released DSpark TP2 controlled | same IDs; two proposals, one accepted | snapshot/restore/fork and semantic text replay |
| Released DFlash2 TP2 controlled | same IDs; three proposals, none accepted | initial and populated-context snapshot/restore/fork and semantic text replay |

The reference pins Transformers `b0cdc61e607f39403e248b11c35aef4edfa7d216`
and ModelOpt `f0daaa241ac6644e9bedd19a1af40c4a2aa8ca04`. Official BF16 is
`0fb02478b02565b2ce7fc57e6051c91172774ac9`; NVFP4 is
`4239e4db19376904f2e82c167e78f70abb5455fe`. The released assistants are
DSpark `6f7aed1cac8d8b1c02a902988876172b39e49fc0` and DFlash2
`46c90f412549c472941a68382d91a99e5e6372a0`. Serving-reference identities and
all community layout revisions remain in the inventory.

The validation machine is an Apple M3 Ultra with 256 GiB RAM. BF16 payloads total
1,644,333,393,864 bytes and are storage-blocked. Verified NVFP4 payloads occupy
533,011,023,472 bytes and exceed resident RAM; completed runs use bounded loading.
Released controlled TP2 runs use target device/host caches of 20/14 GiB per rank,
assistant caches of 8/8 GiB per rank and unused native allocator cache zero.
These parameter budgets exclude native operator workspace and live state.
Forecasting retains the ordinary MLX backend's unknown kernel-workspace and future
allocator-capacity coverage; it supplies no extra Spark-specific finite fit claim.

Full BF16 numerical acceptance, community-layout whole-model reference execution,
native Metal, released distributed media and released coupled native PP/EP
assistant sessions remain unvalidated. Catalog admission is not numerical parity.
These are explicit validation gaps, not architectural restrictions. Shared
execution conformance covers portable partition, state and lifecycle contracts.

Reproduce metadata/payload preparation outside tracked source:

```sh
python3 validation/muse_spark_inventory.py validation/fixtures/muse-spark/inventory.json /tmp/eredu-spark-evidence --repository meta-models/Muse-Spark-Open-816B-A42B-NVFP4 --headers
python3 validation/muse_spark_payload.py --help
```

The payload utility uses the existing Hugging Face credential/profile, verifies
pinned SHA-256 values and retains 64 GiB free disk headroom. It never records a
credential. Fetch the pinned independent sources into external paths, then run:

```sh
python3 validation/muse_spark_reference_run.py "$spark_checkpoint" "$spark_transformers_source" "$spark_modelopt_source" "$spark_reference_output"
cargo run --profile conformance -p eredu-backend-mlx --no-default-features --example muse_spark_reference -- "$spark_checkpoint" "$spark_reference_output" "$spark_native_output" --atol 0 --rtol 0 --device cpu
```

Mixed-media acceptance adds `--multimodal --vision-utils PATH --configuration PATH`
to the Python reference and `--multimodal` to the native comparison. Optional
`--trace-layers` writes detailed diagnostics to the external output directory.
Reference execution requires Python 3.11, PyTorch 2.9.0 and NumPy 2.3.4.

`validation/muse_spark_dspark_released_reference.py` and
`validation/muse_spark_dflash_released_reference.py` run complete released
assistants against their pinned serving sources and retained target captures.
`muse_spark_assistant_reference` compares complete assistant context/body/scoring
outputs with explicit tolerances. The public `eredu` example
`speculative_reference` compares ordinary and assisted generation and accepts
controlled/snapshot and placement options. `validation/muse_spark_facade_text.py`
checks delivered text against the pinned tokenizer bytes. Generated references,
score arrays, debug tests and detailed run records belong outside the repository;
there is no checked-in component numerical-regression fixture suite.

## Checkpoint formats

### SafeTensors directories

Eredu supports single-file and indexed sharded SafeTensors. It validates every
shard header, tensor name, shape, dtype, and required quantization companion
before materialization. Tokenizer and chat-template selection use checkpoint
sidecars. Multimodal models additionally require their processor configuration
and media-tower weights. Indexed tensor mappings must have unique keys; every
shard path must be relative and contained by the checkpoint's canonical access
root, and the index must assign every tensor to the shard header that contains
it without omitting physical shard tensors. The same validation applies to
checkpoint conversion and quantization.

Recognized stored formats include dense F32/F16/BF16, packed affine and MXFP4
layouts, family-specific packed experts, and native dynamic E4M3 block-FP8 for
the supported DeepSeek and Qwen hybrid paths. A stored packed format is kept in
its native representation; it is not silently requantized.

### GGUF

GGUF dispatch uses `general.architecture`, not a filename. Canonical shards are
discovered from any shard path. Projector-based models expect a unique,
family-compatible `mmproj-*.gguf` or other documented companion next to the
language checkpoint unless the architecture-specific API receives the pair
explicitly.

Model execution supports dense F32/F16/BF16 and registered packed paths based
on Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, canonical IQ
encodings, and MXFP4-MoE type 39 where the selected architecture has a matching
kernel and tensor role. Compatibility is checked per tensor; a file-level
recipe name such as `UD-Q2_K_XL` is not itself an encoding.

Unsupported tensor types fail explicitly. Packed weights are not expanded into
a persistent dense copy merely to satisfy loading. See [`eredu-gguf`](../eredu-gguf/)
for the container-level encoding list.

## Tokenizers, chat, and tools

Model directories can use supported tokenizer JSON, SentencePiece, or embedded
GGUF tokenizer metadata. Checkpoint EOS and stop-token metadata are applied to
generation. A present chat template is rendered for structured chat requests;
raw token and text completion APIs remain available separately.

Native tool calling requires independently recognized tool rendering, output
parsing, and constraint capabilities. An ordinary chat template does not imply
tool support. See [Native tool calling](tool-calling.md).

Accordingly, a full checkpoint-support claim covers more than loading tensor
weights: it includes the checkpoint tokenizer and template sidecars, reasoning
controls and history, media placeholders where applicable, and native tool
rendering, constraints, and response parsing. The supported Qwen3.6 and Qwen3.8
checkpoints use the `qwen3_5` / `qwen3_5_text` architecture contract (including
the pinned Qwen3.8-27B configuration fixture). This does not cover
Qwen3.8-Flash-Next, which uses the distinct `qwen4_exp` / `qwen4_exp_text` and
`qwen4exp` contracts. Its pinned template fixtures cover reasoning controls,
tagged parameters, EOS aliases and tool-result replay through ordinary and
controlled drivers. Active tool-grammar snapshots have the typed support limits
in [execution control](execution-control.md). The supported Qwen3.6/3.8 templates
use tagged function parameters.
Qwen3.8 additionally accepts `reasoning_effort` values `low`, `medium`, and
`xhigh`, while Qwen3.6 has no effort selector.

## Execution capabilities

Most registered families support fully resident and bounded weight policies,
load-time quantization of eligible dense parameters, device-resident and paged
request state, and reusable prompt caches. Exact coverage depends on family,
format, quantization, and topology; inspection is authoritative for a concrete
request.

Tensor and pipeline parallelism apply to supported dense families. Registered
MoE families can additionally use expert parallelism and an independent expert
cache. See [Parallel execution](parallel-execution.md) for the topology matrix
and [Model loading, quantization, and memory](model-loading.md) for residency
rules.

## Important boundaries

- Weight budgets exclude activations, request caches, kernels, allocator
  caches, buffered checkpoint shards, and some temporary workspaces.
- On Apple silicon, logical host and device tiers share physical unified memory.
- Full-attention paging reduces logical device residency but still reads all
  retained history and can be I/O intensive.
- GGUF and SafeTensors counters describe Eredu reads and logical residency;
  operating-system page caching means they are not physical disk-I/O meters.
- Realtime Moshi and PersonaPlex session state is not represented by decoder
  prompt-cache persistence.
- A non-replicated topology must use the matching distributed loader rather
  than the complete single-model loader.


## Flash-Next execution

SafeTensors and published GGUF artifacts use architecture-owned target/vision declarations
and exact retained source binding. Vision blocks, experts and table rows are independently
addressable. Ordinary loading supports resident, host-layerwise and disk-streamed weights;
text and image/video construction supports TP, PP, EP and their combinations.

Architecture defaults select one sequence lane, at most 512 tokens per invocation, declared
context capacity, tiled QSA selection and a shared 64 MiB device table cache without a host
table cache. These are execution limits, not eager allocations or physical-fit guarantees.
Generic parameter-bank and state policies remain authoritative.

SafeTensors vision and explicit GGUF projectors support architecture-default media selection.
`MediaLoadRequest::Disabled` selects text-only execution. Prepared/projected media preserve
original token IDs and rotary positions. Raw preparation uses processor JSON or projector
metadata; optional coarse host-processing limits apply before expensive host work.

`prepare_observed_input` accepts a prepared backend prompt plus its explicit original decoder
IDs. Ordinary and controlled generation share the retained request cursor. A controlled step
commits one nonfinal chunk without sampling, and captures retain absolute prompt spans.
Snapshots, restore and fork preserve cursor/model state and immutable encoder products.
Partial-chunk interventions return a typed rejection; single-invocation interventions work.

Explicit embedded drafting selects joint target/prediction execution. GGUF targets require a
matching SafeTensors prediction companion through `ExecutionPlan::with_prediction_source`;
GGUF embedding/output weights remain authoritative. The caller supplies matching training
lineage. MTP owns separate state and streams text/media prefill across invocation boundaries,
using the shared speculative driver's control boundaries.

Memory reports include state history, weights, table caches and operator workspace ranges.
Sparse attention retains full K/V history. Estimates and optional storage diagnostics do not
constitute complete physical-memory accounting. Missing native facts remain unknown.

The [validation report](qwen4-exp-validation.md) specifies numerical fixtures, native
format/topology coverage, pinned reference sources and reproducible commands. Released-payload
parity and throughput, distributed FP8/quantized-table trajectories and physical multi-GPU
execution are unverified.
