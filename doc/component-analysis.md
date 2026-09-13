# Component analysis for LM Inspector

Eredu exposes the complete component-analysis workflow through public portable
APIs: topology, bounded original/effective captures, causal component masks,
effective parameter queries/projections, exact-prefix trials and reversible
coordinated overlays. Acceptance covers ordinary and controlled execution,
applicable residency/parallel modes, supported packed formats and pinned released
models. The [completed checklist](component-analysis-checklist.md) links each
required outcome to implementation and behavioral evidence. Topology metadata
alone is not access to effective loaded parameters; always consult loaded support.

The method motivating these interfaces decomposes a component's write into its
observed signed scalar activation times an output projection column. Keep-only
experiments instead perform new forward computations, with survivors responding
to the modified residual. Calibration, component naming, signed ranking, graph
construction and sufficient-set search belong in consumers. The paper's backward
graph stops at attention channels. See [the paper, sections 2–4, 6 and Appendix
D](https://arxiv.org/pdf/2609.10210); its measured sparsity and editing efficacy are
not implementation acceptance thresholds.

## Topology and actual support

`eredu::api::inspect_architecture` returns an `ArchitectureDescriptor` with
compact `components` groups. `eredu_core::component::ComponentId { group, index }`
identifies one scalar without allocating a descriptor per neuron. Each group
joins an architecture node and parameter group, declares activation paths, read
roles, logical projection rows, a write column, bias terms, normalization and the
activation equation. A gated FFN declares both gate and value reads. Attention
flattens `(query_head, channel)` on the `component` axis; `ComponentRowMapping`
resolves grouped-query sharing and offsets in fused projections. Use `row_range`
for the complete query/key head dependency: one output channel depends on every
query/key row in its head, not just its value-channel coordinate. `row` selects
scalar FFN/value reads and returns `None` for a multirow head dependency.
`ComponentRead::head_normalization` describes Q/K normalization after the affine
read and before positional transformation/scoring, including head width/count,
shared versus independent gains, exact gain identity, groups and epsilon.
An affine Q/K row alone is not the complete normalized attention-read equation.
When `ComponentRead::projection_output` is present, it joins the actual affine
result to any intervening `component_transforms`, including causal convolutions
and shared relative-position projections. Logical repeated
invocations have distinct component IDs and the same `shared_weight` identities.

Architecture descriptor schema 14 carries component schema 13, including
`ComponentRead::input_projections` and explicit shared-state read sources. Check
the descriptor version before interpreting these relationships. An empty chain
reads the source group's normalized input directly; the default source is the
current group.
Otherwise, each stage selects effective matrix rows, adds any declared bias and
applies its declared normalization; its effective output feeds the next stage or
the terminal read. Query effective parameters through the ordinary bounded API.
Normalization factors depend on the current trial, so a chain containing RMSNorm,
L2 normalization or LayerNorm cannot be replaced by a fixed product of matrices. Loaded parameter
input transforms, including dynamic FP8, remain separate numerical facts.
Multiple reads with the same role are distinct required dependencies: V3's key
uses both head-specific latent rows and a direct rotary segment shared by all heads.
Its current normalized KV latent capture is the value inserted into the cache;
it does not recover earlier uncaptured cache positions. `ComponentGroup::write_partition`
distinguishes reduced writes from additive TP terms; capture evidence identifies
the actual assembly arithmetic.

`component_transforms` connects observed boundaries through declared scaling,
normalization and causal depthwise convolution equations. Resolve its parameter
identities through loaded discovery, and follow node ownership to keep prediction
and target invocations separate. These records describe transformations of existing
writes; they are not additional residual terms. Convolution kernels are ordered
oldest to current and include an explicit identity-bypass flag. Current-position
captures do not export historical KV state. Identity normalization uses no
input-dependent denominator; any declared gain and bias remain affine. A readout
can also declare `embedding_normalization`; its captured residual base already
includes that normalization and assembled media, so raw token rows cannot replace it.

Some scalar groups are nested constituents of a complete residual write. For
example, V3's shared experts are children of the sparse FFN node named by an
`other_writes` entry. Follow declared node parents when assembling a score:
include the complete sparse write once, or replace it with a complete expansion
of its branches. Shared-unit contributions can still be inspected and ranked,
but adding them to that complete write would count them twice. The public
reference consumer reports these separately as `nested_components` and
`nested_component_sum`.

`ComponentReadoutEquation::other_writes` identifies whole residual terms such as convolution
outputs and routed-expert sums, with their owning nodes and original/effective
observations. Include them in score reconstruction alongside embedding, scalar
component and bias terms; they do not imply a scalar decomposition of those operators.

Declarations are cold architecture facts. Join their paths to
`LoadedModel::capture_discovery()` and `intervention_discovery()` to check actual
selected support. Missing component groups mean not described, never no components.
`ComponentRowMapping::HeadRows` returns the complete query/key head needed for
attention scores. Its optional `read_head_stride` supports fused interleaved head
segments; `Blocked` maps individual channels within those segments. Use
`row_range` and validate component membership before issuing a parameter query.
Do not infer a query row from the scalar output-channel index alone.

Metadata refers to canonical effective affine geometry, not packed quantization
storage. It does not authorize editing raw checkpoint tensors.

`ComponentGroup::write_output` identifies the complete affine write after any
partition reduction. `output` identifies the contribution after output normalization,
before addition to the residual or an enclosing operator sum. Both have original
and `.effective` observations. Use these declared paths to reconstruct and compare
writes without deriving suffixes from component IDs. Older descriptors can omit
these optional relationships. Architecture-owned partition placement treats the
complete hidden-axis writes as replicas on the ranks executing that invocation;
component scalars retain their own sharded coordinates.

Some attention value reads are mixtures of activated expert projections.
`ComponentGroup::routed_reads` describes these separately from `reads`. Each
`ComponentRoutedRead` joins the routing node and selected-coefficient observations,
the expert bank, channel-to-row mapping, activation, selector weight and optional
selection-only correction bias. Its `read_weight(group, component, expert, loaded)`
and `read_bias(...)` methods produce checked effective parameter regions for the
existing bounded query/projection and overlay APIs. They do not grant execution
authority. The optional field defaults to an empty list when reading older
serialized descriptors.

K2 Horizon's MoVA value read is a selected mixture of SiLU-activated expert rows,
followed by the ordinary attention aggregation and output gate. It has no single
ordinary V matrix. Inspector can query or edit an expert row using the declared
mapping and account for current routing through the linked observations; it must
not replace that input-dependent equation with a fixed affine row. K2 shared FFN
units use their own `model.layers.N.shared.feed_forward.*` boundaries. Their
normalized input and write are separate from the complete routed-plus-shared FFN
contribution. Choose either the complete contribution or its constituent writes
when reconstructing the residual, so shared values are not counted twice.

## Timing and masks

Shared dense decoder boundaries use these suffixes below the logical unit path
(for example `model.layers.0`):

| Suffix | Value |
| --- | --- |
| `attention.input` | Actual normalized attention input |
| `attention.channels` | Aggregated channels after applicable gating, before output projection |
| `attention.write_input` | Read-only actual multiplication input after selected input arithmetic |
| `attention.write` | Affine attention projection, including output bias |
| `attention.output` | Write after optional output normalization |
| `attention.residual` | Residual after the attention addition |
| `feed_forward.input` | Actual normalized FFN input |
| `feed_forward.units` | Activated/gated scalar units before down-projection |
| `feed_forward.write_input` | Read-only actual multiplication input after selected input arithmetic |
| `feed_forward.write` | Affine down-projection, including output bias |
| `feed_forward.output` | Write after optional output normalization |
| `feed_forward.residual` | Residual after the FFN addition, before block normalization |

The `write_input` points are read-only and have no intervention or `.effective`
sibling. Join them through `ComponentGroup::write_input`; they can differ from
post-intervention components when the selected projection quantizes its input.

Each intervenable original path retains `BeforeIntervention` meaning. Its `.effective` sibling
has `AfterIntervention` meaning and is read-only: it reports the value consumed
by the next operation, after all interventions at the original path. An ordinary
capture downstream can already reflect earlier interventions. No baseline pass
is synthesized. Absent observers do no additional tensor evaluation, retention or
materialization.

`InterventionAction::MaskComponents { dtype, indices, keep_selected }` operates
on the complete final `component` axis. IDs are unique and in range. An empty
keep-only list zeros all components; an empty deletion list changes none. The
list applies over the explicitly selected batch/sequence region; use a `sequence`
slice for a particular prompt position. Component-axis slicing is rejected to
keep IDs unambiguous. Existing Zero/Scale/Mask/Replace/Add operations also work at
these boundaries. Keep-only preserves the current forward values; it never copies
baseline activations into surviving slots.

Prediction zero is prefill. Its prediction index does not select a prompt row.
The last row of a fixed token-ID prefix scores the next token. Prepare captures
when creating the run. Start a trial from a snapshot before the tested forward,
use intervention re-admission for future changes, and let the ordinary sampler
choose the result. For historical predictions without captures, replay the exact
prefix IDs; summaries cannot recover old tensors. Use `prepare_observed_token_ids` or `prepare_intervened_token_ids` with the
original `PreparedChat` output/termination contract and the exact prefix IDs.
These methods validate the tokenizer vocabulary and share ordinary admission and
generation; they never decode/re-encode the prefix or force a tested prediction.
The runnable seven-step example below uses this exact replay path.

Embedded prediction components use `prepare_speculative_activations` and
`ControlledSpeculativeOptions::activations`, with declared context, proposal and
prediction-depth scopes. Tentative proposal records are separate from committed
output. Speculative snapshots require a drained canonical boundary after prefill;
initial backend-owned prefill inputs do not have the complete portable copy
contract needed for an earlier snapshot. To test that first computation, prepare
a fresh run with the same exact prefix IDs and the desired activation plan.
Check snapshot support at the actual boundary, and preserve cumulative capture,
copy and transport budgets across later restore/fork trials. See
[execution control](execution-control.md) for the typed capability and lifecycle
rules.

## Score reductions

`ArchitectureDescriptor::component_readout` joins the final normalization and
effective output weight to `readout.embedding`, `readout.residual`,
`readout.normalized` and `readout.linear`. Each has an `.effective` companion.
These are values on the production path, and interventions affect the next
operation. The embedding observation is already scaled. `readout.linear` is the
affine vocabulary score; an optional declared softcap follows it. Final gains,
gain offsets, epsilon, groups, biases, embedding ties and block residual
normalizations are explicit. A missing declaration is not an identity equation.

For LayerNorm, center the gain-weighted readout covector: `P(gain * row)`, not
`gain * P(row)`. Compute the denominator from the same trial's measured residual.
Normalization biases and output biases contribute once per score. Output
softcaps apply to the reconstructed score as a whole. The evaluation helper
`MeasuredReadout` implements this host reference arithmetic with compensated
signed summation; it is not inference or attribution policy in the backend.

`CaptureTransform::TokenScores { token_ids }` at unsliced `model.logits` returns
up to 64 selected raw scores, their strongest different-token competitors,
competition ranks and full-model-vocabulary log probabilities for the final row.
It performs no sampling or forcing. The distribution includes every model output
ID; tokenizer and grammar membership are reported separately. Nonlinear output
transforms already applied by the architecture are included in these scores.
Do not infer an additive pre-transform decomposition from this score stage.
The native primitive uses bounded scalar host reductions; the validation probe
also captures full logits solely to independently check those reductions.

## Accounting and failures

Require `CapturedStep.outcome == CaptureStepOutcome::Committed` when using a batch
as evidence for a committed model prediction. An aborted batch can retain valid
earlier measurements, while `Untracked` supplies no commit evidence. Neither the
batch outcome nor its error establishes native completion or safe session reuse.

Plans permit at most 1024 operations, 32 MiB of unencoded payload and 64 MiB of
compact JSON. One attention and one FFN mask per layer fits 128 layers with
8192 attention channels and 32768 FFN units even when every component is listed:
256 operations and 20 MiB of u32 IDs. A smaller keep-only set costs four bytes
per retained ID. An all-keep trial can also use an empty deletion list.
The backend reserves the native and host mask from the known width before
construction. These wire limits do not replace the separate cumulative capture,
native-work and transport budgets; all-layer experiments still need adequate
execution reservations.

`InterventionEstimator::activation_usage` supplies an explicit logical allowance
for retained source/backing storage, selected values, replacement/update arrays,
mask temporaries and host uploads. The runtime checks it at admission and reserves
it against the shared capture ledger before native edits or evidence copies.
MLX reports conservative array allowances; these are not physical allocator or
private-kernel-workspace guarantees. Evidence transforms and encoded records keep
their separate charges. Failed or replayed work does not refund cumulative charges.

## Loaded parameter queries and overlays

The neutral plans, regions, projections and errors live in
`eredu_core::parameters`; `eredu::api::LoadedModel` exposes the operations.
For a backend implementing `ParameterBackend`, call `parameter_discovery` on the
loaded model. Each slot supplies canonical and shared IDs, effective dtype, logical shape
and implementation status. `query_parameter(identity, name, region, limits)`
returns only the selected effective entries as F32 host values. A row of an
`[output,input]` matrix uses `starts=[row,0], shape=[1,input]`; a write column uses
`starts=[0,column], shape=[output,1]`. Normalization vectors use rank-one regions.
Use the declared normalization `gain_offset` with the queried gain. For example,
Gemma 2 SafeTensors stores a gain offset from one, while its published GGUF stores
the complete gain. Their effective normalization agrees, but the corresponding
query values and offset declarations differ. Do not add one based on a family
name or assume equal serialized discovery equations across those encodings.

The discovery identity changes on every successful parameter publication, including
restoration, so an old query or edit authority cannot silently target new weights.

`project_parameter(identity, name, ParameterProjection, limits)` contracts a
logical region along one axis against finite F32 host directions. Coefficients
have shape `[directions, contracted_width]`; output keeps the other axes in their
original order and appends the direction axis. For a write matrix
`[residual_width, components]`, contract axis zero with a selected-token readout
direction, a difference of two token directions, or a downstream read covector.
Only the reduced output leaves the device. Native F32 matrix multiplication
introduces ordinary reduction rounding; this is not an exact F64 dot product.
Reservations include the source, selected F32 weights, direction upload,
temporary arrays, output and encoded host records. Active overlays are visible
to both rectangular queries and these projections.

Construct `ParameterOverlayPlan` with the exact base identity, provenance,
parameter shapes/dtypes and finite Replace/Add payloads. Admit with
`admit_parameter_overlay`; activate the resulting immutable authority with
`activate_parameter_overlay`. All edits are validated before native work.
Overlapping regions after alias resolution are rejected, including a row/column
intersection through different tied names. Disjoint operations are applied in
plan order. Explicitly remove the current overlay before installing another.

The MLX mechanism copies each affected parameter. Dense updates use F32
arithmetic followed by one rounding to the existing floating dtype. Packed affine,
MXFP4, block-FP8 and native GGUF matrices or grouped banks use their loaded format and declared
companions to obtain effective F32 values. Editing a packed matrix publishes a
complete F32 copy of that affected matrix or bank, with no requantization; other parameters
retain their original encoding. This choice is reported by loaded discovery and
requires reserving the full conversion/copy cost before work. It can change
matmul reduction rounding compared with the original packed kernel, even for a
numerically zero edit. For block-FP8 on both CPU and GPU, the affected projection
also switches to dense input arithmetic, removing its dynamic activation
quantization. That numerical change applies to the complete affected projection.
Public native CPU tests verify this zero-edit transition against an independently
decoded F32 reference in resident, host-layerwise and disk-streamed execution.
F32 replacements can also promote BF16/F16 activations, later projections and
mutable state to F32. Unchanged parameter dtypes do not imply unchanged activation
precision. `CaptureRecord.source_dtype` reports the actual source dtype before
host conversion, so consumers can select the exact dtype for a future mask or
replacement at that boundary. Its absence means unknown; it does not mean F32.
Captured floating payloads still use their existing host representation. Recheck
this evidence after changing parameter versions, and account for reduced-precision
rounding when choosing reconstruction tolerances.

While an overlay is active, MLX runtime-state estimation conservatively uses the
larger of the original state width and the edited parameters' effective width.
A packed-to-F32 edit therefore reserves four-byte floating state, including for a
head-only edit that may leave earlier state at BF16. Removal restores the original
estimator assumption after incompatible state is cleared. This is an upper bound;
native snapshot copy accounting still measures the actual retained tensors.

Non-finite candidates are rejected and every candidate is
evaluated before any slot is published.
Original tensors remain retained for `remove_parameter_overlay`. Both publication
and removal clear KV/recurrent state, invalidate native snapshots and old prepared
facade requests, and require a new exact-token prefill. Existing ordinary and
controlled drivers then consume the edited operators on new inputs. Records carry
`parameter_overlay_id`. Persistent prompt-cache operations reject active overlays
because that cache format does not yet encode their identity.

Query/edit `CaptureUsage` limits are absolute cumulative limits for the loaded
session. Reservation precedes native work and host transfers. Reset, removal and
failed work do not refund those reservations. Edit reservations include originals,
affected-parameter copies, conversions, every slice-update destination and uploads.
Distributed queries also charge catalog preparation, exchange and reconstruction
on the selected topology. Repeated row/column experiments must budget those
charges for every query, including queries after restoration. The usage returned
by consecutive `parameter_discovery` calls exposes the catalog charge; it does
not include the additional parameter conversion and result costs of a query.
These cumulative totals can exceed peak host memory by a large factor.
Queries/projections currently reserve full affected-matrix decoding before slicing;
only the bounded selected result is delivered to the application. Native GGUF decoding uses an
explicitly charged host conversion; affine/MXFP4/FP8 conversion stays native.
FP8 reservations include expansion of scales to complete 128-by-128 blocks,
even when the selected logical matrix is smaller.
The limits describe logical owned arrays and transfers, not native allocator
workspace. Affine public lifecycle verification and the CPU packed-format mechanism checks
passed. Streamed parameter operations use retained recipe metadata and selected
unit loans, independently of forward order. The ordinary residency budget covers
each materialized unit; parameter-work reservations cover retained originals,
replacements and operation outputs. Completed overlay copies are applied again on
every affected unit reload. Dense/affine queries and overlays passed on host-layerwise
and disk-streamed execution, including tied embeddings and repeated invocations.
Parallel queries and projections now use the same public APIs with collective
participation by all ranks. Distributed overlays, cached prediction rollback and
active-overlay controlled capture pass the 24 CPU Ring configurations recorded in
[parallel validation](component-validation.md#effective-parameter-coordinates-for-parallel-execution).
Overlays are installed on an idle `LoadedModel`. A controlled run and all of its
snapshot branches use that immutable parameter version. Installation/removal
invalidates old prepared runs and snapshots, requiring fresh exact-token prefill.
Use independently loaded model owners for trials with different weight versions;
activation authority cannot cross model owners, even for the same artifact.

Qwen hybrid recurrent channels are exposed at `model.layers.N.mixer.channels`,
after recurrent aggregation, RMS normalization and SiLU gating. The descriptor
joins fused Q/K/V reads through their causal convolution and declares the exact
L2 normalization and key-to-value head repetition. Kimi KDA uses its separate
RMS/scaling and key-channel decay declarations at `.attention.channels`.
A channel mask does not rewrite that operator's already updated recurrent state.
It can change later layers' cache entries and future predictions because those
layers consume the modified residual. Earlier token positions remain unchanged.
A mask on the normalized operator input occurs before its own state update.

## Execution coverage

The [implementation checklist](component-analysis-checklist.md) records the
completed requirement audit. [Validation evidence](component-validation.md) preserves exact commands,
fixture revisions, numerical tolerances and the history of individual matrices.
A declaration does not establish support for a loaded execution; use its capture,
intervention and effective-parameter capability reports before admission.

| Execution family | Verified component and parameter workflow |
| --- | --- |
| Shared dense decoders | Gated and unary FFNs, attention channels, normalization/bias/tied-weight equations; native resident/host/disk capture, masks, queries, coordinated edits and controlled replay |
| Qwen, GPT-OSS, Nanbeige, LFM2 and K2 targets | Native CPU Ring coverage includes TP, PP and EP, ordinary and independently cached expert banks; exact encoding/geometry coverage is recorded in validation |
| Qwen Next/3.5 recurrent channels | Neutral exact recurrence and prepared TP/PP/EP placement, including 102 tied/untied conditional placements; public CPU/Metal and all 90 native Ring placements cover masks, capture, coordinated recurrent overlays and controlled replay |
| Qwen Next/3.5 prediction | All 90 F32 text/conditional cases and all 21 conditional MoE FP8 cases pass, including DeepStack, effective queries, edits and controlled replay. All 90 F32 and 21 FP8 placements additionally pass actual extension paging and idle-loan checks |
| Nemotron-H target | Attention, dense/shared/routed ReLU² and complete Mamba/sparse writes; SafeTensors/GGUF CPU Ring capture/query/edit/replay |
| Nemotron-H prediction | All 30 cases each for F32 and MXFP4 pass across dense/routed targets, applicable TP/PP/EP and resident/host/disk placements. All 30 cases in both formats additionally pass actual extension paging and idle-loan checks |
| DeepSeek V3 target and prediction | Dense/sparse and compressed-attention hooks; F32/affine/GGUF target coverage; F32/affine/MXFP4 prediction matrices and coordinated target/prediction edits. All 21 placements each for F32, affine and MXFP4 additionally pass actual prediction paging; published FP8 passes all 42 ordinary and independent-bank Metal placements |
| DeepSeek V4 target, sequential prediction and DSpark | Public CPU/Metal component equations and edits; F32/affine/MXFP4 Ring matrices, phase-specific masks and controlled replay; all 21 placements per format and prediction mode additionally pass actual extension paging, with MXFP4 on GPU compute and local CPU Ring. Mixed encodings pass all 84 ordinary/independent-bank placements; sequential and fused pure FP8 each pass all 84 F32/UE8M0-scale placements |
| Inkling | Neutral target/prediction equations; public CPU/Metal prediction capture and masks; all 30 ordinary target and prediction CPU Ring cases each for F32, affine and MXFP4; effective queries, coordinated edits, controlled replay and scalar-verified paged relative attention. Actual prediction paging passes all 30 placements each for F32, affine and MXFP4; F32/GGUF/affine independent banks pass 21 each |
| Muse-Glimmer | Neutral dense/sparse equations and 102 tied/untied media placements; public CPU/Metal masks and coordinated edits; 30 native cases each for F32, affine, MXFP4 and F32 GGUF, plus 21 independent-bank cases each for F32, affine and F32 GGUF. The additional published-geometry Q8_0/IQ4_NL projector passes all 30 ordinary and 21 independent-bank native placements, including eight-rank execution |
| Qwen3-VL | Shared decoder hooks, DeepStack terms, 102 tied/untied neutral placements, public CPU/Metal; native F32/affine/MXFP4/GGUF matrices (30 each), F32/affine/GGUF independent banks (21 each), and mixed Q8_0/IQ4_NL primary/projector coverage (30 ordinary, 21 independent) |
| Gemma 4 | Shared component hooks and topology, publisher-aware KV reads, branch norms and whole-residual scaling; 90 neutral placements; public CPU/Metal; all 30 SafeTensors, all 30 GGUF, all 30 affine and all 30 MXFP4 native placements, plus 21 each for GGUF, full-precision SafeTensors and affine independent banks, with coordinated edits and controlled replay |
| Kimi Linear | Neutral KDA/MLA, dense/shared/routed hooks, recurrent and latent read declarations, cached masks, residual/score reconstruction and observed PP/TP parity; all 21 native placements each for F32, affine, MXFP4 and F32 GGUF, plus 21 independent-bank cases each for F32, affine and F32 GGUF; public CPU/Metal resident/host/disk KDA/MLA masks, coordinated overlays and controlled replay |

This matrix covers autoregressive text tasks, including prepared media inputs.
Moshi/PersonaPlex selects `LoadingProtocol::Realtime`: delayed codebook frames
and feedback state are required inputs, so a text-token prefix does not specify
the equivalent experiment. Its temporal/depth discovery remains partial, its
prefill/decode observation catalog does not advertise realtime frame identities,
and text construction returns `ReplicatedTextIneligibility::Realtime`. A realtime
component-analysis workflow requires the separate frame protocol.

CPU Ring results establish native collective behavior on the tested local CPU
configuration. Separate distributed GPU suites now cover shared, recurrent,
quantized, prediction and media paths with local Ring transport; all ranks use
one physical Metal device. Multiple physical GPUs and hosts remain hardware
validation gaps. Exact device-specific cases are recorded in the evidence.

## Numerical conventions for packed parameters

FP8 validation now includes native CPU models with inexact activations, several
feature blocks and a partial final-head input block. Both CPU and GPU block-FP8 projections dynamically quantize their input activations;
the CPU fallback also dequantizes the weights. Effective weight columns alone
therefore do not establish an exact additive decomposition using pre-quantization
captures. `LoadedParameter::input_transform` reports `Identity`,
`BlockFp8E4m3 { block_width, floor, magnitude }`, or `Unspecified`. The latter is
also the default for older serialized discovery; it never implies identity.
Current MLX FP8 uses feature blocks of 128, `max(amax, 1e-4) / 448` scales and
nearest E4M3 rounding with ties to even. An active F32 overlay reports `Identity`
for its affected matrices; restoration returns the original selected fact.
`ComponentGroup::write_input` supplies actual multiplication values for use with
effective columns. `ComponentReadoutEquation::projection_input` identifies the final
head's actual input. Its difference from `readout.normalized.effective` gives a
separate measured head correction `row · (actual - normalized)`; do not assign
this nonlinear correction to individual residual components. The example reports
it separately for selected scores and score differences. Raw read/write dot
products remain geometric affinities, not a complete quantized read equation.
The public CPU tests verify this reconstruction for selected scores and margins,
including masked trials and zero/nonzero overlays, across resident/host/disk.
Independent CPU/Metal mechanism comparisons also pass with inexact inputs,
partial matrices, amax-floor inputs and zero-edit/restoration transitions.
The same public session matrix passes on Metal across all three residency modes (22.64s). Native GGUF Metal
kernels, including big-endian IQ block fields, pass the device tests in the
[validation record](component-validation.md).

## Inspector integration and reproduction

Start in Inspector's `crates/engine/src/worker/controlled.rs`, preserving its
prepared-boundary snapshots and controlled driver:

Inspector's `crates/engine/src/inspect.rs` already serializes the complete borrowed
Eredu architecture descriptor. Its `src/lib/types.ts` currently types only the
graph and observation fields. Extend that boundary with `components`,
`routed_components`, `component_readout`, `component_scopes` and
`component_transforms`, retaining the descriptor version. Schema 11/component
schema 10 includes `gated_delta_attention`; consumers must preserve its recurrent
equation and channel normalization instead of interpreting it as softmax
attention. The architecture declaration still requires a separate loaded-support
check before an experiment.

1. Discover stable component identities and actual loaded support. Add bounded
   capture selections when creating the run; retain exact prefix token IDs.
2. Select prediction indices and within-tensor token positions independently.
   Snapshot before the forward computation being tested. Re-admit prospective
   masks on branches, and recompute surviving components on each trial.
3. Join effective component values to effective write columns and the declared
   normalization/readout equation. Include embedding, bias, whole writes and
   projection-input corrections. Preserve typed absent or failed outcomes.
4. Use bounded parameter queries/projections for reader/writer analysis. Install
   coordinated overlays on an idle loaded model, then prepare a fresh exact-ID
   run. Compare trigger, non-trigger and held-out prefixes; remove the overlay
   and repeat baseline verification.

`component_reference_probe` is the runnable public seven-step consumer. Its
helpers perform the research calculations in the example; the library supplies
portable observations, admitted operations and lifecycle guarantees. After
creating the pinned reference fixture with the commands in
[released dense validation](component-validation.md#validation-in-progress), run:

```sh
cargo run -p eredu --no-default-features --features mlx \
  --example component_reference_probe --offline -- \
  /tmp/eredu-component-validation/transformers-reference.json
```

The fixture checks prefill and three cached decode steps against an independent
Transformers reference, signed score and score-difference reconstruction, deletion,
keep-only, backward queries, a three-prefix association and coordinated
attention-plus-FFN edits. The reported editing effect is illustrative; it does
not establish the paper's measured editing efficacy. See also the
[released sparse workflow](component-validation.md#released-sparse-checkpoint-validation).

Muse's readout declares `scaled_softcap`: first reconstruct the unscaled affine
score, then apply `cap * tanh(score * scale / cap)` to the complete score.
Do not move the multiplier into the head input; quantized input transforms
can make that a different computation. `token_embedding_normalization` applies
to text lookup rows before media assembly, while the observed `embedding` is
the completed assembly. Projected-head `output_scale` defaults to one and is
separate from a learned normalization gain.

For Qwen3-VL, the assembled embedding already contains the primary vision
embedding. Its later DeepStack additions are separate residual terms in the
descriptor. Include each declared effective addition once when reconstructing
the score. Their observations require media input and are advertised for
prefill; ordinary transport may retain zero placeholders on text-only steps.
Read the descriptor's paths and terms rather than assuming the text decoder
uses a particular checkpoint prefix.

Gemma4 component declarations distinguish the current attention query from
shared key/value state. Since component schema 12 (architecture descriptor 13),
`ComponentRead::source = PublishedAttentionState { component_group }` points to
the group that actually projects and publishes those keys or values. Resolve
its input and normalization through that publisher group. The effective read
parameters still belong to the publisher. The consuming block has no local
K/V projection. Capturing a current publisher invocation does not recover
historical cached inputs. When keys are reused as values, the value read names
the raw key projection, followed by weightless per-head RMS normalization;
it does not inherit the key's learned normalization or rotary transform.

`ComponentReadoutEquation::block_transforms` references exact transformations
in `ArchitectureDescriptor::component_transforms`. Apply them in listed order
after that layer's writes and `block_normalizations`, before later layers.
Gemma4 uses these references for its learned scalar on the entire residual:
`r_l = s_l * (r_(l-1) + attention_l + ffn_l + per_layer_l)`. Therefore each
scalar changes the embedding contribution and all preceding writes. Applying
it only to the current layer's components gives an incorrect decomposition.
The declaration includes the actual input/output observations and the effective
`[1]` parameter identity for ordinary bounded queries and admitted overlays.
Account for intervention deltas at those boundaries and normal floating-point
rounding when comparing a redistributed sum with the executed residual.

Component schema 13 (architecture descriptor 14) also exposes optional routed
bank `write_output` and `output` paths. The former is the complete route-weighted
write after tensor/expert reduction; the latter includes the bank's declared
output transforms. Neither includes other banks or shared branches. Follow
`component_transforms` between those paths, including the consumed `.effective`
write, before adding the bank to a common residual term.

The dense Gemma4 branch uses exact GELU; the packed routed bank uses its declared
approximate GELU policy. In a sparse block, independently normalized dense and
routed branches form one sum, which receives another normalization before
residual addition. The dense component group is a child of that whole FFN write,
so it must not also be added as an independent residual term. Per-layer inputs
produce an additional complete residual write, including during text-only
execution when the configuration declares those inputs. These declarations
and shared execution hooks pass the neutral and native Gemma matrices listed
above. The requirement audit and exact coverage are recorded in the validation
checklist.

### Distributed caller-owned preparation

Run entry points coordinate ordinary and speculative host/prompt preparation.
Speculative scheduler capacity and completion policy are checked during host
readiness, before native prompt construction. If Inspector prepares media or an
opaque prompt itself, carry the result through an explicit agreement before
constructing the generation request:

```rust,ignore
use eredu::api::{MultimodalPreparationError, PreparedChatInput};
use eredu_core::run_preparation::TextPreparationStage;

let local = model.prepare_chat_multimodal_input(&chat, &bindings);
let prompt = model.finish_text_preparation(
    TextPreparationStage::Prompt,
    local,
    MultimodalPreparationError::Backend,
)?;
let input = PreparedChatInput::prepared_backend_input(&chat, prompt);
```

Every participant must enter the same stages in the same order, including failed
preparation. The cancellable helper accepts `Ok(None)` and reports coordinated
cancellation without treating it as a measured result. Errors from local
inspection or immutable admission also remain local until explicitly agreed.
Preparation usage is cumulative and is not refunded by trial restore. An agreed
rejection preserves error provenance; it does not establish native completion or
remove a session fence.
