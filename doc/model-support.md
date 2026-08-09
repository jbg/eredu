# Model and checkpoint support

This page summarizes the high-level support implemented by `safemlx-lm`. For
API details and the full layerwise-residency matrix, see the
[`safemlx-lm` README](../safemlx-lm/README.md).

Support is determined from checkpoint metadata and validated configuration,
not from a model's display name. Applications can inspect either artifact
format before loading with `inspect_model`. It is the canonical preflight API:
config normalization and architecture validation are internal parts of the
same inspection path used by loading.

```rust,no_run
use safemlx_lm::{inspect_model, InspectionSeverity, ModelInspectionOptions};

let report = inspect_model("model-or-checkpoint", ModelInspectionOptions::default())?;
if !report.is_loadable() {
    for issue in report
        .issues
        .iter()
        .filter(|issue| issue.severity == InspectionSeverity::Error)
    {
        eprintln!("{:?}: {}", issue.code, issue.detail);
    }
    // Route to a fallback engine or return the structured diagnostics.
}
# Ok::<(), safemlx_lm::error::Error>(())
```

Inspection parses configuration, every SafeTensors index and shard header, or
every GGUF shard header and tensor catalog. It validates the same architecture,
quantization, parallel-topology, and residency preflight used by the loader;
reconstructs embedded or sidecar tokenizers; locates chat templates and media
companions; and can behaviorally probe semantic/tool protocols without loading
weights or creating an MLX stream. Supplying a concrete `chat_request` in
`ModelInspectionOptions` validates that request. Without one, native-tool
readiness is based on a bounded synthetic behavior probe and the report marks
real schemas, kwargs, and policies as request-specific.

`is_loadable()` is an authoritative, fail-closed admission result. Container
validity, architecture support, requested load-policy compatibility, and exact
structural binding are reported separately. DeepSeek-V3/R1, Gemma 4 text and multimodal,
GPT-OSS, Inkling, Kimi Linear, Llama/Mistral, Nemotron-H, Qwen2/Qwen2.5 text, Qwen3/Qwen3-MoE,
Qwen3-Next, Qwen3-VL/Qwen3-VL-MoE, Qwen3.5 text and multimodal, PersonaPlex,
and LFM2/LFM2-MoE
SafeTensors, plus DeepSeek2,
Gemma 4, GPT-OSS, Inkling, Llama/Mistral, Qwen2/Qwen2.5 text, Qwen3/Qwen3-MoE,
Qwen3.5/Qwen3.5-MoE, Qwen3-Next, Qwen3-VL, Kimi Linear,
Nemotron-H/Nemotron-H-MoE, and LFM2/LFM2-MoE GGUF
have exact header-only tensor-name, shape, dtype/encoding, tied-head,
hybrid-layer, and quantization-companion validation. Qwen3-MoE validation follows the selected
residency route: fully resident loading requires packed expert banks, while
bounded loading also accepts split or separately packed expert projections.
LFM2-MoE accepts the official split-expert layout on its resident route because
the loader packs it through the same shared plan. Checkpoint-native affine
Qwen3-MoE and LFM2-MoE catalogs validate packed weights and their scale/bias
companions; floating source catalogs requested for on-load quantization use the
same per-operation plan. GPT-OSS additionally validates its native
SafeTensors MXFP4 block/scale/bias catalog and requires canonical type-39 expert
matrices in GGUF. Kimi Linear validates both official split and packed
SafeTensors expert catalogs, its KDA reshape geometry, and its MLA projections.
DeepSeek-V3 validation covers the dense-to-MoE schedule, MLA query variants,
shared experts, official split routed experts, and packed banks on bounded
routes. Native E4M3 block-FP8 weights and inverse-scale companions and native
affine weights/scales/biases are validated per operation. DeepSeek2 GGUF validation shares the loader's parsed geometry
and validates fused or split MLA projections, dense-to-MoE routing, shared and
packed routed experts, and operation-specific GGML encodings.
SafeMLX-written affine checkpoints may use the quantized module's
`.inner.weight` serialization alias, and native FP8 checkpoints may store E4M3
bits either with the SafeTensors E4M3 dtype or the loader's raw-byte `U8`
representation; inspection accepts both exact loader-supported forms while
still requiring the matching scale/bias or inverse-scale companions.
Inkling SafeTensors validation covers the released interleaved dense and expert
weights, canonical loader aliases, local/global attention geometry, short
convolutions, and configured dMel audio and hMLP vision towers.
Inkling GGUF validation shares its loader's pure metadata parser, name
translation, and dense-to-MoE plan. It validates local/global relative
attention, accepted short-convolution layouts, routed and shared expert banks,
paired expert encodings, and the complete optional dMel/hMLP projector catalog.
An absent projector remains text-loadable, while an invalid projector that the
loader would automatically consume blocks admission.
Nemotron-H SafeTensors validation covers its Mamba/attention/dense/MoE pattern,
released and canonical names, both public split and packed ReLU2 experts, and
checkpoint-native affine matrix companions while keeping recurrent state,
convolution, norm, and router operations dense.
Nemotron-H and Nemotron-H-MoE GGUF validation uses the loader's stream-free
metadata parser and name translation. It validates the per-layer
Mamba/attention/dense-or-MoE pattern, recurrent transform tensors, shared and
packed ReLU2 experts, operation-specific GGML encodings, and requested dense or
sparse residency route before payload materialization.
Gemma 4 validation covers released and canonical prefixes, shared-KV layer
omissions, per-layer inputs, dense and SwitchGLU MoE blocks, tied heads, and
route-specific fused expert banks. Multimodal validation additionally covers
every vision and audio tower tensor, clipped-linear bounds, convolutional
geometry, optional standardization, and the modality projections; only the
text-facing modality projections follow checkpoint affine metadata.
Gemma 4 GGUF validation shares the loader's stream-free metadata parser and
name translation. It validates per-layer feed-forward and attention geometry,
sliding/full attention head dimensions, shared-KV omissions, dense and
SwitchGLU MoE catalogs, fused or split expert banks, and operation-specific
GGML encodings before payload materialization.
Qwen3-Next SafeTensors validation covers fused or split Gated DeltaNet input
projections, full-attention layers, shared and routed experts, embedded MTP
layers, checkpoint aliases, and packed or public split expert layouts. Native
FP8 inverse-scale and affine scale/bias catalogs use the same fused-projection
and expert transforms as the loader, including the dense FP8 BA exception.
Qwen3-VL and Qwen3-VL-MoE SafeTensors validation combines the prefixed Qwen3
text plan with the complete vision tower. It validates Conv3D patch and
position embeddings, every vision block, merger and DeepStack parameters,
text/vision geometry, supported vision dtypes, tied or untied text heads, and
route-specific packed or split MoE experts. Checkpoint-native affine text and
expert catalogs are validated without quantizing the vision tower.
Qwen3.5 SafeTensors validation covers dense and MoE hybrid decoders,
split Gated DeltaNet projections, full-attention and MTP layers, public
checkpoint aliases, and packed or split routed experts across resident,
layerwise, and sparse-cache routes. It also validates every configured vision
tower tensor and accepted prefix/merger alias, plus native FP8 inverse scales
and affine companions per text operation.
Qwen3.5 and Qwen3.5-MoE GGUF validation shares the loader's pure metadata
geometry and name-translation plan. It validates split recurrent projections,
full-attention layers, dense or routed/shared experts, paired expert encodings,
operation-specific dense recurrent state tensors, and grouped value-head
quantization alignment. An optional sibling `clip`/`qwen3vl_merger` projector
is validated as a second artifact: exact geometry, complete tensor catalog,
decoder output width, absence of unsupported DeepStack outputs, and dense or
canonical GGUF Q8 projection alignment is checked before either payload is
read.
Qwen3-Next GGUF validation uses the same metadata and decoder plan while
requiring its fused QKVZ/BA recurrent projection catalog and validating affine
input-group alignment before the loader splits those tensors.
Qwen3-VL GGUF validation combines the shared dense or MoE Qwen3 text plan with
its required sibling projector catalog. It validates multimodal RoPE sections
and placeholder tokens, vision-block geometry, both physical patch-convolution
halves, the merger and DeepStack tensors, projector/text hidden-size agreement,
paired packed expert encodings for `qwen3vlmoe`, and dense or canonical GGUF Q8
projector operation encodings. Inspection and the resident, bounded, and
expert-parallel loaders consume the same stream-free argument and structural
plans.
Kimi Linear GGUF validates pure metadata geometry, translated names, KDA/MLA
catalogs, convolution and transition tensor element counts, paired expert
encodings, and per-operation GGML support. The loader's requirement that KDA
transition payload values be negative remains a post-inspection payload-value
failure class.
PersonaPlex 7B-v1 SafeTensors validation covers the released PyTorch catalog:
temporal and depth transformers, packed per-codebook depth attention, norm
reshape aliases, embeddings and projections, and checkpoint-native affine
companions. The resident, quantize-on-load, and bounded realtime loaders call
the same pure catalog validator before model creation. Generic language-model
inspection still reports its requested load route as unsupported because
PersonaPlex must use the realtime API, so `is_loadable()` remains false even
when its separate structural binding is `Ready`. Every supported SafeTensors
`ModelKind` and GGUF architecture has an exact structural policy.

## SafeTensors model directories

The standard loader accepts Hugging Face-style directories containing
`config.json`, tokenizer files, and either a single or sharded SafeTensors
checkpoint. The current architecture dispatch covers:

- DeepSeek-V3 and DeepSeek-R1
- Gemma 4 dense and MoE text and multimodal models
- GPT-OSS
- Thinking Machines Lab Inkling
- Kimi Linear
- Llama and Mistral
- LFM2 and LFM2.5, including MoE variants
- Nemotron-H
- Qwen2 and Qwen2.5 text models (`model_type: "qwen2"`)
- Qwen3, including MoE variants
- Qwen3-Next
- Qwen3-VL and Qwen3-VL-MoE
- Qwen3.5 dense and MoE models

Moshi and PersonaPlex are exposed through the separate realtime
speech-to-speech token API. Their adapter uses the canonical fair scheduler for
request/work identity, isolated temporal/depth state, bounded queues,
cancellation, failure poisoning, and telemetry. The API operates on encoded
audio tokens; codec encoding/decoding is provided separately by
`safemlx-codec`, and audio device I/O remains the application's responsibility.
Released sessions are bound to the SHA-256 content identity of the selected
checkpoint files and a normalized execution identity that includes effective
quantization. The same artifact can hand state between resident and layerwise
loads; a different checkpoint with identical geometry is rejected.
Bounded drains provide cooperative deadline/cancellation boundaries between
frame executions; an already-issued Metal operation is not preempted.

Image preprocessing requires the `safemlx-lm/image-processing` feature. Audio
preprocessing requires `safemlx-lm/audio-processing`. These features are not
enabled by default.

## Capability and request-state estimation

Every high-level `LoadedModel` has an architecture-independent
`capabilities()` report and checked `estimate_runtime_state()`/`admit()` path.
The dispatch is exhaustive over the public `Model` enum, so a new model variant
must explicitly define its context, modalities, and state layout.

The high-level API has one `Model` and one `ModelGenerate` variant per
architecture. `FullyResident`, `LayerwiseHost`, and `DenseDiskStream` are
internal backend policies selected at load time; they share the same forward,
cache, generation, prompt-cache, observation, and reporting dispatch. Load-time affine or MXFP4
conversion hands its immutable transformed tensors to this same execution
engine without writing an intermediate checkpoint or retaining a second
executable model tree.

- Llama/Mistral, Qwen2/Qwen2.5, Qwen3, and Qwen3-VL use KV-head-aware GQA accounting.
- Qwen2 reports full layers and sliding layers grouped by every distinct window;
  cache estimates use the actual KV-head count, derived or explicit head
  dimension, bounded retention for each sliding group, and context-growing
  backing for full layers. `StateMemoryAssumptions::sliding_window_bounds`
  exposes the sorted distinct bounds.
- GPT-OSS reports full layers and sliding layers grouped by each exact window
  from its normalized attention schedule. Gemma 4 accounts for full versus
  sliding attention separately.
- DeepSeek-V3/R1 accounts for compressed MLA latent plus rotary-key state.
- Kimi Linear separates bounded KDA convolution/recurrent state from
  context-growing compressed no-RoPE MLA state.
- LFM2 and Nemotron-H separate bounded convolution/Mamba state from attention
  KV state. Nemotron-H attention groups are context-growing or window-bounded
  according to each scheduled attention policy; dense MLP and MoE layers add
  no persistent decoder state.
- Qwen3-Next and Qwen3.5 separate recurrent linear-attention state from
  full-attention KV.
- Inkling groups full-context and every distinct scheduled sliding window using
  their respective KV-head geometry, then adds four bounded convolution states
  per decoder layer.
- Prepared Gemma 4, Inkling, Qwen3-VL, and multimodal Qwen3.5 inputs derive
  model positions from processor output geometry or masks, not a
  tokens-per-image constant.

Persistent-state formulas are exact under their stated float32-cache and batch
assumptions, including backing-allocation growth units and Gemma 4 shared-KV
layers. Gemma's sliding attention is reported separately from its full-context
KV backing. Prepared multimodal requests report `Conservative`: they add a
checked, architecture-specific media-tower workspace bound to exact decoder
state and media-embedding accounting. Qwen derives it from prepared grids,
full/window attention chunks, mergers, and DeepStack outputs; Gemma 4 derives
it from padded/valid patch and audio-mask geometry plus the loaded tower
configuration; Inkling derives it from the released hMLP folds and dMel
codebooks. `require_complete_estimate` accepts this complete conservative bound.
Callers should still supply a reserve for process-wide allocator cache, driver
allocations, and other memory that is not owned by the request.

System admission signals cover Apple unified memory through
`hw.memsize`/`os_proc_available_memory`, Linux host memory through
`/proc/meminfo`, and Windows host memory through `GlobalMemoryStatusEx`.
CUDA-specific free-device-memory remains unavailable; host and device capacity
are not combined.

Kimi Linear directories may omit `tokenizer.json`: the loader imports the
official `tiktoken.model`, registers the full reserved-token range, and uses
the checkpoint's distinct BOS, tokenizer-EOS, generation-EOS, unknown, and
padding IDs. Ordinary and tool-enabled chat use the official template.
Streaming native-tool events preserve
`functions.<declared_name>:<nonnegative_index>` identifiers and support
repeated or parallel calls.

## GGUF

The high-level loader accepts a GGUF file for these `general.architecture`
values:

- `deepseek2`
- `gemma4`
- `gpt-oss`
- `inkling` (with an optional sibling audio/vision `mmproj` for multimodal input)
- `kimi-linear`
- `llama` and `mistral`
- `lfm2` and `lfm2moe`
- `nemotron_h` and `nemotron_h_moe`
- `qwen2`
- `qwen3` and `qwen3moe`
- `qwen3next`
- `qwen3vl` and `qwen3vlmoe` (with their companion vision projection checkpoint)
- `qwen35` and `qwen35moe` (with an optional sibling Qwen3.5 vision projector)

The tokenizer and chat template are reconstructed from GGUF metadata when
possible. A sibling `tokenizer.json` can supply a tokenizer that is absent from
the file or uses an unsupported embedded tokenizer model. `qwen3vl` and
`qwen3vlmoe` require a
validated sibling vision projector; Inkling remains text-loadable without its
combined audio/vision projector and reports multimodal readiness separately.

For Hugging Face Qwen2/Qwen2.5 directories, `tokenizer.json`,
`tokenizer_config.json`, generation EOS metadata, padding/BOS/EOS ids, added
stop tokens, and checkpoint chat templates flow through the architecture-neutral
tokenizer and chat APIs. Standard Qwen2.5 instruct templates therefore use the
same rendering, streaming, sampling, and EOS/stop handling as other text
models. Native tools are reported only when the supplied template passes the
existing behavioral protocol validation; SafeMLX does not infer tool syntax
from the Qwen family name.

`safemlx-gguf` parses GGUF v1-v3 in either byte order and validates all shard
headers before payload materialization. Its supported dense and quantized
tensor encodings are listed in the
[`safemlx-gguf` README](../safemlx-gguf/README.md).

## Weight loading and residency

Fully resident loading is the default. SafeTensors and registered GGUF families
can also use host-backed layer windows or experimental dense disk streaming.
GGUF bounded loading covers Kimi Linear, DeepSeek2, Gemma 4, Llama/Mistral, LFM2,
Nemotron-H, Qwen2/Qwen2.5, Qwen3, dense or MoE Qwen3-VL with its mmproj,
Qwen3.5, and Qwen3-Next.
Supported MoE families can cache routed experts independently; for GGUF these
are Kimi Linear, DeepSeek2, LFM2-MoE, Nemotron-H-MoE, Qwen3-MoE,
Qwen3.5-MoE, MoE Qwen3-Next, and Qwen3-VL-MoE.

Qwen2/Qwen2.5 text checkpoints share the canonical `architectures::qwen::dense`
decoder with Qwen3. SafeTensors and GGUF support fully resident, layerwise-host,
and dense disk-streamed execution; floating SafeTensors may use supported
load-time affine/MXFP4 quantization. Qwen2 applies the checkpoint-required Q/K/V
biases and layer-selective sliding attention exactly. Qwen2-VL, Qwen2.5-VL,
Qwen2 MoE, and older custom-code Qwen model types are rejected.

Llama/Mistral, dense Qwen, Qwen3-Next/Qwen3.5, DeepSeek-V3/R1, Gemma 4,
Inkling, Kimi Linear, LFM2/LFM2-MoE, Nemotron-H/Nemotron-H-MoE, and GPT-OSS tensor
parallelism use balanced, contiguous logical partitions rather than requiring
every geometry field to divide the rank count. GQA families partition complete
KV groups, with the corresponding query heads, Q/K/V biases, and GPT-OSS
attention sinks kept together. DeepSeek assigns MLA heads, dense and shared
SwiGLU widths, and routed
expert intermediates to independent logical domains; each domain owns all of
the projections and packed companions that must use the same rank-local range.
Dense SwiGLU widths may be uneven. LFM2 preserves its heterogeneous
schedule: fused gated short-convolution projections, depthwise kernels, and
bounded cache state share one rank-local channel range, while attention layers
cache only their rank-local KV heads. Kimi likewise treats each KDA head as one
atomic domain spanning all six head-expanded projections, Q/K/V depthwise
kernels, transition state, recurrent state, and the reduced output projection;
MLA head expansion uses a separate atomic domain while its compressed cache
stays independent of head count. For affine, MXFP4, and native GGML
weights, adjacent semantic units are combined until row boundaries land on
complete quantization blocks; packed values and companions then map the
identical logical range into their respective physical shapes. GPT-OSS native
MXFP4 expert blocks use 32-channel aligned units across fused gate/up and down
projections. Preflight rejects topologies with fewer legal units than ranks.
KV-cache and prompt-cache layouts use each layer's actual rank-local KV-head
count and LFM2 convolution-channel count. Two-rank GPT-OSS SafeTensors and
native MXFP4 GGUF coverage exercises uneven heads, sinks, block-aligned experts,
all three residency policies, bounded GGUF reads, and prompt-cache reload.
DeepSeek coverage crosses its dense-to-MoE boundary under all three residency
policies and verifies uneven MLA heads and dense, routed, and shared-expert
widths. Its SafeTensors fixture also exercises rank-aware prompt-cache reload;
the matching DeepSeek2 GGUF fixture verifies numerical parity and bounded
rank-selective reads. LFM2
coverage additionally includes dense F32 and block-aligned Q8_0 GGUF
checkpoints with bounded-read telemetry. Kimi coverage uses mixed KDA/MLA and
dense/MoE SafeTensors fixtures to verify uneven heads and dense, shared, and
routed intermediates under every parameter residency policy. A matching GGUF
fixture verifies exact name translation, two-rank numerical parity, and bounded
rank-selective reads. Both routes exercise heterogeneous live paging and
rank-aware prompt-cache reload: growing MLA blocks page through the shared
manager while bounded KDA convolution/recurrent state remains resident and is
persisted as fixed state.
Inkling assigns complete GQA KV groups together with their query/relative
heads, output columns, and K/V short-convolution channels. Dense, routed, and
shared-expert intermediates use planner-derived balanced or
quantization-aligned ranges, and hMLP layers slice those same authoritative
logical ranges from folded inputs. Rank-local prompt-cache descriptors record
the actual KV-head counts and convolution widths. SafeTensors and GGUF use the
same plan and all three parameter-residency policies; arbitrary uneven text
and dense folded-vision ranges have deterministic planner coverage.
Nemotron-H treats each complete Mamba B/C group, including all of its heads, as
one atomic domain spanning the fused input projection, depthwise convolution,
transition/state parameters, gated RMSNorm, output projection, and fixed cache.
Its GQA, dense ReLU2, routed-expert, and shared-expert widths use independent
balanced domains. Two-rank SafeTensors tests cover live-paged attention plus
resident rank-local Mamba state under all three parameter-residency policies
and save/drop/reload continuation; the matching GGUF test verifies the same
cache behavior, numerical parity, and bounded rank-selective payload reads.
Replicated expert-parallel coverage also exercises live paging and fixed-state
reload. Packed affine
planner coverage verifies that every row boundary and companion tensor uses the
same aligned semantic range.

Gemma 4 assigns each text or vision GQA KV group together with its query heads
and output columns. Dense gated-GELU widths and routed-expert widths are
independent balanced domains; expert gate/up/down tensors and affine, MXFP4,
or native GGML companions share one quantization-aligned range. Shared-KV text
layers preserve the publisher's local query/KV ratio without allocating cache
state, and prompt-cache descriptors use the planner-authored per-layer KV
geometry. Vision patch/position channels, vision MLP widths, audio attention
heads, both audio feed-forward blocks, light-convolution channels, and audio
input/output modality widths use the same semantic planner. Uneven audio
light-convolution ranks normalize with the global channel count rather than an
equal-shard reconstruction. Deterministic two-rank coverage exercises uneven
text GQA, dense and routed widths, vision GQA/MLP/patch widths, audio heads,
modality ranges, and rank-local cache identity.

Qwen3 MoE, Kimi Linear, and LFM2-MoE use the same logical partition mechanism for routed-expert
intermediates. All expert identities and the router remain replicated, while a
balanced rank-local range is mapped through both fused gate/up segments and
the down projection. Dense expert widths may be uneven. Affine, MXFP4, and
native GGML down projections combine channels into complete quantization units
before balancing; packed weights, scales, and biases share the group-level
range and are validated atomically.

Generalized tensor-parallel family loaders support all three parameter
residency policies through one execution model. `FullyResident` plans the same
rank-local tensors as `LayerwiseHost` and `DenseDiskStream`, then constructs
and populates each local layer once and pins it on the execution device. The
bounded policies continue to construct transient execution modules under
their host/device or disk/host/device windows. Reports distinguish global
unsharded parameter bytes, total local parameter bytes, and currently resident
local device bytes through the residency report; parallel model information
also exposes permanently pinned and maximum planned device bytes. These
parameter totals exclude KV/recurrent caches,
activations, temporary collective buffers, and allocator caches.

The residency-reporting API exposes `LayerwiseModelMetadata::residency`,
`layer_parameter_bytes`, `maximum_device_layer_bytes`, and
`device_layer_capacity`.

Llama and Mistral use the same architecture-neutral
`LayerSchedule<AttentionPolicy>` as their sole normalized attention geometry.
Hugging Face absence or `null` produces an all-full schedule; a positive
`sliding_window` applies that exact window to every layer. GGUF absence or zero
means all-full according to the format, while a positive
`llama.attention.sliding_window` or `mistral.attention.sliding_window` means
all-sliding. Invalid types, zero/negative Hugging Face values, negative GGUF
values, and overflowing values fail in the normalization shared by inspection
and loading.

Resident, bounded-weight, dense-stream, ordinary-cache, paged-cache,
tensor-parallel, pipeline-parallel, fingerprint, and state-estimation paths all
query the ordered schedule. Internally supplied schedules can use arbitrary
full/sliding order and distinct positive windows. Each ordinary or paged cache
receives its layer's exact policy; an `N`-position sliding window includes the
current token and retains at most `N - 1` past positions between calls. Memory
reports count context-growing full layers and group bounded layers by exact
window. Prompt-cache schema v4 persists the complete order, exact per-layer
windows and tensor layouts, and each layer's retained token interval.

JSON configuration is parsed with
`architectures::llama::model::model_args_from_config_value`. Normalized
arguments expose `attention_schedule`, and execution uses one
per-layer-configured cache route.

All dense-Qwen execution consumes one architecture-neutral
`LayerSchedule<AttentionPolicy>`, an ordered list of `AttentionPolicy::Full` or
`AttentionPolicy::Sliding { window }`. `LayerSchedule<P>` supplies generic
layer-count validation and fallback-free borrowed access so hybrid decoders can
adopt architecture-specific policy enums without changing the schedule
container. Hugging Face Qwen2 fields normalize to all-full when
`use_sliding_window` is false; when true,
`max_window_layers` counts the leading full layers and all remaining layers use
the required positive `sliding_window`. Qwen3 remains all-full. GGUF uses
`qwen2.attention.sliding_window_pattern` exactly as an ordered Boolean list,
where `true` means sliding. Arbitrary exact-length patterns, including
alternating and discontiguous patterns, are supported. A GGUF window without a
pattern applies to every layer. Missing windows for enabled layers, invalid or
overflowing windows, wrong encodings, and pattern-length mismatches fail in the
shared inspection/load parser before weights are materialized.

The public architecture-erased variant is `Model::DenseQwen`; residency is an
internal policy of its generalized execution engine. Direct users load through
the high-level `load_model_with_options` API; architecture-level execution uses
the generalized `architectures::qwen::dense::layerwise` loader. Every schedule supports resident,
layerwise-host, dense-streamed, ordinary-cache, and paged-cache execution. The
complete ordered schedule participates in architecture and prompt-cache
fingerprints. Resident models support schema-v4 persisted prompt caches for arbitrary ordered Qwen2
full/sliding patterns and exact per-layer windows; all-full Qwen3 uses the same
route. Qwen2 Q/K/V biases are applied in every execution policy. Tensor and
pipeline parallel Qwen use the generalized dense-Qwen adapter. Pure pipeline
execution supports Qwen2/Qwen2.5, Qwen3, and Qwen3 MoE SafeTensors and canonical
GGUF, with fully resident or dense disk-streamed local layers,
ordinary/paged/persisted caches,
and rank-synchronized generation. Qwen3 MoE routed experts are materialized
from the same direct or derived binding plan as bounded execution; no
MoE-specific pipeline decoder exists. Independent expert caches and
TP+PP+EP pass their stage-, EP-, and TP-local semantic recipes to the shared
bounded affine/MXFP4 materializer. SafeTensors expert ownership and projection
ranges collapse to bounded contiguous spans, conversion scratch is capped by
the final packed local catalog, and expert-cache telemetry reports the exact
selected and written bytes.

GPT-OSS uses the same architecture-neutral
`LayerSchedule<AttentionPolicy>` representation. Hugging Face `layer_types`
entries normalize in decoder order from `full_attention` and
`sliding_attention`; when the field is omitted, the published GPT-OSS default
alternates sliding and full attention beginning with sliding at layer zero.
The positive `sliding_window` supplies the exact window for every declared
sliding layer. GGUF's single `gpt-oss.attention.sliding_window` follows the
format's fixed alternating layer meaning. Invalid entries, count mismatches,
and zero, negative, or overflowing windows fail in the shared inspection/load
parser before weights are materialized.

The schedule is the sole source for resident, layerwise-host, dense-streamed,
ordinary-cache, paged-cache, generation, structural, expert-parallel,
fingerprint, and runtime-state paths. Internally constructed schedules may use
arbitrary ordering and distinct windows. Prompt-cache schema v4 persists the
complete ordered schedule, exact per-layer windows and tensor layouts, and each
layer's retained token interval. JSON callers use
`architectures::gpt_oss::model::model_args_from_config_value`.
Pure GPT-OSS pipeline execution uses the same binding and schedule plans for
SafeTensors and canonical `gpt-oss` GGUF. It supports fully resident and dense
disk-streamed local layers, ordinary/paged/schema-v4 persisted caches, and
rank-synchronized generation. Native MXFP4 expert banks remain packed;
eligible dense matrices may be MXFP4-quantized before resident,
host-layerwise, or dense-streamed stage planning, while affine and packed-input
transcoding are rejected.

Gemma 4 text and assistant models use
`LayerSchedule<architectures::gemma4::model::LayerPolicy>` as their authoritative
decoder geometry. Each ordered entry contains its full/exact-window attention
policy, exact head dimension and KV-head count, KV local/publish/shared role,
projected-value versus key-as-value topology, exact dense MLP width, and
dense-only versus dense-plus-sparse-MoE feed-forward topology.
Hugging Face `layer_types` is an exact decoder-order list of
`full_attention`/`sliding_attention`; an omitted or empty list means all full.
Any enabled sliding entry requires the one positive, executable
`sliding_window`. GGUF reads
`gemma4.attention.sliding_window_pattern` (or the assistant-prefixed equivalent)
as an exact Boolean array; absent pattern metadata means all full, even when an
otherwise valid unused window is present. Arbitrary patterns, including
alternating and discontiguous layouts, are accepted. Wrong encodings, length
mismatches, unknown entries, missing windows, and zero, negative, or overflowing
windows fail in the same normalization used by inspection and loading.

Hugging Face global/full-attention geometry, doubled-final-layer MLP, shared-KV
suffix, key-as-value, and MoE fields are source inputs only. GGUF per-layer
feed-forward lengths and KV-head counts are preserved exactly. Resident,
layerwise-host, multimodal prefill/decode, generation, shared-KV routing, and
Gemma 4 assistant drafting query only the normalized composite schedule.
Multimodal masks are keyed by exact policy, so internally constructed schedules
may use different windows. Gemma sliding layers apply a bounded attention mask
but deliberately retain full KV backing; shared layers reuse state by exact
policy. Ordinary cache slots exist only for KV-owning layers. Capability reports
total full/sliding policy counts and groups sliding layers by each window, while
memory estimates group KV-owning layers by their exact per-position geometry and
separately account for shared layers and context-growing backing. Architecture
fingerprints include the complete ordered composite schedule. Persisted Gemma
prompt caches, live paged Gemma KV caches, and tensor-parallel Gemma execution
use that normalized schedule. Pure pipeline execution supports the Gemma text
decoder for exact SafeTensors catalogs and `gemma4` GGUF files. Its
dependency-aware planner keeps shared-KV publishers and consumers in one
contiguous stage unit and rejects topologies with too many stages. Per-layer
residual inputs travel as declared immutable stage auxiliaries. The common
pipeline runtime owns one type-erased stage shell and materializes standard,
paged, and persisted state from the same canonical per-layer cache schedule.
All supported decoder payloads share one resident/dense-stream layer executor
and consume the generalized architecture adapter's binding plan; stage-role
selection is lazy and does not construct a second source model. There is no
family enum in transport, cache, residency, or persistence paths.
Gemma text stages can requantize dense SafeTensors or compatible dense GGUF
weights to affine or MXFP4 storage when every selected operation satisfies the
requested group alignment. Direct tensors and derived bindings, including
fused GGUF expert gate/up banks, use the same authoritative binding plan before
quantization. Host-layerwise and dense disk-streamed stages create their packed
overlay before residency planning, and matching checkpoint-native encodings
load directly. Implicit transcoding between packed encodings fails closed.
Gemma image and audio encoders are not pipeline capabilities.

Resident, layerwise-host, and tensor-parallel Gemma multimodal execution uses
the common validated execution-group DAG: vision and audio are independent
roots, and the text decoder explicitly depends on both. A root may be skipped
when its modality is absent, but both dependency outputs are resolved before
the adapter performs exact token/embedding and mask assembly at the text
boundary. Execution-group dependencies do not require numeric adjacency, and
ready roots execute serially.

`ModelArgs::layer_schedule` and `ModelArgs::layer_policy` are the normalized
layer APIs. JSON callers use `model_args_from_config_value` and
`gemma4_assistant_config_from_value`; execution uses
`TransformerBlock::layer_policy` and `ModelInput::sliding_masks`.
`CacheStateStrategy::SharedFullKv` reports `full_attention_layers` plus exact
`sliding_attention` window/count groups.

Inkling normalizes both decoder choices into
`LayerSchedule<architectures::inkling::model::LayerPolicy>`. Each ordered entry
contains an `AttentionPolicy` and a `FeedForwardPolicy::{Dense, SparseMoe}`.
For Hugging Face configuration, `local_layer_ids` and exact-length
`layer_types` (`sliding_attention` or `full_attention`) are equivalent sources;
if both are present they must agree. When neither is present, the released
five-sliding/one-full cadence is used. Likewise, exact-length
`mlp_layer_types` (`dense` or `moe`) and `dense_mlp_idx` must agree when both
are present; otherwise the supplied form, or the released all-MoE default, is
normalized once. Invalid indices, duplicates, entries, counts, conflicts, and
non-positive or overflowing windows fail before weight materialization.

GGUF's `inkling.attention.sliding_window_pattern` is read as an exact Boolean
array in decoder order and combined with its one positive window;
`inkling.dense_block_count` supplies the dense feed-forward prefix. Arbitrary
attention ordering is supported. When the pattern is absent, the same released
five-sliding/one-full cadence is used. Wrong pattern encodings, length
mismatches, inconsistent per-policy KV-head geometry, invalid windows, and
impossible dense counts fail closed. The
checkpoint formats currently provide one shared sliding window, but internal
schedules and cache tests support distinct windows.

Resident, layerwise-host, dense-streamed, ordinary-cache, paged-cache,
generation, structural, expert-parallel, fingerprint, and runtime-state paths
consume only this schedule. A sliding window of `N` includes the current token,
so ordinary caches retain at most `N - 1` past positions while paged caches use
the same attention bound over their backing allocation. Runtime reporting
groups every distinct window and accounts separately for full-context KV,
bounded sliding KV, batch/context growth, and four bounded short-convolution
states per layer. The complete ordered attention/feed-forward schedule is in
the architecture fingerprint. Persisted Inkling prompt caches store rank-local
attention state and all four short-convolution histories through the
heterogeneous-state schema. Tensor-parallel preflight, execution, and cache
creation use the exact schedule. Pure text pipeline stages support exact
SafeTensors and canonical `inkling` GGUF checkpoints with fully resident,
host-layerwise, or dense disk-streamed local blocks. Every layer uses the shared
`KeyValueWithFixedState` descriptor: ordinary or paged KV and all four
convolution histories are persisted atomically by global layer and semantic
role. Direct and scheduler-owned multimodal image/audio ingress carry one
prepared payload whose modality and shape identity participate in collective
consensus; stage zero consumes it through the shared model-input path and
pipeline transport preserves matching TP/EP coordinates. Inkling's bounded
multimodal path declares `vision_encoder -> text_decoder` in the same
execution-group DAG and performs projection normalization and folded token
assembly only when the text node becomes ready. Pipeline and Cartesian
load-time affine/MXFP4 conversion use the same rank-local semantic recipes,
including aligned hMLP projections, the dMel embedding, and routed/shared
expert banks.

JSON callers use
`architectures::inkling::model::model_args_from_config_value`, then query
`TextArgs::layer_policy` or `TextArgs::layer_schedule`.

LFM2 and LFM2-MoE normalize both decoder choices into
`LayerSchedule<architectures::lfm2::model::LayerPolicy>`. Each entry contains
an `OperatorPolicy::{CausalConvolution, SelfAttention(AttentionPolicy)}` and a
`FeedForwardPolicy::{Dense, SparseMoe}`. Hugging Face `layer_types` supplies the
operator order; for `lfm2_moe`, `num_dense_layers` supplies a leading dense
count and all later entries become sparse MoE. Dense `lfm2` rejects a nonzero
`num_dense_layers`. GGUF similarly combines its per-layer KV-head metadata with
`leading_dense_block_count`. Counts outside the decoder range fail during the
same normalization used by inspection and loading.

After normalization, the source thresholds are discarded. Resident, bounded,
dense-streamed, sparse-expert, structural, GGUF materialization, and
expert-parallel paths query the ordered schedule. Internally constructed
schedules may freely interleave dense and sparse-MoE blocks independently of
convolution and attention. The full ordered operator/feed-forward schedule is
hashable and available through `ModelArgs::layer_schedule_fingerprint`;
fallback-free lookup uses `ModelArgs::layer_policy`. JSON callers use
`architectures::lfm2::model::model_args_from_config_value`. Persisted LFM2
prompt caches store the ordered causal-convolution histories and attention KV
state under the complete schedule fingerprint.

Pure LFM2 and LFM2-MoE pipeline execution supports exact SafeTensors catalogs
and canonical `lfm2`/`lfm2moe` GGUF checkpoints, with fully resident or dense
disk-streamed local decoder layers. The pipeline cache materializer consumes
the same `LayerCachePolicy` schedule as resident and bounded execution:
causal-convolution history becomes an ordered semantic state slot and
full-attention state uses the shared ordinary or paged KV contract. Paged
prompt-cache publication stores fixed state and attention blocks atomically,
including stages whose local range contains no attention layer, and restores
every slot by global layer and semantic role. LFM2-MoE changes feed-forward
execution only; it does not introduce another cache representation.

Nemotron-H normalizes the Hugging Face `hybrid_override_pattern` and GGUF
per-layer feed-forward/KV-head metadata into
`LayerSchedule<architectures::nemotron_h::model::LayerPolicy>`. Entries are
`Mamba`, `SelfAttention(AttentionPolicy)`, `DenseMlp`, or `SparseMoe`; the
optional positive `sliding_window` becomes the exact policy of every attention
entry, while its absence means full attention. The ordered schedule is the sole
execution geometry for resident, layerwise, dense-streamed, structural,
expert-parallel, cache-validation, fingerprint, and runtime-state paths.
Mamba caches hold bounded convolution/SSM state, attention caches select
context-growing or window-bounded KV storage, and MLP/MoE cache entries are
stateless markers. Invalid markers, layer-count mismatches, and zero, negative,
or overflowing windows fail during normalization before model allocation.
JSON callers use
`architectures::nemotron_h::model::model_args_from_config_value`. Persisted
Nemotron-H prompt caches encode Mamba convolution/SSM tensors, attention KV,
and explicit stateless entries under the complete ordered schedule fingerprint.
Tensor-parallel manifests additionally record the planner-derived rank-local
Mamba head/group and KV-head geometry, so uneven partitions reload exactly.
Nemotron-H live paging keeps bounded Mamba convolution/SSM tensors resident
and pages only attention KV blocks. The same cache works for resident and all
three bounded-weight policies, tensor parallelism, and replicated expert
parallelism; full and sliding attention retain their scheduled semantics.
`CacheStateStrategy::HybridRecurrent` reports `full_attention_layers`,
`sliding_attention`, and `recurrent_layers`, so hybrid state reports preserve
bounded attention groups.

Pure Nemotron-H and Nemotron-H-MoE pipeline stages support SafeTensors and
canonical GGUF with fully resident or dense disk-streamed local blocks. Mamba
convolution and recurrent tensors are descriptor-backed fixed state, attention
uses the shared full/sliding ordinary or paged KV cache, and MLP/MoE-only
layers use the one stateless representation. Prompt-cache persistence and
offset validation consume the same local schedule.

Embedded multi-token prediction is one scheduler contract across
DeepSeek-V3/R1, Inkling, Nemotron-H, Qwen3-Next, and Qwen3.5/3.6. Each adapter
owns only its checkpoint-specific fusion and prediction blocks. Draft cache
clone, verification, partial-accept rollback/replay, cancellation, semantic
streaming, prepared-chat batching, and stochastic acceptance live in the
shared backend. DeepSeek prediction layers use compressed MLA state; Inkling
uses its configured per-depth full/sliding KV plus short-convolution state;
Nemotron-H executes every physical operator in
`mtp_hybrid_override_pattern`. Resident and paged caches include these draft
owners under the same rank identity as the target cache. DeepSeek and
Nemotron-H prediction experts use the ordinary independent expert catalog and
bounded route executor when that residency is selected. Embedded prediction
heads are currently admitted only for single-model execution; parallel model
instances reject the capability until prediction modules have Cartesian
TP/PP/EP ownership plans.

Qwen3.5 and Qwen3-Next normalize Hugging Face `layer_types`, the Qwen3-Next
`full_attention_interval` fallback, and GGUF full-attention intervals into
`LayerSchedule<architectures::qwen::hybrid::qwen3_5::LayerPolicy>`. The ordered
entries distinguish recurrent `LinearAttention` from
`SelfAttention(AttentionPolicy::Full)` and drive resident and bounded execution,
cache construction and validation, structural admission, and exact recurrent/KV
state accounting. Explicit layer lists must match the decoder depth; invalid or
zero interval values fail during normalization. JSON callers use the
architecture module's `model_args_from_config_value` function. Execution uses
`TransformerBlock::layer_policy`, and `Cache::new` returns a validation result.

Qwen3-Next and Qwen3.5 dense/MoE pipeline stages support exact
SafeTensors and canonical `qwen3next`/`qwen35`/`qwen35moe` GGUF with fully
resident or dense disk-streamed local blocks. Linear-attention convolution and
recurrent arrays become fixed semantic slots while full-attention layers use
the shared ordinary or paged KV implementation. Qwen3.5 SafeTensors and
canonical GGUF with a validated sibling projector add a stage-zero vision
execution group and accept direct or scheduler-owned image/video ingress.

Qwen3-VL and Qwen3.5 vision blocks use
`LayerSchedule<architectures::qwen::vl::vision::VisionLayerPolicy>`. Each entry
independently identifies `VisionAttentionPolicy::{Full, Windowed}` and an
optional DeepStack merger-bank index. Qwen3-VL Hugging Face and GGUF metadata
normalize to an all-full attention schedule; `deepstack_visual_indexes` is
validated for exact range and uniqueness, and its source order is preserved as
the merger-bank mapping. Qwen3-VL rejects the Qwen3.5-only `window_size` and
`fullatt_block_indexes` fields instead of silently ignoring them. Qwen3.5
normalizes its positive window and exact full-attention block list into the same
policy type. Qwen3.5 configs that omit both window metadata fields normalize to
full attention at every vision block; an explicit full-attention list selects
the remaining blocks as windowed.

The schedule length is the sole vision depth after normalization. Resident and
bounded vision execution, DeepStack capture, SafeTensors/GGUF structural plans,
GGUF name translation, and prepared-input workspace accounting all use safe
indexed policy access. Ordered schedule fingerprints include both attention and
DeepStack choices. Qwen3-VL prompt-cache persistence stores decoder KV state and
the multimodal RoPE position delta. Prefix identities must cover processed media
and processor settings as well as token IDs. Tensor-parallel snapshots derive
each layer's rank-local KV-head count from the authoritative text parameter
plan, preserve the fixed position state on layer zero, and record exact rank
ownership on every persisted block. Uneven whole-GQA-group partitions therefore
save and reopen without reconstructing geometry through scalar division.
Qwen3-VL and multimodal Qwen3.5 bounded execution declare
`vision_encoder -> text_decoder` through the common execution-group DAG.
Vision merger and DeepStack outputs are assembled at the decoder boundary;
text-only decode skips vision execution while preserving the declared data
dependency. Graph topology is validated before checkpoint materialization.

Kimi Linear normalizes its orthogonal attention and feed-forward choices into
`LayerSchedule<architectures::kimi_linear::model::LayerPolicy>`. Each entry
contains `AttentionKind::{Kda, Mla}` and
`FeedForwardPolicy::{Dense, SparseMoe}`. Hugging Face's one-based `kda_layers`
and `full_attn_layers` must be disjoint and cover the exact decoder depth;
`first_k_dense_replace` and positive `moe_layer_freq` are converted once using
the checkpoint's zero-based sparse-layer rule. GGUF per-layer KV-head metadata
selects MLA for positive entries and KDA otherwise, while the leading dense
block count supplies feed-forward policy. Invalid, duplicate, missing, or
out-of-range layer entries and invalid dense/MoE geometry fail in the shared
inspection/load parser before materialization.

Resident, layerwise-host, dense disk-streamed, sparse-expert-cache,
sparse-expert-with-dense-layers, expert-parallel, cache, structural,
fingerprint, and runtime-state paths consume only the normalized schedule. Its
KDA recurrent state remains bounded while no-RoPE MLA caches grow with context;
cache validation rejects a per-layer state kind that differs from the schedule.
The complete ordered attention/feed-forward policy participates in architecture
identity. Callers use `model_args_from_config_value`, `layer_schedule`,
and `layer_policy`. The GGUF loader accepts split MLA K/V-B or combined MLA KV-B
projections, singleton convolution layouts, dense/K/IQ tensors, and MXFP4-MoE
type 39. Paged and persisted prompt caches represent KDA convolution/recurrent
and compressed-MLA state through the canonical heterogeneous state schema;
owner/role, shape, dtype, ordering, and layer policy are validated before
restoration.

DeepSeek-V3/R1 uses
`LayerSchedule<architectures::deepseek_v3::model::LayerPolicy>` as its sole
normalized dense/MoE topology. Each ordered entry is `DenseMlp` or `SparseMoe`.
Hugging Face `first_k_dense_replace` and positive `moe_layer_freq` normalize
with the published zero-based rule: a layer is sparse only when its index is at
or beyond the threshold and divisible by the frequency. DeepSeek2 GGUF
`leading_dense_block_count` supplies the dense prefix and all remaining layers
are sparse. Invalid decoder counts, negative or oversized thresholds, and
non-positive frequencies fail through the same parser used by inspection and
loading.

Resident, layerwise-host, dense-stream, sparse-expert-cache, structural,
tensor-parallel, pipeline-parallel, expert-parallel, and fingerprint paths
consume the schedule directly. Internally constructed schedules may use any
dense/MoE order. Attention remains uniform MLA, so cache and runtime-state
accounting allocate one context-growing compressed latent-plus-rotary entry for
every scheduled layer; ordinary, paged, and persisted cache routes remain
supported. The complete ordered feed-forward schedule participates in cache
identity even though it does not alter MLA state shape. JSON callers use
`model_args_from_config_value`, `layer_schedule`, and `layer_policy`.

Qwen3-Next supports the official native fine-grained E4M3 checkpoint format
(`fp8`, dynamic activations, 128 x 128 weight blocks) with fully resident,
layerwise, sparse expert-cache, and pure expert-parallel loading. Fused QKVZ
weights and inverse scales are split without dequantization, dense BF16 BA is
preserved, and routed expert weights remain checkpoint-backed at expert
granularity for sparse-cache and expert-parallel execution.

Important boundaries:

- Prompt-cache persistence accepts schema v4. Its ordered
  architecture-neutral layout preserves full/sliding order, each
  positive window, ordinary KV, DeepSeek compressed MLA, fixed convolution and
  recurrent tensors, multimodal prefix state, tensor geometry, and global
  distributed layer indices. Llama/Mistral, dense Qwen, GPT-OSS, DeepSeek,
  Kimi Linear, Qwen3-Next/Qwen3.5, Gemma 4, Inkling, Qwen3-VL, LFM2, and
  Nemotron-H use this shared representation on resident and bounded-weight
  persistence routes.
- GGUF remains fully resident by default. `LayerwiseHost`, `DenseDiskStream`,
  and supported sparse-expert policies use header-only logical catalogs and
  bounded payload materialization.
- Dense Qwen SafeTensors support out-of-core load-time affine/MXFP4 conversion
  with fully resident, layerwise-host, or dense-streamed weights. Supported MoE
  families use the same ordinary-layer overlay together with an independent
  packed expert overlay. Pipeline stages use the same overlay for SafeTensors
  and dense F32/F16/BF16 GGUF recipes before host/device planning. The GGUF
  reader bounds native-axis and reshaped contiguous spans, including composed
  expert and row selections over fused banks. Standalone host-layerwise and
  dense disk-streamed GGUF loads use the same packed overlay before residency
  planning; supported independent expert caches quantize their selected routed
  experts through the same source store. Packed GGUF input is excluded by the
  span type and is never requantized.
- Transfers, cache movement, distributed handoffs, and bounded conversion use
  SafeMLX completion events for exact same-device stream ordering. Host-visible
  route metadata remains an inherently synchronous decision boundary, but its
  readback waits only for the arrays that produce that metadata rather than
  draining an entire stream.
- On Apple silicon, reported host and device residency are logical tiers over
  the same physical unified memory. They do not create additional capacity.
- Parameter budgets do not include activations, KV or recurrent state, kernels,
  allocator caches, checkpoint mappings, or every temporary buffer.
- Kimi KDA and Qwen linear-attention state are persisted as fixed-state tensors,
  not misrepresented as paged KV. Inkling may page attention while keeping its
  convolution histories resident, and schema-v4 publication atomically records
  both parts before reload into an exact resident continuation cache.
- LFM2 persists ordered causal-convolution history and full-attention KV.
  Pure LFM2/LFM2-MoE pipeline stages use the same descriptor-backed live and
  persisted state, including fixed-state-only ranks.
  Nemotron-H keeps bounded Mamba convolution/recurrent state resident while
  paging attention KV, and persists both while explicitly representing
  MLP/MoE-only layers as empty `StateSlots`. Its tensor- and pure expert-parallel routes
  use the same live-paged representation with exact rank topology.
- Realtime Moshi/PersonaPlex temporal/depth session state remains outside schema
  v4. The realtime adapter owns that state in the canonical scheduler and can
  release it for application-level handoff, but persisted timing/depth state is
  not represented by the decoder prompt-cache schema.
- SafeTensors mapping and logical-transfer counters cannot report exact
  physical disk I/O. GGUF additionally reports physical payload read requests
  and bytes issued by its selected-read backend;
  operating-system page caching materially affects disk-backed performance.

The example CLI exposes the common loading policies and their diagnostics. See
its [usage guide](../examples/safemlx-lm-cli/README.md) for concrete commands.

## Parallel execution

The language-model crate contains explicit APIs for pure tensor, pipeline, and
expert parallelism plus tensor + pipeline, tensor + expert, pipeline + expert,
and Qwen3-MoE/GPT-OSS tensor + pipeline + expert execution. A non-replicated topology
must be loaded through the matching API; the ordinary complete-model loader
rejects it. One Cartesian topology owns coordinates and subgroup membership
for every combination. Qwen3-MoE and GPT-OSS triple-axis execution support
complete-layer fully resident and dense-disk-streamed SafeTensors and canonical
GGUF. Their independent expert paths intersect stage-local layers, optional EP
ownership, and optional TP projection shards. With EP inactive a stage owns every routed
expert for its local layers and uses collective-free route recombination;
non-experts may be resident or dense disk-streamed, prompt state remains
rank-local and persistent, and GGUF uses bounded expert reads.
Other families fail architecture preflight before checkpoint payload
materialization until their semantic expert recipes are registered with the
pipeline adapter.

The authoritative family migration backlog and the global versus
family-specific limitations are maintained in the
[combined-topology migration ledger](../safemlx-lm/README.md#authoritative-combined-topology-migration-ledger).

Pure pipeline inference uses the architecture-neutral distributed scheduler.
That canonical runtime owns request/work identity, isolated per-request program
state, bounded stable round-robin queues, cancellation, exact collective work
consensus, failure poisoning, and occupancy/throughput reporting. The decoder
adapter binds each request to one rank-local cache and contributes exact
prefill/decode descriptors, so request ids, work sequence numbers, phases,
batch/sequence dimensions, and mask metadata are compared before point-to-point
traffic. Different requests can occupy different stages concurrently, while
one autoregressive request remains dependent on its preceding sampled token.
Moshi/PersonaPlex uses the same scheduler with temporal/depth state and encoded
or forced-frame work descriptors. It neither inherits nor duplicates the
decoder pipeline's single-hidden-state contract.

Kimi Linear supports pure fully resident and sparse-expert-cache expert
parallelism for SafeTensors: dense/nonexpert weights and the shared expert are
replicated, routed experts are partitioned or loaded through rank-owned sparse
caches, and the shared expert is added once after routed reduction. The
architecture-neutral external expert executor is available for every
registered SafeTensors MoE family. `FullyResident` pins TP-sharded nonexpert
units and every EP-owned routed expert during load;
`WeightResidency::with_expert_cache` composes dense disk-streamed nonexpert
decoder units with an independent rank-owned expert cache. Remote experts are
not materialized under either policy. Tensor
parallelism and pure pipeline parallelism are supported. Pipeline stages load
SafeTensors or canonical `kimi-linear` GGUF with fully resident or dense
disk-streamed local blocks; KDA's three convolution histories plus recurrent
tensor use fixed semantic slots and MLA uses the shared compressed-latent
cache, including paged and persisted prompt-cache routes.

Fully resident GGUF pure expert parallelism is supported for Kimi Linear,
DeepSeek2, Qwen3-MoE, and Qwen3-VL-MoE. Fully resident and sparse-streamed GGUF
TP+EP use the shared catalog and semantic TP plans for Kimi Linear, DeepSeek2,
Qwen3-MoE, GPT-OSS, Inkling, LFM2-MoE, Nemotron-H-MoE, Qwen3-Next,
Qwen3.5-MoE, and Qwen3-VL-MoE. Qwen3-VL GGUF requires its separate multimodal
projection checkpoint; Qwen3.5 discovers an optional family-matching projector.
Both adapters compose language and vision stores before rank-local layer and
expert selection and support stage-local PP+EP where MoE applies.
