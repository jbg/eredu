# Model and checkpoint support

This page summarizes the high-level support implemented by `safemlx-lm`. For
API details and the full layerwise-residency matrix, see the
[`safemlx-lm` README](../safemlx-lm/README.md).

Support is determined from checkpoint metadata and validated configuration,
not from a model's display name. Applications can inspect either artifact
format before loading with `inspect_model`. The compatibility-preserving
`check_model_dir`, `check_model_config`, and `check_model_config_json` helpers
remain available for config-only SafeTensors checks.

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
Nemotron-H/Nemotron-H-MoE, and LFM2/LFM2-MoE GGUF currently
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
text plan with the complete dense vision tower. It validates Conv3D patch and
position embeddings, every vision block, merger and DeepStack parameters,
text/vision geometry, dense-only vision dtypes, tied or untied text heads, and
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
quantization alignment.
Qwen3-Next GGUF validation uses the same metadata and decoder plan while
requiring its fused QKVZ/BA recurrent projection catalog and validating affine
input-group alignment before the loader splits those tensors.
Qwen3-VL GGUF validation combines the exact dense Qwen3 text plan with its
required sibling projector catalog. It validates multimodal RoPE sections and
placeholder tokens, vision-block geometry, both physical patch-convolution
halves, the merger and DeepStack tensors, projector/text hidden-size agreement,
and dense-only projector operation encodings. Inspection and the resident and
bounded loaders consume the same stream-free argument and structural plans.
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
when its separate structural binding is `Ready`. Every currently supported
SafeTensors `ModelKind` and GGUF architecture now has an exact structural
policy; future variants must be added exhaustively and may be marked
`Unverified` only as a fail-closed temporary state.

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
speech-to-speech token API. That API operates on encoded audio tokens; codec
encoding/decoding is provided separately by `safemlx-codec`, and audio device
I/O remains the application's responsibility.

Image preprocessing requires the `safemlx-lm/image-processing` feature. Audio
preprocessing requires `safemlx-lm/audio-processing`. These features are not
enabled by default.

## Capability and request-state estimation

Every high-level `LoadedModel` has an architecture-independent
`capabilities()` report and checked `estimate_runtime_state()`/`admit()` path.
The dispatch is exhaustive over the public `Model` enum, so a new model variant
must explicitly define its context, modalities, and state layout.

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
- `qwen3vl` (with its companion vision projection checkpoint)
- `qwen35` and `qwen35moe`

The tokenizer and chat template are reconstructed from GGUF metadata when
possible. A sibling `tokenizer.json` can supply a tokenizer that is absent from
the file or uses an unsupported embedded tokenizer model. `qwen3vl` requires a
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
Nemotron-H, Qwen2/Qwen2.5, Qwen3, dense Qwen3-VL with its mmproj, Qwen3.5, and Qwen3-Next.
Supported MoE families can cache routed experts independently; for GGUF these
are Kimi Linear, DeepSeek2, LFM2-MoE, Nemotron-H-MoE, Qwen3-MoE,
Qwen3.5-MoE, and MoE Qwen3-Next.

Qwen2/Qwen2.5 text checkpoints share the canonical `architectures::qwen::dense`
decoder with Qwen3. SafeTensors and GGUF support fully resident, layerwise-host,
and dense disk-streamed execution; floating SafeTensors may use supported
load-time affine/MXFP4 quantization. Qwen2 applies the checkpoint-required Q/K/V
biases and layer-selective sliding attention exactly. Qwen2-VL, Qwen2.5-VL,
Qwen2 MoE, and older custom-code Qwen model types are rejected.

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

The residency-reporting API is intentionally canonical rather than retaining
host-only terminology: `LayerwiseModelMetadata` now exposes `residency`,
`layer_parameter_bytes`, `maximum_device_layer_bytes`, and
`device_layer_capacity`. The former host/window-named accessors were removed.

Llama and Mistral now use the same architecture-neutral
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

The migration intentionally removes normalized `ModelArgs.sliding_window`,
direct `ModelArgs` deserialization, `ResidentModel::sliding_window`,
`new_sliding_cache`, the standard/sliding `LlamaCache` split, and the separate
architecture-erased sliding dispatch variants. JSON callers use
`architectures::llama::model::model_args_from_config_value`; execution callers
use `attention_schedule` and the single per-layer-configured device-cache
route.

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

The public architecture-erased variants are `Model::DenseQwen` and
`Model::DenseQwenLayerwise`; the former Qwen3-specific variants and
`architectures::qwen::qwen3` module were removed. The dense-Qwen scalar fields
`use_sliding_window`, `sliding_window`, and `max_window_layers`, plus
`sliding_window_for_layer`, were also removed in favor of the canonical
`attention_schedule`. Direct users now load through
`architectures::qwen::dense::{load_safetensors, load_safetensors_quantized,
load_gguf}` and its `layerwise` module. Every schedule supports resident,
layerwise-host, dense-streamed, ordinary-cache, and paged-cache execution. The
complete ordered schedule participates in architecture and prompt-cache
fingerprints. Resident models support schema-v4 persisted prompt caches for arbitrary ordered Qwen2
full/sliding patterns and exact per-layer windows; all-full Qwen3 uses the same
route. Qwen2 Q/K/V biases are unchanged. Tensor and pipeline parallel Qwen2
remain rejected during topology preflight.

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
layer's retained token interval. Normalized `ModelArgs` no longer implements
`Deserialize` or exposes raw
`layer_types`/`sliding_window`; JSON callers use
`architectures::gpt_oss::model::model_args_from_config_value`.

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
fingerprints include the complete ordered composite schedule. Persisted
Gemma prompt caches, paged Gemma KV caches, and tensor/pipeline parallel Gemma
execution remain unsupported.

This is a breaking API migration: public `LayerType`, raw normalized
`ModelArgs.layer_types`/`sliding_window`, scalar attention/head/KV/shared-layer,
MLP-width, and MoE-enable execution fields, and fallback-returning `layer_type`
were removed. `ModelArgs::layer_schedule` and `ModelArgs::layer_policy` are the
only normalized layer APIs. `ModelArgs` and `Gemma4AssistantConfig` are no longer directly
deserializable; JSON callers use `model_args_from_config_value` and
`gemma4_assistant_config_from_value`. `TransformerBlock::layer_policy` and
`ModelInput::sliding_masks` replace the old scalar/type fields. The public
`CacheStateStrategy::SharedFullKv` now reports `full_attention_layers` plus
`sliding_attention` exact window/count groups instead of one optional global
window.

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
the architecture fingerprint. Persisted Inkling prompt caches remain
unsupported because the persistence schema does not represent its multimodal
prefix and short-convolution state. Tensor and pipeline parallel Inkling remain
unsupported; expert-parallel preflight and cache creation use the exact
schedule.

This is a breaking API migration: normalized `ModelArgs` and `TextArgs` no
longer implement `Deserialize` or expose `local_layer_ids`, `layer_types`,
`sliding_window_size`, `dense_mlp_idx`, or `mlp_layer_types`. The fallback
queries `is_local` and `is_dense` were removed. JSON callers use
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
fallback-free lookup uses `ModelArgs::layer_policy`. This intentionally removes
the normalized `num_dense_layers` field, the model-wide `is_moe` query, and the
old operator-only `LayerPolicy` variants. JSON callers use
`architectures::lfm2::model::model_args_from_config_value`. Persisted LFM2
prompt caches remain unsupported; the complete schedule nevertheless
participates in the loaded-model architecture fingerprint so distinct layouts
cannot share an identity.

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
Normalized `ModelArgs` no longer implements `Deserialize` or exposes
`hybrid_override_pattern`/`sliding_window`; JSON callers use
`architectures::nemotron_h::model::model_args_from_config_value`. Persisted
Nemotron-H prompt caches remain unsupported because the persistence schema does
not represent mixed recurrent and KV state, but the complete ordered schedule
is available in the architecture fingerprint.
`CacheStateStrategy::HybridRecurrent` now reports `full_attention_layers`,
`sliding_attention`, and `recurrent_layers`; the former undifferentiated
`attention_layers` field was removed so hybrid state reports cannot hide
bounded attention groups.

Qwen3.5 and Qwen3-Next normalize Hugging Face `layer_types`, the Qwen3-Next
`full_attention_interval` fallback, and GGUF full-attention intervals into
`LayerSchedule<architectures::qwen::hybrid::qwen3_5::LayerPolicy>`. The ordered
entries distinguish recurrent `LinearAttention` from
`SelfAttention(AttentionPolicy::Full)` and drive resident and bounded execution,
cache construction and validation, structural admission, and exact recurrent/KV
state accounting. Explicit layer lists must match the decoder depth; invalid or
zero interval values fail during normalization. The removed `LayerType`, raw
`ModelArgs.layer_types`, and fallback-returning `ModelArgs::layer_type` APIs are
not retained as compatibility paths. JSON callers use the architecture module's
`model_args_from_config_value` function. `TransformerBlock::layer_type` is now
`layer_policy`, and `Cache::new` returns a validation result.

Qwen3-VL and Qwen3.5 vision blocks use
`LayerSchedule<architectures::qwen::vl::vision::VisionLayerPolicy>`. Each entry
independently identifies `VisionAttentionPolicy::{Full, Windowed}` and an
optional DeepStack merger-bank index. Qwen3-VL Hugging Face and GGUF metadata
normalize to an all-full attention schedule; `deepstack_visual_indexes` is
validated for exact range and uniqueness, and its source order is preserved as
the merger-bank mapping. Qwen3-VL rejects the Qwen3.5-only `window_size` and
`fullatt_block_indexes` fields instead of silently ignoring them. Qwen3.5
normalizes its positive window and exact full-attention block list into the same
policy type.

The schedule length is the sole vision depth after normalization. Resident and
bounded vision execution, DeepStack capture, SafeTensors/GGUF structural plans,
GGUF name translation, and prepared-input workspace accounting all use safe
indexed policy access. Ordered schedule fingerprints include both attention and
DeepStack choices. Qwen3-VL prompt-cache persistence remains unsupported; the
decoder cache cannot be saved independently of its multimodal prefill state.

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
identity. The old normalized `linear_attn_config` layer lists,
`first_k_dense_replace`, `moe_layer_freq`, `is_kda_layer`, and `is_moe_layer`
APIs were removed; callers use `model_args_from_config_value`, `layer_schedule`,
and `layer_policy`. The GGUF loader accepts modern split or legacy unsplit MLA KV-B
projections, singleton convolution layouts, dense/K/IQ tensors, and MXFP4-MoE
type 39. Paged and persisted prompt caches remain unsupported because their
schema cannot represent mixed KDA convolution/recurrent and compressed-MLA
state.

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
identity even though it does not alter MLA state shape. The breaking migration
removes normalized `first_k_dense_replace`, `moe_layer_freq`, direct
`ModelArgs` deserialization, and `is_moe_layer`; JSON callers use
`model_args_from_config_value`, `layer_schedule`, and `layer_policy`.

Qwen3-Next supports the official native fine-grained E4M3 checkpoint format
(`fp8`, dynamic activations, 128 x 128 weight blocks) with fully resident,
layerwise, sparse expert-cache, and pure expert-parallel loading. Fused QKVZ
weights and inverse scales are split without dequantization, dense BF16 BA is
preserved, and routed expert weights remain checkpoint-backed at expert
granularity for sparse-cache and expert-parallel execution.

Important boundaries:

- Prompt-cache schema v4 is intentionally incompatible with schema v3 and older. Its
  ordered architecture-neutral layout preserves full/sliding order, each
  positive window, ordinary KV, DeepSeek compressed MLA, fixed convolution and
  recurrent tensors, multimodal prefix state, tensor geometry, and global
  distributed layer indices. Llama/Mistral, dense Qwen, GPT-OSS, DeepSeek,
  Kimi Linear, Qwen3-Next/Qwen3.5, Gemma 4, Inkling, Qwen3-VL, LFM2, and
  Nemotron-H use this shared representation on resident and bounded-weight
  persistence routes.
- GGUF remains fully resident by default. `LayerwiseHost`, `DenseDiskStream`,
  and supported sparse-expert policies use header-only logical catalogs and
  bounded payload materialization.
- Load-time quantization is incompatible with streamed or sparse-cache loading;
  use a checkpoint-native packed format for those policies.
- Transfers and route inspection are synchronous because the pinned MLX C API
  does not expose the events or fences required for safe cross-stream overlap.
- On Apple silicon, reported host and device residency are logical tiers over
  the same physical unified memory. They do not create additional capacity.
- Parameter budgets do not include activations, KV or recurrent state, kernels,
  allocator caches, checkpoint mappings, or every temporary buffer.
- Kimi KDA and Qwen linear-attention state are persisted as fixed-state tensors,
  not misrepresented as paged KV. Inkling may page attention while keeping its
  convolution histories resident, and schema-v4 publication atomically records
  both parts before reload into an exact resident continuation cache.
- LFM2 persists ordered causal-convolution history and full-attention KV.
  Nemotron-H persists Mamba convolution/recurrent state and attention KV while
  explicitly representing MLP/MoE-only layers as `NoState`. Their pure
  expert-parallel routes persist the same replicated state with exact rank
  topology.
- Realtime Moshi/PersonaPlex temporal/depth session state remains outside schema
  v4 and is intentionally deferred.
- SafeTensors mapping and logical-transfer counters cannot report exact
  physical disk I/O. GGUF additionally reports physical payload read requests
  and bytes issued by its selected-read backend;
  operating-system page caching materially affects disk-backed performance.

The example CLI exposes the common loading policies and their diagnostics. See
its [usage guide](../examples/safemlx-lm-cli/README.md) for concrete commands.

## Parallel execution

The language-model crate contains explicit APIs for pure tensor, pipeline, and
expert parallelism. A non-replicated topology must be loaded through the
matching API; the ordinary complete-model loader rejects it. Hybrid tensor +
pipeline, tensor + expert, and pipeline + expert topologies are not currently
supported.

Kimi Linear supports pure fully resident and sparse-expert-cache expert
parallelism for SafeTensors: dense/nonexpert weights and the shared expert are
replicated, routed experts are partitioned or loaded through rank-owned sparse
caches, and the shared expert is added once after routed reduction. The
architecture-neutral `SparseExpertCacheWithDenseLayers` EP path is available
for every registered SafeTensors MoE family: it disk-streams replicated
decoder units while independently caching rank-owned experts. Tensor
parallelism and pipeline parallelism return capability errors.

Fully resident GGUF expert parallelism is supported for Kimi Linear, DeepSeek2,
and Qwen3-MoE. Sparse-cache GGUF EP uses the shared type-erased expert cache for
those families. The combined streamed-dense policy additionally supports
LFM2-MoE, Nemotron-H-MoE, Qwen3-Next, and Qwen3.5-MoE GGUF checkpoints.
Dense Qwen3-VL GGUF requires its separate multimodal projection checkpoint and
does not expose a pure-EP GGUF adapter.
