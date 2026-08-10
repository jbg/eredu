# safemlx-lm

Native tool calling is documented in
[`NATIVE_TOOL_CALLING.md`](NATIVE_TOOL_CALLING.md), with extension and fixture
guidance in
[`NATIVE_TOOL_CONTRIBUTING.md`](NATIVE_TOOL_CONTRIBUTING.md).

`safemlx-lm` is a Rust runtime for MLX language models.

Use `inspect_model(path, ModelInspectionOptions)` to preflight a local
SafeTensors directory or GGUF checkpoint before allocating model weights. The
structured `ModelInspectionReport` separates container and requested-load
compatibility from architecture support, exact structural binding, tokenizer,
chat, semantic/tool, and multimodal readiness. `is_loadable()` is fail-closed:
it is true only when the selected loader has an exact header/catalog validator;
see [`doc/model-support.md`](../doc/model-support.md) for an example.

The crate is derived from the `mlx-lm` crate in
[`oxideai/mlx-rs`](https://github.com/oxideai/mlx-rs), introduced upstream in
[`oxideai/mlx-rs#281`](https://github.com/oxideai/mlx-rs/pull/281), merged as
commit `7c667cb7`.

The original implementation and authorship belong to the `oxideai/mlx-rs`
contributors.

This fork adds model/runtime support including Gemma 4 loading, Kimi Linear
hybrid KDA/MLA inference, Gemma 4 assistant drafting, expanded model dispatch,
and related generation utilities.

## Persistent checkpoint storage

The public `safemlx_lm::runtime::checkpoint::store` module catalogs SafeTensors and GGUF
checkpoints without materializing tensor arrays. `SafetensorsWeightStore`
accepts a direct payload file, a directory containing `model.safetensors`, or a
Hugging Face-style sharded index. `GgufWeightStore` accepts one or more
validated GGUF checkpoints and translated logical-name catalogs. GGUF payload
readers are opened lazily and retained under the same bounded shard control.

An acquired `WeightLease` pins its mapped bytes. Full tensors, contiguous axis
ranges, and ordered axis indices are selected before the result is copied to a
caller-provided execution stream. `WeightLease::materialize` submits that exact
copy and returns an owning `WeightMaterialization`. The guard exposes
nonblocking query and compatible-stream wait operations, retains mappings and
source arrays through completion, and can be synchronized to take the owned
array. Dropping an unfinished guard waits only for its event, never a whole
stream. Constructing a lazy consumer graph is not enough: insert its stream
wait before evaluating the consumer. Model binding loads retain at most two
submitted materializations and complete the oldest first; a one-shard mapping
bound drains that window before retrying rather than exceeding capacity.
The deterministic mapped-shard cache has a configurable nonzero per-store bound. A live
lease pins its cache entry; if every entry at the bound is leased, acquisition
returns a structured capacity error instead of exceeding the limit.

Rank-aware loading uses the same store and selection implementation. Placement
is resolved before execution-stream materialization, and indexed shards that
contain only remote tensors remain untouched. Cache hits and memory-mapped page
faults are not reported as known physical disk transfers because logical
materialization and storage I/O are different measurements.

`runtime::checkpoint::bounded_quantization` provides the shared out-of-core
materialization path for load-time affine and MXFP4 conversion. A
`BoundedQuantizationPlan` names semantic source recipes and an explicit maximum
conversion working set. Each source is read in bounded row tiles on a CPU
stream, quantized immediately, and written directly into a temporary indexed
SafeTensors store. The store overlays only the final packed `weight`, `scales`,
and optional `biases` tensors; untransformed keys continue to use the original
SafeTensors or GGUF store. A complete dense source tensor and a complete packed
destination tensor are therefore not required in active memory at the same
time. When the working-set ceiling can admit two minimum row tiles, conversion
uses two CPU streams and retains a fixed current-plus-next event window: the
next tile quantizes while the host completes and writes the prior tile. Smaller
ceilings fall back to one tile without exceeding the configured bound.

Admission counts the conservative peak of the selected semantic recipe plus
the packed outputs for that tile. It fails before payload materialization when
even one row does not fit, when the source cannot perform the requested bounded
selection, or when packed geometry and alignment are illegal. Conversion
telemetry reports selected dense bytes, packed output bytes, tile count, peak
in-flight tile count, and the largest admitted source, output, and combined
double-buffered working sets. Downstream
residency planning reads metadata from the packed overlay, so host and device
budgets count the quantized bytes rather than the original dense bytes.

`ModelLoadOptions` selects a compositional parameter-residency policy inside
one generalized execution engine. `LayerWeightResidency` chooses
`FullyResident`, `LayerwiseHost`, or `DenseDiskStream` for ordinary execution
units. `WeightResidency` then either keeps routed experts with those units or
combines `NonExpertWeightResidency` with an independent
`ExpertCacheLoadOptions`. The latter can pin every non-expert parameter, use a
host-backed layer window, or stream non-expert units from disk while routed
experts retain their own cache and budgets. The sum-of-products representation
keeps complete-layer residency distinct from split non-expert/expert ownership.
All ordinary-layer policies
return the same architecture variant and use the same cache, forward,
generation, observation, and capability dispatch. DeepSeek-V3/R1, Gemma 4,
Inkling, Kimi Linear, Llama,
Mistral, GPT-OSS, LFM2/LFM2.5, Nemotron-H, Qwen2/Qwen2.5 text, Qwen3, Qwen3-Next, Qwen3-VL,
Qwen3-VL-MoE, and Qwen3.5 safetensors have registered adapters,
including dense and MoE variants. Moshi and PersonaPlex use the same engine with
independent temporal-layer and depth-codebook-slice windows. A requested family without a registered adapter returns
a specific error and never silently falls back to a separate resident decoder.

Residency is an internal backend choice and is not part of public model
identity.

## Offload planning and observability

Distributed placement decides which tensors a rank owns. Residency is a
separate concern that decides where an owned logical unit lives and for how
long. The public `safemlx_lm::runtime::residency::policy` module provides architecture-independent
configuration, explicit deterministic plans, tier byte totals, and reusable
telemetry. The public `safemlx_lm::runtime::residency::manager` module executes those plans for
caller-defined logical units. Each `OffloadUnit` groups one or more named
checkpoint selections, including companion tensors that must become visible
atomically.

`ResidencyManager` starts disk-planned units without allocations, materializes
host-planned bindings into immutable typed host-transfer buffers, and
materializes device-planned bindings as MLX arrays on an explicit execution
stream. A unit may hold both host and device copies. Dropping a host copy leaves
the checkpoint as its canonical disk-backed source; dropping a device copy
falls back to an existing host copy or the checkpoint. Host and device copies
consume their finite logical budgets independently.

Pinned units cannot be evicted. Windowed units are protected in the active
execution window and preferred for eviction after departure. Cacheable units
remain opportunistically and are evicted by deterministic LRU order, with unit
identifiers breaking ties. RAII `ResidentUnitLease` values explicitly pin the
requested tier while in use. Host leases expose `host_buffer`; device leases
expose `array`. Callers should not retain cloned MLX arrays beyond a device
lease when authoritative residency accounting is required.

Ordinary `LayerwiseHost` prefetch and execution-window lookahead honor
`OffloadConfig::prefetch_depth`, but run synchronously. Experimental dense disk
streaming adds a bounded, joined CPU worker for disk-to-host layer warming.
Dense-stream device promotion uses SafeMLX completion events. A dedicated
same-device transfer stream owns a fixed current-plus-next layer window; the
compute stream waits only on the current layer's completion while the next
transfer may proceed independently. Transfer, stall, eviction, current, and
peak residency observations feed the offload telemetry, while mapped-shard and
process page-fault diagnostics remain separate.

`ResidentLayerGroup` adds named, deterministic ordered-unit preparation and
explicit trimming even under an unlimited device budget. Independent groups
can represent text, vision, audio, temporal, or depth-transformer stacks and
can be cleared without disturbing each other. `LayerwiseModel<A>` and its
canonical `ArchitectureAdapter` contract carry associated input, cache,
heterogeneous runtime-unit, and forward-context types, so retained KV,
convolution, recurrent, and multimodal state is evaluated before a unit lease
is released.

## Experimental dense disk streaming

`WeightResidency::dense_disk_stream(DenseDiskStreamLoadOptions)` keeps ordinary
decoder and execution-stack layers disk planned and array-free at model load.
Its device and host parameter budgets are always finite. Device execution uses
a fixed two-layer transfer window (or one layer for a one-unit execution
group), so the device budget must hold the largest adjacent pair plus pinned
static weights. Host lookahead remains configurable; older cacheable copies
remain resident until deterministic LRU or LFU eviction is needed. A zero host
budget is an explicit direct disk-to-device mode and requires zero host
lookahead and queue capacity. The persistent `SafetensorsWeightStore` remains
the canonical cold source and its mapped-shard cache stays independently
bounded. GGUF uses the
same policy and budget accounting through `GgufWeightStore`; checkpoint-native
affine triples remain packed, outer-axis expert selections issue bounded payload
reads, and the shard control also bounds cached GGUF readers.

This mode is experimental and capacity-oriented. A dense decoder touches
essentially every layer for every token. If neither the logical layer caches nor
the operating system page cache retains the working set, small-token
autoregressive decode can approach reading the checkpoint once per generated
token and may be extremely slow. Background host prefetch can hide some page
fault and materialization latency, but it cannot remove required bytes. The
library promises correctness and bounded logical layer residency, not a
throughput improvement.

Dense disk streaming is mainly appropriate for capacity-first experiments,
large prefill or offline batches, quantized checkpoints, fast local storage,
and workloads that benefit from OS page-cache retention. On Apple silicon, CPU
and GPU arrays share physical unified memory, so logical host/device tiers do
not create additional physical capacity. Pinned embeddings, final norms and
output projections, activations, KV or recurrent state, kernels, allocator
caches, and temporary compact expert banks are outside streamed-layer totals.
Exact physical disk I/O is not observable from mmap telemetry: logical misses,
mapping hits and misses, known logical transfer bytes, and minor/major page
faults are distinct observations.

Direct dense streaming of an MoE checkpoint streams the complete layer,
including its expert bank. To cache experts independently, use
`WeightResidency::with_expert_cache`: select
`NonExpertWeightResidency::FullyResident`, `LayerwiseHost`, or
`DenseDiskStream` for nonexpert units and pass
`ExpertCacheLoadOptions` separately. The expert options contain only expert
budgets and compact-bank scratch controls; they do not duplicate the ordinary
layer policy.

Pure Llama/Mistral and DeepSeek-V3/R1 pipeline stages stream locally: each rank
catalogs its contiguous global layer range and owns a separate worker and
residency report. The first stage alone owns the input embedding, and the final
stage alone owns final normalization and output projection weights.

The `llama_residency` example accepts `--dense-disk-stream` together with
`--stream-host-budget`, `--stream-device-budget`,
`--stream-host-lookahead`, and `--stream-queue-capacity`. It reports load-time residency, first-process and
repeated-process prefill latency, decode latency, cache occupancy, logical
transfers, queue waits, mappings, and available process page-fault samples.
Those labels do not imply physically cold or warm storage, and the example
never drops operating-system caches.

`DerivedWeightRecipe` composes checkpoint selection and renaming,
concatenation, stacking, reshape, axis permutation, and dtype cast. Recipes are
validated from metadata, keep all source leases alive through evaluation, and
materialize the transformed runtime representation on the host before device
promotion. Direct `WeightBinding` construction remains compatible.

CUDA compilation and linkage, including the event integration test target, are
verified by CI. CUDA runtime behavior is not currently exercised because no GPU
runner is configured; local verification in this workspace does not make
discrete-memory capacity or performance claims.

The `safemlx::memory` controls affect process-global MLX-managed allocations.
They do not directly constrain process RSS, checkpoint mappings, or unrelated
native allocations. SafeMLX's patched MLX 0.32.0 surface provides completion
events. Immutable weight residency batches retain their source mappings in a
caller-owned `ResidentTransfer`, and compatible consumer streams use an MLX
event wait rather than a whole-stream handoff.

## Llama-compatible weight residency

`architectures::llama::layerwise::load_llama_model` returns one `LlamaModel`
inference facade. Choose
`LlamaLoadOptions::fully_resident()` for the eager execution-device model or
`LlamaLoadOptions::layerwise_host(...)` for the generic host-backed decoder
engine. Both policies use the same `LlamaCache`, `forward`, `prefill`, `decode`,
and `CausalLm` surface. Dense and already-packed MLX affine/MXFP4 Llama and
Mistral safetensors are supported.

```rust
use safemlx_lm::{
    architectures::llama::layerwise::{load_llama_model, LlamaLoadOptions},
    runtime::execution::layerwise::LayerwiseLoadOptions,
};

let eager = LlamaLoadOptions::fully_resident();
let bounded = LlamaLoadOptions::layerwise_host(
    LayerwiseLoadOptions::new(offload_config),
);
let mut model = load_llama_model(model_dir, bounded, stream, cpu_stream)?;
let mut cache = model.new_cache();
let logits = model.prefill(&token_ids, &mut cache, stream)?;
```

The embedding, final normalization, untied output projection when present,
activations, and KV cache remain on the execution device. Every decoder layer
is initialized on the caller's CPU stream and remains host-resident. Decoder
copies move through a bounded device window whose size is
`OffloadConfig::prefetch_depth()` and includes the current layer. A temporary
unloaded block receives handle clones from its resident lease, executes, and is
dropped after the output and updated cache have been evaluated and the stream
has synchronized. Packed weights, scales, and biases move unchanged; they are
not dequantized or repacked.

Host budgets must contain all decoder weights. Device budgets must contain
pinned static weights plus the largest permitted consecutive layer window.
Residency reports account for parameter copies only; activations, KV state,
kernels, and allocator cache can make MLX peak memory larger.

Checkpoint materialization and promotion use exact MLX completion events;
ordinary layerwise execution keeps its bounded two-layer transfer window.
Registered GGUF checkpoints and bounded load-time affine/MXFP4 conversion from unquantized SafeTensors or
F32/F16/BF16 GGUF sources use the same layerwise residency plan; matching
checkpoint-native packed GGUF tensors load directly and are never implicitly
transcoded. Selecting this weight policy does not enable KV-cache offload.
Weight residency, independent expert caching, and paged KV-cache residency all
use typed host-transfer buffers for their host tiers. Independently owned
request caches can share a `CacheResidencyPool`, which
enforces aggregate device, host, transfer-in-flight, and live-disk limits while
retaining tighter per-request limits. The opt-in
`llama_residency` example accepts a real checkpoint directory and reports
latency, throughput, logical residency, transfer telemetry, allocator samples,
and mapped-shard diagnostics.

## Llama/Mistral attention schedules

Llama and Mistral normalize checkpoint attention metadata into the shared
`LayerSchedule<AttentionPolicy>` geometry. Hugging Face configurations with an
absent or `null` `sliding_window` become all-full; a positive value becomes an
all-sliding schedule with that exact window on every decoder layer. Zero,
negative, and overflowing Hugging Face windows fail during the parser shared by
inspection and loading. GGUF absence or the format-defined zero value means
all-full, while a positive `<architecture>.attention.sliding_window` means
all-sliding. Wrong GGUF types and invalid values fail before weights are
materialized.

After normalization, resident, layerwise-host, dense-streamed, ordinary-cache,
paged-cache, generation, tensor-parallel, pipeline-parallel, fingerprint, and
runtime-state paths query only the ordered schedule. Internally constructed
schedules may freely mix full and sliding layers and may use distinct windows;
ordinary and paged caches are configured independently for each layer. A window
of `N` includes the current token, so a device cache retains at most `N - 1`
past states between calls. Runtime-state estimates count full layers as
context-growing and group bounded layers by exact window.

JSON callers use `model_args_from_config_value`, and execution callers inspect
`attention_schedule`.
Prompt-cache schema v5 persists the complete ordered schedule, exact per-layer
windows, tensor geometry, and retained token intervals. Save/reload therefore
supports non-uniform and distinct-window schedules without weakening cache
identity validation.

## Dense Qwen2/Qwen2.5/Qwen3 weight residency

Qwen2/Qwen2.5 text and dense or sparse-MoE Qwen3 use one
`architectures::qwen::dense` adapter. Token embeddings, final norm, and
the tied or untied output projection stay pinned. Each complete transformer
block, including its routed expert bank, is one `text_decoder` execution unit.
Qwen2 Q/K/V biases are materialized as required tensors. Full-attention layers
grow their GQA KV cache with context; configured Qwen2 sliding layers retain
exactly their declared window. The generic `LayerSchedule<P>` validates an
architecture-defined policy for every decoder layer and provides borrowed,
fallback-free indexed access. Dense Qwen uses
`LayerSchedule<AttentionPolicy>` as its sole normalized decoder geometry after
metadata parsing: each ordered `AttentionPolicy` is either `Full` or
`Sliding { window: NonZeroU32 }`. The schedule accepts arbitrary ordering and
internally supports different windows per layer. Standard KV caches remain
device resident. Matching checkpoint-native affine and MXFP4 parameter trees
load directly. Dense Qwen SafeTensors can also be converted into a disk-backed
packed overlay before residency initialization. Fully resident,
layerwise-host, and dense disk-streamed execution therefore retain only packed
static and layer parameters, and all reported host/device budgets use packed
byte counts. Qwen3-MoE applies the same conversion to non-expert and shared
projections when routed experts use independent residency; routed projections
are converted one expert at a time by the expert overlay.

Hugging Face Qwen2 normalization follows the upstream threshold semantics:
`use_sliding_window=false` produces an all-full schedule; when it is `true`,
`sliding_window` must be positive and `max_window_layers` is the number of
leading full-attention layers, with every remaining layer sliding. Qwen3 remains
all-full. GGUF `qwen2.attention.sliding_window_pattern` is an exact Boolean list
in decoder-layer order (`true` means sliding). Every exact-length pattern is
supported, including alternating and discontiguous patterns. A GGUF window with
no pattern applies to every layer; no window and no enabled pattern means all
full. Enabled layers without a window, invalid windows, length mismatches, and
non-Boolean encodings fail during the shared inspection/load parser.

Callers construct or query `attention_schedule` directly. Ordered policies are included in architecture and prompt-cache
fingerprints. Normal and paged caches support every schedule; schema-v5
persisted prompt caches preserve arbitrary ordered full/sliding policies,
distinct per-layer windows, and each layer's retained token interval. Qwen2
SafeTensors and GGUF tensor parallelism use the generalized family adapter.
Pipeline parallelism supports Llama/Mistral, Qwen2/Qwen2.5, Qwen3, Qwen3 MoE,
and GPT-OSS SafeTensors and GGUF through the same canonical binding and
schedule. Llama/Mistral, Qwen2/Qwen3, GPT-OSS, and Gemma 4 text stages also
compose with TP, while Qwen3-MoE stages compose with EP.

Dense Qwen callers use `architectures::qwen::dense`; architecture identity comes from
validated checkpoint metadata rather than the module selected by the caller.

## GPT-OSS weight residency

GPT-OSS keeps embeddings, final norm, and the output head pinned while complete
sparse decoder blocks move through the `text_decoder` window. The adapter owns
one `LayerSchedule<AttentionPolicy>` containing the exact full/sliding policy
for every layer, plus sink-token mask behavior and RoPE state. Hugging Face
`layer_types` normalizes exactly; when omitted, the published default alternates
sliding and full attention beginning at layer zero. GGUF's required
`gpt-oss.attention.sliding_window` uses that fixed alternating meaning. Invalid
layer entries, length mismatches, and nonpositive or overflowing windows fail
before weight materialization. Checkpoint-native MXFP4 expert blocks and scales
remain packed.

Resident, layerwise-host, dense-streamed, ordinary-cache, paged-cache,
generation, structural, expert-parallel, fingerprint, and runtime-state paths
all consume the canonical schedule. Arbitrary ordering and internally distinct
windows are supported; state reports group sliding layers by exact window and
full layers grow with context. Prompt-cache schema v5 persists the complete
ordered schedule, exact per-layer windows, tensor geometry, and retained token
intervals. JSON callers use `model_args_from_config_value`.

GPT-OSS pipeline stages support SafeTensors and canonical `gpt-oss` GGUF,
including TP+PP with tensor-sharded attention and native MXFP4 expert
intermediates, rank-local KV caches, sharded embedding and output boundaries,
fully resident, host-layerwise, or dense disk-streamed local layers,
ordinary/paged/persisted caches, and rank-synchronized generation. Eligible
dense matrices may be MXFP4-quantized before resident, host-layerwise, or
dense-streamed stage planning; affine and packed-input transcoding fail closed.

## Gemma 4 and assistant weight residency

Gemma 4 normalizes Hugging Face `layer_types` plus `sliding_window`, or GGUF's
exact Boolean `gemma4.attention.sliding_window_pattern`, into one
`LayerSchedule<architectures::gemma4::model::LayerPolicy>`. Each entry includes
the attention policy, exact head/KV geometry, KV local/publish/shared and
key-as-value topology, exact dense MLP width, and dense-only versus
dense-plus-sparse-MoE selection. An omitted or empty pattern is all-full;
enabled sliding layers require a positive executable window. Arbitrary Boolean
ordering is supported, including alternating and discontiguous patterns.
Malformed entries, wrong GGUF encodings, count mismatches, missing windows, and
zero, negative, or overflowing values fail before weights are materialized.
Assistant-prefixed GGUF metadata and assistant JSON use the same normalization.

Hugging Face scalar geometry, shared-suffix, doubled-MLP, key-as-value, and MoE
fields and GGUF metadata are normalized once and do not remain parallel
execution inputs. Resident and layerwise execution, multimodal masks, generation, structural
admission, shared-KV routing, MTP target/assistant handoff, architecture
fingerprints, and runtime-state accounting consume only that schedule.
Multimodal mask maps and shared-KV state are keyed by exact policy, allowing
internally distinct windows. Sliding layers use the exact `N`-position mask
convention while retaining full context-growing KV backing; final shared layers
reuse earlier K/V produced by the same exact policy. Capability reporting gives
the total full-layer count, groups all sliding layers by exact window, and
groups context-growing state by each exact KV-head/head-dimension geometry.
Shared layers have no ordinary cache allocation; publishing layers retain the
full state consumed by matching shared layers.

`ModelArgs::layer_schedule` and `ModelArgs::layer_policy` are the normalized
layer APIs. JSON callers use
`model_args_from_config_value` or `gemma4_assistant_config_from_value`.
`TransformerBlock::layer_policy`, `ModelInput::sliding_masks`, and
`CacheStateStrategy::SharedFullKv::sliding_attention` expose the execution and
reporting state.
The complete ordered policy list participates in architecture identity.
Gemma prompt-cache persistence and safetensors tensor parallelism use the
family layerwise adapter. Tensor-parallel text, vision, and audio layers consume
planner-authored semantic ranges: complete GQA groups, dense and routed
intermediates, vision patch/position channels, audio heads and feed-forward
widths, light-convolution channels, and modality input/output widths may be
uneven across ranks. Packed expert companions remain on the same aligned range,
and prompt-cache identity records the actual rank-local KV geometry. Gemma
text pipeline stages use dependency-safe contiguous placement: a shared-KV
consumer is always colocated with its publisher. Per-layer residual inputs are
prepared once on the ingress stage and relayed as typed immutable auxiliary
state. TP+PP shards the embedding and tied or untied output boundary, per-layer
embedding and projection inputs, attention heads, dense or routed
intermediates, and rank-local publisher caches from the same semantic plan.
SafeTensors and GGUF stages support fully resident, host-layerwise, and
dense-streamed layers.
Image/audio encoder execution is not part of the decoder pipeline API.

## LFM2/LFM2.5 weight residency

Dense and MoE LFM2 variants share one hybrid adapter for full-attention and
short-convolution layers. KV arrays and bounded convolution state are evaluated
before a block lease is released. Public per-expert `w1`/`w2`/`w3` tensors are
concatenated and stacked into runtime expert banks one layer at a time on the
host; already-packed checkpoint representations load directly.

LFM2's authoritative `LayerSchedule<LayerPolicy>` carries both independent
decisions for every decoder layer: `OperatorPolicy` selects convolution or
self-attention, and `FeedForwardPolicy` selects dense SwiGLU or sparse MoE. HF
`num_dense_layers` and GGUF `leading_dense_block_count` are validated
source-format inputs used once to construct the schedule; they are not retained
in normalized `ModelArgs`. Arbitrary internal dense/MoE ordering is supported by
resident, layerwise, dense-stream, sparse-expert, structural, and
expert-parallel paths. `ModelArgs::layer_policy` and
`ModelArgs::layer_schedule_fingerprint` expose the normalized topology. Persisted
LFM2 prompt caches record ordered convolution history and attention KV state.
Tensor-parallel identities preserve fixed convolution state while deriving each
attention layer's exact rank-local KV-head count from the parameter planner.

LFM2 uses the generic `LayerSchedule<P>` runtime geometry. Hugging Face
`layer_types` and GGUF per-layer KV-head metadata
normalize once into `LayerSchedule<architectures::lfm2::model::LayerPolicy>`, whose entries are
`SelfAttention(AttentionPolicy::Full)` or `CausalConvolution`. Resident,
layerwise, structural-admission, cache-validation, and state-accounting paths
consume that schedule directly. Callers with a JSON value use
`architectures::lfm2::model::model_args_from_config_value`.

## DeepSeek-V3/R1 weight residency

DeepSeek keeps embeddings, final normalization, and the output head pinned while
complete MLA blocks move through the `text_decoder` window. Compressed latent and
rotary-key cache arrays are evaluated before each block lease is released. The
dense and routed-plus-shared MoE layers use the same adapter; official
per-expert tensors are stacked per layer for dense, affine, and native 128-by-128
block-FP8 checkpoints. Appended multi-token-prediction non-expert weights are a
pinned static unit with independent compressed caches and execute through the
shared embedded-MTP scheduler; under independent expert residency their routed
banks use the same bounded catalog and route executor as backbone MoE layers.
That catalog identity extends past the backbone layer range and is also the
authoritative EP ownership map for pipeline and triple-axis prediction.

Dense/MoE topology is represented only by
`LayerSchedule<architectures::deepseek_v3::model::LayerPolicy>`, whose entries
are `DenseMlp` or `SparseMoe`. Hugging Face `first_k_dense_replace` and the
positive `moe_layer_freq` follow the checkpoint's zero-based rule: an index is
sparse exactly when it is at or after the dense threshold and divisible by the
frequency. DeepSeek2 GGUF `leading_dense_block_count` normalizes to that same
schedule with every later layer sparse. Invalid counts, negative thresholds,
thresholds beyond decoder depth, and non-positive frequencies fail in the parser
shared by inspection and loading.

JSON callers use
`deepseek_v3::model_args_from_config_value`, then query `layer_schedule` or
`layer_policy`. Resident, layerwise-host, dense-stream, structural-admission,
sparse-expert, tensor/pipeline/expert-parallel, and cache-identity paths use the
ordered policy. Arbitrary internal dense/MoE orders are supported. MLA state
geometry is still model-wide, so ordinary, paged, and persisted caches keep one
compressed latent/rotary entry per scheduled layer; the complete ordered
feed-forward schedule is included in architecture and prompt-cache fingerprints.

## Kimi Linear layer schedules

Kimi Linear normalizes its two independent decoder choices into
`LayerSchedule<architectures::kimi_linear::model::LayerPolicy>`. Every entry
contains `AttentionKind::{Kda, Mla}` and
`FeedForwardPolicy::{Dense, SparseMoe}`. The normalized `ModelArgs` exposes the
ordered `layer_schedule` and shared `kda_config`. JSON callers use
`architectures::kimi_linear::model::model_args_from_config_value` and query
`ModelArgs::layer_policy` or `layer_schedule`.

Hugging Face `linear_attn_config.kda_layers` and `full_attn_layers` are
one-based, must be disjoint, and together must cover every decoder layer.
Feed-forward policy follows the checkpoint's zero-based rule: after
`first_k_dense_replace`, layers whose index is divisible by the positive
`moe_layer_freq` use sparse MoE. GGUF per-layer KV-head metadata, when present,
selects MLA for positive entries and KDA otherwise; its leading dense-block
count is normalized into the same combined policy. Invalid counts, duplicate or
missing indices, out-of-range indices, and invalid prefix/frequency values fail
before weights are materialized.

Resident, layerwise-host, dense-streamed, sparse-expert-cache,
expert-parallel, structural-admission, cache-validation, fingerprint, and
runtime-state paths consume only this schedule. Cache construction selects
bounded KDA recurrent state or context-growing compressed MLA state per exact
entry, and the complete ordered attention/feed-forward policy participates in
architecture identity. Internally constructed schedules may use arbitrary
KDA/MLA and dense/MoE combinations even though current checkpoint formats have
more constrained feed-forward metadata. Paged and persisted prompt caches use
the heterogeneous state schema to represent KDA convolution/recurrent state
alongside compressed MLA state.

## Inkling weight residency

Inkling normalizes its orthogonal attention and feed-forward choices into
`LayerSchedule<inkling::model::LayerPolicy>`. Every ordered entry contains an
exact `AttentionPolicy` plus `FeedForwardPolicy::{Dense, SparseMoe}`. Hugging
Face `local_layer_ids` and exact `layer_types` are accepted as equivalent
attention sources and must agree when both occur; absent either form, the
released five-sliding/one-full cadence applies. `mlp_layer_types` and
`dense_mlp_idx` similarly normalize the feed-forward topology and must agree
when both occur. GGUF consumes the exact Boolean
`inkling.attention.sliding_window_pattern`, its positive shared window, and
`inkling.dense_block_count`; a missing pattern uses the same released cadence.
Arbitrary Boolean ordering is supported; malformed patterns, invalid indices or
entries, conflicts, and invalid windows fail in inspection and load preflight.

Resident, layerwise, dense-streamed, ordinary-cache, paged-cache,
expert-parallel, structural, fingerprint, and memory-accounting paths consume
only the normalized schedule. Internally constructed schedules may use
distinct windows even though current Inkling checkpoint metadata supplies one.
A window of `N` includes the current token: ordinary caches retain `N - 1` past
positions and paged caches enforce the same bound over their backing allocation.
Global and sliding KV arrays plus all four short-convolution states per block
are evaluated before lease release. Released `model.llm` names are rewritten,
short-convolution weights are cast to the runtime dtype, and interleaved dense,
routed, and shared `w13` tensors are selected into runtime gate/up order on the
host. The dMel encoder is a pinned static unit, while the four released hMLP
projection/fold layers use an independent vision window. Typed prompts may
interleave text, discrete audio, precomputed media embeddings, and image
patches.

JSON callers use `inkling::model::model_args_from_config_value` and query
`TextArgs::layer_policy` or `layer_schedule`. The complete ordered schedule participates in
architecture identity. Inkling prompt caches persist rank-local KV state and
all four short-convolution histories using the shared heterogeneous-state
schema. Tensor parallelism covers the text decoder, dMel vocabulary, and hMLP
folded inputs. Pure text pipeline stages support SafeTensors and canonical
`inkling` GGUF through the shared KV-plus-fixed-state cache descriptor, with
fully resident, host-layerwise, or dense disk-streamed local layers. Image/audio
ingress may be direct or scheduler-owned: collective consensus includes its
modality and shape identity, stage zero owns dMel/hMLP preparation, and pipeline
transport connects matching non-PP coordinates. The same semantic bindings
support bounded affine/MXFP4 materialization before resident, host-layerwise,
dense-streamed, independent-expert-cache, and Cartesian execution. Expert
parallelism derives sparse layers and cache policies from the same schedule.

## Nemotron-H weight residency

One hybrid adapter handles Mamba2, attention, dense MLP, and sparse MoE blocks.
Mamba convolution and SSM arrays plus attention KV arrays are evaluated before
lease release. Public `backbone`/`mixer` names are resolved through the same key
rewrite used by eager loading, and split ReLU2 experts are stacked per layer.
Hugging Face `hybrid_override_pattern` and GGUF per-layer operator metadata
normalize once into
`LayerSchedule<architectures::nemotron_h::model::LayerPolicy>`. Its four entry
kinds are `Mamba`, `SelfAttention(AttentionPolicy)`, `DenseMlp`, and
`SparseMoe`; the optional positive `sliding_window` is embedded in every
attention entry. Resident, bounded, structural, expert-parallel, cache,
fingerprint, and memory-accounting paths consume only this schedule. Full
attention uses growing KV state, sliding attention retains its exact window,
Mamba uses bounded convolution/SSM state, and MLP/MoE entries are stateless.
JSON callers use `nemotron_h::model_args_from_config_value`. Persisted Nemotron-H prompt
caches store ordered Mamba convolution/SSM tensors, attention KV, and explicit
stateless MLP/MoE entries. Tensor-parallel manifests use the parameter planner's
exact rank-local Mamba head/group and KV-head geometry, so uneven partitions
save and reopen without scalar division. `CacheResidencyPolicy::Paged` pages
only the context-growing or sliding-window attention blocks; bounded Mamba
convolution/SSM state stays resident. Resident, layerwise-host, dense-stream,
tensor-parallel, and replicated expert-parallel execution all use this same
heterogeneous cache, residency report, and lazy prompt-cache reopen path.
The public `CacheStateStrategy::HybridRecurrent` report separates
`full_attention_layers` from exact `sliding_attention` window groups alongside
`recurrent_layers`.

## Qwen hybrid weight residency

Qwen3-Next and Qwen3.5 share one adapter for recurrent linear attention
and full attention. Hugging Face `layer_types`, the Qwen3-Next
`full_attention_interval` fallback, and GGUF interval metadata normalize once
into
`LayerSchedule<architectures::qwen::hybrid::qwen3_5::LayerPolicy>`. Entries are
`LinearAttention` or `SelfAttention(AttentionPolicy::Full)`; resident,
layerwise, structural-admission, cache-validation, and state-accounting paths
consume only this schedule. JSON callers use
`model_args_from_config_value` from the Qwen3.5 or Qwen3-Next module.
`TransformerBlock::layer_policy` selects execution, and `Cache::new` returns a
validation result.

Qwen3-Next fused QKVZ/BA tensors are selected into runtime
projections without materializing the complete checkpoint, including 128-row
block-space selection for native FP8 QKVZ inverse scales, and public split
SwiGLU experts are packed per layer. The official dynamic E4M3 128 x 128 format
is supported by resident, layerwise, sparse expert-cache, and pure
expert-parallel Qwen3-Next loading. Qwen3.5 dense and packed-MoE checkpoints use
the same block loop. Multimodal checkpoints add an independent Qwen vision-block
group and reuse the resident patch, position, and merger math around that group.

The shared Qwen vision tower also normalizes once into
`LayerSchedule<qwen::vl::vision::VisionLayerPolicy>`. A policy records full or
spatial-window attention plus an optional exact DeepStack merger bank. Qwen3-VL
creates an all-full vision schedule from Hugging Face or GGUF metadata and
preserves the declared DeepStack source order; Qwen3.5 maps its positive
`window_size` and full-attention block indexes into the same representation.
Invalid, duplicate, or out-of-range layer indexes fail before materialization,
and Qwen3-VL rejects Qwen3.5-only window fields. Resident execution, the
independent bounded vision-block group, structural admission, name translation,
and conservative tower-workspace accounting consume only the schedule. Persisted Qwen3-VL prompt
caches store the text KV state together with the multimodal RoPE position delta;
callers must include the processed media and processor settings in the prefix
content fingerprint. Tensor-parallel manifests use the text planner's exact
per-layer rank-local KV-head geometry and record rank ownership on every block.

## Layerwise safetensors coverage

The table records the architecture inventory used by the normal and realtime
dispatch surfaces. “Precise error” means a layerwise request is rejected and is
never replaced by eager loading.

| Family | Eager loader | Layerwise loader | Cache/state | Pinned static modules | Windowed unit | Checkpoint transform / native packing | Parity coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama / Mistral | yes | yes | growing or sliding KV | embedding, norm, head | decoder block | direct affine/MXFP4 | prefill and multi-step decode |
| Qwen2 / Qwen2.5 text | yes | yes | GQA KV, split into full-context and configured sliding layers | embedding, norm, tied/untied head | decoder block | exact Q/K/V biases; direct affine/MXFP4 | full/sliding resident-to-layerwise prefill and decode |
| Qwen3 dense / MoE | yes | yes | growing KV | embedding, norm, head | decoder block with local experts | direct affine/MXFP4 | dense and MoE prefill/decode |
| GPT-OSS | yes | yes | scheduled full/sliding KV | embedding, norm, head | sparse decoder block | native MXFP4 experts | arbitrary schedule, ordinary/paged caches, and multi-step decode |
| LFM2/LFM2.5 dense / MoE | yes | yes | scheduled growing KV or convolution state | embedding, norm, tied/untied head | independently scheduled operator and dense/MoE block | split SwiGLU experts packed for each scheduled sparse layer; packed form accepted | arbitrary dense/MoE schedule plus hybrid prefill/decode |
| DeepSeek-V3/R1 | yes | yes | compressed MLA latent and rotary-key state | embedding, norm, head | MLA decoder block with dense or routed/shared experts | official split experts stacked per layer; direct dense/affine and native block-FP8 banks | dense-to-MoE prefill/decode at two depths; native block-FP8 prefill/decode |
| Kimi Linear | yes | yes | bounded Q/K/V convolution and F32 KDA recurrent state, or growing compressed no-RoPE MLA state | embedding, norm, head | hybrid KDA/MLA block with dense or routed/shared experts | official split experts packed per layer; convolution and transition-state reshaping | prefill/decode primitive, cache, loader, and real-checkpoint smoke coverage |
| Gemma 4 multimodal | yes | yes | scheduled full/sliding KV plus transient shared-KV and media state | patch embedding/pooling, audio subsampling/output, modality projections, token/per-layer embeddings, norm, head | independent vision, audio, and exact-policy text groups | public prefix rewrite; direct affine/MXFP4 text and modality projections | vision/audio/text typed prefill parity; arbitrary-pattern per-layer inputs, shared KV, prefill/decode at two depths |
| Inkling multimodal | yes | yes | global/local KV, four convolution states per layer, transient hMLP activations | dMel embedding/norm, hMLP final norm, text embedding/norm/head | independent hMLP and local/global dense-or-MoE text groups | released-name rewrite, convolution cast, dense/routed/shared w13 deinterleave | audio/text typed prefill parity; local/global and dense/MoE prefill/decode at two depths |
| Nemotron-H | yes | yes | attention KV and Mamba convolution/SSM state | embedding, norm, tied/untied head | hybrid block | public key rewrite and split ReLU2 expert packing | all four block kinds, split MoE, prefill/decode |
| Qwen3-Next / Qwen3.5 | yes | yes | full-attention KV, recurrent linear-attention state, transient vision state | Qwen vision patch/position/merger modules, embedding, norm, tied/untied head | optional vision group plus shared hybrid text group | fused QKVZ/BA selection; split SwiGLU and FP8 expert recipes | Qwen3.5 image/text prefill parity; Qwen3-Next dense/split-MoE and Qwen3.5 dense/MoE prefill/decode |
| Qwen3-VL / Qwen3-VL-MoE | yes | yes | text KV plus multimodal RoPE delta and transient DeepStack state | patch/position embeddings, vision mergers, text embedding/norm/head | independent vision block and dense/MoE text-block groups | direct public DeepStack vision and packed Qwen3 expert trees | image prefill plus multi-step decode for dense/MoE; two depths for dense |
| PersonaPlex / Moshi realtime | yes, realtime API | yes, realtime API | temporal KV plus reset-per-frame depth KV and delayed-stream state | text/audio embeddings, temporal norm and heads | independent temporal layers and per-codebook depth slices | native Moshi layout; released PersonaPlex PyTorch norms, packed attention, embeddings, and projections derived lazily | teacher-forced logits, consecutive realtime frames, offline encoded sequence, forced prompt/cache continuity |

## Linux and CUDA

Enable the `cuda` feature to propagate MLX CUDA support through this crate:

```toml
safemlx-lm = { version = "0.4", features = ["cuda"] }
```

Most model code uses backend-neutral MLX operations. Qwen3.5 MoE's and Kimi
Linear's custom Metal recurrent kernels use portable MLX operation fallbacks on CUDA;
these prioritize correctness and can be slower or use more temporary memory
than the Metal-specialized paths.

## GGUF models

The standard `api::load_model` and `api::LoadedModel::load` entry points
accept Hugging Face-style model directories for Gemma 4, GPT-OSS, Inkling,
Kimi Linear, Llama, dense Mistral,
dense LFM2/LFM2.5 and LFM2-MoE, dense and sparse-MoE Nemotron-H, Qwen2/Qwen2.5 text, Qwen3,
Qwen3-Next, Qwen3-VL, Qwen3-VL-MoE, and dense or MoE Qwen3.5. They also accept the
GGUF architectures listed below. Canonically named sharded GGUF checkpoints
are supported by passing the first
`-00001-of-NNNNN.gguf` shard; the remaining shards are discovered and
validated automatically. Put `tokenizer.json` next to a GGUF file when using
`LoadedModel` or
`load_tokenizer`; adjacent
`tokenizer_config.json` and `chat_template.jinja` files are used when present.

```rust,ignore
use safemlx_lm::api::LoadedModel;

let model = LoadedModel::load(
    "/path/to/model-00001-of-00004.gguf",
    execution_stream,
    cpu_weights_stream,
)?;
```

## Capability and admission checks

`LoadedModel` exposes validated architecture facts without downcasting
`Model`. `capabilities()` reports the configured native/effective context,
input modalities, and cache/state strategy. `count_text`,
`count_prepared_chat`, and `count_prepared_input` distinguish tokenizer IDs
from decoder positions; prepared media positions come from the processor's
patch grids, pooling geometry, or valid-frame masks.

`estimate_runtime_state` models persistent KV, sliding KV, compressed MLA, and
hybrid recurrent state with checked arithmetic. Its result states the assumed
four-byte cache dtype, batch size, every distinct sliding bound, cache growth granularity,
fixed state, context-growing state, and coverage. Standard Llama/Mistral,
Qwen3, GPT-OSS, DeepSeek MLA, LFM2, Nemotron-H, text-only Qwen hybrid, and
Gemma 4 estimates cover persistent state completely. Gemma's shared-KV layers
and full-context backing for sliding-attention layers are accounted separately.
Multimodal models additionally count persistent media embeddings, decoder
positions, and a conservative media-tower execution workspace derived from the
actual prepared tensors. Qwen uses its prepared `grid_thw`, canonical per-layer
vision attention/DeepStack schedule, and merger outputs; Gemma 4 uses padded and
valid patch/audio geometry plus its loaded vision/audio tower configuration;
Inkling uses the released hMLP fold shapes and dMel codebook/mask geometry.
These requests report `Conservative`: decoder state remains exact under the
stated assumptions, while the tower workspace is a complete safe upper bound
for architecture-visible tensors and fused-operation inputs/outputs.

The estimate intentionally remains separate from process-wide MLX allocator
cache, driver allocations, and unrelated application memory. Applications
should retain a safety reserve for those backend and process effects; the MLX
active/cache counters are observational signals, not a request-memory total.

`static_memory()` keeps logical checkpoint/residency accounting separate from
the process-global MLX active and allocator-cache counters. On Apple silicon,
host and device residency are logical tiers over one unified physical
capacity. `available_memory()` uses `hw.memsize` and
`os_proc_available_memory` on macOS, and `MemTotal`/`MemAvailable` on Linux;
Windows uses `GlobalMemoryStatusEx`. Unsupported values are reported as
`Unavailable`, never zero. SafeMLX does not currently expose a trustworthy
CUDA device-free-memory query, so callers on discrete accelerators should pass
an application budget rather than treating host availability as device
capacity.

The admission policy is caller-configurable and does not allocate a model
cache:

```rust,ignore
use safemlx::{Device, DeviceType, ExecutionContext};
use safemlx_lm::api::{
    available_memory, AdmissionRequest, AdmissionResult, ChatTemplateRequest,
    LoadedModel,
};
use serde_json::json;

let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
let mut model = LoadedModel::load(model_dir, execution.stream(), execution.stream())?;

let capabilities = model.capabilities()?;
println!("{capabilities:#?}");

let prepared = model.prepare_chat(ChatTemplateRequest {
    messages: vec![json!({"role": "user", "content": "Summarize this file."})],
    add_generation_prompt: true,
    ..ChatTemplateRequest::default()
})?;
let input = model.count_prepared_chat(&prepared)?;
println!(
    "prompt={} positions, media workspace={} bytes ({:?})",
    input.model_positions,
    input.media_execution_workspace_bytes(),
    input.media_execution_workspace_kind(),
);
let memory = available_memory()?;

let decision = model.admit(
    AdmissionRequest {
        input,
        max_output_tokens: 512,
        batch_size: 1,
        safety_reserve_bytes: 512 * 1024 * 1024,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    },
    Some(&memory),
)?;

match decision {
    AdmissionResult::Admitted(estimate) => {
        println!("admitted: {} incremental bytes", estimate.incremental_required_bytes);
    }
    AdmissionResult::Rejected(reason) => {
        eprintln!("rejected before generation: {reason:?}");
    }
}
```

Dense GGUF tensors are loaded directly. MLX-native packed loading is enabled
for Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0, including checkpoints
that mix packed and dense matrices. Q4_K and Q5_K are losslessly repacked to
MLX's 32-value affine groups, while Q2_K, Q3_K, and Q6_K map exactly to
16-value affine groups. Group-16 K-quants use tiled quantized matrix kernels for
prefill and the corresponding vector kernels for decode. These formats execute
without expanding matrix weights to float16.
Q5_0 and Q5_1 tensors are losslessly repacked into MLX's five-bit affine
layout; unsupported GGUF tensor types return an error. Model dispatch uses
`general.architecture`; the current GGUF adapters support text-only `deepseek2`,
`gpt-oss`, `kimi-linear`, `llama`, `mistral`, `lfm2`, `lfm2moe`, `nemotron_h`,
`nemotron_h_moe`, `qwen2`, `qwen3`, and `qwen3moe` architectures, plus
multimodal `gemma4`, `inkling`, `qwen35`, `qwen35moe`, `qwen3vl`, and
`qwen3vlmoe` with separate projectors; `qwen3next` remains text-only. For
Qwen3-VL or Qwen3.5, put the llama.cpp-style `mmproj-*.gguf` next to the
language-model GGUF. The single-path loaders discover the unique
family-matching projector automatically; explicit Qwen3-VL pairs can use
`architectures::qwen::vl::model::load_qwen3_vl_gguf`.
Nemotron-H routed expert banks retain Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 packed weights
and execute through selected-expert quantized matrix multiplication. Qwen3 MoE
uses the same packed expert-major execution with per-tensor mixed Q2/Q3/Q4/Q5/Q6/Q8
settings. Dense Qwen3.5 uses the hybrid linear/full-attention runtime with
conventional SwiGLU layers; Qwen3.5 MoE keeps its
Q2_K/Q3_K/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0 routed expert banks packed while loading mixed
quantization types. Gemma 4 dense and MoE text weights support fused or separate
GGUF expert projections, native packed affine execution, external MTP assistant
files, and a sibling dense media-projector GGUF. Nemotron-H latent-space MoE and
Omni/multimodal checkpoints remain separate formats. Quantized Qwen3-VL and
Qwen3.5 language GGUFs retain their supported packed weights; their shared Qwen
vision semantic plan accepts dense projector tensors and canonical GGUF Q8
projection matrices, including TP-sharded stage-zero execution. Qwen2-VL,
Qwen2.5-VL, Qwen2 MoE, and older custom-code Qwen
architectures are intentionally unsupported.

Qwen2 GGUF follows GGUF's absence-means-full-attention rule and consumes an
optional Boolean `qwen2.attention.sliding_window_pattern` exactly in layer
order. Alternating and discontiguous patterns are supported. A declared window
without a pattern applies globally. Pattern/type/window conflicts and
attention-affecting unsupported YaRN metadata are rejected during the same
catalog inspection used by loading. Fully resident dense-Qwen models support
normal, paged, and schema-v5 persisted prompt caches for full or mixed
Qwen2/Qwen2.5 schedules. The normalized schedule can carry distinct windows,
and all-full Qwen3 uses the same persistence route. Qwen SafeTensors and GGUF
tensor parallelism use the generalized family adapter. Pure dense-Qwen
pipeline stages support Qwen2/Qwen2.5, Qwen3, and Qwen3 MoE in both formats
with the same schedule, prompt-cache identity, and direct or derived
layer-binding plan.
Qwen3-VL SafeTensors and canonical `qwen3vl` or `qwen3vlmoe` GGUF use the same
pipeline shell for dense and MoE text decoders. GGUF combines the main
checkpoint and sibling mmproj store before rank-local selection. Stage-zero vision ownership,
typed ingress, MRoPE/DeepStack payload construction, and TP- or EP-local text
layers remain adapter-authored semantics rather than format- or
topology-specific runtimes.
Qwen3.5 uses the same Qwen vision groups with its hybrid decoder. Canonical
`qwen35`/`qwen35moe` GGUF discovers and structurally validates an optional
sibling `clip`/`qwen3vl_merger` projector, then adds it to the same bounded
weight store before resident or rank-local selection.

Gemma 4 GGUF discovers a unique nearby `mmproj-*.gguf`, preferring a filename
marked F16, BF16, or F32, or accepts an explicit pair through
`architectures::gemma4::model::load_gemma4_gguf_with_mmproj`. The projector uses
`general.architecture = clip`, Boolean `clip.has_vision_encoder` and
`clip.has_audio_encoder`, and `gemma4` vision/audio projector types. Typed
`clip.vision.*` and `clip.audio.*` geometry reconstructs the native tower
configuration; placeholder and media-boundary token IDs remain `gemma4.*`
metadata in the language GGUF. Projector tensors retain the released
`vision_tower.*`, `embed_vision.*`, `audio_tower.*`, and `embed_audio.*` names.
The loader translates these into the ordinary Gemma module tree, validates the
complete catalog and dense F16/BF16/F32 encoding before materialization, and
combines both files in one bounded-read store. Resident, host-layerwise,
dense-streamed, TP, PP, cached decode, and queued multimodal execution therefore
use the same semantic adapter as SafeTensors.

Kimi Linear GGUF accepts split `attn_k_b`/`attn_v_b` and combined
`attn_kv_b` projections, vector and singleton-ranked convolution tensors, dense
and supported K/IQ formats, and type-39 MXFP4-MoE expert banks. GGUF
`ssm_a = -exp(A_log)` is validated and converted back to canonical `A_log`.
Fully resident, bounded-layer, dense-streamed, and sparse-expert residency
policies share this conversion path.
Embedded `tokenizer.ggml.pre = kimi-k2` uses the official Han-aware
pre-tokenization expression.

GPT-OSS GGUF uses llama.cpp's canonical `gpt-oss` metadata and type-39 MXFP4
routed experts. Split gate/up tensors are reassembled into the checkpoint-native
interleaved expert layout without dequantizing; dense projections retain their
per-tensor GGUF formats. Inkling GGUF follows the draft llama.cpp `inkling`
text contract, including relative-position attention, four short-convolution
streams, split dense/routed/shared SwiGLU tensors, padded-vocabulary masking,
and the `inkling` o200k-family tokenizer pre-type. A sibling llama.cpp-style
`mmproj-*.gguf` supplies the combined hMLP vision and dMel audio towers. It is
discovered automatically or can be passed explicitly to
`architectures::inkling::model::load_gguf_with_mmproj`; dense and supported
checkpoint-native packed projector matrices work with resident, bounded,
sparse-expert-cache, streamed-dense, and streamed expert-parallel execution.

GGUF IQ tensors are also model-loadable: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S,
IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, and IQ1_M. Their nonlinear codebooks cannot be
represented faithfully by MLX affine weights/scales/biases. Safemlx therefore
retains the original GGML blocks as `uint8` device parameters. Linear,
embedding, and grouped routed-expert operations decode codebook values directly
inside Metal kernels, without a persistent dense or affine copy. Resident
storage stays at the GGUF encoding's native bytes-per-weight. Prefill kernels
reuse each decoded block across tiles of eight activation rows. Routed IQ
experts fuse gate/GELU/up and fuse down projection, route weighting, and
reduction, avoiding intermediate dense expert weights. Q4_K, Q5_1, and Q8_0
checkpoint-native linear paths use the same activation-row tiling where
applicable.

CPU execution streams one
packed logical row into F32 scratch and immediately consumes it; scratch memory
is therefore proportional to the tensor width rather than its full element
count. These direct kernels preserve the packed-memory advantage, but they
remain correctness-first implementations and may trail highly tuned dense or
llama.cpp kernels for some shapes. The
`safemlx/benches/native_quantization.rs` benchmark reports packed and dense-F16
decode/prefill medians:

```console
cargo bench -p safemlx --bench native_quantization
```

Names such as IQ2_M, IQ3_M, UD-IQ2_M, UD-Q2_K_XL, and UD-Q3_K_M describe
mixed-precision file recipes rather than extra tensor type codes. Recipe files
are tensor-format compatible when each contained tensor uses a supported
encoding. This includes UD-IQ2_XXS, UD-IQ2_M, UD-Q2_K_XL with IQ4_NL,
UD-IQ3_XXS, UD-IQ3_S, UD-Q3_K_M or UD-Q3_K_XL with IQ4_NL, UD-IQ4_XS, and
UD-IQ4_NL. Codes 36-38 (`IQ4_NL_4_4`, `IQ4_NL_4_8`, and
`IQ4_NL_8_8`) are not canonical GGUF encodings and are rejected.

## Usage

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["image-processing"] }
```

### Executable pipeline parallelism

Runtime parallel topology is configured independently of a model's
`config.json`. `ParallelTopology` uses pipeline-major, tensor, then expert rank
ordering (expert is the fastest-changing coordinate). The process-local device
index is always explicit: a global rank identifies a process in the distributed
group and must not be reused as a local GPU index.

```rust,ignore
use safemlx::{distributed::{self, Backend}, DeviceType, Stream};
use safemlx_lm::{
    architectures::distributed::pipeline::{
        load_pipeline_model_with_options, PipelineInferencePhase,
        PipelineInferenceScheduler, PipelineMicrobatchInput, PipelineStep,
    },
    runtime::scheduler::{RequestId, SchedulerLimits},
    DeviceAssignment, ModelLoadOptions, ParallelTopology,
};

let group = distributed::init(true, Backend::Ring)?;
let topology = ParallelTopology::from_group(
    &group,
    1, // tensor-parallel size
    2, // pipeline-parallel size
    1, // expert-parallel size
    DeviceAssignment::new(DeviceType::Gpu, local_device_index),
)?;
let stream = Stream::new_with_device(&topology.device.device()?);

let options = ModelLoadOptions::default().with_parallel_topology(topology);
let mut model = load_pipeline_model_with_options(
    model_dir,
    options,
    &stream,
    cpu_weights_stream,
)?;
let request = RequestId::new(42);
let mut scheduler = PipelineInferenceScheduler::new(
    &model,
    SchedulerLimits::default(),
)?;
scheduler.register_request(&model, request)?;
let step = PipelineStep::new(1, prompt_length)?;
let input = PipelineMicrobatchInput::new(
    request,
    PipelineInferencePhase::Prefill,
    step,
);
scheduler.enqueue(if group.rank() == 0 {
    input.with_tokens(prompt_tokens)
} else {
    input
})?;
let output = scheduler.run_queued(&mut model, &group, &stream)?;
let completion = &output[0];
completion.synchronize()?; // Exact microbatch event, not the whole stream.
let logits = completion.logits(); // Some only on the final stage.
```

Pipeline execution is explicitly asynchronous. `forward_pipeline`,
`prefill_pipeline`, their Cartesian variants, and scheduler microbatch outputs
return owning completions which submit the relevant lazy MLX graphs. A
completion covers receives, declared auxiliary payloads, stage cache updates,
sends, lane barriers, final logits, and embedded-MTP hidden state for that one
transition. It does not capture unrelated lazy graphs or work submitted later.
Call `wait_on` before evaluating dependent work on another compatible stream,
`is_complete` for a nonblocking query, or `synchronize`/`into_logits` for an
exact host wait. Multiple compatible stream waits are supported. Dropping a
completion while producer work or consumer waits remain outstanding is safe;
host synchronization is required to observe asynchronous backend errors.

The architecture-independent `DistributedCompletion<T>` applies the same
contract to direct Cartesian pipeline sends and receives. This replaces the
former duplicated `eval` plus whole-stream synchronization paths. Pipeline
events are local backend completions: cross-process timing and cross-backend or
incompatible-device waits are not supported.

Scheduled multimodal prefill uses the same queue. Convert borrowed typed input
with `PreparedModelInput::from_model_input`, or use the owned value returned by
a model processor. Every first-stage TP/EP coordinate submits
`with_prepared_input`; later pipeline stages submit the matching
`with_prepared_input_identity`. The scheduler owns the stage-zero payload until
execution, while downstream queues retain only ordered modality, payload-kind,
dtype, shape, and metadata-shape identity. These identities participate in the
same global work consensus before any pipeline send or receive. Combined modes
drain with `run_queued_cartesian`; decode continues to use rank-local token
ingress through the same request cache.

Pipeline execution requires `PP > 1`. The Cartesian planner derives TP
collectives, EP exchanges, and matching-coordinate pipeline lanes from one
topology. Llama/Mistral, DeepSeek-V3/R1, Kimi Linear, Qwen2/Qwen3, Qwen3-VL,
GPT-OSS, LFM2, Inkling, Nemotron-H, Qwen3-Next/Qwen3.5, and Gemma 4 TP+PP
use tensor-sharded stage layers and vocabulary boundaries; DeepSeek-V3/R1,
Inkling, Kimi Linear, Qwen3-MoE, Qwen3-VL-MoE, GPT-OSS, LFM2-MoE,
Nemotron-H-MoE, Gemma 4 MoE, and Qwen3-Next/Qwen3.5-MoE PP+EP keep only the stage
rank's assigned routed experts. DeepSeek preserves its compressed MLA cache
and split, packed-affine, native block-FP8, or GGUF expert recipes. GPT-OSS
preserves its checkpoint-native MXFP4 expert banks and scheduled full/sliding
attention cache, LFM2 preserves its alternating convolution/attention state,
Inkling preserves full/sliding KV state plus all four short-convolution histories,
Nemotron-H preserves rank-local Mamba
convolution/recurrent state plus GQA cache, and the Qwen hybrid adapter
preserves rank-local recurrent convolution/delta-rule state plus full-attention
KV state when composing stage placement with TP or EP ownership.
Qwen3-VL pins its vision patch/position modules, mergers, and vision blocks on
stage zero for every non-PP coordinate. Typed image/video ingress is assembled
there; explicit MRoPE tensors, the persisted position delta, and
sequence-aligned DeepStack residuals then use the ordinary immutable pipeline
payload. Later stages execute only their text-layer ranges.
Qwen3.5 uses the same ownership rule through the shared hybrid semantic
adapter: stage zero owns its patch/position modules, merger banks, and vision
blocks, then sends the assembled text-width sequence through matching TP/EP
pipeline lanes. Vision and recurrent/full-attention decoder units share one
resident, host-layerwise, or dense-streamed plan; dense and MoE decoders use
the same scheduled `PreparedModelInput`, cached decode, generation, and
prompt-cache lifecycle.
Gemma 4 likewise places configured vision and audio roots on stage zero for
every TP/EP coordinate. Pooled image/video and subsampled audio outputs are
assembled through the shared Gemma execution-group adapter; per-layer inputs
and exact full/sliding multimodal masks then travel as declared immutable
pipeline auxiliary tensors. Tower and decoder blocks share one resident,
host-layerwise, or dense-streamed rank-local plan.
Inkling places its folded hMLP vision root, dMel embedding and normalization,
text embedding, and media final normalization on stage zero for every TP/EP
coordinate. Ordered text/image/audio input can be submitted directly or as a
scheduler-owned `PreparedModelInput`; downstream stages retain only its exact
modality and shape identity. Matching-coordinate pipeline lanes carry the
assembled decoder hidden state, while full/sliding KV and all four
short-convolution histories remain stage-local. SafeTensors and canonical
`inkling` GGUF plus a sibling combined hMLP/dMel mmproj use the same resident,
host-layerwise, or dense-streamed semantic plan.
`PipelineModel::prefill_pipeline` and `PipelineModel::prefill_cartesian` perform
typed prefill, after which cached decode, synchronized sampling, cancellation
consensus, and prompt-cache persistence use the common pipeline APIs.
Combined pipeline modes accept `LayerwiseHost` and `DenseDiskStream`. Each
non-resident layer unit is built from the same semantic plan as resident
execution after composing stage placement with its rank-local TP shard or EP
expert selection. `LayerwiseHost` eagerly materializes every rank-local layer
on the host and promotes a bounded ordered window to the execution device;
`DenseDiskStream` leaves those units cold behind independent host and device
caches. SafeTensors and GGUF reads therefore materialize only the local recipe,
while residency and checkpoint diagnostics distinguish pinned stage-boundary
weights from planned non-resident layer bytes.
`PipelineModel::parameter_residency_report` exposes rank-local host/device
placement, transfer, eviction, and active-window telemetry for either
non-resident policy; `dense_stream_report` adds disk-stream pass and cache
statistics only when that policy is active. `PipelineStageInfo` carries the
same report's global rank and TP/PP/EP coordinates and stage ownership.
DeepSeek-V3/R1, Qwen3-MoE, Qwen3-VL-MoE, Kimi Linear, Inkling, GPT-OSS,
LFM2-MoE, Nemotron-H-MoE, Gemma 4 MoE, and Qwen3-Next/Qwen3.5-MoE also compose all
three axes in this runtime. TP
collectives remain inside the stage and EP coordinate, routed exchange remains
inside the stage and TP coordinate, and pipeline transport connects equal
TP/EP coordinates.
The triple-axis path supports arbitrary valid Cartesian degrees on native
subgroup backends and uses topology-planned neighbor routes for stage-local
Ring fallback collectives. Fully resident, host-layerwise, and
dense-disk-streamed SafeTensors and canonical `deepseek2`, `qwen3moe`, `qwen3vlmoe`, `kimi_linear`, `inkling`,
type-39 `gpt-oss`, `gemma4`, `lfm2moe`, `nemotron_h_moe`, `qwen3next`, or `qwen35moe` GGUF checkpoints share the same layer recipes,
cache identity, synchronized generation, and failure consensus. LFM2-MoE carries its
TP-local causal-convolution state and attention KV state through that same
stage cache, while Nemotron-H-MoE carries TP-local Mamba convolution/SSM state
alongside attention KV state, including prompt-cache persistence and reload.
The same independent expert cache works with PP and TP+PP when EP is inactive,
and with PP+EP or TP+PP+EP when EP is active.
In that geometry every stage owns all routed experts for its local layers; TP
coordinates retain semantic projection shards, and route execution uses the
shared collective-free singleton dispatch rather than an artificial EP group.
`ArchitectureAdapter::expert_parallel_assignment`,
`ArchitectureAdapter::new_cartesian_layer`, and
`ArchitectureAdapter::cartesian_layer_bindings` are the shared semantic
boundary for pipeline-local units. The default composition selects ordinary,
TP-local, or EP-local hooks from topology inputs; each family owns its expert
ownership, bank geometry, checkpoint naming, packed companions, and bounded
expert selection. The pipeline runtime does not inspect family layer enums or
expert parameter names.
Unsupported family/policy combinations fail from metadata before checkpoint
payload materialization. The ordinary `Model` loader remains a complete
single-device API and directs non-replicated requests to the explicit pipeline
loader.

### Authoritative combined-topology migration ledger

This table is the persistent source of truth for family migration status. A
family is **complete** only when every applicable pairwise topology and the
generic TP+PP+EP path support prefill, cached decode, synchronized generation,
prompt-cache persistence, resident, host-layerwise, and dense-streamed
non-experts, plus independently cached experts in both registered checkpoint
formats. Dense families have no EP migration.

| Family | Axes | Status | Remaining family-specific work |
|---|---|---|---|
| Qwen2/Qwen3 dense | TP, PP | TP+PP complete | None; EP does not apply |
| Qwen3 MoE | TP, PP, EP | Complete | None |
| GPT-OSS | TP, PP, EP | Complete | None; native MXFP4 SafeTensors and canonical type-39 GGUF are covered |
| Llama/Mistral | TP, PP | TP+PP complete | None; EP does not apply |
| Gemma 4 dense / MoE | TP, PP, EP | Complete | None; routed MoE layers support EP, TP+EP, PP+EP, and TP+PP+EP for SafeTensors and text-plus-projector GGUF, with typed text/image/video/audio ingress, both media towers, cached decode, synchronized generation, prompt-cache persistence, and resident, host-layerwise, or dense-streamed non-experts |
| DeepSeek-V3/R1 | TP, PP, EP | Complete | None; split, packed-affine, native block-FP8 SafeTensors and canonical DeepSeek2 GGUF recipes share the Cartesian path |
| Kimi Linear | TP, PP, EP | Complete | None |
| LFM2 dense | TP, PP | TP+PP complete | None; EP does not apply |
| LFM2-MoE | TP, PP, EP | Complete | None |
| Nemotron-H dense | TP, PP | TP+PP complete | None; EP does not apply |
| Nemotron-H-MoE | TP, PP, EP | Complete | None |
| Qwen3-Next/Qwen3.5 dense | TP, PP | TP+PP complete | None; Qwen3.5 SafeTensors and canonical text-plus-projector GGUF support direct or queued typed image/video ingress with stage-zero TP-sharded vision ownership and all three non-expert residency policies; EP does not apply |
| Qwen3-Next/Qwen3.5-MoE | TP, PP, EP | Complete | None; Qwen3.5 SafeTensors and canonical text-plus-projector GGUF multimodal ingress compose with TP+PP, PP+EP, and TP+PP+EP plus resident or independently cached experts |
| Inkling multimodal | TP, PP, EP | Complete | None; direct or queued text/image/audio ingress, stage-zero hMLP and dMel ownership, SafeTensors and canonical text-plus-mmproj GGUF, cached decode, synchronized generation, prompt-cache persistence, and resident, host-layerwise, dense-streamed, or independently cached expert policies share the Cartesian runtime |
| Qwen3-VL dense | TP, PP | TP+PP complete | None; vision is pinned on stage zero and direct or queued typed image/video ingress composes with arbitrary legal TP/PP degrees; EP does not apply |
| Qwen3-VL-MoE | TP, PP, EP | Complete | None; vision is pinned on stage zero and direct or queued typed image/video ingress composes with arbitrary legal TP/PP/EP degrees |
| Moshi/PersonaPlex | TP | Not applicable | Its temporal/depth runtime has no PP or EP constituent axis |

The remaining global limitations are:

- Route inspection remains a necessary host synchronization. Dense layerwise
  and pipeline execution now use a dedicated transfer stream and a two-layer
  completion-lease window, allowing the next weight transfer to overlap current
  layer computation. Background disk-to-host warming remains bounded and may
  still stall demand. Paged-cache attention now uses a dedicated transfer
  stream and a fixed current-plus-next block window; same-device promotions are
  event-ordered, while a dedicated host-demotion worker retains and charges
  both source and destination allocations until its exact events complete.
  Callers wait only when a finite budget requires the source to be released.
  Expert prediction/prefetch and broader activation double buffering remain
  future work. MTP same-device handoffs are event-ordered independently.
- Load-time affine/MXFP4 conversion applies to semantically eligible grouped
  matrix projections. Convolution kernels, normalization vectors, position
  tables, and projection geometries that do not meet the selected group and
  packed-kernel alignment remain dense; they are not silently padded or
  rewritten into a different operator.

- The shared bounded materialisation store supports out-of-core affine and
  MXFP4 conversion from row-bounded SafeTensors or dense GGUF semantic recipes.
  Static modules, ordinary layers, shared experts, independent expert caches,
  pipeline stages, TP+EP, and TP+PP+EP now transform the authoritative
  rank-local expert catalog rather than rebuilding topology arithmetic: EP
  ownership, TP projection ranges, and the conversion row tile compose into
  one bounded source span. The conversion budget is capped by the final packed
  local semantic recipes, or by one unavoidable aligned semantic row when a
  synthetic model's entire packed stage is smaller than one quantization row;
  executing that projection already requires at least the corresponding row
  workspace. Ordinary residency plans are constructed only after the packed
  overlay exists, so pinned static bytes, layerwise-host budgets, dense-stream
  host/device windows, and expert-cache budgets all count packed bindings.
  `LayerwiseModelMetadata::materialization`,
  `PipelineStageInfo::materialization`, `ResidencyReport::materialization`, and
  `ExpertCacheReport::materialization` expose selected source bytes, output
  bytes, tile counts, the admitted ceiling, and the peak planned working set.
- Fixed-width F32/F16/BF16 GGUF tensors support bounded row/range, indexed, and
  reshaped contiguous-span reads. EP ownership, TP row placement, and
  conversion tiles may therefore compose over a fused dense bank without
  reading the complete bank. Standalone host-layerwise and dense disk-streamed
  GGUF execution uses this same overlay, including independently cached routed
  experts in families that expose standalone expert-cache execution; residency
  admission and runtime windows count only the final packed bytes. The
  contiguous-span type cannot be constructed for packed GGUF
  encodings, and quantizing an already quantized checkpoint is intentionally
  unsupported. Inkling and Nemotron-H now share the affine/MXFP4 grouped rank-3
  expert primitive with the other MoE implementations. Their packed semantic
  adapters, high-level load dispatch, independent expert-cache construction,
  and pipeline-stage construction use the shared packed materialization
  overlay. SafeTensors and dense GGUF conversion compose with standalone EP,
  PP, PP+EP, and TP+PP+EP ownership; route-empty ranks participate without
  materializing experts, and cache reports expose the selected packed format.
  Complete rank-3 routed and shared banks are visited one matrix and bounded
  row tile at a time while their expert-major output geometry is retained, so
  fully resident conversion no longer requires an independent expert cache.
  The packed overlay derives one atomic weight/scale/bias naming layout from
  each target runtime binding, including the underscore companions used by
  grouped expert banks. Qwen3-MoE SafeTensors and every registered dense-GGUF
  MoE adapter therefore use the same complete-bank path for fully resident,
  host-layerwise, and dense-streamed execution without selecting independent
  expert residency. This includes DeepSeek-V3/R1, Kimi Linear,
  Qwen3-Next/Qwen3.5-MoE, Qwen3-VL-MoE, LFM2-MoE, Gemma 4 MoE, and GPT-OSS.
  Inkling and Nemotron-H support affine and MXFP4 conversion for every eligible
  text target. Nemotron-H applies ReLU2 between two shared packed grouped
  projections; MXFP4 scale-only banks use the same route ordering, empty-route,
  cache, and Cartesian ownership contracts as affine weight/scale/bias banks.

TP+PP+EP is executable for DeepSeek-V3/R1, Qwen3-MoE, Qwen3-VL-MoE, Kimi
Linear, Inkling, GPT-OSS, Gemma 4 MoE, LFM2-MoE, Nemotron-H-MoE, and
Qwen3-Next/Qwen3.5-MoE, including Qwen3.5 multimodal SafeTensors and canonical
text-plus-projector GGUF. The Cartesian topology and execution contexts remain
family-neutral and accept arbitrary legal axis sizes.

Decoder layers use balanced contiguous placement. Architectures may declare
unsplittable dependency units; Gemma 4 uses this to keep each shared-KV
publisher with all of its consumers, and preflight rejects a stage count that
cannot receive nonempty legal units. Stage zero owns token embedding and its local
layers. Intermediate stages own only their local layers and constants. The last
stage owns its local layers, final normalization, and the language-model head.
For tied Llama and Gemma weights, the embedding table is present only on stage zero and
the last stage. DeepSeek routed and shared experts stay with their decoder
layer; expert banks for remote layers are filtered before packing.

Indexed safetensors placement is resolved before payload files are opened, so
remote-only shards are skipped and remote tensors never become MLX arrays.
Quantized companions remain colocated. Dense and supported prequantized
safetensors are supported for Llama-compatible and dense-Qwen models. DeepSeek supports its
official split-expert safetensors, native block-FP8 and affine layouts, and
local expert-bank packing. Requested on-load quantization is applied only to
selected local tensors. Gemma 4 accepts exact SafeTensors catalogs and dense or
checkpoint-native quantized GGUF text tensors through the same derived-binding
plan used by bounded execution. Gemma text stages can requantize dense
SafeTensors and compatible dense GGUF tensors to affine or MXFP4 storage,
including derived fused-expert bindings, when operation dimensions satisfy the
requested group alignment. Nonresident pipeline stages build the packed overlay
before host/device budgets are planned. Matching checkpoint-native encodings
load directly and packed formats are never implicitly transcoded.

Qwen2/Qwen2.5, Qwen3, and Qwen3 MoE use their shared structural binding plan
for SafeTensors and canonical GGUF. Resident, host-layerwise, and dense-streamed
stages support aligned affine or MXFP4 load-time quantization, exact Qwen2 Q/K/V
biases, Qwen3 Q/K
norms, GQA, tied or untied heads, routed experts, and every normalized
full/sliding layer policy.
Kimi Linear, Nemotron-H/Nemotron-H-MoE, Inkling, and Qwen3-Next/Qwen3.5 artifacts also
use the generalized pipeline runtime for SafeTensors and canonical GGUF. KDA,
Mamba2, and linear-attention state is represented by semantic fixed slots; MLA
and full/sliding attention use the shared compressed or KV cache. These
families support fully resident, host-layerwise, and dense disk-streamed local
stages, ordinary and paged live caches, persisted prompt caches, and
synchronized generation.

GPT-OSS SafeTensors and GGUF stages use its canonical split/fused expert binding
recipes; TP+PP partitions GQA heads and 32-channel-aligned packed expert
intermediates from the same semantic plan used by pure TP. Native MXFP4 experts
remain packed, while eligible resident dense matrices may be quantized to
MXFP4 on load. Qwen3 MoE likewise uses the shared dense-Qwen stage and canonical
expert binding recipes rather than a separate pipeline implementation.

LFM2 and LFM2-MoE stages accept exact SafeTensors catalogs and canonical
`lfm2`/`lfm2moe` GGUF checkpoints. Fully resident, host-layerwise, and dense
disk-streamed stages use the shared LFM2 binding plan. LFM2-MoE additionally
composes independently resident/cached stage-local experts with PP, TP+PP,
PP+EP, or TP+PP+EP; a dense-only stage represents external expert ownership
without manufacturing an empty cache. Causal-convolution histories are
materialized as descriptor-backed state slots, full-attention layers use the
same ordinary or paged KV implementation as other decoders, and paged prompt
publication saves both representations atomically. A stage containing only
convolution layers still owns a residency manager, so fixed-state-only stage
snapshots are valid and reload with their exact prefix offset.

Nemotron-H-MoE composes the same Cartesian execution and expert-storage
contracts across its ordered Mamba, dense, sparse-MoE, and attention schedule.
Resident routed experts combine EP exchange with TP-sharded expert widths;
independently cached experts remain complete execution units while shared
experts and nonexpert projections retain their TP shards. SafeTensors and
canonical `nemotron_h_moe` GGUF use stage-local expert catalogs, so host and
dense-streamed policies do not inspect or materialize remote-stage expert
payloads. Empty-route coordinates, prompt-cache reload, synchronized
generation, and failure consensus use the shared pipeline runtime.

Qwen3-Next/Qwen3.5-MoE uses that same Cartesian and expert-storage
contract. Full-attention KV and recurrent convolution/delta-rule state remain
ordinary rank-local stage cache entries; resident experts combine TP-sharded
intermediates with EP ownership, while independent expert caches compose with
fully resident, host-layerwise, or dense-streamed non-experts. SafeTensors and
canonical GGUF use the shared packed/split expert catalog. Qwen3.5 multimodal
SafeTensors and canonical GGUF add stage-zero vision units to that same
residency schedule and accept direct or scheduler-owned image/video tensors.
GGUF discovers a unique sibling Qwen3.5 projector, verifies its family,
geometry, complete catalog, and packed alignment before materialization, then
combines it with the language artifact in one bounded-read store. Dense and
canonical GGUF Q8 projector matrices use the same semantic TP recipes; optional
local processor sidecars provide decoded image/video preprocessing.

`PipelineCache` contains only the local global-layer range: standard or
sliding-window KV entries for Llama, dense Qwen, and GPT-OSS;
compressed-latent entries for DeepSeek; and descriptor-backed `StateSlots` for
fixed convolution, recurrent, prefix, and position state. KV and compressed
entries can carry the same ordered slots, so hybrid families do not add cache
variants. Empty `StateSlots` explicitly represent stateless or non-owning
layers such as Gemma shared-KV consumers.
Cache reuse and reset are explicit. Every stage recreates causal mask state from
the shared `PipelineStep` and its local cache offset; explicit masks must be
supplied consistently by every rank.

Hidden stage input and output use `PipelinePayload`, which carries hidden
activations plus an ordered `PipelineAuxiliaryState`.
`PipelineCache` is one architecture-tagged container of semantic
`PipelineLayerCache` entries (`StateSlots`, `KeyValue`, or
`CompressedLatent`); an empty `StateSlots` entry is the stateless
representation. `PipelineModel`
likewise owns one type-erased stage shell rather than an architecture enum.
Every decoder payload runs through the same local-layer executor and load
accumulator, using the corresponding generalized architecture adapter as the
sole source of static and layer bindings. Static units are selected lazily by
stage role, and streamed binding preparation does not count cold shards as
materialized. Family semantics supply only decoder math, auxiliary payload
shapes, and one canonical prompt-cache identity; transport, cache
materialization, persistence, residency, and sampling contain no family
dispatch. `PipelineModel::new_cache` is fallible so a future adapter cannot
silently coerce an unsupported semantic state layout.

Each individual microbatch receives from its predecessor, executes local
layers, and sends to its successor. Lazy point-to-point arrays are submitted
with exact completion events at each boundary; host observations wait only for
the associated arrays, while compatible downstream streams use backend-ordered
event waits. The canonical
`runtime::scheduler::FairScheduler` owns request/work identity,
bounded stable round-robin queues, isolated program state, lifecycle,
cross-rank consensus, poisoning, and telemetry. `PipelineInferenceScheduler`
is its decoder adapter: the program state is an independent `PipelineCache`
and the work descriptors are prefill/decode transitions. After one collective
exact-descriptor preflight, stage zero can advance to a later request while
downstream stages finish an earlier one, filling the pipeline without combining
cache state. Different prompt lengths are supported. Batch size remains fixed
within one request, decode work has sequence length one, and prefill cannot
resume after decode begins.

Queue capacity and active-request capacity provide explicit backpressure.
Cross-rank differences in request id, work sequence, phase, dimensions, or mask
metadata fail before point-to-point traffic. Any execution failure poisons the
scheduler and releases its caches rather than permitting an ambiguous retry.
EOS and cancellation discard queued work and release the corresponding cache;
an idle cache can instead be released for prompt-cache persistence. Scheduler
reports expose current/peak queue occupancy, request count, completed and
failed/discarded work, terminal counts, drain cycles, and poison state.
`PipelineInferenceScheduler::cache_pool_report` separately reports aggregate
cache ownership and high-water marks. Cancellation and EOS release a request's
pool contribution; a cache released for persistence remains charged while the
caller owns it.

Realtime and other programs reuse `FairScheduler` by supplying program state,
a work payload implementing `WorkDescriptor`, and an executor closure.
Single-process realtime drains locally; distributed programs additionally
supply a versioned protocol identity for exact rank consensus. They do not need
pipeline cache, prefill/decode, or hidden-state semantics, and do not duplicate
identity, queueing, cancellation, or accounting. Callers use `RequestId`,
`WorkId`, `SchedulerLimits`, `RequestStatus`, and `SchedulerReport` from the
canonical scheduler module.

The realtime API is `RealtimeInferenceScheduler`; request-owned samplers,
PRNG state, delayed streams, and temporal/depth caches live together in
`RealtimeSession`, while normal and forced prompt frames use `RealtimeStepInput`.

Logits stay on the last stage. `sample_and_synchronize` samples only there,
then all ranks enter the same collectives for the small token id and EOS/stop
flag. Sampling policy is intentionally separate from scheduling: after each
drain, callers enqueue the synchronized token or finish the request on every
rank. A single autoregressive request remains token-dependency limited;
pipeline utilization comes from multiple requests or teacher-forced chunks.

Pipeline training/backward is not supported. Moshi/PersonaPlex uses its
coupled temporal/depth program adapter
over the same generic scheduler rather than the decoder pipeline's
single-hidden-edge adapter. Pipeline GGUF loading is registered for `llama`,
`mistral`, `deepseek2`, `gemma4`, `qwen2`, `qwen3`, `qwen3moe`, `gpt-oss`,
`qwen3vl`, `qwen3vlmoe`, `lfm2`, `lfm2moe`, `nemotron_h`, `nemotron_h_moe`, `qwen3next`, `qwen35`,
`qwen35moe`, `kimi-linear`, and `inkling`. Support is not inferred for other
GGUF architectures.

A checkpoint's DeepSeek `ep_size` remains checkpoint layout/compatibility
metadata and retains its existing validation. Runtime
`expert_parallel_size` only describes this inference job; it does not override
or reinterpret the checkpoint field.

The partition and executable pipeline Ring proofs are opt-in:

```sh
cargo test -p safemlx-lm --test distributed_partition_ring \
  ring_two_process_partition_load -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_pipeline_microbatch_scheduler -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_dense_stream_pipeline_microbatch_scheduler -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_pipeline_schedule_mismatch_fails_closed -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_deepseek_pipeline_persistence -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_gemma_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_multimodal_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_multimodal_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_multimodal_host_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_multimodal_gguf_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_multimodal_gguf_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_multimodal_gguf_host_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring \
  ring_gemma4_moe_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring \
  ring_gemma4_moe_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_moe_pipeline_expert_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_moe_triple_axis_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_moe_streamed_pipeline_expert_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_moe_host_triple_axis_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_gemma4_multimodal_pipeline_ring \
  ring_gemma4_moe_gguf_triple_axis_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_qwen2_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_qwen3_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_qwen3_moe_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_gpt_oss_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_lfm2_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_lfm2_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_lfm2_moe_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_kimi_linear -- --ignored --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_nemotron_h_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_qwen3_next_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_qwen35_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring \
  ring_two_process_inkling_dense_stream_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_qwen3_vl_pipeline_ring \
  ring_qwen3_vl_pipeline_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_qwen3_vl_pipeline_ring \
  ring_qwen3_vl_dense_stream_pipeline_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_qwen3_vl_pipeline_ring \
  ring_qwen3_vl_tensor_pipeline_multimodal -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_qwen3_vl_pipeline_ring \
  ring_qwen3_vl_moe_pipeline_expert_multimodal -- --ignored --exact --nocapture
```

The Kimi command runs both SafeTensors and canonical GGUF fixtures. These
adapter proofs compare two-rank prefill and decode logits with the ordinary
resident loader, validate stage-local semantic state, and assert bounded layer
residency. The GGUF case also checks physical read telemetry before layer
execution, proving that neither rank reads the complete tensor payload while
loading its static modules.

See `cargo run -p safemlx-lm --example pipeline_generate -- MODEL_DIR` for the
minimal rank-aware prefill/decode probe. Launch one process per stage with the
Ring environment (`MLX_RANK` and `MLX_HOSTFILE`) configured for all processes.

### Execution-group dependency graphs

The generalized resident/layerwise engine executes a validated
`ExecutionGroupDag`. Every adapter declares stable group names, named input
dependencies, and one authoritative output; the runtime validates missing or
duplicate names, unknown or repeated dependencies, cycles, disconnected
groups, residency-plan identity, and group lengths before materialization.
Ready groups run in a stable topological order. Root groups receive the initial
activation, single-input groups receive their dependency output, and adapters
with multiple inputs must implement an exact merge instead of relying on
numeric adjacency.
Completed activations are retained only until their final declared consumer,
so a chain does not accumulate every prior group output and a branch keeps only
the roots still needed by its merge.

Gemma multimodal execution declares independent vision and audio roots which
both feed the text decoder. Qwen3-VL, multimodal Qwen3.5, and Inkling declare a
vision-to-text edge. Their media jobs are finalized and assembled only when the
text node becomes ready, so skipped media branches and future non-adjacent
placement cannot change ingress semantics. Ordinary text decoders declare a
single-node graph; Moshi/PersonaPlex declare temporal-to-depth dependencies.
Scheduling is currently topological and serial, not concurrent across ready
roots.

`ArchitectureAdapter::execution_graph`, `begin_execution_group`, and
`complete_execution_group` define group topology and execution boundaries.
`LayerwiseModel::execution_graph` exposes the exact validated graph used by the
loader and executor.

### Executable tensor parallelism

Tensor parallelism uses `TP > 1` and composes with PP, EP, or both through the
same Cartesian execution context.
Use the selected architecture's `load_*_tensor_parallel_model` entry
point (or `load_tensor_parallel_safetensors` for Qwen2/3) with a
`ParallelBuildContext`. For TP+EP DeepSeek-V3/R1, Inkling, Kimi Linear, Qwen3,
Qwen3-VL-MoE, GPT-OSS, LFM2-MoE, Nemotron-H-MoE, or Qwen3-Next/Qwen3.5-MoE,
use the expert loader and
`ExpertParallelModel::forward_cartesian`; the shared layerwise plan shards
nonexpert projections, attention heads or recurrent operators, caches,
embeddings, and the output head across TP while the EP subgroup owns routed
expert work. For TP+PP Llama/Mistral, DeepSeek-V3/R1, Kimi Linear, Qwen2/Qwen3,
Qwen3-VL, GPT-OSS, LFM2, Nemotron-H, Qwen3-Next/Qwen3.5, or Gemma 4, use the pipeline loader
and `PipelineModel::forward_cartesian`. Tensor parallelism is implemented by the
generalized layerwise execution-group engine; there is no
architecture-dispatching TP model wrapper.

The generalized engine accepts three explicit ordinary-parameter policies
through `LayerWeightResidency`: `FullyResident`, `LayerwiseHost`, and
`DenseDiskStream`. `FullyResident` uses the same rank-local planner, cache
layout, collectives, and logits gathering as the bounded policies, but creates
and populates every locally owned layer module once during loading and pins all
local parameters on the execution device. It does not retain a second
architecture-specific resident decoder. `ParallelModelInfo` reports unsharded
global parameter bytes, total rank-local parameter bytes, and the rank-local
subset permanently pinned on device, plus the maximum planned local device
footprint. `LayerwiseModelMetadata` reports the selected policy and its maximum
layer-device footprint. Current bounded-policy residency remains available
through `ResidencyReport`.

All-to-sharded (column-parallel) projections take a complete replicated input,
shard weight/output rows, and keep the output local without communication.
Sharded-to-all (row-parallel) projections take a final-feature input shard,
compute a full-width partial result, all-sum it, and add ordinary linear bias
once after reduction. Affine quantization `biases` are per-group metadata and
stay with their packed weight shards; they are distinct from ordinary linear
bias. Row partitions must align with the affine group, the MXFP4 32-value
group, or DeepSeek's 128-by-128 block-FP8 grid.

Family-specific generalized loaders cover Llama/Mistral, Qwen2/Qwen3
(including Qwen3 MoE), Gemma 4 text and multimodal towers, GPT-OSS,
DeepSeek-V3/R1, Kimi Linear, LFM2
dense/MoE, Nemotron-H, and the text decoders of Qwen3-Next and Qwen3.5
dense/MoE. Qwen vision towers use their own execution groups. DeepSeek,
Inkling, Nemotron-H, and Qwen embedded multi-token-prediction modules retain
family-specific layer math while sharing one transactional draft scheduler.

`load_tensor_parallel_layerwise_model` applies the same typed parameter-role
planner and rank-local checkpoint selection to named layerwise execution
groups. Family entry points are provided for Qwen3-VL, Gemma 4, Inkling,
Moshi, and PersonaPlex. Qwen vision shards QKV heads and MLP intermediates;
Gemma vision shards GQA heads and MLP intermediates; Gemma audio additionally
shards light-convolution channels and computes its RMS statistic across ranks;
Inkling row-shards each folded hMLP input; and Moshi/PersonaPlex shard temporal
and within-frame depth attention heads, caches, and MLP intermediates.
Checkpoint selection happens before host or device materialization. Composite
text decoders use the same family planners as token-only execution. Token,
audio-codebook, and output vocabularies use balanced row ranges; media mergers,
per-layer-input projections, patch projections, and modality projections use
column/row contracts with their required gather or reduction. Gemma vision
position rows and audio input/output projections are rank-local as well.
Quantized Inkling hMLP and Moshi affine companions stay attached to their
packed weight shards. Dense disk streaming operates on rank-local bindings for
every execution group, while pinned static modules retain only their planned
local partitions. Scalars, routers, expert identities, and convolution stems
without a valid feature-local decomposition remain replicated deliberately.

Load-time affine and MXFP4 materialization also covers aligned Qwen3-VL vision
attention, MLP, merger, and DeepStack projections; Gemma 4 vision/audio
attention, feed-forward, relative-position, subsampling, output, and bridge
projections; Inkling hMLP projections and its dMel embedding. The semantic
adapters select these matrices before residency planning, so fully resident,
host-layerwise, dense-streamed, and Cartesian policies budget only their packed
weights and companions. Convolution kernels, normalization vectors, position
tables, and geometrically unaligned patch projections stay dense because they
are not coherent grouped-matrix quantization targets.

Llama/Mistral, dense Qwen, Inkling, Kimi Linear, LFM2 attention, Nemotron-H
attention, GPT-OSS, and Gemma 4 text/vision/audio projections are sharded. RoPE, attention,
activations, and gated products remain local; only attention output and MLP
down projections all-sum. Llama/Mistral, dense Qwen, LFM2, Nemotron-H, and
GPT-OSS and Inkling partition balanced ranges of whole GQA KV groups, so query/KV head
counts need not divide TP and every rank still preserves the exact query-to-KV
ratio. Kimi Linear independently partitions whole KDA or MLA heads. GPT-OSS
learned sinks follow their local query heads. Dense feed-forward intermediates
may also be uneven.
Quantized row ranges combine adjacent
head groups or intermediate channels as needed to align affine, MXFP4, and
checkpoint-native GGML blocks; packed weights and companions consume the same
logical range.
Inkling additionally assigns each dense or routed/shared expert intermediate
one atomic range across its input projections, fused gate/up segments, down
projection, and quantization companions. Its K/V depthwise convolution
channels follow the local KV groups, and each hMLP layer slices the exact
planner-derived folded-input range rather than reconstructing equal rank
chunks. Every rank must receive at least one legal unit. Families not described here retain
their documented exact-divisibility constraints. Each cache contains only its rank's
local K/V heads, including bounded sliding-window caches, and persisted cache
descriptors record the possibly different per-layer local counts. Qwen3-VL
reuses that planner-derived text-cache geometry while retaining its
architecture-owned multimodal position state. Qwen3 MoE and LFM2-MoE
retains every routed expert and the router on every rank while assigning a
balanced logical intermediate range to each rank. The same group-level range
selects both fused gate/up segments, the down-projection input, packed weights,
scales, and quantization biases. Expert intermediate widths therefore need not
divide TP, although every rank must receive at least one complete dense or
quantization-aligned unit.

Gemma 4 uses one planner contract across its related text, vision, and audio
components. Text and vision attention assign balanced ranges of whole GQA KV
groups and retain the corresponding query heads and reduced output columns.
Dense gated-GELU and routed gated-GELU intermediates are separate domains;
packed expert weights, scales, and biases follow the same aligned range.
Vision patch projections and position tables share one hidden-channel range.
Audio attention heads, its paired feed-forward blocks, fused gated
light-convolution input, depthwise kernel/norm, and reduced output consume
authoritative semantic domains; global RMS statistics remain correct when
channel counts differ by rank. Audio and vision modality projections use exact
planner-derived ranges, and persisted prompt caches record local KV heads
without reconstructing them from the rank count.

DeepSeek keeps Q-LoRA input, compressed KV latent projection/normalization,
and routing replicated. One balanced, potentially uneven MLA-head domain owns
the expanded query, fused or split key/value, and reduced output projections,
so every rank's local decoder geometry comes from the planner rather than
integer division of global fields. Compressed-latent caches remain rank-local
and work for both prefill and absorbed decode. Dense MLP, routed-expert, and
shared-expert intermediates use independent balanced domains. Routed and
shared experts retain all expert identities on each rank, while packed weights,
scales, quantization biases, and native block-FP8 companions share the same
aligned local intermediate range; each residual delta is all-summed once.

GPT-OSS assigns complete GQA groups and each retained MXFP4 routed expert's
intermediate dimension through balanced planner ranges. Learned attention sinks
follow their query heads; the router remains replicated. Native expert blocks,
scales, and fused gate/up biases share one 32-channel-aligned intermediate
range, while ordinary output/down bias is added once after reduction. GPT-OSS
TP accepts its native MXFP4 checkpoint representation rather than converting
the expert banks to another load-time format. KV and persisted prompt-cache
geometry comes from the actual local plan, including uneven per-rank head
counts. Two-rank SafeTensors and native MXFP4 GGUF fixtures verify numerical
parity, bounded GGUF reads, and save/drop/reload continuation.

Kimi assigns balanced, potentially uneven MLA and KDA head ranges. One KDA
logical domain jointly partitions Q/K/V, decay/update/gate expansions, all
three depthwise kernels, transition parameters, recurrent state, and the
row-reduced output projection. MLA head-expanded projections share a second
domain while compressed latent state remains head-independent. Dense, routed,
and shared-expert intermediates use independent balanced domains; affine and
MXFP4 down-projection boundaries combine channels into complete quantization
groups before partitioning. KDA's low-rank input bottlenecks and per-head
normalization remain replicated. Rank-local cache descriptors are derived from
the actual KDA plan, and two-rank mixed KDA/MLA fixtures verify all three
parameter-residency policies plus save/drop/reload parity. LFM2 shards complete
GQA groups, balanced dense or routed SwiGLU intermediate ranges, and gated
short-convolution channels. The fused B/C/x projection, depthwise kernel,
bounded convolution state, and row-reduced output projection share one planner
range. Official per-expert safetensors are range-loaded for the local
intermediate slice and then packed into the runtime expert banks, so a rank does
not materialize the global expert intermediate tensors first. Prompt-cache
manifests combine planner-derived per-layer KV and convolution-channel geometry.

Qwen3-Next and Qwen3.5 shard full-attention heads and dense/routed/shared MoE
intermediates together with recurrent key/value heads, convolution channels,
transition parameters, and recurrent state. Qwen3.5's fused query/gate rows
and Qwen3-Next's checkpoint-fused recurrent projections use segmented placement
rules so logical components retain their local geometry before runtime
splitting. These are independent balanced domains: ranks may own different
numbers of complete GQA/recurrent groups and different routed, shared, or dense
intermediate widths. Layer construction and prompt-cache manifests consume the
planner's semantic ranges directly, including packed FP8, affine, MXFP4, and
native GGML companions; they do not infer a shard count from physical tensor
shapes.

Nemotron-H uses its authoritative four-operator schedule to shard Mamba heads,
B/C groups, convolution channels, and SSM state; attention heads; dense MLP
intermediates; and routed/shared ReLU2 expert intermediates. Layer norms,
routers, and expert identities remain replicated, and every operator performs
one row-parallel reduction for its residual delta. A complete Mamba B/C group
is the atomic logical unit: all of that group's heads remain attached across the
fused input segments, depthwise convolution, transition/state parameters,
gated RMSNorm, output projection, and fixed cache. GQA, dense MLP, routed
expert, and shared-expert widths use independent balanced domains. Affine and
native GGML packed boundaries combine adjacent units when required, then apply
the identical semantic range to packed values and companions. Two-rank
SafeTensors coverage exercises fully resident, layerwise-host, and dense-stream
policies plus prompt-cache reload; a matching GGUF fixture verifies numerical
parity and bounded rank-selective reads.

Resident caches work for all of these text models. Kimi and Nemotron-H also
support heterogeneous live paging: growing MLA blocks and Nemotron attention
KV blocks are `SealablePaged`; small KDA, Mamba, Qwen, LFM2, and Inkling
convolution histories are `AlwaysDeviceMutable`; and the large KDA, Mamba, and
Qwen linear-attention matrices are `LayerScopedOffloadable`. Fixed-state policy
uses a narrower sum type without a paged variant, preventing mutable state from
being constructed as sealed attention storage. The same schema-v5 per-layer
descriptors persist these behaviors and lazily reopen paged blocks with exact
rank-local topology. Embedded attention-bearing draft caches join the manager.
LFM2 supports rank-aware persisted prompt-cache snapshots and restores them
into its heterogeneous resident cache.

Embedding and output rows use balanced contiguous vocabulary ranges, including
uneven vocabulary sizes. Embedding masks out non-local ids then all-sums hidden
states. `forward_local_logits` returns the local range; `forward`, `prefill`,
and `decode` pad and gather shards along the final logits axis. Sampling occurs
only on the designated rank, which alone mutates sampler/PRNG state, and only
the selected token and stop flag are synchronized.

Rank-aware safetensors selection happens before execution-device
materialization. Indexed payload shards with no local tensors are not opened.
Dense, MLX affine/MXFP4, DeepSeek native block-FP8, official split-expert, and
local on-load quantization paths are supported subject to alignment. Registered
GGUF families use the same planner to select bounded rank-local ranges before
materialization; unsupported encodings or packed ranges that cannot meet their
quantization alignment are rejected during preflight.

Ring is useful for correctness testing. Practical low-latency TP should use
JACCL or NCCL where available. Run the collective proof with:

```sh
cargo test -p safemlx --test distributed_ring \
  ring_two_process_loopback -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_tensor_parallel -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_deepseek_tensor_parallel_persistence -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_deepseek_gguf_tensor_parallel -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_nemotron_tensor_parallel -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_tensor_parallel_ring \
  ring_two_process_nemotron_gguf_tensor_parallel -- --ignored --exact --nocapture
```

Architecture-neutral planner tests and family-owned rank-local loader fixtures
run with the library test suite:

```sh
cargo test -p safemlx-lm --lib -- --nocapture
```

The model-level probe is:

```sh
cargo run -p safemlx-lm --example tensor_parallel_generate -- MODEL_DIR
```

Launch every rank with the same token input and configured `MLX_RANK` and
`MLX_HOSTFILE`; only rank zero prints generated tokens in the example.

Dense safetensors checkpoints and unquantized F32/F16/BF16 GGUF checkpoints can be affine- or
MXFP4-quantized while loading through the same architecture-dispatched API used for ordinary
loading:

```rust,ignore
use safemlx_lm::{
    models::{LoadedModel, ModelLoadOptions},
    quantization::{AffineQuantization, WeightQuantization},
};

let affine = ModelLoadOptions::with_quantization(WeightQuantization::Affine(
    AffineQuantization::new(64, 4)?,
));
let mxfp4 = ModelLoadOptions::with_quantization(WeightQuantization::MxFp4);
let model = LoadedModel::load_with_options(model_dir, mxfp4, stream, weights_stream)?;
```

The realtime counterpart is `load_realtime_model_with_options`. Both APIs
recognize matching pre-quantized checkpoints and load them directly rather
than quantizing them again. A requested format that differs from existing
checkpoint metadata is an error.

### Multi-token prediction

The `mtp` module provides lossless greedy or stochastic speculative decoding
through architecture-neutral backend, cache-transaction, and sampling-policy
contracts. `LoadedModel::mtp_capability` reports whether drafting weights are
external, embedded, pending an adapter, or unavailable. Gemma 4 resident and
bounded-layer targets accept typed multimodal prefill and an explicit
safetensors or GGUF `LoadedDrafter`.
Gemma assistant GGUF files may embed their JSON config in the
`safemlx.mtp.config` metadata string or provide a sibling `config.json`.
External Gemma assistants may execute through `MtpExecutionStreams` on the
target stream, a second stream on the same GPU, or a stream on another device.
The constructor classifies these as `Single`, `SameDeviceSplit`, or
`CrossDeviceSplit`. Distinct same-device streams share immutable MLX array
storage through backend-ordered completion-event waits without blocking the
host; only cross-device topology performs host-synchronized physical array
copies. Target state, stochastic draft roots, embedding snapshots, and draft
distributions all use that same-device event contract. `MtpScheduler` is the
canonical engine: requests move through explicit prefill, draft,
verification-submission, in-flight lookahead, resolution, and terminal phases.
Verification submission leaves the lazy target graph unresolved. On a distinct
stream the scheduler then drafts one optimistic continuation block, or gives a
round-robin turn to another ready request, before reading target results.
`MtpSchedulerOptions` bounds retained verifications and branches; both default
to one. `MtpSchedulerOptions::with_lookahead(false)` disables all same-request
branch work while leaving the canonical verification, sampler, and target-PRNG
path intact for equivalent A/B runs.

An optimistic Gemma branch shares immutable target shared-KV maps and MLX array
handles while owning its small token delta, exact processed draft
distributions, exact assumed-prefix copy, and assistant progress. Full
acceptance samples and emits the usual target bonus token. When it matches the
first optimistic token, that token is consumed by the bonus and the remaining
token/distribution pairs are promoted without recomputation. The shortened
block is extended from its exact assistant frontier before verification so its
boundary matches ordinary non-lookahead execution. A mismatch, rejection, EOS,
or cancellation drops the branch and commits only canonical state.

Target acceptance and draft sampling use disjoint per-request roots. Draft
randomness is addressed by logical output position rather than scheduler
operation order, so consuming a matching optimistic token, discarding a
mismatch, changing request interleaving, or temporarily exhausting branch
slots cannot move later draft draws. The target cache contains only evaluated
inputs: an emitted bonus remains uncached and leads the next verification.
Fully accepted bonus-emitting rounds commit their entire evaluated input block
without truncating KV storage. Partial rollback moves only the logical frontier
of chunked KV caches, preserving allocated backing capacity so the abandoned
suffix can be overwritten in place.
`MTP_OPTIMISTIC_LOOKAHEAD.md` gives the complete cache, assistant-state,
sampler, RNG, match, mismatch, and terminal transition invariants.

`generate_mtp_text_batch_with_cache_and_streams` submits every independent lane
to the same starvation-free scheduler and returns results in input order.
Single-request APIs are one-request scheduler wrappers. `MtpStats` reports
optimistically drafted, bonus-consumed, reused, and discarded tokens/blocks;
optimistic target bonuses and their non-terminal matches/mismatches; plus
scheduler turns, adaptive-lookahead status, and cross-request draft
opportunities. It also reports optional branch wall time and total
submission-to-resolution intervals for performance diagnosis. After four
resolved branches by default, deterministic adaptive accounting disables
future branch work when no proposal has been reused or when reused proposal
tokens fall below discarded proposal tokens. The consumed matching bonus token
is excluded from both sides. This policy only suppresses optional drafting and
cannot change canonical output.
`MtpBatchOutput::scheduler` reports aggregate turns and peak retained
transactions. A matched first token counts as consumed, not reused and not an
ordinary proposal; only retained tokens promoted into the canonical proposal
block contribute to `draft_tokens`.

Optimistic continuation currently requires an external Gemma assistant,
distinct target/draft streams, and a sampler whose cloned draft processing and
sampling are exact functions of explicit history, immutable configuration, and
the supplied position-addressed PRNG state. Same-GPU split streams are
deliberately opt-in: target and assistant kernels share one device and may
contend for compute or memory bandwidth, so measured throughput can improve or
regress. Cross-device execution can instead overlap otherwise independent
devices but pays synchronization and transfer costs. `MtpStats::stream_topology`
and `MtpBatchOutput::scheduler.stream_topology` report the active assignment.
Mirostat V2 remains lossless and reproducible but does not look ahead because
its next truncation depends on target-committed adaptive state. Embedded
predictors use the same request scheduler and independent lane semantics
without same-request lookahead. DeepSeek-V3/R1, Inkling, Nemotron-H,
Qwen3-Next, and Qwen3.5/3.6 safetensors checkpoints execute their native heads
through `generate_embedded_mtp_input`; resident and bounded-layer loading are
supported, attention-bearing draft caches participate in live paging, and
DeepSeek/Nemotron routed prediction experts participate in independent bounded
expert residency. The same semantic predictor adapters execute through
`generate_embedded_mtp_cartesian` for TP, PP, EP, every pairwise combination,
and TP+PP+EP. Predictor projections use planner-derived TP geometry, routed
predictor experts use the topology-derived EP subgroup and rank-owned expert
catalog, and only the final PP coordinate owns predictor static weights and
draft state. Target verification, predictor output publication, sampling,
failure, and cancellation are synchronized over the Cartesian world while PP
transport remains lane-local to matching non-PP coordinates. Pipeline stage
reports expose `owns_embedded_mtp` and `embedded_mtp_layers` alongside the full
topology coordinates.
Inkling preserves per-depth full/sliding attention and optional chain
normalization; Nemotron-H repeats its configured physical MTP pattern without
assuming a particular number of steps. Text batches use independent backbone
and draft caches, so acceptance lengths and EOS positions may diverge safely.

### Quantized loading coverage

| Architecture | Dense | Existing quantized | Affine / MXFP4 on load | High-level dispatch | Special policy |
|---|---:|---:|---:|---:|---|
| Llama | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Linear, embedding, tied/untied head targets |
| Mistral | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Reuses the Llama-compatible dense decoder; configured sliding attention uses bounded KV caches |
| Qwen2/Qwen2.5 text | yes | MLX affine/MXFP4 and packed GGUF affine | yes / yes | `LoadedModel` | Biased Q/K/V projections, GQA, tied/untied heads, and exact full/sliding layer caches share the dense-Qwen plan |
| LFM2/LFM2.5 and LFM2-MoE | yes | MLX affine/MXFP4 and packed GGUF affine | yes / yes | `LoadedModel` | Alternating short-convolution/attention cache; MoE uses sigmoid top-k routing and packed expert-major SwiGLU execution |
| Kimi Linear | yes | MLX affine/MXFP4 and packed GGUF affine/IQ/MXFP4-MoE | yes / yes | `LoadedModel` | Hybrid KDA/no-RoPE MLA cache; packed routed experts and one shared expert; norms, transition parameters, biases, and convolution weights remain dense |
| Qwen3 | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Linear, embedding, tied/untied head targets |
| Qwen3-VL | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Language targets plus aligned vision attention, MLP, merger, and DeepStack projections share the bounded packed overlay; patch convolution, position embeddings, and norms stay dense |
| Qwen3-VL-MoE | yes | MLX affine/MXFP4 | yes / yes | `LoadedModel` | Reuses Qwen3-VL packed DeepStack/MRoPE vision execution and Qwen3 packed expert-major SwiGLU execution across standalone and Cartesian residency |
| Gemma 4 dense / MoE | yes | MLX affine/MXFP4 and packed GGUF affine | yes / yes | `LoadedModel` | Text, routed experts, bridges, and aligned vision/audio projections share resident, layerwise, streamed, and Cartesian materialization; convolution, position, and normalization state stays dense |
| Gemma 4 assistant | yes | MLX affine/MXFP4 and uniform packed GGUF affine | yes / yes | `LoadedDrafter` with `ModelLoadOptions` | Transformer/projection/head targets; ordered masked-embedding heads return a capability error |
| GPT-OSS | dense attention, MXFP4 experts | checkpoint-native MXFP4 experts plus `gpt-oss` GGUF | no / yes | `LoadedModel` | Canonical GGUF type-39 experts stay packed; mixed dense projections use their exact GGUF formats |
| Inkling | yes | packed GGUF affine/IQ/MXFP4 | yes / yes | `LoadedModel` | `inkling` text GGUF plus sibling combined hMLP/dMel mmproj; aligned hMLP projections, the dMel embedding, eligible text projections, and complete shared/routed expert banks use one packed overlay across fully resident, bounded nonresident, independent-cache, and Cartesian execution |
| Nemotron-H | yes | MLX affine/MXFP4 and checkpoint-native packed GGUF affine/IQ | yes / yes | `LoadedModel` | ReLU2 routed experts execute through the shared packed grouped rank-3 primitive; complete expert banks and ordinary targets use the affine/MXFP4 overlay across fully resident, bounded nonresident, independent-cache, and Cartesian execution |
| Qwen3.5/3.6-MoE | yes | block FP8, MLX affine/MXFP4 | yes / yes, from dense checkpoints | `LoadedModel` | Rank-3 expert banks are quantized row-wise and executed with routed `gather_qmm`; native FP8 checkpoints are never implicitly transcoded |
| Qwen3-Next | yes | native block FP8, MLX affine/MXFP4 | yes / yes, from dense checkpoints | `LoadedModel` | Official dynamic E4M3 128 x 128 checkpoints work with resident, layerwise, sparse expert-cache, and expert-parallel policies; fused weights/scales are split while streaming and native FP8 is never implicitly transcoded |
| Moshi | yes | MLX affine/MXFP4 | yes / yes | realtime loader | Temporal/depth projections and embeddings; no codec dependency |
| PersonaPlex | yes, transformed PyTorch layout | MLX affine/MXFP4 | yes / yes | realtime loader | Preserves per-depth checkpoint transformation; no codec dependency |

On-load selection is driven by the target module parameter tree, not by
blindly quantizing every rank-2 checkpoint tensor. Therefore specialized
convolutions, modality towers, routers, and packed expert banks stay dense only
when the architecture explicitly supports that policy, or the request is
rejected before weights are loaded.

For Gemma 4, Inkling, or Qwen image prompts, pass text and media as ordered processor
segments. Media is inserted where the segment appears; callers do not put
image/video/audio media tokens in rendered prompt text:

```rust,ignore
use safemlx_lm::runtime::media::{MediaInput, ProcessorInput, RgbImageView};

let image = RgbImageView::packed(rgb_pixels, width, height)?;
let prepared = model.prepare_input(
    &[
        ProcessorInput::Text(prompt_before_image),
        ProcessorInput::Media(MediaInput::image_rgb8(image)),
        ProcessorInput::Text(prompt_after_image),
    ],
)?;
let logits = model.prefill_prepared_input_with_cache(
    &prepared,
    &mut cache,
    stream,
)?;
```

Structured multimodal chat uses the same prepared model input without losing
checkpoint-native tools or semantic streaming:

```rust,ignore
use safemlx_lm::{
    api::{PreparedChatGenerationRequest, PreparedChatInput},
    runtime::media::{ChatMediaBinding, MediaInput},
};

let prepared_chat = model.prepare_chat(chat_request)?;
let prepared_input = model.prepare_chat_input(
    &prepared_chat,
    &[ChatMediaBinding::new(
        checkpoint_image_placeholder,
        MediaInput::image_rgb8(image),
    )],
)?;
let output = model.generate_prepared_chat(PreparedChatGenerationRequest {
    input: PreparedChatInput::prepared_model_input(&prepared_chat, &prepared_input),
    // cache, sampling_policy, settings, caller_stop_sequences, stream, on_event
})?;
```

The composer verifies exact placeholder counts and ordering before the
architecture processor inserts media boundary tokens and tensors. Use
`PreparedChatInput::rendered_prompt(&prepared_chat)` for text-only structured
generation.

Decoded videos use the same processor with an ordered frame sequence and source
frame rate. Container decoding remains with the caller:

```rust,ignore
let frames = decoded_rgb_frames
    .iter()
    .map(|frame| RgbImageView::packed(frame, width, height))
    .collect::<Result<Vec<_>, _>>()?;
let prepared = model.prepare_input(
    &[
        ProcessorInput::Text(prompt_before_video),
        ProcessorInput::Media(MediaInput::video_rgb8(&frames, Some(source_fps))),
        ProcessorInput::Text(prompt_after_video),
    ],
)?;
```

The optional `image-processing` feature enables architecture-dispatched Gemma 4,
Inkling, and Qwen processors. Shared code owns decoded-image validation, frame sampling,
and timestamp operations; each processor adds its model-native patch packing,
prompt format, metadata, and ordered media insertion. Inkling divides images into
40-pixel patches and feeds its released four-layer hMLP tower. Gemma samples up to
32 frames by default and encodes each timestamped frame through its vision tower.
Qwen uses its temporal patch packing and timestamp format. Without the feature,
callers can still supply Gemma 4, Inkling, or Qwen `Image/Tensor` and `Video/Tensor`
inputs directly without depending on the `image` crate.

Gemma 4 audio accepts model-native log-mel tensors and Inkling accepts discrete
dMel IDs through the typed input API
without optional dependencies. Enable `audio-processing` to prepare mono `f32`
PCM in the shared processor instead:

```toml
[dependencies]
safemlx-lm = { version = "0.4", features = ["audio-processing"] }
```

```rust,ignore
use safemlx_lm::runtime::media::{MediaInput, ProcessorInput};

let audio = MediaInput::audio_f32(mono_pcm, sample_rate)?;
let prepared = model.prepare_input(&[
    ProcessorInput::Text(prompt_before_audio),
    ProcessorInput::Media(audio),
    ProcessorInput::Text(prompt_after_audio),
])?;
let logits = model.prefill_prepared_input_with_cache(&prepared, &mut cache, stream)?;
```

The common audio processor validates and resamples neither channels nor sample
rate: Gemma 4 and Inkling currently require mono 16 kHz PCM. It computes each
model's log-mel features and valid-frame mask; Inkling then quantizes them to its
16-bin dMel representation. The optional FFT dependency is only enabled by
`audio-processing`; callers that provide `Audio/Tensor` and `audio_mask` directly
do not pay that dependency cost.

## Realtime encoded audio

The `realtime` module defines a codec-free API for realtime speech-to-speech
models. Models consume discrete codec-token frames and emit delay-aligned
generated codec-token frames; callers keep audio encoding, decoding, transport,
and device I/O outside `safemlx-lm`.

Use `load_realtime_model` when the model directory contains a realtime
codec-token model. It dispatches PersonaPlex, Moshi, and future realtime model
families separately from the chat/text `LoadedModel` path:

```rust,ignore
use safemlx_lm::{
    load_realtime_model,
    RealtimeInferenceScheduler, RealtimeSampling, RealtimeStepInput,
    runtime::{
        scheduler::{RequestId, SchedulerLimits},
        generation::sampler::DefaultSampler,
    },
};

let mut model = load_realtime_model(model_dir, stream, weights_stream)?;
let config = model.realtime_config();
let input_codebooks = config.input_audio_codebooks;
let generated_codebooks = config.generated_audio_codebooks;
let audio_samplers = (0..config.depth_audio_codebooks)
    .map(|_| DefaultSampler)
    .collect::<Vec<_>>();
let request = RequestId::new(42);
let mut scheduler = RealtimeInferenceScheduler::new(
    &model,
    SchedulerLimits::new(1, 4)?,
)?;
scheduler.register_request(
    &model,
    request,
    DefaultSampler,
    audio_samplers,
    RealtimeSampling::greedy(),
)?;

// Your codec supplies one user/input-side frame shaped
// [batch, input_codebooks].
scheduler.enqueue(
    &model,
    request,
    RealtimeStepInput::encoded_audio(&encoded_input_frame),
)?;
let output = scheduler
    .run_queued(&mut model, stream)?
    .pop()
    .expect("one queued frame")
    .into_parts()
    .1;

if let Some(codec_tokens) = output.output_audio_tokens {
    // Decode [batch, generated_codebooks] with your codec.
}
```

Pass `ModelLoadOptions::default().with_weight_residency(WeightResidency::layerwise_host(...))`
to `load_realtime_model_with_options` to keep temporal layers and Moshi-family
depth-codebook slices on the host. Text/audio embeddings and temporal output
modules remain pinned, and `residency_report()` exposes the two execution groups.
The scheduler uses the same request-state representation for fully resident,
host-layerwise, and dense-streamed models, so a released session can resume
across residency policies only when the selected checkpoint-file SHA-256 and
normalized execution identity match. Geometry alone is insufficient;
quantization changes and same-shaped checkpoints with different weights are
rejected without a residency-specific state enum.
`run_bounded` limits one drain to a caller-selected number of frames, providing
cooperative deadline and cancellation boundaries; `run_queued` drains all work
that was ready at entry.

The `architectures::moshi::model` module implements Moshi's temporal and depth language
models over pre-tokenized Mimi streams. Production session ownership is through
`RealtimeInferenceScheduler`; `generate_encoded_greedy` is the one-request
offline scheduler convenience API.
Sequence tensors use Mimi's `[batch, codebooks, frames]` layout.

`architectures::moshi::personaplex` exposes PersonaPlex's Moshi-family realtime token API,
published 7B v1 defaults, dual-stream codebook layout, and hybrid system-prompt
helpers. It can load the released Hugging Face PyTorch-layout
`model.safetensors` directly via the shared Moshi-family PyTorch importer.

PersonaPlex consumes 8 user-side codec codebooks per realtime frame and emits 8
agent-side codec codebooks per output frame. Its depth transformer still samples
or teacher-forces 16 codebooks, so realtime sampling requires 16 audio samplers.
Prompt helpers are token-only: use `wrap_system_prompt` before external text
tokenization, then use `enqueue_text_prompt` or `enqueue_system_prompt` on the
same registered request used for live frames. Prompt and live work therefore
share one queue, one delayed-stream state, and the same cancellation and
backpressure contract.

Mimi audio encoding/decoding and audio device I/O deliberately remain outside
`safemlx-lm`. The sibling `safemlx-codec` crate provides safemlx-native codec
building blocks, including Mimi checkpoint loading, PCM encode/decode,
residual-vector quantization, and stateful tokens-to-PCM decode. Audio device
I/O remains optional codec surface rather than an `safemlx-lm` dependency.

Moshi loads dense and MLX affine- or MXFP4-quantized checkpoints. For the original
released Moshika/Moshiko repositories, the loader uses Moshi's built-in v0.1
config when the model directory has no `config.json`.

## Checkpoint quantization

For memory-constrained load-time conversion, use the public
`runtime::checkpoint::bounded_quantization` plan and overlay store described
under persistent checkpoint storage. Its conversion budget can be set to the
final quantized model footprint: for example, the deterministic tests convert
a 2,048-byte F32 matrix under a 320-byte working-set ceiling, equal to that
matrix's complete affine 4-bit weight, scale, and bias representation. The
result then enters `ResidencyManager` as a 320-byte unit. Conversion uses
explicit CPU streams and clears released MLX allocator cache entries as
completed tiles are written. Its fixed two-tile event window remains inside
the admitted combined working set; budgets that cannot hold two minimum tiles
use one slot.

The generic checkpoint converter quantizes eligible two-dimensional
`*.weight` tensors one at a time, writes bounded-size safetensors shards, and
copies tokenizer and other model assets. Affine output has packed `weight`, `scales`, and
`biases`; MXFP4 output has only packed E2M1 `weight` and E8M0 `scales`. In both cases,
`config.json` contains identical `quantization` and `quantization_config`
objects.

```sh
cargo run --release -p safemlx-lm --example quantize_checkpoint -- \
  /path/to/dense-model /path/to/model-4bit \
  --group-size 64 --bits 4

cargo run --release -p safemlx-lm --example quantize_checkpoint -- \
  /path/to/dense-model /path/to/model-mxfp4 --mode mxfp4
```

Use repeatable `--include` and `--exclude` substring filters to experiment on
part of any safetensors checkpoint, `--minimum-elements` to leave small
matrices dense, and `--shard-size-mib` to control peak buffered output and
shard size. The output directory must not already exist.

The checkpoint converter accepts dense safetensors inputs. Load-time conversion also accepts
unquantized F32, F16, and BF16 GGUF inputs through `ModelLoadOptions`. GGUF files containing
packed quantized tensors are rejected rather than being implicitly dequantized and transcoded to
affine or MXFP4 storage.

Library callers can use `runtime::checkpoint::quantization::quantize_checkpoint` for conversion,
the shared `ModelLoadOptions` APIs for architecture dispatch, or
`runtime::checkpoint::load::load_safetensors_dir_quantized_strict` to populate a model that
exposes the standard packed parameter tree. Model-specific
`load_*_model_quantized` helpers remain available. All modes call
`runtime::checkpoint::quantization::quantize_tensor` with a caller-owned explicit stream, so saving
and direct loading use the same numerical transform.
Direct loading materializes each packed weight/scale/bias triple before reading
the next dense tensor. This prevents MLX's lazy graphs from retaining the whole
dense checkpoint during conversion while preserving exact parity with a saved
quantized checkpoint. This older complete-tensor path bounds accumulation
across the checkpoint but still requires one complete dense matrix; use the
bounded overlay store when peak load-time memory must fit inside the final
quantized footprint.

To include direct Q4 conversion in a PersonaPlex load/step benchmark, use the
dense checkpoint with `--quantize-on-load`:

```sh
cargo run --release -p safemlx-lm --example personaplex_step_bench -- \
  /path/to/personaplex-dense 64 --quantize-on-load
```

Generate a deterministic fixture with the upstream `moshi_mlx` package, then
replay it through Rust:

```sh
python safemlx-lm/scripts/moshi_mlx_token_fixture.py \
  /path/to/moshika-mlx-bf16 /tmp/moshi-token-parity.safetensors \
  --require-mlx-version 0.32.0

cargo run -p safemlx-lm --release --example moshi_token_parity -- \
  /path/to/moshika-mlx-bf16 /tmp/moshi-token-parity.safetensors
```

Use the MLX version pinned by `safemlx-sys/src/mlx-c/CMakeLists.txt` when
generating a reference fixture. The version guard prevents comparisons across
different MLX kernel implementations.

The comparator uses standard relative and absolute closeness checks and
defaults to `rtol=0.02` and `atol=0.02`, suitable for BF16 cached inference.
It reports the largest absolute difference observed. Pass explicit tolerances
as the third and fourth arguments.

The fixture contains delayed temporal inputs, teacher-forced depth inputs, the
normalized temporal states, text logits, logits from every depth slice, and an
end-to-end greedy encoded-audio generation sequence. By default the exporter
creates deterministic synthetic tokens; pass `--inputs` with a safetensors file
containing `input.text`, `input.audio`, and `input.depth` to replay a prerecorded
Mimi-token sequence for the teacher-forced portion.

For a lightweight end-to-end check without downloading released weights, add
`--create-tiny --steps 6`. This creates a deterministic miniature BF16
checkpoint in the supplied model directory before exporting its reference
fixture.

Moshi projections preserve their checkpoint dtype and execute with the MLX
0.32.0 NAX kernels.

## Expert-parallel MoE inference

`expert_parallel` provides executable expert parallelism for the
safetensors MoE families supported by expert residency: DeepSeek-V3/R1,
GPT-OSS, Inkling, Kimi Linear, LFM2,
Nemotron-H, Qwen3, Qwen3-Next, Qwen3-VL-MoE, and Qwen3.5-MoE. Every listed SafeTensors family accepts
fully resident execution, or independent expert caching composed with either
host-layerwise or dense disk-streamed nonexpert units.
Fully resident TP+EP uses the same external expert executor as sparse TP+EP:
the generalized layerwise engine pins TP-sharded nonexpert units, while the
residency manager eagerly materializes and pins only the experts owned by the
local EP coordinate. Independent expert caching instead retains or streams
nonexpert units according to its `NonExpertWeightResidency` while caching
only rank-owned experts. The model API requires
`EP > 1`. DeepSeek-V3/R1, Inkling, Kimi Linear, Qwen3, GPT-OSS, LFM2-MoE,
Nemotron-H-MoE, Qwen3-Next/Qwen3.5-MoE, and the Qwen3-VL-MoE text decoder
support TP+EP through tensor-sharded layerwise adapters with topology-scoped TP
and EP groups. All use the same resident-or-sparse executor for TP+EP.
Qwen3-VL-MoE additionally supports SafeTensors and canonical GGUF PP+EP and
TP+PP+EP with typed image/video prefill through the pipeline loader;
its routed expert assignment and packed companions come from the same semantic
adapter as TP+EP. GPT-OSS accepts both SafeTensors and
canonical type-39 MXFP4 GGUF, while LFM2-MoE and Nemotron-H-MoE accept
SafeTensors and GGUF expert layouts through their shared packed-expert recipes.
DeepSeek-V3/R1, Inkling, Kimi Linear, Qwen3-MoE, Qwen3-VL-MoE, GPT-OSS,
LFM2-MoE, Nemotron-H-MoE, and Qwen3-Next/Qwen3.5-MoE PP+EP use the pipeline
loader with stage-local expert assignment.
Dense models and GGUF architectures without a registered EP
adapter are also rejected. Kimi Linear, DeepSeek2, Qwen3-MoE, and
Qwen3-VL-MoE provide
resident GGUF pure EP. Fully resident and sparse-streamed GGUF TP+EP cover
Kimi Linear, DeepSeek2, Qwen3-MoE, GPT-OSS, Inkling, LFM2-MoE,
Nemotron-H-MoE, Qwen3-Next, Qwen3.5-MoE, and Qwen3-VL-MoE. The Qwen3-VL-MoE
adapter composes its sibling mmproj with the language catalog before rank-local
text and expert selection. Qwen3.5-MoE applies the same composition to its
optional sibling projector.
Checkpoint `ep_size`
describes a stored layout and is not the runtime EP degree.

`ExpertAssignment` supports balanced-contiguous (the model default),
round-robin, and explicit owner maps. Pass a non-default assignment to
`load_expert_parallel_model_with_assignment`, or use
`load_expert_parallel_model_with_options_and_assignment` for explicit residency
or ownership. Packed
checkpoints select the exact ordered expert rows for non-contiguous policies;
they do not materialize the enclosing range. Routers and observations always
use checkpoint-global expert ids. Only immediately before a grouped expert
kernel does the dispatcher translate them to dense owner-local ids, so
non-contiguous policies do not depend on `global_id - range.start`.

The pure-EP model path uses replicated-input dispatch. Attention, norms,
routers, embeddings, dense MLPs, heads, and architecture-specific shared
experts are replicated, so every rank already has the same hidden rows and
router result.
Each rank compacts only locally owned routes, executes only its local expert
bank, reduces those routes into a full zero-initialized token buffer, and uses
one all-sum for the routed contribution. Replicated shared experts are computed
once per rank and added *after* that all-sum; they are never multiplied by EP
size. Exact compaction performs one scalar route-count synchronization
per sparse layer, with no per-expert synchronization and no capacity dropping.

Token-sharded execution builds one architecture-neutral `AllToAllVPlan` per
complete dispatch. It all-gathers the tiny source/destination count matrix once,
materializes that matrix once, derives source-major receive counts, and reuses
the plan for hidden rows, global expert ids, original route indices, and
weights. The reverse plan transposes the known count relation without another
consensus and is reused for returned expert outputs and indices. Every rank
enters the same six dependency-ordered payload phases, including route-empty
ranks, and a divergent field row count is rejected before payload traffic.

The payload path is MLX's patched lazy `all_to_all_v`, not padded all-gather.
Native MPI, TCP Ring, JACCL mesh/ring, and NCCL groups transfer compact addressed
rows. Ring uses store-and-forward neighbor packets which stop at their
destinations; physical hop bytes can therefore exceed logical endpoint bytes,
but no rank materializes the global source/destination payload matrix. Logical
Cartesian EP subgroups use their topology-planned neighbor routes with the same
destination-specific property.

`RoutingStatistics` separates useful sent/received bytes, zero native padding,
known or unknown backend physical bytes, measurable staging high water, a
SafeMLX-visible allocation bound, count-consensus time, payload-exchange time,
and native versus logical routed transport. Normal inference keeps MLX's lazy
scheduling, so phase fields primarily measure graph submission apart from the
one required count-matrix wait. The opt-in `profile_expert_parallel_timings`
guard materializes each phase and the final logits before its timer stops; this
changes scheduling and is for measurement only.

For fully resident DeepSeek, Kimi Linear, and Qwen3,
`ExpertParallelModel::forward_with_observer` preserves global router ids and
weights while exposing the rank-local routed contribution, globally reduced
routed contribution, replicated shared-expert contribution, and final combined
MoE output as distinct fields. Detailed activation observation is currently
unavailable on the sparse expert-cache EP path. Qwen3 callers can select the
standard growing cache with `new_cache()` or a bounded cache with
`new_qwen3_sliding_cache(window, paging_options)`; the bounded form uses the
shared paged residency manager and both retain the same EP routing semantics.

Every supported packed or split expert layout is selected by placement before
payload materialization. The ordinary sparse-cache loader avoids opening
remote-only indexed payload shards. Dense streaming catalogs all logical units
up front, so it may touch remote shard metadata, but only rank-owned expert
arrays can enter the expert cache and concurrent readers remain bounded. Dense,
affine/MXFP4, FP8, and ReLU2/SwiGLU banks retain their architecture-specific
physical kernels behind the common replicated dispatch. With sparse caching,
`routed_expert_bytes` is zero and `owned_expert_bytes` describes the rank's
cold, warm, or hot catalog; with fully resident DeepSeek/Qwen3 it scales
approximately with `1 / EP`. `replicated_parameter_bytes` remains constant.
Load-time affine or MXFP4 conversion for sparse-cache EP uses the same
rank-owned semantic catalog as checkpoint-native loading. Each projection is
converted in admitted CPU row tiles into a temporary packed SafeTensors
overlay; remote experts are never read, route-empty ranks do not materialize a
bank, and the retained cache footprint is the packed footprint. Matching
checkpoint-native encodings continue to load directly and implicit packed
transcoding fails closed.

DeepSeek's Cartesian resident path independently shards MLA heads, dense and
shared projections, embeddings, and the output head over TP while assigning
routed expert banks over EP. Its compressed latent and rotary-key cache remains
stage-local. Independent caches keep the routed banks in their checkpoint-native
split, packed-affine, block-FP8, or DeepSeek2 GGUF representation and compose
with fully resident, host-layerwise, and dense-streamed non-experts. Cached
routed results are complete across TP, while only the TP-sharded shared result
is reduced.

Kimi Linear's Cartesian resident path independently shards KDA/MLA heads,
dense and shared projections, embeddings, and the output head over TP while
assigning routed expert banks over EP. The routed contribution is exchanged
within its stage-local EP group and TP-sharded shared projections are reduced
once. Independent expert caches retain replicated TP geometry for each
EP-owned routed bank, so cached routed results do not enter a second TP
reduction. Sparse-cache EP materializes only rank-owned expert payloads.
Kimi GGUF checkpoints use the same partitioning and execution path for dense,
affine, IQ, and MXFP4-MoE expert banks in resident or either sparse-cache mode.
The combined streamed policy keeps rank-local nonexpert layer shards cold,
keeps every owned expert cold until routed, and never adds remote experts to
the rank-local cache. The architecture-neutral GGUF EP dispatcher and
type-erased expert cache also serve the registered DeepSeek2, Qwen3-MoE, LFM2-MoE,
Nemotron-H-MoE, Qwen3-Next, and Qwen3.5-MoE streamed adapters.

Inkling uses the same Cartesian composition for stage-zero hMLP/dMel ingress,
full/sliding relative attention, four fixed short-convolution states, dense
transitions, and routed plus shared experts. Resident execution shards media,
attention, and dense/shared expert intermediates over TP while EP owns routed
banks. Independent caches retain replicated TP geometry for EP-owned routed
banks and remain lazy under fully resident, host-layerwise, and dense-streamed
nonexpert policies. SafeTensors and canonical `inkling` GGUF with a sibling
combined media mmproj preserve stage-local KV/convolution state, bounded reads,
prompt-cache identity, synchronized generation, and failure consensus.

Run a two-process Ring generation probe with the usual MLX Ring host file and
rank environment:

```sh
cargo run --release -p safemlx-lm --example expert_parallel_generate -- /path/to/model
```

The example prints assignment metadata, performs prefill and multiple decode
steps, samples only on rank zero, synchronizes token/stop state, and reports
routing counters. Ring is intended for correctness and functional testing. Its
exact payload can cross several physical hops; JACCL mesh or NCCL is expected
for practical low-latency EP. Small-batch decode is often dominated by expert
imbalance, and replicated-input EP still avoids token exchange entirely. No
speedup is implied without measurements on the target checkpoint and backend.

For a device-complete performance probe comparing a complete model on rank
zero, replicated-input EP on every rank, and a variable-count synthetic
sharded-input exchange, run:

```sh
cargo run --release -p safemlx-lm --example expert_parallel_benchmark -- \
  /path/to/model --backend jaccl --device gpu --warmup 1 --iterations 3
```

The CSV reports prefill and fixed-token decode latency/throughput, every MoE
phase, summed MoE and whole-model time, routes-per-rank imbalance, useful sent
and received bytes, padding bytes, backend physical bytes when known, staging
high water when measurable, allocation bounds, count consensus, routed
transport, peak MLX memory, and complete-versus-EP logit error. The synthetic
case uses uneven destination counts and verifies zero payload padding. Ring can
be used with `--backend ring --device cpu`; MLX Ring collectives execute on CPU.
Results include the synchronization introduced by phase profiling and should
not be presented as unprofiled production throughput or as an automatic EP
speedup.

Useful verification and opt-in probe commands are:

```sh
cargo test -p safemlx-lm expert_parallel --lib
cargo test -p safemlx-lm --test distributed_expert_exchange_ring -- --ignored --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_two_process_model_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_gpt_oss_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_deepseek_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_deepseek_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_deepseek_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_deepseek_triple_axis -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_deepseek_streamed_triple_axis_expert_cache -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_deepseek_layerwise_host_triple_axis_expert_cache -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_deepseek_gguf_triple_axis_expert_cache -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_lfm2_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_lfm2_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_lfm2_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_inkling_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_inkling_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_inkling_gguf_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_inkling_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_inkling_gguf_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_inkling_multimodal_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_inkling_multimodal_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_inkling_multimodal_triple_axis -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_inkling_multimodal_streamed_triple_axis_expert_cache -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_eight_process_inkling_multimodal_layerwise_host_triple_axis_expert_cache -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_qwen3_vl_moe_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_qwen3_vl_moe_gguf_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_qwen3_vl_moe_gguf_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring qwen3_vl_moe_gguf_pipeline_expert_stages_own_rank_local_layers_and_experts -- --exact --nocapture
cargo test -p safemlx-lm --test distributed_qwen3_vl_pipeline_ring ring_qwen3_vl_ -- --ignored --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_kimi_linear_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_kimi_linear_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_kimi_linear_gguf_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_kimi_linear_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_kimi_linear_gguf_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_nemotron_h_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_nemotron_h_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_nemotron_h_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_qwen3_next_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_expert_parallel_ring ring_four_process_qwen35_tensor_expert_parity -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_qwen3_next_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_qwen35_moe_tensor_pipeline -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_qwen3_next_pipeline_expert -- --ignored --exact --nocapture
cargo test -p safemlx-lm --test distributed_pipeline_ring ring_four_process_qwen35_pipeline_expert -- --ignored --exact --nocapture
cargo run --release -p safemlx-lm --example expert_parallel_generate -- /path/to/model
cargo run --release -p safemlx-lm --example expert_parallel_benchmark -- /path/to/model --backend jaccl
```

The exchange Ring test covers variable-count transport plus complete two-rank
forward and reverse sharded dispatch. Its non-monotonic route metadata runs
through packed ReLU2 and native block-FP8 local banks, and a separate case
keeps one rank completely route-empty while every collective is still entered.

The model-parity Ring test uses tiny deterministic complete-model references
and checks prefill, two cached decode steps, and three synchronized tokens for
dense and affine-packed Qwen3/DeepSeek banks, native DeepSeek block-FP8, and
sparse expert-cache EP for every supported MoE family.
The Nemotron-H PP+EP requantization cases select MXFP4 for both fully resident
rank-local expert banks and independent expert caches, and cover prompt-cache
reload plus synchronized decode/generation.
It also runs packed Qwen with round-robin placement and split DeepSeek with an
explicit non-contiguous owner map.
The separate
`ring_two_process_streamed_dense_sparse_expert_cache_parity` case runs
prefill and cached decode with replicated decoder layers streamed and routed
experts cached independently across the supported text/runtime fixture set.
`ring_four_process_gpt_oss_tensor_expert_parity` covers the Cartesian TP=2,
EP=2 path, including tensor-sharded nonexperts, native expert-cache ownership,
axis-local collectives, cached decode, and rank-synchronized generation.
The DeepSeek four-process cases cover TP+PP, TP+EP, and PP+EP across a
dense-to-MoE boundary with tensor-sharded MLA/dense/shared projections,
compressed-cache persistence, stage-local or cached expert ownership,
dense-streamed bounded reads, and single-rank numerical parity.
The DeepSeek eight-process cases add resident TP+PP+EP plus dense-streamed,
host-layerwise, and canonical DeepSeek2 GGUF non-experts with independently
cached routed experts. They also cover EP-inactive TP+PP caching and cached
expert schedule failure consensus.
The LFM2 four-process cases cover TP+PP, TP+EP, and PP+EP with hybrid
convolution/attention caches, dense-streamed rank-local recipes, prompt-cache
reload, dense-to-sparse expert placement, and single-rank numerical parity.
The Kimi Linear four-process cases cover TP+PP, TP+EP, and PP+EP across its KDA
and MLA state transition, TP-sharded dense and shared projections, stage-local
or cached routed experts, SafeTensors and GGUF bounded reads, prompt-cache
reload, synchronized generation, and single-rank numerical parity.
The eight-process cases add resident TP+PP+EP plus dense-streamed,
host-layerwise, and canonical-GGUF non-experts with independently cached routed
experts. They also cover EP-inactive TP+PP caching and cached-expert schedule
failure consensus.
The Inkling four-process cases cover TP+PP, TP+EP, and PP+EP across scheduled
text/image/audio ingress, stage-zero dMel projection, full/sliding attention,
and dense/sparse transitions. They include rank-local KV and four-way
convolution state, stage-local or cached routed experts, shared experts,
dense-streamed bounded reads, prompt-cache reload, synchronized generation,
and single-rank numerical parity.
The eight-process cases add resident TP+PP+EP plus dense-streamed,
host-layerwise, and canonical-GGUF nonexperts with independently cached routed
experts. The multimodal cases repeat resident, streamed, and host-layerwise
triple-axis execution with scheduler-owned payload/identity consensus. They
also cover EP-inactive TP+PP caching, uneven stage placement, and cached-expert
schedule failure consensus.
The Qwen3-VL-MoE four-process cases cover SafeTensors and canonical GGUF TP+EP
text-decoder execution with
rank-local MRoPE KV geometry, tensor-sharded attention and nonexpert
projections, EP-scoped cached routed experts, bounded reads, prompt-cache
reload, synchronized generation, and single-rank numerical parity.
The GGUF suite additionally exercises two-rank pure EP under resident and
sparse-cache policies, four-rank TP+EP under resident and dense/sparse-streamed
policies, sibling-projector telemetry, route-empty participation, and PP+EP
stage-local layer and expert ownership.
The Qwen3-VL-MoE eight-process cases compose scheduler-owned stage-zero vision
and DeepStack ingress with TP-sharded MRoPE attention, matching-coordinate
pipeline lanes, EP-scoped routed experts, prompt-cache reload, route-empty
ranks, and synchronized sampling. Mismatched prepared-input modality or shape
identity fails global work consensus before pipeline transport. SafeTensors
covers resident experts and independently
cached experts with fully resident, host-layerwise, or dense-streamed
nonexperts; canonical GGUF covers dense-streamed nonexperts, bounded read
telemetry, and independently cached experts. TP+PP with EP inactive exercises
the same cache through singleton local dispatch.
The Qwen hybrid four-process cases cover both Qwen3-Next and Qwen3.5 identities
across TP+PP, TP+EP, and PP+EP, including recurrent/full-attention transitions,
packed routed and shared experts, route-empty collective participation,
dense-streamed bounded reads, cache persistence, and synchronized generation.
Its DeepSeek fixture crosses a dense-to-MoE layer boundary, uses two router
groups, and deliberately gives one rank zero routes to exercise imbalance and
empty-local-work behavior. GPU FP8 keeps the packed Metal kernels; CPU Ring
uses the slower dequantized FP8 reference path.

## License

Licensed under either Apache-2.0 or MIT.
Paged attention-cache residency and reusable prompt-cache persistence are
opt-in. Device-resident caches remain the default. See
[`CACHE_RESIDENCY.md`](CACHE_RESIDENCY.md) for configuration, compatibility,
cost, and safety details. Prompt-cache schema v5 records exact ordered per-layer
attention plus convolution, recurrent, compressed-MLA, and multimodal prefix
state, including distinct sliding windows. Prompt-cache loading accepts schema
v4. LFM2 causal-convolution state and Nemotron-H Mamba state use the same
ordered fixed-state representation. Realtime Moshi/PersonaPlex sessions can be
released from their scheduler for application-owned handoff but remain outside
the persisted prompt-cache routes. Run
`paged_prompt_cache` for a deterministic save/drop/reopen parity check.
