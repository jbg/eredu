# Model loading, quantization, and memory

`eredu` separates static weight placement from per-request runtime state.
Choose and budget them independently: moving model layers does not move the KV
cache, and paging a cache does not change where model weights live.

## Preflight

Applications admit a local `ArtifactSource` with `eredu::api::inspect_artifact`,
then resolve text policy once with `PreparedTextModel::from_artifact`.
`PreparedTextModel::inspect` probes the retained tokenizer, template, and EOS
policy without selecting a backend. Its requested-load readiness remains
unverified. Invalid selected text metadata returns a typed preparation error;
missing chat templates still permit raw generation.

`PreparedTextModel::select_execution_plan` validates the exact complete plan
against a backend factory's cold facts and retains target and drafting admission,
including external-assistant vocabulary compatibility. `SelectedTextModel::inspect`
combines that retained selection's report with optional text probes. `load()`
consumes the selection through the same factory without accepting another plan,
reopening text sidecars, or repeating cold selection. Native resource acquisition
and guarded payload access can still fail during realization. Chat probes describe
the supplied request, not every future conversation.

`ArtifactSource::with_projector` accepts `CompanionSelection::Discover`,
`Explicit(path)`, or `Disabled`. Explicit files bypass sibling discovery and have
no fallback. They undergo container, encoding, geometry, and family compatibility
admission. Disabling a required role or selecting an undeclared role is a typed
error. Automatically discovered files follow the same admission rules and reject
ambiguous candidates.

`inspect_safetensors_snapshot` is available without any backend feature. It checks
complete headers, file lengths, and exact index membership using canonical
`eredu-checkpoint` admission. Directory checkpoints use `model.safetensors`, or
`model.safetensors.index.json` for a sharded set; filenames alone cannot prove
completeness. This check does not establish payload checksums, supported model
architecture, text readiness, or native backend availability.

Path-based `LoadedModel` loaders use the same retained pipeline.
`inspect_execution` takes a backend factory and the intended `ExecutionPlan`. It
returns an inspection report and retains any admitted selection. The plan carries
quantization, weight residency and required session capabilities. Native
distributed groups are the backend’s responsibility.

Realtime preparation checks required capabilities against the selected factory
before checkpoint payload materialization. A bound protocol retains the admitted
state and execution resources.

## Gemma 4 MLX conversions

Gemma 4 SafeTensors admission accepts released `model.language_model.*`,
neutral `model.*`, and MLX conversion `language_model.model.*` decoder names.
MLX media towers and projections may use root-level names. Quantized weights
and their scales/biases must satisfy the same exact geometry under every alias;
duplicate physical aliases for a logical tensor are rejected.

Nested per-weight entries in the Hugging Face `quantization` object override
the default format and inherit omitted format fields. Names are normalized
before format selection, including the 8-bit MLP overrides in the 4-bit QAT
conversion. Conflicting aliases for an override are rejected. Boolean `false`
per-weight overrides are rejected explicitly rather than silently quantized.
Media projection encodings are retained in the media configuration consumed by
both checkpoint admission and model construction; tower alignment restrictions
still apply. Audio framing accepts `boa_token_index` and `eoa_token_index` as
fallbacks when the corresponding `_id` field is absent.

Projector-only `gemma4_unified_vision` / `gemma4_unified_audio` metadata without
`num_hidden_layers` admits text execution without advertising those media
modalities. Associated placeholder IDs are cleared and unused projection
weights are permitted. Missing media configurations with live placeholder IDs,
malformed configurations, and incomplete full-tower geometries remain errors.

Header-only validation covers these pinned `mlx-community` artifacts. Counts
include the complete tensor catalogs; this evidence establishes checkpoint
admission and processor selection, not numerical generation parity.

| Artifact | Revision | Tensor headers | Processor |
| --- | --- | ---: | --- |
| `gemma-4-e4b-it-4bit` | `475b9088d29754a3379866cf5aeb6b41acd313c2` | 2,770 | Media enabled |
| `gemma-4-12B-it-4bit` | `73bcf09092aa277861d5a191b989b666f7f32e8f` | 1,341 | Text only |
| `gemma-4-12B-it-qat-4bit` | `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` | 1,341 | Text only |

The QAT regression fixture retains published layer-0 and static header metadata,
with the configured layer count and override map restricted to that layer. It
checks admission, the independent packed shapes, missing companions, and alias
ambiguity without allocating model tensors:


## Weight residency

### Fully resident

`eredu_runtime::WeightResidency::fully_resident()` materializes the complete
selected model or rank-local shard before execution. This is the simplest and
usually fastest policy when it fits. Infrastructure clients configure this
policy through the owning `eredu-runtime` crate.

### Host-layerwise

Host-layerwise execution keeps repeated execution groups in host-transfer
storage and promotes a bounded window to the execution device. Static modules
such as embeddings, final normalization, and the output head remain pinned
when the architecture requires them.

The device-layer window and host/device byte limits are hard admission inputs.
A dedicated same-device transfer stream and completion leases allow the next
group to be prepared while the current group executes. A narrow window may
still stall when a needed transfer does not complete in time.

Some backends expose logical host and device tiers that share the same physical
memory. Host-layerwise placement can still control executable working sets and
lifetimes, but it does not create additional physical capacity; consult the
selected backend's platform guide.

### Dense disk streaming

Disk streaming keeps ordinary execution groups checkpoint-backed and uses
bounded disk, host, and device caches. Host lookahead and the background queue
control how much future materialization may be in flight. This policy reduces
resident parameter bytes at the cost of checkpoint reads and conversion during
execution.

SafeTensors shard buffers and GGUF readers are bounded separately. Logical read
counters describe Eredu's checkpoint access rather than physical storage
traffic.

### Independent expert cache

Registered MoE families can combine any supported non-expert policy with a
separate routed-expert cache. The cache has its own device, host, scratch, and
prefill-bank budgets and deterministic eviction policy.

The portable runtime reads routing demand, selects the exact expert members,
and acquires missing members as a residency batch. MLX reads a compact
device-computed histogram. Candle CPU and Metal read selected indices for host
counting and remapping. Both retain each acquisition through compact-bank
execution and exact completion.

The scratch limit is a hard bound for one compact bank. Multi-token prefill may
be split to stay within the configured prefill-bank target and scratch bound.
Decode remains one routed bank. Exact routing never substitutes a different
expert when the requested one is unavailable.

## Independent media sources

An artifact's admitted tower and processor coverage determines its usable input
modalities. A text-only Spark conversion remains usable for text; an image, audio
or video request without the corresponding components produces a typed
`MissingCompanion` error during preparation. Cold inspection describes the
missing optional source without rejecting text loading.

Use the execution plan to select independent vision and audio sources.

```rust,no_run
use eredu::api::MediaCompanionSources;

let companions = MediaCompanionSources::default()
    .with_vision("/path/to/spark-bf16-vision")
    .with_audio("/path/to/spark-bf16-audio");
let plan = plan.with_media_companions(companions);
```

Vision and audio can share one directory or select different directories. Muse
validates the selected BF16 tower schemas, decoder geometry, token identities,
processor policy and provenance before acquiring payloads. Indexed source
directories require the complete members containing the selected media tensors;
their unrelated decoder members need not be present. Prepared construction keeps
these exact sources and restricted component views through materialization.
The text CLI exposes the same selection through `--vision-companion DIRECTORY`
and `--audio-companion DIRECTORY`, including planning and exact-plan replay.

## Load-time quantization

Eredu can convert eligible dense F32/F16/BF16 weights to an affine or MXFP4
layout while loading. Conversion is driven by the architecture's parameter
plan, not by blindly quantizing every matrix. Norms, convolutions, routers,
position tables, and specialized modality parameters remain dense unless that
architecture explicitly registers a safe transformation.

Bounded policies transform admitted tensor or row selections rather than
materializing the entire dense checkpoint. Independent expert caches use the
same approach for selected expert banks.

Dense quantization requires nonempty matrix geometry. Expert-bank leading
dimensions are flattened with checked shape arithmetic, so banks whose total
element count exceeds a 32-bit integer retain their geometry; only a flattened
row count that the selected quantization operator cannot represent is rejected.

Checkpoint-native packed weights remain native. GGUF K-quant, IQ, affine, and
MXFP4 tensors, or native block-FP8 SafeTensors, are never silently transcoded
into a requested format. An incompatible transformation fails during preflight.

## Runtime-state residency

`CacheResidencyPolicy::Device` keeps request state on the execution device and
is the default. `CacheResidencyPolicy::Paged` divides sealable attention state
into immutable blocks plus a mutable device tail.

A paged cache has finite per-request limits. Multiple caches can share a
`CacheResidencyPool`, which adds process-wide device, host, transfer-in-flight,
and live-disk limits. Per-cache limits remain active so one request cannot
consume the entire shared pool.

Runtime state has explicit behavioral classes:

| Class | Typical state | Behavior |
| --- | --- | --- |
| Sealable paged | KV or compressed MLA blocks | immutable blocks can move between device, host, and optional disk tiers |
| Always-device mutable | rolling convolution histories and small prefix metadata | stays resident because it changes in place |
| Layer-scoped offloadable | large recurrent matrices used by one layer at a time | can be moved at layer boundaries |
| Stateless | MLP- or MoE-only layers | no request-state payload |

Only sealable state enters the paged block store. A cache representation cannot
misclassify mutable recurrent state as immutable KV.

Sliding-window attention discards blocks older than the semantic window unless
persistence retention is requested. Full-attention paging retains history and
uses blockwise online softmax; every decode token still visits all retained
blocks, so host or disk paging can be very expensive.

Optional live-disk backing requires a dedicated directory, finite byte budget,
and finite worker queue. Mutable tails are never written. Temporary live-cache
files are removed when their manager is dropped and are separate from reusable
prompt caches.

## Reusable prompt caches

A prompt cache is a completed immutable prefix. Saving exports resident KV
in bounded slices or seals partial tails for paged KV. It publishes a manifest
and bounded SafeTensors shards atomically. Restore validates the manifest and
owned layer layout. It assembles fresh resident storage one layer at a time or
attaches compatible directory-backed paged shards for on-demand promotion.

`LoadedModel::set_prompt_cache_store` attaches a memory or directory store to a
session with admitted prompt-cache persistence. Request that support with
`with_prompt_cache_persistence(true)` on the execution plan or normalized load
request. Supply the exact checkpoint fingerprint when attaching the store and
the full prompt on every request. The facade shares eligible
prefix lookup, restore, and publication across Candle and MLX. See
[automatic prompt-cache reuse](execution-control.md#automatic-prompt-cache-reuse).
Portable identity and manifest types live in `eredu_core::cache`.

Callers obtain one complete `PromptCacheModelIdentity` from the prepared
session and pass it to `PromptCacheDescriptor::from_model_identity`. Only the
checkpoint fingerprint, processed-prefix fingerprint, and batch size are
caller-supplied; family, effective type, architecture identity, state geometry,
frontier offsets, sink policy, and distributed topology are copied together
from the prepared model.

For Qwen3-Next and Qwen3.5-family checkpoints with embedded prediction layers,
ordinary prefill also advances prediction state to its architecture-declared
prefix-relative frontier. Callers do not need to enter speculative generation
before saving a complete prompt cache.

Schema version 10 records the following.

- model family, effective type, checkpoint identity, and an architecture
  fingerprint;
- the exact ordered per-layer state kind, attention policy, window, tensor
  geometry, and distributed ownership;
- convolution, recurrent, multimodal prefix, and position-delta state where
  applicable;
- block ranges, shapes, dtypes, and payload digests; and
- exact prefix token identity plus the caller's processed-prefix fingerprint.

Load requires schema version 10 and preserves the admitted source precision
for affine companions. Model family, architecture fingerprint,
prefix identity, layer ownership, topology, state representation, tensor
geometry, and payload digests must match before arrays become live. Multimodal
applications should include media bytes and processor settings in the
processed-prefix fingerprint.

The persisted model family is the architecture registry's canonical family
name, identical to `model_family().canonical_name()` on the loaded model. It is
not a backend grouping or shared-implementation label; distinct registered
families such as `qwen2`, `qwen3`, `qwen3_next`, `qwen3_vl_moe`, and `qwen3_5`
remain distinct cache identities.

Distributed prompt caches are rank-local beneath one shared root. A cache
cannot be reopened under a different tensor, pipeline, or expert topology.
Realtime Moshi and PersonaPlex sessions remain outside this decoder prompt-cache
format.

## Accounting and observability

The CLI's `--timing` option reports load completion as soon as weights are ready,
then prints load and generation timings when generation ends. This separates
checkpoint loading from the wait for the first token, which includes prompt
evaluation and any first-use kernel setup. `--verbose` includes these timings too.

Residency reports separate logical device parameter bytes, charged host-buffer
capacity, disk payload bytes, transfers, cache hits and misses, evictions, and
in-flight ownership. Expert, dense-stream, and request-cache reports are
separate because they have different budgets and lifetimes.

Static parameter estimates do not include activations, KV or recurrent state,
kernel memory, allocator caches, buffered checkpoint shards, all temporary workspaces,
or opaque backend-driver allocations. Use backend allocator statistics and
process memory observation alongside Eredu reports when sizing an application.
