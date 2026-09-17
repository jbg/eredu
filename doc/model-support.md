# Model and checkpoint support

`eredu` selects an architecture from checkpoint metadata, normalizes its
configuration, and validates the complete tensor catalog before loading
payloads. A compatible family name alone is not sufficient: unsupported layer
geometry, tensor layout, quantization, tokenizer, processor, or execution
policy produces a structured error.

Family and format support describes functional execution. Finite managed-memory
admission depends on the selected mechanisms, state residency and request. See the
[current managed-admission matrix](bounded-inference.md#current-integration-status)
for validated paths and remaining integration.

## Inspect an artifact

Use backend structural inspection before reserving a device or loading
weights, then apply backend-independent text inspection when tokenizer or chat
readiness matters:

```rust,no_run
use eredu::{
    api::{
        inspect_local_model, inspect_text_model, TextModelOptions, LocalInspectionOptions,
        TextInspectionOptions,
    },
};
use eredu_core::InspectionSeverity;

let structural = inspect_local_model("/path/to/model", LocalInspectionOptions::default())?;
let text_options = TextModelOptions::default();
let report = inspect_text_model(structural, &text_options, TextInspectionOptions::default());
if !report.is_loadable() {
    for issue in report
        .issues
        .iter()
        .filter(|issue| issue.severity == InspectionSeverity::Error)
    {
        eprintln!("{:?}: {}", issue.code, issue.detail);
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Structural inspection reads configuration and bounded checkpoint headers,
validates processor metadata and required media artifacts, and applies the
requested quantization, residency, and topology preflight. Text inspection
reconstructs tokenizer, template, and EOS metadata and behaviorally probes
semantic output and native tools. Neither step creates a backend execution
queue or loads weight payloads. `is_loadable()` is the fail-closed structural
result for the selected backend options.

Configure a single or named override once in `TextModelOptions::chat_template`.
Inspection borrows that options value; pass it to the subsequent
`LoadedModel::*_with_text_options` load. Both operations use the same template
resolver: an override takes precedence over embedded GGUF metadata and
sidecars, while checkpoint tokenizer variables and EOS ids remain in effect.
Invalid selected template metadata is reported without falling back to another
source. `TextInspectionOptions::chat_request` supplies the messages, tools, and
template kwargs to probe before loading.

## Family matrix

The table describes high-level dispatch. “SafeTensors” means a supported
Hugging Face-style model directory with the required configuration, tokenizer,
and processor files. “GGUF” means the canonical architecture metadata and
tensor layout expected by Eredu; multimodal families may require a sibling
projector.

| Family | Inputs | SafeTensors | GGUF | Notable support |
| --- | --- | :---: | :---: | --- |
| Llama / Mistral | text | yes | yes | full or sliding attention, tied or untied head |
| Nanbeige 4.2 | text | yes | `nanbeige` | shared-weight repeated GQA/SwiGLU stack; independent KV caches for each pass |
| Qwen2 / Qwen2.5 | text | yes | `qwen2` | dense text models; exact full/sliding schedules |
| Qwen3 | text | yes | `qwen3`, `qwen3moe` | dense and MoE |
| Qwen3-VL | text, image, video | yes | `qwen3vl`, `qwen3vlmoe` + projector | dense and MoE text decoders, MRoPE and DeepStack |
| Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8 | text; image/video where present | yes | `qwen3next`, `qwen35`, `qwen35moe` | shared Qwen3.5 hybrid recurrent/full-attention contract; dense and MoE, projector and embedded-MTP support; official Qwen3.6/3.8 chat and tagged-parameter tools |
| DeepSeek-V3 / R1 | text | yes | `deepseek2` | compressed MLA, dense-to-MoE schedules, native block FP8 |
| DeepSeek-V4 | text | yes | `deepseek4` | mHC compressed sparse attention, routed/shared experts, native FP8 and mixed FP4; SafeTensors embedded MTP and fused DSpark; base GGUF target only |
| GPT-OSS | text | yes | `gpt-oss` | Harmony chat, scheduled attention, native MXFP4 experts |
| Gemma 2 | text | yes | `gemma2` | alternating sliding/full GQA, four RMSNorms, GELU and attention/output soft caps; [validation status](gemma2-validation.md) |
| Gemma 4 | text, image, video, audio | yes | `gemma4` + projector | dense and MoE variants, shared-KV and media towers |
| Gemma 4 assistant | text draft model | yes | `gemma4_assistant` | external multi-token prediction |
| K2 Horizon | text | yes | `k2-horizon` | dense, MoE and MoVA; independently addressable value/feed-forward experts, grouped RMSNorm, YaRN, IFM reasoning/tools and external drafting; [validation](k2-horizon-validation.md) |
| Kimi Linear | text | yes | `kimi-linear` | KDA/MLA hybrid layers, dense and routed/shared experts |
| LFM2 / LFM2.5 | text | yes | `lfm2`, `lfm2moe` | causal-convolution/attention hybrid, dense and MoE |
| Nemotron-H | text | yes | `nemotron_h`, `nemotron_h_moe` | Mamba, attention, dense MLP, and MoE layers |
| Inkling | text, image, audio | yes | `inkling` + projector | multimodal hMLP ingress and dense/MoE text execution |
| Muse-Glimmer | text, image, video | yes | `muse-glimmer`; optional image-only projector | vision tower, mixed attention, ATEM/DFlash paths |
| Moshi | realtime codec tokens | yes | no | temporal/depth generation without a codec dependency |
| PersonaPlex | realtime speech tokens | yes | no | transformed model layout, voice and text conditioning |

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
ordering. Each logical invocation owns a separately materialized block, so
resident memory includes those replicas; checkpoint storage remains physical.
Bounded residency can admit one logical block at a time.

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
rendering, constraints, and response parsing. Qwen3.6 and Qwen3.8 retain the
`qwen3_5` / `qwen3_5_text` architecture contract. Their released templates use
tagged function parameters rather than the older Qwen JSON-in-XML protocol;
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


### PersonaPlex private numerical evidence

The architecture test suite includes six private reduced-dimension PersonaPlex cases over actual F32 SafeTensors: strict public-gate/source proof; three-residency full-state continuation parity; real delayed/mixed decision demand; wide low-level readout demand and three cached continuations; initialization/observer/failure/cancellation; and physical shared-norm/final packed-slice sensitivity. The fixture keeps the released 16/16/8 topology, eight input channels, delays and normalization ownership. It is separate from public released-profile admission. Dimensions/vocabularies are reduced only through architecture-owned `cfg(test)` normalization. Existing full-checkpoint and independent-reference obligations remain open; these synthetic cases do not replace them. Execution results belong in the integration validation record after central checks.


### Qwen composite retained-ingress conformance

Qwen VL and conditional Qwen have an ordinary selected retained-media source for their existing shared prefill driver. The six added neutral numerical cases cover dense selected resident/host/disk and TP/PP/combined, complete local state, encoder-once behavior, actual Sequence/final rows, ordered raw image/video and projected-text semantics, cancellation and cached rollback parity. Central validation records report executed coverage; these cases supplement existing released-checkpoint evidence and are not native distributed validation.

The native composite visitor continues its existing unsplit media route until original source/graph/control/completion admission is implemented. Other composite families still need architecture-owned retained ingress hooks; this is an implementation gap, not architectural inapplicability. This unit preserves the existing fresh-cache raw-media gate and does not claim populated-cache raw-media append, captured-media attribution or unfinished-source continuation.

## Muse retained-media conformance subset

Muse-Glimmer now supplies its own compact normalized raster-media source to the shared selected media lifecycle. It preserves image/video placeholder IDs, text-embedding normalization and per-layer cache-derived masks, completing all future projected rows under the first span. Received encoder boundaries rebuild placement metadata without repeating patch/learned projection. The nonzero local and selected TP/PP/combined cases cover run/step, resident/host/disk, full/last local rows, mixed image/video 2/2/1 and ordinary cancellation-prefix cached decode. Central validation records describe executed coverage.

This is an ordinary/unfunded mechanism, with no native/public retained-media activation or complete peak claim. Gemma4/Inkling still require a neutral observed-inactive dependency state and their own per-layer/shared-attention or zero-unit audio semantics. Native original source/completion, funded continuation and remaining-family integration remain required follow-ons.

## Gemma4 and Inkling retained-media validation matrix

The selected retained-media cut now distinguishes Unseen, observed Inactive, and Produced dependency outcomes. Existing layered traversal and partition scheduling establish inactivity; later spans import it without fabricating a tensor. Missing active dependencies still reject. Inkling's active zero-unit dMel group executes its existing completion equation. Default hooks preserve other executors.

Gemma4 and Inkling provide architecture-owned retained-source/span hooks through the existing selected media visitor. Gemma4 keeps independent vision/audio projection owners, original placeholders, per-layer inputs, cache positions and shared-KV transport; each input projection runs only at its existing root begin. Received encoder continuations use original metadata, without dummy shape tensors. Inkling preserves exact hMLP folds, dMel offsets/normalization/valid frames and sconv state. Full future media roots settle under the first span; original semantic vectors move into the source while encoder preprocessing temporaries remain under first-span completion.

Five nonzero numerical cases cover: compact optional combinations across resident/host/disk and TP/PP/combined, ordinary versus shared full/last rows, run/step, cancellation at two actual boundaries, complete local state and three cached decodes. The exact large Inkling hMLP receives focused resident/host/disk and bounded PP coverage with 512 MiB fixture budgets. Its remaining raw-image topology/residency cross-products remain explicit unexecuted validation obligations, not unsupported behavior.

Central validation records report executed coverage. This is ordinary/unfunded selected-driver support. Public/native retained-media activation, full original source/control/graph/completion admission, captured-media attribution and funded continuation remain Unit C work. New source and context allocations are not caller-owned or free; no native peak or whole managed execution claim follows from these changes.


### Inkling hMLP cut and input-order regressions

The existing exact-tower numerical case now exercises PP2 and PP4, covering internal hMLP cuts 1/2/3 through the selected transport. Its compact decoder has four alternating sliding/full layers to satisfy the existing one-decoder-unit-per-pipeline-owner rule; the released four-stage vision geometry and fixed 8192×4800 matrix are unchanged. Existing resident/host/disk ordinary versus shared run/step checks, nonzero outputs, complete state and three cached decodes remain. PP cancellation compares committed-prefix state with an independently executed prefix.

An additional request in that case puts two real raw patches before the same ordered text tokens. It compares host-resident ordinary/shared run/step and PP4 execution, checks the complete two-row retained vision root, and compares cached decode state. This exercises the distinction between patch population and decoder batch. These are source-added regression assertions pending central execution, not new released-checkpoint results. Other raw-image TP/combined/topology-residency combinations and native hardware validation remain explicit unexecuted obligations.
