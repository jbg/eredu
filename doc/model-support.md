# Model and checkpoint support

`eredu` selects an architecture from checkpoint metadata, normalizes its
configuration, and validates the complete tensor catalog before loading
payloads. A compatible family name alone is not sufficient: unsupported layer
geometry, tensor layout, quantization, tokenizer, processor, or execution
policy produces a structured error.

## Inspect an artifact

Use backend structural inspection before reserving a device or loading
weights, then apply backend-independent text inspection when tokenizer or chat
readiness matters:

```rust,no_run
use eredu::{
    api::{inspect_text_model, TextInspectionOptions},
    backend::mlx::{inspect_model, MlxInspectionOptions},
    InspectionSeverity,
};

let structural = inspect_model("/path/to/model", MlxInspectionOptions::default())?;
let report = inspect_text_model(structural, TextInspectionOptions::default());
if !report.is_loadable() {
    for issue in report
        .issues
        .iter()
        .filter(|issue| issue.severity == InspectionSeverity::Error)
    {
        eprintln!("{:?}: {}", issue.code, issue.detail);
    }
}
# Ok::<(), eredu::backend::mlx::error::Error>(())
```

Structural inspection reads configuration and bounded checkpoint headers,
validates processor metadata and required media artifacts, and applies the
requested quantization, residency, and topology preflight. Text inspection
reconstructs tokenizer, template, and EOS metadata and behaviorally probes
semantic output and native tools. Neither step creates an MLX stream or loads
weight payloads. `is_loadable()` is the fail-closed structural result for the
selected backend options.

## Family matrix

The table describes high-level dispatch. “SafeTensors” means a supported
Hugging Face-style model directory with the required configuration, tokenizer,
and processor files. “GGUF” means the canonical architecture metadata and
tensor layout expected by Eredu; multimodal families may require a sibling
projector.

| Family | Inputs | SafeTensors | GGUF | Notable support |
| --- | --- | :---: | :---: | --- |
| Llama / Mistral | text | yes | yes | full or sliding attention, tied or untied head |
| Qwen2 / Qwen2.5 | text | yes | `qwen2` | dense text models; exact full/sliding schedules |
| Qwen3 | text | yes | `qwen3`, `qwen3moe` | dense and MoE |
| Qwen3-VL | text, image, video | yes | `qwen3vl`, `qwen3vlmoe` + projector | dense and MoE text decoders, MRoPE and DeepStack |
| Qwen3-Next / Qwen3.5 / Qwen3.6 / Qwen3.8 | text; image/video where present | yes | `qwen3next`, `qwen35`, `qwen35moe` | shared Qwen3.5 hybrid recurrent/full-attention contract; dense and MoE, projector and embedded-MTP support; official Qwen3.6/3.8 chat and tagged-parameter tools |
| DeepSeek-V3 / R1 | text | yes | `deepseek2` | compressed MLA, dense-to-MoE schedules, native block FP8 |
| DeepSeek-V4 | text | yes | `deepseek4` | mHC compressed sparse attention, routed/shared experts, native FP8 and mixed FP4, embedded MTP and fused DSpark |
| GPT-OSS | text | yes | `gpt-oss` | Harmony chat, scheduled attention, native MXFP4 experts |
| Gemma 4 | text, image, video, audio | yes | `gemma4` + projector | dense and MoE variants, shared-KV and media towers |
| Gemma 4 assistant | text draft model | yes | `gemma4_assistant` | external multi-token prediction |
| Kimi Linear | text | yes | `kimi-linear` | KDA/MLA hybrid layers, dense and routed/shared experts |
| LFM2 / LFM2.5 | text | yes | `lfm2`, `lfm2moe` | causal-convolution/attention hybrid, dense and MoE |
| Nemotron-H | text | yes | `nemotron_h`, `nemotron_h_moe` | Mamba, attention, dense MLP, and MoE layers |
| Inkling | text, image, audio | yes | `inkling` + projector | multimodal hMLP ingress and dense/MoE text execution |
| Muse-Glimmer | text, image, video | yes | `muse-glimmer`; optional projector | vision tower, mixed attention, ATEM/DFlash paths |
| Moshi | realtime codec tokens | yes | no | temporal/depth generation without a codec dependency |
| PersonaPlex | realtime speech tokens | yes | no | transformed model layout, voice and text conditioning |

DeepSeek-V4 uses the shared execution infrastructure for fully resident,
host-layerwise, and dense disk-streamed weights; independent expert caches;
device-resident or paged request state; and reusable prompt caches, including
embedded MTP and fused DSpark draft state. Eligible dense SafeTensors weights
can be transformed to affine or MXFP4 layouts at load time. Checkpoint-native
FP8 and packed encodings remain native.

Qwen2 vision-language and Qwen2 MoE checkpoints are not accepted by the Qwen2
text adapter. A model that needs custom Python code or uses an unregistered
tensor layout is not implicitly compatible.

## Checkpoint formats

### SafeTensors directories

Eredu supports single-file and indexed sharded SafeTensors. It validates every
shard header, tensor name, shape, dtype, and required quantization companion
before materialization. Tokenizer and chat-template selection use checkpoint
sidecars. Multimodal models additionally require their processor configuration
and media-tower weights.

Recognized stored formats include dense F32/F16/BF16, MLX affine and MXFP4
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
  caches, memory mappings, and some temporary workspaces.
- On Apple silicon, logical host and device tiers share physical unified memory.
- Full-attention paging reduces logical device residency but still reads all
  retained history and can be I/O intensive.
- GGUF and SafeTensors counters describe Eredu reads and logical residency;
  operating-system page caching means they are not physical disk-I/O meters.
- Realtime Moshi and PersonaPlex session state is not represented by decoder
  prompt-cache persistence.
- A non-replicated topology must use the matching distributed loader rather
  than the complete single-model loader.
