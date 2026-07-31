# Model and checkpoint support

This page summarizes the high-level support implemented by `safemlx-lm`. For
API details and the full layerwise-residency matrix, see the
[`safemlx-lm` README](../safemlx-lm/README.md).

Support is determined from checkpoint metadata and validated configuration,
not from a model's display name. Applications can inspect a SafeTensors model
directory before loading it with `check_model_dir`, `check_model_config`, or
`check_model_config_json`.

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

- Llama/Mistral, Qwen3, and Qwen3-VL use KV-head-aware GQA accounting.
- GPT-OSS and Gemma 4 account for full versus sliding attention separately.
- DeepSeek-V3/R1 accounts for compressed MLA latent plus rotary-key state.
- LFM2 and Nemotron-H separate bounded convolution/Mamba state from
  context-growing attention state.
- Qwen3-Next and Qwen3.5 separate recurrent linear-attention state from
  full-attention KV.
- Inkling accounts for global/sliding KV and bounded convolution state.
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
- `kimi-linear`
- `llama` and `mistral`
- `lfm2` and `lfm2moe`
- `nemotron_h` and `nemotron_h_moe`
- `qwen3` and `qwen3moe`
- `qwen3next`
- `qwen3vl` (with its companion vision projection checkpoint)
- `qwen35` and `qwen35moe`

The tokenizer and chat template are reconstructed from GGUF metadata when
possible. A sibling `tokenizer.json` can supply a tokenizer that is absent from
the file or uses an unsupported embedded tokenizer model.

`safemlx-gguf` parses GGUF v1-v3 in either byte order and validates all shard
headers before payload materialization. Its supported dense and quantized
tensor encodings are listed in the
[`safemlx-gguf` README](../safemlx-gguf/README.md).

## Weight loading and residency

Fully resident loading is the default. SafeTensors and registered GGUF families
can also use host-backed layer windows or experimental dense disk streaming.
GGUF bounded loading covers Kimi Linear, DeepSeek2, Gemma 4, Llama/Mistral, LFM2,
Nemotron-H, Qwen3, dense Qwen3-VL with its mmproj, Qwen3.5, and Qwen3-Next.
Supported MoE families can cache routed experts independently; for GGUF these
are Kimi Linear, DeepSeek2, LFM2-MoE, Nemotron-H-MoE, Qwen3-MoE,
Qwen3.5-MoE, and MoE Qwen3-Next.

Kimi Linear SafeTensors supports fully resident, layerwise-host, dense
disk-streamed, sparse-expert-cache, and sparse-expert-with-dense-layers
loading for both SafeTensors and GGUF. Its KDA recurrent state remains bounded
while no-RoPE MLA caches grow with context. The GGUF loader accepts modern
split or legacy unsplit MLA KV-B projections, singleton convolution layouts,
dense/K/IQ tensors, and MXFP4-MoE type 39.

Qwen3-Next supports the official native fine-grained E4M3 checkpoint format
(`fp8`, dynamic activations, 128 x 128 weight blocks) with fully resident,
layerwise, sparse expert-cache, and pure expert-parallel loading. Fused QKVZ
weights and inverse scales are split without dequantization, dense BF16 BA is
preserved, and routed expert weights remain checkpoint-backed at expert
granularity for sparse-cache and expert-parallel execution.

Important boundaries:

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
- Kimi Linear paged and persisted prompt caches are unavailable because its
  hybrid KDA recurrent state is not represented by the paged KV cache format.
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
caches, and the shared expert is added once after routed reduction. Tensor
parallelism and pipeline parallelism return capability errors. Fully resident
GGUF expert parallelism is supported for Kimi Linear, DeepSeek2, and
Qwen3-MoE through the shared resident-GGUF dispatcher; sparse-cache GGUF EP
remains unavailable.
