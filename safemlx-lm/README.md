# safemlx-lm

`safemlx-lm` is a Rust runtime for loading and running supported language,
multimodal, and realtime speech models on MLX. It provides architecture
inspection, checkpoint validation, tokenization, chat preparation, generation,
structured tool calls, quantization, bounded weight and cache residency, and
distributed execution.

Support is determined from checkpoint metadata and validated tensor structure,
not from a model's display name. Unsupported configurations fail before weight
payloads are materialized.

## Features

- SafeTensors model directories and sharded GGUF checkpoints.
- Text, image, video, audio, and realtime codec-token model inputs where the
  selected architecture supports them.
- Greedy and stochastic generation, prepared chats, reasoning events, and
  constrained native tool calls.
- External and checkpoint-embedded multi-token prediction.
- Load-time affine or MXFP4 quantization for eligible dense weights.
- Fully resident, host-layerwise, disk-streamed, and independent expert-cache
  weight policies.
- Device-resident or paged live caches and reusable prompt-cache persistence.
- Tensor, pipeline, and expert parallel execution for supported families.

See the [model support matrix](../doc/model-support.md) for the supported
families, checkpoint formats, modalities, and major restrictions.

## Installation

```toml
[dependencies]
safemlx-lm = "0.4"
```

Enable `image-processing` when the application wants the built-in image
preprocessing path. Enable `cuda` for an MLX CUDA build on a supported Linux or
Windows host.

## Inspect before loading

`inspect_model` performs the same architecture, tensor, quantization,
residency, and topology preflight used by loading without creating an MLX
stream or loading weight payloads:

```rust,no_run
use safemlx_lm::{inspect_model, ModelInspectionOptions};

let report = inspect_model("/path/to/model", ModelInspectionOptions::default())?;
if !report.is_loadable() {
    for issue in &report.issues {
        eprintln!("{:?}: {}", issue.code, issue.detail);
    }
}
# Ok::<(), safemlx_lm::error::Error>(())
```

Use the high-level `api` module for loading, prepared inputs, generation, cache
creation, and memory admission. Architecture-specific modules expose lower-level
construction and distributed adapters when an application needs them.

## Inputs and generation

`LoadedModel` owns the model, tokenizer, optional processor, and chat-template
metadata. Applications can prepare raw token input, ordered multimodal input,
or a structured chat request. Prepared-chat generation emits protocol-neutral
semantic events for reasoning, visible text, and tool calls.

Raw generation remains available for completion workloads. It deliberately
bypasses chat-template and native-tool guarantees.

## Memory and execution policies

Weights and runtime state have separate policies and budgets:

- Weight residency controls where static parameters live and when they are
  materialized.
- Cache residency controls KV, compressed-latent, convolution, and recurrent
  request state.
- Load-time quantization changes eligible static weights but never silently
  transcodes an already packed checkpoint format.

Read [Model loading, quantization, and memory](../doc/model-loading.md) before
using bounded residency or prompt-cache persistence. Distributed users should
also read [Parallel execution](../doc/parallel-execution.md).

## More documentation

- [Native tool calling](../doc/tool-calling.md)
- [Speculative decoding and MTP](../doc/speculative-decoding.md)
- [Cancellation and bounded execution](../doc/cancellation.md)
- [CLI example](../examples/safemlx-lm-cli/README.md)
- [Platform setup](../doc/platforms.md)

The minimum supported Rust version is 1.89.

## License

Licensed under either Apache-2.0 or MIT.
