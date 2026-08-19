# Eredu

Eredu loads and runs supported text, multimodal, and realtime speech models.
The `eredu` crate is its main application API, covering model inspection,
loading, planning, chat preparation, generation, and telemetry. The default
features include the MLX execution implementation.

Use the [model support guide](../doc/model-support.md) to check supported model
families, checkpoint formats, modalities, and execution capabilities.

## Features

- SafeTensors model directories and sharded GGUF checkpoints.
- Tokenization, chat templates, reasoning events, and constrained native tool
  calls.
- Text, image, video, audio, and realtime codec-token inputs for supported
  architectures.
- Greedy and stochastic generation plus embedded or external speculative
  decoding.
- Load-time quantization and bounded weight, cache, and prompt-cache residency.
- Tensor, pipeline, and expert parallel execution where supported.
- Portable inspection, automatic execution planning, admission, and telemetry
  schemas.

## Installation

The default build uses MLX:

```toml
[dependencies]
eredu = "0.4"
```

A portable facade build does not compile or link MLX:

```toml
[dependencies]
eredu = { version = "0.4", default-features = false }
```

The portable build includes artifact inspection, tokenizer and chat metadata,
generation orchestration, media request descriptions, execution planning, and
the backend contracts. Applications supply an implementation of the required
backend capabilities.

## Loading a model

`LoadedModel<B>` owns one selected backend session together with tokenizer,
EOS, generation-default, and chat-template metadata. The same loading and
generation methods work for every backend:

```rust,ignore
use eredu::{
    api::LoadedModel,
    backend::mlx::{MlxBackend, ModelLoadOptions},
};

let backend = MlxBackend::new(execution_stream, weights_stream);
let mut model = LoadedModel::load(
    backend,
    "/path/to/model",
    ModelLoadOptions::default(),
)?;
```

For MLX, native models, caches, streams, load policies, architecture modules,
and diagnostics are available only under `backend::mlx`. Generic APIs do not
expose MLX arrays, streams, devices, or errors.

Use `LoadedModel::load_execution_plan` or `LoadedModel::plan_and_load` when the
application wants portable plan-to-backend realization. These entry points
construct the selected backend and any embedded or external speculative
drafter described by the plan.

## Cargo features

- `mlx`: MLX model loading and execution; enabled by default.
- `mlx-image`: MLX image and video preprocessing.
- `mlx-audio`: MLX audio preprocessing.
- `cuda`: MLX CUDA execution on supported Linux and Windows hosts.
- `nccl`: NCCL distributed support for CUDA builds.
- `onig`: Oniguruma tokenizer regular expressions; enabled by default.
- `esaxx_fast`: fast SentencePiece-compatible tokenization; enabled by
  default.

Portable tokenization always includes the pure-Rust `fancy-regex` engine.

## Execution model

A backend is selected for an entire model session. The session owns its
executable, cache, processor state, queues, and optional distributed context.
Prefill, decode, multimodal preparation, speculative generation, realtime
execution, capability reporting, and admission are exposed through explicit
backend traits.

Submitted work carries an exact completion object. Generation and scheduler
APIs retain in-flight resources until completion, including after cancellation.
This preserves cache transactions and prevents early release of backend-owned
values.

## Documentation

- [Model loading, quantization, and memory](../doc/model-loading.md)
- [Backend architecture](../doc/backend-architecture.md)
- [Speculative decoding and MTP](../doc/speculative-decoding.md)
- [Native tool calling](../doc/tool-calling.md)
- [Parallel execution](../doc/parallel-execution.md)
- [Platform setup](../doc/platforms.md)
- [Eredu CLI](../eredu-cli/README.md)

The minimum supported Rust version is 1.89.

## License

Licensed under either Apache-2.0 or MIT.
