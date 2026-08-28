# Eredu

Eredu loads and runs supported text, multimodal, and realtime speech models.
The `eredu` crate is its main application API, covering model inspection,
loading, planning, chat preparation, generation, and telemetry. The default
features include the MLX execution implementation.

Use the [model support guide](https://github.com/jbg/eredu/blob/main/doc/model-support.md) to check supported model
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

The portable build includes artifact inspection, tokenizer and chat metadata,
generation orchestration, media request descriptions, execution planning, and
application-facing backend metadata. The facade re-exports selected portable
value types used by those APIs, but generic schedulers, backend traits, and
speculative execution transport contracts are imported from their owning
crates.

## Loading a model

`LocalModel` owns one selected backend session together with tokenizer, EOS,
generation-default, and chat-template metadata. Its backend, session, prompts,
token handles, drafting resources, and native errors remain private:

```rust,ignore
use eredu::{
    api::{local_device_plan, LocalBackendFactory, LocalDevice, LocalModel},
    ExecutionPlan,
};

let device = local_device_plan(LocalDevice::Accelerator(0))?;
let plan = ExecutionPlan::fully_resident(device);
let factory = LocalBackendFactory::default();
let planned = LocalModel::load_execution_plan(&factory, "/path/to/model", &plan)?;
let (mut model, drafting) = planned.into_parts();
```

Model-family architecture modules remain owned by `eredu-architectures`.
Reusable MLX tensors, operators, caches, streams, and family composition live
in `eredu-backend-mlx`. Application clients use the flat selected-local-backend
adapter in `eredu::api`; backend implementers and backend-specific tooling
import the implementation crate directly.

Use `LocalModel::load_execution_plan` or `LocalModel::plan_and_load` when the
application wants portable plan-to-backend realization. These entry points
construct the selected backend and any embedded or external speculative
drafter described by the plan. Backend implementers can use the separate
generic `LoadedModel<B>` API with a backend imported from its owning crate.

Realtime speech models use the selected facade's concrete application API.
`prepare_realtime_model` inspects the artifact, `LocalRealtimeBackendFactory`
loads it, and `LocalRealtimeScheduler` accepts and returns portable host token
frames. These operations require only the `eredu` crate; native backend traits,
streams, tensors, sessions, and completions stay behind the facade.

## Cargo features

- `mlx`: the MLX execution backend; enabled by default.
- `metal`: Metal execution with Accelerate on Apple platforms; enabled by
  default and effective only with `mlx`.
- `cuda`: CUDA execution on supported Linux and Windows hosts; effective only
  with `mlx` and mutually exclusive with `metal`.
- `image`: image and video preprocessing; effective only with `mlx`.
- `audio`: audio preprocessing; effective only with `mlx`.
- `nccl`: NCCL distributed support; effective only with `mlx` and CUDA.
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

- [Model loading, quantization, and memory](https://github.com/jbg/eredu/blob/main/doc/model-loading.md)
- [Backend architecture](https://github.com/jbg/eredu/blob/main/doc/backend-architecture.md)
- [Speculative decoding and MTP](https://github.com/jbg/eredu/blob/main/doc/speculative-decoding.md)
- [Native tool calling](https://github.com/jbg/eredu/blob/main/doc/tool-calling.md)
- [Parallel execution](https://github.com/jbg/eredu/blob/main/doc/parallel-execution.md)
- [Platform setup](https://github.com/jbg/eredu/blob/main/doc/platforms.md)
- [Eredu CLI](https://github.com/jbg/eredu/blob/main/eredu-cli/README.md)

The minimum supported Rust version is 1.89.

## License

Licensed under either Apache-2.0 or MIT.
