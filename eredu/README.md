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

The portable build includes tokenizer and chat metadata plus facade-owned
generation orchestration. Portable architecture, artifact, planning,
generation, media, and scheduling contracts are imported from their owning
crates; the facade does not duplicate those types at its root or under `api`.

## Loading a model

`LoadedModel<B>` is the canonical model API. It owns one backend session with
tokenizer, EOS, generation-default, and chat-template metadata. Pass a concrete
backend factory to infer `B`; the same API also works with mock backends:

```rust,no_run
# #[cfg(feature = "mlx")]
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use eredu::api::{default_local_device, local_device_plan, LoadedModel};
use eredu_core::ExecutionPlan;
use eredu_backend_mlx::MlxBackendFactory;

let device = local_device_plan(default_local_device())?;
let plan = ExecutionPlan::fully_resident(device);
let factory = MlxBackendFactory::default();
let planned = LoadedModel::load_execution_plan(&factory, "/path/to/model", &plan)?;
let (mut model, drafting) = planned.into_parts();
# Ok::<(), Box<dyn std::error::Error>>(())
# }
# #[cfg(not(feature = "mlx"))]
# fn main() {}
```

`default_local_device` selects accelerator zero in CUDA and Apple Metal builds,
and falls back to CPU in CPU-only MLX builds. Pass `LocalDevice::Cpu` or
`LocalDevice::Accelerator(index)` to `local_device_plan` when an application
needs an explicit device.

Model-family architecture modules remain owned by `eredu-architectures`.
Reusable MLX tensors, operators, caches, streams, and family composition live
in `eredu-backend-mlx`. Application clients import the backend factory from that
crate and use the generic facade types in `eredu::api`.

Use `LoadedModel::load_execution_plan` or `LoadedModel::plan_and_load` for
portable plan-to-backend realization. These entry points return
`PlannedModel<B, D>`, which owns the target and `RealizedDrafting<D>`. Automatic
planning can retain its artifact inspection for
`LoadedModel::load_inspected_execution_plan`. Prepared-chat requests, controlled
generation sessions, snapshots, and branches keep their backend parameter too.
Low-level token generation returns `B::Token`; use `eredu_core::TokenOutput` to
obtain its vocabulary ID.

Applications can supply their own builtin or custom Jinja chat template through
`TextModelOptions`:

```rust,ignore
use eredu::api::TextModelOptions;

let text_options = TextModelOptions {
    chat_template: Some(application_template.into()),
};
let planned = LoadedModel::load_execution_plan_with_text_options(
    &factory, "/path/to/model", &plan, text_options,
)?;
```

The same options are accepted by `plan_and_load_with_text_options`,
`load_inspected_execution_plan_with_text_options`, and `load_with_text_options`.
Templates may be a single Jinja string or an `eredu_text::tokenizer::ModelChatTemplate::Named`
collection. The override takes precedence over checkpoint templates and retains
the checkpoint's tokenizer variables, EOS ids, and generation defaults.
`model.set_chat_template(Some(template))` replaces the template for subsequent
preparation; already prepared chats keep their prompt and protocol metadata.

With default text options, loading uses checkpoint templates. When the
checkpoint provides no template, loading still succeeds, raw
token generation remains available, and `prepare_chat` returns
`TextModelError::MissingChatTemplate`. This applies to every model family,
including Gemma 4; Eredu does not supply implicit templates.
Supply an explicit template before preparing chat. Calling
`set_chat_template(None)` disables chat preparation without selecting a fallback.
Custom templates use the same rendering and protocol checks as checkpoint
templates; supplying one does not by itself enable native tools or reasoning.

For templates without recognized semantic support, use
`model.generate_prepared_text(request)` with `PreparedChatGenerationRequest`.
For supported drafting, use `generate_prepared_text_speculative` or
`generate_prepared_text_speculative_batch` with the corresponding prepared-chat
speculative requests. These methods emit literal `TextDelta` and `Finished`
events and return TTFT through `output.timing().time_to_first_token()`.
Inspect `prepared.text_generation_support()` for admission: native tool
declarations and required calls are rejected, and explicit thinking requires
`allow_unparsed_reasoning`. Emulated tools can use ordinary prompt text.

Realtime speech uses `eredu::api::realtime::PreparedRealtimeModel<M>` and
`eredu_runtime::RealtimeSessionScheduler`. For MLX,
`eredu_backend_mlx::create_realtime_execution` loads an architecture-prepared
model and returns its context and execution mechanism. The runtime scheduler
accepts portable host frames and commits completed transitions before host
output is delivered. Released sessions use the generic `ReleasedRealtimeSession`
type.

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
- [MLX backend and platform setup](https://github.com/jbg/eredu/blob/main/eredu-backend-mlx/doc/README.md)
- [Eredu CLI](https://github.com/jbg/eredu/blob/main/eredu-cli/README.md)

The minimum supported Rust version is 1.89.

## License

Licensed under either Apache-2.0 or MIT.
