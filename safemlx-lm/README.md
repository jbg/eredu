# safemlx-lm

`safemlx-lm` is the public Rust facade and MLX backend for loading and
running supported language, multimodal, and realtime speech models. Portable
contracts and orchestration live in the MLX-independent `safemlx-lm-core`
crate. The facade provides architecture
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
creation, and memory admission. `safemlx_lm::load_model_with_options` is the
single model-loading entry point for supported checkpoint formats and
architectures. The MLX backend performs architecture dispatch internally.
Realtime Moshi/PersonaPlex requests use the backend-neutral core scheduler;
only their tensor execution and exact MLX completion adapter remain here.
Distributed pipeline schedule, cancellation, and completion agreement use the
same core scheduler through the MLX collective and completion adapters; the
facade does not maintain a second lifecycle implementation.
Weight-residency plans, atomic admission, ownership leases, protected windows,
eviction decisions, exact transfer generations, and accounting likewise come
directly from `safemlx-lm-core`. The MLX facade mirrors those transitions with
concrete arrays and host buffers and owns native materialization, event/source
retention, physical-capacity queries, and allocator sampling.

## Automatic execution planning

Embedding applications can use the same versioned single-device planner as the
example CLI. The request, policy, selected plan, explanation, and runtime
telemetry all implement Serde serialization:

```rust,no_run
use safemlx_lm::{
    plan_automatic_execution, AutomaticPlanRequest, BackendKind, DevicePlan,
    ExecutionTelemetry,
};

let prior_runs: Vec<ExecutionTelemetry> = load_prior_runs();
let request = AutomaticPlanRequest::new(
    "/path/to/model",
    DevicePlan { backend: BackendKind::Metal, index: 0 },
)
.with_prior_telemetry(prior_runs);
let report = plan_automatic_execution(&request)?;

# fn load_prior_runs() -> Vec<ExecutionTelemetry> { Vec::new() }
# Ok::<(), safemlx_lm::error::Error>(())
```

Matching prior runs are grouped by exact plan and ranked by median decode
throughput. Telemetry from a different model, artifact layout, device, stable
hardware profile, or schema is ignored. Historical plans are also rechecked
against current available memory and header-only loader admission. Use
`AutomaticPlanner::new` with an `AutomaticPlannerPolicy` for explicit policy
bounds, and `execution_plan_load_options` to apply a returned plan to model
loading. Device creation and speculative-generation orchestration remain owned
by the embedding application.

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

- [Backend architecture](../doc/backend-architecture.md)
- [Native tool calling](../doc/tool-calling.md)
- [Speculative decoding and MTP](../doc/speculative-decoding.md)
- [Cancellation and bounded execution](../doc/cancellation.md)
- [CLI example](../examples/safemlx-lm-cli/README.md)
- [Platform setup](../doc/platforms.md)

The minimum supported Rust version is 1.89.

## License

Licensed under either Apache-2.0 or MIT.
