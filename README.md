# Eredu

> **eredu** *noun, Basque* — “model; pattern.”

Eredu is a Rust runtime for local language models. It loads supported
Hugging Face-style SafeTensors repositories and GGUF checkpoints, prepares
tokenizers and chat templates, and provides one API for text, multimodal, and
realtime speech models.

The `eredu` crate provides the application-facing operations. Its default
backend is built on [MLX](https://github.com/ml-explore/mlx), with Metal support on Apple silicon
and CPU or CUDA support on x86-64 Linux and Windows. The runtime contracts,
artifact inspection, planning, scheduling, and generation orchestration remain
backend-neutral and can be built without MLX.

Eredu is independent of Apple and is not an official MLX project.

## What it provides

- Local model loading from SafeTensors directories and sharded GGUF files.
- Text, image, video, audio, and realtime codec-token inputs for supported
  architectures.
- Tokenization, chat templates, reasoning events, constrained native tool
  calls, and streaming generation.
- Greedy and stochastic sampling plus embedded or external speculative
  decoding.
- Load-time quantization and bounded weight, cache, and prompt-cache residency.
- Automatic execution planning, admission, telemetry, and supported tensor,
  pipeline, and expert parallel execution.

See [model and checkpoint support](doc/model-support.md) for the current family,
format, modality, and execution matrix.

## Quick start

The minimum supported Rust version is 1.89. The default features include the
MLX backend; the facade can also be built without a concrete backend.

To run a supported model from this workspace:

```sh
cargo run --release -p eredu-cli -- \
  --model /path/to/model \
  "Write a Rust function that adds two integers."
```

The CLI defaults to `gpu:0` when the build includes CUDA or Apple MLX, and to
`cpu` for CPU-only MLX builds on Linux and Windows. Pass `--device cpu` or
`--device gpu:N` to override that selection.

The CLI accepts local model directories, GGUF files, and model identifiers
already present in the local Hugging Face cache. It does not download models.
See the [CLI guide](eredu-cli/README.md) for automatic
planning, quantization, residency, speculative generation, and tool calling.

## Workspace crates

Applications combine facade-owned operations from `eredu` with portable values
from their owning crates, usually `eredu-core` and `eredu-architectures`.
Implementation crates remain specialized dependencies.

| Crate | Role |
| --- | --- |
| [`eredu`](eredu/) | Main model loading, generation, chat, multimodal, realtime, and planning API |
| [`eredu-cli`](eredu-cli/) | Command-line model runner built on Eredu |
| [`eredu-core`](eredu-core/) | Backend-neutral contracts, artifact inspection, planning, validation, and portable schemas |
| [`eredu-runtime`](eredu-runtime/) | Backend-neutral execution, scheduling, cache I/O, and weight and cache residency orchestration |
| [`eredu-architectures`](eredu-architectures/) | Model-family configuration, checkpoint schemas, parameter topology, and backend-neutral execution |
| [`eredu-text`](eredu-text/) | Backend-neutral tokenization and chat-template utilities |
| [`eredu-gguf`](eredu-gguf/) | Bounded, framework-independent GGUF reading, writing, and tensor conversion |
| [`eredu-backend-mlx`](eredu-backend-mlx/) | Concrete MLX tensor, execution, checkpoint, composition, and media mechanisms |
| [`safemlx`](safemlx/) | Low-level MLX implementation used by Eredu's default backend |
| [`safemlx-sys`](safemlx-sys/) | MLX C bindings and native build integration |
| [`eredu-nn`](eredu-nn/) | Backend-neutral tensor and neural-operation contracts for shared architectures |
| [`eredu-codec`](eredu-codec/) | Backend-neutral neural audio codec architectures used by realtime speech models |
| [`eredu-evaluation`](eredu-evaluation/) | Backend-neutral evidence, parity, quality metrics, performance summaries, and evaluation drivers |

The workspace also contains procedural macros, integration tests, and an iOS
example application.

## Documentation

- [Documentation index](doc/README.md)
- [Model and checkpoint support](doc/model-support.md)
- [Model loading, quantization, and memory](doc/model-loading.md)
- [Backend architecture](doc/backend-architecture.md)
- [Parallel execution](doc/parallel-execution.md)
- [Native tool calling](doc/tool-calling.md)
- [Speculative decoding and MTP](doc/speculative-decoding.md)
- [MLX backend and platform setup](eredu-backend-mlx/doc/README.md)
- [CLI guide](eredu-cli/README.md)

All published crates are pre-1.0, so applications should expect API changes
between minor releases.

## License

The workspace crates are available under MIT or Apache-2.0 unless a crate or
vendored component states otherwise. See [`LICENSE-MIT`](LICENSE-MIT),
[`LICENSE-APACHE`](LICENSE-APACHE), and the notices inside vendored source
trees.
