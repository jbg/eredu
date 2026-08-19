# Eredu

> **eredu** *noun, Basque* — “model; pattern.”

Eredu is a Rust runtime for local language models. It loads supported
Hugging Face-style SafeTensors repositories and GGUF checkpoints, prepares
tokenizers and chat templates, and provides one API for text, multimodal, and
realtime speech models.

The `eredu` crate is the public entry point. Its default backend is built on
[MLX](https://github.com/ml-explore/mlx), with Metal support on Apple silicon
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

The minimum supported Rust version is 1.89. Add the main crate to your project:

```toml
[dependencies]
eredu = "0.4"
```

The default features include the MLX backend. A backend-neutral build is also
available:

```toml
[dependencies]
eredu = { version = "0.4", default-features = false }
```

To run a supported model from this workspace:

```sh
cargo run --release -p eredu-cli -- \
  --model /path/to/model \
  "Write a Rust function that adds two integers."
```

The CLI is also installable with `cargo install eredu-cli`.

The CLI accepts local model directories, GGUF files, and model identifiers
already present in the local Hugging Face cache. It does not download models.
See the [CLI guide](eredu-cli/README.md) for automatic
planning, quantization, residency, speculative generation, and tool calling.

## Workspace crates

Most applications should depend only on `eredu`. The other crates expose
portable components or implementation layers for specialized use.

| Crate | Role |
| --- | --- |
| [`eredu`](eredu/) | Main model loading, generation, chat, multimodal, realtime, and planning API |
| [`eredu-cli`](eredu-cli/) | Command-line model runner built on Eredu |
| [`eredu-core`](eredu-core/) | Backend-neutral runtime contracts, scheduling, validation, residency, and portable schemas |
| [`eredu-text`](eredu-text/) | Backend-neutral tokenization and chat-template utilities |
| [`eredu-gguf`](eredu-gguf/) | Bounded, framework-independent GGUF reading, writing, and tensor conversion |
| [`safemlx`](safemlx/) | Low-level MLX implementation used by Eredu's default backend |
| [`safemlx-sys`](safemlx-sys/) | MLX C bindings and native build integration |
| [`safemlx-codec`](safemlx-codec/) | MLX neural audio codec implementation used by realtime speech models |

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
- [Platform setup](doc/platforms.md)
- [CLI guide](eredu-cli/README.md)

All published crates are pre-1.0, so applications should expect API changes
between minor releases.

## License

The workspace crates are available under MIT or Apache-2.0 unless a crate or
vendored component states otherwise. See [`LICENSE-MIT`](LICENSE-MIT),
[`LICENSE-APACHE`](LICENSE-APACHE), and the notices inside vendored source
trees.
