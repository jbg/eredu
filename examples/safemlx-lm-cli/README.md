# safemlx-lm CLI example

This workspace crate is a script-friendly text generator built on
`safemlx-lm`. It is an example application rather than a published command-line
product.

It accepts:

- a local Hugging Face-style model directory;
- a local GGUF file, including canonical shards; or
- a Hugging Face model identifier already present in the local cache.

The CLI never downloads a model. Cached identifiers are resolved from
`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, or `HF_HOME`.

## Basic use

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  "Write a Rust function that adds two integers."
```

The default device is `gpu:0`. Select CPU or another visible GPU explicitly:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model --device cpu \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 \
  "Tell me a short story."
```

On a supported CUDA host, build the example with its `cuda` feature:

```sh
cargo run --release -p safemlx-lm-cli --features cuda -- \
  --model /path/to/model "Explain MLX."
```

Append a case-insensitive quantization selector to a cached GGUF identifier
when the repository contains multiple files:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model unsloth/Qwen3-0.6B-GGUF:Q4_K_M \
  "Explain imaginary numbers."
```

## Input and output

When the positional prompt is omitted, the command reads it from standard
input. Generated visible text is streamed to standard output, so it can be
piped or redirected. Diagnostics, reasoning output in verbose mode, timing,
memory statistics, and stop reasons are written to standard error.

```sh
printf 'Summarize the purpose of MLX.' | \
  cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model > response.txt
```

Chat templates are applied automatically. Use `--raw` for an untemplated text
completion. `--thinking on|off` overrides templates that expose the standard
thinking control; `auto` preserves the checkpoint's default.

Use `--timing` to print load time, generation time, time to first token, and
token throughput to standard error without enabling the other `--verbose`
diagnostics:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --timing \
  "Explain speculative decoding."
```

## Loading and memory

Eligible dense checkpoints can be quantized while loading:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model --quantize 4 \
  "Explain quantization."
```

Large models can use bounded weight policies:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model --layerwise-host \
  --device-layer-window 1 \
  --host-budget-bytes 24000000000 \
  --device-budget-bytes 8000000000 \
  "Summarize bounded residency."
```

Use `--dense-disk-stream` to stream ordinary layers through bounded disk,
host, and device tiers. Supported MoE families can add `--expert-cache` with
separate `--expert-cache-*-budget-bytes` limits. These modes trade startup or
per-token I/O for a smaller logical parameter working set; on Apple silicon,
host and device tiers still share physical unified memory.

See [Model loading, quantization, and memory](../../doc/model-loading.md) for
the policy contracts and accounting limits.

## Speculative generation

Models with embedded prediction heads use them automatically. An external
compatible assistant can be supplied with `--draft-model`:

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/target \
  --draft-model /path/to/assistant \
  --mtp-draft-device cpu --mtp-draft-tokens 3 \
  "Explain speculative decoding."
```

`--mtp-draft-device target` reuses the target stream. An explicit `cpu` or
`gpu:N` creates a separate draft stream. Use `--disable-mtp-lookahead` for the
canonical no-lookahead comparison. Details and capability limits are in
[Speculative decoding and MTP](../../doc/speculative-decoding.md).

For a cached GGUF repository containing one recognized draft sidecar, the bare
repository ID selects it without requiring a local file path. For example:

```sh
target/release/safemlx-lm \
  --model 'meta-models/Muse-Glimmer-30B-GGUF:17gb' \
  --draft-model meta-models/Muse-Glimmer-30B-GGUF \
  --mtp-draft-tokens 3 \
  "Explain speculative decoding."
```

Append a selector such as `:dflash-kquant` when a repository contains more
than one compatible draft GGUF.

## Native tool calls

Pass an array of OpenAI-shaped function definitions with `--tools`. Native tool
generation is fail-closed: the checkpoint must have a recognized rendering,
parsing, and constraint profile.

```sh
cargo run --release -p safemlx-lm-cli -- \
  --model /path/to/model \
  --tools tools.json --tool-choice required \
  --max-parallel-tool-calls 2 \
  "Look up the weather in Bogotá."
```

Visible text and canonical tool events are written to standard output. See
[Native tool calling](../../doc/tool-calling.md) for supported schema behavior
and event semantics.

Run the command with `--help` for the complete sampling, repetition penalty,
cache, profiling, and diagnostic option list.
