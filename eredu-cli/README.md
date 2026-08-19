# Eredu CLI

`eredu-cli` is the command-line interface for running local models with Eredu.
The package installs an `eredu` executable and supports both interactive use
and script-friendly input, output, planning, and telemetry.

It accepts a local Hugging Face-style model directory, a local GGUF file or
canonical shard, or a model identifier already present in the local Hugging
Face cache. It does not download models.

## Basic use

```sh
cargo install eredu-cli

eredu --model /path/to/model \
  "Write a Rust function that adds two integers."
```

From a workspace checkout, use `cargo run --release -p eredu-cli --` in place
of `eredu`.

The default device is `gpu:0`. Select another device and sampling policy with
command-line options:

```sh
eredu --model /path/to/model --device cpu \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 \
  "Tell me a short story."
```

On supported CUDA hosts, install with `cargo install eredu-cli --features cuda`.

Run the command with `--help` for the complete option list.

## Input and output

If the positional prompt is absent, the command reads standard input.
Generated visible text is written to standard output. Reasoning, diagnostics,
timing, memory statistics, and stop reasons are written to standard error, so
the response can be piped safely:

```sh
printf 'Summarize the purpose of MLX.' | \
  eredu --model /path/to/model > response.txt
```

Chat templates are applied automatically. Use `--raw` for an untemplated text
completion and `--thinking on|off` to override templates with a standard
thinking control.

Use `--timing` for concise performance statistics, `--verbose` for execution
details, and `--telemetry-json PATH` for a versioned machine-readable report.

## Automatic planning

Normal generation selects a single-device execution plan automatically. The
planner considers available memory, checkpoint resources, residency policy,
expert caching, and supported embedded drafting.

Inspect the selected plan without loading the model:

```sh
eredu --model /path/to/model --device gpu:0 --auto plan > plan.json
```

Use `--auto benchmark` to evaluate admitted candidates in isolated child
processes. `--auto-cache PATH` stores reusable plan selections, and repeated
`--auto-feedback PATH` options supply compatible execution telemetry to the
planner. Every reused plan is checked against the current artifact, hardware,
available memory, and loader admission.

Explicit residency and budget options override the corresponding automatic
choice. Use `--no-auto` when every execution setting should come from explicit
flags and defaults.

## Loading and memory

Eligible dense checkpoints can be quantized while loading:

```sh
eredu --model /path/to/model --quantize 4 \
  "Explain quantization."
```

Large models can use host-layerwise or dense disk-streamed weight residency.
Supported mixture-of-experts models can use an independent expert cache. These
policies accept explicit host and device budgets and trade transfer or I/O work
for a smaller resident parameter set.

See [Model loading, quantization, and memory](../doc/model-loading.md) for
the policy contracts and accounting model.

## Speculative generation

Supported embedded prediction heads are selected by the execution plan. Supply
an external compatible assistant with `--draft-model`:

```sh
eredu --model /path/to/target \
  --draft-model /path/to/assistant \
  --mtp-draft-device cpu --mtp-draft-tokens 3 \
  "Explain speculative decoding."
```

Speculative generation requires a recognized executable chat protocol and is
not available with `--raw`. See [Speculative decoding and
MTP](../doc/speculative-decoding.md) for compatibility and placement rules.

## Native tool calls

Pass OpenAI-shaped function definitions with `--tools`. Tool generation is
enabled only when the checkpoint has a recognized rendering, constraint, and
parsing profile.

```sh
eredu --model /path/to/model \
  --tools tools.json --tool-choice required \
  "Look up the weather in Bogota."
```

Visible text and canonical tool events are written to standard output. See
[Native tool calling](../doc/tool-calling.md) for schemas and event
semantics.
