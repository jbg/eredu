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
piped or redirected. When standard output is a terminal, reasoning is shown on
standard error before the visible response and dimmed when standard error is
also a terminal. Reasoning remains hidden when standard output is piped or
redirected. Diagnostics, verbose reasoning output, timing, memory statistics,
and stop reasons are also written to standard error.

`--verbose` also writes the complete effective `ExecutionPlan` as JSON before
model loading. This includes device placement, residency budgets, checkpoint
transformation, mapped-shard limit, expert-cache allocation, drafting, and the
MLX allocator-cache limit. The plan remains on standard error, so generated
output on standard output stays machine-readable or pipeable.

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

Use `--telemetry-json PATH` to write a versioned machine-readable report of the
concrete execution plan, timings, MLX allocator memory, bounded residency,
expert-cache occupancy, and speculative-decoding statistics. Generated text
continues to use standard output and the telemetry file is written only after a
successful run:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --telemetry-json run.json \
  "Explain automatic model placement."
```

The plan and telemetry schemas are public `safemlx-lm` API types. Hardware
discovery is available through `discover_hardware`, and `inspect_model` reports
header-only model resource accounting in `ModelInspectionReport::resources`.
Unknown materialized, per-layer, expert, or device-memory values are represented
as unavailable rather than zero.

## Automatic single-device planning

Use `--auto plan` to inspect the selected device and checkpoint headers, print a
versioned `ExecutionPlanReport` as JSON, and exit without loading the model or
reading a prompt:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --device gpu:0 --auto plan > plan.json
```

Normal generation uses the quick automatic policy by default. `--auto quick`
is therefore optional and can still be written explicitly:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --device gpu:0 --auto quick \
  "Explain automatic model placement."
```

The quick policy is intentionally limited to one execution device. It reserves
30% of currently available memory, prefers fully resident execution, then
host-backed layerwise execution, then dense disk streaming. Bounded candidates
undergo an exact metadata-only capacity probe for pinned parameters and their
device window before selection. For admitted nonresident MoE checkpoints it
assigns 40% of each bounded tier budget to the independent expert cache. It also
enables three-token adaptive drafting when a supported SafeTensors configuration
advertises embedded MTP layers.

If live availability cannot be observed on a unified-memory system, the
prototype derives a budget from physical capacity while retaining the 30%
headroom. If neither live nor unified physical memory can be observed, it uses
documented 4 GiB device and 16 GiB host fallbacks. The JSON explanation records
the chosen basis and every rejected candidate. `--device` chooses the single
device being planned; the remaining explicit performance knobs are layered over
the selected plan as described below.

Explicit performance flags are applied after quick selection and take
precedence over the corresponding planned values. For example, this keeps the
automatically chosen plan but overrides its mapped-shard limit and, when the
selected residency uses one, its device budget:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --device-budget-bytes 34359738368 \
  --mapped-shards 8 "Explain automatic model placement."
```

An explicit residency flag such as `--layerwise-host` or
`--dense-disk-stream` replaces the automatically selected residency mode while
retaining other planned settings unless they are also overridden. Use
`--no-auto` to bypass planning entirely and recover the explicit/default-only
CLI behavior. Overrides are reflected in `--auto plan`, verbose plan output,
and telemetry. Isolated `--auto benchmark` runs reject overrides because each
candidate must be benchmarked as one exact plan.

Use `--auto benchmark` to time every resource-admitted residency, expert-cache,
and embedded-MTP variant in a fresh child process. A failed or out-of-memory
candidate is recorded without contaminating the allocator state of later
trials. The command prints a JSON benchmark report and chooses the plan with the
highest median generation rate:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --device gpu:0 --auto benchmark \
  --auto-benchmark-tokens 32 --auto-benchmark-runs 3 \
  --auto-benchmark-timeout-seconds 300 \
  --auto-cache .cache/safemlx-plans.json > benchmark.json
```

The optional `--auto-cache PATH` is shared by all automatic modes. Benchmark
stores its winner; `plan` and `quick` reuse a matching entry and otherwise
store their heuristic choice. Cache keys include the planner schema, resolved
artifact path, checkpoint file sizes and modification times, tensor catalog,
hardware capacity and selected device. Cache publication uses a temporary file
and atomic rename, so an interrupted writer does not leave a partial JSON file.
Hits are rechecked against current available memory and exact loader admission.
Each benchmark child is also terminated after the configured timeout.

The CLI delegates heuristic selection and plan-to-loader conversion to the
public `safemlx-lm` automatic planner. Feed one or more telemetry documents from
earlier runs back into a later session with `--auto-feedback PATH` (the option
may be repeated). Each path may contain one `ExecutionTelemetry` object or an
array. Matching plans are ranked by median decode throughput and revalidated
against the current artifact, hardware, free memory, and loader admission. A
feedback request bypasses an existing cache hit and publishes the newly selected
plan afterward:

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --device gpu:0 --auto quick \
  --auto-feedback previous-run.json --telemetry-json next-run.json \
  "Explain telemetry-guided planning."
```

```sh
cargo run --release -q -p safemlx-lm-cli -- \
  --model /path/to/model --auto quick \
  --auto-cache .cache/safemlx-plans.json \
  "Explain plan caching."
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
