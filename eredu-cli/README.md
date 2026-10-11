# Eredu CLI

`eredu-cli` is the command-line interface for running local models with Eredu.
The package installs an `eredu` executable and supports both interactive use
and script-friendly input, output, planning, and telemetry.

It accepts a local Hugging Face-style model directory, a local GGUF file or
canonical shard, or a model identifier already present in the local Hugging
Face cache. It does not download models.

## Basic use

```sh
eredu --model /path/to/model \
  "Write a Rust function that adds two integers."
```

From a workspace checkout, use `cargo run --release -p eredu-cli --` in place
of `eredu`.

The default device is `gpu:0` when the executable is built with CUDA or Apple
MLX, and `cpu` for CPU-only MLX builds on Linux and Windows. Pin a device and
select a sampling policy with command-line options:

```sh
eredu --model /path/to/model --device cpu \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 \
  "Tell me a short story."
```

Generation settings, including repetition, frequency, and presence penalties,
use values from the checkpoint's `generation_config.json` when supplied there.
Only explicit command-line arguments override those values. For example,
`--presence-penalty 0` disables a checkpoint's presence penalty, and
`--repeat-penalty 1` disables its repetition penalty. Use `--verbose` to see
the effective settings.

On supported CUDA hosts, disabling the default features and enabling `cuda`
replaces the default Metal/Accelerate backend selection instead of combining
native backends.

The `mlx` feature makes the executable available. The `metal` and `cuda`
features both imply `mlx`, while `nccl` implies `cuda`. A bare
`--no-default-features` package check is supported but intentionally does not
build or install an `eredu` executable.

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
Unrecognized templates use literal text generation with TTFT and supported
speculative drafting. They retain EOS and caller stop sequences; native tools
and parsed reasoning still require a recognized protocol.

## Automatic planning

Normal generation selects a single-device execution plan automatically. The
planner considers available memory, checkpoint resources, residency policy,
expert caching, and supported embedded drafting.

Inspect the selected plan without loading the model:

```sh
eredu --model /path/to/model --auto plan > plan.json
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

See [Model loading, quantization, and memory](https://github.com/jbg/eredu/blob/main/doc/model-loading.md) for
the policy contracts and accounting model.

## Speculative generation

Supported embedded prediction heads are selected by the execution plan. Supply
an external compatible assistant with `--draft-model`:

```sh
eredu --model /path/to/target \
  --draft-model /path/to/assistant \
  --speculative-draft-device cpu --speculative-draft-tokens 3 \
  "Explain speculative decoding."
```

Speculative generation requires a recognized executable chat protocol and is
not available with `--raw`. See [Speculative decoding and
MTP](https://github.com/jbg/eredu/blob/main/doc/speculative-decoding.md) for
compatibility and placement rules.

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
[Native tool calling](https://github.com/jbg/eredu/blob/main/doc/tool-calling.md) for schemas and event
semantics.

## Request memory forecasts

Use `--estimate-memory-tokens N --max-tokens M` to inspect input-dependent memory
before loading, or `--memory-report PATH` to report the actual tokenized request
before generation. `--memory-budget-bytes` is a total model-plus-request planning
budget; `--memory-policy warn|refuse` controls predicted shortfalls. The default
reserve is 256 MiB, configurable with `--memory-reserve-bytes`.

Ordinary text prefill defaults to chunks of 512 tokens. Set
`--prefill-chunk-size 0` for one pass. Reports preserve unknown costs and identify
full-prompt fallbacks. See [formulas, coverage and calibration](../doc/generation-memory.md).

## Realtime speech

Build with `realtime,system-audio` for microphone/speaker conversations; both
features are enabled by default. `system-audio` uses CPAL's native platform host.
Linux builds require the host audio development libraries used by CPAL (normally
ALSA). A WAV-only build uses `--no-default-features --features mlx,metal,realtime`
on Apple silicon. CUDA hosts use `--no-default-features --features cuda,realtime,system-audio`.

```sh
eredu realtime --list-audio-devices
MLX_MAX_OPS_PER_BUFFER=25 MLX_MAX_MB_PER_BUFFER=256 \
  eredu realtime --model /models/voicechat-official \
  --base-assets /models/nemotron-base --device gpu:0 \
  --quantize 8 --speech-weights f16 --speech-compute f16 \
  --seconds 60 --output speech.wav --events events.jsonl
```

The command-buffer settings in this example identify the measured M3 Ultra
configuration; other hardware can need different batching.

The default microphone and speaker are selected automatically. Use exact names
from `--list-audio-devices` with `--input-device` and `--output-device` to select
other devices. Headphones avoid acoustic feedback. First Ctrl-C closes capture
and drains the response tail; a second Ctrl-C cancels. Queue storage is bounded
by `--audio-buffer-ms`; capture overflow reports a continuity error.

On-load quantization targets eligible backbone/readout, function-head and causal
speech-transformer projections, uses the bounded prepared-checkpoint cache, and
keeps sensitive floating operations intact. `--quantize-components` selects a
subset. Quantized causal projections retain F32 arithmetic. Released packed
checkpoints load without `--quantize`; re-quantization is rejected.

Use `--config voicechat.json` for tool declarations, split GGUF sources and precise
resource configuration. Device, sampling, seed, mixed precision, parameter budgets
and bounded host/disk residency have command-line overrides. Model identifiers
must already be cached; auxiliary assets are admitted from the exact official
and base tokenizer/configuration sources described in
[VoiceChat validation](../doc/voicechat-validation.md).

Recorded input and unattended measurements use the same controlled driver:

```sh
eredu realtime --config voicechat.json --input conversation.wav \
  --offline --no-playback --quantize 8 \
  --speech-weights f16 --speech-compute f16 \
  --output speech.wav --events events.jsonl
```

Input WAV must be mono 16 kHz; output is mono 22.05 kHz float32. Output paths must
be new. Agent text is written to stdout; diagnostics and warm execution percentiles
are written to stderr. JSONL records committed text, transcription, tools, turns,
wall time and invocation execution time. Execution timing excludes input pacing
and output delivery; it does not measure complete microphone-to-speaker latency.

Run `eredu realtime --help` for the complete option list.
