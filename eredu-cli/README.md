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
for a smaller resident parameter set. Eligible on-load quantization composes with
these residency policies; exact architecture/backend admission checks support.

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

## Typed decision reads

`--decisions PATH` reads evidence and typed questions from JSON. Use `-` for
standard input. This mode requires a model exposing block refinement, including
DiffusionGemma. The existing device, quantization and weight-residency options
apply. No generated answer text is parsed: the CLI reads model scores directly
at the compiled answer positions.

```json
{
  "state": {"ticket": "I was charged twice; please refund the duplicate payment."},
  "steps": 2,
  "seed": 42,
  "questions": [
    {"id": "department", "type": "choice", "question": "Which department should handle this?", "options": ["billing", "technical", "sales"]},
    {"id": "refund", "type": "noul", "question": "Does the customer request a refund?"},
    {"id": "priority", "type": "score", "question": "Rate the urgency.", "levels": [
      {"label": "routine", "value": 0, "description": "Can wait for ordinary processing"},
      {"label": "urgent", "value": 10, "description": "Requires immediate attention"}
    ]}
  ]
}
```

Save this as `request.json`, then run:

```sh
eredu --model /path/to/diffusiongemma --decisions request.json --timing
eredu --model /path/to/diffusiongemma --decisions - < request.json
```

Standard output contains one JSON object with `answers` keyed by question ID:

- `choice` returns the selected option, its confidence and every option's probability.
- `noul` returns a boolean `value` and a `probability` of yes, plus both alternatives.
- `score` returns the expected numeric `score` over the declared levels and the
  highest-probability `level`, plus the level distribution. Values must be finite
  and strictly increasing; level descriptions define their meaning.

`normalization: "allowed_labels"` identifies probabilities normalized over the
specified alternatives. `label_mass` reports how much of the original full-model
probability those labels cover. These preferences are not calibrated correctness
estimates. Ties choose the first declared alternative. Questions sharing a canvas
can affect one another; their distributions are not independent joint probabilities.

The CLI encodes each alternative as a verified single-token letter within its
answer template, pins the scaffold and padding, and leaves answer slots as noise.
It executes exactly `steps` refinement passes, then reads raw position scores and
stops without committing generated tokens. `steps` defaults to 1 and accepts
1..=64; `seed` defaults to 0, with `--seed` overriding the request value.
Requests are limited to 1 MiB, 1..=32 unique questions and 2..=26 alternatives
per question. The complete template must fit the model's trained canvas. Invalid
schemas or unsuitable label tokenization produce errors rather than repaired output.

Generation sampling flags, speculative drafting, ordinary prompt/tool options,
and generation-specific memory/telemetry reports do not apply to decision reads
and conflict with this mode. `--auto quick`, `--auto plan`, `--no-auto`, and
`--timing` remain available; generation benchmarks do not measure decision reads.
Timing output separates request preparation, context encoding, refinement and
completion. Refinement timing includes score capture and sampler processing;
context timing includes controlled-session setup. Prompt and canvas token counts
help identify the amount of work performed. Preparation is further divided into
canvas tokenization, chat preparation, capture discovery and capture admission.
First capture discovery establishes content-exact checkpoint provenance by hashing
all source files; large multi-shard checkpoints use bounded parallel hashing.
The result is retained for subsequent reads on the same loaded model.

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

System-audio builds include WebRTC echo cancellation and require Meson, Ninja,
a C/C++ toolchain and `rustup component add llvm-tools-preview`. On macOS,
`brew install meson ninja pkg-config` supplies the additional build tools. The
bundled library downloads pinned Abseil sources when a suitable system dependency
is unavailable.

```sh
eredu realtime --list-audio-devices
MLX_MAX_OPS_PER_BUFFER=25 MLX_MAX_MB_PER_BUFFER=256 \
  eredu realtime --model /models/voicechat-official \
  --base-assets /models/nemotron-base --device gpu:0 \
  --quantize 8 --speech-weights f16 --speech-compute f16 \
  --seconds 60 --output speech.wav --events events.jsonl
```

Cached official repositories can be passed directly:

```sh
eredu realtime --model nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --base-assets nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --quantize 8 --speech-weights f16 --speech-compute f16
```

`--base-assets` supplies `config.json`, `tokenizer.json` and
`tokenizer_config.json` from the base model. VoiceChat supplies the backbone
weights. The validated revision is `6533e8de2c68e4536bf7c411d7a3ce5734111476`. If these
files are missing from the cache, download only the assets:

```sh
hf download nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  config.json tokenizer.json tokenizer_config.json \
  --revision 6533e8de2c68e4536bf7c411d7a3ce5734111476
```

The resulting snapshot directory is also accepted by `--base-assets`.
`--official-assets` defaults to the `--model` repository for direct model requests.
Text and realtime commands share model/revision, device, allocator cache,
quantization, sampling and common residency argument definitions and normalization.
Omitted sampling controls preserve each command's defaults or supplied configuration.
Both commands default to four cached checkpoint readers.

The command-buffer settings above identify the measured M3 Ultra
configuration; other hardware can need different batching.

The default microphone and speaker are selected automatically. Use exact names
from `--list-audio-devices` with `--input-device` and `--output-device` to select
other devices. Headphones avoid acoustic feedback. First Ctrl-C closes capture
and drains the response tail; a second Ctrl-C cancels. Queue storage is bounded
by `--audio-buffer-ms`; capture overflow reports a continuity error.

Live microphone-and-speaker sessions enable WebRTC AEC3 echo cancellation by
default. The reference is PCM actually sent to the speaker, including playback
silence, clipping, underruns and interruptions. Cancellation runs in 10 ms host
frames after microphone resampling, while capture stays live for user barge-in.
`--no-echo-cancellation` disables it, for example when comparing headphone input.
File input and `--no-playback` do not use echo cancellation. Automatic gain
control is disabled. Cancellation needs time to
adapt to the acoustic path; it does not guarantee removal of all speaker leakage.

Push-to-talk, noise suppression and speech admission are opt-in. They configure
host input processing; the ordinary controlled conversation driver continues to
receive timed PCM, including silence when input is muted.

- `--push-to-talk` or `--push-to-talk toggle` starts muted. Tap Space to alternate
  between listening and muted; do not hold it in toggle mode because ordinary
  terminals can report autorepeat as repeated presses. `--ptt-key enter` or a
  single ASCII letter/digit selects a different key. PTT requires interactive
  stdin, stdout and stderr.
- `--push-to-talk hold` admits capture while the key is held. Unix terminals must
  support keyboard-enhancement key-release events; unsupported terminals report
  an error before model loading. Windows console key-release events are native.
  Terminal focus loss closes admission when reported. The CLI restores terminal
  mode on normal exit, cancellation and errors. Ctrl-C retains its two-stage
  close/cancel behavior. Permission follows captured samples through buffering;
  the echo/noise filters continue operating on real microphone PCM while muted.
- `--noise-suppression` enables WebRTC moderate suppression. Choose `off`, `low`,
  `moderate`, `high` or `very-high` explicitly to compare strengths. It operates
  after microphone resampling, in the same processor as echo cancellation, and
  works with `--no-playback` or `--no-echo-cancellation`. It is microphone-only;
  stronger suppression can distort speech and does not identify the intended speaker.
- `--speech-admission` enables a sustained-level gate for microphone or WAV input.
  `--speech-threshold-dbfs` sets the opening RMS level after filtering (default
  `-40`), and `--speech-hysteresis-db` sets the reduction in closing level (default
  `6`). `--speech-min-ms` rejects shorter peaks (default `80`);
  `--speech-preroll-ms` retains preceding audio (default `160`);
  `--speech-hangover-ms` retains quieter endings (default `300`). Pre-roll and
  hangover use whole 10 ms windows. Admission adds pre-roll plus minimum-duration
  lookahead, 240 ms with these settings, and preserves the exact source sample
  count at shutdown. Rejected windows become silence. Distant speech above the
  threshold still passes; this is level admission, not speaker recognition.

These options combine, so speech admission can reject quiet material while PTT
is open. Tune the threshold using `--record-input` with admission disabled and
separate quiet-background and normal-speaking recordings; a whole-session RMS
includes silence and cannot identify a useful speech threshold by itself.

`--system-prompt` overrides the conversation's system prompt, including one from
`--config`, without changing input processing. Prompts must contain ASCII text.
For example:

```sh
eredu realtime \
  --model nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --base-assets nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --quantize 8 --speech-weights f16 --speech-compute f16 \
  --noise-suppression moderate --speech-admission --speech-threshold-dbfs -35 \
  --system-prompt 'You are a helpful voice assistant. Wait silently until the user makes a clear request addressed to you. Do not respond to background conversation or incomplete fragments. If a request is unclear, ask a brief clarifying question.'
```

Add `--push-to-talk` for manual admission. Prompt instructions are an experiment;
the model may not follow them reliably and they do not replace audio admission.

In an interactive terminal, conversation text appears on stderr as separate
`Heard:` and `Agent:` lines. Token fragments are joined into readable chunks;
transcription is flushed after an 800 ms pause, at a turn boundary or at shutdown.
Long chunks are split at word boundaries where available, with a 240-byte limit.
Agent text is flushed at sentence and turn boundaries. Redirected stdout preserves the raw
streaming agent text for scripts; transcription remains on stderr.

Transcription is shown by default. It uses the model's existing perception output
and requires no additional inference. `--no-transcription` hides it. The
transcription is the model's speech recognition output, not an explanation of its
response. The model can initiate an agent turn without recognized user speech;
transcription is not an admission gate for generated responses. The backbone
consumes audio features directly, so a response preceding transcript text does
not imply that the model has received no audio.

For microphone diagnostics, use `--record-input microphone.wav` to save the
resampled and filtered source audio accepted by the session as mono 16 kHz
float32 WAV.
The recording includes PTT/admission silence in place of rejected input.
The recording excludes injected silence and response-tail padding. The CLI reports
the selected microphone device and native format at startup, and accepted sample
count, RMS level and peak level at shutdown. Background dialogue can reach the
model even when the operator is silent.

To distinguish live capture from model behavior, record a run with
`--record-input microphone.wav --events microphone.jsonl`, then listen to the
recording and replay it with the same model/precision options plus
`--input microphone.wav --offline --no-playback --events replay.jsonl`.
JSONL preserves individual text deltas and source-frame attribution regardless
of terminal grouping.

On-load quantization targets eligible backbone/readout, function-head and causal
speech-transformer projections, uses the bounded prepared-checkpoint cache, and
keeps sensitive floating operations intact. `--quantize-components` selects a
subset. Quantized causal projections retain F32 arithmetic. Released packed
checkpoints load without `--quantize`; re-quantization is rejected.

Startup reports planned memory for device-side weights, separate host-side weight
pools, loading/conversion workspace, and cached computations. These are admission
allowances, not measured process RAM usage. Conversation history and temporary
computation buffers require additional memory. On Apple silicon, device and host
allocations share physical RAM; the loading workspace is a loading reservation,
not an additional permanent weight copy.

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
be new. Redirected agent text is written to stdout; terminal conversation text,
diagnostics and warm execution percentiles are written to stderr. JSONL records
committed text, transcription, tools, turns, wall time and invocation execution
time. Execution timing excludes input pacing and output delivery; it does not measure complete
microphone-to-speaker latency.

Run `eredu realtime --help` for the complete option list.
