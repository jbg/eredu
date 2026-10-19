# NemotronLabs VoiceChat component and artifact validation

`eredu_architectures::voicechat` provides portable configuration, checkpoint
schemas, canonical source recipes, retained component sources, host PCM/mel
scheduling, causal perception, RNNT neural prediction/joint execution and shared
Nemotron input/readout components, composed cached EAR-TTS generation, character
conditioning, fixed-Aria prompt preparation and codec residual quantization.
Prepared speech materialization binds individual learned units and consumes them
into the components used by the coupled executor, retaining their bound weights.
`eredu_codec::ear` supplies finite STFT/ConvNeXt waveform encoding and the
ConvNeXt/ISTFT code-to-waveform decoder. The shared architecture registry
recognizes nested NeMo and component-based VoiceChat configurations and complete
`nemotron_voicechat` GGUF metadata as realtime VoiceChat. Generic realtime
preparation retains the sources used to construct a single-device speech
session. MLX CPU has RNNT, perception, channel-boundary, cached TTS, character
conditioning, MoG/RVQ, codec encoding/decoding and retained-sampling component
evidence; complete-model residency/parallel execution has no complete VoiceChat
validation results. The realtime CLI completes the pinned released checkpoint on
Metal with speech, text, transcription and drain, as recorded below. Independent
complete-model numerical parity and speech quality remain unvalidated. No
hardware latency claim follows from storage admission alone; resident performance
measurements have the explicit precision, workload and timing scope below.

## Realtime audio, WAV and recorded tool responses

`eredu realtime` consumes mono 16 kHz WAV input through
`BoundVoiceChatSession`, writes mono 22.05 kHz float32 speech and records committed
frame, agent-text, transcription, tool and turn events as JSONL. PCM and output
delivery are bounded. Speech sample offsets are contiguous in the output WAV;
JSONL retains model/audio positions and codec delay separately from the explicit
invocation timestamp (`at_ms`). The default uses the system microphone and speaker
with wall-clock pacing; `--input` selects a file, and `--offline --no-playback`
selects recorded-time execution without opening audio streams. External tool
execution is application-owned; `--script` supplies recorded tool results.

`--quantize 8 --quantization-group-size 64` selects bounded preparation of
backbone/readout, function-head and causal speech-transformer matrices.
`--quantize-components` restricts those groups. Packed checkpoints load through
the same path without `--quantize`; implicit re-quantization is rejected.
`--speech-weights f16 --speech-compute f16` selects narrow floating execution
for eligible unquantized speech/perception work. Quantized causal projections
retain F32 companions and arithmetic. The bounded prepared-checkpoint cache
persists encoded weights; reuse checks source identity and exact recipes.

System audio uses optional CPAL devices and bounded lock-free queues. Native
sample formats convert to mono and band-limited streaming resampling supplies
16 kHz input and device-rate playback of 22.05 kHz output. Audio callbacks
perform scalar conversion and queue access only. Playback starts after a
160 ms preroll, or at a flushed phrase boundary; input overflow is a continuity
error. Interruption drops queued older speech; samples already submitted to
the operating system remain subject to device latency. First Ctrl-C closes
input and drains the response tail; a second Ctrl-C cancels the bound session.
`--seconds` and `--max-frames` bound capture duration. `--audio-buffer-ms` bounds
native queue storage. Headphones avoid feeding generated speech back into capture.

JSONL adds `wall_ms` and per-invocation `execution_ms`. The latter spans
advancement through committed-output retrieval, excluding input pacing and
output delivery. The stderr summary excludes the first ten live frames and
compares mean/p95/p99 execution with the 80 ms frame budget; it is not a complete
callback-to-speaker latency measurement.

The CLI uses the shared loading entry points: `prepare_realtime_model` performs
architecture-owned family discovery, source admission and auxiliary-asset retention;
`prepare_mlx_realtime` selects cache mechanisms and parameter pools without native
allocation; consuming `PreparedMlxRealtime::load` constructs the selected execution.
The same preparation and loading interfaces admit Moshi/PersonaPlex categorical
frames. `RealtimeLoadOptions` distinguishes categorical frames from PCM speech;
a protocol mismatch fails before binding. The serialized checkpoint-source contract
is shared with the architecture layer. The CLI supplies paths and policy, not
family detection, required component lists, source recipes or cache selection.

All artifacts are local. `official_assets` contains `config.json` and the three
`rnnt_tokenizer/{tokenizer.model,tokenizer_config.json,vocab.json}` files from
official revision `443794ea956ef0065f001967ffd00e77f519cb39`. `base_assets` contains
`config.json`, `tokenizer.json` and `tokenizer_config.json` from base revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`. Preparation verifies exact auxiliary
asset digests, tensor geometry, control values and tokenizer declarations.
Checkpoint storage remains outside the source tree. Native construction consumes
retained source views. The CLI accepts local paths or repositories already present in the Hugging Face cache;
it does not download artifacts.

A configuration file has the following shape. These byte limits are illustrative
admission ceilings, not measured memory requirements or hardware recommendations.
Parameter admission prints selected totals before native allocation; mutable
state, activations and process overhead are separate.

```json
{
  "official_assets": "/models/voicechat-official",
  "base_assets": "/models/nemotron-base",
  "source": {"format": "safetensors", "path": "/models/voicechat-official"},
  "system_prompt": "You are a helpful assistant.",
  "tools": [],
  "device": "cpu",
  "parameter_residency": {"mode": "resident"},
  "device_parameter_bytes": 68719476736,
  "host_parameter_bytes": 68719476736,
  "max_branch_bytes": 8589934592,
  "max_total_copy_bytes": 68719476736,
  "max_frames": 4096,
  "max_model_steps": 8192,
  "seed": 0
}
```

`source.format="safetensors"` accepts the official dense tensors and released
community affine matrices through one admission and construction path. Matrix
packing and group geometry determine each format; strict admission checks every
scale/bias companion and any explicit per-module declaration. Packed matrices
remain encoded through the ordinary parameter providers and prepared-checkpoint
cache. Nested NeMo and component-based inference configurations normalize to the
same released equations; unknown equation overrides are rejected. Community
sources use the same pinned `official_assets` and `base_assets` above for auxiliary
configuration, RNNT tokenization and prompt preparation. They require no on-load
quantization or alternate native loader.

Artifact paths resolve relative to the configuration file. `source.format="gguf"`
accepts one complete container. `source.format="split_gguf"` requires `perception`,
`backbone`, `function_head`, `tts` and `companion` paths. The companion is a complete
official SafeTensors source or complete GGUF supplying missing RNNT/codec-encoder
roles. `parameter_residency.mode` also accepts `host` or `disk`, with explicit
`backbone_device_bytes`, `backbone_host_bytes`, `speech_device_bytes` and
`speech_host_bytes`. Speech limits apply to each independent provider; enclosing
budgets account for all providers together. Host mode accepts `depth` (default 1);
disk mode with zero component host budgets uses direct streaming. Cache realization
uses actual MLX mechanism facts with device storage. `device="gpu"` requires a
compiled, available accelerator.

```sh
CARGO_INCREMENTAL=0 cargo run -p eredu-cli --no-default-features --features mlx,realtime -- realtime --offline --no-playback --config voicechat.json --input conversation.wav \
  --output speech.wav --events events.jsonl --script responses.json \
  --chunk-samples 640 --tail-frames 200 --max-steps 100000
```

For Metal, add `metal` to the features and select `device="gpu"`. Output paths
must be new. The optional script maps zero-based committed tool-batch ordinals to
responses passed verbatim to the public tool-response API. Tool definitions use
the existing chat-template JSON types.

```json
[
  {
    "call_index": 0,
    "after_ms": 800,
    "response": "<TOOL_RESPONSE>The temperature is 20 degrees.</TOOL_RESPONSE>"
  }
]
```

`after_ms` is relative to the batch's committed publication time. The optional
`initial` script string overrides that call's configured initial reminder;
omission uses its configured selection. Optional `generic` accepts at most two
distinct recorded phrases. The portable API additionally supports suppressing
initial speech per call through `VoiceChatInitialReminder::Disabled`, alongside
`Configured` and `Text` choices. System prompts, responses and speech phrases must
be ASCII. Responses bind to exact
request/turn/batch tickets; interrupted or stale responses are discarded. Input
continues through tool waits. Completion polling holds recorded time, and tool
substeps do not advance the input-audio clock. Missing responses use the facade's
default timeout policy.

By default, each new tool call selects uniformly with replacement from the pinned
streaming configuration's 20 acknowledgement phrases and ten initial reminders.
The CLI initializes independent retained host streams with `seed` for
acknowledgements and `seed XOR 0x52454d494e444552` for initial reminders. Both
selections commit at SOTC. Discarded transitions consume no canonical draw, and
snapshots/forks preserve complete generator state. The streams use ChaCha8 with
the Rand 0.9 uniform distribution, not Python's random sequence or neural RNG.
`VoiceChatStartOptions::{acknowledgement, initial_reminder}` expose these choices
through `VoiceChatPhrasePolicy::{Disabled, Fixed, Streaming}`.

An optional configuration field `"acknowledgement": "Happy to help with that."`
selects a fixed acknowledgement. With no fixed phrase,
`"random_acknowledgement": false` disables that role. The corresponding
`initial_reminder` string and `random_initial_reminder` boolean configure initial
speech. Preparation uses the retained tokenizer and validates ASCII and complete
framed token bounds. Acknowledgement BOS, text, trailing PADs and EOS render
through ordinary Tool TTS/codec invocations, while agent feedback remains PAD.
Only committed Tool work advances its cursor; perception does not. Collection end
or interruption clears the remaining acknowledgement suffix, and response
injection does not repeat it. The initial-reminder selection survives EOTC and
runs through the existing Reminder path. Even an already available result waits
until initial speech finishes recorded playback; generic reminders that have not
started are skipped once a result arrives.

`interruption_speech_frames` optionally overrides the number of consecutive RNNT
nonblank frames required to interrupt agent speech or active tool work. Each
frame is 80 ms; omission retains the pinned 40-frame default. Blank frames break
the consecutive run. This is a speech-decision threshold, not utterance duration.

Configuration and script reads are bounded to 1 MiB and 4 MiB. Chunk size is bounded
to 1,048,576 samples; RIFF output has its standard 32-bit size limit. After EOF,
the application submits bounded zero PCM to complete the final frame and keep
perception active during committed tool work. The response window
(`--tail-frames`, default 200, or 16 seconds) counts ordinary live frames after
the source ends, excluding the padded final source frame. Completing or
interrupting tool work resets that window, including completion of response
injection through EOTR. The application then closes input with no additional
continuation and advances the ordinary drain path. The default exceeds
the released 40-frame end-of-utterance and 50-frame first-turn fallback windows.
The suffix does not guarantee TTS EOS
for every response. Explicit frame, model-step, publication and copy limits remain
enforced; exhaustion is an error. Driver errors cancel using recorded time and
the caller finalizes any committed waveform already written. `input_silence`
events record the inserted samples and whether they complete source padding,
support tool work or provide the response tail. The final event distinguishes
WAV `source_samples`, inserted `silence_samples`, and total `input_samples`
accepted by the session.

Portable fixtures cover independent PCM24 decoding, malformed/truncated WAVs,
exact float32 output, fragmented/backpressured input, pending completion, tool
substeps and drain through neutral session fixtures. Native compiler coverage does not
establish released-weight conversations, speech quality, latency or distributed
execution.

### Released-checkpoint single-device execution

The resident F32 measurement dated 2026-10-05 uses the resident WAV driver on
Apple M3 Ultra with 256 GiB unified memory, macOS 26.6.2 and Metal. The official checkpoint is
44,382,749,892 bytes, SHA-256
`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`, at the pinned
revision above. Auxiliary asset digests pass ordinary preparation. Input is
`examples/speechlm2/sample_audio/sample_general.wav` from Speech revision
`097dfe9e2f55baf653b83035868bdc89849f1b47`, SHA-256
`481f422a961fb160ddeba9824d55cb7c190c57acb7dc1730a2d595fd078dcb04`:
249,734 mono PCM16 samples at 16 kHz.

The configuration above uses `device="gpu"`, `max_frames=512`,
`max_model_steps=1024`, `max_total_copy_bytes=1099511627776`, seed zero, resident
parameters and the same system prompt. The cumulative copy ceiling is an admission
budget, not simultaneously allocated memory. Artifacts and run outputs reside in
`/private/tmp/eredu-voicechat-reference` outside the repository.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu-cli --no-default-features --features mlx,realtime,metal
/usr/bin/time -l target/debug/eredu realtime --offline --no-playback \
  --config /private/tmp/eredu-voicechat-reference/voicechat.json \
  --input /private/tmp/eredu-voicechat-reference/input-general.wav \
  --output /private/tmp/eredu-voicechat-reference/speech-2.wav \
  --events /private/tmp/eredu-voicechat-reference/events-2.jsonl \
  --chunk-samples 640 --tail-frames 200 --max-steps 2000
```

The run commits eight prompt positions, 396 audio frames (including 200 declared
silence frames), 404 total backbone positions, and terminal drain. It publishes
24 agent-text fragments, 396 transcription events, four turn events and one
finished event. Agent text is “Hi! How can I help you today?” followed by
“Yes, the sky appears blue due to a phenomenon called Rayleigh scattering.”
The observed transcription is `o do you know what color the sky is`; transcript
accuracy has no independent comparison here.

The WAV contains 698,552 contiguous float32 samples at 22.05 kHz, including an
eight-sample codec tail: 31.68036 seconds, all finite, absolute peak 0.18182363,
RMS 0.0078710742 and no samples outside [-1,1]. Its SHA-256 is
`79a27903f6875dd642c57f6dcaf4c7238f9f6ce3d02923e89d81772a9fe6a4ac`.
The retained fixture records output checks in `run-summary.json` and
`run-2.log`. It establishes released-checkpoint execution through the public
session, not independent complete-model reference parity or perceptual quality.

### Speech precision and cache contracts

`VoiceChatStartOptions::speech_precision` selects the portable
`SpeechPrecision { weights, compute }` policy, defaulting to F32. The realtime CLI
accepts `speech_weights` and `speech_compute` as `f32`, `f16` or `bf16`. Narrow
compute requires matching narrow weight storage. `"speech_weights":"f16",
"speech_compute":"f32"` selects eligible narrow weights with F32 arithmetic;
`"speech_compute":"f16"` also narrows projection/convolution execution and
perception convolution weights. Normalization, residuals, retained perception
histories/TTS K/V state, recurrent transcription, mixture readout/banks, RVQ scoring
and codec execution remain F32. Selected precision participates in state identity
and ordinary controlled/scheduled execution. Bindings account for selected storage
and conversion workspace before materialization.

Speech lifecycle and output accounting use the shared execution contracts.
Independent complete-model speech precision comparisons remain a validation gap.

## Conversational quality limits

Correct transcription does not guarantee a relevant agent response. A clean
Samantha-synthesized recording of "Get out from under the table, please." produces
a sexual-content refusal under both resident Q8/F16 speech execution and
unquantized F32 execution. Both runs recognize `get out from under the table
please`, start agent text at 3,920 ms, and produce the same 391-character response,
beginning `I cannot fulfill this request.` The fixed mono PCM16 fixture contains
341,119 samples at 16 kHz, including one second of leading silence and 18 seconds
of trailing silence. Its WAV SHA-256 is
`b25db6092a611fae32c7626241f3f0c95fb765ebbb930b78f3ddad2fc10f6a0c`.

These measurements use the pinned official checkpoint and base assets above,
seed zero, the system prompt `You are a helpful assistant.`, Metal on M3 Ultra,
`MLX_MAX_OPS_PER_BUFFER=25`, `MLX_MAX_MB_PER_BUFFER=256`, offline WAV input and
`--tail-frames 0`. Q8 uses `--quantize 8 --speech-weights f16 --speech-compute f16`;
the floating control omits quantization and uses
`--speech-weights f32 --speech-compute f32`.

Quantization and narrow speech precision are not necessary to trigger this
fixture's false refusal. Both controls use Eredu's execution mechanisms; they
do not establish independent complete-model parity with NVIDIA's runtime.
[NVIDIA's known limitations](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat#known-limitations)
include irrelevant canned replies, refusal loops, agent continuations without
new input and missing transcription even on clear audio. These published
limitations match the observed class of behavior without proving the cause of
every incorrect response.

## MLX Community SafeTensors validation

The shared SafeTensors path admits these pinned community artifacts:

| Repository | Revision | Physical matrices |
| --- | --- | --- |
| `mlx-community/NemotronLabs-VoiceChat-11B-8bit` | `446fc83f6cde720e7fd9ad7e29d7acb3adbb8967` | 123 affine Q8/group64 |
| `mlx-community/NemotronLabs-VoiceChat-11B-mlx-8bit` | `ea8e66e303e50babacd640f38c1e3a9312e1b622` | Same three weight shards as the Q8 artifact above |
| `mlx-community/NemotronLabs-VoiceChat-11B-4bit` | `dffd203fef7a380081da89479c55848e50b98413` | 123 affine Q4/group64 |

The complete Q8 and Q4 downloads are verified against the sizes and SHA256
digests in each pinned Hugging Face LFS catalog. Both contain all 1,632 released
primary identities plus 246 scale/bias companions. Metadata admission accepts
both published configuration forms, preserves the released equations, and rejects
a declaration whose bit width disagrees with the physical catalog. Quantization
covers 9,473,228,800 logical backbone/embedding/output-head parameters; speech and
perception source weights remain BF16. No load-time quantization is applied.

The retained compatibility fixtures use the M3 Ultra 80-core GPU with 256 GiB
unified memory and the pinned mono 16 kHz input (249,734 samples, SHA-256
`481f422a961fb160ddeba9824d55cb7c190c57acb7dc1730a2d595fd078dcb04`), followed by
800 response-tail frames. Each timeline contains 996 live frames (79.68 seconds).
Resident execution uses seed zero, the helpful-assistant prompt, no tools,
640-sample input chunks and the ordinary prepared cache. Eligible speech weights
and arithmetic use F32/F32, F16/F32, F16/F16 or BF16/BF16 policies; backbone
recurrent and normalization arithmetic remain F32. These fixtures establish
admission, execution and output repeatability for the retained formats and
precision policies.

Q8 F16/F32 produces byte-identical waveform and committed events to the Q8
F32/F32 run. Both transcribe `o do you know what color the sky is` and answer
`Hi! How can I help you today?Yes, the sky appears blue due to Rayleigh scattering.`
F16/F16 preserves that assistant text but transcribes `o the sky is`; its repeated
runs have identical waveform and events. Q4 repeats are also identical, but include
unrelated and malformed assistant text before answering the question. Every run
produces 1,756,952 finite float samples without clipping. These checks establish
repeatability and the observed fixture behavior, not broad perceptual or language
quality. Independent complete-model reference parity remains unvalidated.

The official NVIDIA F32 source retains exact regression parity through the same
loader: the 396-frame fixture produces 698,552 samples with WAV SHA256
`79a27903f6875dd642c57f6dcaf4c7238f9f6ce3d02923e89d81772a9fe6a4ac` and identical
committed events. Focused neutral tests reject malformed affine packing and missing
companions, and reject changed component equations. Affine function-head tests use
independent nonzero expected logits through resident, host and disk execution and
preserve companions during TP2 partitioning. Existing GGUF backbone admission tests
pass with F32 sensitive-vector recipes. Native test consumers and the backend without
native features compile; a native distributed Q4/Q8 execution run is not part of
this evidence.

Reproduction uses the ordinary configuration above with `source.path` set to the
downloaded repository directory and `speech_weights` / `speech_compute` set to the
selected policy. The `"f16"` / `"f32"` policy preserves the Q8 F32/F32 fixture's
waveform and committed events.

```sh
hf download mlx-community/NemotronLabs-VoiceChat-11B-8bit \
  --revision 446fc83f6cde720e7fd9ad7e29d7acb3adbb8967 \
  --local-dir /models/voicechat-community-8bit
cargo build --release -p eredu-cli --no-default-features --features mlx,realtime,metal
MLX_MAX_OPS_PER_BUFFER=25 MLX_MAX_MB_PER_BUFFER=256 \
  target/release/eredu realtime --offline --no-playback --config voicechat-community.json \
  --input input-general.wav --output speech.wav --events events.jsonl \
  --chunk-samples 640 --tail-frames 800 --max-steps 2000
```

Pinned catalogs, full shard hashes, configuration files, audio, committed event
records and binary digests are retained in `/private/tmp/eredu-mlx-compat` on the
validation host.


## Numerical acceptance

The released CLI workloads above establish execution evidence. Independent
complete-model numerical parity remains unvalidated for official SafeTensors,
community packed SafeTensors and GGUF. A complete reference run must use the same
PCM, tokenizer, tools, prompt, seed and checkpoint encoding, compare backbone and
speech decision score rows over several cached frames, and compare committed
text/audio tokens and drained waveform output. Component equations and extracted
NVIDIA methods do not establish that result.

Pinned NVIDIA source identities are retained in
[`sources.json`](../eredu-evaluation/fixtures/voicechat/sources.json). Checkpoint
payloads, PCM, waveform outputs and score traces stay outside the tracked tree.
