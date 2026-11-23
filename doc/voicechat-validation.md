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

`eredu audio realtime` consumes mono 16 kHz WAV input through
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
persists encoded weights and verifies source identity and exact recipes on reuse.
The CLI defaults to 32 GiB under the user cache directory. Override it with
both `--prepared-cache-dir` and `--prepared-cache-bytes`, or set
`prepared_checkpoint_cache` with `directory` and `max_bytes` in the realtime JSON.

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
`prepare_realtime` selects cache mechanisms and parameter pools without native
allocation; consuming `bind_realtime` constructs the selected execution.
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

The official F32 checkpoint at the pinned revision above has 44,382,749,892
bytes and SHA-256
`d553750c29434a6bb524377e17634c6cafdbf621892e643a77f406e51570354b`.
The recorded-input fixtures use `examples/speechlm2/sample_audio/sample_general.wav`
from NVIDIA Speech revision `097dfe9e2f55baf653b83035868bdc89849f1b47`.
The source WAV contains 249,734 mono PCM16 samples at 16 kHz and has SHA-256
`481f422a961fb160ddeba9824d55cb7c190c57acb7dc1730a2d595fd078dcb04`.
See [local cross-backend controlled execution](#local-cross-backend-controlled-execution)
for released-checkpoint frame, replay, and drain results with the shared sampler.

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

To run a community checkpoint through the realtime CLI, set `source.path` to
the pinned downloaded directory and choose a `speech_weights` and
`speech_compute` policy as described above.

## Local cross-backend controlled execution

The community 4-bit checkpoint at revision
`dffd203fef7a380081da89479c55848e50b98413` runs with resident weights,
device-state residency, F16 speech weights, F32 speech computation, and
seed 7. These fixtures use the shared host sampler.

| Input fixture | Placements | Result |
| --- | --- | --- |
| One silent 1,280-sample input frame | Candle CPU, Candle Metal, MLX Metal | 31/31 codec IDs agree. Maximum pairwise PCM difference is `1.01e-9`. |
| Frame 23 of the pinned 16 kHz NVIDIA `sample_general.wav`, empty system prompt | Candle CPU, Candle Metal, MLX Metal | Candle Metal restore and fork reproduce all 31 codec IDs and 1,764 PCM samples bit for bit. Candle CPU and MLX Metal agree on all 31 codec IDs with a maximum PCM difference of `8.01e-10`. |
| Sustained recorded speech | Candle CPU, MLX Metal | Both commit 81 live frames and drain an eight-sample codec tail. Turn events, delivered text, frame attribution, and consumed input agree. The 142,892 output samples differ by at most `4.89e-6` with relative L2 `1.35e-5`. |
| Recorded speech with overlap | Candle CPU, MLX Metal | Both commit an interruption at live frame 21 and drain the codec. Maximum absolute PCM difference is `2.89e-6`. |

Use zero-based 80 ms frames 10–26 and 101–148 from the recording for the
sustained case, followed by 16 continuation frames. The end-of-utterance
thresholds are eight frames and the interruption threshold is three frames.
Snapshot with four input frames queued, advance four commits without delivery,
and compare restored execution with an isolated fork. Both backends replay
the four publications bit for bit within their own placement.

For the overlap fixture, start the second excerpt at source frame 103 just
as the agent begins, with six frames of lookback and a one-frame interruption
threshold.

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
