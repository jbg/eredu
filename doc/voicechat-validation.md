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
recognizes the official nested NeMo configuration and complete
`nemotron_voicechat` GGUF metadata as realtime VoiceChat. Generic realtime
preparation retains the sources used to construct a single-device speech
session. MLX CPU has RNNT, perception, channel-boundary, cached TTS, character
conditioning, MoG/RVQ, codec encoding/decoding and retained-sampling component
evidence; complete-model residency/parallel execution has no complete VoiceChat
validation results. The WAV example completes the pinned released checkpoint on
Metal with speech, text, transcription and drain, as recorded below. Independent
complete-model numerical parity and speech quality remain unvalidated. No
hardware latency claim follows from storage admission or the unoptimized
execution measurement.

## WAV and recorded tool responses

`eredu/examples/voicechat_wav.rs` consumes mono 16 kHz WAV input through
`BoundVoiceChatSession`, writes mono 22.05 kHz float32 speech and records committed
frame, agent-text, transcription, tool and turn events as JSONL. PCM and output
delivery are bounded. Speech sample offsets are contiguous in the output WAV;
JSONL retains model/audio positions and codec delay separately from the explicit
invocation timestamp (`at_ms`). There is no wall-clock pacing or tool execution.

The example uses the shared loading entry points: `prepare_realtime_model` performs
architecture-owned family discovery, source admission and auxiliary-asset retention;
`prepare_mlx_realtime` selects cache mechanisms and parameter pools without native
allocation; consuming `PreparedMlxRealtime::load` constructs the selected execution.
The same preparation and loading interfaces admit Moshi/PersonaPlex categorical
frames. `RealtimeLoadOptions` distinguishes categorical frames from PCM speech;
a protocol mismatch fails before binding. The serialized checkpoint-source contract
is shared with the architecture layer. The example supplies paths and policy, not
family detection, required component lists, source recipes or cache selection.

All artifacts are local. `official_assets` contains `config.json` and the three
`rnnt_tokenizer/{tokenizer.model,tokenizer_config.json,vocab.json}` files from
official revision `443794ea956ef0065f001967ffd00e77f519cb39`. `base_assets` contains
`config.json`, `tokenizer.json` and `tokenizer_config.json` from base revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`. Preparation verifies exact auxiliary
asset digests, tensor geometry, control values and tokenizer declarations.
Checkpoint storage remains outside the source tree. Native construction consumes
retained source views. The example does not download artifacts.

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
CARGO_INCREMENTAL=0 cargo run -p eredu --no-default-features --features mlx,audio \
  --example voicechat_wav -- --config voicechat.json --input conversation.wav \
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
substeps and drain without an MLX dependency. Native compiler coverage does not
establish released-weight conversations, speech quality, latency or distributed
execution.

### Released-checkpoint single-device execution

The resident F32 measurement dated 2026-10-05 uses the public WAV example on
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
CARGO_INCREMENTAL=0 cargo build -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/debug/examples/voicechat_wav \
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
The dev-profile run takes 172.39 seconds including startup; maximum process RSS is
123,601,731,584 bytes and peak process footprint is 125,483,392,336 bytes. These
are observations, not real-time performance or minimum-residency claims; a test
build and a brief scalar-kernel check overlap execution. `run-summary.json` and
`run-2.log` retain the output checks and process measurements.

The resident release-profile performance measurement on 2026-10-08 uses the
same pinned checkpoint, configuration, input and 200-frame silence tail on an
M3 Ultra with 80 GPU cores and 256 GB unified memory. Both measured executables
use ordered Metal RVQ lookup fusion, retained codebook norms, bounded immutable
character conditioning and shared immutable frame-state roots. Parameters, five
active refinement slots and sampling policy are unchanged. The 421,888-byte
derived-tensor allowance is included in cold device admission. The current
implementation additionally fuses F32 single-step selective recurrence updates
and packs agent/function IDs for joint evaluation and transfer. Projection uses
MLX's ordinary reduction order; explicit snapshots/forks retain physical isolation.
The unoptimized pinned commit `d5683bb3a454a627b1023713a053d0a0d5a3c74a`
measures 217.079 ms per warmed frame in the separate retained baseline artifact.

| Full-path resident Metal measurement | Ordinary scan/readbacks | Fused update/grouped IDs |
| --- | ---: | ---: |
| Mean warmed 80 ms frame | 170.446 ms | 169.122 ms |
| p95 warmed frame | 174.779 ms | 173.792 ms |
| Warmed real-time factor, lower is better | 2.131 | 2.114 |
| Warmed deadline misses | 386 / 386 | 386 / 386 |
| Process wall time including startup and drain | 77.77 s | 76.12 s |
| Peak process footprint | 98.463 GB | 98.445 GB |

Warmed statistics exclude the first ten of 396 live frames and include input
processing, inference, transaction preparation, GPU completion, host output and
publication. No builds or tests overlap this complete paired measurement. The
mean falls by 1.325 ms (0.78%), giving 1.008x throughput; 367 of 386 matched
warmed frames are faster, with a median difference of 1.228 ms. This is a modest
full-path result from one pair, not isolated recurrence or readback timing. An
initial unpaired fused run measures 189.471 ms/frame with identical output; the
variation limits inference about gains across machine states. The current paired
mean exceeds the deadline by 89.122 ms and requires another 2.114x throughput
improvement (52.70% lower latency) for sustained real-time processing. It is not
a system microphone/speaker or acoustic echo-cancellation measurement. Maximum
process RSS is 44,987,269,120 bytes before and 44,968,296,448 bytes after. Neither
paired run swaps.

The complete 698,552-sample float32 WAV is byte-for-byte identical to the baseline,
with SHA-256 `79a27903f6875dd642c57f6dcaf4c7238f9f6ce3d02923e89d81772a9fe6a4ac`.
All 1,421 published events match after removal of instrumentation timestamps. This
is local regression parity, not an independent full-session NVIDIA comparison.

External `/private/tmp/eredu-voicechat-mamba` contains `REPORT.md`, `VALIDATION.md`,
the source patch, preserved before/after executables, WAVs, event logs, process
logs, per-frame summaries and `comparison.json`. `before.*` and `after-clean.*`
retain the paired measurement above; `after.*` retains the slower unpaired run.
External `/private/tmp/eredu-voicechat-state-sharing` retains the separate candidate
sharing comparison: 191.737 to 170.731 ms/frame, with byte-identical speech.
That 21.006 ms full-path effect is not an isolated copy-kernel measurement; the
earlier approximately 24 ms outside execution includes completion and other work.
External `/private/tmp/eredu-voicechat-fusion` retains the codebook/conditioning
comparison against the unoptimized commit. Its `timing.patch` wraps the event
writer with monotonic publication timestamps and a load timer; it adds no GPU
wait or model decision. Timing instrumentation is absent from the example source.

```sh
CARGO_INCREMENTAL=0 cargo build --release --offline -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/release/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat.json \
  --input /private/tmp/eredu-voicechat-reference/input-general.wav \
  --output /private/tmp/eredu-voicechat-mamba/after-clean.wav \
  --events /private/tmp/eredu-voicechat-mamba/after-clean.jsonl \
  --chunk-samples 640 --tail-frames 200 --max-steps 2000
cargo test --offline -p eredu-backend-mlx --no-default-features \
  --features audio,metal --test codebook_fusion -- --test-threads=1
cargo test --offline -p eredu-backend-mlx --no-default-features \
  --features audio,metal --test selective_scan_fusion -- --test-threads=1
cargo test --offline -p eredu-architectures --test reference_numeric \
  -- voicechat_channel_decision voicechat_dynamic_channel_policy \
  voicechat_channel_coordination_owner_policy_rollback_and_replay --test-threads=1
cargo test --offline -p eredu-backend-mlx --no-default-features \
  --features audio,metal --lib -- mlx_selective_scan_chunk_boundaries \
  coupled_native_execution failed_and_unwinding_voicechat --test-threads=1
cargo test --offline -p eredu-architectures --test reference_numeric voicechat_rvq
cargo test --offline -p eredu-architectures --lib character_cache
cargo test --offline -p eredu-architectures --lib \
  retained_execution_cache_cannot_consume_parameter_capacity_twice
```

A separate resident on-load affine quantization experiment on the same M3 Ultra,
pinned checkpoint and full conversation fixture uses group size 64 and the existing
bounded native quantizer. The experimental architecture projection declares packed
Nemotron parameter slots and retained recipes; the backend packs 122 matrices with a
64 MiB conversion working-set bound. Norms, convolutions, activations, caches and
all speech-provider weights remain F32. No converted checkpoint file is generated.
The public speech loader does not expose this experimental request. The experiment
retains conservative F32 cold admission and does not establish a production quantized
residency or capability contract.

| Resident on-load experiment, in run order | F32 control | 8-bit backbone | 4-bit backbone | F32 control |
| --- | ---: | ---: | ---: | ---: |
| Mean warmed 80 ms frame | 189.058 ms | 137.189 ms | 131.452 ms | 168.662 ms |
| p95 warmed frame | 194.142 ms | 141.125 ms | 136.080 ms | 173.427 ms |
| Warmed real-time factor | 2.363 | 1.715 | 1.643 | 2.108 |
| Backbone parameter storage | 35.553 GB | 10.006 GB | 5.563 GB | 35.553 GB |
| Startup, including on-load packing | 9.875 s | 81.991 s | 77.765 s | 8.470 s |
| Peak process RSS | 44.977 GB | 32.117 GB | 23.548 GB | 44.982 GB |

All runs have 396 live frames, 386 warmed observations, 698,552 output samples
and zero swaps. Builds and tests do not overlap the sequential runs. The F32
control variation limits precise speedup attribution: 8-bit reduces latency by
18.7–27.4% against the controls, and 4-bit by 22.1–30.5%. All warmed frames still
miss the deadline. The 8-bit mean exceeds it by 57.189 ms, requiring another
1.715x throughput (41.69% less latency). The 4-bit gap is 51.452 ms, requiring
1.643x throughput (39.14% less latency). The 5.737 ms difference between quantized
runs is smaller than the F32 control variation. These measurements support a
substantial bandwidth benefit from backbone quantization, without establishing
real-time performance or the effect of quantizing the remaining speech matrices.

The transformed dense matrices total 35,544,104,960 bytes; packed outputs are
9,996,779,520 bytes at 8-bit and 5,553,766,400 bytes at 4-bit. Another 8,804,352
backbone bytes remain dense. The backbone accounts for approximately 80% of
learned storage; the speech providers retain approximately 8.827 GB. Process
peaks include the retained in-memory packed source overlay, native weights,
conversion work and mapped source pages, rather than just packed parameter bytes.

Both F32 controls reproduce the complete baseline WAV and all 1,421 events after
removing timestamps. RNNT transcription is identical across all four runs, and
all waveforms have finite samples with no samples outside [-1,1]. The 8-bit
response remains coherent for the fixture: “Yes, the sky appears blue due to
Rayleigh scattering.” The 4-bit response contains an unrelated fragment about a
“popular dish” and a malformed continuation before correctly answering the sky
question. This is visible response degradation. Neither quantized output is
asserted equivalent to F32, and this fixture is not a broad quality or perceptual
speech evaluation. The 8-bit run has the same active/inactive-agent frame counts
as F32; the 4-bit conversation trajectory differs.

External `/private/tmp/eredu-voicechat-quantization` retains `REPORT.md`, the
reproducible experimental loader/timing patch, preserved executable, release
build log, original source inputs, run/analysis scripts, all four WAVs and event
logs, native conversion telemetry, process logs and `comparison.json`. The
experimental source is restored after building; the preserved executable is
used for all four runs. `REPORT.md` records commands, revisions, hashes and
quality limitations. The fixture does not exercise microphone/speaker hardware
or acoustic echo cancellation.

The selective on-load 8-bit experiment on 2026-10-09 extends packing to the
independent function head and causal EAR-TTS projections, keeping the backbone
8-bit in every run. It uses the same pinned checkpoint, resident Metal execution,
group size 64 and 64 MiB bounded converter. Architecture-owned retained recipes and
linear format declarations include exact affine companions; native materialization
packs them before ordinary provider construction. The full function vocabulary,
greedy decision policy, guidance lanes, state and completion paths remain intact.
Character encoding, MoG, perception, codec, activations, caches and norms remain F32.
These switches are experimental, with conservative F32 cold admission; the public
loader exposes no additional production quantization contract.

| Selective experiment | Backbone-only control after | + 8-bit function head | + 8-bit causal TTS |
| --- | ---: | ---: | ---: |
| Mean warmed 80 ms frame | 137.123 ms | 134.776 ms | 126.446 ms |
| p95 warmed frame | 141.845 ms | 139.391 ms | 130.778 ms |
| Warmed real-time factor | 1.714 | 1.685 | 1.581 |
| Function-head storage | 2.349 GB | 0.661 GB | 0.661 GB |
| Causal-TTS storage | 2.379 GB | 2.379 GB | 0.669 GB |
| Startup, including packing | 81.945 s | 98.013 s | 97.899 s |
| Peak process RSS | 33.452 GB | 32.066 GB | 30.515 GB |

Run order is an initial backbone-only control (157.501 ms), function-head packing,
combined function-head/TTS packing and the repeated backbone-only control above.
The initial control uses the earlier benchmark build; its 20.378 ms difference
prevents attributing its full difference to quantization. All later runs use the
same final binary. Relative to the later control, the measured function-head saving
is 2.347 ms (1.71%); adding causal TTS saves another 8.330 ms versus the head-only
run. The combined comparison is 10.677 ms (7.79%) or 1.084x throughput, rather than
an isolated component GPU measurement. All 386 warmed intervals still miss the
80 ms deadline. The combined mean gap is 46.446 ms, requiring another 1.581x
throughput or 36.73% latency reduction. Sequential runs have no overlapping builds
or tests. These local comparisons do not establish repeatability across workloads.

The function matrix packs 2,348,810,240 bytes to 660,602,880 bytes. The 196 causal-TTS
projections pack 2,378,170,368 bytes to 668,860,416 bytes, retaining another 536,832
F32 bytes. Their combined incremental parameter saving is 3,397,517,312 bytes.
Native conversion uses two in-flight tiles and produces no checkpoint file.

Both controls and the function-head run reproduce the same 1,418 semantic events
excluding wall measurement and the same complete WAV, SHA256
`4fd9b2e18cd783b6374d52da04d619db2c08c2bc87bc411b088a80a0044bc7c0`.
The TTS run preserves those events, transcription, agent text, turn boundaries and
698,552 output samples, but changes the waveform, SHA256
`8b26df28ebbf05e711a32967d27d86a40825adc900fa7a9cbc63e69167ea1c81`.
All samples remain finite and inside [-1,1]. RMS changes from 0.0067413 to 0.0059829;
peak changes from 0.137963 to 0.165811. These checks establish neither perceptual
speech equivalence nor absence of quality degradation. MoG and perception packing,
microphone/speaker hardware and acoustic echo cancellation are not measured.

The 8-bit-backbone tool fixture compares F32 and 8-bit function heads with F32
TTS. Both reproduce all 1,005 semantic events and the exact 460,412-sample WAV,
SHA256 `dfcc5e63a916d1d0cde0100de9aac44a025902b3f03da3f6f521320a840e7089`.
The call is `generate_random_number` with `{"min":1,"max":50}` at audio time
10,560 ms; the scripted result arrives at 11,360 ms and the answer is “Your random
number is twenty.” No script entries or results remain unused. The 166 warmed
live-audio intervals measure 136.248 ms with the F32 head and 132.681 ms with the
8-bit head (3.566 ms saving); p95 is 140.094 and 135.651 ms respectively. These
intervals exclude non-live tool-processing invocations. This single call establishes
fixture parity, not broad function-channel accuracy; quantized TTS has no tool-fixture
or perceptual quality comparison here.

External `/private/tmp/eredu-voicechat-function-quantization` retains `REPORT.md`,
the combined experimental patch, exact source originals, successful release build
log, preserved executable, independent switches in `run.sh`, analysis and comparison
scripts, native conversion/process logs, WAVs, JSONL events and pairwise quality
reports. Source inputs are restored after building. The preserved final binary
SHA256 is `71c13382d7831e286f977d599c384602acd6f944e300671be239dee01c077000`.
Reproduction uses the same Metal/audio release command above and
`run.sh NAME HEAD_BITS FIXTURE [TTS_BITS]`, with each added component set to `f32`
or `8` and `FIXTURE` set to `general` or `tool`. `REPORT.md` records commands,
pinned revisions, hashes and limitations.

The mixed-precision measurement on 2026-10-09 uses the same checkpoint and M3
Ultra, resident Metal release execution, and one preserved executable. The backbone
and function head use the experimental 8-bit bindings in every run; causal TTS
remains dense. `VoiceChatStartOptions::speech_precision` selects a portable
`SpeechPrecision { weights, compute }`, defaulting to F32. The WAV example accepts
`speech_weights` and `speech_compute` as `f32`, `f16` or `bf16`. Narrow compute
requires matching narrow weight storage. For example, `"speech_weights":"f16",
"speech_compute":"f32"` selects narrow matrices with F32 arithmetic;
`"speech_compute":"f16"` also narrows projection/convolution inputs and perception
convolution weights. Norms, residuals, retained histories/KV state, recurrent
transcription, mixture readout/banks, RVQ scoring and codec remain F32.

| Speech weights / compute | Mean warmed 80 ms frame | p95 frame | Real-time factor |
| --- | ---: | ---: | ---: |
| F32 / F32, first control | 136.376 ms | 143.186 ms | 1.705 |
| F16 / F32 | 132.881 ms | 137.457 ms | 1.661 |
| BF16 / F32 | 132.849 ms | 138.310 ms | 1.661 |
| F16 / F16 | 129.991 ms | 135.295 ms | 1.625 |
| BF16 / BF16 | 144.693 ms | 149.494 ms | 1.809 |
| F32 / F32, repeated control | 135.041 ms | 139.582 ms | 1.688 |

Sequential runs have no overlapping builds or tests. Timing uses completed
live-frame publication intervals and includes preparation, execution, GPU
completion and host/event work; statistics discard the first ten intervals and
retain 386. F16 compute saves 5.050–6.386 ms (3.74–4.68%) against the two controls.
Narrow matrices alone save about 2.2–3.5 ms; their F16/BF16 timings are tied here.
BF16 compute regresses by 8.3–9.7 ms. All warmed intervals still miss the deadline.
The best mean gap is 49.991 ms, requiring another 1.625x throughput or 38.46%
latency reduction. These sequential controls quantify drift, not confidence
intervals or broad workload repeatability. The separate 126.446 ms selective
8-bit causal-TTS result above is faster than dense F16 compute; this sweep does
not establish an improvement over that configuration or test packed TTS together
with narrow compute.

Exact converted parameter bindings save 2,402,156,544 bytes for narrow matrices
with F32 compute, or 2,553,869,824 bytes when perception convolutions also narrow.
Causal-TTS parameter storage decreases from 2,378,707,200 to 1,189,622,016 bytes.
Converted binding sizes and conversion peaks enter admission before materialization;
the independent experimental 8-bit switches retain conservative F32 admission.

All four general mixed-precision runs preserve the same 1,418 semantic events,
transcription, agent text, turns, audio timing and 698,552 output samples, excluding
only wall timestamps. Both F32 controls reproduce the same complete WAV.
Narrow profiles change the waveform; samples remain finite and inside [-1,1].
The tool fixture preserves all 1,005 semantic events, the exact tool arguments
and audio-time delivery, final answer and 460,412 output samples. F32/F32 measures
134.126 ms mean (137.664 ms p95); F16/F16 measures 127.768 ms (131.612 ms p95), a
6.359 ms or 4.74% improvement over 166 warmed live-audio intervals. All miss their
deadlines. The tool waveform changes. These checks establish fixture parity, not
perceptual equivalence, speech quality or broad function accuracy. Listening,
external ASR, microphone/speaker hardware and echo cancellation are unmeasured.

Focused native conformance uses the existing nonzero coupled fixture with F16/F32,
F16/F16 and BF16/BF16 profiles. Explicit and scheduled publication agree on output,
retained caches, RNG and clocks; snapshot restoration and fork continuation replay
identically, rejected incomplete input leaves completion unsubmitted, and codec
completion is preserved. Independent projection rounding checks cover F16 and BF16
inputs, ordinary/observed projection and singleton row reduction with F32 outputs.
Row-parallel observer failure/unwind and bias-once coverage passes. Cold perception
selection checks converted byte accounting, F32 norms, convolution policy and
two-rank shard geometry. Native multi-rank precision execution and performance are
unvalidated. Portable facade and native Metal/audio WAV compiler checks pass.

External `/private/tmp/eredu-voicechat-mixed-precision` retains `REPORT.md`, exact
build inputs, `benchmark.patch`, configs, release build log, run/analysis scripts,
WAVs, JSONL records, latency/memory statistics and pairwise quality comparisons.
The executable SHA256 is
`20951c58aae2102d500965de536b9470402d38aa92beb4725721d0414068b4ca`.
`run.sh NAME PRECISION [FIXTURE]` selects `f32`, `f16-weights`, `bf16-weights`,
`f16-compute` or `bf16-compute`, and `general` or `tool`. Experimental quantization
and timing source changes are restored after building. The persistent loader
exposes the speech precision policy without benchmark switches.

The combined quantization/mixed-precision measurement on 2026-10-09 retains
8-bit backbone, function-head and all 196 causal-TTS projections, group size 64,
with the same 64 MiB bounded converter. Eligible perception, character-encoder and
MoG projections plus perception convolutions use F16. Quantized TTS has separate
F32-input/F32-companion and F16-input/F16-companion variants. TTS packing consumes
the original F32 source recipe directly, removing the narrow-storage cast before
quantization; both variants retain identical packed-weight byte counts. Norms,
residuals, attention/perception caches, recurrent accumulation/transcription,
mixture prediction/banks, RVQ scoring and codec stay F32.

| Combined experiment | Mean warmed 80 ms frame | p95 frame | Real-time factor |
| --- | ---: | ---: | ---: |
| 8-bit control, repeated | 127.057 ms | 131.807 ms | 1.588 |
| F16 remaining speech/perception, F32 quantized TTS | 123.930 ms | 128.739 ms | 1.549 |
| Same combination, repeat | 123.860 ms | 128.219 ms | 1.548 |
| F16 remaining speech/perception and quantized TTS inputs/companions | 131.309 ms | 172.522 ms | 1.641 |

One preserved release executable runs all seven measurements sequentially, with
no overlapping builds/tests. General statistics retain 386 completed live-frame
intervals after discarding ten; the timing includes GPU completion and host/event
work. The initial quantized control measures 147.503 ms, 20.446 ms slower than
the repeated control despite identical events and complete waveform. Its difference
has no precision attribution. Against the stable repeated control, the two fastest
combination runs save 3.127–3.197 ms (2.46–2.52%). The F16-input quantized-TTS run
has large early delays and no demonstrated additional performance win; their cause
is unmeasured. The fastest repeat remains 43.860 ms above budget, requiring another
1.548x throughput or 35.41% latency reduction. All warmed frames miss the deadline.
Sequential local comparisons establish no statistical confidence interval.

F16 storage outside packed causal TTS saves another 1,364,784,640 parameter bytes.
Causal TTS, including F32 norms, occupies 669,397,248 bytes with F32 affine
companions or 632,238,336 bytes with F16 companions. All seven runs report zero
swaps. Experimental quantization retains conservative floating cold admission;
these results introduce no production on-load quantization capability contract.

Both combined conversation variants preserve all 1,418 semantic events and
698,552 output samples. The fastest repeat reproduces the exact combined waveform.
Both controls reproduce the earlier selective 8-bit TTS WAV. The tool control and
fastest combination preserve all 1,005 semantic events, exact arguments/delivery,
final answer and 460,412 output samples, with no unused or undelivered script
results. The 166 warmed tool intervals measure 125.807 ms mean (129.229 ms p95)
for the quantized control and 123.254 ms (126.993 ms p95) for the combination:
2.553 ms or 2.03% saving. The 8-bit TTS tool control also preserves the prior
F32-TTS tool events. All audio remains finite and inside [-1,1], but mixed precision
changes the waveform relative to quantized controls. User listening evaluation
accepts the fastest combined waveform. External ASR, microphone/speaker hardware
and echo cancellation remain unmeasured.

External `/private/tmp/eredu-voicechat-combined-precision` retains `REPORT.md`,
`benchmark.patch`, exact source originals/hashes, configs, scripts, successful
release build log, conversion/process telemetry, WAVs, JSONL events and quality/
latency comparisons. Executable SHA256 is
`3b6ec61acedb1e162d6061cf5913394592b3a74079d4c8b4733c02cc0d971af0`.
All seven experimental source inputs are restored byte-for-byte and hash-verified.
`run.sh NAME SPEECH_PRECISION FIXTURE TTS_COMPUTE` uses `f32` or `f16-compute`,
`general` or `tool`, and `f32` or `f16` respectively. `sweep.py` selects the faster
combination for the tool fixture; `repeat-best.py` repeats that configuration.
`REPORT.md` records commands, pinned revisions, timing policy and limitations.

The accepted combined configuration also has released-checkpoint packed-projection
replay and complete-conversation comparisons on the M3 Ultra. Native kernel traces
show that two-row causal TTS already uses MLX's weight-reusing affine vector path.
Forcing that selection preserves projection outputs and establishes no additional
speedup. Pre-padded layouts make several large single-row projections 33–51%
faster in isolated warmed replay. A guarded partial-tile kernel uses the original
matrices and preserves close numerical agreement, but retains little of that
isolated benefit. Representative replay is not direct full-frame GPU attribution;
its padded matmul timing excludes activation padding and parameter retention.

| Packed-projection comparison | Mean 80 ms frame | p95 frame |
| --- | ---: | ---: |
| Ordinary control before | 125.318 ms | 130.716 ms |
| Guarded partial-tile kernel | 125.337 ms | 130.494 ms |
| Padded matrices and activations | 125.276 ms | 129.747 ms |
| Ordinary control after | 124.088 ms | 128.546 ms |

Each sequential run has 386 warmed intervals after discarding ten. The candidates
and final control use one preserved binary; the initial control uses the preceding
benchmark binary. The candidates establish no useful end-to-end improvement:
their differences from the initial control are smaller than the control spread,
and both are slower than the same-binary final control. Every warmed interval
misses its deadline. All 1,418 semantic events and complete WAV bytes match the
accepted combined audio, and all runs report zero swaps. Neither candidate is a
production execution mechanism. External
`/private/tmp/eredu-voicechat-packed-projections` retains `REPORT.md`, exact source
snapshots/restoration hashes, generic native instrumentation and kernel patches,
commands, binaries, kernel traces, raw replay/phase/frame timings, WAVs and quality
comparisons. The final ordinary critical-path spans are 24.600 ms input,
43.920 ms backbone/heads/decisions, 21.461 ms conditioning/causal TTS, and
25.388 ms speech refinement/render submission. These host spans include queued
work through existing synchronization points and are not isolated GPU timings.

Complete execution profiling of the accepted combined precision configuration on
M3 Ultra uses the same general conversation, 396 live frames and 386 warmed
intervals. The final same-binary control with profiling disabled averages
124.145 ms per 80 ms frame (p95 130.019 ms, p99 131.659 ms); all warmed intervals
miss the deadline. Throughput is 0.644 times realtime. The mean gap is 44.145 ms,
requiring a 35.56% wall-time reduction to reach the deadline. Startup/on-load
conversion takes 102.484 seconds and is excluded from frame latency; the control
reports zero swaps.

Portable host timing without added GPU fences gives 25.294 ms for
input/perception/RNNT, 43.924 ms for backbone/heads/decisions, 21.689 ms for
character conditioning/causal TTS and 25.450 ms for refinement/codec submission.
Execution assembly adds 0.024 ms; candidate/preflight/submission around execution
adds 2.118 ms, completion/observation/commit adds 6.373 ms and publication/other
work adds 0.112 ms, totaling 124.984 ms. These spans include queued work through
existing synchronization points; the time outside execution is not a copy-only
budget.

A finalized Metal System Trace and CPU sampling window covers 213 complete late
frames. The process-attributed GPU interval union is 87.410 ms/frame, 67.5% of
traced wall time, with 248.643 distinct command buffers/frame. Inclusive native
progress stacks account for 23.498 sampled ms/frame and overlap recovery and GPU
work. Relative-position construction accounts for 9.559 sampled ms/frame,
confirmed by direct clocks at 9.512 ms/frame across 24 perception layers.

Separate intrusive timing at actual native submissions includes asynchronously
submitted speech work. Its GPU regional duration sum is 90.081 ms/frame:
41.747 ms backbone/heads, 11.280 ms perception/RNNT, 0.062 ms character,
23.037 ms causal TTS, 7.475 ms MoG prediction/latent work, 0.566 ms random draws,
0.994 ms RVQ and 4.920 ms codec/render. The main 28-layer causal transformer
accounts for 22.503 ms. Existing native kernel-selection evidence shows a median
2,478 kernel requests/frame in the warmed speech regions; counts do not establish
individual kernel durations. Fine-fenced diagnostics expose 32.805 ms in Mamba
input projections and dense MLP blocks (including normalization, activation and
residual work), 1.155 ms in fused recurrence, 1.014 ms in perception position
projections and 1.755 ms in perception K/V projections.

GPU regional sums, traced unions, CPU samples and host spans use different
profiling modes and are not additive savings estimates. Host logging averages
124.984 ms, Instruments 128.470 ms, native-submission timing 141.314 ms and fine
fences 223.938 ms; intrusive timings do not represent ordinary wall performance.
All runs preserve all 1,418 semantic events after excluding wall clocks and the
complete accepted WAV bytes. All 23 temporarily instrumented source files match
their original hashes. External `/private/tmp/eredu-voicechat-full-profile`
retains `REPORT.md`, plots, finalized `complete.trace`, exact source snapshots,
restoration hashes, binaries, reproduction scripts, raw timings and quality
comparisons. The profile covers offline serial PCM execution; live mic/speaker,
echo cancellation, external tool-service latency and individual shader durations
are outside its evidence.

Perception caches the single-frame relative-position table and its projection in
each parameter owner. The position equation and attention arithmetic are
unchanged. Larger chunks bypass this fixed-size cache; parameter mutation and
precision changes invalidate projected values, context/dtype changes rebuild
constants and cloned owners start cold. These immutable values are separate from
causal state and snapshots. Cold admission reserves 27,721,728 additional bytes
for the released 24-layer geometry, separately from parameter pools.

Same-binary complete-conversation comparisons with accepted combined precision
use sequential uncached/cached/uncached runs and 386 warmed frames each:

| Position cache comparison | Mean 80 ms frame | p95 frame |
| --- | ---: | ---: |
| Uncached control before | 144.606 ms | 149.084 ms |
| Cached positions and projections | 113.440 ms | 118.624 ms |
| Uncached control after | 124.003 ms | 128.622 ms |

Caching saves 10.564 ms/frame (8.52%) against the faster control;
the first control is anomalously slow and is excluded from the savings claim.
The final control agrees with the preceding profiling-disabled 124.145 ms control
and a fresh preserved-binary calibration (123.865 ms, p95 128.865 ms). Throughput
is 0.705 times realtime and the remaining mean gap is 33.440 ms. All warmed intervals miss the deadline.
All 1,418 semantic events and complete accepted WAV bytes match. Thirteen existing
neutral perception tests cover pinned outputs, retries, snapshots, bounded units
and TP/PP; the focused nonzero position regression checks projection reuse and
rebinding against a fresh owner. The native Metal/audio release build passes.
External `/private/tmp/eredu-voicechat-position-cache` retains `REPORT.md`, exact
implementation/benchmark patches, hashes, commands, binary, timings, events, WAVs
and quality comparisons. The benchmark-only bypass and on-load quantization edits
are restored to their implementation hashes; no bypass flag is retained in source.

Resident serial-group execution uses the caller's executor and defers group
submissions to the surrounding forward completion. Eligibility is portable:
every group must lie on one dependency chain, and the parameter policy must
explicitly guarantee retained unit storage. Branched graphs, bounded policies
and parallel execution retain exact group submission boundaries. Unit and group
hooks, policy completion/abort, candidate state and final publication still use
the shared drivers; no runtime feature flag controls this mechanism.

With the accepted combined precision and cached perception positions, complete
serial PCM conversation measurements on the same M3 Ultra use 386 warmed frames:

| Resident chain comparison | Mean 80 ms frame | p95 frame |
| --- | ---: | ---: |
| Preserved control before | 113.287 ms | 118.458 ms |
| Deferred resident chain | 98.345 ms | 102.575 ms |
| Preserved control after | 113.219 ms | 118.545 ms |

Against the faster control, batching saves 14.874 ms/frame (13.14%). Throughput
is 0.813 times realtime, real-time factor is 1.229, and the remaining mean gap
is 18.345 ms; another 18.65% reduction is needed to reach 80 ms without scheduling
headroom. All 386 warmed intervals still miss the deadline. All 1,418 semantic
events and the complete accepted WAV bytes match, with SHA-256
`369f1334dea5e5d06fc450c03c06eb446a8eb673b60cc1e173e35b2641c1d282`.
The neutral resident-chain regression checks repeated values, state and group
order against ordinary execution, eliminates intermediate submissions, and
preserves branched scheduling. Pinned MoG resident/bounded numerical parity and
selected-bank failure/RNG retry coverage pass. Six native VoiceChat lifecycle
tests cover preparation and frame failure/unwind, retained owners, host delivery,
virtual continuation and publication. The explicit Metal mixed-precision test
confirms controlled and scheduled cached publication parity. The Metal/audio
release build passes. External `/private/tmp/eredu-completion-optimisation` retains the report,
reproduction scripts, binary hashes, exact benchmark and implementation patches,
logs, complete events and WAVs. All seven temporary on-load quantization source
edits match their implementation hashes after the build. This measures offline
serial inference; live audio scheduling and echo cancellation are not measured.

Generic Metal F32 RMS and `InputProducts` rotary fusion preserves the accepted
conversation under the same combined precision and resident serial execution.
RMS keeps its ordered reduction and separate F32 square, then applies
normalization and gain across the threadgroup. Rotary retains precise
trigonometric functions, separate product rounding and runtime sequence extents.
These native operators consume existing neutral specifications; family equations,
cache state, RNG and completion ownership retain their existing contracts.

Complete-conversation measurements on the same M3 Ultra use 386 warmed frames:

| Generic speech operator comparison | Mean 80 ms frame | p95 frame |
| --- | ---: | ---: |
| Preserved committed control before | 98.286 ms | 102.432 ms |
| Fused RMS and rotary | 88.264 ms | 92.618 ms |
| Preserved committed control after | 98.315 ms | 102.448 ms |

Against the faster bracketing control, fusion saves 10.021 ms/frame (10.20%).
Throughput is 0.906 times realtime, real-time factor is 1.103, and the mean
remaining gap is 8.264 ms; another 9.36% reduction reaches 80 ms without scheduling
headroom. The p99 is 95.127 ms and maximum 95.582 ms. All 386 warmed intervals
still miss the deadline. All 1,418 semantic events and the complete accepted
698,552-sample WAV bytes match, with SHA-256
`369f1334dea5e5d06fc450c03c06eb446a8eb673b60cc1e173e35b2641c1d282`.
This combined end-to-end measurement does not attribute isolated operator or
shader durations. Microphone/speaker scheduling and echo cancellation are not
measured.

Four focused Metal tests pass: independently computed grouped RMS values across
strided rows and widths 17/72/1152 with reduction tails; learned-offset RMS
input-precision and F32 scale boundaries; strided rotary pairs, partial tails and
F32 position rounding beyond `2^24`; and controlled/scheduled mixed-precision
cached publication. Reproducible native validation uses:

```sh
cargo test --offline --release -p eredu-backend-mlx --no-default-features \
  --features audio,metal --lib -- fused_grouped_rms fused_input_products_rotary \
  mlx_learned_offset_rms_preserves_precision_and_float32_scale_arithmetic \
  coupled_mixed_precision_matches_controlled_and_scheduled_cached_publication \
  --include-ignored --nocapture
cargo check --offline -p eredu-backend-mlx --no-default-features
```

The Metal/audio release build and non-Metal fallback compiler check pass.
External `/private/tmp/eredu-speech-fusion` retains reproduction scripts, the
report, exact benchmark and implementation patches, binaries and hashes, complete
logs, events and WAVs. All seven temporary on-load quantization source edits
match their implementation hashes after the build.

Full-conversation backbone projection comparisons retain native affine 8-bit
weights. The historical intrusive profile identifies 16.096 ms GPU/frame in 27
Mamba input projections and 16.709 ms in 25 dense MLP blocks, including their
normalization, activation and residual work. The 32.805 ms union identifies a
candidate subset; it is not a promised saving or a current shader-duration
measurement.

The same pinned checkpoint, corpus and accepted combined precision execute 396
live frames, with 386 warmed intervals after discarding the first ten. Each run
executes the complete conversation across successive model weights and cached
states. Loading and prefill are excluded; hot single-projection replay is not
used to select an implementation. Six generic packed-matvec layouts compare
indexed and pointer-based traversal, each with 8/16/32 output rows per 64-thread
group. Ordered F32 affine arithmetic is retained, and aligned native fast paths
remain selected. Three precision configurations replace selected Q8 source
weights with F16 weights using F32 projection arithmetic; normalization and
recurrent accumulation remain F32.

| Backbone comparison | Mean 80 ms frame | p95 frame |
| --- | ---: | ---: |
| Native steady controls, range | 88.278–88.478 ms | 92.349–93.016 ms |
| Indexed rows 8 / 16 / 32 | 108.680 / 89.453 / 90.315 ms | 113.623 / 93.586 / 94.540 ms |
| Indexed rows 8 repeat | 108.694 ms | 113.767 ms |
| Pointer rows 8 / 16 / 32 | 88.687 / 88.998 / 89.757 ms | 92.931 / 93.297 / 94.074 ms |
| F16 Mamba input weights | 91.900 ms | 96.105 ms |
| F16 dense MLP up/down weights | 93.082 ms | 97.117 ms |
| Both F16 selections | 96.436 ms | 100.368 ms |

None improves complete-conversation latency. An initial same-binary dense
control measures 108.284 ms, whereas its repeated unchanged configuration
measures 88.369 ms. The cause is unmeasured; that initial control is not used to
claim an optimization gain. Native throughput remains approximately 0.904–0.906
times realtime, with an 8.278–8.478 ms mean gap and another 9.38–9.58% reduction
required to reach 80 ms without scheduling headroom. All 386 warmed intervals
miss the deadline in every configuration. No live audio scheduling measurement
or isolated kernel attribution is implied.

All six packed layouts, the repeated indexed layout, and the input-only and
MLP-only F16 selections preserve all 1,418 semantic events and the complete
accepted WAV bytes. The combined F16 selection changes the answer to
“the sky appears blue due to a phenomenon called Rayleigh scattering,” producing
1,421 events and different audio. First lexical divergence is at invocation 162,
12,880 ms audio. Its longer wording changes the workload, so the combined timing
does not isolate precision overhead. Retained backbone binding requests are
10,005,583,872 bytes for Q8, 12,403,494,912 for F16 inputs, 13,078,863,872 for F16
MLPs, and 15,476,774,912 for both. These are parameter binding bytes, not measured
GPU traffic or total process memory.

Both generic kernel implementations pass focused Metal numerical comparisons
against native bit patterns and independent dequantized expectations, including
strided activations, nonzero affine companions and partial reduction tails.
External `/private/tmp/eredu-backbone-projections` retains `REPORT.md`,
`comparison.json`, pinned binary hashes, exact source/benchmark patches,
reproduction scripts, build/test logs, full event streams and WAVs. Restoration
manifests confirm all temporary source edits are restored. The production tree
contains neither these candidate kernels nor their experiment flags; native
8-bit selection and existing architecture/backend ownership remain unchanged.

Generic Metal grouped F32 RMS distributes its sixteen independent cascade
accumulator chains across a 32-thread SIMD group. Each chain retains the CPU
cascade's local addition/carry order; ordered SIMD exchanges retain its component
merge and tail order. The separate F32 square, precise reciprocal square root,
normalization product and learned-gain product retain their rounding boundaries.
Group and offset semantics remain neutral specifications. This mechanism changes
neither model equations nor RNG, cache, rollback or completion ownership, and
requires no feature flag. Weightless RMS and ordinary row sums retain their
scalar-per-row mechanism.

The same pinned checkpoint, corpus and accepted on-load Q8/mixed-precision
configuration provide complete-conversation validation on the M3 Ultra. Each run
contains 396 live frames and 386 warmed intervals after discarding the first ten.
Loading and prefill are excluded. All 1,418 semantic events and the complete
698,552-sample WAV match the accepted bytes, with SHA-256
`369f1334dea5e5d06fc450c03c06eb446a8eb673b60cc1e173e35b2641c1d282`.

| Grouped RMS comparison | Mean 80 ms frame | p95 frame | p99 frame | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Preserved control before | 88.316 ms | 92.285 ms | 94.727 ms | 95.370 ms |
| Grouped RMS only | 69.667 ms | 74.130 ms | 76.143 ms | 76.996 ms |
| Preserved final control | 88.217 ms | 92.505 ms | 94.506 ms | 95.464 ms |
| Grouped RMS only, repeat | 69.588 ms | 73.998 ms | 76.098 ms | 77.428 ms |

Against the fastest control and slower retained run, the reduction is
18.549 ms/frame (21.03%). Throughput is 1.148 times realtime, with real-time factor
0.871. Mean headroom is 10.333 ms (12.92% of the 80 ms budget). All 772 warmed
intervals across both retained runs meet the deadline. The slowest observed frame
leaves 2.572 ms of headroom. These figures do not establish performance at longer
conversation histories.

Exact GPU/CPU bit comparisons cover signed, varying-magnitude row sums at widths
1/7/17/255/257/4095/4097/65537, including partial vectors/chains and carries into
all three upper cascade levels. Independently computed grouped RMS comparisons
cover strided rows, learned offset and widths 17/72/1152. The retained
grouped-only mechanism passes the carry/tail and strided-group regressions. Reproducible focused native validation uses:

```sh
cargo test --offline --release -p eredu-backend-mlx --no-default-features \
  --features audio,metal --lib -- \
  parallel_cascade_matches_cpu_addition_order_at_carries_and_tails \
  fused_grouped_rms_handles_strided_rows_groups_and_reduction_tails --nocapture
```

A dependent 112-operation chain at width 1152 measures 4.73–4.74 ms GPU time for
the parallel variants versus 10.27 ms for the scalar cascade. The narrow
16-row/72-column chain is essentially unchanged. Extending the per-row
threadgroup strategy to weightless RMS and row sums measures 89.03–89.13 ms/frame
for the complete conversation, despite the isolated chain improvement. Those
extensions are absent from the production tree. These measurements do not
attribute isolated shader durations or distinguish the two extension costs.

External `/private/tmp/eredu-rms-parallel` retains the report, comparison data,
exact implementation/benchmark patches, reproduction scripts, binary/source
hashes, numerical logs, full event streams and WAVs. Restoration manifests
confirm all temporary on-load quantization and profiling source edits are restored.
This measures offline serial inference; live microphone/speaker scheduling and
echo cancellation are not measured.

The pinned native MLX command encoder exposes process-level batching through
`MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`, read during device
initialization. The Ultra defaults are 50 encoded operations and 50 MiB of
unique input buffers. Either threshold can trigger a commit after a primitive;
these are batching triggers rather than strict command-buffer size limits.
Per-primitive completion handlers and input retention remain intact.

A full-conversation sweep with the same grouped-RMS binary, accepted precision
recipe and pinned fixture varies resource thresholds across 128/256/512/1024 MiB
at 50 operations, and operation thresholds across 10/25/50/100/200/500 at
512 MiB, with additional 25-operation/256 MiB runs. All 1,418 semantic events
and accepted WAV bytes remain exact in all fifteen unprofiled sweep runs.

| Native batching selection | Mean 80 ms frame | p95 frame |
| --- | ---: | ---: |
| Four default controls | 69.606 / 69.693 / 69.670 / 69.588 ms | 73.944 / 74.048 / 74.136 / 73.807 ms |
| 50 operations / 512 MiB | 69.364 ms | 73.662 ms |
| 25 operations / 256 MiB, two runs | 68.885 / 69.078 ms | 73.544 / 74.091 ms |
| 10 operations / 512 MiB | 70.305 ms | 74.350 ms |
| 500 operations / 512 MiB | 70.995 ms | 75.289 ms |

The confirmed 25-operation/256 MiB selection averages 68.982 ms, versus
69.640 ms across four controls: 0.658 ms (0.94%) less mean wall time.
Mean headroom is 11.018 ms (13.77% of the frame budget), or 1.160 times realtime.
All 772 selected warmed intervals meet the deadline; the maximum is 76.889 ms.
The p95 ranges overlap, so these runs do not establish a tail-latency gain.
The selection is an explicit workload-specific launch override; native global
defaults remain unchanged. No new portable API, model-specific backend policy
or feature flag is required:

```sh
MLX_MAX_OPS_PER_BUFFER=25 MLX_MAX_MB_PER_BUFFER=256 <voicechat-command>
```

Bounded warmed Metal traces observe 142.737 command buffers/frame at default
and 230.023 at the selected setting. Counts range over 142–145 and 230–231
within 57 and 43 complete observed frames respectively. Agent-active counts
are 143.286 and 230.024, so the different conversation windows do not explain
the increase. The faster unprofiled setting increases command-buffer count;
its gain is consistent with timely GPU supply, rather than fewer submissions.
The traces do not isolate the cause. Profiling strongly perturbs the selected
run, so trace wall times are excluded from performance comparisons.

The existing Metal controlled/scheduled cached-publication test passes at
500 operations / 512 MiB. No completion, rollback or resource-retention code
changes. External `/private/tmp/eredu-mlx-batching` retains `REPORT.md`,
`comparison.json`, reproduction scripts, binary and artifact hashes, full event
streams/WAVs, native traces and per-frame command-buffer counts. These timings
exclude loading and prefill, use 386 warmed intervals per run, and do not
establish live I/O scheduling or longer-history performance.

Explicit attention retains the physical K/V head axis. Native batched products
broadcast a singleton RHS axis over contiguous query-head groups, avoiding
expanded K/V storage. Equal-head products consume native batch strides directly
without forced contiguous preparation. BF16 prepared products retain physical
RHS banks and map query groups onto those banks with the existing reduction
order. Blockwise attention uses the same grouped product. Native memory facts
price physical heads and blockwise F32 conversion; portable conservative scratch
allowances remain unchanged. InputScores scale, mask, softmax and probability
rounding boundaries, cache snapshots, rollback and completion ownership remain
unchanged. No feature flag or new public API is required.

The released speech transformers have sixteen query heads and sixteen K/V heads.
They receive no head-replication saving. Contiguous preparation can already alias
contiguous inputs, so removing it does not establish a particular byte-traffic
reduction. Complete-conversation measurements use the same pinned corpus,
accepted precision recipe and 386 warmed intervals per run:

| Shared K/V layout selection | Mean 80 ms frame | p95 frame | Maximum |
| --- | ---: | ---: | ---: |
| Fresh preserved control, default batching | 69.711 ms | 74.374 ms | 79.894 ms |
| Shared layouts, initial default run | 69.409 ms | 73.568 ms | 77.516 ms |
| Retained implementation, default batching | 69.271 ms | 73.809 ms | 76.585 ms |
| Retained implementation, 25 operations / 256 MiB | 68.873 ms | 73.083 ms | 78.648 ms |

The final default comparison saves 0.441 ms/frame (0.63%). The combined batching
selection leaves 11.127 ms mean headroom (13.91% of the budget), or 1.162 times
realtime. All warmed intervals in these four runs meet their deadline; all
1,418 semantic events and accepted WAV bytes remain exact. These measurements
show a modest gain for this equal-head checkpoint; the single combined run does
not establish an additional gain over the prior batching selection.

Independent scalar fixtures cover nonzero grouped attention with two batches,
six query heads, two K/V heads, strided sequence origins and unequal key/value
widths on CPU and Metal. Published BF16 attention and contiguous/paged
InputScores references pass, as does controlled/scheduled cached publication.
A BF16 allocator fixture with eight query heads, two K/V heads, 128 queries,
1,024 keys, key width 64 and value width 256 measures 22,159,424 bytes peak for
shared preparation versus 37,883,968 for separately prepared query tiles:
41.51% less with exact output. This measures grouped-attention preparation,
rather than whole-model memory use.

External `/private/tmp/eredu-speech-attention` retains `REPORT.md`, comparison
data, source/binary hashes, restoration manifests, reproduction commands,
native test logs and complete event streams/WAVs. Score-fusion experiments
preserve complete output but provide no end-to-end benefit; their code and
plumbing are absent from production. Timings exclude loading and prefill and
do not establish microphone/speaker scheduling or longer-history performance.

The native tests cover ordered F32 rounding, strided code views, signed gather
indices, PAD, synchronous and completion-owned invalid-index rejection, and
batched/per-depth norm parity at `[31, 1024, 512]` geometry. Neutral tests cover
cache flags, provider identity changes, eviction, failed work, long-row bypass,
cache admission, parameter rebinding, bounded and tensor-parallel RVQ execution,
first-tie behavior, retries and retained RNG. The independent spoken-text fixture
also passes. These fourteen focused tests supplement the complete local WAV
comparison; they do not certify native distributed performance.

Selective recurrence fixtures compare cold zero-state and multiple cached updates
against the independent scalar recurrence within `2e-5 * (1 + abs(expected))`.
Fused GPU carry and output match ordinary GPU prefill exactly across two batch
lanes, three heads, state widths 1/7/128 and strided vectors. Predecessor carry
remains unchanged. Grouped decision fixtures preserve ID order, scalar shape
errors and domain rejection. Ten focused tests additionally cover channel policy,
owner/follower rollback and replay, controlled/ordinary coupled parity, native
failure/unwind retention and prefill chunk boundaries. Native Metal/audio and
no-default-feature compiler checks pass.


The shared Metal scalar activation regression checks sigmoid and SiLU across
scalar, singleton and vector inputs without changing the exponential arithmetic:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --features audio,metal --lib native_pointwise_scalar_and_singleton_preserve_shape_and_rounding
```

Separate original NVIDIA backbone execution on the same released weights produces
finite hidden states and both heads for prefill and three full-history extensions.
External `REFERENCE.md`, `reference-backbone-recompute.json` and
`reference-backbone-recompute.npz` retain commands, source hashes and tensors.
It uses PyTorch 2.11.0, Transformers 4.56.0, the upstream Mamba CPU RMSNorm
reference and removal of the unconditional CUDA stream context. It does not alter
the backbone equations. The original cached CPU path fails on a missing
`conv_kernel_size` attribute; full-history recomputation supplies the reference.
These tensors have no numerical comparison with Eredu and establish no
tolerance or parity result. Complete NVIDIA speech comparison,
released-weight snapshots and distributed/residency comparisons remain unvalidated.

### Allocator and prepared-checkpoint memory

VoiceChat component and speech-provider realization use the ordinary allocator
policy: 1/32 of currently available memory, capped at 8 GiB, with the shared
256 MiB fallback. Explicit and preserved-native choices remain authoritative.
Completed binding and fixed prompt/silence preparation trim unused buffers;
inference retains reusable buffers. Final executable retirement trims after
nested owners finish ordinary-host destruction, preserving unresolved completion
roots. The generic bounded quantizer persists encoded tiles through the neutral
prepared-checkpoint cache and existing bounded SafeTensors leases, eliminating
complete packed host payload vectors. The default prepared-file budget is 32 GiB;
active source leases prevent eviction. `doc/backend-architecture.md` describes
location, identity, integrity, publication and budget contracts.

The M3 Ultra comparison uses the pinned checkpoint and accepted experimental
8-bit/group-64 backbone/function/causal-TTS recipe, F16 remaining speech/perception
weights with F32 arithmetic, F32 quantized TTS inputs/companions, and 25-operation /
256 MiB batching. Each run contains 996 live frames (79.68 s audio); 986 warmed
intervals exclude loading, prefill, drain and the first ten live frames. Native
memory probes and 250 ms macOS footprint sampling run without vmmap pauses or
concurrent native compilation.

| Long-conversation source/cache realization | Peak process footprint | Load | Warm mean frame | p95 | Frames above 80 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Memory-backed packed sources, explicit 8 GiB allocator control | 36.205 GiB | 98.379 s | 74.207 ms | 84.590 ms | 118 |
| Persisted packed sources, cold conversion, adaptive allocator | 22.773 GiB | 125.155 s | 74.097 ms | 84.806 ms | 117 |
| Persisted packed sources, reused cache, adaptive allocator | 22.501 GiB | 24.914 s | 74.092 ms | 84.766 ms | 117 |

The matched cold run reduces peak footprint by 13.432 GiB (37.10%). Three packed
sources contain 11,326,242,816 payload bytes (10.548 GiB), without permanent host
vectors. Fresh preparation includes source-content fingerprinting, file writes
and checksum work; reuse performs no native quantization tiles. Automatically
selected allocator limits are about 7.24 GiB with `ManagedDefault` provenance.
Unused native cache is zero after completed startup preparation. After session
drop and ordinary retirement, live and cached MLX counters are both exactly zero.
File-backed pages remain subject to operating-system caching, separate from those
counters. Legitimate history growth and the remaining late-frame deadline misses
remain; mean throughput does not certify hard realtime or microphone/speaker
pacing.

Both persisted-source runs reproduce all 3,818 semantic events and full accepted
WAV bytes exactly. The WAV SHA-256 is
`6cedc9b561f2fc98690678bd9bd61a9d099b3b010cc41ebef1f55df8a84b3884`.
External `/private/tmp/eredu-memory-cache` retains `REPORT.md`, `comparison.json`,
reproduction/build scripts, source/binary hashes, restoration manifests, full
WAV/JSONL output, sampled memory and per-frame latency. Native bounded-quantization,
retirement/completion and controlled/scheduled publication checks pass; neutral
cache coverage includes selected-content identity, bounded reload reads, lease
capacity, eviction, reduced-budget hits, unrelated-file protection, corruption
repair and incomplete-write cleanup. The portable facade check passes.

### Released-checkpoint tool round trip

The tool-round-trip measurement dated 2026-10-05 uses the resident Metal WAV
path, pinned `sample_fc.wav`, and the same checkpoint, machine and dev profile.
Input is `examples/speechlm2/sample_audio/sample_fc.wav` at the pinned Speech
revision: 190,278 mono PCM16 samples at 16 kHz, SHA-256
`265f9e5f58bff1e71f4354f7d83e2ff405a8405b29c2cfeb50eb9085042e9136`.
`voicechat-tool-silent.json` disables acknowledgements with
`random_acknowledgement=false` and `random_initial_reminder=false`, and uses
`DEFAULT_SYSTEM_MESSAGE` and `DEFAULT_TOOLS` from
`offline_voicechat_fc_infer.py`. Each definition adds the facade schema's outer
`"type":"function"`; this does not change the released template's rendered text.
The facade renders that template once. The framed prompt contains 709 tokens;
other configuration limits match the single-device run above.

`responses-tool.json` supplies the pinned `random_number_response.json` result
through the public response API, without executing an external tool:

```json
[{"call_index":0,"after_ms":800,"response":"<TOOL_RESPONSE>[{\"result\": 20}]</TOOL_RESPONSE>"}]
```

The explicit 800 ms wait exercises streaming input during tool waiting; it is not
the pinned offline helper's immediate-injection timing.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-silent.json \
  --input /private/tmp/eredu-voicechat-reference/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_fc.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-2.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-2.jsonl \
  --chunk-samples 640 --tail-frames 200 --max-steps 2000
```

One complete committed batch at 10,560 ms requests `generate_random_number` with
`{"min":1,"max":50}`. The matching request/turn/batch ticket is accepted at
11,360 ms, after nine perception-only frames during the wait. Agent text resumes
with “Your random number is twenty.” The trace contains 284 live frames,
65 perception-only frames, 59 tool substeps and terminal drain, finishing at
349 audio frames and 1,052 backbone positions. Every tool substep preserves its
preceding audio frontier; every perception-only step preserves its backbone
frontier. No scripted response is discarded or left unused. No external tool
execution, overlapping user speech during the wait or timeout is tested.

The waveform contains 605,060 contiguous finite float32 samples, including an
eight-sample codec tail: 27.44036 seconds at 22.05 kHz, absolute peak 0.14620356,
RMS 0.0065569352 and no samples outside [-1,1]. SHA-256 is
`c4a832224bd4d52cca0c5ed7e0a555b12fa62c7d0583f2610e03b57bc9b1ba78`.
The run takes 143.53 seconds including startup, with maximum RSS
105,873,637,376 bytes and peak process footprint 109,433,154,920 bytes. A unit-test
build overlaps this unoptimized run; these are process measurements, not a
real-time performance claim. `run-tool-summary.json` and `run-tool-2.log` retain
event assertions, waveform statistics and resource measurements.

Long prompts use the shared selective scan's existing chunk size (128 for the
released backbone). MLX evaluates recurrent carry and each chunk's output roots
before constructing the next chunk, preserving the original token equations and
the enclosing session's single commit. The focused CPU regression compares a
partial final chunk and an unchunked scan against independent scalar-recurrence
values and cached continuation. The test requires native device access.
The Metal-enabled test binary needs that access even with its CPU execution stream.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --features audio,metal --lib mlx_selective_scan_chunk_boundaries_preserve_outputs_and_cached_state
```

The WAV driver feeds queued audio between tool substeps. After source EOF, that
input uses bounded inserted silence; the live response tail begins after tool
work completes. Overlapping speech during tool waits is not exercised by this
round-trip recording. Independent NVIDIA full-conversation timing and numerical
parity remain unestablished.

### Released-checkpoint acknowledgement during tool serialization

The same resident F32 Metal setup renders the recorded phrase “Happy to help with
that.” during function-call collection and completes the subsequent result
response. The phrase is the first entry in the pinned streaming pipeline's
`data/fc_ack_messages.json`. The retained tokenizer produces content IDs
`[64866, 1317, 3508, 1454, 1455, 1046]`; BOS 1, seventeen PAD 12 tokens and EOS 2
complete its 25-position framing. The configuration is the full tool example
with `acknowledgement` set to that phrase. Its script supplies result 20 after
800 ms, with no initial or generic reminder.
`voicechat-tool-ack-no-reminder.json` selects this speech policy with the fixed
acknowledgement and `random_initial_reminder=false`.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-ack-no-reminder.json \
  --input /private/tmp/eredu-voicechat-reference/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_fc.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool-ack.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-ack.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-ack.jsonl \
  --chunk-samples 640 --tail-frames 80 --max-steps 2000
```

Thirty-seven committed Tool frames publish 65,268 finite samples during call
serialization, with peak magnitude 0.09493857 and RMS 0.0083107997. The complete
call publishes at 10,560 ms with `generate_random_number(min=1,max=50)`; its
matching result is accepted at 11,360 ms. Agent text subsequently contains
“Your random number is twenty.” Acknowledgement speech does not appear in agent
text or enter backbone feedback. The session drains at 19,280 ms with 241 audio
frames and 944 backbone positions, with no unused or undelivered script entry.

The full WAV contains 414,548 contiguous finite float32 samples, including the
eight-sample codec tail. Peak magnitude is 0.14620356, RMS is 0.0093572236, and no
sample reaches full scale. SHA-256 is
`e4d47d0845cd54ecc84534605f6dfda1ba487bb3f5f577c481418f4702ae05de`.
`speech-tool-ack-serialization.wav` contains the exact 2.96 seconds of Tool-frame
PCM for listening, SHA-256
`9a9c6e5905033ddaed226ff0d600b8c51ddef7a80afa1015367942dac0c7dff7`.
The dev-profile run takes 99.04 seconds including startup, with maximum RSS
80,435,118,080 bytes and peak process footprint 83,998,025,376 bytes. These are
process observations, not real-time latency claims. `run-tool-ack-summary.json`
retains the checks and measurements. Nonzero PCM establishes speech publication,
not intelligibility or independent NVIDIA numerical parity; random phrase
selection is not exercised.

Focused portable tests cover committed phrase progress, rejection/discard,
interleaved perception, complete PAD/EOS handling, snapshot replay, and matching
controlled/iterator streams through call, response and drain:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib acknowledgement_is_committed_speech_independent_of_function_feedback
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib iterator_and_controlled_tool_acknowledgement_publish_the_same_stream
```

### Released-checkpoint repeated calls with automatic acknowledgements

The default seeded pool runs through two calls in one resident F32 Metal session.
`input-tool-random.wav` contains the pinned `sample_fc.wav`'s 190,278 PCM16
samples, silence through sample 352,000 (22 seconds), then the same complete
recording again. The 542,278-sample input has SHA-256
`6274ed467350c61f36d045550aaf8b9a367ab0e833fb6427b4fce9267812a8b7`.
`voicechat-tool-random-no-reminder.json` uses the full tool prompt, seed zero, 768 audio frames
and 1,536 model steps as explicit capacities. It omits a fixed acknowledgement
and leaves random selection enabled. The two script entries return 20 and 37,
respectively, each 800 ms after its batch publishes, with no reminder phrases.
The configuration sets `random_initial_reminder=false`.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-random-no-reminder.json \
  --input /private/tmp/eredu-voicechat-reference/input-tool-random.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool-random.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-random.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-random.jsonl \
  --chunk-samples 640 --tail-frames 80 --max-steps 2000
```

Both batches request `generate_random_number(min=1,max=50)`. They publish at
10,560 ms and 32,480 ms; their distinct batch tickets accept results at 11,360 ms
and 33,280 ms. Each call serializes through 37 Tool frames and publishes 65,268
finite samples before its complete batch. The extracted clips
`speech-tool-random-serialization-{0,1}.wav` have peaks 0.19684838 and 0.19940870,
respectively. Agent text includes “Your random number is twenty.” and “Your random
number is thirty seven.”, with “What would you like to do next?” between requests.
The session drains at 41,200 ms, audio frontier 515 and backbone position 1,211,
with no unused or undelivered scripted result.

The full WAV has 885,536 contiguous finite float32 samples, including its
eight-sample codec tail, at 22.05 kHz (40.1603628 seconds). Peak magnitude is
0.19940870, RMS is 0.0078732340, and no sample reaches full scale. SHA-256 is
`471ea5e9b685ae796312a02d7f2feb85a1adbba10a0855d7f23e9e994f560368`.
The dev-profile process takes 197.82 seconds including startup, with maximum RSS
152,063,737,856 bytes and peak footprint 155,645,964,000 bytes. These measurements
do not establish real-time latency. `run-tool-random-summary.json` retains
per-call, waveform and resource evidence. This run establishes repeated tool
response and acknowledgement publication; it does not establish phrase
intelligibility, Python random-sequence parity, or independent numerical parity.
Automatic initial-reminder selection is outside this recorded path.

Focused portable tests cover fixed framing, seeded phrase selection across eight
call starts with discarded candidates, restore/fork replay, and equality of
controlled and iterator publication streams under both fixed and seeded policies:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib acknowledgement
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib streaming_phrase_selections_replay_across_discard_restore_and_fork
```

### Released-checkpoint automatic initial reminder with an early result

The resident F32 Metal path runs with both released speech defaults enabled in
`voicechat-tool-reminder.json`: seed zero, the full tool prompt, 512 audio frames
and 1,024 model steps. No fixed acknowledgement or initial reminder is supplied.
`responses-tool-reminder.json` returns result 20 after 800 ms, without an explicit
initial or generic phrase. The input is the pinned 190,278-sample `sample_fc.wav`
above. Checkpoint, base and reference revisions are those recorded at the start
of this section; `tool-reminder-fixture.json` retains configuration/input hashes.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-reminder.json \
  --input /private/tmp/eredu-voicechat-reference/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_fc.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool-reminder.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-reminder.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-reminder.jsonl \
  --chunk-samples 640 --tail-frames 80 --max-steps 2000 \
  > /private/tmp/eredu-voicechat-reference/run-tool-reminder.log 2>&1
python3 /private/tmp/eredu-voicechat-reference/analyze-tool-reminder.py
```

The complete `generate_random_number(min=1,max=50)` call publishes at 10,560 ms.
Its result is accepted at 11,360 ms while the reminder is speaking. The last
reminder frame commits at 12,640 ms; recorded playback finishes at 12,720 ms.
No response Tool frame precedes that receipt. Injection starts at 12,720 ms,
and agent text subsequently contains “Your random number is twenty.” RNNT
transcribes “okay can you give me a random number between one and fifty”.

The 26 Reminder frames produce 45,864 samples (2.08 seconds), with peak magnitude
0.08628845 and RMS 0.01216302. Their backbone position remains 842 while perception
continues to consume input. `speech-tool-reminder-initial.wav` retains this audio,
SHA-256 `d467f555beb21353ec9d36608d41621402f73adbeaabd7a1818fecf3fb593058`.
The 37 serialization frames also publish 65,268 acknowledgement samples.
The complete stream contains 460,412 contiguous finite float32 samples, including
the eight-sample codec tail, with peak 0.19684838, RMS 0.0089305244 and no full-scale
samples. WAV SHA-256 is
`dfcc5e63a916d1d0cde0100de9aac44a025902b3f03da3f6f521320a840e7089`.
Drain commits at 20,800 ms with 260 audio frames and 944 backbone positions;
no script entry or response remains undelivered.

The dev-profile process takes 105.54 seconds including startup, with maximum RSS
86,479,323,136 bytes and peak footprint 90,027,669,352 bytes. These measurements
are not a real-time latency claim. `run-tool-reminder-summary.json` records event
ordering, audio metrics and artifact hashes. This run establishes automatic
reminder publication and early-result gating through the public session; phrase
intelligibility, independent NVIDIA numerical parity and native distributed
execution are not established by it.

Focused portable coverage verifies both retained random streams across discard,
restore and fork, retention through EOTC, early-result playback gating and matching
controlled/iterator publication through acknowledgement, reminder, response and drain:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib configured_initial_reminder
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib streaming_phrase_selections_replay_across_discard_restore_and_fork
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib iterator_and_controlled_tool_acknowledgement_publish_the_same_stream
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::wait::tests
```

### Shared discovery and loading evidence

Generic metadata inspection admits the pinned official nested NeMo configuration
and its 1,632-tensor SafeTensors header, plus the published complete Q4 GGUF header.
Both resolve to realtime VoiceChat and expose the coupled component/feedback graph.
The tests exercise the ordinary artifact registry entry point. Default GGUF
catalog admission preserves affine matrix companions while lazily decoding
quantized buffers without `.weight` names through bounded F32 conversion.
The shared preparation tests also cover malformed configuration, missing text
assets for all three source layouts and explicit missing-configuration errors.
Moshi/PersonaPlex selection, source retention and portable scheduling regressions
use the same preparation contracts.

The generic loader completes the pinned default-reminder tool conversation with
this command on the Metal setup above:

```sh
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-reminder.json \
  --input /private/tmp/eredu-voicechat-reference/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_fc.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool-reminder.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-generic-loading.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-generic-loading.jsonl \
  --chunk-samples 640 --tail-frames 80 --max-steps 2000 \
  > /private/tmp/eredu-voicechat-reference/run-tool-generic-loading.log 2>&1
cmp /private/tmp/eredu-voicechat-reference/speech-tool-reminder.wav /private/tmp/eredu-voicechat-reference/speech-tool-generic-loading.wav
cmp /private/tmp/eredu-voicechat-reference/events-tool-reminder.jsonl /private/tmp/eredu-voicechat-reference/events-tool-generic-loading.jsonl
```

Both comparisons are byte-exact: 460,412 finite speech samples, the same agent
answer, tool ticket/result/playback ordering and final drain at 20,800 ms.
The WAV SHA-256 is
`dfcc5e63a916d1d0cde0100de9aac44a025902b3f03da3f6f521320a840e7089`;
the JSONL SHA-256 is
`d3123a39b37e0fca6c40dc0c1ede8e883d88a5c3bc009a46a76e56a1a6663166`.
`run-tool-generic-loading-summary.json` retains these comparisons. Process
observations are 106.79 seconds including startup, 86,494,806,016 bytes maximum
RSS and 90,079,033,192 bytes peak footprint. Concurrent compilation makes this
unsuitable for latency comparison. This is loading/output consistency evidence,
not independent NVIDIA numerical parity or native distributed validation.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib voicechat_nested_configuration_and_pinned_headers_use_generic_discovery
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib voicechat_complete_gguf_uses_generic_configuration_and_discovery
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib realtime::tests
CARGO_INCREMENTAL=0 cargo test -p eredu-gguf --test block_rows default_catalog_decodes_quantized_buffers_and_preserves_linear_weights
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --test portable_realtime_facade
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --features mlx,audio,metal --test portable_realtime_facade shared_realtime_loader_rejects_unavailable_observation_before_source_binding
```

### Released-checkpoint result after WAV EOF

The same resident F32 Metal setup completes a 12-second tool wait with the
default turn policy and no reminder phrases. The input is the same pinned
190,278-sample `sample_fc.wav` above. `voicechat-tool-late-silent.json` contains the
full tool prompt and configuration with `random_acknowledgement=false` and
`random_initial_reminder=false`;
`responses-tool-late.json` contains
one entry with `call_index: 0`, `after_ms: 12000`, and response
`<TOOL_RESPONSE>[{"result": 20}]</TOOL_RESPONSE>`.

```sh
CARGO_INCREMENTAL=0 cargo build -p eredu --no-default-features \
  --features mlx,audio,metal --example voicechat_wav
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-late-silent.json \
  --input /private/tmp/eredu-voicechat-reference/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/sample_audio/sample_fc.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool-late.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-late.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-late.jsonl \
  --chunk-samples 640 --tail-frames 80 --max-steps 2000
```

The batch publishes at 10,560 ms and its matching result is accepted at 22,560 ms,
after both the 11.892375-second WAV and an ordinary 80-frame tail would end
(18,320 ms). Fresh perception continues during that wait. The final response
Tool frame commits at 24,240 ms, with audio frontier 303 and backbone position
864. Exactly 80 live frames follow it before drain at 30,640 ms, audio frontier
383 and backbone position 944. JSONL records the committed work kinds rather than
individual function tokens, so this identifies completed response injection
without directly observing the EOTR token. Agent text is “Your random number is
twenty.” The run consumes every source sample, inserts 299,962 zero samples and
leaves no scripted result undelivered. Inserted samples include 442 for source
padding; one frame submitted during tool work is consumed as the first live
response frame, so ingress-purpose counts and executed frame kinds are distinct.

The output has 414,548 contiguous finite float32 samples, including the
eight-sample codec tail, at 22.05 kHz (18.8003628 seconds). Peak magnitude is
0.14620356, RMS is 0.0079215961, and no sample reaches full scale. WAV SHA-256 is
`b63f93d3ca741e9cb819071589745919018be406febf2a497dbf80cc8e0e8f2e`.
The dev-profile process takes 113.31 seconds including startup, with maximum RSS
80,441,901,056 bytes and peak footprint 84,007,167,600 bytes; these are not
real-time latency measurements. `run-tool-late-summary.json` and
`run-tool-late.log` retain the event, waveform and process evidence. The result
arrives before the default timeout, so this run does not validate timeout speech
or independent numerical parity.

The portable WAV-driver tests cover a late result through committed response-end
processing, partial PCM, backpressure, interruption, stale results and bounded
cancellation. The session snapshot test covers committed tool activity during
incomplete collection, restore, injection and interruption:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --test voicechat_wav_driver
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib snapshot_retains_complete_tool_calls_but_never_replays_delivered_calls
```

### Released-checkpoint tool-wait interruption and audio handoff

The same resident F32 Metal setup completes an overlapping-input run with an
explicit `interruption_speech_frames=3` override (the default remains 40).
`input-tool-interruption.wav` contains all 190,278 samples of `sample_fc.wav`,
silence through sample 192,000, then samples `[124000,196000)` of the pinned
`sample_general.wav`, copied without resampling or gain changes. The resulting
264,000-sample mono PCM16 WAV has SHA-256
`2a8d3c50ce47a783de1b41ff6a2c6f3e0ea4d509f4dcbcefbe9a6ce7c21c29e1`.
The inserted question is “do you know what color the sky is”. Configuration is
`voicechat-tool-interruption-silent.json`; its other options match the tool round trip.
The script delays the original tool result by 8,000 ms and selects initial
reminder “Let me check that for you while you wait.”

```sh
/usr/bin/time -l target/debug/examples/voicechat_wav \
  --config /private/tmp/eredu-voicechat-reference/voicechat-tool-interruption-silent.json \
  --input /private/tmp/eredu-voicechat-reference/input-tool-interruption.wav \
  --script /private/tmp/eredu-voicechat-reference/responses-tool-interruption.json \
  --output /private/tmp/eredu-voicechat-reference/speech-tool-handoff.wav \
  --events /private/tmp/eredu-voicechat-reference/events-tool-handoff.jsonl \
  --chunk-samples 640 --tail-frames 200 --max-steps 2000
```

The call publishes at 10,560 ms. Thirty-three reminder frames complete playback
at 13,280 ms. Three consecutive nonblank RNNT frames commit interruption at
15,920 ms in perception-only invocation 269, with audio frontier 199 and
unchanged backbone frontier 842. The next 103 invocations replay the queued
projected audio in order at the same recorded time, advancing only the backbone
to position 945. They publish no speech, transcription, text, tool calls or turn
events. On resuming live input, the agent answers “The sky is blue.” The
original request/turn/batch ticket reaches the public response API at 18,560 ms
and is rejected as stale. No response injection or later reminder frame is
published. These times are recorded application time, not wall-clock latency.
This run establishes interruption during a silent tool wait; the reminder has
already finished when the threshold is reached.

The run drains at 407 audio frames and 1,153 backbone positions. Its 659,744
float32 output samples include the eight-sample codec tail, are all finite, have
absolute peak 0.14620356 and RMS 0.0064277738, and have no samples outside [-1,1].
WAV SHA-256 is
`1e27d8ce456699fd0202661f7bf51b8826c8acf8c86e4a0e1ab35ee2d3b7ae2e`.
The dev-profile process takes 162.23 seconds including startup, with maximum RSS
115,548,651,520 bytes and peak footprint 119,112,878,328 bytes. Portable test
compilation overlaps the run; these are not performance guarantees. External
`input-tool-interruption.metadata.json`, `run-tool-handoff-summary.json`,
`analyze-handoff.py` and `run-tool-handoff.log` retain provenance,
frontier/late-ticket checks, waveform metrics and measurements.

The queue is bounded by `max_frames`; with the selected F32 activation precision,
each 4,480-wide frame occupies 17,920 payload bytes, for an admitted maximum of
9,175,040 bytes at 512 frames. Native handles, allocator overhead and independent
snapshot/fork copies are separate. Replay uses EOS/PAD input feedback at its first
position and PAD/PAD thereafter, without running either vocabulary head or
sampling. Its forced PAD positions age the ordinary feedback history while audio
time stays fixed. Normal tool completion releases queued features without replay
when fresh live processing resumes. This lossless interruption handoff is a
facade-selected extension of the pinned latest-frame-only async source behavior.

RNNT transcribes only “color the sky is” from the inserted question; its
accuracy is not repaired or established by replaying already projected audio.
The reference EOS lookback can affect short interrupted utterances with
few queued frames. This recording establishes a correct answer to its overlapping
question, not broad conversational quality or independent NVIDIA numerical
parity. Active reminder termination, default-threshold interruption and native
distributed handoff remain unvalidated.

Focused portable checks cover tool-phase interruption, failed/discarded
publication, neural commit rejection, feedback/history replacement at capacity,
unchanged caches/RNG/frontiers, and delayed scripted arrival through the driver:

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib interrupt
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --test voicechat_wav_driver delayed_script_arrives_after_interruption_or_is_reported_at_finish
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_interruption_replaces_feedback_without_advancing_candidate_state
```

The nonzero coupled replay fixture compares chronological direct backbone state
with replay, checks unchanged RNG and non-backbone clocks, and compares resident,
bounded, controlled and uninterrupted execution. Queue tests cover capacity,
ordered consumption and rejected/discarded branches; the facade test checks
replay selection and retry across serialization, waiting and response injection.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib sustained_tool_speech_interrupts_only_when_its_publication_commits
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_state::
```

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-media --no-default-features audio::wav::tests
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --test voicechat_wav_driver
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio --example voicechat_wav
CARGO_INCREMENTAL=0 cargo run -p eredu --no-default-features --features mlx,audio --example voicechat_wav -- --help
```

The portable facade exposes bounded host text preparation and a committed
function-channel policy. Text preparation uses the exact retained tokenizer
without padding, truncation or automatic special tokens. It validates ASCII,
byte/token capacities and released control IDs before execution; architecture
prompt policy applies BOS/EOS framing and repetition. Whitespace-only system
prompts omit text prefill. Empty tool-response encodings are rejected, including
nonempty input erased by a tokenizer normalizer. Host tokenizer failures retain
their original source through `BackendFailure`.

System/tool rendering follows the pinned [function-calling
template](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/function_calling/template.jinja),
using the facade's existing function definitions and schema validation. Its
bounded writer charges Unicode and HTML escaping as output is written. Complete
function payloads accept an object or list, optional textual TOOLCALL tags,
`parameters` as an alternative to `arguments`, and JSON-encoded argument
objects, then pass through the existing semantic parser. This normalization
corresponds to the pinned [streaming
pipeline](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/pipelines/streaming_s2s_pipeline.py)
and [offline
helper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py).
No tool executes inside this policy. Call events are returned together after a
committed EOTC, and response tokens advance only on matching committed feedback.
After the supplied response is exhausted, unforced function decoding continues
with the batch pending until committed EOTR. Callers deliver committed
observations exactly once and in order within each turn. Turn stamps and batch
tickets reject observations or results invalidated by an interruption. Input
rejection preserves the pending payload or response cursor.

Portable host-policy tests cover exact template text and escaping, retained
tokenizer errors, capacities, schema rejection, complete batch publication,
response peeking/commit and stale identities. The shared JSON-template
regression checks that the general bounded-writer seam preserves Python JSON
formatting. These tests establish host-policy behavior, not neural session
execution. Responses are already formatted ASCII text: the helpers add no
response tags or EOTR and do not perform the selected streaming configuration's
number conversion. Phrase selection and complete session snapshot/replay belong
to the enclosing conversation and session, respectively.

`PreparedVoiceChatPhrase` bounds the entire spoken sequence: BOS, text IDs,
`max(17, ceil(ASCII characters / 2))` tail PADs, then EOS. Its exact short/long
fixtures cover the ceiling and expansion bound. `VoiceChatToolWait` uses those
phrases with recorded timestamps and the existing committed tool channel. The
initial reminder completes despite early results; up to two distinct generic
selections have one-second gaps, and the final fifteen-second timeout begins
after playback. These defaults and ordering follow the pinned [streaming
configuration](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml#L104-L125)
and [tool-wait
pipeline](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/pipelines/streaming_s2s_pipeline.py#L1582-L1698).
Reminder output preserves generated feedback and waveform, including the PAD
tail and EOS, as in the pinned [reminder
executor](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py#L2016-L2092).

Wait tests separate generation from recorded playback completion: phrases played
out at 2, 5 and 8 seconds produce a final timeout at 23 seconds. They check early
results, unchanged admitted speech, duplicate receipts, same-time discard/retry,
backward-time rejection, interruption and natural EOTR after timeout injection.
Recorded replay compares identical committed speech/function actions with and
without abandoned reminder admissions. This is host-policy evidence; no native
tool-wait conversation, playback device, latency measurement or session snapshot
result follows from it. Random phrase choices are supplied explicitly, and
perception-only actions remain available while timing or playback is pending.

`VoiceChatTurnTaking` implements the selected RNNT core over first-joint-decision
blank evidence, with independent fresh-audio and generated-position counters.
Defaults include forty-frame EOU/BOU and prior-agent-token windows, first-user
minimum speech of two frames, later minimum of three, and low-density minimum of
six. An immutable resolver runs inside ordinary channel selection after raw
sampling, before effective feedback and speech. It publishes no host successor
until the caller observes matching committed architecture output. Invocation and
turn admissions reject duplicate or stale output and retain the function-channel
activity seen by the candidate, even if a tool response subsequently finishes.
Main-loop PAD-tail gating uses prior committed speech counters and preserves the
generated waveform separately from feedback substitution.

The [turn
fixture](../eredu-evaluation/fixtures/voicechat/README.md#rnnt-host-turn-policy)
executes NVIDIA's original `_apply_rnnt_turn_taking` method unchanged on
seventeen explicit host states. Expected tokens, represented RNNT state and
interruption flags compare exactly; no learned weights or tensor numerical
kernels execute. Separate tests exercise first-blank acoustic counters and
density freezing, unpublished/replayed resolutions, EOU and barge-in commitment,
independent clocks, interruption of tool waits, step-specific capacity
admission, strict prior-state PAD-tail gating, and a result completing between
admission and host observation. Architecture tests exercise same-frame evidence
through the coupled executor and rejection before RNG/history adoption. MLX
compiler checks cover the borrowed decision interface. The existing compact
native coupled CPU test passes through the same interface with fixed overrides;
these host fixtures are not native conversational or released-weight parity
evidence.

The pinned
[wrapper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py#L2253-L2307)
suppresses turn overrides during active tool work. The facade explicitly applies
the configured consecutive nonblank threshold to fresh perception while tool
serialization, waiting or response injection is active, including silent waits.
This is a facade policy, not a claim that the wrapper executes that check in the
same place. Committing the threshold frame invalidates the old tool ticket and
clears its wait; failed or discarded candidates preserve both. The pinned
streaming pipeline's quit-to-normal boundary supplies the EOS/PAD feedback
substitution while retaining neural caches. Already delivered speech remains
delivered. The wrapper's optional watchdog, redirect and ASR-head policies are
not implemented. Explicit host interruption remains available.

`VoiceChatConversation` stages turn, function and reminder successors through the
ordinary runtime prepublication gate. One owned admission fences canonical host
mutation with typed `Busy` backpressure. Private candidates share immutable text
and phrases while copying bounded mutable state. Mapping or capacity failures
discard the candidate; adoption after neural commit performs no fallible work.
Reminder receipts are compared with the actual spoken token retained by the
architecture executor and its host observer.

Portable conformance covers a persistent shared scheduler committing live SOTC,
retained-audio payload and EOTC invocations with automatic host adoption and queued
delivery, then cancelling perception work whose native completion is delayed.
Dropping scheduler progress preserves complete tool calls and speech. An
independent receiver delivers earlier output after request cancellation, and
reserved credit remains charged until native retirement. Queue saturation rejects
permits before enqueueing; delivery permits retry at the same frontier. Unused
allowances return at commit and exact payload charges remain until delivery.
Foreign permits and stale branches are rejected. Neural commit failure preserves
host state and queue credit, retains its original error source, and permits a
same-frame retry in the semantic transaction.
Focused host cases reject malformed EOTC after valid acoustic evidence, reject
exhausted speech/tool publication allowances, discard staged calls, and replay
reminders through recorded playback completion. Shared-runtime cases verify
one-shot mapping of pending, ready and failed observations, source-error
retention and prevention of neural commit on mapping failure.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib host_output_mapping
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::conversation
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::publication::tests
```

These tests establish transactional host composition and bounded publication.
Applications reserve output permits before enqueueing and consume the receiver;
the publication owner alone does not supply session snapshots or released-weight
conversation validation.

`prepare_prefill` and `prepare_next` select reserved work for the ordinary shared
scheduler. Portable driver cases compare immediate and controlled execution
with delayed completion and a fully observed discarded candidate; both publish
the same stream and one terminal codec tail. Other cases cover admission held
while queued, capacity rejection, scheduler cancellation, fresh/retained tool
selection, PCM during waits, forced result injection, reminder/perception fairness,
80 ms recorded reminder cadence, early-result playback gating and automatic drain
after input finish. Scheduler descriptors remain stable after moving permits into
the branch. Queue rejection and dropped work preserve accepted input and policy.

Tool selection defaults to prepared silence, matching `fc_async_use_real_audio`
in the pinned wrapper; the explicit real-audio override uses fresh or retained
features, with silence before an encoder result exists. Separate perception work
preserves every accepted frame during silence-backed tool steps. The test with
queued PCM verifies alternation rather than silently dropping older frames as
the reference UI queue can do. These tests use neutral fixed-output mechanisms;
they establish selection and transaction behavior, not native conversational or
distributed numerical parity. Prepared prompt/Aria/silence assets remain
responsibilities of the composition adapter. Finish declares a finite silence
suffix; advancement drains codec overlap after that suffix and pending tool work.

All five lifecycle cases execute `VoiceChatWork::submit`, checking the actual
architecture invocation, forced tool policy, source features, prompt framing and
distinct main/tool/reminder waveform behavior. Two additional cases reject missing
silence or mismatched prompt counts before executor entry and retry the same work,
and preserve original typed submission/observation errors without publication.
The committed-turn test distinguishes active-turn PAD-tail suppression from
idle-only tool suppression. Shared-runtime coverage verifies concrete observer
errors remain reachable through the cached `Arc` source chain.

`submit_mlx_voicechat` is checked with MLX and audio features. Its returned owned
completion borrows neither execution nor prepared input views. This compiler
evidence establishes native adapter compatibility, not a native full-session run.
The existing compact native numerical fixture has a sixteen-token head and uses
component submission; the facade resolver requires the released 131,072-token
domain. Production validation is not relaxed to fit that fixture.

```sh
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::driver::tests
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib main_pad_tail_and_tool_idle_gates_use_prior_committed_state
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib host_output_mapping_preserves_pending_ready_and_failed_observations
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio
```

The bound facade owns prepared execution and the ordinary session endpoints.
Focused neutral coverage compares controlled and iterator-driven finished streams,
and checks shared-executor backpressure before admission with same-time retry.
Iteration retains access to PCM ingress and output delivery between ticks. Cold
startup coverage rejects mismatched PCM precision and insufficient publication
reservation using the ordinary queue validators. Aria host planning reuses the
same subword schedule as prepared native prompt inputs.

Native startup tests retain prompt roots and execution ownership through exact
completion, and inject returned errors and unwinds after eager work. Failed
preparation publishes no output and keeps recovery ownership until safe
retirement. The retained-session MLX loader is compiler-checked with audio
enabled; a complete released-weight facade conversation is not established by
these component tests.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib -- iterator_and_controlled_advancement_publish_the_same_finished_stream bound_shared_execution_waits_before_admission_and_retries cold_start_admits_matching_input_and_complete_publication_reservation
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib composition::mlx::voicechat::submission::tests::preparation
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_aria_prompt_matches_original_initialization_and_batch_expansion
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio
```

The portable `VoiceChatSession` has neutral conformance cases using the
production prepared-work submission bridge. Recorded-tick iteration and explicit
advancement publish identical streams, including a 220-sample partial input and
one terminal codec tail. Pending work accepts PCM without duplicate submission;
cancellation preserves earlier output and keeps input/output reservations until
completion, with exactly 240 ms of recorded retirement latency. Output saturation
rejects submission, then resumes drain after receiver delivery. Submission and
neural-commit failures preserve concrete source downcasts and publish no candidate.
Backward session timestamps are rejected. Two core scheduler cases separately
cover explicit cancellation timing and failure abandonment of submitted siblings.
These tests use fixed-output neutral mechanisms; they do not establish native
checkpoint-backed conversation or waveform parity.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::session::tests
CARGO_INCREMENTAL=0 cargo test -p eredu-core --lib cancellation_retains_submitted_resources_until_completion
CARGO_INCREMENTAL=0 cargo test -p eredu-core --lib submission_failure_marks_work_and_request_failed
```

Finish conformance covers a padded partial frame followed by two lazily generated
silence frames, identical iterator/controlled event streams, finish declared while
the last source frame is pending, output backpressure, snapshot/fork replay and
cancellation of a pending continuation with no original PCM. Cancelled or rejected
work consumes no suffix frames and publishes no candidate. Source counters exclude
padding and continuation; acoustic timestamps include both. One terminal codec
tail follows the declared suffix and resolved tool work, without a separate drain
flag. Repeating a finish policy does not restart its suffix. Media tests reject
timeline overflow before closing admission and retain a large suffix as a counter.
Frontend tests compare exact mel output against explicitly submitted zero PCM.

The silence count is an explicit finite application choice. The pinned NVIDIA
[streaming
configuration](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml)
leaves its optional input-padding settings unset. Training channel delays do not
specify a universal finish duration, and finite silence does not establish
response EOS for arbitrary speech or a tool response.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-media --no-default-features --test streaming_audio pcm_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test voicechat_frontend finite_silence_finish
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::ingress::tests
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::session::tests
```

Committed agent-text and RNNT transcript deltas use the same staged successor and
bounded publication queue. Agent decoding reuses the ordinary facade tokenizer
stream, including byte-fallback carry; it consumes effective agent feedback and
filters declared control IDs. Ordinary EOS and drain reject incomplete byte
sequences, while interruption discards unpublished agent carry. Function and
reminder speech do not appear as agent text. RNNT decoding consumes only newly
emitted IDs, replacing word markers and trimming whitespace exactly as the pinned
wrapper; retained trailing whitespace is bounded. Both streams retain their
containing frame's model or source-sample attribution.

Nine cases in `transcription-nvidia.json` execute NVIDIA's original
`_rnnt_decode_text` unchanged and compare every prefix exactly, including Unicode,
Python information-separator whitespace, empty pieces and literal unknown or
byte-looking pieces. These are host presentation results, not learned RNNT parity.
Portable facade tests cover split UTF-8, controls, capacity retry, commit-only
delivery, exact byte charges and rejection before or during neural commit.

```sh
python3 eredu-evaluation/scripts/voicechat_transcription_reference.py \
  --wrapper /private/tmp/eredu-voicechat-turn-reference.py \
  --output eredu-evaluation/fixtures/voicechat/transcription-nvidia.json
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib voicechat::transcription::tests
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::
```

The optional ingress port has portable scheduler conformance for arbitrary partial
chunks, arrivals while completion is pending, full-queue backpressure including
the reserved head, and commit-driven capacity release. Host rejection, mismatched
source sample counts and neural commit failure preserve the head plus later
arrivals. Retained-tool and reminder work leave queued PCM unchanged while
perception-only work consumes it. Final partial input preserves original sample
counts; finish declared during a pending full frame requires a later finish-only
receipt. Empty finish produces no synthetic perception frame. Cancellation closes
the sender while retaining pending PCM until exact native retirement. These use
neutral mock neural transactions and the real shared PCM queue, not a second
inference engine or native speech-quality fixture.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat::ingress::tests
CARGO_INCREMENTAL=0 cargo test -p eredu-media --no-default-features --test streaming_audio pcm_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test voicechat_frontend
```

The media cases cover stable head peeking under new arrivals and allocation-free
prefix consumption. Frontend cases check F16/BF16 overflow rejection before
queueing, preservation of accepted PCM, and existing NVIDIA feature/window
fixtures through BF16 input and rolling-window advancement. The input facade
remains available without optional host-audio dependencies; actual mel extraction
uses the existing audio feature.

```sh
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib runtime::voicechat
CARGO_INCREMENTAL=0 cargo test -p eredu-text --no-default-features --test json_templates tojson_matches_python_json_dumps
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- voicechat_dynamic_channel_policy voicechat_execution_couples_components
CARGO_INCREMENTAL=0 cargo check -p eredu-backend-mlx --no-default-features --features audio
```

The portable coupled component executor joins those equations on the shared
generation branch. Its compact nonzero numerical test compares explicit controlled
advancement with `execute_frame_transition` and a bounded Nemotron runtime through
prefill, live input, perception-only work, a retained-audio tool step, reminder
speech and drain. It compares PCM, code frames, effective decisions, retained
perception/RNNT/Nemotron/TTS/codec tensors and RNG progress, with independently
specified expected clocks. The fixture preserves 37 startup positions, 1,764
samples per frame and eight tail samples. Character/TTS parameters come from the
independent component fixture with extended Aria prompt time; compact perception
and cadence-preserving codec parameters are deterministic nonzero test weights.
These are composition and transaction comparisons, not a full NVIDIA reference
conversation or released-weight parity measurement. Bounded execution here covers
Nemotron traversal, not residency of the entire speech graph.

The native coupled controlled run saves its generation owner after the first live
step, retaining 157 pending final PCM samples and nonzero perception, RNNT, both
TTS guidance lanes and codec state. It continues through perception-only, tool,
reminder and terminal drain, then restores and repeats that continuation. Decisions,
generated codes, waveform/transcription outputs, component clocks, all retained
tensor values, RNNT hypotheses, agent history and RNG keys compare exactly. The
shared publication revision advances on restore and cumulative copying increases.
An independently forked owner retains that same live boundary while the source
continues and restores. Its perception/tool/reminder/drain continuation compares
exactly under the same decisions, including full retained state and RNG. Releasing
the child returns its branch reservation while the source and snapshot remain.
This establishes model-generation snapshot/fork replay with compact numerical fixtures;
it excludes facade conversation/queue delivery, external tool execution and
released-checkpoint or distributed snapshot validation.

The native CPU coupled fixture also compares one nonzero input frame followed by
virtual silence with the same input followed by explicit zero PCM. Both silence
frames execute the live perception/RNNT, Nemotron, two TTS guidance lanes and codec
path, followed by tool, reminder and drain invocations. Decisions, codes, waveform,
RNNT emissions, retained tensor values, sampler history and RNG keys match exactly.
Only input provenance and original-sample attribution differ: continuation has
zero original samples at cadence sample 1,280; explicit input has 1,280. This is
compact-fixture mechanism parity, not released-weight conversation, distributed
validation or a measurement of speech quality or latency.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib coupled_native_
```

Shared native completion tests hold the runtime lock on a separate thread after
successful validation. A cloned completion remains publishable without blocking;
resource retirement remains guarded. Subsequent unresolved or failed observation
work invalidates the cached proof and prevents publication until its own terminal
conditions are established. This protects ordinary semantic commit between two
polls of the same completed invocation.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib realtime::completion::observation_tests
```

Portable facade snapshot conformance compares uninterrupted output with restored
continuation while 220 source PCM samples and an undelivered nonzero speech chunk
are retained. Complete tool calls survive queue copying, incomplete function
payloads resume correctly, and delivery fences prevent replaying an exposed tool
call. Tests reject newly accepted input/finish, explicit interruption and foreign
owners before copy admission. Injected neural-install failure preserves live PCM,
queued output and recorded time while retaining the original error source.
Scheduler counters and cumulative copy charges do not rewind; snapshot limits
cannot be reset after configuration.

Separate host cases replay a partially spoken reminder through receipt, playback,
gap and timeout decisions, and invoke reentrant input/delivery callbacks during
snapshot estimation, copying and native installation. Busy rejections protect both
queues. A retired diagnostic destructor submits new PCM after restoration; those
samples remain queued. These cases use neutral fixed-output mechanisms, not native
checkpoint-backed facade or distributed snapshot execution.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib snapshot
```

Portable fork cases exercise the ordinary session and host-policy paths.
They compare independent continuation from 220 queued PCM samples, preserve only
undelivered events, rekey queued tool calls, and map delivered pending tickets
without reissuing calls. Separate checks reject longer tool IDs that exceed the
original output bound before copy admission, preserve response-injection and
reminder/playback progress, retain branch credit through restore and pending
cancellation, and reject reused request IDs after failed copies or child release.
The shared budget remains configured across descendants.

Retention checks cover future child growth: oversized neural candidates preserve
queued PCM and publish no output, oversized diagnostic payloads leave the queue
unchanged, and rejected response injection preserves its pending ticket for retry.
Diagnostic admission includes reserved vector capacity even when only one element
is populated; standard tensor observations account for owned storage as well.
Empty ingress reserves full configured capacity. High-water storage survives
restore and native-retirement ownership without spending or refunding copy credit.
The native coupled test checks that each completed candidate's estimate covers its
committed successor's retained storage.

A native joint-state case drops the child and unpublished branch while keeping its
completion retention alive. The branch reservation remains charged until that
retention is released, independently of the source owner's cumulative native copy
counter. These cases do not establish released-weight facade or distributed fork
parity.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib fork_
CARGO_INCREMENTAL=0 cargo test -p eredu --no-default-features --lib rejected_external_response_growth
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib fork_growth_is_bounded
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib diagnostic_spare_capacity
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib fork_budget_remains_owned_by_native_completion_after_child_and_branch_drop
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib coupled_native_execution_matches_explicit_and_scheduler_publication
```

The host frontend's existing continuation test copies a partial queue after mel
window rollover and compares independent continuation, bounded admission and final
partial attribution. The portable PCM queue case also copies pending input through
its explicit host snapshot method without enabling FFT/audio dependencies.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-media --no-default-features --test streaming_audio pcm_partial_chunks_backpressure_finish_and_fork_preserve_every_sample
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test voicechat_frontend frontend_backpressure_snapshot_and_explicit_partial_finish_preserve_audio
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --test dense_sampling isolated_rng_snapshot_preserves_strided_key_words
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib coupled_native_execution_matches_explicit_and_scheduler_publication
```

A late character-capacity failure after perception and backbone execution leaves
canonical queued PCM, caches and RNG unchanged after candidate discard. Reusing
that partially advanced candidate rejects. Incomplete PCM returns `NeedsInput`
before claiming the candidate; explicitly finishing the partial frame permits
advancement on the same branch. A separate state-owner test rejects a joint neural
commit and checks that host input and feedback are not partially adopted.

MLX's joint cache owner has six direct native CPU tests with nonzero hybrid recurrent
state and two-row attention state. They check isolated candidate mutation,
discard, foreign-owner rejection, atomic publication and copy accounting for
both guidance rows. A finite shared paging pool rejects the second cache copy
after the first succeeds; canonical values remain unchanged, consumed copy
usage is retained, and releasing the competing allocation permits retry. The
successful paged branch has an independent namespace in that same pool and
releases the old canonical namespace on commit. These tests validate native
cache ownership; they do not execute the complete learned speech graph.

The isolated-copy cases capture through the portable `IsolatedStateSnapshot`,
continue canonical inference while retaining the saved state, and repeatedly
restore exact nonzero recurrent and two-row attention values and offsets.
Separately mutable copied owners share a cumulative copy allowance with their
origin; continued child mutation leaves the parent unchanged. Foreign or busy
restores reject before replacement, and exhausted copying cannot be reset by
copying or dropping descendants. A neutral injected-copy failure verifies that
recovery retains logical storage charges, failed restore leaves the target
unchanged, concrete error sources survive, and successful retry never refunds
prior copying. These component results do not establish full-session replay,
externally delivered speech/tool semantics or distributed snapshot behavior.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib failed_restore_retains_recovery_credit_and_never_refunds_copying
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test joint_state
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio,metal --lib -- candidate_updates_preserve isolated_native_hybrid_snapshots isolated_native_kv_snapshots coupled_native_execution failed_and_unwinding_voicechat --test-threads=1
```

Shared realtime snapshot conformance has two neutral cases: committed continuation
replays the same schedule and decisions after restore, and injected model-install
failure preserves canonical model, clock, sampler and RNG values. Restore advances
the live publication revision and never refunds cumulative copying; outstanding
model work rejects restoration before budget admission.

The native CPU composition case combines the joint cache owner, actual VoiceChat
schedule and nonempty agent history, and MLX RNG in the shared generation owner.
It advances both cache families, a retained-audio tool substep, effective history
and random state, then restores the saved cache values and all clocks/history and
reproduces the next nine Gaussian values exactly. This is a component-copy test,
not a coupled learned forward. Two RNG cases independently compare Gaussian,
Gumbel and uniform continuation across a copied key and verify that a stride-two
key retains the correct words and produces identical draws. Backing observations
confirm the saved key has independent storage. No complete VoiceChat frontend,
perception/codec carry, delivered-event replay or distributed snapshot result
follows from these tests.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib realtime::snapshot::tests
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test joint_state --test dense_sampling -- realtime_snapshot_restores_native_draws_with_voicechat_clocks_and_history dense_draws_replay_and_survive_decision_driver_adoption isolated_rng_snapshot_preserves_strided_key_words
```

The retained backbone construction driver selects exact static and per-layer
recipes before materialization and binds the ordinary Nemotron layer runtime.
Focused admission coverage verifies per-unit physical-source restrictions,
preserved conversion expressions and rejection of mismatched module identities.
The MLX build type-checks the joint cache adapter against VoiceChat's portable
neural-state contract.
Cold geometry coverage checks hybrid Nemotron slots, the released 28-layer
TTS local/global schedule, both guidance rows and mismatched-layout rejection.
Generation-preparation coverage checks retained control IDs, mandatory startup
capacity, PAD/PAD feedback, unchanged supplied RNG and explicit prefill. Native
initialization coverage uses compact declared geometry with nonzero two-row
TTS keys/values: selected device and paged placements both support joint discard
and commit, while mismatched selections receive typed errors. The shared selector
already guarantees checkpoint and rollback support for its opaque selections.
Shared completion recovery coverage checks that an erased execution owner is
poisoned on failure and retained until native observation resources can safely
be released. This owner is also used by existing Moshi composition.
It also instantiates the retained selected-component construction driver with the actual
MLX neural and parameter backends and hybrid cache type. The portable facade
build checks that the same construction declarations remain available without
a concrete backend. These compiler checks do not establish released-weight
materialization or coupled native numerical parity.

Native submission coverage injects nonzero eager CPU work into the coupled
adapter. Returned errors and unwinds attach failed completion, poison the owner
and leave canonical caches unchanged. A successful synthetic prefill passes the
architecture host observer, commits retained nonzero tensors and holds joint
admission until the native frame is consumed. Separate scope tests verify that
failed eager work requires no final evaluation event, token-validation scope
closes during unwind and invalid tokens block publication. Joint-retention
coverage checks that a dropped candidate cannot release admission still held by
completion, and that release does not refund copy usage. These tests establish
submission lifetimes with compact native state, not learned coupled numerical
parity. Codec reference coverage also checks traversal of neural carry and both
real and imaginary spectral histories against the independent fixture caches.
The two coupled submission tests enable `audio` to construct ordinary host input
state; the joint cache and completion-scope tests require no host-audio feature.

A compact MLX CPU test executes the learned coupled components through the native
submission scope and host observer. It compares explicit branch advancement with
ordinary shared-scheduler publication for prompt prefill, live speech,
perception-only tool waiting, a retained-audio tool step, reminder speech and
drain. Same-seed runs compare PCM, codes, transcription, all retained component
and neural tensors, effective-token history and RNG state exactly. Independent
expected clocks require 37 startup positions, 1,764 samples per speech frame and
eight drain samples. Input arrives in 157-sample chunks, including a padded final
frame; a preflight `NeedsInput` rejection leaves the same candidate reusable.
Forced nonblank RNNT output and nonzero Nemotron Mamba parameters exercise retained
LSTM, convolution and recurrent state. Tool and reminder steps leave the input
frontier unchanged, and perception-only work and drain consume no RNG draws.

`eredu-evaluation` supplies the shared compact perception, character/TTS and codec
fixture builders used by neutral and native tests. Native Nemotron blocks use
deterministic nonzero parameters, including Mamba state parameters that differ
from the scalar fixture's initialization. Exact equality here compares execution
paths on one backend; it is not an independent NVIDIA coupled oracle. The test
uses the production coupled equations and private native submission mechanism.
It does not exercise released-artifact materialization or the public resident
aggregate constructor, whose admission deliberately requires released geometry.
Native full-graph residency, distributed execution and released-weight parity
have no validation evidence from this test.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test joint_state
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib voicechat::backbone::tests
CARGO_INCREMENTAL=0 cargo check -p eredu-backend-mlx --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib --test reference_numeric -- voicechat_neural_geometry_preserves_guidance_lanes_and_attention_windows voicechat_generation_preparation_admits_startup_and_retains_cold_feedback
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_initialization
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib -- failed_observation_becomes_releasable_only_after_its_native_scope_settles completion_resources_share_the_executions_poison_authority
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib -- joint_retention_keeps_branch_admission_until_completion_owner_releases_it failed_invocation_scope_retains_eager_work_without_a_final_event unwinding_invocation_scope_poison_closes_token_validation completed_invalid_token_scope_fails_readiness_before_publication
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib -- failed_and_unwinding_voicechat_invocations_attach_completion_without_publication successful_voicechat_submission_releases_next_branch_after_host_delivery
CARGO_INCREMENTAL=0 cargo test -p eredu-codec --test ear_reference ear_neural_spectral_streams_match_original_nvidia_cache_and_flush
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib coupled_native_execution_matches_explicit_and_scheduler_publication
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone voicechat_publication_gates_owned_speech_transcription_and_failed_candidates
```

Owned host observation crosses the ordinary core scheduler's completion gate
with actual coupled component output. The test delays completion and verifies
that PCM conversion, queued-input consumption and RNG publication remain pending.
Successful publication returns independent 1,764-sample PCM, complete codes and
two RNNT symbols with the original 157 source samples of a padded final frame.
Injected nonfinite PCM, incorrect source coordinates, host-copy failure,
completion failure and failed event creation publish no candidate. The latter
retains and waits its quarantine completion before discard. Cancellation also
delivers no candidate. These submission tests use synchronous numerical mechanisms
with explicit completion controls, not native asynchronous recovery proof.

The public MLX channel observer has separate CPU coverage for floating PCM,
integer IDs, aligned coordinates and F16 diagnostic conversion. Invalid tensor
shapes and storage kinds reject. Moshi's existing observer uses this same
mechanism before its architecture projection; VoiceChat's owned observer accepts
the generic host envelope.

```sh
cargo test -p eredu-architectures --test reference_numeric -- voicechat_execution voicechat_state
cargo test -p eredu-architectures --test reference_numeric voicechat_publication
cargo test -p eredu-backend-mlx --no-default-features --test realtime_observation
```

Shared realtime ingress retains channel-keyed categorical and dense inputs and
selected overrides, admitted together before native materialization. Moshi's
categorical adapter preserves batch ordering and omits unselected forcing
sentinels. Shared realtime completion supports categorical and dense outputs,
including distinct current and aligned coordinates. Neutral scheduler tests
verify that nonfinite dense output prevents commitment after exact completion;
Moshi/PersonaPlex use the same envelope with portable output projection. The
existing frame interpreter and ordered-decision schedule remain categorical
policies; these shared contracts alone do not execute the
coupled VoiceChat graph or publish its transcription/tool events.

The shared generation transaction accepts architecture-owned schedule state,
sampling controls and transition inputs with an explicit sampler-count contract.
Neutral conformance exercises independent audio/model/tool clocks through the
same ordinary and explicit-advancement transition, schedule rejection before
model publication, and stale-branch rejection after commit or reseeding. These
checks establish transaction behavior, not VoiceChat conversational policy.
Zero-sampler branch coverage checks dense RNG publication, discard and synchronous
draw failure through the same mechanism used by categorical decision drivers.

Cold selection retains canonical policy descriptors, channel layouts and sampler
counts without requiring a categorical speech configuration. The shared session
scheduler accepts architecture-defined inputs and state under that retained
contract. Neutral tests exercise separate audio, model and TTS progress, exact
admission, release/resume, cancellation and failed completion. Moshi/PersonaPlex
construction and the portable facade use this same generic scheduler. No
checkpoint-backed VoiceChat session or native distributed VoiceChat result follows from
these mechanism tests.

`voicechat::schedule` provides the architecture's checked invocation policy with
explicit capacity limits and exact component-progress receipts. Neutral tests use
independently advanced mock components through `RealtimeFrameTransition`; they
compare ordinary and explicit branch execution, reject inconsistent receipts and
check capacity before component calls. This is schedule conformance, not numerical
or released-checkpoint coupled execution evidence.
Retained-audio steps require earlier perception. Empty drain is terminal without
emitting samples, and stale architecture predecessors are rejected even inside a
branch created at the current shared transaction revision.

| Invocation | Live audio / perception / RNNT | Nemotron | TTS guidance lanes | Codec |
| --- | --- | --- | --- | --- |
| Prefill | unchanged | framed prompt length | 37 Aria positions | unchanged |
| Live audio | one frame | one position | one position | one frame |
| Perception only | one frame | unchanged | unchanged | unchanged |
| Tool with fresh audio | one frame | one position | one position | one frame |
| Tool with cached silence or retained audio | unchanged | one position | one position | one frame |
| Reminder | unchanged | unchanged | one position | one frame |
| Drain | unchanged | unchanged | unchanged | eight tail samples for a nonempty stream |

Each codec frame emits 1,764 samples; the eight-sample startup latency is retained
as attribution metadata rather than subtracted from emitted samples. Plans add no
runtime FIFO from the training text/speech delay fields. Initial agent/function
feedback is PAD (12), not the token suggested by the reference helper's BOS name.
Decoded agent sampling precedes effective-token overrides; function prediction is
greedy. The plan declares one agent sampler and preserves the distinct main/tool/
reminder rendering modes. It does not execute sampling, choose tool/turn policy,
or resolve the differing native/vLLM prefill RNG behavior.

`voicechat::agent_sampling` executes agent selection through shared sampling
mechanisms. The streaming defaults are temperature zero, top-p one and repetition
penalty one. Original-logit BOS/EOS/PAD winners bypass stochastic processing;
otherwise unique nonspecial effective feedback receives sign-dependent repetition
penalties before temperature and nucleus filtering. Raw selection does not append
history: callers accept the effective token after overrides. History capacity
exhaustion is typed and never silently evicts feedback.

`agent-sampling-nvidia.json` executes the hash-verified NVIDIA
`ModelInterface._sample_text_token` method on CPU with PyTorch 2.14.0. Ten cases
cover greedy bypasses, sign-dependent penalties, history prefixes, effective
feedback and nucleus crossing. The oracle replaces only multinomial selection
with a supplied-uniform inverse CDF and records final probabilities. Its declared
tolerance is `2e-7` absolute plus `2e-6` relative; widening the exact float32 inputs
measures maximum probability error `5.3338e-8`. Neutral comparisons use those
probabilities and count random draws. Shared-branch coverage checks that raw
selection, effective feedback and subsequent dense draws publish together,
including failure without completion, replay and stale forks.

The MLX CPU test compares native sampling against the fixed NVIDIA probabilities
under identical native seeds, then compares the following Gaussian draws. This
checks native policy and RNG progress, not PyTorch/MLX random-stream parity or
released-weight conversation behavior.

`voicechat::channel_decision` composes agent selection, greedy function selection,
explicit feedback overrides and independent spoken text in the shared unpublished
branch. Tool acknowledgments do not replace PAD agent feedback. Neutral tests
check failed function selection after an agent draw, replay and effective feedback
through the next cached Nemotron invocation against independently indexed channel
embeddings and retained caches. MLX CPU coverage checks rank-three native logits,
signed feedback materialization and subsequent Gaussian RNG position.

Coordinated channel selection uses the same sampler, override rules and shared
decision driver. Neutral coverage compares a nonzero NVIDIA nucleus-crossing
case with ordinary execution. Identity, domain, perception-evidence and retained
history mismatches reject work before draws. Owner resolver failure after a draw
and invalid follower publication preserve both RNG and effective history; fresh
authority retries reproduce the expected raw token, PAD tool feedback and spoken
override. Followers retain no RNG and invoke neither selection nor host policy.
The native two-rank CPU Ring test selects rank one as owner and compares Tool
and Main decisions, retained history and subsequent Gaussian draws with ordinary
execution. These component checks do not establish a complete distributed
conversation or released-weight parity.

Token identities and same-backend
RNG continuation compare exactly; independent NVIDIA probabilities retain the
fixture tolerances stated above.

```sh
python eredu-evaluation/scripts/voicechat_agent_sampling_reference.py "$SPEECH_CHECKOUT" eredu-evaluation/fixtures/voicechat/agent-sampling-nvidia.json
cargo test -p eredu-architectures --test reference_numeric voicechat_agent
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_agent_sampling
cargo test -p eredu-architectures --test reference_numeric voicechat_channel_decision
cargo test -p eredu-architectures --test reference_numeric voicechat_channel_coordination
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_agent_sampling coordination::native_agent_sampling_owner_decisions_preserve_feedback_and_speech_rng -- --ignored --exact
```

Typed realtime output packaging accepts component-owned output and resource
bundles through the existing completion and prepublication boundary. Runtime
coverage distinguishes failures before submission from quarantine after
submission, and checks that mixed-output host validation gates commitment.
Categorical Moshi/PersonaPlex callers use the same generalized contracts.

`voicechat::output` assembles executed components and an observed schedule receipt
into that shared envelope. Neutral coverage checks prefill, live audio,
perception-only work, retained-audio tool steps, reminders and both empty and
nonempty drain. Nonzero PCM and categorical values verify native shape
normalization, independent invocation/audio/output clocks, raw versus effective
decision identity and exact eight-sample tail attribution. Corrupted observer
results with mismatched decision IDs, coordinates, nonfinite PCM or wrong lengths
are rejected before projection. This is output-contract evidence, not a coupled
neural execution or native publication measurement.

```sh
cargo test -p eredu-architectures --test reference_numeric voicechat_output
```

The pinned [streaming
wrapper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py)
executes TTS and codec on every async model step and runs reminders without the
backbone. Perception-only advancement during API waits expresses the portable
queue-preserving policy; the wrapper's worker lifecycle and dropping of older
queued perception results are not reproduced. Codec-tail accounting follows the
validated EAR codec component; the wrapper does not itself call codec flush.

The input composition tests join the actual frontend, perception and RNNT calls.
They compare whole PCM submission with arbitrary chunks, preserve attribution
for final padding, and verify retry without consuming parent PCM or caches.
The shared-encoder test compares nonzero perception outputs and caches against
the NVIDIA fixture and drives an analytic RNNT decision boundary using raw
encoder features. It is not an independent full-stack NVIDIA reference run.

The speech composition test carries pinned TTS decisions through feedback policy
and the actual codec. `speech-nvidia.json` records original NVIDIA codec output
for ordinary/main-EOS and tool-EOS waveforms plus flush tails, using compact
nonzero parameters. Source and input fixture hashes are checked by its generator.
The predeclared codec tolerance is `2e-5` absolute plus `5e-5` relative; the
independent widened-neural, float32-spectral comparison measures maximum absolute
waveform error `9.1735e-8`. The test also compares controlled refinement with
uninterrupted refinement and rejects incompatible retained component policies.
These compositions do not establish released-weight, full session, residency,
or native distributed validation.

```sh
cargo test -p eredu-architectures --test reference_numeric -- voicechat_perception_rnnt voicechat_input voicechat_speech
python eredu-evaluation/scripts/voicechat_speech_reference.py "$SPEECH_CHECKOUT" eredu-evaluation/fixtures/voicechat/speech-nvidia.json
```

Backbone composition has neutral numerical coverage for framed prompt prefill
and repeated cached audio/feedback steps through resident, bounded and TP2
execution. Tests use independent manual channel fusion, the pinned nonzero
embedding/head fixture, ordinary Nemotron readouts, and scalar function-head
projection; both outputs and retained cache values are compared. Provider events
and positions check that two heads do not cause two backbone traversals.
Signed VoiceChat ingress also crosses the existing TP2/PP2 boundary test, with
normalized-hidden readout retained across stages. The default unsigned text
boundary has its own regression. These are compact neutral component tests,
not released-weight conversation or native distributed validation.

Progress observation is checked against actual backbone cache positions, including
stateless MLP slots, and the existing speech fixture's retained guidance and codec
state. Mismatches between TTS caches and retained speech are rejected. A complete
local receipt requires a stateful local backbone slot; no remote frontier is
inferred from absent state.

```sh
cargo test -p eredu-architectures --test reference_numeric -- voicechat_backbone_composition voicechat_speech nemotron_hidden_readout_crosses_tp2_pp2_boundaries dense_nemotron_h_partitioned_tp2_pp2_matches_prefill_and_repeated_decode
cargo check -p eredu --no-default-features
cargo check -p eredu-backend-mlx --no-default-features
```

Shared scheduling and contract checks are:

```sh
cargo test -p eredu-architectures --lib voicechat::schedule
cargo test -p eredu-runtime --lib -- realtime_completed realtime_interpreter realtime_executor realtime_channel_ingress realtime_ingress
cargo test -p eredu-runtime --lib -- realtime::tests realtime_session realtime_executor
cargo test -p eredu-runtime --lib -- realtime_contract realtime_selection realtime_model
cargo test -p eredu-architectures --lib native_metadata_selects_exact_dense_realization_before_construction
cargo test -p eredu-architectures --lib projection_preserves_batch_rows_depth_order_and_delayed_channel_identity
cargo test -p eredu-architectures --test reference_realtime_construction -- bounded_independent_adapter_consumes_the_complete_neutral_preparation_stack declared_realtime_interventions_causally_reach_decisions_and_output_once personaplex_prompt_plan_materializes_portable_frames_in_architecture_order
cargo test -p eredu --no-default-features --test portable_realtime_facade
cargo check -p eredu-backend-mlx --no-default-features --tests --example personaplex_step_bench
cargo test -p eredu-backend-mlx --no-default-features --test dense_sampling dense_draws_replay_and_survive_decision_driver_adoption
cargo test -p eredu-backend-mlx --no-default-features --lib mixed_realtime_observation_preserves_values_and_rejects_reshaped_payloads
cargo test -p eredu-backend-mlx --no-default-features --lib mixed_realtime_channels_materialize_exact_native_values
```

## Release contract

| Source | Immutable revision |
| --- | --- |
| [Official checkpoint](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/tree/443794ea956ef0065f001967ffd00e77f519cb39) | `443794ea956ef0065f001967ffd00e77f519cb39` |
| [NVIDIA reference](https://github.com/NVIDIA-NeMo/Speech/tree/097dfe9e2f55baf653b83035868bdc89849f1b47) | `097dfe9e2f55baf653b83035868bdc89849f1b47` |
| [Base configuration/tokenizer](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/tree/6533e8de2c68e4536bf7c411d7a3ce5734111476) | `6533e8de2c68e4536bf7c411d7a3ce5734111476` |
| [Community GGUF](https://huggingface.co/hoidhxd/NVIDIA-NemotronLabs-VoiceChat-11B-GGUF/tree/89883a05a031557729771f94abb9998e4facdd45) | `89883a05a031557729771f94abb9998e4facdd45` |

Configuration admission accepts the official nested NeMo document without a
top-level `model_type`. It never executes `_target_` strings. Explicit normalized
policy includes 80 ms frames, 16 kHz input, 22.05 kHz output, fixed Aria conditioning,
RNNT neural geometry, and the pinned reference's Transformers 4.56.0 defaults.
For example, Gemma3 attention uses query scaling 256 with head dimension 72;
substituting the head dimension changes the equation.

The strict physical catalog covers all 1,632 official tensors, including
preprocessing buffers and six I64 control/lookup buffers. Complete GGUF F16,
Q8_0 and Q4_0 layouts preserve these identities. The four-file bundle uses
component-specific names and transformations. Its missing canonical inputs are
15 RNNT tensors, 76 codec encoder tensors and two scalar TTS padding buffers.
`VoiceChatArtifactPlan::new` rejects an incomplete selected component with
`AdmissionError::MissingComponent`; an explicit complete companion fills only
missing parameters. Duplicate primary providers are rejected.

`AdmittedVoiceChatSource` validates physical headers and metadata. Canonical
recipes retain recurrent `-exp(A_log)` inversion, Gemma norm offsets, declared
reshapes and exact integer casts. Quantized recipes consume decoded F32 source
views, including convolution parameters. Per-parameter reports distinguish
physical stored bytes, canonical output bytes and a conservative conversion peak
including input, the decoder's additional F32 vector and recipe temporaries.
These are parameter conversion bounds, not model residency or native allocator
measurements.

`VoiceChatArtifactPlan::retain` binds admitted stores once, preserving their file
identity guards and restricted component views. Seven auxiliary assets retain
exact bytes, length, SHA-256, repository and revision. A caller supplies resolution;
construction has no network dependency. Embedded split backbone tokenizer arrays,
merges, token types, special IDs and chat template must match those assets.
Selected integer buffers and split character tables are validated with reads of
at most 4,096 integers per callback. Their canonical I64 hashes also admit exact
I32 storage in the split bundle. The released TTS special-flags buffer marks EOS
at base token 12; the live text-channel EOS override at token 2 does not rewrite it.

Validated integer payloads are retained from the same bounded read pass, with
canonical host payload bytes reported separately. `PreparedHostConditioning`
uses those values and retained tokenizer bytes to construct the 256-character
mapping, speech feedback controls and live text/user/tool markers, including
listening token 11. It performs no source-store reads. `PreparedSpeechParameterUnit`
uses a module's parameter traversal to select canonical recipes and restrict its
source to their physical inputs. Generic runtime selection and atomic binding
preserve recipe transforms and aliases; host buffers are rejected as learned
parameters. These independently selectable units do not constitute a full graph
residency or parallel execution policy.

`SpeechConstructionPlan` validates component geometry before creating neural
modules and exposes independently constructible perception, RNNT, function-head,
character/TTS, MoG and codec units. `PreparedSpeechConstruction` admits only units
whose component sources are retained and pairs each unloaded module with its
canonical bindings. Perception, TTS and codec F16/BF16 recipes include an explicit
F32 execution cast; materialization bounds include its overlapping input and
output storage. RNNT and function-head recipes retain their component precision.
These factories do not allocate session state or select a full-graph residency
or parallel policy.

`SpeechConstructionPlan::assemble_with` constructs each declared unit once and
offers it to an explicit binder before consuming it into
`LearnedSpeechComponents`. `PreparedSpeechConstruction::materialize` supplies the
retained-source binder through generic runtime selection, completion-safe
materialization and atomic parameter replacement. Missing required companion
roles fail before native construction or payload reads; binding errors stop
assembly without returning a partial aggregate and retain their original source
through `BackendFailure`.

The aggregate borrows `execution::Components` around a caller-owned Nemotron
runtime and host conditioning. Its startup encoder uses the decoder's quantizer,
with one retained codec codebook set. Consuming codec constructors check exact
unit order and geometry and preserve supplied parameter objects without
redeclaring learned slots. Materializing one unit at a time does not bound the
aggregate's final residency: it retains all bound speech weights. The assembly
supplies learned components; the enclosing prepared realtime loader constructs
the native joint Nemotron/TTS transaction and public session.

Perception admission retains the saved 400-value analysis window and 128-by-257
mel filter bank using bounded reads of at most 4,096 floating values. F16, BF16
and F32 values convert to canonical finite F32 host coefficients, with singleton
reshapes preserved. The retained payload is 133,184 bytes. Prepared frontend
construction copies these coefficients into a padded 512-value window and the
filter table (133,632 bytes); FFT workspace and feature caches are additional.
Host preprocessing coefficients are excluded from learned parameter units.

Focused construction coverage reconciles every learned speech parameter name and
shape with the pinned official header, verifies unique ownership across units,
and rejects out-of-range unit indices. The assembled census preserves bound
tensor sentinels and unique parameter ownership, including the single codec
codebook set. An injected binding failure stops further unit construction.
Codec assembly coverage rejects incomplete, reordered and incompatible units;
the nonzero NVIDIA decoder fixture exercises consumed, independently bound units
through cached waveform execution and drain. Precision tests preserve source authority
while checking F16/BF16 execution casts and their allocation bounds. Saved
coefficient tests cover F16/BF16 SafeTensors, bounded big-endian F32/F16 GGUF reads,
and nonfinite rejection. Host FFT tests use independently calculated powers for
explicit coefficients; the existing NVIDIA mel fixtures cover the frontend's
rolling-window and PCM precision behavior.

```sh
cargo test -p eredu-architectures --lib voicechat::preprocessing::tests
cargo test -p eredu-architectures --lib voicechat::parameters::tests
cargo test -p eredu-architectures --test reference_realtime_construction \
  voicechat_construction
cargo test -p eredu-codec --test ear_reference ear_assembly
cargo test -p eredu-codec --test ear_reference \
  ear_code_to_waveform_forks_retries_failures_and_drains_without_new_codes
cargo test -p eredu-media -p eredu-architectures --features eredu-media/audio \
  --test streaming_audio --test voicechat_frontend
```

Dense TTS RVQ tables and quantized codec codebooks remain distinct parameters.
Their geometry is validated, but their numerical equivalence is not established
by header admission. Full floating payload integrity and released-weight
learned-equation parity are outside the evidence below.

## Neural and streaming component contracts

`voicechat::rnnt::Rnnt` uses the shared `eredu_nn::recurrent::Lstm`, with exact
checkpoint parameter identities and separately declared embedding, recurrent-layer
and joint-projection groups. It accepts one unprojected encoder row per stream;
no perception computation is repeated inside transcription. The pinned streaming
wrapper primes with two zero-input prediction steps. Each emission updates hidden
and cell state; blank retains prediction state. Frame-level blank evidence is the
first decision, even when a nonblank frame later terminates on blank. The final
allowed emission updates prediction state and evaluates the following joint
decision before returning a symbol-limit result.

Explicit frame advancement and token-only prediction use the same equations.
The returned state is unpublished, cloneable and independent of its parent. Frame
and token advancement are separate: injected prediction tokens do not increment
the encoder frame count. A caller supplies hypothesis capacity and per-frame
symbol limits; the pinned per-frame limit is 10. Capacity exhaustion returns a
typed error without changing the parent state. Punctuation bias and conversational
turn policy are not provided by this neural component.

`voicechat::frontend::VoiceChatFrontend` uses the released 1,280-sample input
cadence and a configurable finite PCM history (default 71 frames, minimum three).
The cached wrapper disables dither and feature padding. Host extraction uses a
symmetric 400-sample Hann window within a centered 512-point FFT, pre-emphasis
0.97, 160-sample hops, Slaney filters and additive log floor `2^-24`. Interior FFT
rows are reused; moving the retained window invalidates affected boundary rows.
Input admission has an explicit sample bound and atomic backpressure. PCM rounds
to the declared F32/F16/BF16 representation before feature extraction; BF16 is the
pinned native-wrapper default. The F32 mode isolates preprocessing numerics. Advancement
returns a candidate state; finish explicitly declares whether to pad partial PCM.
`VoiceChatFrontend::from_retained` and `InputState::from_retained` use the exact
saved coefficients without source reads. The explicit-coefficient media
constructor validates geometry and finiteness before creating processing state;
the configuration-only constructor generates the same declared window/filter
geometry for standalone component use.

The default first perception step selects one mel row and drops no encoded rows.
Subsequent steps select nine preceding and eight current mel rows, prefix zeros
when necessary, and drop two subsampled outputs. The finite-window rollover clamp
matches the reference. The optional `pad_and_drop_preencoded` policy changes the
first step to nine zero rows plus eight real rows with two outputs dropped.

`voicechat::perception::Perception` composes three causal subsampling stages,
independently parameterized FastConformer layers and the modality projection.
Causal padding on both convolution axes gives frequency extents
`128 → 65 → 33 → 17`; the output affine flattens channels before frequency.
Each layer retains normalized attention inputs and post-GLU convolution carry.
Relative-position attention, causal masks, layer normalization and half-weight
feed-forward residuals follow the pinned equations. Perception requires float32
parameters and inputs, matching the streaming wrapper. One invocation supplies
both unprojected RNNT features and projected Nemotron features, with candidate
caches returned only after both branches succeed.

The same equations execute through the shared resident and layerwise runtimes.
Subsampling, each Conformer block and the modality projection have separate
parameter leases; no learned perception parameters are pinned as static modules.
The final group's retained forward context captures raw encoder features for
RNNT. Cold prepared bindings restrict source authority to learned perception
inputs and apply the architecture's F32 execution recipes, including F16 storage
conversion. The shared PCM/RNNT and startup-silence compositions accept this
provider-backed execution.

Perception TP2 component checks use the same pinned NVIDIA fixture for full
input and six cached windows. The shared planner selects complete attention
heads, including relative-position projections and biases, and the intermediate
channels of both feed-forward branches. Neutral resident and bounded execution
compare encoder features, modality projections and both streaming caches with
the independent outputs. Wrong communicator geometry and excessive overlap are
rejected without changing the parent cache; a subsequent valid invocation uses
the same retained parent. Cold official-header admission verifies source
retention, local projection recipes and complete replicated convolution storage
without opening parameter payloads.

The native two-process CPU Ring test exercises resident, host-layerwise and
disk-streamed providers with the same output/cache oracle and per-rank parameter
accounting. Subsampling, normalization, convolution and modality projection are
replicated; normalized-input and convolution caches retain full hidden width on
each rank. Comparisons use the fixture's independently fixed absolute `3e-6`
plus relative `3e-5` tolerance, based on the pinned implementation's F32/F64
comparison. No tolerance is fitted to Eredu output.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_perception_parallel_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_perception_tp_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_perception_residency native_perception \
  -- --include-ignored --test-threads=1 --nocapture
```

The Ring launcher requires loopback sockets and bounds
worker runtime. This component evidence does not establish coupled
VoiceChat distributed commitment, accelerator transport, released-weight
conversations or measured real-time latency.

Perception pipeline coverage uses the same pinned NVIDIA component outputs for
cuts after subsampling and before modality projection. The neutral matrix also
cuts between two distinct Conformer blocks and compares with ordinary execution;
the second block's parameters are deterministic transforms of the fixture, so
that internal-cut case is an execution-parity check rather than an independent
NVIDIA two-block oracle. PP2 and TP2/PP2 use the shared scheduler, typed transport,
output publication and exact commit agreement. Full input and six cached windows
exercise subsampling's changing encoded count and overlap removal, with cache
ownership restricted to each selected stage. A provider failure prevents
publication and permits retry from the retained parent. Literal malformed/stale
frontiers leave populated caches unchanged, and stateless stage identities cannot
be exchanged.

Native CPU Ring checks exercise both component boundaries under resident,
host-layerwise and disk-streamed policies, including descriptor rejection and
retry. Output comparisons retain the absolute `3e-6` plus relative `3e-5`
fixture tolerance. Perception and TTS use the same native component-pipeline
driver; the TTS PP2/TP2PP2 check covers that shared completion/commit path.
Pinned-header preparation checks restricted stage sources and missing ingress
without reading weight payloads. The shared Qwen pipeline check covers the
architecture-resolved transport dimension interface for text ingress.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_perception_distributed_ voicechat_perception_partition_rejects_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  released_perception_pipeline_preparation_ released_pipeline_admission_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_perception_residency \
  pipeline::native_perception_pipeline_pp2_and_tp2_pp2_match_nvidia_outputs_and_cache \
  -- --ignored --exact --nocapture
```

Perception TP, TTS pipeline and Qwen pipeline regressions cover the shared
mechanisms. Native Ring workers use
two and four local CPU processes and require loopback sockets.

These local CPU component runs do not establish native accelerator transport,
full coupled distributed VoiceChat sessions, released-checkpoint conversations
or measured speech latency.

The shared runtime pipeline executor also has a three-rank neutral geometry
fixture. Successive cuts carry `[batch, 2*time, 3]` and
`[batch, 5, 3*time]` with distinct auxiliary roles; final publication carries
`[batch, 1, 6*time]`. Two request shapes compare literal output values on every
rank through actual transport, publication and commit. Malformed resolved
schemas fail before receive allocation, and an incorrect final-owner layout
prevents broadcast. Cold validation compares each manifest route against its
independent architecture declaration. Nemotron's publication shape follows the
request's logits or normalized-hidden readout; the numerical boundary fixture
uses different vocabulary and hidden widths.

These checks validate shared transport geometry, not an EAR pipeline selection.
Codec families can implement the runtime contracts without importing language
model architectures. Existing perception and TTS component pipelines and the
Kimi routed pipeline retain their ordinary execution contracts.

```sh
CARGO_INCREMENTAL=0 cargo check -p eredu-architectures -p eredu-codec --tests --all-features
CARGO_INCREMENTAL=0 cargo check -p eredu-backend-mlx --no-default-features --tests
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  pipeline_geometry voicechat_perception_distributed_ voicechat_transformer_distributed_ \
  kimi_observed_pipeline_cut_ nemotron_hidden_readout_crosses_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_tts_transformer_residency \
  pipeline::native_tts_transformer_pipeline_pp2_and_tp2_pp2_preserves_hf_outputs_and_cache \
  -- --ignored --exact --nocapture
```

The native run uses local CPU Ring
transport; the changing-layout fixture uses the neutral numerical backend.

`voicechat::channels::ChannelFusion` reuses the Nemotron token embedding table
for agent and function feedback. It casts projected perception features to the
embedding precision, then computes agent + audio + 2 × function with intermediate
rounding in that precision. Initial feedback uses PAD 12 for both channels.

Tool silence startup uses `PreparedSilenceConditioning`, retained during resident
construction from the admitted preprocessing buffers. Its cold constructor needs
no host FFT feature or native resources. Execution processes 16,000 zero samples
as one full perception input, with fresh caches and no overlap drop, then compacts
the first projected frame and casts it to the selected backbone precision. This
operation does not run RNNT or change live input, conversation clocks or RNG.
`MlxVoiceChatExecution::prepare_silence` uses the shared completion and recovery
owner before returning the tensor for `VoiceChatExecutionInputs::silence`.

`tool-silence-nvidia.json` executes hash-verified NVIDIA preprocessing, causal
subsampling, Conformer and wrapper cache-selection methods at Speech revision
`097dfe9e2f55baf653b83035868bdc89849f1b47`. The compact nonzero learned stack uses
128 mel bins and produces 101 mel rows and 14 projected rows. Its finite one-hot
filter bank is supplied directly to the original preprocessing module; zero PCM
has zero power, so this fixture does not measure filter-bank generation. The
original wrapper selects the first row and casts to BF16. PyTorch 2.13.0 CPU
Float32/Float64 comparisons differ by at most `1.852865189055919e-7`, fixing the
comparison policy before Eredu evaluation at `3e-6 + 3e-5 * abs(reference)`.
The neutral and MLX CPU tests compare Float32 conditioning within that bound and
BF16 conditioning exactly. Neutral rejection cases cover malformed coefficients,
nonfloating output precision and incompatible mel widths. RNNT parameter slots
are deliberately nonexecutable in the neutral integration test.

```sh
python3 eredu-evaluation/scripts/voicechat_tool_silence_reference.py \
  --checkout "$SPEECH_CHECKOUT" \
  --wrapper "$SPEECH_CHECKOUT/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py" \
  --output eredu-evaluation/fixtures/voicechat/tool-silence-nvidia.json
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_startup_silence
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --test voicechat_silence
```

This establishes startup component numerics with synthetic weights. It does not
establish released-weight tool-response conversations, full-graph bounded residency
or native distributed startup parity. The released-checkpoint WAV run above uses
this retained-session entry point and completes startup and conversation; it does
not compare the startup tensors against an independent implementation.
Prompt preparation frames tokens with BOS 1 and EOS 2, repeats the complete
sequence under an explicit capacity, and combines it with both PAD feedback
channels. It does not use the last prompt prediction as first-frame feedback.

The ordinary Nemotron target input accepts those embeddings and can return the
final normalized hidden state. Agent and function projections share that result;
large heads can be materialized independently. Function tokens use greedy argmax.
Target RMS normalization reduces and scales in float32 before casting back,
including BF16 execution. Supplied embeddings, target readout selection and cache
state use existing resident, bounded and partitioned drivers. The version-2 target
boundary carries an exact I32 readout control through pipeline transport.

`PreparedBackboneParameters` projects all 341 backbone parameters onto existing
Nemotron names and retains the same restricted source and conversion accounting.
Its static recipes and 56 layer groups consume admitted canonical recipes, so
split-GGUF `-exp(A_log)` is restored exactly once. This projection does not choose
native materialization or construct the complete speech execution graph.

Training configuration declares text/source delays, but the cached inference
wrapper does not introduce an additional two-frame host text queue from those
fields. Runtime feedback uses policy-adjusted token IDs. No turn policy, forced
response injection, tool serialization or TTS scheduling is supplied by the
channel components.

The shared codec convolution operators support grouped/dilated causal filtering,
arbitrary chunk boundaries, compact carry, explicit final-stride padding and
transposed overlap-add with a declared trim/emit tail. Mimi consumes the same
operators with its released geometry and parameter identities. EAR's ConvNeXt
blocks use the same parameter-free convolution carry, with exact GELU and
channel LayerNorm. Kernel-equals-stride transposed convolutions introduce no
overlap state. Each neural stage and residual codebook is independently visitable.
The spectral decoder retains four complex-equivalent frames at the released
FFT16/hop4 geometry, represented by portable real/imaginary tensors. It performs
periodic-Hann inverse transform, per-frame constraint before windowing and
squared-window overlap normalization. Ordinary decoding and streamed
startup/flush follow their distinct reference edge policies. The composed
decoder proposes both neural and spectral carry atomically and drains without
inventing another code frame.

`voicechat::tts_mog` implements the released residual MLP stack with offset RMS
normalization, tanh-GELU gating and classifier-free guidance after the final
normalization. The conditional lane precedes the unconditional lane. A selected
component projects through its low-rank mean bank; the expanded mean is scaled
by `exp(log_std)` before adding the residual projection. Supplied Gaussian noise
uses the same standard deviation and the explicit noise multiplier. Layers and
large mean banks have separate parameter groups; this component consumes dense
F32 canonical parameters and does not establish full speech residency or
distributed execution.

`voicechat::tts_rvq` implements the released eight-slot cubic masking schedule.
The assigned depths per slot are `[0, 0, 0, 1, 1, 3, 4, 22]`, yielding five
uniform/Gaussian draw pairs per frame. Each active slot quantizes a fresh latent
over its contiguous depth range; existing prefix codes are not subtracted from
that latent. PAD 1024 contributes a zero vector to summed feedback. BOS/EOS
codes are not valid codebook indices. The released configuration disables the
separate binary EOS head; text-channel EOS and silence policy belong to the
coupled streaming driver, which these components do not implement.

`eredu_codec::prvq` supplies the shared nearest-mean and ordered residual
operations. Released tables contain 31 codebooks of 1,024 vectors with 512
channels. Codec encoding includes the residual squared norm in its distances;
TTS refinement omits that constant, matching the respective pinned equations.
Codec decoding sums selected means in depth order. Variance scalars remain
canonical parameters but do not affect nearest-code selection or decoding.

Shared `DenseSamplingBackend` mechanisms use the existing sequential decision
driver's retained random state. They supply F32 uniforms, standard normals and
Gumbel-Max selection, including ascending cumulative top-p filtering with at
least one retained candidate. Mixture decisions do not advance the ordinary
categorical channel ordinal or sampler history. Supplied draws permit comparison
without claiming that MLX and PyTorch generate the same random sequence.
Native argmin and Gumbel selections return exact I32 indices, so partial-code
concatenation preserves its declared representation.

`voicechat::tts_transformer` builds Gemma3 causal and T5Gemma bidirectional blocks
through the shared decoder. Gemma3 uses offset Q/K RMS normalization and distinct
rotary bases for local and global layers. T5Gemma omits Q/K normalization, scales
supplied embeddings by the rounded square root of hidden width, and uses key
padding plus bidirectional local/full masks. Both retain four residual RMS
normalizations, tanh-GELU gating and their declared attention scaling. Causal
execution consumes existing caller-owned cache branches; it does not create a
second session engine. Positional capacity and all layer frontiers are validated
before the first append. Block construction, parameter groups, tensor-parallel
partitioning and final normalization remain individually accessible.

`tts_transformer_layered` supplies the same block equations to resident and
bounded traversal. Each block and final normalization is a separate execution
unit, with no learned static allocation. `TransformerExecution` composes selected
providers with the ordinary TTS generator, speech rendering and coupled transition.
Causal calls borrow the enclosing transaction's cache branch and retain its exact
key/value roots through completion. A late failure can advance that unpublished
branch; the enclosing owner discards it before retry. Character calls carry key
masks and scaled embeddings without creating causal caches. Cold preparation
uses shared supplied-embedding decoder topology and restricted exact recipes,
selecting bounded conversion to ordinary F32 projection weights.

`tts_conditioning` packs valid token characters using retained vocabulary and
checkpoint flag arrays. Mean pooling masks padded characters, then projects and
scatters results before adding continuation and BOS/EOS embeddings. Flags apply
even to unselected rows in a partially masked batch. The caller skips the complete
character module when every subword mask is false. Packing has an explicit cell
limit. `tts_ingress` shares the RVQ feedback projection, implements the prompt
code shift and BOS masks, and selects the fixed Aria latent before the first BOS.
The frozen speaker projection uses `embedding @ W`. Conditional rows precede
null-conditioned rows, and gated fusion divides audio embeddings by 31 before
its two biased projections.

## Evidence and reproducibility

### Composed cached TTS transition

`voicechat::tts_generator` connects character conditioning, prompt ingress,
gated fusion, Gemma and MoG/RVQ execution. Prepared speaker prompts prefill
without sampling; the final unshifted prompt code supplies initial recurrent
feedback. One cached invocation retains both guidance hidden lanes. Partial-frame
advancement and uninterrupted generation then share the same sampled refinement
transition, consuming draws only for the five active slots. The released
configuration has no context projection or binary EOS head.

The [composed
fixture](../eredu-evaluation/fixtures/voicechat/README.md#composed-cached-ear-tts-transition)
executes original NVIDIA forward/generation and
`DuplexEARTTS.infer_codes_one_step` around actual Transformers 4.56.0 Gemma3
with nonzero synthetic weights. It covers five prefill positions and three
cached steps, both guidance KV lanes, all mixture logits and exact
sampled/codebook decisions. Character conditioning is an explicitly supplied
boundary; the separate character fixtures establish those equations. Float32
versus independently widened arithmetic differs by at most `3.42310228e-5`, with
identical decisions. Tolerance is fixed before Eredu evaluation at absolute
`5e-5` plus relative `5e-5`.

`tts-character-transition-nvidia.json` extends the same pinned oracle through
original NVIDIA character mapping, pooling and saved flags around actual T5Gemma
blocks. `SpeechComponents::begin_text_frame` is compared with its cached
ordinary text, EOS and PAD steps, all with true live character masks. The test
checks both guidance hidden/cache lanes, mixture logits, exact RVQ decisions and
effective speech feedback. Character-capacity exhaustion rejects before causal
cache advancement. The independent float32-versus-widened maximum error is
`6.43021943e-5`; tolerance is fixed before Eredu comparison at absolute `1e-4`
plus relative `5e-5`. This is synthetic component evidence, not released-weight
speech or native distributed validation. The script's default mode reproduces
the supplied-conditioning fixture byte for byte.

Neutral coverage compares retained-slot advancement, retry and fork with
uninterrupted frame generation. An injected failure after cache appends mutates
only the caller's unpublished branch; the retained parent remains replayable.
Feedback tests compare source-generated EOS input and elementwise speech-control
replacement, with separate cases for raw-waveform preservation and explicit
per-row PAD/tool-wait substitutions.

`voicechat::speech_sampling` runs refinement on the shared realtime branch RNG.
The neutral test replays the composed fixture's recorded NVIDIA uniform and
Gaussian tensors through actual mixture selection and MoG/RVQ execution. It
reproduces exact partial and final RVQ codes, compares controlled and uninterrupted
codec output, and verifies ten draws across the five active slots. Injected
Gaussian failures, a later-slot failure and malformed draw output preserve the
pre-call RNG and parent frame; native error sources survive the portable error
chain. Missing RNG and submitted branches have typed rejections.
The combined `render_sampled_frame` path also injects a codec failure after all
ten draws. It preserves pre-call RNG, parent codec carry and the partial frame;
retry with valid codec parameters reproduces the separate refinement/render
path's waveform, generated codes and effective feedback. Rendering a completed
frame requires no random stream. This rollback coverage uses the neutral numeric
backend; the native RNG-order test below covers refinement without codec rendering.

The MLX CPU integration test runs agent-channel selection and speech refinement
on the same retained stream. Controlled and uninterrupted execution produce
identical codes and subsequent Gaussian draws, matching an independently
advanced native stream with one agent draw followed by five uniform/Gaussian
pairs. Both configured and zero-noise policies consume those pairs. Refinement
leaves the causal cache frontier unchanged. This checks native stream ordering
and component composition, not PyTorch/MLX random-stream equality or complete
session publication.

Coordinated refinement uses the same generation loop and branch RNG transaction.
The neutral test compares every active code prefix with the recorded NVIDIA
fixture and checks controlled/uninterrupted replay with a follower that has no
RNG. Identity, operation-mode and slot mismatches reject work before prediction.
Missing owner RNG, owner Gaussian failure, and a peer failure after three sampled
slots preserve pre-call RNG and the parent frame without communication timeouts.
Consumed authority rejects reuse. The native two-process CPU Ring test selects
rank one as owner, compares exact codes and subsequent Gaussian draws
with ordinary execution, and supplies fresh recorded identity for each controlled
advance. This exercises retained bounded agreement and tensor publication, not
the complete distributed conversation loader or released checkpoint weights.

Coordinated rendering checks an output-policy mismatch after all ten draws and
before either rank enters the codec. A malformed peer codebook then exercises
late rendering failure: both ranks discard the candidate and the owner retains
its pre-call RNG. A fresh retry matches ordinary waveform, feedback and retained
neural/spectral state. Parent codec and causal-cache frontiers remain unchanged.

Comparisons use exact
integer codes and same-backend replay equality; the existing NVIDIA fixture
tolerances govern independent numerical comparisons.

```sh
cargo test -p eredu-architectures --test reference_numeric -- voicechat_tts_generator voicechat_tts_feedback voicechat_mog
cargo test -p eredu-architectures --test reference_numeric voicechat_spoken_text
cargo test -p eredu-architectures --test reference_numeric voicechat_speech_sampling
cargo test -p eredu-architectures --test reference_numeric voicechat_speech_coordination
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_speech_sampling
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_speech_sampling coordination::native_speech_sampling_owner_draws_match_controlled_and_uninterrupted -- --ignored --exact
cargo test -p eredu-backend-mlx --no-default-features --lib cpu_voicechat_tts_generator_matches_original_cached_refinement
cargo check -p eredu --no-default-features
```

This component accepts prepared prompt codes and caller-owned attention branches.
The fixed Aria latent does not eliminate the reference's silent-carrier codec
encoding: the final prompt BOS position still consumes its shifted encoded code.
`tts_prompt` prepares those codes through the finite codec encoder and applies
the released prompt masks. Its retained frame is not a complete conversational snapshot;
released-weight parity and native distributed composed-TTS validation are absent.

### EAR code-to-waveform decoder

The [codec
fixture](../eredu-evaluation/fixtures/voicechat/README.md#ear-neural-decoder-and-streaming-spectral-reconstruction)
executes hash-verified NVIDIA `Latent2Wav`, `ConvNeXt1d`, `LayerNormNd`,
`CausalConv1dCache`, `spec_to_wav` and probabilistic-codebook decode equations.
Nonzero synthetic parameters cover a compact decoder and a small-width decoder
with the released rates `[9,7,7]`, FFT 16 and hop four. Neutral and MLX CPU
tests compare raw spectra, ordinary and chunked PCM, every convolution cache and
split complex spectral history. Direct spectra exercise constrained and
unconstrained inverse transforms. Code-to-waveform comparisons include state
forks, invalid input or injected computation failure, retry, empty-stream
finish, terminal rejection and idempotent drain without a new code frame.

The layered decoder tests compare ordinary, resident-provider and bounded
execution against those same nonzero spectra, convolution carries and waveform
fixtures. A late unit-acquisition failure occurs after earlier candidate caches
advance; the parent remains reusable and retry matches the reference. Waveform
tests cover replay, fork, startup overlap and idempotent drain, and verify that
finish acquires no learned parameters. The waveform test combines independently
bounded codebook and neural decoder providers.

The MLX CPU decoder residency test uses F32 and F16 SafeTensors inputs under
resident, host-layerwise and disk-streamed policies. It compares raw spectra,
ordinary and chunked PCM, retained state and final drain across policies. F32
also uses the pinned fixture tolerance below; F16 checks execution parity at
identical stored precision. Cold preparation reads no payload, excludes unrelated
and quantizer tensors, and accounts for execution conversions. Bounded runs check
the unit/window budget and confirm that drain performs no parameter reads.
Shared dense prefill/decode telemetry classifies physical axis one, which is
channels for this decoder; only their combined invocation count measures codec
passes. These counters do not represent its audio frontier.


EAR decoder tensor-parallel checks reuse the pinned nonzero compact decoder's
spectra and every retained convolution carry. Neutral resident and bounded
providers compare complete and cached execution, parent replay, late acquisition
failure and retry. Wrong communicator, ordinary use of local units, malformed
ingress and finished states are rejected. A separate literal oracle uses two
convolution groups with three output channels each, split 2/1 across ranks. Its
distinct group/time kernels, identity output projection and nonzero ConvNeXt
output bias exercise grouped channel reconstruction and exactly one post-reduction
bias addition over two cached steps.

Native MLX CPU Ring coverage composes the partitioned PRVQ and neural decoder
through the existing waveform driver. Resident, host-layerwise and disk-streamed
F32 providers compare raw spectra, carried state, startup PCM and final drain
with the pinned NVIDIA outputs. A focused F16 disk case compares ordinary
execution over identical stored precision. Replay preserves the parent; drain
performs no learned reads. The compact decoder uses 1,240 parameter bytes per
rank, including 336 bytes of replicated depthwise, normalization and output-bias
parameters. Its quantizer uses 132 bytes per rank. Bounded peaks and rejection
before acquisition are checked through the shared provider reports.

Cold pinned-header preparation verifies upsample, MLP and spectral shards,
coherent projection coordinates, exact retained sources and F32 conversion
costs. Released decoder parameter replicas occupy 354,816 bytes per rank. The
full depthwise carry is also replicated: 193,536 F32 data bytes per batch row
once all nine convolution caches are populated. These are component storage
figures, not measured session peak residency.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_decoder_tp_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_codec_decoder
CARGO_INCREMENTAL=0 cargo test -p eredu-codec --test ear_reference --test ear_encoder_reference -- \
  ear_assembly ear_decoder_preparation ear_encoder_matches
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_decoder_residency native_ \
  -- --include-ignored --test-threads=1 --nocapture
```

Native Ring validation requires loopback sockets. These component
checks do not establish full coupled distributed VoiceChat, accelerator transport,
released-weight conversation parity or measured streaming latency.

Streaming neural decoder pipeline checks use the same pinned NVIDIA spectra and
convolution-cache oracle through PP2 and combined TP2/PP2. Cuts on either side of
an upsampler exercise changing channel width and frame expansion. Resident and
bounded neutral runs compare all four cached steps, local owned carry, parent
replay and exact commits. Published spectra pass through the existing spectral
decoder and match startup PCM, retained spectral history and final flush. A late
execution failure produces no publication or commit, preserves the parent and
permits retry. Malformed boundary metadata and foreign stage state reject before
parameter acquisition. Header-only preparation verifies the released decoder's
restricted stage sources, retained tensor placement and exact local cache slice;
a stage cannot retain caches for upstream units.

Native MLX CPU Ring checks cover PP2 and TP2/PP2 with resident, host-layerwise and
disk-streamed F32 providers at the first cut, plus resident execution at the
second cut. Ordinary and cached spectra match the NVIDIA fixture; stage-local
carry matches each non-flushing reference step. Replay preserves the parent and
reproduces spectra and carry. Finish clears neural carry without parameter reads
and emits no spectral tail. Cold admission rejects inconsistent route cuts and
different unit ownership within a tensor cohort. Descriptor disagreement rejects
before learned reads. Parameter residency charges only the selected units;
convolution carry is replicated only within its owning stage's tensor cohort.
These checks use the independently fixed absolute `2e-5` plus relative `5e-5`
tolerance below. Spectral reconstruction remains a separate component and these
runs do not select a complete distributed VoiceChat session.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_decoder_pipeline_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_codec_decoder voicechat_codec_encoder --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_decoder_residency --test voicechat_codec_encoder_residency -- \
  --include-ignored --test-threads=1 --nocapture \
  native_codec_decoder_residency_ native_ear_decoder_pipeline_pp2_ native_ear_encoder_pipeline_pp2_
```

Native Ring uses local CPU ranks and loopback sockets;
accelerator transport and released-weight waveform parity remain unvalidated.

Whole-waveform pipeline coverage composes the selected codebook provider, neural
pipeline and shared spectral kernel. The neutral two-participant test rejects
different code values, different spectral geometry and incompatible neural
lifecycle before learned reads. Asymmetric codebook and spectral failures leave
the parent unchanged; retries use fresh authority and match ordinary waveform
execution. Step/drain disagreement, final overlap and idempotent finish exercise
the same agreement protocol. A separate coupled-executor check uses a selected
codec-state type, injects failures after actual rendering and drain, and compares
controlled and automatic commitment with unchanged parent state on failure.

The native waveform matrix uses the same nonzero NVIDIA code-to-waveform fixture
and unchanged absolute `2e-5` plus relative `5e-5` bounds. PP2 and combined TP2/PP2
run resident, host-layerwise and disk-streamed providers, with cuts on either
side of an upsampler. Comparisons include startup PCM, owned convolution carry,
spectral history, snapshot replay with fresh authority and parameter-free drain.
Malformed depth and incompatible retained batch reject before learned reads.
The combined TP2/PP2 disk-streamed case partitions codebooks within each decoder
stage's tensor cohort; the other waveform cases use replicated codebooks.
Codebook and neural-decoder residency allowances are independent; their reports
do not claim an aggregate session peak. These compact component fixtures do not
establish released-weight conversation parity or full distributed session selection.

Guarded TP codebook coverage rejects differing invocation modes before acquisition
and agrees asymmetric acquisition and malformed-codebook failures before data
reductions. Repaired invocations reproduce the independent quantization and
decode fixture. The native TP2 matrix covers quantization, selected ranges and
all three residency modes. Guarded native reductions complete before subsequent
control agreement, preserving data/status ordering on the shared Ring transport.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_execution voicechat_codec_decoder waveform_pipeline
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  waveform_pipeline voicechat_codec_quantizer voicechat_rvq_resident_and_bounded_feedback
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_decoder_residency --test voicechat_codec_quantizer_residency -- \
  --include-ignored --nocapture \
  native_ear_waveform_pp2_tp2pp2_preserves_startup_replay_and_drain \
  native_prvq_tp2_residency_matches_nvidia_quantization_and_ranges
```

Both native matrices use local MLX CPU Ring ranks;
these results do not validate accelerator transport.

The released-rate fixture emits 1,764 samples per latent frame, with eight samples
of startup delay and eight final drain samples. Its aligned offline and streaming
interiors agree within `4.20e-9`; endpoint normalization differs by up to `0.00520`
in the original source. Both edge policies are independently recorded.
Tolerance is absolute `2e-5` plus relative `5e-5`, fixed from independent
Float32/Float64 comparisons before Eredu output assessment: neural maximum
difference `2.84e-7`, reconstructed-waveform difference `9.46e-8`, and widened
inverse-transform difference `7.15e-8` for direct spectra.

```sh
cargo test -p eredu-codec --test ear_reference
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_codec
cargo test -p eredu-architectures --test reference_numeric voicechat_codec_decoder
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_codec_decoder_residency
cargo test -p eredu-codec --test streaming_convolution
cargo test -p eredu-codec --test mimi_artifact_conformance constructs_real_mimi_and_runs_numeric_latent_pcm_and_streaming_paths
cargo test -p eredu-architectures --lib nested_release_resolves_numerically_material_defaults
cargo check -p eredu --no-default-features
```

Released spectral geometry retains 1,088 bytes of F32 window/routing constants
and 288 bytes of spectral carry per batch row. Convolution carry, materialized
weights, parameter-layout conversion, output/overlap temporaries and native FFT
scratch are separate costs; these figures do not establish peak residency.
Complete speech-graph residency/distributed construction and released-weight
waveform parity have no validation evidence here. Accelerator pipeline transport
is unvalidated. Tiny equation fixtures do not
establish speech quality, hardware latency or a complete VoiceChat session.

### Finite EAR encoding and fixed-Aria preparation

The [encoder fixture](../eredu-evaluation/fixtures/voicechat/README.md#ear-waveform-encoder)
executes NVIDIA's hash-verified STFT, Wav2Latent, ConvNeXt and residual quantizer
with nonzero synthetic parameters. Neutral and MLX CPU tests compare real and
imaginary spectra, individual neural stages, final latents and exact codes for
silence and nonzero PCM. A narrow released-rate case uses `[7,7,9]`, FFT 16 and
hop four, retaining the 1,764-sample cadence. Direct spectral cases cover odd
and even FFT sizes and asymmetric zero padding. The original encoder rejects
streaming caches; this component accepts complete finite waveforms.

Finite encoder residency tests compare the same pinned silent and nonzero PCM
rows through ordinary, resident-provider and bounded traversal. The composed
encoder borrows a bounded PRVQ provider and reproduces reference latents and
codes. Single-row calls in reversed order verify that finite execution retains
no convolution carry. A late encoder acquisition failure leaves input unchanged,
starts no quantizer work and permits a reference-matching retry. Malformed PCM
is rejected before unit acquisition. The frame-major output transpose runs in
the final learned unit, within its completion lifetime.

MLX CPU tests use F32/F16 SafeTensors sources under resident, host-layerwise and
disk-streamed policies, with independently budgeted encoder and PRVQ providers.
F32 latents and codes use the NVIDIA oracle; F16 compares identical stored
precision across policies. Preparation rejects missing or mismatched projections
without payload reads, restricts source authority to encoder tensors and charges
F32 conversion. Bounded runs verify unit/window device budgets, and malformed
input adds no parameter reads before a successful retry. Dense telemetry counts
finite calls through its combined prefill/decode total; the encoder uses
channel-first activations rather than text positions. These are component
residency checks, not a complete speech-graph schedule or measured session peak
memory.

Tolerance is absolute `2e-5` plus relative `5e-5`, fixed before Eredu comparison.
Independent Float32/Float64 runs differ by at most `3.93e-7` in neural outputs
and `1.85e-7` in direct spectra, with identical quantizer decisions. These are
equation fixtures, not released-weight encoder parity or peak-memory measurements.

Finite encoder tensor-parallel checks use the same pinned silence/nonzero batch
through resident and bounded providers. They compare latents and exact borrowed
quantizer codes, reject wrong communicators and ordinary shard execution before
acquisition, and inject a late encoder acquisition failure before quantizer work.
Repeated finite calls retain no convolution carry. An independent literal-value
case uses two convolution groups with three output rows each, split unevenly
across two ranks; distinct filters and residual biases check segmented row
ordering, frame-major output order and single bias addition.

Native MLX CPU Ring checks compose the partitioned finite encoder with a
partitioned PRVQ under resident, host-layerwise and disk-streamed F32 policies.
The pinned NVIDIA fixture supplies the F32 oracle; F16 disk streaming compares
against ordinary execution at identical stored precision. A separate grouped
checkpoint compares native segmented materialization against its ordinary
execution, without claiming an independent NVIDIA grouped reference. The compact
encoder charges 1,240 F32 parameter bytes per rank (1,176 in that grouped case),
including 336 replicated bytes; its borrowed three-depth quantizer charges 132
bytes. Header-only admission checks the released encoder's exact retained source,
all local binding geometry, conversion bounds and 354,816 replicated F32 bytes
per rank, with no payload reads.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_encoder_tp_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_codec_encoder
CARGO_INCREMENTAL=0 cargo test -p eredu-codec --test ear_encoder_reference --test ear_reference -- \
  ear_encoder_matches ear_neural_spectral_streams
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_encoder_residency native_ -- --include-ignored --test-threads=1 --nocapture
```

Finite encoder pipeline checks use the pinned packed spectrum and latent oracle
through PP2 and combined TP2/PP2. Cuts before and after downsampling carry
different channel widths and frame counts. Resident and bounded neutral runs
compare repeated finite calls, inject a completed-stage failure before publication
and verify a reference-matching retry with exact commit. Malformed cut, request
geometry, metadata dtype and activation shape reject before parameter acquisition.
Header-only preparation retains precisely the selected stage's recipes and source
keys, including tensor placement, and rejects mutable state ownership. The shared
runtime admits an empty state plan only for an entirely `NoState` layout.

Native MLX CPU Ring checks cover PP2 and TP2/PP2 with resident, host-layerwise and
disk-streamed F32 providers. The first cut uses all three policies; the second cut
checks the changed downsampling boundary under resident execution. Each rank
charges only its selected unit bindings, including replicated small parameters.
Cold preparation rejects inconsistent cuts and different unit ownership within a
tensor collective without reading payloads. Two finite committed calls match the
NVIDIA latent oracle; mismatched recorded request descriptors reject before disk
reads and permit retry. Spectral preprocessing and subsequent quantization remain
separate from this finite neural pipeline.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib empty_state_plan_admits_only_stateless_architectures
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_encoder_pipeline_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_codec_encoder
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_encoder_residency -- --include-ignored --test-threads=1 --nocapture \
  native_codec_encoder_residency_ native_ear_encoder_pipeline_pp2_
```

Pipeline comparisons use the same independently fixed absolute `2e-5` plus
relative `5e-5` tolerance. Native Ring validation requires loopback sockets.
These component checks do not validate a coupled distributed waveform or speech
session, accelerator transport, released-weight parity or real-time latency.

The [Aria
fixture](../eredu-evaluation/fixtures/voicechat/README.md#fixed-aria-prompt-preparation)
executes the original initialization and batch-expansion methods with supplied
encoded codes and a tiny supplied speaker table. It records the actual zero
carrier request and exact integer/boolean prompt tensors. The released geometry
encodes 67,032 samples (38 frames), retains 37 positions and expands the
prepared single row without re-encoding. Neutral tests compare PAD overrides,
subword IDs and masks, initial feedback and batch expansion, and exercise typed
carrier and prompt-position capacity failures. This fixture isolates prompt
policy from the separately validated encoder; it does not execute a complete
speech session.

```sh
cargo test -p eredu-codec --test ear_encoder_reference --test ear_reference
cargo test -p eredu-architectures --test reference_numeric voicechat_codec_encoder
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_codec_encoder_residency
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_codec_encoder --test voicechat_codec
cargo test -p eredu-architectures --test reference_numeric voicechat_aria_prompt_matches_original_initialization_and_batch_expansion
cargo check -p eredu --no-default-features
```

### TTS transformers, character conditioning and ingress

The [transformer fixture](../eredu-evaluation/fixtures/voicechat/README.md)
executes actual Transformers 4.56.0 `Gemma3TextModel` and `T5GemmaEncoderModel`
modules with tiny nonzero parameters and hash-verified source files. Local/global
layers, GQA, Q/K gains, distinct rotary bases, bidirectional padding and the
encoder's input scaling are exercised. A six-position whole Gemma pass is
compared with prefill two and cached chunks of one, two and one positions,
crossing a three-position local window. Tests compare outputs and retained K/V
values, not just final token decisions.

Absolute `2e-5` plus relative `5e-5` transformer tolerance is fixed from independent
precision evidence: wider linear arithmetic with the source's F32 normalization
and rotary computations differs by at most `2.50339508e-6`; whole versus cached
reference output differs by at most `1.90734863e-6`. Separate original NVIDIA AST
fixtures cover character lookup/pooling/scatter/flags, gated fusion and prompt
ingress. Their character or backbone stand-ins are explicitly disclosed; they
isolate boundaries and do not replace the actual transformer fixture. Character
tolerance is absolute `3e-6` plus relative `3e-5` (widening delta `1.49011612e-7`);
ingress tolerance is absolute `2e-5` plus relative `5e-5` (delta `3.57627869e-7`).

```sh
cargo test -p eredu-architectures --test reference_numeric voicechat_t
cargo test -p eredu-architectures --test reference_numeric voicechat_conditioning
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_tts_ingress --test voicechat_conditioning
cargo test -p eredu-backend-mlx --no-default-features --lib cpu_voicechat_transformers_match_pinned_hf_cached_and_bidirectional_outputs
cargo test -p eredu-architectures --test reference_numeric -- gemma2_matches_scalar_oracle_across_windows_chunked_prefill_and_cached_decode gemma4_tp2_text_matches_replicated_composite_graph
cargo check -p eredu --no-default-features
```

Neutral transformer coverage includes TP2 through the existing parameter
partition and collective mechanisms, plus positional-capacity rejection without
cache mutation. The single fixture KV head is duplicated into equivalent logical
heads for ordinary TP sharding; each rank's retained values still compare with
the independent HF oracle. This does not establish a complete speech graph's
resident/host/disk, pipeline-parallel or native distributed behavior. Aria
conditioning uses a synthetic time-aligned latent, not the released payload.
Released-checkpoint speech parity and waveform measurements remain outside this
component evidence.

The layered transformer tests reuse these independent fixtures for resident and
bounded execution. They compare causal prefill and cached K/V against the pinned
HF outputs, masked bidirectional output, and both guidance lanes against the
NVIDIA TTS transition fixture. A final-normalization acquisition failure leaves
the canonical parent untouched while advancing only the supplied unpublished
cache branch; discarding that branch and retrying reproduces the reference.
The generator test performs one prompt prefill and two cached invocations through
the same bounded provider and skips character acquisition for supplied conditioning.

The MLX CPU residency test covers both transformer stacks, F32/F16 SafeTensors
storage and resident/host/disk policies. F32 output uses the fixed independent
HF tolerance above; output and retained-cache equality across residency is
checked separately for each storage precision. Cold preparation reads no tensor
payloads and excludes neighboring component keys. Converted storage is charged
at F32, static leases are zero, and host device residency is limited to the
largest individual block or normalization unit. The disk budget covers its
declared two-unit transfer window within a group; the separate normalization
group shares that budget. The disk policy disables host caching and host
lookahead. Causal tests fail during the second block, discard the candidate,
then reuse the provider for full and cached execution. The tests reuse
synthetic-weight fixtures; no released-weight or native distributed result is
implied.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric -- voicechat_transformer voicechat_tts_generator voicechat_gemma_tp2
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_realtime_construction voicechat_construction_covers_released_learned_parameters_once_by_unit
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_tts_transformer_residency
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio
```

The complete causal and character transformer stacks also execute with tensor
parallelism through `LayerwiseRuntime::forward_parallel_with_state`. The
architecture derives local projection widths and KV-cache geometry from retained
physical layouts. Prepared bindings use the shared placement planner and bounded
recipe selections; the native adapter derives selection from its communicator's
rank and size. Normalization parameters remain replicated and their storage is
included in each rank's bindings. Shared attention/MLP reductions precede the
ordinary residual updates, including the Gemma3 post-projection normalizations.
Ordinary and parallel calls share masks, rotary policy, character scaling and
unit completion handling.

Two neutral TP2 tests compare the pinned HF nonzero fixture's cached causal
outputs, retained K/V and bidirectional character masks under resident and bounded
providers. They exercise explicit Q2/KV1 replication and standard KV-group
sharding using duplicated identical KV heads. The duplication preserves the
independent oracle equation; it does not substitute a zero-valued fixture. The
released stacks have Q16/KV16. Automatic selection partitions KV groups;
automatic replication when TP exceeds KV cardinality is not implemented. The
low-level selected-layout contract can express that replication explicitly.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_transformer_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_transformer_tp_preparation_retains_source_and_partitions_recipes
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_tts_transformer_residency \
  native_tts_transformer_tp2_preserves_hf_outputs_masks_and_cached_state \
  -- --ignored --exact --nocapture
```

The opt-in native test uses two local CPU Ring processes with a finite timeout.
It exercises exact SafeTensors slicing, resident, host-layerwise and disk-streamed providers,
causal prefill/cached outputs and local K/V against ordinary native execution,
plus independent HF output and character-mask comparisons. The cache comparison
and output comparisons use the fixture's independently fixed absolute tolerance
`2e-5` plus relative tolerance `5e-5`. A separate cold preparation test uses pinned
official headers to check local projection sizes, F32 storage, replicated norms,
exact source retention and single application of partitioning without payload reads.
This component route does not select coupled distributed VoiceChat sessions or
pipeline boundaries, does not choose random outputs on individual ranks, and
does not establish released-weight speech or hardware-latency measurements.

Native TP2 CPU Ring coverage runs both stacks under
resident, host-layerwise and disk-streamed policies on two local processes.
Loopback socket access is required for that opt-in test. This is component-level
native distributed evidence; full coupled VoiceChat TP/PP, accelerator transport
and released-checkpoint conversations remain unvalidated.

TTS pipeline component checks use separate block/normalization stages, a block
suffix with normalization, and a multi-block stage with separate normalization.
`LayerwisePartitionRuntime` handles the complete local stage through
`LayeredPartitionDriver` and the existing `LayerwisePolicyForward` leases. Each
boundary round-trips through the architecture's typed wire schema, carrying the
original causal position or character key visibility. The checks exercise
PP2/PP3 and combined tensor/pipeline partitions, compare outputs and retained
K/V with the pinned HF fixture under its existing tolerances, and reject a
mismatched causal frontier or malformed character mask before cache mutation.
Character scaling occurs once at ingress; normalization has no cache slots. A
late normalization acquisition failure aborts the provider, preserves the
canonical parent cache and permits retry with a fresh candidate branch.

Cold pinned-header checks retain canonical stage addresses and restrict prepared
sources to the selected recipes. They verify that a middle block cannot access
the normalization recipe and that a normalization-only stage has no attention
cache layout and only its 4,608-byte F32 parameter. Payload access panics in the
header fixture.

```sh
CARGO_INCREMENTAL=0 cargo check -p eredu-runtime
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_transformer_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_transformer_
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib selected_unit_layout_preserves_canonical_addresses_and_local_storage
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_tts_transformer_residency
```

Neutral component partition checks use in-process tensor collectives and
explicitly passed pipeline boundaries. Native MLX tests use the same independent
HF fixture for both stacks under resident, host-layerwise and disk-streamed
providers. They exercise nonzero canonical block addresses, compound and
standalone normalization stages, causal cache parity and selected-unit residency
bounds. The opt-in two-process CPU Ring check also runs these stage cuts with
tensor-parallel projections. Stage boundaries are passed locally within each
tensor rank; this does not establish cross-process pipeline transport,
full-session TP/PP commitment or released-checkpoint parity. Native comparisons retain the
fixture's absolute `2e-5` plus relative `5e-5` tolerance. The Ring test requires
loopback socket access; no accelerator transport or latency measurement is implied.

Cross-process transformer component coverage uses the existing distributed graph
driver, typed pipeline transport, output publication and exact commit agreement.
PP2 places all blocks before a separate normalization rank; TP2/PP2 cuts between
blocks and gives the second stage its block suffix plus normalization. Both
causal and bidirectional stacks use the independent nonzero HF fixture above.
The neutral test compares outputs and retained cache state, rejects a provider
failure before output publication, and retries from the retained parent state.
Native CPU Ring coverage exercises resident, host-layerwise and disk-streamed
providers, local cache parity, residency limits, and invocation-descriptor
rejection followed by retry. Commit agreement compares exact integer epoch IDs;
the enclosing scheduler owns their allocation.
The two- and four-process native checks use the independent absolute `2e-5`
plus relative `5e-5` fixture tolerance. They require loopback
socket access and run on local CPU Ring transport with finite worker timeouts.

Selected causal execution also runs through the ordinary TTS generator. Its
native PP2 and TP2/PP2 test uses the independent NVIDIA TTS transition fixture
for prompt fusion, three cached frames, both guidance lanes and every active
RVQ refinement slot with identical supplied decisions and Gaussian draws.
Character embeddings are supplied at the fixture boundary; conditioning, MoG
and RVQ providers are replicated. The causal transformer runs resident,
host-layerwise and disk-streamed policies with the same selected-unit residency
bounds as its component checks. Comparisons use the transition fixture's existing
absolute `5e-5` plus relative `5e-5` bounds, including hidden output, local caches,
mixture logits and exact generated codes. Asymmetric invocation identities and
spent causal authority reject before cache mutation.

Neutral partition and distributed tests retain the logical position on every
stage, including normalization-only stages. They reject a changed boundary
position, preserve the position when independently copying local caches, and
compare retry output and unchanged parent state. The shared speech and coupled
transaction checks cover ordinary prefill, cached execution, rendering and drain
through the selected-state contract. They do not establish full conversational
preparation agreement, coordinator-owned speech sampling, accelerator transport
or released-weight conversation parity.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_tts_generator voicechat_transformer voicechat_speech \
  voicechat_execution voicechat_backbone_composition
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_tts_transformer_residency -- --include-ignored --nocapture \
  native_tts_generator_pipeline_pp2_tp2pp2_matches_cached_refinement \
  native_tts_transformer_pipeline_pp2_and_tp2_pp2_preserves_hf_outputs_and_cache
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_tts_transformer_residency \
  partition::native_tts_partition_stages_preserve_hf_outputs_caches_and_residency \
  -- --exact --nocapture
```

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_transformer_distributed_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_pipeline_admission_
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --test backend_independence final_agreement_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_tts_transformer_residency \
  pipeline::native_tts_transformer_pipeline_pp2_and_tp2_pp2_preserves_hf_outputs_and_cache \
  -- --ignored --exact --nocapture
```

Cold admission uses pinned official headers and rejects missing stage ingress or
publication with the wrong activation precision without reading weight payloads.
The shared text-pipeline regression checks nonzero embedding ingress,
visibility, prefill and cached execution using
`qwen4_pipeline_executor_preserves_embedding_input_ids_and_visibility`. The
epoch test distinguishes adjacent IDs above `2^60`, rejects disagreement, and
accepts a matching retry; the completion test retains the existing
indeterminate-failure and poison checks. Component pipeline coverage does not
establish full coupled VoiceChat TP/PP, released-weight conversations,
accelerator transport or measured speech latency.

Learned-conditioning residency reuses the independent character and ingress
fixtures. Neutral tests compare character lookup, pooling/flags, fusion,
fixed-Aria and projected-code prompts, cached decode and null guidance across
ordinary, resident-provider and bounded execution. Empty character selections
skip both learned phases and their transformer callback. The composed generator
test selects bounded providers for conditioning, both transformer stacks, MoG
and RVQ together. It compares one prompt and two cached frames, retained K/V,
mixture logits and generated codes with the pinned TTS transition. A fusion
acquisition failure after completed ingress leaves the causal cache unchanged;
retry uses the same provider and matches the reference.
That composed fixture supplies subword conditioning and verifies that character
providers remain idle; the separate character fixture exercises their learned phases.

The MLX CPU conditioning test covers F32/F16 storage under resident,
host-layerwise and disk-streamed policies. F32 uses the independent fixture
tolerances above; F16 output equality is checked between residency policies
separately. Exact cold binding excludes neighboring RVQ tensors and reads no
payloads, with converted F32 storage charged explicitly. Each conditioning group
contains one unit, so bounded device admission uses the largest unit without
pinned learned static leases. Disk telemetry checks that ingress/fusion groups
remain unmaterialized during character execution. Feedback uses a separate RVQ
provider with the same residency policy and completes before ingress acquisition.

These tests use
synthetic nonzero weights and do not establish released-checkpoint conversation
parity, a complete speech-graph residency schedule or native distributed execution.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric -- voicechat_conditioning voicechat_character_mapping_pooling_flags_and_fusion_match_nvidia voicechat_tts_generator voicechat_tts_ingress voicechat_speech_sampling voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_realtime_construction voicechat_construction_covers_released_learned_parameters_once_by_unit
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_conditioning_residency --test voicechat_speech_sampling
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio
```

Conditioning tensor-parallel component checks use the same pinned NVIDIA
character, fusion and ingress outputs with the unchanged tolerances above.
Neutral resident and bounded providers exercise replicated flag additions, Aria
and projected-code prompts, decode guidance, communicator preflight and
acquisition retry. A construction regression rejects each shard unit when
repackaged into ordinary resident execution. A separate scalar oracle uses
character width three and speech width five, with uneven TP2 slices, biased
projections and a global RMS normalization; its expected embeddings, pooled
projections and fusion values are independent literals.

The MLX CPU Ring matrix uses two processes with resident, host-layerwise and
disk-streamed policies. F32 compares against NVIDIA outputs; a focused F16 disk
case compares ordinary execution over identical stored precision. It checks
rank-local parameter bytes including replicated controls, bounded residency,
wrong-rank source rejection and invalid ingress before payload acquisition.
Pinned-header selection verifies retained source authority, matrix/bias slice
coherence and F32 conversion overhead. The full per-rank control replicas occupy
211,972 bytes. Released speaker projection uses 1,152 bounded row spans per rank;
its loading latency is unmeasured.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_conditioning_tp_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_conditioning voicechat_character_mapping_pooling_flags_and_fusion_match_nvidia voicechat_tts_ingress
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_conditioning_residency native_conditioning \
  -- --include-ignored --test-threads=1 --nocapture
```

Native Ring
validation requires loopback sockets. These component checks do not establish
full coupled distributed sessions, released-weight conversation parity,
accelerator transport or streaming latency.

### Speech-provider preparation and assembly

`PreparedSpeechProviders` prepares eleven independent speech execution providers,
including RNNT, from retained sources. Its cold census test uses the
pinned official SafeTensors header with a source that rejects all payload reads.
Summed converted provider bytes match the independent catalog's learned
speech parameters, excluding backbone, control codes and host preprocessing.
The function head contributes exactly 2,348,810,240 bytes; RNNT contributes
35,772,420 bytes. Layered stacks have smaller acquisition units than their complete
parameter sets. RNNT retains exactly its 15 bindings and cannot access the function
head; changed recipe geometry is rejected.

Residency tests exercise resident, finite host-layerwise and disk-streamed policies.
They check summed independent device/host allowances, bounded RNNT, conservative
foreground/background conversion reserves, insufficient aggregate capacities,
unbounded pools and arithmetic overflow. These are logical parameter-pool bounds,
not measurements of whole-session or unified physical memory. Backend host alignment
and multi-unit window admission remain separate shared-materializer checks.

The ordinary released-shape assembly census runs through the consuming provider
factory, preserving each bound parameter once. A deferred provider with a
mismatched function vocabulary is rejected without invocation. Assembly tests
cover spectral mechanism rejection and binding-failure short-circuiting. A
complete native aggregate load, distributed run and released-weight conversation
are not validated by these header/shape fixtures.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  voicechat::provider_residency released_provider_census rnnt_retains_exact selected_target_coexists
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_realtime_construction voicechat
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio --example voicechat_wav
```

Complete coupled parameter admission also accounts for the selected Nemotron
backbone provider and its pinned static weights. A payload-rejecting
pinned-header source verifies 35,552,909,312 parameter bytes and a conservative
9,395,276,800-byte largest-unit materialization reserve. The pinned static
payload is 4,697,638,400 bytes, the blocks total 30,855,270,912 bytes, and the
largest block is 589,802,496 bytes. A one-block device window therefore requires
at least 5,287,440,896 logical bytes before native alignment. Disk streaming
uses the shared two-block transfer window, requiring 5,877,243,392 bytes for
this catalog. Cold admission checks the largest consecutive window for the
selected policy. Both vocabulary matrices are included. Tied readout cannot
retain a separate head; its single embedding owner is charged once. Total
admission tests combine the selected backbone allocation with independent speech
pools and loading reserves, reject one-byte insufficient host/device limits, and
reject arithmetic overflow. This is a logical parameter/loading bound, not a
physical whole-session peak measurement.

The native CPU backbone residency fixture compares the ordinary resident runtime
with selected resident, host-layerwise and disk-streamed providers from the same
small SafeTensors file. It checks exact F32 equality for normalized hidden
states, both output heads, every retained hybrid-state tensor and cache position
after prefill and two cached audio steps. It also checks source restrictions,
rejects missing or malformed readout recipes before payload reads, and verifies
selected device windows and retained leases. Host loading uses one block; disk
streaming uses the shared two-block transfer window. These comparisons establish
mechanism parity for identical compact weights, without measuring released
speech quality. Native distributed and released-checkpoint conversation parity
are not established for this selected aggregate.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- voicechat::component_residency voicechat::backbone::tests
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_backbone_residency
```

The selected native components use the existing submission owner. The coupled CPU
fixture compares a bounded host-layerwise function head during controlled
snapshot/replay/fork with its resident provider during scheduler-driven execution,
checking outputs, retained state and RNG. Its head pool peaks at 384 logical device
bytes. This exercises selected-provider acquisition inside the ordinary internal
submission driver. Complete released-weight aggregate construction has the WAV
evidence above; this fixture establishes neither that construction nor independent
released-weight conversation parity. Failure and unwind tests cover the same
completion/discard machinery. The failure test permits pending or terminal
failure from nonblocking completion observation; both reject publication and retain
safe eventual retirement, without retrying execution.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- voicechat::component_residency voicechat::backbone::tests::released_backbone_residency
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib composition::mlx::voicechat::submission::tests
```

### EAR-TTS mixture and residual quantization

The [independent fixtures](../eredu-evaluation/fixtures/voicechat/README.md)
execute original AST classes/functions from the pinned NVIDIA source and
Transformers 4.56.0 filtering code, with hash-verified source files. MoG fixtures
use nonzero tiny parameters, 31 codebooks, the released eight-slot schedule and
identical supplied uniforms/Gaussians. They record guided hidden states, component
logits and decisions, means, log standard deviations, sampled latents, every
updated code prefix and summed feedback. A separate codec fixture executes the
original probabilistic quantizer class, including its distinct distance equation,
first-minimum ties and ordered full/prefix decoding.

Codec PRVQ residency tests reuse these fixtures for ordinary, resident-provider
and bounded execution. They compare integer codes, residuals, quantized latents
and ordered decoded sums. Acquisition traces cover full, prefix and nonzero-start
ranges, including fresh-residual suffix decisions. A late selected-depth failure
leaves the source unchanged and retry reproduces the independent reference. The
waveform tests compose bounded codebooks with bounded neural decoding and retain
the same replay, fork, startup and drain assertions.

MLX CPU coverage uses F32/F16 SafeTensors storage across resident, host-layerwise
and disk-streamed policies. F32 uses the independent codec tolerance below; F16
compares identical stored precision across policies. Disk group telemetry checks
that an isolated middle-range request leaves unselected codebooks unloaded.
Exact preparation retains both means and scalar variances, rejects missing
variance or mismatched mean geometry before payload reads, excludes neighboring
parameters and charges F32 conversions. Waveform composition reports codebook
and decoder budgets separately; drain reads neither provider. These component
checks do not establish a unified speech-graph residency schedule or measured
whole-session peak memory.


Codec PRVQ tensor-parallel checks use the pinned nonzero quantizer fixture through
resident and bounded neutral execution. They compare global codes, residuals,
quantized values and ordered decoded means, including selected prefixes. A
separate five-channel oracle splits channels 3/2: each rank prefers a different
local code, while the global distance selects a third code. Its subsequent
first-index tie and fresh suffix request exercise both distance metrics without
substituting local decisions for the global reduction. Communicator and malformed
input rejection precede acquisition; late acquisition failure preserves the
input for retry.

The MLX CPU Ring matrix uses two processes with resident, host-layerwise and
disk-streamed providers. F32 uses the unchanged codec tolerance of absolute
`1e-6` plus relative `1e-5`; F16 disk compares ordinary execution over identical
stored precision. The three-depth fixture uses 44 bytes per depth on each rank,
including the scalar variance replica. Disk telemetry checks that a middle-depth
request leaves both neighboring depths unloaded. Cold pinned-header preparation
selects `[1024,256]` mean tables at all 31 depths, charges 124 replicated variance
bytes per rank, retains exact source authority and rejects inconsistent channel
ownership. F32 conversion overhead is included in the selected resource estimate.

The shared parameter-layout tests cover exact TP ranges, packed expert ownership,
short final chunks and matching quantization scales. Retained architecture layout
coverage checks shared K/V replicas and rejects projection onto another rank.
These tests exercise the runtime mechanism used by model and codec declarations.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib parameter_layout::tests
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  released_quantizer_tp_preparation_ retained_rank_projection_preserves_shared_kv_and_rejects_other_ranks
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_codec_quantizer
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_quantizer_residency native_prvq \
  -- --include-ignored --test-threads=1 --nocapture
CARGO_INCREMENTAL=0 cargo check -p eredu-architectures --test reference_structural
CARGO_INCREMENTAL=0 cargo check -p eredu-backend-mlx --no-default-features \
  --test voicechat_codec_decoder_residency --test voicechat_codec_encoder_residency
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib --no-run
```

The affected structural, codec integration and backend unit-test targets also
compile. Native Ring validation requires loopback sockets. These component
checks do not establish a complete distributed codec or coupled VoiceChat
session, accelerator transport, released-weight conversation parity or measured
streaming latency.

MoG tolerances are absolute `2e-5` plus relative `5e-5`. Independent widened
linear/RVQ arithmetic, with the source's explicitly F32 RMS normalization,
differs by at most `2.5984129345957285e-6` and preserves all discrete decisions.
Codec tolerances are absolute `1e-6` plus relative `1e-5`; its independent
F32/F64 maximum difference is `3.725290298461914e-8`, with identical codes.
These tolerances are fixed from reference precision evidence before Eredu
comparison. Fixture generation commands and complete hashes are in the linked
provenance documentation.

Focused validation commands:

```sh
cargo test -p eredu-architectures --test reference_numeric voicechat_mog
cargo test -p eredu-codec --test prvq
cargo test -p eredu-architectures --test reference_numeric voicechat_codec_quantizer
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_codec_quantizer_residency --test voicechat_codec_decoder_residency
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_mog --test voicechat_prvq --test dense_sampling
cargo check -p eredu --no-default-features
```

Coverage includes neutral and MLX CPU equations, actual native Gumbel selection
from recorded uniforms, five draw requests per frame, malformed-input retry,
partial-frame replay/fork, and ordinary generation through the same refinement
transition. Shared RNG tests compare interleaved uniform, Gaussian and
categorical draws after retained-state adoption. Mixed precision and invalid
codebook indices are rejected. These tests use synthetic nonzero weights;
released-checkpoint TTS/codec parity, native distributed execution, waveform
quality, latency and peak residency are not established by them.

MoG residency tests use the same pinned nonzero fixtures through ordinary,
resident-provider and bounded-provider execution. Prediction leases residual
blocks and the normalization/logits head; selected-component evaluation leases
only its parameter bank. Acquisition traces check that selection does not repeat
prediction. Bounded refinement compares each code prefix and the completed frame
with the NVIDIA oracle. A selected-bank acquisition failure after both random
draws leaves the parent frame and retained RNG unchanged; controlled retry and
uninterrupted generation reproduce the same reference codes.

The MLX CPU test covers F32/F16 SafeTensors storage with resident, host-layerwise
and disk-streamed policies. F32 outputs use the independent MoG tolerance above;
each precision separately compares outputs across residency policies. Exact cold
bindings exclude neighboring RVQ/component tensors and charge converted F32
storage. Host residency uses the largest single-unit budget; disk residency uses
the declared within-group transfer window. No learned static leases are pinned.
Disk telemetry checks that prediction leaves the selected bank unmaterialized
and selection does not rerun either prediction group. Malformed draw rejection
leaves the provider reusable.

These are component results with synthetic weights;
they do not establish a complete speech-graph residency schedule, native
distributed speech or released-checkpoint session parity.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric -- voicechat_mog voicechat_rvq voicechat_speech_sampling voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_realtime_construction voicechat_construction_covers_released_learned_parameters_once_by_unit
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_mog_residency
```

MoG tensor-parallel component checks use the same pinned guidance-on and
guidance-off steps, mixture IDs and Gaussian draws. They compare guided hidden
states, logits, means, log standard deviations and latents with the unchanged
absolute `2e-5` plus relative `5e-5` tolerance. The residual MLP is reduced before
post-normalization; per-mixture low-rank contractions are reduced before applying
scale, residual and noise. Norms and small readouts are explicit per-rank replicas.
Neutral resident and bounded checks exercise real collectives, preflight
communicator rejection, acquisition failure and retry. Provider traces verify
that prediction excludes the selected bank and selected evaluation excludes the
prediction stack.

An additional bounded TP2 case exercises uneven rank widths 2 and 1. It expands
each fixture mean's two rows `[a, b]` into `[a, b, a]` and each low-rank matrix's
columns `[x, y]` into `[x/2, y, x/2]`. This preserves the original nonzero
contraction and its reference outputs; it is an execution-parity transformation,
not a separate NVIDIA rank-three fixture.

MLX CPU Ring coverage uses two processes and the same resident, host-layerwise
and disk-streamed providers. F32 cases use the independent outputs; a focused
F16 disk case compares identical stored precision with ordinary projection.
Local unit byte counts include replicated projections and norms, and bounded
residency peaks remain within selected capacities. Invalid input and retained
wrong-rank placement are rejected before payload reads. Pinned-header preparation
checks coherent segmented mean rows and low-rank columns, exact source retention,
converted storage and replica charges without acquiring weight payloads.
Selected source reads are bounded; released-size strided loading latency is
unmeasured.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_mog voicechat_rvq
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_mog_tp_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_mog_residency native_mog \
  -- --include-ignored --test-threads=1 --nocapture
```

Native
Ring validation requires loopback sockets. These component checks do not
establish synchronized full-session sampling, coupled distributed VoiceChat
commitment, accelerator transport or released-checkpoint conversation parity.

RVQ residency uses the same pinned MoG/RVQ fixture for ordered feedback sums,
embedded guidance input and all five active refinement slots. Neutral tests
compare ordinary, resident-provider and bounded-provider results against the
recorded prefixes and complete frame. Acquisition traces distinguish bank-only
sum/quantization from bank-plus-projection feedback. A quantization acquisition
failure after both random draws preserves the parent frame, cached TTS values
and retained RNG; controlled retry and uninterrupted generation produce the
recorded codes through the shared speech sampler.

The MLX CPU RVQ test exercises resident, host-layerwise and disk-streamed policies
over F32/F16 storage. F32 uses the existing independent MoG tolerance and exact
discrete codes. F16 compares residency policies separately, without claiming
released-precision parity. Cold bindings retain only the canonical RVQ bank and
feedback projection, perform no payload reads, and charge F32 conversion storage.
There are no learned static leases; both bounded policies use the larger of the
two independent units as their device budget. Disk telemetry checks that
quantization leaves the projection unmaterialized. Invalid code/latent geometry
does not prevent retry from the retained parent.

The fixtures use synthetic nonzero weights. Full speech-graph
residency, native distributed execution and released-checkpoint conversation
parity are not established by this component evidence.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric -- voicechat_rvq voicechat_speech_sampling voicechat_tts_ingress voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_realtime_construction voicechat_construction_covers_released_learned_parameters_once_by_unit
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_rvq_residency
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio
```

RVQ tensor-parallel checks reuse the pinned MoG/RVQ code prefixes, depth sums and
guided feedback under resident and bounded neutral execution. Native CPU Ring
coverage adds host-layerwise and disk-streamed policies; F32 uses the existing
absolute `2e-5` plus relative `5e-5` tolerance and exact discrete codes. The F16
disk case compares ordinary execution over identical stored precision. Both bank
and projection are partitioned along latent channels, with no learned replicas.
The compact TP2 fixture charges 1,272 bytes per rank; released-header preparation
checks bank `[31,1024,256]` and projection `[1152,256]` selections and retains
encoded storage/conversion overhead separately.

An independent uneven-width oracle uses five latent channels split 3/2. The ranks
prefer different local codes, while the reduced distance has a third global
winner. A subsequent fresh-residual slot has a first-index tie. These assertions
exercise score reduction before argmin and retain the prior code prefix. PAD
feedback, communicator mismatch, malformed ingress and bank-refusal retry use
the same provider paths. Acquisition traces distinguish sum/quantization from
embedded feedback. The shared codec tests reject reducers that change score
geometry or precision and preserve ordinary codec distances and ordered decoding.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_rvq voicechat_mog_bounded_refinement
CARGO_INCREMENTAL=0 cargo test -p eredu-codec --test prvq
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib released_rvq_tp_preparation_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rvq_residency native_rvq \
  -- --include-ignored --test-threads=1 --nocapture
```

The native two-process Ring check requires loopback sockets. These component
checks do not establish full coupled distributed sessions, accelerator
transport, released-weight speech quality or measured streaming latency.

Guarded MoG and TTS RVQ checks inject asymmetric failures with two participants.
MoG rejects a late layer acquisition, an invalid local down projection and an
incompatible selected low-rank bank before peers enter the next reduction.
RVQ rejects differing sum/embed requests before leases, a late projection
acquisition, malformed local projection precision and invalid late-depth bank
geometry before score reduction. Both checks assert no coordination timeout and
compare repaired execution with the pinned nonzero fixture; RVQ also checks
unchanged parent codes. Native TP2 uses the same retained authority across the
three residency policies and includes one-rank malformed Gaussian rejection
before selected-bank reads, followed by successful retry. F32/F16 comparisons
retain the tolerances and stored-precision distinctions described above.

The shared cold-admission check rejects missing failure agreement, absent bounded
completion and insufficient payload capacity. Family preparation supplies its
own exact required tensor geometry; actual invocation sizes remain enforced by
the retained communication contract. These checks cover component execution,
not coordinator-owned random draws or complete conversational rank scheduling.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib \
  coordinated_collectives_require_failure_completion_and_exact_payload_capacity
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_mog voicechat_rvq voicechat_tts_generator
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_mog_residency --test voicechat_rvq_residency -- \
  --include-ignored --test-threads=1 --nocapture \
  native_mog_tp2_residency_matches_nvidia_guidance_and_selected_draws \
  native_rvq_tp2_residency_matches_nvidia_prefixes_and_feedback
```

Native validation uses local MLX CPU
Ring workers and loopback sockets; accelerator transport remains unvalidated.

### Artifact and input components

The [fixture provenance](../eredu-architectures/tests/fixtures/voicechat/README.md)
records exact SafeTensors header bytes and extracted tensor/metadata directories
from all seven published GGUF files. Tests reject missing tensors, altered
geometry, wrong integer/quantized types, conflicting numerical metadata and
special tokens. No multi-gigabyte weights are stored in the repository.

Focused portable checks:

```sh
cargo test -p eredu-gguf --test block_rows
cargo test -p eredu-checkpoint --lib gguf_explicit_storage_contracts_preserve_integer_and_quantized_types
cargo test -p eredu-core --lib canonical_prepared_safetensors_open_rejects_substitution_without_payload_reads
cargo test -p eredu-architectures --lib voicechat::
cargo check -p eredu --no-default-features
cargo check -p eredu-backend-mlx --no-default-features
```

Downloaded pinned assets and bounded real integer payload ranges reside outside
the source tree. Optional integration tests consume the exact bytes and digests;
no random fixture substitutes for these checks:

Host-conditioning validation reconstructs all 131,072 raw token spellings from
the digest-pinned tokenizer and compares their character sequences exactly with
the independently published split-GGUF tables (891,139 IDs and 131,073 offsets).
The source integer buffers and comparison tables pass their pinned hashes before
comparison. Live marker IDs and missing host-buffer rejection are checked in the
same test. Parameter-unit tests cover restricted physical leases, missing learned
identities, retained transforms/provenance, host-buffer rejection and aliases.
These checks exercise preparation, not released-weight neural execution.

```sh
VOICECHAT_REFERENCE=/private/tmp/eredu-voicechat-reference \
cargo test -p eredu-architectures --lib \
  voicechat::host::tests::released_host_conditioning_matches_published_character_tables -- --ignored
cargo test -p eredu-architectures --lib voicechat::parameters::tests
cargo check -p eredu --no-default-features
```

```sh
VOICECHAT_OFFICIAL_ASSETS=/private/tmp/eredu-voicechat-reference/assets \
VOICECHAT_BASE_ASSETS=/private/tmp/eredu-voicechat-reference/base-assets \
VOICECHAT_REFERENCE=/private/tmp/eredu-voicechat-reference \
cargo test -p eredu-architectures --lib voicechat:: -- --include-ignored
```

Compiler checks cover portable facade and MLX adapter feature boundaries with
default features disabled. GGUF coverage includes nonzero rank-three Q4 bounded
conversion, allocation limits, and exact signed-integer reads through restricted
views from big-endian storage.

Header and integer validation use exact comparisons, not floating tolerances.
The complete released model has no independent comparison against NVIDIA's
implementation. The single-device WAV run above records waveform statistics and
process resource measurements; intermediate-tensor, cached-decode and native
distributed parity are not established.

## Component numerical evidence

The [independent RNNT fixture](../eredu-evaluation/fixtures/voicechat/README.md)
executes hash-verified original NVIDIA function ASTs with tiny nonzero PyTorch
operators. It covers nonzero recurrent initial state, both affine biases,
two-zero priming, joint logits, repeated emissions, blank retention, symbol limits
and feedback termination. The tolerance is fixed from an independent float32 vs
float64 comparison before evaluating Eredu: absolute `2e-6` plus relative `2e-5`,
with measured reference maximum absolute difference `2.640381628893351e-7`.

```sh
cargo test -p eredu-architectures --test reference_numeric voicechat_rnnt
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_rnnt
cargo test -p eredu-codec --test streaming_convolution
cargo test -p eredu-codec --test mimi_artifact_conformance constructs_real_mimi_and_runs_numeric_latent_pcm_and_streaming_paths
```

Three neutral RNNT tests compare reference outputs, controlled continuation,
restore/fork and capacity rollback. Four MLX CPU tests compare the same neural
fixture, preserve BF16 recurrent initialization and verify encoder input rounding
to F16 joint parameters. Three shared-convolution tests compare nonzero independent
scalar equations across all 64 partitions of a seven-sample input and all four
partitions of a three-frame transposed input, plus reset, fork, drain and retry.
The Mimi regression exercises numerical construction and streaming with the
same shared convolution operations. Compiler checks cover the portable facade
with default features disabled.

RNNT provider tests use the same pinned fixture through resident and bounded
execution. They compare two-zero priming, every retained hidden/cell tensor,
token-only prediction, joint logits, greedy emissions and blank/limit evidence.
Acquisition traces verify that prediction does not acquire joint projections and
that the encoder projection runs once per frame. A late recurrent-layer failure
preserves the parent and permits a reference-matching retry; malformed input,
invalid tokens and an unprimed joint request reject before acquisition. The
existing perception integration also composes bounded RNNT and perception,
checking one perception traversal per audio frame.

MLX CPU coverage uses resident, host-layerwise and disk-streamed F32 execution
against that oracle. F16 storage is converted through the prepared F32 recipes;
disk execution compares against resident execution over identical converted
values. Native checks include recurrent replay, source restrictions, rejected
input before learned reads and shared residency bounds. No learned RNNT parameters
remain outside its units. The released header retains six units and 15 tensors,
totalling 35,772,420 F32 bytes; each of its two LSTM units occupies 13,127,680 bytes.
These logical parameter figures exclude recurrent state and temporary tensors.
The native precision regression also compares ordinary and layered predictors
with F16 input, F32 first-layer parameters and F16 second-layer parameters across
initialization and a cached step. Every layer initializes its zeros from the
original input precision, even when an earlier layer promotes its output.

These fixture results do not establish released-weight conversation
parity, a complete distributed speech session or accelerator latency.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  voicechat::provider_residency released_provider_census rnnt_retains_exact selected_target_coexists
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_rnnt voicechat_perception_rnnt voicechat_input voicechat_execution
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric \
  voicechat_bounded_perception_feeds_the_same_cached_rnnt_once_per_audio_frame
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rnnt --test voicechat_rnnt_residency -- --include-ignored --test-threads=1
```

RNNT tensor-parallel preparation uses the pinned 1,025-row vocabulary (including
blank), with 513/512 rows on two ranks. Embedding and output weight/bias rows are
partitioned; recurrent and encoder/predictor joint parameters contribute
30,520,320 replicated F32 bytes on each rank. The resulting parameter payloads
are 33,148,932 and 33,143,808 bytes. Retained source authority is unchanged, repeat
selection does not slice twice, and incompatible layout, bias ownership or
communication contracts reject before materialization. Logical collective limits
cover the complete embedding reduction, gathered vocabulary and I32 decisions;
backend gather padding remains separate mechanism storage.

Neutral TP2 checks compare priming, retained recurrent tensors and every supplied
prediction/joint fixture call against NVIDIA, through leased units. An independent
five-row fixture checks uneven ordering, blank ownership and bias. Coordinated
decoding uses the same greedy loop: explicit invocation and hypothesis agreement
precedes neural work, only the designated owner selects symbols, and integer
publication precedes feedback. Failure tests exercise descriptor mismatch before
leases, a late recurrent lease failure, and an asymmetric tensor error between
neural phases. Both participants preserve their parent and replay successfully,
without an extra agreement that could strand a peer. These checks concern RNNT
component coordination, not full coupled distributed session selection.

Native Ring coverage uses the same bounded communication authority for neural
collectives and decision coordination. Its TP2 matrix covers resident,
host-layerwise and disk-streamed F32 execution, uneven vocabulary and F16 storage
conversion. Owner rank one drives pinned frames and replay; conflicting request
identities or decision owners reject without additional parameter reads. Shared
I32 broadcast coverage uses both signed extrema and values beyond the F32 exact
integer range, and checks completion ownership and wrong-contract rejection.

Native Ring runs require local loopback socket access. These results do not
validate full coupled distributed sessions, accelerator transport,
released-weight conversational parity or hardware latency. The portable facade
and MLX WAV example also compile with the selected feature boundaries:

```sh
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio --example voicechat_wav
```

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  released_rnnt_tp_preparation rnnt_retains_exact released_provider_census rnnt_tp_communication
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_rnnt voicechat_bounded_perception_feeds_the_same_cached_rnnt_once_per_audio_frame
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rnnt_residency native_rnnt_tp2_residency_matches_nvidia_and_uneven_vocabulary \
  -- --include-ignored --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib -- \
  ring_point_to_point_bundle_retains_exact_resources \
  mlx_manifest_capabilities_admit_publication_but_fail_closed_on_unsupported_dtype \
  --include-ignored --test-threads=1 --nocapture
```

RNNT predictor pipeline checks use the same independent NVIDIA fixture and its
absolute `2e-6` plus relative `2e-5` tolerance. PP2 and combined TP2/PP2 execute
two zero-input priming steps and cached token feedback, compare each stage's
hidden/cell tensors and the published readout, and replay unchanged parents.
Neutral checks inject an execution failure and reject foreign ownership,
non-F32 carry, malformed cuts and mismatched low/high words of a 64-bit step
before acquiring parameters. Valid steps above the signed 64-bit range survive
boundary transport. Phase state retains only local carry; the completed state
additionally retains a readout on nonfinal stages.

Pinned official header admission restricts a stage's retained source to its four
LSTM tensors. Neighboring recurrent layers and joint parameters are
inaccessible. Each released predictor layer contributes 13,127,680 F32 parameter
bytes per TP replica. The two PP2 stages retain 7,680 and 5,120 completed-state
payload bytes per rank, including the first stage's extra readout. These are
logical tensor payloads, excluding snapshot-copy scratch, descriptors and
allocator overhead.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib \
  released_predictor_stage_retains_only_owned_lstm_sources_and_complete_replica_bytes
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_rnnt voicechat_bounded_perception_feeds_the_same_cached_rnnt_once_per_audio_frame
```

Native CPU Ring coverage exercises PP2 and TP2/PP2 under resident,
host-layerwise and disk-streamed policies. Tiny stages retain 192 parameter
bytes and 24/16 completed-state payload bytes per rank. Tests compare all local
carry and output values, copy snapshots, replay cached steps and reject
mismatched caller/step identities without additional reads. An undersized
final-readout broadcast rejects during preparation with a typed limit error.
Ordinary mixed-precision, RNNT residency and coordinated TP2 regressions cover
the shared predictor mechanisms.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rnnt --test voicechat_rnnt_residency -- \
  --include-ignored --test-threads=1 --nocapture
```

The complete RNNT frame driver composes that predictor with a compact projection
provider and the existing coordinated greedy loop. Its header checks retain four
units and seven tensors, with no recurrent source access. Ordinary F32 projection
parameters occupy 9,517,060 bytes; TP2 participants retain 6,893,572 and 6,888,448
bytes, including 4,264,960 replicated joint-input bytes on each rank. These
projection payloads repeat on each pipeline stage and are charged independently
from the stage's recurrent parameters.

Neutral tests cover pinned frame decisions, local carry, blank and symbol-bound
behavior, non-refundable step/epoch reservations, F32 restored-state preflight,
aborted predictor outcome rejection and parent-preserving retry. A TP lease
failure is agreed before the vocabulary collective. Native PP2 and TP2/PP2 use
all three residency policies, with a separate projection pool bounded to 64
learned bytes in the tiny fixture. They compare the same pinned outputs and
carry, reject cursor exhaustion without reads, and recover from asymmetric
ingress and post-construction checkpoint-read failures using fresh epochs. The
native coordinated TP2 matrix uses the shared vocabulary agreement authority.
Its focused Ring communication check rejects a one-rank invalid embedding token
and one-rank linear/tied-projection shape errors before data collectives, then
verifies successful nonzero retries. Pending completion and resource ownership
use the existing bounded communication authority.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  rnnt_projection_sources rnnt_tp_communication
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_rnnt_pipeline voicechat_rnnt_projection voicechat_rnnt_residency \
  voicechat_rnnt_predictor voicechat_rnnt_coordination
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rnnt_residency native_rnnt_greedy_pp2_and_tp2_pp2_match_pinned_frames_and_replay \
  -- --include-ignored --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rnnt_residency native_rnnt_tp2_residency_matches_nvidia_and_uneven_vocabulary \
  -- --include-ignored --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib \
  ring_point_to_point_bundle_retains_exact_resources -- \
  --include-ignored --test-threads=1 --nocapture
```

These tests do not validate a coupled distributed speech session, accelerator
execution or released-weight conversational parity. The fixture measures neither
real-time latency nor full-model peak residency.

The paired input fixture applies the original pinned NVIDIA RNNT methods to all
six raw streaming perception outputs, using a compatible four-wide encoder
projection and nonzero two-layer predictor. Its recorded Float32-versus-Float64
maximum difference is `1.2518894130231217e-6`, with identical greedy decisions;
RNNT comparisons use the independently fixed `2e-6` absolute plus `2e-5` relative
tolerance. Perception outputs and caches retain their own fixture tolerances.
The fixture records the input perception artifact's SHA-256 and PyTorch 2.13.0
CPU provenance; it does not contain the released 11B weights.

Paired component checks compare raw/projected features, local Conformer caches,
RNNT decisions and recurrent carry. They exercise immutable parents, fresh-epoch
replay, bounded snapshots, capacity before reads and rejection of a perception
outcome before RNNT. The native PP2 and TP2/PP2 matrix uses resident,
host-layerwise and disk-streamed providers, with cuts before and after the
Conformer stack. An asymmetric disk acquisition failure is injected separately
in perception and RNNT, with parent preservation and a fresh-authority retry. A
neutral executor check runs only one stage's two tensor ranks in a declared
four-rank world to verify that acquisition agreement excludes inactive stages
and precedes the Conformer collective. Cold preparation checks exact source
retention, role restrictions, parameter/cache payloads and decision authority
without weight reads. These results establish component composition and failure
ownership, not full-session distributed selection, accelerator parity or
released-weight conversation quality.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  coupled_pipeline_preparation
CARGO_INCREMENTAL=0 cargo test -p eredu-runtime --lib \
  pipeline_cohort_requires_exact_status_before_parameter_realization
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_perception_rnnt_pipeline \
  voicechat_perception_pipeline_tp_acquisition_failure_excludes_inactive_stages
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_rnnt_residency \
  native_perception_and_rnnt_pp2_tp2pp2_share_frames_and_replay \
  -- --include-ignored --nocapture
```

PCM input composition compares ordinary and pipeline execution with the retained
128-bin preprocessing buffers and nonzero perception parameters from the
tool-silence fixture, plus the compatible paired RNNT fixture. This comparison
uses the existing independent component bounds: perception `3e-6` absolute plus
`3e-5` relative, and RNNT `2e-6` absolute plus `2e-5` relative. It is a comparison
between Eredu execution paths, not an independent NVIDIA full-PCM-stack oracle.
The neutral pipeline check rejects differing readiness and processed feature
content before neural reads, retains authority across incomplete submissions,
and replays failed and discarded ready attempts with fresh epochs. Coupled
transaction checks retain a selected local cache through the same live,
perception-only and tool-audio branches, comparing controlled and automatic runs.

The MLX CPU PCM check consumes the retained aggregate in PP2 and combined
TP2/PP2, under resident, host-layerwise and disk-streamed policies. It compares
arbitrary 157-sample submissions with whole-input execution over two complete
frames, a padded 317-sample final frame and one continuation-silence frame. It
covers frontend/local-cache snapshot replay, capacity rejection before reads,
unchanged parents, exact sample attribution and drain. Cold checks also reject a
projection cohort that supports neural collectives but lacks the exact integer
control publication needed by its coordinator. The standalone distributed greedy
RNNT regression exercises ordinary coordinator construction. The portable facade
and MLX/audio WAV example compile with the selected-input contract.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib --test reference_numeric -- \
  coupled_pipeline_preparation voicechat_input voicechat_execution
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio \
  --test voicechat_rnnt_residency \
  native_pcm_aggregate_pp2_tp2pp2_matches_ordinary_chunks_and_drain \
  -- --include-ignored --nocapture
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio \
  --test voicechat_rnnt_residency \
  native_rnnt_greedy_pp2_and_tp2_pp2_match_pinned_frames_and_replay \
  -- --include-ignored --nocapture
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features --features mlx,audio \
  --example voicechat_wav
```

The perception fixture executes hash-verified NVIDIA subsampling, relative
attention, convolution and Conformer classes with nonzero tiny parameters. It
records both feature branches and both caches for a full invocation and six
cached steps. The separate layer fixture includes a two-row chunk and cache
rollover. Independent float32/float64 maximum difference is
`8.284204126574046e-7`; tolerance is absolute `3e-6` plus relative `3e-5`, fixed
before Eredu comparison. The coupled reference segmented/full difference is
`2.384185791015625e-7`.

The host fixture executes NVIDIA's original filterbank class and cache-manager
step method, using a recording encoder for mel selection. It covers finite-window
reflection, rolling history, the default and padded first-step policies, and
nonzero exactly representable PCM. Independent float32/float64 maximum log-mel
difference is `9.1552734375e-5`; the fixed absolute tolerance is `4e-4`.
Regeneration commands, source hashes and Torch/librosa versions are in the
[independent fixture provenance](../eredu-evaluation/fixtures/voicechat/README.md).

```sh
cargo test -p eredu-media --features audio --test streaming_audio
cargo test -p eredu-media --features audio --lib leading_slaney_frontend_uses_input_divided_by_hop_frames
cargo test -p eredu-media --no-default-features --test streaming_audio
cargo test -p eredu-architectures --test voicechat_frontend
cargo test -p eredu-architectures --test reference_numeric voicechat_perception
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_conformer --test voicechat_subsampling
cargo check -p eredu --no-default-features
```

Perception, frontend and host streaming tests cover independent outputs/caches,
both initial phases, rollover, arbitrary PCM chunking, snapshots, backpressure,
explicit partial finish and mixed-precision rejection. Shared coverage includes
leading-Slaney extraction and host processing without optional audio features;
compiler checks cover the portable facade. Independent GPU perception numerics
are unvalidated; CPU Ring component evidence is described in the perception
parallel sections.

Layered perception conformance reuses the pinned full-invocation and six-step
cached fixture. Neutral tests compare both output branches, channel/convolution
caches and frame counters across resident and rebuilt bounded units. They inject
a projection-acquisition failure after encoder execution, check the unchanged
parent and retry, and exercise RNNT and BF16 startup silence through the same
provider contract. The coupled execution regression and released-header assembly
test cover shared invocation dispatch and preservation of already-bound modules.

The MLX CPU residency test stores those nonzero fixture parameters in temporary
SafeTensors files as F32 and F16. For each storage precision it compares resident,
host-layerwise and disk-streamed output/caches exactly across six cached steps,
after an aborted subsampling invocation. F32 output also uses the independent
NVIDIA tolerance above; F16 storage parity is assessed separately. The test checks
zero static leases, cold preparation without payload reads, restricted sources,
F32 materialized-byte accounting and actual disk-stream activity. Bounded modes
use a device budget equal to the largest individual unit and assert that their
reported peak stays within it. Disk execution disables host caching and lookahead.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric -- voicechat_perception voicechat_bounded_perception voicechat_startup_silence_uses_the_same_bounded_perception_units
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_realtime_construction voicechat_construction_covers_released_learned_parameters_once_by_unit
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_perception_residency
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib both_text_architectures_retain_exact_sharded_admission_with_one_cached_payload
CARGO_INCREMENTAL=0 cargo check -p eredu --no-default-features
```

The shared disk-streamed Mistral/Qwen3 regression covers nonempty
static parameter bindings and ordinary single-group completion.
These component fixtures do not establish released-weight conversations,
whole-speech-stack residency, native TP/PP, GPU performance or latency measurements.

The channel-boundary fixture executes pinned NVIDIA `AddFusion`, final
`NemotronHRMSNorm`, prompt preparation and `DuplexSTTModel.forward` with nonzero
weights. A disclosed backbone stand-in supplies residual hidden states; this
oracle checks the fusion/normalization/readout boundary and does not execute
attention or Mamba. The fixed F32 tolerance is absolute `2e-6` plus relative
`2e-5`; the independent widened-reference maximum difference is
`1.7369699012448336e-7`. BF16 head comparisons allow one BF16 ULP plus that F32
bound, with identical quantized inputs/weights audited against float64
accumulation. Audio cast-before-fusion is compared exactly in BF16.

Function-head residency reuses the pinned F32 normalized-hidden and function-logit
fixture with the same absolute `2e-6` plus relative `2e-5` tolerance. Ordinary,
resident-provider and bounded projections consume that exact hidden tensor.
Acquisition refusal leaves it unchanged for retry; malformed rank, width, empty
geometry and integer inputs fail before provider activity. The backbone composition
fixture checks one traversal and one function-head lease per readout. The coupled
transaction fixture compares a bounded backbone/function provider with resident
controlled and uninterrupted runs, including outputs, retained state and RNG.

Selected-backbone conformance routes the same coupled executor through
`ParallelBackbone` and independently selected `ParallelFunctionHead`. Two neutral
ranks execute nonzero Mamba/attention/MLP parameters under bounded layer loading.
Prompt prefill, cached live/tool steps, perception-only work, reminder speech and
codec drain match ordinary execution in token/code decisions, clocks, PCM and
shared retained carry. The test checks smaller rank-local backbone state and
exact layer acquisitions for the three backbone invocations. Float comparisons
use the existing reference-numeric absolute `2e-4` tolerance; discrete output,
clock and RNG comparisons are exact. The standalone selected-provider test also
compares normalized hidden and both gathered vocabulary heads at every cached
position. These are compact equation-level comparisons, not released-weight
speech or native coupled TP/PP validation. Native coupled parallel construction
and whole-graph owner coordination are not exercised here.

Native MLX CPU
ordinary submission matches controlled and scheduler-driven publication, and
the backbone residency test preserves cached dual logits and hybrid state under
resident, host-layerwise and disk-streamed execution. This evidence does not
include a native distributed run of the selected coupled backbone path.

Coupled owner coordination uses the same compact nonzero components and ordinary
providers on both participants. The owner advances explicitly while the follower
uses the uninterrupted transition driver. Prefill, live audio, perception-only,
retained-audio tool steps, reminder speech and drain match ordinary execution
exactly for output samples, codes, timestamps, feedback, sampler history and
retained neural carry. The owner retains the ordinary RNG sequence; the follower
has no RNG and rejects any resolver invocation.

The coordinated test rejects different valid invocation kinds before audio
consumption, preserves a follower's typed `NeedsInput` error, and injects an
owner-only character-capacity failure after perception/backbone execution and
channel selection. Both participants discard the failed candidate through the
ordinary transaction owner. Canonical carry, history and RNG remain unchanged;
fresh authority and branches complete the same recorded continuation without
transport timeouts. This is replicated ordinary-provider evidence; it does not
validate failures inside TP collectives or native distributed session loading.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_execution_coordinated_owner_policy_failure_discard_and_retry
```

The ordinary MLX coupled submission regression exercises explicit advancement,
scheduler publication, snapshot restore and fork on one device. Coordinated
comparison is exact within the same neutral backend and fixture precision,
independently of the NVIDIA component tolerances above. These checks do not
establish native coordinated execution or released-weight conversation parity.

MLX CPU component coverage stores the same nonzero parameters as F32 and F16
SafeTensors and selects resident, host-layerwise and disk-streamed policies. F32
uses the independent tolerance above; F16 compares identical stored precision
across policies. The prepared provider selects dense F32 execution and charges
its widened storage, while retaining encoded storage and conservative conversion
peak accounting separately. Source views exclude neighboring heads; malformed
hidden width fails before reads. Missing role/weight and mismatched checkpoint
geometry are rejected before native construction. These component checks do not
establish aggregate speech-stack peak memory. These results use compact numerical
fixtures and pinned headers, not a full released-weight conversation.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_function_head
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_backbone_composition
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --features audio --test reference_numeric voicechat_execution_couples
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric voicechat_execution_selected_tp_backbone_matches_coupled_cached_transitions
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --lib coupled_native_execution_matches_explicit_and_scheduler_publication
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --features audio --test voicechat_backbone_residency
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --test voicechat_function_head_residency
```

Function-head TP2 uses the same retained normalized-hidden fixture and unchanged
F32 tolerance. Resident and bounded neutral execution and resident, host-layerwise
and disk-streamed MLX CPU execution gather the full logits from actual rank-local
projections. Removing the final fixture weight row and corresponding expected
logit tests an uneven 12/11 split without changing the independent reference for
the retained rows. The native F16 uneven-vocabulary disk case compares with
ordinary projection over identical stored precision, separately from NVIDIA F32
parity. Repeated projection checks completed head leases and per-rank weight
accounting; invalid hidden input and wrong-rank retained selection fail before
payload reads. Neutral checks reject communicator drift before acquisition and
permit retry after an acquisition refusal.

Guarded function-head TP coverage rejects asymmetric malformed hidden input,
independently valid but different batch extents, one-rank acquisition failure and
wrong leased-unit ownership before any vocabulary gather. The failing acquisition
retains its original error source; the successful peer's lease reaches the shared
abort lifecycle. A healthy retry matches the pinned nonzero NVIDIA logits using
the unchanged component tolerance. Cold admission rejects missing bounded
failure/gather authority, insufficient local or gathered output capacity and
incompatible dtype declarations.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib --test reference_numeric -- \
  function_head_collective_admission_requires_failure_and_exact_f32_gather_bounds \
  voicechat_function_head_parallel voicechat_function_head_residency_matches_pinned_normalized_readout
```

Native preparation and ordinary residency have separate component coverage. The
CPU two-process Ring test exercises retained failure authority across resident,
host-layerwise and disk-streamed F32 heads, balanced and uneven vocabulary rows,
and the existing F16 uneven-vocabulary disk case. One-rank malformed input
prevents payload reads on both participants. A retained-source wrapper injects a
one-rank disk acquisition failure without changing the checkpoint file; the
original source failure remains observable and a healthy provider retries
through the same retained coordinator. The native distributed test requires
loopback socket access.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_function_head_residency -- --include-ignored --test-threads=1 --nocapture
```

These function-head guards do not establish failure coordination inside the
Nemotron backbone, a complete distributed conversation or released-weight parity.

Pinned-header admission retains the exact restricted source, selected row recipes
and encoded/conversion overhead without reading payloads. Matching retained
selection is reused without slicing twice. Cold physical-format coverage admits
balanced affine weight/scale/bias rows for TP1 and TP2 and rejects an opposite-rank
scale even when its shape matches. This format check does not claim native packed
affine execution through the dedicated dense F32 prepared head.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_function_head_ voicechat_backbone_composition
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --lib -- \
  released_function_head_tp_preparation_ function_head_parallel_layout_
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_function_head_residency native_function_head \
  -- --include-ignored --test-threads=1 --nocapture
```

The local two-process Ring check requires loopback sockets. These component
checks do not establish full coupled distributed VoiceChat sessions, native
accelerator transport or released-weight conversation parity.

Guarded channel ingress compares operation identity, tensor metadata, fusion
weights and exact prompt framing before vocabulary lookup. The neutral fixture
rejects different prompt/audio operations, invalid prompt capacity, different valid
prompt IDs and malformed feedback geometry without a lookup or tensor collective.
An injected status-agreement failure after the first completed lookup preserves
its original source and prevents the second lookup; this is an agreement failure,
not a simulated failure inside a numerical backend. Healthy audio and repeated
prompt retries match the pinned NVIDIA values under the existing tolerance.
The selected `ParallelBackbone` callback also rejects ingress with position zero
and no learned-unit acquisition. Cached TP backbone and coupled transition
regressions preserve their outputs and retained state.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-architectures --test reference_numeric -- \
  voicechat_channel_ingress_guarded voicechat_channels_match_nvidia \
  voicechat_backbone_composition_tp2 voicechat_execution_selected_tp
```

The MLX CPU channel and prompt regressions check BF16 cast/add order. A
two-process Ring run over actual vocabulary shards matches the NVIDIA audio and
prompt values, rejects asymmetric feedback geometry and prompt capacity as well
as different invocation kinds, and completes a healthy retry. This test requires
loopback socket access.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test voicechat_channels -- --include-ignored --test-threads=1 --nocapture
```

Ingress agreement protects channel preparation before layered execution; it does
not guard fallible preparation or collectives inside the Nemotron blocks. These
component fixtures do not establish released-weight or full-session parity.

Shared native row-parallel projection has a separate two-process Ring fixture
using retained failure authority. Unequal local input widths and nonzero,
rank-distinct contributions have an independent exact arithmetic oracle, including
replicated bias applied once. Asymmetric observer refusal preserves its typed
source. Invalid local projection geometry, incompatible output or bias metadata,
and selected reduction-limit rejection converge before the dependent collective;
a failed bias broadcast is agreed after a completed reduction, and a healthy retry
uses the same coordinator. Existing single-rank dense, affine and block-FP8
regressions cover bias/weight identity across observer errors and unwind.
Unwind coverage is local parameter preservation, not distributed panic recovery.

```sh
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features --lib -- \
  row_parallel_observer_error_or_unwind_preserves_bias_and_retry \
  ordinary_and_row_parallel_observers_apply_bias_once_or_leave_absence_unchanged
CARGO_INCREMENTAL=0 cargo test -p eredu-backend-mlx --no-default-features \
  --test row_parallel_agreement -- --include-ignored --test-threads=1 --nocapture
```

The distributed test requires loopback socket access and uses exact
F32 arithmetic for this compact fixture; it does not measure accelerator latency.

This native operator coverage does not establish failure agreement in preceding
Nemotron normalization, attention/cache or recurrent preparation, whole-session
distributed execution, or released-checkpoint parity.

The tests also exercise the existing Nemotron layer path with supplied embeddings:
resident and rebuilt-layer traversal, prefill and multiple cached steps, matching
retained state, normalized readout over a TP2/PP2 pipeline, and exact rejection of
malformed readout controls. Independent channel fixtures run through TP2 vocabulary
lookup and both head gathers. Rebuilt-layer traversal establishes shared driver
behavior; it is not a measurement of disk I/O or native distributed residency.

```sh
cargo test -p eredu-architectures --lib voicechat::backbone::tests
cargo test -p eredu-architectures --test reference_numeric nemotron
cargo test -p eredu-architectures --test reference_numeric voicechat_channels
cargo test -p eredu-architectures --test voicechat_frontend
cargo test -p eredu-backend-mlx --no-default-features --test voicechat_channels
cargo check -p eredu --no-default-features
```

The BF16 normalization comparison
uses float32 scale multiplication before output rounding and matches the
independent tolerance.

The native streaming reference recomputes fused-embedding history for its
Nemotron branch; the VLLM path owns a cache. Eredu's cached/full equivalence
evidence here comes from its existing shared layer paths and is distinct from an
independent released-weight cached-backbone comparison. Complete-session
execution and released-checkpoint parity are not established by these fixtures.
CPU Ring channel/perception component results and released single-device session
measurements have separate coverage above.

These Nemotron and channel tests use synthetic weights. Their isolated component
results do not establish complete-session or released-weight waveform parity.
