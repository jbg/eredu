# LFM3 text execution and acceptance

The experimental LFM3 text route admits `lfm3` SafeTensors and GGUF checkpoints.
Its declaration owns width-seven gated convolutions, global and sliding attention,
per-head sigmoid gates, separate rotary policies, and sandwich normalization before
residual addition. Global rotary pairs retain full-head frequency spacing; only the
highest-frequency half rotates. Rotary tables, products and sums use FP32 before
the final input-dtype cast.

Resident, host-layerwise and disk-streamed execution use independently leased decoder
layers. TP/PP construction consumes the same declaration and local state geometry.
Ordinary and controlled text generation use the shared drivers. The existing tagged
tool dialect consumes the released template's function/parameter tags; the LFM2
Python-call dialect does not describe this template.

The public LFM3 route supports text. Vision has standalone neural components.
The Audio architecture admits the complete floating checkpoint and its codec
companion and supports resident mixed-input traversal. Both this path and
standalone LiquidCodec have independent F32 reference acceptance. The explicit
resident Audio session exposes controlled text and speech events. Public
vision/video generation, omni and d1 decision execution remain unimplemented. The generic GGUF reader does not
admit published NVFP4 encoding.

## Released SafeTensors evidence

The source manifest is `validation/fixtures/lfm3/sources.json`. Instruction weights
from `LiquidAI/LFM3-1.7B` revision
`1da06bec6a819da10b54c67e5c90972b46d81871` have SHA-256
`135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.
Exact widening from BF16 to F32 produces SHA-256
`6f1823dc67f4f6367f61702379ac5505124e7e4719013753e2ff0ebb08230c98`.
This measures execution of the widened weights, not native BF16 numerical parity.

The independent publisher implementation uses PyTorch 2.15.0 and Transformers
5.22.2, eager attention and F32 execution. MLX CPU compares every vocabulary score
after prefill and after each of eight teacher-forced cached inputs
`[5,8,13,21,34,55,3,2]`, with maximum absolute tolerance `5e-4`.

| Prompt | Maximum absolute error across all nine rows | Argmax agreement |
| --- | ---: | --- |
| `[1,3,2,5,8,13,21,34,55]` | `6.318092346191406e-5` | 9/9 |
| 269 tokens, `(13*i + 1) % 128000` | `1.544952392578125e-4` | 9/9 |

A greedy CPU run also matches all nine predicted IDs and full score rows, with
maximum absolute error `6.318092346191406e-5`.

Metal F32 execution with `MLX_ENABLE_TF32=0` matches the same independent reference:
greedy maximum error is `6.67572021484375e-5`; the 269-token input has maximum error
`1.1399388313293457e-4`. All nine argmax IDs agree in both cases. Default TF32
execution exceeds the `5e-4` F32 tolerance (maximum decode error `0.0238952637`);
the precise setting is part of this acceptance configuration.

The longer input crosses the 256-token sliding window. Full tensors and diagnostic
outputs live outside the repository. The shared `chunked_prefill` conformance driver
passes on CPU and Metal with a compact convolution/sliding/global LFM3 fixture,
using the same
residency, affine loading and controlled/uninterrupted checks as existing fixtures.
Shared execution agreement does not substitute for released-checkpoint acceptance.

## Same-artifact GGUF evidence

`LiquidAI/LFM3-1.7B-GGUF` revision
`cd0c74c575b5c416f3011b5b4c298077f26ca8c2`, file
`LFM3-1.7B-Q4_K_M.gguf`, has SHA-256
`3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4`.
Eredu Metal with `MLX_ENABLE_TF32=0` and llama.cpp Metal execute this identical
encoded artifact. The reference is llama.cpp `cd959bae99ff4bed22bef2604f45241c09bbb8c8`
(0.9.0, build 12110), with F32 KV and flash attention disabled. This reference pin
is the installed version used for measurement.

The short prompt and eight teacher-forced cached inputs compare every vocabulary
score. Maximum relative L2 is `0.01882853`, minimum cosine is `0.99983101`, minimum
top-five overlap is 4/5, and all nine argmax IDs agree. These meet the existing shared
logit profile (`relative_l2 <= 0.02`, `cosine >= 0.999`, top-five overlap at least four,
and matching unambiguous argmax). Maximum absolute difference is `0.24158096`;
quantized evidence does not use the F32 absolute tolerance.

llama.cpp CPU uses Q8_K activation dots for Q4_K weights. The CPU-to-CPU comparison
has relative L2 `0.052–0.088` and does not meet the shared score profile, despite
matching token IDs. It is not recorded as passing numerical acceptance. This result
is separate from the passing Metal-to-Metal comparison.

Compact measured rows, precision settings and token IDs are retained in
`validation/fixtures/lfm3/acceptance-results.json`. Other published GGUF encodings
have no recorded numerical acceptance.

## Reproduce

Download with the pinned shared helper, keeping `HF_HOME` and all outputs outside
the checkout. Custom Python code is fetched from the same immutable revision.

```sh
python eredu-evaluation/scripts/download_component_fixture.py /private/tmp/lfm3-source --model lfm3
python eredu-evaluation/scripts/prefill_reference.py /private/tmp/lfm3-source/provenance.json /private/tmp/lfm3-reference.json --f32-checkpoint /private/tmp/lfm3-f32 --trust-remote-code --decode 5,8,13,21,34,55,3,2
cargo run -p eredu-backend-mlx --no-default-features --example checkpoint_probe -- --model /private/tmp/lfm3-f32 --device cpu --input-ids 1,3,2,5,8,13,21,34,55 --teacher-forced-ids 5,8,13,21,34,55,3,2 --warmup-runs 0 --output /private/tmp/lfm3-actual
EREDU_PREPARED_CHECKPOINT_CACHE=/private/tmp/lfm3-prepared-cache cargo test -p eredu-backend-mlx --no-default-features --test chunked_prefill cpu_chunked_prefill_matches_full_observed_logits_and_controlled_generation -- --exact
```

`validation/reference_runner.py` also produces the ordinary shared comparison
artifact from a probe JSON using `--trust-remote-code --local-files-only --device cpu
--dtype float32`. A same-artifact GGUF reference uses
`validation/gguf_reference.cpp`, built against the exact llama.cpp revision recorded
with its result. The exporter emits full prefill and cached-decode vocabulary rows;
generated token agreement alone is insufficient.

Native BF16, the base checkpoint and physical multi-device numerical parity
have no recorded acceptance here. Vision, speech and decision results are absent.

For precise Metal acceptance, build the probe with `--features metal`, select
`--device gpu`, and set `MLX_ENABLE_TF32=0`. Metal execution needs device access.
The same probe accepts the GGUF file as `--model`; use
`--output /private/tmp/lfm3-gguf-actual` for the comparison command below. Build the independent exporter
against the pinned llama.cpp headers and library:

```sh
clang++ -std=c++17 validation/gguf_reference.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lllama -Wl,-rpath,/opt/homebrew/lib -o /private/tmp/gguf_reference
/private/tmp/gguf_reference /private/tmp/LFM3-1.7B-Q4_K_M.gguf /private/tmp/lfm3-gguf-reference.safetensors 1,3,2,5,8,13,21,34,55 5,8,13,21,34,55,3,2 gpu
cargo run -p eredu-evaluation --bin eredu-parity -- --actual /private/tmp/lfm3-gguf-actual.json --reference /private/tmp/lfm3-gguf-reference.json --output /private/tmp/lfm3-gguf-parity.json
EREDU_TOOL_CHECKPOINT=/private/tmp/lfm3-f32 cargo test -p eredu --no-default-features --test portable_facade lfm_checkpoint_native_tool_activation_with_portable_backend -- --exact --ignored
```

The portable tokenizer/template case uses scripted backend outputs. It verifies
native tool grammar activation and decoded semantic events, not model tool quality.

## Independent vision components

`lfm3::vision` contains the released encoder's linear patchifier, gated spatial
depthwise-convolution blocks, frame-local bidirectional GQA blocks with axial RoPE,
and LayerNorm/spatial-merge/GELU projector. Each encoder layer and the input/output
units is constructible independently. The host patch-packing utility accepts
already normalized frames; it duplicates still images across temporal slots and
repeats the last frame when a video ends with an incomplete temporal patch.

`validation/fixtures/lfm3/vision-sources.json` pins the VL instruction checkpoint,
its complete weight hash and the publisher source-file hashes. The reference
entry point executes the pinned publisher encoder, projector and image/video
patchifiers on an exact F32 widening of the released component weights. Its cases
are a rectangular image, an odd-length three-frame clip, and two differently
shaped images. Inputs are already sized; resizing and frame sampling are outside
this comparison. It does not establish end-to-end conditional-generation parity.

The compact measured results in
`validation/fixtures/lfm3/vision-component-results.json` record exact host packing
agreement for all three cases. CPU encoder maximum absolute error is
`8.7738037109375e-5`; precise Metal encoder maximum is `1.3017654418945312e-4`.
Projector maxima are `6.705522537231445e-7` on CPU and
`1.259148120880127e-6` on Metal. Every encoder/projector comparison meets the preset
`5e-4` maximum absolute tolerance. These measurements use F32 widening; they do
not establish native BF16 component parity.

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("LiquidAI/LFM3-VL-2B", revision="23387beda43c60310348e83e25f78f33c3140d3a", local_dir="/private/tmp/lfm3-vl-source", allow_patterns=["*.py", "*.json", "*.jinja", "*.safetensors"])'
HF_HOME=/private/tmp/lfm3-hf HF_MODULES_CACHE=/private/tmp/lfm3-hf/modules python validation/lfm3_vision_reference.py /private/tmp/lfm3-vl-source /private/tmp/lfm3-vl-reference
cargo run -p eredu-backend-mlx --no-default-features --example lfm3_vision_reference -- /private/tmp/lfm3-vl-reference/f32-component /private/tmp/lfm3-vl-reference /private/tmp/lfm3-vl-reference/cpu-results.json --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features metal --example lfm3_vision_reference -- /private/tmp/lfm3-vl-reference/f32-component /private/tmp/lfm3-vl-reference /private/tmp/lfm3-vl-reference/metal-results.json --device gpu
```

The reference runner needs PyTorch, Transformers, SafeTensors, torchvision and
Pillow, and verifies the immutable weights and source hashes before executing
publisher Python code. The native runner validates source geometry and checks the
derived F32 weight hash before comparing values. Payloads and detailed outputs
remain outside the checkout.

These components have no public VL task admission, conditional text composition,
GGUF admission, selected memory/residency reports, TP/PP construction or controlled
media-session integration. No corresponding support is advertised by their
existence or by encoder/projector comparison.

The speech input path admits the released Audio SafeTensors encoder and projector
geometry. Its host processor accepts 16-kHz mono PCM and computes 128-bin Slaney
power-mel features with an uncentered 400-sample window and 160-sample hop.
Right padding completes the final window. Three convolutional stages reduce
time by a factor of eight before the causal LFM2 backbone, whose attention
uses a 32-token window. The encoder can advance over complete stride groups
and emit the final partial group on flush. Finite batches contain equal-length
unpadded clips. The standalone encoder and projector do not establish full
audio language-model acceptance.

The standalone speech output path admits the released depth and feedback
SafeTensors scopes. The three depth blocks score all twelve codebook positions
bidirectionally. The checkpoint schedule reveals four codebooks per pass in a
request-supplied permutation. Codebook tables are shared between input lookup
and output scoring. A frame containing any EOS code terminates the speech
segment. The completed frame exposes either ordinary codec codes or EOS. The
language feedback table embeds ordinary frames and canonical all-EOS frames.
The resident driver uses these components for controlled speech output.
Native BF16 numerical parity remains unvalidated.

Standalone LiquidCodec neural encoding and finite/cached decoding, CPU/Metal
F32 reference results and integration limits are documented in
[LiquidCodec validation](liquid-codec-validation.md). Resident codec acceptance
does not establish bounded/distributed codec or speech-model acceptance.

## Mixed speech input and feedback

`PreparedAudio` admits the full released Audio SafeTensors checkpoint and
bundled LiquidCodec. It retains restricted language, speech encoder, depth
and codec sources, as well as the tokenizer, template, generation defaults
and processor policy. `AudioPrompt` checks the number of projected rows
against speech placeholders and uses the shared LFM3 decoder traversal to
replace those positions. Its readout returns normalized language states for
depth scoring or the tied text head. A caller can also use the same input
hook with an already configured layerwise language runtime.

The independent publisher case uses the Audio revision
`2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1`. It compares 128,000-entry
prefill score rows from a prompt with both input speech and replayed speech,
eight cached language steps including EOS feedback, all three depth score
passes, one synthesized frame and the subsequent feedback step. The fixture
uses a fixed reveal order and greedily selects from the codec vocabulary to
guarantee a speech frame. Source BF16 values widen exactly to F32 when bound.

Both CPU and precise Metal pass the preset `5e-4` maximum absolute tolerance
and match all 17 greedy text decisions and all twelve sampled speech codes.
The synthetic speech frame has 1,920 PCM samples and a reference peak of
`1.272e-6`. Artifact hashes, inputs, package versions and per-phase maxima
are retained in
[`audio-acceptance-results.json`](../validation/fixtures/lfm3/audio-acceptance-results.json).

| Measurement | CPU maximum absolute error | Metal maximum absolute error |
| --- | ---: | ---: |
| Prefill scores | `1.149e-4` | `2.399e-4` |
| Eight cached score rows | `2.694e-5` | `5.007e-5` |
| Depth scores | `2.289e-5` | `2.575e-5` |
| PCM samples | `2.786e-12` | `4.406e-12` |
| Scores after generated speech feedback | `4.054e-6` | `6.170e-6` |

Keep checkpoint payloads and generated traces outside the checkout. The Python
runner verifies source and weight hashes before loading the publisher code.

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("LiquidAI/LFM3-Audio-2B", revision="2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1", local_dir="/private/tmp/lfm3-audio", allow_patterns=["*.py", "*.json", "*.jinja", "model.safetensors", "audio_tokenizer/*"])'
HF_HOME=/private/tmp/lfm3-hf HF_MODULES_CACHE=/private/tmp/lfm3-hf/modules python validation/lfm3_audio_reference.py /private/tmp/lfm3-audio /private/tmp/lfm3-audio-reference
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-audio-reference /private/tmp/lfm3-audio-reference/cpu.json --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features audio,metal --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-audio-reference /private/tmp/lfm3-audio-reference/metal.json --device gpu
```

The retained evidence covers resident conditional execution with a fixed
sampler. It does not compare the publisher's random sampling stream or realtime
mode transitions. Official audio GGUF, native BF16, and selected bounded and
distributed audio execution remain outside this path.


## Resident Audio sessions

`eredu::api::realtime::prepare_realtime_model` discovers `lfm3_audio` from
the model assets and admits the specified SafeTensors file or directory.
`prepare_mlx_realtime` with `RealtimeLoadOptions::Lfm3Audio` selects the
resident model and reports the logical F32 parameter bytes by component.
The report excludes session state and temporary allocations. Load consumes
the selection with explicit weight and execution contexts. The bound model's
`start` accepts a batch-one `AudioRequest` and sampling seed. The request supplies
token IDs with expanded speech placeholders, finite mel features in prompt
order, optional replayed speech codes, a mode, and a slot budget. The retained
processor transforms mono 16-kHz PCM to the input features.

The session emits text token IDs, completed codebook frames with 24-kHz
mono PCM, and speech EOS. Both `advance` and `events` use the same controlled
driver. Idle snapshots and restore use explicit copy limits and preserve
unfinished refinement and codec carry. Live forks resume the settled state
on the same resident graph, with independent restore origins and a shared
copy budget. The audio feature also exposes controlled mono 16-kHz PCM
ingress for one streamed utterance. The caller supplies token spans around
the utterance and finalizes it before generating a response. Distributed
execution and bounded residency remain unimplemented for Audio.

A CPU loader check uses the pinned Audio checkpoint with the weight file and
asset directory supplied separately. The selected language, encoder, depth
and codec bindings total 6,915,206,144, 461,112,576, 533,873,408 and
294,236,360 bytes respectively. After the configuration, processor,
generation defaults and tokenizer paths are removed from the temporary
asset directory, the selected model loads and commits token 13 for prompt
IDs `[124894,5,13,21,34]` with seed 41 and a one-slot text budget.

The released-checkpoint session check uses seed 41, a three-slot speech
budget, and prompt IDs `[124894,5,125031,125031,13,21,34]` before marker
insertion. The input is 2,151 samples of 413-Hz and 863-Hz sine waves
with amplitudes 0.12 and 0.05 at 16 kHz. Controlled advancement and iterator
draining produce identical events within each device on CPU and precise Metal.
Each run
commits three speech frames after six intermediate depth advances and emits
5,760 samples, including the final codec tail. Cancelling a separate session
after its first depth pass leaves zero committed slots.

The snapshot replay check uses the same fixture. It saves the first speech
frame after one depth pass, then saves the second after one pass with codec
carry from the first committed frame. On CPU and precise Metal, restoring
either boundary reproduces all three codebook frames and all 5,760 PCM
samples exactly within each device, including the final flush. A third
snapshot is rejected at the two-snapshot limit without charging a copy.
Dropping both handles releases all snapshot retention while preserving
cumulative copy usage. Explicit cancellation rejects restore.

The fork check uses the same fixture and forks after the first depth pass.
Interleaved parent and child advances reproduce all three codebook frames
and all 5,760 PCM samples exactly within each device on CPU and precise
Metal. After the first speech commit, child retention grows from 8,914,005
to 9,505,358 logical bytes. The branch count limit and cross-origin restore
reject before charging. A CPU run capped at the initial retention rejects
the child's first speech commit on both attempts, leaves its slot count at
zero, and lets the source publish that frame. Dropping the child releases
its retention and keeps cumulative copying charged.

The ingress check sends the same 2,151 samples as three chunks ending at
samples 399, 1,520, and 2,151. One projected speech position commits before
finish and a second enters the prompt on flush. The streamed and finite
prompts produce identical codebooks and all 5,760 PCM samples within each
device on CPU and precise Metal. Invalid sample rates and non-finite samples
leave progress unchanged. A separate CPU check reserves one audio position,
accepts its exactly aligned 1,520 samples, rejects an additional sample,
and finalizes with zero new mel frames before emitting a text token.

The native runner's `--ingress-replay` mode saves at 1,055 samples, before
the first audio position, and at 1,520 samples, after that position enters
the language cache. It restores each boundary and resubmits only the later
samples. On CPU and precise Metal, the restored response matches all three
speech frames and all 5,760 PCM samples from finite prompting within each
device. The response keeps the ingress copy budget, and dropping saved
ingress states releases their retention while preserving cumulative copy work.
Run it with the same checkpoint and reference directory as the conditional
reference case.

The same runner forks at 1,055 samples and advances the source and child
independently through finalization and speech generation. Both match all
three finite speech frames and all 5,760 PCM samples on CPU and precise
Metal. The check covers branch
limits and restore origins, retained growth during input, and the transfer
of the same branch slot and cumulative copy charges into the child's
response. Dropping the child releases its retention.

## LiveASR

The resident LiveASR route takes mono 16-kHz PCM or unpadded mel blocks and
publishes greedy English transcription tokens before the stream finishes.
A context restart keeps the encoder continuation and begins a fresh language prompt. The
publisher defines the cadence and closing-token policy. [Liquid AI LiveASR](https://huggingface.co/LiquidAI/LFM3-Audio-2B#usage-examples)

The LiveASR reference runner takes mono 16-kHz signed 16-bit WAV with more
than two full blocks of audio. It keeps the PCM and token traces outside the
checkout. The native runner applies Eredu's portable processor to the same
PCM and compares projected audio-position counts, closing status, and
generated IDs for every block.

CPU and precise Metal match all seven blocks and all 20 generated IDs from
the pinned Python reference. The fixture begins with seven mel frames that
complete no audio position, restarts the language context in the third
block, and flushes a 16-frame final block. Its mono WAV is generated by the
macOS Samantha voice reading “The quick brown fox jumps over the lazy dog.
The bright red boat sailed across the lake.” The WAV SHA-256 is
`9ad5d0e647cbbc25d9008c26a006805c4f79dcd136e7e5c28e5398978f9eacee`.
The LiveASR comparison measures block and token parity. The conditional
reference case above provides the full-vocabulary score comparisons.

The native runner's `--live-asr-pcm` option feeds 100 ms PCM chunks and lets
the session assemble its own mel blocks. It uses the same Python trace and
compares blocks that reach the transcript driver. CPU and precise Metal match
all six transcript blocks and all 20 generated IDs. The runner also checks
sample backpressure, finish, and rejection of further input.

Both native input paths compare replay after restore. The mel check saves
after the seven-frame block that completes no audio position. The PCM check
saves after the first published block with queued mel frames and a partial
PCM window, restores before replaying the next block, and repeats the final
flush from a second snapshot. It also checks the snapshot count limit,
release of retained storage, and cumulative copying across restore.

The mel comparison also forks after the initial seven frames and compares
the first transcript block in both continuations. It checks that the
branch limit and foreign snapshots are rejected before charging, and that
growth charges retention while preserving cumulative copy usage. The PCM
comparison forks before the final flush and closes the parent and child
independently with the same transcript.

```sh
python validation/lfm3_audio_reference.py /private/tmp/lfm3-audio /private/tmp/lfm3-live --live-asr-wav /private/tmp/speech.wav
cargo run -p eredu-backend-mlx --features audio --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-live /private/tmp/lfm3-live/cpu.json --live-asr --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features audio,metal --example lfm3_audio_reference -- /private/tmp/lfm3-audio /private/tmp/lfm3-live /private/tmp/lfm3-live/metal.json --live-asr --device gpu
```

Add `--live-asr-pcm` to the native command to compare the PCM path.
