# VoiceChat component references

## RNNT transcript presentation

`transcription-nvidia.json` executes the original `_rnnt_decode_text` method from
the same pinned NVIDIA wrapper and SHA256 described below. Nine cases compare
every accumulated prefix, covering word-marker replacement, leading/trailing and
internal whitespace, Unicode, Python information separators, punctuation, empty
pieces and literal unknown/byte-looking pieces. No SentencePiece runtime or RNNT
neural computation executes. Expected strings compare exactly.

```sh
python3 eredu-evaluation/scripts/voicechat_transcription_reference.py \
  --wrapper /private/tmp/eredu-voicechat-turn-reference.py \
  --output eredu-evaluation/fixtures/voicechat/transcription-nvidia.json
```

## RNNT host turn policy

`turn-nvidia.json` executes the original `_apply_rnnt_turn_taking` method from
NVIDIA Speech revision `097dfe9e2f55baf653b83035868bdc89849f1b47` on seventeen explicit
host states. The generator extracts the method with Python AST; a small
scalar/vector adapter supplies indexing, assignment, equality, `.item()`, `.any()`
and shape access. It does not reimplement turn policy or execute RNNT neural
decoding. The source file SHA256 is
`b50aeab4b36285cf041330b315e35a01788e1edb639380bb6105446787cc6a1a`.

```sh
python3 eredu-evaluation/scripts/voicechat_turn_reference.py \
  /private/tmp/eredu-voicechat-turn-reference.py \
  eredu-evaluation/fixtures/voicechat/turn-nvidia.json
```

The external source file is the pinned
[inference wrapper](https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py).
Cases include confirmed EOU, first-turn fallback, minimum and low-density speech
evidence, history-window boundaries, silence reset and barge-in. Expected agent
tokens, represented RNNT state and forced-EOS flags use exact comparison.
RNNT counters are supplied already updated; separate portable tests check their
first-prediction updates and float32 density accumulation. Optional ASR-head,
redirect and duration-watchdog paths are outside these fixtures. The fixture
records the source's EOS-after-BOS history ordering, including current BOS with
retained `agent_speaking=false` when an EOS remains in the preceding window.

## Backbone input and output boundaries

`backbone-nvidia.json` executes the pinned Speech implementation's `AddFusion`,
`DuplexSTTModel.forward`, first-feedback lookup and system-prompt preparation,
plus `NemotronHRMSNorm` from base revision
`6533e8de2c68e4536bf7c411d7a3ce5734111476`. A disclosed LLM stand-in returns the
final-normalized **supplied residual hidden state**. It does not execute attention,
Mamba or feed-forward blocks, or implement their caches. This fixture validates
the boundaries around the Nemotron backbone, not backbone parity.

```sh
python eredu-evaluation/scripts/voicechat_backbone_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  /path/to/base-6533e8de/modeling_nemotron_h.py \
  eredu-evaluation/fixtures/voicechat/backbone-nvidia.json
```

The base modeling file comes from the pinned base repository's `resolve` URL.
All executed source hashes are verified by the generator. Width four and
vocabulary 24 preserve the released PAD/BOS/EOS and tool-control IDs while keeping
the synthetic parameter tables small. Agent and function feedback use the same
embedding table. Ordered fusion adds agent text with weight one, user audio with
weight one, then function feedback with weight two. No separate user-text head
is selected. Both distinct bias-free heads consume the same final-normalized
hidden tensor, through one original forward invocation. Its raw function argmax
sequence is recorded independently.

The prompt cases execute the original preparation method with explicit tokenizer
result `[5, 7]`, producing `[1, 5, 7, 2]` in the user-audio channel and PAD 12 in
both feedback channels. Cases include one and two repetitions of the complete
wrapped sequence. The original first-feedback helper also selects PAD 12.
Tokenizer fidelity and external tool parsing are outside this fixture.

The float32 tolerance is fixed before Eredu comparison at absolute `2e-6` plus
relative `2e-5`; widening the same inputs and weights produces maximum difference
`1.7369699012448336e-7`. The original RMSNorm intentionally retains float32
accumulation in that comparison. The BF16 case rounds learned parameters and
casts float32 audio **before** fusion; casting after fusion differs by
`0.00390625`. BF16 fused intermediates compare exactly. Its head outputs agree
with float64 accumulation of the same BF16 inputs/weights rounded back to BF16;
the readout allowance is one BF16 ULP plus the float32 tolerance. All measurements
use PyTorch 2.14.0 CPU and establish no released-weight or end-to-end speech result.

## RNNT

`rnnt-nvidia.json` contains nonzero synthetic component weights and outputs from
the NVIDIA-NeMo/Speech revision
`097dfe9e2f55baf653b83035868bdc89849f1b47`. The generator verifies source SHA-256
hashes and executes the original ASTs of `RNNTDecoder.predict`, the RNNT joint
projection/combination methods, and the streaming wrapper's `_rnnt_step` and
`_rnnt_init_state`. Tiny PyTorch modules supply the released equations' embedding,
two-layer LSTM and biased linear operators. Adapters, training dropout and display
punctuation bias are disabled. The fixture is batch-one float32 CPU evidence;
it is not a full NeMo import or released-checkpoint inference result.

```sh
python eredu-evaluation/scripts/voicechat_rnnt_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/rnnt-nvidia.json
```

The recorded interpreter uses PyTorch 2.14.0. `parameters` uses the exact canonical
checkpoint names and flattened row-major values. Predictor states are flattened
`[layers, batch, hidden]`; the standalone `lstm_sequence` input/output uses
`[time, batch, hidden]`. PyTorch's gate order is input, forget, candidate, output.
Both stored bias vectors participate; construction-time forget-bias initialization
is not added again to loaded checkpoint values.

`perception-rnnt-nvidia.json` applies the same original RNNT methods to all six
unprojected streaming outputs in `perception-nvidia.json`. Its four-wide encoder
projection uses nonzero synthetic weights, two recurrent layers and the same
three-label vocabulary. The first frame emits three labels; the following five
frames retain the prediction on blank. The input fixture SHA-256 is retained in
provenance. PyTorch 2.13.0 CPU produces a maximum Float32-versus-Float64 difference
of `1.2518894130231217e-6` with identical decisions. The RNNT tolerance is
`2e-6` absolute plus `2e-5` relative, fixed before comparison with Eredu. Perception
outputs and caches use their own recorded tolerances. This paired component
fixture does not establish released-checkpoint or full-conversation parity.

```sh
python eredu-evaluation/scripts/voicechat_rnnt_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/perception-rnnt-nvidia.json \
  --perception-fixture eredu-evaluation/fixtures/voicechat/perception-nvidia.json
```

The streaming wrapper primes with two zero vectors: `predict(y=None)` creates
one zero vector, and `add_sos=True` prepends another. The last prediction is
retained. Every emitted label advances that prediction and both recurrent states;
blank retains them. The first joint decision determines frame-level speech
activity, even when the label loop later terminates on blank. A capped frame
retains the predictor after the last allowed emission and evaluates one additional
joint decision. The fixture uses an override of three emitted symbols per frame;
the pinned wrapper default is ten.

`streaming_frames` is one continuous four-frame sequence: initial blank, two
capped nonblank frames, then blank. `feedback_terminates_frame` is a separate run
from a fresh state that emits one symbol before feedback produces blank.
`predictor_steps` starts with two-zero priming, then tokens 0, 2 and 1.
`lstm_sequence` additionally exercises three steps with nonzero initial hidden
and cell values. `joint_calls.logits` records raw pre-softmax logits; the pinned
CPU joint subsequently applies log-softmax, preserving the argmax decisions.

Tolerance is fixed before comparison with Eredu: absolute `2e-6` plus relative
`2e-5`. Repeating the fixture with identical float32 weights and inputs widened
to float64 gives maximum absolute difference `2.640381628893351e-7` across the
LSTM states, predictions and raw joint logits, with identical greedy decisions.
The fixture records this measurement and the exact source hashes. No latency,
waveform, full-model, or distributed inference claim follows from these values.

## Streaming perception

`perception-nvidia.json` executes the original `ConvSubsampling`, `CausalConv2D`,
`ConformerLayer`, attention/position/convolution classes, and encoder mask method
from the same pinned NVIDIA revision. Only adapter/access bookkeeping is replaced
by inert mixins. Training dropout and autocast are disabled. Tiny nonzero weights
exercise both learned positional biases, normalization scales/biases, depthwise
convolution, and every causal subsampling stage.

```sh
python eredu-evaluation/scripts/voicechat_perception_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/perception-nvidia.json
```

The subsampling fixture has eight input features, two convolution channels, a
factor of eight and four output features. NVIDIA's causal padding is left two
and right one on **both** spatial axes of each stride-two convolution. The
frequency extents are therefore 8, 5, 3, 2. It retains the released cached schedule:
one feature column on the first call, then nine preceding columns plus eight new
columns per call, discarding the first two encoded rows. All six retained outputs
match the full 41-column invocation exactly in the recorded float32 execution.
This checks supplied features; the fixture does not compute host mel features or
execute the wrapper's rolling PCM buffer.

The Conformer fixture has width four, two heads, feed-forward width eight,
convolution kernel three and left attention context three. `use_bias=false`
applies to linear/convolution projections; learned layer-normalization biases
and relative attention biases remain present. `steps` partitions six input rows
as 1, 1, 2, 1, 1 and records each input, position tensor, attention/padding masks,
output and retained caches. Channel caches have upstream layout `[B, left, D]`;
convolution caches use `[B, D, kernel-1]`. Channel entries store normalized
attention inputs; convolution entries store GLU outputs. Initial channel slots
are masked until filled, and full caches retain their newest entries.

The `perception` section composes those same subsampling and Conformer parameters
with a deterministic nonzero four-to-six linear projection. Six one-row steps
record the unprojected encoder output, projected output and both retained caches.
The projection parameters use the canonical `stt_model.perception.proj` prefix;
the subsampling and block parameters remain in their component sections. The
coupled segmented/full maximum absolute difference is `2.384185791015625e-7`.

The recorded PyTorch 2.14.0 float32 segmented/full maximum absolute difference
is `9.834766387939453e-7`. Using identical float32 parameters and inputs widened
to float64 gives maximum difference `8.284204126574046e-7`, including output and
retained caches. Tolerance is fixed before Eredu comparison at absolute `3e-6`
plus relative `3e-5`. These synthetic component values establish no waveform,
released-weight, native distributed, or real-time latency result.

## Host mel windows and cadence selection

`mel-nvidia.json` executes the hash-verified `FilterbankFeatures` class and
`normalize_batch` method from the pinned NVIDIA revision with CPU PyTorch 2.14.0,
librosa 0.11.0 and NumPy 2.5.3. The input is nonzero synthetic PCM with exactly
representable sample values; no learned weights are involved.

```sh
python eredu-evaluation/scripts/voicechat_mel_reference.py \
  --checkout /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  --output eredu-evaluation/fixtures/voicechat/mel-nvidia.json
```

The filter uses a symmetric (`periodic=False`) 400-sample Hann window centered
within a 512-point FFT, centered reflection padding, 160-sample hops, pre-emphasis
0.97 before reflection, squared magnitudes, 128 area-normalized Slaney filters and
natural log with additive floor `2^-24`. Normalization `NA` performs no feature
normalization. Dither and feature padding are disabled by the cached wrapper.
The cases cover initial reflection, live right-edge reflection, retained-window
rollover and interior frames. The first pre-emphasized sample belongs to the
current retained window, so changing its origin can invalidate cached rows.

The `schedule` section runs the original `PerceptionCacheManager.step` method for
four 1,280-sample ticks with an explicit three-tick rolling buffer. A recording
encoder captures the selected feature rows and `drop_extra_pre_encoded`; identity
projection/adaptation and a CPU no-op for CUDA synchronization allow the slicing
method to run unchanged. It records the first one-column input, subsequent
17-column inputs with leading zeros where necessary, and the rollover clamp.
`padded_schedule` repeats these four calls with `pad_and_drop_preencoded=true`:
the first input has nine zero rows plus eight mel rows and drops two encoded
outputs; subsequent calls retain the same frame origins as the default schedule.
`bf16_schedule` supplies BF16-rounded PCM to the original manager, matching the
pinned native wrapper's default compute dtype before its F32 preprocessor cast.
It records four ticks with the default unpadded first phase. The F32 cases use
explicit precision overrides. This section checks host scheduling, not learned
encoder execution.

Absolute log-mel tolerance is `4e-4`, fixed before Eredu comparison. Widening the
same float32 PCM and filter/window tables to float64 in the independent NVIDIA
method yields maximum absolute difference `9.1552734375e-5`; log scaling exposes
FFT rounding at low-energy bins. BF16-ingress PCM yields a corresponding F32/F64
feature difference of `8.869171142578125e-5`, within the same bound. The fixture
records these precision measurements,
source hashes and tool versions. These values establish no released-checkpoint,
waveform quality, native distributed or real-time latency result.

## EAR-TTS mixture sampling and residual quantization

`mog-nvidia.json` executes original `RMSNorm`, `MLP`, `MLPLayer`, `MoGHead`,
`gumbel_like`, `get_masking_rate`, `depthsum_encoding_step`,
`RVQEARTTSModel.generate_step` and `depthsum_embedding` ASTs from the pinned
Speech revision. Matrix selection uses the source's original CPU fallback.
Top-p and top-k use original Transformers **4.56.0** ASTs, matching the pinned
Speech README environment. The generator verifies hashes of both source trees;
its inert docstring decorator does not alter warper equations.

```sh
python eredu-evaluation/scripts/voicechat_mog_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  /path/to/transformers-4.56.0/src/transformers/generation/logits_process.py \
  eredu-evaluation/fixtures/voicechat/mog-nvidia.json
```

The three MoG MLP layers retain the released offset RMSNorm and tanh-GELU
semantics. Tiny hidden/latent width four, intermediate width eight, rank two,
four mixtures and five entries per codebook bound fixture size. The actual
31-codebook, eight-slot cubic schedule has masked counts
`[31,31,31,31,30,29,26,22]` and depth counts `[0,0,0,1,1,3,4,22]`.
Only slots 3–7 execute MoG and consume random draws. Parameters use canonical
checkpoint names. The release case selects guidance `0.2`, top-p `0.95` and
noise `0.001`; a second case exercises top-k two without guidance and larger
noise. Explicit uniform and Gaussian values are supplied identically to both
precision runs; the fixture does not validate a backend RNG algorithm.

Each stage records unconditioned predictor outputs, post-MLP guided hidden,
mixture logits and low-rank means, filtering masks/probabilities, uniform and
Gumbel values, selected and expanded mean, clamped log deviation, Gaussian
values, latent sample, updated codes and ordered summed feedback. Filtered
negative-infinity logits appear as JSON `null` with a separate boolean mask.
Guidance combines the conditional lane first and unconditional lane second,
after the MoG MLP stack. Selected expanded means are multiplied by the
predicted standard deviation before adding the residual projection; Gaussian
noise receives that same deviation. The released `-4` log-deviation clamp and
actual removal by top-p are both exercised.

Tolerance is fixed before Eredu comparison at absolute `2e-5` plus relative
`5e-5`. Widening linear and RVQ arithmetic while retaining the original explicit
Float32 RMSNorm gives maximum absolute difference `2.5984129345957285e-6` with
identical selected mixtures and all RVQ codes. The source's RMSNorm conversion
means this is a mixed-precision comparison, not a fully Float64 reference.
The standalone depthsum cases additionally exercise first-index ties, suffix
quantization from a fresh residual, and zero feedback for the extra vocabulary
sentinel. They do not admit BOS/EOS indices as ordinary RVQ entries.

The released configuration disables the optional binary EOS head, so recorded
EOS/logit outputs are null. Wrapper-level silence replacement, turn policy,
Gemma, character conditioning, codec execution and released checkpoint parity
are outside this fixture.

## Fixed-mean codec residual quantization

`prvq-nvidia.json` executes the original `PreTrainedEMAVariance` and
`PreTrainedProbabilisticVQ` class ASTs from the pinned Speech revision. The
generator verifies the SHA256 of `ear_tts_vae_codec.py` before extracting them.
All random initialization is overwritten with deterministic nonzero synthetic
means and variances. Three depths, five entries and four latent channels bound
the fixture; duplicate first/second entries exercise first-minimum ties. The
released component has 31 depths, 1,024 entries and 512 latent channels.

```sh
python eredu-evaluation/scripts/voicechat_prvq_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/prvq-nvidia.json
```

The fixture records exact integer choices, the original encoder's final
residual, its `input - residual` quantized return, and ordered full/prefix
decoding. A Python profile callback captures the local residual without changing
the original equations. Unlike EAR-TTS's depthsum scoring, codec encoding keeps
the residual squared norm in every nearest-mean score. Variance parameters are
retained in their canonical scalar slots and are not used by encode/decode.

CPU PyTorch 2.14.0 produces a maximum Float32-versus-Float64 absolute delta of
`3.725290298461914e-8`, with identical discrete choices. Comparison tolerances
are fixed before Eredu comparison at absolute `1e-6` plus relative `1e-5`.
This is component evidence with synthetic parameters, not neural codec
decoder, startup/flush, or released-checkpoint parity evidence.

## Character static conditioning and gated fusion

`conditioning-nvidia.json` executes original `build_vocabs`,
`CharAwareSubwordEncoder.prepare_inputs`/`forward`, flag-embedding classes, and
`GatedProjectedSumRMSNorm` from the pinned Speech revision. Source hashes are
verified before AST extraction. The character backbone is a disclosed nonzero
affine stub; its supplied hidden states isolate character lookup, masking,
pooling, projection and scatter from the separately tested T5Gemma encoder.

```sh
python eredu-evaluation/scripts/voicechat_conditioning_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/conditioning-nvidia.json
```

The mixed-mask batch covers base EOS 12 versus a different live EOS token,
custom padding, out-of-vocabulary IDs, Unicode characters absent from the
single-character alphabet, unequal character lengths, and post-scatter flags
on unselected positions. Synthetic flag tables include nonzero regular rows;
execution must read these rows instead of assuming their initialization is
zero. The fusion uses nonzero projection biases, feature gates, scalar residual
scale, division by 31 and final offset RMSNorm. Its fourth positional constructor
argument is retained as in the source; it selects a truthy final norm while the
codebook divisor is the default 31.

CPU PyTorch 2.14.0 gives maximum absolute widening error
`1.4901161193847656e-7`. The widened run retains the original explicit Float32
RMSNorm casts. Tolerances are fixed before Eredu comparison at absolute `3e-6`
plus relative `3e-5`. This fixture does not establish released-weight parity.

## EAR-TTS causal and character transformers

`transformer-hf.json` runs actual Transformers **4.56.0** `Gemma3TextModel` and
`T5GemmaEncoderModel` modules in eager attention mode. The generator verifies
model/configuration, cache, mask, rotary and activation source hashes. Supplied
nonzero embeddings and deterministic tiny learned parameters bypass tokenizer
and embedding-table construction. The fixture does not establish released-weight
or complete speech-stack parity.

```sh
PYTHONPATH=/path/to/isolated-transformers-4.56 \
  python eredu-evaluation/scripts/voicechat_transformer_reference.py \
  eredu-evaluation/fixtures/voicechat/transformer-hf.json
```

The isolated package uses the pinned Speech README dependencies: Transformers
4.56.0, tokenizers 0.22.0 and huggingface-hub 0.34.4. The oracle uses PyTorch
2.14.0 CPU. Both models have hidden width eight, head width four, two query heads,
one KV head, feed-forward width twelve, and two layers (local then global).
The shortened local window is three, so six input positions cross the local
cache boundary. Other inference equation defaults use the pinned values, including
query pre-attention scalar 256, offset RMSNorm and tanh-GELU. Gemma uses local
RoPE base 10,000 and global base 1,000,000 with Q/K normalization; T5 uses base
10,000, no Q/K normalization, and attention logit softcap 50.

Gemma's six-token whole pass is compared with prefill two followed by cached
chunks of one, two and one tokens. Each call records post-rotary Q/K/V, additive
masks, attention probabilities, outputs and retained cache. The upstream dynamic
sliding cache retains `window - 1` keys for the next call while tracking the full
position count; global cache retains all keys. T5's two five-position character
rows have valid lengths five and two. Its original encoder multiplies supplied
embeddings by `sqrt(hidden_size)` and combines padding with bidirectional
local/global masks. Gemma leaves supplied embeddings unscaled.

Tolerance is fixed before Eredu comparison at absolute `2e-5` plus relative
`5e-5`. Whole versus cached Gemma output differs by at most `1.90734863e-6`.
Widening learned linear arithmetic while preserving the original explicit
Float32 normalization and rotary computations gives maximum output difference
`2.50339508e-6`. The distinction is retained in the fixture's comparison metadata.

## EAR-TTS ingress boundaries

`ingress-nvidia.json` executes the pinned Speech module's original
`RVQEARTTSModel.forward`, `_prepare_conditioning`, `depthsum_embedding`,
`GatedProjectedSumRMSNorm`, offset `RMSNorm`, and precision context. A disclosed
subword table supplies character-model outputs, and a backbone recorder returns
its input. The oracle isolates ingress equations; actual transformer execution
is covered separately by `transformer-hf.json`.

```sh
python eredu-evaluation/scripts/voicechat_ingress_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/ingress-nvidia.json
```

Tiny width four and five codebook entries retain all 31 codebook depths.
Nonzero canonical learned parameters include the two biased fusion projections,
gate, scalar residual scale, final normalization, BOS/null vectors and frozen
speaker projection. The audio mask `[false,true,true,false,true]` exercises
pre-BOS speaker substitution and two BOS transitions. Cases include a supplied
retained speaker latent, the original computed frozen projection, recurrent
unshifted code, and fully masked subword conditioning. Initial code shifting
pads with code **zero**, not the PAD sentinel. The conditional lane precedes the
null-conditioned lane. The original gated fusion divides audio by 31 and scales
the entire projected mixture by the sigmoid residual scalar before final norm.

Inputs, pre-fusion audio/conditioning, duplicated masks/positions, recorder
inputs, and returned speaker latent are recorded. Absolute `2e-5` plus relative
`5e-5` tolerance is fixed before Eredu comparison; wider linear arithmetic with
the original Float32 normalization gives maximum output difference
`3.57627869e-7`. This fixture covers no wrapper EOS/silence policy, tokenizer,
character transformer, generated codes, codec or released-weight parity.

## EAR neural decoder and streaming spectral reconstruction

`codec-nvidia.json` executes hash-verified original `Latent2Wav`, `ConvNeXt1d`,
`LayerNormNd`, `CausalConv1dCache` and `spec_to_wav` ASTs. All learned parameters
are synthetic and nonzero, including ConvNeXt final projections that otherwise
initialize to zero. It covers the neural decoder and spectral reconstruction,
not released-weight inference or codec-encoder parity.

```sh
python eredu-evaluation/scripts/voicechat_codec_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/codec-nvidia.json
```

The compact decoder uses latent width four, base width two, multipliers `[2,1]`,
rates `[3,2]`, two ConvNeXt blocks per stage, kernel three, FFT eight and hop two.
Six latent frames are submitted as `[1,2,1,2]`. A separate case preserves the
released reversed rates `[9,7,7]`, three stages/blocks, FFT 16 and hop four with
tiny widths: four latent frames submitted as `[1,2,1]` each produce the released
**1,764 samples per frame**. Raw spectrogram projections, ordinary/streaming
waveforms and every convolution/spectral cache are recorded. The compact case
also records magnitude, phase, padded complex spectrum and overlap envelope.

Upsampling has kernel equal to stride and no overlap. ConvNeXt uses exact
(erf) GELU and channel LayerNorm. Magnitude is bounded using the original
softplus/exponential expression; DC and Nyquist have real cosine phase only.
Inverse FFT uses periodic Hann, constrains each reconstructed frame to the
positive/negative window before window multiplication when requested, then
normalizes overlap sums by the squared-window envelope. Direct nonzero complex
spectra exercise both constrained and unconstrained reconstruction.

Streaming retains four complex frames in both configurations. The compact
startup delay is four samples; the released-rate delay is eight. Flushing adds
that many samples and clears every original cache. After removing startup delay,
streaming and ordinary interior samples agree within `4.20e-9`, while edge
normalization differs by up to `0.00520` in these inputs. The fixture records
this original boundary behavior; it does not claim exact offline edge parity.

Absolute `2e-5` plus relative `5e-5` tolerance is fixed before Eredu comparison.
Float32 versus Float64 neural output differs by at most `2.84e-7`; original
Float32 spectral reconstruction of those neural outputs differs by at most
`9.46e-8`. Independently widening inverse FFT while retaining the original
Float32 Hann window changes the direct reconstruction by at most `7.15e-8`.
No waveform quality, latency or peak-residency claim follows from tiny fixtures.

The `composition` case executes original `PreTrainedProbabilisticVQ.decode`
with three tiny codebooks before the compact decoder, yielding an independent
code-to-waveform expectation. It records both ordinary streaming calls and a
last-call flush. Their common final-call prefix is exactly equal; the remaining
four samples form `finish_tail`, validating a drain without invented code input.
The unflushed convolution and spectral carries remain available for snapshot,
fork and failed-transition retry comparisons.

## Composed cached EAR-TTS transition

`tts-transition-nvidia.json` executes original NVIDIA
`RVQEARTTSModel.forward`, `_prepare_conditioning`, `generate_step`, depth-sum
embedding/quantization and MoG methods around actual Transformers 4.56.0
`Gemma3TextModel`. The original `DuplexEARTTS.infer_codes_one_step` controls
recurrent input. Source hashes cover NVIDIA methods, the HF model, cache,
masking, rotary, activations and top-p implementation.

```sh
PYTHONPATH=/path/to/isolated-transformers-4.56 \
  python eredu-evaluation/scripts/voicechat_tts_transition_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/tts-transition-nvidia.json
```

Five prompt positions warm both conditional and unconditional caches without
sampling. Initial feedback is the unshifted final prompt code. Three recurrent
steps cover ordinary text, live EOS and completely masked character input.
The live EOS step replaces prior feedback with the supplied silence frame
before embedding. Raw generated codes remain independently recorded; the
fixture does not impose streaming-facade feedback or waveform policy.
A separate original-source case replaces each special speech code with the
corresponding silence-codebook entry.

Each recurrent step records five active refinement slots, their exact uniform
and Gaussian draws, mixture decisions, intermediate latents and partial codes.
All calls record fused input, final hidden states and both guidance lanes of
retained local/global K/V state. Parameters are nonzero synthetic tiny values:
hidden width eight, latent width four, two transformer layers and 31 codebooks
with five entries. A supplied embedding table explicitly replaces character
encoding; the independent character and T5Gemma fixtures cover that boundary.
This fixture does not establish released-weight or complete VoiceChat parity.

CPU PyTorch 2.14.0 compares Float32 execution against widened linear/codebook
arithmetic while preserving source explicit Float32 normalization and rotary
operations. The maximum absolute difference across recorded floating-point
outputs and caches is `3.42310228e-5`; mixture and code decisions are identical.
Comparison tolerance is fixed before Eredu evaluation at absolute `5e-5`
plus relative `5e-5`. Draws are rounded once to Float32 and supplied identically
in both executions.

## Character-conditioned cached EAR-TTS transition

`tts-character-transition-nvidia.json` uses the cached TTS generator script's
`--character-encoder` mode. It replaces the supplied embedding boundary with
original NVIDIA character mapping, pooling, continuation/special flags and an
actual two-layer Transformers 4.56.0 T5Gemma encoder. The fixture includes exact
vocabulary and saved flag tables. Its character local window spans 16 positions;
the separate transformer fixture exercises shorter-window attention boundaries.
Five prompt positions precede ordinary, EOS and PAD cached steps. Live masks
remain true, and explicit main-path EOS feedback replacement precedes the next
step. Recorded code generation uses original NVIDIA equations and identical
Float32-rounded random draws across precision runs.

```sh
PYTHONPATH=/path/to/isolated-transformers-4.56 \
python eredu-evaluation/scripts/voicechat_tts_transition_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/tts-character-transition-nvidia.json \
  --character-encoder
```

The independent Float32-versus-widened maximum absolute error is
`6.43021943e-5`, with identical mixture and code decisions. Comparison tolerance
is fixed before Eredu evaluation at absolute `1e-4` plus relative `5e-5`.
Parameters are nonzero synthetic values; this is neither released-checkpoint
nor complete conversational validation. Default script mode retains the original
`tts-transition-nvidia.json` supplied-conditioning oracle.

## EAR waveform encoder

`codec-encoder-nvidia.json` executes the pinned codec's original `spectrogram`,
`Wav2Latent`, `ConvNeXt1d` and `PreTrainedProbabilisticVQ` methods. It verifies
manual zero padding, periodic Hann analysis and real-bin/imaginary-bin channel
concatenation before the neural input projection. No magnitude/phase conversion
occurs in this encoder. Direct spectral cases cover odd and even FFT sizes and
asymmetric left/right padding.

```sh
python eredu-evaluation/scripts/voicechat_codec_encoder_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/codec-encoder-nvidia.json
```

The compact batch combines all-zero carrier PCM and a nonzero waveform, with
nonzero learned weights and biases. Every learned layer output, final latent,
residual-quantizer codes and decoded latent are recorded. A separate tiny-width
case preserves released FFT size 16, hop 4 and downsampling rates 7, 7, 9,
producing one code frame per 1,764 samples. Both cases use three small codebooks;
they establish finite-input equations, not released-weight or cached-encoder
parity. The pinned `Wav2Latent` explicitly rejects cache use.

The original spectral boundary always casts to Float32. Widening only learned
arithmetic after that boundary changes recorded neural outputs by at most
`3.92724422e-7`, with identical discrete codes. Float64 STFT with the original
Float32 Hann values differs from Float32 STFT by at most `1.84973610e-7`.
Tolerances are fixed before Eredu comparison at absolute `2e-5` plus relative
`5e-5`. The neutral test implements a separate scalar DFT with Float64
accumulation; the native test uses the MLX real FFT mechanism.

## Fixed-Aria prompt preparation

`aria-prompt-nvidia.json` executes the original pinned `DuplexEARTTS`
`set_init_inputs`, `get_init_inputs` and `get_audio_prompt_latent` methods.
The codec returns explicitly supplied discrete codes and records the requested
silent carrier; the Aria table contains supplied tiny values. This boundary
isolates preparation policy and does not establish waveform encoding or
released-weight parity.

```sh
python eredu-evaluation/scripts/voicechat_aria_prompt_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/aria-prompt-nvidia.json
```

The released geometry uses a three-second prompt, 22,050 Hz and 1,764 samples
per code frame. The original method requests 38 silent carrier frames and
retains 37 prompt frames. Speech PAD occupies rows 0 and 36; the audio mask
selects only the final retained row, while the subword mask selects the final
two. Text PAD fills the first 36 positions and EOS fills the final position.
The fixture also covers compact geometry, retained-buffer clone independence,
and batch-one expansion to three rows. Integer, Boolean and supplied speaker
values compare exactly; there is no floating-point numerical tolerance.

## Agent-text sampling

`agent-sampling-nvidia.json` executes the original `ModelInterface._sample_text_token`
method from the pinned NVIDIA source. The script verifies its source hash before
extracting the method AST. Only `torch.multinomial` is replaced: the replacement
records the original final probabilities and selects their inverse CDF with a
supplied uniform. This validates policy independently of accelerator RNG streams
and does not execute model weights.

```sh
python eredu-evaluation/scripts/voicechat_agent_sampling_reference.py \
  /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  eredu-evaluation/fixtures/voicechat/agent-sampling-nvidia.json
```

The ten cases cover streaming greedy defaults, the identity-policy greedy path,
zero-temperature bypass, original-logit special-token bypass, unique sign-aware
repetition penalties excluding BOS/EOS/PAD, nucleus threshold crossing, retained
history prefixes, and effective forced feedback. A single surviving nucleus token
still consumes a draw; the greedy paths consume none. Float32 versus Float64
probabilities differ by at most `5.33377536e-8`, with identical supplied-uniform
decisions. Probability tolerances are fixed at absolute `2e-7` plus relative `2e-6`;
token IDs and draw counts compare exactly.

## Prepared tool silence

`tool-silence-nvidia.json` executes the pinned NVIDIA `FilterbankFeatures.forward`,
causal subsampling, relative-position Conformer, and the wrapper's original
`_create_silence_embedding_cache` method. One second of zero PCM yields 101 valid
centered mel rows and 14 causal subsampling rows. The wrapper retains the first
projected row; the fixture records that embedding in Float32 and after its final
BF16 cast. Learned computation uses Float32 with compact deterministic nonzero
parameters (128 input features, two subsampling channels, width four, two heads,
and projected width six).

```sh
python eredu-evaluation/scripts/voicechat_tool_silence_reference.py \
  --checkout /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47 \
  --wrapper /path/to/Speech-097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py \
  --output eredu-evaluation/fixtures/voicechat/tool-silence-nvidia.json
```

The generator requires CPU PyTorch, verifies every upstream source hash, and
imports the perception oracle's source extraction and initialization
helpers. It initializes the preprocessing module's buffers directly, retaining
its original forward method. The exported symmetric Hann window and finite
one-hot filter bank are shared exactly with numerical consumers. Filter values
cannot change zero PCM's zero power spectrum; these synthetic filters do not
claim parity with released filter coefficients. This avoids an optional librosa
dependency without replacing NVIDIA's FFT, log-floor, masking, or length logic.

With PyTorch 2.13.0, widening the same Float32 parameters and buffers to Float64
produces maximum absolute difference `1.852865189055919e-7` across mel, encoder,
and projected outputs. The fixed tolerance is absolute `3e-6` plus relative
`3e-5`, established before Eredu comparisons. The BF16 fixture covers the final
wrapper cast only. This is independent synthetic component evidence, not
released-weight, full-conversation, distributed, or latency validation.
