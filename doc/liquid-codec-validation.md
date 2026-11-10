# LiquidCodec execution and acceptance

`eredu-codec::liquid` owns the released LiquidCodec declaration, neural encoder
and decoder, finite-scalar quantizer, and spectral analysis/reconstruction. It is
available without a concrete backend and has no language-model architecture
dependency. `Tokenizer` implements the shared `AudioTokenizer` contract using an
already bound codec and a caller-supplied neutral state factory. Finite calls and
caller-advanced decoding use the same neural and spectral equations.

The configuration uses 24-kHz mono PCM, FFT/window 1280, hop 320, six spectral
frames per 1920-sample codec frame, and twelve codebooks. Each codebook combines
scalar alphabets `[8,7,6,6]`, giving 2016 entries. Learned projections map width
256 to and from 48 scalar features. Encode uses bounded tanh, ties-to-even
rounding and exact mixed-radix I32 codes; decode rejects invalid code IDs.

The convolutional frontend feeds an eight-layer, width-512 causal backbone.
The decoder repeats each codec embedding across six spectral positions before
its eight-layer backbone. Both alternate width-seven gated convolution and
gated attention with sandwich normalization; decoder attention has a thirty-frame
window. Encoder and decoder state layouts independently declare convolution
carry and KV history. Layer declarations and `BackboneBlock` construction expose
individual parameter units without selecting a residency or distribution policy.
The strict SafeTensors schema includes canonical names, published aliases and
the fixed F32 synthesis window.

Analysis zero-pads the final PCM frame before reflecting 480 samples at each
spectral endpoint. F32 Hann coefficients use explicit range reduction and fused
polynomial arithmetic. Analysis selects the neutral host-F32 FFT mechanism;
on Metal this transfers windowed frames to the CPU and Cartesian results back
to the device. These transfers and temporary host arrays are real execution
costs. Synthesis uses the checkpoint's F32 window and native inverse FFT.
Streaming overlap-add retains a 960-sample tail for sample and envelope sums,
consumes the initial 480-sample trim once, and flushes the final 480 samples.
Empty intermediate chunks preserve neural state. Finite empty-code decoding
returns empty audio. A synchronous streaming error invalidates the tokenizer's
owned stream; native completion and transactional state publication belong to
the execution owner.

## Reference evidence

`validation/fixtures/lfm3/liquid-sources.json` pins the codec bundled with
`LiquidAI/LFM3-Audio-2B` revision
`2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1`, including publisher source hashes.
The original codec weight SHA-256 is
`45984a872fdcf0b94cd0ec68822bbf26f0c6fcb580cfaba0f7fa3ee60355137f`.
The whole-checkpoint exact F32 widening has SHA-256
`72fcfe81d104add75035d21012bde1f034facab4eeb6015e18d8ff677a45511f`.

The independent runner executes the complete publisher codec in F32 with eager
attention. Native acceptance executes full PCM encoding and finite/cached decoding
through `AudioTokenizer`, and compares encoder latent scores, all code IDs,
waveforms, streaming continuity and emitted sample counts. It also compares
spectral features, scalar projections and reconstruction from reference spectra.
Inputs include one-sample silence, a 2305-sample tone mixture and a 17397-sample
mixture. The last case crosses the decoder's attention window; silence includes
an empty intermediate chunk. Every stream flushes independently.

The preset maximum absolute tolerance is `5e-4`; code IDs and sample counts
require exact agreement. CPU and precise Metal acceptance results are retained
in `validation/fixtures/lfm3/liquid-acceptance-results.json`. All criteria pass.
The largest encoder latent errors are `5.007e-6` on CPU and `4.292e-6` on
Metal. Maximum finite/streamed PCM errors are `5.208e-6`/`6.333e-6` on CPU and
`5.186e-6`/`5.410e-6` on Metal.

This evidence covers F32 execution of exact widened publisher weights, not native
BF16 execution or an audio language-model task.

## Reproduce

Keep checkpoints, derived weights and detailed outputs outside the checkout.
The reference verifies immutable source and weight hashes before executing
publisher Python code. Package versions and native runtime revision are recorded
with the results. Metal commands require access to native devices.

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("LiquidAI/LFM3-Audio-2B", revision="2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1", local_dir="/private/tmp/lfm3-audio-source", allow_patterns=["*.py", "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors"])'
HF_HOME=/private/tmp/lfm3-hf HF_MODULES_CACHE=/private/tmp/lfm3-hf/modules python validation/lfm3_liquid_reference.py /private/tmp/lfm3-audio-source /private/tmp/lfm3-liquid-reference
cargo run -p eredu-backend-mlx --no-default-features --example lfm3_liquid_reference -- /private/tmp/lfm3-liquid-reference/f32-neural /private/tmp/lfm3-liquid-reference /private/tmp/lfm3-liquid-reference/cpu-results.json --device cpu --neural
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features metal --example lfm3_liquid_reference -- /private/tmp/lfm3-liquid-reference/f32-neural /private/tmp/lfm3-liquid-reference /private/tmp/lfm3-liquid-reference/metal-results.json --device gpu --neural
```

## Integration scope

Resident codec execution and explicit decode continuation are implemented. Public
prepared codec selection, GGUF admission, host-layerwise/disk-streamed adapters,
TP/PP construction, selected memory reports and controlled-session/capture
integration are not implemented. The independent units and caller-owned state
are building blocks for these paths; they do not establish support for them.
The LFM3 audio encoder, depth decoder, audio language-model feedback and omni
orchestration have no implementation here. No complete audio-family support is
advertised by this standalone codec acceptance.
