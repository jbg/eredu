# LiquidCodec component execution and comparison

`eredu-codec::liquid` contains configuration normalization, finite-scalar
quantization with learned input/output projections, and spectral analysis and
synthesis components. It is available without a concrete backend and has no
dependency on language-model architectures. This is not a complete neural codec
or LFM3 speech-model integration.

The released component configuration uses 24-kHz mono PCM, FFT/window 1280,
hop 320, six spectral frames per 1920-sample codec frame, and twelve codebooks.
Each codebook combines scalar alphabets `[8,7,6,6]`, giving 2016 entries. Learned
projections map width 256 to and from 48 scalar features. Encode uses bounded
tanh, ties-to-even rounding and exact mixed-radix I32 codes. Decode rejects
negative and out-of-range codes instead of applying negative-index wrapping.
Quantizer source admission retains the canonical projection names and the
published `quantizer.fsq` aliases.

Analysis zero-pads the final PCM frame before reflecting 480 samples at each
spectral endpoint. It forms log-magnitude and normalized Cartesian phase features.
Synthesis uses F32 polar spectra and inverse FFT, retaining sample and envelope
sums until the endpoint owner divides them. Streaming overlap-add retains a
960-sample tail for each sum, consumes the initial 480-sample trim once, and flushes
the final 480 samples before closing. Carry is independent of learned parameters.

## Reference scope and results

`validation/fixtures/lfm3/liquid-sources.json` pins the codec bundled with
`LiquidAI/LFM3-Audio-2B` revision
`2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1`, including publisher source hashes.
`audio_tokenizer/model.safetensors` has SHA-256
`45984a872fdcf0b94cd0ec68822bbf26f0c6fcb580cfaba0f7fa3ee60355137f`.
The exact F32 widening of the scalar projection weights has SHA-256
`b1747a7a014db24f70682e9ab0a4d845e14c18700bfdf4b7fee3e803082dc22e`.

The independent reference executes the complete publisher codec to produce
trained latents, codes and spectral predictions. Native comparison consumes
those intermediate inputs; it does not execute native neural encoder/decoder
backbones. Two tone-mixture inputs have 2305 and 17397 real PCM samples. The longer
case decodes sixty spectral frames, crossing the publisher decoder's thirty-frame
attention window. The reference's own streamed/uninterrupted waveform difference
is at most `2.951e-6`.

The preset maximum absolute tolerance is `5e-4`; code IDs and emitted sample counts
require exact agreement. Both CPU and precise Metal match all reference scalar
codes. Scalar decode, raw synthesis, finite waveform reconstruction, streaming
continuity and empty final flush meet the tolerance. Maximum finite/streamed PCM
differences are `1.789e-7`/`2.936e-6` on CPU and `7.004e-7`/`2.951e-6` on Metal.
The final flush emits 480 samples in each case.

**The combined component comparison fails.** Normalized spectral-analysis phase
features exceed the unchanged `5e-4` threshold: maximum errors are `0.1908–0.2862`
on CPU and `0.5201–0.5238` on Metal. This remains an analysis validation gap;
matching scalar codes from reference latents and matching synthesis do not
establish native end-to-end encoding parity. The compact result manifest records
the failing analysis rows and overall failure alongside the passing components.

Temporary numerical localization finds the largest phase differences in bins
whose reference magnitudes are below or near the `1e-5` phase denominator floor.
On the short input, log-magnitude maximum errors are `1.634e-6` on CPU and
`6.208e-6` on Metal. CPU FFT output matches PyTorch for identical windowed input;
the regenerated analysis window differs slightly in F32 arithmetic. Metal FFT
components on identical windowed input differ by at most `1.717e-5` in this case.
No phase tolerance is relaxed, and native neural encoder output/code parity from
native analysis features remains unestablished.

## Reproduce

Keep checkpoints, derived weights and detailed outputs outside the checkout.
The reference verifies immutable source and weight hashes before executing
publisher Python code. It needs the package versions recorded with the results.

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download("LiquidAI/LFM3-Audio-2B", revision="2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1", local_dir="/private/tmp/lfm3-audio-source", allow_patterns=["*.py", "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors"])'
HF_HOME=/private/tmp/lfm3-hf HF_MODULES_CACHE=/private/tmp/lfm3-hf/modules python validation/lfm3_liquid_reference.py /private/tmp/lfm3-audio-source /private/tmp/lfm3-liquid-reference
cargo run -p eredu-backend-mlx --no-default-features --example lfm3_liquid_reference -- /private/tmp/lfm3-liquid-reference/f32-component /private/tmp/lfm3-liquid-reference /private/tmp/lfm3-liquid-reference/cpu-results.json --device cpu
MLX_ENABLE_TF32=0 cargo run -p eredu-backend-mlx --features metal --example lfm3_liquid_reference -- /private/tmp/lfm3-liquid-reference/f32-component /private/tmp/lfm3-liquid-reference /private/tmp/lfm3-liquid-reference/metal-results.json --device gpu
```

The native runner returns failure when any component misses its declared
tolerance. Spectra used for its waveform comparison come from the reference
neural decoder. This evidence covers neither native BF16 nor complete native
codec encode/decode. Neural codec backbones/frontend, full artifact admission,
bounded/distributed construction, capture, controlled-session integration and
LFM3 audio/omni orchestration have no implementation here.
