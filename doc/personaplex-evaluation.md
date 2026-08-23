# PersonaPlex quantization evaluation

The `personaplex_quantization_eval` example compares dense and quantized
PersonaPlex checkpoints on the same recorded input. It reports realtime
deadline behavior, teacher-forced text and audio distribution drift,
free-running token divergence, and a blinded pair of decoded responses.

This is an evaluation aid, not a universal quality score. Listening results,
input coverage, checkpoint compatibility, and the target deployment hardware
all matter.

## Prepare audio

The evaluator expects mono 24 kHz raw `f32le` PCM for the user audio and voice
prompt. FFmpeg can produce the files:

```sh
ffmpeg -i input.wav -f f32le -ac 1 -ar 24000 /tmp/input.f32le
ffmpeg -i voice-prompt.wav -f f32le -ac 1 -ar 24000 /tmp/voice-prompt.f32le
```

PersonaPlex requires both voice and text conditioning. Omitting either can
produce codec noise rather than meaningful speech.

## Run one comparison

```sh
cargo run --release -p eredu-backend-mlx --features codec \
  --example personaplex_quantization_eval -- \
  /path/to/personaplex-dense \
  /path/to/personaplex-quantized \
  /path/to/tokenizer.safetensors \
  /path/to/tokenizer_spm_32k_3.model \
  /tmp/voice-prompt.f32le \
  /tmp/input.f32le \
  /tmp/personaplex-eval \
  128
```

The final required argument is the maximum number of 80 ms frames. Optional
arguments can override the assistant text prompt and sampling seed.

The output directory contains:

- `metrics.json` with performance and drift measurements;
- `input.wav` and codec round-trip diagnostics;
- randomized `sample_a.wav` and `sample_b.wav` responses;
- `listening_manifest.json` and a separate `answer_key.json`; and
- `token_diagnostics.json` with conditioning and generated token traces.

Teacher-forced metrics run both models on the dense model's token history so
their distributions are comparable. Listening samples are independent
free-running generations. The evaluator warns when the selected frame limit
cuts off active input or generated speech.

## Run a suite

The suite runner expands cases across seeds, launches each trial in a fresh
process, aggregates metrics, and creates one blinded listening manifest. A
suite file has this shape:

```json
{
  "format_version": 1,
  "dense_model": "/path/to/personaplex-dense",
  "quantized_model": "/path/to/personaplex-q4",
  "mimi": "/path/to/tokenizer.safetensors",
  "text_tokenizer": "/path/to/tokenizer_spm_32k_3.model",
  "voice_prompt": "/path/to/voice-prompt.f32le",
  "sampling_seeds": [20260713, 20260714],
  "cases": [
    {
      "id": "procedural_question",
      "category": "procedural",
      "input": "/path/to/input-mono-24khz.f32le"
    }
  ]
}
```

Run and summarize it with:

```sh
python eredu-codec/scripts/personaplex_quantization_suite.py run \
  suite.json /tmp/personaplex-quantization-suite

python eredu-codec/scripts/personaplex_quantization_suite.py summarize \
  /tmp/personaplex-quantization-suite \
  /tmp/personaplex-quantization-suite/human_ratings.json
```

The runner rejects silent inputs and byte-identical cases by default. Override
those checks per case only when silence or duplication is intentional. Fill in
the generated ratings file before opening answer keys.

## Compare with a PyTorch backend

`token_diagnostics.json` contains the exact conditioning tokens and dense
traces needed to remove tokenizer and encoder differences from a backend
comparison. The included reference runner consumes that file:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 \
PYTHONPATH=/path/to/upstream/moshi:/path/to/python/dependencies \
python eredu-codec/scripts/personaplex_pytorch_backend_reference.py \
  --moshi-source /path/to/upstream/moshi \
  --model /path/to/personaplex/model.safetensors \
  --mimi /path/to/tokenizer.safetensors \
  --tokenizer /path/to/tokenizer_spm_32k_3.model \
  --eredu-eval-dir /tmp/personaplex-eval \
  --output-dir /tmp/personaplex-backend-comparison \
  --device mps
```

The runner produces a short greedy parity trace and another blinded sampled
pair. MLX and PyTorch use different random-number implementations, so sampled
tokens are not expected to match even with the same seed. PyTorch MPS may also
fall back to CPU for cache updates; treat its timing as diagnostic rather than
a direct backend benchmark.
