# safemlx-codec

`safemlx-codec` is the MLX neural-audio implementation used by Eredu's realtime
speech models. Applications should normally access it through
[`eredu`](../eredu/); direct use remains available for codec-only pipelines
that do not need the model runtime.

## Mimi

The `mimi` module implements the Mimi encoder, residual vector quantizer, and
decoder used by Moshi-family speech models. It supports:

- SafeTensors checkpoint loading;
- selecting an active subset of a checkpoint's codebooks;
- PCM-to-token and token-to-PCM conversion;
- latent-to-token and token-to-latent conversion; and
- stateful one-frame decoding for realtime playback.

```rust,ignore
use safemlx_codec::{mimi::Mimi, AudioTokenizer};

let mut mimi = Mimi::load("/path/to/tokenizer.safetensors", Some(8), stream)?;
let tokens = mimi.encode(&pcm, stream)?;
let reconstructed = mimi.decode(&tokens, stream)?;
```

Tensor shapes follow `[batch, channels, samples_or_frames]`. Audio capture,
playback, resampling, and device selection remain application concerns.

## Evaluation tools

The crate includes an example and suite runner for comparing dense and
quantized PersonaPlex checkpoints with teacher-forced metrics, realtime
deadlines, and blinded listening samples. See the [PersonaPlex quantization
evaluation guide](../doc/personaplex-evaluation.md).

## License

Licensed under either Apache-2.0 or MIT.
