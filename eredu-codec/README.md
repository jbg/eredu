# eredu-codec

`eredu-codec` contains backend-neutral neural-audio architectures used with
Eredu's realtime speech models. Backends construct `Mimi<T>`, use
`checkpoint_tensor_plan` to map released checkpoint tensors, and populate the
model through `Mimi::load_parameters`.

## Mimi

The `mimi` module implements the Mimi encoder, residual vector quantizer, and
decoder used by Moshi-family speech models. It supports:

- backend-neutral checkpoint name and tensor-layout planning;
- selecting an active subset of a checkpoint's codebooks;
- PCM-to-token and token-to-PCM conversion;
- latent-to-token and token-to-latent conversion; and
- stateful one-frame decoding for realtime playback.

```rust,ignore
use eredu_codec::{mimi::{Config, Mimi}, AudioTokenizer};

let mut mimi = Mimi::new(Config::v0_1(Some(8)), context)?;
// Load backend-native tensors with `Mimi::load_parameters`.
let tokens = mimi.encode(&pcm, stream)?;
let reconstructed = mimi.decode(&tokens, stream)?;
```

Tensor shapes follow `[batch, channels, samples_or_frames]`. Audio capture,
playback, resampling, and device selection remain application concerns.

## Evaluation tools

Concrete backend integrations own executable Mimi benchmarks and PersonaPlex
evaluation tools. The MLX integration and runnable examples live in
`eredu-backend-mlx` behind its `codec` feature; `eredu-codec` itself has no MLX
feature or accelerator dependency. See the [PersonaPlex quantization evaluation
guide](https://github.com/jbg/eredu/blob/main/doc/personaplex-evaluation.md).

## License

Licensed under either Apache-2.0 or MIT.
