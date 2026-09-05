# eredu-codec

`eredu-codec` contains backend-neutral neural-audio architectures used with
Eredu's realtime speech models. Mimi checkpoint preparation validates the
released SafeTensors catalog and produces exact neutral parameter recipes.
`construct` materializes and atomically binds those recipes through any
general `ParameterBackend`, returning the ordinary `Mimi<B::Tensor>` type.

## Mimi

The `mimi` module implements the Mimi encoder, residual vector quantizer, and
decoder used by Moshi-family speech models. It supports:

- exact backend-neutral checkpoint admission and tensor-layout recipes;
- selecting an active subset of a checkpoint's codebooks;
- PCM-to-token and token-to-PCM conversion;
- latent-to-token and token-to-latent conversion; and
- stateful one-frame decoding for realtime playback.

```rust,no_run
use eredu_codec::mimi::Mimi;
use eredu_nn::Tensor;

fn round_trip<T: Tensor>(
    mimi: &mut Mimi<T>,
    pcm: &T,
    context: &T::Context,
) -> Result<T, eredu_codec::Error> {
    let tokens = mimi.encode(pcm, context)?;
    mimi.decode(&tokens, context)
}
```

Tensor shapes follow `[batch, channels, samples_or_frames]`. Audio capture,
playback, resampling, and device selection remain application concerns.

## Evaluation tools

Executable Mimi benchmarks and PersonaPlex evaluation entry points select a
backend's general parameter mechanisms and use this crate's neutral
constructor. This crate has no concrete-backend feature or accelerator
dependency. See the [PersonaPlex quantization evaluation
guide](https://github.com/jbg/eredu/blob/main/eredu-evaluation/doc/personaplex-quantization.md).

## License

Licensed under either Apache-2.0 or MIT.
