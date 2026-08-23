//! MLX materialization of PersonaPlex conditioning frames.

use safemlx::{error::Exception, ops::broadcast_to, Array, Stream};

pub use eredu_architectures::moshi::personaplex_prompt::{AUDIO_TOKENS_PER_STREAM, SINE_TOKENS};

/// Creates a repeated sine-conditioning frame shaped `[batch, 8]`.
pub fn sine_frame(batch: i32, stream: &Stream) -> Result<Array, Exception> {
    broadcast_to(
        Array::from_slice(&SINE_TOKENS, &[1, AUDIO_TOKENS_PER_STREAM as i32]),
        &[batch, AUDIO_TOKENS_PER_STREAM as i32],
        stream,
    )
}
