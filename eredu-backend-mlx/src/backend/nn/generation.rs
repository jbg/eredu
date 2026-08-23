//! Causal language-model generation traits, sampling, and iterators.

use safemlx::{
    argmax_axis, array,
    error::Exception,
    random::{self, RandomState},
    Array, Stream,
};

/// Samples a token id from logits.
///
/// A temperature of `0.0` uses greedy argmax; non-zero temperatures use
/// categorical sampling and require `prng_state`.
pub fn sample(
    logits: &Array,
    temp: f32,
    prng_state: Option<&mut RandomState>,
    stream: &Stream,
) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1, stream = stream),
        _ => {
            let prng_state = prng_state.ok_or_else(|| {
                Exception::custom("random operations require an explicit PRNG key")
            })?;
            let key = prng_state.next_key(stream)?;
            let logits = logits.multiply(array!(1.0 / temp), stream)?;
            random::categorical(&logits, None, None, &key, stream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sample;
    use safemlx::{Array, Device, DeviceType, ExecutionContext};

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn non_greedy_sample_requires_prng_key() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let logits = Array::from_slice(&[0.0f32, 1.0], &[1, 2]);

        let error = sample(&logits, 1.0, None, ctx.stream()).unwrap_err();

        assert!(error
            .to_string()
            .contains("random operations require an explicit PRNG key"));
    }
}
