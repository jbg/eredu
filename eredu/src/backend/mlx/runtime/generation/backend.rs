//! SafeMLX realization of backend-neutral token-sampling primitives.

use std::collections::HashMap;

use eredu_runtime::{PenaltyConfig, SamplingBackend, TokenDomain};
use safemlx::{
    argmax_axis, array,
    error::Exception,
    ops::indexing::TryIndexOp,
    random::{self, RandomState},
    Array, Dtype, Stream,
};

use crate::{backend::mlx::nn::tensor::validate_token_domain, core::TokenFilter};

/// SafeMLX token-sampling capability implementation.
#[derive(Debug, Clone, Copy)]
pub struct MlxSamplingBackend;

impl SamplingBackend for MlxSamplingBackend {
    type Logits = Array;
    type Token = Array;
    type RandomState = RandomState;
    type Context = Stream;
    type Error = Exception;

    fn error(message: String) -> Self::Error {
        Exception::custom(message)
    }

    fn validate_token(
        token: &Self::Token,
        domain: TokenDomain,
        stream: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        if !matches!(token.dtype(), Dtype::Int32 | Dtype::Uint32) {
            return Err(Exception::custom(format!(
                "token IDs must use int32 or uint32 storage, got {:?}",
                token.dtype()
            )));
        }
        let cardinality = i32::try_from(domain.cardinality())
            .map_err(|_| Exception::custom("token domain exceeds MLX int32 range"))?;
        if cardinality <= 0 {
            return Err(Exception::custom("token domain must be non-empty"));
        }
        validate_token_domain(token, cardinality, None, stream)
    }

    fn scale_temperature(
        logits: &Array,
        temperature: f32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        logits.multiply(array!(1.0 / temperature), stream)
    }

    fn apply_penalties(
        logits: &Array,
        history: &[u32],
        penalties: PenaltyConfig,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if history.is_empty() || penalties.is_identity() {
            return Ok(logits.clone());
        }

        let vocab_size = logits.dim(-1) as usize;
        if vocab_size == 0 {
            return Ok(logits.clone());
        }
        let row_count = logits.size() / vocab_size;
        let mut repeat_mask = vec![false; logits.size()];
        let mut additive = vec![0.0f32; logits.size()];
        let start = if penalties.repeat_last_n < 0 {
            0
        } else {
            history
                .len()
                .saturating_sub(penalties.repeat_last_n as usize)
        };
        let mut counts = HashMap::<u32, usize>::new();
        for &token in &history[start..] {
            *counts.entry(token).or_default() += 1;
        }
        for (token, count) in counts {
            let token = token as usize;
            if token >= vocab_size {
                continue;
            }
            for row in 0..row_count {
                let index = row * vocab_size + token;
                repeat_mask[index] = true;
                additive[index] =
                    penalties.frequency_penalty * count as f32 + penalties.presence_penalty;
            }
        }

        let mut adjusted = logits.clone();
        if penalties.repeat_penalty != 1.0 {
            let mask = Array::from_slice(&repeat_mask, logits.shape());
            let positive = adjusted.divide(array!(penalties.repeat_penalty), stream)?;
            let negative = adjusted.multiply(array!(penalties.repeat_penalty), stream)?;
            let penalized = safemlx::ops::r#where(
                adjusted.gt(Array::from_f32(0.0), stream)?,
                positive,
                negative,
                stream,
            )?;
            adjusted = safemlx::ops::r#where(mask, penalized, adjusted, stream)?;
        }
        if penalties.frequency_penalty != 0.0 || penalties.presence_penalty != 0.0 {
            adjusted = adjusted.subtract(Array::from_slice(&additive, logits.shape()), stream)?;
        }
        Ok(adjusted)
    }

    fn apply_top_k(logits: Array, top_k: i32, stream: &Stream) -> Result<Array, Exception> {
        let vocab_size = logits.dim(-1);
        if top_k <= 0 || top_k >= vocab_size {
            return Ok(logits);
        }
        let top_values = safemlx::ops::indexing::topk_axis(&logits, top_k, -1, stream)?;
        let threshold = top_values.min_axis(-1, true, stream)?;
        mask_logits(logits.lt(threshold, stream)?, logits, stream)
    }

    fn apply_top_p(logits: Array, top_p: f32, stream: &Stream) -> Result<Array, Exception> {
        if top_p >= 1.0 {
            return Ok(logits);
        }
        let descending = safemlx::ops::argsort_axis(logits.negative(stream)?, -1, stream)?;
        let sorted = safemlx::ops::indexing::take_along_axis(&logits, &descending, -1, stream)?;
        let probabilities = safemlx::ops::softmax_axis(&sorted, -1, true, stream)?;
        let cumulative = probabilities.cumsum(-1, None, None, stream)?;
        let before = cumulative.subtract(probabilities, stream)?;
        let masked = mask_logits(
            before.gt(Array::from_f32(top_p.max(0.0)), stream)?,
            sorted,
            stream,
        )?;
        let fill = Array::full::<f32>(
            logits.shape(),
            Array::from_f32(logits.dtype().finfo_min()? as f32),
            stream,
        )?
        .as_dtype(logits.dtype(), stream)?;
        safemlx::ops::indexing::put_along_axis(&fill, &descending, &masked, -1, stream)
    }

    fn apply_min_p(logits: Array, min_p: f32, stream: &Stream) -> Result<Array, Exception> {
        if min_p <= 0.0 {
            return Ok(logits);
        }
        let probabilities = safemlx::ops::softmax_axis(&logits, -1, true, stream)?;
        let maximum = probabilities.max_axis(-1, true, stream)?;
        let threshold = maximum.multiply(Array::from_f32(min_p), stream)?;
        mask_logits(probabilities.lt(threshold, stream)?, logits, stream)
    }

    fn apply_token_filter(
        logits: &Array,
        filter: &TokenFilter,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let Some(allowed) = filter.allowed_mask() else {
            return Ok(logits.clone());
        };
        let vocab_size = logits.dim(-1) as usize;
        let allowed = effective_allowed_mask(allowed, vocab_size).map_err(Exception::custom)?;
        let row_count = logits.size() / vocab_size;
        let invalid = (0..row_count)
            .flat_map(|_| allowed.iter().map(|allowed| !allowed))
            .collect::<Vec<_>>();
        mask_logits(
            Array::from_slice(&invalid, logits.shape()),
            logits.clone(),
            stream,
        )
    }

    fn apply_mirostat(
        logits: &Array,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        mu: f32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let vocab_size = logits.dim(-1) as usize;
        if vocab_size == 0 || logits.size() / vocab_size != 1 {
            return Err(Exception::custom(
                "Mirostat V2 currently requires logits for exactly one sequence",
            ));
        }
        let logits = Self::apply_penalties(logits, history, penalties, stream)?;
        let scaled = Self::scale_temperature(&logits, temperature, stream)?;
        let probabilities = safemlx::ops::softmax_axis(&scaled, -1, true, stream)?;
        let cutoff = Array::from_f32((-mu).exp2());
        let maximum = probabilities.max_axis(-1, true, stream)?;
        let cutoff_mask = probabilities.lt(&cutoff, stream)?;
        let best =
            argmax_axis!(&probabilities, -1, stream = stream)?.expand_dims_axes(&[-1], stream)?;
        let fallback = Array::full::<bool>(logits.shape(), Array::from_bool(true), stream)?;
        let keep_best = Array::full::<bool>(best.shape(), Array::from_bool(false), stream)?;
        let fallback =
            safemlx::ops::indexing::put_along_axis(&fallback, &best, &keep_best, -1, stream)?;
        let mask =
            safemlx::ops::r#where(cutoff.gt(maximum, stream)?, fallback, cutoff_mask, stream)?;
        mask_logits(mask, scaled, stream)
    }

    fn sample_raw(
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if temperature == 0.0 {
            argmax_axis!(logits, -1, stream = stream)
        } else {
            let scaled = Self::scale_temperature(logits, temperature, stream)?;
            sample_categorical(&scaled, random, stream)
        }
    }

    fn sample_processed(
        logits: &Array,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if temperature == 0.0 {
            argmax_axis!(logits, -1, stream = stream)
        } else {
            sample_categorical(logits, random, stream)
        }
    }

    fn token_id(token: &Array, stream: &Stream) -> Result<u32, Exception> {
        Ok(token.clone().item::<u32>(stream))
    }

    fn token_probability(logits: &Array, token: u32, stream: &Stream) -> Result<f32, Exception> {
        let vocab_size = logits.dim(-1) as usize;
        if token as usize >= vocab_size {
            return Err(Exception::custom(format!(
                "sampled token {token} exceeds vocabulary size {vocab_size}"
            )));
        }
        let probabilities = safemlx::ops::softmax_axis(logits, -1, true, stream)?;
        let selected = match probabilities.ndim() {
            1 => probabilities.try_index_device(token as i32, stream)?,
            2 => probabilities.try_index_device((0, token as i32), stream)?,
            3 => probabilities.try_index_device((0, 0, token as i32), stream)?,
            rank => {
                return Err(Exception::custom(format!(
                    "Mirostat V2 processed logits must have rank 1, 2, or 3, got rank {rank}"
                )))
            }
        };
        Ok(selected.item::<f32>(stream))
    }
}

fn effective_allowed_mask(allowed: &[bool], vocab_size: usize) -> Result<&[bool], String> {
    let Some(allowed) = allowed.get(..vocab_size) else {
        return Err(format!(
            "token filter vocabulary size {} is smaller than logits vocabulary size {vocab_size}",
            allowed.len()
        ));
    };
    if !allowed.iter().any(|allowed| *allowed) {
        return Err(format!(
            "token filter permits no token in the logits vocabulary prefix of size {vocab_size}"
        ));
    }
    Ok(allowed)
}

fn sample_categorical(
    logits: &Array,
    random: Option<&mut RandomState>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let random = random
        .ok_or_else(|| Exception::custom("random operations require an explicit PRNG key"))?;
    let key = random.next_key(stream)?;
    random::categorical(logits, None, None, &key, stream)
}

fn mask_logits(mask: Array, logits: Array, stream: &Stream) -> Result<Array, Exception> {
    let minimum = Array::from_f32(logits.dtype().finfo_min()? as f32);
    safemlx::ops::r#where(mask, minimum, logits, stream)
}

#[cfg(test)]
mod tests {
    use super::{effective_allowed_mask, MlxSamplingBackend};
    use eredu_runtime::{SamplingBackend, TokenDomain};
    use safemlx::{transforms::async_eval_with_event, Array, Device, DeviceType, ExecutionContext};

    use crate::backend::mlx::nn::tensor::TokenValidationScope;

    #[test]
    fn token_filter_accepts_a_truncated_output_vocabulary_prefix() {
        assert_eq!(
            effective_allowed_mask(&[false, true, false, true], 3).unwrap(),
            &[false, true, false]
        );
        assert!(effective_allowed_mask(&[false, false, true], 2).is_err());
        assert!(effective_allowed_mask(&[true], 2).is_err());
    }

    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_token_domain_validation_is_deferred_to_completion() {
        let execution = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
        let stream = execution.stream();
        let scope = TokenValidationScope::begin().unwrap();
        let valid = MlxSamplingBackend::validate_token(
            &Array::from_slice(&[0_u32, 4], &[2]),
            TokenDomain::new(5),
            stream,
        )
        .unwrap();
        assert_eq!(valid.dtype(), safemlx::Dtype::Int32);
        let validations = scope.finish();
        let event =
            async_eval_with_event(std::iter::once(&valid).chain(validations.arrays())).unwrap();
        event.synchronize().unwrap();
        validations.validate_completed().unwrap();

        for invalid in [-1_i32, 5] {
            let scope = TokenValidationScope::begin().unwrap();
            let token = MlxSamplingBackend::validate_token(
                &Array::from_slice(&[invalid], &[1]),
                TokenDomain::new(5),
                stream,
            )
            .expect("lazy device validation must not synchronize while building the graph");
            let validations = scope.finish();
            let event =
                async_eval_with_event(std::iter::once(&token).chain(validations.arrays())).unwrap();
            event.synchronize().unwrap();
            assert!(validations.validate_completed().is_err());
        }
    }
}
