//! MLX realization of backend-neutral token-sampling primitives.

use eredu_runtime::{PenaltyConfig, SamplingBackend, TokenDomain};
use safemlx::{Array, Dtype, Stream, argmax_axis, error::Exception, random};

use crate::MlxTensor;
use crate::backend::{nn::tensor::validate_token_domain, random::RandomState};
use eredu_core::TokenFilter;

mod penalties;

/// MLX token-sampling capability implementation.
#[derive(Debug, Clone, Copy)]
pub struct MlxSamplingBackend;

impl SamplingBackend for MlxSamplingBackend {
    type Logits = MlxTensor;
    type Token = MlxTensor;
    type RandomState = RandomState;
    type Context = Stream;
    type Error = Exception;

    fn clone_token_with_host_source(
        value: &Self::Token,
        funding: &eredu_core::HostMetadataFunding,
        _context: &Self::Context,
    ) -> Result<Self::Token, eredu_core::BackendFailure> {
        value.clone_with_host_source(funding)
    }
    fn clone_logits_with_host_source(
        value: &Self::Logits,
        funding: &eredu_core::HostMetadataFunding,
        _context: &Self::Context,
    ) -> Result<Self::Logits, eredu_core::BackendFailure> {
        value.clone_with_host_source(funding)
    }
    fn clone_random_with_host_source(
        value: &Self::RandomState,
        funding: &eredu_core::HostMetadataFunding,
        _context: &Self::Context,
    ) -> Result<Self::RandomState, eredu_core::BackendFailure> {
        value.clone_with_host_source(funding)
    }

    fn error(message: String) -> Self::Error {
        Exception::custom(message)
    }

    fn validate_token(
        token: &Self::Token,
        domain: TokenDomain,
        stream: &Self::Context,
    ) -> Result<Self::Token, Self::Error> {
        if !matches!(token.as_array().dtype(), Dtype::Int32 | Dtype::Uint32) {
            return Err(Exception::custom(format!(
                "token IDs must use int32 or uint32 storage, got {:?}",
                token.as_array().dtype()
            )));
        }
        let cardinality = i32::try_from(domain.cardinality())
            .map_err(|_| Exception::custom("token domain exceeds MLX int32 range"))?;
        if cardinality <= 0 {
            return Err(Exception::custom("token domain must be non-empty"));
        }
        validate_token_domain(token.as_array(), cardinality, None, stream)
            .map(MlxTensor::from_array)
    }

    fn scale_temperature(
        logits: &MlxTensor,
        temperature: f32,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        logits
            .as_array()
            .multiply(Array::try_from_f32(1.0 / temperature)?, stream)
            .map(MlxTensor::from_array)
    }

    fn apply_penalties(
        logits: &MlxTensor,
        history: &[u32],
        penalties: PenaltyConfig,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        penalties::apply(logits, history, penalties, stream)
    }

    fn apply_top_k(logits: MlxTensor, top_k: i32, stream: &Stream) -> Result<MlxTensor, Exception> {
        let logits = logits.into_array();
        let vocab_size = logits.dim(-1);
        if top_k <= 0 || top_k >= vocab_size {
            return Ok(MlxTensor::from_array(logits));
        }
        let top_values = safemlx::ops::indexing::topk_axis(&logits, top_k, -1, stream)?;
        let threshold = top_values.min_axis(-1, true, stream)?;
        mask_logits(logits.lt(threshold, stream)?, logits, stream).map(MlxTensor::from_array)
    }

    fn apply_top_p(logits: MlxTensor, top_p: f32, stream: &Stream) -> Result<MlxTensor, Exception> {
        let logits = logits.into_array();
        if top_p >= 1.0 {
            return Ok(MlxTensor::from_array(logits));
        }
        let descending = safemlx::ops::argsort_axis(logits.negative(stream)?, -1, stream)?;
        let sorted = safemlx::ops::indexing::take_along_axis(&logits, &descending, -1, stream)?;
        let probabilities = safemlx::ops::softmax_axis(&sorted, -1, true, stream)?;
        let cumulative = probabilities.cumsum(-1, None, None, stream)?;
        let before = cumulative.subtract(probabilities, stream)?;
        let masked = mask_logits(
            before.gt(Array::try_from_f32(top_p.max(0.0))?, stream)?,
            sorted,
            stream,
        )?;
        let fill = Array::full::<f32>(
            logits.shape(),
            Array::try_from_f32(logits.dtype().finfo_min()? as f32)?,
            stream,
        )?
        .as_dtype(logits.dtype(), stream)?;
        safemlx::ops::indexing::put_along_axis(&fill, &descending, &masked, -1, stream)
            .map(MlxTensor::from_array)
    }

    fn apply_min_p(logits: MlxTensor, min_p: f32, stream: &Stream) -> Result<MlxTensor, Exception> {
        let logits = logits.into_array();
        if min_p <= 0.0 {
            return Ok(MlxTensor::from_array(logits));
        }
        let probabilities = safemlx::ops::softmax_axis(&logits, -1, true, stream)?;
        let maximum = probabilities.max_axis(-1, true, stream)?;
        let threshold = maximum.multiply(Array::try_from_f32(min_p)?, stream)?;
        mask_logits(probabilities.lt(threshold, stream)?, logits, stream).map(MlxTensor::from_array)
    }

    fn apply_token_filter(
        logits: &MlxTensor,
        filter: &TokenFilter,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let plan =
            eredu_runtime::generation::TokenMaskPlan::new(filter, logits.as_array().shape(), None)
                .map_err(|cause| Exception::custom(cause.to_string()))?;
        if plan.is_identity() {
            return Ok(logits.clone());
        }
        let mut invalid = Vec::with_capacity(plan.elements());
        plan.fill(&mut invalid)
            .map_err(|cause| Exception::custom(cause.to_string()))?;
        apply_token_mask(logits, &invalid, stream)
    }

    fn apply_mirostat(
        logits: &MlxTensor,
        history: &[u32],
        penalties: PenaltyConfig,
        temperature: f32,
        mu: f32,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        let vocab_size = logits.as_array().dim(-1) as usize;
        if vocab_size == 0 || logits.as_array().size() / vocab_size != 1 {
            return Err(Exception::custom(
                "Mirostat V2 currently requires logits for exactly one sequence",
            ));
        }
        let logits = Self::apply_penalties(logits, history, penalties, stream)?;
        let scaled = Self::scale_temperature(&logits, temperature, stream)?;
        let scaled = scaled.into_array();
        let probabilities = safemlx::ops::softmax_axis(&scaled, -1, true, stream)?;
        let cutoff = Array::try_from_f32((-mu).exp2())?;
        let maximum = probabilities.max_axis(-1, true, stream)?;
        let cutoff_mask = probabilities.lt(&cutoff, stream)?;
        let best =
            argmax_axis!(&probabilities, -1, stream = stream)?.expand_dims_axes(&[-1], stream)?;
        let fallback = Array::full::<bool>(
            logits.as_array().shape(),
            Array::try_from_bool(true)?,
            stream,
        )?;
        let keep_best = Array::full::<bool>(best.shape(), Array::try_from_bool(false)?, stream)?;
        let fallback =
            safemlx::ops::indexing::put_along_axis(&fallback, &best, &keep_best, -1, stream)?;
        let mask =
            safemlx::ops::r#where(cutoff.gt(maximum, stream)?, fallback, cutoff_mask, stream)?;
        mask_logits(mask, scaled, stream).map(MlxTensor::from_array)
    }

    fn sample_raw(
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        if temperature == 0.0 {
            argmax_axis!(logits.as_array(), -1, stream = stream).map(MlxTensor::from_array)
        } else {
            let scaled = Self::scale_temperature(logits, temperature, stream)?;
            sample_categorical(scaled.as_array(), random, stream).map(MlxTensor::from_array)
        }
    }

    fn sample_processed(
        logits: &MlxTensor,
        temperature: f32,
        random: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<MlxTensor, Exception> {
        if temperature == 0.0 {
            argmax_axis!(logits.as_array(), -1, stream = stream).map(MlxTensor::from_array)
        } else {
            sample_categorical(logits.as_array(), random, stream).map(MlxTensor::from_array)
        }
    }

    fn token_id(token: &MlxTensor, stream: &Stream) -> Result<u32, Exception> {
        token.as_array().clone().try_item::<u32>(stream)
    }

    fn token_probability(
        logits: &MlxTensor,
        token: u32,
        stream: &Stream,
    ) -> Result<f32, Exception> {
        token_probability_array(logits, token, stream)?.try_item::<f32>(stream)
    }
}

// The ordinary and admitted samplers construct the identical probability graph.
// Completion and scalar observation belong to their existing execution contexts.
pub(super) fn token_probability_array(
    logits: &MlxTensor,
    token: u32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let logits = logits.as_array();
    let vocab_size = logits.dim(-1) as usize;
    if token as usize >= vocab_size {
        return Err(Exception::custom(format!(
            "sampled token {token} exceeds vocabulary size {vocab_size}"
        )));
    }
    let probabilities = safemlx::ops::softmax_axis(logits, -1, true, stream)?;
    let rank = probabilities.ndim();
    if !(1..=3).contains(&rank) {
        return Err(Exception::custom(format!(
            "Mirostat V2 processed logits must have rank 1, 2, or 3, got rank {rank}"
        )));
    }
    // All supported ranks select coordinate [0, ..., token]. Use the
    // existing borrowed static Slice worker, followed by scalar reshape,
    // so ordinary/admitted execution need no general-index Vec or eager
    // index tensor. Neither operation changes the selected probability.
    let mut starts = [0; 3];
    let mut stops = [1; 3];
    starts[rank - 1] = token as i32;
    stops[rank - 1] = token as i32 + 1;
    let selected = probabilities
        .try_slice(&starts[..rank], &stops[..rank], &[1; 3][..rank], stream)?
        .reshape(&[], stream)?;
    Ok(selected)
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

// Named Rust transports of the shared TopK/TopP/MinP workers. TopP is the
// largest live set; source/configuration and native primitive owners are
// supplied by the selected pipeline and Graph producers respectively.
pub(super) fn filter_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        eredu_runtime::generation::TokenMaskPlan::control_bytes(),
        size_of::<[Array; 8]>(),
        size_of::<[MlxTensor; 2]>(),
        size_of::<Result<MlxTensor, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Option<i32>>(),
        size_of::<[Option<bool>; 2]>(),
        size_of::<Option<Dtype>>(),
        size_of::<[f32; 3]>(),
        size_of::<i32>(),
        size_of::<f64>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(super) fn mirostat_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<[Array; 11]>(),
        size_of::<[MlxTensor; 2]>(),
        size_of::<[&Array; 4]>(),
        size_of::<&Stream>(),
        size_of::<PenaltyConfig>(),
        size_of::<&[u32]>(),
        size_of::<&[i32]>(),
        size_of::<[f32; 4]>(),
        size_of::<[usize; 2]>(),
        size_of::<[[i32; 3]; 3]>(),
        size_of::<[bool; 2]>(),
        size_of::<Result<MlxTensor, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<f32, Exception>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(super) fn penalty_control_bytes() -> Option<usize> {
    penalties::control_bytes()
}

/// Consumes the already-expanded exact invalid mask through the same native
/// upload/where worker. The caller retains its paid Vec until upload completes.
pub(crate) fn apply_token_mask(
    logits: &MlxTensor,
    invalid: &[bool],
    stream: &Stream,
) -> Result<MlxTensor, Exception> {
    mask_logits(
        Array::try_from_slice(invalid, logits.as_array().shape())?,
        logits.as_array().clone(),
        stream,
    )
    .map(MlxTensor::from_array)
}

/// Fixed source frames for both ordinary token-filter branches and the shared
/// expanded-mask worker; the selected workspace report owns Vec payload bytes.
pub(super) fn token_mask_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        eredu_runtime::generation::TokenMaskPlan::control_bytes(),
        size_of::<(&MlxTensor, &TokenFilter, &Stream)>(),
        size_of::<(&MlxTensor, &[bool], &Stream)>(),
        size_of::<(Array, Array, &Stream)>(),
        size_of::<Vec<bool>>(),
        size_of::<eredu_runtime::generation::TokenMaskPlan<'_>>(),
        size_of::<[Array; 3]>(),
        size_of::<MlxTensor>(),
        size_of::<Result<MlxTensor, Exception>>(),
        size_of::<Result<Array, Exception>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

fn mask_logits(mask: Array, logits: Array, stream: &Stream) -> Result<Array, Exception> {
    let minimum = Array::try_from_f32(f32::NEG_INFINITY)?;
    safemlx::ops::r#where(mask, minimum, logits, stream)
}

#[cfg(test)]
mod tests;
