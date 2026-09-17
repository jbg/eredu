//! Shared repetition/frequency/presence worker and its fixed host transports.
use crate::MlxTensor;
use eredu_runtime::PenaltyConfig;
use safemlx::{error::Exception, Array, Stream};

pub(super) fn apply(
    logits: &MlxTensor,
    history: &[u32],
    penalties: PenaltyConfig,
    stream: &Stream,
) -> Result<MlxTensor, Exception> {
    if history.is_empty() || penalties.is_identity() {
        return Ok(logits.clone());
    }

    let logits = logits.as_array();
    let vocab_size = logits.dim(-1) as usize;
    if vocab_size == 0 {
        return Ok(MlxTensor::from_array(logits.clone()));
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
    // A bounded contiguous history buffer replaces hash-table buckets.
    // In-place unstable sorting allocates no additional heap payload. The
    // quote can therefore price exact history and vocabulary dimensions,
    // independently of repeated token values or a hash-table growth edge.
    let mut sorted_history = history[start..].to_vec();
    sorted_history.sort_unstable();
    for repeated in sorted_history.chunk_by(equal_token_ids) {
        let token = repeated[0] as usize;
        if token >= vocab_size {
            continue;
        }
        let count = repeated.len();
        for row in 0..row_count {
            let index = row * vocab_size + token;
            repeat_mask[index] = true;
            additive[index] =
                penalties.frequency_penalty * count as f32 + penalties.presence_penalty;
        }
    }

    let mut adjusted = logits.clone();
    if penalties.repeat_penalty != 1.0 {
        let mask = Array::try_from_slice(&repeat_mask, logits.shape())?;
        let positive = adjusted.divide(Array::try_from_f32(penalties.repeat_penalty)?, stream)?;
        let negative = adjusted.multiply(Array::try_from_f32(penalties.repeat_penalty)?, stream)?;
        let penalized = safemlx::ops::r#where(
            adjusted.gt(Array::try_from_f32(0.0)?, stream)?,
            positive,
            negative,
            stream,
        )?;
        adjusted = safemlx::ops::r#where(mask, penalized, adjusted, stream)?;
    }
    if penalties.frequency_penalty != 0.0 || penalties.presence_penalty != 0.0 {
        adjusted = adjusted.subtract(Array::try_from_slice(&additive, logits.shape())?, stream)?;
    }
    Ok(MlxTensor::from_array(adjusted))
}

fn equal_token_ids(left: &u32, right: &u32) -> bool {
    left == right
}

// Payloads are already supplied by sampling::emit_host: Bool/F32 vocabulary
// buffers and the exact U32 history suffix. This names only their actual
// headers, borrowed iterators, scalar and safe-wrapper call transports. The
// shared unstable sort allocates no additional heap buffer.
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let empty: &[u32] = &[];
    [
        size_of::<Vec<bool>>(),
        size_of::<Vec<f32>>(),
        size_of::<Vec<u32>>(),
        size_of_val(&empty.chunk_by(equal_token_ids)),
        size_of::<&'static [u32]>(),
        size_of::<&'static MlxTensor>(),
        size_of::<&'static Array>(),
        size_of::<&'static Stream>(),
        size_of::<PenaltyConfig>(),
        // adjusted/mask/positive/negative/penalized plus three scalar/result
        // call arguments; their native C shells belong to the Graph bank.
        size_of::<[Array; 8]>(),
        size_of::<MlxTensor>(),
        size_of::<Result<MlxTensor, Exception>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<[usize; 7]>(),
        size_of::<std::ops::Range<usize>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}
