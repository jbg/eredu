//! Finite-row normalization and selected host scores.
use safemlx::ops::indexing::TryIndexOp;
use safemlx::{error::Exception, Array, Dtype, Stream};

pub(super) const CHUNK: u64 = 1024;

pub(super) fn normalized_row(
    tensor: &Array,
    ids: &[u32],
    position: Option<u64>,
    stream: &Stream,
    host_read: impl Fn(usize),
) -> Result<(Array, i32, f64, f64), Exception> {
    let vocabulary = *tensor
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("scalar logits"))?;
    if vocabulary <= 0 || ids.iter().any(|id| *id >= vocabulary as u32) {
        return Err(Exception::custom("invalid selected score vocabulary"));
    }
    let shape = tensor.shape();
    let axes = &shape[..shape.len() - 1];
    let positions = axes
        .iter()
        .try_fold(1u64, |total, &extent| {
            total.checked_mul(u64::try_from(extent).ok()?)
        })
        .ok_or_else(|| Exception::custom("score position geometry overflow"))?;
    if position.is_some_and(|p| p >= positions) {
        return Err(Exception::custom(
            "selected score position lies outside the invocation",
        ));
    }
    let row_index = position
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Exception::custom("selected score position exceeds native index range"))?
        .unwrap_or(-1);
    let row = tensor
        .reshape(&[-1, vocabulary], stream)?
        .try_index_device((row_index, ..), stream)?
        .as_dtype(Dtype::Float32, stream)?;
    if count_mask(row.is_finite(stream)?, stream, &host_read)? != vocabulary as u64 {
        return Err(Exception::custom(
            "full-vocabulary scoring requires finite raw logits",
        ));
    }
    let maximum = scalar_f32(row.max(false, stream)?, &host_read)?;
    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    let mut accumulate = |sum: f32| {
        let sum = f64::from(sum);
        let next = total + sum;
        correction += if total.abs() >= sum.abs() {
            (total - next) + sum
        } else {
            (sum - next) + total
        };
        total = next;
    };
    // Native F32 chunk sums feed a compensated F64 total. Transfer batches
    // of at most CHUNK sums independently of the vocabulary extent.
    for start in (0..vocabulary).step_by((CHUNK * CHUNK) as usize) {
        let end = start.saturating_add((CHUNK * CHUNK) as i32).min(vocabulary);
        let complete = (end - start) / CHUNK as i32;
        let boundary = start + complete * CHUNK as i32;
        if complete > 0 {
            let sums = row
                .try_index_device(start..boundary, stream)?
                .reshape(&[complete, CHUNK as i32], stream)?
                .subtract(Array::from_f32(maximum as f32), stream)?
                .exp(stream)?
                .sum_axis(1, false, stream)?;
            host_read(complete as usize);
            let sums = sums.evaluated()?;
            for &sum in sums.as_slice::<f32>() {
                accumulate(sum);
            }
        }
        if boundary < end {
            let sum = scalar_f32(
                row.try_index_device(boundary..end, stream)?
                    .subtract(Array::from_f32(maximum as f32), stream)?
                    .exp(stream)?
                    .sum(false, stream)?,
                &host_read,
            )?;
            accumulate(sum as f32);
        }
    }
    let log_mass = (total + correction).ln();
    Ok((row, vocabulary, maximum, log_mass))
}

pub(super) fn scalar_f32(array: Array, host_read: impl Fn(usize)) -> Result<f64, Exception> {
    host_read(1);
    Ok(f64::from(array.evaluated()?.as_slice::<f32>()[0]))
}

pub(super) fn count_mask(
    mask: Array,
    stream: &Stream,
    host_read: impl Fn(usize),
) -> Result<u64, Exception> {
    host_read(1);
    let sum = mask.as_dtype(Dtype::Uint32, stream)?.sum(false, stream)?;
    Ok(u64::from(sum.evaluated()?.as_slice::<u32>()[0]))
}

pub(super) fn read_selected(
    scores: &Array,
    selections: Vec<eredu_core::refinement::RefinementScoreSelection>,
    stream: &Stream,
) -> Result<eredu_core::refinement::RefinementScoreRows, Exception> {
    selections
        .into_iter()
        .map(|selection| {
            let (row, _, maximum, log_mass) = normalized_row(
                scores,
                &selection.token_ids,
                Some(u64::from(selection.position)),
                stream,
                |_| {},
            )?;
            selection
                .token_ids
                .into_iter()
                .map(|token_id| {
                    let score =
                        scalar_f32(row.try_index_device(token_id as i32, stream)?, |_| {})? as f32;
                    Ok(eredu_core::refinement::RefinementSelectedScore {
                        token_id,
                        score,
                        log_probability: (f64::from(score) - maximum) - log_mass,
                    })
                })
                .collect()
        })
        .collect()
}
