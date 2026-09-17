//! Shared attention inputs, projection transforms, cache updates, and kernels.

use safemlx::{
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::{
        broadcast_to, concatenate_axis,
        indexing::{NewAxis, TryIndexOp},
        r#where, softmax_axis,
    },
    Array, Dtype, Stream,
};

use crate::backend::nn::tensor::create_causal_mask;

mod pooling;
pub(in crate::backend::nn) mod indexed;
pub(in crate::backend::nn) mod pooled_positions;
pub(in crate::backend::nn) mod gather_mask;
pub(super) use pooling::{pooled_mask_shapes, pooled_mask_shapes_fixed, pooled_attention_control_bytes};

// One native policy shared by execution and cold allocation facts.
pub(super) const INPUT_SCORE_ROW_BUDGET: i32 = 8192;
pub(super) const INPUT_SCORE_KEY_TILE: i32 = 256;
pub(super) const SLIDING_QUERY_TILE: i32 = 256;
pub(super) fn input_score_query_step(keys: i32) -> i32 {
    (INPUT_SCORE_ROW_BUDGET / keys).clamp(1, 32)
}

// Every selected tile retains rank four. General tuple indexing constructs
// host index-plan Vecs; this shared view uses fixed bounds on the actual array.
fn sequence_slice(
    input: &Array,
    start: i32,
    end: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    input.try_slice(
        &[0, 0, start, 0],
        &[input.dim(0), input.dim(1), end, input.dim(3)],
        &[1; 4],
        stream,
    )
}

/// Sparse attention over a bounded local window and indexed compressed tokens.
///
/// Queries are `[batch, heads, query, key_dim]`, local and pooled keys/values
/// are `[batch, tokens, key_or_value_dim]`, and `pooled_indices` is
/// `[batch, query, selected]`.
/// A learned sink contributes to the shared softmax denominator with a zero
/// value. The implementation materializes scores only for the local window and
/// selected compressed positions, never for the complete source context.
#[allow(clippy::too_many_arguments)]
pub fn indexed_sparse_attention(
    queries: &Array,
    local_keys: &Array,
    local_values: &Array,
    pooled_keys: &Array,
    pooled_values: &Array,
    pooled_indices: &Array,
    scale: f32,
    local_mask: Option<&Array>,
    pooled_mask: Option<&Array>,
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    indexed::run(indexed::Input {
        values: [queries, local_keys, local_values, pooled_keys, pooled_values, pooled_indices],
        scale, local_mask, pooled_mask, sinks,
    }, stream)
}

// Actual Rust owners in apply_rotary_embeddings; numerical descriptors and
// buffers are supplied by the shared explicit-embedding recipe separately.
pub(super) fn apply_rotary_embeddings_control_bytes() -> Option<usize> {
    use std::mem::size_of;
    [
        size_of::<(&Array, &Array, &Array, &Stream)>(),
        size_of::<(&Array, &Stream)>(), // add_batch_axis closure
        size_of::<&Stream>(), // rotate_half closure
        size_of::<(&Array, i32)>(), // rotate_half arguments and split
        size_of::<[Array; 2]>(), // batch-normalized cosine and sine
        size_of::<[Array; 2]>(), // cast cosine and sine temporaries
        size_of::<[Array; 2]>(), // head-expanded cosine and sine
        size_of::<[Array; 2]>(), // first and second halves
        size_of::<[Array; 2]>(), // scalar and negated second half
        size_of::<[Array; 2]>(), // concatenate's moved operands
        size_of::<[Array; 4]>(), // two products, rotated half and sum
        size_of::<Result<Array, Exception>>(),
        size_of::<eredu_nn::RotaryPosition<'_, crate::MlxTensor>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
        safemlx::ops::indexing::inline_basic_index_control_bytes()?.checked_mul(4)?,
        safemlx::ops::concatenate_axis_control_bytes()?,
    ].into_iter().try_fold(0usize, usize::checked_add)
}

/// Applies caller-provided rotary cosine and sine tensors to one head view.
pub fn apply_rotary_embeddings(
    value: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let add_batch_axis = |embedding: &Array| -> Result<Array, Exception> {
        match embedding.shape().len() {
            2 => embedding.expand_dims(0, stream),
            3 => Ok(embedding.clone()),
            rank => Err(Exception::custom(format!(
                "explicit rotary embeddings must have rank 2 or 3, got {rank}"
            ))),
        }
    };
    let cos = add_batch_axis(cos)?;
    let sin = add_batch_axis(sin)?;
    let cos = cos
        .as_dtype(value.dtype(), stream)?
        .try_index_device((.., NewAxis, .., ..), stream)?;
    let sin = sin
        .as_dtype(value.dtype(), stream)?
        .try_index_device((.., NewAxis, .., ..), stream)?;
    let rotate_half = |x: &Array| -> Result<Array, Exception> {
        let half = x.dim(-1) / 2;
        let first = x.try_index_device((.., .., .., ..half), stream)?;
        let second = x.try_index_device((.., .., .., half..), stream)?;
        concatenate_axis(
            &[second.multiply(Array::try_from_f32(-1.0)?, stream)?, first],
            -1,
            stream,
        )
    };
    value
        .multiply(&cos, stream)?
        .add(rotate_half(value)?.multiply(&sin, stream)?, stream)
}

#[allow(clippy::too_many_arguments)]
/// Computes causal sliding-window attention without a prompt-sized square mask.
pub fn sliding_window_prefill_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    window_size: i32,
    query_position_offset: i32,
    batch: i32,
    seq_len: i32,
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    sliding_window_prefill_attention_with_softcap(
        queries,
        keys,
        values,
        scale,
        window_size,
        query_position_offset,
        batch,
        seq_len,
        sinks,
        None,
        eredu_nn::AttentionArithmetic::Fused,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
/// Runs bounded sliding attention with an optional pre-mask score cap.
pub fn sliding_window_prefill_attention_with_softcap(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    window_size: i32,
    query_position_offset: i32,
    batch: i32,
    seq_len: i32,
    sinks: Option<&Array>,
    softcap: Option<f32>,
    arithmetic: eredu_nn::AttentionArithmetic,
    stream: &Stream,
) -> Result<Array, Exception> {
    if window_size <= 0 {
        return Err(Exception::custom(
            "sliding attention window must be positive",
        ));
    }
    let q_shape = queries.shape();
    let k_shape = keys.shape();
    if q_shape.len() != 4 || k_shape.len() != 4 || values.shape().len() != 4 {
        return Err(Exception::custom(
            "sliding prefill attention expects rank-4 Q/K/V",
        ));
    }
    let key_len = k_shape[2];
    if q_shape[2] != seq_len || values.shape()[2] != key_len {
        return Err(Exception::custom(
            "sliding prefill attention received inconsistent sequence lengths",
        ));
    }
    let key_position_offset = eredu_nn::operation_geometry::SlidingAttentionGeometry::new(
        seq_len,
        key_len,
        window_size,
        query_position_offset,
    )
    .map_err(|error| Exception::custom(error.to_string()))?
    .key_origin();

    if arithmetic == eredu_nn::AttentionArithmetic::Fused
        && softcap.is_none()
        && query_position_offset == 0
        && seq_len <= window_size
    {
        return safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            sinks,
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream);
    }

    let max_past = window_size - 1;
    let chunk_size = SLIDING_QUERY_TILE;
    let mut chunks = crate::backend::nn::tensor::GroupedChunkOutputs::prepare(
        (seq_len as usize).div_ceil(chunk_size as usize),
    )?;
    let mut start = 0;
    while start < seq_len {
        let end = start + chunk_size.min(seq_len - start);
        let query_abs_start = query_position_offset + start;
        let wanted_key_start = (query_abs_start - max_past).max(key_position_offset);
        let key_start = wanted_key_start - key_position_offset;
        let key_end = query_position_offset + end - key_position_offset;
        let relative_offset = query_abs_start - wanted_key_start;
        let query_chunk = sequence_slice(&queries, start, end, stream)?;
        let key_chunk = sequence_slice(&keys, key_start, key_end, stream)?;
        let value_chunk = sequence_slice(&values, key_start, key_end, stream)?;
        let mask = create_causal_mask(
            end - start,
            Some(relative_offset),
            Some(max_past),
            None,
            stream,
        )?;
        chunks.push(attention_with_softcap(
            &query_chunk,
            &key_chunk,
            &value_chunk,
            scale,
            Some(&mask),
            sinks,
            softcap,
            arithmetic,
            stream,
        )?)?;
        start = end;
    }

    concatenate_axis(chunks.as_slice(), 2, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream)
}

#[cfg(test)]
mod tests;

/// Scaled attention with an optional score transform, before masks and sink logits.
#[allow(clippy::too_many_arguments)]
pub fn attention_with_softcap(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: Option<&Array>,
    sinks: Option<&Array>,
    softcap: Option<f32>,
    arithmetic: eredu_nn::AttentionArithmetic,
    stream: &Stream,
) -> Result<Array, Exception> {
    if softcap.is_none() && arithmetic == eredu_nn::AttentionArithmetic::Fused {
        return safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            sinks,
            stream,
        );
    }
    if softcap.is_some_and(|cap| !cap.is_finite() || cap <= 0.0) {
        return Err(Exception::custom(
            "attention score cap must be positive and finite",
        ));
    }
    if queries.ndim() != 4
        || keys.ndim() != 4
        || values.ndim() != 4
        || keys.dim(1) <= 0
        || queries.dim(1) <= 0
        || queries.dim(1) % keys.dim(1) != 0
        || queries.dim(0) != keys.dim(0)
        || queries.dim(0) != values.dim(0)
        || keys.dim(1) != values.dim(1)
        || keys.dim(2) != values.dim(2)
        || queries.dim(3) != keys.dim(3)
    {
        return Err(Exception::custom(
            "incompatible soft-capped attention geometry",
        ));
    }
    if arithmetic == eredu_nn::AttentionArithmetic::InputScores
        && i64::from(queries.dim(2)) * i64::from(keys.dim(2)) > i64::from(INPUT_SCORE_ROW_BUDGET)
    {
        return bounded_input_score_attention(
            queries, keys, values, scale, mask, sinks, softcap, stream,
        );
    }
    let batch = queries.dim(0);
    let heads = queries.dim(1);
    let kv_heads = keys.dim(1);
    let tokens = keys.dim(2);
    let repeat = heads / kv_heads;
    let expand = |x: &Array| -> Result<Array, Exception> {
        safemlx::ops::broadcast_to(
            &x.reshape(&[batch, kv_heads, 1, tokens, x.dim(3)], stream)?,
            &[batch, kv_heads, repeat, tokens, x.dim(3)],
            stream,
        )?
        .reshape(&[batch, heads, tokens, x.dim(3)], stream)
    };
    let score_dtype = if arithmetic == eredu_nn::AttentionArithmetic::InputScores {
        queries.dtype()
    } else {
        Dtype::Float32
    };
    let keys = expand(keys)?.as_dtype(score_dtype, stream)?;
    let values = expand(values)?;
    let query = queries.as_dtype(score_dtype, stream)?;
    let key = keys.swap_axes(-1, -2, stream)?;
    let mut scores = match super::matrix::bf16_batched_product(&query, &key, false, stream)? {
        Some(scores) => scores,
        None => safemlx::ops::matmul(&query, &key, stream)?,
    };
    scores = scores
        .as_dtype(Dtype::Float32, stream)?
        .multiply(Array::try_from_f32(scale)?, stream)?
        .as_dtype(score_dtype, stream)?;
    if let Some(cap) = softcap {
        scores = safemlx::ops::tanh(
            &scores.multiply(Array::try_from_f32(cap.recip())?, stream)?,
            stream,
        )?
        .multiply(Array::try_from_f32(cap)?, stream)?
        .as_dtype(score_dtype, stream)?;
    }
    if let Some(mask) = mask {
        scores = if mask.dtype() == safemlx::Dtype::Bool {
            safemlx::ops::r#where(
                mask,
                &scores,
                &Array::try_from_f32(f32::NEG_INFINITY)?,
                stream,
            )?
        } else {
            scores
                .add(mask.as_dtype(score_dtype, stream)?, stream)?
                .as_dtype(score_dtype, stream)?
        };
    }
    if let Some(sinks) = sinks {
        let sinks = safemlx::ops::broadcast_to(
            &sinks
                .as_dtype(score_dtype, stream)?
                .reshape(&[1, heads, 1, 1], stream)?,
            &[batch, heads, queries.dim(2), 1],
            stream,
        )?;
        scores = concatenate_axis(&[&scores, &sinks], -1, stream)?;
    }
    let scores = scores.as_dtype(Dtype::Float32, stream)?;
    let probabilities = match super::arithmetic::softmax_last(&scores, stream)? {
        Some(value) => value,
        None => safemlx::ops::softmax_axis(&scores, -1, true, stream)?,
    };
    // The direct worker keeps rank four and removes only the optional sink
    // column. Borrowed fixed bounds avoid general indexing's host plan vectors.
    let probabilities = probabilities
        .try_slice(
            &[0; 4],
            &[batch, heads, queries.dim(2), tokens],
            &[1; 4],
            stream,
        )?
        .as_dtype(queries.dtype(), stream)?;
    match super::matrix::bf16_batched_product(&probabilities, &values, true, stream)? {
        Some(output) => Ok(output),
        None => safemlx::ops::matmul(&probabilities, &values, stream),
    }
}

/// Explicit rounding need not materialize the complete Q-by-K score matrix.
/// Complete key rows use the ordinary softmax and product reduction when they
/// fit the score budget. Larger rows use the same two-pass recurrence as paged
/// storage, with bounded views and caller-provided (possibly noncausal) masks.
#[allow(clippy::too_many_arguments)]
fn bounded_input_score_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: Option<&Array>,
    sinks: Option<&Array>,
    softcap: Option<f32>,
    stream: &Stream,
) -> Result<Array, Exception> {
    use crate::backend::runtime::cache::kv::{
        BlockwiseAttentionAccumulator, KeyValueAttentionBlock,
    };
    let mask = mask
        .map(|mask| {
            broadcast_to(
                mask,
                &[queries.dim(0), queries.dim(1), queries.dim(2), keys.dim(2)],
                stream,
            )
        })
        .transpose()?;
    let query_step = input_score_query_step(keys.dim(2));
    let mut outputs = crate::backend::nn::tensor::GroupedChunkOutputs::prepare(
        (queries.dim(2) as usize).div_ceil(query_step as usize),
    )?;
    for start in (0..queries.dim(2)).step_by(query_step as usize) {
        let end = start + query_step.min(queries.dim(2) - start);
        let query = sequence_slice(queries, start, end, stream)?;
        let mask = mask
            .as_ref()
            .map(|mask| sequence_slice(mask, start, end, stream))
            .transpose()?;
        if keys.dim(2) <= INPUT_SCORE_ROW_BUDGET {
            outputs.push(attention_with_softcap(
                &query,
                keys,
                values,
                scale,
                mask.as_ref(),
                sinks,
                softcap,
                eredu_nn::AttentionArithmetic::InputScores,
                stream,
            )?)?;
            continue;
        }
        let mut accumulator = BlockwiseAttentionAccumulator::new(
            &query,
            scale,
            mask.as_ref(),
            0,
            None,
            0,
            sinks,
            i64::from(keys.dim(2)),
            stream,
        )?;
        accumulator.set_arithmetic(eredu_nn::AttentionArithmetic::InputScores)?;
        accumulator.set_softcap(softcap)?;
        accumulator.use_explicit_mask_only()?;
        for pass in 0..2 {
            if pass == 1 {
                accumulator.begin_value_pass()?;
            }
            for key_start in (0..keys.dim(2)).step_by(INPUT_SCORE_KEY_TILE as usize) {
                let key_end = key_start + INPUT_SCORE_KEY_TILE.min(keys.dim(2) - key_start);
                let block = KeyValueAttentionBlock::unleased(
                    i64::from(key_start),
                    i64::from(key_end),
                    sequence_slice(keys, key_start, key_end, stream)?,
                    sequence_slice(values, key_start, key_end, stream)?,
                );
                accumulator.accumulate(&block, stream)?;
            }
        }
        outputs.push(accumulator.finish(stream)?)?;
    }
    concatenate_axis(outputs.as_slice(), 2, stream)
}
