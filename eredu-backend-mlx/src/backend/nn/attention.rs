//! Shared attention inputs, projection transforms, cache updates, and kernels.

use safemlx::{
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::{
        broadcast_to, concatenate_axis, einsum,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        r#where, softmax_axis,
    },
    Array, Dtype, Stream,
};

use crate::backend::nn::tensor::create_causal_mask;

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
    if queries.ndim() != 4
        || local_keys.ndim() != 3
        || local_values.ndim() != 3
        || pooled_keys.ndim() != 3
        || pooled_values.ndim() != 3
        || pooled_indices.ndim() != 3
        || queries.dim(0) != local_keys.dim(0)
        || queries.dim(0) != local_values.dim(0)
        || queries.dim(0) != pooled_keys.dim(0)
        || queries.dim(0) != pooled_values.dim(0)
        || queries.dim(0) != pooled_indices.dim(0)
        || queries.dim(2) != pooled_indices.dim(1)
        || queries.dim(3) != local_keys.dim(2)
        || queries.dim(3) != pooled_keys.dim(2)
        || local_keys.dim(1) != local_values.dim(1)
        || pooled_keys.dim(1) != pooled_values.dim(1)
        || local_values.dim(2) != pooled_values.dim(2)
    {
        return Err(Exception::custom(format!(
            "indexed sparse attention received incompatible shapes q={:?}, local_keys={:?}, local_values={:?}, pooled_keys={:?}, pooled_values={:?}, indices={:?}",
            queries.shape(),
            local_keys.shape(),
            local_values.shape(),
            pooled_keys.shape(),
            pooled_values.shape(),
            pooled_indices.shape()
        )));
    }
    let batch = queries.dim(0);
    let heads = queries.dim(1);
    let query_tokens = queries.dim(2);
    let key_dim = queries.dim(3);
    let value_dim = local_values.dim(2);
    let selected = pooled_indices.dim(2);
    let pooled_tokens = pooled_keys.dim(1);
    if selected <= 0 || pooled_tokens <= 0 {
        return Err(Exception::custom(
            "indexed sparse attention requires at least one pooled token and selected index",
        ));
    }

    let expanded_pooled_keys = broadcast_to(
        &pooled_keys.try_index_device((.., NewAxis, .., ..), stream)?,
        &[batch, query_tokens, pooled_tokens, key_dim],
        stream,
    )?;
    let expanded_key_indices = broadcast_to(
        &pooled_indices.try_index_device((.., .., .., NewAxis), stream)?,
        &[batch, query_tokens, selected, key_dim],
        stream,
    )?;
    let selected_pooled_keys =
        take_along_axis(expanded_pooled_keys, &expanded_key_indices, 2, stream)?;
    let expanded_pooled_values = broadcast_to(
        &pooled_values.try_index_device((.., NewAxis, .., ..), stream)?,
        &[batch, query_tokens, pooled_tokens, value_dim],
        stream,
    )?;
    let expanded_value_indices = broadcast_to(
        &pooled_indices.try_index_device((.., .., .., NewAxis), stream)?,
        &[batch, query_tokens, selected, value_dim],
        stream,
    )?;
    let selected_pooled_values =
        take_along_axis(expanded_pooled_values, &expanded_value_indices, 2, stream)?;

    let scaled_queries = queries.multiply(Array::from_f32(scale), stream)?;
    let mut local_scores = einsum("bhld,btd->bhlt", [&scaled_queries, local_keys], stream)?;
    let mut pooled_scores = einsum(
        "bhld,blkd->bhlk",
        [&scaled_queries, &selected_pooled_keys],
        stream,
    )?;
    apply_score_mask(&mut local_scores, local_mask, stream)?;
    apply_score_mask(&mut pooled_scores, pooled_mask, stream)?;

    let mut score_parts = vec![local_scores, pooled_scores];
    if let Some(sinks) = sinks {
        if sinks.shape() != [heads] {
            return Err(Exception::custom(format!(
                "attention sinks require shape [{heads}], got {:?}",
                sinks.shape()
            )));
        }
        score_parts.push(broadcast_to(
            &sinks
                .as_dtype(score_parts[0].dtype(), stream)?
                .reshape(&[1, heads, 1, 1], stream)?,
            &[batch, heads, query_tokens, 1],
            stream,
        )?);
    }
    let scores = concatenate_axis(&score_parts, -1, stream)?;
    let weights = softmax_axis(scores, -1, true, stream)?;
    let local_tokens = local_keys.dim(1);
    let local_weights = weights.try_index_device((.., .., .., ..local_tokens), stream)?;
    let pooled_weights =
        weights.try_index_device((.., .., .., local_tokens..local_tokens + selected), stream)?;
    let local_context = einsum("bhlt,btv->bhlv", [&local_weights, local_values], stream)?;
    let pooled_context = einsum(
        "bhlk,blkv->bhlv",
        [&pooled_weights, &selected_pooled_values],
        stream,
    )?;
    local_context.add(pooled_context, stream)
}

fn apply_score_mask(
    scores: &mut Array,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<(), Exception> {
    let Some(mask) = mask else {
        return Ok(());
    };
    *scores = if mask.dtype() == Dtype::Bool {
        r#where(
            mask,
            &*scores,
            Array::from_f32(scores.dtype().finfo_min()? as f32),
            stream,
        )?
    } else {
        scores.add(mask.as_dtype(scores.dtype(), stream)?, stream)?
    };
    Ok(())
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
            &[second.multiply(Array::from_f32(-1.0), stream)?, first],
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
    let key_position_offset = query_position_offset + seq_len - key_len;
    if key_position_offset < 0 {
        return Err(Exception::custom(
            "sliding prefill attention key origin precedes position zero",
        ));
    }

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
    let chunk_size = 256;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < seq_len {
        let end = (start + chunk_size).min(seq_len);
        let query_abs_start = query_position_offset + start;
        let wanted_key_start = (query_abs_start - max_past).max(key_position_offset);
        let key_start = wanted_key_start - key_position_offset;
        let key_end = query_position_offset + end - key_position_offset;
        let relative_offset = query_abs_start - wanted_key_start;
        let query_chunk = queries.try_index_device((.., .., start..end, ..), stream)?;
        let key_chunk = keys.try_index_device((.., .., key_start..key_end, ..), stream)?;
        let value_chunk = values.try_index_device((.., .., key_start..key_end, ..), stream)?;
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
        )?);
        start = end;
    }

    let refs = chunks.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 2, stream)?
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
        && i64::from(queries.dim(2)) * i64::from(keys.dim(2)) > 8192
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
        .multiply(Array::from_f32(scale), stream)?
        .as_dtype(score_dtype, stream)?;
    if let Some(cap) = softcap {
        scores = safemlx::ops::tanh(
            &scores.multiply(Array::from_f32(cap.recip()), stream)?,
            stream,
        )?
        .multiply(Array::from_f32(cap), stream)?
        .as_dtype(score_dtype, stream)?;
    }
    if let Some(mask) = mask {
        scores = if mask.dtype() == safemlx::Dtype::Bool {
            safemlx::ops::r#where(mask, &scores, &Array::from_f32(f32::NEG_INFINITY), stream)?
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
    let probabilities = probabilities
        .try_index_device((.., .., .., ..tokens), stream)?
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
    let mut outputs = Vec::new();
    let query_step = (8192 / keys.dim(2)).clamp(1, 32);
    for start in (0..queries.dim(2)).step_by(query_step as usize) {
        let end = (start + query_step).min(queries.dim(2));
        let query = queries.try_index_device((.., .., start..end, ..), stream)?;
        let mask = mask
            .as_ref()
            .map(|mask| mask.try_index_device((.., .., start..end, ..), stream))
            .transpose()?;
        if keys.dim(2) <= 8192 {
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
            )?);
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
            for key_start in (0..keys.dim(2)).step_by(256) {
                let key_end = (key_start + 256).min(keys.dim(2));
                let block = KeyValueAttentionBlock::unleased(
                    i64::from(key_start),
                    i64::from(key_end),
                    keys.try_index_device((.., .., key_start..key_end, ..), stream)?,
                    values.try_index_device((.., .., key_start..key_end, ..), stream)?,
                );
                accumulator.accumulate(&block, stream)?;
            }
        }
        outputs.push(accumulator.finish(stream)?);
    }
    concatenate_axis(&outputs, 2, stream)
}
