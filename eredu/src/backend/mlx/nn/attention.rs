//! Shared attention inputs, projection transforms, cache updates, and kernels.

use safemlx::{
    builder::Builder,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    module::Module,
    nn,
    ops::{
        broadcast_to, concatenate_axis, einsum,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        matmul, r#where, softmax_axis,
    },
    Array, Dtype, Stream,
};

use crate::{
    backend::mlx::nn::tensor::{
        create_causal_mask, rope::RopeVariant, scaled_dot_product_attention,
    },
    backend::mlx::runtime::cache::KeyValueCache,
};

/// Common attention-layer input.
pub struct AttentionInput<'a, C> {
    /// Hidden states with shape `[batch, sequence, hidden]`.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional mutable key/value cache.
    pub cache: Option<&'a mut C>,
}

/// Returns the batch size and sequence length from a hidden-state tensor.
pub fn batch_seq(x: &Array) -> (i32, i32) {
    let shape = x.shape();
    (shape[0], shape[1])
}

/// Reshapes a projected Q/K/V tensor to `[batch, heads, sequence, head_dim]`.
pub fn reshape_attention_projection(
    projection: Array,
    batch: i32,
    seq_len: i32,
    heads: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    projection
        .reshape(&[batch, seq_len, heads, -1], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)
}

/// Computes explicit attention probabilities for inspection views.
pub fn attention_probabilities(
    queries: &Array,
    keys: &Array,
    scale: f32,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    let queries_shape = queries.shape();
    let keys_shape = keys.shape();
    let batch = queries_shape[0];
    let query_heads = queries_shape[1];
    let key_heads = keys_shape[1];
    let key_len = keys_shape[2];
    let head_dim = keys_shape[3];
    let keys = if query_heads == key_heads {
        keys.clone()
    } else if query_heads % key_heads == 0 {
        let repeats = query_heads / key_heads;
        broadcast_to(
            &keys.reshape(&[batch, key_heads, 1, key_len, head_dim], stream)?,
            &[batch, key_heads, repeats, key_len, head_dim],
            stream,
        )?
        .reshape(&[batch, query_heads, key_len, head_dim], stream)?
    } else {
        return Err(Exception::custom(
            "query attention heads are not divisible by key/value heads",
        ));
    };

    let mut scores = matmul(
        &queries.multiply(Array::from_f32(scale), stream)?,
        &keys.swap_axes(-1, -2, stream)?,
        stream,
    )?;
    if let Some(mask) = mask {
        if mask.dtype() == Dtype::Bool {
            let finfo_min = scores.dtype().finfo_min()?;
            scores = r#where(mask, scores, Array::from_f32(finfo_min as f32), stream)?;
        } else {
            scores = scores.add(mask, stream)?;
        }
    }
    softmax_axis(&scores, -1, true, stream)
}

/// Sparse attention over a bounded local window and indexed compressed tokens.
///
/// Queries are `[batch, heads, query, dim]`, local and pooled values are
/// `[batch, tokens, dim]`, and `pooled_indices` is `[batch, query, selected]`.
/// A learned sink contributes to the shared softmax denominator with a zero
/// value. The implementation materializes scores only for the local window and
/// selected compressed positions, never for the complete source context.
#[allow(clippy::too_many_arguments)]
pub fn indexed_sparse_attention(
    queries: &Array,
    local: &Array,
    pooled: &Array,
    pooled_indices: &Array,
    scale: f32,
    local_mask: Option<&Array>,
    pooled_mask: Option<&Array>,
    sinks: Option<&Array>,
    stream: &Stream,
) -> Result<Array, Exception> {
    if queries.ndim() != 4
        || local.ndim() != 3
        || pooled.ndim() != 3
        || pooled_indices.ndim() != 3
        || queries.dim(0) != local.dim(0)
        || queries.dim(0) != pooled.dim(0)
        || queries.dim(0) != pooled_indices.dim(0)
        || queries.dim(2) != pooled_indices.dim(1)
        || queries.dim(3) != local.dim(2)
        || queries.dim(3) != pooled.dim(2)
    {
        return Err(Exception::custom(format!(
            "indexed sparse attention received incompatible shapes q={:?}, local={:?}, pooled={:?}, indices={:?}",
            queries.shape(),
            local.shape(),
            pooled.shape(),
            pooled_indices.shape()
        )));
    }
    let batch = queries.dim(0);
    let heads = queries.dim(1);
    let query_tokens = queries.dim(2);
    let head_dim = queries.dim(3);
    let selected = pooled_indices.dim(2);
    let pooled_tokens = pooled.dim(1);
    if selected <= 0 || pooled_tokens <= 0 {
        return Err(Exception::custom(
            "indexed sparse attention requires at least one pooled token and selected index",
        ));
    }

    let expanded_pooled = broadcast_to(
        &pooled.try_index_device((.., NewAxis, .., ..), stream)?,
        &[batch, query_tokens, pooled_tokens, head_dim],
        stream,
    )?;
    let expanded_indices = broadcast_to(
        &pooled_indices.try_index_device((.., .., .., NewAxis), stream)?,
        &[batch, query_tokens, selected, head_dim],
        stream,
    )?;
    let selected_pooled = take_along_axis(expanded_pooled, &expanded_indices, 2, stream)?;

    let scaled_queries = queries.multiply(Array::from_f32(scale), stream)?;
    let mut local_scores = einsum("bhld,btd->bhlt", [&scaled_queries, local], stream)?;
    let mut pooled_scores = einsum(
        "bhld,blkd->bhlk",
        [&scaled_queries, &selected_pooled],
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
    let local_tokens = local.dim(1);
    let local_weights = weights.try_index_device((.., .., .., ..local_tokens), stream)?;
    let pooled_weights =
        weights.try_index_device((.., .., .., local_tokens..local_tokens + selected), stream)?;
    let local_context = einsum("bhlt,btd->bhld", [&local_weights, local], stream)?;
    let pooled_context = einsum(
        "bhlk,blkd->bhld",
        [&pooled_weights, &selected_pooled],
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

/// Applies RoPE to queries and keys, then updates a cache when provided.
pub fn apply_rope_and_update_cache<C>(
    rope: &mut RopeVariant,
    mut queries: Array,
    mut keys: Array,
    mut values: Array,
    cache: &mut Option<&mut C>,
    stream: &Stream,
) -> Result<(Array, Array, Array), Exception>
where
    C: KeyValueCache + ?Sized,
{
    if let Some(cache) = cache.as_mut() {
        let offset = cache.offset();
        queries = rope.forward(
            nn::RopeInputBuilder::new(&queries).offset(offset).build()?,
            stream,
        )?;
        keys = rope.forward(
            nn::RopeInputBuilder::new(&keys).offset(offset).build()?,
            stream,
        )?;
        (keys, values) = cache.update_for_attention(keys, values, stream)?;
    } else {
        queries = rope.forward(nn::RopeInput::new(&queries), stream)?;
        keys = rope.forward(nn::RopeInput::new(&keys), stream)?;
    }

    Ok((queries, keys, values))
}

/// Applies caller-provided rotary cosine and sine tensors to one head view.
pub(crate) fn apply_rotary_embeddings(
    value: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
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
/// Runs scaled dot-product attention and reshapes the output back to hidden states.
pub fn finish_attention<C>(
    queries: Array,
    keys: Array,
    values: Array,
    cache: Option<&mut C>,
    scale: f32,
    mask: Option<&Array>,
    batch: i32,
    seq_len: i32,
    stream: &Stream,
) -> Result<Array, Exception>
where
    C: KeyValueCache + ?Sized,
{
    let attention = if let Some(cache) = cache {
        match cache.paged_attention(&queries, scale, mask, None, stream)? {
            Some(output) => output,
            None => scaled_dot_product_attention(
                queries,
                keys,
                values,
                Some(cache),
                scale,
                mask,
                stream,
            )?,
        }
    } else {
        scaled_dot_product_attention(
            queries,
            keys,
            values,
            Option::<&mut C>::None,
            scale,
            mask,
            stream,
        )?
    };
    attention
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, seq_len, -1], stream)
}

#[allow(clippy::too_many_arguments)]
/// Computes causal sliding-window attention without a prompt-sized square mask.
pub(crate) fn sliding_window_prefill_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    window_size: i32,
    query_position_offset: i32,
    batch: i32,
    seq_len: i32,
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

    if query_position_offset == 0 && seq_len <= window_size {
        return safemlx::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale,
            Some(ScaledDotProductAttentionMask::Causal),
            None,
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
        chunks.push(safemlx::fast::scaled_dot_product_attention(
            query_chunk,
            key_chunk,
            value_chunk,
            scale,
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
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
mod tests {
    use super::{
        attention_probabilities, indexed_sparse_attention, sliding_window_prefill_attention,
    };
    use safemlx::{
        fast::ScaledDotProductAttentionMask, Array, Device, DeviceType, Dtype, ExecutionContext,
    };

    use crate::backend::mlx::nn::tensor::create_causal_mask;

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn bool_attention_mask_keeps_attention_probabilities_float32() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let queries = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        let keys = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2, 1]);
        let mask = Array::from_slice(&[false, true, false, false], &[1, 1, 2, 2]);

        let probs = attention_probabilities(&queries, &keys, 1.0, Some(&mask), stream).unwrap();

        assert_eq!(probs.dtype(), Dtype::Float32);
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn chunked_sliding_prefill_matches_full_masked_gqa_attention() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let queries = Array::from_slice(
            &[
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
                0.4, 0.3, 0.2, 0.1,
            ],
            &[1, 2, 5, 2],
        );
        let keys = Array::from_slice(
            &[0.2f32, 0.4, 0.6, 0.8, 1.0, 0.9, 0.7, 0.5, 0.3, 0.1],
            &[1, 1, 5, 2],
        );
        let values = Array::from_slice(
            &[1.0f32, 0.0, 0.8, 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8],
            &[1, 1, 5, 2],
        );
        let mask = create_causal_mask(5, None, Some(2), None, stream).unwrap();
        let reference = safemlx::fast::scaled_dot_product_attention(
            queries.clone(),
            keys.clone(),
            values.clone(),
            2.0f32.sqrt().recip(),
            Some(ScaledDotProductAttentionMask::Array(&mask)),
            None,
            stream,
        )
        .unwrap()
        .transpose_axes(&[0, 2, 1, 3], stream)
        .unwrap()
        .reshape(&[1, 5, 4], stream)
        .unwrap();

        let chunked = sliding_window_prefill_attention(
            queries,
            keys,
            values,
            2.0f32.sqrt().recip(),
            3,
            0,
            1,
            5,
            stream,
        )
        .unwrap();

        assert!(chunked
            .all_close(&reference, 1e-5, 1e-5, None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn sliding_prefill_matches_independent_scalar_softmax_reference() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let query_values = [1.0f32, 2.0, 3.0, 4.0];
        let key_values = [0.5f32, 1.0, 1.5, 2.0];
        let value_values = [10.0f32, 20.0, 30.0, 40.0];
        let actual = sliding_window_prefill_attention(
            Array::from_slice(&query_values, &[1, 1, 4, 1]),
            Array::from_slice(&key_values, &[1, 1, 4, 1]),
            Array::from_slice(&value_values, &[1, 1, 4, 1]),
            1.0,
            2,
            0,
            1,
            4,
            stream,
        )
        .unwrap();
        let actual = actual.evaluated().unwrap();

        let expected = (0..query_values.len())
            .map(|query| {
                let first = query.saturating_sub(1);
                let logits = (first..=query)
                    .map(|key| query_values[query] * key_values[key])
                    .collect::<Vec<_>>();
                let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = logits
                    .iter()
                    .map(|logit| (logit - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                weights
                    .iter()
                    .zip(&value_values[first..=query])
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
                    / denominator
            })
            .collect::<Vec<_>>();
        for (actual, expected) in actual.as_slice::<f32>().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn indexed_attention_shares_softmax_with_sink() {
        let ctx = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let queries = Array::from_slice(&[1.0f32], &[1, 1, 1, 1]);
        let local = Array::from_slice(&[1.0f32], &[1, 1, 1]);
        let pooled = Array::from_slice(&[2.0f32, 3.0], &[1, 2, 1]);
        let indices = Array::from_slice(&[1i32], &[1, 1, 1]);
        let sinks = Array::from_slice(&[0.0f32], &[1]);
        let output = indexed_sparse_attention(
            &queries,
            &local,
            &pooled,
            &indices,
            1.0,
            None,
            None,
            Some(&sinks),
            stream,
        )
        .unwrap();
        let output = output.evaluated().unwrap();
        let denominator = 1.0f32.exp() + 3.0f32.exp() + 1.0;
        let expected = (1.0f32.exp() + 3.0 * 3.0f32.exp()) / denominator;
        assert!((output.as_slice::<f32>()[0] - expected).abs() < 1e-5);
    }
}
