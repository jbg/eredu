//! Blockwise attention over ordered paged-cache blocks.

use super::*;

pub struct KeyValueAttentionBlock {
    pub start: i64,
    pub end: i64,
    pub keys: Array,
    pub values: Array,
    pub bytes: u64,
}

impl KeyValueAttentionBlock {
    pub fn unleased(start: i64, end: i64, keys: Array, values: Array) -> Self {
        let bytes = keys.nbytes() as u64 + values.nbytes() as u64;
        Self {
            start,
            end,
            keys,
            values,
            bytes,
        }
    }
}

/// MLX online-softmax recurrence used behind the neutral blockwise-attention
/// backend contract.
pub struct BlockwiseAttentionAccumulator {
    queries: Array,
    output_dtype: Dtype,
    scale: f32,
    explicit_mask: Option<Array>,
    query_start: i64,
    sliding_window: Option<i32>,
    prefix_tokens: i64,
    sinks: Option<Array>,
    mask_origin: i64,
    batch: i32,
    query_heads: i32,
    query_len: i32,
    head_dim: i32,
    running_max: Option<Array>,
    running_sum: Option<Array>,
    accumulator: Option<Array>,
}

impl BlockwiseAttentionAccumulator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        queries: &Array,
        scale: f32,
        explicit_mask: Option<&Array>,
        query_start: i64,
        sliding_window: Option<i32>,
        prefix_tokens: i64,
        sinks: Option<&Array>,
        context_end: i64,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        if queries.ndim() != 4 {
            return Err(Exception::custom(
                "blockwise attention requires rank-4 queries",
            ));
        }
        if let Some(mask) = explicit_mask {
            if mask.ndim() != 2 {
                return Err(Exception::custom(
                    "paged attention supports only rank-2 explicit attention masks",
                ));
            }
            if mask.dim(0) != queries.dim(-2) {
                return Err(Exception::custom(
                    "paged attention mask query dimension does not match the active query",
                ));
            }
        }
        let batch = queries.dim(0);
        let query_heads = queries.dim(1);
        let query_len = queries.dim(2);
        let head_dim = queries.dim(3);
        Ok(Self {
            queries: queries.as_dtype(Dtype::Float32, stream)?,
            output_dtype: queries.dtype(),
            scale,
            explicit_mask: explicit_mask.cloned(),
            query_start,
            sliding_window,
            prefix_tokens,
            sinks: sinks.cloned(),
            mask_origin: explicit_mask.map_or(0, |mask| context_end - mask.dim(1) as i64),
            batch,
            query_heads,
            query_len,
            head_dim,
            running_max: None,
            running_sum: None,
            accumulator: None,
        })
    }

    pub fn accumulate(
        &mut self,
        block: &KeyValueAttentionBlock,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.accumulate_with_bias(block, None, stream)
    }

    /// Submits the current recurrence before the next cache-block dependency
    /// is inserted, allowing the dedicated transfer stream to prefetch the
    /// following block while this block's attention executes.
    pub fn submit(&self) -> Result<(), Exception> {
        safemlx::transforms::async_eval(
            self.running_max
                .iter()
                .chain(self.running_sum.iter())
                .chain(self.accumulator.iter()),
        )
    }

    pub fn accumulate_with_bias(
        &mut self,
        block: &KeyValueAttentionBlock,
        additive_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let block_start = block.start;
        let block_end = block.end;
        let keys = &block.keys;
        let values = &block.values;
        if keys.ndim() != 4
            || values.ndim() != 4
            || values.dim(0) != keys.dim(0)
            || values.dim(1) != keys.dim(1)
            || values.dim(2) != keys.dim(2)
        {
            return Err(Exception::custom(
                "paged key/value block shapes are inconsistent",
            ));
        }
        let key_heads = keys.dim(1);
        if self.query_heads % key_heads != 0 {
            return Err(Exception::custom(
                "query attention heads are not divisible by paged key/value heads",
            ));
        }
        let key_len = keys.dim(2);
        let value_dim = values.dim(3);
        let repeats = self.query_heads / key_heads;
        let (keys, values) = if repeats == 1 {
            (keys.clone(), values.clone())
        } else {
            let keys = broadcast_to(
                &keys.reshape(&[self.batch, key_heads, 1, key_len, self.head_dim], stream)?,
                &[self.batch, key_heads, repeats, key_len, self.head_dim],
                stream,
            )?
            .reshape(
                &[self.batch, self.query_heads, key_len, self.head_dim],
                stream,
            )?;
            let values = broadcast_to(
                &values.reshape(&[self.batch, key_heads, 1, key_len, value_dim], stream)?,
                &[self.batch, key_heads, repeats, key_len, value_dim],
                stream,
            )?
            .reshape(&[self.batch, self.query_heads, key_len, value_dim], stream)?;
            (keys, values)
        };
        let keys = keys.as_dtype(Dtype::Float32, stream)?;
        let values = values.as_dtype(Dtype::Float32, stream)?;
        let mut scores = matmul(
            &self.queries.multiply(Array::from_f32(self.scale), stream)?,
            &keys.swap_axes(-1, -2, stream)?,
            stream,
        )?;
        if let Some(bias) = additive_bias {
            scores = scores.add(bias, stream)?;
        }
        let allowed = absolute_attention_mask(
            self.query_start,
            self.query_len,
            block_start,
            block_end,
            self.sliding_window,
            self.prefix_tokens,
        );
        let allowed = Array::from_slice(&allowed, &[self.query_len, key_len]);
        let effective_mask = if let Some(mask) = &self.explicit_mask {
            let relative_start = block_start - self.mask_origin;
            let relative_end = block_end - self.mask_origin;
            if relative_start < 0 || relative_end > mask.dim(1) as i64 {
                return Err(Exception::custom(
                    "paged attention mask does not cover every visible cache block",
                ));
            }
            let mask =
                mask.try_index_device((.., relative_start as i32..relative_end as i32), stream)?;
            if mask.dtype() == Dtype::Bool {
                let combined = allowed.logical_and(&mask, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(f32::MIN), stream)?;
                combined
            } else {
                let combined =
                    allowed.logical_and(&mask.is_neg_inf(stream)?.logical_not(stream)?, stream)?;
                scores = scores.add(mask, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(f32::MIN), stream)?;
                combined
            }
        } else {
            scores = r#where(&allowed, scores, Array::from_f32(f32::MIN), stream)?;
            allowed
        };
        let block_max = scores.max_axis(-1, true, stream)?;
        let mut weights = scores.subtract(&block_max, stream)?.exp(stream)?;
        weights = weights.multiply(effective_mask.as_dtype(Dtype::Float32, stream)?, stream)?;
        let block_sum = sum_axis(&weights, -1, true, stream)?;
        let block_accumulator = matmul(&weights, &values, stream)?;
        match (&self.running_max, &self.running_sum, &self.accumulator) {
            (Some(old_max), Some(old_sum), Some(old_accumulator)) => {
                let new_max = maximum(old_max, &block_max, stream)?;
                let old_scale = old_max.subtract(&new_max, stream)?.exp(stream)?;
                let block_scale = block_max.subtract(&new_max, stream)?.exp(stream)?;
                self.running_sum = Some(
                    old_sum
                        .multiply(&old_scale, stream)?
                        .add(block_sum.multiply(&block_scale, stream)?, stream)?,
                );
                self.accumulator = Some(
                    old_accumulator
                        .multiply(&old_scale, stream)?
                        .add(block_accumulator.multiply(&block_scale, stream)?, stream)?,
                );
                self.running_max = Some(new_max);
            }
            _ => {
                if let Some(sinks) = &self.sinks {
                    if sinks.ndim() != 1 || sinks.dim(0) != self.query_heads {
                        return Err(Exception::custom(
                            "paged attention sinks must have one value per query head",
                        ));
                    }
                    let sink = sinks
                        .as_dtype(Dtype::Float32, stream)?
                        .reshape(&[1, self.query_heads, 1, 1], stream)?;
                    let sink = broadcast_to(
                        &sink,
                        &[self.batch, self.query_heads, self.query_len, 1],
                        stream,
                    )?;
                    let new_max = maximum(&sink, &block_max, stream)?;
                    let sink_sum = sink.subtract(&new_max, stream)?.exp(stream)?;
                    let block_scale = block_max.subtract(&new_max, stream)?.exp(stream)?;
                    self.running_max = Some(new_max);
                    self.running_sum =
                        Some(sink_sum.add(block_sum.multiply(&block_scale, stream)?, stream)?);
                    self.accumulator = Some(block_accumulator.multiply(block_scale, stream)?);
                } else {
                    self.running_max = Some(block_max);
                    self.running_sum = Some(block_sum);
                    self.accumulator = Some(block_accumulator);
                }
            }
        }
        safemlx::transforms::eval([
            self.running_max
                .as_ref()
                .expect("blockwise attention initialized row maximum"),
            self.running_sum
                .as_ref()
                .expect("blockwise attention initialized normalization"),
            self.accumulator
                .as_ref()
                .expect("blockwise attention initialized accumulator"),
        ])?;
        Ok(())
    }

    pub fn finish(self, stream: &Stream) -> Result<Array, Exception> {
        let accumulator = self
            .accumulator
            .ok_or_else(|| Exception::custom("blockwise attention received no cache blocks"))?;
        let running_sum = self
            .running_sum
            .ok_or_else(|| Exception::custom("blockwise attention normalization is empty"))?;
        let nonzero = running_sum.gt(Array::from_f32(0.0), stream)?;
        let safe_sum = r#where(&nonzero, &running_sum, Array::from_f32(1.0), stream)?;
        let output = accumulator.divide(safe_sum, stream)?;
        output.as_dtype(self.output_dtype, stream)
    }
}

pub(super) fn absolute_attention_mask(
    query_start: i64,
    query_len: i32,
    key_start: i64,
    key_end: i64,
    sliding_window: Option<i32>,
    prefix_tokens: i64,
) -> Vec<bool> {
    let mut mask = Vec::with_capacity(query_len as usize * (key_end - key_start) as usize);
    for query in query_start..query_start + query_len as i64 {
        for key in key_start..key_end {
            let causal = key <= query;
            let visible = sliding_window
                .is_none_or(|window| key < prefix_tokens || key >= query - (window - 1) as i64);
            mask.push(causal && visible);
        }
    }
    mask
}
