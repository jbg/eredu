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
    softcap: Option<f32>,
    arithmetic: eredu_nn::AttentionArithmetic,
    value_pass: bool,
    causal: bool,
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
            if mask.ndim() == 0 || mask.ndim() > 4 {
                return Err(Exception::custom(
                    "attention masks must broadcast to rank-4 scores",
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
            softcap: None,
            arithmetic: eredu_nn::AttentionArithmetic::Fused,
            value_pass: false,
            causal: true,
            explicit_mask: explicit_mask
                .map(|mask| {
                    broadcast_to(mask, &[batch, query_heads, query_len, mask.dim(-1)], stream)
                })
                .transpose()?,
            query_start,
            sliding_window,
            prefix_tokens,
            sinks: sinks.cloned(),
            mask_origin: explicit_mask.map_or(0, |mask| context_end - mask.dim(-1) as i64),
            batch,
            query_heads,
            query_len,
            head_dim,
            running_max: None,
            running_sum: None,
            accumulator: None,
        })
    }

    /// Sets the score transform used by every scanned block before masking.
    pub fn set_softcap(&mut self, cap: Option<f32>) -> Result<(), Exception> {
        if self.running_max.is_some() {
            return Err(Exception::custom(
                "attention score policy cannot change after accumulation starts",
            ));
        }
        if cap.is_some_and(|c| !c.is_finite() || c <= 0.0) {
            return Err(Exception::custom(
                "attention score cap must be positive and finite",
            ));
        }
        self.softcap = cap;
        Ok(())
    }

    /// Chooses score rounding before any cache block is consumed.
    pub fn set_arithmetic(
        &mut self,
        arithmetic: eredu_nn::AttentionArithmetic,
    ) -> Result<(), Exception> {
        if self.running_max.is_some() {
            return Err(Exception::custom(
                "attention arithmetic cannot change after accumulation starts",
            ));
        }
        self.arithmetic = arithmetic;
        Ok(())
    }

    /// Contiguous callers supply their own complete mask, including causality.
    pub fn use_explicit_mask_only(&mut self) -> Result<(), Exception> {
        if self.running_max.is_some() {
            return Err(Exception::custom(
                "attention masking cannot change after accumulation starts",
            ));
        }
        self.causal = false;
        Ok(())
    }

    /// Retains the global softmax normalization and starts a second bounded
    /// scan, rounding normalized probabilities before their value products.
    pub fn begin_value_pass(&mut self) -> Result<(), Exception> {
        if self.arithmetic != eredu_nn::AttentionArithmetic::InputScores
            || self.value_pass
            || self.running_max.is_none()
            || self.running_sum.is_none()
        {
            return Err(Exception::custom(
                "rounded-probability attention requires a completed normalization pass",
            ));
        }
        self.value_pass = true;
        self.accumulator = None;
        Ok(())
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
        let mut scores = if self.arithmetic == eredu_nn::AttentionArithmetic::InputScores {
            matmul(&self.queries, &keys.swap_axes(-1, -2, stream)?, stream)?
                .as_dtype(self.output_dtype, stream)?
                .as_dtype(Dtype::Float32, stream)?
                .multiply(Array::from_f32(self.scale), stream)?
                .as_dtype(self.output_dtype, stream)?
        } else {
            matmul(
                &self.queries.multiply(Array::from_f32(self.scale), stream)?,
                &keys.swap_axes(-1, -2, stream)?,
                stream,
            )?
        };
        if let Some(cap) = self.softcap {
            scores = safemlx::ops::tanh(
                &scores.multiply(Array::from_f32(cap.recip()), stream)?,
                stream,
            )?
            .multiply(Array::from_f32(cap), stream)?;
        }
        if let Some(bias) = additive_bias {
            scores = scores.add(bias.as_dtype(scores.dtype(), stream)?, stream)?;
        }
        let allowed = if self.causal {
            absolute_attention_mask(
                self.query_start,
                self.query_len,
                block_start,
                block_end,
                self.sliding_window,
                self.prefix_tokens,
            )
        } else {
            vec![true; self.query_len as usize * key_len as usize]
        };
        let allowed = Array::from_slice(&allowed, &[self.query_len, key_len]);
        let effective_mask = if let Some(mask) = &self.explicit_mask {
            let relative_start = block_start - self.mask_origin;
            let relative_end = block_end - self.mask_origin;
            if relative_start < 0 || relative_end > mask.dim(-1) as i64 {
                return Err(Exception::custom(
                    "paged attention mask does not cover every visible cache block",
                ));
            }
            let mask = mask.try_index_device(
                (.., .., .., relative_start as i32..relative_end as i32),
                stream,
            )?;
            if mask.dtype() == Dtype::Bool {
                let combined = allowed.logical_and(&mask, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(f32::MIN), stream)?;
                combined
            } else {
                let combined =
                    allowed.logical_and(&mask.is_neg_inf(stream)?.logical_not(stream)?, stream)?;
                scores = scores.add(mask.as_dtype(scores.dtype(), stream)?, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(f32::MIN), stream)?;
                combined
            }
        } else {
            scores = r#where(&allowed, scores, Array::from_f32(f32::MIN), stream)?;
            allowed
        };
        let scores = scores.as_dtype(Dtype::Float32, stream)?;
        if self.value_pass {
            let maximum = self
                .running_max
                .as_ref()
                .expect("normalization pass established maxima");
            let sum = self
                .running_sum
                .as_ref()
                .expect("normalization pass established denominators");
            let safe_sum = r#where(
                &sum.gt(Array::from_f32(0.0), stream)?,
                sum,
                Array::from_f32(1.0),
                stream,
            )?;
            let probabilities = scores
                .subtract(maximum, stream)?
                .exp(stream)?
                .multiply(effective_mask.as_dtype(Dtype::Float32, stream)?, stream)?
                .divide(&safe_sum, stream)?
                .as_dtype(self.output_dtype, stream)?;
            // Each page contributes to one FP32 product accumulator; rounding
            // partial page outputs would make the equation depend on page size.
            let product = matmul(
                &probabilities.as_dtype(Dtype::Float32, stream)?,
                &values,
                stream,
            )?;
            self.accumulator = Some(match &self.accumulator {
                Some(previous) => previous.add(&product, stream)?,
                None => product,
            });
            safemlx::transforms::eval(self.accumulator.iter())?;
            return Ok(());
        }
        let block_max = scores.max_axis(-1, true, stream)?;
        let mut weights = scores.subtract(&block_max, stream)?.exp(stream)?;
        weights = weights.multiply(effective_mask.as_dtype(Dtype::Float32, stream)?, stream)?;
        let block_sum = sum_axis(&weights, -1, true, stream)?;
        let block_accumulator = if self.arithmetic == eredu_nn::AttentionArithmetic::InputScores {
            safemlx::ops::zeros::<f32>(
                &[self.batch, self.query_heads, self.query_len, value_dim],
                stream,
            )?
        } else {
            matmul(&weights, &values, stream)?
        };
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
        if self.arithmetic == eredu_nn::AttentionArithmetic::InputScores {
            if !self.value_pass {
                return Err(Exception::custom(
                    "rounded-probability attention is missing its value pass",
                ));
            }
            return accumulator.as_dtype(self.output_dtype, stream);
        }
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
