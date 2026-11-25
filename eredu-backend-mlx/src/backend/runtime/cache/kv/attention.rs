//! Blockwise attention over ordered paged-cache blocks.

use super::*;
use crate::backend::nn::{
    attention::{grouped_head_product, input_score_product},
    matrix::{BatchedDotReduction, PreparedBf16BatchedProduct},
    ordered_bf16_attention::OrderedBf16Attention,
};
use crate::MlxTensor;
use eredu_core::cache::mechanisms::AttentionScan;
use eredu_nn::paged_attention::{
    mask_columns, merge_softmax, page_visibility, requires_value_pass, retained_page_ranges,
    SoftmaxPartial,
};

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
    pass: usize,
    ordered: Option<OrderedBf16Attention>,
    causal: bool,
    retained_start: Option<i64>,
    explicit_mask: Option<Array>,
    query_start: i64,
    context_end: i64,
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
            pass: 0,
            ordered: None,
            causal: true,
            retained_start: None,
            explicit_mask: explicit_mask
                .map(|mask| {
                    broadcast_to(mask, &[batch, query_heads, query_len, mask.dim(-1)], stream)
                })
                .transpose()?,
            query_start,
            context_end,
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
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.running_max.is_some() {
            return Err(Exception::custom(
                "attention arithmetic cannot change after accumulation starts",
            ));
        }
        self.arithmetic = arithmetic;
        self.ordered = (requires_value_pass(arithmetic)
            && self.output_dtype == Dtype::Bfloat16
            && self
                .context_end
                .saturating_add(i64::from(self.sinks.is_some()))
                >= 4
            && OrderedBf16Attention::supported(stream)?)
        .then(OrderedBf16Attention::default);
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

    /// Reads the logically retained suffix and any permanent prefix without
    /// extending the sliding window across an uncommitted query span.
    pub fn use_retained_context(&mut self, start: i64) -> Result<(), Exception> {
        self.use_explicit_mask_only()?;
        self.retained_start = Some(start);
        Ok(())
    }

    /// Returns the sweep plan selected for the admitted arithmetic and device.
    pub fn scan(&self) -> AttentionScan {
        if self.ordered.is_some() {
            AttentionScan::MaximumThenMassThenValues
        } else if requires_value_pass(self.arithmetic) {
            AttentionScan::OnlineThenValues
        } else {
            AttentionScan::Online
        }
    }

    pub fn ordered_workspace_bytes(&self, page_score_bytes: u64) -> u64 {
        self.ordered.as_ref().map_or(0, |ordered| {
            ordered
                .workspace_bytes()
                .saturating_add(page_score_bytes / 2)
        })
    }

    /// Transitions to the next sweep after establishing its inputs.
    pub fn begin_pass(&mut self, pass: usize, stream: &Stream) -> Result<(), Exception> {
        if pass != self.pass + 1 || pass >= self.scan().passes() || self.running_max.is_none() {
            return Err(Exception::custom("attention sweep is out of order"));
        }
        if self.ordered.is_some() {
            if pass == 2 {
                if let Some(sink) = self.sink_scores(self.output_dtype, stream)? {
                    let ordered = self.ordered.as_mut().expect("ordered BF16 attention");
                    ordered.accumulate_mass(
                        &sink,
                        self.running_max.as_ref().expect("attention has maxima"),
                        i32::try_from(self.context_end)
                            .map_err(|_| Exception::custom("attention context exceeds i32"))?,
                        stream,
                    )?;
                    safemlx::transforms::eval(ordered.roots())?;
                }
            }
        } else {
            self.accumulator = None;
        }
        self.pass = pass;
        Ok(())
    }

    fn sink_scores(&self, dtype: Dtype, stream: &Stream) -> Result<Option<Array>, Exception> {
        self.sinks
            .as_ref()
            .map(|sink| {
                if sink.ndim() != 1 || sink.dim(0) != self.query_heads {
                    return Err(Exception::custom(
                        "paged attention sinks must have one value per query head",
                    ));
                }
                broadcast_to(
                    &sink
                        .as_dtype(dtype, stream)?
                        .as_dtype(Dtype::Float32, stream)?
                        .reshape(&[1, self.query_heads, 1, 1], stream)?,
                    &[self.batch, self.query_heads, self.query_len, 1],
                    stream,
                )
            })
            .transpose()
    }

    pub fn accumulate(
        &mut self,
        block: &KeyValueAttentionBlock,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.accumulate_with_bias(block, None, stream)
    }

    /// Appends an unleased contiguous tile. The caller completes each bounded
    /// group at its last tile, including any tail before changing sweeps.
    pub(crate) fn accumulate_contiguous(
        &mut self,
        block: &KeyValueAttentionBlock,
        complete_group: bool,
        stream: &Stream,
    ) -> Result<(), Exception> {
        debug_assert!(!self.causal && self.retained_start.is_none());
        self.accumulate_visible_with_bias(block, None, complete_group, stream)
    }

    /// Submits the current recurrence before the next cache-block dependency
    /// is inserted, allowing the dedicated transfer stream to prefetch the
    /// following block while this block's attention executes.
    pub fn submit(&self) -> Result<(), Exception> {
        safemlx::transforms::async_eval(
            self.running_max
                .iter()
                .chain(self.running_sum.iter())
                .chain(self.accumulator.iter())
                .chain(self.ordered.iter().flat_map(OrderedBf16Attention::roots)),
        )
    }

    pub fn accumulate_with_bias(
        &mut self,
        block: &KeyValueAttentionBlock,
        additive_bias: Option<&Array>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if let Some(start) = self.retained_start {
            // Physical pages can overlap discarded logical history. Split out
            // only the permanent prefix and retained suffix before interpreting
            // the caller's positional mask; transient positions are in suffix.
            for range in retained_page_ranges(block.start..block.end, self.prefix_tokens, start) {
                let (begin, end) = (range.start, range.end);
                let range = (begin - block.start) as i32..(end - block.start) as i32;
                let keys = block
                    .keys
                    .try_index_device((.., .., range.clone(), ..), stream)?;
                let values = block
                    .values
                    .try_index_device((.., .., range.clone(), ..), stream)?;
                let bias = additive_bias
                    .map(|bias| bias.try_index_device((.., .., .., range), stream))
                    .transpose()?;
                let part = KeyValueAttentionBlock::unleased(begin, end, keys, values);
                self.accumulate_visible_with_bias(&part, bias.as_ref(), true, stream)?;
            }
            return Ok(());
        }
        self.accumulate_visible_with_bias(block, additive_bias, true, stream)
    }

    fn accumulate_visible_with_bias(
        &mut self,
        block: &KeyValueAttentionBlock,
        additive_bias: Option<&Array>,
        complete_group: bool,
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
            || keys.dim(1) <= 0
            || keys.dim(2) <= 0
            || keys.dim(3) != self.head_dim
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
        let rounded = requires_value_pass(self.arithmetic);
        let ordered_scores = rounded && self.output_dtype == Dtype::Bfloat16;
        let dot_dtype = if ordered_scores {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        let keys = keys
            .as_dtype(dot_dtype, stream)?
            .swap_axes(-1, -2, stream)?;
        let mut scores = if ordered_scores {
            let query = self.queries.as_dtype(dot_dtype, stream)?;
            let reduction =
                BatchedDotReduction::for_scores(eredu_nn::arithmetic::Bf16DotGeometry {
                    rows: self.query_len as u64,
                    width: self.head_dim as u64,
                    columns: self.context_end as u64,
                });
            let prepared =
                PreparedBf16BatchedProduct::new(&keys, false, reduction, self.query_len, stream)?;
            input_score_product(&query, &keys, prepared.as_ref(), reduction, stream)?
                .as_dtype(Dtype::Float32, stream)?
                .multiply(Array::from_f32(self.scale), stream)?
                .as_dtype(self.output_dtype, stream)?
        } else if rounded {
            grouped_head_product(&self.queries, &keys, stream)?
                .as_dtype(self.output_dtype, stream)?
                .as_dtype(Dtype::Float32, stream)?
                .multiply(Array::from_f32(self.scale), stream)?
                .as_dtype(self.output_dtype, stream)?
        } else if self.arithmetic == eredu_nn::AttentionArithmetic::Float32 {
            grouped_head_product(&self.queries, &keys, stream)?
                .multiply(Array::from_f32(self.scale), stream)?
        } else {
            grouped_head_product(
                &self.queries.multiply(Array::from_f32(self.scale), stream)?,
                &keys,
                stream,
            )?
        };
        if let Some(cap) = self.softcap {
            scores = safemlx::ops::tanh(
                &scores.multiply(Array::from_f32(cap.recip()), stream)?,
                stream,
            )?
            .multiply(Array::from_f32(cap), stream)?;
            if ordered_scores {
                scores = scores.as_dtype(self.output_dtype, stream)?;
            }
        }
        if let Some(bias) = additive_bias {
            scores = scores.add(bias.as_dtype(scores.dtype(), stream)?, stream)?;
        }
        let allowed = if self.causal {
            page_visibility(
                self.query_start,
                self.query_len,
                block_start..block_end,
                self.sliding_window,
                self.prefix_tokens,
            )
            .map_err(|error| Exception::custom(error.to_string()))?
        } else {
            vec![true; self.query_len as usize * key_len as usize]
        };
        let allowed = Array::from_slice(&allowed, &[self.query_len, key_len]);
        let hidden = if self.ordered.is_some() {
            f32::NEG_INFINITY
        } else {
            f32::MIN
        };
        let effective_mask = if let Some(mask) = &self.explicit_mask {
            let columns = mask_columns(block_start..block_end, self.mask_origin, mask.dim(-1))
                .map_err(|error| Exception::custom(error.to_string()))?;
            let mask = mask.try_index_device((.., .., .., columns), stream)?;
            if mask.dtype() == Dtype::Bool {
                let combined = allowed.logical_and(&mask, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(hidden), stream)?;
                combined
            } else {
                let combined =
                    allowed.logical_and(&mask.is_neg_inf(stream)?.logical_not(stream)?, stream)?;
                scores = scores.add(mask.as_dtype(scores.dtype(), stream)?, stream)?;
                scores = r#where(&combined, scores, Array::from_f32(hidden), stream)?;
                combined
            }
        } else {
            scores = r#where(&allowed, scores, Array::from_f32(hidden), stream)?;
            allowed
        };
        let scores = scores.as_dtype(Dtype::Float32, stream)?;
        if self.ordered.is_some() {
            return self.accumulate_ordered(&scores, values, block_start, stream);
        }
        let values = values.as_dtype(Dtype::Float32, stream)?;
        if self.pass == 1 {
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
            let product = grouped_head_product(
                &probabilities.as_dtype(Dtype::Float32, stream)?,
                &values,
                stream,
            )?;
            self.accumulator = Some(match &self.accumulator {
                Some(previous) => previous.add(&product, stream)?,
                None => product,
            });
            if complete_group {
                safemlx::transforms::eval(self.accumulator.iter())?;
            }
            return Ok(());
        }
        let block_max = scores.max_axis(-1, true, stream)?;
        let mut weights = scores.subtract(&block_max, stream)?.exp(stream)?;
        weights = weights.multiply(effective_mask.as_dtype(Dtype::Float32, stream)?, stream)?;
        let block_sum = sum_axis(&weights, -1, true, stream)?;
        let block_accumulator = if requires_value_pass(self.arithmetic) {
            safemlx::ops::zeros::<f32>(
                &[self.batch, self.query_heads, self.query_len, value_dim],
                stream,
            )?
        } else {
            grouped_head_product(&weights, &values, stream)?
        };
        match (&self.running_max, &self.running_sum, &self.accumulator) {
            (Some(old_max), Some(old_sum), Some(old_accumulator)) => {
                let retained = SoftmaxPartial {
                    maximum: MlxTensor::from_array(old_max.clone()),
                    mass: MlxTensor::from_array(old_sum.clone()),
                    weighted_values: MlxTensor::from_array(old_accumulator.clone()),
                };
                let page = SoftmaxPartial {
                    maximum: MlxTensor::from_array(block_max),
                    mass: MlxTensor::from_array(block_sum),
                    weighted_values: MlxTensor::from_array(block_accumulator),
                };
                let merged = merge_softmax(
                    &retained,
                    &page,
                    stream,
                    |left, right, stream| {
                        maximum(left.as_array(), right.as_array(), stream)
                            .map(MlxTensor::from_array)
                            .map_err(eredu_nn::Error::backend_source)
                    },
                    |input, stream| {
                        input
                            .as_array()
                            .exp(stream)
                            .map(MlxTensor::from_array)
                            .map_err(eredu_nn::Error::backend_source)
                    },
                )
                .map_err(|error| Exception::custom(error.to_string()))?;
                self.running_sum = Some(merged.mass.into_array());
                self.accumulator = Some(merged.weighted_values.into_array());
                self.running_max = Some(merged.maximum.into_array());
            }
            _ => {
                if let Some(sink) = self.sink_scores(Dtype::Float32, stream)? {
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
        if complete_group {
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
        }
        Ok(())
    }

    fn accumulate_ordered(
        &mut self,
        scores: &Array,
        values: &Array,
        block_start: i64,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.pass == 0 {
            let block = scores.max_axis(-1, true, stream)?;
            let previous = match &self.running_max {
                Some(maximum) => Some(maximum.clone()),
                None => self.sink_scores(self.output_dtype, stream)?,
            };
            self.running_max = Some(match previous {
                Some(previous) => maximum(previous, block, stream)?,
                None => block,
            });
            return safemlx::transforms::eval(self.running_max.iter());
        }
        let position = |value| {
            i32::try_from(value).map_err(|_| Exception::custom("attention context exceeds i32"))
        };
        let maximum = self.running_max.as_ref().expect("attention has maxima");
        let ordered = self.ordered.as_mut().expect("ordered BF16 attention");
        let start = position(block_start)?;
        if self.pass == 1 {
            ordered.accumulate_mass(scores, maximum, start, stream)?;
        } else {
            ordered.accumulate_values(
                scores,
                maximum,
                values,
                start,
                position(self.context_end)?,
                stream,
            )?;
        }
        // Bound the number of page-local graphs and their retained cache views.
        safemlx::transforms::eval(ordered.roots())
    }

    pub fn finish(self, stream: &Stream) -> Result<Array, Exception> {
        if let Some(ordered) = self.ordered {
            return ordered.finish(stream);
        }
        let accumulator = self
            .accumulator
            .ok_or_else(|| Exception::custom("blockwise attention received no cache blocks"))?;
        if requires_value_pass(self.arithmetic) {
            if self.pass != 1 {
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
