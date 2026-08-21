//! Backend-neutral DeepSeek-V4 local, compressed, and indexed attention.

use eredu_nn::{
    AttentionRequest, Error, Index, IndexedAttentionInput, LinearOperator, LinearSpec,
    LowRankProjection, NeuralBackend, NormalizationOperator, NormalizationSpec, Parameter,
    ParameterSpec, Parameterized, PooledAttentionInput, PooledPositionInput, PoolingAttentionCache,
    PoolingOverlap, PoolingWindows, Tensor,
};

use crate::deepseek::{projection::ProjectionPolicy, V4Args, V4AttentionPolicy};
use eredu_runtime::ActivationObserver;

#[derive(Debug, Clone)]
struct V4Rotary<T: Tensor> {
    rotary_dimensions: i32,
    frequency_scale: i32,
    frequencies: T,
}

impl<T: Tensor> V4Rotary<T> {
    fn new(
        args: &V4Args,
        base: f32,
        yarn: bool,
        frequency_scale: i32,
        context: &T::Context,
    ) -> Result<Self, Error> {
        let dimensions = args.qk_rope_head_dim;
        let mut inverse = (0..dimensions)
            .step_by(2)
            .map(|index| 1.0 / base.powf(index as f32 / dimensions as f32))
            .collect::<Vec<_>>();
        if yarn {
            if let Some(config) = &args.rope_scaling {
                let correction = |rotations: f32| {
                    dimensions as f32
                        * (config.original_max_position_embeddings as f32
                            / (rotations * 2.0 * std::f32::consts::PI))
                            .ln()
                        / (2.0 * base.ln())
                };
                let low = correction(config.beta_fast).floor().max(0.0);
                let mut high = correction(config.beta_slow)
                    .ceil()
                    .min((dimensions - 1) as f32);
                if low == high {
                    high += 0.001;
                }
                for (index, frequency) in inverse.iter_mut().enumerate() {
                    let ramp = ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    let smooth = 1.0 - ramp;
                    *frequency = *frequency / config.factor * (1.0 - smooth) + *frequency * smooth;
                }
            }
        }
        let frequencies = inverse
            .into_iter()
            .map(|frequency| 1.0 / frequency / frequency_scale as f32)
            .collect::<Vec<_>>();
        Ok(Self {
            rotary_dimensions: dimensions,
            frequency_scale,
            frequencies: T::from_f32_slice(&frequencies, &[dimensions / 2], context)?,
        })
    }

    fn apply(
        &self,
        input: &T,
        offset: i32,
        inverse: bool,
        context: &T::Context,
    ) -> Result<T, Error> {
        let head_dimensions = *input
            .shape()
            .last()
            .ok_or_else(|| Error::backend("V4 rotary input has no feature axis"))?;
        let inactive_pairs = (head_dimensions - self.rotary_dimensions) / 2;
        if inactive_pairs < 0 || head_dimensions % 2 != 0 {
            return Err(Error::backend(format!(
                "V4 rotary width {head_dimensions} cannot contain {} rotary dimensions",
                self.rotary_dimensions
            )));
        }
        let active = if inverse {
            self.frequencies.multiply_scalar(-1.0, context)?
        } else {
            self.frequencies.clone()
        };
        let frequencies = if inactive_pairs == 0 {
            active
        } else {
            T::concatenate(
                &[
                    T::full_f32(f32::INFINITY, &[inactive_pairs], context)?,
                    active,
                ],
                0,
                context,
            )?
        };
        input.rope_with_frequencies(
            head_dimensions,
            true,
            offset / self.frequency_scale,
            &frequencies,
            context,
        )
    }
}

/// One learned gated compressor shared by ordinary compressed and sparse
/// index streams.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Compressor<B: NeuralBackend> {
    #[parameter(skip)]
    ratio: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    overlapping: bool,
    wkv: B::Linear,
    wgate: B::Linear,
    ape: Parameter<B::Tensor>,
    norm: B::Normalization,
    #[parameter(skip)]
    rope: V4Rotary<B::Tensor>,
}

impl<B: NeuralBackend> Compressor<B> {
    fn new(
        args: &V4Args,
        ratio: i32,
        head_dimensions: i32,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let overlapping = ratio == 4;
        let output = head_dimensions * if overlapping { 2 } else { 1 };
        Ok(Self {
            ratio,
            head_dimensions,
            overlapping,
            wkv: linear::<B>(
                format!("{root}.wkv.weight"),
                args.hidden_size,
                output,
                args.linear_format_for(&format!("{root}.wkv.weight")),
                context,
            )?,
            wgate: linear::<B>(
                format!("{root}.wgate.weight"),
                args.hidden_size,
                output,
                args.linear_format_for(&format!("{root}.wgate.weight")),
                context,
            )?,
            ape: Parameter::unloaded(parameter(format!("{root}.ape"))?, &[ratio, output], context)?,
            norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: head_dimensions,
                    epsilon: args.rms_norm_eps,
                    weight: parameter(format!("{root}.norm.weight"))?,
                },
                context,
            )?,
            rope: V4Rotary::new(args, args.compress_rope_theta, true, ratio, context)?,
        })
    }

    fn forward<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mut cache: Option<&mut C>,
        stream: u32,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if let Some(cache) = cache.as_deref_mut() {
            if cache.pooling_ratio(stream) != Some(self.ratio) {
                return Err(Error::backend(format!(
                    "V4 pooling stream {stream} does not have ratio {}",
                    self.ratio
                )));
            }
        }
        let values = self.wkv.forward(input, context)?;
        let gates = self.wgate.forward(input, context)?;
        let batch = values.dim(0);
        let windows = match cache.as_deref_mut() {
            Some(cache) => {
                cache.accumulate_pooling_windows(stream, values, gates, offset, context)?
            }
            None => complete_windows(values, gates, self.ratio, offset, context)?,
        };
        let complete = windows.values.dim(1);
        let mut pooled = if complete == 0 {
            B::Tensor::full_f32(0.0, &[batch, 0, self.head_dimensions], context)?
        } else {
            let count = complete / self.ratio;
            let output = self.head_dimensions * if self.overlapping { 2 } else { 1 };
            let values = windows
                .values
                .reshape(&[batch, count, self.ratio, output], context)?;
            let gates = windows
                .gates
                .reshape(&[batch, count, self.ratio, output], context)?;
            let ape = self
                .ape
                .as_ref()
                .reshape(&[1, 1, self.ratio, output], context)?
                .broadcast_to(&[batch, count, self.ratio, output], context)?;
            let gates = gates.add(&ape, context)?;
            if self.overlapping {
                overlap_pool::<B, C>(
                    &values,
                    &gates,
                    cache.as_deref_mut(),
                    stream,
                    self.head_dimensions,
                    context,
                )?
            } else {
                let weights = gates.softmax_axis(-2, true, context)?;
                B::Tensor::sum_axis(&values.multiply(&weights, context)?, -2, false, context)?
            }
        };
        pooled = self.norm.forward(&pooled, context)?;
        pooled = self.rope.apply(
            &pooled.expand_dims(1, context)?,
            windows.base_position,
            false,
            context,
        )?;
        pooled = pooled.squeeze_axes(&[1], context)?;
        match cache {
            Some(cache) => cache.append_pooled(stream, pooled, context),
            None => Ok(pooled),
        }
    }
}

fn complete_windows<T: Tensor>(
    values: T,
    gates: T,
    ratio: i32,
    offset: i32,
    context: &T::Context,
) -> Result<PoolingWindows<T>, Error> {
    let usable = values.dim(1) / ratio * ratio;
    Ok(PoolingWindows {
        values: slice_axis(&values, 1, 0, usable, context)?,
        gates: slice_axis(&gates, 1, 0, usable, context)?,
        base_position: offset,
    })
}

fn overlap_pool<B, C>(
    values: &B::Tensor,
    gates: &B::Tensor,
    cache: Option<&mut C>,
    stream: u32,
    head_dimensions: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error>
where
    B: NeuralBackend,
    C: PoolingAttentionCache<B::Tensor>,
{
    let batch = values.dim(0);
    let windows = values.dim(1);
    let ratio = values.dim(2);
    let current_values = slice_last(
        &slice_axis(values, 1, windows - 1, windows, context)?,
        0,
        head_dimensions,
        context,
    )?
    .squeeze_axes(&[1], context)?;
    let current_gates = slice_last(
        &slice_axis(gates, 1, windows - 1, windows, context)?,
        0,
        head_dimensions,
        context,
    )?
    .squeeze_axes(&[1], context)?;
    let previous = match cache {
        Some(cache) => cache.replace_pooling_overlap(stream, current_values, current_gates)?,
        None => PoolingOverlap {
            values: None,
            gates: None,
        },
    };
    let first_values = previous.values.unwrap_or(B::Tensor::full_f32(
        0.0,
        &[batch, ratio, head_dimensions],
        context,
    )?);
    let first_gates = previous.gates.unwrap_or(B::Tensor::full_f32(
        f32::NEG_INFINITY,
        &[batch, ratio, head_dimensions],
        context,
    )?);
    let previous_values = B::Tensor::concatenate(
        &[
            first_values.expand_dims(1, context)?,
            slice_last(
                &slice_axis(values, 1, 0, windows - 1, context)?,
                0,
                head_dimensions,
                context,
            )?,
        ],
        1,
        context,
    )?;
    let current_values = slice_last(values, head_dimensions, 2 * head_dimensions, context)?;
    let values = B::Tensor::concatenate(&[previous_values, current_values], 2, context)?;
    let previous_gates = B::Tensor::concatenate(
        &[
            first_gates.expand_dims(1, context)?,
            slice_last(
                &slice_axis(gates, 1, 0, windows - 1, context)?,
                0,
                head_dimensions,
                context,
            )?,
        ],
        1,
        context,
    )?;
    let current_gates = slice_last(gates, head_dimensions, 2 * head_dimensions, context)?;
    let gates = B::Tensor::concatenate(&[previous_gates, current_gates], 2, context)?;
    let weights = gates.softmax_axis(-2, true, context)?;
    B::Tensor::sum_axis(&values.multiply(&weights, context)?, -2, false, context)
}

/// Sparse indexer selecting pooled positions for ratio-four layers.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Indexer<B: NeuralBackend> {
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    top_k: i32,
    wq_b: B::Linear,
    weights_projection: B::Linear,
    compressor: Compressor<B>,
}

impl<B: NeuralBackend> Indexer<B> {
    fn new(
        args: &V4Args,
        ratio: i32,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            heads: args.index_n_heads,
            head_dimensions: args.index_head_dim,
            top_k: args.index_topk,
            wq_b: linear::<B>(
                format!("{root}.wq_b.weight"),
                args.q_lora_rank,
                args.index_n_heads * args.index_head_dim,
                args.linear_format_for(&format!("{root}.wq_b.weight")),
                context,
            )?,
            weights_projection: linear::<B>(
                format!("{root}.weights_proj.weight"),
                args.hidden_size,
                args.index_n_heads,
                args.linear_format_for(&format!("{root}.weights_proj.weight")),
                context,
            )?,
            compressor: Compressor::new(
                args,
                ratio,
                args.index_head_dim,
                &format!("{root}.compressor"),
                context,
            )?,
        })
    }

    fn forward<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        query_residual: &B::Tensor,
        rope: &V4Rotary<B::Tensor>,
        cache: &mut C,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error> {
        let pooled = self
            .compressor
            .forward(input, Some(cache), 1, offset, context)?;
        if pooled.dim(1) == 0 {
            return Ok((pooled, None));
        }
        let batch = input.dim(0);
        let tokens = input.dim(1);
        let query = self
            .wq_b
            .forward(query_residual, context)?
            .reshape(&[batch, tokens, self.heads, self.head_dimensions], context)?;
        let query = query.transpose_axes(&[0, 2, 1, 3], context)?;
        let query = rope.apply(&query, offset, false, context)?;
        let head_weights = self.weights_projection.forward(input, context)?;
        let mask = cache.pooling_mask(1, tokens, offset, context)?;
        let positions = B::select_pooled_positions(
            PooledPositionInput {
                queries: &query,
                pooled_keys: &pooled,
                head_weights: &head_weights,
                mask: mask.as_ref(),
                top_k: self.top_k.min(pooled.dim(1)),
                scale: (self.head_dimensions as f32).sqrt().recip(),
                head_scale: (self.heads as f32).sqrt().recip(),
            },
            context,
        )?;
        Ok((pooled, Some(positions)))
    }
}

/// Canonical V4 attention scheduled as local, compressed, or indexed sparse
/// without changing the surrounding block implementation.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend> {
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    groups: i32,
    #[parameter(skip)]
    output_rank: i32,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    normalization_epsilon: f32,
    #[parameter(skip)]
    policy: V4AttentionPolicy,
    query: LowRankProjection<B>,
    wkv: B::Linear,
    kv_norm: B::Normalization,
    wo_a: B::Linear,
    wo_b: B::Linear,
    sinks: Parameter<B::Tensor>,
    compressor: Option<Compressor<B>>,
    indexer: Option<Indexer<B>>,
    #[parameter(skip)]
    rope: V4Rotary<B::Tensor>,
}

impl<B: NeuralBackend> Attention<B> {
    /// Builds one unloaded V4 attention layer from its scheduled policy.
    pub fn new(
        args: &V4Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(args, layer, &format!("layers.{layer}.attn"), context)
    }

    pub(crate) fn new_at(
        args: &V4Args,
        layer: usize,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let policy = args
            .attention_policy(layer)
            .ok_or_else(|| Error::backend(format!("missing V4 attention policy {layer}")))?;
        let ratio = match policy {
            V4AttentionPolicy::Local => 0,
            V4AttentionPolicy::Compressed { ratio } => ratio,
        };
        let query = ProjectionPolicy {
            first_weight: Some(format!("{root}.wq_a.weight")),
            normalization_weight: format!("{root}.q_norm.weight"),
            second_weight: format!("{root}.wq_b.weight"),
            input_dimensions: args.hidden_size,
            rank: args.q_lora_rank,
            output_dimensions: args.num_attention_heads * args.head_dim,
            epsilon: args.rms_norm_eps,
            first_format: args.linear_format_for(&format!("{root}.wq_a.weight")),
            second_format: args.linear_format_for(&format!("{root}.wq_b.weight")),
        }
        .build(context)?;
        Ok(Self {
            heads: args.num_attention_heads,
            head_dimensions: args.head_dim,
            groups: args.o_groups,
            output_rank: args.o_lora_rank,
            scale: (args.head_dim as f32).sqrt().recip(),
            normalization_epsilon: args.rms_norm_eps,
            policy,
            query,
            wkv: linear::<B>(
                format!("{root}.wkv.weight"),
                args.hidden_size,
                args.head_dim,
                args.linear_format_for(&format!("{root}.wkv.weight")),
                context,
            )?,
            kv_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: args.head_dim,
                    epsilon: args.rms_norm_eps,
                    weight: parameter(format!("{root}.kv_norm.weight"))?,
                },
                context,
            )?,
            wo_a: linear::<B>(
                format!("{root}.wo_a.weight"),
                args.num_attention_heads * args.head_dim / args.o_groups,
                args.o_groups * args.o_lora_rank,
                args.linear_format_for(&format!("{root}.wo_a.weight")),
                context,
            )?,
            wo_b: linear::<B>(
                format!("{root}.wo_b.weight"),
                args.o_groups * args.o_lora_rank,
                args.hidden_size,
                args.linear_format_for(&format!("{root}.wo_b.weight")),
                context,
            )?,
            sinks: Parameter::unloaded(
                parameter(format!("{root}.attn_sink"))?,
                &[args.num_attention_heads],
                context,
            )?,
            compressor: (ratio != 0)
                .then(|| {
                    Compressor::new(
                        args,
                        ratio,
                        args.head_dim,
                        &format!("{root}.compressor"),
                        context,
                    )
                })
                .transpose()?,
            indexer: (ratio == 4)
                .then(|| Indexer::new(args, ratio, &format!("{root}.indexer"), context))
                .transpose()?,
            rope: V4Rotary::new(
                args,
                if ratio == 0 {
                    args.rope_theta
                } else {
                    args.compress_rope_theta
                },
                ratio != 0,
                1,
                context,
            )?,
        })
    }

    /// Executes the scheduled V4 attention policy over one neutral cache.
    pub fn forward<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_selected(input, mask, cache, context, |_| Ok(()))
    }

    /// Executes attention while reporting sparse pooled-position selections.
    pub fn forward_observed<C, O>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_with_selected(input, mask, cache, context, |positions| {
            observer.observe(&format!("{path}.selected_indexes"), positions)
        })
    }

    fn forward_with_selected<C, F>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        mut selected: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        F: FnMut(&B::Tensor) -> Result<(), Error>,
    {
        let batch = input.dim(0);
        let tokens = input.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let query_residual = match &mut self.query.first {
            Some(first) => first.forward(input, context)?,
            None => input.clone(),
        };
        let query_residual = self.query.normalization.forward(&query_residual, context)?;
        let query = self
            .query
            .second
            .forward(&query_residual, context)?
            .reshape(&[batch, tokens, self.heads, self.head_dimensions], context)?;
        let query = B::rms_norm_without_weight(&query, self.normalization_epsilon, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let query = self.rope.apply(&query, offset, false, context)?;
        let kv = self
            .kv_norm
            .forward(&self.wkv.forward(input, context)?, context)?;
        let kv = self
            .rope
            .apply(&kv.expand_dims(1, context)?, offset, false, context)?;
        let kv = kv.squeeze_axes(&[1], context)?;

        let mut output = match cache {
            None => {
                let keys = kv.expand_dims(1, context)?;
                let generated;
                let mask = match mask {
                    Some(mask) => Some(mask),
                    None if tokens > 1 => {
                        generated = B::causal_mask(tokens, 0, None, context)?;
                        Some(&generated)
                    }
                    None => None,
                };
                B::attention_with_sinks(
                    AttentionRequest {
                        queries: query,
                        keys: keys.clone(),
                        values: keys,
                        scale: self.scale,
                        mask,
                        sinks: Some(self.sinks.as_ref()),
                    },
                    context,
                )?
            }
            Some(cache) => {
                let local = cache.append_local(kv, context)?;
                let generated_local_mask;
                let local_mask = match mask {
                    Some(mask) => mask,
                    None => {
                        generated_local_mask = cache.local_mask(tokens, offset, context)?;
                        &generated_local_mask
                    }
                };
                match self.policy {
                    V4AttentionPolicy::Local => {
                        let keys = local.expand_dims(1, context)?;
                        B::attention_with_sinks(
                            AttentionRequest {
                                queries: query,
                                keys: keys.clone(),
                                values: keys,
                                scale: self.scale,
                                mask: Some(local_mask),
                                sinks: Some(self.sinks.as_ref()),
                            },
                            context,
                        )?
                    }
                    V4AttentionPolicy::Compressed { ratio: 4 } => {
                        let pooled = self
                            .compressor
                            .as_mut()
                            .expect("ratio-four V4 layer has a compressor")
                            .forward(input, Some(cache), 0, offset, context)?;
                        let (_, positions) = self
                            .indexer
                            .as_mut()
                            .expect("ratio-four V4 layer has an indexer")
                            .forward(input, &query_residual, &self.rope, cache, offset, context)?;
                        if pooled.dim(1) == 0
                            || pooled.dim(1) <= self.indexer.as_ref().expect("indexer").top_k
                        {
                            let pooled_mask = cache.pooling_mask(0, tokens, offset, context)?;
                            B::pooled_attention(
                                PooledAttentionInput {
                                    queries: &query,
                                    local: &local,
                                    pooled: &pooled,
                                    scale: self.scale,
                                    local_mask: Some(local_mask),
                                    pooled_mask: pooled_mask.as_ref(),
                                    sinks: Some(self.sinks.as_ref()),
                                },
                                context,
                            )?
                        } else {
                            let positions = positions
                                .as_ref()
                                .expect("non-empty V4 pool has selected positions");
                            selected(positions)?;
                            let full_pool_mask = cache.pooling_mask(0, tokens, offset, context)?;
                            let selected_pool_mask = full_pool_mask
                                .as_ref()
                                .map(|mask| B::gather_pooled_mask(mask, positions, context))
                                .transpose()?;
                            B::indexed_attention(
                                IndexedAttentionInput {
                                    queries: &query,
                                    local_keys: &local,
                                    local_values: &local,
                                    pooled_keys: &pooled,
                                    pooled_values: &pooled,
                                    selected_positions: positions,
                                    scale: self.scale,
                                    local_mask: Some(local_mask),
                                    pooled_mask: selected_pool_mask.as_ref(),
                                    sinks: Some(self.sinks.as_ref()),
                                },
                                context,
                            )?
                        }
                    }
                    V4AttentionPolicy::Compressed { .. } => {
                        let pooled = self
                            .compressor
                            .as_mut()
                            .expect("compressed V4 layer has a compressor")
                            .forward(input, Some(cache), 0, offset, context)?;
                        let pooled_mask = cache.pooling_mask(0, tokens, offset, context)?;
                        B::pooled_attention(
                            PooledAttentionInput {
                                queries: &query,
                                local: &local,
                                pooled: &pooled,
                                scale: self.scale,
                                local_mask: Some(local_mask),
                                pooled_mask: pooled_mask.as_ref(),
                                sinks: Some(self.sinks.as_ref()),
                            },
                            context,
                        )?
                    }
                }
            }
        };
        output = self.rope.apply(&output, offset, true, context)?;
        let heads_per_group = self.heads / self.groups;
        output = output
            .reshape(
                &[
                    batch,
                    self.groups,
                    heads_per_group,
                    tokens,
                    self.head_dimensions,
                ],
                context,
            )?
            .transpose_axes(&[0, 1, 3, 2, 4], context)?
            .reshape(
                &[
                    batch,
                    self.groups,
                    tokens,
                    heads_per_group * self.head_dimensions,
                ],
                context,
            )?;
        output = B::grouped_linear(
            &mut self.wo_a,
            &output,
            self.groups,
            self.output_rank,
            context,
        )?;
        output = output
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, tokens, self.groups * self.output_rank], context)?;
        self.wo_b.forward(&output, context)
    }
}

fn slice_axis<T: Tensor>(
    value: &T,
    axis: usize,
    start: i32,
    end: i32,
    context: &T::Context,
) -> Result<T, Error> {
    let mut indexes = vec![Index::Full; value.shape().len()];
    indexes[axis] = Index::Range(start, end);
    value.index(&indexes, context)
}

fn slice_last<T: Tensor>(
    value: &T,
    start: i32,
    end: i32,
    context: &T::Context,
) -> Result<T, Error> {
    slice_axis(value, value.shape().len() - 1, start, end, context)
}

fn linear<B: NeuralBackend>(
    name: impl Into<String>,
    input: i32,
    output: i32,
    format: eredu_checkpoint::LinearFormat,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    B::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(name)?,
            bias: None,
            format,
        },
        context,
    )
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
