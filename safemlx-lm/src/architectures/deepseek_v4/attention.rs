//! Local, compressed, and indexed sparse attention for DeepSeek V4.

use safemlx::{
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        argpartition_axis, broadcast_to, concatenate_axis, einsum,
        indexing::{take_along_axis, NewAxis, TryIndexOp},
        maximum, softmax_axis, zeros_dtype,
    },
    Array, Dtype, Stream,
};

use crate::{
    api::qwen3_5::{QwenLinear as Linear, QwenWeightFormat as WeightFormat},
    nn::attention::indexed_sparse_attention,
    runtime::cache::{ConcatKeyValueCache, KeyValueCache, PoolingCache},
};

use super::{
    layers::{projection_format, rms_norm},
    model::ModelArgs,
};

#[derive(Debug, Clone)]
pub(crate) struct V4Rope {
    rotary_dimensions: i32,
    frequency_scale: i32,
    frequencies: Array,
}

impl V4Rope {
    fn new(
        args: &ModelArgs,
        base: f32,
        yarn: bool,
        frequency_scale: i32,
    ) -> Result<Self, Exception> {
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
            frequencies: Array::from_slice(&frequencies, &[dimensions / 2]),
        })
    }

    fn apply(
        &self,
        input: &Array,
        offset: i32,
        inverse: bool,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let head_dim = input.dim(-1);
        let non_rotary_pairs = (head_dim - self.rotary_dimensions) / 2;
        let rotary = if inverse {
            self.frequencies.multiply(Array::from_f32(-1.0), stream)?
        } else {
            self.frequencies.clone()
        };
        let frequencies = if non_rotary_pairs > 0 {
            let inactive = Array::from_slice(
                &vec![f32::INFINITY; non_rotary_pairs as usize],
                &[non_rotary_pairs],
            );
            concatenate_axis(&[inactive, rotary], 0, stream)?
        } else {
            rotary
        };
        safemlx::fast::rope(
            input,
            head_dim,
            true,
            None::<f32>,
            1.0,
            offset / self.frequency_scale,
            &frequencies,
            stream,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct GroupedOutput {
    pub(crate) groups: i32,
    #[param]
    pub(crate) projections: Vec<Linear>,
}

impl GroupedOutput {
    fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let input_dims = args.num_attention_heads * args.head_dim / args.o_groups;
        Ok(Self {
            groups: args.o_groups,
            projections: (0..args.o_groups)
                .map(|_| {
                    Linear::new(
                        input_dims,
                        args.o_lora_rank,
                        false,
                        projection_format(args),
                        stream,
                    )
                })
                .collect::<Result<_, _>>()?,
        })
    }

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let mut outputs = Vec::with_capacity(self.groups as usize);
        for group in 0..self.groups {
            let selected = input.try_index_device((.., group, .., ..), stream)?;
            outputs.push(
                self.projections[group as usize]
                    .forward(&selected, stream)?
                    .try_index_device((.., NewAxis, .., ..), stream)?,
            );
        }
        concatenate_axis(&outputs, 1, stream)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Compressor {
    pub(crate) ratio: i32,
    pub(crate) head_dim: i32,
    pub(crate) overlap: bool,
    #[param]
    pub(crate) wkv: Linear,
    #[param]
    pub(crate) wgate: Linear,
    #[param]
    pub(crate) ape: Param<Array>,
    #[param]
    pub(crate) norm: nn::RmsNorm,
    pub(crate) rope: V4Rope,
}

impl Compressor {
    fn new(
        args: &ModelArgs,
        ratio: i32,
        head_dim: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let overlap = ratio == 4;
        let output = head_dim * if overlap { 2 } else { 1 };
        Ok(Self {
            ratio,
            head_dim,
            overlap,
            wkv: Linear::new(args.hidden_size, output, false, WeightFormat::Dense, stream)?,
            wgate: Linear::new(args.hidden_size, output, false, WeightFormat::Dense, stream)?,
            ape: Param::unloaded(&[ratio, output], Dtype::Float32, stream)?,
            norm: rms_norm(head_dim, args.rms_norm_eps, stream)?,
            rope: V4Rope::new(args, args.compress_rope_theta, true, ratio)?,
        })
    }

    fn forward(
        &mut self,
        input: &Array,
        mut cache: Option<&mut PoolingCache>,
        offset: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let values = self.wkv.forward(input, stream)?;
        let gates = self.wgate.forward(input, stream)?;
        let batch = values.dim(0);
        let windows = if let Some(cache) = cache.as_deref_mut() {
            cache.accumulate_windows(values, gates, offset, stream)?
        } else {
            let usable = values.dim(1) / self.ratio * self.ratio;
            crate::runtime::cache::PoolingWindows {
                values: values.try_index_device((.., ..usable, ..), stream)?,
                gates: gates.try_index_device((.., ..usable, ..), stream)?,
                base_position: offset,
            }
        };
        let mut pooled = if windows.values.dim(1) == 0 {
            zeros_dtype(&[batch, 0, self.head_dim], input.dtype(), stream)?
        } else {
            let count = windows.values.dim(1) / self.ratio;
            let values = windows
                .values
                .reshape(&[batch, count, self.ratio, -1], stream)?;
            let gates = windows
                .gates
                .reshape(&[batch, count, self.ratio, -1], stream)?
                .add(self.ape.as_ref(), stream)?;
            if self.overlap {
                let current_values =
                    values.try_index_device((.., -1, .., ..self.head_dim), stream)?;
                let current_gates =
                    gates.try_index_device((.., -1, .., ..self.head_dim), stream)?;
                let (previous_values, previous_gates) = match cache.as_deref_mut() {
                    Some(cache) => cache.replace_overlap(current_values, current_gates),
                    None => (None, None),
                };
                overlap_pool(
                    &values,
                    &gates,
                    previous_values.as_ref(),
                    previous_gates.as_ref(),
                    self.head_dim,
                    stream,
                )?
            } else {
                let weights =
                    softmax_axis(gates, -2, true, stream)?.as_dtype(values.dtype(), stream)?;
                values
                    .multiply(weights, stream)?
                    .sum_axis(-2, false, stream)?
            }
        };
        pooled = self.norm.forward(&pooled, stream)?;
        pooled = self.rope.apply(
            &pooled.try_index_device((.., NewAxis, .., ..), stream)?,
            windows.base_position,
            false,
            stream,
        )?;
        pooled = pooled.try_index_device((.., 0, .., ..), stream)?;
        if let Some(cache) = cache {
            cache.update_and_fetch(pooled, stream)
        } else {
            Ok(pooled)
        }
    }
}

fn overlap_pool(
    values: &Array,
    gates: &Array,
    cached_values: Option<&Array>,
    cached_gates: Option<&Array>,
    head_dim: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let batch = values.dim(0);
    let windows = values.dim(1);
    let ratio = values.dim(2);
    let first_values = match cached_values {
        Some(values) => values.try_index_device((.., NewAxis, .., ..), stream)?,
        None => zeros_dtype(&[batch, 1, ratio, head_dim], values.dtype(), stream)?,
    };
    let first_gates = match cached_gates {
        Some(gates) => gates.try_index_device((.., NewAxis, .., ..), stream)?,
        None => Array::full::<f32>(
            &[batch, 1, ratio, head_dim],
            Array::from_f32(f32::NEG_INFINITY),
            stream,
        )?,
    };
    let previous_values = concatenate_axis(
        &[
            first_values,
            values.try_index_device((.., ..windows - 1, .., ..head_dim), stream)?,
        ],
        1,
        stream,
    )?;
    let current_values = values.try_index_device((.., .., .., head_dim..), stream)?;
    let values = concatenate_axis(&[previous_values, current_values], 2, stream)?;
    let previous_gates = concatenate_axis(
        &[
            first_gates,
            gates.try_index_device((.., ..windows - 1, .., ..head_dim), stream)?,
        ],
        1,
        stream,
    )?;
    let current_gates = gates.try_index_device((.., .., .., head_dim..), stream)?;
    let gates = concatenate_axis(&[previous_gates, current_gates], 2, stream)?;
    let weights = softmax_axis(gates, -2, true, stream)?.as_dtype(values.dtype(), stream)?;
    values
        .multiply(weights, stream)?
        .sum_axis(-2, false, stream)
}

#[derive(Debug, Clone)]
pub(crate) enum AttentionCache {
    Local(ConcatKeyValueCache),
    Compressed {
        local: ConcatKeyValueCache,
        pool: PoolingCache,
    },
    Sparse {
        local: ConcatKeyValueCache,
        pool: PoolingCache,
        index_pool: PoolingCache,
    },
}

impl AttentionCache {
    pub(crate) fn offset(&self) -> i32 {
        match self {
            Self::Local(cache)
            | Self::Compressed { local: cache, .. }
            | Self::Sparse { local: cache, .. } => cache.offset(),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Indexer {
    pub(crate) heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) top_k: i32,
    #[param]
    pub(crate) wq_b: Linear,
    #[param]
    pub(crate) weights_proj: Linear,
    #[param]
    pub(crate) compressor: Compressor,
}

impl Indexer {
    fn new(args: &ModelArgs, ratio: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            heads: args.index_n_heads,
            head_dim: args.index_head_dim,
            top_k: args.index_topk,
            wq_b: Linear::new(
                args.q_lora_rank,
                args.index_n_heads * args.index_head_dim,
                false,
                projection_format(args),
                stream,
            )?,
            weights_proj: Linear::new(
                args.hidden_size,
                args.index_n_heads,
                false,
                WeightFormat::Dense,
                stream,
            )?,
            compressor: Compressor::new(args, ratio, args.index_head_dim, stream)?,
        })
    }

    fn forward(
        &mut self,
        input: &Array,
        query_residual: &Array,
        rope: &V4Rope,
        cache: &mut PoolingCache,
        offset: i32,
        stream: &Stream,
    ) -> Result<(Array, Option<Array>), Exception> {
        let pooled = self
            .compressor
            .forward(input, Some(cache), offset, stream)?;
        if pooled.dim(1) == 0 {
            return Ok((pooled, None));
        }
        let mut query = self.wq_b.forward(query_residual, stream)?.reshape(
            &[input.dim(0), input.dim(1), self.heads, self.head_dim],
            stream,
        )?;
        query = query.transpose_axes(&[0, 2, 1, 3], stream)?;
        query = rope.apply(&query, offset, false, stream)?;
        let scores = einsum(
            "bhld,bpd->bhlp",
            [
                &query.as_dtype(Dtype::Float32, stream)?,
                &pooled.as_dtype(Dtype::Float32, stream)?,
            ],
            stream,
        )?;
        let scores = maximum(scores, Array::from_f32(0.0), stream)?.multiply(
            Array::from_f32((self.head_dim as f32).sqrt().recip()),
            stream,
        )?;
        let head_weights = self
            .weights_proj
            .forward(input, stream)?
            .as_dtype(Dtype::Float32, stream)?
            .multiply(Array::from_f32((self.heads as f32).sqrt().recip()), stream)?
            .transpose_axes(&[0, 2, 1], stream)?
            .try_index_device((.., .., .., NewAxis), stream)?;
        let mut scores = scores
            .multiply(head_weights, stream)?
            .sum_axis(1, false, stream)?;
        if let Some(mask) = cache.make_mask(input.dim(1), offset, stream)? {
            scores =
                safemlx::ops::r#where(mask, scores, Array::from_f32(f32::NEG_INFINITY), stream)?;
        }
        let top_k = self.top_k.min(pooled.dim(1));
        let indices = argpartition_axis(&scores, -top_k, -1, stream)?
            .try_index_device((.., .., -top_k..), stream)?;
        Ok((pooled, Some(indices)))
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct Attention {
    pub(crate) ratio: i32,
    pub(crate) heads: i32,
    pub(crate) head_dim: i32,
    pub(crate) groups: i32,
    pub(crate) scale: f32,
    pub(crate) norm_epsilon: f32,
    #[param]
    pub(crate) wq_a: Linear,
    #[param]
    pub(crate) q_norm: nn::RmsNorm,
    #[param]
    pub(crate) wq_b: Linear,
    #[param]
    pub(crate) wkv: Linear,
    #[param]
    pub(crate) kv_norm: nn::RmsNorm,
    #[param]
    pub(crate) wo_a: GroupedOutput,
    #[param]
    pub(crate) wo_b: Linear,
    #[param]
    pub(crate) attn_sink: Param<Array>,
    #[param]
    pub(crate) compressor: Option<Compressor>,
    #[param]
    pub(crate) indexer: Option<Indexer>,
    pub(crate) rope: V4Rope,
}

impl Attention {
    pub(crate) fn new(
        args: &ModelArgs,
        layer_index: usize,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let ratio = args.compress_ratios[layer_index];
        Ok(Self {
            ratio,
            heads: args.num_attention_heads,
            head_dim: args.head_dim,
            groups: args.o_groups,
            scale: (args.head_dim as f32).sqrt().recip(),
            norm_epsilon: args.rms_norm_eps,
            wq_a: Linear::new(
                args.hidden_size,
                args.q_lora_rank,
                false,
                projection_format(args),
                stream,
            )?,
            q_norm: rms_norm(args.q_lora_rank, args.rms_norm_eps, stream)?,
            wq_b: Linear::new(
                args.q_lora_rank,
                args.num_attention_heads * args.head_dim,
                false,
                projection_format(args),
                stream,
            )?,
            wkv: Linear::new(
                args.hidden_size,
                args.head_dim,
                false,
                projection_format(args),
                stream,
            )?,
            kv_norm: rms_norm(args.head_dim, args.rms_norm_eps, stream)?,
            wo_a: GroupedOutput::new(args, stream)?,
            wo_b: Linear::new(
                args.o_groups * args.o_lora_rank,
                args.hidden_size,
                false,
                projection_format(args),
                stream,
            )?,
            attn_sink: Param::unloaded(&[args.num_attention_heads], Dtype::Float32, stream)?,
            compressor: if ratio == 0 {
                None
            } else {
                Some(Compressor::new(args, ratio, args.head_dim, stream)?)
            },
            indexer: if ratio == 4 {
                Some(Indexer::new(args, ratio, stream)?)
            } else {
                None
            },
            rope: V4Rope::new(
                args,
                if ratio == 0 {
                    args.rope_theta
                } else {
                    args.compress_rope_theta
                },
                ratio != 0,
                1,
            )?,
        })
    }

    pub(crate) fn new_cache(&self, sliding_window: i32) -> Result<AttentionCache, Exception> {
        let local = ConcatKeyValueCache::new_for_sliding_attention(sliding_window);
        match self.ratio {
            0 => Ok(AttentionCache::Local(local)),
            4 => Ok(AttentionCache::Sparse {
                local,
                pool: PoolingCache::new(4)?,
                index_pool: PoolingCache::new(4)?,
            }),
            ratio => Ok(AttentionCache::Compressed {
                local,
                pool: PoolingCache::new(ratio)?,
            }),
        }
    }

    pub(crate) fn forward(
        &mut self,
        input: &Array,
        _mask: Option<&Array>,
        cache: Option<&mut AttentionCache>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = input.dim(0);
        let tokens = input.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let query_residual = self
            .q_norm
            .forward(&self.wq_a.forward(input, stream)?, stream)?;
        let mut query = self
            .wq_b
            .forward(&query_residual, stream)?
            .reshape(&[batch, tokens, self.heads, self.head_dim], stream)?;
        query = rms_without_weight(&query, self.norm_epsilon, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        query = self.rope.apply(&query, offset, false, stream)?;
        let mut kv = self
            .kv_norm
            .forward(&self.wkv.forward(input, stream)?, stream)?
            .reshape(&[batch, 1, tokens, self.head_dim], stream)?;
        kv = self.rope.apply(&kv, offset, false, stream)?;

        let mut output = match cache {
            Some(AttentionCache::Local(local)) => {
                let (kv, _) = local.update_and_fetch(
                    kv,
                    zeros_dtype(&[batch, 1, tokens, 0], input.dtype(), stream)?,
                    stream,
                )?;
                let local_mask = causal_local_mask(tokens, offset, kv.dim(2), stream)?;
                dense_attention(
                    &query,
                    &kv,
                    Some(&local_mask),
                    self.scale,
                    self.attn_sink.as_ref(),
                    stream,
                )?
            }
            Some(AttentionCache::Compressed { local, pool }) => {
                let (local_kv, _) = local.update_and_fetch(
                    kv,
                    zeros_dtype(&[batch, 1, tokens, 0], input.dtype(), stream)?,
                    stream,
                )?;
                let pooled = self
                    .compressor
                    .as_mut()
                    .expect("compressed layer")
                    .forward(input, Some(pool), offset, stream)?;
                let local_tokens = local_kv.dim(2);
                let full = concatenate_axis(
                    &[
                        local_kv,
                        pooled.try_index_device((.., NewAxis, .., ..), stream)?,
                    ],
                    2,
                    stream,
                )?;
                let local_mask = causal_local_mask(tokens, offset, local_tokens, stream)?;
                let extended = extend_mask(
                    Some(&local_mask),
                    pool.make_mask(tokens, offset, stream)?,
                    full.dim(2),
                    stream,
                )?;
                dense_attention(
                    &query,
                    &full,
                    extended.as_ref(),
                    self.scale,
                    self.attn_sink.as_ref(),
                    stream,
                )?
            }
            Some(AttentionCache::Sparse {
                local,
                pool,
                index_pool,
            }) => {
                let (local_kv, _) = local.update_and_fetch(
                    kv,
                    zeros_dtype(&[batch, 1, tokens, 0], input.dtype(), stream)?,
                    stream,
                )?;
                let pooled = self
                    .compressor
                    .as_mut()
                    .expect("sparse compressor")
                    .forward(input, Some(pool), offset, stream)?;
                let (_, topk) = self.indexer.as_mut().expect("sparse indexer").forward(
                    input,
                    &query_residual,
                    &self.rope,
                    index_pool,
                    offset,
                    stream,
                )?;
                if pooled.dim(1) == 0 || pooled.dim(1) <= self.indexer.as_ref().unwrap().top_k {
                    let local_tokens = local_kv.dim(2);
                    let full = concatenate_axis(
                        &[
                            local_kv,
                            pooled.try_index_device((.., NewAxis, .., ..), stream)?,
                        ],
                        2,
                        stream,
                    )?;
                    let local_mask = causal_local_mask(tokens, offset, local_tokens, stream)?;
                    let extended = extend_mask(
                        Some(&local_mask),
                        pool.make_mask(tokens, offset, stream)?,
                        full.dim(2),
                        stream,
                    )?;
                    dense_attention(
                        &query,
                        &full,
                        extended.as_ref(),
                        self.scale,
                        self.attn_sink.as_ref(),
                        stream,
                    )?
                } else {
                    let topk = topk.as_ref().expect("non-empty pooled index");
                    let local_mask = causal_local_mask(tokens, offset, local_kv.dim(2), stream)?;
                    let pooled_mask = pool
                        .make_mask(tokens, offset, stream)?
                        .map(|pool_mask| -> Result<Array, Exception> {
                            let pool_mask = broadcast_to(
                                &pool_mask.try_index_device((NewAxis, .., ..), stream)?,
                                &[batch, tokens, pooled.dim(1)],
                                stream,
                            )?;
                            take_along_axis(&pool_mask, topk, 2, stream)?
                                .try_index_device((.., NewAxis, .., ..), stream)
                        })
                        .transpose()?;
                    indexed_sparse_attention(
                        &query,
                        &local_kv.try_index_device((.., 0, .., ..), stream)?,
                        &pooled,
                        topk,
                        self.scale,
                        Some(&local_mask),
                        pooled_mask.as_ref(),
                        Some(self.attn_sink.as_ref()),
                        stream,
                    )?
                }
            }
            None => {
                let local_mask = causal_local_mask(tokens, offset, kv.dim(2), stream)?;
                dense_attention(
                    &query,
                    &kv,
                    Some(&local_mask),
                    self.scale,
                    self.attn_sink.as_ref(),
                    stream,
                )?
            }
        };
        output = self.rope.apply(&output, offset, true, stream)?;
        output = output
            .reshape(&[batch, self.groups, -1, tokens, self.head_dim], stream)?
            .transpose_axes(&[0, 1, 3, 2, 4], stream)?
            .reshape(&[batch, self.groups, tokens, -1], stream)?;
        output = self
            .wo_a
            .forward(&output, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[batch, tokens, -1], stream)?;
        self.wo_b.forward(&output, stream)
    }
}

fn dense_attention(
    query: &Array,
    kv: &Array,
    mask: Option<&Array>,
    scale: f32,
    sinks: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    safemlx::fast::scaled_dot_product_attention(
        query,
        kv,
        kv,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
        sinks,
        stream,
    )
}

fn causal_local_mask(
    query_tokens: i32,
    query_offset: i32,
    key_tokens: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let key_offset = query_offset + query_tokens - key_tokens;
    let queries = Array::arange::<i32, i32>(
        Some(query_offset),
        query_offset + query_tokens,
        None,
        stream,
    )?
    .try_index_device((.., NewAxis), stream)?;
    let keys = Array::arange::<i32, i32>(Some(key_offset), key_offset + key_tokens, None, stream)?
        .try_index_device((NewAxis, ..), stream)?;
    queries.ge(keys, stream)
}

fn extend_mask(
    mask: Option<&Array>,
    pool_mask: Option<Array>,
    total: i32,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    let Some(mask) = mask else { return Ok(None) };
    let local = mask.dim(-1);
    let pooled = total - local;
    if pooled <= 0 {
        return Ok(Some(mask.clone()));
    }
    let prefix = mask.shape()[..mask.ndim() - 1].to_vec();
    let pool = match pool_mask {
        Some(pool) => {
            let mut shape = vec![1; mask.ndim()];
            shape[mask.ndim() - 2] = pool.dim(0);
            shape[mask.ndim() - 1] = pool.dim(1);
            broadcast_to(
                &pool.reshape(&shape, stream)?,
                &[prefix, vec![pooled]].concat(),
                stream,
            )?
        }
        None => Array::ones::<bool>(&[prefix, vec![pooled]].concat(), stream)?,
    };
    Ok(Some(concatenate_axis(&[mask.clone(), pool], -1, stream)?))
}

fn rms_without_weight(input: &Array, epsilon: f32, stream: &Stream) -> Result<Array, Exception> {
    let dtype = input.dtype();
    let variance = input.square(stream)?.mean_axis(-1, true, stream)?;
    input
        .multiply(
            variance
                .add(Array::from_f32(epsilon), stream)?
                .rsqrt(stream)?,
            stream,
        )?
        .as_dtype(dtype, stream)
}
