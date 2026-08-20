//! Kimi Linear no-positional multi-head latent attention.

use eredu_nn::{
    AttentionMask, BlockwiseAttentionBackend, BlockwiseAttentionSpec, CompressedAttentionCache,
    CompressedAttentionState, Error, Index, LinearOperator, LinearSpec, LowRankProjection,
    LowRankProjectionSpec, NormalizationOperator, NormalizationSpec, ParameterSpec, Parameterized,
    Tensor,
};

use super::ModelArgs;

/// Direct or normalized low-rank Kimi query projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum QueryProjection<B: BlockwiseAttentionBackend> {
    /// Direct hidden-to-query projection.
    Direct(B::Linear),
    /// Normalized two-stage query projection.
    LowRank(LowRankProjection<B>),
}

/// Fused or split physical MLA latent reconstruction.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum LatentProjection<B: BlockwiseAttentionBackend> {
    /// One projection emits interleaved non-positional keys and values.
    Fused(B::Linear),
    /// Independent projections emit all per-head keys and values.
    Split {
        /// Latent-to-key projection.
        key: B::Linear,
        /// Latent-to-value projection.
        value: B::Linear,
    },
}

impl<B: BlockwiseAttentionBackend> QueryProjection<B> {
    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Direct(projection) => projection.forward(input, context),
            Self::LowRank(projection) => projection.forward(input, context),
        }
    }
}

/// Backend-neutral Kimi MLA retaining head-independent latent state.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct KimiLatentAttention<B: BlockwiseAttentionBackend> {
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    nope_dimensions: i32,
    #[parameter(skip)]
    positional_dimensions: i32,
    #[parameter(skip)]
    value_dimensions: i32,
    #[parameter(skip)]
    latent_dimensions: i32,
    #[parameter(skip)]
    scale: f32,
    query: QueryProjection<B>,
    kv_a: B::Linear,
    kv_norm: B::Normalization,
    kv_b: LatentProjection<B>,
    output: B::Linear,
}

impl<B: BlockwiseAttentionBackend> KimiLatentAttention<B> {
    /// Creates unloaded no-positional MLA parameters for one physical layer.
    pub fn new(
        args: &ModelArgs,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_heads(args, layer, args.num_attention_heads, context)
    }

    /// Creates unloaded MLA for a placement-resolved query-head count.
    pub fn new_with_heads(
        args: &ModelArgs,
        layer: usize,
        heads: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let root = format!("model.layers.{layer}.self_attn");
        let query_dimensions = heads
            .checked_mul(args.qk_nope_head_dim + args.qk_rope_head_dim)
            .ok_or_else(|| Error::backend("Kimi MLA query width overflowed"))?;
        let linear_spec = |name: String, input, output| -> Result<LinearSpec, Error> {
            let format = args.weight_quantization_for(&name).into();
            Ok(LinearSpec {
                input,
                output,
                weight: parameter(name)?,
                bias: None,
                format,
            })
        };
        let query = if let Some(rank) = args.q_lora_rank {
            QueryProjection::LowRank(LowRankProjection::new(
                LowRankProjectionSpec {
                    first: Some(linear_spec(
                        format!("{root}.q_a_proj.weight"),
                        args.hidden_size,
                        rank,
                    )?),
                    normalization: NormalizationSpec {
                        dimensions: rank,
                        epsilon: args.rms_norm_eps,
                        weight: parameter(format!("{root}.q_a_layernorm.weight"))?,
                    },
                    second: linear_spec(format!("{root}.q_b_proj.weight"), rank, query_dimensions)?,
                },
                context,
            )?)
        } else {
            QueryProjection::Direct(B::linear(
                linear_spec(
                    format!("{root}.q_proj.weight"),
                    args.hidden_size,
                    query_dimensions,
                )?,
                context,
            )?)
        };
        Ok(Self {
            heads,
            nope_dimensions: args.qk_nope_head_dim,
            positional_dimensions: args.qk_rope_head_dim,
            value_dimensions: args.v_head_dim,
            latent_dimensions: args.kv_lora_rank,
            scale: ((args.qk_nope_head_dim + args.qk_rope_head_dim) as f32)
                .sqrt()
                .recip(),
            query,
            kv_a: B::linear(
                linear_spec(
                    format!("{root}.kv_a_proj_with_mqa.weight"),
                    args.hidden_size,
                    args.kv_lora_rank + args.qk_rope_head_dim,
                )?,
                context,
            )?,
            kv_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: args.kv_lora_rank,
                    epsilon: args.rms_norm_eps,
                    weight: parameter(format!("{root}.kv_a_layernorm.weight"))?,
                },
                context,
            )?,
            kv_b: if args.split_kv_b {
                LatentProjection::Split {
                    key: B::linear(
                        linear_spec(
                            format!("{root}.k_b_proj.weight"),
                            args.kv_lora_rank,
                            heads * args.qk_nope_head_dim,
                        )?,
                        context,
                    )?,
                    value: B::linear(
                        linear_spec(
                            format!("{root}.v_b_proj.weight"),
                            args.kv_lora_rank,
                            heads * args.v_head_dim,
                        )?,
                        context,
                    )?,
                }
            } else {
                LatentProjection::Fused(B::linear(
                    linear_spec(
                        format!("{root}.kv_b_proj.weight"),
                        args.kv_lora_rank,
                        heads * (args.qk_nope_head_dim + args.v_head_dim),
                    )?,
                    context,
                )?)
            },
            output: B::linear(
                linear_spec(
                    format!("{root}.o_proj.weight"),
                    heads * args.v_head_dim,
                    args.hidden_size,
                )?,
                context,
            )?,
        })
    }

    /// Executes no-positional MLA against resident or block-addressable state.
    pub fn forward<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_inner(input, mask, cache, None, context)
    }

    /// Executes MLA with a row-parallel output projection.
    pub fn forward_parallel<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_inner(input, mask, cache, Some(parallel), context)
    }

    fn forward_inner<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let batch = input.dim(0);
        let tokens = input.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let query = self.query.forward(input, context)?.reshape(
            &[
                batch,
                tokens,
                self.heads,
                self.nope_dimensions + self.positional_dimensions,
            ],
            context,
        )?;
        let queries = query.transpose_axes(&[0, 2, 1, 3], context)?;
        let kv = self.kv_a.forward(input, context)?;
        let latent = self.kv_norm.forward(
            &slice_last(&kv, 0, self.latent_dimensions, context)?,
            context,
        )?;
        let rotary = slice_last(
            &kv,
            self.latent_dimensions,
            self.latent_dimensions + self.positional_dimensions,
            context,
        )?;
        let current = CompressedAttentionState { latent, rotary };

        let (latent, positional) = match cache {
            None => (current.latent, current.rotary),
            Some(cache) => {
                let view = cache.append(current, context)?;
                if !view.is_paged() {
                    let state = view.observable();
                    (state.latent.clone(), state.rotary.clone())
                } else {
                    let mut accumulator = B::begin_blockwise_attention(
                        BlockwiseAttentionSpec {
                            queries: &queries,
                            scale: self.scale,
                            mask,
                            query_start: i64::from(offset),
                            context_end: i64::from(offset) + i64::from(tokens),
                            sliding_window: None,
                            prefix_tokens: 0,
                            sinks: None,
                        },
                        context,
                    )?;
                    cache.visit_blocks(tokens, context, |block| {
                        let (keys, values) =
                            self.reconstruct(&block.state.latent, &block.state.rotary, context)?;
                        B::accumulate_blockwise_attention(
                            &mut accumulator,
                            block.start,
                            block.end,
                            keys,
                            values,
                            context,
                        )
                    })?;
                    let attended = B::finish_blockwise_attention(accumulator, context)?
                        .transpose_axes(&[0, 2, 1, 3], context)?
                        .reshape(
                            &[batch, tokens, self.heads * self.value_dimensions],
                            context,
                        )?;
                    return self.project_output(&attended, parallel, context);
                }
            }
        };
        let (keys, values) = self.reconstruct(&latent, &positional, context)?;
        let attention_mask = match mask {
            Some(mask) => AttentionMask::Tensor(mask),
            None if tokens > 1 => AttentionMask::Causal,
            None => AttentionMask::None,
        };
        let attended = B::Tensor::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            self.scale,
            attention_mask,
            context,
        )?
        .transpose_axes(&[0, 2, 1, 3], context)?
        .reshape(
            &[batch, tokens, self.heads * self.value_dimensions],
            context,
        )?;
        self.project_output(&attended, parallel, context)
    }

    fn project_output(
        &mut self,
        attended: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match parallel {
            Some(parallel) => B::row_parallel_linear(&mut self.output, attended, parallel, context),
            None => self.output.forward(attended, context),
        }
    }

    fn reconstruct(
        &mut self,
        latent: &B::Tensor,
        positional: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        let batch = latent.dim(0);
        let tokens = latent.dim(1);
        let (key_nope, values) = match &mut self.kv_b {
            LatentProjection::Fused(projection) => {
                let projected = projection.forward(latent, context)?.reshape(
                    &[
                        batch,
                        tokens,
                        self.heads,
                        self.nope_dimensions + self.value_dimensions,
                    ],
                    context,
                )?;
                (
                    slice_last(&projected, 0, self.nope_dimensions, context)?,
                    slice_last(
                        &projected,
                        self.nope_dimensions,
                        self.nope_dimensions + self.value_dimensions,
                        context,
                    )?
                    .transpose_axes(&[0, 2, 1, 3], context)?,
                )
            }
            LatentProjection::Split { key, value } => (
                key.forward(latent, context)?
                    .reshape(&[batch, tokens, self.heads, self.nope_dimensions], context)?,
                value
                    .forward(latent, context)?
                    .reshape(&[batch, tokens, self.heads, self.value_dimensions], context)?
                    .transpose_axes(&[0, 2, 1, 3], context)?,
            ),
        };
        let positional = positional.expand_dims(2, context)?;
        let positional = B::Tensor::concatenate(
            &vec![positional; usize::try_from(self.heads).map_err(Error::backend)?],
            2,
            context,
        )?;
        let keys = B::Tensor::concatenate(&[key_nope, positional], -1, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        Ok((keys, values))
    }
}

fn slice_last<T: Tensor>(
    value: &T,
    start: i32,
    end: i32,
    context: &T::Context,
) -> Result<T, Error> {
    let mut indexes = vec![Index::Full; value.shape().len()];
    let last = indexes.len() - 1;
    indexes[last] = Index::Range(start, end);
    value.index(&indexes, context)
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
