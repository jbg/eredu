//! Backend-neutral DeepSeek-V3/R1 multi-head latent attention.

use eredu_nn::{
    AttentionMask, BlockwiseAttentionBackend, BlockwiseAttentionSpec, CompressedAttentionCache,
    CompressedAttentionState, Error, Index, LinearOperator, LinearSpec, LowRankProjection,
    NormalizationOperator, NormalizationSpec, ParameterSpec, Parameterized, RotaryOperator,
    RotaryPosition, RotarySpec, Tensor,
};

use crate::deepseek::{projection::ProjectionPolicy, V3Args};

/// Direct or normalized low-rank query projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum QueryProjection<B: BlockwiseAttentionBackend> {
    /// Published checkpoints without query LoRA.
    Direct(B::Linear),
    /// Published checkpoints with query LoRA.
    LowRank(LowRankProjection<B>),
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

/// Canonical V3 MLA module. Fused and split physical KV-B layouts bind to the
/// same `kv_b` parameter identity and therefore share this execution path.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: BlockwiseAttentionBackend> {
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    nope: i32,
    #[parameter(skip)]
    rope_dimensions: i32,
    #[parameter(skip)]
    value_dimensions: i32,
    #[parameter(skip)]
    latent_dimensions: i32,
    #[parameter(skip)]
    scale: f32,
    query: QueryProjection<B>,
    kv_a: B::Linear,
    kv_norm: B::Normalization,
    kv_b: B::Linear,
    output: B::Linear,
    rotary: B::Rotary,
}

impl<B: BlockwiseAttentionBackend> Attention<B> {
    /// Builds one unloaded target or MTP MLA layer under its canonical global
    /// parameter root.
    pub fn new(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let root = format!("model.layers.{layer}.self_attn");
        let query_width = args
            .num_attention_heads
            .checked_mul(args.qk_nope_head_dim + args.qk_rope_head_dim)
            .ok_or_else(|| Error::backend("V3 query width overflowed"))?;
        let linear = |name: String, input, output| {
            let format = args.linear_format_for(&name);
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
        };
        let query = if let Some(rank) = args.q_lora_rank {
            QueryProjection::LowRank(
                ProjectionPolicy {
                    first_weight: Some(format!("{root}.q_a_proj.weight")),
                    normalization_weight: format!("{root}.q_a_layernorm.weight"),
                    second_weight: format!("{root}.q_b_proj.weight"),
                    input_dimensions: args.hidden_size,
                    rank,
                    output_dimensions: query_width,
                    epsilon: args.rms_norm_eps,
                    first_format: args.linear_format_for(&format!("{root}.q_a_proj.weight")),
                    second_format: args.linear_format_for(&format!("{root}.q_b_proj.weight")),
                }
                .build(context)?,
            )
        } else {
            QueryProjection::Direct(linear(
                format!("{root}.q_proj.weight"),
                args.hidden_size,
                query_width,
            )?)
        };
        let rope_scaling = args.rope_scaling.as_ref().map(|yarn| yarn.rope_scaling());
        let scale = ((args.qk_nope_head_dim + args.qk_rope_head_dim) as f32)
            .sqrt()
            .recip()
            * args
                .rope_scaling
                .as_ref()
                .map_or(1.0, |yarn| yarn.attention_multiplier());
        Ok(Self {
            heads: args.num_attention_heads,
            nope: args.qk_nope_head_dim,
            rope_dimensions: args.qk_rope_head_dim,
            value_dimensions: args.v_head_dim,
            latent_dimensions: args.kv_lora_rank,
            scale,
            query,
            kv_a: linear(
                format!("{root}.kv_a_proj_with_mqa.weight"),
                args.hidden_size,
                args.kv_lora_rank + args.qk_rope_head_dim,
            )?,
            kv_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: args.kv_lora_rank,
                    epsilon: args.rms_norm_eps,
                    weight: parameter(format!("{root}.kv_a_layernorm.weight"))?,
                },
                context,
            )?,
            kv_b: linear(
                format!("{root}.kv_b_proj.weight"),
                args.kv_lora_rank,
                args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
            )?,
            output: linear(
                format!("{root}.o_proj.weight"),
                args.num_attention_heads * args.v_head_dim,
                args.hidden_size,
            )?,
            rotary: B::rotary(
                RotarySpec {
                    dimensions: args.qk_rope_head_dim,
                    base: args.rope_theta,
                    traditional: false,
                    max_positions: args.max_position_embeddings,
                    scaling: rope_scaling.as_ref(),
                },
                context,
            )?,
        })
    }

    /// Runs MLA while retaining only head-independent latent and rotary state.
    pub fn forward<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let batch = input.dim(0);
        let tokens = input.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let query = self.query.forward(input, context)?.reshape(
            &[batch, tokens, self.heads, self.nope + self.rope_dimensions],
            context,
        )?;
        let query_nope = slice_last(&query, 0, self.nope, context)?;
        let query_rope = slice_last(&query, self.nope, self.nope + self.rope_dimensions, context)?;
        let query_rope =
            self.rotary
                .forward(&query_rope, RotaryPosition::Offset(offset), context)?;
        let queries =
            B::Tensor::concatenate(&[query_nope.clone(), query_rope.clone()], -1, context)?
                .transpose_axes(&[0, 2, 1, 3], context)?;
        let kv = self.kv_a.forward(input, context)?;
        let latent = self.kv_norm.forward(
            &slice_last(&kv, 0, self.latent_dimensions, context)?,
            context,
        )?;
        let rotary = slice_last(
            &kv,
            self.latent_dimensions,
            self.latent_dimensions + self.rope_dimensions,
            context,
        )?;
        let rotary = self
            .rotary
            .forward(&rotary, RotaryPosition::Offset(offset), context)?;
        let current = CompressedAttentionState { latent, rotary };

        let (latent, rotary) = match cache {
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
                        let cached_tokens = block.state.latent.dim(1);
                        let projected = self.kv_b.forward(&block.state.latent, context)?.reshape(
                            &[
                                batch,
                                cached_tokens,
                                self.heads,
                                self.nope + self.value_dimensions,
                            ],
                            context,
                        )?;
                        let key_nope = slice_last(&projected, 0, self.nope, context)?;
                        let values = slice_last(
                            &projected,
                            self.nope,
                            self.nope + self.value_dimensions,
                            context,
                        )?
                        .transpose_axes(&[0, 2, 1, 3], context)?;
                        let rotary = block.state.rotary.expand_dims(2, context)?;
                        let rotary = B::Tensor::concatenate(
                            &vec![rotary; usize::try_from(self.heads).map_err(Error::backend)?],
                            2,
                            context,
                        )?;
                        let keys = B::Tensor::concatenate(&[key_nope, rotary], -1, context)?
                            .transpose_axes(&[0, 2, 1, 3], context)?;
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
                    return self.output.forward(&attended, context);
                }
            }
        };
        let cached_tokens = latent.dim(1);
        let projected = self.kv_b.forward(&latent, context)?.reshape(
            &[
                batch,
                cached_tokens,
                self.heads,
                self.nope + self.value_dimensions,
            ],
            context,
        )?;
        let key_nope = slice_last(&projected, 0, self.nope, context)?;
        let values = slice_last(
            &projected,
            self.nope,
            self.nope + self.value_dimensions,
            context,
        )?
        .transpose_axes(&[0, 2, 1, 3], context)?;
        let rotary = rotary.expand_dims(2, context)?;
        let rotary = B::Tensor::concatenate(
            &vec![rotary; usize::try_from(self.heads).map_err(Error::backend)?],
            2,
            context,
        )?;
        let keys = B::Tensor::concatenate(&[key_nope, rotary], -1, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
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
        self.output.forward(&attended, context)
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
