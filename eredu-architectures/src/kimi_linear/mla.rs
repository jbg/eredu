//! Kimi Linear no-positional multi-head latent attention.

use eredu_nn::{
    AttentionMask, BlockwiseAttentionBackend, BlockwiseAttentionSpec, CompressedAttentionCache,
    CompressedAttentionState, Error, Index, LinearOperator, LowRankProjection,
    NormalizationOperator, Parameterized, Tensor,
};

use crate::decoder::ComponentInstrumentation;

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
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Direct(projection) => projection.forward(input, context),
            Self::LowRank(projection) => {
                projection.forward_with_normalized_rank(input, context, |rank, _| {
                    instrumentation.apply("attention.query.latent", rank)
                })
            }
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
    pub(crate) fn from_spec(
        spec: super::specs::MlaSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        use eredu_nn::DeclareLogicalParameters;
        spec.logical_parameters()?;
        let geometry = spec.geometry;
        let query = match spec.query {
            super::specs::QuerySpec::Direct(projection) => {
                QueryProjection::Direct(B::linear(projection, context)?)
            }
            super::specs::QuerySpec::LowRank(projection) => {
                QueryProjection::LowRank(LowRankProjection::new(projection, context)?)
            }
        };
        Ok(Self {
            heads: geometry.heads(),
            nope_dimensions: geometry.non_positional_dimensions(),
            positional_dimensions: geometry.positional_dimensions(),
            value_dimensions: geometry.value_dimensions(),
            latent_dimensions: geometry.latent_dimensions(),
            scale: spec.scale,
            query,
            kv_a: B::linear(spec.kv_a, context)?,
            kv_norm: B::normalization(spec.kv_norm, context)?,
            kv_b: match spec.kv_b {
                super::specs::LatentSpec::Fused(projection) => {
                    LatentProjection::Fused(B::linear(projection, context)?)
                }
                super::specs::LatentSpec::Split { key, value } => LatentProjection::Split {
                    key: B::linear(key, context)?,
                    value: B::linear(value, context)?,
                },
            },
            output: B::linear(spec.output, context)?,
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
        self.forward_instrumented(
            input,
            mask,
            cache,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
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
        self.forward_instrumented(
            input,
            mask,
            cache,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes resident or paged MLA with the same latent and channel hooks.
    pub fn forward_instrumented<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let batch = input.dim(0);
        let tokens = input.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let query = self
            .query
            .forward(input, context, instrumentation)?
            .reshape(
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
        let latent = instrumentation.apply("attention.key_value.latent", latent)?;
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
                    return self.project_output(attended, parallel, context, instrumentation);
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
        self.project_output(attended, parallel, context, instrumentation)
    }

    fn project_output(
        &mut self,
        attended: B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let channels = instrumentation.apply("attention.channels", attended)?;
        instrumentation.project::<B>(
            "attention.write_input",
            &mut self.output,
            &channels,
            parallel,
            context,
        )
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
