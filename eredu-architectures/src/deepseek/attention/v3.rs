//! Backend-neutral DeepSeek-V3/R1 multi-head latent attention.

use eredu_nn::{
    AttentionMask, BlockwiseAttentionBackend, BlockwiseAttentionSpec, CompressedAttentionCache,
    CompressedAttentionState, Error, Index, LinearOperator, LinearSpec, LowRankProjection,
    NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Parameterized,
    RotaryOperator, RotaryPosition, RotarySpec, Tensor,
};

use crate::{
    decoder::ComponentInstrumentation,
    deepseek::{projection::ProjectionPolicy, V3Args},
};

mod construction;
pub(crate) use construction::AttentionSpec;

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

/// Canonical V3 MLA module. Fused and split physical KV-B layouts bind to the
/// same `kv_b` parameter identity and therefore share this execution path.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: BlockwiseAttentionBackend> {
    #[parameter(skip, metadata)]
    heads: i32,
    #[parameter(skip, metadata)]
    nope: i32,
    #[parameter(skip, metadata)]
    rope_dimensions: i32,
    #[parameter(skip, metadata)]
    value_dimensions: i32,
    #[parameter(skip, metadata)]
    latent_dimensions: i32,
    #[parameter(skip, metadata)]
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
        if B::construction_metadata(context).is_some_and(|metadata| metadata.uses_checked_metadata()) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        AttentionSpec::new(args, layer)?.instantiate::<B>(context)
    }

    /// Runs MLA while retaining only head-independent latent and rotary state.
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
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes and edits the actual post-aggregation channels before the output
    /// projection. Resident and paged compressed caches share this same boundary.
    /// The caller owns normalization, residual addition and any parallel reduction.
    pub fn forward_instrumented<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(input, mask, cache, context, instrumentation)?;
        let attended = instrumentation.apply("attention.channels", attended)?;
        instrumentation.project::<B>(
            "attention.write_input",
            &mut self.output,
            &attended,
            None,
            context,
        )
    }

    fn attend<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
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
                &[batch, tokens, self.heads, self.nope + self.rope_dimensions],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        // Rotary operators index positions along the penultimate axis. Keep
        // sequence there so the phase is independent of the local head count.
        let query_nope = slice_last(&query, 0, self.nope, context)?;
        let query_rope = slice_last(&query, self.nope, self.nope + self.rope_dimensions, context)?;
        let query_rope =
            self.rotary
                .forward(&query_rope, RotaryPosition::Offset(offset), context)?;
        let queries = B::Tensor::concatenate(&[query_nope, query_rope], -1, context)?;
        let kv = self.kv_a.forward(input, context)?;
        let latent = self.kv_norm.forward(
            &slice_last(&kv, 0, self.latent_dimensions, context)?,
            context,
        )?;
        let latent = instrumentation.apply("attention.key_value.latent", latent)?;
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
                    return Ok(attended);
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
        Ok(attended)
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
