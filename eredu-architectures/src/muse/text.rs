//! Backend-neutral Muse-Glimmer decoder equations.

use eredu_nn::{
    AttentionCache, AttentionRequest, DeclareLogicalParameters, EmbeddingOperator, Error,
    GroupedGatedProductOperator, GroupedGatedProductSpec, GroupedNeuralBackend, LinearOperator,
    NormalizationOperator, Parameterized, RotaryOperator, RotaryPosition, Tensor,
};
use eredu_runtime::{
    ExpertPass, ParameterProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

use crate::decoder::{ComponentInstrumentation, DecoderStaticResources};

use super::config::{AttentionEquation, VocabularyArithmetic};
use super::equations::{weightless_rms_norm, BranchNorm, LearnedRmsNorm, ResidualCombination};
use super::{DecoderConfig, DecoderPolicy, LocalGeometry};

mod mask;
/// Source-free routed construction, ownership, and placement declarations.
pub mod routed;
pub(crate) mod specs;

/// Gated grouped-query attention with per-layer RoPE/NoPE policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    key_value_heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    window: Option<i32>,
    #[parameter(skip)]
    uses_rope: bool,
    #[parameter(skip)]
    qk_norm_epsilon: f32,
    #[parameter(skip)]
    equation: AttentionEquation,
    #[parameter(skip)]
    arithmetic: eredu_nn::AttentionArithmetic,
    #[parameter(skip)]
    qk_scale_factor: f32,
    #[parameter(skip)]
    query_scale: Option<B::Tensor>,
    /// Query projection.
    pub query: B::Linear,
    /// Key projection.
    pub key: B::Linear,
    /// Value projection.
    pub value: B::Linear,
    /// Per-head sigmoid gate projection.
    pub gate: B::Linear,
    /// Output projection.
    pub output: B::Linear,
    /// Optional synthesized GGUF query norm.
    pub query_norm: Option<B::Normalization>,
    /// Optional synthesized GGUF key norm.
    pub key_norm: Option<B::Normalization>,
    /// Full-head rotary operator, bypassed by NoPE layers.
    pub rotary: B::Rotary,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Attention<B> {
    /// Builds one unloaded scheduled attention unit.
    pub fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::from_spec(args.selected_attention(layer)?, context)
    }

    fn from_spec(
        spec: specs::AttentionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.parameters.logical_parameters()?;
        let window = spec.policy.sliding_window_i32().map_err(Error::backend)?;
        let geometry = spec.parameters.projection.geometry;
        let mut norms = spec.parameters.norms.into_iter();
        Ok(Self {
            query_heads: geometry.query_heads(),
            key_value_heads: geometry.key_value_heads(),
            head_dimensions: geometry.key_dimensions(),
            scale: spec.scale,
            window,
            uses_rope: spec.uses_rope,
            qk_norm_epsilon: spec.epsilon,
            equation: spec.equation,
            arithmetic: spec.arithmetic,
            qk_scale_factor: spec.qk_scale_factor,
            query_scale: spec
                .query_scale
                .map(|scale| B::Tensor::full_f32(scale, &[geometry.key_dimensions()], context))
                .transpose()?,
            query: B::linear(spec.parameters.projection.query, context)?,
            key: B::linear(spec.parameters.projection.key, context)?,
            value: B::linear(spec.parameters.projection.value, context)?,
            gate: B::linear(spec.parameters.gate, context)?,
            output: B::linear(spec.parameters.projection.output, context)?,
            query_norm: norms
                .next()
                .map(|norm| B::normalization(norm, context))
                .transpose()?,
            key_norm: norms
                .next()
                .map(|norm| B::normalization(norm, context))
                .transpose()?,
            rotary: B::rotary(spec.rotary, context)?,
        })
    }

    fn normalize_keys(
        &mut self,
        keys: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self.key_norm.as_mut() {
            Some(norm) => norm.forward(keys, context),
            None if matches!(
                self.equation,
                AttentionEquation::NormalizedHeads | AttentionEquation::NormalizedHeadsScoreScaled
            ) =>
            {
                weightless_rms_norm::<B>(keys, self.qk_norm_epsilon, context)
            }
            None => B::rms_norm_without_weight(keys, self.qk_norm_epsilon, context),
        }
    }

    /// Projects independently encoded committed context using the same key
    /// normalization and rotary policy as an ordinary decoder invocation.
    pub(crate) fn project_context(
        &mut self,
        hidden: &B::Tensor,
        offset: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::DFlashLayerContext<B::Tensor>, Error> {
        let shape = [
            hidden.dim(0),
            hidden.dim(1),
            self.key_value_heads,
            self.head_dimensions,
        ];
        let keys = self
            .key
            .forward(hidden, context)?
            .reshape(&shape, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let keys = self.normalize_keys(&keys, context)?;
        let keys = if self.uses_rope {
            self.rotary
                .forward(&keys, RotaryPosition::Offset(offset), context)?
        } else {
            keys
        };
        let values = self
            .value
            .forward(hidden, context)?
            .reshape(&shape, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        Ok(super::DFlashLayerContext { keys, values })
    }

    fn attend<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let head_dimensions = self.head_dimensions;
        let reshape = |value: B::Tensor, heads| {
            value
                .reshape(&[batch, sequence, heads, head_dimensions], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let mut queries = reshape(self.query.forward(hidden, context)?, self.query_heads)?;
        let mut keys = reshape(self.key.forward(hidden, context)?, self.key_value_heads)?;
        let values = reshape(self.value.forward(hidden, context)?, self.key_value_heads)?;
        queries = match self.query_norm.as_mut() {
            Some(norm) => norm.forward(&queries, context)?,
            None if self.equation == AttentionEquation::NormalizedHeads => {
                weightless_rms_norm::<B>(&queries, self.qk_norm_epsilon, context)?
                    .multiply_scalar(self.qk_scale_factor, context)?
            }
            None if self.equation == AttentionEquation::NormalizedHeadsScoreScaled => {
                weightless_rms_norm::<B>(&queries, self.qk_norm_epsilon, context)?
            }
            None => B::rms_norm_with_weight(
                &queries,
                self.query_scale
                    .as_ref()
                    .ok_or_else(|| Error::backend("Muse-Glimmer query scale is missing"))?,
                self.qk_norm_epsilon,
                context,
            )?,
        };
        keys = self.normalize_keys(&keys, context)?;
        if self.uses_rope {
            queries = self
                .rotary
                .forward(&queries, RotaryPosition::Offset(offset), context)?;
            keys = self
                .rotary
                .forward(&keys, RotaryPosition::Offset(offset), context)?;
        }
        instrumentation.observe("attention.queries", &queries)?;
        let blockwise = cache
            .as_ref()
            .is_some_and(|cache| cache.uses_blockwise_attention());
        let (keys, values, cache) = match cache {
            Some(cache) => {
                let (keys, values) = cache.update_for_attention(keys, values, context)?;
                (keys, values, Some(cache))
            }
            None => (keys, values, None),
        };
        // Paged attention slices a complete logical mask per block. Contiguous
        // sliding caches instead return only retained past plus this invocation.
        let mask_keys = if blockwise {
            offset
                .checked_add(sequence)
                .ok_or_else(|| Error::backend("Muse attention position overflow"))?
        } else {
            keys.dim(2)
        };
        let mask_window = if blockwise { None } else { self.window };
        let mask =
            mask::attention_mask::<B>(explicit_mask, sequence, mask_keys, mask_window, context)?;
        instrumentation.observe("attention.keys", &keys)?;
        instrumentation.observe("attention.values", &values)?;
        let request = AttentionRequest {
            arithmetic: self.arithmetic,
            softcap: None,
            queries,
            keys,
            values,
            scale: self.scale,
            mask: mask.as_ref(),
            sinks: None,
        };
        let attended = match cache {
            Some(cache) => cache.attention(request, context)?,
            None => B::attention_with_sinks(request, context)?,
        };
        instrumentation.observe("attention.attended_heads", &attended)?;
        if self.equation == AttentionEquation::NormalizedHeadsScoreScaled {
            let gate = reshape(self.gate.forward(hidden, context)?, self.query_heads)?;
            return super::equations::weightless_rms_sigmoid_gate::<B>(
                &attended,
                &gate,
                self.qk_norm_epsilon,
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(
                &[batch, sequence, self.query_heads * self.head_dimensions],
                context,
            );
        }
        let attended = match self.equation {
            AttentionEquation::QueryScaled => attended,
            AttentionEquation::NormalizedHeads | AttentionEquation::NormalizedHeadsScoreScaled => {
                weightless_rms_norm::<B>(&attended, self.qk_norm_epsilon, context)?
            }
        };
        let attended = attended.transpose_axes(&[0, 2, 1, 3], context)?.reshape(
            &[batch, sequence, self.query_heads * self.head_dimensions],
            context,
        )?;
        let gate = B::sigmoid(self.gate.forward(hidden, context)?, context)?;
        attended.multiply(&gate, context)
    }

    /// Runs cache-aware attention and gates each attended head before output.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            hidden,
            explicit_mask,
            cache,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs rank-local gated attention and reduces the output projection.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            hidden,
            explicit_mask,
            cache,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn forward_instrumented<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        explicit_mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(hidden, explicit_mask, cache, context, instrumentation)?;
        let attended = instrumentation.apply("attention.channels", attended)?;
        instrumentation.project::<B>(
            "attention.write_input",
            &mut self.output,
            &attended,
            parallel,
            context,
        )
    }
}

/// Dense SwiGLU branch.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Gate projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Mlp<B> {
    fn from_parameters(
        parameters: eredu_nn::SplitGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            gate: B::linear(parameters.gate, context)?,
            up: B::linear(parameters.up, context)?,
            down: B::linear(parameters.down, context)?,
            policy: parameters.policy,
        })
    }

    fn forward_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(hidden, context)?;
        let up = self.up.forward(hidden, context)?;
        let hidden = B::gated_product(gate, up, self.policy, context)?;
        let hidden = instrumentation.apply("feed_forward.units", hidden)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &hidden,
            parallel,
            context,
        )
    }
}

/// Projection and normalization around the selected expert bank.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum ExpertComposition<B: GroupedNeuralBackend> {
    /// Experts consume and produce decoder-width activations directly.
    Direct,
    /// Experts operate in a separately normalized latent space.
    Compressed {
        /// Decoder-to-latent projection, replicated across expert/TP ranks.
        pre_projection: B::Linear,
        /// Centered latent input norm.
        pre_norm: LearnedRmsNorm<B>,
        /// Scale-before-RMS applied separately to each complete expert output.
        expert_norm: LearnedRmsNorm<B>,
        /// Latent-to-decoder projection after weighted accumulation.
        post_projection: B::Linear,
    },
}

impl<B: GroupedNeuralBackend> ExpertComposition<B> {
    fn new(
        spec: Option<routed::CompressionSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let Some(spec) = spec else {
            return Ok(Self::Direct);
        };
        let [input_norm, output_norm] = spec.norms;
        Ok(Self::Compressed {
            pre_projection: B::linear(spec.input, context)?,
            pre_norm: LearnedRmsNorm::new(input_norm, context)?,
            expert_norm: LearnedRmsNorm::new(output_norm, context)?,
            post_projection: B::linear(spec.output, context)?,
        })
    }

    fn prepare_input(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Direct => Ok(input.clone()),
            Self::Compressed {
                pre_projection,
                pre_norm,
                ..
            } => {
                let latent = pre_projection.forward(input, context)?;
                let latent = instrumentation.apply("feed_forward.compressed_input", latent)?;
                let latent = pre_norm.forward(&latent, context)?;
                instrumentation.apply("feed_forward.expert_input", latent)
            }
        }
    }

    fn finish_output(
        &mut self,
        output: B::Tensor,
        routes: &eredu_nn::GroupSelection<B::Tensor>,
        bank: &B::GatedProductGroups,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        match self {
            Self::Direct => Ok(output),
            Self::Compressed {
                expert_norm,
                post_projection,
                ..
            } => {
                let output = instrumentation.apply("feed_forward.expert_outputs", output)?;
                let normalized = expert_norm.forward(&output, context)?;
                let normalized =
                    instrumentation.apply("feed_forward.normalized_experts", normalized)?;
                let combined = bank.combine_outputs(&normalized, routes, context)?;
                let combined = instrumentation.apply("feed_forward.expert_mixture", combined)?;
                // Expert outputs have already been reconstructed across TP.
                // This replicated latent projection is evaluated exactly once.
                post_projection.forward(&combined, context)
            }
        }
    }
}

/// Top-k routed gated-product branch with typed expert composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SparseMoe<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Learned router with the normalized scoring and precision policy.
    pub router: B::Selector,
    /// Packed routed expert bank.
    pub experts: B::GatedProductGroups,
    /// Decoder-width or compressed expert composition.
    pub composition: ExpertComposition<B>,
    #[parameter(skip)]
    hidden_size: i32,
    #[parameter(skip)]
    expert_count: i32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SparseMoe<B> {
    fn from_declaration(
        declaration: routed::Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        declaration.validate()?;
        let (selector, bank, compression) = declaration.into_parts();
        Ok(Self {
            hidden_size: selector.input_dimensions(),
            expert_count: selector.selection().group_count(),
            router: B::top_k_group_selector(selector, context)?,
            composition: ExpertComposition::new(compression, context)?,
            experts: B::grouped_gated_product(bank, context)?,
        })
    }

    fn forward_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if hidden.shape().len() != 3 || hidden.dim(2) != self.hidden_size {
            return Err(Error::backend("invalid Muse-Glimmer MoE hidden geometry"));
        }
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, self.hidden_size], context)?;
        let routes = instrumentation.select_groups::<B, P>(
            "routing",
            &mut self.router,
            &flat,
            provider,
            eredu_runtime::RoutedBankId::new(0),
            context,
        )?;
        let expert_input = self
            .composition
            .prepare_input(&flat, context, instrumentation)?;
        let output = instrumentation.routed(
            "routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer,
                input: &expert_input,
                routes: &routes,
                pass,
            },
            |request| match &self.composition {
                ExpertComposition::Direct => provider
                    .forward_grouped(&mut self.experts, request, context)
                    .map_err(Error::backend_source),
                ExpertComposition::Compressed { .. } => {
                    provider.forward_grouped_uncombined(&mut self.experts, request, context)
                }
            },
        )?;
        let output = self
            .composition
            .finish_output(output, &routes, &self.experts, context, instrumentation)?
            .reshape(&shape, context)?;
        instrumentation.routing_output("routing", &routes, output, self.expert_count)
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if hidden.shape().len() != 3 || hidden.dim(2) != self.hidden_size {
            return Err(Error::backend("invalid Muse-Glimmer MoE hidden geometry"));
        }
        let shape = hidden.shape().to_vec();
        let flat = hidden.reshape(&[-1, self.hidden_size], context)?;
        let routes = instrumentation.select_groups::<B, P>(
            "routing",
            &mut self.router,
            &flat,
            provider,
            eredu_runtime::RoutedBankId::new(0),
            context,
        )?;
        let expert_input = self
            .composition
            .prepare_input(&flat, context, instrumentation)?;
        let output = instrumentation.routed(
            "routing",
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer,
                input: &expert_input,
                routes: &routes,
                pass,
            },
            |request| match &self.composition {
                ExpertComposition::Direct => provider
                    .forward_grouped_tensor_parallel(
                        &mut self.experts,
                        request,
                        B::parallel_size(parallel),
                        context,
                    )
                    .map_err(Error::backend_source),
                ExpertComposition::Compressed { .. } => provider
                    .forward_grouped_uncombined_tensor_parallel(
                        &mut self.experts,
                        request,
                        B::parallel_size(parallel),
                        context,
                    ),
            },
        )?;
        let output =
            eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(output, parallel, context)?;
        let output = self
            .composition
            .finish_output(output, &routes, &self.experts, context, instrumentation)?
            .reshape(&shape, context)?;
        instrumentation.routing_output("routing", &routes, output, self.expert_count)
    }
}

/// Dense or routed feed-forward branch selected by normalized checkpoint geometry.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU branch.
    Dense(Mlp<B>),
    /// Softmax top-k routed gated-product branch.
    Sparse(SparseMoe<B>),
}

enum PreparedFeedForward {
    Dense(eredu_nn::SplitGatedProductSpec),
    Routed(routed::Declaration),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    fn from_prepared(
        declaration: PreparedFeedForward,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        match declaration {
            PreparedFeedForward::Routed(declaration) => Ok(Self::Sparse(
                SparseMoe::from_declaration(declaration, context)?,
            )),
            PreparedFeedForward::Dense(parameters) => {
                Ok(Self::Dense(Mlp::from_parameters(parameters, context)?))
            }
        }
    }

    fn forward_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(hidden, None, context, instrumentation)
            }
            Self::Sparse(sparse) => sparse.forward_with_provider(
                hidden,
                layer,
                pass,
                provider,
                context,
                instrumentation,
            ),
        }
    }

    fn forward_parallel_with_provider<P>(
        &mut self,
        hidden: &B::Tensor,
        layer: usize,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(dense) => {
                dense.forward_instrumented(hidden, Some(parallel), context, instrumentation)
            }
            Self::Sparse(sparse) => sparse.forward_parallel_with_provider(
                hidden,
                layer,
                pass,
                provider,
                parallel,
                context,
                instrumentation,
            ),
        }
    }
}

/// One post-normalized decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    /// Gated self-attention.
    pub attention: Attention<B>,
    /// Dense SwiGLU branch.
    pub feed_forward: FeedForward<B>,
    /// Pre-attention norm.
    pub input_norm: LearnedRmsNorm<B>,
    /// Attention-delta norm before residual addition.
    pub post_attention_norm: BranchNorm<B>,
    /// Pre-feed-forward norm.
    pub pre_feed_forward_norm: LearnedRmsNorm<B>,
    /// Feed-forward-delta norm before residual addition.
    pub post_feed_forward_norm: LearnedRmsNorm<B>,
    /// Attention branch residual arithmetic and optional learned gate.
    pub attention_residual: ResidualCombination<B>,
    /// Feed-forward branch residual arithmetic and optional learned gate.
    pub feed_forward_residual: ResidualCombination<B>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TransformerBlock<B> {
    /// Constructs the released dense assistant topology over the same block
    /// equation driver. Its input norms return compute precision while the
    /// caller supplies and retains an FP32 residual stream.
    pub(crate) fn from_dspark_parameters(
        args: &DecoderPolicy,
        parameters: specs::DsparkBlockParameters,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let specs::DsparkBlockParameters {
            attention,
            dense,
            scalar,
        } = parameters;
        let [attention_residual, feed_forward_residual] = scalar.residuals;
        Ok(Self {
            layer,
            attention: Attention::from_spec(
                specs::AttentionSpec::with_parameters(args, layer, attention)?,
                context,
            )?,
            feed_forward: FeedForward::Dense(Mlp::from_parameters(dense, context)?),
            input_norm: LearnedRmsNorm::new(scalar.input, context)?,
            post_attention_norm: BranchNorm::new(scalar.post_attention, context)?,
            pre_feed_forward_norm: LearnedRmsNorm::new(scalar.pre_feed_forward, context)?,
            post_feed_forward_norm: LearnedRmsNorm::new(scalar.post_feed_forward, context)?,
            attention_residual: ResidualCombination::new(attention_residual, context)?,
            feed_forward_residual: ResidualCombination::new(feed_forward_residual, context)?,
        })
    }

    /// Builds one unloaded block.
    pub fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_routed_spec(args, layer, None, context)
    }

    pub(crate) fn new_with_routed_spec(
        args: &DecoderConfig,
        layer: usize,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let declaration = if args.is_moe() {
            let source = args.selected_routed(layer)?;
            PreparedFeedForward::Routed(match routed_spec {
                Some(bank) => source.with_bank(bank)?,
                None => source,
            })
        } else {
            if routed_spec.is_some() {
                return Err(Error::backend("dense Muse invocation has no routed bank"));
            }
            PreparedFeedForward::Dense(args.selected_dense(layer)?)
        };
        let attention = args.selected_attention(layer)?;
        let scalar = args.scalar(layer)?.clone();
        let [attention_residual, feed_forward_residual] = scalar.residuals;
        Ok(Self {
            layer,
            attention: Attention::from_spec(attention, context)?,
            feed_forward: FeedForward::from_prepared(declaration, context)?,
            input_norm: LearnedRmsNorm::new(scalar.input, context)?,
            post_attention_norm: BranchNorm::new(scalar.post_attention, context)?,
            pre_feed_forward_norm: LearnedRmsNorm::new(scalar.pre_feed_forward, context)?,
            post_feed_forward_norm: LearnedRmsNorm::new(scalar.post_feed_forward, context)?,
            attention_residual: ResidualCombination::new(attention_residual, context)?,
            feed_forward_residual: ResidualCombination::new(feed_forward_residual, context)?,
        })
    }

    /// Executes exact pre/post normalization and residual order.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_observed(
            hidden,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the same block with component observations and interventions.
    pub fn forward_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.forward_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            instrumentation,
        )
    }

    /// Executes the canonical block while runtime policy owns routed experts.
    pub fn forward_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes the exact routed request and consumed sublayer values.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_observed<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            hidden,
            mask,
            cache,
            None,
            context,
            instrumentation,
            |feed_forward, hidden, layer, instrumentation| {
                feed_forward.forward_with_provider(
                    hidden,
                    layer,
                    pass,
                    provider,
                    context,
                    instrumentation,
                )
            },
        )
    }

    /// Executes the canonical block with rank-local projections and resident experts.
    pub fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        self.forward_parallel_observed(
            hidden,
            mask,
            cache,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Executes the same TP block with local component evidence.
    pub fn forward_parallel_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.forward_parallel_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
            instrumentation,
        )
    }

    /// Runs a dense rank-local block without requiring routed-expert collectives.
    /// Projection sums precede branch normalization and residual combination.
    pub(crate) fn forward_dense_parallel_observed<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.forward_inner(
            hidden,
            mask,
            cache,
            Some(parallel),
            context,
            instrumentation,
            |feed_forward, hidden, _, observed| match feed_forward {
                FeedForward::Dense(mlp) => {
                    mlp.forward_instrumented(hidden, Some(parallel), context, observed)
                }
                FeedForward::Sparse(_) => Err(Error::backend(
                    "dense tensor-parallel invocation requires a dense feed-forward block",
                )),
            },
        )
    }

    /// Executes the canonical TP block while runtime owns routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_parallel_with_provider_observed(
            hidden,
            mask,
            cache,
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes and intervenes in the same TP/provider block driver.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider_observed<C, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_inner(
            hidden,
            mask,
            cache,
            Some(parallel),
            context,
            instrumentation,
            |feed_forward, hidden, layer, instrumentation| {
                feed_forward.forward_parallel_with_provider(
                    hidden,
                    layer,
                    pass,
                    provider,
                    parallel,
                    context,
                    instrumentation,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner<C, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            usize,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let observed_input = if instrumentation.enabled() {
            Some(instrumentation.apply("input", hidden.clone())?)
        } else {
            None
        };
        let hidden = observed_input.as_ref().unwrap_or(hidden);
        let normalized = self.input_norm.forward(hidden, context)?;
        let normalized = instrumentation.apply("attention.input", normalized)?;
        let attention = self.attention.forward_instrumented(
            &normalized,
            mask,
            cache,
            parallel,
            context,
            instrumentation,
        )?;
        let attention = instrumentation.apply("attention.write", attention)?;
        let attention = self.post_attention_norm.forward(&attention, context)?;
        let attention = instrumentation.apply("attention.output", attention)?;
        let hidden = self
            .attention_residual
            .forward(hidden, &attention, context)?;
        let hidden = instrumentation.apply("attention.residual", hidden)?;
        let normalized = self.pre_feed_forward_norm.forward(&hidden, context)?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
        let feed_forward = feed_forward(
            &mut self.feed_forward,
            &normalized,
            self.layer,
            instrumentation,
        )?;
        let feed_forward = instrumentation.apply("feed_forward.write", feed_forward)?;
        let feed_forward = self
            .post_feed_forward_norm
            .forward(&feed_forward, context)?;
        let feed_forward = instrumentation.apply("feed_forward.output", feed_forward)?;
        let hidden = self
            .feed_forward_residual
            .forward(&hidden, &feed_forward, context)?;
        let hidden = instrumentation.apply("feed_forward.residual", hidden)?;
        instrumentation.apply("output", hidden)
    }
}

/// Pinned token embedding, final norm, and tied/untied vocabulary head.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Canonical pinned owners actually retained by this component.
    pub resources: crate::decoder::PartitionStaticModules<B>,
    #[parameter(skip)]
    tied_head: bool,
    #[parameter(skip)]
    embedding_epsilon: f32,
    #[parameter(skip)]
    vocabulary_arithmetic: VocabularyArithmetic,
    #[parameter(skip)]
    output_multiplier: f32,
    #[parameter(skip)]
    logit_cap: f32,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> StaticModules<B> {
    /// Builds all canonical pinned resources for ordinary execution.
    pub fn new(
        args: &DecoderConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::materialize(args, None, |_| true, context)
    }

    /// Builds all canonical pinned resources with planner-derived vocabulary rows.
    pub fn new_parallel(
        args: &DecoderConfig,
        geometry: &LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        geometry.validate_for(args).map_err(Error::backend_source)?;
        Self::materialize(args, Some(geometry), |_| true, context)
    }

    /// Materializes complete canonical text resources from the retained component and bindings.
    pub fn from_parameter_bindings(
        args: &DecoderConfig,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: Option<&LocalGeometry>,
        bindings: &[eredu_runtime::OwnedParameterGroupSpec],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = super::model::select_definition(args, definition, geometry)?;
        let roles =
            definition.select_complete_pinned_roles(bindings, &["embedding", "norm", "output"])?;
        Self::materialize(args, geometry, |role| roles.contains(role), context)
    }

    fn materialize(
        args: &DecoderConfig,
        geometry: Option<&LocalGeometry>,
        stores: impl Fn(&str) -> bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let resources = crate::decoder::PartitionStaticModules::<B>::from_selected_construction(
            args.pinned().clone(),
            geometry.map(|geometry| {
                (
                    geometry.embedding_range().clone(),
                    geometry.output_range().cloned(),
                )
            }),
            stores,
            context,
        )?;
        Ok(Self {
            resources,
            tied_head: args.tie_word_embeddings,
            embedding_epsilon: args.rms_norm_eps,
            vocabulary_arithmetic: args.vocabulary_arithmetic(),
            output_multiplier: args.output_multiplier,
            logit_cap: args.final_logit_softcapping,
        })
    }

    /// Borrows the actually retained input or tied-output table.
    pub fn embedding(&self) -> Result<&B::Embedding, Error> {
        self.resources
            .embeddings
            .as_ref()
            .ok_or_else(|| Error::backend("Muse text does not own the embedding table"))
    }

    /// Mutably borrows that same canonical table.
    pub fn embedding_mut(&mut self) -> Result<&mut B::Embedding, Error> {
        self.resources
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("Muse text does not own the embedding table"))
    }

    fn require_output(&self) -> Result<(), Error> {
        if (self.tied_head && self.resources.embeddings.is_some())
            || (!self.tied_head && self.resources.lm_head.is_some())
        {
            Ok(())
        } else {
            Err(Error::backend(
                "Muse text does not own its vocabulary projection",
            ))
        }
    }

    /// Embeds IDs and applies the released weightless normalization.
    pub fn embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.raw_embeddings(tokens, None, context)?;
        self.normalize_embeddings(&hidden, context)
    }

    /// Looks up the raw token table, reducing rank-local vocabulary contributions first.
    pub fn raw_embeddings(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                self.embedding_mut()?,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => self.embedding_mut()?.forward(tokens, context),
        }
    }

    /// Projects already-normalized assistant states through the raw target head.
    /// Vocabulary shards are gathered before candidate selection or output transforms.
    pub fn raw_vocabulary_projection(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.project_vocabulary_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Applies released weightless input normalization to gathered TP embeddings.
    pub fn normalize_embeddings(
        &self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self.vocabulary_arithmetic {
            VocabularyArithmetic::Backend => {
                B::rms_norm_without_weight(hidden, self.embedding_epsilon, context)
            }
            VocabularyArithmetic::StageRounded => {
                weightless_rms_norm::<B>(hidden, self.embedding_epsilon, context)
            }
            VocabularyArithmetic::Float32 => B::rms_norm_without_weight(
                &hidden.cast_float(eredu_nn::TensorElementType::F32, context)?,
                self.embedding_epsilon,
                context,
            ),
        }
    }

    /// Applies the replicated final norm before the vocabulary projection.
    pub fn final_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.resources.decoder_resources_mut().normalize(
            hidden,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Applies output scaling and softcapping after vocabulary shards are gathered.
    pub fn finish_logits(
        &self,
        logits: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match self.vocabulary_arithmetic {
            VocabularyArithmetic::Backend => logits
                .multiply_scalar(self.output_multiplier / self.logit_cap, context)?
                .tanh(context)?
                .multiply_scalar(self.logit_cap, context),
            VocabularyArithmetic::Float32 => logits
                .cast_float(eredu_nn::TensorElementType::F32, context)?
                .multiply_scalar(self.output_multiplier, context)?
                .multiply_scalar(self.logit_cap.recip(), context)?
                .tanh(context)?
                .multiply_scalar(self.logit_cap, context),
            VocabularyArithmetic::StageRounded => {
                let dtype = logits
                    .element_type()
                    .ok_or_else(|| Error::backend("Muse logit precision is unknown"))?;
                let scaled = logits
                    .cast_float(eredu_nn::TensorElementType::F32, context)?
                    .multiply_scalar(self.output_multiplier, context)?
                    .cast_float(dtype, context)?;
                let capped = scaled
                    .cast_float(eredu_nn::TensorElementType::F32, context)?
                    .divide(&B::Tensor::full_f32(self.logit_cap, &[], context)?, context)?
                    .cast_float(dtype, context)?;
                capped
                    .cast_float(eredu_nn::TensorElementType::F32, context)?
                    .tanh(context)?
                    .cast_float(dtype, context)?
                    .cast_float(eredu_nn::TensorElementType::F32, context)?
                    .multiply_scalar(self.logit_cap, context)?
                    .cast_float(dtype, context)
            }
        }
    }

    /// Applies final norm, head, multiplier, then tanh softcap.
    pub fn logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.logits_instrumented(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Applies the same readout with evidence from the actual projection input.
    pub(crate) fn logits_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.require_output()?;
        let logits = self.resources.decoder_resources_mut().finish_parallel(
            hidden,
            parallel,
            context,
            instrumentation,
        )?;
        self.finish_logits(logits, context)
    }

    fn project_vocabulary_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.require_output()?;
        self.resources
            .decoder_resources_mut()
            .project(hidden, parallel, context, instrumentation)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> DecoderStaticResources<B>
    for StaticModules<B>
{
    fn decoder_resources(&self) -> crate::decoder::DecoderResources<'_, B> {
        self.resources.decoder_resources()
    }
    fn decoder_resources_mut(&mut self) -> crate::decoder::DecoderResourcesMut<'_, B> {
        self.resources.decoder_resources_mut()
    }
}
