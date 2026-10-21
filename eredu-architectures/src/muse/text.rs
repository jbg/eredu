//! Backend-neutral Muse-Glimmer decoder equations.

use eredu_nn::{
    AttentionCache, AttentionRequest, EmbeddingOperator, EmbeddingSpec, Error,
    GatedProductGroupLayout, GroupScoring, GroupedGatedProductOperator, GroupedGatedProductSpec,
    GroupedNeuralBackend, LinearOperator, LinearSpec, NormalizationConstructionSpec,
    NormalizationOperator, ParameterSpec, Parameterized, RotaryOperator, RotaryPosition,
    RotarySpec, Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ParameterProvider, RoutedExpertRequest, TensorParallelParameterProvider,
};

use crate::decoder::{ComponentInstrumentation, DecoderStaticResources};
use crate::linear_format::standard_expert_projection;

use super::config::{
    AttentionEquation, BlockNormalization, ResidualEquation, RoutedFeedForwardEquation,
    VocabularyArithmetic,
};
use super::equations::{
    weightless_rms_norm, BranchNorm, LearnedResidualCombination, LearnedRmsNorm,
    ResidualCombination, RmsNormEquation,
};
use super::{DecoderConfig, LocalGeometry, WeightConvention};

mod mask;

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
        Self::new_with_layout(
            args,
            layer,
            &format!("model.layers.{layer}.self_attn"),
            "gate_proj",
            context,
        )
    }

    pub(crate) fn new_with_layout(
        args: &DecoderConfig,
        layer: usize,
        prefix: &str,
        gate_projection: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = args
            .attention_schedule
            .get(layer)
            .copied()
            .ok_or_else(|| Error::backend(format!("missing Muse-Glimmer layer {layer}")))?;
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )
        };
        let gguf = args.weight_convention == WeightConvention::Gguf;
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.head_dim,
                    args.rms_norm_eps,
                    ParameterSpec::trainable(format!("{prefix}.{field}.weight"))
                        .map_err(Error::backend)?,
                ),
                context,
            )
        };
        Ok(Self {
            query_heads: args.num_attention_heads,
            key_value_heads: args.num_key_value_heads,
            head_dimensions: args.head_dim,
            scale: match args.attention_equation {
                super::config::AttentionEquation::QueryScaled => {
                    (args.head_dim as f32).sqrt().recip()
                }
                super::config::AttentionEquation::NormalizedHeads => {
                    (args.head_dim as f64).powf(-0.5) as f32
                }
                super::config::AttentionEquation::NormalizedHeadsScoreScaled => {
                    ((args.head_dim as f64).powf(-0.5) * args.qk_scale_factor as f64) as f32
                }
            },
            window: policy.window().map(|window| window.get() as i32),
            uses_rope: *args
                .layer_uses_rope
                .get(layer)
                .ok_or_else(|| Error::backend("missing Muse-Glimmer RoPE policy"))?,
            qk_norm_epsilon: args.rms_norm_eps,
            equation: args.attention_equation,
            arithmetic: args.attention_arithmetic(),
            qk_scale_factor: args.qk_scale_factor,
            query_scale: (!gguf)
                .then(|| B::Tensor::full_f32(args.qk_scale_factor, &[args.head_dim], context))
                .transpose()?,
            query: linear(
                "q_proj",
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
            )?,
            key: linear(
                "k_proj",
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
            )?,
            value: linear(
                "v_proj",
                args.hidden_size,
                args.num_key_value_heads * args.head_dim,
            )?,
            gate: linear(
                gate_projection,
                args.hidden_size,
                args.num_attention_heads * args.head_dim,
            )?,
            output: linear(
                "o_proj",
                args.num_attention_heads * args.head_dim,
                args.hidden_size,
            )?,
            query_norm: gguf.then(|| norm("q_norm")).transpose()?,
            key_norm: gguf.then(|| norm("k_norm")).transpose()?,
            rotary: B::rotary(
                RotarySpec {
                    arithmetic: args.rotary_arithmetic(),
                    dimensions: args.head_dim,
                    base: args.rope_theta,
                    traditional: args.weight_convention.uses_traditional_rope(),
                    algorithm: crate::rotary::normalize_algorithm(args.rope_scaling.as_ref())
                        .expect("validated Muse-Glimmer RoPE algorithm"),
                },
                context,
            )?,
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
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Mlp<B> {
    fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_prefix(args, &format!("model.layers.{layer}.mlp"), context)
    }

    pub(crate) fn new_with_prefix(
        args: &DecoderConfig,
        prefix: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )
        };
        Ok(Self {
            gate: linear("gate_proj", args.hidden_size, args.intermediate_size)?,
            up: linear("up_proj", args.hidden_size, args.intermediate_size)?,
            down: linear("down_proj", args.intermediate_size, args.hidden_size)?,
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
        let hidden = B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
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
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } =
            args.routed_feed_forward
        else {
            return Ok(Self::Direct);
        };
        let prefix = format!("model.layers.{layer}.mlp");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        args.linear_format_for(&weight),
                    )?,
                },
                context,
            )
        };
        Ok(Self::Compressed {
            pre_projection: linear("pre_expert_proj", args.hidden_size, expert_width)?,
            pre_norm: LearnedRmsNorm::new(
                expert_width,
                args.rms_norm_eps,
                RmsNormEquation::OutputScaledFp32(args.weight_convention),
                format!("{prefix}.pre_expert_norm.weight"),
                context,
            )?,
            expert_norm: LearnedRmsNorm::new(
                expert_width,
                args.post_norm_eps,
                RmsNormEquation::InputScaled(args.weight_convention),
                format!("{prefix}.experts.post_expert_norm.weight"),
                context,
            )?,
            post_projection: linear("post_expert_proj", expert_width, args.hidden_size)?,
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
    fn new(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_spec(args, layer, expert_bank_spec(args, layer)?, context)
    }

    pub(crate) fn new_with_spec(
        args: &DecoderConfig,
        layer: usize,
        spec: GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.mlp");
        let selection = routing_selection_spec(args)?;
        let weight = format!("{prefix}.gate.weight");
        let mut selector = TopKGroupSelectorSpec::new(
            args.hidden_size,
            ParameterSpec::trainable(&weight).map_err(Error::backend)?,
            crate::linear_format::standard_linear_format(&weight, args.linear_format_for(&weight))?,
            selection,
        )?;
        if let RoutedFeedForwardEquation::CompressedSigmoid { logit_scale, .. } =
            args.routed_feed_forward
        {
            let mut correction =
                ParameterSpec::trainable(format!("{prefix}.gate.e_score_correction_bias"))
                    .map_err(Error::backend)?;
            correction.trainable = false;
            selector = selector
                .with_logit_scale(logit_scale)?
                .with_correction_bias(correction)?;
        }
        Ok(Self {
            router: B::top_k_group_selector(selector, context)?,
            composition: ExpertComposition::new(args, layer, context)?,
            experts: B::grouped_gated_product(spec, context)?,
            hidden_size: args.hidden_size,
            expert_count: args.num_experts,
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

/// Complete neutral router policy shared by construction and cold intervention admission.
pub(crate) fn routing_selection_spec(
    args: &DecoderConfig,
) -> Result<TopKGroupSelectionSpec, Error> {
    Ok(match args.routed_feed_forward {
        RoutedFeedForwardEquation::DirectSoftmax => TopKGroupSelectionSpec::new(
            args.num_experts,
            args.num_experts_per_tok,
            GroupScoring::Softmax,
            args.norm_topk_prob,
        )?,
        RoutedFeedForwardEquation::CompressedSigmoid {
            coefficient_scale, ..
        } => TopKGroupSelectionSpec::new(
            args.num_experts,
            args.num_experts_per_tok,
            GroupScoring::Sigmoid,
            true,
        )?
        .with_weight_policy(1e-15, coefficient_scale)?
        .with_order(if args.weight_convention == WeightConvention::HuggingFace {
            eredu_nn::GroupSelectionOrder::Partition
        } else {
            eredu_nn::GroupSelectionOrder::Ranked
        }),
    })
}

/// Returns the architecture-owned routed expert specification for one layer.
pub fn expert_bank_spec(
    args: &DecoderConfig,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    let prefix = format!("model.layers.{layer}.mlp.experts");
    let gate_up = format!("{prefix}.gate_up_proj");
    let down = format!("{prefix}.down_proj");
    let width = match args.routed_feed_forward {
        RoutedFeedForwardEquation::DirectSoftmax => args.hidden_size,
        RoutedFeedForwardEquation::CompressedSigmoid { expert_width, .. } => expert_width,
    };
    let spec = GroupedGatedProductSpec::new(
        args.num_experts,
        width,
        args.moe_intermediate_size,
        width,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(&gate_up, None, args.linear_format_for(&gate_up))?,
            down: standard_expert_projection(&down, None, args.linear_format_for(&down))?,
        },
    )?;
    Ok(match args.routed_feed_forward {
        RoutedFeedForwardEquation::DirectSoftmax => spec,
        RoutedFeedForwardEquation::CompressedSigmoid { .. } => {
            spec.with_reduction(eredu_nn::GroupReduction::SequentialGroupOrder)
        }
    })
}

/// Returns the same architecture-owned bank at placement-resolved geometry.
pub(crate) fn localized_expert_bank_spec(
    args: &DecoderConfig,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(args, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
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

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    fn new_with_routed_spec(
        args: &DecoderConfig,
        layer: usize,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if args.is_moe() {
            Ok(Self::Sparse(match routed_spec {
                Some(spec) => SparseMoe::new_with_spec(args, layer, spec, context)?,
                None => SparseMoe::new(args, layer, context)?,
            }))
        } else {
            Ok(Self::Dense(Mlp::new(args, layer, context)?))
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
    pub(crate) fn new_dspark(
        args: &DecoderConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let root = format!("layers.{layer}");
        let norm = |name: &str, epsilon, equation| {
            LearnedRmsNorm::new(
                args.hidden_size,
                epsilon,
                equation,
                format!("{root}.{name}.weight"),
                context,
            )
        };
        let ResidualEquation::Learned {
            temperature,
            minimum_variance,
        } = args.residual_equation
        else {
            return Err(Error::backend(
                "DSpark requires learned FP32 residual gates",
            ));
        };
        let residual = |name: &str| {
            LearnedResidualCombination::new(
                args.hidden_size,
                temperature,
                minimum_variance,
                format!("{root}.{name}.gate"),
                context,
            )
            .map(ResidualCombination::Learned)
        };
        Ok(Self {
            layer,
            attention: Attention::new_with_layout(
                args,
                layer,
                &format!("{root}.self_attn"),
                "o_gate_proj",
                context,
            )?,
            feed_forward: FeedForward::Dense(Mlp::new_with_prefix(
                args,
                &format!("{root}.feed_forward"),
                context,
            )?),
            input_norm: norm(
                "self_attn.pre_qkv_proj_norm",
                args.rms_norm_eps,
                RmsNormEquation::OutputScaledComputeFp32(super::equations::RmsGain::Centered),
            )?,
            post_attention_norm: BranchNorm::Weightless {
                epsilon: args.post_norm_eps,
            },
            pre_feed_forward_norm: norm(
                "pre_ffn_norm",
                args.rms_norm_eps,
                RmsNormEquation::OutputScaledComputeFp32(super::equations::RmsGain::Centered),
            )?,
            post_feed_forward_norm: norm(
                "post_ffn_norm",
                args.post_norm_eps,
                RmsNormEquation::InputScaledFp32(super::equations::RmsGain::Centered),
            )?,
            attention_residual: residual("add_attn")?,
            feed_forward_residual: residual("add_ffn")?,
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
        let root = format!("model.layers.{layer}");
        let norm = |field: &str, epsilon, equation| {
            LearnedRmsNorm::new(
                args.hidden_size,
                epsilon,
                equation,
                format!("{root}.{field}.weight"),
                context,
            )
        };
        let pre_equation = match args.block_normalization {
            BlockNormalization::OutputScaled => {
                RmsNormEquation::OutputScaled(args.weight_convention)
            }
            BlockNormalization::InputScaledFeedForward => {
                RmsNormEquation::OutputScaledFp32(args.weight_convention)
            }
        };
        let (post_attention_norm, post_ff_equation) = match args.block_normalization {
            BlockNormalization::OutputScaled => (
                BranchNorm::Learned(norm(
                    "post_attention_layernorm",
                    args.post_norm_eps,
                    pre_equation,
                )?),
                pre_equation,
            ),
            BlockNormalization::InputScaledFeedForward => (
                BranchNorm::Weightless {
                    epsilon: args.post_norm_eps,
                },
                RmsNormEquation::InputScaled(args.weight_convention),
            ),
        };
        let residual = |field| match args.residual_equation {
            ResidualEquation::Add => Ok(ResidualCombination::Add),
            ResidualEquation::StoredCoefficients => {
                super::equations::StoredResidualCombination::new(
                    args.hidden_size,
                    &format!("{root}.{field}"),
                    context,
                )
                .map(ResidualCombination::Stored)
            }
            ResidualEquation::Learned {
                temperature,
                minimum_variance,
            } => LearnedResidualCombination::new(
                args.hidden_size,
                temperature,
                minimum_variance,
                format!("{root}.{field}.gate"),
                context,
            )
            .map(ResidualCombination::Learned),
        };
        Ok(Self {
            layer,
            attention: Attention::new(args, layer, context)?,
            feed_forward: FeedForward::new_with_routed_spec(args, layer, routed_spec, context)?,
            input_norm: norm("input_layernorm", args.rms_norm_eps, pre_equation)?,
            post_attention_norm,
            pre_feed_forward_norm: norm(
                "pre_feedforward_layernorm",
                args.rms_norm_eps,
                pre_equation,
            )?,
            post_feed_forward_norm: norm(
                "post_feedforward_layernorm",
                args.post_norm_eps,
                post_ff_equation,
            )?,
            attention_residual: residual("post_attention_residual_gate")?,
            feed_forward_residual: residual("post_feedforward_residual_gate")?,
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
    fn final_norm_spec(args: &DecoderConfig) -> Result<NormalizationConstructionSpec, Error> {
        let spec = NormalizationConstructionSpec::learned(
            args.hidden_size,
            args.rms_norm_eps,
            ParameterSpec::trainable("model.norm.weight").map_err(Error::backend)?,
        );
        match args.vocabulary_arithmetic() {
            VocabularyArithmetic::Backend => Ok(spec),
            VocabularyArithmetic::StageRounded | VocabularyArithmetic::Float32 => {
                spec.with_groups(1)
            }
        }
    }
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

    /// Materializes only complete canonical text resources present in the selected bindings.
    pub fn from_parameter_bindings(
        args: &DecoderConfig,
        geometry: Option<&LocalGeometry>,
        bindings: &[eredu_runtime::OwnedParameterGroupSpec],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = super::component_definition(args, geometry)?;
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
        let embedding = "model.embed_tokens.weight";
        let head = "lm_head.weight";
        let embeddings = stores("embedding")
            .then(|| {
                let spec = EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable(embedding).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        embedding,
                        args.linear_format_for(embedding),
                    )?,
                };
                match geometry {
                    Some(geometry) => B::vocabulary_parallel_embedding(
                        spec,
                        geometry.embedding_range().clone(),
                        context,
                    ),
                    None => B::embedding(spec, context),
                }
            })
            .transpose()?;
        let norm = stores("norm")
            .then(|| B::normalization(Self::final_norm_spec(args)?, context))
            .transpose()?;
        let lm_head = (stores("output") && !args.tie_word_embeddings)
            .then(|| {
                let spec = LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: ParameterSpec::trainable(head).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        head,
                        args.linear_format_for(head),
                    )?,
                };
                match geometry {
                    Some(geometry) => B::vocabulary_parallel_linear(
                        spec,
                        geometry
                            .output_range()
                            .cloned()
                            .ok_or_else(|| Error::backend("missing Muse-Glimmer output range"))?,
                        context,
                    ),
                    None => B::linear(spec, context),
                }
            })
            .transpose()?;
        Ok(Self {
            resources: crate::decoder::PartitionStaticModules {
                embeddings,
                norm,
                lm_head,
            },
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
