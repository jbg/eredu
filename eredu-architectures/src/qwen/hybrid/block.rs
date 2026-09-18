//! Shared Qwen3-Next/Qwen3.5 decoder block.

use eredu_nn::{
    AttentionCache, Error, GatedProductGroupLayout, GroupScoring, GroupSelectionOperator,
    GroupedGatedProductSpec, GroupedNeuralBackend, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, NormalizationScale, ParameterSpec,
    Parameterized, RotarySpec, Tensor, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use eredu_runtime::{
    ExpertPass, ResidentExpertProvider, RoutedExpertProvider, RoutedExpertRequest,
    RuntimeStateComponents, TensorParallelRoutedExpertProvider,
};

use crate::{
    decoder::{
        Attention, AttentionInput, ComponentInstrumentation, DecoderProjectionOperator, Mlp,
        TensorParallelProjectionOperator,
    },
    linear_format::standard_expert_projection,
};

use super::{HybridConfig, HybridLayerPolicy, LinearAttention};
pub(crate) mod construction;
pub(crate) use construction::{PredictionBlockSpec, TargetBlockSpec};

/// Scheduled hybrid token mixer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum TokenMixer<B: NeuralBackend> {
    /// Gated-delta recurrent attention.
    Linear(LinearAttention<B>),
    /// Gated grouped-query self attention.
    Attention(Attention<B>),
}

/// Routed experts plus the always-on Qwen shared expert.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct SharedRoutedGatedProduct<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip, metadata)]
    layer: usize,
    #[parameter(skip, metadata)]
    resident_unit_coordinates: Option<(
        std::sync::Arc<eredu_core::component::ComponentCoordinateMap>,
        bool,
    )>,
    /// Learned top-k router.
    pub router: B::Selector,
    /// Packed routed expert bank.
    pub experts: B::GatedProductGroups,
    /// Always-on dense shared expert.
    pub shared_expert: Mlp<B>,
    /// Scalar gate applied to the shared expert output.
    pub shared_expert_gate: B::Linear,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> SharedRoutedGatedProduct<B> {
    fn new(
        config: &HybridConfig,
        layer: usize,
        prefix: &str,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        construction::require_source_compiler::<B>(context)?;
        construction::RoutedSpec::new(config, layer, prefix, routed_spec)?.instantiate::<B>(context)
    }

    /// Retains global scalar coordinates for a prepared resident prediction bank.
    pub(crate) fn bind_resident_unit_coordinates(
        &mut self,
        coordinates: eredu_core::component::ComponentCoordinateMap,
        partitioned: bool,
    ) {
        self.bind_shared_resident_unit_coordinates(std::sync::Arc::new(coordinates), partitioned);
    }

    pub(crate) fn bind_shared_resident_unit_coordinates(
        &mut self,
        coordinates: std::sync::Arc<eredu_core::component::ComponentCoordinateMap>,
        partitioned: bool,
    ) {
        self.resident_unit_coordinates = Some((coordinates, partitioned));
    }

    fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: eredu_runtime::RoutedBankId::new(0),
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass: pass(input),
                },
                context,
            )
            .map_err(Error::backend_retained_source)?;
        let shared =
            self.forward_shared(input, context, &mut ComponentInstrumentation::disabled())?;
        routed.add(&shared, context)
    }

    /// Executes shared/routed feed-forward work with pre-dispatch routing controls
    /// and separately attributable shared contributions.
    pub fn forward_observed_with_provider<P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoints,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let point = point
            .bank(eredu_runtime::RoutedBankId::new(0))
            .ok_or_else(|| Error::backend("missing feed-forward routing observation"))?;
        let routes = eredu_runtime::select_routes_with_observer(
            &mut self.router,
            input,
            context,
            point.path(),
            observer,
        )?;
        let routed = eredu_runtime::with_routed_unit_observer(
            observer,
            point.path(),
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: self.layer,
                input,
                routes: &routes,
                pass: pass(input),
            },
            |request| {
                eredu_runtime::with_borrowed_resident_unit_coordinates(
                    self.resident_unit_coordinates
                        .as_ref()
                        .map(|(coordinates, partitioned)| (coordinates.as_ref(), *partitioned)),
                    request,
                    |request| provider.forward_grouped(&mut self.experts, request, context),
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let shared = {
            let path = format!("{}.shared_expert", point.path());
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.forward_shared(
                input,
                context,
                &mut ComponentInstrumentation::new(&path, &mut borrowed),
            )?
        };
        let combined = routed.add(&shared, context)?;
        observer.observe_routing(eredu_runtime::RoutingObservation {
            path: point.path(),
            selected_experts: routes.group_indices(),
            selected_scores: routes.selected_scores(),
            coefficients: routes.coefficients(),
            routed_output: &routed,
            local_routed_output: None,
            reduced_routed_output: None,
            shared_output: Some(&shared),
            combined_output: Some(&combined),
            expert_count: point.expert_count(),
        })?;
        eredu_runtime::observe_and_intervene(
            observer,
            &format!("{}.output", point.path()),
            &combined,
        )
    }

    fn forward_tensor_parallel_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let routes = self.router.select(input, context)?;
        let routed = provider
            .forward_grouped_tensor_parallel(
                &mut self.experts,
                RoutedExpertRequest {
                    unit_observer: None,
                    bank: eredu_runtime::RoutedBankId::new(0),
                    layer: self.layer,
                    input,
                    routes: &routes,
                    pass: pass(input),
                },
                B::parallel_size(parallel),
                context,
            )
            .map_err(Error::backend_retained_source)?;
        let shared =
            self.forward_shared(input, context, &mut ComponentInstrumentation::disabled())?;
        Self::combine_tensor_parallel(routed, shared, parallel, context)
    }

    fn combine_tensor_parallel(
        routed: eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>,
        shared: B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::RoutedExpertTensorParallelOutput<B::Tensor>, Error> {
        match routed {
            eredu_runtime::RoutedExpertTensorParallelOutput::Complete(routed) => {
                let shared = B::sum_parallel(shared, parallel, context)?;
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Complete(
                    routed.add(&shared, context)?,
                ))
            }
            eredu_runtime::RoutedExpertTensorParallelOutput::Partial(routed) => {
                let (reducible, post_reduce) = routed.into_parts();
                Ok(eredu_runtime::RoutedExpertTensorParallelOutput::Partial(
                    eredu_nn::TensorParallelGroupedOutput::new(
                        reducible.add(&shared, context)?,
                        post_reduce,
                    ),
                ))
            }
        }
    }

    fn forward_tensor_parallel_observed_with_provider<P, O>(
        &mut self,
        points: eredu_runtime::RoutedObservationPoints,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let point = points
            .bank(eredu_runtime::RoutedBankId::new(0))
            .ok_or_else(|| Error::backend("missing feed-forward routing observation"))?;
        let routes = eredu_runtime::select_routes_with_observer(
            &mut self.router,
            input,
            context,
            point.path(),
            observer,
        )?;
        let routed = eredu_runtime::with_routed_unit_observer(
            observer,
            point.path(),
            RoutedExpertRequest {
                unit_observer: None,
                bank: eredu_runtime::RoutedBankId::new(0),
                layer: self.layer,
                input,
                routes: &routes,
                pass: pass(input),
            },
            |request| {
                eredu_runtime::with_borrowed_resident_unit_coordinates(
                    self.resident_unit_coordinates
                        .as_ref()
                        .map(|(coordinates, partitioned)| (coordinates.as_ref(), *partitioned)),
                    request,
                    |request| {
                        provider.forward_grouped_tensor_parallel(
                            &mut self.experts,
                            request,
                            B::parallel_size(parallel),
                            context,
                        )
                    },
                )
            },
        )
        .map_err(eredu_runtime::ObservedExpertProviderError::into_neural_error)?;
        let shared = {
            let path = format!("{}.shared_expert", point.path());
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            self.forward_shared(
                input,
                context,
                &mut ComponentInstrumentation::new(&path, &mut borrowed),
            )?
        };
        let combined = eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
            Self::combine_tensor_parallel(routed, shared, parallel, context)?,
            parallel,
            context,
        )?;
        eredu_runtime::observe_and_intervene(
            observer,
            &format!("{}.output", point.path()),
            &combined,
        )
    }

    fn forward_shared(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let shared =
            self.shared_expert
                .forward_feed_forward_observed(input, context, instrumentation)?;
        let shared = instrumentation.apply("feed_forward.write", shared)?;
        instrumentation.observe("gate.input", input)?;
        let gate = instrumentation.project::<B>(
            "gate.projection_input",
            &mut self.shared_expert_gate,
            input,
            None,
            context,
        )?;
        let gate = instrumentation.apply("gate", B::sigmoid(gate, context)?)?;
        instrumentation.apply("feed_forward.output", shared.multiply(&gate, context)?)
    }
}

/// Returns the architecture-owned routed expert specification for a target or MTP layer.
pub fn expert_bank_spec(
    config: &HybridConfig,
    layer: usize,
) -> Result<GroupedGatedProductSpec, Error> {
    let target = config.num_hidden_layers as usize;
    let root = if layer < target {
        format!("model.layers.{layer}.mlp.experts")
    } else {
        format!("mtp.layers.{}.mlp.experts", layer - target)
    };
    expert_bank_spec_at(config, &root)
}

fn expert_bank_spec_at(
    config: &HybridConfig,
    expert_prefix: &str,
) -> Result<GroupedGatedProductSpec, Error> {
    let gate_up_name = format!("{expert_prefix}.gate_up_proj");
    let down_name = format!("{expert_prefix}.down_proj");
    GroupedGatedProductSpec::new(
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
        config.hidden_size,
        eredu_nn::GatedProductPolicy::ordinary_silu(),
        GatedProductGroupLayout::Packed {
            gate_up: standard_expert_projection(
                &gate_up_name,
                None,
                config.linear_format(&gate_up_name),
            )?,
            down: standard_expert_projection(&down_name, None, config.linear_format(&down_name))?,
        },
    )
}

/// Returns the canonical expert bank at rank-local cardinality and width.
pub(crate) fn localized_expert_bank_spec(
    config: &HybridConfig,
    layer: usize,
    expert_count: i32,
    intermediate_dimensions: i32,
) -> Result<GroupedGatedProductSpec, Error> {
    expert_bank_spec(config, layer)?.with_group_geometry(expert_count, intermediate_dimensions)
}

/// Dense or routed/shared-expert feed-forward policy selected by configuration.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense SwiGLU.
    Dense(Mlp<B>),
    /// Routed SwiGLU plus an always-on shared expert.
    Routed(SharedRoutedGatedProduct<B>),
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> FeedForward<B> {
    /// Executes through a runtime-owned expert provider.
    pub fn forward_with_provider<P>(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => moe.forward_with_provider(input, context, provider),
        }
    }

    fn forward_observed_with_provider<P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoints,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match self {
            Self::Dense(mlp) => mlp.forward_feed_forward(input, context),
            Self::Routed(moe) => {
                moe.forward_observed_with_provider(point, input, context, provider, observer)
            }
        }
    }
}

/// One pre-normalized hybrid decoder block for every dense/MoE family form.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Block<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Recurrent or full-attention policy.
    pub mixer: TokenMixer<B>,
    /// Dense or routed/shared-expert policy.
    pub feed_forward: FeedForward<B>,
    /// Learned-offset token-mixer normalization.
    pub input_norm: B::Normalization,
    /// Learned-offset feed-forward normalization.
    pub post_attention_norm: B::Normalization,
}

/// Dense-only Qwen hybrid unit used by replicated text composition.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(crate) struct ReplicatedBlock<B: NeuralBackend> {
    mixer: TokenMixer<B>,
    feed_forward: Mlp<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

impl<B: NeuralBackend> ReplicatedBlock<B> {
    pub(crate) fn new(
        config: &HybridConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &HybridConfig, usize, String, HybridLayerPolicy, TokenMixer<B>,
            NormalizationConstructionSpec, &<B::Tensor as Tensor>::Context)>()?;
        if config.is_moe() {
            return Err(metadata.error(format_args!("replicated Qwen hybrid unit rejects routed computation")));
        }
        let policy = config
            .layer_schedule
            .get(layer)
            .copied()
            .ok_or_else(|| metadata.error(format_args!("Qwen hybrid has no layer {layer}")))?;
        let root = metadata.text(format_args!("model.layers.{layer}"))?;
        let mixer = match policy {
            HybridLayerPolicy::LinearAttention => {
                TokenMixer::Linear(LinearAttention::new(config, layer, context)?)
            }
            HybridLayerPolicy::SelfAttention(_) => {
                TokenMixer::Attention(new_attention(config, &root, context)?)
            }
        };
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec {
                    groups: None,
                    dimensions: config.hidden_size,
                    epsilon: config.rms_norm_eps,
                    scale: NormalizationScale::LearnedOffset {
                        weight: metadata.named_parameter(format_args!("{root}.{field}.weight"))?,
                        offset: 1.0,
                    },
                },
                context,
            )
        };
        metadata.borrowed_controls(&norm)?;
        Ok(Self {
            mixer,
            feed_forward: new_mlp(
                config,
                &metadata.text(format_args!("{root}.mlp"))?,
                config.intermediate_size,
                context,
            )?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    pub(crate) fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_instrumented(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    pub(crate) fn forward_instrumented<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let hidden = forward_mixer::<B, S>(
            &mut self.mixer,
            &mut self.input_norm,
            hidden,
            mask,
            state,
            context,
            instrumentation,
        )?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let feed_forward = self.feed_forward.forward_feed_forward_observed(
            &normalized,
            context,
            instrumentation,
        )?;
        finish_feed_forward::<B>(&hidden, feed_forward, context, instrumentation)
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Block<B> {
    /// Builds one global-geometry physical decoder layer.
    pub fn new(
        config: &HybridConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_routed_spec(config, layer, None, context)
    }

    pub(crate) fn new_with_routed_spec(
        config: &HybridConfig,
        layer: usize,
        routed_spec: Option<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        construction::require_source_compiler::<B>(context)?;
        TargetBlockSpec::new(config, layer, routed_spec)?.instantiate::<B>(config, context)
    }

    /// Builds one configured MTP prediction block at its checkpoint path.
    pub fn new_mtp(
        config: &HybridConfig,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        construction::require_source_compiler::<B>(context)?;
        PredictionBlockSpec::new(config, depth)?.instantiate::<B>(context)
    }

    /// Executes one block with resident routed experts.
    pub fn forward<S>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        self.forward_with_provider(hidden, mask, state, context, &mut ResidentExpertProvider)
    }

    /// Executes one block through a runtime-owned routed-expert provider.
    pub fn forward_with_provider<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
            |ff, input, context, _| ff.forward_with_provider(input, context, provider),
        )
    }

    /// Executes one block while exposing the complete routed/shared contribution.
    pub fn forward_observed_with_provider<S, P, O>(
        &mut self,
        point: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut ComponentInstrumentation::disabled(),
            |ff, input, context, _| {
                ff.forward_observed_with_provider(point, input, context, provider, observer)
            },
        )
    }

    /// Component and routed observations use the same architecture-owned boundaries.
    pub(crate) fn forward_components_with_provider<S, P, O>(
        &mut self,
        path: &str,
        point: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(path, &mut borrowed);
        self.forward_instrumented_with_feed_forward(
            hidden,
            mask,
            state,
            context,
            &mut instrumentation,
            |ff, input, context, instrumentation| match ff {
                FeedForward::Dense(mlp) => {
                    mlp.forward_feed_forward_observed(input, context, instrumentation)
                }
                FeedForward::Routed(moe) => match instrumentation.observer() {
                    Some(observer) => moe
                        .forward_observed_with_provider(point, input, context, provider, observer),
                    None => moe.forward_with_provider(input, context, provider),
                },
            },
        )
    }

    fn forward_instrumented_with_feed_forward<S, F>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let hidden = forward_mixer::<B, S>(
            &mut self.mixer,
            &mut self.input_norm,
            hidden,
            mask,
            state,
            context,
            instrumentation,
        )?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let output = feed_forward(
            &mut self.feed_forward,
            &normalized,
            context,
            instrumentation,
        )?;
        finish_feed_forward::<B>(&hidden, output, context, instrumentation)
    }

    /// Executes local projections and one row reduction per parallel output.
    pub fn forward_parallel<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_parallel_instrumented(
            hidden,
            mask,
            state,
            parallel,
            context,
            provider,
            &mut ComponentInstrumentation::disabled(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_components_parallel_with_provider<S, P, O>(
        &mut self,
        path: &str,
        points: eredu_runtime::RoutedObservationPoints,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_parallel_instrumented(
            hidden,
            mask,
            state,
            parallel,
            context,
            provider,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            Some(points),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_parallel_instrumented<S, P>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        provider: &mut P,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        points: Option<eredu_runtime::RoutedObservationPoints>,
    ) -> Result<B::Tensor, Error>
    where
        S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        P: TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let normalized = self.input_norm.forward(hidden, context)?;
        let (mixed, boundary) = match &mut self.mixer {
            TokenMixer::Linear(linear) => {
                let normalized = instrumentation.apply("mixer.input", normalized)?;
                (
                    linear.forward_parallel_instrumented(
                        &normalized,
                        state,
                        parallel,
                        context,
                        instrumentation,
                    )?,
                    "mixer",
                )
            }
            TokenMixer::Attention(attention) => {
                let normalized = instrumentation.apply("attention.input", normalized)?;
                (
                    attention.forward_instrumented(
                        AttentionInput {
                            hidden: &normalized,
                            mask,
                            cache: Some(&mut *state),
                            allow_sliding_prefill: true,
                            rotary_position: None,
                        },
                        Some(parallel),
                        context,
                        instrumentation,
                    )?,
                    "attention",
                )
            }
        };
        let mixed = instrumentation.apply(
            if boundary == "attention" {
                "attention.write"
            } else {
                "mixer.write"
            },
            mixed,
        )?;
        let mixed = instrumentation.apply(
            if boundary == "attention" {
                "attention.output"
            } else {
                "mixer.output"
            },
            mixed,
        )?;
        let hidden = instrumentation.apply(
            if boundary == "attention" {
                "attention.residual"
            } else {
                "mixer.residual"
            },
            hidden.add(&mixed, context)?,
        )?;
        let normalized = instrumentation.apply(
            "feed_forward.input",
            self.post_attention_norm.forward(&hidden, context)?,
        )?;
        let feed_forward = match &mut self.feed_forward {
            FeedForward::Dense(mlp) => mlp.forward_feed_forward_parallel_observed(
                &normalized,
                parallel,
                context,
                instrumentation,
            )?,
            FeedForward::Routed(moe) => {
                if let Some(observer) = instrumentation.observer() {
                    moe.forward_tensor_parallel_observed_with_provider(
                        points.expect("observed routed block declares its routing points"),
                        &normalized,
                        parallel,
                        context,
                        provider,
                        observer,
                    )?
                } else {
                    let output = moe.forward_tensor_parallel_with_provider(
                        &normalized,
                        parallel,
                        context,
                        provider,
                    )?;
                    eredu_runtime::reduce_routed_expert_tensor_parallel::<B>(
                        output, parallel, context,
                    )?
                }
            }
        };
        finish_feed_forward::<B>(&hidden, feed_forward, context, instrumentation)
    }
}

fn new_attention<B: NeuralBackend>(
    config: &HybridConfig,
    root: &str,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Attention<B>, Error> {
    construction::AttentionSpec::new_with_metadata(
        config, root, crate::decoder::ModuleMetadata::new::<B>(context),
    )?.instantiate::<B>(context)
}

fn new_mlp<B: NeuralBackend>(
    config: &HybridConfig,
    prefix: &str,
    intermediate: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Mlp<B>, Error> {
    construction::MlpSpec::new_with_metadata(
        config, prefix, intermediate, crate::decoder::ModuleMetadata::new::<B>(context),
    )?.instantiate::<B>(context)
}

fn parameter(name: impl AsRef<str>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name.as_ref()).map_err(Error::backend)
}

fn pass<T: Tensor>(input: &T) -> ExpertPass {
    if input
        .shape()
        .get(input.shape().len().saturating_sub(2))
        .copied()
        .unwrap_or(1)
        > 1
    {
        ExpertPass::Prefill
    } else {
        ExpertPass::Decode
    }
}

impl HybridConfig {
    /// Selection policy shared by module construction and intervention discovery.
    pub(crate) fn routing_spec(&self) -> Result<TopKGroupSelectionSpec, Error> {
        TopKGroupSelectionSpec::new(
            self.num_experts,
            self.num_experts_per_tok,
            GroupScoring::Softmax,
            self.norm_topk_prob,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_mixer<B: NeuralBackend, S>(
    mixer: &mut TokenMixer<B>,
    norm: &mut B::Normalization,
    hidden: &B::Tensor,
    mask: Option<&B::Tensor>,
    state: &mut S,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error>
where
    S: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    let normalized = norm.forward(hidden, context)?;
    let (mixed, boundary) = match mixer {
        TokenMixer::Linear(linear) => {
            let normalized = instrumentation.apply("mixer.input", normalized)?;
            (
                linear.forward_instrumented(&normalized, state, context, instrumentation)?,
                "mixer",
            )
        }
        TokenMixer::Attention(attention) => {
            let normalized = instrumentation.apply("attention.input", normalized)?;
            (
                attention.forward_instrumented(
                    AttentionInput {
                        hidden: &normalized,
                        mask,
                        cache: Some(state),
                        allow_sliding_prefill: true,
                        rotary_position: None,
                    },
                    None,
                    context,
                    instrumentation,
                )?,
                "attention",
            )
        }
    };
    let (write, output, residual) = if boundary == "attention" {
        ("attention.write", "attention.output", "attention.residual")
    } else {
        ("mixer.write", "mixer.output", "mixer.residual")
    };
    let mixed = instrumentation.apply(write, mixed)?;
    let mixed = instrumentation.apply(output, mixed)?;
    instrumentation.apply(residual, hidden.add(&mixed, context)?)
}
fn finish_feed_forward<B: NeuralBackend>(
    hidden: &B::Tensor,
    output: B::Tensor,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
) -> Result<B::Tensor, Error> {
    let output = instrumentation.apply("feed_forward.write", output)?;
    let output = instrumentation.apply("feed_forward.output", output)?;
    instrumentation.apply("feed_forward.residual", hidden.add(&output, context)?)
}
