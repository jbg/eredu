//! Shared text-stack lifecycle for resident and bounded execution mechanisms.

use super::{
    text::{TextLayer, TextStatic},
    FamilyConfig, TextInvocationPhase,
};
use crate::{
    decoder::ComponentInstrumentation,
    gemma4::{BlockInput, SharedAttentionStates},
};
use eredu_nn::{
    AttentionCache, Error, GroupedNeuralBackend, ReadOnlyAttentionCache, RotaryPosition, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, ArchitectureParameters, ExecutionGraph, LayerRuntimeState,
    LayeredArchitecture, LayeredForwardState, ParameterProvider, PartitionedLayeredArchitecture,
    ResidentExpertProvider, RoutedLayeredArchitecture, StateLayout,
};

/// Stable identity of the shared encoder/refinement execution stack.
pub const TEXT_EXECUTION_GROUP: &str = "shared_text";

/// Shared-stack placement, with decoder-only feedback retained on the input owner.
pub(crate) fn text_transport() -> eredu_runtime::ArchitectureGroupTransport {
    eredu_runtime::ArchitectureGroupTransport {
        placement: eredu_runtime::ArchitectureGroupPlacement::Pipeline,
        kind: eredu_runtime::ArchitectureGroupKind::Decoder,
        first_owner_static_roles: vec!["embedding".into(), "conditioning".into()],
        last_owner_static_roles: vec!["norm".into(), "output".into()],
        merge_destination: eredu_runtime::ArchitectureMergeDestination::OutputOwner,
        parallel_subgroup: Some(eredu_runtime::ArchitectureParallelSubgroup::Decoder),
        request_optional: false,
    }
}

/// One architecture invocation; refinement reads the context owned by the state.
pub enum TextInput<'a, T> {
    /// Authoritative tokens whose projections are appended to context.
    Context {
        /// Prepared token identities, `[batch, positions]`.
        tokens: &'a T,
        /// Complete architecture-prepared visibility, when supplied.
        mask: Option<&'a T>,
    },
    /// Uncommitted canvas and previous processed position scores.
    Refinement {
        /// Tentative token identities, `[batch, canvas]`.
        tokens: &'a T,
        /// Scores retained in embedding arithmetic, `[batch, canvas, vocabulary]`.
        feedback: Option<&'a T>,
        /// Complete prefix/canvas visibility, when supplied.
        mask: Option<&'a T>,
    },
}

/// Prepared ingress plus the explicit mask supplied to each pipeline stage.
/// Masks remain request values; phase/frontier metadata travels with activations.
pub struct PreparedPartitionInvocation<T: 'static> {
    /// Owned residual input, with typed shared-stack invocation metadata.
    pub input: eredu_runtime::LayeredPartitionInput<'static, T, super::boundary::TextBoundary<T>>,
    /// Visibility retained from the request and used by every stage.
    pub mask: Option<T>,
}

/// Invocation-local positions and publications; shared weights are not copied.
pub struct ForwardContext<T> {
    phase: TextInvocationPhase,
    mask: Option<T>,
    position: i32,
    shared: SharedAttentionStates<T>,
}

/// Family-owned text equations traversed through ordinary runtime residency.
pub struct TextModel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    config: FamilyConfig,
    statics: TextStatic<B>,
    geometry: Option<super::parallel::LocalGeometry>,
    experts: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Clone for TextModel<B> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            statics: self.statics.clone(),
            geometry: self.geometry.clone(),
            experts: self.experts.clone(),
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    eredu_nn::Parameterized<B::Tensor> for TextModel<B>
{
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, B::Tensor>>(
        &'a self,
        visitor: &mut V,
    ) {
        self.statics.visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.statics.visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.statics.set_trainable(trainable);
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TextModel<B> {
    fn context_mask(
        &self,
        forward: &ForwardContext<B::Tensor>,
        query: i32,
        window: Option<i32>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error> {
        if forward.phase != TextInvocationPhase::Context {
            return Ok(forward.mask.clone());
        }
        if let Some(mask) = &forward.mask {
            let Some(window) = window else {
                return Ok(Some(mask.clone()));
            };
            let keys = *mask
                .shape()
                .last()
                .ok_or_else(|| Error::backend("context visibility has no key axis"))?;
            let origin = forward.position + query - keys;
            let values = (0..query)
                .flat_map(|q| {
                    (0..keys).map(move |k| i32::from(origin + k > forward.position + q - window))
                })
                .collect::<Vec<_>>();
            let visible = B::Tensor::from_i32_slice(&values, &[1, 1, query, keys], context)?
                .equal_i32(1, context)?;
            let merged =
                match mask.element_type() {
                    Some(eredu_nn::TensorElementType::Bool) => {
                        let excluded =
                            B::Tensor::from_i32_slice(&[0], &[], context)?.equal_i32(1, context)?;
                        B::Tensor::where_condition(mask, &visible, &excluded, context)?
                    }
                    Some(
                        eredu_nn::TensorElementType::F16
                        | eredu_nn::TensorElementType::Bf16
                        | eredu_nn::TensorElementType::F32
                        | eredu_nn::TensorElementType::F64,
                    ) => {
                        let excluded = B::Tensor::full_f32(f32::NEG_INFINITY, &[], context)?;
                        B::Tensor::where_condition(&visible, mask, &excluded, context)?
                    }
                    _ => return Err(Error::backend(
                        "context mask requires known boolean visibility or floating attention bias",
                    )),
                };
            return Ok(Some(merged));
        }
        if self.config.encoder_attention == super::EncoderAttention::CausalWithVisionSpans {
            return Ok(Some(B::causal_mask(
                query,
                forward.position,
                window.map(|n| n - 1),
                context,
            )?));
        }
        Ok(None)
    }

    pub(super) fn begin_context_embeddings<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
        state: &mut S,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                phase: TextInvocationPhase::Context,
                mask,
                position: self.position(state)?,
                shared: SharedAttentionStates::new(),
            },
        })
    }

    pub(super) fn begin_invocation_embeddings<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<B::Tensor>,
        phase: TextInvocationPhase,
        state: &mut S,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                phase,
                mask,
                position: self.position(state)?,
                shared: SharedAttentionStates::new(),
            },
        })
    }

    pub(super) fn encode_context_tokens(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.statics.encode_input(tokens, context)
    }
    pub(super) fn encode_context_tokens_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: eredu_nn::TensorParallelGroupedNeuralBackend,
    {
        self.statics
            .encode_input_parallel(tokens, parallel, context)
    }
    /// Declares pinned text modules independently from per-layer materialization.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            statics: TextStatic::new(&config, context)?,
            config,
            geometry: None,
            experts: None,
        })
    }

    /// Constructs only parameters and state owned by the retained parallel partition.
    pub fn new_parallel(
        config: FamilyConfig,
        geometry: super::parallel::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        geometry.validate_for(&config).map_err(Error::backend)?;
        Ok(Self {
            statics: TextStatic::new_parallel(&config, &geometry, context)?,
            config,
            geometry: Some(geometry),
            experts: None,
        })
    }

    /// Derives ownership of the shared routed banks for an admitted rank topology.
    pub fn expert_realization_plan(
        &self,
        topology: eredu_core::ParallelRankTopology,
    ) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, Error>
    {
        let Some(count) = self.config.text.num_experts else {
            return Ok(None);
        };
        let global = usize::try_from(count).map_err(Error::backend)?;
        let local = eredu_core::balanced_contiguous_range(
            global,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(Error::backend)?;
        let owner =
            eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP).map_err(Error::backend)?;
        let mut specs = std::collections::BTreeMap::new();
        for (index, policy) in self.config.text.layer_schedule.iter().enumerate() {
            if policy.feed_forward != crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe {
                continue;
            }
            let args = self
                .geometry
                .as_ref()
                .and_then(|g| g.block_args(index))
                .unwrap_or(&self.config.text);
            let width = if self
                .geometry
                .as_ref()
                .is_some_and(|g| g.block_args(index).is_none())
            {
                let global_width = self
                    .config
                    .text
                    .moe_intermediate_size
                    .ok_or_else(|| Error::backend("missing routed intermediate geometry"))?;
                let owned = eredu_core::balanced_contiguous_range(
                    global_width as usize,
                    topology.tensor_parallel_size(),
                    topology.tensor_parallel_rank(),
                    false,
                )
                .map_err(Error::backend)?;
                i32::try_from(owned.len()).map_err(Error::backend)?
            } else {
                args.moe_intermediate_size
                    .ok_or_else(|| Error::backend("missing routed intermediate geometry"))?
            };
            let spec = crate::gemma4::text::expert_bank_spec_at(
                &self.config.text,
                &format!("model.decoder.layers.{index}.experts.switch_glu"),
            )?
            .with_group_geometry(i32::try_from(local.len()).map_err(Error::backend)?, width)?;
            specs.insert((owner.clone(), index), spec);
        }
        if specs.is_empty() {
            return Ok(None);
        }
        crate::ExpertRealizationPlan::balanced(global, topology, specs)
            .map(Some)
            .map_err(Error::backend)
    }

    /// Retains exact shared-bank ownership for native or neutral routed providers.
    pub fn with_expert_realization(
        mut self,
        plan: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        if self.config.text.num_experts.map(|n| n as usize) != Some(plan.global_expert_count()) {
            return Err(Error::backend(
                "expert realization differs from the admitted global bank",
            ));
        }
        let owner =
            eredu_runtime::ExecutionGroupId::new(TEXT_EXECUTION_GROUP).map_err(Error::backend)?;
        let expected = self
            .config
            .text
            .layer_schedule
            .iter()
            .enumerate()
            .filter(|(_, p)| p.feed_forward == crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe)
            .map(|(i, _)| (owner.clone(), i))
            .collect::<std::collections::BTreeSet<_>>();
        if plan
            .unit_specs()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            != expected
        {
            return Err(Error::backend(
                "expert realization differs from the shared invocation owners",
            ));
        }
        self.experts = Some(plan);
        Ok(self)
    }

    fn state_index(&self, index: usize) -> Result<usize, Error> {
        match &self.geometry {
            Some(geometry) if geometry.units().contains(&index) => {
                Ok(index - geometry.units().start)
            }
            Some(_) => Err(Error::backend(
                "diffusion unit lies outside the local state partition",
            )),
            None if index < self.config.text.num_hidden_layers() => Ok(index),
            None => Err(Error::backend(
                "diffusion unit lies outside the shared stack",
            )),
        }
    }

    /// Exact retained family semantics.
    pub fn config(&self) -> &FamilyConfig {
        &self.config
    }

    fn graph(&self) -> Result<ExecutionGraph, Error> {
        ExecutionGraph::chain([TEXT_EXECUTION_GROUP]).map_err(Error::backend)
    }

    fn position<S>(&self, state: &mut S) -> Result<i32, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        let expected = self
            .geometry
            .as_ref()
            .map(|geometry| geometry.state_layout().clone())
            .unwrap_or(super::state_layout(&self.config).map_err(Error::backend)?);
        if state.layout() != &expected {
            return Err(Error::backend(
                "diffusion state does not match the retained local geometry",
            ));
        }
        if expected.is_empty() {
            return Ok(0);
        }
        let position = state.layer(0).map_err(Error::backend)?.offset();
        for index in 1..expected.len() {
            if state.layer(index).map_err(Error::backend)?.offset() != position {
                return Err(Error::backend(
                    "diffusion shared context frontiers disagree",
                ));
            }
        }
        Ok(position)
    }

    fn run_unit<S, P>(
        &mut self,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let cache = state
            .layer(self.state_index(index)?)
            .map_err(Error::backend)?;
        let generated = self.context_mask(forward, hidden.dim(1), cache.max_size(), context)?;
        let input = BlockInput {
            hidden,
            mask: generated.as_ref(),
            cache: Some(cache),
            shared: &mut forward.shared,
            per_layer_input: None,
            rotary_position: Some(RotaryPosition::Offset(forward.position)),
        };
        match forward.phase {
            TextInvocationPhase::Context => {
                unit.encode(input, pass, provider, context, instrumentation)
            }
            TextInvocationPhase::Refinement => {
                unit.refine(input, pass, provider, context, instrumentation)
            }
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> ArchitectureParameters<B>
    for TextModel<B>
{
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        match &self.geometry {
            Some(geometry) => Ok(geometry.state_layout().clone()),
            None => super::state_layout(&self.config).map_err(Error::backend),
        }
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        eredu_runtime::ModelStateIdentity::new(
            "diffusion_gemma",
            "diffusion_gemma_text",
            self.config.architecture_fingerprint(),
            self.config.text.num_hidden_layers(),
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend)
    }
    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ArchitectureParameterDescription, Error> {
        super::parallel::parameter_description(&self.config).map_err(Error::backend)
    }
    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(module) = &self.statics.embeddings {
            visitor.visit("embedding", module)?;
        }
        if let Some(module) = &self.statics.norm {
            visitor.visit("norm", module)?;
        }
        if let Some(head) = &self.statics.head {
            visitor.visit("output", head)?;
        }
        if let Some(module) = &self.statics.conditioning {
            visitor.visit("conditioning", module)?;
        }
        Ok(())
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        if let Some(module) = &mut self.statics.embeddings {
            visitor.visit_mut("embedding", module)?;
        }
        if let Some(module) = &mut self.statics.norm {
            visitor.visit_mut("norm", module)?;
        }
        if let Some(head) = &mut self.statics.head {
            visitor.visit_mut("output", head)?;
        }
        if let Some(module) = &mut self.statics.conditioning {
            visitor.visit_mut("conditioning", module)?;
        }
        Ok(())
    }
}

impl<B, S> LayeredArchitecture<B, S> for TextModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(false, true, true)
            .with_routed_units(true)
    }

    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let pass = <Self as RoutedLayeredArchitecture<B, S>>::expert_pass_for_unit(
            self, group, index, hidden, forward,
        );
        <Self as RoutedLayeredArchitecture<B, S>>::forward_unit_observed_with_provider(
            self,
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut ResidentExpertProvider,
            context,
            observer,
        )
    }

    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        match forward.phase {
            TextInvocationPhase::Context => None,
            TextInvocationPhase::Refinement => {
                Some(eredu_core::refinement::PredictionScoreKind::RefinementPosition)
            }
        }
    }

    type Input<'a>
        = TextInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = TextStatic<B>;
    type Unit = TextLayer<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        text_transport()
    }
    fn primary_execution_group(&self) -> &str {
        TEXT_EXECUTION_GROUP
    }
    fn state_partition_plan(
        &self,
        layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        eredu_runtime::ArchitectureStatePartitionPlan::new([
            eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..layout.len()),
        ])
    }
    fn execution_graph(&self) -> Result<ExecutionGraph, Error> {
        self.graph()
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        if group != 0 {
            return Err(Error::backend("invalid diffusion text group"));
        }
        Ok(self.config.text.num_hidden_layers())
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if index >= <Self as LayeredArchitecture<B, S>>::group_unit_count(self, group)? {
            return Err(Error::backend(
                "diffusion unit lies outside the shared stack",
            ));
        }
        Ok(format!("model.decoder.layers.{index}"))
    }
    fn static_modules(&self) -> &TextStatic<B> {
        &self.statics
    }
    fn static_modules_mut(&mut self) -> &mut TextStatic<B> {
        &mut self.statics
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextLayer<B>, Error> {
        <Self as LayeredArchitecture<B, S>>::unit_path(self, group, index)?;
        let args = match &self.geometry {
            Some(geometry) => geometry
                .block_args(index)
                .ok_or_else(|| Error::backend("diffusion unit is outside this partition"))?,
            None => &self.config.text,
        };
        let experts = self
            .experts
            .as_ref()
            .and_then(|plan| plan.unit_spec(TEXT_EXECUTION_GROUP, index))
            .cloned();
        TextLayer::new_with_experts(args, index, experts, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: TextInput<'a, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let position = self.position(state)?;
        let (hidden, mask, phase) = match input {
            TextInput::Context { tokens, mask } => (
                self.statics.encode_input(tokens, context)?,
                mask,
                TextInvocationPhase::Context,
            ),
            TextInput::Refinement {
                tokens,
                feedback,
                mask,
            } => {
                if tokens.shape().len() != 2
                    || tokens.dim(1) <= 0
                    || tokens.dim(1) as u32 > self.config.canvas_length.get()
                {
                    return Err(Error::backend("diffusion canvas exceeds admitted geometry"));
                }
                (
                    self.statics.refine_input(tokens, feedback, context)?,
                    mask,
                    TextInvocationPhase::Refinement,
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                phase,
                mask: mask.cloned(),
                position,
                shared: SharedAttentionStates::new(),
            },
        })
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut ForwardContext<B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend(
                "diffusion text group has no external execution dependencies",
            ));
        }
        Ok(initial.clone())
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 {
            return Err(Error::backend("invalid diffusion text group"));
        }
        self.run_unit(
            index,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut borrowed);
        match forward.phase {
            TextInvocationPhase::Context => {
                self.statics
                    .normalize_observed(hidden, context, &mut instrumentation)
            }
            TextInvocationPhase::Refinement => {
                self.statics
                    .readout_observed(hidden, None, context, &mut instrumentation)
            }
        }
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match forward.phase {
            TextInvocationPhase::Context => self.statics.normalize(hidden, context),
            TextInvocationPhase::Refinement => self.statics.readout(hidden, context),
        }
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a ForwardContext<B::Tensor>,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, S> RoutedLayeredArchitecture<B, S> for TextModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }

    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if group != 0 {
            return Err(Error::backend("invalid diffusion text group"));
        }
        let path = format!("model.decoder.layers.{index}");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.run_unit(
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }

    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if group != 0 {
            return Err(Error::backend("invalid diffusion text group"));
        }
        self.run_unit(
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
}

impl<B, S> eredu_runtime::ParallelLayeredArchitecture<B, S> for TextModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let pass = <Self as RoutedLayeredArchitecture<B, S>>::expert_pass_for_unit(
            self, group, index, hidden, forward,
        );
        <Self as eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>>::forward_unit_parallel_observed_with_provider(
            self, group, index, unit, hidden, state, forward, pass, &mut ResidentExpertProvider, parallel, context, observer)
    }

    fn begin_forward_parallel<'a>(
        &mut self,
        input: TextInput<'a, B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        self.geometry.as_ref().ok_or_else(|| {
            Error::backend("tensor-parallel diffusion requires retained local geometry")
        })?;
        let position = self.position(state)?;
        let (hidden, mask, phase) = match input {
            TextInput::Context { tokens, mask } => (
                self.statics
                    .encode_input_parallel(tokens, parallel, context)?,
                mask,
                TextInvocationPhase::Context,
            ),
            TextInput::Refinement {
                tokens,
                feedback,
                mask,
            } => {
                if tokens.shape().len() != 2
                    || tokens.dim(1) <= 0
                    || tokens.dim(1) as u32 > self.config.canvas_length.get()
                {
                    return Err(Error::backend("diffusion canvas exceeds admitted geometry"));
                }
                (
                    self.statics
                        .refine_input_parallel(tokens, feedback, parallel, context)?,
                    mask,
                    TextInvocationPhase::Refinement,
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                phase,
                mask: mask.cloned(),
                position,
                shared: SharedAttentionStates::new(),
            },
        })
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        <Self as eredu_runtime::ParallelRoutedLayeredArchitecture<B, S>>::forward_unit_parallel_with_provider(self, group, index, unit, hidden, state, forward, eredu_runtime::ExpertPass::Prefill, &mut ResidentExpertProvider, parallel, context)
    }
    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut borrowed);
        match forward.phase {
            TextInvocationPhase::Context => {
                self.statics
                    .normalize_observed(hidden, context, &mut instrumentation)
            }
            TextInvocationPhase::Refinement => {
                self.statics
                    .readout_observed(hidden, Some(parallel), context, &mut instrumentation)
            }
        }
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match forward.phase {
            TextInvocationPhase::Context => self.statics.normalize(hidden, context),
            TextInvocationPhase::Refinement => {
                self.statics.readout_parallel(hidden, parallel, context)
            }
        }
    }
}

impl<B, S> eredu_runtime::ParallelRoutedLayeredArchitecture<B, S> for TextModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let path = format!("model.decoder.layers.{index}");
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.run_unit_parallel(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut borrowed),
        )
    }

    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.run_unit_parallel(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
}

impl<B> TextModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Prepares context or feedback ingress once on its owner, before the shared
    /// partition driver traverses the stack. The typed boundary retains phase.
    pub fn prepare_partition_input<S>(
        &mut self,
        input: TextInput<'_, B::Tensor>,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedPartitionInvocation<B::Tensor>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        let prepared = match parallel {
            Some(parallel) => {
                <Self as eredu_runtime::ParallelLayeredArchitecture<B, S>>::begin_forward_parallel(
                    self, input, state, parallel, context,
                )?
            }
            None => {
                <Self as LayeredArchitecture<B, S>>::begin_forward(self, input, state, context)?
            }
        };
        Ok(PreparedPartitionInvocation {
            input: eredu_runtime::LayeredPartitionInput::Hidden {
                hidden: prepared.hidden,
                auxiliary: Self::boundary_value(&prepared.context, context)?,
            },
            mask: prepared.context.mask,
        })
    }

    fn boundary_value(
        forward: &ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::boundary::TextBoundary<B::Tensor>, Error> {
        let phase = match forward.phase {
            TextInvocationPhase::Context => 0,
            TextInvocationPhase::Refinement => 1,
        };
        Ok(super::boundary::TextBoundary {
            invocation: B::Tensor::from_i32_slice(&[phase, forward.position], &[2], context)?,
        })
    }

    fn resume_partition<S>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            '_,
            B::Tensor,
            super::boundary::TextBoundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        if expected != state.layout() || first != 0 {
            return Err(Error::backend(
                "diffusion pipeline state ownership disagrees with local geometry",
            ));
        }
        let position = self.position(state)?;
        match input {
            eredu_runtime::LayeredPartitionInput::Tokens(tokens) => match parallel {
                Some(parallel) => <Self as eredu_runtime::ParallelLayeredArchitecture<B, S>>::begin_forward_parallel(self, TextInput::Context { tokens, mask }, state, parallel, context),
                None => <Self as LayeredArchitecture<B, S>>::begin_forward(self, TextInput::Context { tokens, mask }, state, context),
            },
            eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if auxiliary.invocation.shape() != [2] { return Err(Error::backend("invalid diffusion invocation boundary geometry")); }
                let words = auxiliary.invocation.to_i32_vec(context)?;
                if words.len() != 2 || words[1] != position { return Err(Error::backend("diffusion pipeline context frontier differs from the transported invocation")); }
                let phase = match words[0] { 0 => TextInvocationPhase::Context, 1 => TextInvocationPhase::Refinement, _ => return Err(Error::backend("unknown diffusion invocation phase")) };
                if hidden.shape().len() != 3 || hidden.dim(0) <= 0 || hidden.dim(1) <= 0 || hidden.dim(2) != self.config.text.hidden_size || (phase == TextInvocationPhase::Refinement && hidden.dim(1) as u32 > self.config.canvas_length.get()) { return Err(Error::backend("invalid diffusion partition activation geometry")); }
                Ok(LayeredForwardState { hidden, context: ForwardContext { phase, position, mask: mask.cloned(), shared: SharedAttentionStates::new() } })
            },
        }
    }
}

impl<B, S> eredu_runtime::PartitionedLayeredArchitecture<B, S> for TextModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = super::boundary::TextBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        Ok(Self::Boundary::new(self.config.text.hidden_size))
    }
    fn begin_partition<'a>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            'a,
            B::Tensor,
            super::boundary::TextBoundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        self.resume_partition(input, mask, state, expected, first, None, context)
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: eredu_runtime::LayeredPartitionInput<
            'a,
            B::Tensor,
            super::boundary::TextBoundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        self.resume_partition(input, mask, state, expected, first, Some(parallel), context)
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ForwardContext<B::Tensor>,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        eredu_runtime::LayeredPartitionOutput<B::Tensor, super::boundary::TextBoundary<B::Tensor>>,
        Error,
    > {
        if owns_output {
            let output = match parallel {
                Some(parallel) => <Self as eredu_runtime::ParallelLayeredArchitecture<B, S>>::finish_forward_parallel(self, hidden, state, forward, parallel, context)?,
                None => <Self as LayeredArchitecture<B, S>>::finish_forward(self, hidden, state, forward, context)?,
            };
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: Self::boundary_value(forward, context)?,
            })
        }
    }
}

impl<B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend>
    TextModel<B>
{
    #[allow(clippy::too_many_arguments)]
    fn run_unit_parallel<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        if group != 0 {
            return Err(Error::backend("invalid diffusion text group"));
        }
        let cache = state
            .layer(self.state_index(index)?)
            .map_err(Error::backend)?;
        let generated = self.context_mask(forward, hidden.dim(1), cache.max_size(), context)?;
        let input = BlockInput {
            hidden,
            mask: generated.as_ref(),
            cache: Some(cache),
            shared: &mut forward.shared,
            per_layer_input: None,
            rotary_position: Some(RotaryPosition::Offset(forward.position)),
        };
        match forward.phase {
            TextInvocationPhase::Context => {
                unit.encode_parallel(input, pass, provider, parallel, context, instrumentation)
            }
            TextInvocationPhase::Refinement => {
                unit.refine_parallel(input, pass, provider, parallel, context, instrumentation)
            }
        }
    }
}

impl<B, S> eredu_runtime::generation::refinement::architecture::BlockRefinementArchitecture<B, S>
    for TextModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn refinement_geometry(
        &self,
    ) -> eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
        eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry {
            canvas_length: self.config.canvas_length,
            vocabulary_size: std::num::NonZeroU32::new(self.config.text.vocab_size as u32)
                .expect("validated vocabulary"),
        }
    }
    fn context_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        TextInput::Context { tokens, mask }
    }
    fn refinement_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
    ) -> Self::Input<'a> {
        TextInput::Refinement {
            tokens,
            feedback,
            mask,
        }
    }
}

impl<B, S> eredu_runtime::PipelinePartitionArchitecture<B, S> for TextModel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn partition_input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
        let tokens = match input {
            TextInput::Context { tokens, .. } | TextInput::Refinement { tokens, .. } => tokens,
        };
        if tokens.shape().len() != 2 || tokens.dim(0) <= 0 || tokens.dim(1) <= 0 {
            return Err(Error::backend(
                "shared-stack pipeline input requires nonempty batched tokens",
            ));
        }
        Ok((tokens.dim(0), tokens.dim(1)))
    }

    fn begin_pipeline_partition<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, super::boundary::TextBoundary<B::Tensor>)>,
        state: &mut S,
        expected: Option<&StateLayout>,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let expected = expected.ok_or_else(|| {
            Error::backend("shared-stack pipeline requires selected state geometry")
        })?;
        let mask = match &input {
            TextInput::Context { mask, .. } | TextInput::Refinement { mask, .. } => mask.cloned(),
        };
        let prepared = match incoming {
            Some((hidden, auxiliary)) => {
                eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary }
            }
            None => {
                self.prepare_partition_input(input, state, parallel, context)?
                    .input
            }
        };
        self.begin_partition_observed(
            prepared,
            mask.as_ref(),
            state,
            expected,
            first_state_ordinal,
            parallel,
            context,
            observer,
        )
    }

    fn partition_output_width(&self) -> i32 {
        self.config.text.vocab_size
    }

    fn partition_output_shape(
        &self,
        input: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Error> {
        let width = match input {
            TextInput::Context { .. } => self.config.text.hidden_size,
            TextInput::Refinement { .. } => self.config.text.vocab_size,
        };
        Ok(vec![batch, sequence, width])
    }

    fn partition_routed_tensor_reductions(
        &self,
        _unit: usize,
        routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Error> {
        // Attention and the always-present dense branch reduce before routing;
        // the selected-expert write reduces afterward, before normalization.
        Ok(eredu_runtime::RoutedTensorReductions::hidden(
            if routed { 2 } else { 1 },
            1,
        ))
    }
}

impl<B, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for TextModel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn supports_chunked_prefill() -> bool {
        true
    }

    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        TextInput::Context { tokens, mask }
    }
    fn block_refinement_geometry(
        &self,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        Some(<Self as eredu_runtime::generation::refinement::architecture::BlockRefinementArchitecture<B, S>>::refinement_geometry(self))
    }
    fn invocation_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
        invocation: eredu_core::decoding::AttentionInvocation,
    ) -> Option<Self::Input<'a>> {
        if invocation == eredu_core::decoding::AttentionInvocation::CAUSAL && feedback.is_none() {
            return Some(TextInput::Context { tokens, mask });
        }
        if invocation != eredu_core::decoding::AttentionInvocation::DIFFUSION {
            return None;
        }
        Some(TextInput::Refinement {
            tokens,
            feedback,
            mask,
        })
    }
    fn text_output_selection(&self) -> eredu_runtime::ReplicatedTextOutputSelection {
        eredu_runtime::ReplicatedTextOutputSelection::AllSequencePositions
    }
}
