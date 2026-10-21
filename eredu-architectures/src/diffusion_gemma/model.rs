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
    LayerRuntimeState, LayeredForwardState, ParameterProvider, ResidentExpertProvider, StateLayout,
};

use eredu_runtime::PartitionComponentKernel as _;

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

/// Validates shared text storage, context slots, and invocation addressing without native resources.
pub fn component_definition(
    config: &FamilyConfig,
    geometry: Option<&super::parallel::LocalGeometry>,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let state = match geometry {
        Some(geometry) => geometry
            .complete_state_layout(config)
            .map_err(Error::backend_source)?,
        None => super::state_layout(config).map_err(Error::backend_source)?,
    };
    let layers = config.text.num_hidden_layers();
    eredu_runtime::ComponentDefinition::new(
        "diffusion_gemma_text",
        super::parallel::parameter_description(config).map_err(Error::backend_source)?,
        (0..layers)
            .map(|index| eredu_runtime::ComponentUnit {
                path: format!("model.decoder.layers.{index}"),
                state: index..index + 1,
            })
            .collect(),
        vec![text_transport()],
        TEXT_EXECUTION_GROUP,
        vec![],
        state,
        text_state_partition_plan(layers),
    )
    .map_err(Error::from)
}

fn text_state_partition_plan(layers: usize) -> eredu_runtime::ArchitectureStatePartitionPlan {
    eredu_runtime::ArchitectureStatePartitionPlan::new([
        eredu_runtime::ArchitectureStatePartitionRule::group_units(0, 0..layers),
    ])
}

/// Derives ownership of the shared routed banks for an admitted rank topology.
pub fn expert_realization_plan(
    config: &FamilyConfig,
    geometry: Option<&super::parallel::LocalGeometry>,
    topology: eredu_core::ParallelRankTopology,
) -> Result<Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>, Error> {
    if let Some(geometry) = geometry {
        geometry
            .validate_for(config)
            .map_err(Error::backend_source)?;
    }
    let Some(count) = config.text.num_experts else {
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
    for (index, policy) in config.text.layer_schedule.iter().enumerate() {
        if policy.feed_forward != crate::gemma4::FeedForwardPolicy::DenseWithSparseMoe {
            continue;
        }
        let args = geometry
            .as_ref()
            .and_then(|g| g.block_args(index))
            .unwrap_or(&config.text);
        let width = if geometry
            .as_ref()
            .is_some_and(|g| g.block_args(index).is_none())
        {
            let global_width = config
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
            &config.text,
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

/// Shared runtime contracts over DiffusionGemma's context and refinement equations.
pub type TextModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    TextKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Shared storage and family-owned context/refinement equations.
#[derive(eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TextKernel<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    config: FamilyConfig,
    statics: TextStatic<B>,
    #[parameter(skip)]
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    #[parameter(skip)]
    geometry: Option<super::parallel::LocalGeometry>,
    #[parameter(skip)]
    owns_input: bool,
    #[parameter(skip)]
    experts: Option<crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>>,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> Clone for TextKernel<B> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            statics: self.statics.clone(),
            definition: std::sync::Arc::clone(&self.definition),
            geometry: self.geometry.clone(),
            experts: self.experts.clone(),
            owns_input: self.owns_input,
        }
    }
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> TextKernel<B> {
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

    /// Materializes one complete text component over the retained declaration.
    pub fn new(
        config: FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextModel<B>, Error> {
        let kernel = Self::materialize(config, None, context)?;
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(
            std::sync::Arc::clone(&kernel.definition),
            kernel,
        )
        .map_err(Error::from)
    }

    /// Materializes a tensor-local stack using the roles retained by its geometry.
    pub fn new_parallel(
        config: FamilyConfig,
        geometry: super::parallel::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextModel<B>, Error> {
        let definition = std::sync::Arc::new(component_definition(&config, Some(&geometry))?);
        let units = geometry.units();
        let mut roles = vec![];
        if geometry.embedding_range().is_some() {
            roles.push("embedding".to_owned());
        }
        if geometry.owns_norm() {
            roles.push("norm".to_owned());
        }
        if geometry.output_range().is_some() {
            roles.push("output".to_owned());
        }
        if geometry.conditioning_width().is_some() {
            roles.push("conditioning".to_owned());
        }
        let ownership = eredu_runtime::PartitionOwnership::new(
            units.start == 0,
            units.end == config.text.num_hidden_layers(),
            roles,
        )
        .map_err(Error::backend_source)?;
        let partition = eredu_runtime::ArchitecturePartition::from_description(
            definition.parameters(),
            [(TEXT_EXECUTION_GROUP, units)],
            ownership,
            definition.state_layout(),
            definition.state_partition_plan(),
            geometry,
            super::boundary::TextBoundarySchema::new(config.text.hidden_size),
        )
        .map_err(Error::backend_source)?;
        Self::materialize_selected(config, &partition, definition, context)
    }

    /// Constructs the exact canonical resources and independent local state selected by a partition.
    pub fn from_partition(
        config: FamilyConfig,
        partition: &eredu_runtime::ArchitecturePartition<
            super::parallel::LocalGeometry,
            super::boundary::TextBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextModel<B>, Error> {
        let definition = std::sync::Arc::new(component_definition(
            &config,
            Some(partition.local_geometry()),
        )?);
        Self::materialize_selected(config, partition, definition, context)
    }

    fn materialize_selected(
        config: FamilyConfig,
        partition: &eredu_runtime::ArchitecturePartition<
            super::parallel::LocalGeometry,
            super::boundary::TextBoundarySchema,
        >,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextModel<B>, Error> {
        let geometry = partition.local_geometry();
        definition.select_partition(partition)?;
        if partition.boundary_schema()
            != &super::boundary::TextBoundarySchema::new(config.text.hidden_size)
        {
            return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                component: definition.id().to_owned(),
                resource: "partition boundary geometry".into(),
            }
            .into());
        }
        let units = geometry.units();
        if partition.groups().len() != 1
            || partition.groups()[0].group_index() != 0
            || partition.groups()[0].global_units() != units
        {
            return Err(Error::backend(
                "diffusion text partition differs from retained local units",
            ));
        }
        let geometry = geometry.select_pinned_resources(
            &config,
            &definition,
            partition.parameter_bindings(),
        )?;
        let kernel = Self {
            statics: TextStatic::new_parallel(&config, &geometry, context)?,
            config,
            geometry: Some(geometry),
            owns_input: partition.ownership().owns_input(),
            definition: std::sync::Arc::clone(&definition),
            experts: None,
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }

    // Composite equations borrow the same kernel directly, under their enclosing component lifecycle.
    pub(super) fn materialize(
        config: FamilyConfig,
        geometry: Option<super::parallel::LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = std::sync::Arc::new(component_definition(&config, geometry.as_ref())?);
        let statics = match &geometry {
            Some(geometry) => TextStatic::new_parallel(&config, geometry, context)?,
            None => TextStatic::new(&config, context)?,
        };
        let owns_input = geometry
            .as_ref()
            .is_none_or(|geometry| geometry.units().start == 0);
        Ok(Self {
            config,
            statics,
            definition,
            geometry,
            owns_input,
            experts: None,
        })
    }

    /// Retains exact shared-bank ownership for native or neutral routed providers.
    pub fn set_expert_realization(
        &mut self,
        plan: crate::ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<(), Error> {
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
        Ok(())
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

    /// Borrows the validated semantic definition retained before parameter construction.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Selected local context slots retained by the family geometry.
    pub fn local_state_layout(&self) -> Result<StateLayout, Error> {
        match &self.geometry {
            Some(geometry) => self
                .definition
                .state_layout()
                .slice(geometry.units())
                .map_err(Error::backend_source),
            None => Ok(self.definition.state_layout().clone()),
        }
    }

    fn position<S>(&self, state: &mut S) -> Result<i32, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
    {
        if self
            .geometry
            .as_ref()
            .is_some_and(|geometry| geometry.units().is_empty())
        {
            if state.optional_layout().is_some() {
                return Err(Error::backend(
                    "stateless image partition retains text cache slots",
                ));
            }
            return Ok(0);
        }
        let expected = self.local_state_layout()?;
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

impl<B> eredu_runtime::ComponentKernel<B> for TextKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Input<'a> = TextInput<'a, B::Tensor>;
    type Unit = TextLayer<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
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
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextLayer<B>, Error> {
        let index = address.index();

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
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a ForwardContext<B::Tensor>,
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, S> eredu_runtime::ComponentEquations<B, S> for TextKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(false, true, true)
            .with_routed_units(true)
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
            self, address, hidden, forward,
        );
        <Self as eredu_runtime::RoutedComponentKernel<B, S>>::execute_routed_observed(
            self,
            address,
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
    fn begin<'a>(
        &mut self,
        input: TextInput<'a, B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        if !self.owns_input {
            return Err(Error::backend(
                "diffusion text receiver requires transported invocation ingress",
            ));
        }
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
    fn begin_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut ForwardContext<B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let index = address.index();
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
    fn finish_observed<O>(
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
    fn finish(
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
}

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for TextKernel<B>
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
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let index = address.index();
        let path = self.definition.unit(address.group(), index)?.path.clone();
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
    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let index = address.index();
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

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for TextKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let pass = <Self as eredu_runtime::RoutedComponentKernel<B, S>>::expert_pass(
            self, address, hidden, forward,
        );
        <Self as eredu_runtime::ParallelRoutedComponentKernel<B, S>>::execute_parallel_routed_observed(
            self, address, unit, hidden, state, forward, pass, &mut ResidentExpertProvider, parallel, context, observer)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: TextInput<'a, B::Tensor>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        self.geometry.as_ref().ok_or_else(|| {
            Error::backend("tensor-parallel diffusion requires retained local geometry")
        })?;
        if !self.owns_input {
            return Err(Error::backend(
                "diffusion text receiver requires transported invocation ingress",
            ));
        }
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
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        <Self as eredu_runtime::ParallelRoutedComponentKernel<B, S>>::execute_parallel_routed(
            self,
            address,
            unit,
            hidden,
            state,
            forward,
            eredu_runtime::ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
        )
    }
    fn finish_parallel_observed<O>(
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
    fn finish_parallel(
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

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for TextKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let index = address.index();
        let path = self.definition.unit(address.group(), index)?.path.clone();
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.run_unit_parallel(
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
    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
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
        let index = address.index();
        self.run_unit_parallel(
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

impl<B> TextKernel<B>
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
                <Self as eredu_runtime::ParallelComponentKernel<B, S>>::begin_parallel(
                    self, input, state, parallel, context,
                )?
            }
            None => <Self as eredu_runtime::ComponentEquations<B, S>>::begin(
                self, input, state, context,
            )?,
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

    pub(super) fn resume_partition<S>(
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
                Some(parallel) => {
                    <Self as eredu_runtime::ParallelComponentKernel<B, S>>::begin_parallel(
                        self,
                        TextInput::Context { tokens, mask },
                        state,
                        parallel,
                        context,
                    )
                }
                None => <Self as eredu_runtime::ComponentEquations<B, S>>::begin(
                    self,
                    TextInput::Context { tokens, mask },
                    state,
                    context,
                ),
            },
            eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                if auxiliary.invocation.shape() != [2] {
                    return Err(Error::backend(
                        "invalid diffusion invocation boundary geometry",
                    ));
                }
                let words = auxiliary.invocation.to_i32_vec(context)?;
                if words.len() != 2 || words[1] != position {
                    return Err(Error::backend("diffusion pipeline context frontier differs from the transported invocation"));
                }
                let phase = match words[0] {
                    0 => TextInvocationPhase::Context,
                    1 => TextInvocationPhase::Refinement,
                    _ => return Err(Error::backend("unknown diffusion invocation phase")),
                };
                if hidden.shape().len() != 3
                    || hidden.dim(0) <= 0
                    || hidden.dim(1) <= 0
                    || hidden.dim(2) != self.config.text.hidden_size
                    || (phase == TextInvocationPhase::Refinement
                        && hidden.dim(1) as u32 > self.config.canvas_length.get())
                {
                    return Err(Error::backend(
                        "invalid diffusion partition activation geometry",
                    ));
                }
                Ok(LayeredForwardState {
                    hidden,
                    context: ForwardContext {
                        phase,
                        position,
                        mask: mask.cloned(),
                        shared: SharedAttentionStates::new(),
                    },
                })
            }
        }
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for TextKernel<B>
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
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error> {
        let expected = selected
            .expected
            .ok_or_else(|| Error::backend("shared stack requires selected state geometry"))?;
        self.resume_partition(
            input,
            mask,
            state,
            expected,
            selected.first_state_ordinal,
            selected.parallel,
            selected.native,
        )
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
                Some(parallel) => {
                    <Self as eredu_runtime::ParallelComponentKernel<B, S>>::finish_parallel(
                        self, hidden, state, forward, parallel, context,
                    )?
                }
                None => <Self as eredu_runtime::ComponentEquations<B, S>>::finish(
                    self, hidden, state, forward, context,
                )?,
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
    TextKernel<B>
{
    #[allow(clippy::too_many_arguments)]
    fn run_unit_parallel<S, P>(
        &mut self,
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

impl<B> eredu_runtime::RefinementComponentKernel<B> for TextKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
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

impl<B, S> eredu_runtime::PipelineComponentKernel<B, S> for TextKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
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
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, super::boundary::TextBoundary<B::Tensor>)>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let expected = selected.expected.ok_or_else(|| {
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
                self.prepare_partition_input(input, state, selected.parallel, selected.native)?
                    .input
            }
        };
        self.begin_partition_observed(
            prepared,
            mask.as_ref(),
            state,
            eredu_runtime::ComponentPartitionContext {
                expected: Some(expected),
                first_state_ordinal: selected.first_state_ordinal,
                parallel: selected.parallel,
                native: selected.native,
            },
            observer,
        )
    }
    fn output_width(&self) -> i32 {
        self.config.text.vocab_size
    }
    fn output_shape(
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
    fn routed_tensor_reductions(
        &self,
        _address: eredu_runtime::ExecutionUnitAddress,
        routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Error> {
        // Attention and the always-present dense branch reduce before routing;
        // the selected-expert write reduces afterward, before normalization.
        Ok(eredu_runtime::RoutedTensorReductions::hidden(
            if routed { 2 } else { 1 },
            1,
        ))
    }
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
}

impl<B> eredu_runtime::TextComponentKernel<B> for TextKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
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
        Some(<Self as eredu_runtime::RefinementComponentKernel<B>>::refinement_geometry(self))
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

impl<B> eredu_runtime::ComponentPinnedResources<B> for TextKernel<B>
where
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Modules = TextStatic<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.statics
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.statics
    }
    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
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
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
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
