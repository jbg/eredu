//! Shared invocation transport across independently owned text-layer partitions.
use super::{
    model::{TextInput, TextLayer},
    FamilyConfig,
};
use crate::decoder::{
    self, ComponentInstrumentation, Config, PartitionStaticModules, PartitionedLayeredModel,
};
use eredu_core::decoding::{AttentionInvocation, CacheWritePolicy};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Error, NeuralBackend, ReadOnlyAttentionCache, Tensor,
};
use eredu_runtime::{
    ArchitectureParameters, LayerRuntimeState, LayeredArchitecture, LayeredForwardState,
    LayeredPartitionInput, LayeredPartitionOutput, NoAuxiliaryBoundary,
    ParallelLayeredArchitecture, PartitionedLayeredArchitecture, StateLayout,
};

/// Immutable mode and original prefix position carried with the complete canvas.
#[derive(Debug, Clone)]
pub struct TextBoundary<T> {
    /// Three I32 values: mode code, prefix position and activation precision.
    pub invocation: T,
}
/// Checked mode/frontier wire geometry over full residual activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextBoundarySchema {
    hidden: i32,
}
impl TextBoundarySchema {
    /// Declares the global residual width.
    pub fn new(hidden: i32) -> Self {
        Self { hidden }
    }
}
impl eredu_runtime::ArchitectureBoundary for TextBoundarySchema {
    type Boundary<T> = TextBoundary<T>;
    const IDENTITY: &'static str = "nemotron_labs_diffusion.shared_text";
    fn primary_tensor_spec(&self) -> eredu_runtime::BoundaryTensorSpec {
        eredu_runtime::BoundaryTensorSpec::primary_activation(self.hidden)
    }
    fn auxiliary_tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        vec![eredu_runtime::BoundaryTensorSpec::new(
            "invocation",
            [eredu_runtime::BoundaryTensorDimension::Fixed(3)],
            eredu_runtime::BoundaryTensorDtype::Int32,
        )]
    }
    fn encode<T>(
        &self,
        boundary: TextBoundary<T>,
    ) -> Result<
        Vec<eredu_runtime::ArchitectureBoundaryValue<T>>,
        eredu_runtime::ArchitectureBoundaryError,
    > {
        Ok(vec![eredu_runtime::ArchitectureBoundaryValue::new(
            "invocation",
            boundary.invocation,
        )?])
    }
    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<TextBoundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        Ok(TextBoundary {
            invocation: tensors
                .into_iter()
                .next()
                .expect("validated boundary cardinality"),
        })
    }
}
fn mode_code(invocation: AttentionInvocation) -> Result<i32, Error> {
    match invocation {
        AttentionInvocation::CAUSAL => Ok(0),
        AttentionInvocation::DIFFUSION => Ok(1),
        AttentionInvocation::DRAFT => Ok(2),
        _ => Err(Error::backend("invalid shared text invocation")),
    }
}
fn decode<T: Tensor>(
    boundary: TextBoundary<T>,
    context: &T::Context,
) -> Result<(AttentionInvocation, i32, eredu_nn::TensorElementType), Error> {
    if boundary.invocation.shape() != [3]
        || boundary.invocation.element_type() != Some(eredu_nn::TensorElementType::I32)
    {
        return Err(Error::backend("invalid shared text invocation wire"));
    }
    let values = boundary.invocation.to_i32_vec(context)?;
    if values.len() != 3 || values[1] < 0 {
        return Err(Error::backend("invalid shared text prefix frontier"));
    }
    let invocation = match values[0] {
        0 => AttentionInvocation::CAUSAL,
        1 => AttentionInvocation::DIFFUSION,
        2 => AttentionInvocation::DRAFT,
        _ => return Err(Error::backend("invalid shared text mode code")),
    };
    Ok((invocation, values[1], precision_from_code(values[2])?))
}

pub(crate) fn precision_code<T: Tensor>(tensor: &T) -> Result<i32, Error> {
    match tensor.element_type() {
        Some(eredu_nn::TensorElementType::F32) => Ok(0),
        Some(eredu_nn::TensorElementType::F16) => Ok(1),
        Some(eredu_nn::TensorElementType::Bf16) => Ok(2),
        _ => Err(Error::backend(
            "unsupported shared-backbone activation precision",
        )),
    }
}
pub(crate) fn precision_from_code(code: i32) -> Result<eredu_nn::TensorElementType, Error> {
    match code {
        0 => Ok(eredu_nn::TensorElementType::F32),
        1 => Ok(eredu_nn::TensorElementType::F16),
        2 => Ok(eredu_nn::TensorElementType::Bf16),
        _ => Err(Error::backend(
            "invalid shared-backbone activation precision",
        )),
    }
}
/// Invocation values retained separately from shared partition parameters.
pub struct PartitionContext<T> {
    pub(crate) mask: Option<T>,
    pub(crate) invocation: AttentionInvocation,
    pub(crate) position: i32,
}
/// Prepared first-owner embeddings plus complete immutable invocation metadata.
pub struct PreparedPartitionInvocation<T: 'static> {
    /// Owned residual ingress, ready for the ordinary partition driver.
    pub input: LayeredPartitionInput<'static, T, TextBoundary<T>>,
    /// Request visibility used unchanged by every owner.
    pub mask: Option<T>,
}
/// One retained neutral text partition, with no unowned embedding or readout storage.
pub struct PartitionedTextModel<B: NeuralBackend> {
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    base: PartitionedLayeredModel<B, FamilyConfig>,
}
impl<B: DistributedNeuralBackend> eredu_nn::Parameterized<B::Tensor> for PartitionedTextModel<B> {
    fn set_trainable(&mut self, trainable: bool) {
        self.base
            .kernel_mut()
            .static_modules_mut()
            .set_trainable(trainable);
    }
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, B::Tensor>>(
        &'a self,
        visitor: &mut V,
    ) {
        self.base
            .kernel()
            .static_modules()
            .visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, B::Tensor>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.base
            .kernel_mut()
            .static_modules_mut()
            .visit_parameters_mut(visitor);
    }
}

impl<B: DistributedNeuralBackend> PartitionedTextModel<B> {
    /// Retained semantic declaration used by execution and inspection.
    pub fn definition(&self) -> &eredu_runtime::ComponentDefinition {
        &self.definition
    }

    /// Builds only statics and units authorized by the selected neutral partition.
    pub fn new(
        config: FamilyConfig,
        parameters: &eredu_runtime::ArchitectureParameterDescription,
        partition: &eredu_runtime::ArchitecturePartition<
            decoder::PartitionLocalGeometry<FamilyConfig>,
            TextBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let definition = std::sync::Arc::new(super::definition::text_partition(
            &config,
            parameters,
            partition.local_geometry(),
        )?);
        definition.select_partition(partition)?;
        if partition.boundary_schema() != &TextBoundarySchema::new(config.hidden_size()) {
            return Err(eredu_runtime::ComponentDefinitionError::Invalid {
                component: definition.id().to_owned(),
                resource: "partition boundary geometry".into(),
            }
            .into());
        }
        Ok(Self {
            definition,
            base: decoder::DecoderKernel::from_partition(config, parameters, partition, context)?,
        })
    }
    /// Complete family equations, including optional draft adapter topology.
    pub fn config(&self) -> &FamilyConfig {
        self.base.kernel().args()
    }
    /// Constructs exactly one locally owned layer and its optional low-rank matrices.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TextLayer<B>, Error> {
        let base = self.base.kernel().construct_unit(index, context)?;
        let draft_adapter = self
            .config()
            .draft_adapter()
            .map(|a| {
                a.layer(
                    index,
                    base.self_attention.query_heads * self.config().head_dim(),
                    self.config().hidden_size(),
                    context,
                )
            })
            .transpose()?;
        Ok(TextLayer {
            base,
            draft_adapter,
        })
    }
    fn prepare_mask<S: LayerRuntimeState<B>>(
        &self,
        mask: Option<&B::Tensor>,
        length: i32,
        invocation: AttentionInvocation,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Error>
    where
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if let Some(mask) = mask {
            return Ok(Some(mask.clone()));
        }
        if invocation.cache_write == CacheWritePolicy::Transient {
            let frontier = state.layer(0).map_err(Error::backend)?.offset();
            return Ok(Some(B::Tensor::full_f32(
                0.0,
                &[
                    length,
                    frontier
                        .checked_add(length)
                        .ok_or_else(|| Error::backend("partition canvas visibility overflow"))?,
                ],
                context,
            )?));
        }
        Ok(None)
    }
    #[allow(clippy::too_many_arguments)]
    fn begin<S, O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, PartitionContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let frontier = state.layer(0).map_err(Error::backend)?.offset();
        let (input, invocation, position, length) = match input {
            LayeredPartitionInput::Tokens(tokens) => (
                LayeredPartitionInput::Tokens(tokens),
                AttentionInvocation::CAUSAL,
                frontier,
                tokens.dim(1),
            ),
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let (invocation, position, element) = decode(auxiliary, context)?;
                let hidden = hidden.cast_float(element, context)?;
                if position != frontier {
                    return Err(Error::backend(
                        "pipeline owners disagree on canonical prefix frontier",
                    ));
                }
                let length = hidden.dim(1);
                (
                    LayeredPartitionInput::Hidden {
                        hidden,
                        auxiliary: NoAuxiliaryBoundary,
                    },
                    invocation,
                    position,
                    length,
                )
            }
        };
        let mask = self.prepare_mask(mask, length, invocation, state, context)?;
        let forward=<PartitionedLayeredModel<B,FamilyConfig> as PartitionedLayeredArchitecture<B,S>>::begin_partition_observed(&mut self.base,input,mask.as_ref(),state,expected,first,parallel,context,observer)?;
        Ok(LayeredForwardState {
            hidden: forward.hidden,
            context: PartitionContext {
                mask: forward.context.visibility_mask().cloned(),
                invocation,
                position,
            },
        })
    }
    /// Embeds the first-owner token span once and transports its request semantics.
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
        let expected = state.layout().clone();
        let forward = self.begin_input(
            input,
            None,
            state,
            &expected,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )?;
        let boundary = TextBoundary {
            invocation: B::Tensor::from_i32_slice(
                &[
                    mode_code(forward.context.invocation)?,
                    forward.context.position,
                    precision_code(&forward.hidden)?,
                ],
                &[3],
                context,
            )?,
        };
        Ok(PreparedPartitionInvocation {
            input: LayeredPartitionInput::Hidden {
                hidden: forward.hidden,
                auxiliary: boundary,
            },
            mask: forward.context.mask,
        })
    }
    fn begin_input<S, O>(
        &mut self,
        input: TextInput<'_, B::Tensor>,
        incoming: Option<(B::Tensor, TextBoundary<B::Tensor>)>,
        state: &mut S,
        expected: &StateLayout,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, PartitionContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        mode_code(input.invocation)?;
        if input.feedback.is_some()
            || input.tokens.shape().len() != 2
            || input.tokens.dim(0) <= 0
            || input.tokens.dim(1) <= 0
        {
            return Err(Error::backend("invalid shared text partition input"));
        }
        if let Some((hidden, boundary)) = incoming {
            if hidden.shape()
                != [
                    input.tokens.dim(0),
                    input.tokens.dim(1),
                    self.config().hidden_size(),
                ]
            {
                return Err(Error::backend(
                    "pipeline activation does not carry the complete requested token span",
                ));
            }
            let (actual, position, _) = decode(boundary.clone(), context)?;
            if actual != input.invocation
                || position != state.layer(0).map_err(Error::backend)?.offset()
            {
                return Err(Error::backend(
                    "pipeline owners disagree on invocation mode or causal frontier",
                ));
            }
            let mut forward = self.begin(
                LayeredPartitionInput::Hidden {
                    hidden,
                    auxiliary: boundary,
                },
                input.mask,
                state,
                expected,
                0,
                parallel,
                context,
                observer,
            )?;
            if forward.context.invocation != input.invocation {
                return Err(Error::backend(
                    "pipeline owners disagree on requested attention mode",
                ));
            }
            forward.context.invocation = input.invocation;
            return Ok(forward);
        }
        let position = state.layer(0).map_err(Error::backend)?.offset();
        let mask = self.prepare_mask(
            input.mask,
            input.tokens.dim(1),
            input.invocation,
            state,
            context,
        )?;
        let forward=<PartitionedLayeredModel<B,FamilyConfig> as PartitionedLayeredArchitecture<B,S>>::begin_partition_observed(&mut self.base,LayeredPartitionInput::Tokens(input.tokens),mask.as_ref(),state,expected,0,parallel,context,observer)?;
        Ok(LayeredForwardState {
            hidden: forward.hidden,
            context: PartitionContext {
                mask: forward.context.visibility_mask().cloned(),
                invocation: input.invocation,
                position,
            },
        })
    }
}

impl<B: DistributedNeuralBackend> ArchitectureParameters<B> for PartitionedTextModel<B> {
    type DefinitionError = Error;
    fn state_layout(&self) -> Result<StateLayout, Error> {
        Ok(self.definition.state_layout().clone())
    }
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        self.base.state_identity(state, topology)
    }
    fn parameter_description(
        &self,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::ArchitectureParameterDescription, Error> {
        Ok(self.definition.parameters().clone())
    }
    fn static_parameter_recipes(
        &self,
        source: &dyn eredu_checkpoint::store::CheckpointSource,
    ) -> Result<
        std::collections::BTreeMap<String, eredu_checkpoint::recipe::DerivedWeightRecipe>,
        String,
    > {
        super::checkpoint::module_source_recipes(self.base.kernel().static_modules(), source)
    }
    fn visit_static_parameters<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.base.visit_static_parameters(visitor)
    }
    fn visit_static_parameters_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.base.visit_static_parameters_mut(visitor)
    }
}
impl<B, S> LayeredArchitecture<B, S> for PartitionedTextModel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Input<'a> = TextInput<'a, B::Tensor>;
    type StaticModules = PartitionStaticModules<B>;
    type Unit = TextLayer<B>;
    type ForwardContext = PartitionContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        self.definition
            .group_transport(group)
            .expect("validated execution group")
            .clone()
    }
    fn primary_execution_group(&self) -> &str {
        self.definition.primary_group()
    }
    fn state_partition_plan(
        &self,
        _layout: &StateLayout,
    ) -> eredu_runtime::ArchitectureStatePartitionPlan {
        self.definition.state_partition_plan().clone()
    }
    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        Ok(self.definition.parameters().graph().clone())
    }
    fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
        self.definition.unit_count(group).map_err(Error::from)
    }
    fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        Ok(self.definition.unit(group, index)?.path.clone())
    }
    fn static_modules(&self) -> &Self::StaticModules {
        self.base.kernel().static_modules()
    }
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        <PartitionedLayeredModel<B, FamilyConfig> as LayeredArchitecture<B, S>>::static_modules_mut(
            &mut self.base,
        )
    }
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        if group != 0 {
            return Err(Error::backend("invalid text partition execution group"));
        }
        self.construct_unit(index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = state.layout().clone();
        self.begin_input(
            input,
            None,
            state,
            &expected,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let expected = state.layout().clone();
        self.begin_input(input, None, state, &expected, None, context, observer)
    }
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _: &mut S,
        _: &mut Self::ForwardContext,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend("invalid text partition dependencies"));
        }
        Ok(initial.clone())
    }
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_unit_observed(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.run_unit(
            group, index, unit, hidden, state, forward, None, context, observer,
        )
    }
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.base.kernel_mut().finish_local(hidden, None, context)
    }
    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.base.kernel_mut().finish_local_instrumented(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: usize,
        _: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}
impl<B: DistributedNeuralBackend> PartitionedTextModel<B> {
    #[allow(clippy::too_many_arguments)]
    fn run_unit<S, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut TextLayer<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut PartitionContext<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let range = self.base.kernel().local_geometry().owned_units();
        if group != 0 || !range.contains(&index) {
            return Err(Error::backend(
                "shared text partition attempted an unowned unit",
            ));
        }
        let cache = state.layer(index - range.start).map_err(Error::backend)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward(
            hidden,
            forward.mask.as_ref(),
            forward.invocation,
            cache,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&format!("model.layers.{index}"), &mut observer),
        )
    }
}
impl<B, S> ParallelLayeredArchitecture<B, S> for PartitionedTextModel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as LayeredArchitecture<B, S>>::observation_hooks(self)
    }
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = state.layout().clone();
        self.begin_input(
            input,
            None,
            state,
            &expected,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_forward_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let expected = state.layout().clone();
        self.begin_input(
            input,
            None,
            state,
            &expected,
            Some(parallel),
            context,
            observer,
        )
    }
    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.run_unit(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn forward_unit_parallel_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.run_unit(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            Some(parallel),
            context,
            observer,
        )
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.base
            .kernel_mut()
            .finish_local(hidden, Some(parallel), context)
    }
    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        _: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.base.kernel_mut().finish_local_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}
impl<B, S> PartitionedLayeredArchitecture<B, S> for PartitionedTextModel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    type Boundary = TextBoundarySchema;
    fn partition_observation_hooks(
        &self,
        _: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        <Self as LayeredArchitecture<B, S>>::observation_hooks(self)
    }
    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        Ok(TextBoundarySchema::new(self.config().hidden_size()))
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin(
            input,
            mask,
            state,
            expected,
            first,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.begin(
            input,
            mask,
            state,
            expected,
            first,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, TextBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.begin(
            input, mask, state, expected, first, parallel, context, observer,
        )
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error> {
        self.finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredPartitionOutput<B::Tensor, TextBoundary<B::Tensor>>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        if !owns_output {
            return Ok(LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: TextBoundary {
                    invocation: B::Tensor::from_i32_slice(
                        &[
                            mode_code(forward.invocation)?,
                            forward.position,
                            precision_code(hidden)?,
                        ],
                        &[3],
                        context,
                    )?,
                },
            });
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        Ok(LayeredPartitionOutput::Final {
            output: self.base.kernel_mut().finish_local_instrumented(
                hidden,
                parallel,
                context,
                &mut ComponentInstrumentation::new("readout", &mut observer),
            )?,
            retained: None,
        })
    }
}
impl<B, S> eredu_runtime::PipelinePartitionArchitecture<B, S> for PartitionedTextModel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn partition_output_width(&self) -> i32 {
        self.config().vocabulary_size()
    }
    fn partition_input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Error> {
        if input.tokens.shape().len() != 2 || input.tokens.dim(0) <= 0 || input.tokens.dim(1) <= 0 {
            return Err(Error::backend("invalid shared text pipeline token shape"));
        }
        Ok((input.tokens.dim(0), input.tokens.dim(1)))
    }
    fn begin_pipeline_partition<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, TextBoundary<B::Tensor>)>,
        state: &mut S,
        expected: Option<&StateLayout>,
        _: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.begin_input(
            input,
            incoming,
            state,
            expected.ok_or_else(|| {
                Error::backend("shared text pipeline requires selected state geometry")
            })?,
            parallel,
            context,
            observer,
        )
    }
}

impl<B, S> eredu_runtime::ReplicatedTextArchitecture<B, S> for PartitionedTextModel<B>
where
    B: DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: ReadOnlyAttentionCache<B::Tensor>,
{
    fn supports_chunked_prefill() -> bool {
        true
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        TextInput {
            tokens,
            mask,
            invocation: AttentionInvocation::CAUSAL,
            feedback: None,
        }
    }
    fn invocation_input<'a>(
        tokens: &'a B::Tensor,
        feedback: Option<&'a B::Tensor>,
        mask: Option<&'a B::Tensor>,
        invocation: AttentionInvocation,
    ) -> Option<Self::Input<'a>> {
        <super::model::TextModel<B> as eredu_runtime::ReplicatedTextArchitecture<B, S>>::invocation_input(tokens, feedback, mask, invocation)
    }
    fn block_refinement_geometry(
        &self,
    ) -> Option<eredu_runtime::generation::refinement::architecture::BlockRefinementGeometry> {
        Some(self.config().generation_geometry())
    }
}
