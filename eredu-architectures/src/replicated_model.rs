//! Dense-only layered models for architecture-owned replicated text dispatch.

use std::{marker::PhantomData, sync::Arc};

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AttentionCache, AttentionRequest, BlockwiseAttentionBackend, CompressedAttentionBlock,
    CompressedAttentionCache, CompressedAttentionScan, CompressedAttentionState,
    CompressedAttentionView, Error, NeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    ComponentAdapter, ComponentDefinition, ComponentEquations, ComponentKernel,
    ComponentPinnedResources, ExecutionUnitAddress, KernelParameterModules, LayerRuntimeState,
    LayeredForwardState, ModelStateIdentity, RuntimeLayerState, RuntimeStateComponents, StateError,
    StateLayout, StaticParameterVisitor, StaticParameterVisitorMut, TextComponentKernel,
};

use crate::decoder::{DecoderStaticResources, LayeredInput, StaticModuleSpec, StaticModules};

pub(crate) struct ReplicatedForwardContext<T> {
    pub(crate) mask: Option<T>,
    prediction_capture: Option<T>,
}

pub(crate) trait FixedReplicatedFamily<B: NeuralBackend>: 'static {
    /// Validated equivalence of consecutive causal prefix passes.
    const CHUNKED_PREFILL: bool = false;
    type Config: Clone + 'static;
    type Unit: Parameterized<B::Tensor>;

    fn validate(config: &Self::Config) -> Result<(), Error>;
    fn definition(config: &Self::Config) -> Result<ComponentDefinition, Error>;
    fn static_spec(config: &Self::Config) -> StaticModuleSpec;
    fn state_identity(
        config: &Self::Config,
        layout: &StateLayout,
        global_layer_start: usize,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error>;
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error>;
    fn mask_layer(config: &Self::Config) -> Option<usize>;
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>;
    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let _ = (path, observer);
        Self::forward_unit(unit, hidden, state, forward, context)
    }
}

pub(crate) trait CompressedReplicatedFamily<B: BlockwiseAttentionBackend>: 'static {
    type Config: Clone + 'static;
    type Unit: Parameterized<B::Tensor>;

    fn validate(config: &Self::Config) -> Result<(), Error>;
    fn definition(config: &Self::Config) -> Result<ComponentDefinition, Error>;
    fn static_spec(config: &Self::Config) -> StaticModuleSpec;
    fn state_identity(
        config: &Self::Config,
        layout: &StateLayout,
        global_layer_start: usize,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error>;
    fn build_unit(
        config: &Self::Config,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error>;
    fn mask_layer(config: &Self::Config) -> Option<usize>;
    fn forward_unit<C>(
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>;

    fn forward_unit_observed<C, O>(
        unit: &mut Self::Unit,
        path: &str,
        hidden: &B::Tensor,
        state: &mut C,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let _ = (path, observer);
        Self::forward_unit(unit, hidden, state, forward, context)
    }
}

fn select_definition(
    definition: Arc<ComponentDefinition>,
    expected: ComponentDefinition,
) -> Result<Arc<ComponentDefinition>, Error> {
    let invalid = |resource: &str| eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    };
    if definition.id() != expected.id() || definition.state_layout() != expected.state_layout() {
        return Err(invalid("replicated component identity or state geometry").into());
    }
    let units = (0..expected.parameters().unit_layout().len())
        .map(|ordinal| {
            let address = expected
                .parameters()
                .unit_layout()
                .address(ordinal)
                .expect("declared invocation");
            expected.unit(address.group(), address.index()).cloned()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transport = (0..expected.parameters().graph().groups().len())
        .map(|group| expected.group_transport(group).cloned())
        .collect::<Result<Vec<_>, _>>()?;
    definition.validate_execution_contract(
        expected.primary_group(),
        expected.prediction_groups(),
        &units,
        &transport,
        expected.state_partition_plan(),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    let mut parameters = expected.parameters().clone();
    if let Some((rank, layout)) = definition.parameters().partition_layout() {
        let incomplete = |layout: &eredu_runtime::LocalModelLayout| {
            layout.tensors().find_map(|(target, tensor)| {
                let complete = |placement: &eredu_runtime::TensorPlacement| {
                    use eredu_runtime::TensorPlacement;
                    match placement {
                        TensorPlacement::Replicated | TensorPlacement::Local => true,
                        TensorPlacement::Rank { rank: owner } => *owner == rank.global_rank(),
                        TensorPlacement::Shard { index, parts, .. } => *index == 0 && *parts == 1,
                        TensorPlacement::Range { axis, start, end } => {
                            *start == 0 && tensor.global_shape().get(*axis) == Some(end)
                        }
                        TensorPlacement::Indices { axis, indices } => tensor
                            .global_shape()
                            .get(*axis)
                            .is_some_and(|width| indices.iter().copied().eq(0..*width)),
                        TensorPlacement::Omit => false,
                    }
                };
                (tensor.local_shape() != tensor.global_shape()
                    || !complete(tensor.placement())
                    || !tensor.additional_placements().iter().all(complete))
                .then(|| target.to_owned())
            })
        };
        if let Some(target) = incomplete(layout) {
            return Err(invalid(&format!(
                "replicated parameter {target} requires complete geometry"
            ))
            .into());
        }
        let selected = eredu_runtime::derive_partitioned_local_layout(&parameters, rank)
            .map_err(Error::backend)?;
        if let Some(target) = incomplete(&selected) {
            return Err(invalid(&format!(
                "replicated selected parameter {target} requires complete geometry"
            ))
            .into());
        }
        parameters = parameters
            .with_partition_layout(rank, selected)
            .map_err(Error::backend_source)?;
    }
    definition
        .project_parameters(parameters)
        .map_err(Error::from)
}

pub(crate) struct MixedState;
pub(crate) struct AttentionState;
pub(crate) struct FixedState;
pub(crate) struct Stateless;
pub(crate) struct MixedCompressedState;
pub(crate) struct CompressedState;

struct AttentionLayer<'a, B: NeuralBackend, C> {
    inner: &'a mut C,
    backend: PhantomData<B>,
}

impl<B: NeuralBackend, C: RuntimeLayerState<B>> RuntimeLayerState<B> for AttentionLayer<'_, B, C> {
    type RetainedValues<'a>
        = C::RetainedValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.inner.retained_values()
    }
}

impl<B: NeuralBackend, C: AttentionCache<B::Tensor> + RuntimeLayerState<B>>
    AttentionCache<B::Tensor> for AttentionLayer<'_, B, C>
{
    fn offset(&self) -> i32 {
        self.inner.offset()
    }
    fn max_size(&self) -> Option<i32> {
        self.inner.max_size()
    }
    fn update_for_attention(
        &mut self,
        keys: B::Tensor,
        values: B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        self.inner.update_for_attention(keys, values, context)
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.inner.attention(request, context)
    }
    fn uses_blockwise_attention(&self) -> bool {
        self.inner.uses_blockwise_attention()
    }
    fn relative_attention<N: NeuralBackend<Tensor = B::Tensor>>(
        &mut self,
        request: eredu_nn::RelativeAttentionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.inner.relative_attention::<N>(request, context)
    }
}

impl<B: NeuralBackend, C: AttentionCache<B::Tensor> + RuntimeLayerState<B>>
    RuntimeStateComponents<B> for AttentionLayer<'_, B, C>
{
    fn position(&self) -> i32 {
        self.inner.offset()
    }
    fn fixed_component(
        &mut self,
        role: eredu_core::cache::StateTensorRole,
    ) -> Result<&mut Option<B::Tensor>, StateError> {
        Err(StateError::UnknownComponent { role })
    }
    fn replace_fixed_components(
        &mut self,
        values: Vec<(eredu_core::cache::StateTensorRole, Option<B::Tensor>)>,
    ) -> Result<(), StateError> {
        eredu_runtime::state::replace_fixed_components(std::iter::empty(), values)
    }
    fn advance_fixed(&mut self, _tokens: i32) -> Result<(), StateError> {
        Err(StateError::InvalidAdvance(
            "attention-only state has no fixed frontier".into(),
        ))
    }
}

struct FixedLayer<'a, B: NeuralBackend, C> {
    inner: &'a mut C,
    backend: PhantomData<B>,
}

impl<B: NeuralBackend, C: RuntimeStateComponents<B>> RuntimeLayerState<B> for FixedLayer<'_, B, C> {
    type RetainedValues<'a>
        = C::RetainedValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.inner.retained_values()
    }
}

impl<B: NeuralBackend, C: RuntimeStateComponents<B>> RuntimeStateComponents<B>
    for FixedLayer<'_, B, C>
{
    fn position(&self) -> i32 {
        self.inner.position()
    }
    fn fixed_component(
        &mut self,
        role: eredu_core::cache::StateTensorRole,
    ) -> Result<&mut Option<B::Tensor>, StateError> {
        self.inner.fixed_component(role)
    }
    fn replace_fixed_components(
        &mut self,
        values: Vec<(eredu_core::cache::StateTensorRole, Option<B::Tensor>)>,
    ) -> Result<(), StateError> {
        self.inner.replace_fixed_components(values)
    }
    fn advance_fixed(&mut self, tokens: i32) -> Result<(), StateError> {
        self.inner.advance_fixed(tokens)
    }
}

impl<B: NeuralBackend, C: RuntimeStateComponents<B>> AttentionCache<B::Tensor>
    for FixedLayer<'_, B, C>
{
    fn offset(&self) -> i32 {
        self.inner.position()
    }
    fn max_size(&self) -> Option<i32> {
        None
    }
    fn update_for_attention(
        &mut self,
        _keys: B::Tensor,
        _values: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        Err(Error::backend("fixed-only state cannot update attention"))
    }
    fn attention(
        &mut self,
        _request: AttentionRequest<'_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Err(Error::backend("fixed-only state cannot execute attention"))
    }
}

struct StatelessLayer<'a, B: NeuralBackend, C> {
    inner: &'a mut C,
    backend: PhantomData<B>,
}

impl<B: NeuralBackend, C: RuntimeLayerState<B>> RuntimeLayerState<B> for StatelessLayer<'_, B, C> {
    type RetainedValues<'a>
        = C::RetainedValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;

    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.inner.retained_values()
    }
}

impl<B: NeuralBackend, C: RuntimeLayerState<B>> RuntimeStateComponents<B>
    for StatelessLayer<'_, B, C>
{
    fn position(&self) -> i32 {
        0
    }
    fn fixed_component(
        &mut self,
        role: eredu_core::cache::StateTensorRole,
    ) -> Result<&mut Option<B::Tensor>, StateError> {
        Err(StateError::UnknownComponent { role })
    }
    fn replace_fixed_components(
        &mut self,
        values: Vec<(eredu_core::cache::StateTensorRole, Option<B::Tensor>)>,
    ) -> Result<(), StateError> {
        eredu_runtime::state::replace_fixed_components(std::iter::empty(), values)
    }
    fn advance_fixed(&mut self, _tokens: i32) -> Result<(), StateError> {
        Err(StateError::InvalidAdvance(
            "stateless layer has no frontier".into(),
        ))
    }
}

impl<B: NeuralBackend, C: RuntimeLayerState<B>> AttentionCache<B::Tensor>
    for StatelessLayer<'_, B, C>
{
    fn offset(&self) -> i32 {
        0
    }
    fn max_size(&self) -> Option<i32> {
        None
    }
    fn update_for_attention(
        &mut self,
        _keys: B::Tensor,
        _values: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor), Error> {
        Err(Error::backend("stateless layer cannot update attention"))
    }
    fn attention(
        &mut self,
        _request: AttentionRequest<'_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Err(Error::backend("stateless layer cannot execute attention"))
    }
}

struct CompressedLayer<'a, B: NeuralBackend, C> {
    inner: &'a mut C,
    backend: PhantomData<B>,
}

impl<B: NeuralBackend, C> std::fmt::Debug for CompressedLayer<'_, B, C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CompressedLayer")
    }
}

impl<B: NeuralBackend, C: RuntimeLayerState<B>> RuntimeLayerState<B> for CompressedLayer<'_, B, C> {
    type RetainedValues<'a>
        = C::RetainedValues<'a>
    where
        Self: 'a,
        B::Tensor: 'a;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        self.inner.retained_values()
    }
}

impl<B: NeuralBackend, C: CompressedAttentionCache<B::Tensor> + RuntimeLayerState<B>>
    RuntimeStateComponents<B> for CompressedLayer<'_, B, C>
{
    fn position(&self) -> i32 {
        self.inner.offset()
    }
    fn fixed_component(
        &mut self,
        role: eredu_core::cache::StateTensorRole,
    ) -> Result<&mut Option<B::Tensor>, StateError> {
        Err(StateError::UnknownComponent { role })
    }
    fn replace_fixed_components(
        &mut self,
        values: Vec<(eredu_core::cache::StateTensorRole, Option<B::Tensor>)>,
    ) -> Result<(), StateError> {
        eredu_runtime::state::replace_fixed_components(std::iter::empty(), values)
    }
    fn advance_fixed(&mut self, _tokens: i32) -> Result<(), StateError> {
        Err(StateError::InvalidAdvance(
            "compressed-only state has no fixed frontier".into(),
        ))
    }
}

impl<B: NeuralBackend, C: CompressedAttentionCache<B::Tensor> + RuntimeLayerState<B>>
    CompressedAttentionCache<B::Tensor> for CompressedLayer<'_, B, C>
{
    type Checkpoint = C::Checkpoint;
    fn offset(&self) -> i32 {
        self.inner.offset()
    }
    fn is_paged(&self) -> bool {
        self.inner.is_paged()
    }
    fn append(
        &mut self,
        state: CompressedAttentionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<CompressedAttentionView<B::Tensor>, Error> {
        self.inner.append(state, context)
    }
    fn visit_blocks<F>(
        &mut self,
        query_tokens: i32,
        context: &<B::Tensor as Tensor>::Context,
        visitor: F,
    ) -> Result<CompressedAttentionScan, Error>
    where
        F: FnMut(CompressedAttentionBlock<B::Tensor>) -> Result<u64, Error>,
    {
        self.inner.visit_blocks(query_tokens, context, visitor)
    }
    fn checkpoint(&self) -> Self::Checkpoint {
        self.inner.checkpoint()
    }
    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        self.inner.restore(checkpoint, context)
    }
    fn finalize(&mut self) -> Result<(), Error> {
        self.inner.finalize()
    }
    fn clear(&mut self) -> Result<(), Error> {
        self.inner.clear()
    }
}

pub(crate) type FixedReplicatedModel<B, F, P = MixedState> = ComponentAdapter<
    Arc<ComponentDefinition>,
    FixedReplicatedKernel<B, F, P>,
    KernelParameterModules,
>;
pub(crate) struct FixedReplicatedKernel<
    B: NeuralBackend,
    F: FixedReplicatedFamily<B>,
    P = MixedState,
> {
    config: F::Config,
    definition: Arc<ComponentDefinition>,
    static_modules: StaticModules<B>,
    prediction_capture: bool,
    family: PhantomData<(F, P)>,
}
impl<B: NeuralBackend, F: FixedReplicatedFamily<B>, P: 'static> FixedReplicatedKernel<B, F, P> {
    pub fn from_definition(
        config: F::Config,
        definition: Arc<ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        F::validate(&config)?;
        let definition = select_definition(definition, F::definition(&config)?)?;
        let static_modules = StaticModules::from_spec(F::static_spec(&config), context)?;
        Ok(Self {
            config,
            definition,
            static_modules,
            prediction_capture: false,
            family: PhantomData,
        })
    }
    pub fn into_model(self) -> Result<FixedReplicatedModel<B, F, P>, Error> {
        ComponentAdapter::with_pinned_resources::<B>(self.definition.clone(), self)
            .map_err(Error::from)
    }
    pub(crate) fn enable_prediction_capture(&mut self) {
        self.prediction_capture = true;
    }
}
impl<B: NeuralBackend, F: FixedReplicatedFamily<B>, P: 'static> ComponentKernel<B>
    for FixedReplicatedKernel<B, F, P>
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error> {
        F::state_identity(
            &self.config,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<F::Unit, Error> {
        F::build_unit(&self.config, address.index(), context)
    }
    fn executes_group(&self, _group: usize, _forward: &Self::ForwardContext) -> bool {
        true
    }
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.prediction_capture.as_ref()
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter().chain(forward.prediction_capture.iter())
    }
}
impl<B: NeuralBackend, F: FixedReplicatedFamily<B>, P: 'static> ComponentPinnedResources<B>
    for FixedReplicatedKernel<B, F, P>
{
    type Modules = StaticModules<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V: StaticParameterVisitor<B>>(&self, visitor: &mut V) -> Result<(), V::Error> {
        self.static_modules.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.static_modules.decoder_resources_mut().visit(visitor)
    }
}
impl<B: NeuralBackend, F: FixedReplicatedFamily<B>, P: 'static> TextComponentKernel<B>
    for FixedReplicatedKernel<B, F, P>
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
    fn supports_chunked_prefill() -> bool {
        F::CHUNKED_PREFILL
    }
}
trait FixedExecutionProfile<B: NeuralBackend, S: LayerRuntimeState<B>, F: FixedReplicatedFamily<B>>:
    'static
{
    fn validate(layout: &StateLayout, mask_layer: Option<usize>) -> Result<(), Error>;
    fn position(state: &mut S, layer: usize) -> Result<i32, Error>;
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error>;
}
impl<B, S, F, P> ComponentEquations<B, S> for FixedReplicatedKernel<B, F, P>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
    P: FixedExecutionProfile<B, S, F>,
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        P::validate(self.definition.state_layout(), F::mask_layer(&self.config))?;
        if state.layout() != self.definition.state_layout() {
            return Err(Error::backend(
                "replicated state layout differs from architecture",
            ));
        }
        let hidden = self.static_modules.embed(input.tokens, context)?;
        let position = F::mask_layer(&self.config)
            .map(|layer| P::position(state, layer))
            .transpose()?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, position, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
        })
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut forward = self.begin(input, state, context)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }
    fn begin_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut F::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        P::execute::<eredu_runtime::NoopObserver>(
            unit,
            &self.definition.unit(address.group(), address.index())?.path,
            address.index(),
            hidden,
            state,
            forward,
            context,
            None,
        )
    }
    fn execute_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut F::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        P::execute(
            unit,
            &self.definition.unit(address.group(), address.index())?.path,
            address.index(),
            hidden,
            state,
            forward,
            context,
            Some(observer),
        )
    }
    fn complete_group(
        &mut self,
        _group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if self.prediction_capture {
            forward.prediction_capture = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.finish_instrumented(
            hidden,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.static_modules.finish_instrumented(
            hidden,
            context,
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
        )
    }
    fn projects_final_text_position() -> bool {
        true
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish(
            &crate::decoder::final_hidden_position(hidden, context)?,
            state,
            forward,
            context,
        )
    }
}

pub(crate) type CompressedReplicatedModel<B, F, P = MixedCompressedState> = ComponentAdapter<
    Arc<ComponentDefinition>,
    CompressedReplicatedKernel<B, F, P>,
    KernelParameterModules,
>;
pub(crate) struct CompressedReplicatedKernel<
    B: BlockwiseAttentionBackend,
    F: CompressedReplicatedFamily<B>,
    P = MixedCompressedState,
> {
    config: F::Config,
    definition: Arc<ComponentDefinition>,
    static_modules: StaticModules<B>,
    family: PhantomData<(F, P)>,
}
impl<B: BlockwiseAttentionBackend, F: CompressedReplicatedFamily<B>, P: 'static>
    CompressedReplicatedKernel<B, F, P>
{
    pub fn from_definition(
        config: F::Config,
        definition: Arc<ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        F::validate(&config)?;
        let definition = select_definition(definition, F::definition(&config)?)?;
        let static_modules = StaticModules::from_spec(F::static_spec(&config), context)?;
        Ok(Self {
            config,
            definition,
            static_modules,
            family: PhantomData,
        })
    }
    pub fn into_model(self) -> Result<CompressedReplicatedModel<B, F, P>, Error> {
        ComponentAdapter::with_pinned_resources::<B>(self.definition.clone(), self)
            .map_err(Error::from)
    }
}
impl<B: BlockwiseAttentionBackend, F: CompressedReplicatedFamily<B>, P: 'static> ComponentKernel<B>
    for CompressedReplicatedKernel<B, F, P>
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;
    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error> {
        F::state_identity(
            &self.config,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }
    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<F::Unit, Error> {
        F::build_unit(&self.config, address.index(), context)
    }
    fn executes_group(&self, _group: usize, _forward: &Self::ForwardContext) -> bool {
        true
    }

    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter().chain(forward.prediction_capture.iter())
    }
}
impl<B: BlockwiseAttentionBackend, F: CompressedReplicatedFamily<B>, P: 'static>
    ComponentPinnedResources<B> for CompressedReplicatedKernel<B, F, P>
{
    type Modules = StaticModules<B>;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V: StaticParameterVisitor<B>>(&self, visitor: &mut V) -> Result<(), V::Error> {
        self.static_modules.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.static_modules.decoder_resources_mut().visit(visitor)
    }
}
impl<B: BlockwiseAttentionBackend, F: CompressedReplicatedFamily<B>, P: 'static>
    TextComponentKernel<B> for CompressedReplicatedKernel<B, F, P>
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}
trait CompressedExecutionProfile<
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    F: CompressedReplicatedFamily<B>,
>: 'static
{
    fn validate(layout: &StateLayout, mask_layer: Option<usize>) -> Result<(), Error>;
    fn position(state: &mut S, layer: usize) -> Result<i32, Error>;
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error>;
}
impl<B, S, F, P> ComponentEquations<B, S> for CompressedReplicatedKernel<B, F, P>
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    F: CompressedReplicatedFamily<B>,
    P: CompressedExecutionProfile<B, S, F>,
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        P::validate(self.definition.state_layout(), F::mask_layer(&self.config))?;
        if state.layout() != self.definition.state_layout() {
            return Err(Error::backend(
                "replicated state layout differs from architecture",
            ));
        }
        let hidden = self.static_modules.embed(input.tokens, context)?;
        let position = F::mask_layer(&self.config)
            .map(|layer| P::position(state, layer))
            .transpose()?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, position, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
        })
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut forward = self.begin(input, state, context)?;
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }
    fn begin_group(
        &mut self,
        _group: usize,
        initial: &B::Tensor,
        _dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut F::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        P::execute::<eredu_runtime::NoopObserver>(
            unit,
            &self.definition.unit(address.group(), address.index())?.path,
            address.index(),
            hidden,
            state,
            forward,
            context,
            None,
        )
    }
    fn execute_observed<O>(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut F::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        P::execute(
            unit,
            &self.definition.unit(address.group(), address.index())?.path,
            address.index(),
            hidden,
            state,
            forward,
            context,
            Some(observer),
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.finish_instrumented(
            hidden,
            context,
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.static_modules.finish_instrumented(
            hidden,
            context,
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
        )
    }
    fn projects_final_text_position() -> bool {
        true
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish(
            &crate::decoder::final_hidden_position(hidden, context)?,
            state,
            forward,
            context,
        )
    }
}

impl<B, S, F> FixedExecutionProfile<B, S, F> for MixedState
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    fn validate(_layout: &StateLayout, _mask_layer: Option<usize>) -> Result<(), Error> {
        Ok(())
    }
    fn position(state: &mut S, layer: usize) -> Result<i32, Error> {
        state
            .layer(layer)
            .map(|state| RuntimeStateComponents::position(&*state))
            .map_err(Error::backend)
    }
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error> {
        match observer {
            Some(observer) => F::forward_unit_observed(
                unit,
                path,
                hidden,
                state.layer(index).map_err(Error::backend)?,
                forward,
                context,
                observer,
            ),
            None => F::forward_unit(
                unit,
                hidden,
                state.layer(index).map_err(Error::backend)?,
                forward,
                context,
            ),
        }
    }
}

impl<B, S, F> FixedExecutionProfile<B, S, F> for AttentionState
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    F: FixedReplicatedFamily<B>,
{
    fn validate(_layout: &StateLayout, _mask_layer: Option<usize>) -> Result<(), Error> {
        Ok(())
    }
    fn position(state: &mut S, layer: usize) -> Result<i32, Error> {
        state
            .layer(layer)
            .map(|state| AttentionCache::offset(&*state))
            .map_err(Error::backend)
    }
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error> {
        match observer {
            Some(observer) => {
                let mut layer = AttentionLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit_observed(unit, path, hidden, &mut layer, forward, context, observer)
            }
            None => {
                let mut layer = AttentionLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit(unit, hidden, &mut layer, forward, context)
            }
        }
    }
}

impl<B, S, F> FixedExecutionProfile<B, S, F> for FixedState
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    fn validate(_layout: &StateLayout, _mask_layer: Option<usize>) -> Result<(), Error> {
        if _mask_layer.is_some() {
            return Err(Error::backend(
                "fixed-only profile has an attention mask layer",
            ));
        }
        Ok(())
    }
    fn position(state: &mut S, layer: usize) -> Result<i32, Error> {
        let _ = (state, layer);
        Err(Error::backend("profile has no attention frontier"))
    }
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error> {
        match observer {
            Some(observer) => {
                let mut layer = FixedLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit_observed(unit, path, hidden, &mut layer, forward, context, observer)
            }
            None => {
                let mut layer = FixedLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit(unit, hidden, &mut layer, forward, context)
            }
        }
    }
}

impl<B, S, F> FixedExecutionProfile<B, S, F> for Stateless
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
{
    fn validate(_layout: &StateLayout, _mask_layer: Option<usize>) -> Result<(), Error> {
        if _mask_layer.is_some()
            || _layout
                .layers()
                .iter()
                .any(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty())
        {
            return Err(Error::backend("stateless profile contains mutable state"));
        }
        Ok(())
    }
    fn position(state: &mut S, layer: usize) -> Result<i32, Error> {
        let _ = (state, layer);
        Err(Error::backend("profile has no attention frontier"))
    }
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error> {
        match observer {
            Some(observer) => {
                let mut layer = StatelessLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit_observed(unit, path, hidden, &mut layer, forward, context, observer)
            }
            None => {
                let mut layer = StatelessLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit(unit, hidden, &mut layer, forward, context)
            }
        }
    }
}

impl<B, S, F> CompressedExecutionProfile<B, S, F> for MixedCompressedState
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F: CompressedReplicatedFamily<B>,
{
    fn validate(_layout: &StateLayout, _mask_layer: Option<usize>) -> Result<(), Error> {
        Ok(())
    }
    fn position(state: &mut S, layer: usize) -> Result<i32, Error> {
        state
            .layer(layer)
            .map(|state| RuntimeStateComponents::position(&*state))
            .map_err(Error::backend)
    }
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error> {
        match observer {
            Some(observer) => F::forward_unit_observed(
                unit,
                path,
                hidden,
                state.layer(index).map_err(Error::backend)?,
                forward,
                context,
                observer,
            ),
            None => F::forward_unit(
                unit,
                hidden,
                state.layer(index).map_err(Error::backend)?,
                forward,
                context,
            ),
        }
    }
}

impl<B, S, F> CompressedExecutionProfile<B, S, F> for CompressedState
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
    F: CompressedReplicatedFamily<B>,
{
    fn validate(_layout: &StateLayout, _mask_layer: Option<usize>) -> Result<(), Error> {
        Ok(())
    }
    fn position(state: &mut S, layer: usize) -> Result<i32, Error> {
        state
            .layer(layer)
            .map(|state| CompressedAttentionCache::offset(&*state))
            .map_err(Error::backend)
    }
    fn execute<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        unit: &mut F::Unit,
        path: &str,
        index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &ReplicatedForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Error> {
        match observer {
            Some(observer) => {
                let mut layer = CompressedLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit_observed(unit, path, hidden, &mut layer, forward, context, observer)
            }
            None => {
                let mut layer = CompressedLayer::<B, _> {
                    inner: state.layer(index).map_err(Error::backend)?,
                    backend: PhantomData,
                };
                F::forward_unit(unit, hidden, &mut layer, forward, context)
            }
        }
    }
}

impl<B, F, P: 'static> FixedReplicatedKernel<B, F, P>
where
    B: BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    F: FixedReplicatedFamily<B>,
{
    fn embed_prediction_shared(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match parallel {
            Some(parallel) => B::vocabulary_parallel_lookup(
                &mut self.static_modules.embeddings,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => self.static_modules.embed(tokens, context),
        }
    }

    fn project_prediction_shared(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .project_instrumented(hidden, parallel, context, instrumentation)
    }
}
impl<B, P: 'static> crate::prediction_extension::NemotronHPredictionTarget<B>
    for FixedReplicatedModel<B, crate::replicated_text::NemotronHReplicated, P>
where
    B: BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
{
    fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.kernel_mut()
            .embed_prediction_shared(tokens, parallel, context)
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.kernel_mut()
            .project_prediction_shared(hidden, parallel, context, instrumentation)
    }
}
impl<B, P: 'static> crate::prediction_extension::QwenHybridPredictionTarget<B>
    for FixedReplicatedModel<B, crate::replicated_text::QwenHybridReplicated, P>
where
    B: BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
{
    fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.kernel_mut()
            .embed_prediction_shared(tokens, parallel, context)
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.kernel_mut().static_modules.project_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }
}

fn causal_mask<B: NeuralBackend>(
    sequence: i32,
    supplied: Option<&B::Tensor>,
    position: Option<i32>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Option<B::Tensor>, Error> {
    if let Some(mask) = supplied {
        return Ok(Some(mask.clone()));
    }
    match (sequence > 1, position) {
        (true, Some(position)) => B::causal_mask(sequence, position, None, context).map(Some),
        _ => Ok(None),
    }
}
