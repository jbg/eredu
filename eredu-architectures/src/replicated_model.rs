//! Dense-only layered models for architecture-owned replicated text dispatch.

use std::marker::PhantomData;

use eredu_core::cache::PromptCacheTopology;
use eredu_nn::{
    AttentionCache, AttentionRequest, BlockwiseAttentionBackend, CompressedAttentionBlock,
    CompressedAttentionCache, CompressedAttentionScan, CompressedAttentionState,
    CompressedAttentionView, EmbeddingOperator, Error, NeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    module_parameter_group, ArchitectureParameterDescription, ExecutionUnitLayout,
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, MemberSharding,
    ModelStateIdentity, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterRole,
    ReplicatedTextArchitecture, RuntimeLayerState, RuntimeStateComponents, StateError, StateLayout,
    StaticParameterVisitor, StaticParameterVisitorMut,
};

use crate::{
    decoder::{LayeredInput, StaticModuleSpec, StaticModules, TARGET_EXECUTION_GROUP},
    hybrid_decoder::HybridDecoder,
};

pub(crate) struct ReplicatedForwardContext<T> {
    pub(crate) mask: Option<T>,
}

pub(crate) trait FixedReplicatedFamily<B: NeuralBackend>: 'static {
    type Config: Clone;
    type Unit: Parameterized<B::Tensor>;

    fn validate(config: &Self::Config) -> Result<(), Error>;
    fn layer_count(config: &Self::Config) -> Result<usize, Error>;
    fn static_spec(config: &Self::Config) -> StaticModuleSpec;
    fn state_layout(config: &Self::Config) -> Result<StateLayout, Error>;
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
}

pub(crate) trait CompressedReplicatedFamily<B: BlockwiseAttentionBackend>: 'static {
    type Config: Clone;
    type Unit: Parameterized<B::Tensor>;

    fn validate(config: &Self::Config) -> Result<(), Error>;
    fn layer_count(config: &Self::Config) -> Result<usize, Error>;
    fn static_spec(config: &Self::Config) -> StaticModuleSpec;
    fn state_layout(config: &Self::Config) -> Result<StateLayout, Error>;
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

pub(crate) struct FixedReplicatedModel<
    B: NeuralBackend,
    F: FixedReplicatedFamily<B>,
    P = MixedState,
> {
    config: F::Config,
    decoder: HybridDecoder<B>,
    family: PhantomData<(F, P)>,
}

pub(crate) struct CompressedReplicatedModel<
    B: BlockwiseAttentionBackend,
    F: CompressedReplicatedFamily<B>,
    P = MixedCompressedState,
> {
    config: F::Config,
    decoder: HybridDecoder<B>,
    family: PhantomData<(F, P)>,
}

impl<B: NeuralBackend, F: FixedReplicatedFamily<B>, P> FixedReplicatedModel<B, F, P> {
    pub fn new(config: F::Config, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        F::validate(&config)?;
        let layers = F::layer_count(&config)?;
        Ok(Self {
            decoder: HybridDecoder::new(F::static_spec(&config), "model.layers", layers, context)?,
            config,
            family: PhantomData,
        })
    }
}

impl<B: BlockwiseAttentionBackend, F: CompressedReplicatedFamily<B>, P>
    CompressedReplicatedModel<B, F, P>
{
    pub fn new(config: F::Config, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        F::validate(&config)?;
        let layers = F::layer_count(&config)?;
        Ok(Self {
            decoder: HybridDecoder::new(F::static_spec(&config), "model.layers", layers, context)?,
            config,
            family: PhantomData,
        })
    }
}

fn parameter_description<B: NeuralBackend, U: Parameterized<B::Tensor>>(
    decoder: &HybridDecoder<B>,
    layers: usize,
    tied_head: bool,
    mut build: impl FnMut(usize) -> Result<U, Error>,
) -> Result<ArchitectureParameterDescription, Error> {
    let graph = decoder.execution_graph()?;
    let layout = ExecutionUnitLayout::new(&graph, [layers]).map_err(Error::backend)?;
    let mut expected = Vec::new();
    let mut owned = Vec::new();
    let modules = decoder.static_modules();
    fn add_static<B: NeuralBackend, M: Parameterized<B::Tensor>>(
        expected: &mut Vec<eredu_runtime::ParameterGroupSpec>,
        owned: &mut Vec<OwnedParameterGroupSpec>,
        name: &str,
        role: &str,
        parameter_role: ParameterRole,
        module: &M,
        tied_head: bool,
    ) -> Result<(), Error> {
        let group = module_parameter_group::<B::Tensor, _>(name, parameter_role, module, |_, _| {
            Ok(MemberSharding::Replicated)
        })
        .map_err(Error::backend)?;
        expected.push(group.clone());
        let owner = if tied_head && role == "embedding" {
            ParameterGroupOwner::static_any_of(["embedding", "output"])
        } else {
            ParameterGroupOwner::static_role(role)
        };
        owned.push(OwnedParameterGroupSpec::new(owner, group));
        Ok(())
    }
    add_static::<B, _>(
        &mut expected,
        &mut owned,
        "embedding",
        "embedding",
        ParameterRole::Vocabulary,
        &modules.embeddings,
        tied_head,
    )?;
    add_static::<B, _>(
        &mut expected,
        &mut owned,
        "norm",
        "norm",
        ParameterRole::Replicated,
        &modules.norm,
        tied_head,
    )?;
    if let Some(head) = &modules.lm_head {
        add_static::<B, _>(
            &mut expected,
            &mut owned,
            "output",
            "output",
            ParameterRole::Vocabulary,
            head,
            tied_head,
        )?;
    }
    let owner_group = layout.group_id(0).expect("replicated target group").clone();
    for index in 0..layers {
        let unit = build(index)?;
        let group = module_parameter_group::<B::Tensor, _>(
            format!("model.layers.{index}"),
            ParameterRole::Replicated,
            &unit,
            |_, _| Ok(MemberSharding::Replicated),
        )
        .map_err(Error::backend)?;
        expected.push(group.clone());
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(owner_group.clone(), index),
            group,
        ));
    }
    ArchitectureParameterDescription::new(&graph, &layout, expected, owned).map_err(Error::backend)
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

macro_rules! architecture_parameters {
    ($model:ident, $family:ident, $backend:path) => {
        impl<B: $backend, F: $family<B>, P> eredu_runtime::ArchitectureParameters<B>
            for $model<B, F, P>
        {
            type DefinitionError = Error;

            fn state_layout(&self) -> Result<StateLayout, Error> {
                F::state_layout(&self.config)
            }

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

            fn parameter_description(
                &self,
                context: &<B::Tensor as Tensor>::Context,
            ) -> Result<ArchitectureParameterDescription, Error> {
                parameter_description(
                    &self.decoder,
                    F::layer_count(&self.config)?,
                    self.decoder.static_modules().lm_head.is_none(),
                    |index| F::build_unit(&self.config, index, context),
                )
            }

            fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
            where
                V: StaticParameterVisitor<B>,
            {
                let modules = self.decoder.static_modules();
                visitor.visit("embedding", &modules.embeddings)?;
                visitor.visit("norm", &modules.norm)?;
                if let Some(head) = &modules.lm_head {
                    visitor.visit("output", head)?;
                }
                Ok(())
            }

            fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
            where
                V: StaticParameterVisitorMut<B>,
            {
                let modules = self.decoder.static_modules_mut();
                visitor.visit_mut("embedding", &mut modules.embeddings)?;
                visitor.visit_mut("norm", &mut modules.norm)?;
                if let Some(head) = &mut modules.lm_head {
                    visitor.visit_mut("output", head)?;
                }
                Ok(())
            }
        }
    };
}

architecture_parameters!(FixedReplicatedModel, FixedReplicatedFamily, NeuralBackend);
architecture_parameters!(
    CompressedReplicatedModel,
    CompressedReplicatedFamily,
    BlockwiseAttentionBackend
);

macro_rules! common_layered_methods {
    () => {
        fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
            crate::transport::decoder()
        }
        fn primary_execution_group(&self) -> &str {
            TARGET_EXECUTION_GROUP
        }
        fn state_partition_plan(
            &self,
            layout: &StateLayout,
        ) -> eredu_runtime::ArchitectureStatePartitionPlan {
            crate::transport::pipeline_state(0, layout)
        }
        fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
            self.decoder.execution_graph()
        }
        fn group_unit_count(&self, group: usize) -> Result<usize, Error> {
            self.decoder.group_unit_count(group)
        }
        fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
            self.decoder.unit_path(group, index)
        }
        fn static_modules(&self) -> &Self::StaticModules {
            self.decoder.static_modules()
        }
        fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
            self.decoder.static_modules_mut()
        }
        fn begin_execution_group(
            &mut self,
            group: usize,
            initial: &B::Tensor,
            dependencies: &[&B::Tensor],
            _state: &mut S,
            _forward: &mut Self::ForwardContext,
            _context: &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error> {
            self.decoder.begin_group(group, initial, dependencies)
        }
        fn finish_forward(
            &mut self,
            hidden: &B::Tensor,
            _state: &mut S,
            _forward: &Self::ForwardContext,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error> {
            self.decoder.finish_logits(hidden, context)
        }
        fn retained_context_values<'a>(
            &'a self,
            forward: &'a Self::ForwardContext,
            _group: usize,
            _index: usize,
        ) -> Self::RetainedContextValues<'a> {
            forward.mask.iter()
        }
    };
}

impl<B, S, F> LayeredArchitecture<B, S> for FixedReplicatedModel<B, F, MixedState>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.decoder.unit_path(group, index)?;
        F::build_unit(&self.config, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = F::state_layout(&self.config)?;
        if state.layout() != &expected {
            return Err(Error::backend(
                "replicated fixed-state layout differs from architecture",
            ));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        let position = F::mask_layer(&self.config)
            .map(|layer| {
                state
                    .layer(layer)
                    .map(|state| RuntimeStateComponents::position(&*state))
                    .map_err(Error::backend)
            })
            .transpose()?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, position, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext { mask },
        })
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
        self.decoder.unit_path(group, index)?;
        F::forward_unit(
            unit,
            hidden,
            state.layer(index).map_err(Error::backend)?,
            forward,
            context,
        )
    }
}

impl<B, S, F> ReplicatedTextArchitecture<B, S> for FixedReplicatedModel<B, F, MixedState>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, F> LayeredArchitecture<B, S> for FixedReplicatedModel<B, F, AttentionState>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    F: FixedReplicatedFamily<B>,
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.decoder.unit_path(group, index)?;
        F::build_unit(&self.config, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = F::state_layout(&self.config)?;
        if state.layout() != &expected {
            return Err(Error::backend(
                "replicated attention-state layout differs from architecture",
            ));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        let position = F::mask_layer(&self.config)
            .map(|layer| {
                state
                    .layer(layer)
                    .map(|state| AttentionCache::offset(&*state))
                    .map_err(Error::backend)
            })
            .transpose()?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, position, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext { mask },
        })
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
        self.decoder.unit_path(group, index)?;
        let mut layer = AttentionLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit(unit, hidden, &mut layer, forward, context)
    }
}

impl<B, S, F> ReplicatedTextArchitecture<B, S> for FixedReplicatedModel<B, F, AttentionState>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
    F: FixedReplicatedFamily<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, F> LayeredArchitecture<B, S> for FixedReplicatedModel<B, F, FixedState>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.decoder.unit_path(group, index)?;
        F::build_unit(&self.config, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = F::state_layout(&self.config)?;
        if state.layout() != &expected || F::mask_layer(&self.config).is_some() {
            return Err(Error::backend(
                "replicated fixed-only layout differs from architecture profile",
            ));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, None, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext { mask },
        })
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
        self.decoder.unit_path(group, index)?;
        let mut layer = FixedLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit(unit, hidden, &mut layer, forward, context)
    }
}

impl<B, S, F> ReplicatedTextArchitecture<B, S> for FixedReplicatedModel<B, F, FixedState>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B>,
    F: FixedReplicatedFamily<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, F> LayeredArchitecture<B, S> for FixedReplicatedModel<B, F, Stateless>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.decoder.unit_path(group, index)?;
        F::build_unit(&self.config, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = F::state_layout(&self.config)?;
        let has_components = expected
            .layers()
            .iter()
            .any(|layer| layer.attention().is_some() || !layer.fixed_state().is_empty());
        if state.layout() != &expected || has_components || F::mask_layer(&self.config).is_some() {
            return Err(Error::backend(
                "replicated stateless layout differs from architecture profile",
            ));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, None, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext { mask },
        })
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
        self.decoder.unit_path(group, index)?;
        let mut layer = StatelessLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit(unit, hidden, &mut layer, forward, context)
    }
}

impl<B, S, F> ReplicatedTextArchitecture<B, S> for FixedReplicatedModel<B, F, Stateless>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    F: FixedReplicatedFamily<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, F> LayeredArchitecture<B, S> for CompressedReplicatedModel<B, F, MixedCompressedState>
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F: CompressedReplicatedFamily<B>,
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.decoder.unit_path(group, index)?;
        F::build_unit(&self.config, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = F::state_layout(&self.config)?;
        if state.layout() != &expected {
            return Err(Error::backend(
                "replicated compressed-state layout differs from architecture",
            ));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        let position = F::mask_layer(&self.config)
            .map(|layer| {
                state
                    .layer(layer)
                    .map(|state| RuntimeStateComponents::position(&*state))
                    .map_err(Error::backend)
            })
            .transpose()?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, position, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext { mask },
        })
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
        self.decoder.unit_path(group, index)?;
        F::forward_unit(
            unit,
            hidden,
            state.layer(index).map_err(Error::backend)?,
            forward,
            context,
        )
    }
}

impl<B, S, F> ReplicatedTextArchitecture<B, S>
    for CompressedReplicatedModel<B, F, MixedCompressedState>
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    F: CompressedReplicatedFamily<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, F> LayeredArchitecture<B, S> for CompressedReplicatedModel<B, F, CompressedState>
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
    F: CompressedReplicatedFamily<B>,
{
    type Input<'a>
        = LayeredInput<'a, B::Tensor>
    where
        Self: 'a;
    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.decoder.unit_path(group, index)?;
        F::build_unit(&self.config, index, context)
    }
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let expected = F::state_layout(&self.config)?;
        if state.layout() != &expected {
            return Err(Error::backend(
                "replicated compressed-only layout differs from architecture",
            ));
        }
        let hidden = self
            .decoder
            .static_modules_mut()
            .embeddings
            .forward(input.tokens, context)?;
        let position = F::mask_layer(&self.config)
            .map(|layer| {
                state
                    .layer(layer)
                    .map(|state| CompressedAttentionCache::offset(&*state))
                    .map_err(Error::backend)
            })
            .transpose()?;
        let mask = causal_mask::<B>(hidden.dim(1), input.mask, position, context)?;
        Ok(LayeredForwardState {
            hidden,
            context: ReplicatedForwardContext { mask },
        })
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
        self.decoder.unit_path(group, index)?;
        let mut layer = CompressedLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit(unit, hidden, &mut layer, forward, context)
    }
}

impl<B, S, F> ReplicatedTextArchitecture<B, S> for CompressedReplicatedModel<B, F, CompressedState>
where
    B: BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
    F: CompressedReplicatedFamily<B>,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}
