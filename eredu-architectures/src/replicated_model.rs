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
    decoder::{LayeredInput, StaticModuleSpecView, StaticModules, TARGET_EXECUTION_GROUP},
    hybrid_decoder::HybridDecoder,
};

pub(crate) struct ReplicatedForwardContext<T> {
    pub(crate) mask: Option<T>,
    prediction_capture: Option<T>,
}

pub(crate) trait FixedReplicatedFamily<B: NeuralBackend>: 'static {
    type Config: Clone;
    type Unit: Parameterized<B::Tensor>;

    /// Family proof for the validated ordinary target configuration. Internal
    /// hook availability is independent of causal row equivalence.
    const CAUSAL_PREFILL_ROWS: bool = false;
    const OBSERVATION_HOOKS: eredu_runtime::inspection::ObservationHookSupport =
        eredu_runtime::inspection::ObservationHookSupport::internal(false, false, false);

    fn validate(
        config: &Self::Config,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<(), Error>;
    fn layer_count(
        config: &Self::Config,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<usize, Error>;
    fn static_spec(config: &Self::Config) -> StaticModuleSpecView<'_>;
    fn state_layout(config: &Self::Config) -> Result<StateLayout, Error>;
    fn state_layout_with_metadata(
        config: &Self::Config,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<StateLayout, Error>;
    fn state_identity(
        config: &Self::Config,
        layout: &StateLayout,
        global_layer_start: usize,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error>;
    fn state_identity_with_metadata(
        config: &Self::Config,
        layout: &StateLayout,
        global_layer_start: usize,
        topology: PromptCacheTopology,
        metadata: &eredu_nn::workspace::WorkspaceContext,
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
    type Config: Clone;
    type Unit: Parameterized<B::Tensor>;

    /// Family proof for the validated ordinary target configuration. Internal
    /// hook availability is independent of causal row equivalence.
    const CAUSAL_PREFILL_ROWS: bool = false;
    const OBSERVATION_HOOKS: eredu_runtime::inspection::ObservationHookSupport =
        eredu_runtime::inspection::ObservationHookSupport::internal(false, false, false);

    fn validate(
        config: &Self::Config,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<(), Error>;
    fn layer_count(
        config: &Self::Config,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<usize, Error>;
    fn static_spec(config: &Self::Config) -> StaticModuleSpecView<'_>;
    fn state_layout(config: &Self::Config) -> Result<StateLayout, Error>;
    fn state_layout_with_metadata(
        config: &Self::Config,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<StateLayout, Error>;
    fn state_identity(
        config: &Self::Config,
        layout: &StateLayout,
        global_layer_start: usize,
        topology: PromptCacheTopology,
    ) -> Result<ModelStateIdentity, Error>;
    fn state_identity_with_metadata(
        config: &Self::Config,
        layout: &StateLayout,
        global_layer_start: usize,
        topology: PromptCacheTopology,
        metadata: &eredu_nn::workspace::WorkspaceContext,
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
    decoder: HybridDecoder<B>,
    prediction_capture: bool,
    family: PhantomData<(F, P)>,
    // The actual source outlives every module retained by this model.
    config: crate::replicated_text::ConfigOwner<F::Config>,
}

pub(crate) struct CompressedReplicatedModel<
    B: BlockwiseAttentionBackend,
    F: CompressedReplicatedFamily<B>,
    P = MixedCompressedState,
> {
    decoder: HybridDecoder<B>,
    family: PhantomData<(F, P)>,
    // The actual source outlives every module retained by this model.
    config: crate::replicated_text::ConfigOwner<F::Config>,
}

impl<B: NeuralBackend, F: FixedReplicatedFamily<B>, P> FixedReplicatedModel<B, F, P> {
    pub fn new(config: F::Config, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Self::new_with_config(config.into(), context)
    }

    pub(crate) fn new_with_config(
        config: crate::replicated_text::ConfigOwner<F::Config>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        F::validate(&config, metadata)?;
        let layers = F::layer_count(&config, metadata)?;
        // The family returns its actual borrowed declarations before any owned
        // names are constructed. The shared sink owns their checked copying.
        let static_spec = F::static_spec(&config);
        let static_spec = match B::construction_metadata(context) {
            Some(metadata) => static_spec.to_owned_with_metadata(metadata)?,
            None => static_spec.to_owned(),
        };
        Ok(Self {
            decoder: HybridDecoder::new(static_spec, "model.layers", layers, context)?,
            config,
            prediction_capture: false,
            family: PhantomData,
        })
    }

    pub(crate) fn with_prediction_capture(mut self) -> Self {
        self.prediction_capture = true;
        self
    }
}

impl<B, F, P> FixedReplicatedModel<B, F, P>
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
                &mut self.decoder.static_modules_mut().embeddings,
                tokens,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                parallel,
                context,
            ),
            None => self
                .decoder
                .static_modules_mut()
                .embeddings
                .forward(tokens, context),
        }
    }

    fn project_prediction_shared(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.decoder.static_modules_mut().project_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }
}

impl<B, P> crate::prediction_extension::NemotronHPredictionTarget<B>
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
        self.embed_prediction_shared(tokens, parallel, context)
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.project_prediction_shared(hidden, parallel, context, instrumentation)
    }
}

impl<B, P> crate::prediction_extension::QwenHybridPredictionTarget<B>
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
        self.embed_prediction_shared(tokens, parallel, context)
    }
    fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.decoder.static_modules_mut().project_instrumented(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }
}

impl<B: BlockwiseAttentionBackend, F: CompressedReplicatedFamily<B>, P>
    CompressedReplicatedModel<B, F, P>
{
    pub fn new(config: F::Config, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Self::new_with_config(config.into(), context)
    }

    pub(crate) fn new_with_config(
        config: crate::replicated_text::ConfigOwner<F::Config>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        F::validate(&config, metadata)?;
        let layers = F::layer_count(&config, metadata)?;
        // The family returns its actual borrowed declarations before any owned
        // names are constructed. The shared sink owns their checked copying.
        let static_spec = F::static_spec(&config);
        let static_spec = match B::construction_metadata(context) {
            Some(metadata) => static_spec.to_owned_with_metadata(metadata)?,
            None => static_spec.to_owned(),
        };
        Ok(Self {
            decoder: HybridDecoder::new(static_spec, "model.layers", layers, context)?,
            config,
            family: PhantomData,
        })
    }
}

fn parameter_description<B: NeuralBackend, U: Parameterized<B::Tensor>>(
    decoder: &HybridDecoder<B>,
    layers: usize,
    tied_head: bool,
    context: &<B::Tensor as Tensor>::Context,
    mut build: impl FnMut(usize) -> Result<U, Error>,
) -> Result<ArchitectureParameterDescription, Error> {
    let metadata = B::construction_metadata(context).filter(|context| context.uses_checked_metadata());
    crate::decoder::ModuleMetadata::new::<B>(context).controls::<(
        ArchitectureParameterDescription, eredu_runtime::ExecutionGraph, ExecutionUnitLayout,
        Vec<OwnedParameterGroupSpec>, ParameterGroupOwner, U, [usize; 1],
        Option<&eredu_nn::workspace::WorkspaceContext>,
    )>()?;
    if let Some(metadata) = metadata {
        metadata.charge_metadata(std::mem::size_of_val(&build))?;
    }
    let graph = match metadata {
        Some(metadata) => decoder.execution_graph_with_metadata(metadata)?
            .into_owned_with_metadata(metadata)?,
        None => decoder.execution_graph()?,
    };
    let layout = match metadata {
        Some(metadata) => ExecutionUnitLayout::new_with_metadata(&graph, &[layers], metadata)?,
        None => ExecutionUnitLayout::new(&graph, [layers]).map_err(Error::backend)?,
    };
    let modules = decoder.static_modules();
    let count = layers.checked_add(2 + usize::from(modules.lm_head.is_some()))
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    let mut owned = match metadata {
        Some(metadata) => metadata.metadata_vec(count)?,
        None => Vec::with_capacity(count),
    };
    fn group<B: NeuralBackend, M: Parameterized<B::Tensor>>(
        name: std::fmt::Arguments<'_>,
        role: ParameterRole,
        module: &M,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<eredu_runtime::ParameterGroupSpec, Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                std::fmt::Arguments<'_>, ParameterRole, &M,
                Result<eredu_runtime::ParameterGroupSpec, Error>,
                Option<&eredu_nn::workspace::WorkspaceContext>,
            )>())?;
        }
        match metadata {
            Some(metadata) => eredu_runtime::module_parameter_group_with_metadata::<B::Tensor, _>(
                name, role, module, metadata, |_, _| Ok(MemberSharding::Replicated),
            ),
            None => module_parameter_group::<B::Tensor, _>(
                name.to_string(), role, module, |_, _| Ok(MemberSharding::Replicated),
            ).map_err(Error::backend),
        }
    }
    fn add_static<B: NeuralBackend, M: Parameterized<B::Tensor>>(
        owned: &mut Vec<OwnedParameterGroupSpec>, name: &str, role: &str,
        parameter_role: ParameterRole, module: &M, tied_head: bool,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(), Error> {
        if let Some(metadata) = metadata {
            metadata.charge_metadata(std::mem::size_of::<(
                &mut Vec<OwnedParameterGroupSpec>, &str, &str, ParameterRole, &M, bool,
                Option<&eredu_nn::workspace::WorkspaceContext>, eredu_runtime::ParameterGroupSpec,
                ParameterGroupOwner, Vec<String>, Result<(), Error>,
            )>())?;
        }
        let group = group::<B, M>(format_args!("{name}"), parameter_role, module, metadata)?;
        let owner = if tied_head && role == "embedding" {
            match metadata {
                Some(metadata) => {
                    let mut roles = metadata.metadata_vec(2)?;
                    roles.push(metadata.metadata_string(format_args!("embedding"))?);
                    roles.push(metadata.metadata_string(format_args!("output"))?);
                    ParameterGroupOwner::StaticAnyOf(roles)
                }
                None => ParameterGroupOwner::static_any_of(["embedding", "output"]),
            }
        } else {
            match metadata {
                Some(metadata) => ParameterGroupOwner::static_role(
                    metadata.metadata_string(format_args!("{role}"))?,
                ),
                None => ParameterGroupOwner::static_role(role),
            }
        };
        owned.push(OwnedParameterGroupSpec::new(owner, group));
        Ok(())
    }
    add_static::<B, _>(&mut owned, "embedding", "embedding", ParameterRole::Vocabulary,
        &modules.embeddings, tied_head, metadata)?;
    add_static::<B, _>(&mut owned, "norm", "norm", ParameterRole::Replicated,
        &modules.norm, tied_head, metadata)?;
    if let Some(head) = &modules.lm_head {
        add_static::<B, _>(&mut owned, "output", "output", ParameterRole::Vocabulary,
            head, tied_head, metadata)?;
    }
    let owner_group = layout.group_id(0).expect("replicated target group");
    for index in 0..layers {
        let unit = build(index)?;
        let group = group::<B, _>(format_args!("model.layers.{index}"),
            ParameterRole::Replicated, &unit, metadata)?;
        let owner_group = match metadata {
            Some(metadata) => eredu_runtime::ExecutionGroupId::new(
                metadata.metadata_string(format_args!("{}", owner_group.as_str()))?,
            ).map_err(|cause| metadata.metadata_source(cause))?,
            None => owner_group.clone(),
        };
        owned.push(OwnedParameterGroupSpec::new(
            ParameterGroupOwner::execution_unit(owner_group, index), group,
        ));
    }
    // The authoritative groups are these exact emitted groups. The shared
    // consuming validator avoids deep-cloning every name and shape solely for
    // duplicate expected/owned collections.
    match metadata {
        Some(metadata) => ArchitectureParameterDescription::from_owned_with_metadata(
            graph, layout, owned, metadata,
        ),
        None => ArchitectureParameterDescription::from_owned(graph, layout, owned)
            .map_err(Error::backend),
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

macro_rules! architecture_parameters {
    ($model:ident, $family:ident, $backend:path) => {
        impl<B: $backend, F: $family<B>, P> eredu_runtime::ArchitectureParameters<B>
            for $model<B, F, P>
        {
            type DefinitionError = Error;

            fn state_layout(&self) -> Result<StateLayout, Error> {
                F::state_layout(&self.config)
            }

            fn state_layout_with_metadata(
                &self,
                context: &eredu_nn::workspace::WorkspaceContext,
            ) -> Result<StateLayout, Error> {
                F::state_layout_with_metadata(&self.config, context)
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

            fn state_identity_with_metadata(
                &self,
                state: &eredu_runtime::PartitionState,
                topology: PromptCacheTopology,
                metadata: &eredu_nn::workspace::WorkspaceContext,
            ) -> Result<ModelStateIdentity, Error> {
                F::state_identity_with_metadata(
                    &self.config,
                    state.layout(),
                    state.global_layer_offset(),
                    topology,
                    metadata,
                )
            }

            fn parameter_description(
                &self,
                context: &<B::Tensor as Tensor>::Context,
            ) -> Result<ArchitectureParameterDescription, Error> {
                parameter_description(
                    &self.decoder,
                    F::layer_count(
                        &self.config,
                        crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
                    )?,
                    self.decoder.static_modules().lm_head.is_none(),
                    context,
                    |index| F::build_unit(&self.config, index, context),
                )
            }

            fn retained_static_value_slot_bound(&self) -> Option<usize> {
                eredu_nn::Parameterized::retained_value_slot_bound(self.decoder.static_modules())
            }

            fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
                eredu_nn::Parameterized::visit_retained_values(
                    self.decoder.static_modules(),
                    visitor,
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

// Every fixed state profile uses the same target capture semantics. Selecting
// a narrower cache representation must not remove the requested hidden output.
macro_rules! fixed_prediction_target_methods {
    () => {
        fn complete_execution_group(
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

        fn retain_prediction_target_capture(&mut self) {
            self.prediction_capture = true;
        }

        fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
            forward.prediction_capture.as_ref()
        }
    };
}

macro_rules! common_layered_methods {
    () => {
        fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
            F::OBSERVATION_HOOKS
        }
        fn prefill_observation_declarations(
            &self,
        ) -> Result<Vec<eredu_runtime::layered::PrefillObservationDeclaration>, Error> {
            if !F::CAUSAL_PREFILL_ROWS {
                return Ok(Vec::new());
            }
            // These paths belong to this selected shell, including its actual
            // state profile and checkpoint transformation. No direct family
            // model is reconstructed to supply a different path owner.
            let count = self.decoder.group_unit_count(0)?;
            crate::decoder::ordinary_prefill_observation_declarations(
                (0..count).map(|index| self.decoder.unit_path(0, index)),
                true,
            )
        }
        fn group_transport(&self, _group: usize) -> eredu_runtime::ArchitectureGroupTransport {
            crate::transport::decoder()
        }
        fn group_transport_matches(
            &self,
            _group: usize,
            expected: &eredu_runtime::ArchitectureGroupTransport,
        ) -> bool {
            crate::transport::decoder_declaration().matches(expected)
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
        fn execution_graph_with_metadata(
            &self,
            context: &eredu_nn::workspace::WorkspaceContext,
        ) -> Result<eredu_runtime::ArchitectureExecutionGraph<'_>, Error> {
            self.decoder.execution_graph_with_metadata(context)
        }
        fn group_unit_count_with_metadata(
            &self,
            group: usize,
            context: &eredu_nn::workspace::WorkspaceContext,
        ) -> Result<usize, Error> {
            self.decoder.group_unit_count_with_metadata(group, context)
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
            let mut forward = self.begin_forward(input, state, context)?;
            let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden =
                crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed)
                    .apply("embedding", forward.hidden)?;
            Ok(forward)
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
        fn select_readout_positions(
            &self,
            hidden: &B::Tensor,
            _forward: &Self::ForwardContext,
            demand: eredu_core::OutputDemand,
            context: &<B::Tensor as Tensor>::Context,
        ) -> Result<Option<B::Tensor>, Self::Error> {
            crate::readout::select_readout_positions(hidden, demand, 1, context)
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
        fn finish_forward_observed<O>(
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
            self.decoder.finish_logits_instrumented(
                hidden,
                context,
                &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
            )
        }
        fn retained_context_values<'a>(
            &'a self,
            forward: &'a Self::ForwardContext,
            _group: usize,
            _index: usize,
        ) -> Self::RetainedContextValues<'a> {
            forward.mask.iter().chain(forward.prediction_capture.iter())
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

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();
    fixed_prediction_target_methods!();


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
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
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
        let path = self.decoder.unit_path(group, index)?;
        F::forward_unit_observed(
            unit,
            &path,
            hidden,
            state.layer(index).map_err(Error::backend)?,
            forward,
            context,
            observer,
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

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();
    fixed_prediction_target_methods!();

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
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
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
        let path = self.decoder.unit_path(group, index)?;
        let mut layer = AttentionLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit_observed(unit, &path, hidden, &mut layer, forward, context, observer)
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

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();
    fixed_prediction_target_methods!();

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
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
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
        let path = self.decoder.unit_path(group, index)?;
        let mut layer = FixedLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit_observed(unit, &path, hidden, &mut layer, forward, context, observer)
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

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
    where
        Self: 'a,
        B::Tensor: 'a;
    type Error = Error;

    common_layered_methods!();
    fixed_prediction_target_methods!();

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
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
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
        let path = self.decoder.unit_path(group, index)?;
        let mut layer = StatelessLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit_observed(unit, &path, hidden, &mut layer, forward, context, observer)
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

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
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
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
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
        let path = self.decoder.unit_path(group, index)?;
        F::forward_unit_observed(
            unit,
            &path,
            hidden,
            state.layer(index).map_err(Error::backend)?,
            forward,
            context,
            observer,
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

    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error> {
        crate::prefill::token_shape(input.tokens).map(Some)
    }

    type StaticModules = StaticModules<B>;
    type Unit = F::Unit;
    type ForwardContext = ReplicatedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::iter::Chain<std::option::Iter<'a, B::Tensor>, std::option::Iter<'a, B::Tensor>>
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
            context: ReplicatedForwardContext {
                mask,
                prediction_capture: None,
            },
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
        let path = self.decoder.unit_path(group, index)?;
        let mut layer = CompressedLayer::<B, _> {
            inner: state.layer(index).map_err(Error::backend)?,
            backend: PhantomData,
        };
        F::forward_unit_observed(unit, &path, hidden, &mut layer, forward, context, observer)
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
