//! Shared complete and selected LFM2 equations over mixed state geometry.

use eredu_nn::{AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, Tensor};
use eredu_runtime::{
    ArchitectureParameterDescription, LayerRuntimeState, LayeredForwardState,
    LayeredPartitionInput, ParameterProvider, RuntimeStateComponents, StateLayout,
};

use eredu_runtime::PartitionComponentKernel as _;

use super::{ForwardContext, ModelArgs, PartitionLocalGeometry};
use crate::decoder::{
    ComponentInstrumentation, DecoderStaticResources, LayeredInput, PartitionStaticModules,
    StaticModules,
};

/// Shared execution adapter for one exact selected LFM2 ownership view.
pub type PartitionedLayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    Lfm2Kernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// LFM2 block equations and mixed attention/convolution geometry for one rank.
pub struct Lfm2Kernel<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    M = PartitionStaticModules<B>,
    G = PartitionLocalGeometry,
> {
    args: ModelArgs,
    static_modules: M,
    geometry: G,
    backend: std::marker::PhantomData<fn() -> B>,
}

/// Complete LFM2 equations with complete pinned modules and admitted local geometry.
pub type CompleteLfm2Kernel<B> = Lfm2Kernel<B, StaticModules<B>, CompleteLfm2Geometry>;

/// Architecture geometry independent of native parameter ownership.
pub trait Lfm2Geometry: 'static {
    /// Admission for chunked text prefill.
    const CHUNKED_PREFILL: bool = false;
    /// Whether text readout projects only the final hidden position.
    const FINAL_TEXT_POSITION: bool = false;
    /// Whether public routed discovery supplies the family's logical seams.
    const ROUTING_POINTS: bool = false;
    /// Complete rank-local state before pipeline slicing.
    fn complete_state_layout(&self) -> &StateLayout;
    /// Complete embedding and optional independent output vocabulary ownership.
    fn vocabulary_ranges(
        &self,
        args: &ModelArgs,
    ) -> (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    );
    /// Architecture-global invocations physically owned here.
    fn owned_units(&self) -> std::ops::Range<usize>;
    /// Local block geometry for one admitted global invocation.
    fn block(&self, args: &ModelArgs, index: usize) -> Result<super::BlockGeometry, Error>;
    /// Selected routed bank geometry, when ownership requires specialization.
    fn routed_spec(&self, _index: usize) -> Option<eredu_nn::GroupedGatedProductSpec> {
        None
    }
    /// Ordinary execution retains its global state admission on a TP construction.
    fn ordinary_state_layout<'a>(
        &'a self,
        _args: &ModelArgs,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        Ok(std::borrow::Cow::Borrowed(self.complete_state_layout()))
    }
    /// Tensor-parallel execution requires admitted rank-local geometry.
    fn parallel_state_layout(&self) -> Result<&StateLayout, Error> {
        Ok(self.complete_state_layout())
    }
    /// Maps a validated global invocation to its actual local mutable-state slot.
    fn state_index(&self, index: usize) -> Result<usize, Error> {
        let owned = self.owned_units();
        if !owned.contains(&index) {
            return Err(Error::backend(format!(
                "LFM2 unit {index} is outside owned range {owned:?}"
            )));
        }
        Ok(index - owned.start)
    }
}

/// Complete state and optional tensor-local block geometry retained before materialization.
pub struct CompleteLfm2Geometry {
    parallel: Option<std::sync::Arc<super::LocalGeometry>>,
    state: StateLayout,
    units: usize,
}
impl Lfm2Geometry for CompleteLfm2Geometry {
    const CHUNKED_PREFILL: bool = super::CHUNKED_TEXT_PREFILL;
    const FINAL_TEXT_POSITION: bool = true;
    const ROUTING_POINTS: bool = true;
    fn complete_state_layout(&self) -> &StateLayout {
        &self.state
    }
    fn vocabulary_ranges(
        &self,
        args: &ModelArgs,
    ) -> (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    ) {
        if let Some(local) = &self.parallel {
            return (
                local.embedding_range().clone(),
                local.output_range().cloned(),
            );
        }
        let full = eredu_nn::VocabularyParallelRange {
            global_vocabulary: args.vocab_size as usize,
            local: 0..args.vocab_size as usize,
        };
        (full.clone(), (!args.tie_word_embeddings).then_some(full))
    }
    fn owned_units(&self) -> std::ops::Range<usize> {
        0..self.units
    }
    fn block(&self, args: &ModelArgs, index: usize) -> Result<super::BlockGeometry, Error> {
        self.state_index(index)?;
        match &self.parallel {
            Some(geometry) => geometry.block(index).copied().ok_or_else(|| {
                Error::backend(format!("missing rank-local LFM2 block geometry {index}"))
            }),
            None => Ok(super::BlockGeometry::replicated(args)),
        }
    }
    fn ordinary_state_layout<'a>(
        &'a self,
        args: &ModelArgs,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        match self.parallel {
            Some(_) => super::state_layout(args)
                .map(std::borrow::Cow::Owned)
                .map_err(Error::backend),
            None => Ok(std::borrow::Cow::Borrowed(&self.state)),
        }
    }
    fn parallel_state_layout(&self) -> Result<&StateLayout, Error> {
        self.parallel
            .as_ref()
            .map(|geometry| geometry.state_layout())
            .ok_or_else(|| Error::backend("LFM2 model was not built with local geometry"))
    }
}
impl Lfm2Geometry for PartitionLocalGeometry {
    fn complete_state_layout(&self) -> &StateLayout {
        self.complete_state_layout()
    }
    fn vocabulary_ranges(
        &self,
        _args: &ModelArgs,
    ) -> (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    ) {
        (self.embedding_range().clone(), self.output_range().cloned())
    }
    fn owned_units(&self) -> std::ops::Range<usize> {
        self.owned_units()
    }
    fn block(&self, _args: &ModelArgs, index: usize) -> Result<super::BlockGeometry, Error> {
        self.block(index).copied().ok_or_else(|| {
            Error::backend(format!(
                "LFM2 unit {index} is outside owned range {:?}",
                self.owned_units()
            ))
        })
    }
    fn routed_spec(&self, index: usize) -> Option<eredu_nn::GroupedGatedProductSpec> {
        self.expert_realization()
            .and_then(|plan| plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, index))
            .cloned()
    }
}

pub(crate) fn component_definition(
    parameters: ArchitectureParameterDescription,
    state: StateLayout,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let units = component_units(&parameters);
    let partition = crate::transport::pipeline_state(0, &state);
    eredu_runtime::ComponentDefinition::new(
        "lfm2",
        parameters,
        units,
        vec![crate::transport::decoder()],
        crate::decoder::TARGET_EXECUTION_GROUP,
        vec![],
        state,
        partition,
    )
    .map_err(Error::from)
}

fn component_units(
    parameters: &ArchitectureParameterDescription,
) -> Vec<eredu_runtime::ComponentUnit> {
    (0..parameters.unit_layout().len())
        .map(|ordinal| {
            let address = parameters
                .unit_layout()
                .address(ordinal)
                .expect("declared unit");
            eredu_runtime::ComponentUnit {
                path: format!("model.layers.{}", address.index()),
                state: address.index()..address.index() + 1,
            }
        })
        .collect()
}
fn invalid(definition: &eredu_runtime::ComponentDefinition, resource: impl Into<String>) -> Error {
    eredu_runtime::ComponentDefinitionError::Invalid {
        component: definition.id().into(),
        resource: resource.into(),
    }
    .into()
}
fn validate_identity(definition: &eredu_runtime::ComponentDefinition) -> Result<(), Error> {
    if definition.id() != "lfm2" {
        return Err(invalid(definition, "LFM2 component identity"));
    }
    Ok(())
}
/// Projects admitted formats while preserving logical consumers and observation seams.
pub fn project_parameters(
    args: &ModelArgs,
    definition: &std::sync::Arc<eredu_runtime::ComponentDefinition>,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    validate_identity(definition)?;
    let parameters = super::parameter_description(args).map_err(Error::backend)?;
    definition
        .project_parameters(parameters)
        .map_err(Error::from)
}
fn select_definition(
    args: &ModelArgs,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: &impl Lfm2Geometry,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    validate_identity(&definition)?;
    definition.validate_parameter_topology(
        &super::parameter_description(args).map_err(Error::backend)?,
    )?;
    let state = geometry.complete_state_layout();
    let global = super::state_layout(args).map_err(Error::backend)?;
    if definition.state_layout() != &global && definition.state_layout() != state {
        return Err(invalid(
            &definition,
            "LFM2 retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained = super::local_geometry(args, layout).map_err(Error::backend_source)?;
        let (embedding, output) = geometry.vocabulary_ranges(args);
        if retained.state_layout() != state
            || retained.embedding_range() != &embedding
            || retained.output_range() != output.as_ref()
        {
            return Err(invalid(
                &definition,
                "LFM2 retained state or vocabulary placement differs from selected geometry",
            ));
        }
        for index in geometry.owned_units() {
            if retained.block(index) != Some(&geometry.block(args, index)?) {
                return Err(invalid(
                    &definition,
                    format!("LFM2 retained block {index} placement differs from selected geometry"),
                ));
            }
        }
    }
    definition.validate_execution_contract(
        crate::decoder::TARGET_EXECUTION_GROUP,
        &[],
        &component_units(definition.parameters()),
        &[crate::transport::decoder()],
        &crate::transport::pipeline_state(0, definition.state_layout()),
        |_, _, boundaries| !boundaries.is_kernel_owned(),
    )?;
    definition
        .select_state_layout(state)
        .map_err(Error::from)
}


impl<B> Lfm2Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Constructs only modules already selected by one exact LFM2 partition.
    pub fn from_partition<A>(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedLayeredModel<B>, Error> {
        args.validate().map_err(Error::backend)?;
        partition
            .local_geometry()
            .validate_for(&args)
            .map_err(Error::backend)?;
        let [group] = partition.groups() else {
            return Err(Error::backend(
                "partitioned LFM2 must own exactly one execution group",
            ));
        };
        let owned = partition.local_geometry().owned_units();
        if group.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP
            || group.global_units() != owned
        {
            return Err(Error::backend(
                "LFM2 construction range differs from selected partition",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let definition = select_definition(&args, definition, &geometry)?;
        let selection = definition.select_partition(partition)?;
        let static_modules = PartitionStaticModules::from_selected_spec(
            super::static_spec(&args),
            Some((
                geometry.embedding_range().clone(),
                geometry.output_range().cloned(),
            )),
            |role| {
                selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(role.into()))
            },
            context,
        )?;
        let kernel = Self {
            args,
            static_modules,
            geometry,
            backend: std::marker::PhantomData,
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }
}

impl<B, M, G> Lfm2Kernel<B, M, G>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
{
    /// Normalized LFM2 policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Exact local TP/PP geometry.
    pub const fn local_geometry(&self) -> &G {
        &self.geometry
    }

    /// Pinned modules physically allocated on this pipeline partition.
    pub const fn static_modules(&self) -> &M {
        &self.static_modules
    }

    /// Mutably borrows the actual pinned resources for parameter editing.
    pub fn static_modules_mut(&mut self) -> &mut M {
        &mut self.static_modules
    }

    /// Constructs one owned block and rejects every other global index.
    pub fn construct_unit(
        &self,
        global_unit: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::Block<B>, Error> {
        let geometry = self.geometry.block(&self.args, global_unit)?;
        super::Block::new_with_geometry_and_routed_spec(
            &self.args,
            global_unit,
            geometry,
            self.geometry.routed_spec(global_unit),
            context,
        )
    }

    fn begin_hidden<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.layout() != expected || first_state_ordinal >= expected.len() {
            return Err(Error::backend(
                "LFM2 runtime state does not match its partition-local layout",
            ));
        }
        let sequence = hidden.dim(1);
        let mask =
            if let Some(mask) = mask {
                Some(mask.clone())
            } else if sequence > 1
                && self.args.layer_schedule.iter().any(|policy| {
                    matches!(policy.operator, super::OperatorPolicy::SelfAttention(_))
                })
            {
                Some(B::causal_mask(
                    sequence,
                    state
                        .layer(first_state_ordinal)
                        .map_err(Error::backend)?
                        .position(),
                    None,
                    context,
                )?)
            } else {
                None
            };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext { mask },
        })
    }

    fn finish_local(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_local_instrumented(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn finish_local_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        if parallel.is_some() {
            self.geometry.parallel_state_layout()?;
        }
        self.static_modules.decoder_resources_mut().finish_parallel(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }

    fn local_state_ordinal(&self, global_unit: usize) -> Result<usize, Error> {
        self.geometry.state_index(global_unit)
    }
}

impl<B, M, G> eredu_runtime::ComponentKernel<B> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;
    type Unit = super::Block<B>;
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
    ) -> Result<eredu_runtime::ModelStateIdentity, Self::Error> {
        super::state_identity(
            &self.args,
            state.layout(),
            state.global_layer_offset(),
            topology,
        )
    }
    fn build_unit(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        self.construct_unit(address.index(), context)
    }
    fn executes_group(&self, _: usize, _: &Self::ForwardContext) -> bool {
        true
    }
    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}

impl<B, M, G> eredu_runtime::ComponentPinnedResources<B> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Modules = M;
    fn pinned_modules(&self) -> &Self::Modules {
        &self.static_modules
    }
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        &mut self.static_modules
    }
    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.static_modules.decoder_resources().visit(visitor)
    }
    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.static_modules.decoder_resources_mut().visit(visitor)
    }
}

impl<B, S, M, G> eredu_runtime::ComponentEquations<B, S> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.geometry.ordinary_state_layout(&self.args)?;
        if state.layout() != expected.as_ref() {
            return Err(Error::backend(
                "LFM2 runtime state does not match its admitted layout",
            ));
        }
        let hidden = self
            .static_modules
            .decoder_resources_mut()
            .embedding
            .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?
            .forward(input.tokens, context)?;
        self.begin_hidden(hidden, input.mask, state, &expected, 0, context)
    }
    fn begin_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin(input, state, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
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
    ) -> Result<B::Tensor, Self::Error> {
        Ok(initial.clone())
    }
    fn execute(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block(address.index(), unit, hidden, state, forward, context)
    }
    fn execute_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let index = address.index();
        let path = format!("model.layers.{}", address.index());
        let ordinal = self.local_state_ordinal(index)?;
        let points = G::ROUTING_POINTS
            .then(|| self.args.routed_observation_points(&path, index))
            .flatten();
        let pass = if hidden.dim(1) > 1 {
            eredu_runtime::ExpertPass::Prefill
        } else {
            eredu_runtime::ExpertPass::Decode
        };
        let mut provider = eredu_runtime::ResidentExpertProvider;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            |policy, normalized, context, instrumentation| {
                policy.forward_observed_with_provider(
                    normalized,
                    pass,
                    context,
                    &mut provider,
                    instrumentation,
                    points,
                )
            },
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, None, context)
    }
    fn projects_final_text_position() -> bool {
        G::FINAL_TEXT_POSITION
    }
    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if G::FINAL_TEXT_POSITION {
            self.finish_local(
                &crate::decoder::final_hidden_position(hidden, context)?,
                None,
                context,
            )
        } else {
            self.finish(hidden, state, forward, context)
        }
    }
    fn finish_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_local_instrumented(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S, M, G> eredu_runtime::ParallelComponentKernel<B, S> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.geometry.parallel_state_layout()?;
        let hidden = B::vocabulary_parallel_lookup(
            self.static_modules
                .decoder_resources_mut()
                .embedding
                .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_hidden(hidden, input.mask, state, expected, 0, context)
    }
    fn begin_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut forward = self.begin_parallel(input, state, parallel, context)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
            .apply("embedding", forward.hidden)?;
        Ok(forward)
    }
    fn execute_parallel(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block_parallel(
            address.index(),
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
        )
    }
    fn execute_parallel_observed<O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let index = address.index();
        let path = format!("model.layers.{}", address.index());
        let ordinal = self.local_state_ordinal(index)?;
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_parallel_observed(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
        )
    }
    fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, Some(parallel), context)
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_local_instrumented(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S, M, G> eredu_runtime::RoutedComponentKernel<B, S> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn routing_points(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Self::Error> {
        self.geometry.state_index(address.index())?;
        Ok(if G::ROUTING_POINTS {
            self.args.routed_observation_points(
                &format!("model.layers.{}", address.index()),
                address.index(),
            )
        } else {
            None
        })
    }
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
    }
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let index = address.index();
        let path = format!("model.layers.{}", address.index());
        let ordinal = self.local_state_ordinal(index)?;
        let points = self.args.routed_observation_points(&path, index);
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_partition_instrumented_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(&path, &mut observer),
            |policy, normalized, context, instrumentation| {
                policy.forward_observed_with_provider(
                    normalized,
                    pass,
                    context,
                    provider,
                    instrumentation,
                    points,
                )
            },
        )
    }
    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_block_with_feed_forward(
            address.index(),
            unit,
            hidden,
            state,
            forward,
            context,
            |policy, normalized, context| {
                policy.forward_with_provider(normalized, pass, context, provider)
            },
        )
    }
}

impl<B, S, M, G> eredu_runtime::ParallelRoutedComponentKernel<B, S> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let index = address.index();
        let path = format!("model.layers.{}", address.index());
        let ordinal = self.local_state_ordinal(index)?;
        let points = self.args.routed_observation_points(&path, index);
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_partition_instrumented_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            Some(parallel),
            context,
            &mut crate::decoder::ComponentInstrumentation::new(&path, &mut observer),
            |policy, normalized, context, instrumentation| {
                policy.forward_parallel_observed_with_provider(
                    normalized,
                    pass,
                    parallel,
                    context,
                    provider,
                    instrumentation,
                    points,
                )
            },
        )
    }
    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: eredu_runtime::TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_block_parallel_with_feed_forward(
            address.index(),
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
            |policy, normalized, parallel, context| {
                policy.forward_parallel_with_provider(normalized, pass, parallel, context, provider)
            },
        )
    }
}

impl<B, S, M, G> eredu_runtime::PartitionComponentKernel<B, S> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args.hidden_size,
        ))
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = selected
            .expected
            .ok_or_else(|| Error::backend("LFM2 partition requires selected state geometry"))?;
        let context = selected.native;
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let embedding = self
                    .static_modules
                    .decoder_resources_mut()
                    .embedding
                    .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?;
                match selected.parallel {
                    Some(parallel) => B::vocabulary_parallel_lookup(
                        embedding,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                    None => embedding.forward(tokens, context)?,
                }
            }
            LayeredPartitionInput::Hidden { hidden, .. } => hidden,
        };
        self.begin_hidden(
            hidden,
            mask,
            state,
            expected,
            selected.first_state_ordinal,
            context,
        )
    }
    fn begin_partition_observed<O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let embedded = matches!(&input, LayeredPartitionInput::Tokens(_));
        let mut forward = self.begin_partition(input, mask, state, selected)?;
        if embedded {
            let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
            forward.hidden = ComponentInstrumentation::new("readout", &mut observer)
                .apply("embedding", forward.hidden)?;
        }
        Ok(forward)
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output: self.finish_local(hidden, parallel, context)?,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut observer);
        let output =
            self.finish_local_instrumented(hidden, parallel, context, &mut instrumentation)?;
        Ok(eredu_runtime::LayeredPartitionOutput::Final {
            output,
            retained: None,
        })
    }
}

impl<B, M, G> eredu_runtime::TextComponentKernel<B> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    fn supports_chunked_prefill() -> bool {
        G::CHUNKED_PREFILL
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, M, G> eredu_runtime::PipelineComponentKernel<B, S> for Lfm2Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        Ok((input.tokens.dim(0), input.tokens.dim(1)))
    }
    fn output_width(&self) -> i32 {
        self.args.vocab_size
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(B::Tensor, eredu_runtime::NoAuxiliaryBoundary)>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let transported = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input.tokens),
        };
        self.begin_partition_observed(transported, input.mask, state, selected, observer)
    }
}

impl<B, M, G> Lfm2Kernel<B, M, G>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    M: DecoderStaticResources<B>,
    G: Lfm2Geometry,
{
    /// Begins from embedded inputs with explicit admitted state geometry.
    pub fn begin_embedded_with_layout<S>(
        &self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.begin_hidden(hidden, mask, state, expected, 0, context)
    }
    /// Executes one owned block through its local mutable-state slot.
    pub fn forward_block<S>(
        &mut self,
        index: usize,
        block: &mut super::Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    {
        let ordinal = self.geometry.state_index(index)?;
        block.forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
        )
    }
    /// Executes an owned block with a supplied feed-forward operation.
    pub fn forward_block_with_feed_forward<S, F>(
        &mut self,
        index: usize,
        block: &mut super::Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut super::FeedForward<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let ordinal = self.geometry.state_index(index)?;
        block.forward_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            feed_forward,
        )
    }
    /// Executes the same owned block with tensor-parallel operators.
    pub fn forward_block_parallel<S>(
        &mut self,
        index: usize,
        block: &mut super::Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    {
        let ordinal = self.geometry.state_index(index)?;
        block.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
        )
    }
    /// Executes an owned parallel block with a supplied feed-forward operation.
    pub fn forward_block_parallel_with_feed_forward<S, F>(
        &mut self,
        index: usize,
        block: &mut super::Block<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        F: FnOnce(
            &mut super::FeedForward<B>,
            &B::Tensor,
            &B::ParallelContext,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let ordinal = self.geometry.state_index(index)?;
        block.forward_parallel_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            feed_forward,
        )
    }
}

impl<B> CompleteLfm2Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Materializes complete resources from the cold retained declaration.
    pub fn from_definition(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::LayeredModel<B>, Error> {
        Self::materialize(args, definition, None, context)
    }
    /// Materializes the same equations with admitted tensor-local geometry.
    pub fn from_parallel_definition(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        geometry: super::LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::LayeredModel<B>, Error> {
        Self::materialize(args, definition, Some(geometry), context)
    }
    fn materialize(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        local: Option<super::LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::LayeredModel<B>, Error> {
        args.validate().map_err(Error::backend)?;
        if let Some(local) = &local {
            local.validate_for(&args).map_err(Error::backend)?;
        }
        let geometry = CompleteLfm2Geometry {
            state: match &local {
                Some(local) => local.state_layout().clone(),
                None => super::state_layout(&args).map_err(Error::backend)?,
            },
            parallel: local.map(std::sync::Arc::new),
            units: args.num_hidden_layers as usize,
        };
        let definition = select_definition(&args, definition, &geometry)?;
        let static_modules = match &geometry.parallel {
            Some(local) => StaticModules::from_parallel_spec(
                super::static_spec(&args),
                local.embedding_range().clone(),
                local.output_range().cloned(),
                context,
            )?,
            None => StaticModules::from_spec(super::static_spec(&args), context)?,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(
            definition,
            Self {
                args,
                static_modules,
                geometry,
                backend: std::marker::PhantomData,
            },
        )
        .map_err(Error::from)
    }
    /// Returns planner-derived geometry for a tensor-local realization.
    pub fn parallel_geometry(&self) -> Option<&super::LocalGeometry> {
        self.geometry.parallel.as_deref()
    }
    /// Shares admitted local geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<super::LocalGeometry>> {
        self.geometry.parallel.as_ref().map(std::sync::Arc::clone)
    }
    /// Returns the ordinary global state declaration for low-level callers.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        super::state_layout(&self.args).map_err(Error::backend)
    }
}
