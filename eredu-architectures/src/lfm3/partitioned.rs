//! Shared complete and selected LFM3 equations over mixed state geometry.

use eredu_nn::{AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, Tensor};
use eredu_runtime::{
    ArchitectureParameterDescription, LayerRuntimeState, LayeredForwardState,
    LayeredPartitionInput, ParameterProvider, RuntimeStateComponents, StateLayout,
};

use eredu_runtime::PartitionComponentKernel as _;

use super::{ForwardContext, Lfm3Input, ModelArgs, PartitionLocalGeometry};
use crate::decoder::{
    ComponentInstrumentation, DecoderStaticResources, LayeredInput, PartitionStaticModules,
    StaticModules,
};

/// Shared execution adapter for one exact selected LFM3 ownership view.
pub type PartitionedLayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    Lfm3Kernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// LFM3 block equations and mixed attention/convolution geometry for one rank.
pub struct Lfm3Kernel<
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    M = PartitionStaticModules<B>,
    G = PartitionLocalGeometry,
> {
    args: ModelArgs,
    declaration: std::sync::Arc<super::Declaration>,
    static_modules: M,
    geometry: G,
    backend: std::marker::PhantomData<fn() -> B>,
}

/// Complete LFM3 equations with complete pinned modules and admitted local geometry.
pub type CompleteLfm3Kernel<B> = Lfm3Kernel<B, StaticModules<B>, CompleteLfm3Geometry>;

/// Architecture geometry independent of native parameter ownership.
pub trait Lfm3Geometry: 'static {
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
        declaration: &super::Declaration,
    ) -> (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    );
    /// Architecture-global invocations physically owned here.
    fn owned_units(&self) -> std::ops::Range<usize>;
    /// Local block geometry for one admitted global invocation.
    fn block(
        &self,
        declaration: &super::Declaration,
        index: usize,
    ) -> Result<super::BlockGeometry, Error>;

    /// Ordinary execution retains its global state admission on a TP construction.
    fn ordinary_state_layout<'a>(
        &'a self,
        _declaration: &'a super::Declaration,
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
                "LFM3 unit {index} is outside owned range {owned:?}"
            )));
        }
        Ok(index - owned.start)
    }
}

/// Complete state and optional tensor-local block geometry retained before materialization.
pub struct CompleteLfm3Geometry {
    parallel: Option<std::sync::Arc<super::LocalGeometry>>,
    state: StateLayout,
    units: usize,
}
impl Lfm3Geometry for CompleteLfm3Geometry {
    const CHUNKED_PREFILL: bool = super::CHUNKED_TEXT_PREFILL;
    const FINAL_TEXT_POSITION: bool = true;
    const ROUTING_POINTS: bool = false;
    fn complete_state_layout(&self) -> &StateLayout {
        &self.state
    }
    fn vocabulary_ranges(
        &self,
        declaration: &super::Declaration,
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
            global_vocabulary: declaration.statics.embedding_spec().vocabulary as usize,
            local: 0..declaration.statics.embedding_spec().vocabulary as usize,
        };
        (
            full.clone(),
            declaration.statics.head_spec().is_some().then_some(full),
        )
    }
    fn owned_units(&self) -> std::ops::Range<usize> {
        0..self.units
    }
    fn block(
        &self,
        declaration: &super::Declaration,
        index: usize,
    ) -> Result<super::BlockGeometry, Error> {
        self.state_index(index)?;
        match &self.parallel {
            Some(geometry) => geometry.block(index).copied().ok_or_else(|| {
                Error::backend(format!("missing rank-local LFM3 block geometry {index}"))
            }),
            None => declaration
                .blocks
                .get(index)
                .map(|scope| scope.geometry(declaration.args()))
                .ok_or_else(|| Error::backend("retained LFM3 scope omits requested geometry")),
        }
    }
    fn ordinary_state_layout<'a>(
        &'a self,
        declaration: &'a super::Declaration,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        match self.parallel {
            Some(_) => Ok(std::borrow::Cow::Borrowed(
                declaration.definition().state_layout(),
            )),
            None => Ok(std::borrow::Cow::Borrowed(&self.state)),
        }
    }
    fn parallel_state_layout(&self) -> Result<&StateLayout, Error> {
        self.parallel
            .as_ref()
            .map(|geometry| geometry.state_layout())
            .ok_or_else(|| Error::backend("LFM3 model was not built with local geometry"))
    }
}
impl Lfm3Geometry for PartitionLocalGeometry {
    fn complete_state_layout(&self) -> &StateLayout {
        self.complete_state_layout()
    }
    fn vocabulary_ranges(
        &self,
        _declaration: &super::Declaration,
    ) -> (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    ) {
        (self.embedding_range().clone(), self.output_range().cloned())
    }
    fn owned_units(&self) -> std::ops::Range<usize> {
        self.owned_units()
    }
    fn block(
        &self,
        _declaration: &super::Declaration,
        index: usize,
    ) -> Result<super::BlockGeometry, Error> {
        self.block(index).copied().ok_or_else(|| {
            Error::backend(format!(
                "LFM3 unit {index} is outside owned range {:?}",
                self.owned_units()
            ))
        })
    }
}

pub(crate) fn component_definition(
    parameters: ArchitectureParameterDescription,
    state: StateLayout,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let units = component_units(&parameters);
    let partition = crate::transport::pipeline_state(0, &state);
    eredu_runtime::ComponentDefinition::new(
        "lfm3",
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
    if definition.id() != "lfm3" {
        return Err(invalid(definition, "LFM3 component identity"));
    }
    Ok(())
}
fn select_definition(
    declaration: &super::Declaration,
    definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
    geometry: &impl Lfm3Geometry,
) -> Result<std::sync::Arc<eredu_runtime::ComponentDefinition>, Error> {
    validate_identity(&definition)?;
    definition.validate_parameter_topology(declaration.definition().parameters())?;
    let state = geometry.complete_state_layout();
    let global = declaration.definition().state_layout();
    if definition.state_layout() != global && definition.state_layout() != state {
        return Err(invalid(
            &definition,
            "LFM3 retained state differs from global or selected geometry",
        ));
    }
    if let Some((_, layout)) = definition.parameters().partition_layout() {
        let retained = super::local_geometry(declaration, layout).map_err(Error::backend_source)?;
        let (embedding, output) = geometry.vocabulary_ranges(declaration);
        if retained.state_layout() != state
            || retained.embedding_range() != &embedding
            || retained.output_range() != output.as_ref()
        {
            return Err(invalid(
                &definition,
                "LFM3 retained state or vocabulary placement differs from selected geometry",
            ));
        }
        for index in geometry.owned_units() {
            if retained.block(index) != Some(&geometry.block(declaration, index)?) {
                return Err(invalid(
                    &definition,
                    format!("LFM3 retained block {index} placement differs from selected geometry"),
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
    definition.select_state_layout(state).map_err(Error::from)
}

impl<B> Lfm3Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Constructs only modules already selected by one exact LFM3 partition.
    pub fn from_partition<A>(
        declaration: std::sync::Arc<super::Declaration>,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedLayeredModel<B>, Error> {
        let args = declaration.args().clone();
        partition
            .local_geometry()
            .validate_for(&declaration)
            .map_err(Error::backend)?;
        let [group] = partition.groups() else {
            return Err(Error::backend(
                "partitioned LFM3 must own exactly one execution group",
            ));
        };
        let owned = partition.local_geometry().owned_units();
        if group.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP
            || group.global_units() != owned
        {
            return Err(Error::backend(
                "LFM3 construction range differs from selected partition",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let definition = select_definition(&declaration, definition, &geometry)?;
        let selection = definition.select_partition(partition)?;
        let owns = |role: &str| {
            selection.owns_storage(&eredu_runtime::ParameterStorageId::StaticRole(role.into()))
        };
        use eredu_runtime::VocabularyModulePlacement as V;
        let embedding = if owns("embedding") {
            V::VocabularyParallel(geometry.embedding_range().clone())
        } else {
            V::Omitted
        };
        let head = match (owns("output"), geometry.output_range()) {
            (true, Some(range)) => V::VocabularyParallel(range.clone()),
            _ => V::Omitted,
        };
        let static_modules = PartitionStaticModules::from_construction(
            declaration.statics.clone(),
            embedding,
            head,
            owns("norm"),
            context,
        )?;
        let kernel = Self {
            args,
            declaration,
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

impl<B, M, G> Lfm3Kernel<B, M, G>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
{
    /// Normalized LFM3 policy.
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
        let geometry = self.geometry.block(&self.declaration, global_unit)?;
        let parameters = self
            .declaration
            .blocks
            .get(global_unit)
            .cloned()
            .ok_or_else(|| Error::backend("retained LFM3 declaration omits requested block"))?
            .local(geometry)?;
        super::Block::from_parameters(global_unit, parameters, context)
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
                "LFM3 runtime state does not match its partition-local layout",
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
            context: ForwardContext {
                mask,
                normalized_audio_output: false,
            },
        })
    }

    fn finish_requested(
        &mut self,
        hidden: &B::Tensor,
        forward: &ForwardContext<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        if forward.normalized_audio_output {
            self.static_modules
                .decoder_resources_mut()
                .normalize(hidden, context, instrumentation)
        } else {
            self.finish_local_instrumented(hidden, parallel, context, instrumentation)
        }
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

    /// Projects an Audio hidden position on the stage that owns vocabulary weights.
    pub fn project_audio_position(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.decoder_resources_mut().project(
            hidden,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn local_state_ordinal(&self, global_unit: usize) -> Result<usize, Error> {
        self.geometry.state_index(global_unit)
    }
}

impl<B, M, G> eredu_runtime::ComponentKernel<B> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Input<'a> = Lfm3Input<'a, B::Tensor>;
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

impl<B, M, G> eredu_runtime::ComponentPinnedResources<B> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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

impl<B, S, M, G> eredu_runtime::ComponentEquations<B, S> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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
        let expected = self.geometry.ordinary_state_layout(&self.declaration)?;
        if state.layout() != expected.as_ref() {
            return Err(Error::backend(
                "LFM3 runtime state does not match its admitted layout",
            ));
        }
        let hidden = self
            .static_modules
            .decoder_resources_mut()
            .embedding
            .ok_or_else(|| Error::backend("LFM3 partition does not own input embedding"))?
            .forward(input.tokens(), context)?;
        let mut forward = self.begin_hidden(hidden, input.mask(), state, &expected, 0, context)?;
        condition_prompt(input, true, &mut forward, context)?;
        Ok(forward)
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
        ComponentInstrumentation::apply_embedding(self.begin(input, state, context)?, observer)
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
            .then(|| self.declaration.routed_observation_points(&path, index))
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
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_requested(
            hidden,
            forward,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
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
        if G::FINAL_TEXT_POSITION && !forward.normalized_audio_output {
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
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut observer);
        self.finish_requested(hidden, forward, None, context, &mut instrumentation)
    }
}

impl<B, S, M, G> eredu_runtime::ParallelComponentKernel<B, S> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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
                .ok_or_else(|| Error::backend("LFM3 partition does not own input embedding"))?,
            input.tokens(),
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        let mut forward = self.begin_hidden(hidden, input.mask(), state, expected, 0, context)?;
        condition_prompt(input, true, &mut forward, context)?;
        Ok(forward)
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
        ComponentInstrumentation::apply_embedding(
            self.begin_parallel(input, state, parallel, context)?,
            observer,
        )
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
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_requested(
            hidden,
            forward,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
    fn finish_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut observer);
        self.finish_requested(
            hidden,
            forward,
            Some(parallel),
            context,
            &mut instrumentation,
        )
    }
}

impl<B, S, M, G> eredu_runtime::RoutedComponentKernel<B, S> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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
            self.declaration.routed_observation_points(
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
        let points = self.declaration.routed_observation_points(&path, index);
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

impl<B, S, M, G> eredu_runtime::ParallelRoutedComponentKernel<B, S> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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
        let points = self.declaration.routed_observation_points(&path, index);
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

impl<B, S, M, G> eredu_runtime::PartitionComponentKernel<B, S> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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
            .ok_or_else(|| Error::backend("LFM3 partition requires selected state geometry"))?;
        let context = selected.native;
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let embedding = self
                    .static_modules
                    .decoder_resources_mut()
                    .embedding
                    .ok_or_else(|| Error::backend("LFM3 partition does not own input embedding"))?;
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
            forward = ComponentInstrumentation::apply_embedding(forward, observer)?;
        }
        Ok(forward)
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<eredu_runtime::LayeredPartitionOutput<B::Tensor>, Self::Error> {
        if owns_output {
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output: self.finish_requested(
                    hidden,
                    forward,
                    parallel,
                    context,
                    &mut ComponentInstrumentation::disabled(),
                )?,
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
            self.finish_requested(hidden, forward, parallel, context, &mut instrumentation)?;
        Ok(eredu_runtime::LayeredPartitionOutput::Final {
            output,
            retained: None,
        })
    }
}

impl<B, M, G> eredu_runtime::TextComponentKernel<B> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    fn supports_chunked_prefill() -> bool {
        G::CHUNKED_PREFILL
    }
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        Lfm3Input::Text(LayeredInput { tokens, mask })
    }
}

impl<B, S, M, G> eredu_runtime::PipelineComponentKernel<B, S> for Lfm3Kernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        Ok((input.tokens().dim(0), input.tokens().dim(1)))
    }
    fn output_width(&self) -> i32 {
        self.args.vocab_size
    }
    fn output_shape(
        &self,
        input: &Self::Input<'_>,
        batch: i32,
        sequence: i32,
    ) -> Result<Vec<i32>, Self::Error> {
        let width = match input {
            Lfm3Input::Text(_) | Lfm3Input::Vision(_) => self.args.vocab_size,
            Lfm3Input::Audio(_) => self.args.hidden_size,
        };
        Ok(vec![batch, sequence, width])
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
        let embedded = incoming.is_none();
        let transported = match incoming {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input.tokens()),
        };
        let context = selected.native;
        let mut forward = self.begin_partition(transported, input.mask(), state, selected)?;
        condition_prompt(input, embedded, &mut forward, context)?;
        if embedded {
            ComponentInstrumentation::apply_embedding(forward, observer)
        } else {
            Ok(forward)
        }
    }
}

fn condition_prompt<T: Tensor>(
    input: Lfm3Input<'_, T>,
    embedding_owner: bool,
    forward: &mut LayeredForwardState<T, ForwardContext<T>>,
    context: &T::Context,
) -> Result<(), Error> {
    if let Lfm3Input::Vision(prompt) = &input {
        if embedding_owner {
            forward.hidden = prompt.replace(&forward.hidden, context)?;
        }
    }
    if let Lfm3Input::Audio(prompt) = input {
        if embedding_owner {
            forward.hidden = prompt.replace(&forward.hidden, context)?;
        }
        forward.context.normalized_audio_output = true;
    }
    Ok(())
}

impl<B, M, G> Lfm3Kernel<B, M, G>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    M: DecoderStaticResources<B>,
    G: Lfm3Geometry,
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

impl<B> CompleteLfm3Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Materializes retained scopes against exact selected ownership and optional local geometry.
    pub fn from_selected_declaration(
        declaration: std::sync::Arc<super::Declaration>,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        local: Option<super::LocalGeometry>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::LayeredModel<B>, Error> {
        let args = declaration.args().clone();
        if let Some(local) = &local {
            local.validate_for(&declaration).map_err(Error::backend)?;
        }
        let geometry = CompleteLfm3Geometry {
            state: match &local {
                Some(local) => local.state_layout().clone(),
                None => declaration.definition().state_layout().clone(),
            },
            parallel: local.map(std::sync::Arc::new),
            units: args.num_hidden_layers as usize,
        };
        let definition = select_definition(&declaration, definition, &geometry)?;
        let statics = &declaration.statics;
        let boundary =
            crate::decoder::NormalizedBoundary::new(statics.normalization_spec().clone(), context)?;
        let static_modules = match &geometry.parallel {
            Some(local) => StaticModules::from_parallel_boundary(
                statics.embedding_spec().clone(),
                statics.head_spec().cloned(),
                boundary,
                local.embedding_range().clone(),
                local.output_range().cloned(),
                context,
            )?,
            None => StaticModules::from_boundary(
                statics.embedding_spec().clone(),
                statics.head_spec().cloned(),
                boundary,
                context,
            )?,
        };
        eredu_runtime::ComponentAdapter::with_pinned_resources::<B>(
            definition,
            Self {
                args,
                declaration,
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
        Ok(self.declaration.definition().state_layout().clone())
    }
}
