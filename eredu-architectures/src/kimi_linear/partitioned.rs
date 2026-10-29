//! Shared complete and selected Kimi equations over recurrent and compressed state.

use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, EmbeddingLookupPolicy, EmbeddingOperator,
    Error, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, LayerRuntimeState, LayeredForwardState,
    LayeredPartitionInput, ParameterProvider, RuntimeStateComponents, StateLayout,
};
use eredu_runtime::{
    ComponentEquations as _, ParallelComponentKernel as _, ParallelRoutedComponentKernel as _,
    PartitionComponentKernel as _, RoutedComponentKernel as _,
};

use super::{AttentionKind, ForwardContext, ModelArgs, PartitionLocalGeometry};
use crate::decoder::{DecoderStaticResources, LayeredInput, PartitionStaticModules, StaticModules};
use eredu_runtime::ExpertPass;

/// Shared typed adapter for one exact selected Kimi Linear ownership view.
pub type PartitionedLayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    KimiKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Kimi equations over complete or selected native resource ownership.
pub struct KimiKernel<B, M = PartitionStaticModules<B>, G = PartitionLocalGeometry>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
{
    args: ModelArgs,
    static_modules: M,
    geometry: G,
    backend: std::marker::PhantomData<fn() -> B>,
}

/// Complete Kimi equations with complete pinned modules and admitted local geometry.
pub type CompleteKimiKernel<B> = KimiKernel<B, StaticModules<B>, CompleteKimiGeometry>;

/// Stateful Kimi geometry independent of native parameter ownership.
pub trait KimiGeometry: 'static {
    /// Complete rank-local state declaration before pipeline slicing.
    fn complete_state_layout(&self) -> &StateLayout;
    /// Complete embedding and optional independent output vocabulary ownership.
    fn vocabulary_ranges(
        &self,
        args: &ModelArgs,
    ) -> (
        eredu_nn::VocabularyParallelRange,
        Option<eredu_nn::VocabularyParallelRange>,
    );
    /// Architecture-global units physically owned by the kernel.
    fn owned_units(&self) -> std::ops::Range<usize>;
    /// Admitted geometry for one global invocation.
    fn block(&self, args: &ModelArgs, index: usize) -> Result<super::BlockGeometry, Error>;
    /// Selected routed bank geometry, when ownership specializes the bank.
    fn routed_spec(&self, _index: usize) -> Option<eredu_nn::GroupedGatedProductSpec> {
        None
    }
    /// Ordinary execution retains global-state admission on a TP construction.
    fn ordinary_state_layout<'a>(
        &'a self,
        _args: &ModelArgs,
    ) -> Result<std::borrow::Cow<'a, StateLayout>, Error> {
        Ok(std::borrow::Cow::Borrowed(self.complete_state_layout()))
    }
    /// Parallel ingress and readout require admitted local geometry.
    fn parallel_state_layout(&self) -> Result<&StateLayout, Error> {
        Ok(self.complete_state_layout())
    }
    /// Maps one global invocation to its actual mutable-state ordinal.
    fn state_index(&self, index: usize) -> Result<usize, Error> {
        let owned = self.owned_units();
        if !owned.contains(&index) {
            return Err(Error::backend(format!(
                "Kimi unit {index} is outside owned range {owned:?}"
            )));
        }
        Ok(index - owned.start)
    }
}

/// Complete state and optional tensor-local KDA/MLA block geometry.
pub struct CompleteKimiGeometry {
    parallel: Option<std::sync::Arc<super::LocalGeometry>>,
    state: StateLayout,
    units: usize,
}
impl KimiGeometry for CompleteKimiGeometry {
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
                Error::backend(format!("missing rank-local Kimi block geometry {index}"))
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
            .ok_or_else(|| Error::backend("Kimi model was not built with local geometry"))
    }
}
impl KimiGeometry for PartitionLocalGeometry {
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
                "Kimi unit {index} is outside owned range {:?}",
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
        "kimi_linear",
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
    if definition.id() != "kimi_linear" {
        return Err(invalid(definition, "Kimi Linear component identity"));
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
    geometry: &impl KimiGeometry,
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
            "Kimi Linear retained state differs from global or selected geometry",
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
                "Kimi Linear retained state or vocabulary placement differs from selected geometry",
            ));
        }
        for index in geometry.owned_units() {
            if retained.block(index) != Some(&geometry.block(args, index)?) {
                return Err(invalid(
                    &definition,
                    format!("Kimi Linear retained block {index} placement differs from selected geometry"),
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

impl<B> KimiKernel<B>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
{
    /// Constructs only statics and blocks selected by an exact dense Kimi partition.
    pub fn from_partition<A>(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry, A>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedLayeredModel<B>, Error> {
        crate::operator_requirements::require::<B>(
            "Kimi Linear",
            crate::operator_requirements::KIMI_LINEAR,
        )?;
        args.validate().map_err(Error::backend)?;
        if args.num_nextn_predict_layers != 0 {
            return Err(Error::backend(
                "partitioned resident Kimi does not accept embedded prediction layers",
            ));
        }

        partition
            .local_geometry()
            .validate_for(&args)
            .map_err(Error::backend)?;
        let [group] = partition.groups() else {
            return Err(Error::backend(
                "partitioned Kimi must own exactly one execution group",
            ));
        };
        let owned = partition.local_geometry().owned_units();
        if group.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP
            || group.global_units() != owned
        {
            return Err(Error::backend(
                "Kimi construction range differs from selected partition",
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

impl<B, M, G> KimiKernel<B, M, G>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
{
    /// Normalized Kimi policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Exact local TP/PP geometry.
    pub const fn local_geometry(&self) -> &G {
        &self.geometry
    }

    /// Pinned modules physically retained by this ownership view.
    pub const fn static_modules(&self) -> &M {
        &self.static_modules
    }

    /// Mutably borrows the actual pinned modules for parameter editing.
    pub fn static_modules_mut(&mut self) -> &mut M {
        &mut self.static_modules
    }

    /// Constructs one owned global unit and rejects all other indices.
    pub fn construct_unit(
        &self,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::Block<B>, Error> {
        super::Block::new_with_geometry_and_routed_spec(
            &self.args,
            index,
            self.geometry.block(&self.args, index)?,
            self.geometry.routed_spec(index),
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
                "Kimi runtime state does not match its partition-local layout",
            ));
        }
        let sequence = hidden.dim(1);
        let requires_mask = self.geometry.owned_units().any(|unit| {
            self.args
                .layer_policy(unit)
                .is_some_and(|policy| policy.attention == AttentionKind::Mla)
        });
        let mask = if let Some(mask) = mask {
            Some(mask.clone())
        } else if sequence > 1 && requires_mask {
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
            &mut crate::decoder::ComponentInstrumentation::disabled(),
        )
    }

    fn finish_local_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut crate::decoder::ComponentInstrumentation<'_, B::Tensor>,
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

impl<B, M, G> eredu_runtime::ComponentKernel<B> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
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
    ) -> Result<Self::Unit, Self::Error> {
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

impl<B, M, G> eredu_runtime::ComponentPinnedResources<B> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
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

impl<B, S, M, G> eredu_runtime::ComponentEquations<B, S> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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
        let hidden = self
            .static_modules
            .decoder_resources_mut()
            .embedding
            .ok_or_else(|| Error::backend("Kimi partition does not own input embedding"))?
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
        crate::decoder::ComponentInstrumentation::apply_embedding(
            self.begin(input, state, context)?,
            observer,
        )
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
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.execute_routed_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            context,
            observer,
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
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.finish_local_instrumented(
            hidden,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut borrowed),
        )
    }
}

impl<B, S, M, G> eredu_runtime::ParallelComponentKernel<B, S> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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
                .ok_or_else(|| Error::backend("Kimi partition does not own input embedding"))?,
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
        crate::decoder::ComponentInstrumentation::apply_embedding(
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
        let pass = if hidden.dim(1) > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        };
        self.execute_parallel_routed_observed(
            address,
            unit,
            hidden,
            state,
            forward,
            pass,
            &mut eredu_runtime::ResidentExpertProvider,
            parallel,
            context,
            observer,
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
            &mut crate::decoder::ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S, M, G> eredu_runtime::RoutedComponentKernel<B, S> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
{
    fn routing_points(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
    ) -> Result<Option<eredu_runtime::RoutedObservationPoints>, Self::Error> {
        let path = format!("model.layers.{}", address.index());
        Ok(self.args.routed_observation_points(&path, address.index()))
    }
    fn routed_unit_observations(&self) -> bool {
        true
    }
    fn routed_sparse_observations(&self) -> bool {
        true
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
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let index = address.index();
        let path = format!("model.layers.{}", address.index());
        let points = self.args.routed_observation_points(&path, index);
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_partition_instrumented_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state
                .layer(self.local_state_ordinal(index)?)
                .map_err(Error::backend)?,
            None,
            context,
            &mut crate::decoder::ComponentInstrumentation::new(&path, &mut borrowed),
            |policy, normalized, context, instrumentation| {
                policy.forward_instrumented_with_provider(
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
}

impl<B, S, M, G> eredu_runtime::ParallelRoutedComponentKernel<B, S> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
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
                policy.forward_parallel_instrumented_with_provider(
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
}

impl<B, S, M, G> eredu_runtime::PartitionComponentKernel<B, S> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
{
    type Boundary = eredu_runtime::NoAuxiliaryBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(eredu_runtime::NoAuxiliaryBoundarySchema::new(
            self.args.hidden_size,
        ))
    }
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
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
            .ok_or_else(|| Error::backend("Kimi partition requires selected state geometry"))?;
        if selected.parallel.is_some() {
            self.geometry.parallel_state_layout()?;
        }
        let context = selected.native;
        let hidden = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let embedding = self
                    .static_modules
                    .decoder_resources_mut()
                    .embedding
                    .ok_or_else(|| Error::backend("Kimi partition does not own input embedding"))?;
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
            forward = crate::decoder::ComponentInstrumentation::apply_embedding(forward, observer)?;
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
        if owns_output {
            let output = match parallel {
                Some(parallel) => self.finish_parallel_observed(
                    hidden, state, forward, parallel, context, observer,
                )?,
                None => self.finish_observed(hidden, state, forward, context, observer)?,
            };
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: eredu_runtime::NoAuxiliaryBoundary,
            })
        }
    }
}

impl<B, M, G> eredu_runtime::TextComponentKernel<B> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S, M, G> eredu_runtime::PipelineComponentKernel<B, S> for KimiKernel<B, M, G>
where
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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
    fn routed_tensor_reductions(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Self::Error> {
        let unit = address.index();
        let policy = self.args.layer_schedule.get(unit).ok_or_else(|| {
            Error::backend(format!("Kimi has no collective order for unit {unit}"))
        })?;
        if routed != matches!(policy.feed_forward, super::FeedForwardPolicy::SparseMoe) {
            return Err(Error::backend(format!(
                "Kimi routed collective selection drifted at unit {unit}"
            )));
        }
        // KDA/MLA each reduce their row-parallel output before feed-forward.
        // A sparse unit then reduces both its routed result and its shared
        // expert projection; a dense unit has only the ordinary FFN reduction.
        Ok(eredu_runtime::RoutedTensorReductions::hidden(
            1,
            if routed { 2 } else { 1 },
        ))
    }
}

impl<B, M, G> KimiKernel<B, M, G>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
    M: DecoderStaticResources<B>,
    G: KimiGeometry,
{
    /// Begins from embedded input using explicit admitted state geometry.
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
    /// Executes one owned block through its actual local state slot.
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
        S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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
        S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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
    /// Executes the same owned block using tensor-parallel operators.
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
        S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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
        S::LayerState: RuntimeStateComponents<B> + CompressedAttentionCache<B::Tensor>,
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

impl<B> CompleteKimiKernel<B>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + BlockwiseAttentionBackend,
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
        crate::operator_requirements::require::<B>(
            "Kimi Linear",
            crate::operator_requirements::KIMI_LINEAR,
        )?;
        args.validate().map_err(Error::backend)?;
        if let Some(local) = &local {
            local.validate_for(&args).map_err(Error::backend)?;
        }
        let geometry = CompleteKimiGeometry {
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
    /// Returns admitted tensor-local block and state geometry.
    pub fn parallel_geometry(&self) -> Option<&super::LocalGeometry> {
        self.geometry.parallel.as_deref()
    }
    /// Shares authoritative geometry with a backend residency policy.
    pub fn shared_parallel_geometry(&self) -> Option<std::sync::Arc<super::LocalGeometry>> {
        self.geometry.parallel.as_ref().map(std::sync::Arc::clone)
    }
    /// Returns the ordinary global-state declaration for low-level execution.
    pub fn state_layout(&self) -> Result<StateLayout, Error> {
        super::state_layout(&self.args).map_err(Error::backend)
    }
}
