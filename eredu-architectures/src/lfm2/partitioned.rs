//! Selected LFM2 equations over mixed attention/convolution geometry.

use eredu_nn::{
    AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, EmbeddingSpec, Error, LinearSpec,
    NormalizationConstructionSpec, ParameterSpec, Tensor,
};
use eredu_runtime::{
    ArchitectureParameterDescription, LayerRuntimeState, LayeredForwardState,
    LayeredPartitionInput, ParameterProvider, RuntimeStateComponents, StateLayout,
};

use eredu_runtime::PartitionComponentKernel as _;

use super::{ForwardContext, ModelArgs, PartitionLocalGeometry};
use crate::decoder::{
    ComponentInstrumentation, DecoderProjectionOperator, DecoderStaticResources, LayeredInput,
    PartitionStaticModules,
};

/// Shared execution adapter for one exact selected LFM2 ownership view.
pub type PartitionedLayeredModel<B> = eredu_runtime::ComponentAdapter<
    eredu_runtime::ComponentDefinition,
    Lfm2Kernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// LFM2 block equations and mixed attention/convolution geometry for one rank.
pub struct Lfm2Kernel<B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: ModelArgs,
    static_modules: PartitionStaticModules<B>,
    geometry: PartitionLocalGeometry,
}

pub(crate) fn component_definition(
    parameters: ArchitectureParameterDescription,
    state: StateLayout,
) -> Result<eredu_runtime::ComponentDefinition, Error> {
    let units = (0..parameters.unit_layout().len())
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
        .collect();
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

fn partition_static_modules<B, A>(
    args: &ModelArgs,
    geometry: &PartitionLocalGeometry,
    partition: &eredu_runtime::ArchitecturePartition<PartitionLocalGeometry, A>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PartitionStaticModules<B>, Error>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    let ownership = partition.ownership();
    let embedding_name = "model.embed_tokens.weight";
    let embeddings = (ownership.owns_input()
        || (ownership.owns_output() && args.tie_word_embeddings))
        .then(|| {
            B::vocabulary_parallel_embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: args.hidden_size,
                    weight: ParameterSpec::trainable(embedding_name).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        embedding_name,
                        args.weight_quantization_for(embedding_name).into(),
                    )?,
                },
                geometry.embedding_range().clone(),
                context,
            )
        })
        .transpose()?;
    let norm = ownership
        .owns_output()
        .then(|| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.norm_eps,
                    ParameterSpec::trainable("model.embedding_norm.weight")
                        .map_err(Error::backend)?,
                ),
                context,
            )
        })
        .transpose()?;
    let lm_head = (ownership.owns_output() && !args.tie_word_embeddings)
        .then(|| {
            let name = "lm_head.weight";
            B::vocabulary_parallel_linear(
                LinearSpec {
                    input: args.hidden_size,
                    output: args.vocab_size,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        name,
                        args.weight_quantization_for(name).into(),
                    )?,
                },
                geometry.output_range().cloned().ok_or_else(|| {
                    Error::backend("untied LFM2 output owner has no vocabulary range")
                })?,
                context,
            )
        })
        .transpose()?;
    Ok(PartitionStaticModules {
        embeddings,
        norm,
        lm_head,
    })
}

impl<B> Lfm2Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Constructs only modules already selected by one exact LFM2 partition.
    pub fn from_partition<A>(
        args: ModelArgs,
        parameters: &ArchitectureParameterDescription,
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
        let state = partition
            .state()
            .ok_or_else(|| Error::backend("partitioned LFM2 has no selected state"))?;
        let expected_state = partition
            .local_geometry()
            .complete_state_layout()
            .slice(owned.clone())
            .map_err(Error::backend)?;
        if state.global_layer_offset() != owned.start || state.layout() != &expected_state {
            return Err(Error::backend(
                "LFM2 partition state does not match its global unit range",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let definition =
            component_definition(parameters.clone(), geometry.complete_state_layout().clone())?;
        let static_modules = partition_static_modules(&args, &geometry, partition, context)?;
        let kernel = Self {
            args,
            static_modules,
            geometry,
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }

    /// Normalized LFM2 policy.
    pub const fn args(&self) -> &ModelArgs {
        &self.args
    }

    /// Exact local TP/PP geometry.
    pub const fn local_geometry(&self) -> &PartitionLocalGeometry {
        &self.geometry
    }

    /// Pinned modules physically allocated on this pipeline partition.
    pub const fn static_modules(&self) -> &PartitionStaticModules<B> {
        &self.static_modules
    }

    /// Constructs one owned block and rejects every other global index.
    pub fn construct_unit(
        &self,
        global_unit: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::Block<B>, Error> {
        let geometry = self.geometry.block(global_unit).ok_or_else(|| {
            Error::backend(format!(
                "LFM2 unit {global_unit} is not owned by local range {:?}",
                self.geometry.owned_units()
            ))
        })?;
        let routed_spec = self
            .geometry
            .expert_realization()
            .and_then(|plan| plan.unit_spec(crate::decoder::TARGET_EXECUTION_GROUP, global_unit))
            .cloned();
        super::Block::new_with_geometry_and_routed_spec(
            &self.args,
            global_unit,
            *geometry,
            routed_spec,
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
        self.static_modules
            .finish_instrumented(hidden, parallel, context, instrumentation)
    }

    fn local_state_ordinal(&self, global_unit: usize) -> Result<usize, Error> {
        let owned = self.geometry.owned_units();
        owned
            .contains(&global_unit)
            .then(|| global_unit - owned.start)
            .ok_or_else(|| Error::backend("LFM2 attempted state access for an unowned unit"))
    }
}

impl<B> eredu_runtime::ComponentKernel<B> for Lfm2Kernel<B>
where
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

impl<B> eredu_runtime::ComponentPinnedResources<B> for Lfm2Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Modules = PartitionStaticModules<B>;
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

impl<B, S> eredu_runtime::ComponentEquations<B, S> for Lfm2Kernel<B>
where
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
        let hidden = self
            .static_modules
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?
            .forward(input.tokens, context)?;
        self.begin_hidden(
            hidden,
            input.mask,
            state,
            self.geometry.complete_state_layout(),
            0,
            context,
        )
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
        let index = address.index();
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
        )
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
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        unit.forward_instrumented_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            &mut ComponentInstrumentation::new(&path, &mut observer),
            |policy, normalized, context, instrumentation| {
                policy.forward_feed_forward_observed(normalized, context, instrumentation)
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

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for Lfm2Kernel<B>
where
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
        let hidden = B::vocabulary_parallel_lookup(
            self.static_modules
                .embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("LFM2 partition does not own input embedding"))?,
            input.tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_hidden(
            hidden,
            input.mask,
            state,
            self.geometry.complete_state_layout(),
            0,
            context,
        )
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
        let index = address.index();
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
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

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for Lfm2Kernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
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
        let index = address.index();
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            |policy, normalized, context| {
                policy.forward_with_provider(normalized, pass, context, provider)
            },
        )
    }
}

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for Lfm2Kernel<B>
where
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
        let index = address.index();
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_parallel_with_feed_forward(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            |policy, normalized, parallel, context| {
                policy.forward_parallel_with_provider(normalized, pass, parallel, context, provider)
            },
        )
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for Lfm2Kernel<B>
where
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
                let embedding =
                    self.static_modules.embeddings.as_mut().ok_or_else(|| {
                        Error::backend("LFM2 partition does not own input embedding")
                    })?;
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

impl<B> eredu_runtime::TextComponentKernel<B> for Lfm2Kernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        LayeredInput { tokens, mask }
    }
}

impl<B, S> eredu_runtime::PipelineComponentKernel<B, S> for Lfm2Kernel<B>
where
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
