//! Exact TP/PP target construction and retained prediction-shared roles for Nemotron-H.

use eredu_nn::{AttentionCache, EmbeddingLookupPolicy, EmbeddingOperator, Error, Tensor};
use eredu_runtime::PartitionComponentKernel as _;
use eredu_runtime::{
    LayerRuntimeState, LayeredForwardState,
    LayeredPartitionInput, ParameterProvider, RuntimeStateComponents, StateLayout,
};

use super::{
    EmbeddedInput, LayerPolicy, ModelArgs, PartitionLocalGeometry, RetainedValues, TargetBoundary,
    TargetBoundarySchema, TargetReadout,
};
use crate::decoder::{ComponentInstrumentation, DecoderStaticResources, PartitionStaticModules};

/// Request-local target values retained across one partitioned pass.
pub struct PartitionedForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    target_capture: Option<T>,
    readout: TargetReadout,
}

/// Shared typed adapter for one selected Nemotron-H target ownership view.
pub type PartitionedLayeredModel<B> = eredu_runtime::ComponentAdapter<
    std::sync::Arc<eredu_runtime::ComponentDefinition>,
    NemotronKernel<B>,
    eredu_runtime::KernelParameterModules,
>;

/// Selected target equations, retained prediction context and recurrent geometry.
pub struct NemotronKernel<B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    args: ModelArgs,
    static_modules: PartitionStaticModules<B>,
    geometry: PartitionLocalGeometry,
    boundary: TargetBoundarySchema,
}


impl<B> NemotronKernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    /// Constructs only the statics and target units selected by one exact partition.
    pub fn from_partition(
        args: ModelArgs,
        definition: std::sync::Arc<eredu_runtime::ComponentDefinition>,
        partition: &eredu_runtime::ArchitecturePartition<
            PartitionLocalGeometry,
            TargetBoundarySchema,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PartitionedLayeredModel<B>, Error> {
        crate::operator_requirements::require::<B>(
            "Nemotron-H",
            crate::operator_requirements::NEMOTRON_H,
        )?;
        args.validate().map_err(Error::backend)?;
        if partition.boundary_schema().hidden_size() != args.hidden_size {
            return Err(Error::backend(
                "Nemotron-H partition boundary width differs from its configuration",
            ));
        }
        if args.num_nextn_predict_layers != 0 {
            return Err(Error::backend(
                "partitioned resident Nemotron-H does not accept embedded prediction",
            ));
        }

        partition
            .local_geometry()
            .validate_for(&args)
            .map_err(Error::backend)?;
        let [group] = partition.groups() else {
            return Err(Error::backend(
                "partitioned Nemotron-H must own exactly one execution group",
            ));
        };
        let owned = partition.local_geometry().owned_units();
        if group.group().as_str() != crate::decoder::TARGET_EXECUTION_GROUP
            || group.global_units() != owned
        {
            return Err(Error::backend(
                "Nemotron-H construction range differs from selected partition",
            ));
        }
        let geometry = partition.local_geometry().clone();
        let definition = super::parameters::select_definition(
            &args,
            definition,
            geometry.complete_state_layout(),
            (
                geometry.embedding_range().clone(),
                geometry.output_range().cloned(),
            ),
            geometry
                .owned_units()
                .map(|index| (index, *geometry.unit(index).expect("validated local unit"))),
        )?;
        let selection = definition.select_partition(partition)?;
        let static_modules = PartitionStaticModules::from_selected_spec(
            super::static_spec(&args, super::FinalNormalization::Float32),
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
            boundary: *partition.boundary_schema(),
        };
        eredu_runtime::ComponentAdapter::with_partition_resources::<B, _, _>(
            definition, kernel, partition,
        )
        .map_err(Error::from)
    }

    /// Normalized dense prediction-free Nemotron-H policy.
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

    /// Constructs one owned global target unit and rejects all other indices.
    pub fn construct_unit(
        &self,
        global_unit: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::Block<B>, Error> {
        let geometry = self.geometry.unit(global_unit).ok_or_else(|| {
            Error::backend(format!(
                "Nemotron-H unit {global_unit} is not owned by local range {:?}",
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

    fn local_state_ordinal(&self, global_unit: usize) -> Result<usize, Error> {
        let owned = self.geometry.owned_units();
        if owned.contains(&global_unit) {
            Ok(global_unit - owned.start)
        } else {
            Err(Error::backend(
                "Nemotron-H attempted state access for an unowned unit",
            ))
        }
    }

    fn mask_state_ordinal(&self, first_state_ordinal: usize) -> Option<usize> {
        self.geometry
            .owned_units()
            .enumerate()
            .find(|(_, global)| {
                matches!(
                    self.args.layer_schedule.get(*global),
                    Some(LayerPolicy::SelfAttention(_))
                )
            })
            .map(|(local, _)| first_state_ordinal + local)
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_hidden<S>(
        &self,
        tokens: B::Tensor,
        embedded: B::Tensor,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, PartitionedForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        self.boundary
            .validate_input(&tokens, &embedded)
            .map_err(Error::backend_source)?;
        super::mtp::validate_target_embeddings(&tokens, &hidden, self.args.hidden_size)?;
        if state.layout() != expected || first_state_ordinal >= expected.len() {
            return Err(Error::backend(
                "Nemotron-H runtime state does not match its partition-local layout",
            ));
        }
        let mask = if let Some(mask) = mask {
            Some(mask.clone())
        } else if hidden.dim(1) > 1 {
            self.mask_state_ordinal(first_state_ordinal)
                .map(|ordinal| {
                    B::causal_mask(
                        hidden.dim(1),
                        state.layer(ordinal).map_err(Error::backend)?.position(),
                        None,
                        context,
                    )
                })
                .transpose()?
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: PartitionedForwardContext {
                tokens,
                embedded,
                mask,
                target_capture: None,
                readout: TargetReadout::Logits,
            },
        })
    }

    fn finish_local(
        &mut self,
        hidden: &B::Tensor,
        readout: TargetReadout,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_local_instrumented(
            hidden,
            readout,
            parallel,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    fn finish_local_instrumented(
        &mut self,
        hidden: &B::Tensor,
        readout: TargetReadout,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        readout.apply(
            self.static_modules.decoder_resources_mut(),
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }

    pub(crate) fn project_prediction_instrumented(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.decoder_resources_mut().project(
            hidden,
            parallel,
            context,
            instrumentation,
        )
    }

    /// Embeds one prediction token through the target-owned vocabulary table.
    pub fn embed_prediction(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("Nemotron-H prediction target has no embedding"))?
            .forward(tokens, context)
    }

    /// Embeds one prediction token through the TP-sharded target vocabulary table.
    pub fn embed_prediction_parallel(
        &mut self,
        tokens: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::vocabulary_parallel_lookup(
            self.static_modules
                .embeddings
                .as_mut()
                .ok_or_else(|| Error::backend("Nemotron-H prediction target has no embedding"))?,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )
    }

    /// Projects prediction hidden state without the target-only final normalization.
    pub fn project_prediction(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.decoder_resources_mut().project(
            hidden,
            None,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Projects prediction hidden state through the TP-sharded target vocabulary table.
    pub fn project_prediction_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.decoder_resources_mut().project(
            hidden,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }
}

impl<B> eredu_runtime::ComponentKernel<B> for NemotronKernel<B>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    type Input<'a> = EmbeddedInput<'a, B::Tensor>;
    type Unit = super::Block<B>;
    type ForwardContext = PartitionedForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
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
    fn prediction_target_capture(forward: &Self::ForwardContext) -> Option<&B::Tensor> {
        forward.target_capture.as_ref()
    }
    fn prediction_target_placeholder_shape(
        &self,
        forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(Some(vec![
            forward.tokens.dim(0),
            forward.tokens.dim(1),
            self.args.hidden_size,
        ]))
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
        _address: eredu_runtime::ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.tokens),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            None,
        ])
    }
}

impl<B> eredu_runtime::ComponentPinnedResources<B> for NemotronKernel<B>
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

impl<B, S> eredu_runtime::ComponentEquations<B, S> for NemotronKernel<B>
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
        let (tokens, mask) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, mask),
            EmbeddedInput::TargetEmbedded {
                tokens,
                embedded,
                mask,
                readout,
            } => {
                let mut forward = self.begin_hidden(
                    tokens.clone(),
                    embedded.clone(),
                    embedded.clone(),
                    mask,
                    state,
                    self.geometry.complete_state_layout(),
                    0,
                    context,
                )?;
                forward.context.readout = readout;
                return Ok(forward);
            }
            EmbeddedInput::Draft { .. } => {
                return Err(Error::backend(
                    "partitioned resident Nemotron-H does not execute draft input",
                ))
            }
        };
        let embedded = self
            .static_modules
            .embeddings
            .as_mut()
            .ok_or_else(|| Error::backend("Nemotron-H partition does not own input embedding"))?
            .forward(tokens, context)?;
        self.begin_hidden(
            tokens.clone(),
            embedded.clone(),
            embedded,
            mask,
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
        forward.context.embedded = forward.hidden.clone();
        Ok(forward)
    }
    fn complete_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 {
            forward.target_capture = Some(hidden.clone());
        }
        Ok(hidden.clone())
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
        unit.forward_observed_with_provider(
            &path,
            self.args.n_routed_experts,
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            observer,
            &mut eredu_runtime::ResidentExpertProvider,
        )
    }
    fn finish(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_local(hidden, forward.readout, None, context)
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
        self.finish_local_instrumented(
            hidden,
            forward.readout,
            None,
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S> eredu_runtime::ParallelComponentKernel<B, S> for NemotronKernel<B>
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
        let (tokens, mask) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, mask),
            EmbeddedInput::TargetEmbedded {
                tokens,
                embedded,
                mask,
                readout,
            } => {
                let mut forward = self.begin_hidden(
                    tokens.clone(),
                    embedded.clone(),
                    embedded.clone(),
                    mask,
                    state,
                    self.geometry.complete_state_layout(),
                    0,
                    context,
                )?;
                forward.context.readout = readout;
                return Ok(forward);
            }
            EmbeddedInput::Draft { .. } => {
                return Err(Error::backend(
                    "partitioned resident Nemotron-H does not execute draft input",
                ))
            }
        };
        let embedded = B::vocabulary_parallel_lookup(
            self.static_modules.embeddings.as_mut().ok_or_else(|| {
                Error::backend("Nemotron-H partition does not own input embedding")
            })?,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        self.begin_hidden(
            tokens.clone(),
            embedded.clone(),
            embedded,
            mask,
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
        forward.context.embedded = forward.hidden.clone();
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
            &mut eredu_runtime::ResidentExpertProvider,
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
        unit.forward_parallel_observed_with_provider(
            &path,
            self.args.n_routed_experts,
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            observer,
            &mut eredu_runtime::ResidentExpertProvider,
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
        self.finish_local(hidden, forward.readout, Some(parallel), context)
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
        self.finish_local_instrumented(
            hidden,
            forward.readout,
            Some(parallel),
            context,
            &mut ComponentInstrumentation::new("readout", &mut observer),
        )
    }
}

impl<B, S> eredu_runtime::RoutedComponentKernel<B, S> for NemotronKernel<B>
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
    fn execute_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: ParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        let index = address.index();
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_with_provider(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            provider,
        )
    }
    fn execute_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
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
        let ordinal = self.local_state_ordinal(index)?;
        unit.forward_observed_with_provider(
            &path,
            self.args.n_routed_experts,
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            context,
            observer,
            provider,
        )
    }
}

impl<B, S> eredu_runtime::ParallelRoutedComponentKernel<B, S> for NemotronKernel<B>
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
    fn execute_parallel_routed<P>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
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
        unit.forward_parallel(
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            provider,
        )
    }
    fn execute_parallel_routed_observed<P, O>(
        &mut self,
        address: eredu_runtime::ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _pass: eredu_runtime::ExpertPass,
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
        unit.forward_parallel_observed_with_provider(
            &path,
            self.args.n_routed_experts,
            hidden,
            forward.mask.as_ref(),
            state.layer(ordinal).map_err(Error::backend)?,
            parallel,
            context,
            observer,
            provider,
        )
    }
}

impl<B, S> eredu_runtime::PartitionComponentKernel<B, S> for NemotronKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    type Boundary = TargetBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Self::Error> {
        Ok(self.boundary)
    }
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
    }
    fn begin_partition(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TargetBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = selected.expected.ok_or_else(|| {
            Error::backend("Nemotron-H partition requires selected state geometry")
        })?;
        let context = selected.native;
        let (tokens, embedded, hidden, readout) = match input {
            LayeredPartitionInput::Tokens(tokens) => {
                let embedding = self.static_modules.embeddings.as_mut().ok_or_else(|| {
                    Error::backend("Nemotron-H partition does not own input embedding")
                })?;
                let embedded = match selected.parallel {
                    Some(parallel) => B::vocabulary_parallel_lookup(
                        embedding,
                        tokens,
                        EmbeddingLookupPolicy::Strict,
                        parallel,
                        context,
                    )?,
                    None => embedding.forward(tokens, context)?,
                };
                (
                    tokens.clone(),
                    embedded.clone(),
                    embedded,
                    TargetReadout::Logits.encode::<B::Tensor>(context)?,
                )
            }
            LayeredPartitionInput::Hidden { hidden, auxiliary } => {
                let (tokens, embedded, readout) = auxiliary.into_parts();
                (tokens, embedded, hidden, readout)
            }
        };
        let readout = TargetReadout::decode(&readout, context)?;
        let mut forward = self.begin_hidden(
            tokens,
            embedded,
            hidden,
            mask,
            state,
            expected,
            selected.first_state_ordinal,
            context,
        )?;
        forward.context.readout = readout;
        Ok(forward)
    }
    fn begin_partition_observed<O>(
        &mut self,
        input: LayeredPartitionInput<'_, B::Tensor, TargetBoundary<B::Tensor>>,
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
            forward.context.embedded = forward.hidden.clone();
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
    ) -> Result<
        eredu_runtime::LayeredPartitionOutput<B::Tensor, TargetBoundary<B::Tensor>>,
        Self::Error,
    > {
        if owns_output {
            Ok(eredu_runtime::LayeredPartitionOutput::Final {
                output: self.finish_local(hidden, forward.readout, parallel, context)?,
                retained: None,
            })
        } else {
            Ok(eredu_runtime::LayeredPartitionOutput::Boundary {
                hidden: hidden.clone(),
                auxiliary: TargetBoundary::new(
                    forward.tokens.clone(),
                    forward.embedded.clone(),
                    forward.readout.encode::<B::Tensor>(context)?,
                ),
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
    ) -> Result<
        eredu_runtime::LayeredPartitionOutput<B::Tensor, TargetBoundary<B::Tensor>>,
        Self::Error,
    >
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition(hidden, state, forward, false, parallel, context);
        }
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new("readout", &mut observer);
        let output = self.finish_local_instrumented(
            hidden,
            forward.readout,
            parallel,
            context,
            &mut instrumentation,
        )?;
        Ok(eredu_runtime::LayeredPartitionOutput::Final {
            output,
            retained: None,
        })
    }
}

impl<B> eredu_runtime::TextComponentKernel<B> for NemotronKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
{
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        EmbeddedInput::target(tokens, mask)
    }
}

impl<B, S> eredu_runtime::PipelineComponentKernel<B, S> for NemotronKernel<B>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn tensor_vocabulary_sharded(&self) -> bool {
        true
    }
    fn input_dimensions(&self, input: &Self::Input<'_>) -> Result<(i32, i32), Self::Error> {
        let tokens = match input {
            EmbeddedInput::Target { tokens, .. }
            | EmbeddedInput::TargetEmbedded { tokens, .. }
            | EmbeddedInput::Draft { tokens, .. } => tokens,
        };
        Ok((tokens.dim(0), tokens.dim(1)))
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
            EmbeddedInput::TargetEmbedded {
                readout: TargetReadout::NormalizedHidden,
                ..
            } => self.args.hidden_size,
            _ => self.args.vocab_size,
        };
        Ok(vec![batch, sequence, width])
    }
    fn begin_pipeline<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        incoming: Option<(
            B::Tensor,
            <Self::Boundary as eredu_runtime::ArchitectureBoundary>::Boundary<B::Tensor>,
        )>,
        state: &mut S,
        selected: eredu_runtime::ComponentPartitionContext<'_, B>,
        observer: &mut O,
    ) -> Result<eredu_runtime::LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: eredu_runtime::ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        let context = selected.native;
        let (tokens, mask, supplied) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, mask, None),
            EmbeddedInput::TargetEmbedded {
                tokens,
                embedded,
                mask,
                readout,
            } => (tokens, mask, Some((embedded, readout))),
            EmbeddedInput::Draft { tokens, .. } => (tokens, None, None),
        };
        let input = match incoming {
            Some((hidden, auxiliary)) => {
                eredu_runtime::LayeredPartitionInput::Hidden { hidden, auxiliary }
            }
            None => match supplied {
                Some((embedded, readout)) => eredu_runtime::LayeredPartitionInput::Hidden {
                    hidden: embedded.clone(),
                    auxiliary: TargetBoundary::new(
                        tokens.clone(),
                        embedded.clone(),
                        readout.encode::<B::Tensor>(context)?,
                    ),
                },
                None => eredu_runtime::LayeredPartitionInput::Tokens(tokens),
            },
        };
        self.begin_partition_observed(input, mask, state, selected, observer)
    }
    fn routed_tensor_reductions(
        &self,
        address: eredu_runtime::ExecutionUnitAddress,
        routed: bool,
    ) -> Result<eredu_runtime::RoutedTensorReductions, Self::Error> {
        let unit = address.index();
        let policy = self.args.layer_schedule.get(unit).ok_or_else(|| {
            Error::backend(format!(
                "Nemotron-H has no collective order for unit {unit}"
            ))
        })?;
        if routed != matches!(policy, LayerPolicy::SparseMoe) {
            return Err(Error::backend(format!(
                "Nemotron-H routed collective selection drifted at unit {unit}"
            )));
        }
        Ok(eredu_runtime::RoutedTensorReductions::hidden(
            0,
            if routed { 2 } else { 1 },
        ))
    }
}
