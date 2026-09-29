//! Shared vision and target equations under retained TP/PP/EP placement.
use super::super::{
    input::{RequestBoundary, RequestBoundarySchema},
    target::TargetTensorPartition,
};
use super::*;
use crate::composite_execution::{
    CompositeArchitecture, ParallelCompositeArchitecture, PreparedCompositeInput,
};
use eredu_nn::{GroupedGatedProductSpec, TensorParallelGroupedNeuralBackend};

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ConditionalModel<B> {
    /// Constructs modules from the exact retained target, vision and state placement.
    pub fn new_partitioned(
        spec: BoundTargetSpec,
        target: TargetTensorPartition,
        ingress: MediaIngress,
        vision: VisionConfig,
        topology: eredu_core::ParallelRankTopology,
        layout: &LocalModelLayout,
        state: &PartitionState,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if topology.tensor_parallel_rank() != target.rank()
            || topology.tensor_parallel_size() != target.ranks()
        {
            return Err(Error::backend(
                "conditional retained topology differs from target tensor placement",
            ));
        }
        let mut model = Self::new(spec.clone(), ingress, vision.clone(), context)?;
        let parameters = model
            .parameter_description(context)?
            .with_partition_layout(topology, layout.clone())
            .map_err(Error::backend)?;
        let blocks = (0..vision.layer_count())
            .map(|index| {
                crate::qwen::vision::local_block_geometry(&vision, "model.visual", index, layout)
                    .map_err(Error::backend)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mergers = crate::qwen::vision::local_merger_widths(&vision, "model.visual", layout)
            .map_err(Error::backend)?;
        model.modules.target = TargetModel::new_tensor_parallel(spec, target, context)?;
        model.modules.target.set_partition_state(state)?;
        model.modules.vision =
            VisionStatic::new_parallel_with_root(vision, "model.visual", &mergers, context)?;
        model.vision_local = Some(blocks);
        model.vision_input_owner = topology.pipeline_parallel_rank() == 0;
        model.partition_state = Some(state.clone());
        model.global_parameters = Some(parameters);
        Ok(model)
    }

    /// Binds local routed operators while preserving global router identities.
    pub fn set_expert_realization(
        &mut self,
        realization: &crate::ExpertRealizationPlan<GroupedGatedProductSpec>,
    ) -> Result<(), Error> {
        self.modules.target.set_expert_realization(realization)
    }

    fn validate_parallel(&self, parallel: &B::ParallelContext) -> Result<(), Error> {
        let target = self.modules.target.tensor_partition().ok_or_else(|| {
            Error::backend("conditional parallel execution has no retained geometry")
        })?;
        if B::parallel_rank(parallel) != target.rank()
            || B::parallel_size(parallel) != target.ranks()
        {
            return Err(Error::backend(
                "conditional parallel group differs from retained geometry",
            ));
        }
        Ok(())
    }

    fn validate_partition_state_origin(&self, global_state_offset: usize) -> Result<(), Error> {
        let expected = self
            .partition_state
            .as_ref()
            .map_or(0, PartitionState::global_layer_offset);
        if global_state_offset != expected {
            return Err(Error::backend(
                "conditional partition state origin differs from retained placement",
            ));
        }
        Ok(())
    }

    fn wrap_target(
        next: LayeredForwardState<B::Tensor, TargetForward<B::Tensor>>,
    ) -> LayeredForwardState<B::Tensor, ConditionalForward<B::Tensor>> {
        LayeredForwardState {
            hidden: next.hidden,
            context: ConditionalForward {
                media: None,
                range: None,
                vision: None,
                initial: None,
                projected: None,
                target: Some(next.context),
                incoming_target: None,
            },
        }
    }
}

impl<B, S> ParallelLayeredArchitecture<B, S> for ConditionalModel<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn parallel_observation_hooks(&self) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.validate_parallel(parallel)?;
        // Patch embeddings, vocabulary and residual ingress are replicated.
        self.begin_forward(input, state, context)
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
        self.forward_unit_parallel_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
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
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.forward_unit_parallel_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            ExpertPass::Prefill,
            &mut ResidentExpertProvider,
            parallel,
            context,
            observer,
        )
    }
    fn complete_execution_group_parallel(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.validate_parallel(parallel)?;
        if group == VISION_INDEX {
            let Some(vision) = forward.vision.as_mut() else {
                return Ok(hidden.clone());
            };
            let output = self
                .modules
                .vision
                .finish_parallel(hidden, vision, parallel, context)?;
            forward.projected = Some(output.embeddings.clone());
            Ok(output.embeddings)
        } else {
            self.complete_execution_group(group, hidden, state, forward, context)
        }
    }
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.validate_parallel(parallel)?;
        self.finish_forward(hidden, state, forward, context)
    }
    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.validate_parallel(parallel)?;
        self.finish_forward_observed(hidden, state, forward, context, observer)
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for ConditionalModel<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn parallel_routed_unit_observations(&self) -> bool {
        true
    }
    fn parallel_routed_sparse_observations(&self) -> bool {
        true
    }
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_unit_parallel_observed_with_provider(
            group,
            index,
            unit,
            hidden,
            state,
            forward,
            pass,
            provider,
            parallel,
            context,
            &mut NoopObserver,
        )
    }
    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        P: TensorParallelParameterProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.validate_parallel(parallel)?;
        self.check_unit(group, index)?;
        match (group, unit) {
            (VISION_INDEX, ConditionalUnit::Vision(block)) => {
                self.modules.vision.forward_block_parallel(
                    block,
                    index,
                    hidden,
                    forward
                        .vision
                        .as_mut()
                        .ok_or_else(|| Error::backend("missing vision context"))?,
                    parallel,
                    context,
                )
            }
            (TARGET_INDEX, ConditionalUnit::Target(unit)) => self
                .modules
                .target
                .forward_unit_parallel_observed_with_provider(
                    0,
                    index,
                    unit,
                    hidden,
                    state,
                    forward
                        .target
                        .as_mut()
                        .ok_or_else(|| Error::backend("missing target context"))?,
                    pass,
                    provider,
                    parallel,
                    context,
                    observer,
                ),
            _ => Err(Error::backend("conditional unit/group mismatch")),
        }
    }
}

impl<B, S> ParallelCompositeArchitecture<B, S> for ConditionalModel<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    fn begin_composite_forward_parallel<'a>(
        &mut self,
        input: PreparedCompositeInput<'a, B::Tensor, Self::InputPartPlan>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.validate_parallel(parallel)?;
        self.begin_composite_forward(input, state, context)
    }
}

impl<B, S> PartitionedLayeredArchitecture<B, S> for ConditionalModel<B>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B> + RuntimeAppendStreams<B>,
{
    type Boundary = RequestBoundarySchema;
    fn boundary_schema(&self) -> Result<Self::Boundary, Error> {
        self.modules.target.spec.boundary_schema()
    }
    fn partition_observation_hooks(
        &self,
        _: bool,
    ) -> eredu_runtime::inspection::ObservationHookSupport {
        eredu_runtime::inspection::ObservationHookSupport::internal(true, true, true)
            .with_routed_units(true)
    }
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, RequestBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        global_state_offset: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.validate_partition_state_origin(global_state_offset)?;
        // Composite entry identifies the global state origin. The retained
        // target already owns this exact local view and indexes it from zero.
        self.modules
            .target
            .begin_partition(input, mask, state, expected, 0, context)
            .map(Self::wrap_target)
    }
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<'a, B::Tensor, RequestBoundary<B::Tensor>>,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        global_state_offset: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        self.validate_parallel(parallel)?;
        self.validate_partition_state_origin(global_state_offset)?;
        self.modules
            .target
            .begin_partition_parallel(input, mask, state, expected, 0, parallel, context)
            .map(Self::wrap_target)
    }
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if group != TARGET_INDEX {
            return Err(Error::backend(
                "conditional decoder boundary entered a non-target group",
            ));
        }
        if let Some(auxiliary) = forward.incoming_target.take() {
            let input = LayeredPartitionInput::Hidden {
                hidden: initial.clone(),
                auxiliary,
            };
            let expected = state.layout().clone();
            let next = match parallel {
                Some(parallel) => self.modules.target.begin_partition_parallel(
                    input, None, state, &expected, 0, parallel, context,
                )?,
                None => self
                    .modules
                    .target
                    .begin_partition(input, None, state, &expected, 0, context)?,
            };
            forward.target = Some(next.context);
            return Ok(next.hidden);
        }
        Ok(initial.clone())
    }
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredPartitionOutput<B::Tensor, RequestBoundary<B::Tensor>>, Error> {
        self.finish_partition_observed(
            hidden,
            state,
            forward,
            owns_output,
            parallel,
            context,
            &mut NoopObserver,
        )
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
    ) -> Result<LayeredPartitionOutput<B::Tensor, RequestBoundary<B::Tensor>>, Error>
    where
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        self.modules.target.finish_partition_observed(
            hidden,
            state,
            forward
                .target
                .as_ref()
                .ok_or_else(|| Error::backend("missing target context"))?,
            owns_output,
            parallel,
            context,
            observer,
        )
    }
}
