use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{LayerRuntimeState, SelectedReplicatedTextRealization};

use super::*;
use crate::{
    composite_partitioned::*,
    partitioned_execution::*,
    prediction_extension::{MaterializedPredictionExtension, PredictionExtensionMaterializer},
    replicated_text::*,
    routed_text::*,
};

/// Architecture-owned authority for publishing an already selected prediction placement.
#[doc(hidden)]
pub struct RetainedPredictionBinding {
    facts: PredictionBinding,
    pub(crate) placement: (
        crate::prediction_extension::PredictionPlacementSlot,
        ParallelRankTopology,
    ),
}

impl RetainedPredictionBinding {
    pub(crate) fn new(
        selected: eredu_runtime::SelectedSpeculativeRealization,
        capability: CapabilityEstimate,
        placement: crate::prediction_extension::PredictionPlacementSlot,
        topology: ParallelRankTopology,
    ) -> Self {
        Self {
            facts: PredictionBinding::new(selected, capability),
            placement: (placement, topology),
        }
    }

    pub(crate) fn into_facts(self) -> PredictionBinding {
        self.facts
    }
}

impl sealed::Sealed for WithoutPrediction {}
impl<M, S, V> sealed::Sealed for PredictionMechanisms<M, S, V> {}

/// Architecture-owned optional prediction construction for a selected route.
#[doc(hidden)]
pub trait PredictionConstruction<B: NeuralBackend, S, C, E, F>: sealed::Sealed {
    /// Consumes mutually agreeing selected prediction and source roles.
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        branch: PreparedConstructionBranch<S, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>>;
}
impl<B: NeuralBackend, S, C, E, F> PredictionConstruction<B, S, C, E, F> for WithoutPrediction {
    fn construct(
        self,
        _: PreparedPredictionSelection,
        _: PreparedConstructionBranch<S, C>,
        _: &<B::Tensor as Tensor>::Context,
        _: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        Err(PreparedExecutionError::UnavailablePrediction)
    }
}

fn materialize<B, M, E>(
    prediction: PreparedPredictionSelection,
    source_context: &<B::Tensor as Tensor>::Context,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(MaterializedPredictionExtension<B, M>, PredictionBinding), PreparedExecutionError<E>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B, Error = E>,
{
    let prepared = crate::prediction_extension::prepare::<B>(
        &prediction.parameters,
        prediction.topology,
        source_context,
        context,
    )
    .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?
    .with_residency(prediction.residency);
    let placement = prepared.retained_placement(prediction.topology);
    let mut materialization = M::bind_source(prediction.source, context, source_context);
    let extension = prepared
        .materialize::<M>(&mut materialization)
        .map_err(PreparedExecutionError::Backend)?;
    prediction.placement.set(placement).map_err(|_| {
        PreparedExecutionError::Architecture(
            "prediction placement was already bound for these prepared sources".into(),
        )
    })?;
    Ok((
        extension,
        PredictionBinding::new(prediction.realization, prediction.capability),
    ))
}

impl<B, M, VF, V, C, E, F> PredictionConstruction<B, SelectedReplicatedTextRealization, C, E, F>
    for PredictionMechanisms<M, (), VF>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B, Error = F>,
    VF: FnOnce(PredictionBinding) -> V,
    V: ReplicatedPredictionProfileDispatcher<B, M, Output = E, Error = F>,
    <V::State as LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        branch: PreparedConstructionBranch<SelectedReplicatedTextRealization, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let (extension, binding) = materialize::<B, M, _>(prediction, source_context, context)?;
        dispatch_replicated_prediction_target_architecture::<B, M, _>(
            branch.inspection.architecture_plan(),
            branch.selected,
            extension,
            branch.target,
            context,
            (self.visitor)(binding),
        )
        .map_err(replicated_error)
    }
}

impl<B, M, S, PS, VF, V, C, E, F> PredictionConstruction<B, SelectedRoutedTextRealization, C, E, F>
    for PredictionMechanisms<M, (S, PS), VF>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B, Error = F>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    VF: FnOnce(PredictionBinding) -> V,
    V: RoutedPredictionProfileDispatcher<
        B,
        M,
        crate::SelectedRoutedBank,
        GatedState = S,
        PoolingState = PS,
        Output = E,
        Error = F,
    >,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        branch: PreparedConstructionBranch<SelectedRoutedTextRealization, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let (extension, binding) = materialize::<B, M, _>(prediction, source_context, context)?;
        dispatch_routed_prediction_target_architecture::<B, M, _>(
            &branch.inspection,
            branch.selected,
            extension,
            branch.target,
            context,
            (self.visitor)(binding),
        )
        .map_err(routed_error)
    }
}

impl<B, M, S, VF, V, C, E, F> PredictionConstruction<B, SelectedCompositeTextRealization, C, E, F>
    for PredictionMechanisms<M, S, VF>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    M: PredictionExtensionMaterializer<B, Error = F>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    VF: FnOnce(PredictionBinding) -> V,
    V: CompositePredictionTargetVisitor<
        B,
        S,
        M,
        crate::SelectedRoutedBank,
        crate::replicated_text::CompositeTextRequirements,
        crate::replicated_text::RoutedCompositeRequirements,
        Output = E,
        Error = F,
    >,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        branch: PreparedConstructionBranch<SelectedCompositeTextRealization, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let requirements = composite_text_requirements(&branch.inspection)
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let (extension, binding) = materialize::<B, M, _>(prediction, source_context, context)?;
        visit_composite_prediction_target_text_architecture::<B, S, M, _>(
            requirements,
            branch.selected,
            extension,
            branch.target,
            context,
            (self.visitor)(binding),
        )
        .map_err(replicated_error)
    }
}

impl<B, M, S, VF, V, C, E, F> PredictionConstruction<B, SelectedDensePartitionedExecution, C, E, F>
    for PredictionMechanisms<M, S, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B, Error = F>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    VF: FnOnce(PreparedPartitionPredictionResources<C>) -> V,
    V: PartitionedPredictionTargetVisitor<B, S, M, Output = E, Error = F>,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        mut branch: PreparedConstructionBranch<SelectedDensePartitionedExecution, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let partition = branch.partition_resources()?;
        let (extension, prediction) = materialize::<B, M, _>(prediction, source_context, context)?;
        visit_resident_partitioned_prediction_target_architecture::<B, S, M, _>(
            &branch.inspection,
            branch.selected,
            extension,
            branch.target,
            context,
            (self.visitor)(PreparedPartitionPredictionResources::new(
                partition, prediction,
            )),
        )
        .map_err(partitioned_error)
    }
}

impl<B, M, S, PS, VF, V, C, E, F>
    PredictionConstruction<B, SelectedRoutedPartitionedExecution, C, E, F>
    for PredictionMechanisms<M, (S, PS), VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B, Error = F>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    VF: FnOnce(PreparedPartitionPredictionResources<C>) -> V,
    V: RoutedPartitionedPredictionTargetProductionVisitor<
            B,
            S,
            M,
            crate::SelectedRoutedBank,
            Output = E,
            Error = F,
        > + RoutedPartitionedPredictionTargetProductionVisitor<
            B,
            PS,
            M,
            crate::SelectedRoutedBank,
            Output = E,
            Error = F,
        >,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        mut branch: PreparedConstructionBranch<SelectedRoutedPartitionedExecution, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let partition = branch.partition_resources()?;
        let (extension, prediction) = materialize::<B, M, _>(prediction, source_context, context)?;
        let visitor = (self.visitor)(PreparedPartitionPredictionResources::new(
            partition, prediction,
        ));
        dispatch_routed_partitioned_production(
            &branch.inspection,
            branch.selected,
            (branch.target, extension, visitor),
            |(target, extension, visitor), inspection, selected| {
                visit_routed_partitioned_prediction_target_production::<B, S, M, _>(
                    inspection, selected, extension, target, context, visitor,
                )
                .map_err(partitioned_error)
            },
            |(target, extension, visitor), inspection, selected| {
                visit_relu2_routed_partitioned_prediction_target_production::<B, S, M, _>(
                    inspection, selected, extension, target, context, visitor,
                )
                .map_err(partitioned_error)
            },
            |(target, extension, visitor), inspection, selected| {
                visit_pooling_routed_partitioned_prediction_target_production::<B, PS, M, _>(
                    inspection, selected, extension, target, context, visitor,
                )
                .map_err(partitioned_error)
            },
        )
    }
}

impl<B, M, S, VF, V, C, E, F>
    PredictionConstruction<B, SelectedCompositePartitionedExecution, C, E, F>
    for PredictionMechanisms<M, S, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B, Error = F>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    VF: FnOnce(PreparedPartitionPredictionResources<C>) -> V,
    V: AuthoritativeCompositePartitionPredictionTargetVisitor<
        B,
        S,
        M,
        crate::SelectedRoutedBank,
        Output = E,
        Error = F,
    >,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        mut branch: PreparedConstructionBranch<SelectedCompositePartitionedExecution, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let partition = branch.partition_resources()?;
        let (extension, prediction) = materialize::<B, M, _>(prediction, source_context, context)?;
        visit_authoritative_composite_prediction_target_partition::<B, S, M, _>(
            branch.selected,
            extension,
            context,
            (self.visitor)(PreparedPartitionPredictionResources::new(
                partition, prediction,
            )),
        )
        .map_err(composite_partitioned_error)
    }
}
