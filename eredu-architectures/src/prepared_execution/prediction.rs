use std::marker::PhantomData;

use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{LayerRuntimeState, SelectedReplicatedTextRealization};

use super::*;
use crate::{
    composite_partitioned::*,
    partitioned_execution::*,
    prediction_extension::{
        MaterializedPredictionExtension, PredictionExtensionMaterializer,
        PreparedPredictionExtension,
    },
    replicated_text::*,
    routed_text::*,
};

/// Exact architecture-owned facts used by a native prediction target visitor.
pub struct PredictionBinding {
    selected: eredu_runtime::SelectedSpeculativeRealization,
    capability: CapabilityEstimate,
}
impl PredictionBinding {
    /// Selected speculative contract paired with this materialized extension.
    pub const fn selected(&self) -> &eredu_runtime::SelectedSpeculativeRealization {
        &self.selected
    }
    /// Architecture capability estimate including the selected prediction extension.
    pub const fn capability(&self) -> &CapabilityEstimate {
        &self.capability
    }
}

/// An absent optional prediction construction capability.
#[derive(Clone, Copy, Debug, Default)]
pub struct WithoutPrediction;
impl sealed::Sealed for WithoutPrediction {}

/// Native extension materialization and a typed target visitor factory.
///
/// Construction routes create this value through their `with_prediction` methods.
pub struct PredictionMechanisms<M, S, F, V> {
    materialize: F,
    visitor: V,
    marker: PhantomData<fn() -> (M, S)>,
}
impl<M, S, F, V> PredictionMechanisms<M, S, F, V> {
    pub(super) fn new(materialize: F, visitor: V) -> Self {
        Self {
            materialize,
            visitor,
            marker: PhantomData,
        }
    }
}
impl<M, S, F, V> sealed::Sealed for PredictionMechanisms<M, S, F, V> {}

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

fn materialize<B, M, F, E>(
    prediction: PreparedPredictionSelection,
    source_context: &<B::Tensor as Tensor>::Context,
    context: &<B::Tensor as Tensor>::Context,
    mechanism: F,
) -> Result<(MaterializedPredictionExtension<B, M>, PredictionBinding), PreparedExecutionError<E>>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    F: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, E>,
{
    let prepared = crate::prediction_extension::prepare::<B>(
        &prediction.extension,
        prediction.topology,
        &prediction.tasks,
        source_context,
        context,
    )
    .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
    let extension =
        mechanism(prepared, prediction.source).map_err(PreparedExecutionError::Backend)?;
    Ok((
        extension,
        PredictionBinding {
            selected: prediction.realization,
            capability: prediction.capability,
        },
    ))
}

impl<B, M, MF, VF, V, C, E, F> PredictionConstruction<B, SelectedReplicatedTextRealization, C, E, F>
    for PredictionMechanisms<M, (), MF, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    MF: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, F>,
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
        let (extension, binding) =
            materialize::<B, M, _, _>(prediction, source_context, context, self.materialize)?;
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

impl<B, M, S, PS, MF, VF, V, C, E, F>
    PredictionConstruction<B, SelectedRoutedTextRealization, C, E, F>
    for PredictionMechanisms<M, (S, PS), MF, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    MF: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, F>,
    VF: FnOnce(PredictionBinding) -> V,
    V: RoutedPredictionProfileDispatcher<
        B,
        M,
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
        let (extension, binding) =
            materialize::<B, M, _, _>(prediction, source_context, context, self.materialize)?;
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

impl<B, M, S, MF, VF, V, C, E, F>
    PredictionConstruction<B, SelectedCompositeTextRealization, C, E, F>
    for PredictionMechanisms<M, S, MF, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend
        + Clone,
    M: PredictionExtensionMaterializer<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    MF: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, F>,
    VF: FnOnce(PredictionBinding) -> V,
    V: CompositePredictionTargetVisitor<B, S, M, Output = E, Error = F>,
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
        let (extension, binding) =
            materialize::<B, M, _, _>(prediction, source_context, context, self.materialize)?;
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

impl<B, M, S, MF, VF, V, C, E, F>
    PredictionConstruction<B, SelectedDensePartitionedExecution, C, E, F>
    for PredictionMechanisms<M, S, MF, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    MF: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, F>,
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
        let (extension, prediction) =
            materialize::<B, M, _, _>(prediction, source_context, context, self.materialize)?;
        visit_resident_partitioned_prediction_target_architecture::<B, S, M, _>(
            &branch.inspection,
            branch.selected,
            extension,
            branch.target,
            context,
            (self.visitor)(PreparedPartitionPredictionResources {
                partition,
                prediction,
            }),
        )
        .map_err(partitioned_error)
    }
}

impl<B, M, S, PS, MF, VF, V, C, E, F>
    PredictionConstruction<B, SelectedRoutedPartitionedExecution, C, E, F>
    for PredictionMechanisms<M, (S, PS), MF, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    MF: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, F>,
    VF: FnOnce(PreparedPartitionPredictionResources<C>) -> V,
    V: RoutedPartitionedPredictionTargetProductionVisitor<B, S, M, Output = E, Error = F>
        + RoutedPartitionedPredictionTargetProductionVisitor<B, PS, M, Output = E, Error = F>,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        mut branch: PreparedConstructionBranch<SelectedRoutedPartitionedExecution, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let partition = branch.partition_resources()?;
        let (extension, prediction) =
            materialize::<B, M, _, _>(prediction, source_context, context, self.materialize)?;
        let visitor = (self.visitor)(PreparedPartitionPredictionResources {
            partition,
            prediction,
        });
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

impl<B, M, S, MF, VF, V, C, E, F>
    PredictionConstruction<B, SelectedCompositePartitionedExecution, C, E, F>
    for PredictionMechanisms<M, S, MF, VF>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: PredictionExtensionMaterializer<B>,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    MF: FnOnce(
        PreparedPredictionExtension<B>,
        SharedCheckpointSource,
    ) -> Result<MaterializedPredictionExtension<B, M>, F>,
    VF: FnOnce(PreparedPartitionPredictionResources<C>) -> V,
    V: AuthoritativeCompositePartitionPredictionTargetVisitor<B, S, M, Output = E, Error = F>,
{
    fn construct(
        self,
        prediction: PreparedPredictionSelection,
        mut branch: PreparedConstructionBranch<SelectedCompositePartitionedExecution, C>,
        source_context: &<B::Tensor as Tensor>::Context,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<E, PreparedExecutionError<F>> {
        let partition = branch.partition_resources()?;
        let (extension, prediction) =
            materialize::<B, M, _, _>(prediction, source_context, context, self.materialize)?;
        visit_authoritative_composite_prediction_target_partition::<B, S, M, _>(
            branch.selected,
            extension,
            context,
            (self.visitor)(PreparedPartitionPredictionResources {
                partition,
                prediction,
            }),
        )
        .map_err(composite_partitioned_error)
    }
}
