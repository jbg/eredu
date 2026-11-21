use eredu_nn::NeuralBackend;
use eredu_runtime::LayerRuntimeState;

use super::*;
use crate::{composite_partitioned::*, partitioned_execution::*};

impl<B: NeuralBackend, S, V, P> sealed::Sealed for PartitionedDenseRoute<'_, B, S, V, P> {}

impl<B, S, VF, V, P, C, E, F> PreparedExecutionRoute<SelectedDensePartitionedExecution, C, E, F>
    for PartitionedDenseRoute<'_, B, S, VF, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    VF: FnOnce(PreparedPartitionResources<C>) -> V,
    V: PartitionedArchitectureVisitor<B, S, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedDensePartitionedExecution, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedDensePartitionedExecution, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        let resources = branch.partition_resources()?;
        visit_resident_partitioned_architecture::<B, S, _>(
            &branch.inspection,
            branch.selected,
            branch.target,
            self.context,
            (self.visitor)(resources),
        )
        .map_err(partitioned_error)
    }
}

impl<B: NeuralBackend, S, PS, SS, G, T, U, P> sealed::Sealed
    for PartitionedRoutedRoute<'_, B, S, PS, SS, G, T, U, P>
{
}

impl<B, S, PS, SS, GF, TF, UF, G, T, U, P, C, E, F>
    PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, E, F>
    for PartitionedRoutedRoute<'_, B, S, PS, SS, GF, TF, UF, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    SS: LayerRuntimeState<B>,
    SS::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>,
    GF: FnOnce(PreparedPartitionResources<C>) -> G,
    TF: FnOnce(PreparedPartitionResources<C>) -> T,
    UF: FnOnce(PreparedPartitionResources<C>) -> U,
    G: RoutedPartitionedProductionVisitor<B, S, crate::SelectedRoutedBank, Output = E, Error = F>,
    T: RoutedPartitionedProductionVisitor<B, PS, crate::SelectedRoutedBank, Output = E, Error = F>,
    U: RoutedPartitionedProductionVisitor<B, SS, crate::SelectedRoutedBank, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedRoutedPartitionedExecution, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedRoutedPartitionedExecution, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        let resources = branch.partition_resources()?;
        if let Some(retained) = branch.retained_construction.take() {
            return match retained {
                RetainedArchitectureConstruction::Qwen4Partition(prepared) => {
                    let visitor = (self.streams)(resources);
                    if let Some(binding) = branch.retained_prediction.take() {
                        prepared.visit_prediction::<B, SS, U>(self.context, visitor, binding)
                    } else {
                        prepared.visit::<B, SS, U>(self.context, visitor)
                    }
                    .map_err(partitioned_error)
                }
                _ => Err(PreparedExecutionError::Architecture(
                    "partitioned route received nonpartitioned construction authority".into(),
                )),
            };
        }
        dispatch_routed_partitioned_production(
            &branch.inspection,
            branch.selected,
            (branch.target, resources, self.gated, self.pooling),
            |(target, resources, gated, _), inspection, selected| {
                visit_routed_partitioned_production::<B, S, _>(
                    inspection,
                    selected,
                    target,
                    self.context,
                    gated(resources),
                )
                .map_err(partitioned_error)
            },
            |(target, resources, gated, _), inspection, selected| {
                visit_relu2_routed_partitioned_production::<B, S, _>(
                    inspection,
                    selected,
                    target,
                    self.context,
                    gated(resources),
                )
                .map_err(partitioned_error)
            },
            |(target, resources, _, pooling), inspection, selected| {
                visit_pooling_routed_partitioned_production::<B, PS, _>(
                    inspection,
                    selected,
                    target,
                    self.context,
                    pooling(resources),
                )
                .map_err(partitioned_error)
            },
        )
    }
}

impl<B: NeuralBackend, S, V, P> sealed::Sealed for PartitionedCompositeRoute<'_, B, S, V, P> {}

impl<B, S, VF, V, P, C, E, F> PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, E, F>
    for PartitionedCompositeRoute<'_, B, S, VF, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    VF: FnOnce(PreparedPartitionResources<C>) -> V,
    V: AuthoritativeCompositePartitionVisitor<
        B,
        S,
        crate::SelectedRoutedBank,
        Output = E,
        Error = F,
    >,
    P: PredictionConstruction<B, SelectedCompositePartitionedExecution, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedCompositePartitionedExecution, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(retained) = branch.retained_construction.take() {
            if branch.prediction.is_some() {
                return Err(PreparedExecutionError::UnavailablePrediction);
            }
            // Row providers and bound integer controls retain their own source
            // authority. The module visitor receives only its declared recipes,
            // as on the ordinary routed and replicated composite paths.
            branch.target = eredu_runtime::restrict_checkpoint_handoff(
                branch.selected.base().execution().requirements(),
                branch.target,
                eredu_runtime::SourceHandoffScope::Primary,
            )
            .map_err(|handoff| PreparedExecutionError::Architecture(handoff.to_string()))?;
            let resources = branch.partition_resources()?;
            return match retained {
                RetainedArchitectureConstruction::Qwen4ConditionalPartition(prepared) => {
                    let visitor = (self.visitor)(resources);
                    if let Some(binding) = branch.retained_prediction.take() {
                        prepared.visit_prediction::<B, S, V>(self.context, visitor, binding)
                    } else {
                        prepared.visit::<B, S, V>(self.context, visitor)
                    }
                    .map_err(composite_partitioned_error)
                }
                _ => Err(PreparedExecutionError::Architecture(
                    "composite partition route received incompatible construction authority".into(),
                )),
            };
        }
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        let resources = branch.partition_resources()?;
        visit_authoritative_composite_partition::<B, S, _>(
            branch.selected,
            self.context,
            (self.visitor)(resources),
        )
        .map_err(composite_partitioned_error)
    }
}
