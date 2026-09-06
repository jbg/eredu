use std::marker::PhantomData;

use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::LayerRuntimeState;

use super::*;
use crate::{composite_partitioned::*, partitioned_execution::*};

/// Direct partitioned construction from an exact native visitor factory.
pub struct PartitionedDenseRoute<'a, B: NeuralBackend, S, V, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    visitor: V,
    prediction: P,
    state: PhantomData<fn() -> S>,
}
impl<'a, B: NeuralBackend, S, V> PartitionedDenseRoute<'a, B, S, V> {
    /// Binds a factory receiving only checked native communication and source roles.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        visitor: V,
    ) -> Self {
        Self {
            context,
            source_context,
            visitor,
            prediction: WithoutPrediction,
            state: PhantomData,
        }
    }
}
impl<'a, B: NeuralBackend, S, V, P> PartitionedDenseRoute<'a, B, S, V, P> {
    /// Adds native extension materialization and a partitioned prediction visitor factory.
    pub fn with_prediction<M, F, V2>(
        self,
        materialize: F,
        visitor: V2,
    ) -> PartitionedDenseRoute<'a, B, S, V, PredictionMechanisms<M, S, F, V2>> {
        PartitionedDenseRoute {
            context: self.context,
            source_context: self.source_context,
            visitor: self.visitor,
            prediction: PredictionMechanisms::new(materialize, visitor),
            state: PhantomData,
        }
    }
}
impl<B: NeuralBackend, S, V, P> sealed::Sealed for PartitionedDenseRoute<'_, B, S, V, P> {}

impl<B, S, VF, V, P, C, E, F> PreparedExecutionRoute<SelectedDensePartitionedExecution, C, E, F>
    for PartitionedDenseRoute<'_, B, S, VF, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
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

/// Routed partitioned construction with architecture-owned operator/state selection.
pub struct PartitionedRoutedRoute<'a, B: NeuralBackend, S, PS, G, T, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    gated: G,
    pooling: T,
    prediction: P,
    states: PhantomData<fn() -> (S, PS)>,
}
impl<'a, B: NeuralBackend, S, PS, G, T> PartitionedRoutedRoute<'a, B, S, PS, G, T> {
    /// Binds factories for mixed-state grouped execution and pooling execution.
    ///
    /// The first visitor implements both gated-product and ReLU-squared contracts.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        gated: G,
        pooling: T,
    ) -> Self {
        Self {
            context,
            source_context,
            gated,
            pooling,
            prediction: WithoutPrediction,
            states: PhantomData,
        }
    }

    /// Restricts construction to gated/pooling equations without requiring a ReLU visitor.
    ///
    /// This narrow route has no prediction materializer. Unavailable equations
    /// and extensions fail before architecture constructors or native binding.
    pub fn without_relu2(self) -> GatedPartitionedRoute<'a, B, S, PS, G, T> {
        GatedPartitionedRoute(self)
    }
}

/// A routed partition adapter that implements gated/pooling but not ReLU-squared execution.
pub struct GatedPartitionedRoute<'a, B: NeuralBackend, S, PS, G, T>(
    PartitionedRoutedRoute<'a, B, S, PS, G, T>,
);

impl<B: NeuralBackend, S, PS, G, T> sealed::Sealed for GatedPartitionedRoute<'_, B, S, PS, G, T> {}

impl<B, S, PS, GF, TF, G, T, C, E, F>
    PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, E, F>
    for GatedPartitionedRoute<'_, B, S, PS, GF, TF>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    GF: FnOnce(PreparedPartitionResources<C>) -> G,
    TF: FnOnce(PreparedPartitionResources<C>) -> T,
    G: RoutedPartitionedProductionVisitor<B, S, Output = E, Error = F>,
    T: RoutedPartitionedProductionVisitor<B, PS, Output = E, Error = F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedRoutedPartitionedExecution, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if branch.prediction.is_some() {
            return Err(PreparedExecutionError::UnavailablePrediction);
        }
        let resources = branch.partition_resources()?;
        dispatch_routed_partitioned_production(
            &branch.inspection,
            branch.selected,
            (branch.target, resources, self.0.gated, self.0.pooling),
            |(target, resources, gated, _), inspection, selected| {
                visit_routed_partitioned_production::<B, S, _>(
                    inspection,
                    selected,
                    target,
                    self.0.context,
                    gated(resources),
                )
                .map_err(partitioned_error)
            },
            |_, _, _| Err(PreparedExecutionError::UnavailableExecution),
            |(target, resources, _, pooling), inspection, selected| {
                visit_pooling_routed_partitioned_production::<B, PS, _>(
                    inspection,
                    selected,
                    target,
                    self.0.context,
                    pooling(resources),
                )
                .map_err(partitioned_error)
            },
        )
    }
}
impl<'a, B: NeuralBackend, S, PS, G, T, P> PartitionedRoutedRoute<'a, B, S, PS, G, T, P> {
    /// Adds native extension materialization and a typed prediction visitor factory.
    pub fn with_prediction<M, F, V>(
        self,
        materialize: F,
        visitor: V,
    ) -> PartitionedRoutedRoute<'a, B, S, PS, G, T, PredictionMechanisms<M, (S, PS), F, V>> {
        PartitionedRoutedRoute {
            context: self.context,
            source_context: self.source_context,
            gated: self.gated,
            pooling: self.pooling,
            prediction: PredictionMechanisms::new(materialize, visitor),
            states: PhantomData,
        }
    }
}
impl<B: NeuralBackend, S, PS, G, T, P> sealed::Sealed
    for PartitionedRoutedRoute<'_, B, S, PS, G, T, P>
{
}

impl<B, S, PS, GF, TF, G, T, P, C, E, F>
    PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, E, F>
    for PartitionedRoutedRoute<'_, B, S, PS, GF, TF, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    GF: FnOnce(PreparedPartitionResources<C>) -> G,
    TF: FnOnce(PreparedPartitionResources<C>) -> T,
    G: RoutedPartitionedProductionVisitor<B, S, Output = E, Error = F>
        + RoutedPartitionedProductionVisitor<B, S, eredu_nn::GroupedRelu2Spec, Output = E, Error = F>,
    T: RoutedPartitionedProductionVisitor<B, PS, Output = E, Error = F>,
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

/// Composite partitioned construction from checked native resources.
pub struct PartitionedCompositeRoute<'a, B: NeuralBackend, S, V, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    visitor: V,
    prediction: P,
    state: PhantomData<fn() -> S>,
}
impl<'a, B: NeuralBackend, S, V> PartitionedCompositeRoute<'a, B, S, V> {
    /// Binds a visitor factory receiving only checked communication and sources.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        visitor: V,
    ) -> Self {
        Self {
            context,
            source_context,
            visitor,
            prediction: WithoutPrediction,
            state: PhantomData,
        }
    }
}
impl<'a, B: NeuralBackend, S, V, P> PartitionedCompositeRoute<'a, B, S, V, P> {
    /// Adds native extension materialization and composite prediction binding.
    pub fn with_prediction<M, F, V2>(
        self,
        materialize: F,
        visitor: V2,
    ) -> PartitionedCompositeRoute<'a, B, S, V, PredictionMechanisms<M, S, F, V2>> {
        PartitionedCompositeRoute {
            context: self.context,
            source_context: self.source_context,
            visitor: self.visitor,
            prediction: PredictionMechanisms::new(materialize, visitor),
            state: PhantomData,
        }
    }
}
impl<B: NeuralBackend, S, V, P> sealed::Sealed for PartitionedCompositeRoute<'_, B, S, V, P> {}

impl<B, S, VF, V, P, C, E, F> PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, E, F>
    for PartitionedCompositeRoute<'_, B, S, VF, P>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    VF: FnOnce(PreparedPartitionResources<C>) -> V,
    V: AuthoritativeCompositePartitionVisitor<B, S, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedCompositePartitionedExecution, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedCompositePartitionedExecution, C>,
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
        visit_authoritative_composite_partition::<B, S, _>(
            branch.selected,
            self.context,
            (self.visitor)(resources),
        )
        .map_err(composite_partitioned_error)
    }
}
