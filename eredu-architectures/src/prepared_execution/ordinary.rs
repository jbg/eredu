use std::marker::PhantomData;

use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{LayerRuntimeState, SelectedReplicatedTextRealization};

use super::*;
use crate::{replicated_text::*, routed_text::*};

/// Ordinary key/value construction requiring only the base neural backend.
///
/// Other state profiles and prediction extensions are rejected before any model
/// constructor or binding visitor runs. Backends add a broader route only when
/// they implement its corresponding optional mechanisms.
pub struct KeyValueRoute<'a, B: NeuralBackend, S, V> {
    context: &'a <B::Tensor as Tensor>::Context,
    visitor: V,
    state: PhantomData<fn() -> S>,
}

impl<'a, B: NeuralBackend, S, V> KeyValueRoute<'a, B, S, V> {
    /// Binds the native context and one ordinary attention-state visitor.
    pub fn new(context: &'a <B::Tensor as Tensor>::Context, visitor: V) -> Self {
        Self {
            context,
            visitor,
            state: PhantomData,
        }
    }
}
impl<B: NeuralBackend, S, V> sealed::Sealed for KeyValueRoute<'_, B, S, V> {}

impl<B, S, V, C, E, F> PreparedExecutionRoute<SelectedReplicatedTextRealization, C, E, F>
    for KeyValueRoute<'_, B, S, V>
where
    B: NeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
    V: ReplicatedTextArchitectureVisitor<B, S, Output = E, Error = F>,
{
    fn construct(
        self,
        branch: PreparedConstructionBranch<SelectedReplicatedTextRealization, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if branch.prediction.is_some() {
            return Err(PreparedExecutionError::UnavailablePrediction);
        }
        dispatch_replicated_key_value_text_architecture::<B, S, V>(
            branch.inspection.architecture_plan(),
            branch.selected,
            branch.target,
            self.context,
            self.visitor,
        )
        .map_err(replicated_error)
    }
}

/// Ordinary replicated construction with architecture-selected typed state access.
pub struct ReplicatedRoute<'a, B: NeuralBackend, D, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    dispatcher: D,
    prediction: P,
}

impl<'a, B: NeuralBackend, D> ReplicatedRoute<'a, B, D> {
    /// Binds native contexts and a typed profile visitor without selecting a profile.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        dispatcher: D,
    ) -> Self {
        Self {
            context,
            source_context,
            dispatcher,
            prediction: WithoutPrediction,
        }
    }
}

impl<'a, B: NeuralBackend, D, P> ReplicatedRoute<'a, B, D, P> {
    /// Adds native extension materialization and a prediction-aware target visitor.
    pub fn with_prediction<M, F, V>(
        self,
        materialize: F,
        visitor: V,
    ) -> ReplicatedRoute<'a, B, D, PredictionMechanisms<M, (), F, V>> {
        ReplicatedRoute {
            context: self.context,
            source_context: self.source_context,
            dispatcher: self.dispatcher,
            prediction: PredictionMechanisms::new(materialize, visitor),
        }
    }
}
impl<B: NeuralBackend, D, P> sealed::Sealed for ReplicatedRoute<'_, B, D, P> {}

impl<B, D, P, C, E, F> PreparedExecutionRoute<SelectedReplicatedTextRealization, C, E, F>
    for ReplicatedRoute<'_, B, D, P>
where
    B: eredu_nn::BlockwiseAttentionBackend,
    D: ReplicatedTextProfileDispatcher<B, Output = E, Error = F>,
    <D::AttentionState as LayerRuntimeState<B>>::LayerState: eredu_nn::AttentionCache<B::Tensor>,
    <D::ComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_runtime::RuntimeStateComponents<B>,
    <D::AttentionComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    <D::CompressedState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor>,
    <D::CompressedComponentState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::CompressedAttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    P: PredictionConstruction<B, SelectedReplicatedTextRealization, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedReplicatedTextRealization, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        dispatch_replicated_text_architecture::<B, _>(
            branch.inspection.architecture_plan(),
            branch.selected,
            branch.target,
            self.context,
            self.dispatcher,
        )
        .map_err(replicated_error)
    }
}

/// Routed text construction with architecture-owned equation and state dispatch.
pub struct RoutedRoute<'a, B: NeuralBackend, S, PS, G, R, T, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    gated: G,
    relu2: R,
    pooling: T,
    prediction: P,
    states: PhantomData<fn() -> (S, PS)>,
}

impl<'a, B: NeuralBackend, S, PS, G, R, T> RoutedRoute<'a, B, S, PS, G, R, T> {
    /// Binds exact gated, ReLU-squared, and pooling visitors; architecture code chooses one.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        gated: G,
        relu2: R,
        pooling: T,
    ) -> Self {
        Self {
            context,
            source_context,
            gated,
            relu2,
            pooling,
            prediction: WithoutPrediction,
            states: PhantomData,
        }
    }
}

impl<'a, B: NeuralBackend, S, PS, G, R, T, P> RoutedRoute<'a, B, S, PS, G, R, T, P> {
    /// Adds native extension materialization and a routed prediction-profile visitor.
    pub fn with_prediction<M, F, V>(
        self,
        materialize: F,
        visitor: V,
    ) -> RoutedRoute<'a, B, S, PS, G, R, T, PredictionMechanisms<M, (S, PS), F, V>> {
        RoutedRoute {
            context: self.context,
            source_context: self.source_context,
            gated: self.gated,
            relu2: self.relu2,
            pooling: self.pooling,
            prediction: PredictionMechanisms::new(materialize, visitor),
            states: PhantomData,
        }
    }
}
impl<B: NeuralBackend, S, PS, G, R, T, P> sealed::Sealed for RoutedRoute<'_, B, S, PS, G, R, T, P> {}

impl<B, S, PS, G, R, T, P, C, E, F> PreparedExecutionRoute<SelectedRoutedTextRealization, C, E, F>
    for RoutedRoute<'_, B, S, PS, G, R, T, P>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    G: GatedRoutedTextArchitectureVisitor<B, S, Output = E, Error = F>,
    R: Relu2RoutedTextArchitectureVisitor<B, S, Output = E, Error = F>,
    T: GatedRoutedTextArchitectureVisitor<B, PS, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedRoutedTextRealization, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedRoutedTextRealization, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        if branch.selected.plan().relu2().is_some() {
            return visit_relu2_routed_text_architecture::<B, S, _>(
                &branch.inspection,
                branch.selected,
                branch.target,
                self.context,
                self.relu2,
            )
            .map_err(routed_error);
        }
        let pooling = branch
            .selected
            .text()
            .state()
            .components()
            .iter()
            .any(|component| {
                matches!(
                    component.component().role(),
                    eredu_core::cache::StateComponentRole::Fixed(
                        eredu_core::cache::StateTensorRole::Pooling { .. }
                    )
                )
            });
        if pooling {
            visit_pooling_routed_text_architecture::<B, PS, _>(
                &branch.inspection,
                branch.selected,
                branch.target,
                self.context,
                self.pooling,
            )
            .map_err(routed_error)
        } else {
            visit_gated_routed_text_architecture::<B, S, _>(
                &branch.inspection,
                branch.selected,
                branch.target,
                self.context,
                self.gated,
            )
            .map_err(routed_error)
        }
    }
}

/// Composite text construction with one selected processor and ingress contract.
pub struct CompositeRoute<'a, B: NeuralBackend, S, V, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    visitor: V,
    prediction: P,
    state: PhantomData<fn() -> S>,
}
impl<'a, B: NeuralBackend, S, V> CompositeRoute<'a, B, S, V> {
    /// Binds one native visitor to architecture-owned composite construction.
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
impl<'a, B: NeuralBackend, S, V, P> CompositeRoute<'a, B, S, V, P> {
    /// Adds native extension materialization and a composite prediction visitor.
    pub fn with_prediction<M, F, V2>(
        self,
        materialize: F,
        visitor: V2,
    ) -> CompositeRoute<'a, B, S, V, PredictionMechanisms<M, S, F, V2>> {
        CompositeRoute {
            context: self.context,
            source_context: self.source_context,
            visitor: self.visitor,
            prediction: PredictionMechanisms::new(materialize, visitor),
            state: PhantomData,
        }
    }
}
impl<B: NeuralBackend, S, V, P> sealed::Sealed for CompositeRoute<'_, B, S, V, P> {}

impl<B, S, V, P, C, E, F> PreparedExecutionRoute<SelectedCompositeTextRealization, C, E, F>
    for CompositeRoute<'_, B, S, V, P>
where
    B: eredu_nn::GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    V: CompositeTextArchitectureVisitor<B, S, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedCompositeTextRealization, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedCompositeTextRealization, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        let requirements = composite_text_requirements(&branch.inspection)
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        visit_composite_text_architecture::<B, S, _>(
            requirements,
            branch.selected,
            branch.target,
            self.context,
            self.visitor,
        )
        .map_err(replicated_error)
    }
}
