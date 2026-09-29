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

/// Architecture-owned construction authority retained through cold selection.
///
/// The value owns the exact prepared source, equations, state geometry and selected
/// mechanisms. Backends choose typed bindings without rebuilding artifact plans.
#[derive(Clone)]
pub enum RetainedArchitectureConstruction {
    /// Routed partition with exact rank sources, placement and local state authority.
    Qwen4Partition(Box<crate::qwen4_exp::prepared::PreparedTargetPartition>),
    /// Conditional partition with exact target, projector, local state and row sources.
    Qwen4ConditionalPartition(Box<crate::qwen4_exp::prepared::PreparedConditionalPartition>),
    /// Ordinary attention, fixed components and named append-only streams.
    Qwen4Exp(Box<crate::qwen4_exp::prepared::SelectedTargetExecution>),
    /// Joint image/video and target execution with retained processor admission.
    Qwen4Conditional(Box<crate::qwen4_exp::prepared::SelectedConditionalExecution>),
}

/// Routed text construction with architecture-owned equation and state dispatch.
pub struct RoutedRoute<'a, B: NeuralBackend, S, PS, SS, G, R, T, U, P = WithoutPrediction> {
    context: &'a <B::Tensor as Tensor>::Context,
    source_context: &'a <B::Tensor as Tensor>::Context,
    gated: G,
    relu2: R,
    pooling: T,
    streams: U,
    prediction: P,
    states: PhantomData<fn() -> (S, PS, SS)>,
}

impl<'a, B: NeuralBackend, S, PS, SS, G, R, T, U> RoutedRoute<'a, B, S, PS, SS, G, R, T, U> {
    /// Binds exact gated, ReLU-squared, pooling and append-stream visitors.
    /// Architecture code chooses the selected state profile.
    pub fn new(
        context: &'a <B::Tensor as Tensor>::Context,
        source_context: &'a <B::Tensor as Tensor>::Context,
        gated: G,
        relu2: R,
        pooling: T,
        streams: U,
    ) -> Self {
        Self {
            context,
            source_context,
            gated,
            relu2,
            pooling,
            streams,
            prediction: WithoutPrediction,
            states: PhantomData,
        }
    }
}

impl<'a, B: NeuralBackend, S, PS, SS, G, R, T, U, P> RoutedRoute<'a, B, S, PS, SS, G, R, T, U, P> {
    /// Adds native extension materialization and a routed prediction-profile visitor.
    pub fn with_prediction<M, F, V>(
        self,
        materialize: F,
        visitor: V,
    ) -> RoutedRoute<'a, B, S, PS, SS, G, R, T, U, PredictionMechanisms<M, (S, PS), F, V>> {
        RoutedRoute {
            context: self.context,
            source_context: self.source_context,
            gated: self.gated,
            relu2: self.relu2,
            pooling: self.pooling,
            streams: self.streams,
            prediction: PredictionMechanisms::new(materialize, visitor),
            states: PhantomData,
        }
    }
}
impl<'a, B, S, PS, SS, G, R, T, U, P> RoutedRoute<'a, B, S, PS, SS, G, R, T, U, P>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    SS: LayerRuntimeState<B>,
    SS::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>,
    U: RoutedTextArchitectureVisitor<B, SS>,
{
    /// Constructs an exact retained target with its selected typed state profile.
    ///
    /// Prediction owners use the same selected target visitor and shared driver.
    pub fn construct_retained(
        self,
        retained: RetainedArchitectureConstruction,
        prediction: Option<PredictionBinding>,
    ) -> Result<U::Output, PreparedExecutionError<U::Error>> {
        match retained {
            RetainedArchitectureConstruction::Qwen4Conditional(_)
            | RetainedArchitectureConstruction::Qwen4Partition(_)
            | RetainedArchitectureConstruction::Qwen4ConditionalPartition(_) => {
                Err(PreparedExecutionError::Architecture(
                    "construction authority does not match nonpartitioned routed route".into(),
                ))
            }
            RetainedArchitectureConstruction::Qwen4Exp(selected) => {
                let selected = *selected;
                if let Some(binding) = prediction {
                    let (target, weights, target_source, provider_source) = selected
                        .prepare_prediction_execution::<B, SS>(self.context)
                        .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?
                        .into_parts();
                    let weights = weights
                        .with_discovery_binding(&binding)
                        .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
                    return self
                        .streams
                        .visit_prediction(target, weights, target_source, provider_source, binding)
                        .map_err(routed_error);
                }
                if selected.prediction_state().is_some() {
                    return Err(PreparedExecutionError::PredictionSourceMismatch);
                }
                selected
                    .visit::<B, SS, U>(self.context, self.streams)
                    .map_err(routed_error)
            }
        }
    }
}

impl<B: NeuralBackend, S, PS, SS, G, R, T, U, P> sealed::Sealed
    for RoutedRoute<'_, B, S, PS, SS, G, R, T, U, P>
{
}

impl<B, S, PS, SS, G, R, T, U, P, C, E, F>
    PreparedExecutionRoute<SelectedRoutedTextRealization, C, E, F>
    for RoutedRoute<'_, B, S, PS, SS, G, R, T, U, P>
where
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    SS: LayerRuntimeState<B>,
    SS::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>,
    G: RoutedTextArchitectureVisitor<B, S, Output = E, Error = F>,
    R: Relu2RoutedTextArchitectureVisitor<B, S, Output = E, Error = F>,
    T: RoutedTextArchitectureVisitor<B, PS, Output = E, Error = F>,
    U: RoutedTextArchitectureVisitor<B, SS, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedRoutedTextRealization, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedRoutedTextRealization, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(retained) = branch.retained_construction.take() {
            if branch.prediction.is_some() {
                return Err(PreparedExecutionError::UnavailablePrediction);
            }
            return self.construct_retained(retained, branch.retained_prediction.take());
        }
        if let Some(prediction) = branch.prediction.take() {
            return self.prediction.construct(
                prediction,
                branch,
                self.source_context,
                self.context,
            );
        }
        if branch
            .selected
            .banks()
            .values()
            .all(|bank| bank.plan().relu2().is_some())
        {
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
            visit_routed_text_architecture::<B, S, _>(
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
    B: eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + Clone
        + 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    V: CompositeTextArchitectureVisitor<B, S, Output = E, Error = F>,
    P: PredictionConstruction<B, SelectedCompositeTextRealization, C, E, F>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<SelectedCompositeTextRealization, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if let Some(retained) = branch.retained_construction.take() {
            if branch.prediction.is_some() {
                return Err(PreparedExecutionError::UnavailablePrediction);
            }
            return match retained {
                RetainedArchitectureConstruction::Qwen4Conditional(selected) => {
                    if let Some(binding) = branch.retained_prediction.take() {
                        selected.visit_composite_prediction::<B, S, V>(
                            self.context,
                            self.visitor,
                            binding,
                        )
                    } else {
                        selected.visit_composite::<B, S, V>(self.context, self.visitor)
                    }
                }
                RetainedArchitectureConstruction::Qwen4Exp(_)
                | RetainedArchitectureConstruction::Qwen4Partition(_)
                | RetainedArchitectureConstruction::Qwen4ConditionalPartition(_) => {
                    Err(PreparedExecutionError::Architecture(
                        "construction authority does not match nonpartitioned composite route"
                            .into(),
                    ))
                }
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
