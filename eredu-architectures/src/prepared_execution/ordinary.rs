use eredu_nn::{NeuralBackend, Tensor};
use eredu_runtime::{
    ComponentRankMechanisms, DecoderPipelineState, LayerRuntimeState, LocalComponentMechanisms,
    SelectedReplicatedTextRealization,
};

use super::*;
use crate::{replicated_text::*, routed_text::*};
use eredu_runtime::executable_binding::{
    routed_session::{construct_retained_routed_prediction, RetainedRoutedTextPrediction},
    ReplicatedTextProfileDispatcher,
};

impl<B: NeuralBackend, S, V> sealed::Sealed for KeyValueRoute<'_, B, S, V> {}

impl<B, S, V, C, E, F> PreparedExecutionRoute<SelectedReplicatedTextRealization, C, E, F>
    for KeyValueRoute<'_, B, S, V>
where
    B: eredu_core::SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
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

impl<B: NeuralBackend, D, P> sealed::Sealed for ReplicatedRoute<'_, B, D, P> {}

impl<B, D, P, C, E, F> PreparedExecutionRoute<SelectedReplicatedTextRealization, C, E, F>
    for ReplicatedRoute<'_, B, D, P>
where
    B: eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::GroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_core::SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    D: ReplicatedTextProfileDispatcher<B, Output = E, Error = F>,
    <D::AttentionState as LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor> + eredu_nn::ReadOnlyAttentionCache<B::Tensor>,
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
    /// Dense VL graph with retained component bindings and processor authority.
    Lfm3Vl(Box<crate::lfm3::vision::PreparedVlGraph>),
    /// Routed partition with exact rank sources, placement and local state authority.
    Qwen4Partition(Box<crate::qwen4_exp::prepared::PreparedTargetPartition>),
    /// Conditional partition with exact target, projector, local state and row sources.
    Qwen4ConditionalPartition(Box<crate::qwen4_exp::prepared::PreparedConditionalPartition>),
    /// Ordinary attention, fixed components and named append-only streams.
    Qwen4Exp(Box<crate::qwen4_exp::prepared::SelectedTargetExecution>),
    /// Joint image/video and target execution with retained processor admission.
    Qwen4Conditional(Box<crate::qwen4_exp::prepared::SelectedConditionalExecution>),
}

fn construct_retained_routed<B, SS, U>(
    context: &<B::Tensor as Tensor>::Context,
    streams: U,
    retained: RetainedArchitectureConstruction,
    prediction: Option<RetainedPredictionBinding>,
) -> Result<U::Output, PreparedExecutionError<U::Error>>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    SS: LayerRuntimeState<B>
        + eredu_runtime::PredictionModelState<B, LayerState: eredu_runtime::RuntimeAppendStreams<B>>,
    <SS as LayerRuntimeState<B>>::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>,
    U: RetainedRoutedTextPrediction<B, SS, SelectedRoutedBank>,
{
    match retained {
        RetainedArchitectureConstruction::Lfm3Vl(_)
        | RetainedArchitectureConstruction::Qwen4Conditional(_)
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
                    .prepare_prediction_execution::<B, SS>(context)
                    .map_err(|e| PreparedExecutionError::Architecture(e.to_string()))?
                    .into_parts();
                let weights = weights
                    .with_discovery_binding(&binding)
                    .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
                let executable = construct_retained_routed_prediction(
                    &streams,
                    target,
                    weights,
                    target_source,
                    provider_source,
                    (),
                    binding.into_facts(),
                )?;
                return Ok(eredu_runtime::executable_binding::install_text_route::<
                    U::BoundRetainedPrediction<
                        _,
                        crate::prediction_extension::PreparedQwen4PredictionWeights<B>,
                    >,
                    _,
                    _,
                >(executable));
            }
            if selected.prediction_state().is_some() {
                return Err(PreparedExecutionError::PredictionSourceMismatch);
            }
            selected
                .visit::<B, SS, U>(context, streams)
                .map_err(routed_error)
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
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + 'static,
    S: LayerRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_nn::CompressedAttentionCache<B::Tensor>
        + eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>,
    PS: LayerRuntimeState<B>,
    PS::LayerState: eredu_nn::PoolingAttentionCache<B::Tensor>,
    SS: LayerRuntimeState<B>
        + eredu_runtime::PredictionModelState<B, LayerState: eredu_runtime::RuntimeAppendStreams<B>>,
    <SS as LayerRuntimeState<B>>::LayerState: eredu_nn::AttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>,
    G: RoutedTextArchitectureVisitor<B, S, SelectedRoutedBank, Output = E, Error = F>,
    R: RoutedTextArchitectureVisitor<B, S, SelectedRoutedBank, Output = E, Error = F>,
    T: RoutedTextArchitectureVisitor<B, PS, SelectedRoutedBank, Output = E, Error = F>,
    U: RoutedTextArchitectureVisitor<B, SS, SelectedRoutedBank, Output = E, Error = F>
        + RetainedRoutedTextPrediction<B, SS, SelectedRoutedBank>,
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
            return construct_retained_routed::<B, SS, U>(
                self.context,
                self.streams,
                retained,
                branch.retained_prediction.take(),
            );
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

/// Constructed VL rank with portable vision traversal and bound native mechanisms.
type RealizedVlRank<B, S, C, M> = crate::lfm3::vision::BoundVlRankSession<
    crate::lfm3::vision::PolicyVision<
        B,
        <M as eredu_runtime::ParameterUnitMechanisms<B>>::UnitPolicy<
            crate::lfm3::vision::VisionModule<B>,
        >,
    >,
    crate::lfm3::vision::PreparedVlDecoderStage<
        <M as ComponentRankMechanisms<B, S, C>>::Pipeline<crate::lfm3::PartitionedLayeredModel<B>>,
    >,
    S,
>;

pub use eredu_runtime::executable_binding::retained_session::RetainedCompositeVisitor;

/// Constructs the retained rank using generic native mechanisms.
pub fn construct_vl_rank<B, S, C, M>(
    prepared: crate::lfm3::vision::PreparedVlRankGraph,
    context: &<B::Tensor as Tensor>::Context,
    communication: &C,
    mechanisms: &M,
) -> Result<
    super::RealizedRank<impl super::PreparedRetainedTextSession<B::Tensor, S> + 'static>,
    eredu_core::BackendFailure,
>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: DecoderPipelineState<B> + 'static,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    M: ComponentRankMechanisms<B, S, C>,
{
    let mut bound: RealizedVlRank<B, S, C, M> = prepared.construct_session::<B, S, _, _>(
        context,
        |pipeline| {
            pipeline
                .bind(mechanisms, communication)
                .map(|(execution, _)| execution)
        },
        |vision, residency| vision.bind(context, residency, mechanisms),
        S::allocate_pipeline_state,
    )?;
    Ok(super::RealizedRank {
        raw_processor: bound.raw_processor.take(),
        capabilities: bound.session,
        floating_state_bytes: bound.dtype,
        session: crate::lfm3::vision::rank_session::VlRankSession::new(bound),
    })
}

impl<B: NeuralBackend, S, V, P> sealed::Sealed for CompositeRoute<'_, B, S, V, P> {}

impl<B, S, V, P, C, E, F> PreparedExecutionRoute<SelectedCompositeTextRealization, C, E, F>
    for CompositeRoute<'_, B, S, V, P>
where
    B: eredu_core::SubmissionBackend<
            Executor = <<B as eredu_nn::NeuralBackend>::Tensor as eredu_nn::Tensor>::Context,
        > + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::HyperNeuralBackend
        + Clone
        + 'static,
    S: DecoderPipelineState<B>
        + eredu_runtime::PredictionModelState<B, LayerState: eredu_runtime::RuntimeAppendStreams<B>>,
    <S as LayerRuntimeState<B>>::LayerState: eredu_nn::ReadOnlyAttentionCache<B::Tensor>
        + eredu_runtime::RuntimeStateComponents<B>
        + eredu_runtime::RuntimeAppendStreams<B>
        + eredu_nn::AuxiliaryConvolutionState<B::Tensor>,
    B::Tensor: 'static,
    S: 'static,
    C: 'static,
    V: CompositeTextArchitectureVisitor<
            B,
            S,
            crate::SelectedRoutedBank,
            crate::replicated_text::CompositeTextRequirements,
            crate::replicated_text::RoutedCompositeRequirements,
            Output = E,
            Error = F,
        > + RetainedCompositeVisitor<B, S, C, F>
        + eredu_runtime::executable_binding::routed_session::RetainedCompositePrediction<
            B,
            S,
            crate::SelectedRoutedBank,
            Published = E,
        >,
    <B::Tensor as eredu_nn::Tensor>::Context: Sized + 'static,
    E: From<
        eredu_runtime::executable_binding::InstalledTextExecutable<
            <V as RetainedCompositeVisitor<B, S, C, F>>::RetainedTypes,
        >,
    >,
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
                RetainedArchitectureConstruction::Lfm3Vl(prepared) => {
                    let admission = prepared
                        .prompt_admission()
                        .map_err(eredu_core::BackendFailure::from_error)
                        .map_err(PreparedExecutionError::Binding)?;
                    let capabilities = prepared
                        .capabilities()
                        .map_err(eredu_core::BackendFailure::from_error)
                        .map_err(PreparedExecutionError::Binding)?;
                    let execution = prepared.selected().execution();
                    if execution.prompt_cache() {
                        return Err(PreparedExecutionError::UnavailableExecution);
                    }
                    eredu_core::SessionAdmission::new(execution.session())
                        .validate(crate::lfm3::vision::VL_SESSION_CAPABILITIES)
                        .map_err(eredu_core::BackendFailure::from_error)
                        .map_err(PreparedExecutionError::Binding)?;
                    let processor = prepared.selected().processor().clone();
                    let residency = execution.residency();
                    let state = execution.state().clone();
                    let conversion_retention = execution.parameter_conversion_retention();
                    let components = self.visitor.components();
                    use eredu_core::checkpoint_mechanisms::unit_binding::ComponentPreparationScope as _;
                    let scope = components
                        .begin_local_preparation()
                        .map_err(PreparedExecutionError::Binding)?;
                    let prepared = (|| {
                        let (architecture, component, _, _) = prepared
                            .into_layerwise::<B>(self.context)
                            .map_err(eredu_core::BackendFailure::from_error)?;
                        eredu_runtime::PreparedLocalComponent::new(
                            architecture,
                            component,
                            eredu_runtime::LocalComponentPolicy {
                                residency,
                                state: Some(state),
                                conversion_retention,
                                conversion_eligibility: eredu_core::residency::ParameterConversionRetentionEligibility::Eligible,
                            },
                        )
                        .map_err(eredu_core::BackendFailure::from_error)
                    })();
                    let prepared = scope
                        .finish(prepared)
                        .map_err(PreparedExecutionError::Binding)?;
                    let graph = prepared
                        .bind(&components, self.context)
                        .map_err(PreparedExecutionError::Binding)?;
                    let session = crate::lfm3::vision::local_vl_session::<B, S, _>(
                        graph,
                        admission,
                        capabilities,
                        processor,
                        residency,
                    )
                    .map_err(PreparedExecutionError::Binding)?;
                    eredu_runtime::executable_binding::retained_executable::install::<
                        V::RetainedTypes,
                        _,
                    >(session, None, self.visitor.retained_context())
                    .map(E::from)
                    .map_err(eredu_core::BackendFailure::from_error)
                    .map_err(PreparedExecutionError::Binding)
                }
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

use crate::lfm3::vision::PreparedVlRankGraph;
use eredu_runtime::executable_binding::route_assembly::RetainedCompositeRank;

impl<B: NeuralBackend, S, V> sealed::Sealed for RetainedCompositeRank<'_, B, S, V> {}

impl<B, S, V, C, E, F> PreparedExecutionRoute<PreparedVlRankGraph, C, E, F>
    for RetainedCompositeRank<'_, B, S, V>
where
    B: eredu_nn::TensorParallelGroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    S: DecoderPipelineState<B> + 'static,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor> + eredu_runtime::RuntimeStateComponents<B>,
    C: 'static,
    V: RetainedCompositeVisitor<B, S, C, F, Components: ComponentRankMechanisms<B, S, C>>,
    <B::Tensor as eredu_nn::Tensor>::Context: Sized + 'static,
    E: From<eredu_runtime::executable_binding::InstalledTextExecutable<V::RetainedTypes>>,
{
    fn construct(
        self,
        mut branch: PreparedConstructionBranch<PreparedVlRankGraph, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        if branch.prediction.is_some() || branch.retained_prediction.is_some() {
            return Err(PreparedExecutionError::UnavailablePrediction);
        }
        let communication = branch
            .communication
            .take()
            .ok_or(PreparedExecutionError::MissingCommunication)?;
        let session = construct_vl_rank::<B, S, _, _>(
            branch.selected,
            self.context,
            &communication,
            &self.visitor.components(),
        )
        .map_err(PreparedExecutionError::Binding)?;
        eredu_runtime::executable_binding::retained_executable::install::<V::RetainedTypes, _>(
            session.session,
            Some(communication),
            self.visitor.retained_context(),
        )
        .map(E::from)
        .map_err(eredu_core::BackendFailure::from_error)
        .map_err(PreparedExecutionError::Binding)
    }
}
