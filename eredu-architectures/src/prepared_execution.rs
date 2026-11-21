//! Total construction of selected architectures through typed native mechanisms.

use std::{collections::BTreeSet, sync::Arc};

use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::{ArtifactInspection, ParallelRankTopology, ParallelTopology};
use eredu_runtime::executable_binding::{
    KeyValueRoute, PreparedExecutableConstruction, PreparedExecutableDiscovery,
    PreparedExecutablePlan,
};

use crate::{
    capability::CapabilityEstimate,
    prepared_sources::{PreparedModelDiscovery, PreparedModelSources},
    processor_execution::PreparedProcessor,
    processor_plan::ArtifactArchitecturePlan,
    selected_execution::*,
};

mod direct_session;
mod ordinary;
pub(crate) mod partition_facts;
mod partitioned;
mod prediction;
mod retained_session;
mod routed_partition;
pub use retained_session::{PreparedRetainedTextSession, RealizedRank};

pub use direct_session::{construct_selected_composite_session, construct_selected_text_session};
pub use eredu_runtime::executable_binding::{
    routed_session::construct_selected_routed_composite_session, PreparedCompositeSessionFacts,
    PreparedTextSessionFacts,
};
pub use ordinary::{construct_vl_rank, RetainedArchitectureConstruction, RetainedCompositeVisitor};
pub use partition_facts::PreparedPartitionSessionFacts;
pub use prediction::PredictionConstruction;
pub use routed_partition::{
    construct_partition_bank_providers, PartitionBankMechanisms, PartitionBankProviders,
    PreparedPartitionBanks,
};

pub(crate) mod sealed {
    pub trait Sealed {}
}

pub use eredu_runtime::executable_binding::{
    CompositeRoute, PartitionedCompositeRoute, PartitionedDenseRoute, PartitionedRoutedRoute,
    PredictionMechanisms, ReplicatedRoute, RoutedRoute, WithoutPrediction,
};
pub use eredu_runtime::executable_binding::{
    PredictionBinding, PreparedExecutionError, PreparedExecutionRoutes,
    PreparedPartitionPredictionResources, PreparedPartitionResources, Unavailable,
};
#[doc(hidden)]
pub use prediction::RetainedPredictionBinding;

/// Opaque, mutually agreeing prediction selection and source roles.
#[doc(hidden)]
pub struct PreparedPredictionSelection {
    placement: crate::prediction_extension::PredictionPlacementSlot,
    parameters: Arc<crate::prediction_extension::SelectedPredictionParameters>,
    source: SharedCheckpointSource,
    realization: eredu_runtime::SelectedSpeculativeRealization,
    capability: CapabilityEstimate,
    topology: ParallelRankTopology,
    residency: eredu_runtime::LayerWeightResidency,
}

/// Opaque branch input created only by the total construction driver.
#[doc(hidden)]
pub struct PreparedConstructionBranch<S, C> {
    selected: S,
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    target: SharedCheckpointSource,
    communication: Option<C>,
    extension_sources: BTreeSet<String>,
    prediction: Option<PreparedPredictionSelection>,
    retained_prediction: Option<RetainedPredictionBinding>,
    retained_construction: Option<RetainedArchitectureConstruction>,
    retained_composite_rank: Option<crate::lfm3::vision::PreparedVlRankGraph>,
}

impl<S, C> PreparedConstructionBranch<S, C> {
    fn partition_resources<E>(
        &mut self,
    ) -> Result<PreparedPartitionResources<C>, PreparedExecutionError<E>> {
        Ok(PreparedPartitionResources::new(
            self.communication
                .take()
                .ok_or(PreparedExecutionError::MissingCommunication)?,
            Arc::clone(&self.target),
            std::mem::take(&mut self.extension_sources),
        ))
    }
}

/// Architecture-owned implementation of one exact typed construction route.
///
/// This trait is sealed. Backend authors supply visitors to the route values;
/// they do not implement selected-execution orchestration.
#[doc(hidden)]
pub trait PreparedExecutionRoute<S, C, E, F>: sealed::Sealed {
    /// Consumes a branch whose common source and communication contracts are checked.
    fn construct(
        self,
        branch: PreparedConstructionBranch<S, C>,
    ) -> Result<E, PreparedExecutionError<F>>;
}

impl sealed::Sealed for Unavailable {}
impl<S, C, E, F> PreparedExecutionRoute<S, C, E, F> for Unavailable {
    fn construct(
        self,
        _: PreparedConstructionBranch<S, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        Err(PreparedExecutionError::UnavailableExecution)
    }
}

impl PreparedExecutableDiscovery for PreparedModelDiscovery {
    fn bind_executable(
        self,
        parameters: Option<Arc<eredu_runtime::ArchitectureParameterDescription>>,
        hooks: Option<eredu_runtime::inspection::ObservationHookSupport>,
    ) -> Result<Self, eredu_core::capture::CaptureError> {
        self.bind_partition_parameters(parameters)?
            .bind_partition_observation_hooks(hooks)
    }
}

impl<C, R, T, X, PD, PT, PX, E, F>
    PreparedExecutableConstruction<C, E, F, PreparedExecutionRoutes<R, T, X, PD, PT, PX>>
    for PreparedModelSources
where
    R: PreparedExecutionRoute<eredu_runtime::SelectedReplicatedTextRealization, C, E, F>,
    T: PreparedExecutionRoute<crate::SelectedRoutedTextRealization, C, E, F>,
    X: PreparedExecutionRoute<crate::replicated_text::SelectedCompositeTextRealization, C, E, F>,
    PD: PreparedExecutionRoute<SelectedDensePartitionedExecution, C, E, F>,
    PT: PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, E, F>,
    PX: PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, E, F>,
{
    type Processor = PreparedProcessor;
    type Discovery = PreparedModelDiscovery;

    fn prepare(
        self,
        routes: PreparedExecutionRoutes<R, T, X, PD, PT, PX>,
        observation: eredu_core::ObservationMechanisms,
        capture: eredu_core::capture::CaptureCapabilities,
    ) -> Result<
        PreparedExecutablePlan<
            Self::Processor,
            Self::Discovery,
            impl FnOnce(Option<C>) -> Result<E, PreparedExecutionError<F>>
                + use<C, R, T, X, PD, PT, PX, E, F>,
        >,
        PreparedExecutionError<F>,
    > {
        let discovery = self.prepare_discovery(observation, capture);
        let (selected, inspection, mut graph, vl_rank) = self.into_parts();
        let retained_construction = graph.take_retained_construction();
        let topology = selected.execution().parallel_topology().unwrap_or(
            ParallelTopology::new(1, 1, 1, 1)
                .and_then(|topology| ParallelRankTopology::new(topology, 0))
                .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?,
        );
        let retained_prediction = match (&retained_construction, selected.prediction_realization())
        {
            (Some(RetainedArchitectureConstruction::Qwen4Exp(target)), Some(realization))
                if target.prediction_state().is_some() && graph.extension().is_some() =>
            {
                Some(RetainedPredictionBinding::new(
                    realization.clone(),
                    target.capability_estimate().clone(),
                    Arc::clone(&graph.prediction_placement),
                    topology,
                ))
            }
            (Some(RetainedArchitectureConstruction::Qwen4Exp(target)), None)
                if target.prediction_state().is_some() =>
            {
                return Err(PreparedExecutionError::PredictionSourceMismatch);
            }
            (
                Some(RetainedArchitectureConstruction::Qwen4Conditional(target)),
                Some(realization),
            ) if target.prediction_state().is_some() && graph.extension().is_some() => {
                Some(RetainedPredictionBinding::new(
                    realization.clone(),
                    target.capability_estimate().clone(),
                    Arc::clone(&graph.prediction_placement),
                    topology,
                ))
            }
            (Some(RetainedArchitectureConstruction::Qwen4Conditional(target)), None)
                if target.prediction_state().is_some() =>
            {
                return Err(PreparedExecutionError::PredictionSourceMismatch);
            }
            (Some(RetainedArchitectureConstruction::Qwen4Partition(target)), Some(realization))
                if target.prediction_state().is_some() && graph.extension().is_some() =>
            {
                Some(RetainedPredictionBinding::new(
                    realization.clone(),
                    target.capability_estimate().clone(),
                    Arc::clone(&graph.prediction_placement),
                    topology,
                ))
            }
            (
                Some(RetainedArchitectureConstruction::Qwen4ConditionalPartition(target)),
                Some(realization),
            ) if target.prediction_state().is_some() && graph.extension().is_some() => {
                Some(RetainedPredictionBinding::new(
                    realization.clone(),
                    target.capability_estimate().clone(),
                    Arc::clone(&graph.prediction_placement),
                    topology,
                ))
            }
            (Some(RetainedArchitectureConstruction::Qwen4Partition(target)), None)
                if target.prediction_state().is_some() =>
            {
                return Err(PreparedExecutionError::PredictionSourceMismatch)
            }
            (Some(RetainedArchitectureConstruction::Qwen4ConditionalPartition(target)), None)
                if target.prediction_state().is_some() =>
            {
                return Err(PreparedExecutionError::PredictionSourceMismatch)
            }
            _ => None,
        };
        let state_residency = selected.text_realization().state().policy().clone();
        let prediction = match (
            selected.prediction_extension(),
            graph.prediction_extension(),
            graph.extension(),
            selected.prediction_realization(),
        ) {
            (Some(extension), Some(source_extension), Some(source), Some(realization))
                if extension.same_admission(source_extension)
                    && selected
                        .execution()
                        .prediction_parameters()
                        .is_some_and(|parameters| parameters.same_admission(extension)) =>
            {
                Some(PreparedPredictionSelection {
                    placement: Arc::clone(&graph.prediction_placement),
                    parameters: selected
                        .execution()
                        .prediction_parameters()
                        .cloned()
                        .ok_or_else(|| {
                            PreparedExecutionError::Architecture(
                                "prediction selection has no retained executable parameters".into(),
                            )
                        })?,
                    source: Arc::clone(source),
                    realization: realization.clone(),
                    capability: crate::prediction_extension::prediction_extension_capability(
                        extension,
                    )
                    .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?,
                    topology,
                    residency: selected.text_realization().residency(),
                })
            }
            (None, None, None, None) => None,
            (None, None, Some(_), Some(_)) if retained_prediction.is_some() => None,
            _ => return Err(PreparedExecutionError::PredictionSourceMismatch),
        };
        let processor = if let Some(prepared) = &vl_rank {
            prepared.processor()
        } else {
            match &retained_construction {
            Some(RetainedArchitectureConstruction::Lfm3Vl(prepared)) => prepared.processor(),
            _ => match selected.execution().processor() {
                Some(processor) if processor.raw_media() => Some(match selected.qwen4_construction() {
                    Some(crate::selected_execution::SelectedQwen4Construction::Conditional {
                        selected: conditional,
                        ..
                    }) => {
                        let budget = conditional.header_plan().processor_budget();
                        conditional.processor().prepared_processor(budget)
                    }
                    Some(
                        crate::selected_execution::SelectedQwen4Construction::ConditionalPartitioned {
                            selected: conditional,
                            ..
                        },
                    ) => {
                        let budget = conditional.header_plan().processor_budget();
                        conditional.processor().prepared_processor(budget)
                    }
                    _ => PreparedProcessor::from_artifact(graph.architecture()).ok_or_else(|| {
                        PreparedExecutionError::Architecture(
                            "selected raw-media execution has no retained architecture processor"
                                .into(),
                        )
                    })?,
                }),
                _ => None,
            },
        }
        };
        let inspection = inspection.map_architecture_plan(|_| graph.architecture().clone());
        let source = crate::preparation::prepared_floating_state_dtype_source(&inspection)
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
        let expected_dtype = selected.text_realization().state().floating_dtype();
        let mut context = BranchContext {
            retained_construction,
            retained_composite_rank: vl_rank,
            retained_prediction,
            inspection,
            target: Arc::clone(graph.target()),
            extension_sources: graph
                .extension()
                .map(|source| source.source_keys().into_iter().collect())
                .unwrap_or_default(),
            communication: None,
            prediction,
        };
        let (execution, _, _, _) = selected.into_parts();
        Ok(PreparedExecutablePlan {
            floating_source: source,
            expected_dtype,
            state_residency,
            processor,
            discovery,
            dispatch: move |communication| {
                context.communication = communication;
                execution.dispatch(ConstructionDispatcher {
                    routes,
                    context,
                    marker: std::marker::PhantomData,
                })
            },
        })
    }
}

struct BranchContext<C> {
    inspection: ArtifactInspection<ArtifactArchitecturePlan>,
    target: SharedCheckpointSource,
    communication: Option<C>,
    extension_sources: BTreeSet<String>,
    prediction: Option<PreparedPredictionSelection>,
    retained_prediction: Option<RetainedPredictionBinding>,
    retained_construction: Option<RetainedArchitectureConstruction>,
    retained_composite_rank: Option<crate::lfm3::vision::PreparedVlRankGraph>,
}
impl<C> BranchContext<C> {
    fn branch<S>(self, selected: S) -> PreparedConstructionBranch<S, C> {
        PreparedConstructionBranch {
            selected,
            inspection: self.inspection,
            target: self.target,
            communication: self.communication,
            extension_sources: self.extension_sources,
            prediction: self.prediction,
            retained_prediction: self.retained_prediction,
            retained_construction: self.retained_construction,
            retained_composite_rank: self.retained_composite_rank,
        }
    }
}

struct ConstructionDispatcher<R, T, X, PD, PT, PX, C, E, F> {
    routes: PreparedExecutionRoutes<R, T, X, PD, PT, PX>,
    context: BranchContext<C>,
    marker: std::marker::PhantomData<fn() -> (E, F)>,
}

impl<R, T, X, PD, PT, PX, C, E, F> SelectedExecutionDispatcher
    for ConstructionDispatcher<R, T, X, PD, PT, PX, C, E, F>
where
    R: PreparedExecutionRoute<eredu_runtime::SelectedReplicatedTextRealization, C, E, F>,
    T: PreparedExecutionRoute<crate::SelectedRoutedTextRealization, C, E, F>,
    X: PreparedExecutionRoute<crate::replicated_text::SelectedCompositeTextRealization, C, E, F>,
    PD: PreparedExecutionRoute<SelectedDensePartitionedExecution, C, E, F>,
    PT: PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, E, F>,
    PX: PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, E, F>,
{
    type Output = E;
    type Error = PreparedExecutionError<F>;
    fn replicated(
        self,
        selected: eredu_runtime::SelectedReplicatedTextRealization,
    ) -> Result<E, Self::Error> {
        self.routes
            .replicated
            .construct(self.context.branch(selected))
    }
    fn routed(self, selected: crate::SelectedRoutedTextRealization) -> Result<E, Self::Error> {
        self.routes.routed.construct(self.context.branch(selected))
    }
    fn composite(
        self,
        selected: crate::replicated_text::SelectedCompositeTextRealization,
    ) -> Result<E, Self::Error> {
        self.routes
            .composite
            .construct(self.context.branch(selected))
    }
    fn partitioned_dense(
        self,
        selected: SelectedDensePartitionedExecution,
    ) -> Result<E, Self::Error> {
        self.routes
            .partitioned_dense
            .construct(self.context.branch(selected))
    }
    fn partitioned_routed(
        self,
        selected: SelectedRoutedPartitionedExecution,
    ) -> Result<E, Self::Error> {
        self.routes
            .partitioned_routed
            .construct(self.context.branch(selected))
    }
    fn partitioned_composite(
        self,
        selected: SelectedCompositePartitionedExecution,
    ) -> Result<E, Self::Error> {
        self.routes
            .partitioned_composite
            .construct(self.context.branch(selected))
    }
}

fn replicated_error<E>(
    error: crate::replicated_text::ReplicatedTextDispatchError<E>,
) -> PreparedExecutionError<E> {
    use crate::replicated_text::ReplicatedTextDispatchError::*;
    match error {
        Backend(error) => PreparedExecutionError::Backend(error),
        Architecture(error) => PreparedExecutionError::Architecture(error),
        Ineligible(error) => PreparedExecutionError::Architecture(error.to_string()),
    }
}
fn routed_error<E>(
    error: crate::routed_text::RoutedTextDispatchError<E>,
) -> PreparedExecutionError<E> {
    use crate::routed_text::RoutedTextDispatchError::*;
    match error {
        Backend(error) => PreparedExecutionError::Backend(error),
        Architecture(error) => PreparedExecutionError::Architecture(error),
    }
}
fn partitioned_error<E>(
    error: crate::partitioned_execution::DenseDecoderPartitionedDispatchError<E>,
) -> PreparedExecutionError<E> {
    use crate::partitioned_execution::DenseDecoderPartitionedDispatchError::*;
    match error {
        Visitor(error) => PreparedExecutionError::Backend(error),
        Architecture(error) => PreparedExecutionError::Architecture(error),
    }
}
fn composite_partitioned_error<E>(
    error: crate::composite_partitioned::CompositePartitionPreparationError<E>,
) -> PreparedExecutionError<E> {
    use crate::composite_partitioned::CompositePartitionPreparationError::*;
    match error {
        Visitor(error) => PreparedExecutionError::Backend(error),
        Architecture(error) => PreparedExecutionError::Architecture(error),
    }
}
