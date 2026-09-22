//! Total construction of selected architectures through typed native mechanisms.

use std::{collections::BTreeSet, num::NonZeroU8, sync::Arc};

use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_core::{ArtifactInspection, ParallelRankTopology, ParallelTopology, SessionCapabilities};
use eredu_runtime::{
    CacheResidencyPolicy, CommunicationManifest, ReplicatedTextMaterializationTask,
};

use crate::{
    capability::CapabilityEstimate, configuration::PredictionExtensionPlan,
    preparation::FloatingStateDtypeSource, prepared_sources::PreparedModelSources,
    processor_execution::PreparedProcessor, processor_plan::ArtifactArchitecturePlan,
    selected_execution::*,
};

mod direct_session;
mod ordinary;
mod partition_facts;
mod partitioned;
mod prediction;
mod routed_partition;
mod routed_session;
mod workspace;
pub(crate) use workspace::layerwise::WorkspaceLayerwisePolicy;
pub(crate) use workspace::parallel::{
    PreparedCompositeModelSource, PreparedDirectPartitionSource, PreparedFamilyPartitionModelSource,
};

pub use workspace::{
    AddressableBindingDestinations, BorrowedTextSamplingWorkspace,
    EmbeddedPredictionWorkspaceObservation, EmbeddedTargetWorkspaceObservation,
    InferenceEquationTraceObserver, InvocationWorkspaceObservation, MediaEquationInterval,
    OriginalMediaWorkspaceInput, OriginalMediaWorkspaceInputError, OriginalMediaWorkspaceReport,
    OriginalMediaWorkspaceTraceError, OriginalMediaWorkspaceTraceFailure,
    PartitionedTextBindingDestinations, PreparedInferenceBlueprint, PreparedMediaWorkspaceTensor,
    PreparedTextGenerationWorkspace, ReplicatedTextBindingDestinations,
    WorkspaceLayerwiseParameters, WorkspacePredictionEquationTails,
    project_addressable_binding_destinations, project_partitioned_text_binding_destinations,
    project_replicated_text_binding_destinations,
};

pub use direct_session::{construct_selected_composite_session, construct_selected_text_session};
pub use ordinary::{CompositeRoute, KeyValueRoute, ReplicatedRoute, RoutedRoute};
pub use partition_facts::PreparedPartitionSessionFacts;
pub use partitioned::{PartitionedCompositeRoute, PartitionedDenseRoute, PartitionedRoutedRoute};
pub use prediction::{
    PredictionBinding, PredictionConstruction, PredictionMechanisms, WithoutPrediction,
};
pub use routed_partition::{
    PartitionBankMechanisms, PartitionBankProviders, PreparedPartitionBanks,
    construct_partition_bank_providers, construct_selected_partition_providers,
};
pub use routed_session::{
    PreparedCompositeSessionFacts, PreparedTextSessionFacts,
    construct_selected_routed_composite_session, construct_selected_routed_session,
};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Failure before publication of a fully constructed executable.
#[derive(Debug, thiserror::Error)]
pub enum PreparedExecutionError<E> {
    /// Selected sources or architecture construction violated their contract.
    #[error("prepared execution construction failed: {0}")]
    Architecture(String),
    /// A participating portable host constructor preserved its typed failure.
    #[error(transparent)]
    Metadata(eredu_nn::Error),
    /// A native materialization, communication, or publication mechanism failed.
    #[error("prepared execution mechanism failed: {0}")]
    Backend(#[source] E),
    /// A partitioned selection has no realized communication.
    #[error("partitioned construction has no realized communication")]
    MissingCommunication,
    /// Ordinary execution was paired with partition communication.
    #[error("ordinary construction has unexpected realized communication")]
    UnexpectedCommunication,
    /// Extension selection and its exact prepared source do not agree.
    #[error("prepared prediction selection and source role disagree")]
    PredictionSourceMismatch,
    /// The selected class has no supplied typed construction mechanism.
    #[error("selected execution has no supplied typed construction mechanism")]
    UnavailableExecution,
    /// An embedded extension was selected without a native extension materializer.
    #[error("selected prediction extension has no supplied construction mechanism")]
    UnavailablePrediction,
}

/// Native facts and final outer adaptation after architecture construction.
pub trait PreparedExecutableAssembler<C>: Sized {
    /// Completely bound executable returned by each typed visitor.
    type Executable;
    /// Backend-owned outer prepared model.
    type Output;
    /// Native mechanism failure.
    type Error;

    /// Reports the physical floating-state dtype of this exact admitted source.
    /// Construction compares it with the representation accepted during cold selection.
    fn floating_state_dtype(
        &mut self,
        source: &FloatingStateDtypeSource,
    ) -> Result<eredu_runtime::StateStorageDtype, Self::Error>;

    /// Checks the realized native handle against the exact selected manifest.
    fn validate_communication(
        &mut self,
        manifest: &CommunicationManifest,
        communication: &C,
    ) -> Result<(), Self::Error>;

    /// Publishes one outer model only after every typed construction step succeeds.
    fn finish(
        self,
        parts: PreparedExecutableParts<Self::Executable, C>,
    ) -> Result<Self::Output, Self::Error>;
}

/// Fully checked pieces consumed by a backend's final model adapter.
pub struct PreparedExecutableParts<E, C> {
    executable: E,
    floating_state_bytes: NonZeroU8,
    state_residency: CacheResidencyPolicy,
    communication: Option<C>,
    processor: Option<PreparedProcessor>,
    capabilities: SessionCapabilities,
    inference: PreparedInferenceBlueprint,
}

impl<E, C> PreparedExecutableParts<E, C> {
    /// Exact architecture selection and shared sources used for request quotes.
    /// This contains no native tensors, devices or submission authority.
    pub fn inference_blueprint(&self) -> &PreparedInferenceBlueprint {
        &self.inference
    }
    /// Physical width reported for the architecture-selected floating-state source.
    pub const fn floating_state_bytes(&self) -> NonZeroU8 {
        self.floating_state_bytes
    }
    /// Exact selected mutable-state residency.
    pub const fn state_residency(&self) -> &CacheResidencyPolicy {
        &self.state_residency
    }
    /// Exact retained session admission.
    pub const fn capabilities(&self) -> SessionCapabilities {
        self.capabilities
    }
    /// Removes the already checked native communication handle.
    pub fn take_communication(&mut self) -> Option<C> {
        self.communication.take()
    }
    /// Removes the architecture-prepared raw-media processor, when selected.
    pub fn take_processor(&mut self) -> Option<PreparedProcessor> {
        self.processor.take()
    }
    /// Consumes these parts after extracting the other adapter-owned resources.
    pub fn into_executable(self) -> E {
        self.executable
    }
}

/// Checked native resources for a partition-specific typed binding visitor.
pub struct PreparedPartitionResources<C> {
    communication: C,
    target: RetainedCheckpointSource,
    extension_sources: BTreeSet<String>,
}

impl<C> PreparedPartitionResources<C> {
    /// Exact communication handle checked against the selected manifest.
    pub const fn communication(&self) -> &C {
        &self.communication
    }
    /// Exact target source, excluding prediction-only parameters.
    pub const fn target(&self) -> &RetainedCheckpointSource {
        &self.target
    }
    /// Exact physical keys claimed by the separately materialized extension.
    pub const fn extension_sources(&self) -> &BTreeSet<String> {
        &self.extension_sources
    }
    /// Moves the checked communication handle into a native visitor.
    pub fn into_communication(self) -> C {
        self.communication
    }
}

/// Checked facts and native resources for a partitioned prediction binding visitor.
pub struct PreparedPartitionPredictionResources<C> {
    partition: PreparedPartitionResources<C>,
    prediction: PredictionBinding,
}

impl<C> PreparedPartitionPredictionResources<C> {
    /// Exact prediction realization and architecture capability estimate.
    pub const fn prediction(&self) -> &PredictionBinding {
        &self.prediction
    }
    /// Checked target source, claimed extension keys, and native communication.
    pub const fn partition(&self) -> &PreparedPartitionResources<C> {
        &self.partition
    }
    /// Moves the checked partition resources into a native visitor.
    pub fn into_partition(self) -> PreparedPartitionResources<C> {
        self.partition
    }
}

/// Opaque, mutually agreeing prediction selection and source roles.
#[doc(hidden)]
pub struct PreparedPredictionSelection {
    placement: crate::prediction_extension::PredictionPlacementSlot,
    publish_placement: bool,
    extension: PredictionExtensionPlan,
    source: RetainedCheckpointSource,
    realization: eredu_runtime::SelectedSpeculativeRealization,
    capability: CapabilityEstimate,
    topology: ParallelRankTopology,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    residency: eredu_runtime::LayerWeightResidency,
}

/// Opaque branch input created only by the total construction driver.
#[doc(hidden)]
pub struct PreparedConstructionBranch<S, C> {
    selected: S,
    inspection: PreparedConstructionInspection,
    target: RetainedCheckpointSource,
    communication: Option<C>,
    extension_sources: BTreeSet<String>,
    prediction: Option<PreparedPredictionSelection>,
}

impl<S, C> PreparedConstructionBranch<S, C> {
    fn partition_resources<E>(
        &mut self,
    ) -> Result<PreparedPartitionResources<C>, PreparedExecutionError<E>> {
        Ok(PreparedPartitionResources {
            communication: self
                .communication
                .take()
                .ok_or(PreparedExecutionError::MissingCommunication)?,
            target: self.target.clone(),
            extension_sources: std::mem::take(&mut self.extension_sources),
        })
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

/// An explicitly absent optional construction capability.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unavailable;
impl sealed::Sealed for Unavailable {}
impl<S, C, E, F> PreparedExecutionRoute<S, C, E, F> for Unavailable {
    fn construct(
        self,
        _: PreparedConstructionBranch<S, C>,
    ) -> Result<E, PreparedExecutionError<F>> {
        Err(PreparedExecutionError::UnavailableExecution)
    }
}

/// Typed architecture construction routes with optional capabilities absent by default.
pub struct PreparedExecutionRoutes<
    R = Unavailable,
    T = Unavailable,
    X = Unavailable,
    PD = Unavailable,
    PT = Unavailable,
    PX = Unavailable,
> {
    replicated: R,
    routed: T,
    composite: X,
    partitioned_dense: PD,
    partitioned_routed: PT,
    partitioned_composite: PX,
}

impl Default for PreparedExecutionRoutes {
    fn default() -> Self {
        Self::new()
    }
}
impl PreparedExecutionRoutes {
    /// Creates an empty capability bundle; add only implemented typed routes.
    pub const fn new() -> Self {
        Self {
            replicated: Unavailable,
            routed: Unavailable,
            composite: Unavailable,
            partitioned_dense: Unavailable,
            partitioned_routed: Unavailable,
            partitioned_composite: Unavailable,
        }
    }
}

impl<R, T, X, PD, PT, PX> PreparedExecutionRoutes<R, T, X, PD, PT, PX> {
    /// Supplies ordinary replicated construction.
    pub fn with_replicated<N>(self, route: N) -> PreparedExecutionRoutes<N, T, X, PD, PT, PX> {
        PreparedExecutionRoutes {
            replicated: route,
            routed: self.routed,
            composite: self.composite,
            partitioned_dense: self.partitioned_dense,
            partitioned_routed: self.partitioned_routed,
            partitioned_composite: self.partitioned_composite,
        }
    }
    /// Supplies routed text construction.
    pub fn with_routed<N>(self, route: N) -> PreparedExecutionRoutes<R, N, X, PD, PT, PX> {
        PreparedExecutionRoutes {
            replicated: self.replicated,
            routed: route,
            composite: self.composite,
            partitioned_dense: self.partitioned_dense,
            partitioned_routed: self.partitioned_routed,
            partitioned_composite: self.partitioned_composite,
        }
    }
    /// Supplies composite text construction.
    pub fn with_composite<N>(self, route: N) -> PreparedExecutionRoutes<R, T, N, PD, PT, PX> {
        PreparedExecutionRoutes {
            replicated: self.replicated,
            routed: self.routed,
            composite: route,
            partitioned_dense: self.partitioned_dense,
            partitioned_routed: self.partitioned_routed,
            partitioned_composite: self.partitioned_composite,
        }
    }
    /// Supplies direct partitioned construction.
    pub fn with_partitioned_dense<N>(
        self,
        route: N,
    ) -> PreparedExecutionRoutes<R, T, X, N, PT, PX> {
        PreparedExecutionRoutes {
            replicated: self.replicated,
            routed: self.routed,
            composite: self.composite,
            partitioned_dense: route,
            partitioned_routed: self.partitioned_routed,
            partitioned_composite: self.partitioned_composite,
        }
    }
    /// Supplies routed partitioned construction.
    pub fn with_partitioned_routed<N>(
        self,
        route: N,
    ) -> PreparedExecutionRoutes<R, T, X, PD, N, PX> {
        PreparedExecutionRoutes {
            replicated: self.replicated,
            routed: self.routed,
            composite: self.composite,
            partitioned_dense: self.partitioned_dense,
            partitioned_routed: route,
            partitioned_composite: self.partitioned_composite,
        }
    }
    /// Supplies composite partitioned construction.
    pub fn with_partitioned_composite<N>(
        self,
        route: N,
    ) -> PreparedExecutionRoutes<R, T, X, PD, PT, N> {
        PreparedExecutionRoutes {
            replicated: self.replicated,
            routed: self.routed,
            composite: self.composite,
            partitioned_dense: self.partitioned_dense,
            partitioned_routed: self.partitioned_routed,
            partitioned_composite: route,
        }
    }
}

/// Constructs one exact selected model through architecture-owned typed routes.
///
/// All common agreement checks precede typed construction. No native tensor or
/// optional neural trait is required by this total outer driver.
pub fn construct_prepared_execution<C, A, R, T, X, PD, PT, PX>(
    sources: PreparedModelSources,
    communication: Option<C>,
    routes: PreparedExecutionRoutes<R, T, X, PD, PT, PX>,
    assembler: A,
) -> Result<A::Output, PreparedExecutionError<A::Error>>
where
    C: Clone,
    A: PreparedExecutableAssembler<C>,
    R: PreparedExecutionRoute<
            eredu_runtime::SelectedReplicatedTextRealization,
            C,
            A::Executable,
            A::Error,
        >,
    T: PreparedExecutionRoute<crate::SelectedRoutedTextRealization, C, A::Executable, A::Error>,
    X: PreparedExecutionRoute<
            crate::replicated_text::SelectedCompositeTextRealization,
            C,
            A::Executable,
            A::Error,
        >,
    PD: PreparedExecutionRoute<SelectedDensePartitionedExecution, C, A::Executable, A::Error>,
    PT: PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, A::Executable, A::Error>,
    PX: PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, A::Executable, A::Error>,
{
    construct_prepared_execution_impl(
        sources,
        communication,
        routes,
        assembler,
        ConstructionPurpose::Executable,
    )
}

// These private metadata consumers share selection and typed construction.
// Only executable construction publishes native-bound prediction placement.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConstructionPurpose {
    Executable,
    TargetEquations,
    BindingDestinations,
}
fn construct_prepared_execution_impl<C, A, R, T, X, PD, PT, PX>(
    sources: PreparedModelSources,
    communication: Option<C>,
    routes: PreparedExecutionRoutes<R, T, X, PD, PT, PX>,
    mut assembler: A,
    purpose: ConstructionPurpose,
) -> Result<A::Output, PreparedExecutionError<A::Error>>
where
    C: Clone,
    A: PreparedExecutableAssembler<C>,
    R: PreparedExecutionRoute<
            eredu_runtime::SelectedReplicatedTextRealization,
            C,
            A::Executable,
            A::Error,
        >,
    T: PreparedExecutionRoute<crate::SelectedRoutedTextRealization, C, A::Executable, A::Error>,
    X: PreparedExecutionRoute<
            crate::replicated_text::SelectedCompositeTextRealization,
            C,
            A::Executable,
            A::Error,
        >,
    PD: PreparedExecutionRoute<SelectedDensePartitionedExecution, C, A::Executable, A::Error>,
    PT: PreparedExecutionRoute<SelectedRoutedPartitionedExecution, C, A::Executable, A::Error>,
    PX: PreparedExecutionRoute<SelectedCompositePartitionedExecution, C, A::Executable, A::Error>,
{
    let target_equations = purpose == ConstructionPurpose::TargetEquations;
    let inference = PreparedInferenceBlueprint::new(sources.clone());
    // The borrowed dispatcher selects the same admitted branch. Retain a closed
    // source alias while the construction branch consumes its original owner.
    let selection_owner = sources.clone();
    let selected = selection_owner.selected();
    let graph = sources.graph();
    match (selected.communication_manifest(), communication.as_ref()) {
        (Some(manifest), Some(communication)) => assembler
            .validate_communication(manifest, communication)
            .map_err(PreparedExecutionError::Backend)?,
        (Some(_), None) => return Err(PreparedExecutionError::MissingCommunication),
        (None, Some(_)) => return Err(PreparedExecutionError::UnexpectedCommunication),
        (None, None) => {}
    }
    let state_residency = selected.text_realization().state().policy().clone();
    let capabilities = selected.session_capabilities();
    let topology = selected.execution().parallel_topology().unwrap_or(
        ParallelTopology::new(1, 1, 1, 1)
            .and_then(|topology| ParallelRankTopology::new(topology, 0))
            .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?,
    );
    let prediction = match (
        selected.prediction_extension(),
        graph.prediction_extension(),
        graph.extension(),
        selected.prediction_realization(),
    ) {
        (Some(extension), Some(source_extension), Some(source), Some(realization))
            if extension.same_admission(source_extension) =>
        {
            if target_equations {
                None
            } else {
                Some(PreparedPredictionSelection {
                    placement: Arc::clone(&graph.prediction_placement),
                    publish_placement: purpose == ConstructionPurpose::Executable,
                    extension: extension.clone(),
                    source: source.clone(),
                    realization: realization.clone(),
                    capability: crate::prediction_extension::prediction_extension_capability(
                        extension,
                    )
                    .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?,
                    topology,
                    residency: selected.text_realization().residency(),
                    tasks: selected
                        .text_realization()
                        .auxiliary_materialization_tasks()
                        .to_vec(),
                })
            }
        }
        (None, None, None, None) => None,
        _ => return Err(PreparedExecutionError::PredictionSourceMismatch),
    };
    let processor = match selected.execution().processor() {
        Some(processor) if processor.raw_media() => Some(
            PreparedProcessor::from_artifact(graph.architecture()).ok_or_else(|| {
                PreparedExecutionError::Architecture(
                    "selected raw-media execution has no retained architecture processor".into(),
                )
            })?,
        ),
        _ => None,
    };
    let source = crate::preparation::prepared_floating_state_dtype_source_parts(
        sources.inspection().format(),
        graph.architecture(),
        sources.inspection().tensors(),
    )
    .map_err(|error| PreparedExecutionError::Architecture(error.to_string()))?;
    let floating_state_dtype = assembler
        .floating_state_dtype(&source)
        .map_err(PreparedExecutionError::Backend)?;
    if !floating_state_dtype.is_floating()
        || selected
            .text_realization()
            .state()
            .floating_dtype()
            .is_some_and(|dtype| dtype != floating_state_dtype)
    {
        return Err(PreparedExecutionError::Architecture(
            "native floating-state dtype differs from the admitted storage dtype".into(),
        ));
    }
    let floating_state_bytes = floating_state_dtype.bytes();
    let retained_communication = communication.clone();
    let target = graph.target().clone();
    // Equation-only routes neither build extension materialization resources
    // nor consume their claimed-key set. Ordinary total construction preserves
    // its existing exact extension inventory for the materializer.
    let extension_sources = if target_equations {
        BTreeSet::new()
    } else {
        graph
            .extension()
            .map(|source| source.source_keys().into_iter().collect())
            .unwrap_or_default()
    };
    let context = BranchContext {
        inspection: PreparedConstructionInspection { sources },
        target,
        extension_sources,
        communication,
        prediction,
    };
    let executable = selected.execution().dispatch_ref(ConstructionDispatcher::<
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        A::Executable,
        A::Error,
    > {
        routes,
        context,
        marker: std::marker::PhantomData,
    })?;
    assembler
        .finish(PreparedExecutableParts {
            executable,
            floating_state_bytes,
            state_residency,
            communication: retained_communication,
            processor,
            capabilities,
            inference,
        })
        .map_err(PreparedExecutionError::Backend)
}

/// Retains the exact admitted catalog and target graph without cloning either.
/// Plain replicated dispatch only needs the target architecture. Other typed
/// routes borrow the exact full target inspection already validated by source
/// preparation, preserving the same catalog/target relationship without clones.
struct PreparedConstructionInspection {
    sources: PreparedModelSources,
}

impl PreparedConstructionInspection {
    fn architecture_plan(&self) -> &ArtifactArchitecturePlan {
        self.sources.architecture()
    }

    fn retained(&self) -> &ArtifactInspection<ArtifactArchitecturePlan> {
        self.sources.execution_inspection()
    }
}

struct BranchContext<C> {
    inspection: PreparedConstructionInspection,
    target: RetainedCheckpointSource,
    communication: Option<C>,
    extension_sources: BTreeSet<String>,
    prediction: Option<PreparedPredictionSelection>,
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
        }
    }
}

struct ConstructionDispatcher<R, T, X, PD, PT, PX, C, E, F> {
    routes: PreparedExecutionRoutes<R, T, X, PD, PT, PX>,
    context: BranchContext<C>,
    marker: std::marker::PhantomData<fn() -> (E, F)>,
}

impl<'a, R, T, X, PD, PT, PX, C, E, F> SelectedExecutionBorrowedDispatcher<'a>
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
        selected: &'a eredu_runtime::SelectedReplicatedTextRealization,
    ) -> Result<E, Self::Error> {
        self.routes
            .replicated
            .construct(self.context.branch(selected.clone()))
    }
    fn routed(self, selected: &'a crate::SelectedRoutedTextRealization) -> Result<E, Self::Error> {
        self.routes
            .routed
            .construct(self.context.branch(selected.clone()))
    }
    fn composite(
        self,
        selected: &'a crate::replicated_text::SelectedCompositeTextRealization,
    ) -> Result<E, Self::Error> {
        self.routes
            .composite
            .construct(self.context.branch(selected.clone()))
    }
    fn partitioned_dense(
        self,
        selected: &'a SelectedDensePartitionedExecution,
    ) -> Result<E, Self::Error> {
        self.routes
            .partitioned_dense
            .construct(self.context.branch(selected.clone()))
    }
    fn partitioned_routed(
        self,
        selected: &'a SelectedRoutedPartitionedExecution,
    ) -> Result<E, Self::Error> {
        self.routes
            .partitioned_routed
            .construct(self.context.branch(selected.clone()))
    }
    fn partitioned_composite(
        self,
        selected: &'a SelectedCompositePartitionedExecution,
    ) -> Result<E, Self::Error> {
        self.routes
            .partitioned_composite
            .construct(self.context.branch(selected.clone()))
    }
}

fn replicated_error<E>(
    error: crate::replicated_text::ReplicatedTextDispatchError<E>,
) -> PreparedExecutionError<E> {
    use crate::replicated_text::ReplicatedTextDispatchError::*;
    match error {
        Backend(error) => PreparedExecutionError::Backend(error),
        Architecture(error) => PreparedExecutionError::Architecture(error),
        Metadata(error) => PreparedExecutionError::Metadata(error),
        Ineligible(error) => PreparedExecutionError::Architecture(error.to_string()),
    }
}
fn routed_error<E>(
    error: crate::routed_text::RoutedTextDispatchError<E>,
) -> PreparedExecutionError<E> {
    use crate::routed_text::RoutedTextDispatchError::*;
    match error {
        Metadata(error) => PreparedExecutionError::Metadata(error),
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

#[cfg(test)]
mod failure_tests {
    use super::PreparedExecutionError;
    use std::error::Error as _;

    #[test]
    fn construction_preserves_the_mechanism_source_chain() {
        let cause = eredu_nn::Error::backend_retained_source(std::io::Error::from(
            std::io::ErrorKind::OutOfMemory,
        ));
        let failure = PreparedExecutionError::Backend(cause);
        let neural = failure.source().unwrap();
        assert!(neural.downcast_ref::<eredu_nn::Error>().is_some());
        assert_eq!(
            neural
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::OutOfMemory,
        );
    }
}
