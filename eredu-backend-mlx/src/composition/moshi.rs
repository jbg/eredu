//! Production MLX composition for the backend-neutral Moshi-family model.

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
    sync::Arc,
};

#[cfg(test)]
use eredu_architectures::moshi::MoshiConfig;
use eredu_architectures::moshi::{self};
use eredu_checkpoint::store::{CheckpointSource, SharedCheckpointSource};
use eredu_core::artifact::ArtifactIdentity;
use eredu_nn::Parameterized;
use eredu_runtime::{
    construct_realtime_model, ConstructedRealtimeExecution, DenseDiskStreamReport,
    LayeredArchitecture, LayerwiseModelMetadata, LayerwiseRuntime, ParameterGroupOwner,
    RealizedRealtimePolicy, RealizedRealtimeState, RealtimeLayerwiseRuntime,
    RealtimeMaterializationTask, RealtimeModelConstructionMechanisms, ResidencyReport,
    ResidentLayerGroupReport, SelectedRealtimeRealization, SelectedRealtimeStateRealization,
    SequentialDecisionDriver, WeightBinding,
};
use safemlx::Stream;

use crate::backend::{
    error::Error,
    nn::shared::MlxNeuralBackend,
    ordinary_retirement::{self, OrdinaryRetirement},
    runtime::{
        cache::state::MlxKeyValueState,
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            },
            layerwise::{quantize_exact_realtime_tasks, shard_layer_bindings},
        },
        generation::MlxSamplingBackend,
    },
    submission_recovery::{self, Recovery, Retention, Status},
};
type SelectedLayerwiseRuntime<A, P> = LayerwiseRuntime<A, MlxNeuralBackend, MlxKeyValueState, P>;
type SelectedPartitionRuntime<A, P> = eredu_runtime::PartitionedTextRuntime<
    A,
    MlxNeuralBackend,
    MlxKeyValueState,
    (),
    eredu_runtime::LayerwiseTraversalPartitionExecutor<A, MlxNeuralBackend, MlxKeyValueState, P>,
    crate::backend::runtime::distributed::Group,
    crate::backend::runtime::distributed::topology::CommunicationRouteRealization,
    crate::backend::nn::shared::MlxCommunicationTensorMetadata,
    eredu_runtime::NoBoundaryTransport,
    eredu_runtime::NoOutputPublisher,
    eredu_runtime::NoCommitAgreement,
>;
type SelectedTraversalRuntime<A, P> = eredu_runtime::LayerwiseTraversalRuntime<
    SelectedLayerwiseRuntime<A, P>,
    Box<SelectedPartitionRuntime<A, P>>,
>;
trait ErasedRealtimeExecutionContract {
    fn selected(&self) -> &SelectedRealtimeRealization;
    fn residency_report(&self) -> Result<ResidencyReport, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error>;
    fn execute_decisions(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, moshi::ForwardContext<crate::MlxTensor>), Error>;
}

struct DirectRealtimeExecution<A>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>,
    A::Error: std::fmt::Display,
    A::Unit: 'static,
{
    selected: SelectedRealtimeRealization,
    execution: ConstructedRealtimeExecution<
        A,
        MlxNeuralBackend,
        MlxRealtimeConstructionMechanisms<A::Unit>,
    >,
}

struct PartitionedRealtimeExecution<A, P>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>,
    A::Error: std::fmt::Display,
    P: eredu_runtime::LayerwisePolicy<MlxNeuralBackend, A::Unit>,
{
    selected: SelectedRealtimeRealization,
    execution: SelectedTraversalRuntime<A, P>,
}

impl<A> ErasedRealtimeExecutionContract for DirectRealtimeExecution<A>
where
    A: moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend, MlxKeyValueState>
        + eredu_runtime::ParallelLayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>
        + 'static,
    A::Error: std::fmt::Display,
{
    fn selected(&self) -> &SelectedRealtimeRealization {
        &self.selected
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        match self.execution.execution() {
            RealtimeLayerwiseRuntime::Resident(runtime) => runtime.policy().residency_report(),
            RealtimeLayerwiseRuntime::Bounded(runtime) => runtime.policy().residency_report(),
        }
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        match self.execution.execution() {
            RealtimeLayerwiseRuntime::Resident(_) => Ok(None),
            RealtimeLayerwiseRuntime::Bounded(runtime) => runtime.policy().dense_stream_report(),
        }
    }

    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        match self.execution.execution() {
            RealtimeLayerwiseRuntime::Resident(runtime) => {
                runtime.policy().execution_group_reports()
            }
            RealtimeLayerwiseRuntime::Bounded(runtime) => {
                runtime.policy().execution_group_reports()
            }
        }
    }

    fn execute_decisions(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, moshi::ForwardContext<crate::MlxTensor>), Error> {
        moshi::execute_detached_replicated_moshi_realtime(
            &mut self.execution,
            state,
            temporal,
            driver,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

trait RealtimePolicyReports {
    fn residency_report(&self) -> Result<ResidencyReport, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error>;
}

impl<U> RealtimePolicyReports for MlxResidentPolicy<U> {
    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        Ok(None)
    }

    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.execution_group_reports()
    }
}

impl<U: 'static> RealtimePolicyReports for MlxLayerwisePolicy<U> {
    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.dense_stream_report()
    }

    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.execution_group_reports()
    }
}

impl<A, P> ErasedRealtimeExecutionContract for PartitionedRealtimeExecution<A, P>
where
    A: moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend, MlxKeyValueState>
        + eredu_runtime::ParallelLayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>
        + 'static,
    A::Error: std::fmt::Display,
    P: eredu_runtime::LayerwisePolicy<MlxNeuralBackend, A::Unit> + RealtimePolicyReports + 'static,
    P::Error: std::fmt::Display,
{
    fn selected(&self) -> &SelectedRealtimeRealization {
        &self.selected
    }

    fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.policy().residency_report()
    }

    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.execution.policy().dense_stream_report()
    }

    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.execution.policy().execution_group_reports()
    }

    fn execute_decisions(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, moshi::ForwardContext<crate::MlxTensor>), Error> {
        moshi::execute_detached_partitioned_moshi_realtime(
            &mut self.execution,
            state,
            temporal,
            driver,
            stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))
    }
}

/// Generic MLX storage/materialization mechanisms used by the neutral constructor.
pub struct MlxRealtimeConstructionMechanisms<U: 'static> {
    store: SharedCheckpointSource,
    residency: eredu_runtime::LayerWeightResidency,
    weights_stream: Stream,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    pending: Option<MlxLayerwisePolicy<U>>,
    metadata: Option<LayerwiseModelMetadata>,
    local_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
}

impl<U: 'static> MlxRealtimeConstructionMechanisms<U> {
    fn new(
        store: SharedCheckpointSource,
        residency: eredu_runtime::LayerWeightResidency,
        transform: Option<eredu_checkpoint::WeightQuantization>,
        weights_stream: &Stream,
        local_layout: Option<eredu_runtime::LocalModelLayout>,
    ) -> Self {
        Self {
            store,
            residency,
            weights_stream: weights_stream.clone(),
            transform,
            materialization: None,
            pending: None,
            metadata: None,
            local_layout: local_layout.map(Arc::new),
        }
    }

    /// Returns construction metadata after a policy has been realized.
    pub fn metadata(&self) -> Option<&LayerwiseModelMetadata> {
        self.metadata.as_ref()
    }
}
fn selected_task_bindings(
    tasks: &[RealtimeMaterializationTask],
    store: &dyn CheckpointSource,
    local_layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<
    (
        Vec<WeightBinding>,
        BTreeMap<ParameterGroupOwner, Vec<WeightBinding>>,
    ),
    Error,
> {
    let (mut pinned, mut units) = eredu_runtime::realtime_task_binding_plan(tasks, store)
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?
        .into_parts();
    if let Some(layout) = local_layout {
        pinned = shard_layer_bindings(pinned, store, layout)?;
        for bindings in units.values_mut() {
            *bindings = shard_layer_bindings(std::mem::take(bindings), store, layout)?;
        }
    }
    Ok((pinned, units))
}

impl<A, U> RealtimeModelConstructionMechanisms<A, MlxNeuralBackend>
    for MlxRealtimeConstructionMechanisms<U>
where
    A: LayeredArchitecture<MlxNeuralBackend, MlxKeyValueState, Unit = U>,
    A::Error: std::fmt::Display,
    U: Parameterized<crate::MlxTensor> + 'static,
{
    type State = MlxKeyValueState;
    type PolicyError = Error;
    type ResidentPolicy = MlxResidentPolicy<U>;
    type BoundedPolicy = MlxLayerwisePolicy<U>;
    type Error = Error;

    fn prepare_resident_materialization(
        &mut self,
        architecture: &mut A,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &Stream,
    ) -> Result<(), Self::Error> {
        eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
            tasks,
            self.store.as_ref(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if source_architecture.is_some() || source_units.is_some() {
            let quantization = self.transform.ok_or_else(|| {
                Error::ArchitectureModel(
                    "source-format architecture was supplied without a selected MLX transform"
                        .into(),
                )
            })?;
            let (store, report) = quantize_exact_realtime_tasks(
                Arc::clone(&self.store),
                architecture.static_modules(),
                units,
                quantization,
                tasks,
                context,
            )?;
            self.store = store;
            self.materialization = Some(report);
        } else if self.transform.is_some() {
            return Err(Error::ArchitectureModel(
                "selected MLX transform has no source-format architecture".into(),
            ));
        }
        let (static_bindings, mut unit_bindings) =
            selected_task_bindings(tasks, self.store.as_ref(), self.local_layout.as_deref())?;
        let layout = selected.execution_units().clone();
        let (policy, metadata) = prepare_layerwise_policy_with_bindings(
            Arc::clone(&self.store),
            architecture,
            (),
            std::marker::PhantomData::<MlxKeyValueState>,
            self.residency,
            context,
            &self.weights_stream,
            |_| false,
            move |_modules, _store| Ok(static_bindings),
            move |_ordinal, address, _path, _unit, _store, _stream| {
                let group = layout
                    .group_id(address.group())
                    .ok_or_else(|| {
                        Error::ArchitectureModel("selected unit group is missing".into())
                    })?
                    .clone();
                unit_bindings
                    .remove(&ParameterGroupOwner::execution_unit(group, address.index()))
                    .ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "selected unit bindings are missing for {address:?}"
                        ))
                    })
            },
        )?;
        self.pending = Some(policy);
        let mut metadata = metadata;
        metadata.set_materialization(self.materialization.clone());
        self.metadata = Some(metadata);
        Ok(())
    }

    fn resident_policy(
        &mut self,
        _architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedRealtimeRealization,
        context: &Stream,
    ) -> Result<RealizedRealtimePolicy<Self::ResidentPolicy>, Self::Error> {
        let policy = self
            .pending
            .take()
            .ok_or_else(|| Error::ArchitectureModel("resident MLX policy was not prepared".into()))?
            .into_resident_units(units, context)?;
        Ok(RealizedRealtimePolicy::new(policy, selected.residency()))
    }

    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        source_architecture: Option<&mut A>,
        tasks: &[RealtimeMaterializationTask],
        selected: &SelectedRealtimeRealization,
        context: &Stream,
    ) -> Result<RealizedRealtimePolicy<Self::BoundedPolicy>, Self::Error> {
        eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
            tasks,
            self.store.as_ref(),
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        if source_architecture.is_some() {
            let quantization = self.transform.ok_or_else(|| {
                Error::ArchitectureModel(
                    "source-format architecture was supplied without a selected MLX transform"
                        .into(),
                )
            })?;
            let units = (0..selected.execution_units().len())
                .map(|ordinal| {
                    let address = selected
                        .execution_units()
                        .address(ordinal)
                        .expect("selected execution ordinal has an address");
                    architecture
                        .build_unit(address.group(), address.index(), context)
                        .map_err(|error| Error::ArchitectureModel(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (store, report) = quantize_exact_realtime_tasks(
                Arc::clone(&self.store),
                architecture.static_modules(),
                &units,
                quantization,
                tasks,
                context,
            )?;
            self.store = store;
            self.materialization = Some(report);
        } else if self.transform.is_some() {
            return Err(Error::ArchitectureModel(
                "selected MLX transform has no source-format architecture".into(),
            ));
        }
        let (static_bindings, mut unit_bindings) =
            selected_task_bindings(tasks, self.store.as_ref(), self.local_layout.as_deref())?;
        let layout = selected.execution_units().clone();
        let (policy, metadata) = prepare_layerwise_policy_with_bindings(
            Arc::clone(&self.store),
            architecture,
            (),
            std::marker::PhantomData::<MlxKeyValueState>,
            self.residency,
            context,
            &self.weights_stream,
            |_| false,
            move |_modules, _store| Ok(static_bindings),
            move |_ordinal, address, _path, _unit, _store, _stream| {
                let group = layout
                    .group_id(address.group())
                    .ok_or_else(|| {
                        Error::ArchitectureModel("selected unit group is missing".into())
                    })?
                    .clone();
                unit_bindings
                    .remove(&ParameterGroupOwner::execution_unit(group, address.index()))
                    .ok_or_else(|| {
                        Error::ArchitectureModel(format!(
                            "selected unit bindings are missing for {address:?}"
                        ))
                    })
            },
        )?;
        let mut metadata = metadata;
        metadata.set_materialization(self.materialization.clone());
        self.metadata = Some(metadata);
        Ok(RealizedRealtimePolicy::new(policy, selected.residency()))
    }

    fn realize_state(
        &mut self,
        selected: &SelectedRealtimeStateRealization,
        _context: &Stream,
    ) -> Result<RealizedRealtimeState<Self::State>, Self::Error> {
        let state = MlxKeyValueState::device(selected.layout().clone())?;
        Ok(RealizedRealtimeState::new(state, selected.clone()))
    }
}

/// MLX mechanisms bound to one architecture-constructed realtime execution.
///
/// Architecture-owned configuration, realization, state geometry, and frame
/// semantics remain on the surrounding neutral execution handle. This value
/// erases only the concrete MLX traversal and resource mechanisms needed by
/// that handle.
pub struct MlxRealtimeExecution {
    artifact_identity: ArtifactIdentity,
    metadata: LayerwiseModelMetadata,
    payload: Rc<OrdinaryRetirement<RealtimeExecutionPayload>>,
    poisoned: Rc<Cell<bool>>,
}

struct RealtimeExecutionPayload {
    execution: Box<dyn ErasedRealtimeExecutionContract>,
    resources: Arc<SelectedRealtimeResources>,
}

struct RealtimeSubmissionRetention {
    payload: RefCell<Option<Rc<OrdinaryRetirement<RealtimeExecutionPayload>>>>,
    poisoned: Rc<Cell<bool>>,
}

impl Retention for RealtimeSubmissionRetention {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.poisoned.set(true);
        }
        if status.settled {
            self.payload.borrow_mut().take();
        }
    }
}

/// Installed before eager input/model/sampling work. Its payload clone is made
/// only after the mutable execution borrow has ended, including on unwinding.
struct RealtimeSubmissionGuard<'a> {
    model: &'a mut MlxRealtimeExecution,
    recovery: Recovery<Rc<RealtimeSubmissionRetention>>,
    succeeded: bool,
}

impl RealtimeSubmissionGuard<'_> {
    fn finish(&mut self) -> Status {
        self.recovery.seal();
        let status = self.recovery.progress();
        if !self.succeeded || status.failed || status.blocked {
            self.model.poisoned.set(true);
        }
        if !status.settled {
            *self.recovery.retention().payload.borrow_mut() = Some(Rc::clone(&self.model.payload));
        }
        status
    }
}

impl Drop for RealtimeSubmissionGuard<'_> {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Native resources whose lifetime must cover every submitted completion.
pub(crate) struct SelectedRealtimeResources {
    inner: OrdinaryRetirement<RealtimeResourcePayload>,
}

struct RealtimeResourcePayload {
    _store: SharedCheckpointSource,
    world: Option<Arc<crate::backend::runtime::distributed::Group>>,
    stream: Stream,
    _weights_stream: Stream,
    poisoned: Rc<Cell<bool>>,
}

impl SelectedRealtimeResources {
    pub(crate) fn poison(&self) {
        self.inner.poisoned.set(true);
    }

    fn validate_stream(&self, stream: &Stream) -> Result<(), Error> {
        crate::backend::validate_native_execution_target(
            &self.inner.stream.get_device()?,
            None,
            stream,
            None,
        )
    }

    fn validate_context(
        &self,
        stream: &Stream,
        world: Option<&crate::backend::runtime::distributed::Group>,
    ) -> Result<(), Error> {
        crate::backend::validate_native_execution_target(
            &self.inner.stream.get_device()?,
            self.inner
                .world
                .as_deref()
                .map(|group| group.native_group()),
            stream,
            world.map(|group| group.native_group()),
        )
    }
}

impl MlxRealtimeExecution {
    fn ensure_healthy(&self) -> Result<(), Error> {
        if self.poisoned.get() {
            return Err(Error::ArchitectureModel(
                "MLX realtime execution is poisoned after a failed submission; construct a new execution".into(),
            ));
        }
        Ok(())
    }

    /// Covers a complete host operation, including work before/after traversal.
    pub(crate) fn with_submission<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
        self.ensure_healthy()?;
        if Rc::strong_count(&self.payload) != 1 {
            return Err(Error::ArchitectureModel(
                "MLX realtime execution still has unresolved native work".into(),
            ));
        }
        let recovery = Recovery::begin(Rc::new(RealtimeSubmissionRetention {
            payload: RefCell::new(None),
            poisoned: Rc::clone(&self.poisoned),
        }))?;
        let mut guard = RealtimeSubmissionGuard {
            model: self,
            recovery,
            succeeded: false,
        };
        let result = operation(guard.model);
        guard.succeeded = result.is_ok();
        let status = guard.finish();
        if result.is_ok() && (status.failed || status.blocked) {
            return Err(Error::ArchitectureModel(
                "MLX realtime native work failed; unresolved resources remain retained".into(),
            ));
        }
        result
    }

    /// Parameter topology and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.ensure_healthy()?;
        self.payload.execution.residency_report()
    }

    /// Disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.ensure_healthy()?;
        self.payload.execution.dense_stream_report()
    }

    /// Per-execution-group residency reports.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.ensure_healthy()?;
        self.payload.execution.execution_group_reports()
    }

    /// Identity of the checkpoint payload bound to these mechanisms.
    pub fn artifact_identity(&self) -> &ArtifactIdentity {
        &self.artifact_identity
    }

    /// Executes one neutral prepared decision traversal on selected construction.
    pub fn execute_selected_realtime(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
    ) -> Result<(crate::MlxTensor, moshi::ForwardContext<crate::MlxTensor>), Error> {
        self.validate_stream(stream)?;
        self.with_submission(|model| {
            Rc::get_mut(&mut model.payload)
                .ok_or_else(|| {
                    Error::ArchitectureModel(
                        "MLX realtime execution has unresolved native owners".into(),
                    )
                })?
                .execution
                .execute_decisions(state, temporal, driver, stream)
        })
    }

    /// Creates request-local resident key/value state from the neutral layout.
    pub fn new_realtime_state(&self) -> Result<MlxKeyValueState, Error> {
        self.ensure_healthy()?;
        MlxKeyValueState::device(self.payload.execution.selected().state().layout().clone())
            .map_err(Into::into)
    }

    /// Clones exact store, stream, and collective ownership into a completion.
    pub(crate) fn completion_resources(&self) -> Arc<SelectedRealtimeResources> {
        Arc::clone(&self.payload.resources)
    }

    pub(crate) fn validate_stream(&self, stream: &Stream) -> Result<(), Error> {
        self.ensure_healthy()?;
        self.payload.resources.validate_stream(stream)
    }

    pub(crate) fn validate_context(
        &self,
        stream: &Stream,
        world: Option<&crate::backend::runtime::distributed::Group>,
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        self.payload.resources.validate_context(stream, world)
    }
}

struct SelectedMoshiRealtimeMechanismVisitor {
    artifact_identity: ArtifactIdentity,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    target_quantization: Option<eredu_checkpoint::WeightQuantization>,
    effective_model_type: String,
    residency: eredu_runtime::LayerWeightResidency,
    stream: Stream,
    weights_stream: Stream,
    distributed: Option<crate::backend::MlxDistributedSession>,
    resources: Arc<SelectedRealtimeResources>,
}

impl moshi::MoshiRealtimeArchitectureVisitor<MlxNeuralBackend, MlxKeyValueState>
    for SelectedMoshiRealtimeMechanismVisitor
{
    type Output = MlxRealtimeExecution;
    type Error = Error;

    fn visit<A>(
        self,
        mut prepared: moshi::PreparedMoshiRealtimeArchitecture<A>,
        store: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend, MlxKeyValueState>
            + eredu_runtime::RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display,
    {
        let parallel = prepared.take_parallel();
        let mechanisms = MlxRealtimeConstructionMechanisms::new(
            store,
            self.residency,
            self.transform,
            &self.weights_stream,
            parallel.as_ref().map(|parallel| parallel.layout().clone()),
        );
        let constructed = construct_realtime_model::<A, MlxNeuralBackend, _>(
            prepared.take_architecture(),
            prepared.take_source_architecture(),
            prepared.take_contract(),
            mechanisms,
            &self.stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let (execution, _initial_state) = constructed.into_execution_and_state();
        let mut metadata = execution.mechanisms().metadata().cloned().ok_or_else(|| {
            Error::ArchitectureModel("neutral MLX construction produced no metadata".into())
        })?;
        metadata.set_effective_model_type(&self.effective_model_type);
        metadata.set_quantization(self.target_quantization);
        let selected = execution.selected().clone();
        let execution: Box<dyn ErasedRealtimeExecutionContract> = match parallel {
            None => Box::new(DirectRealtimeExecution {
                selected,
                execution,
            }),
            Some(parallel) => {
                let (local_layout, _geometry, communication, tensor_group, execution_plan) =
                    parallel.into_parts();
                if local_layout.is_empty() {
                    return Err(Error::Parallel(
                        "selected Moshi tensor-parallel layout is empty".into(),
                    ));
                }
                let distributed = self.distributed.ok_or_else(|| {
                    Error::Parallel(
                        "selected Moshi tensor parallelism has no realized MLX communication"
                            .into(),
                    )
                })?;
                let (partition_communication, parallel, _sampling, communication_executor) =
                    distributed.into_partition_communication(
                        communication.clone(),
                        Some(tensor_group),
                        tensor_group,
                    )?;
                let parallel = parallel.ok_or_else(|| {
                    Error::Parallel("selected Moshi tensor group was not realized".into())
                })?;
                let (selected, runtime, _mechanisms) = execution.into_parts();
                let residency = selected.residency().execution_residency();
                match runtime {
                    RealtimeLayerwiseRuntime::Resident(runtime) => {
                        let executor = eredu_runtime::LayerwiseTraversalPartitionExecutor::new(
                            runtime, parallel,
                        );
                        Box::new(PartitionedRealtimeExecution {
                            selected,
                            execution: eredu_runtime::LayerwiseTraversalRuntime::partitioned(
                                Box::new(
                                    eredu_runtime::PartitionedTextRuntime::new(
                                        execution_plan,
                                        executor,
                                        partition_communication,
                                        communication_executor,
                                        eredu_runtime::NoBoundaryTransport,
                                        eredu_runtime::NoOutputPublisher,
                                        eredu_runtime::NoCommitAgreement,
                                        residency,
                                        None,
                                    )
                                    .map_err(|error| Error::Parallel(error.to_string()))?,
                                ),
                            ),
                        })
                    }
                    RealtimeLayerwiseRuntime::Bounded(runtime) => {
                        let executor = eredu_runtime::LayerwiseTraversalPartitionExecutor::new(
                            runtime, parallel,
                        );
                        Box::new(PartitionedRealtimeExecution {
                            selected,
                            execution: eredu_runtime::LayerwiseTraversalRuntime::partitioned(
                                Box::new(
                                    eredu_runtime::PartitionedTextRuntime::new(
                                        execution_plan,
                                        executor,
                                        partition_communication,
                                        communication_executor,
                                        eredu_runtime::NoBoundaryTransport,
                                        eredu_runtime::NoOutputPublisher,
                                        eredu_runtime::NoCommitAgreement,
                                        residency,
                                        Some(()),
                                    )
                                    .map_err(|error| Error::Parallel(error.to_string()))?,
                                ),
                            ),
                        })
                    }
                }
            }
        };
        Ok(MlxRealtimeExecution {
            artifact_identity: self.artifact_identity,
            metadata,
            poisoned: Rc::clone(&self.resources.inner.poisoned),
            payload: Rc::new(OrdinaryRetirement::new(RealtimeExecutionPayload {
                execution,
                resources: self.resources,
            })),
        })
    }
}

/// Constructs one already selected model through neutral construction and its
/// architecture-selected replicated or pure-TP executor.
pub fn materialize_selected(
    prepared: moshi::PreparedMoshiRealtimeSource,
    world: Option<Arc<crate::backend::runtime::distributed::Group>>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<moshi::MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
    ordinary_retirement::reclaim();
    let execution_descriptor = prepared.selected().execution_descriptor();
    let target_config = prepared.selected().execution_config().clone();
    let selected = prepared.selected().selected().clone();
    let effective_model_type = target_config.effective_model_type().as_str().to_owned();
    let parallel_manifest = prepared
        .selected()
        .parallel()
        .map(|parallel| parallel.communication().clone());
    let artifact_identity = prepared.artifact_identity();
    let lowering = prepared.lowering();
    let store = Arc::clone(prepared.source());
    let transform = lowering.transform();
    let target_quantization = lowering.target();
    let distributed = parallel_manifest
        .as_ref()
        .map(|communication| {
            let world = world.as_deref().ok_or_else(|| {
                Error::Parallel(
                    "selected Moshi tensor parallelism requires a native world group".into(),
                )
            })?;
            crate::backend::MlxDistributedSession::from_manifest(
                communication,
                world.native_group(),
                stream,
            )
        })
        .transpose()?;
    let resources = Arc::new(SelectedRealtimeResources {
        inner: OrdinaryRetirement::new(RealtimeResourcePayload {
            _store: Arc::clone(&store),
            world: parallel_manifest.as_ref().and(world.clone()),
            stream: stream.clone(),
            _weights_stream: weights_stream.clone(),
            poisoned: Rc::new(Cell::new(false)),
        }),
    });
    let execution =
        moshi::visit_selected_moshi_realtime_architecture::<MlxNeuralBackend, MlxKeyValueState, _>(
            prepared,
            stream,
            SelectedMoshiRealtimeMechanismVisitor {
                artifact_identity,
                transform,
                target_quantization,
                effective_model_type,
                residency: selected.residency(),
                stream: stream.clone(),
                weights_stream: weights_stream.clone(),
                distributed,
                resources,
            },
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
    Ok(execution_descriptor.bind(execution))
}

#[cfg(test)]
#[path = "moshi_recovery_tests.rs"]
mod recovery_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::{AffineQuantization, WeightQuantization};

    #[test]
    fn source_and_target_configuration_identities_are_distinct() {
        let source =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        let target = source
            .with_native_quantization(Some(WeightQuantization::Affine(
                AffineQuantization::new(32, 4).unwrap(),
            )))
            .unwrap();
        assert_eq!(source.native_quantization(), None);
        assert_eq!(source.checkpoint_layout(), target.checkpoint_layout());
        assert_ne!(
            source.architecture_fingerprint(),
            target.architecture_fingerprint()
        );
    }
}
