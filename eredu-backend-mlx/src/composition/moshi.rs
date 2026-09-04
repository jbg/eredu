//! Production MLX composition for the backend-neutral Moshi-family model.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use eredu_architectures::moshi::{self, MoshiConfig};
use eredu_checkpoint::store::{
    CheckpointSource, ResolvedCheckpointSource, SharedCheckpointSource, TensorSelection,
};
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
    runtime::{
        cache::state::MlxKeyValueState,
        checkpoint::artifact::{fingerprint_artifact, ArtifactFile, LoadedArtifactIdentity},
        execution::{
            generic::{
                prepare_layerwise_policy_with_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
            },
            layerwise::{
                open_safetensors_weight_store, quantize_exact_realtime_tasks, shard_layer_bindings,
            },
        },
        generation::MlxSamplingBackend,
    },
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

impl<U> RealtimePolicyReports for MlxLayerwisePolicy<U> {
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
pub struct MlxRealtimeConstructionMechanisms<U> {
    store: SharedCheckpointSource,
    residency: eredu_runtime::LayerWeightResidency,
    weights_stream: Stream,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    pending: Option<MlxLayerwisePolicy<U>>,
    metadata: Option<LayerwiseModelMetadata>,
    local_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
}

impl<U> MlxRealtimeConstructionMechanisms<U> {
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
    let mut pinned = Vec::new();
    let mut units = BTreeMap::<ParameterGroupOwner, Vec<WeightBinding>>::new();
    for task in tasks {
        let destination = match task.owner() {
            ParameterGroupOwner::StaticRole(_) | ParameterGroupOwner::StaticAnyOf(_) => &mut pinned,
            ParameterGroupOwner::ExecutionUnit { .. } => {
                units.entry(task.owner().clone()).or_default()
            }
            _ => {
                return Err(Error::ArchitectureModel(
                    "unsupported future realtime parameter owner".into(),
                ))
            }
        };
        let transformed = matches!(
            task.lowering().kind(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        );
        for component in task.components() {
            let requirement = component.requirement();
            let target = requirement.target().as_str();
            let binding = if transformed {
                let metadata = store.source_metadata(target)?;
                WeightBinding::new(
                    target,
                    target,
                    TensorSelection::Full,
                    metadata.encoded_byte_len,
                )?
            } else {
                let output = component.recipe_output().ok_or_else(|| {
                    Error::ArchitectureModel(format!(
                        "selected MLX recipe component {target:?} has no exact output metadata"
                    ))
                })?;
                match requirement.recipe_owner() {
                    Some(owner) if owner != requirement.target() => {
                        WeightBinding::alias(target, owner.as_str(), output.byte_len())?
                    }
                    _ => WeightBinding::from_recipe(
                        target,
                        component.recipe().cloned().ok_or_else(|| {
                            Error::ArchitectureModel(format!(
                                "selected MLX component {target:?} has no source recipe"
                            ))
                        })?,
                        output.byte_len(),
                    )?,
                }
            }
            .with_logical_target(task.lowering().target().as_str())?;
            destination.push(binding);
        }
    }
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
    U: Parameterized<crate::MlxTensor>,
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
    artifact_identity: LoadedArtifactIdentity,
    metadata: LayerwiseModelMetadata,
    execution: Box<dyn ErasedRealtimeExecutionContract>,
    resources: Arc<SelectedRealtimeResources>,
}

/// Native resources whose lifetime must cover every submitted completion.
pub(crate) struct SelectedRealtimeResources {
    _store: SharedCheckpointSource,
    _world: Option<Arc<crate::backend::runtime::distributed::Group>>,
    _stream: Stream,
    _weights_stream: Stream,
}

impl MlxRealtimeExecution {
    /// Parameter topology and residency metadata.
    pub fn metadata(&self) -> &LayerwiseModelMetadata {
        &self.metadata
    }

    /// Logical residency and transfer telemetry.
    pub fn residency_report(&self) -> Result<ResidencyReport, Error> {
        self.execution.residency_report()
    }

    /// Disk-stream telemetry when that policy is active.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error> {
        self.execution.dense_stream_report()
    }

    /// Per-execution-group residency reports.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        self.execution.execution_group_reports()
    }

    /// Identity of the checkpoint payload bound to these mechanisms.
    pub fn artifact_identity(&self) -> &LoadedArtifactIdentity {
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
        self.execution
            .execute_decisions(state, temporal, driver, stream)
    }

    /// Creates request-local resident key/value state from the neutral layout.
    pub fn new_realtime_state(&self) -> Result<MlxKeyValueState, Error> {
        MlxKeyValueState::device(self.execution.selected().state().layout().clone())
            .map_err(Into::into)
    }

    /// Clones exact store, stream, and collective ownership into a completion.
    pub(crate) fn completion_resources(&self) -> Arc<SelectedRealtimeResources> {
        Arc::clone(&self.resources)
    }
}

struct SelectedMlxConstructionVisitor {
    artifact_identity: LoadedArtifactIdentity,
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
    for SelectedMlxConstructionVisitor
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
            execution,
            resources: self.resources,
        })
    }
}

/// Constructs one already selected model through neutral construction and its
/// architecture-selected replicated or pure-TP executor.
pub fn materialize_selected(
    prepared: moshi::PreparedMoshiRealtime,
    world: Option<Arc<crate::backend::runtime::distributed::Group>>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<moshi::MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
    let execution_descriptor = prepared.execution_descriptor();
    let model_dir = prepared.artifact_root().to_owned();
    let source_path = prepared.checkpoint_source().to_owned();
    let checkpoint_plan = prepared.checkpoint_plan().clone();
    let source_config = prepared.source_config().clone();
    let target_config = prepared.execution_config().clone();
    let selected = prepared.selected().clone();
    let (transform, target_quantization) = selected_quantization(&selected)?;
    let effective_model_type = target_config.effective_model_type().as_str().to_owned();
    let distributed = prepared
        .parallel()
        .map(|parallel| {
            let world = world.as_deref().ok_or_else(|| {
                Error::Parallel(
                    "selected Moshi tensor parallelism requires a native world group".into(),
                )
            })?;
            crate::backend::MlxDistributedSession::from_manifest(
                parallel.communication(),
                world.native_group(),
                stream,
            )
        })
        .transpose()?;
    let artifact_identity = artifact_identity(&model_dir, &source_path, &source_config)?;
    let source_store =
        open_safetensors_weight_store(&source_path, selected.residency().max_cached_shards())?;
    let checkpoint_contract = eredu_checkpoint::validation::resolve_safetensors_plan(
        source_store.as_ref(),
        &checkpoint_plan,
    )
    .map_err(|validation| {
        Error::ArchitectureModel(format!(
            "selected Moshi checkpoint contract no longer resolves: {validation:?}"
        ))
    })?;
    let store: SharedCheckpointSource = Arc::new(ResolvedCheckpointSource::new(
        source_store,
        checkpoint_contract,
    ));
    let resources = Arc::new(SelectedRealtimeResources {
        _store: Arc::clone(&store),
        _world: world.clone(),
        _stream: stream.clone(),
        _weights_stream: weights_stream.clone(),
    });
    let execution =
        moshi::visit_selected_moshi_realtime_architecture::<MlxNeuralBackend, MlxKeyValueState, _>(
            prepared,
            store,
            stream,
            SelectedMlxConstructionVisitor {
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

fn selected_quantization(
    selected: &SelectedRealtimeRealization,
) -> Result<
    (
        Option<eredu_checkpoint::WeightQuantization>,
        Option<eredu_checkpoint::WeightQuantization>,
    ),
    Error,
> {
    if selected.weight_lowerings().is_empty() {
        return Err(Error::ArchitectureModel(
            "selected realtime execution has no weight lowerings".into(),
        ));
    }
    let mut target = None;
    for quantization in selected
        .weight_lowerings()
        .iter()
        .filter_map(|lowering| lowering.descriptor().executable().weight_quantization())
    {
        if target.is_some_and(|selected| selected != quantization) {
            return Err(Error::ArchitectureModel(
                "selected realtime execution mixes incompatible executable quantization formats"
                    .into(),
            ));
        }
        target = Some(quantization);
    }

    let mut transform = None;
    for lowering in selected.weight_lowerings().iter().filter(|lowering| {
        matches!(
            lowering.kind(),
            eredu_runtime::WeightLoweringKind::Transform
                | eredu_runtime::WeightLoweringKind::DerivedTransform
        )
    }) {
        let quantization = lowering
            .descriptor()
            .executable()
            .weight_quantization()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "selected transforming realtime lowering has no executable quantization".into(),
                )
            })?;
        if transform.is_some_and(|selected| selected != quantization) {
            return Err(Error::ArchitectureModel(
                "selected realtime transform lowerings disagree on quantization".into(),
            ));
        }
        transform = Some(quantization);
    }
    Ok((transform, target))
}

fn artifact_identity(
    model_dir: &Path,
    source: &Path,
    config: &MoshiConfig,
) -> Result<LoadedArtifactIdentity, Error> {
    let paths = if source.is_dir() {
        crate::backend::runtime::checkpoint::load::safetensors_files(source)?
    } else {
        vec![source.to_owned()]
    };
    let files = paths.into_iter().map(|path| {
        let logical = path
            .strip_prefix(model_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        ArtifactFile::new(logical, path)
    });
    fingerprint_artifact(config.effective_model_type().as_str(), files)
}

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
