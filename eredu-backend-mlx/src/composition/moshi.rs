//! Production MLX composition for the backend-neutral Moshi-family model.

use eredu_runtime::parameter_operations::LayeredParameterOwner;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
    sync::Arc,
};

#[cfg(test)]
use eredu_architectures::moshi::MoshiConfig;
use eredu_architectures::moshi::{self};
use eredu_checkpoint::store::{CheckpointSource, RetainedCheckpointSource};
use eredu_core::artifact::{ArtifactIdentity, DeferredArtifactIdentity};
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
                prepare_layerwise_policy_with_bindings,
                prepare_layerwise_policy_with_prepared_manager, MlxLayerwisePolicy,
                MlxResidentPolicy, PreparedLayerwiseManager,
            },
            layerwise::{quantize_exact_realtime_tasks, shard_layer_bindings},
        },
        generation::MlxSamplingBackend,
    },
    submission_recovery::{self, Recovery, Retention, Status},
};
mod bounded_source;
mod model_source;
mod operation_source;
mod workspace;
use operation_source::RealtimeOperationPolicy;
pub(crate) use operation_source::{RealtimeOperationPlan, RealtimeOperationRecipe};
mod execution_failure;
mod original_frame;
pub(crate) use original_frame::OriginalRealtimeModelLease;
pub(crate) use workspace::RealtimeWorkspaceVisitor;

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
    fn parallel_communication(
        &self,
    ) -> Option<(
        &crate::backend::MlxDistributedSession,
        eredu_core::CollectiveGroupId,
    )> {
        None
    }
    fn collect_retained_module_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error>;
    fn realtime_operation_plan(
        &self,
        stream: &Stream,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<RealtimeOperationPlan, Error>;

    fn with_workspace_frame(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        context: &eredu_nn::workspace::WorkspaceContext,
        visitor: &mut dyn RealtimeWorkspaceVisitor,
    ) -> Result<(), Error>;

    fn execute_original_decisions(
        &mut self,
        _state: &mut MlxKeyValueState,
        _temporal: &[crate::MlxTensor],
        _driver: &mut SequentialDecisionDriver<
            MlxSamplingBackend,
            eredu_runtime::GenerationSampler,
        >,
        _stream: &Stream,
        _parallel: &crate::backend::runtime::distributed::Group,
        _funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        Err(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
        ))
    }
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
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    >;
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
    communication: crate::backend::MlxDistributedSession,
    tensor_group: eredu_core::CollectiveGroupId,
}

impl<A> ErasedRealtimeExecutionContract for DirectRealtimeExecution<A>
where
    A: moshi::MoshiRealtimeExecutionArchitecture<MlxNeuralBackend, MlxKeyValueState>
        + eredu_runtime::ParallelLayeredArchitecture<MlxNeuralBackend, MlxKeyValueState>
        + 'static,
    A::Error: std::fmt::Display,
{
    fn collect_retained_module_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        storage.include_retained_values(|visitor| {
            Ok::<_, Error>(match self.execution.execution() {
                RealtimeLayerwiseRuntime::Resident(runtime) => {
                    runtime.visit_retained_values(visitor)
                }
                RealtimeLayerwiseRuntime::Bounded(runtime) => {
                    runtime.visit_retained_values(visitor)
                }
            })
        })
    }
    fn realtime_operation_plan(
        &self,
        stream: &Stream,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<RealtimeOperationPlan, Error> {
        match self.execution.execution() {
            RealtimeLayerwiseRuntime::Resident(runtime) => runtime
                .policy()
                .realtime_operation_plan(stream, allocation, pool, context),
            RealtimeLayerwiseRuntime::Bounded(runtime) => runtime
                .policy()
                .realtime_operation_plan(stream, allocation, pool, context),
        }
    }
    fn with_workspace_frame(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        context: &eredu_nn::workspace::WorkspaceContext,
        visitor: &mut dyn RealtimeWorkspaceVisitor,
    ) -> Result<(), Error> {
        match self.execution.execution() {
            RealtimeLayerwiseRuntime::Resident(runtime) => {
                workspace::resident(runtime, context, visitor)
            }
            RealtimeLayerwiseRuntime::Bounded(runtime) => {
                workspace::layerwise(runtime, allocation, context, visitor)
            }
        }
    }
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
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        moshi::execute_detached_replicated_moshi_frame(
            &mut self.execution,
            state,
            temporal,
            driver,
            stream,
        )
        .map_err(|cause| match funding {
            Some(funding) => {
                Error::Neural(funding.metadata_source(execution_failure::Source(cause)))
            }
            None => Error::ArchitectureModel(cause.to_string()),
        })
    }
}

trait RealtimePolicyReports {
    fn residency_report(&self) -> Result<ResidencyReport, Error>;
    fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, Error>;
    fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error>;
}

impl<U: 'static> RealtimePolicyReports for MlxResidentPolicy<U> {
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
    P: eredu_runtime::LayerwisePolicy<MlxNeuralBackend, A::Unit, Error = Error>
        + RealtimePolicyReports
        + workspace::RealtimeWorkspacePolicy<A>
        + RealtimeOperationPolicy<A::Unit>
        + 'static,
    P::Error: std::fmt::Display,
{
    fn parallel_communication(
        &self,
    ) -> Option<(
        &crate::backend::MlxDistributedSession,
        eredu_core::CollectiveGroupId,
    )> {
        Some((&self.communication, self.tensor_group))
    }
    fn collect_retained_module_storage(
        &self,
        storage: &mut crate::backend::runtime::residency::storage::RetainedStorage,
    ) -> Result<(), Error> {
        self.communication.collect_retained_buffers(storage)?;
        storage.include_retained_values(|visitor| {
            Ok::<_, Error>(match &self.execution {
                eredu_runtime::LayerwiseTraversalRuntime::Direct(runtime) => {
                    runtime.visit_retained_values(visitor)
                }
                eredu_runtime::LayerwiseTraversalRuntime::Partitioned(runtime) => runtime
                    .traversal_executor()
                    .runtime()
                    .visit_retained_values(visitor),
            })
        })
    }
    fn realtime_operation_plan(
        &self,
        stream: &Stream,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<RealtimeOperationPlan, Error> {
        match &self.execution {
            eredu_runtime::LayerwiseTraversalRuntime::Direct(runtime) => runtime
                .policy()
                .realtime_operation_plan(stream, allocation, pool, context),
            eredu_runtime::LayerwiseTraversalRuntime::Partitioned(runtime) => {
                if !runtime.is_fully_local_traversal() {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    ));
                }
                runtime
                    .traversal_executor()
                    .runtime()
                    .policy()
                    .realtime_operation_plan(stream, allocation, pool, context)
            }
        }
    }
    fn with_workspace_frame(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        context: &eredu_nn::workspace::WorkspaceContext,
        visitor: &mut dyn RealtimeWorkspaceVisitor,
    ) -> Result<(), Error> {
        match &self.execution {
            eredu_runtime::LayerwiseTraversalRuntime::Direct(runtime) => {
                P::with_workspace_frame(runtime, allocation, context, None, visitor)
            }
            eredu_runtime::LayerwiseTraversalRuntime::Partitioned(runtime) => {
                if !runtime.is_fully_local_traversal() {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    ));
                }
                let executor = runtime.traversal_executor();
                let group = executor.parallel_context();
                let parallel =
                    eredu_nn::workspace::WorkspaceParallelContext::new(group.rank(), group.size())
                        .map_err(Error::Neural)?;
                P::with_workspace_frame(
                    executor.runtime(),
                    allocation,
                    context,
                    Some(parallel),
                    visitor,
                )
            }
        }
    }
    fn execute_original_decisions(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
        parallel: &crate::backend::runtime::distributed::Group,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        let eredu_runtime::LayerwiseTraversalRuntime::Partitioned(runtime) = &self.execution else {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        };
        let expected = runtime.traversal_executor().parallel_context();
        if !runtime.is_fully_local_traversal()
            || !parallel.has_original_parallel()
            || !parallel
                .native_group()
                .shares_native_handle(expected.native_group())
            || parallel.rank() != expected.rank()
            || parallel.size() != expected.size()
            || !parallel
                .retained_source()
                .zip(expected.retained_source())
                .is_some_and(|(actual, expected)| actual.same_source(expected))
        {
            return Err(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        moshi::execute_detached_partitioned_moshi_frame_with_parallel(
            &mut self.execution,
            state,
            temporal,
            driver,
            stream,
            parallel,
        )
        .map_err(|cause| Error::Neural(funding.metadata_source(execution_failure::Source(cause))))
    }
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
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        moshi::execute_detached_partitioned_moshi_frame(
            &mut self.execution,
            state,
            temporal,
            driver,
            stream,
        )
        .map_err(|cause| match funding {
            Some(funding) => {
                Error::Neural(funding.metadata_source(execution_failure::Source(cause)))
            }
            None => Error::ArchitectureModel(cause.to_string()),
        })
    }
}

/// Generic MLX storage/materialization mechanisms used by the neutral constructor.
pub struct MlxRealtimeConstructionMechanisms<U: 'static> {
    store: RetainedCheckpointSource,
    residency: eredu_runtime::LayerWeightResidency,
    weights_stream: Stream,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    materialization: Option<eredu_runtime::WeightMaterializationReport>,
    pending: Option<MlxLayerwisePolicy<U>>,
    loaded_residency: Option<crate::backend::runtime::residency::manager::ResidencyManager>,
    metadata: Option<LayerwiseModelMetadata>,
    local_layout: Option<Arc<eredu_runtime::LocalModelLayout>>,
    prepared_manager: Option<PreparedLayerwiseManager>,
}

impl<U: 'static> MlxRealtimeConstructionMechanisms<U> {
    fn new(
        store: RetainedCheckpointSource,
        residency: eredu_runtime::LayerWeightResidency,
        transform: Option<eredu_checkpoint::WeightQuantization>,
        weights_stream: &Stream,
        local_layout: Option<eredu_runtime::LocalModelLayout>,
        prepared_manager: Option<PreparedLayerwiseManager>,
    ) -> Self {
        Self {
            store,
            residency,
            weights_stream: weights_stream.clone(),
            transform,
            materialization: None,
            pending: None,
            loaded_residency: None,
            metadata: None,
            local_layout: local_layout.map(Arc::new),
            prepared_manager,
        }
    }

    /// Returns construction metadata after a policy has been realized.
    pub fn metadata(&self) -> Option<&LayerwiseModelMetadata> {
        self.metadata.as_ref()
    }
}
fn selected_task_bindings(
    prepared: eredu_runtime::PreparedRealtimeTaskBindingPlan<'_>,
    store: &dyn CheckpointSource,
    local_layout: Option<&eredu_runtime::LocalModelLayout>,
) -> Result<
    (
        Vec<WeightBinding>,
        BTreeMap<ParameterGroupOwner, Vec<WeightBinding>>,
    ),
    Error,
> {
    let (mut pinned, mut units) = prepared
        .materialized(store)
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
        let bindings = eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
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
                self.store.clone(),
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
            selected_task_bindings(bindings, self.store.as_ref(), self.local_layout.as_deref())?;
        let layout = selected.execution_units().clone();
        let (policy, metadata) = prepare_layerwise_policy_with_bindings(
            self.store.clone(),
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
        self.loaded_residency = Some(policy.residency_manager().clone());
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
        let bindings = eredu_runtime::preflight_realtime_materialization_tasks::<MlxNeuralBackend>(
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
                self.store.clone(),
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
        let (static_bindings, unit_bindings) =
            selected_task_bindings(bindings, self.store.as_ref(), self.local_layout.as_deref())?;
        let layout = selected.execution_units().clone();
        let ordered = bounded_source::ordered_units(&layout, unit_bindings)?;
        let (policy, metadata) = prepare_layerwise_policy_with_prepared_manager(
            self.store.clone(),
            architecture,
            (),
            std::marker::PhantomData::<MlxKeyValueState>,
            self.residency,
            context,
            &self.weights_stream,
            |_| false,
            layout,
            static_bindings,
            ordered,
            Vec::new(),
            self.prepared_manager.take(),
        )?;
        let mut metadata = metadata;
        metadata.set_materialization(self.materialization.clone());
        self.metadata = Some(metadata);
        self.loaded_residency = Some(policy.residency_manager().clone());
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
    artifact_identity: DeferredArtifactIdentity,
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
    loaded: std::cell::OnceCell<model_source::LoadedRealtimeSource>,
    execution_identity: eredu_runtime::working_memory::InferenceExecutionIdentity,
    ingress: eredu_runtime::RealtimeIngressContract,
    _store: RetainedCheckpointSource,
    world: Option<Arc<crate::backend::runtime::distributed::Group>>,
    backend: Rc<crate::backend::MlxBackend<'static>>,
    poisoned: Rc<Cell<bool>>,
}

impl SelectedRealtimeResources {
    /// Stable identity of this exact selected loaded owner, shared by its frame
    /// sources and completions. A fresh model creates a distinct identity.
    pub(crate) fn execution_identity(
        &self,
    ) -> &eredu_runtime::working_memory::InferenceExecutionIdentity {
        &self.inner.execution_identity
    }

    /// Proves that these exact loaded resources completed their source
    /// publication into this backend pool. This grants no frame allocation.
    pub(crate) fn validate_loaded_source(&self) -> Result<(), Error> {
        self.inner
            .loaded
            .get()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?
            .validate(self.execution_identity(), self.backend().memory_ledger())
    }

    /// The architecture-validated schedule is constructed once at load time.
    /// Managed frames only inspect this retained source.
    pub(crate) fn ingress_contract(&self) -> &eredu_runtime::RealtimeIngressContract {
        &self.inner.ingress
    }

    pub(crate) fn original_copy_environment(
        &self,
    ) -> Result<
        crate::backend::OriginalCopyEnvironment<'_>,
        crate::backend::OriginalCopyEnvironmentError,
    > {
        self.inner.backend.original_copy_environment()
    }

    pub(crate) fn backend(&self) -> &crate::backend::MlxBackend<'static> {
        &self.inner.backend
    }

    pub(crate) fn poison(&self) {
        self.inner.poisoned.set(true);
    }

    fn validate_stream(&self, stream: &Stream) -> Result<(), Error> {
        crate::backend::validate_native_execution_target(
            &self.inner.backend.stream().get_device()?,
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
            &self.inner.backend.stream().get_device()?,
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

    /// Borrows the selected immutable architecture and actual native parameter
    /// owners for one cold frame trace. This does not acquire execution units,
    /// mutate residency, evaluate tensors or grant native submission authority.
    pub(crate) fn parallel_communication(
        &self,
    ) -> Option<(
        &crate::backend::MlxDistributedSession,
        eredu_core::CollectiveGroupId,
    )> {
        self.payload.execution.parallel_communication()
    }

    pub(crate) fn with_workspace_frame(
        &self,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        context: &eredu_nn::workspace::WorkspaceContext,
        visitor: &mut dyn RealtimeWorkspaceVisitor,
    ) -> Result<(), Error> {
        self.ensure_healthy()?;
        self.payload
            .execution
            .with_workspace_frame(allocation, context, visitor)
    }

    /// Retains the selected policy's actual operation slot and stream source.
    /// This descriptive plan grants no frame occurrence or native submission.
    pub(crate) fn realtime_operation_plan(
        &self,
        stream: &Stream,
        allocation: crate::backend::nn::workspace::NativeAllocationFacts,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<RealtimeOperationPlan, Error> {
        self.ensure_healthy()?;
        self.validate_stream(stream)?;
        self.payload
            .execution
            .realtime_operation_plan(stream, allocation, pool, context)
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
    pub fn artifact_identity(
        &self,
    ) -> Result<ArtifactIdentity, std::sync::Arc<eredu_core::artifact::ArtifactError>> {
        self.artifact_identity.resolve()
    }

    /// Executes one neutral prepared decision traversal on selected construction.
    pub fn execute_selected_realtime(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        self.validate_stream(stream)?;
        self.with_submission(|model| {
            model.execute_realtime_body(state, temporal, driver, stream, None)
        })
    }

    fn execute_realtime_body(
        &mut self,
        state: &mut MlxKeyValueState,
        temporal: &[crate::MlxTensor],
        driver: &mut SequentialDecisionDriver<MlxSamplingBackend, eredu_runtime::GenerationSampler>,
        stream: &Stream,
        funding: Option<&eredu_nn::workspace::HostMetadataFunding>,
    ) -> Result<
        (
            Option<crate::MlxTensor>,
            moshi::ForwardContext<crate::MlxTensor>,
        ),
        Error,
    > {
        Rc::get_mut(&mut self.payload)
            .ok_or(Error::PrefillScopeUnavailable)?
            .execution
            .execute_decisions(state, temporal, driver, stream, funding)
    }

    /// Creates request-local resident key/value state from the neutral layout.
    pub fn new_realtime_state(&self) -> Result<MlxKeyValueState, Error> {
        self.ensure_healthy()?;
        MlxKeyValueState::device(self.payload.execution.selected().state().layout().clone())
            .map_err(Into::into)
    }

    /// Clones exact store, stream, and collective ownership into a completion.
    pub(crate) fn completion_resources(&self) -> Arc<SelectedRealtimeResources> {
        self.payload.resources.clone()
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
    artifact_identity: DeferredArtifactIdentity,
    transform: Option<eredu_checkpoint::WeightQuantization>,
    target_quantization: Option<eredu_checkpoint::WeightQuantization>,
    effective_model_type: String,
    residency: eredu_runtime::LayerWeightResidency,
    stream: Stream,
    weights_stream: Stream,
    distributed: Option<crate::backend::MlxDistributedSession>,
    resources: Arc<SelectedRealtimeResources>,
    loading: crate::backend::managed_memory::NativeMemoryOwner,
    prepared_manager: Option<PreparedLayerwiseManager>,
}

impl moshi::MoshiRealtimeArchitectureVisitor<MlxNeuralBackend, MlxKeyValueState>
    for SelectedMoshiRealtimeMechanismVisitor
{
    type Output = MlxRealtimeExecution;
    type Error = Error;

    fn visit<A>(
        self,
        mut prepared: moshi::PreparedMoshiRealtimeArchitecture<A>,
        store: RetainedCheckpointSource,
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
            self.prepared_manager,
        );
        let constructed = construct_realtime_model::<A, MlxNeuralBackend, _>(
            prepared.take_architecture(),
            prepared.take_source_architecture(),
            prepared.take_contract(),
            mechanisms,
            &self.stream,
        )
        .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        let (execution, initial_state) = constructed.into_execution_and_state();
        // This construction scratch is not the later request-owned decoder.
        drop(initial_state);
        let loaded_residency = execution
            .mechanisms()
            .loaded_residency
            .as_ref()
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))?
            .clone();
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
                let (partition_communication, parallel, communication_executor) =
                    distributed.retained_partition_communication(communication, tensor_group)?;
                let (selected, runtime, _mechanisms) = execution.into_parts();
                let residency = selected.residency().execution_residency();
                match runtime {
                    RealtimeLayerwiseRuntime::Resident(runtime) => {
                        let executor = eredu_runtime::LayerwiseTraversalPartitionExecutor::new(
                            runtime, parallel,
                        );
                        Box::new(PartitionedRealtimeExecution {
                            selected,
                            communication: distributed,
                            tensor_group,
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
                            communication: distributed,
                            tensor_group,
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
        let loaded = model_source::LoadedRealtimeSource::publish(
            execution.as_ref(),
            loaded_residency,
            &self.resources,
            &self.loading,
        )?;
        self.resources.inner.loaded.set(loaded).map_err(|_| {
            Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::AlreadyStarted)
        })?;
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
    materialize_selected_with_backend(
        prepared,
        world,
        Rc::new(crate::backend::MlxBackend::new(stream, weights_stream)),
    )
}

/// Shares the actual selected native owners with the returned model and every
/// completion, including failure retirement after the public context is dropped.
pub(crate) fn materialize_selected_with_backend(
    prepared: moshi::PreparedMoshiRealtimeSource,
    world: Option<Arc<crate::backend::runtime::distributed::Group>>,
    backend: Rc<crate::backend::MlxBackend<'static>>,
) -> Result<moshi::MoshiRealtimeExecution<MlxRealtimeExecution>, Error> {
    ordinary_retirement::reclaim();
    let prepared_manager = bounded_source::prepare(&prepared, &backend)?;
    let loading =
        crate::backend::managed_memory::NativeMemoryOwner::acquire(backend.memory_ledger())?;
    submission_recovery::detached_retained(loading.clone(), || {
        let stream = backend.stream();
        let weights_stream = backend.weights_stream();
        let execution_descriptor = prepared.selected().execution_descriptor();
        let target_config = prepared.selected().execution_config().clone();
        let ingress =
            moshi::realtime_ingress_contract(&target_config).map_err(Error::ArchitectureModel)?;
        let selected = prepared.selected().selected().clone();
        let effective_model_type = target_config.effective_model_type().as_str().to_owned();
        let parallel_manifest = prepared
            .selected()
            .parallel()
            .map(|parallel| parallel.communication().clone());
        let artifact_identity = prepared.deferred_artifact_identity();
        let lowering = prepared.lowering();
        let store = prepared.source().clone();
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
                loaded: std::cell::OnceCell::new(),
                execution_identity: Default::default(),
                ingress,
                _store: store.clone(),
                world: parallel_manifest.as_ref().and(world.clone()),
                backend: Rc::clone(&backend),
                poisoned: Rc::new(Cell::new(false)),
            }),
        });
        let execution = moshi::visit_selected_moshi_realtime_architecture::<
            MlxNeuralBackend,
            MlxKeyValueState,
            _,
        >(
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
                loading: loading.clone(),
                prepared_manager,
            },
        )
        .map_err(|error| match error {
            moshi::MoshiRealtimeDispatchError::Mechanism(cause) => cause,
            other => Error::ArchitectureModel(other.to_string()),
        })?;
        Ok(execution_descriptor.bind(execution))
    })
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

use eredu_nn::workspace::WorkspaceMetadataAllocation;
