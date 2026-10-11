//! MLX realization of architecture-prepared, independently leased function head.

use eredu_architectures::voicechat::{
    channels::{ChannelError, FunctionHead},
    function_head_layered::{
        FunctionHeadArchitecture, FunctionHeadDimensions, FunctionHeadExecution,
        FunctionHeadRuntimeState, LayerwiseFunctionHead,
    },
    function_head_preparation::{validate_parallel_communication, PreparedFunctionHeadBindings},
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, LocalModelLayout,
    ResidencyReport,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::{
            coordinator::{MlxLayeredAgreement, MlxReplicatedCoordinator},
            Group,
        },
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = FunctionHead<MlxNeuralBackend>;
enum Execution {
    Resident(LayerwiseFunctionHead<MlxNeuralBackend, MlxResidentPolicy<Unit>>),
    Bounded(LayerwiseFunctionHead<MlxNeuralBackend, MlxLayerwisePolicy<Unit>>),
}

/// Native function vocabulary head over the selected shared resident or bounded parameter policy.
/// Projection equations remain architecture-owned.
pub struct MlxFunctionHead {
    execution: Execution,
    parallel: Option<Group>,
}

impl MlxFunctionHead {
    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.runtime().policy().residency_report(),
            Execution::Bounded(model) => model.runtime().policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Disk-stream telemetry, present only when that policy was selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .runtime()
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }

    /// Number of pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.runtime().policy().static_lease_count(),
            Execution::Bounded(model) => model.runtime().policy().static_lease_count(),
        }
    }
}

impl FunctionHeadExecution<MlxNeuralBackend> for MlxFunctionHead {
    fn dimensions(&self) -> FunctionHeadDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }
    fn project(
        &mut self,
        normalized: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, ChannelError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model.project_parallel_guarded(
                normalized,
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
                |descriptor, success, context| {
                    parallel
                        .agree_local_phase(descriptor, success, context)
                        .map_err(ChannelError::from)
                },
            ),
            (Execution::Bounded(model), Some(parallel)) => model.project_parallel_guarded(
                normalized,
                parallel,
                context,
                &mut MlxLayeredAgreement::new(parallel),
                |descriptor, success, context| {
                    parallel
                        .agree_local_phase(descriptor, success, context)
                        .map_err(ChannelError::from)
                },
            ),
            (Execution::Resident(model), None) => model.project(normalized, context),
            (Execution::Bounded(model), None) => model.project(normalized, context),
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The architecture supplies geometry, unit order, and all storage conversions.
/// This component constructor does not select full VoiceChat capabilities or
/// allocate live conversation, RNNT, or codec state.
pub fn materialize_voicechat_function_head(
    prepared: PreparedFunctionHeadBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxFunctionHead, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds the architecture-selected vocabulary shard to its native communicator.
/// The retained coordinator supplies bounded failure agreement for acquisition,
/// architecture preparation and local projection before vocabulary gathering.
/// The independently acquired head lease stays owned through these checks.
pub fn materialize_voicechat_function_head_parallel(
    prepared: PreparedFunctionHeadBindings,
    coordinator: &MlxReplicatedCoordinator,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxFunctionHead, LayerwiseModelMetadata), BackendFailure> {
    validate_parallel_communication(
        prepared.dimensions(),
        coordinator.manifest(),
        coordinator.group(),
    )
    .map_err(BackendFailure::from_error)?;
    let parallel = coordinator.parallel_group();
    let topology = ParallelTopology::new(parallel.size(), 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, parallel.rank()))
        .map_err(BackendFailure::from_error)?;
    let (prepared, layout) = prepared
        .select_parallel(topology)
        .map_err(BackendFailure::from_error)?;
    materialize(
        prepared,
        residency,
        Some((&layout, parallel)),
        weights_stream,
        execution_stream,
    )
}

fn materialize(
    prepared: PreparedFunctionHeadBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxFunctionHead, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(ChannelError::Configuration(
            "function head parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (dimensions, format, local_layout, layout, source, units) = prepared.into_parts();
            let mut architecture = match local_layout.as_ref() {
                Some(local) => FunctionHeadArchitecture::<MlxNeuralBackend>::new_parallel(
                    dimensions, format, local,
                ),
                None => FunctionHeadArchitecture::<MlxNeuralBackend>::new(dimensions, format),
            }
            .map_err(BackendFailure::from_error)?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<FunctionHeadRuntimeState>,
                residency,
                execution_stream,
                weights_stream,
                |_| false,
                layout,
                Vec::new(),
                units,
            )
            .map_err(BackendFailure::from_error)?;
            let execution = if residency.is_fully_resident() {
                let policy = policy
                    .into_resident(
                        &architecture,
                        execution_stream,
                        std::marker::PhantomData::<FunctionHeadRuntimeState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseFunctionHead::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseFunctionHead::new(architecture, policy))
            };
            Ok((
                MlxFunctionHead {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
