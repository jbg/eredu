//! MLX realization of architecture-prepared, independently leased perception.

use eredu_architectures::voicechat::{
    perception::{PerceptionDimensions, PerceptionError, PerceptionOutput, PerceptionState},
    perception_layered::{
        ParallelPerceptionExecution, PerceptionExecution, PerceptionKernel, PerceptionRuntimeState,
        PerceptionUnit,
    },
    perception_preparation::PreparedPerceptionBindings,
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ArchitectureParameters, ComponentAdapter, ComponentDefinition, DenseDiskStreamReport,
    LayerWeightResidency, LayerwiseModelMetadata, LayerwiseRuntime, LocalModelLayout,
    ResidencyReport, StateLayout,
};
use safemlx::{Array, Stream};

use crate::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::distributed::Group,
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = PerceptionUnit<MlxNeuralBackend>;
type Runtime<P> = LayerwiseRuntime<
    ComponentAdapter<ComponentDefinition, PerceptionKernel<MlxNeuralBackend>>,
    MlxNeuralBackend,
    PerceptionRuntimeState<MlxTensor>,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native perception over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain architecture-owned.
pub struct MlxPerception {
    execution: Execution,
    parallel: Option<Group>,
    state_layout: StateLayout,
}

impl MlxPerception {
    /// Architecture-declared cache geometry; Conformer history is replicated across tensor ranks.
    pub fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }

    /// Current physical residency and retained-source accounting.
    pub fn residency_report(&self) -> Result<ResidencyReport, BackendFailure> {
        match &self.execution {
            Execution::Resident(model) => model.policy().residency_report(),
            Execution::Bounded(model) => model.policy().residency_report(),
        }
        .map_err(BackendFailure::from_error)
    }

    /// Disk-stream telemetry, present only when that policy was selected.
    pub fn dense_stream_report(&self) -> Result<Option<DenseDiskStreamReport>, BackendFailure> {
        match &self.execution {
            Execution::Resident(_) => Ok(None),
            Execution::Bounded(model) => model
                .policy()
                .dense_stream_report()
                .map_err(BackendFailure::from_error),
        }
    }

    /// Number of pinned parameter leases outside the independently loaded units.
    pub fn static_lease_count(&self) -> usize {
        match &self.execution {
            Execution::Resident(model) => model.policy().static_lease_count(),
            Execution::Bounded(model) => model.policy().static_lease_count(),
        }
    }
}

impl PerceptionExecution<MlxNeuralBackend> for MlxPerception {
    fn dimensions(&self) -> PerceptionDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }

    fn forward(
        &mut self,
        mel: &MlxTensor,
        drop_preencoded: usize,
        previous: &PerceptionState<MlxTensor>,
        context: &Stream,
    ) -> Result<PerceptionOutput<MlxTensor>, PerceptionError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => {
                ParallelPerceptionExecution::forward_parallel(
                    model,
                    mel,
                    drop_preencoded,
                    previous,
                    parallel,
                    context,
                )
            }
            (Execution::Bounded(model), Some(parallel)) => {
                ParallelPerceptionExecution::forward_parallel(
                    model,
                    mel,
                    drop_preencoded,
                    previous,
                    parallel,
                    context,
                )
            }
            (Execution::Resident(model), None) => {
                PerceptionExecution::forward(model, mel, drop_preencoded, previous, context)
            }
            (Execution::Bounded(model), None) => {
                PerceptionExecution::forward(model, mel, drop_preencoded, previous, context)
            }
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The architecture supplies geometry, unit order, and all storage conversions.
/// This component constructor does not select full VoiceChat capabilities or
/// allocate live conversation, RNNT, TTS, or codec state.
pub fn materialize_voicechat_perception(
    prepared: PreparedPerceptionBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPerception, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds a selected perception tensor partition to its native communicator.
/// Architecture-owned recipes partition attention and feed-forward projections;
/// replicated parameters and full-width caches remain charged on each rank.
/// This component binding does not select a distributed VoiceChat session.
pub fn materialize_voicechat_perception_parallel(
    prepared: PreparedPerceptionBindings,
    parallel: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPerception, LayerwiseModelMetadata), BackendFailure> {
    let parallel = Group::uncontracted(parallel);
    let topology = ParallelTopology::new(parallel.size(), 1, 1, 1)
        .and_then(|topology| ParallelRankTopology::new(topology, parallel.rank()))
        .map_err(BackendFailure::from_error)?;
    let (prepared, layout) = prepared
        .select_parallel(topology)
        .map_err(BackendFailure::from_error)?;
    materialize(
        prepared,
        residency,
        Some((&layout, &parallel)),
        weights_stream,
        execution_stream,
    )
}

fn materialize(
    prepared: PreparedPerceptionBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPerception, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(PerceptionError::State(
            "perception parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let precision = prepared.precision();
            let (dimensions, definition, layout, source, units) = prepared.into_parts();
            let kernel = match parallel {
                Some((local, _)) => {
                    PerceptionKernel::<MlxNeuralBackend>::new_parallel(dimensions, local)
                }
                None => PerceptionKernel::<MlxNeuralBackend>::new(dimensions),
            }
            .and_then(|architecture| architecture.with_precision(precision))
            .map_err(BackendFailure::from_error)?;
            let mut architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
                .map_err(|error| BackendFailure::from_error(PerceptionError::from(error)))?;
            let state_layout = architecture
                .state_layout()
                .map_err(BackendFailure::from_error)?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<PerceptionRuntimeState<MlxTensor>>,
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
                        std::marker::PhantomData::<PerceptionRuntimeState<MlxTensor>>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRuntime::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
            };
            Ok((
                MlxPerception {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                    state_layout,
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
