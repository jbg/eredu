//! MLX realization of architecture-prepared, independently leased TTS MoG.

use eredu_architectures::voicechat::{
    tts_mog::{MogDimensions, MogPrediction, MogSample, TtsError},
    tts_mog_layered::{MogExecution, MogInput, MogKernel, MogUnit, ParallelMogExecution},
    tts_mog_preparation::{validate_parallel_communication, PreparedMogBindings},
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ComponentAdapter, ComponentDefinition, DenseDiskStreamReport, LayerWeightResidency,
    LayerwiseModelMetadata, LayerwiseRuntime, LocalModelLayout, ResidencyReport,
    StatelessComponentState,
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

type Unit = MogUnit<MlxNeuralBackend>;
type Runtime<P> = LayerwiseRuntime<
    ComponentAdapter<ComponentDefinition, MogKernel<MlxNeuralBackend>>,
    MlxNeuralBackend,
    StatelessComponentState,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native EAR-TTS MoG over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain architecture-owned.
pub struct MlxMog {
    execution: Execution,
    parallel: Option<Group>,
}

impl MlxMog {
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

impl MogExecution<MlxNeuralBackend> for MlxMog {
    fn dimensions(&self) -> MogDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }
    fn predict(
        &mut self,
        hidden: &MlxTensor,
        guidance: f32,
        context: &Stream,
    ) -> Result<MogPrediction<MlxTensor>, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    MogInput::Predict { hidden, guidance },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(TtsError::from)
                    },
                )
                .and_then(|(_, forward)| forward.into_prediction()),
            (Execution::Bounded(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    MogInput::Predict { hidden, guidance },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(TtsError::from)
                    },
                )
                .and_then(|(_, forward)| forward.into_prediction()),
            (Execution::Resident(model), None) => model.predict(hidden, guidance, context),
            (Execution::Bounded(model), None) => model.predict(hidden, guidance, context),
        }
    }
    fn selected(
        &mut self,
        prediction: &MogPrediction<MlxTensor>,
        mixture: &MlxTensor,
        gaussian: &MlxTensor,
        noise_scale: f32,
        context: &Stream,
    ) -> Result<MogSample<MlxTensor>, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    MogInput::Selected {
                        prediction,
                        mixture,
                        gaussian,
                        noise_scale,
                    },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(TtsError::from)
                    },
                )
                .and_then(|(_, forward)| forward.into_sample()),
            (Execution::Bounded(model), Some(parallel)) => model
                .invoke_parallel_guarded(
                    MogInput::Selected {
                        prediction,
                        mixture,
                        gaussian,
                        noise_scale,
                    },
                    parallel,
                    context,
                    &mut MlxLayeredAgreement::new(parallel),
                    |descriptor, success, context| {
                        parallel
                            .agree_local_phase(descriptor, success, context)
                            .map_err(TtsError::from)
                    },
                )
                .and_then(|(_, forward)| forward.into_sample()),
            (Execution::Resident(model), None) => {
                model.selected(prediction, mixture, gaussian, noise_scale, context)
            }
            (Execution::Bounded(model), None) => {
                model.selected(prediction, mixture, gaussian, noise_scale, context)
            }
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The architecture supplies geometry, unit order, and all storage conversions.
/// This component constructor does not select full VoiceChat capabilities or
/// allocate live conversation, RNNT, or codec state.
pub fn materialize_voicechat_mog(
    prepared: PreparedMogBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxMog, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds the architecture-selected feed-forward and low-rank banks to native TP.
/// Mixture decisions and Gaussian draws are supplied explicitly by the caller.
pub fn materialize_voicechat_mog_parallel(
    prepared: PreparedMogBindings,
    coordinator: &MlxReplicatedCoordinator,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxMog, LayerwiseModelMetadata), BackendFailure> {
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
    prepared: PreparedMogBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxMog, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(TtsError::Configuration(
            "MoG parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let precision = prepared.precision();
            let (dimensions, definition, local_layout, layout, source, units) =
                prepared.into_parts();
            let kernel = match local_layout.as_ref() {
                Some(local) => MogKernel::<MlxNeuralBackend>::new_parallel(dimensions, local),
                None => MogKernel::<MlxNeuralBackend>::new(dimensions),
            }
            .and_then(|kernel| kernel.with_precision(precision))
            .map_err(BackendFailure::from_error)?;
            let mut architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
                .map_err(|error| BackendFailure::from_error(TtsError::from(error)))?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<eredu_runtime::StatelessComponentState>,
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
                        std::marker::PhantomData::<eredu_runtime::StatelessComponentState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRuntime::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRuntime::new(architecture, policy))
            };
            Ok((
                MlxMog {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
