//! MLX realization of architecture-prepared, independently leased TTS RVQ.

use eredu_architectures::voicechat::{
    tts_mog::TtsError,
    tts_rvq::{RefinementState, RvqDimensions},
    tts_rvq_layered::{LayerwiseRvq, RvqExecution, RvqForward, RvqInput, RvqKernel, RvqUnit},
    tts_rvq_preparation::{validate_parallel_communication, PreparedRvqBindings},
};
use eredu_core::{BackendFailure, ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    DenseDiskStreamReport, LayerWeightResidency, LayerwiseModelMetadata, LocalModelLayout,
    ResidencyReport, StatelessComponentState,
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

type Unit = RvqUnit<MlxNeuralBackend>;
enum Execution {
    Resident(LayerwiseRvq<MlxNeuralBackend, MlxResidentPolicy<Unit>>),
    Bounded(LayerwiseRvq<MlxNeuralBackend, MlxLayerwisePolicy<Unit>>),
}

/// Native EAR-TTS RVQ over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain architecture-owned.
pub struct MlxRvq {
    execution: Execution,
    parallel: Option<Group>,
}

impl MlxRvq {
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

impl MlxRvq {
    fn invoke_guarded(
        &mut self,
        input: RvqInput<'_, MlxTensor>,
        context: &Stream,
    ) -> Result<(MlxTensor, RvqForward<MlxTensor>), TtsError> {
        let parallel = self
            .parallel
            .as_ref()
            .expect("selected tensor-parallel provider");
        let mut hook = MlxLayeredAgreement::new(parallel);
        let agree = |descriptor, success, context: &Stream| {
            parallel
                .agree_local_phase(descriptor, success, context)
                .map_err(TtsError::from)
        };
        match &mut self.execution {
            Execution::Resident(model) => {
                model.invoke_parallel_guarded(input, parallel, context, &mut hook, agree)
            }
            Execution::Bounded(model) => {
                model.invoke_parallel_guarded(input, parallel, context, &mut hook, agree)
            }
        }
    }
}
impl RvqExecution<MlxNeuralBackend> for MlxRvq {
    fn dimensions(&self) -> RvqDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }
    fn depthsum(&mut self, codes: &MlxTensor, context: &Stream) -> Result<MlxTensor, TtsError> {
        if self.parallel.is_some() {
            return self
                .invoke_guarded(RvqInput::Sum(codes), context)
                .map(|(output, _)| output);
        }
        match &mut self.execution {
            Execution::Resident(model) => model.depthsum(codes, context),
            Execution::Bounded(model) => model.depthsum(codes, context),
        }
    }
    fn embed_codes(&mut self, codes: &MlxTensor, context: &Stream) -> Result<MlxTensor, TtsError> {
        if self.parallel.is_some() {
            return self
                .invoke_guarded(RvqInput::Embed(codes), context)
                .map(|(output, _)| output);
        }
        match &mut self.execution {
            Execution::Resident(model) => model.embed_codes(codes, context),
            Execution::Bounded(model) => model.embed_codes(codes, context),
        }
    }
    fn advance(
        &mut self,
        state: &RefinementState<MlxTensor>,
        latent: &MlxTensor,
        context: &Stream,
    ) -> Result<RefinementState<MlxTensor>, TtsError> {
        if self.parallel.is_some() {
            return self
                .invoke_guarded(RvqInput::Advance { state, latent }, context)?
                .1
                .into_state();
        }
        match &mut self.execution {
            Execution::Resident(model) => model.advance(state, latent, context),
            Execution::Bounded(model) => model.advance(state, latent, context),
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The architecture supplies geometry, unit order, and all storage conversions.
/// This component constructor does not select full VoiceChat capabilities or
/// allocate live conversation, RNNT, or codec state.
pub fn materialize_voicechat_rvq(
    prepared: PreparedRvqBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRvq, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds architecture-selected latent shards to native tensor parallel execution.
/// Candidate codes and latent feedback remain replicated across the selected group.
pub fn materialize_voicechat_rvq_parallel(
    prepared: PreparedRvqBindings,
    coordinator: &MlxReplicatedCoordinator,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRvq, LayerwiseModelMetadata), BackendFailure> {
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
    prepared: PreparedRvqBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxRvq, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(TtsError::Configuration(
            "RVQ parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (dimensions, definition, local_layout, layout, source, units) =
                prepared.into_parts();
            let kernel = match local_layout.as_ref() {
                Some(local) => RvqKernel::<MlxNeuralBackend>::new_parallel(dimensions, local),
                None => RvqKernel::<MlxNeuralBackend>::new(dimensions),
            }
            .map_err(BackendFailure::from_error)?;
            let mut architecture = eredu_runtime::ComponentAdapter::new(definition, kernel)
                .map_err(|error| BackendFailure::from_error(TtsError::from(error)))?;
            let (policy, metadata) = prepare_layerwise_policy_from_bindings(
                source,
                &mut architecture,
                (),
                std::marker::PhantomData::<StatelessComponentState>,
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
                        std::marker::PhantomData::<StatelessComponentState>,
                    )
                    .map_err(BackendFailure::from_error)?;
                Execution::Resident(LayerwiseRvq::new(architecture, policy))
            } else {
                Execution::Bounded(LayerwiseRvq::new(architecture, policy))
            };
            Ok((
                MlxRvq {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
