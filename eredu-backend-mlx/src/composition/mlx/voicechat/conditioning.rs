//! MLX realization of architecture-prepared, independently leased TTS conditioning.

use eredu_architectures::voicechat::{
    tts_conditioning::CharacterBatch,
    tts_conditioning_layered::{
        ConditioningDimensions, ConditioningExecution, ConditioningKernel, ConditioningUnit,
        ParallelConditioningExecution,
    },
    tts_conditioning_preparation::PreparedConditioningBindings,
    tts_ingress::{ConditioningLanes, PromptLayout, SpeakerPrompt},
    tts_mog::TtsError,
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
        runtime::distributed::Group,
        runtime::execution::generic::{
            prepare_layerwise_policy_from_bindings, MlxLayerwisePolicy, MlxResidentPolicy,
        },
        submission_recovery::detached_preparation,
    },
    MlxTensor,
};

type Unit = ConditioningUnit<MlxNeuralBackend>;
type Runtime<P> = LayerwiseRuntime<
    ComponentAdapter<ComponentDefinition, ConditioningKernel<MlxNeuralBackend>>,
    MlxNeuralBackend,
    StatelessComponentState,
    P,
>;
enum Execution {
    Resident(Runtime<MlxResidentPolicy<Unit>>),
    Bounded(Runtime<MlxLayerwisePolicy<Unit>>),
}

/// Native EAR-TTS conditioning over the selected shared resident or bounded parameter policy.
/// All equations and candidate-state construction remain architecture-owned.
pub struct MlxConditioning {
    execution: Execution,
    parallel: Option<Group>,
    immutable_identity: eredu_core::resources::ResourceIdentity,
    immutable_stream_index: i32,
}

impl MlxConditioning {
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

impl ConditioningExecution<MlxNeuralBackend> for MlxConditioning {
    fn immutable_parameters_identity(
        &self,
        context: &Stream,
    ) -> Option<&eredu_core::resources::ResourceIdentity> {
        (matches!(self.execution, Execution::Resident(_))
            && self.parallel.is_none()
            && context.get_index().ok() == Some(self.immutable_stream_index))
        .then_some(&self.immutable_identity)
    }

    fn dimensions(&self) -> ConditioningDimensions {
        match &self.execution {
            Execution::Resident(model) => model.dimensions(),
            Execution::Bounded(model) => model.dimensions(),
        }
    }
    fn character_embed(
        &mut self,
        batch: &CharacterBatch,
        context: &Stream,
    ) -> Result<MlxTensor, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(group)) => {
                model.character_embed_parallel(batch, group, context)
            }
            (Execution::Bounded(model), Some(group)) => {
                model.character_embed_parallel(batch, group, context)
            }
            (Execution::Resident(model), None) => model.character_embed(batch, context),
            (Execution::Bounded(model), None) => model.character_embed(batch, context),
        }
    }
    fn character_finish(
        &mut self,
        batch: &CharacterBatch,
        hidden: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(group)) => {
                model.character_finish_parallel(batch, hidden, group, context)
            }
            (Execution::Bounded(model), Some(group)) => {
                model.character_finish_parallel(batch, hidden, group, context)
            }
            (Execution::Resident(model), None) => model.character_finish(batch, hidden, context),
            (Execution::Bounded(model), None) => model.character_finish(batch, hidden, context),
        }
    }
    fn prefill_embedded(
        &mut self,
        embedded: &MlxTensor,
        layout: &PromptLayout,
        speaker: SpeakerPrompt,
        conditioning: Option<&MlxTensor>,
        guidance: bool,
        context: &Stream,
    ) -> Result<ConditioningLanes<MlxTensor>, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(group)) => model.prefill_embedded_parallel(
                embedded,
                layout,
                speaker,
                conditioning,
                guidance,
                group,
                context,
            ),
            (Execution::Bounded(model), Some(group)) => model.prefill_embedded_parallel(
                embedded,
                layout,
                speaker,
                conditioning,
                guidance,
                group,
                context,
            ),
            (Execution::Resident(model), None) => {
                model.prefill_embedded(embedded, layout, speaker, conditioning, guidance, context)
            }
            (Execution::Bounded(model), None) => {
                model.prefill_embedded(embedded, layout, speaker, conditioning, guidance, context)
            }
        }
    }
    fn decode_embedded(
        &mut self,
        embedded: &MlxTensor,
        conditioning: Option<&MlxTensor>,
        guidance: bool,
        context: &Stream,
    ) -> Result<ConditioningLanes<MlxTensor>, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(group)) => {
                model.decode_embedded_parallel(embedded, conditioning, guidance, group, context)
            }
            (Execution::Bounded(model), Some(group)) => {
                model.decode_embedded_parallel(embedded, conditioning, guidance, group, context)
            }
            (Execution::Resident(model), None) => {
                model.decode_embedded(embedded, conditioning, guidance, context)
            }
            (Execution::Bounded(model), None) => {
                model.decode_embedded(embedded, conditioning, guidance, context)
            }
        }
    }
    fn fuse(
        &mut self,
        audio: &MlxTensor,
        text: &MlxTensor,
        context: &Stream,
    ) -> Result<MlxTensor, TtsError> {
        match (&mut self.execution, &self.parallel) {
            (Execution::Resident(model), Some(group)) => {
                model.fuse_parallel(audio, text, group, context)
            }
            (Execution::Bounded(model), Some(group)) => {
                model.fuse_parallel(audio, text, group, context)
            }
            (Execution::Resident(model), None) => model.fuse(audio, text, context),
            (Execution::Bounded(model), None) => model.fuse(audio, text, context),
        }
    }
}

/// Realizes exact retained bindings without source discovery or family mapping.
///
/// The architecture supplies geometry, unit order, and all storage conversions.
/// This component constructor does not select full VoiceChat capabilities or
/// allocate live conversation, RNNT, or codec state.
pub fn materialize_voicechat_conditioning(
    prepared: PreparedConditioningBindings,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxConditioning, LayerwiseModelMetadata), BackendFailure> {
    materialize(prepared, residency, None, weights_stream, execution_stream)
}

/// Binds architecture-selected conditioning shards to native tensor parallel execution.
/// Character embeddings and conditional/guidance lanes remain replicated across the group.
pub fn materialize_voicechat_conditioning_parallel(
    prepared: PreparedConditioningBindings,
    parallel: &safemlx::distributed::Group,
    residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxConditioning, LayerwiseModelMetadata), BackendFailure> {
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
    prepared: PreparedConditioningBindings,
    residency: LayerWeightResidency,
    parallel: Option<(&LocalModelLayout, &Group)>,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxConditioning, LayerwiseModelMetadata), BackendFailure> {
    if prepared.local_layout() != parallel.map(|(layout, _)| layout) {
        return Err(BackendFailure::from_error(TtsError::Configuration(
            "conditioning parameter placement differs from the selected execution",
        )));
    }
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let (dimensions, definition, local_layout, layout, source, units) =
                prepared.into_parts();
            let kernel = match local_layout.as_ref() {
                Some(local) => {
                    ConditioningKernel::<MlxNeuralBackend>::new_parallel(dimensions, local)
                }
                None => ConditioningKernel::<MlxNeuralBackend>::new(dimensions),
            }
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
                MlxConditioning {
                    execution,
                    parallel: parallel.map(|(_, group)| group.clone()),
                    immutable_stream_index: execution_stream
                        .get_index()
                        .map_err(BackendFailure::from_error)?,
                    immutable_identity: {
                        static NEXT: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(1);
                        eredu_core::resources::ResourceIdentity {
                            scope: "mlx.immutable.conditioning".into(),
                            key: NEXT
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                .to_string(),
                        }
                    },
                },
                metadata,
            ))
        },
        BackendFailure::from_error,
    )
}
