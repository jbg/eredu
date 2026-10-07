//! Cold selection followed by native binding of a retained realtime artifact.

use super::RealtimeLoadOptions;
use crate::runtime::voicechat::{
    start_mlx_voicechat, MlxVoiceChatLoadOptions, MlxVoiceChatSession, PreparedVoiceChatSession,
    VoiceChatLoadingError, VoiceChatStartError,
};
use eredu_architectures::{
    moshi::{MoshiPreparationPlan, MoshiRealtimeExecution},
    realtime::{PreparedRealtimeArtifact, RealtimeArtifactVisitor},
    voicechat::{
        artifact::RetainedVoiceChatArtifacts,
        component_residency::{
            VoiceChatResidencyError, VoiceChatResidencyPolicy, VoiceChatResidencyReport,
        },
    },
};
use eredu_backend_mlx::{
    native::{
        state_capabilities, ExecutionContext, MlxPreparedRealtimeExecution, MlxRealtimeExecution,
        MlxRealtimeExecutionContext,
    },
    MlxLoadRequest,
};
use eredu_core::{checkpoint::TensorDtype, BackendFailure, SessionCapabilities};
use eredu_runtime::{
    select_state_realization, CacheResidencyPolicy, LayerWeightResidency,
    ReplicatedTextSelectionError, ReplicatedTextSelectionRequest, ReplicatedTextStateAccess,
    SelectedStateRealization, StateLayout, StateRealizationRequirements,
};

/// Portable failure during cold selection or native binding.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeLoadError {
    /// The requested host protocol differs from the discovered architecture.
    #[error("realtime artifact requires {required} input, but loading requested {requested}")]
    Protocol {
        /// Protocol declared by the discovered architecture.
        required: &'static str,
        /// Protocol selected by the application.
        requested: &'static str,
    },
    /// Speech host preparation failed before construction.
    #[error(transparent)]
    SpeechPreparation(#[from] VoiceChatLoadingError),
    /// Exact parameter pools cannot admit the selected graph.
    #[error(transparent)]
    Residency(#[from] VoiceChatResidencyError),
    /// The architecture state requirements are invalid.
    #[error(transparent)]
    StateContract(#[from] eredu_runtime::ReplicatedTextContractError),
    /// Selected cache mechanisms cannot implement the declared state.
    #[error(transparent)]
    State(#[from] ReplicatedTextSelectionError),
    /// Speech session binding failed.
    #[error(transparent)]
    SpeechStart(#[from] VoiceChatStartError),
    /// Native failure with its original error retained as source.
    #[error("realtime backend failed: {0}")]
    Backend(#[source] BackendFailure),
}

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> RealtimeLoadError {
    RealtimeLoadError::Backend(BackendFailure::from_error(error))
}

struct SpeechSelection<'a> {
    prepared: PreparedVoiceChatSession<'a>,
    parameters: VoiceChatResidencyPolicy,
    report: VoiceChatResidencyReport,
    backbone: SelectedStateRealization,
    tts: SelectedStateRealization,
    max_branch_bytes: u64,
    max_total_copy_bytes: u64,
    seed: u64,
}
enum Selection<'a> {
    Categorical(MlxPreparedRealtimeExecution),
    Speech(Box<SpeechSelection<'a>>),
}

/// Complete cold selection. Loading accepts no replacement policy or source.
pub struct PreparedMlxRealtime<'a> {
    selection: Selection<'a>,
}

/// Bound execution retaining the architecture's declared input protocol.
pub enum LoadedMlxRealtime {
    /// Categorical frames use the shared realtime scheduler and model state.
    Categorical {
        /// Native stream mechanisms retained for submissions and state creation.
        context: MlxRealtimeExecutionContext,
        /// Architecture-owned selected categorical execution.
        model: MoshiRealtimeExecution<MlxRealtimeExecution>,
    },
    /// PCM speech uses the shared controlled/iterator conversation session.
    Speech(MlxVoiceChatSession),
}

/// Selects from a discovered artifact without creating a native device or tensor.
pub fn prepare_mlx_realtime<'a>(
    artifact: &'a PreparedRealtimeArtifact,
    options: &RealtimeLoadOptions,
) -> Result<PreparedMlxRealtime<'a>, RealtimeLoadError> {
    artifact.visit(Select { options })
}

struct Select<'a> {
    options: &'a RealtimeLoadOptions,
}
impl<'a, 'o> RealtimeArtifactVisitor<'a> for Select<'o> {
    type Output = PreparedMlxRealtime<'a>;
    type Error = RealtimeLoadError;
    fn moshi(self, artifact: &'a MoshiPreparationPlan) -> Result<Self::Output, Self::Error> {
        let RealtimeLoadOptions::Categorical(request) = self.options else {
            return Err(RealtimeLoadError::Protocol {
                required: "categorical",
                requested: "PCM",
            });
        };
        let selected = MlxRealtimeExecutionContext::select_realtime_execution(
            artifact.clone(),
            &MlxLoadRequest::from_normalized(request.clone()),
            false,
        )
        .map_err(backend)?;
        Ok(PreparedMlxRealtime {
            selection: Selection::Categorical(selected),
        })
    }
    fn voicechat(
        self,
        artifacts: &'a RetainedVoiceChatArtifacts,
    ) -> Result<Self::Output, Self::Error> {
        let RealtimeLoadOptions::Speech(options) = self.options else {
            return Err(RealtimeLoadError::Protocol {
                required: "PCM",
                requested: "categorical",
            });
        };
        let prepared = PreparedVoiceChatSession::from_retained(artifacts, &options.conversation)?;
        let construction = prepared.construction();
        let geometry = construction.neural_state_geometry();
        let backbone = select_state(
            geometry.backbone_layout(),
            ReplicatedTextStateAccess::AttentionWithFixed,
            &options.state,
        )?;
        let tts = select_state(
            geometry.tts_layout(),
            ReplicatedTextStateAccess::KeyValue,
            &options.state,
        )?;
        let report = options.parameters.validate(
            construction.backbone_parameters(),
            construction.speech_providers(),
        )?;
        Ok(PreparedMlxRealtime {
            selection: Selection::Speech(Box::new(SpeechSelection {
                prepared,
                parameters: options.parameters.clone(),
                report,
                backbone,
                tts,
                max_branch_bytes: options.max_branch_bytes,
                max_total_copy_bytes: options.max_total_copy_bytes,
                seed: options.seed,
            })),
        })
    }
}

fn select_state(
    layout: &StateLayout,
    access: ReplicatedTextStateAccess,
    policy: &CacheResidencyPolicy,
) -> Result<SelectedStateRealization, RealtimeLoadError> {
    let requirements = StateRealizationRequirements::new(
        layout.clone(),
        access,
        Some(TensorDtype::F32),
        Vec::new(),
    )?;
    let capabilities = state_capabilities(&requirements, policy);
    let request =
        ReplicatedTextSelectionRequest::new(LayerWeightResidency::FullyResident, policy.clone())
            .with_session(SessionCapabilities::new(true, true, true))
            .with_exact_completion(true);
    Ok(select_state_realization(
        &requirements,
        &request,
        &capabilities,
    )?)
}

impl PreparedMlxRealtime<'_> {
    /// Exact aggregate speech parameter admission, when the protocol uses PCM.
    pub fn speech_residency_report(&self) -> Option<VoiceChatResidencyReport> {
        match &self.selection {
            Selection::Speech(s) => Some(s.report),
            Selection::Categorical(_) => None,
        }
    }
    /// Materializes the retained selection through the existing submission owners.
    /// Sources and host policy cannot change between selection and this binding.
    pub fn load(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<LoadedMlxRealtime, RealtimeLoadError> {
        match self.selection {
            Selection::Categorical(selected) => {
                let context =
                    MlxRealtimeExecutionContext::new(execution.stream(), weights.stream());
                let model = context
                    .materialize_realtime_execution(selected)
                    .map_err(backend)?;
                Ok(LoadedMlxRealtime::Categorical { context, model })
            }
            Selection::Speech(s) => Ok(LoadedMlxRealtime::Speech(start_mlx_voicechat(
                s.prepared,
                &s.parameters,
                MlxVoiceChatLoadOptions {
                    backbone: &s.backbone,
                    tts: &s.tts,
                    max_branch_bytes: s.max_branch_bytes,
                    max_total_copy_bytes: s.max_total_copy_bytes,
                    seed: s.seed,
                },
                weights,
                execution,
            )?)),
        }
    }
}
