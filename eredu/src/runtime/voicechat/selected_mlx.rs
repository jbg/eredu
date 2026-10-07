//! Selected native mechanisms for the portable prepared-session driver.

use std::sync::Arc;

use eredu_architectures::voicechat::{
    component_residency::{VoiceChatResidencyPolicy, VoiceChatResidencyReport},
    components::VoiceChatConstructionError,
    perception_rnnt::PerceptionRnntLimits,
    preparation::GenerationPreparationError,
    state::NeuralStateGeometryError,
    tts_prompt::PreparedTtsPrompt,
};
use eredu_backend_mlx::{
    backend::runtime::cache::state::JointCopyLimits,
    native::{
        initialize_voicechat_neural_state, materialize_voicechat_components, ExecutionContext,
        MlxVoiceChatBranch, MlxVoiceChatExecution, MlxVoiceChatGeneration, RandomState,
    },
    MlxTensor,
};
use eredu_core::{BackendFailure, RealtimeDecisionDiagnostics};
use eredu_nn::TensorElementType;
use eredu_runtime::SelectedStateRealization;

use super::{
    submit_mlx_voicechat, BoundVoiceChatFork, BoundVoiceChatSession, PreparedVoiceChatPrompt,
    PreparedVoiceChatSession, VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatExecutor,
    VoiceChatPublicationBranch, VoiceChatSessionError, VoiceChatSpeechOptions,
    VoiceChatSubmittedWork, VoiceChatWork,
};

/// Selected cache storage and finite copy budgets for native startup.
/// Both layouts are checked against the prepared architecture before binding.
pub(crate) struct MlxVoiceChatLoadOptions<'a> {
    /// Selected Nemotron hybrid cache realization for one conversation lane.
    pub backbone: &'a SelectedStateRealization,
    /// Selected TTS attention realization, including both guidance rows.
    pub tts: &'a SelectedStateRealization,
    /// Maximum retained storage and temporary bytes for one neural branch.
    pub max_branch_bytes: u64,
    /// Cumulative copy allowance, shared by isolated descendants.
    pub max_total_copy_bytes: u64,
    /// Explicit initial RNG seed; replay state belongs to the session.
    pub seed: u64,
}

/// Failure while adopting a cold request into the ordinary native session.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatStartError {
    /// Selected cache layouts disagree with the retained architecture.
    #[error(transparent)]
    Geometry(#[from] NeuralStateGeometryError),
    /// Exact parameter admission or component construction failed.
    #[error(transparent)]
    Construction(#[from] VoiceChatConstructionError),
    /// Shared generation-state initialization failed.
    #[error(transparent)]
    Generation(#[from] GenerationPreparationError),
    /// Session registration or bounded queue admission failed.
    #[error(transparent)]
    Session(#[from] VoiceChatSessionError),
    /// Native preparation failure, preserving its original source.
    #[error("VoiceChat native startup failed: {0}")]
    Backend(#[source] BackendFailure),
}

struct Inputs {
    prompt: PreparedVoiceChatPrompt,
    speaker: PreparedTtsPrompt<MlxTensor>,
    silence: MlxTensor,
}

/// Native execution resources and immutable startup inputs for a bound session.
/// Neural caches, RNG, conversation policy and delivered events live in the
/// ordinary session transaction and are never copied by this executor.
pub struct MlxVoiceChatExecutor {
    execution: MlxVoiceChatExecution,
    inputs: Arc<Inputs>,
    input_limits: PerceptionRnntLimits,
    max_character_cells: usize,
    speech: VoiceChatSpeechOptions,
}

impl MlxVoiceChatExecutor {
    /// Shares immutable startup inputs and the same admitted parameter pools.
    /// Use with `fork_with_executor`; pending sibling work causes retryable
    /// admission backpressure rather than creating another parameter allocation.
    pub fn share(&self) -> Self {
        Self {
            execution: self.execution.share(),
            inputs: Arc::clone(&self.inputs),
            input_limits: self.input_limits,
            max_character_cells: self.max_character_cells,
            speech: self.speech,
        }
    }

    /// Complete admitted parameter pools; cache and activation storage is separate.
    pub fn residency_report(&self) -> VoiceChatResidencyReport {
        self.execution.residency_report()
    }
}

impl VoiceChatExecutor<MlxVoiceChatGeneration, RealtimeDecisionDiagnostics>
    for MlxVoiceChatExecutor
{
    fn is_busy(&self) -> bool {
        self.execution.is_busy()
    }

    fn submit(
        &mut self,
        work: &VoiceChatWork<RealtimeDecisionDiagnostics>,
        branch: &mut VoiceChatPublicationBranch<MlxVoiceChatBranch, RealtimeDecisionDiagnostics>,
    ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
        let inputs = VoiceChatExecutionInputs {
            prompt: Some(&self.inputs.prompt),
            speaker_codes: self.inputs.speaker.codes(),
            speaker_layout: self.inputs.speaker.layout(),
            speaker_conditioning: None,
            silence: Some(&self.inputs.silence),
            speech: self.speech,
        };
        submit_mlx_voicechat(
            work,
            branch,
            &mut self.execution,
            &inputs,
            self.input_limits,
            self.max_character_cells,
        )
    }
}

/// Selected MLX session using the portable controlled and iterator API.
pub type MlxVoiceChatSession = BoundVoiceChatSession<
    MlxVoiceChatGeneration,
    RealtimeDecisionDiagnostics,
    MlxVoiceChatExecutor,
>;

impl MlxVoiceChatSession {
    /// Forks mutable session state while sharing the same prepared execution.
    /// Sibling invocations use ordinary completion backpressure; weights and
    /// startup tensors are retained once, and cache copies keep their budgets.
    pub fn fork_shared(
        &mut self,
        request: eredu_core::scheduler::RequestId,
    ) -> Result<
        BoundVoiceChatFork<
            MlxVoiceChatGeneration,
            RealtimeDecisionDiagnostics,
            MlxVoiceChatExecutor,
        >,
        VoiceChatSessionError,
    > {
        let executor = self.executor().share();
        self.fork_with_executor(request, executor)
    }
}

/// Binds an admitted request without reopening artifacts or advancing live conversation.
/// Fixed-Aria and silence preparation establish native completion before their
/// tensors are exposed. Prompt prefill is queued on the ordinary session and
/// executes only when the application advances it with an explicit timestamp.
pub(crate) fn start_mlx_voicechat(
    prepared: PreparedVoiceChatSession<'_>,
    residency: &VoiceChatResidencyPolicy,
    options: MlxVoiceChatLoadOptions<'_>,
    weights: &ExecutionContext,
    execution: &ExecutionContext,
) -> Result<MlxVoiceChatSession, VoiceChatStartError> {
    let parts = prepared.into_parts();
    let geometry = parts.construction.neural_state_geometry().clone();
    geometry.validate_layouts(options.backbone.layout(), options.tts.layout())?;
    let (components, input) = materialize_voicechat_components(
        parts.construction,
        parts.frontend,
        residency,
        weights.stream(),
        execution.stream(),
    )?;
    let mut native = MlxVoiceChatExecution::new(components, execution.stream());
    let map_native = |error| VoiceChatStartError::Backend(BackendFailure::from_error(error));
    let speaker = native
        .prepare_prompt(1, parts.max_prompt_positions)
        .map_err(map_native)?;
    // Host feature extraction produces F32 tensors after the admitted PCM rounding.
    let silence = native
        .prepare_silence(TensorElementType::F32)
        .map_err(map_native)?;
    let neural = initialize_voicechat_neural_state(
        &geometry,
        options.backbone,
        options.tts,
        JointCopyLimits {
            max_branch_bytes: options.max_branch_bytes,
            max_total_copy_bytes: options.max_total_copy_bytes,
        },
        execution.stream(),
    )
    .map_err(|error| VoiceChatStartError::Backend(BackendFailure::from_error(error)))?;
    let random = RandomState::with_seed(options.seed)
        .map_err(|error| VoiceChatStartError::Backend(BackendFailure::from_error(error)))?;
    let state = parts.generation.initialize(input, neural, random)?;
    let executor = MlxVoiceChatExecutor {
        execution: native,
        inputs: Arc::new(Inputs {
            prompt: parts.prompt,
            speaker,
            silence,
        }),
        input_limits: parts.input_limits,
        max_character_cells: parts.max_character_cells,
        speech: parts.speech,
    };
    Ok(BoundVoiceChatSession::start(
        state,
        parts.conversation,
        parts.input_config,
        parts.publication_limits,
        parts.session_config,
        executor,
    )?)
}
