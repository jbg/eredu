//! Portable construction of an admitted VoiceChat session.

use eredu_architectures::voicechat::{
    component_residency::VoiceChatResidencyPolicy,
    components::{
        PreparedVoiceChatConstruction, VoiceChatConstructionError, VoiceChatStartupComponents,
    },
    frontend::FrontendOptions,
    input::InputState,
    perception_rnnt::PerceptionRnntLimits,
    preparation::{Generation, GenerationPreparationError},
    state::NeuralStateGeometryError,
    tts_prompt::PreparedTtsPrompt,
};
use eredu_core::realtime_mechanisms::{RealtimeQueue, SpeechRealtimeNative};
use eredu_core::{
    sampling::host_random::HostRandomState, BackendFailure, RealtimeDecisionDiagnostics,
};
use eredu_nn::{RetainedTensorRoots, Tensor, TensorElementType};
use eredu_runtime::{
    JointRealtimeState, JointStateError, RealtimeCopyLimits, SelectedStateRealization,
    SharedRealtimeExecution, SpeechRealtimeMechanisms, SpeechRealtimeState,
};

use super::{
    BoundVoiceChatSession, PreparedVoiceChatPrompt, PreparedVoiceChatSession, VoiceChatExecutor,
    VoiceChatSessionError, VoiceChatSpeechOptions,
};

/// Selected cache storage and finite copy budgets for native startup.
/// Both layouts are checked against the prepared architecture before binding.
pub struct VoiceChatLoadSelection<'a> {
    /// Selected Nemotron hybrid cache realization for one conversation lane.
    pub backbone: &'a SelectedStateRealization,
    /// Selected TTS attention realization, including both guidance rows.
    pub tts: &'a SelectedStateRealization,
    /// Finite branch and descendant copy budgets.
    pub copy_limits: RealtimeCopyLimits,
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
    /// Coupled state transaction could not adopt the selected caches.
    #[error(transparent)]
    State(#[from] JointStateError),
    /// Native preparation failure, preserving its original source.
    #[error("VoiceChat native startup failed: {0}")]
    Backend(#[source] BackendFailure),
}

/// Immutable startup inputs supplied to the session executor.
pub struct VoiceChatStartupInputs<T: Tensor> {
    /// Host-prepared conversation prompt.
    pub prompt: PreparedVoiceChatPrompt,
    /// Completed fixed-Aria reference prompt.
    pub speaker: PreparedTtsPrompt<T>,
    /// Completed tool-silence conditioning.
    pub silence: T,
    /// Input execution limits.
    pub input_limits: PerceptionRnntLimits,
    /// Maximum character cells per invocation.
    pub max_character_cells: usize,
    /// Selected output waveform policy.
    pub speech: VoiceChatSpeechOptions,
}

struct Silence<T>(T);
impl<T> RetainedTensorRoots<T> for Silence<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        visitor(&self.0);
    }
}

type SessionGeneration<T, M> = Generation<
    T,
    SpeechRealtimeState<M>,
    HostRandomState,
    <<M as SpeechRealtimeNative>::Queue as RealtimeQueue>::Completion,
>;

impl PreparedVoiceChatSession<'_> {
    /// Binds selected resources and queues portable prefill on the ordinary session.
    pub fn start<T, M, C, Q, E>(
        self,
        residency: &VoiceChatResidencyPolicy,
        selection: VoiceChatLoadSelection<'_>,
        mechanisms: &M,
        bind: impl FnOnce(
            PreparedVoiceChatConstruction<'_>,
            FrontendOptions,
            &VoiceChatResidencyPolicy,
        ) -> Result<
            (SharedRealtimeExecution<C, Q>, InputState<T>),
            VoiceChatConstructionError,
        >,
        executor: impl FnOnce(SharedRealtimeExecution<C, Q>, VoiceChatStartupInputs<T>) -> E,
    ) -> Result<
        BoundVoiceChatSession<SessionGeneration<T, M>, RealtimeDecisionDiagnostics, E>,
        VoiceChatStartError,
    >
    where
        T: Tensor,
        M: SpeechRealtimeMechanisms,
        C: VoiceChatStartupComponents<T> + 'static,
        Q: RealtimeQueue<Tensor = T>,
        E: VoiceChatExecutor<SessionGeneration<T, M>, RealtimeDecisionDiagnostics>,
    {
        let parts = self.into_parts();
        parts
            .construction
            .neural_state_geometry()
            .validate_layouts(selection.backbone.layout(), selection.tts.layout())?;
        let backbone_plan = selection
            .backbone
            .allocation()
            .map_err(VoiceChatStartError::Backend)?;
        let tts_plan = selection
            .tts
            .allocation()
            .map_err(VoiceChatStartError::Backend)?;
        let (execution, input) = bind(parts.construction, parts.frontend, residency)?;
        let speaker = execution
            .prepare_startup(|components, context| {
                components
                    .prepare_prompt(1, parts.max_prompt_positions, context)
                    .map_err(BackendFailure::from_error)
            })
            .map_err(VoiceChatStartError::Backend)?;
        // The admitted PCM frontend produces F32 features.
        let silence = execution
            .prepare_startup(|components, context| {
                components
                    .prepare_silence(TensorElementType::F32, context)
                    .map(Silence)
                    .map_err(BackendFailure::from_error)
            })
            .map_err(VoiceChatStartError::Backend)?
            .0;
        let backbone_cache = selection
            .backbone
            .create_live_cache::<M::CacheBackend>()
            .map_err(VoiceChatStartError::Backend)?;
        let backbone = eredu_runtime::state::HybridState::allocate_layers(
            &backbone_plan,
            backbone_cache,
            None,
            0,
        )
        .map_err(BackendFailure::from_error)
        .map_err(VoiceChatStartError::Backend)?;
        let tts_cache = selection
            .tts
            .create_live_cache::<M::CacheBackend>()
            .map_err(VoiceChatStartError::Backend)?;
        let tts =
            eredu_runtime::state::KeyValueState::allocate_layers(&tts_plan, tts_cache, None, 0)
                .map_err(BackendFailure::from_error)
                .map_err(VoiceChatStartError::Backend)?;
        let neural = JointRealtimeState::new(
            backbone,
            tts,
            mechanisms.state_copy_queue(),
            selection.copy_limits,
        )?;
        let random = HostRandomState::with_seed(selection.seed);
        let state = parts.generation.initialize(input, neural, random)?;
        Ok(BoundVoiceChatSession::start(
            state,
            parts.conversation,
            parts.input_config,
            parts.publication_limits,
            parts.session_config,
            executor(
                execution,
                VoiceChatStartupInputs {
                    prompt: parts.prompt,
                    speaker,
                    silence,
                    input_limits: parts.input_limits,
                    max_character_cells: parts.max_character_cells,
                    speech: parts.speech,
                },
            ),
        )?)
    }
}
