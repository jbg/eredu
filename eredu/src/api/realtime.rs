//! Backend-neutral realtime preparation and scheduling facade.
//!
//! These types remain available when no concrete execution backend feature is
//! enabled. Architecture and runtime crates own the underlying selection,
//! transaction, and scheduling semantics; this module gives applications one
//! facade namespace for composing those semantics with reference, mock, or
//! selected native mechanisms.

pub use crate::runtime::voicechat::{
    BoundVoiceChatFork, BoundVoiceChatSession, PreparedVoiceChatPhrase, PreparedVoiceChatPrompt,
    PreparedVoiceChatSession, PreparedVoiceChatSessionParts, PreparedVoiceChatToolResponse,
    VoiceChatAdvance, VoiceChatAgentText, VoiceChatBlocked, VoiceChatConversation,
    VoiceChatConversationError, VoiceChatConversationOutput, VoiceChatDecodeError,
    VoiceChatDecodeLimits, VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatExecutor,
    VoiceChatFork, VoiceChatInitialReminder, VoiceChatInputConfig, VoiceChatInputError,
    VoiceChatInputPermit, VoiceChatInputSender, VoiceChatInputStatus, VoiceChatLoadingError,
    VoiceChatPhrasePolicy, VoiceChatPreparation, VoiceChatPublicationAdmission,
    VoiceChatPublicationBranch, VoiceChatPublicationCapacity, VoiceChatPublicationError,
    VoiceChatPublicationLimits, VoiceChatPublicationPermit, VoiceChatPublicationReceiver,
    VoiceChatPublicationState, VoiceChatPublicationUsage, VoiceChatReminderPlayback,
    VoiceChatReminderStep, VoiceChatSession, VoiceChatSessionConfig, VoiceChatSessionError,
    VoiceChatSessionFailure, VoiceChatSessionProgress, VoiceChatSessionSnapshot,
    VoiceChatSessionStatus, VoiceChatSnapshotError, VoiceChatSpeechOptions, VoiceChatStartOptions,
    VoiceChatSubmittedWork, VoiceChatTextError, VoiceChatTextKind, VoiceChatTextLimits,
    VoiceChatTextPreparation, VoiceChatToolAudioPolicy, VoiceChatToolBatch, VoiceChatToolChannel,
    VoiceChatToolError, VoiceChatToolLimits, VoiceChatToolTicket, VoiceChatToolWait,
    VoiceChatToolWaitAction, VoiceChatToolWaitError, VoiceChatToolWaitMessages,
    VoiceChatToolWaitPolicy, VoiceChatTranscriptionText, VoiceChatTurnAdmission,
    VoiceChatTurnError, VoiceChatTurnEvent, VoiceChatTurnIdentity, VoiceChatTurnLimits,
    VoiceChatTurnPolicy, VoiceChatTurnResolver, VoiceChatTurnTaking, VoiceChatWork,
    VoiceChatWorkKind,
};

mod loading;
pub use loading::{RealtimeLoadOptions, SpeechSessionOptions};
#[cfg(feature = "mlx")]
mod selected;
#[cfg(feature = "mlx")]
pub use crate::runtime::voicechat::{
    submit_mlx_voicechat, MlxVoiceChatExecutor, MlxVoiceChatSession, VoiceChatStartError,
};
#[cfg(feature = "mlx")]
pub use selected::{
    prepare_mlx_realtime, LoadedMlxRealtime, PreparedMlxRealtime, RealtimeLoadError,
};

pub use eredu_architectures::realtime::{
    prepare_realtime_model, PreparedRealtimeArtifact, RealtimeArtifactRequest,
    RealtimeArtifactSource, RealtimeAssetDirectories, RealtimePreparationError,
};

pub use eredu_architectures::voicechat::silence::{
    PreparedSilenceConditioning, SilenceConditioningError,
};
pub use eredu_architectures::voicechat::transcription::{RnntTextDecoder, RnntTextError};
pub use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy, PcmFrameSource};

pub use eredu_core::{
    execution_control::{SnapshotLimits, SnapshotUsage},
    scheduler::{
        RequestId, RequestStatus, SchedulerCapabilities, SchedulerLimits, SchedulerReport,
    },
    RealtimeChannelCadence, RealtimeChannelDeclaration, RealtimeChannelError,
    RealtimeChannelFeedback, RealtimeChannelId, RealtimeChannelLayout, RealtimeChannelPayload,
    RealtimeFrameScheduleState, RealtimeInputFrame, RealtimeOutputFrame, RealtimeSampling,
    RealtimeSpeechConfig, SessionCapabilities,
};
pub use eredu_runtime::{
    RealtimeChannelIngressError, RealtimeChannelValues, RealtimeFrameContract,
    RealtimeFrameContractError, RealtimeGenerationContract, RealtimeGenerationState,
    RealtimeModelSessionIdentity, RealtimeScheduleDescriptor, RealtimeScheduleState,
    RealtimeSessionInput, RealtimeSessionScheduler, ReleasedRealtimeSession,
    ValidatedRealtimeChannels,
};

/// A backend-neutral prepared realtime model bound to one authoritative
/// selected realization.
///
/// `M` may be an architecture-owned constructed model, a reference mechanism,
/// or an opaque selected-native execution object. Session identity and
/// capabilities stay portable regardless of that mechanism type.
pub struct PreparedRealtimeModel<M> {
    mechanism: M,
    identity: RealtimeModelSessionIdentity,
    capabilities: SessionCapabilities,
}

impl<M> PreparedRealtimeModel<M> {
    /// Binds a mechanism to the exact identity selected before construction.
    pub fn new(
        mechanism: M,
        selected: &eredu_runtime::SelectedRealtimeRealization,
        capabilities: SessionCapabilities,
    ) -> Self {
        Self {
            mechanism,
            identity: RealtimeModelSessionIdentity::from_selected(selected),
            capabilities,
        }
    }

    /// Returns the exact portable identity required by every session.
    pub const fn session_identity(&self) -> &RealtimeModelSessionIdentity {
        &self.identity
    }

    /// Returns capabilities of the exact selected realization.
    pub const fn session_capabilities(&self) -> SessionCapabilities {
        self.capabilities
    }

    /// Borrows the architecture or execution mechanism retained by this model.
    pub const fn mechanism(&self) -> &M {
        &self.mechanism
    }

    /// Mutably borrows the retained mechanism for frame execution.
    pub const fn mechanism_mut(&mut self) -> &mut M {
        &mut self.mechanism
    }

    /// Consumes the facade model into its retained mechanism.
    pub fn into_mechanism(self) -> M {
        self.mechanism
    }
}
