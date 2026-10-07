//! Portable VoiceChat host policy around committed shared realtime transitions.

mod bound;
mod conversation;
mod decoded_text;
mod driver;
mod ingress;
mod loading;
mod phrase;
mod phrase_selection;
mod prompt;
mod publication;
mod publication_queue;
mod session;
mod submission;
mod text;
mod tools;
mod turn;
mod wait;

pub use bound::{BoundVoiceChatFork, BoundVoiceChatSession, VoiceChatExecutor};
pub use loading::{
    PreparedVoiceChatSession, PreparedVoiceChatSessionParts, VoiceChatLoadingError,
    VoiceChatStartOptions,
};
pub use phrase_selection::VoiceChatPhrasePolicy;

pub use ingress::{
    VoiceChatInputConfig, VoiceChatInputError, VoiceChatInputPermit, VoiceChatInputSender,
    VoiceChatInputStatus,
};

pub use conversation::{
    VoiceChatConversation, VoiceChatConversationError, VoiceChatConversationOutput,
    VoiceChatPublicationCapacity, VoiceChatTranscriptionText,
};

pub use decoded_text::{VoiceChatAgentText, VoiceChatDecodeError, VoiceChatDecodeLimits};
pub use driver::{
    VoiceChatAdvance, VoiceChatBlocked, VoiceChatPreparation, VoiceChatToolAudioPolicy,
    VoiceChatWork, VoiceChatWorkKind,
};
pub use session::{
    VoiceChatFork, VoiceChatSession, VoiceChatSessionConfig, VoiceChatSessionError,
    VoiceChatSessionFailure, VoiceChatSessionProgress, VoiceChatSessionSnapshot,
    VoiceChatSessionStatus, VoiceChatSnapshotError,
};
#[cfg(feature = "mlx")]
pub use submission::submit_mlx_voicechat;
pub use submission::{
    VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatSpeechOptions,
    VoiceChatSubmittedWork,
};

pub use publication::{
    VoiceChatPublicationAdmission, VoiceChatPublicationBranch, VoiceChatPublicationPermit,
    VoiceChatPublicationState,
};
pub use publication_queue::{
    VoiceChatPublicationError, VoiceChatPublicationLimits, VoiceChatPublicationReceiver,
    VoiceChatPublicationUsage,
};

pub use turn::{
    VoiceChatTurnAdmission, VoiceChatTurnError, VoiceChatTurnEvent, VoiceChatTurnLimits,
    VoiceChatTurnPolicy, VoiceChatTurnResolver, VoiceChatTurnTaking,
};

pub use phrase::PreparedVoiceChatPhrase;
pub use wait::{
    VoiceChatInitialReminder, VoiceChatReminderPlayback, VoiceChatReminderStep, VoiceChatToolWait,
    VoiceChatToolWaitAction, VoiceChatToolWaitError, VoiceChatToolWaitMessages,
    VoiceChatToolWaitPolicy,
};

pub use text::{
    PreparedVoiceChatPrompt, PreparedVoiceChatToolResponse, VoiceChatTextError, VoiceChatTextKind,
    VoiceChatTextLimits, VoiceChatTextPreparation,
};
pub use tools::{
    VoiceChatToolBatch, VoiceChatToolChannel, VoiceChatToolError, VoiceChatToolLimits,
    VoiceChatToolTicket, VoiceChatTurnIdentity,
};

#[cfg(feature = "mlx")]
mod selected_mlx;
#[cfg(feature = "mlx")]
pub(crate) use selected_mlx::{start_mlx_voicechat, MlxVoiceChatLoadOptions};
#[cfg(feature = "mlx")]
pub use selected_mlx::{MlxVoiceChatExecutor, MlxVoiceChatSession, VoiceChatStartError};
