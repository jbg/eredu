//! Protocol-specific options for architecture-dispatched realtime loading.

use eredu_architectures::voicechat::component_residency::VoiceChatResidencyPolicy;
use eredu_runtime::{CacheResidencyPolicy, NormalizedLoadRequest};

use super::VoiceChatStartOptions;

/// Host and resource policy for a PCM speech conversation.
#[derive(Debug, Clone)]
pub struct SpeechSessionOptions {
    /// Eligible matrix transformations, realized through bounded prepared storage.
    pub quantization: eredu_architectures::voicechat::quantization::SpeechQuantizationPolicy,
    /// Conversation, queue, sampling and recorded-time controls.
    pub conversation: VoiceChatStartOptions,
    /// Exact independent parameter pools for the selected speech graph.
    pub parameters: VoiceChatResidencyPolicy,
    /// Requested realization of both backbone and TTS attention state.
    pub state: CacheResidencyPolicy,
    /// Maximum storage and temporary bytes for one unpublished neural branch.
    pub max_branch_bytes: u64,
    /// Cumulative copy budget shared by isolated descendants.
    pub max_total_copy_bytes: u64,
    /// Initial neural random state; host phrase streams have their own explicit seeds.
    pub seed: u64,
}

/// Required input protocol and its policy. Artifact discovery chooses the family;
/// an incompatible protocol fails before native resources are allocated.
#[derive(Debug, Clone)]
pub enum RealtimeLoadOptions {
    /// Categorical text/audio-code frames, including Moshi and PersonaPlex.
    Categorical(NormalizedLoadRequest),
    /// PCM input with speech, transcription, text and conversational tool events.
    Speech(SpeechSessionOptions),
}
