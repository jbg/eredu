//! Host conditioning constructed from exact retained assets and integer payloads.
//!
//! Preparation allocates no native tensor and performs no artifact reads. Saved
//! character flags retain their checkpoint semantics independently of live EOS.

use std::{collections::BTreeMap, sync::Arc};

use eredu_text::tokenizer::{token_id_vocabulary, Tokenizer};

use super::{
    artifact::RetainedVoiceChatArtifacts,
    assets::{PreparedVoiceChatAssets, VoiceChatAssetRole},
    checkpoint::Component,
    config::VoiceChatConfig,
    controls::VOICECHAT_INTEGER_BUFFERS,
    tts_conditioning::{CharacterFlags, CharacterMapping},
    tts_feedback::FeedbackControls,
    tts_mog::TtsError,
};

/// Text controls resolved against the exact retained tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct TextControls {
    pub bos: u32,
    pub eos: u32,
    pub pad: u32,
    pub listening: u32,
    pub user_bos: u32,
    pub user_eos: u32,
    pub tool_call_start: u32,
    pub tool_call_end: u32,
    pub tool_response_end: u32,
}

/// Invalid or incomplete host conditioning; no native resource is allocated.
#[derive(Debug, thiserror::Error)]
pub enum HostPreparationError {
    /// The selected task does not retain a component needed for speech.
    #[error("VoiceChat host conditioning requires component {0:?}")]
    MissingComponent(Component),
    /// A saved integer buffer is absent or cannot satisfy its host representation.
    #[error("invalid retained VoiceChat host buffer {0}")]
    IntegerBuffer(&'static str),
    /// The text tokenizer could not be reconstructed, preserving its source.
    #[error("VoiceChat tokenizer construction failed: {0}")]
    Tokenizer(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A configured marker is absent from the canonical vocabulary.
    #[error("VoiceChat tokenizer is missing control token {0}")]
    MissingToken(String),
    /// The saved character/control geometry is inconsistent.
    #[error(transparent)]
    Conditioning(#[from] TtsError),
}

/// Native-independent inputs shared by prompt preparation and cached speech.
pub struct PreparedHostConditioning {
    /// Reconstructed base tokenizer; no path or resolver is needed to use it.
    pub tokenizer: Tokenizer,
    /// Raw token spelling to character IDs, with saved continuation/BOS/EOS flags.
    pub characters: CharacterMapping,
    /// Validated speech controls and codec silence feedback.
    pub feedback: FeedbackControls,
    /// Live text, user-turn and tool markers.
    pub tokens: TextControls,
}

impl PreparedHostConditioning {
    /// Constructs from retained host payloads, without reopening any source store.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
    ) -> Result<Self, HostPreparationError> {
        for component in [Component::Tts, Component::Controls] {
            if artifacts.component_source(component).is_none() {
                return Err(HostPreparationError::MissingComponent(component));
            }
        }
        Self::prepare(
            artifacts.plan().config(),
            artifacts.assets(),
            artifacts.integer_buffers(),
        )
    }

    fn prepare(
        config: &VoiceChatConfig,
        assets: &PreparedVoiceChatAssets,
        buffers: &BTreeMap<String, Arc<[i64]>>,
    ) -> Result<Self, HostPreparationError> {
        let read = |index: usize| -> Result<Vec<i32>, HostPreparationError> {
            let contract = VOICECHAT_INTEGER_BUFFERS[index];
            let values = buffers
                .get(contract.key)
                .filter(|values| values.len() == contract.values)
                .ok_or(HostPreparationError::IntegerBuffer(contract.key))?;
            values
                .iter()
                .map(|&value| {
                    i32::try_from(value)
                        .map_err(|_| HostPreparationError::IntegerBuffer(contract.key))
                })
                .collect()
        };
        let flags = CharacterFlags {
            special_pad: read(2)?[0],
            special: read(3)?,
            continuation: read(4)?,
            continuation_pad: read(5)?[0],
        };
        let tokenizer =
            Tokenizer::from_bytes(assets.asset(VoiceChatAssetRole::TextTokenizer).bytes())
                .map_err(HostPreparationError::Tokenizer)?;
        let resolve = |token: &str| {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| HostPreparationError::MissingToken(token.to_owned()))
        };
        let policy = &config.tokens;
        let tokens = TextControls {
            bos: resolve(&policy.bos)?,
            eos: resolve(&policy.eos)?,
            pad: resolve(&policy.pad)?,
            listening: resolve("<SPECIAL_11>")?,
            user_bos: resolve(&policy.user_bos)?,
            user_eos: resolve(&policy.user_eos)?,
            tool_call_start: resolve(&policy.tool_call_start)?,
            tool_call_end: resolve(&policy.tool_call_end)?,
            tool_response_end: resolve(&policy.tool_response_end)?,
        };
        let characters = CharacterMapping::new(&token_id_vocabulary(&tokenizer), flags)?;
        if characters.characters().len() != 256 {
            return Err(TtsError::Configuration(
                "released character alphabet must contain 256 entries",
            )
            .into());
        }
        let feedback = FeedbackControls::new(
            config.tts.codebook_size as i32,
            tokens.eos as i32,
            tokens.pad as i32,
            config.tts.force_silence_on_eos,
            &read(0)?,
            &read(1)?,
        )?;
        Ok(Self {
            tokenizer,
            characters,
            feedback,
            tokens,
        })
    }
}
