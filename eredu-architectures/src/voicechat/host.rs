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

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact release bytes live outside the tracked source tree.
    #[test]
    #[ignore = "requires VOICECHAT_REFERENCE with pinned assets and integer payload ranges"]
    fn released_host_conditioning_matches_published_character_tables() {
        let root = std::path::PathBuf::from(
            std::env::var_os("VOICECHAT_REFERENCE").expect("reference directory"),
        );
        let assets = PreparedVoiceChatAssets::from_local_directories(
            &root.join("assets"),
            &root.join("base-assets"),
        )
        .unwrap();
        let config = VoiceChatConfig::from_value(assets.configuration()).unwrap();
        let header: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("official-header.json")).unwrap())
                .unwrap();
        let integers = std::fs::read(root.join("official-integer-buffers.bin")).unwrap();
        let buffers = super::super::controls::retain_integer_buffers::<std::io::Error>(
            &VOICECHAT_INTEGER_BUFFERS,
            |key, range| {
                assert!(range.len() <= super::super::controls::CONTROL_READ_VALUES);
                let offset = header[key]["data_offsets"][0].as_u64().unwrap() as usize;
                Ok(integers[offset + 8 * range.start..offset + 8 * range.end]
                    .chunks_exact(8)
                    .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect())
            },
        )
        .unwrap();
        // The constructor receives owned admitted values, never the source file.
        drop(integers);
        let prepared = PreparedHostConditioning::prepare(&config, &assets, &buffers).unwrap();
        assert_eq!(
            prepared.tokens,
            TextControls {
                bos: 1,
                eos: 2,
                pad: 12,
                listening: 11,
                user_bos: 1094,
                user_eos: 1036,
                tool_call_start: 20,
                tool_call_end: 21,
                tool_response_end: 22
            }
        );
        assert_eq!(prepared.characters.characters().len(), 256);
        assert_eq!(prepared.feedback.codebooks(), 31);
        // Check every raw spelling against the independently published split GGUF tables.
        let tables = std::fs::read(root.join("split-character-buffers.bin")).unwrap();
        let read_table =
            |key: &str, range: std::ops::Range<usize>| -> Result<Vec<i64>, std::io::Error> {
                let start = match key {
                    "tts.subword.char_ids" => 0,
                    "tts.subword.char_offsets" => 3564576,
                    _ => unreachable!(),
                };
                Ok(tables[start + 4 * range.start..start + 4 * range.end]
                    .chunks_exact(4)
                    .map(|value| i64::from(i32::from_le_bytes(value.try_into().unwrap())))
                    .collect())
            };
        super::super::controls::validate_split_character_tables(read_table).unwrap();
        let table = |key, count| {
            read_table(key, 0..count)
                .unwrap()
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>()
        };
        let ids = table("tts.subword.char_ids", 891139);
        let offsets = table("tts.subword.char_offsets", 131073);
        for token in 0..131072 {
            let batch = prepared
                .characters
                .prepare(&[token], None, 1, 1, 1024)
                .unwrap();
            assert_eq!(
                batch.ids(),
                &ids[offsets[token as usize] as usize..offsets[token as usize + 1] as usize],
                "token {token}"
            );
        }
        let mut missing = buffers.clone();
        missing.remove(VOICECHAT_INTEGER_BUFFERS[3].key);
        assert!(matches!(
            PreparedHostConditioning::prepare(&config, &assets, &missing),
            Err(HostPreparationError::IntegerBuffer(_))
        ));
    }
}
