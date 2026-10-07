//! Bounded spoken acknowledgement/reminder framing.
//!
//! Framing follows NVIDIA Speech `streaming_s2s_pipeline.py` at
//! `097dfe9e2f55baf653b83035868bdc89849f1b47`:
//! <https://github.com/NVIDIA-NeMo/Speech/blob/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/pipelines/streaming_s2s_pipeline.py#L172-L271>.

use super::{VoiceChatTextError, VoiceChatTextKind, VoiceChatTextPreparation};

/// A complete spoken phrase, including its speech-tail PAD frames and final EOS.
///
/// The trailing PAD tokens remain inside the utterance. They must pass through
/// the architecture's reminder rendering policy so that speech can finish.
#[derive(Debug, Clone)]
pub struct PreparedVoiceChatPhrase {
    tokens: Vec<i32>,
}

impl PreparedVoiceChatPhrase {
    pub(super) fn snapshot_retention(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, eredu_core::BackendFailure> {
        let mut estimate = super::conversation::host_storage::<Self>(
            self.tokens.capacity(),
            std::mem::size_of::<i32>(),
        )?;
        estimate.copy_bytes = 0;
        Ok(estimate)
    }

    /// Encodes ASCII text and admits the complete framed utterance before allocation.
    ///
    /// The retained preparation's ordinary byte and unframed-token limits apply.
    /// `max_tokens` additionally bounds BOS, all content tokens, trailing PADs
    /// and EOS together. Text validation uses the existing tool-response errors.
    pub fn prepare(
        preparation: &VoiceChatTextPreparation,
        text: &str,
        max_tokens: usize,
    ) -> Result<Self, VoiceChatTextError> {
        if max_tokens == 0 {
            return Err(VoiceChatTextError::InvalidLimit("spoken phrase token"));
        }
        let content = preparation.prepare_tool_response(text)?;
        // The shared preparation has already established ASCII, so bytes equal
        // characters. Integer ceiling avoids floating-point length conversion.
        let trailing_pads = 17.max(text.len().div_ceil(2));
        let required = content
            .tokens()
            .len()
            .checked_add(trailing_pads)
            .and_then(|length| length.checked_add(2))
            .ok_or(VoiceChatTextError::TokenCapacity {
                kind: VoiceChatTextKind::ToolResponse,
                required: usize::MAX,
                capacity: max_tokens,
            })?;
        if required > max_tokens {
            return Err(VoiceChatTextError::TokenCapacity {
                kind: VoiceChatTextKind::ToolResponse,
                required,
                capacity: max_tokens,
            });
        }
        let controls = preparation.controls();
        let mut tokens = Vec::with_capacity(required);
        tokens.push(controls.bos as i32);
        tokens.extend_from_slice(content.tokens());
        tokens.resize(required - 1, controls.pad as i32);
        tokens.push(controls.eos as i32);
        Ok(Self { tokens })
    }

    /// Exact TTS input sequence: BOS, content, speech-tail PADs, EOS.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::voicechat::VoiceChatTextLimits;
    use eredu_architectures::voicechat::host::TextControls;
    use eredu_text::tokenizer::Tokenizer;
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit};

    fn preparation() -> VoiceChatTextPreparation {
        let words = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]", 0),
                    ("<s>", 1),
                    ("</s>", 2),
                    ("user-start", 3),
                    ("user-end", 4),
                    ("listen", 11),
                    ("pad", 12),
                    ("call-start", 20),
                    ("call-end", 21),
                    ("response-end", 22),
                    ("hello", 30),
                    ("world", 31),
                ]
                .into_iter()
                .map(|(word, id)| (word.to_owned(), id))
                .collect(),
            )
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(words);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        VoiceChatTextPreparation::from_tokenizer(
            &Tokenizer::from_tokenizer(tokenizer),
            TextControls {
                bos: 1,
                eos: 2,
                pad: 12,
                listening: 11,
                user_bos: 3,
                user_eos: 4,
                tool_call_start: 20,
                tool_call_end: 21,
                tool_response_end: 22,
            },
            VoiceChatTextLimits {
                max_bytes: 64,
                max_tokens: 16,
            },
        )
        .unwrap()
    }

    #[test]
    fn spoken_phrase_retains_content_and_scales_drain_before_eos() {
        let preparation = preparation();
        let short = PreparedVoiceChatPhrase::prepare(&preparation, "hello world", 21).unwrap();
        let mut expected = vec![1, 30, 31];
        expected.extend([12; 17]);
        expected.push(2);
        assert_eq!(short.tokens(), expected);

        // Thirty-five ASCII characters require eighteen drain frames, even
        // though this fixture represents the whole phrase in only six tokens.
        let long = PreparedVoiceChatPhrase::prepare(
            &preparation,
            "hello hello hello hello hello hello",
            26,
        )
        .unwrap();
        let mut expected = vec![1, 30, 30, 30, 30, 30, 30];
        expected.extend([12; 18]);
        expected.push(2);
        assert_eq!(long.tokens(), expected);
    }

    #[test]
    fn spoken_phrase_admits_expanded_capacity_and_shared_text_validation() {
        let preparation = preparation();
        assert!(matches!(
            PreparedVoiceChatPhrase::prepare(&preparation, "hello world", 20),
            Err(VoiceChatTextError::TokenCapacity {
                required: 21,
                capacity: 20,
                ..
            })
        ));
        assert!(matches!(
            PreparedVoiceChatPhrase::prepare(&preparation, "hello", 0),
            Err(VoiceChatTextError::InvalidLimit(_))
        ));
        assert!(matches!(
            PreparedVoiceChatPhrase::prepare(&preparation, "héllo", 64),
            Err(VoiceChatTextError::NonAscii { byte_offset: 1, .. })
        ));
        assert!(matches!(
            PreparedVoiceChatPhrase::prepare(&preparation, " \n ", 64),
            Err(VoiceChatTextError::EmptyToolResponse)
        ));
        assert!(matches!(
            PreparedVoiceChatPhrase::prepare(&preparation, &"x".repeat(65), 64),
            Err(VoiceChatTextError::ByteCapacity {
                required: 65,
                capacity: 64,
                ..
            })
        ));
    }
}
