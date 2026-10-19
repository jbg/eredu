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
