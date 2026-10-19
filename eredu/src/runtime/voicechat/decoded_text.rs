//! Bounded incremental decoding of committed effective agent-channel tokens.

use std::sync::Arc;

use eredu_architectures::voicechat::{
    host::TextControls, observation::OwnedHostOutput, schedule::Step,
};
use eredu_core::BackendFailure;
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use crate::api::TextDecoder;

use super::VoiceChatTextPreparation;

/// Independent bounds on one incremental agent-text decoding window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatDecodeLimits {
    /// Retained lookbehind IDs plus the next ID, before decoding that ID.
    pub max_pending_tokens: usize,
    /// Maximum retained text prefix or one emitted UTF-8 delta, in bytes.
    pub max_pending_bytes: usize,
}

impl Default for VoiceChatDecodeLimits {
    fn default() -> Self {
        Self {
            max_pending_tokens: 256,
            max_pending_bytes: 64 * 1024,
        }
    }
}

/// Invalid or exhausted incremental text decoding, without a native error type.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatDecodeError {
    /// Both limits must be positive.
    #[error("VoiceChat text decoding {0} limit must be positive")]
    Configuration(&'static str),
    /// More lookbehind would exceed the admitted token bound.
    #[error("VoiceChat text decoding pending-token capacity exhausted")]
    PendingTokens,
    /// A retained prefix or delta exceeds the admitted byte bound.
    #[error("VoiceChat text decoding pending-byte capacity exhausted")]
    PendingBytes,
    /// A generated ID is absent from the exact retained tokenizer.
    #[error("VoiceChat agent token {0} is absent from its tokenizer")]
    UnknownToken(u32),
    /// Tokenizer or incomplete UTF-8 failure with its original source retained.
    #[error("VoiceChat agent text decoding failed: {0}")]
    Decoder(#[source] BackendFailure),
}

/// A visible UTF-8 delta attributed to its containing committed model frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceChatAgentText {
    /// Complete UTF-8 text; channel control markers are excluded.
    pub text: String,
}

/// Copied with the unpublished host candidate and adopted with neural state.
/// Immutable tokenizer configuration is shared; private lookbehind is bounded.
#[derive(Clone)]
pub(super) struct VoiceChatAgentDecoder {
    decoder: TextDecoder,
    controls: TextControls,
    limits: VoiceChatDecodeLimits,
}

impl SnapshotValue for VoiceChatAgentDecoder {
    fn estimate_snapshot_value(
        &self,
    ) -> Result<eredu_core::execution_control::SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            super::conversation::host_storage::<Vec<u32>>(
                self.decoder.ids.capacity(),
                std::mem::size_of::<u32>(),
            ),
            super::conversation::host_storage::<String>(self.decoder.prefix.capacity(), 1),
        ])
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut ids = Vec::new();
        ids.try_reserve_exact(self.decoder.ids.len())
            .map_err(BackendFailure::from_error)?;
        ids.extend_from_slice(&self.decoder.ids);
        let mut prefix = String::new();
        prefix
            .try_reserve_exact(self.decoder.prefix.len())
            .map_err(BackendFailure::from_error)?;
        prefix.push_str(&self.decoder.prefix);
        Ok(Self {
            decoder: TextDecoder {
                tokenizer: Arc::clone(&self.decoder.tokenizer),
                skip_special_tokens: self.decoder.skip_special_tokens,
                ids,
                prefix,
                prefix_index: self.decoder.prefix_index,
            },
            controls: self.controls,
            limits: self.limits,
        })
    }
}

impl VoiceChatAgentDecoder {
    pub(super) fn new(
        text: &VoiceChatTextPreparation,
        limits: VoiceChatDecodeLimits,
    ) -> Result<Self, VoiceChatDecodeError> {
        if limits.max_pending_tokens == 0 {
            return Err(VoiceChatDecodeError::Configuration("tokens"));
        }
        if limits.max_pending_bytes == 0 {
            return Err(VoiceChatDecodeError::Configuration("bytes"));
        }
        Ok(Self {
            decoder: TextDecoder::from_shared_tokenizer(
                Arc::new((**text.tokenizer()).clone()),
                true,
            ),
            controls: text.controls(),
            limits,
        })
    }

    /// Decodes effective agent feedback, not rejected samples, independent TTS
    /// overrides or reminder inputs. Frame attribution belongs to the caller.
    pub(super) fn observe<D>(
        &mut self,
        frame: &OwnedHostOutput<D>,
    ) -> Result<Option<VoiceChatAgentText>, VoiceChatDecodeError> {
        if matches!(frame.step, Step::Drain) {
            self.finish()?;
            return Ok(None);
        }
        let Some(decisions) = frame.decisions else {
            return Ok(None);
        };
        self.push(decisions.feedback.agent())
    }

    /// Interruption discards only incomplete, unpublished lookbehind. Delivered
    /// text remains owned by its committed output and is never retracted.
    pub(super) fn interrupt(&mut self) {
        self.decoder.reset();
    }

    fn finish(&mut self) -> Result<(), VoiceChatDecodeError> {
        self.decoder.finish().map_err(decoder_failure)?;
        self.decoder.reset();
        Ok(())
    }

    fn push(&mut self, token: u32) -> Result<Option<VoiceChatAgentText>, VoiceChatDecodeError> {
        if token == self.controls.bos || token == self.controls.eos {
            self.finish()?;
            return Ok(None);
        }
        if [
            self.controls.pad,
            self.controls.listening,
            self.controls.user_bos,
            self.controls.user_eos,
            self.controls.tool_call_start,
            self.controls.tool_call_end,
            self.controls.tool_response_end,
        ]
        .contains(&token)
        {
            return Ok(None);
        }
        if self.decoder.tokenizer.id_to_token(token).is_none() {
            return Err(VoiceChatDecodeError::UnknownToken(token));
        }
        // Gate ID growth before tokenizers allocates its next window. Decoding
        // temporaries are bounded by this window and immutable tokenizer data;
        // only admitted prefix and delta bytes can enter retained host state.
        if self.decoder.ids.len() >= self.limits.max_pending_tokens {
            return Err(VoiceChatDecodeError::PendingTokens);
        }
        let mut candidate = self.decoder.clone();
        let delta = candidate.step(token).map_err(decoder_failure)?;
        if candidate.prefix.len() > self.limits.max_pending_bytes
            || delta
                .as_ref()
                .is_some_and(|text| text.len() > self.limits.max_pending_bytes)
        {
            return Err(VoiceChatDecodeError::PendingBytes);
        }
        self.decoder = candidate;
        Ok(delta
            .filter(|text| !text.is_empty())
            .map(|text| VoiceChatAgentText { text }))
    }
}

fn decoder_failure(error: crate::api::TextDecoderError) -> VoiceChatDecodeError {
    VoiceChatDecodeError::Decoder(
        BackendFailure::from_error(error).with_operation("VoiceChat agent text decoding"),
    )
}
