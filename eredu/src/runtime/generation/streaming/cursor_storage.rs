//! Private storage dispatch for the existing commitment driver. These methods
//! grant no admission: the caller must already own the original core sequence.
use super::*;
use eredu_core::{
    GenerationSequenceStorage, GenerationTokenIds, RetainedGenerationSequence,
    RetainedGenerationSequenceStorage, RetainedSequencePreparationError,
};

pub(crate) trait CursorStorage: GenerationSequenceStorage + Sized {
    type PreparationError: fmt::Debug;
    fn decode_token(
        _sequence: &mut GenerationSequence<Self>,
        _id: u32,
    ) -> Option<Result<Option<&str>, eredu_core::GenerationDecoderError>> {
        None
    }
    fn finish_decoder(
        _sequence: &mut GenerationSequence<Self>,
    ) -> Option<Result<(), eredu_core::GenerationDecoderError>> {
        None
    }
    fn prepare(
        sequence: GenerationSequence<Self>,
    ) -> Result<GenerationSequence<Self>, Self::PreparationError>;
}
impl CursorStorage for OwnedGenerationStorage {
    type PreparationError = std::convert::Infallible;
    fn prepare(
        sequence: GenerationSequence<Self>,
    ) -> Result<GenerationSequence<Self>, Self::PreparationError> {
        Ok(sequence)
    }
}
impl CursorStorage for RetainedGenerationSequenceStorage {
    type PreparationError = RetainedSequencePreparationError;
    fn decode_token(
        sequence: &mut GenerationSequence<Self>,
        id: u32,
    ) -> Option<Result<Option<&str>, eredu_core::GenerationDecoderError>> {
        if sequence.matches_decoder_input(None) {
            None
        } else {
            Some(sequence.decode_token(id))
        }
    }
    fn finish_decoder(
        sequence: &mut GenerationSequence<Self>,
    ) -> Option<Result<(), eredu_core::GenerationDecoderError>> {
        if sequence.matches_decoder_input(None) {
            None
        } else {
            Some(sequence.finish_decoder())
        }
    }
    fn prepare(
        sequence: GenerationSequence<Self>,
    ) -> Result<GenerationSequence<Self>, Self::PreparationError> {
        sequence.prepare_storage()
    }
}
impl CommittedGenerationCursor<RetainedGenerationSequenceStorage> {
    /// Move the actual dormant original provider into the same cursor. This
    /// allocates no token slots and neither quotes nor validates outer owners.
    pub(crate) fn from_retained_sequence(sequence: RetainedGenerationSequence) -> Self {
        Self {
            sequence: Some(sequence),
            finish_reason: None,
            failed: false,
        }
    }

    /// Freeze the terminal original token owner without copying or allocating.
    /// A nonterminal/failed cursor is returned intact; no result is fabricated.
    pub(crate) fn into_retained_tokens(self) -> Result<(GenerationTokenIds, FinishReason), Self> {
        if self.failed || self.finish_reason.is_none() {
            return Err(self);
        }
        let reason = self.finish_reason.expect("terminal retained cursor");
        Ok((
            self.sequence
                .expect("terminal retained sequence")
                .into_token_ids(),
            reason,
        ))
    }
}

mod consumer;
pub(crate) use consumer::{RetainedConsumerCursor, RetainedCursorFailure};

#[cfg(test)]
mod tests;
