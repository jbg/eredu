//! Fixed mappings shared by original sequence and speculative semantic decoders.
use eredu_core::GenerationDecoderError;
use eredu_text::decoder_storage::{DecodeStorageError, OwnedDecodeStorageError};
use eredu_text::stop_storage::StopStorageError;
pub(super) fn transition_error(error: OwnedDecodeStorageError) -> GenerationDecoderError {
    match error {
        OwnedDecodeStorageError::Unprepared => GenerationDecoderError::Unprepared,
        OwnedDecodeStorageError::Storage(error) => match error {
            DecodeStorageError::CallLimit => GenerationDecoderError::CallLimit,
            DecodeStorageError::HistoryLimit => GenerationDecoderError::HistoryLimit,
            DecodeStorageError::InvalidPrefix {
                token_id,
                expected_bytes,
                actual_bytes,
            } => GenerationDecoderError::InvalidPrefix {
                token_id,
                expected_bytes,
                actual_bytes,
            },
            DecodeStorageError::IncompleteByteSequence => {
                GenerationDecoderError::IncompleteByteSequence
            }
            DecodeStorageError::Overflow | DecodeStorageError::Extent { .. } => {
                GenerationDecoderError::InvalidStorage
            }
        },
    }
}

pub(super) fn stop_error(error: StopStorageError) -> GenerationDecoderError {
    match error {
        StopStorageError::Unprepared => GenerationDecoderError::Unprepared,
        StopStorageError::Overflow
        | StopStorageError::InvalidStorage
        | StopStorageError::InputLimit => GenerationDecoderError::InvalidStorage,
    }
}
