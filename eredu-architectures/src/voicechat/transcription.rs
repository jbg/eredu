//! Incremental transcript policy of the pinned NVIDIA streaming wrapper.
//!
//! The wrapper decodes its accumulated RNNT vocabulary pieces by replacing
//! SentencePiece word markers with spaces and applying Python `str.strip`.
//! It does not invoke SentencePiece model decoding: unknown and byte-looking
//! pieces remain literal strings. The decoder below implements that policy
//! without retaining or decoding the accumulated hypothesis again.

use std::sync::Arc;

use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    BackendFailure,
};
use eredu_runtime::execution_control::{SnapshotReservation, SnapshotValue};

use super::assets::PreparedVoiceChatAssets;

/// Rejected transcript input or explicit retained/output text capacity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RnntTextError {
    /// A decoder requires a vocabulary and a positive pending-space capacity.
    #[error("invalid RNNT transcript configuration: {0}")]
    Configuration(&'static str),
    /// Emitted IDs must index the vocabulary; the RNNT blank is not text.
    #[error("RNNT transcript token {token} is outside vocabulary of size {vocabulary_size}")]
    Token {
        /// Invalid emitted token, including the blank if incorrectly supplied.
        token: usize,
        /// Number of nonblank vocabulary entries.
        vocabulary_size: usize,
    },
    /// Whitespace waiting for subsequent content exceeds its retained budget.
    #[error("RNNT transcript pending whitespace exceeds {0} bytes")]
    PendingCapacity(usize),
    /// The next event exceeds its caller-supplied UTF-8 byte budget.
    #[error("RNNT transcript output exceeds {0} bytes")]
    OutputCapacity(usize),
    /// A finalized transcript cannot accept additional emitted IDs.
    #[error("RNNT transcript is finished")]
    Finished,
}

/// Append-only text deltas matching the released RNNT transcript presentation.
///
/// Clones share the immutable vocabulary and copy only bounded trailing
/// whitespace. No prior tokens or delivered text are retained. A rejected
/// append leaves all state unchanged, allowing an unpublished host candidate
/// to be discarded or retried. In contrast to the reference display helper,
/// invalid IDs are errors rather than silently discarded decoder corruption.
#[derive(Debug, Clone)]
pub struct RnntTextDecoder {
    vocabulary: Arc<[String]>,
    max_pending_bytes: usize,
    pending: String,
    started: bool,
    finished: bool,
}

impl SnapshotValue for RnntTextDecoder {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        // A saved decoder can outlive its original owner. Charge the shared
        // vocabulary's retained storage, without charging a nonexistent copy.
        let bytes = std::mem::size_of::<Self>()
            .checked_add(self.pending.capacity())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        let mut retained_bytes = bytes
            .checked_add((2 * std::mem::size_of::<usize>()) as u64)
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        for piece in self.vocabulary.iter() {
            let storage = std::mem::size_of::<String>()
                .checked_add(piece.capacity())
                .and_then(|storage| u64::try_from(storage).ok())
                .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
            retained_bytes = retained_bytes
                .checked_add(storage)
                .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        }
        Ok(SnapshotEstimate {
            retained_bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut pending = String::new();
        pending
            .try_reserve_exact(self.pending.len())
            .map_err(BackendFailure::from_error)?;
        pending.push_str(&self.pending);
        Ok(Self {
            vocabulary: Arc::clone(&self.vocabulary),
            max_pending_bytes: self.max_pending_bytes,
            pending,
            started: self.started,
            finished: self.finished,
        })
    }
}

impl RnntTextDecoder {
    /// Constructs from the exact already-validated retained RNNT vocabulary.
    /// No artifacts are reopened and no network access occurs.
    pub fn from_assets(
        assets: &PreparedVoiceChatAssets,
        max_pending_bytes: usize,
    ) -> Result<Self, RnntTextError> {
        Self::from_vocabulary(assets.rnnt_vocabulary(), max_pending_bytes)
    }

    /// Constructs from an explicitly supplied nonblank vocabulary.
    ///
    /// This constructor also supports neutral fixtures. Production release
    /// preparation uses [`Self::from_assets`] to preserve validated ordering.
    pub fn from_vocabulary(
        vocabulary: &[String],
        max_pending_bytes: usize,
    ) -> Result<Self, RnntTextError> {
        if vocabulary.is_empty() {
            return Err(RnntTextError::Configuration("empty vocabulary"));
        }
        if max_pending_bytes == 0 {
            return Err(RnntTextError::Configuration("zero pending byte capacity"));
        }
        Ok(Self {
            vocabulary: Arc::from(vocabulary),
            max_pending_bytes,
            pending: String::new(),
            started: false,
            finished: false,
        })
    }

    /// Decodes newly committed nonblank IDs into one bounded UTF-8 delta.
    ///
    /// Leading whitespace is removed. Trailing whitespace is retained until
    /// later non-whitespace proves it belongs inside the transcript. Thus a
    /// standalone word-boundary token never publishes a retractable space.
    pub fn append(
        &mut self,
        emitted: &[usize],
        max_output_bytes: usize,
    ) -> Result<String, RnntTextError> {
        if self.finished {
            return Err(RnntTextError::Finished);
        }
        let mut pending = self.pending.clone();
        let mut started = self.started;
        let mut output = String::new();
        for &token in emitted {
            let piece = self.vocabulary.get(token).ok_or(RnntTextError::Token {
                token,
                vocabulary_size: self.vocabulary.len(),
            })?;
            for character in piece.chars() {
                let character = if character == '\u{2581}' {
                    ' '
                } else {
                    character
                };
                if python_whitespace(character) {
                    if started {
                        if pending.len().saturating_add(character.len_utf8())
                            > self.max_pending_bytes
                        {
                            return Err(RnntTextError::PendingCapacity(self.max_pending_bytes));
                        }
                        pending.push(character);
                    }
                } else {
                    if output
                        .len()
                        .saturating_add(pending.len())
                        .saturating_add(character.len_utf8())
                        > max_output_bytes
                    {
                        return Err(RnntTextError::OutputCapacity(max_output_bytes));
                    }
                    output.push_str(&pending);
                    pending.clear();
                    output.push(character);
                    started = true;
                }
            }
        }
        self.pending = pending;
        self.started = started;
        Ok(output)
    }

    /// Finalizes `strip` by discarding trailing whitespace; emits no new text.
    /// Repeated finish calls are harmless.
    pub fn finish(&mut self) {
        self.pending.clear();
        self.finished = true;
    }
}

fn python_whitespace(character: char) -> bool {
    // Python includes these information separators in addition to Unicode's
    // White_Space property used by Rust's char::is_whitespace.
    character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn committed_deltas_match_pinned_nvidia_transcripts_across_piece_boundaries() {
        let fixture: serde_json::Value =
            serde_json::from_str(eredu_evaluation::voicechat::TRANSCRIPTION_REFERENCE_JSON)
                .unwrap();
        let vocabulary: Vec<String> =
            serde_json::from_value(fixture["vocabulary"].clone()).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let tokens: Vec<usize> = serde_json::from_value(case["tokens"].clone()).unwrap();
            let expected: Vec<String> = serde_json::from_value(case["prefixes"].clone()).unwrap();
            let mut decoder = RnntTextDecoder::from_vocabulary(&vocabulary, 64).unwrap();
            let mut delivered = String::new();
            for (token, expected) in tokens.iter().zip(expected) {
                delivered.push_str(&decoder.append(&[*token], 256).unwrap());
                assert_eq!(delivered, expected, "{}", case["name"]);
            }
            decoder.finish();
            assert_eq!(delivered, case["text"].as_str().unwrap());
            let mut together = RnntTextDecoder::from_vocabulary(&vocabulary, 64).unwrap();
            assert_eq!(together.append(&tokens, 256).unwrap(), delivered);
        }
    }

    #[test]
    fn capacity_and_invalid_ids_leave_the_candidate_unchanged() {
        let vocabulary = ["▁hello", "▁", "世界", "    "].map(String::from);
        let mut decoder = RnntTextDecoder::from_vocabulary(&vocabulary, 3).unwrap();
        assert_eq!(decoder.append(&[0, 1], 5).unwrap(), "hello");
        let mut fork = decoder.clone();
        assert_eq!(
            decoder.append(&[2], 6),
            Err(RnntTextError::OutputCapacity(6))
        );
        assert_eq!(
            decoder.append(&[3], 20),
            Err(RnntTextError::PendingCapacity(3))
        );
        assert!(matches!(
            decoder.append(&[2, 4], 20),
            Err(RnntTextError::Token { token: 4, .. })
        ));
        assert_eq!(decoder.append(&[2], 7).unwrap(), " 世界");
        assert_eq!(fork.append(&[2], 7).unwrap(), " 世界");
        decoder.finish();
        assert_eq!(decoder.append(&[], 0), Err(RnntTextError::Finished));
    }
}
