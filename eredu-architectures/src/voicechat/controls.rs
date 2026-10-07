//! Bounded admission of published integer buffers and derived character tables.
//!
//! Hashes describe canonical little-endian I64 values, allowing the published
//! split GGUF's I32 storage without weakening value checks. In particular, the
//! saved BOS/EOS embedding flags retain the base tokenizer's EOS at index 12;
//! the live text channel's EOS override at index 2 does not rewrite that buffer.

use std::{error::Error, ops::Range};

use sha2::{Digest, Sha256};

/// Maximum values requested at once during integer admission (32 KiB of I64).
pub const CONTROL_READ_VALUES: usize = 4096;

/// Exact source identity and value contract for a published integer buffer.
#[derive(Debug, Clone, Copy)]
pub struct IntegerBufferContract {
    /// Canonical key, or physical derived-table key for split character tables.
    pub key: &'static str,
    /// Number of scalar values, including singleton scalar buffers.
    pub values: usize,
    /// SHA-256 of canonical little-endian I64 values.
    pub sha256: &'static str,
}

/// Integer buffers from the official release, including both padding scalars.
pub const VOICECHAT_INTEGER_BUFFERS: [IntegerBufferContract; 6] = [
    IntegerBufferContract {
        key: "tts_model._control_codes",
        values: 3,
        sha256: "485b8db0f7741c9f8ccd7846c81ee206042f37a486cfb510775c5119ad6e23f4",
    },
    IntegerBufferContract {
        key: "tts_model.codec_silence_tokens",
        values: 31,
        sha256: "504404eab90d4269c1b0c26cfc958e125874fc1b78913234df4b2cb831448e36",
    },
    IntegerBufferContract {
        key: "tts_model.tts_model.embed_subword.bos_eos_emb.pad_tensor",
        values: 1,
        sha256: "32fda152378480c8e250e2b871b95b5e317f0caa0d3308866682583c0186c3ae",
    },
    IntegerBufferContract {
        key: "tts_model.tts_model.embed_subword.bos_eos_emb.special_flags",
        values: 131072,
        sha256: "9b9988f36c53280a69a79dacdbd662dde6cead2d2b986292be54203f385072f6",
    },
    IntegerBufferContract {
        key: "tts_model.tts_model.embed_subword.subword_flag_emb.is_continuation",
        values: 131073,
        sha256: "381cbab359259c628d2ed5ad17011319179cf47df40c19e35574ae3107ba4485",
    },
    IntegerBufferContract {
        key: "tts_model.tts_model.embed_subword.subword_flag_emb.pad_tensor",
        values: 1,
        sha256: "446bfbd9ec163187d37929dc2e44cc86f8403d574c2d841c9a5e4f3b5a0c20e3",
    },
];

/// Published split tables, independently checked against the pinned tokenizer's
/// character vocabulary ordered by single-character token IDs.
pub const VOICECHAT_SPLIT_CHARACTER_BUFFERS: [IntegerBufferContract; 2] = [
    IntegerBufferContract {
        key: "tts.subword.char_ids",
        values: 891139,
        sha256: "2b77dee497672766f14776a4c11bf77c4368873f8bbd491eb8280bf4252331ab",
    },
    IntegerBufferContract {
        key: "tts.subword.char_offsets",
        values: 131073,
        sha256: "de6ed590145f44ec9fba7b888161c2a7dd7e58c5a5d2fda20c4cec3b55febd7d",
    },
];

/// Integer semantic admission failure, retaining original source failures.
#[derive(Debug, thiserror::Error)]
pub enum IntegerBufferError {
    /// The exact bounded source read failed.
    #[error("could not read VoiceChat integer buffer {key}: {source}")]
    Read {
        /// Buffer whose read failed.
        key: &'static str,
        /// Original portable source error.
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
    /// A source did not honor the requested exact bounded range.
    #[error(
        "VoiceChat integer buffer {key} returned {actual} values for a {expected}-value range"
    )]
    RangeLength {
        /// Buffer whose read was invalid.
        key: &'static str,
        /// Requested values.
        expected: usize,
        /// Returned values.
        actual: usize,
    },
    /// Geometry was valid but values differed from the released semantics.
    #[error(
        "VoiceChat integer buffer {key} differs from the pinned release (SHA-256 {actual_sha256})"
    )]
    ValueMismatch {
        /// Buffer whose contents differed.
        key: &'static str,
        /// Digest of the resolved integer values.
        actual_sha256: String,
    },
}

/// Validates all canonical integer control buffers before publishing preparation.
///
/// `read` returns the requested flattened range as exact integers. The caller
/// reads from retained admitted sources and must reject non-integer encodings;
/// decoding I64 through floating-point storage is not part of this contract.
pub fn validate_control_tensors<E: Error + Send + Sync + 'static>(
    read: impl FnMut(&str, Range<usize>) -> Result<Vec<i64>, E>,
) -> Result<(), IntegerBufferError> {
    validate_integer_buffers(&VOICECHAT_INTEGER_BUFFERS, read)
}

/// Validates the split TTS character tables before publishing preparation.
pub fn validate_split_character_tables<E: Error + Send + Sync + 'static>(
    read: impl FnMut(&str, Range<usize>) -> Result<Vec<i64>, E>,
) -> Result<(), IntegerBufferError> {
    validate_integer_buffers(&VOICECHAT_SPLIT_CHARACTER_BUFFERS, read)
}

/// Validates the exact subset retained by a selected component plan.
pub fn validate_integer_buffers<E: Error + Send + Sync + 'static>(
    contracts: &[IntegerBufferContract],
    mut read: impl FnMut(&str, Range<usize>) -> Result<Vec<i64>, E>,
) -> Result<(), IntegerBufferError> {
    for contract in contracts {
        let mut hash = Sha256::new();
        for start in (0..contract.values).step_by(CONTROL_READ_VALUES) {
            let end = (start + CONTROL_READ_VALUES).min(contract.values);
            let values =
                read(contract.key, start..end).map_err(|source| IntegerBufferError::Read {
                    key: contract.key,
                    source: Box::new(source),
                })?;
            if values.len() != end - start {
                return Err(IntegerBufferError::RangeLength {
                    key: contract.key,
                    expected: end - start,
                    actual: values.len(),
                });
            }
            for value in values {
                hash.update(value.to_le_bytes());
            }
        }
        let actual_sha256 = hash
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if actual_sha256 != contract.sha256 {
            return Err(IntegerBufferError::ValueMismatch {
                key: contract.key,
                actual_sha256,
            });
        }
    }
    Ok(())
}

// Retain exactly the bytes whose semantics were admitted, in the same bounded
// read pass. Construction must not reread a potentially changed source buffer.
pub(super) fn retain_integer_buffers<E: Error + Send + Sync + 'static>(
    contracts: &[IntegerBufferContract],
    mut read: impl FnMut(&str, Range<usize>) -> Result<Vec<i64>, E>,
) -> Result<std::collections::BTreeMap<String, std::sync::Arc<[i64]>>, IntegerBufferError> {
    let mut retained = std::collections::BTreeMap::<String, Vec<i64>>::new();
    validate_integer_buffers(contracts, |key, range| {
        let values = read(key, range)?;
        retained
            .entry(key.to_owned())
            .or_default()
            .extend_from_slice(&values);
        Ok::<_, E>(values)
    })?;
    Ok(retained
        .into_iter()
        .map(|(key, values)| (key, values.into()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_shape_control_values_must_preserve_code_order() {
        let error = validate_control_tensors::<std::io::Error>(|_, _| Ok(vec![1024, 1025, 1026]))
            .unwrap_err();
        assert!(matches!(
            error,
            IntegerBufferError::ValueMismatch {
                key: "tts_model._control_codes",
                ..
            }
        ));
    }

    #[test]
    fn released_special_flags_keep_base_eos_while_reads_remain_bounded() {
        let contract = VOICECHAT_INTEGER_BUFFERS[3];
        let mut reads = 0;
        validate_integer_buffers::<std::io::Error>(&[contract], |_, range| {
            reads += 1;
            assert!(range.len() <= CONTROL_READ_VALUES);
            Ok(range
                .map(|i| match i {
                    1 => 1,
                    12 => 2,
                    _ => 0,
                })
                .collect())
        })
        .unwrap();
        assert_eq!(reads, 32);
        let error = validate_integer_buffers::<std::io::Error>(&[contract], |_, range| {
            Ok(range
                .map(|i| match i {
                    1 => 1,
                    2 => 2,
                    _ => 0,
                })
                .collect())
        })
        .unwrap_err();
        assert!(matches!(error, IntegerBufferError::ValueMismatch { .. }));
    }

    #[test]
    fn truncated_source_ranges_fail_before_content_comparison() {
        let error =
            validate_control_tensors::<std::io::Error>(|_, _| Ok(vec![1026, 1025])).unwrap_err();
        assert!(matches!(
            error,
            IntegerBufferError::RangeLength {
                expected: 3,
                actual: 2,
                ..
            }
        ));
    }
}
