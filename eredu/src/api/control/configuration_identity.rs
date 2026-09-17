//! Keep the existing serialized configuration identity without a JSON buffer.
use serde::Serialize;
use sha2::{Digest, Sha256};

struct DigestWriter(Sha256);
impl std::io::Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn digest(value: &impl Serialize) -> Result<[u8; 32], serde_json::Error> {
    let mut writer = DigestWriter(Sha256::new());
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.0.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::control::OutputMode;
    use eredu_core::TextInferencePolicy;

    #[test]
    fn streamed_identity_preserves_legacy_bytes_for_controlled_policy_and_text() {
        let long_stop = "é\n\"\\🙂".repeat(8192);
        for mode in [OutputMode::Text, OutputMode::Semantic] {
            for capacity in [None, Some(0), Some(16 << 20), Some(u64::MAX)] {
                let input = (
                    2_u32,
                    mode,
                    [0xabu8; 32],
                    (0.75_f64, 0.9_f32, Some(128_usize)),
                    u64::MAX,
                    TextInferencePolicy {
                        prefill_chunk_positions: std::num::NonZeroU64::new(3),
                        managed_memory_capacity_bytes: capacity,
                        ..Default::default()
                    },
                    Some(("profile", "constraint", "auto", ["", "\0", "é🙂"])),
                    [0_u32, 123, u32::MAX],
                    ["", long_stop.as_str()],
                );
                // The pre-change representation is the compatibility oracle.
                let prior: [u8; 32] = Sha256::digest(serde_json::to_vec(&input).unwrap()).into();
                assert_eq!(digest(&input).unwrap(), prior);
            }
        }
    }

    #[test]
    fn streamed_identity_preserves_serialization_failure() {
        struct Refusal;
        impl Serialize for Refusal {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("configuration source refused"))
            }
        }
        let prior = serde_json::to_vec(&("prefix", Refusal)).unwrap_err();
        let actual = digest(&("prefix", Refusal)).unwrap_err();
        assert_eq!(actual.classify(), prior.classify());
        assert_eq!(actual.to_string(), prior.to_string());
    }
}
