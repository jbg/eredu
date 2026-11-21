//! Prompt-cache identity for prepared multimodal input.

use eredu_core::{PreparedInputError, PreparedInputIdentity};
use sha2::{Digest, Sha256};

/// Cache identity coupling an ordered prepared-input description to caller-owned content.
///
/// [`PreparedInputIdentity`] deliberately excludes tensor payload bytes. Prompt caches need
/// both that stable description and a digest of the semantic content that produced the native
/// tensors, so equal shapes alone can never make two media requests cache-equivalent.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedInputCacheIdentity {
    prepared: PreparedInputIdentity,
    semantic_content_fingerprint: String,
    prefix_content_fingerprint: String,
}

impl PreparedInputCacheIdentity {
    /// Couples one prepared-input description to a nonempty semantic-content fingerprint.
    pub fn new(
        prepared: PreparedInputIdentity,
        semantic_content_fingerprint: impl Into<String>,
    ) -> Result<Self, PreparedInputCacheIdentityError> {
        let semantic_content_fingerprint = semantic_content_fingerprint.into();
        if semantic_content_fingerprint.trim().is_empty() {
            return Err(PreparedInputCacheIdentityError::EmptySemanticContent);
        }
        let words = prepared
            .encode_words()
            .map_err(PreparedInputCacheIdentityError::Prepared)?;
        let mut digest = Sha256::new();
        digest.update(b"eredu-prepared-input-cache-v1\0");
        digest.update((words.len() as u64).to_le_bytes());
        for word in words {
            digest.update(word.to_le_bytes());
        }
        digest.update((semantic_content_fingerprint.len() as u64).to_le_bytes());
        digest.update(semantic_content_fingerprint.as_bytes());
        let prefix_content_fingerprint = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(Self {
            prepared,
            semantic_content_fingerprint,
            prefix_content_fingerprint,
        })
    }

    /// Exact ordered payload-free prepared-input description.
    pub const fn prepared(&self) -> &PreparedInputIdentity {
        &self.prepared
    }

    /// Caller-owned digest identifying the semantic tensor payloads.
    pub fn semantic_content_fingerprint(&self) -> &str {
        &self.semantic_content_fingerprint
    }

    /// Canonical fingerprint stored in [`eredu_core::cache::PromptCacheDescriptor`].
    pub fn prefix_content_fingerprint(&self) -> &str {
        &self.prefix_content_fingerprint
    }

    /// Logical host storage retained by a copied identity, including descriptions
    /// and both fingerprints. Allocator capacity/overhead is excluded.
    pub fn logical_metadata_bytes(&self) -> Option<u64> {
        u64::try_from(std::mem::size_of::<Self>())
            .ok()?
            .checked_add(self.prepared.logical_metadata_bytes()?)?
            .checked_add(u64::try_from(self.semantic_content_fingerprint.len()).ok()?)?
            .checked_add(u64::try_from(self.prefix_content_fingerprint.len()).ok()?)
    }
}

/// Invalid cache identity for a prepared model input.
#[derive(Debug, thiserror::Error)]
pub enum PreparedInputCacheIdentityError {
    /// Semantic tensor content was not identified by the caller or processor.
    #[error("prepared-input semantic content fingerprint must not be empty")]
    EmptySemanticContent,
    /// The ordered prepared-input description could not be encoded canonically.
    #[error("prepared-input cache identity is invalid: {0}")]
    Prepared(PreparedInputError),
}

#[cfg(test)]
mod tests {
    use eredu_core::{
        checkpoint::TensorDtype, input::PreparedInputTensorRole, InputExtent, InputMetadataKey,
        InputModality, InputPayloadKind, InputTensorIdentity, PreparedInputPart,
        PreparedInputPayload, PreparedModelInput,
    };

    use super::*;

    #[derive(Debug, Clone, Eq, PartialEq)]
    struct FakeTensor {
        dtype: TensorDtype,
        shape: Vec<usize>,
        marker: u8,
    }

    fn fake(dtype: TensorDtype, shape: &[usize], marker: u8) -> FakeTensor {
        FakeTensor {
            dtype,
            shape: shape.to_vec(),
            marker,
        }
    }

    fn describe(value: &FakeTensor) -> Result<InputTensorIdentity, PreparedInputError> {
        InputTensorIdentity::new(value.dtype.clone(), value.shape.clone())
    }

    #[test]
    fn storage_roles_borrow_every_payload_and_metadata_in_wire_order() {
        let input = PreparedModelInput::new(
            vec![
                PreparedInputPart::new(
                    InputModality::Text,
                    PreparedInputPayload::TokenIds(fake(TensorDtype::U32, &[1, 2], 1)),
                    [],
                )
                .unwrap(),
                PreparedInputPart::new(
                    InputModality::Image,
                    PreparedInputPayload::Tensor(fake(TensorDtype::F32, &[4, 12], 2)),
                    [
                        (
                            InputMetadataKey::PatchPositions,
                            fake(TensorDtype::I32, &[4, 3], 4),
                        ),
                        (
                            InputMetadataKey::PatchGrid,
                            fake(TensorDtype::I32, &[1, 3], 3),
                        ),
                    ],
                )
                .unwrap(),
            ],
            describe,
        )
        .unwrap();
        let expected = [
            (PreparedInputTensorRole::Payload { part: 0 }, 1),
            (PreparedInputTensorRole::Payload { part: 1 }, 2),
            (
                PreparedInputTensorRole::Metadata {
                    part: 1,
                    key: InputMetadataKey::PatchGrid,
                },
                3,
            ),
            (
                PreparedInputTensorRole::Metadata {
                    part: 1,
                    key: InputMetadataKey::PatchPositions,
                },
                4,
            ),
        ];
        let actual = input
            .storage_values()
            .map(|(role, value)| (role, value.marker))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        let wire = input.wire_values();
        for ((_, value), wire_value) in input.storage_values().zip(&wire) {
            assert!(std::ptr::eq(value, *wire_value));
        }
        assert!(std::ptr::eq(wire[1], input.parts()[1].payload().value()));
        assert!(std::ptr::eq(
            wire[2],
            input.parts()[1]
                .metadata_value(InputMetadataKey::PatchGrid)
                .unwrap()
        ));
    }

    #[test]
    fn composite_input_extension_binds_typed_parts_to_a_multi_group_graph() {
        let graph = crate::ExecutionGraph::new(
            vec![
                crate::ExecutionGroupSpec::root("vision"),
                crate::ExecutionGroupSpec::with_dependencies("text", ["vision"]),
            ],
            "text",
        )
        .unwrap();
        let input = PreparedModelInput::new(
            vec![
                PreparedInputPart::new(
                    InputModality::Text,
                    PreparedInputPayload::TokenIds(fake(TensorDtype::U32, &[1, 2], 1)),
                    [],
                )
                .unwrap(),
                PreparedInputPart::new_with_extents(
                    InputModality::Image,
                    PreparedInputPayload::Tensor(fake(TensorDtype::F32, &[4, 12], 2)),
                    [(
                        InputMetadataKey::PatchGrid,
                        fake(TensorDtype::I32, &[1, 3], 3),
                    )],
                    [InputExtent::PatchGrid {
                        time: 1,
                        height: 2,
                        width: 2,
                    }],
                )
                .unwrap(),
            ],
            describe,
        )
        .unwrap();
        let identity = input.identity().clone();
        let values = input.wire_values().into_iter().cloned().collect();

        let rebuilt =
            PreparedModelInput::from_identity_wire_values(identity, values, describe).unwrap();
        assert_eq!(rebuilt, input);
        assert_eq!(graph.execution_order(), [0, 1]);
        assert_eq!(graph.output(), 1);
        assert_eq!(rebuilt.wire_values()[2].marker, 3);
        assert_eq!(
            rebuilt.parts()[1].extents(),
            &[InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }]
        );
    }

    #[test]
    fn rejects_payload_geometry_that_disagrees_with_wire_identity() {
        let input = PreparedModelInput::new(
            vec![PreparedInputPart::new(
                InputModality::Text,
                PreparedInputPayload::TokenIds(fake(TensorDtype::U32, &[1, 2], 1)),
                [],
            )
            .unwrap()],
            describe,
        )
        .unwrap();
        let wrong = vec![fake(TensorDtype::U32, &[1, 3], 1)];

        assert!(matches!(
            PreparedModelInput::from_identity_wire_values(
                input.identity().clone(),
                wrong,
                describe
            ),
            Err(PreparedInputError::WireIdentityMismatch)
        ));
    }

    #[test]
    fn rejects_incompatible_payload_at_part_construction() {
        let result = PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::Tensor(fake(TensorDtype::F32, &[1, 2], 1)),
            [],
        );

        assert!(matches!(
            result,
            Err(PreparedInputError::IncompatiblePayload {
                modality: InputModality::Text,
                payload: InputPayloadKind::Tensor,
            })
        ));
    }

    #[test]
    fn rejects_incompatible_metadata_at_part_construction() {
        let result = PreparedInputPart::new(
            InputModality::Text,
            PreparedInputPayload::TokenIds(fake(TensorDtype::U32, &[1, 2], 1)),
            [(
                InputMetadataKey::PatchGrid,
                fake(TensorDtype::I32, &[1, 3], 2),
            )],
        );

        assert!(matches!(
            result,
            Err(PreparedInputError::IncompatibleMetadata {
                modality: InputModality::Text,
                key: InputMetadataKey::PatchGrid,
            })
        ));
    }

    #[test]
    fn prompt_cache_identity_requires_both_prepared_description_and_semantic_content() {
        let first = PreparedModelInput::new(
            vec![PreparedInputPart::new(
                InputModality::Image,
                PreparedInputPayload::Tensor(fake(TensorDtype::F32, &[1, 3, 4, 4], 1)),
                [],
            )
            .unwrap()],
            describe,
        )
        .unwrap();
        let reshaped = PreparedModelInput::new(
            vec![PreparedInputPart::new(
                InputModality::Image,
                PreparedInputPayload::Tensor(fake(TensorDtype::F32, &[1, 3, 8, 8], 2)),
                [],
            )
            .unwrap()],
            describe,
        )
        .unwrap();

        let image_a =
            PreparedInputCacheIdentity::new(first.identity().clone(), "sha256:image-a").unwrap();
        let image_b =
            PreparedInputCacheIdentity::new(first.identity().clone(), "sha256:image-b").unwrap();
        let reshaped_a =
            PreparedInputCacheIdentity::new(reshaped.identity().clone(), "sha256:image-a").unwrap();

        assert_ne!(
            image_a.prefix_content_fingerprint(),
            image_b.prefix_content_fingerprint()
        );
        assert_ne!(
            image_a.prefix_content_fingerprint(),
            reshaped_a.prefix_content_fingerprint()
        );
        assert_eq!(image_a.prepared(), first.identity());
        assert!(PreparedInputCacheIdentity::new(first.identity().clone(), " ").is_err());
    }
}
