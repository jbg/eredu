//! Backend-neutral ownership of prepared multimodal tensors.

use std::collections::BTreeMap;

use eredu_core::{
    CapabilityError, InputExtent, InputMetadataKey, InputModality, InputPartDescriptor,
    InputPayloadKind, InputTensorIdentity, PreparedInputError, PreparedInputIdentity,
};
use sha2::{Digest, Sha256};

/// Mechanism for describing native prepared tensors and reading bounded metadata.
///
/// Architecture admission owns which metadata values are semantically relevant.
/// Implementations expose only tensor identity and evaluation of the small integer
/// or Boolean arrays requested by that admission logic.
pub trait PreparedInputInspector<Tensor> {
    /// Returns the portable identity of a native tensor.
    fn identity(&self, tensor: &Tensor) -> Result<InputTensorIdentity, PreparedInputError>;

    /// Reads an evaluated signed-integer metadata tensor in row-major order.
    fn i32_values(&self, tensor: &Tensor) -> Result<Vec<i32>, CapabilityError>;

    /// Reads an evaluated Boolean metadata tensor in row-major order.
    fn bool_values(&self, tensor: &Tensor) -> Result<Vec<bool>, CapabilityError>;
}

/// Primary tensor and its semantic role for one prepared input part.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum PreparedInputPayload<Tensor> {
    /// Tokenizer vocabulary IDs.
    TokenIds(Tensor),
    /// Model-native features or patches that still require an encoder.
    Tensor(Tensor),
    /// Already projected decoder-width embeddings.
    Embeddings(Tensor),
}

impl<Tensor> PreparedInputPayload<Tensor> {
    /// Semantic payload kind.
    pub const fn kind(&self) -> InputPayloadKind {
        match self {
            Self::TokenIds(_) => InputPayloadKind::TokenIds,
            Self::Tensor(_) => InputPayloadKind::Tensor,
            Self::Embeddings(_) => InputPayloadKind::Embeddings,
        }
    }

    /// Borrows the backend-native tensor.
    pub const fn value(&self) -> &Tensor {
        match self {
            Self::TokenIds(value) | Self::Tensor(value) | Self::Embeddings(value) => value,
        }
    }
}

/// One owned, typed, ordered prepared input part.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedInputPart<Tensor> {
    modality: InputModality,
    payload: PreparedInputPayload<Tensor>,
    metadata: BTreeMap<InputMetadataKey, Tensor>,
    extents: Vec<InputExtent>,
}

impl<Tensor> PreparedInputPart<Tensor> {
    /// Creates a part with compatible payload and unique, compatible metadata.
    pub fn new(
        modality: InputModality,
        payload: PreparedInputPayload<Tensor>,
        metadata: impl IntoIterator<Item = (InputMetadataKey, Tensor)>,
    ) -> Result<Self, PreparedInputError> {
        Self::new_with_extents(modality, payload, metadata, [])
    }

    /// Creates a part with compatible host-known execution extents.
    pub fn new_with_extents(
        modality: InputModality,
        payload: PreparedInputPayload<Tensor>,
        metadata: impl IntoIterator<Item = (InputMetadataKey, Tensor)>,
        extents: impl IntoIterator<Item = InputExtent>,
    ) -> Result<Self, PreparedInputError> {
        let payload_kind = payload.kind();
        if !payload_kind.accepts(modality) {
            return Err(PreparedInputError::IncompatiblePayload {
                modality,
                payload: payload_kind,
            });
        }
        let mut typed_metadata = BTreeMap::new();
        for (key, value) in metadata {
            if !key.accepts(modality) {
                return Err(PreparedInputError::IncompatibleMetadata { modality, key });
            }
            if typed_metadata.insert(key, value).is_some() {
                return Err(PreparedInputError::DuplicateMetadata { key });
            }
        }
        let extents = extents.into_iter().collect::<Vec<_>>();
        for (index, extent) in extents.iter().copied().enumerate() {
            if !extent.accepts(modality) {
                return Err(PreparedInputError::IncompatibleExtent { modality, extent });
            }
            if extents[..index]
                .iter()
                .any(|prior| std::mem::discriminant(prior) == std::mem::discriminant(&extent))
            {
                return Err(PreparedInputError::DuplicateExtent { extent });
            }
        }
        Ok(Self {
            modality,
            payload,
            metadata: typed_metadata,
            extents,
        })
    }

    /// Part modality.
    pub const fn modality(&self) -> InputModality {
        self.modality
    }

    /// Primary tensor and semantic role.
    pub const fn payload(&self) -> &PreparedInputPayload<Tensor> {
        &self.payload
    }

    /// Typed metadata tensors in stable key order.
    pub const fn metadata(&self) -> &BTreeMap<InputMetadataKey, Tensor> {
        &self.metadata
    }

    /// Looks up one metadata tensor.
    pub fn metadata_value(&self, key: InputMetadataKey) -> Option<&Tensor> {
        self.metadata.get(&key)
    }

    /// Host-known extents needed by accelerator execution.
    pub fn extents(&self) -> &[InputExtent] {
        &self.extents
    }

    /// Builds and validates the core descriptor for this exact tensor part.
    pub fn descriptor(
        &self,
        describe: &impl Fn(&Tensor) -> Result<InputTensorIdentity, PreparedInputError>,
    ) -> Result<InputPartDescriptor, PreparedInputError> {
        InputPartDescriptor::new_with_extents(
            self.modality,
            self.payload.kind(),
            describe(self.payload.value())?,
            self.metadata
                .iter()
                .map(|(key, value)| Ok((*key, describe(value)?)))
                .collect::<Result<Vec<_>, PreparedInputError>>()?,
            self.extents.iter().copied(),
        )
    }
}

/// Backend-neutral prepared input that owns backend-native tensor handles.
///
/// The identity is validated at construction and remains coupled to the exact
/// ordered payload and metadata values used by runtime and distributed paths.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedModelInput<Tensor> {
    parts: Vec<PreparedInputPart<Tensor>>,
    identity: PreparedInputIdentity,
}

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

impl<Tensor> PreparedModelInput<Tensor> {
    /// Validates and owns ordered prepared input parts.
    pub fn new(
        parts: Vec<PreparedInputPart<Tensor>>,
        describe: impl Fn(&Tensor) -> Result<InputTensorIdentity, PreparedInputError>,
    ) -> Result<Self, PreparedInputError> {
        let identity = PreparedInputIdentity::new(
            parts
                .iter()
                .map(|part| part.descriptor(&describe))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        Ok(Self { parts, identity })
    }

    /// Exact payload-free identity used for rank agreement and persistence.
    pub const fn identity(&self) -> &PreparedInputIdentity {
        &self.identity
    }

    /// Derives the canonical prompt-cache content identity for these exact prepared parts.
    pub fn cache_identity(
        &self,
        semantic_content_fingerprint: impl Into<String>,
    ) -> Result<PreparedInputCacheIdentity, PreparedInputCacheIdentityError> {
        PreparedInputCacheIdentity::new(self.identity.clone(), semantic_content_fingerprint)
    }

    /// Ordered owned parts.
    pub fn parts(&self) -> &[PreparedInputPart<Tensor>] {
        &self.parts
    }

    /// Number of ordered parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// This input is always non-empty after construction.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Borrows payloads and metadata tensors in deterministic wire order.
    pub fn wire_values(&self) -> Vec<&Tensor> {
        let mut values = Vec::new();
        for part in &self.parts {
            values.push(part.payload.value());
            values.extend(part.metadata.values());
        }
        values
    }

    /// Reconstructs and validates input received in deterministic wire order.
    pub fn from_identity_wire_values(
        identity: PreparedInputIdentity,
        values: Vec<Tensor>,
        describe: impl Fn(&Tensor) -> Result<InputTensorIdentity, PreparedInputError>,
    ) -> Result<Self, PreparedInputError> {
        let expected_values = identity
            .parts()
            .iter()
            .map(|part| 1 + part.metadata().len())
            .sum::<usize>();
        if values.len() != expected_values {
            return Err(PreparedInputError::WireValueCount {
                expected: expected_values,
                actual: values.len(),
            });
        }
        let mut values = values.into_iter();
        let mut parts = Vec::with_capacity(identity.len());
        for descriptor in identity.parts() {
            let payload = values.next().expect("validated prepared-input value count");
            let payload = match descriptor.payload_kind() {
                InputPayloadKind::TokenIds => PreparedInputPayload::TokenIds(payload),
                InputPayloadKind::Tensor => PreparedInputPayload::Tensor(payload),
                InputPayloadKind::Embeddings => PreparedInputPayload::Embeddings(payload),
                payload_kind => {
                    return Err(PreparedInputError::IncompatiblePayload {
                        modality: descriptor.modality(),
                        payload: payload_kind,
                    });
                }
            };
            let metadata = descriptor
                .metadata()
                .keys()
                .copied()
                .map(|key| {
                    (
                        key,
                        values.next().expect("validated prepared-input value count"),
                    )
                })
                .collect::<Vec<_>>();
            parts.push(PreparedInputPart::new_with_extents(
                descriptor.modality(),
                payload,
                metadata,
                descriptor.extents(),
            )?);
        }
        let actual = Self::new(parts, describe)?;
        if actual.identity != identity {
            return Err(PreparedInputError::WireIdentityMismatch);
        }
        Ok(actual)
    }

    /// Consumes the lifecycle container and returns its ordered parts.
    pub fn into_parts(self) -> Vec<PreparedInputPart<Tensor>> {
        self.parts
    }
}

#[cfg(test)]
mod tests {
    use eredu_core::{checkpoint::TensorDtype, PreparedInputError};

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

        let image_a = first.cache_identity("sha256:image-a").unwrap();
        let image_b = first.cache_identity("sha256:image-b").unwrap();
        let reshaped_a = reshaped.cache_identity("sha256:image-a").unwrap();

        assert_ne!(
            image_a.prefix_content_fingerprint(),
            image_b.prefix_content_fingerprint()
        );
        assert_ne!(
            image_a.prefix_content_fingerprint(),
            reshaped_a.prefix_content_fingerprint()
        );
        assert_eq!(image_a.prepared(), first.identity());
        assert!(first.cache_identity(" ").is_err());
    }
}
