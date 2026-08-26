//! Backend-neutral ownership of prepared multimodal tensors.

use std::collections::BTreeMap;

use eredu_core::{
    InputExtent, InputMetadataKey, InputModality, InputPartDescriptor, InputPayloadKind,
    InputTensorIdentity, PreparedInputError, PreparedInputIdentity,
};

/// Primary tensor and its semantic role for one prepared input part.
#[derive(Debug, Clone, Eq, PartialEq)]
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
    fn owns_ordered_parts_and_round_trips_wire_values_without_backend_types() {
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
}
