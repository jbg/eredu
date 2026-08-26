//! Portable identity for ordered, prepared model input.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::checkpoint::TensorDtype;

/// Modality of one ordered model-input part.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputModality {
    /// Text token IDs or precomputed text embeddings.
    Text,
    /// Still-image patches or precomputed image embeddings.
    Image,
    /// Video patches or precomputed video embeddings.
    Video,
    /// Audio features or precomputed audio embeddings.
    Audio,
}

impl InputModality {
    /// Stable lowercase diagnostic name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Video => "video",
            Self::Audio => "audio",
        }
    }

    const fn wire_tag(self) -> u32 {
        match self {
            Self::Text => 0,
            Self::Image => 1,
            Self::Video => 2,
            Self::Audio => 3,
        }
    }

    fn from_wire_tag(tag: u32) -> Result<Self, PreparedInputError> {
        match tag {
            0 => Ok(Self::Text),
            1 => Ok(Self::Image),
            2 => Ok(Self::Video),
            3 => Ok(Self::Audio),
            _ => Err(PreparedInputError::InvalidWireValue {
                field: "modality",
                value: tag,
            }),
        }
    }
}

/// Semantic role of a prepared part's primary tensor.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputPayloadKind {
    /// Tokenizer vocabulary IDs.
    TokenIds,
    /// Model-native media features or patches that still require an encoder.
    Tensor,
    /// Already projected decoder-width embeddings.
    Embeddings,
}

impl InputPayloadKind {
    /// Returns whether this payload role is meaningful for `modality`.
    pub const fn accepts(self, modality: InputModality) -> bool {
        match self {
            Self::TokenIds => matches!(modality, InputModality::Text),
            Self::Tensor => !matches!(modality, InputModality::Text),
            Self::Embeddings => true,
        }
    }

    const fn wire_tag(self) -> u32 {
        match self {
            Self::TokenIds => 0,
            Self::Tensor => 1,
            Self::Embeddings => 2,
        }
    }

    fn from_wire_tag(tag: u32) -> Result<Self, PreparedInputError> {
        match tag {
            0 => Ok(Self::TokenIds),
            1 => Ok(Self::Tensor),
            2 => Ok(Self::Embeddings),
            _ => Err(PreparedInputError::InvalidWireValue {
                field: "payload kind",
                value: tag,
            }),
        }
    }
}

/// Architecture-neutral metadata attached to a prepared input part.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMetadataKey {
    /// One or more `(time, height, width)` patch-grid rows.
    PatchGrid,
    /// Explicit spatial or temporal patch coordinates, including padding.
    PatchPositions,
    /// Valid-frame or valid-feature mask for audio.
    AudioMask,
}

/// Host-known extent needed to plan device execution without reading a tensor
/// back from the accelerator.
#[derive(Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputExtent {
    /// Exact `(time, height, width)` of one image or video patch grid.
    PatchGrid {
        /// Temporal extent (one for an independently packed frame).
        time: usize,
        /// Unpadded patch-row count.
        height: usize,
        /// Unpadded patch-column count.
        width: usize,
    },
    /// Number of valid (unpadded) input audio frames.
    AudioValidFrames(usize),
}

impl InputExtent {
    /// Returns whether this extent is meaningful for `modality`.
    pub const fn accepts(self, modality: InputModality) -> bool {
        match self {
            Self::PatchGrid { .. } => {
                matches!(modality, InputModality::Image | InputModality::Video)
            }
            Self::AudioValidFrames(_) => matches!(modality, InputModality::Audio),
        }
    }

    const fn wire_tag(self) -> u32 {
        match self {
            Self::PatchGrid { .. } => 0,
            Self::AudioValidFrames(_) => 1,
        }
    }

    const fn key(self) -> u32 {
        self.wire_tag()
    }

    fn encode_words(self, output: &mut Vec<u32>) -> Result<(), PreparedInputError> {
        output.push(self.wire_tag());
        let values: &[usize] = match &self {
            Self::PatchGrid {
                time,
                height,
                width,
            } => &[*time, *height, *width],
            Self::AudioValidFrames(frames) => &[*frames],
        };
        for value in values {
            output.push(
                u32::try_from(*value)
                    .map_err(|_| PreparedInputError::WireValueOverflow("input extent"))?,
            );
        }
        Ok(())
    }

    fn decode_words(cursor: &mut WordCursor<'_>) -> Result<Self, PreparedInputError> {
        match cursor.next("input extent")? {
            0 => Ok(Self::PatchGrid {
                time: cursor.usize("patch grid time")?,
                height: cursor.usize("patch grid height")?,
                width: cursor.usize("patch grid width")?,
            }),
            1 => Ok(Self::AudioValidFrames(cursor.usize("valid audio frames")?)),
            value => Err(PreparedInputError::InvalidWireValue {
                field: "input extent",
                value,
            }),
        }
    }
}

impl InputMetadataKey {
    /// Returns whether this metadata is meaningful for `modality`.
    pub const fn accepts(self, modality: InputModality) -> bool {
        match self {
            Self::PatchGrid | Self::PatchPositions => {
                matches!(modality, InputModality::Image | InputModality::Video)
            }
            Self::AudioMask => matches!(modality, InputModality::Audio),
        }
    }

    const fn wire_tag(self) -> u32 {
        match self {
            Self::PatchGrid => 0,
            Self::PatchPositions => 1,
            Self::AudioMask => 2,
        }
    }

    fn from_wire_tag(tag: u32) -> Result<Self, PreparedInputError> {
        match tag {
            0 => Ok(Self::PatchGrid),
            1 => Ok(Self::PatchPositions),
            2 => Ok(Self::AudioMask),
            _ => Err(PreparedInputError::InvalidWireValue {
                field: "metadata key",
                value: tag,
            }),
        }
    }
}

/// Payload-free shape and element-type identity for a backend tensor.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputTensorIdentity {
    dtype: TensorDtype,
    shape: Vec<usize>,
}

impl InputTensorIdentity {
    /// Validates a non-scalar tensor identity with non-zero dimensions.
    pub fn new(dtype: TensorDtype, shape: Vec<usize>) -> Result<Self, PreparedInputError> {
        if shape.is_empty() || shape.contains(&0) {
            return Err(PreparedInputError::InvalidTensorShape { shape });
        }
        Ok(Self { dtype, shape })
    }

    /// Logical element type.
    pub const fn dtype(&self) -> &TensorDtype {
        &self.dtype
    }

    /// Row-major logical shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn encode_words(&self, output: &mut Vec<u32>) -> Result<(), PreparedInputError> {
        encode_dtype(&self.dtype, output)?;
        output.push(
            u32::try_from(self.shape.len())
                .map_err(|_| PreparedInputError::WireValueOverflow("tensor rank"))?,
        );
        for dimension in &self.shape {
            output.push(
                u32::try_from(*dimension)
                    .map_err(|_| PreparedInputError::WireValueOverflow("tensor dimension"))?,
            );
        }
        Ok(())
    }

    fn decode_words(cursor: &mut WordCursor<'_>) -> Result<Self, PreparedInputError> {
        let dtype = decode_dtype(cursor)?;
        let rank = cursor.usize("tensor rank")?;
        if rank == 0 || rank > 8 {
            return Err(PreparedInputError::InvalidWireRank(rank));
        }
        let shape = (0..rank)
            .map(|_| cursor.usize("tensor dimension"))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(dtype, shape)
    }
}

/// Payload-free identity for one ordered prepared input part.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputPartDescriptor {
    modality: InputModality,
    payload_kind: InputPayloadKind,
    payload: InputTensorIdentity,
    metadata: BTreeMap<InputMetadataKey, InputTensorIdentity>,
    extents: BTreeMap<u32, InputExtent>,
}

impl InputPartDescriptor {
    /// Validates modality/payload compatibility and unique typed metadata.
    pub fn new(
        modality: InputModality,
        payload_kind: InputPayloadKind,
        payload: InputTensorIdentity,
        metadata: impl IntoIterator<Item = (InputMetadataKey, InputTensorIdentity)>,
    ) -> Result<Self, PreparedInputError> {
        Self::new_with_extents(modality, payload_kind, payload, metadata, [])
    }

    /// Validates tensor metadata plus host-known execution extents.
    pub fn new_with_extents(
        modality: InputModality,
        payload_kind: InputPayloadKind,
        payload: InputTensorIdentity,
        metadata: impl IntoIterator<Item = (InputMetadataKey, InputTensorIdentity)>,
        extents: impl IntoIterator<Item = InputExtent>,
    ) -> Result<Self, PreparedInputError> {
        if !payload_kind.accepts(modality) {
            return Err(PreparedInputError::IncompatiblePayload {
                modality,
                payload: payload_kind,
            });
        }
        let mut typed_metadata = BTreeMap::new();
        for (key, identity) in metadata {
            if !key.accepts(modality) {
                return Err(PreparedInputError::IncompatibleMetadata { modality, key });
            }
            if typed_metadata.insert(key, identity).is_some() {
                return Err(PreparedInputError::DuplicateMetadata { key });
            }
        }
        let mut typed_extents = BTreeMap::new();
        for extent in extents {
            if !extent.accepts(modality) {
                return Err(PreparedInputError::IncompatibleExtent { modality, extent });
            }
            if typed_extents.insert(extent.key(), extent).is_some() {
                return Err(PreparedInputError::DuplicateExtent { extent });
            }
        }
        Ok(Self {
            modality,
            payload_kind,
            payload,
            metadata: typed_metadata,
            extents: typed_extents,
        })
    }

    /// Part modality.
    pub const fn modality(&self) -> InputModality {
        self.modality
    }

    /// Primary tensor role.
    pub const fn payload_kind(&self) -> InputPayloadKind {
        self.payload_kind
    }

    /// Primary tensor identity.
    pub const fn payload(&self) -> &InputTensorIdentity {
        &self.payload
    }

    /// Typed metadata identities in stable key order.
    pub const fn metadata(&self) -> &BTreeMap<InputMetadataKey, InputTensorIdentity> {
        &self.metadata
    }

    /// Host-known execution extents in stable wire order.
    pub fn extents(&self) -> impl ExactSizeIterator<Item = InputExtent> + '_ {
        self.extents.values().copied()
    }

    /// Looks up one typed metadata identity.
    pub fn metadata_value(&self, key: InputMetadataKey) -> Option<&InputTensorIdentity> {
        self.metadata.get(&key)
    }

    /// Requires metadata selected by family policy.
    pub fn require_metadata(
        &self,
        part: usize,
        key: InputMetadataKey,
    ) -> Result<&InputTensorIdentity, PreparedInputError> {
        self.metadata
            .get(&key)
            .ok_or(PreparedInputError::MissingMetadata {
                part,
                modality: self.modality,
                key,
            })
    }
}

/// Exact payload-free identity of an ordered prepared model input.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct PreparedInputIdentity {
    parts: Vec<InputPartDescriptor>,
}

impl PreparedInputIdentity {
    /// Validates a non-empty ordered set of input-part descriptors.
    pub fn new(parts: Vec<InputPartDescriptor>) -> Result<Self, PreparedInputError> {
        if parts.is_empty() {
            return Err(PreparedInputError::EmptyInput);
        }
        Ok(Self { parts })
    }

    /// Ordered input-part descriptors.
    pub fn parts(&self) -> &[InputPartDescriptor] {
        &self.parts
    }

    /// Number of ordered parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// This identity is always non-empty after construction.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Encodes the identity for backend-independent rank agreement.
    pub fn encode_words(&self) -> Result<Vec<u32>, PreparedInputError> {
        let mut output = Vec::new();
        output.push(
            u32::try_from(self.parts.len())
                .map_err(|_| PreparedInputError::WireValueOverflow("part count"))?,
        );
        for part in &self.parts {
            output.extend_from_slice(&[part.modality.wire_tag(), part.payload_kind.wire_tag()]);
            part.payload.encode_words(&mut output)?;
            output.push(
                u32::try_from(part.metadata.len())
                    .map_err(|_| PreparedInputError::WireValueOverflow("metadata count"))?,
            );
            for (key, identity) in &part.metadata {
                output.push(key.wire_tag());
                identity.encode_words(&mut output)?;
            }
            output.push(
                u32::try_from(part.extents.len())
                    .map_err(|_| PreparedInputError::WireValueOverflow("extent count"))?,
            );
            for extent in part.extents.values().copied() {
                extent.encode_words(&mut output)?;
            }
        }
        Ok(output)
    }

    /// Decodes and validates a rank-agreement descriptor.
    pub fn decode_words(words: &[u32]) -> Result<Self, PreparedInputError> {
        let mut cursor = WordCursor { words, offset: 0 };
        let part_count = cursor.usize("part count")?;
        if part_count == 0 {
            return Err(PreparedInputError::EmptyInput);
        }
        let mut parts = Vec::with_capacity(part_count);
        for _ in 0..part_count {
            let modality = InputModality::from_wire_tag(cursor.next("modality")?)?;
            let payload_kind = InputPayloadKind::from_wire_tag(cursor.next("payload kind")?)?;
            let payload = InputTensorIdentity::decode_words(&mut cursor)?;
            let metadata_count = cursor.usize("metadata count")?;
            if metadata_count > 3 {
                return Err(PreparedInputError::InvalidMetadataCount(metadata_count));
            }
            let metadata = (0..metadata_count)
                .map(|_| {
                    let key = InputMetadataKey::from_wire_tag(cursor.next("metadata key")?)?;
                    Ok((key, InputTensorIdentity::decode_words(&mut cursor)?))
                })
                .collect::<Result<Vec<_>, PreparedInputError>>()?;
            let extent_count = cursor.usize("extent count")?;
            if extent_count > 2 {
                return Err(PreparedInputError::InvalidExtentCount(extent_count));
            }
            let extents = (0..extent_count)
                .map(|_| InputExtent::decode_words(&mut cursor))
                .collect::<Result<Vec<_>, _>>()?;
            parts.push(InputPartDescriptor::new_with_extents(
                modality,
                payload_kind,
                payload,
                metadata,
                extents,
            )?);
        }
        if cursor.offset != words.len() {
            return Err(PreparedInputError::TrailingWireValues);
        }
        Self::new(parts)
    }
}

/// Invalid portable prepared-input identity.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum PreparedInputError {
    /// At least one ordered part is required.
    #[error("prepared model input must contain at least one part")]
    EmptyInput,
    /// Tensor identities must be non-scalar and have non-zero dimensions.
    #[error("prepared input tensor has invalid shape {shape:?}")]
    InvalidTensorShape {
        /// Invalid logical dimensions.
        shape: Vec<usize>,
    },
    /// A modality cannot consume the selected primary payload kind.
    #[error("{modality:?} input is incompatible with {payload:?} payload")]
    IncompatiblePayload {
        /// Declared modality.
        modality: InputModality,
        /// Declared payload kind.
        payload: InputPayloadKind,
    },
    /// Metadata belongs to a different modality.
    #[error("{key:?} metadata is incompatible with {modality:?} input")]
    IncompatibleMetadata {
        /// Declared modality.
        modality: InputModality,
        /// Metadata key.
        key: InputMetadataKey,
    },
    /// A typed metadata key occurred more than once.
    #[error("prepared input contains duplicate {key:?} metadata")]
    DuplicateMetadata {
        /// Duplicated key.
        key: InputMetadataKey,
    },
    /// A host execution extent belongs to a different modality.
    #[error("{extent:?} extent is incompatible with {modality:?} input")]
    IncompatibleExtent {
        /// Declared modality.
        modality: InputModality,
        /// Incompatible extent.
        extent: InputExtent,
    },
    /// An execution extent kind occurred more than once.
    #[error("prepared input contains duplicate {extent:?} extent")]
    DuplicateExtent {
        /// Duplicated extent.
        extent: InputExtent,
    },
    /// Family policy required absent metadata.
    #[error("prepared input part {part} ({modality:?}) is missing {key:?} metadata")]
    MissingMetadata {
        /// Ordered part index.
        part: usize,
        /// Part modality.
        modality: InputModality,
        /// Required key.
        key: InputMetadataKey,
    },
    /// A descriptor tag was not recognized.
    #[error("prepared-input descriptor has invalid {field} value {value}")]
    InvalidWireValue {
        /// Descriptor field.
        field: &'static str,
        /// Invalid value.
        value: u32,
    },
    /// Descriptor ended before the announced data was present.
    #[error("prepared-input descriptor ended while reading {0}")]
    TruncatedWireDescriptor(&'static str),
    /// Descriptor carried an unsupported tensor rank.
    #[error("prepared-input descriptor tensor rank {0} is outside 1..=8")]
    InvalidWireRank(usize),
    /// A part advertised more metadata keys than the closed vocabulary contains.
    #[error("prepared-input descriptor metadata count {0} exceeds 3")]
    InvalidMetadataCount(usize),
    /// A part advertised more host extents than the closed vocabulary contains.
    #[error("prepared-input descriptor extent count {0} exceeds 2")]
    InvalidExtentCount(usize),
    /// Descriptor has data after the complete identity.
    #[error("prepared-input descriptor has trailing values")]
    TrailingWireValues,
    /// A host value cannot be represented by the u32 wire format.
    #[error("prepared-input {0} exceeds descriptor range")]
    WireValueOverflow(&'static str),
    /// The number of received tensors does not match the identity.
    #[error("prepared-input wire payload has {actual} values; expected {expected}")]
    WireValueCount {
        /// Expected payload and metadata tensor count.
        expected: usize,
        /// Received tensor count.
        actual: usize,
    },
    /// Received tensor shape or dtype disagrees with the advertised identity.
    #[error("prepared-input wire payload does not match its identity")]
    WireIdentityMismatch,
    /// Encoded/quantized checkpoint storage is not a runtime tensor dtype.
    #[error("encoded dtype {0:?} cannot identify a prepared runtime tensor")]
    EncodedRuntimeDtype(String),
    /// A backend could not describe a prepared tensor without materializing it.
    #[error("backend prepared-tensor identity failed: {0}")]
    BackendTensorIdentity(String),
}

struct WordCursor<'a> {
    words: &'a [u32],
    offset: usize,
}

impl WordCursor<'_> {
    fn next(&mut self, field: &'static str) -> Result<u32, PreparedInputError> {
        let value = self
            .words
            .get(self.offset)
            .copied()
            .ok_or(PreparedInputError::TruncatedWireDescriptor(field))?;
        self.offset += 1;
        Ok(value)
    }

    fn usize(&mut self, field: &'static str) -> Result<usize, PreparedInputError> {
        usize::try_from(self.next(field)?).map_err(|_| PreparedInputError::WireValueOverflow(field))
    }
}

fn encode_dtype(dtype: &TensorDtype, output: &mut Vec<u32>) -> Result<(), PreparedInputError> {
    let tag = match dtype {
        TensorDtype::Bool => 0,
        TensorDtype::U8 => 1,
        TensorDtype::U16 => 2,
        TensorDtype::U32 => 3,
        TensorDtype::U64 => 4,
        TensorDtype::I8 => 5,
        TensorDtype::I16 => 6,
        TensorDtype::I32 => 7,
        TensorDtype::I64 => 8,
        TensorDtype::F16 => 9,
        TensorDtype::F32 => 10,
        TensorDtype::F64 => 11,
        TensorDtype::Bf16 => 12,
        TensorDtype::Complex64 => 13,
        TensorDtype::Encoded(name) => {
            return Err(PreparedInputError::EncodedRuntimeDtype(name.clone()))
        }
    };
    output.push(tag);
    Ok(())
}

fn decode_dtype(cursor: &mut WordCursor<'_>) -> Result<TensorDtype, PreparedInputError> {
    let tag = cursor.next("dtype")?;
    match tag {
        0 => Ok(TensorDtype::Bool),
        1 => Ok(TensorDtype::U8),
        2 => Ok(TensorDtype::U16),
        3 => Ok(TensorDtype::U32),
        4 => Ok(TensorDtype::U64),
        5 => Ok(TensorDtype::I8),
        6 => Ok(TensorDtype::I16),
        7 => Ok(TensorDtype::I32),
        8 => Ok(TensorDtype::I64),
        9 => Ok(TensorDtype::F16),
        10 => Ok(TensorDtype::F32),
        11 => Ok(TensorDtype::F64),
        12 => Ok(TensorDtype::Bf16),
        13 => Ok(TensorDtype::Complex64),
        value => Err(PreparedInputError::InvalidWireValue {
            field: "dtype",
            value,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(dtype: TensorDtype, shape: &[usize]) -> InputTensorIdentity {
        InputTensorIdentity::new(dtype, shape.to_vec()).unwrap()
    }

    #[test]
    fn identity_round_trip_preserves_order_geometry_and_typed_metadata() {
        let identity = PreparedInputIdentity::new(vec![
            InputPartDescriptor::new(
                InputModality::Text,
                InputPayloadKind::TokenIds,
                tensor(TensorDtype::U32, &[1, 2]),
                [],
            )
            .unwrap(),
            InputPartDescriptor::new_with_extents(
                InputModality::Image,
                InputPayloadKind::Tensor,
                tensor(TensorDtype::F32, &[4, 12]),
                [(
                    InputMetadataKey::PatchGrid,
                    tensor(TensorDtype::I32, &[1, 3]),
                )],
                [InputExtent::PatchGrid {
                    time: 1,
                    height: 2,
                    width: 2,
                }],
            )
            .unwrap(),
        ])
        .unwrap();

        let words = identity.encode_words().unwrap();
        assert_eq!(
            PreparedInputIdentity::decode_words(&words).unwrap(),
            identity
        );
        assert_eq!(
            identity.parts()[1].extents().collect::<Vec<_>>(),
            [InputExtent::PatchGrid {
                time: 1,
                height: 2,
                width: 2,
            }]
        );
    }

    #[test]
    fn rejects_duplicate_missing_and_modality_incompatible_metadata() {
        let grid = tensor(TensorDtype::I32, &[1, 3]);
        let duplicate = InputPartDescriptor::new(
            InputModality::Image,
            InputPayloadKind::Tensor,
            tensor(TensorDtype::F32, &[2, 4]),
            [
                (InputMetadataKey::PatchGrid, grid.clone()),
                (InputMetadataKey::PatchGrid, grid.clone()),
            ],
        );
        assert!(matches!(
            duplicate,
            Err(PreparedInputError::DuplicateMetadata { .. })
        ));

        let image = InputPartDescriptor::new(
            InputModality::Image,
            InputPayloadKind::Tensor,
            tensor(TensorDtype::F32, &[2, 4]),
            [],
        )
        .unwrap();
        assert!(matches!(
            image.require_metadata(0, InputMetadataKey::PatchGrid),
            Err(PreparedInputError::MissingMetadata { .. })
        ));

        assert!(matches!(
            InputPartDescriptor::new(
                InputModality::Audio,
                InputPayloadKind::Tensor,
                tensor(TensorDtype::F32, &[2, 4]),
                [(InputMetadataKey::PatchGrid, grid)]
            ),
            Err(PreparedInputError::IncompatibleMetadata { .. })
        ));
    }

    #[test]
    fn malformed_wire_descriptors_fail_closed() {
        assert!(matches!(
            PreparedInputIdentity::decode_words(&[]),
            Err(PreparedInputError::TruncatedWireDescriptor(_))
        ));
        assert!(matches!(
            PreparedInputIdentity::decode_words(&[1, 99]),
            Err(PreparedInputError::InvalidWireValue {
                field: "modality",
                ..
            })
        ));
    }
}
