//! Shared prepared-input observations and text-only admission.

use eredu_core::input::PreparedInputInspector;
use eredu_core::{
    CapabilityError, InputExtent, InputMetadataKey, InputModality, InputPartDescriptor,
    InputPayloadKind, InputTensorIdentity, PreparedInputPart,
};

use super::retained_session::PreparedInputPartPlan;

/// Observed shape and values of a prepared metadata tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataValues<T> {
    /// Observed metadata tensor shape.
    pub shape: Vec<u64>,
    /// Observed metadata values in tensor order.
    pub values: Vec<T>,
}

/// Native observations used by portable prepared-input admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaAdmissionInput {
    /// Exact descriptor of the prepared input.
    pub descriptor: InputPartDescriptor,
    /// Payload dimensions used in capability accounting.
    pub payload_shape: Vec<u64>,
    /// Declared patch grid when present.
    pub patch_grid: Option<MetadataValues<i32>>,
    /// Declared patch positions when present.
    pub patch_positions: Option<MetadataValues<i32>>,
    /// Declared audio validity mask when present.
    pub audio_mask: Option<MetadataValues<bool>>,
}

impl MediaAdmissionInput {
    /// Returns the modality retained by the prepared input.
    pub fn modality(&self) -> InputModality {
        self.descriptor.modality()
    }

    /// Returns the declared patch grid extent when present.
    pub fn patch_extent(&self) -> Option<[usize; 3]> {
        self.descriptor.extents().find_map(|extent| match extent {
            InputExtent::PatchGrid {
                time,
                height,
                width,
            } => Some([time, height, width]),
            InputExtent::AudioValidFrames(_) => None,
            _ => None,
        })
    }

    /// Returns the declared audio frame extent when present.
    pub fn audio_valid_frames(&self) -> Option<usize> {
        self.descriptor.extents().find_map(|extent| match extent {
            InputExtent::AudioValidFrames(frames) => Some(frames),
            InputExtent::PatchGrid { .. } => None,
            _ => None,
        })
    }
}

/// Returns the diagnostic name used for an admitted payload kind.
pub fn payload_name(kind: InputPayloadKind) -> &'static str {
    match kind {
        InputPayloadKind::TokenIds => "token-ID",
        InputPayloadKind::Tensor => "tensor",
        InputPayloadKind::Embeddings => "embedding",
        _ => "unknown",
    }
}

/// Converts exact prepared input dimensions for capability accounting.
pub fn u64_shape(identity: &InputTensorIdentity) -> Result<Vec<u64>, CapabilityError> {
    identity
        .shape()
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| CapabilityError::ArithmeticOverflow {
                operation: "prepared-input tensor dimension",
            })
        })
        .collect()
}

/// Observes the input descriptor and the metadata values consumed by admission.
pub fn inspect_part<Tensor>(
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<MediaAdmissionInput, CapabilityError> {
    let descriptor = input
        .descriptor(&|tensor| inspector.identity(tensor))
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    let inspect_metadata = descriptor.payload_kind() == InputPayloadKind::Tensor
        && descriptor.modality() != InputModality::Text;

    let metadata_shape = |key| descriptor.metadata_value(key).map(u64_shape).transpose();
    let i32_metadata = |key| -> Result<Option<MetadataValues<i32>>, CapabilityError> {
        if !inspect_metadata {
            return Ok(None);
        }
        input
            .metadata_value(key)
            .map(|tensor| {
                Ok(MetadataValues {
                    shape: metadata_shape(key)?.expect("runtime metadata has a descriptor"),
                    values: inspector.i32_values(tensor)?,
                })
            })
            .transpose()
    };
    let bool_metadata = |key| -> Result<Option<MetadataValues<bool>>, CapabilityError> {
        if !inspect_metadata {
            return Ok(None);
        }
        input
            .metadata_value(key)
            .map(|tensor| {
                Ok(MetadataValues {
                    shape: metadata_shape(key)?.expect("runtime metadata has a descriptor"),
                    values: inspector.bool_values(tensor)?,
                })
            })
            .transpose()
    };
    let payload_shape = u64_shape(descriptor.payload())?;
    let patch_grid = i32_metadata(InputMetadataKey::PatchGrid)?;
    let patch_positions = i32_metadata(InputMetadataKey::PatchPositions)?;
    let audio_mask = bool_metadata(InputMetadataKey::AudioMask)?;
    let media = MediaAdmissionInput {
        descriptor,
        payload_shape,
        patch_grid,
        patch_positions,
        audio_mask,
    };
    Ok(media)
}

/// Reports an input rejected by portable admission.
pub fn unsupported(architecture: &str, reason: impl Into<String>) -> CapabilityError {
    CapabilityError::UnsupportedInput {
        architecture: architecture.into(),
        reason: reason.into(),
    }
}

/// Validates a nonempty batch-one input and returns its sequence extent.
pub fn batch_one_sequence(
    shape: &[u64],
    rank: usize,
    name: &str,
    architecture: &str,
) -> Result<u64, CapabilityError> {
    if shape.len() != rank || shape.first() != Some(&1) || shape.get(1).copied().unwrap_or(0) == 0 {
        return Err(unsupported(
            architecture,
            format!("prepared {name} must be batch-one with rank {rank}, got {shape:?}"),
        ));
    }
    Ok(shape[1])
}

/// Validates one prepared input part for a text-only architecture.
pub fn text_only_input_part<Tensor>(
    architecture: &str,
    input: &PreparedInputPart<Tensor>,
    inspector: &impl PreparedInputInspector<Tensor>,
) -> Result<PreparedInputPartPlan, CapabilityError> {
    let descriptor = input
        .descriptor(&|tensor| inspector.identity(tensor))
        .map_err(|error| CapabilityError::Observation(error.to_string()))?;
    match (descriptor.modality(), descriptor.payload_kind()) {
        (InputModality::Text, InputPayloadKind::TokenIds) => Ok(PreparedInputPartPlan::Text {
            positions: batch_one_sequence(
                &u64_shape(descriptor.payload())?,
                2,
                "text token IDs",
                architecture,
            )?,
        }),
        (modality, payload) => Err(unsupported(
            architecture,
            format!(
                "text-only architecture does not support a {} {} payload",
                modality.as_str(),
                payload_name(payload)
            ),
        )),
    }
}
