//! Typed runtime inputs for model prefill.

use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp},
    Array, Stream,
};

/// Ordered runtime input for model prefill.
#[derive(Debug, Clone, Copy)]
pub struct ModelInput<'a> {
    /// Ordered input parts consumed by the model.
    pub parts: &'a [InputPart<'a>],
}

impl<'a> ModelInput<'a> {
    /// Creates a typed input from ordered parts.
    pub fn new(parts: &'a [InputPart<'a>]) -> Self {
        Self { parts }
    }
}

/// One ordered input part with an explicit modality.
#[derive(Debug, Clone, Copy)]
pub struct InputPart<'a> {
    /// The modality of this part.
    pub modality: Modality,
    /// The payload for this part.
    pub payload: InputPayload<'a>,
    /// Optional typed metadata needed by some model families.
    pub metadata: InputMetadata<'a>,
}

impl<'a> InputPart<'a> {
    /// Creates a text token-id part.
    pub fn text_token_ids(token_ids: &'a Array) -> Self {
        Self {
            modality: Modality::Text,
            payload: InputPayload::TokenIds(token_ids),
            metadata: InputMetadata::default(),
        }
    }

    /// Creates an image tensor part.
    pub fn image_tensor(tensor: &'a Array, metadata: InputMetadata<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: InputPayload::Tensor(tensor),
            metadata,
        }
    }

    /// Creates a video tensor part.
    pub fn video_tensor(tensor: &'a Array, metadata: InputMetadata<'a>) -> Self {
        Self {
            modality: Modality::Video,
            payload: InputPayload::Tensor(tensor),
            metadata,
        }
    }

    /// Creates an audio feature tensor part.
    pub fn audio_tensor(tensor: &'a Array, metadata: InputMetadata<'a>) -> Self {
        Self {
            modality: Modality::Audio,
            payload: InputPayload::Tensor(tensor),
            metadata,
        }
    }
}

/// Runtime modality.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Modality {
    /// Text token input.
    Text,
    /// Image tensor input.
    Image,
    /// Audio tensor input.
    Audio,
    /// Video tensor input.
    Video,
}

impl Modality {
    /// Returns a stable lowercase name for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Payload for one input part.
#[derive(Debug, Clone, Copy)]
pub enum InputPayload<'a> {
    /// Token ids shaped `[batch, sequence]`.
    TokenIds(&'a Array),
    /// Model-native tensor input for non-text modalities.
    Tensor(&'a Array),
    /// Already-projected embeddings shaped `[batch, sequence, hidden]`.
    Embeddings(&'a Array),
}

/// Optional metadata carried by an input part.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputMetadata<'a> {
    /// Architecture-neutral `(time, height, width)` vision grids shaped `[items, 3]`.
    pub patch_grid: Option<&'a Array>,
    /// Image or video-frame patch positions shaped `[batch, patches, 2]`, with negative coordinates for padding.
    pub patch_positions: Option<&'a Array>,
    /// Valid-frame mask for model-native audio features.
    pub audio_mask: Option<&'a Array>,
    /// Host-known `(time, height, width)` patch extent. This avoids a device
    /// synchronization when planning pooling and placeholder counts.
    pub patch_extent: Option<[i32; 3]>,
    /// Host-known number of valid, unpadded audio frames.
    pub audio_valid_frames: Option<i32>,
}

impl<'a> InputMetadata<'a> {
    /// Creates empty metadata.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates metadata carrying architecture-neutral vision `(t, h, w)` grids.
    pub fn patch_grid(grid_thw: &'a Array) -> Self {
        Self {
            patch_grid: Some(grid_thw),
            ..Self::default()
        }
    }

    /// Creates metadata carrying generic 2-D patch positions.
    pub fn patch_positions(position_ids: &'a Array) -> Self {
        Self {
            patch_grid: None,
            patch_positions: Some(position_ids),
            audio_mask: None,
            patch_extent: None,
            audio_valid_frames: None,
        }
    }

    /// Creates metadata carrying a valid-frame mask for audio features.
    pub fn audio_mask(mask: &'a Array) -> Self {
        Self {
            patch_grid: None,
            patch_positions: None,
            audio_mask: Some(mask),
            patch_extent: None,
            audio_valid_frames: None,
        }
    }

    /// Creates complete vision layout metadata with a host-known grid extent.
    pub fn patch_layout(grid_thw: &'a Array, position_ids: &'a Array, extent: [i32; 3]) -> Self {
        Self {
            patch_grid: Some(grid_thw),
            patch_positions: Some(position_ids),
            audio_mask: None,
            patch_extent: Some(extent),
            audio_valid_frames: None,
        }
    }

    /// Creates complete audio metadata with a host-known valid-frame count.
    pub fn audio_layout(mask: &'a Array, valid_frames: i32) -> Self {
        Self {
            patch_grid: None,
            patch_positions: None,
            audio_mask: Some(mask),
            patch_extent: None,
            audio_valid_frames: Some(valid_frames),
        }
    }
}

pub fn ensure_hidden_size(array: &Array, hidden_size: i32, name: &str) -> Result<(), Exception> {
    let shape = array.shape();
    if shape.len() != 3 || shape[2] != hidden_size {
        return Err(Exception::custom(format!(
            "{name} must be shaped [batch, sequence, {hidden_size}], got {shape:?}"
        )));
    }
    Ok(())
}

/// Validates basic modality/payload compatibility.
pub fn validate(input: ModelInput<'_>) -> Result<(), Exception> {
    if input.parts.is_empty() {
        return Err(Exception::custom(
            "model input must contain at least one part",
        ));
    }
    for part in input.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => validate_token_ids(tokens)?,
            (Modality::Text, InputPayload::Embeddings(embeddings)) => {
                validate_embeddings(embeddings, "text embeddings")?
            }
            (Modality::Text, InputPayload::Tensor(_)) => {
                return Err(Exception::custom(
                    "text input does not accept tensor payloads",
                ));
            }
            (Modality::Image | Modality::Audio | Modality::Video, InputPayload::Tensor(tensor)) => {
                validate_rank_at_least(tensor, 2, part.modality.as_str())?;
            }
            (
                Modality::Image | Modality::Audio | Modality::Video,
                InputPayload::Embeddings(embeddings),
            ) => validate_embeddings(embeddings, part.modality.as_str())?,
            (Modality::Image | Modality::Audio | Modality::Video, InputPayload::TokenIds(_)) => {
                return Err(Exception::custom(format!(
                    "{} input does not accept token-id payloads",
                    part.modality.as_str()
                )));
            }
        }
    }
    Ok(())
}

/// Builds a `[batch, sequence]` token array from text-only typed input.
pub fn text_token_ids(input: ModelInput<'_>, stream: &Stream) -> Result<Array, Exception> {
    validate(input)?;
    let mut parts = Vec::new();
    for part in input.parts {
        match (part.modality, part.payload) {
            (Modality::Text, InputPayload::TokenIds(tokens)) => parts.push(tokens.clone()),
            (Modality::Text, InputPayload::Embeddings(_)) => {
                return Err(Exception::custom(
                    "text embeddings are not supported by this model",
                ));
            }
            _ => {
                return Err(Exception::custom(format!(
                    "{} input is not supported by this model",
                    part.modality.as_str()
                )));
            }
        }
    }
    concatenate_token_parts(&parts, stream)
}

/// Converts a slice of token IDs to a batch-1 text input array.
pub fn token_ids_array(token_ids: &[u32], stream: &Stream) -> Result<Array, Exception> {
    Array::from(token_ids).try_index_device(NewAxis, stream)
}

/// Extracts backend-neutral media geometry from one native runtime input.
///
/// This function only observes array facts. Family admission and all derived
/// execution geometry remain owned by `eredu-architectures` media plans.
pub fn prepared_media_input(
    modality: Modality,
    payload: &Array,
    metadata: InputMetadata<'_>,
) -> Result<eredu_architectures::media_plan::PreparedMediaInput, Exception> {
    use eredu_architectures::media_plan::{MediaMetadata, MediaModality, PreparedMediaInput};

    fn shape(array: &Array) -> Result<Vec<u64>, Exception> {
        array
            .shape()
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension)
                    .map_err(|_| Exception::custom("prepared media dimension is negative"))
            })
            .collect()
    }

    fn i32_metadata(array: Option<&Array>) -> Result<Option<MediaMetadata<i32>>, Exception> {
        array
            .map(|array| {
                let evaluated = array.evaluated()?;
                let values = evaluated
                    .try_as_slice::<i32>()
                    .map_err(|error| Exception::custom(error.to_string()))?;
                Ok(MediaMetadata {
                    shape: shape(array)?,
                    values: values.to_vec(),
                })
            })
            .transpose()
    }

    fn bool_metadata(array: Option<&Array>) -> Result<Option<MediaMetadata<bool>>, Exception> {
        array
            .map(|array| {
                let evaluated = array.evaluated()?;
                let values = evaluated
                    .try_as_slice::<bool>()
                    .map_err(|error| Exception::custom(error.to_string()))?;
                Ok(MediaMetadata {
                    shape: shape(array)?,
                    values: values.to_vec(),
                })
            })
            .transpose()
    }

    let modality = match modality {
        Modality::Image => MediaModality::Image,
        Modality::Audio => MediaModality::Audio,
        Modality::Video => MediaModality::Video,
        Modality::Text => {
            return Err(Exception::custom(
                "prepared media geometry is unavailable for text input",
            ))
        }
    };
    Ok(PreparedMediaInput {
        modality,
        payload_shape: shape(payload)?,
        patch_grid: i32_metadata(metadata.patch_grid)?,
        patch_positions: i32_metadata(metadata.patch_positions)?,
        audio_mask: bool_metadata(metadata.audio_mask)?,
    })
}

/// Extracts the portable payload facts used by architecture input admission.
pub fn prepared_input_part(
    part: InputPart<'_>,
) -> Result<eredu_architectures::media_plan::PreparedInputPart, Exception> {
    use eredu_architectures::media_plan::{
        PreparedInputModality, PreparedInputPart, PreparedInputPayload,
    };

    let shape = |array: &Array| {
        array
            .shape()
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension)
                    .map_err(|_| Exception::custom("prepared input dimension is negative"))
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let modality = match part.modality {
        Modality::Text => PreparedInputModality::Text,
        Modality::Image => PreparedInputModality::Image,
        Modality::Audio => PreparedInputModality::Audio,
        Modality::Video => PreparedInputModality::Video,
    };
    let payload = match part.payload {
        InputPayload::TokenIds(tokens) => PreparedInputPayload::TokenIds(shape(tokens)?),
        InputPayload::Embeddings(embeddings) => {
            PreparedInputPayload::Embeddings(shape(embeddings)?)
        }
        InputPayload::Tensor(tensor) => PreparedInputPayload::Tensor {
            shape: shape(tensor)?,
            media: if part.modality == Modality::Text {
                None
            } else {
                Some(prepared_media_input(part.modality, tensor, part.metadata)?)
            },
        },
    };
    Ok(PreparedInputPart { modality, payload })
}

fn concatenate_token_parts(parts: &[Array], stream: &Stream) -> Result<Array, Exception> {
    if parts.is_empty() {
        return Err(Exception::custom("text input must contain token ids"));
    }
    if parts.len() == 1 {
        return Ok(parts[0].clone());
    }
    let refs = parts.iter().collect::<Vec<_>>();
    concatenate_axis(&refs, 1, stream)
}

fn validate_token_ids(tokens: &Array) -> Result<(), Exception> {
    let shape = tokens.shape();
    if shape.len() != 2 {
        return Err(Exception::custom(format!(
            "token ids must be shaped [batch, sequence], got {shape:?}"
        )));
    }
    if shape[0] <= 0 || shape[1] <= 0 {
        return Err(Exception::custom(format!(
            "token ids must have non-empty batch and sequence dimensions, got {shape:?}"
        )));
    }
    Ok(())
}

fn validate_embeddings(embeddings: &Array, name: &str) -> Result<(), Exception> {
    let shape = embeddings.shape();
    if shape.len() != 3 {
        return Err(Exception::custom(format!(
            "{name} must be shaped [batch, sequence, hidden], got {shape:?}"
        )));
    }
    if shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 {
        return Err(Exception::custom(format!(
            "{name} must have non-empty dimensions, got {shape:?}"
        )));
    }
    Ok(())
}

fn validate_rank_at_least(tensor: &Array, min_rank: usize, name: &str) -> Result<(), Exception> {
    if tensor.shape().len() < min_rank {
        return Err(Exception::custom(format!(
            "{name} tensor must have rank at least {min_rank}, got shape {:?}",
            tensor.shape()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate, InputMetadata, InputPart, InputPayload, Modality, ModelInput};
    use safemlx::Array;

    #[test]
    fn validates_text_token_part() {
        let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let parts = [InputPart::text_token_ids(&tokens)];

        validate(ModelInput::new(&parts)).unwrap();
    }

    #[test]
    fn rejects_empty_input() {
        let err = validate(ModelInput::new(&[])).unwrap_err();

        assert!(err.to_string().contains("at least one part"));
    }

    #[test]
    fn rejects_text_tensor_payload() {
        let tensor = Array::from_slice(&[0.0_f32, 1.0], &[1, 2]);
        let parts = [InputPart {
            modality: Modality::Text,
            payload: InputPayload::Tensor(&tensor),
            metadata: InputMetadata::empty(),
        }];

        let err = validate(ModelInput::new(&parts)).unwrap_err();

        assert!(err
            .to_string()
            .contains("text input does not accept tensor"));
    }

    #[test]
    fn accepts_future_modality_tensor_payloads() {
        let tensor = Array::from_slice(&[0.0_f32, 1.0], &[1, 2]);
        let parts = [
            InputPart {
                modality: Modality::Audio,
                payload: InputPayload::Tensor(&tensor),
                metadata: InputMetadata::empty(),
            },
            InputPart {
                modality: Modality::Video,
                payload: InputPayload::Tensor(&tensor),
                metadata: InputMetadata::empty(),
            },
        ];

        validate(ModelInput::new(&parts)).unwrap();
    }
}
