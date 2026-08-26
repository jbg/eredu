//! Typed runtime inputs for model prefill.

use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp},
    Array, Dtype, Stream,
};

use eredu_core::{
    checkpoint::TensorDtype, CapabilityError, InputExtent, InputMetadataKey, InputModality,
    InputTensorIdentity, PreparedInputError,
};
use eredu_runtime::{PreparedInputPart as RuntimeInputPart, PreparedInputPayload};

/// MLX specialization of the backend-neutral runtime input part.
pub type InputPart = RuntimeInputPart<Array>;

/// MLX specialization of the backend-neutral runtime payload container.
pub type InputPayload = PreparedInputPayload<Array>;

/// Ordered runtime input for model prefill.
#[derive(Debug, Clone, Copy)]
pub struct ModelInput<'a> {
    /// Ordered input parts consumed by the model.
    pub parts: &'a [InputPart],
}

impl<'a> ModelInput<'a> {
    /// Creates a typed input from ordered parts.
    pub fn new(parts: &'a [InputPart]) -> Self {
        Self { parts }
    }
}

/// Creates a structurally validated runtime part from MLX arrays.
pub fn input_part(
    modality: InputModality,
    payload: InputPayload,
    metadata: impl IntoIterator<Item = (InputMetadataKey, Array)>,
    extents: impl IntoIterator<Item = InputExtent>,
) -> Result<InputPart, Exception> {
    RuntimeInputPart::new_with_extents(modality, payload, metadata, extents)
        .map_err(|error| Exception::custom(error.to_string()))
}

/// Clones a token array handle into a validated text input part.
pub fn token_ids_part(token_ids: &Array) -> Result<InputPart, Exception> {
    input_part(
        InputModality::Text,
        InputPayload::TokenIds(token_ids.clone()),
        [],
        [],
    )
}

/// Describes MLX tensors and evaluates architecture-required metadata.
pub struct MlxInputInspector;

impl eredu_architectures::media_plan::PreparedInputInspector<Array> for MlxInputInspector {
    fn identity(&self, tensor: &Array) -> Result<InputTensorIdentity, PreparedInputError> {
        let shape = tensor
            .shape()
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    PreparedInputError::BackendTensorIdentity(format!(
                        "prepared-input tensor has negative dimension {dimension}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        InputTensorIdentity::new(portable_dtype(tensor.dtype()), shape)
    }

    fn i32_values(&self, tensor: &Array) -> Result<Vec<i32>, CapabilityError> {
        let evaluated = tensor
            .evaluated()
            .map_err(|error| CapabilityError::Observation(error.to_string()))?;
        evaluated
            .try_as_slice::<i32>()
            .map(|values| values.to_vec())
            .map_err(|error| CapabilityError::Observation(error.to_string()))
    }

    fn bool_values(&self, tensor: &Array) -> Result<Vec<bool>, CapabilityError> {
        let evaluated = tensor
            .evaluated()
            .map_err(|error| CapabilityError::Observation(error.to_string()))?;
        evaluated
            .try_as_slice::<bool>()
            .map(|values| values.to_vec())
            .map_err(|error| CapabilityError::Observation(error.to_string()))
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
        match (part.modality(), part.payload()) {
            (InputModality::Text, InputPayload::TokenIds(tokens)) => validate_token_ids(tokens)?,
            (InputModality::Text, InputPayload::Embeddings(embeddings)) => {
                validate_embeddings(embeddings, "text embeddings")?
            }
            (InputModality::Text, InputPayload::Tensor(_)) => {
                return Err(Exception::custom(
                    "text input does not accept tensor payloads",
                ));
            }
            (
                InputModality::Image | InputModality::Audio | InputModality::Video,
                InputPayload::Tensor(tensor),
            ) => {
                validate_rank_at_least(tensor, 2, part.modality().as_str())?;
            }
            (
                InputModality::Image | InputModality::Audio | InputModality::Video,
                InputPayload::Embeddings(embeddings),
            ) => validate_embeddings(embeddings, part.modality().as_str())?,
            (
                InputModality::Image | InputModality::Audio | InputModality::Video,
                InputPayload::TokenIds(_),
            ) => {
                return Err(Exception::custom(format!(
                    "{} input does not accept token-id payloads",
                    part.modality().as_str()
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
        match (part.modality(), part.payload()) {
            (InputModality::Text, InputPayload::TokenIds(tokens)) => parts.push(tokens.clone()),
            (InputModality::Text, InputPayload::Embeddings(_)) => {
                return Err(Exception::custom(
                    "text embeddings are not supported by this model",
                ));
            }
            _ => {
                return Err(Exception::custom(format!(
                    "{} input is not supported by this model",
                    part.modality().as_str()
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

fn portable_dtype(dtype: Dtype) -> TensorDtype {
    match dtype {
        Dtype::Bool => TensorDtype::Bool,
        Dtype::Uint8 => TensorDtype::U8,
        Dtype::Uint16 => TensorDtype::U16,
        Dtype::Uint32 => TensorDtype::U32,
        Dtype::Uint64 => TensorDtype::U64,
        Dtype::Int8 => TensorDtype::I8,
        Dtype::Int16 => TensorDtype::I16,
        Dtype::Int32 => TensorDtype::I32,
        Dtype::Int64 => TensorDtype::I64,
        Dtype::Float16 => TensorDtype::F16,
        Dtype::Bfloat16 => TensorDtype::Bf16,
        Dtype::Float32 => TensorDtype::F32,
        Dtype::Float64 => TensorDtype::F64,
        Dtype::Complex64 => TensorDtype::Complex64,
    }
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
    use super::{input_part, validate, InputPayload, ModelInput};
    use eredu_core::InputModality;
    use safemlx::Array;

    #[test]
    fn validates_text_token_part() {
        let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let parts =
            [input_part(InputModality::Text, InputPayload::TokenIds(tokens), [], []).unwrap()];

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
        assert!(input_part(InputModality::Text, InputPayload::Tensor(tensor), [], []).is_err());
    }

    #[test]
    fn accepts_future_modality_tensor_payloads() {
        let tensor = Array::from_slice(&[0.0_f32, 1.0], &[1, 2]);
        let parts = [
            input_part(
                InputModality::Audio,
                InputPayload::Tensor(tensor.clone()),
                [],
                [],
            )
            .unwrap(),
            input_part(InputModality::Video, InputPayload::Tensor(tensor), [], []).unwrap(),
        ];

        validate(ModelInput::new(&parts)).unwrap();
    }
}
