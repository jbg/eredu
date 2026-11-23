//! Typed runtime inputs for model prefill.

use eredu_core::input::PreparedInputInspector;
use safemlx::{
    error::Exception,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp},
    Array, Dtype, Stream,
};

use eredu_core::{
    checkpoint::TensorDtype, CapabilityError, InputExtent, InputMetadataKey, InputModality,
    InputTensorIdentity, PreparedInputError,
};
use eredu_core::{PreparedInputPart as RuntimeInputPart, PreparedInputPayload};

/// MLX specialization of the backend-neutral runtime input part.
pub type InputPart = RuntimeInputPart<Array>;

/// MLX specialization of the backend-neutral runtime payload container.
pub type InputPayload = PreparedInputPayload<Array>;

/// Reads original or architecture-declared token IDs, without interpreting embeddings.
pub(crate) fn host_token_ids(tokens: &Array) -> Result<Vec<u32>, crate::backend::error::Error> {
    use crate::backend::error::Error;
    let evaluated = tokens.evaluated()?;
    match tokens.dtype() {
        Dtype::Uint32 => evaluated
            .try_to_vec::<u32>()
            .map_err(|error| Error::Other(Box::new(error))),
        Dtype::Int32 => evaluated
            .try_to_vec::<i32>()
            .map_err(|error| Error::Other(Box::new(error)))?
            .into_iter()
            .map(|id| {
                u32::try_from(id).map_err(|_| {
                    Error::ArchitectureModel(
                        "decoder alignment contains a negative token ID".into(),
                    )
                })
            })
            .collect(),
        // Concatenating unsigned token parts with signed placeholder parts
        // can promote the assembled decoder sequence to Int64.
        Dtype::Int64 => evaluated
            .try_to_vec::<i64>()
            .map_err(|error| Error::Other(Box::new(error)))?
            .into_iter()
            .map(|id| {
                u32::try_from(id).map_err(|_| {
                    Error::ArchitectureModel("decoder alignment token ID is outside u32".into())
                })
            })
            .collect(),
        _ => Err(Error::ArchitectureModel(
            "decoder alignment requires integer token IDs".into(),
        )),
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

impl PreparedInputInspector<Array> for MlxInputInspector {
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
        if tensor.dtype() == Dtype::Uint32 {
            return evaluated
                .try_as_slice::<u32>()
                .map_err(|error| CapabilityError::Observation(error.to_string()))?
                .iter()
                .map(|value| {
                    i32::try_from(*value).map_err(|_| {
                        CapabilityError::Observation(format!(
                            "unsigned input value {value} exceeds signed 32-bit inspection range"
                        ))
                    })
                })
                .collect();
        }
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

/// Describes neutral MLX tensor wrappers for architecture-owned admission.
pub struct MlxTensorInputInspector;

impl PreparedInputInspector<crate::MlxTensor> for MlxTensorInputInspector {
    fn identity(
        &self,
        tensor: &crate::MlxTensor,
    ) -> Result<InputTensorIdentity, PreparedInputError> {
        MlxInputInspector.identity(tensor.as_array())
    }

    fn i32_values(&self, tensor: &crate::MlxTensor) -> Result<Vec<i32>, CapabilityError> {
        MlxInputInspector.i32_values(tensor.as_array())
    }

    fn token_ids(&self, tensor: &crate::MlxTensor) -> Result<Vec<u32>, eredu_core::BackendFailure> {
        host_token_ids(tensor.as_array()).map_err(eredu_core::BackendFailure::from_error)
    }

    fn bool_values(&self, tensor: &crate::MlxTensor) -> Result<Vec<bool>, CapabilityError> {
        MlxInputInspector.bool_values(tensor.as_array())
    }
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

pub(crate) fn concatenate_token_parts(
    parts: &[&Array],
    stream: &Stream,
) -> Result<Array, Exception> {
    for tokens in parts {
        validate_token_ids(tokens)?;
    }
    match parts {
        [] => Err(Exception::custom("text input must contain token ids")),
        [tokens] => Ok((*tokens).clone()),
        parts => concatenate_axis(parts, 1, stream),
    }
}

fn validate_token_ids(tokens: &Array) -> Result<(), Exception> {
    validate_decoder_tensor(eredu_core::InputPayloadKind::TokenIds, tokens)
}

fn validate_decoder_tensor(
    kind: eredu_core::InputPayloadKind,
    tensor: &Array,
) -> Result<(), Exception> {
    kind.validate_decoder_tensor(&portable_dtype(tensor.dtype()), tensor.shape())
        .map_err(|error| Exception::custom(error.to_string()))
}
