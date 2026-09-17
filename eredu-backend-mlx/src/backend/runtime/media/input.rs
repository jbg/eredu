mod original_semantics;
pub(crate) use original_semantics::OriginalMediaPacket;
#[cfg(test)]
pub(crate) use original_semantics::{
    original_semantic_preparations, record_original_semantic_preparation,
    reset_original_semantic_preparations,
};

// Typed runtime inputs for model prefill.

use safemlx::{
    Array, Dtype, Stream,
    error::Exception,
    ops::{concatenate_axis, indexing::NewAxis, indexing::TryIndexOp},
};

use eredu_core::{
    CapabilityError, InputExtent, InputMetadataKey, InputModality, InputTensorIdentity,
    PreparedInputError, checkpoint::TensorDtype,
};
use eredu_runtime::{
    PreparedInputCacheIdentity, PreparedInputInspector, PreparedInputPart as RuntimeInputPart,
    PreparedInputPayload,
};

/// MLX specialization of the backend-neutral runtime input part.
pub type InputPart = RuntimeInputPart<Array>;

/// MLX specialization of the backend-neutral runtime payload container.
pub type InputPayload = PreparedInputPayload<Array>;

/// Ordered runtime input for model prefill.
#[derive(Debug, Clone, Copy)]
pub struct ModelInput<'a> {
    /// Ordered input parts consumed by the model.
    pub parts: &'a [InputPart],
    cache_identity: Option<&'a PreparedInputCacheIdentity>,
    shared_cache_identity: Option<&'a eredu_runtime::SharedPreparedInputCacheIdentity>,
    memory_owner: Option<&'a crate::backend::managed_memory::NativeMemoryOwner>,
    prefill_chunk_positions: Option<std::num::NonZeroU64>,
    inference_request: Option<&'a eredu_runtime::working_memory::InferenceRequest>,
    original_media: Option<&'a OriginalMediaPacket>,
    original_media_metadata: Option<&'a eredu_nn::workspace::WorkspaceContext>,
    copied_media_semantics:
        Option<&'a eredu_architectures::media_plan::BoundPreparedMediaSemantics>,
}

impl<'a> ModelInput<'a> {
    pub(crate) fn with_copied_media_semantics(
        mut self,
        semantics: &'a eredu_architectures::media_plan::BoundPreparedMediaSemantics,
    ) -> Self {
        self.copied_media_semantics = Some(semantics);
        self
    }
    pub(crate) fn copied_media_semantics(
        self,
    ) -> Option<&'a eredu_architectures::media_plan::BoundPreparedMediaSemantics> {
        self.copied_media_semantics
    }

    pub(crate) fn with_original_media_metadata(
        mut self,
        metadata: &'a eredu_nn::workspace::WorkspaceContext,
    ) -> Self {
        self.original_media_metadata = Some(metadata);
        self
    }
    pub(crate) fn original_media_metadata(
        self,
    ) -> Option<&'a eredu_nn::workspace::WorkspaceContext> {
        self.original_media_metadata
    }

    pub(crate) fn with_original_media(mut self, packet: &'a OriginalMediaPacket) -> Self {
        self.original_media = Some(packet);
        self
    }
    pub(crate) fn original_media(self) -> Option<&'a OriginalMediaPacket> {
        self.original_media
    }

    /// Creates a typed input from ordered parts.
    pub fn new(parts: &'a [InputPart]) -> Self {
        Self {
            parts,
            cache_identity: None,
            shared_cache_identity: None,
            memory_owner: None,
            prefill_chunk_positions: None,
            inference_request: None,
            original_media: None,
            original_media_metadata: None,
            copied_media_semantics: None,
        }
    }

    /// Creates a typed input carrying its exact cache-relevant semantic identity.
    pub fn with_cache_identity(
        parts: &'a [InputPart],
        cache_identity: &'a PreparedInputCacheIdentity,
    ) -> Self {
        Self {
            parts,
            cache_identity: Some(cache_identity),
            shared_cache_identity: None,
            memory_owner: None,
            prefill_chunk_positions: None,
            inference_request: None,
            original_media: None,
            original_media_metadata: None,
            copied_media_semantics: None,
        }
    }

    /// Borrows an existing immutable identity without copying its payload or
    /// changing the accounting authority retained by its aliases.
    pub fn with_shared_cache_identity(
        parts: &'a [InputPart],
        identity: &'a eredu_runtime::SharedPreparedInputCacheIdentity,
    ) -> Self {
        Self {
            parts,
            cache_identity: Some(identity.as_ref()),
            shared_cache_identity: Some(identity),
            memory_owner: None,
            prefill_chunk_positions: None,
            inference_request: None,
            original_media: None,
            original_media_metadata: None,
            copied_media_semantics: None,
        }
    }

    /// Existing physical identity owner, when supplied by owned preparation.
    pub const fn shared_cache_identity(
        self,
    ) -> Option<&'a eredu_runtime::SharedPreparedInputCacheIdentity> {
        self.shared_cache_identity
    }

    pub(crate) fn with_memory_owner(
        mut self,
        owner: &'a crate::backend::managed_memory::NativeMemoryOwner,
    ) -> Self {
        self.memory_owner = Some(owner);
        self
    }

    pub(crate) fn memory_owner(
        self,
    ) -> Option<&'a crate::backend::managed_memory::NativeMemoryOwner> {
        self.memory_owner
    }

    /// Sets the ordinary prefill span size; selected media keeps its typed ingress.
    pub fn with_prefill_chunk_positions(mut self, positions: std::num::NonZeroU64) -> Self {
        self.prefill_chunk_positions = Some(positions);
        self
    }

    /// Requested ordinary span size. No setting preserves the existing ingress default.
    pub const fn prefill_chunk_positions(self) -> Option<std::num::NonZeroU64> {
        self.prefill_chunk_positions
    }

    /// Exact internal request admitted before native preparation. Borrowing a
    /// prompt preserves its charge owner; it does not create another admission.
    pub(crate) fn with_inference_request(
        mut self,
        request: &'a eredu_runtime::working_memory::InferenceRequest,
    ) -> Self {
        self.inference_request = Some(request);
        self
    }

    pub(crate) const fn inference_request(
        self,
    ) -> Option<&'a eredu_runtime::working_memory::InferenceRequest> {
        self.inference_request
    }

    /// Returns the identity coupled to these exact prepared values, when supplied.
    pub const fn cache_identity(self) -> Option<&'a PreparedInputCacheIdentity> {
        self.cache_identity
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

    fn identity_with_metadata(
        &self,
        tensor: &crate::MlxTensor,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<InputTensorIdentity, eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(
            InputTensorIdentity,
            Result<InputTensorIdentity, PreparedInputError>,
            &crate::MlxTensor,
        )>())?;
        let mut shape = context.metadata_vec(tensor.as_array().shape().len())?;
        for &dimension in tensor.as_array().shape() {
            shape.push(usize::try_from(dimension).map_err(|_| {
                context.metadata_error(format_args!(
                    "prepared-input tensor has negative dimension {dimension}"
                ))
            })?);
        }
        InputTensorIdentity::new(portable_dtype(tensor.as_array().dtype()), shape)
            .map_err(|cause| context.metadata_source(cause))
    }

    fn i32_values(&self, tensor: &crate::MlxTensor) -> Result<Vec<i32>, CapabilityError> {
        MlxInputInspector.i32_values(tensor.as_array())
    }

    fn bool_values(&self, tensor: &crate::MlxTensor) -> Result<Vec<bool>, CapabilityError> {
        MlxInputInspector.bool_values(tensor.as_array())
    }
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
            _ => {
                return Err(Exception::custom(
                    "model input uses an unsupported modality or payload",
                ));
            }
        }
    }
    Ok(())
}

/// Builds a `[batch, sequence]` token array from text-only typed input.
pub fn text_token_ids(input: ModelInput<'_>, stream: &Stream) -> Result<Array, Exception> {
    validate(input)?;
    if let [part] = input.parts {
        return text_token_part(part).cloned();
    }
    let parts = input
        .parts
        .iter()
        .map(text_token_part)
        .collect::<Result<Vec<_>, _>>()?;
    concatenate_axis(&parts, 1, stream)
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

fn text_token_part(part: &InputPart) -> Result<&Array, Exception> {
    match (part.modality(), part.payload()) {
        (InputModality::Text, InputPayload::TokenIds(tokens)) => Ok(tokens),
        (InputModality::Text, InputPayload::Embeddings(_)) => Err(Exception::custom(
            "text embeddings are not supported by this model",
        )),
        _ => Err(Exception::custom(format!(
            "{} input is not supported by this model",
            part.modality().as_str()
        ))),
    }
}

fn validate_token_ids(tokens: &Array) -> Result<(), Exception> {
    validate_token_ids_with_diagnostic(tokens, |message| Exception::custom(message.to_string()))
}

pub(crate) fn validate_token_ids_with_diagnostic<E>(
    tokens: &Array,
    mut diagnostic: impl FnMut(std::fmt::Arguments<'_>) -> E,
) -> Result<(), E> {
    let shape = tokens.shape();
    if shape.len() != 2 {
        return Err(diagnostic(format_args!(
            "token ids must be shaped [batch, sequence], got {shape:?}"
        )));
    }
    if shape[0] <= 0 || shape[1] <= 0 {
        return Err(diagnostic(format_args!(
            "token ids must have non-empty batch and sequence dimensions, got {shape:?}"
        )));
    }
    if !matches!(tokens.dtype(), Dtype::Int32 | Dtype::Uint32) {
        return Err(diagnostic(format_args!(
            "public token ids must use int32 or uint32 storage, got {:?}",
            tokens.dtype()
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
    use super::{InputPayload, ModelInput, input_part, text_token_ids, token_ids_part, validate};
    use eredu_core::InputModality;
    use safemlx::{Array, Device, DeviceType, Stream};

    #[test]
    fn text_token_extraction_preserves_single_backing_and_ordered_segments() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let first = Array::from_slice(&[11_u32, 7, 19], &[1, 3]);
        let second = Array::from_slice(&[2_u32, 23], &[1, 2]);
        first.evaluated().unwrap();
        let backing = first.allocation_info().unwrap().unwrap();
        let parts = [
            token_ids_part(&first).unwrap(),
            token_ids_part(&second).unwrap(),
        ];
        let single = text_token_ids(ModelInput::new(&parts[..1]), &stream).unwrap();
        single.evaluated().unwrap();
        assert_eq!(single.allocation_info().unwrap(), Some(backing));
        let joined = text_token_ids(ModelInput::new(&parts), &stream).unwrap();
        drop(parts);
        drop(first);
        drop(second);
        assert_eq!(single.shape(), [1, 3]);
        assert_eq!(
            single.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
            [11, 7, 19]
        );
        assert_eq!(joined.shape(), [1, 5]);
        assert_eq!(
            joined.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
            [11, 7, 19, 2, 23]
        );
    }

    #[test]
    fn text_token_extraction_validates_all_parts_before_selecting_payloads() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let embeddings = input_part(
            InputModality::Text,
            InputPayload::Embeddings(Array::from_slice(&[0.5_f32, 0.25], &[1, 1, 2])),
            [],
            [],
        )
        .unwrap();
        let invalid = input_part(
            InputModality::Text,
            InputPayload::TokenIds(Array::from_slice(&[1_i64, 2], &[1, 2])),
            [],
            [],
        )
        .unwrap();
        let parts = [embeddings, invalid];
        assert!(
            text_token_ids(ModelInput::new(&parts), &stream)
                .unwrap_err()
                .to_string()
                .contains("public token ids must use int32 or uint32")
        );
        assert!(
            text_token_ids(ModelInput::new(&parts[..1]), &stream)
                .unwrap_err()
                .to_string()
                .contains("text embeddings are not supported by this model")
        );
        assert!(
            text_token_ids(ModelInput::new(&[]), &stream)
                .unwrap_err()
                .to_string()
                .contains("at least one part")
        );
    }

    #[test]
    fn validates_text_token_part() {
        let tokens = Array::from_slice(&[1_u32, 2, 3], &[1, 3]);
        let parts =
            [input_part(InputModality::Text, InputPayload::TokenIds(tokens), [], []).unwrap()];

        validate(ModelInput::new(&parts)).unwrap();
    }

    #[test]
    fn rejects_public_int64_token_part() {
        let tokens = Array::from_slice(&[1_i64, 2], &[1, 2]);
        let parts =
            [input_part(InputModality::Text, InputPayload::TokenIds(tokens), [], []).unwrap()];

        let error = validate(ModelInput::new(&parts)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("public token ids must use int32 or uint32")
        );
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
