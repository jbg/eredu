//! MLX conversion of validated portable media buffers before typed prefill.

/// Typed runtime inputs for model prefill.
pub mod input;

#[cfg(any(test, feature = "image", feature = "audio"))]
use eredu_core::InputModality;
use eredu_core::{checkpoint::TensorDtype, InputTensorIdentity, PreparedInputIdentity};
use eredu_runtime::{PreparedInputCacheIdentity, PreparedModelInput as RuntimePreparedModelInput};
use safemlx::{Array, Dtype};

#[cfg(any(test, feature = "image", feature = "audio"))]
use crate::backend::runtime::media::input::InputPayload;
use crate::{backend::error::Error, backend::runtime::media::input::ModelInput};

/// Backend-neutral runtime input part specialized to MLX arrays.
pub(crate) use crate::backend::runtime::media::input::InputPart;

#[cfg(any(test, feature = "image", feature = "audio"))]
fn text_input_part(ids: &[u32]) -> Result<InputPart, Error> {
    let width = i32::try_from(ids.len())
        .map_err(|_| Error::Processor("text input exceeds MLX dimension capacity".into()))?;
    input::input_part(
        InputModality::Text,
        InputPayload::TokenIds(Array::from_slice(ids, &[1, width])),
        [],
        [],
    )
    .map_err(Into::into)
}

/// Owned runtime input produced by a media processor.
#[derive(Debug, Clone)]
pub struct PreparedModelInput {
    inner: RuntimePreparedModelInput<Array>,
    cache_identity: Option<PreparedInputCacheIdentity>,
}

mod transfer;

impl PreparedModelInput {
    fn new(parts: Vec<InputPart>) -> Result<Self, Error> {
        let inner = RuntimePreparedModelInput::new(parts, |array| {
            portable_tensor_identity(array).map_err(|error| {
                eredu_core::PreparedInputError::BackendTensorIdentity(error.to_string())
            })
        })
        .map_err(prepared_input_error)?;
        Ok(Self {
            inner,
            cache_identity: None,
        })
    }

    /// Couples processor output to the exact semantic request that produced it.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn from_runtime_with_semantic_content(
        inner: RuntimePreparedModelInput<Array>,
        semantic_content_fingerprint: impl Into<String>,
    ) -> Result<Self, Error> {
        let cache_identity = inner
            .cache_identity(semantic_content_fingerprint)
            .map_err(|error| Error::Processor(error.to_string()))?;
        Ok(Self {
            inner,
            cache_identity: Some(cache_identity),
        })
    }

    /// Retains an observed prepared result without asserting an unchanged semantic identity.
    #[cfg(any(feature = "image", feature = "audio"))]
    pub(crate) fn from_observed_runtime(inner: RuntimePreparedModelInput<Array>) -> Self {
        Self {
            inner,
            cache_identity: None,
        }
    }

    /// Copies borrowed typed input into scheduler-safe owned storage.
    pub fn from_model_input(input: ModelInput<'_>) -> Result<Self, Error> {
        input::validate(input)?;
        Self::new(input.parts.to_vec())
    }

    /// Returns the payload-free collective identity of this prepared input.
    pub const fn identity(&self) -> &PreparedInputIdentity {
        self.inner.identity()
    }

    /// Exact prepared-description and semantic-content cache identity, when supplied.
    pub const fn cache_identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.cache_identity.as_ref()
    }

    /// Returns the number of ordered runtime parts.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true when no parts are present.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrows the prepared data as ordinary typed runtime input parts.
    ///
    /// The returned slice borrows this prepared input.
    pub fn input_parts(&self) -> &[InputPart] {
        self.inner.parts()
    }

    /// Calls `function` with a borrowed typed runtime input.
    pub fn with_model_input<T>(&self, function: impl FnOnce(ModelInput<'_>) -> T) -> T {
        function(ModelInput::new(self.input_parts()))
    }

    /// Clones payload and metadata arrays in the identity's deterministic wire order.
    pub fn wire_arrays(&self) -> Vec<Array> {
        let mut arrays = Vec::new();
        arrays.extend(self.inner.wire_values().into_iter().cloned());
        arrays
    }

    /// Describes the selected serial host-staged copy without submitting native work.
    /// Destination payload/count and the peak physical staging capacity are independent.
    /// Source backing, destination allocator capacity and evaluation scratch remain outside
    /// this per-copy report; a logical view may retain a larger source allocation.
    pub fn transfer_resources(
        &self,
    ) -> Result<eredu_runtime::input_transfer::InputTransferResources, Error> {
        use eredu_runtime::input_transfer::{InputTransferError, InputTransferResources};
        Ok(InputTransferResources::from_identity_with_staging(
            self.identity(),
            |tensor| {
                // The neutral preflight has already checked all dense wire extents.
                let bytes =
                    eredu_runtime::input_transfer::InputTransferResources::tensor_payload_bytes(
                        tensor,
                    )?;
                let bytes = usize::try_from(bytes).map_err(|_| InputTransferError::Overflow)?;
                safemlx::host_transfer_capacity_upper_bound(
                    bytes,
                    safemlx::HostTransferPolicy::Transfer,
                )
                .map(|bytes| bytes as u64)
                .map_err(|error| InputTransferError::StagingUnavailable {
                    detail: error.to_string(),
                })
            },
        )?)
    }

    /// Copies every payload and metadata tensor through completion-owned host staging.
    /// The entire request is admitted before cloning roots or allocating staging.
    /// Each staging buffer is retired after download and upload completion, so its
    /// physical peak is the largest single buffer rather than the sum of wire bytes.
    /// Source and all destination arrays remain retained through final completion.
    /// Exact structural and semantic cache identities survive the value-preserving copy.
    /// These bounds exclude source/destination allocator capacity and graph scratch.
    pub fn transfer_to_stream(
        &self,
        stream: &safemlx::Stream,
        budget: eredu_runtime::input_transfer::InputTransferBudget,
    ) -> Result<(Self, eredu_runtime::input_transfer::InputTransferResources), Error> {
        let resources = self.transfer_resources()?;
        budget.admit(resources)?;
        transfer::copy(self, stream, resources)
    }

    /// Rebuilds owned prepared ingress from a validated identity and wire arrays.
    pub fn from_identity_wire_arrays(
        identity: &PreparedInputIdentity,
        arrays: Vec<Array>,
    ) -> Result<Self, Error> {
        let expected = prepared_identity_wire_arrays(identity)?;
        if arrays.len() != expected.len()
            || arrays
                .iter()
                .zip(&expected)
                .any(|(array, (dtype, shape))| array.dtype() != *dtype || array.shape() != shape)
        {
            return Err(Error::Parallel(
                "prepared-input wire payload does not match its identity".into(),
            ));
        }
        let inner = RuntimePreparedModelInput::from_identity_wire_values(
            identity.clone(),
            arrays,
            |array| {
                portable_tensor_identity(array).map_err(|error| {
                    eredu_core::PreparedInputError::BackendTensorIdentity(error.to_string())
                })
            },
        )
        .map_err(prepared_input_error)?;
        Ok(Self {
            inner,
            cache_identity: None,
        })
    }
}

fn prepared_input_error(error: eredu_core::PreparedInputError) -> Error {
    Error::Parallel(error.to_string())
}

fn portable_tensor_identity(array: &Array) -> Result<InputTensorIdentity, Error> {
    let shape = array
        .shape()
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Error::Parallel(format!(
                    "prepared-input tensor has negative dimension {dimension}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    InputTensorIdentity::new(portable_dtype(array.dtype()), shape).map_err(prepared_input_error)
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
        Dtype::Float32 => TensorDtype::F32,
        Dtype::Float64 => TensorDtype::F64,
        Dtype::Bfloat16 => TensorDtype::Bf16,
        Dtype::Complex64 => TensorDtype::Complex64,
    }
}

fn mlx_tensor_identity(identity: &InputTensorIdentity) -> Result<(Dtype, Vec<i32>), Error> {
    let dtype = match identity.dtype() {
        TensorDtype::Bool => Dtype::Bool,
        TensorDtype::U8 => Dtype::Uint8,
        TensorDtype::U16 => Dtype::Uint16,
        TensorDtype::U32 => Dtype::Uint32,
        TensorDtype::U64 => Dtype::Uint64,
        TensorDtype::I8 => Dtype::Int8,
        TensorDtype::I16 => Dtype::Int16,
        TensorDtype::I32 => Dtype::Int32,
        TensorDtype::I64 => Dtype::Int64,
        TensorDtype::F16 => Dtype::Float16,
        TensorDtype::F32 => Dtype::Float32,
        TensorDtype::F64 => Dtype::Float64,
        TensorDtype::Bf16 => Dtype::Bfloat16,
        TensorDtype::Complex64 => Dtype::Complex64,
        TensorDtype::Encoded(name) => {
            return Err(Error::Parallel(format!(
                "prepared-input identity contains encoded dtype {name}"
            )))
        }
    };
    let shape = identity
        .shape()
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| {
                Error::Parallel(format!(
                    "prepared-input dimension {dimension} exceeds MLX range"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((dtype, shape))
}

/// Returns the MLX dtypes and shapes encoded by a prepared-input identity.
pub fn prepared_identity_wire_arrays(
    identity: &PreparedInputIdentity,
) -> Result<Vec<(Dtype, Vec<i32>)>, Error> {
    identity
        .parts()
        .iter()
        .flat_map(|part| {
            std::iter::once(part.payload())
                .chain(part.metadata().values())
                .map(mlx_tensor_identity)
        })
        .collect()
}

#[derive(Debug)]
#[cfg(any(feature = "image", feature = "audio"))]
/// Failure while converting application input into backend-ready model input.
pub enum ProcessorPreparationError<E> {
    /// MLX media preparation failed.
    Backend(Error),
    #[cfg_attr(not(feature = "image"), allow(dead_code))]
    /// Application text preparation failed.
    Text(E),
}

#[cfg(any(feature = "image", feature = "audio"))]
impl<E> From<Error> for ProcessorPreparationError<E> {
    fn from(error: Error) -> Self {
        Self::Backend(error)
    }
}

#[cfg(any(test, feature = "image", feature = "audio"))]
/// Validates and assembles owned prepared model-input parts.
pub fn prepared_model_input(parts: Vec<InputPart>) -> Result<PreparedModelInput, Error> {
    if parts.is_empty() {
        return Err(Error::Processor(
            "prepared model input must not be empty".to_string(),
        ));
    }
    PreparedModelInput::new(parts)
}

#[cfg(any(test, feature = "image", feature = "audio"))]
/// Appends a non-empty text-token input part.
pub fn push_text_token_ids(parts: &mut Vec<InputPart>, token_ids: &[u32]) -> Result<(), Error> {
    if !token_ids.is_empty() {
        parts.push(text_input_part(token_ids)?);
    }
    Ok(())
}

#[cfg(test)]
mod prepared_input_identity_tests {
    use super::{
        input::{input_part, InputPayload, ModelInput},
        PreparedModelInput,
    };
    use eredu_core::{InputMetadataKey, InputModality};
    use safemlx::Array;

    fn descriptor(input: ModelInput<'_>) -> Vec<u32> {
        PreparedModelInput::from_model_input(input)
            .unwrap()
            .identity()
            .encode_words()
            .unwrap()
    }

    #[test]
    fn identity_covers_modality_payload_shape_and_metadata() {
        let image = Array::from_slice(&[0.0_f32; 8], &[2, 4]);
        let differently_shaped = Array::from_slice(&[0.0_f32; 8], &[1, 8]);
        let grid = Array::from_slice(&[1_i32, 1, 2], &[1, 3]);
        let other_grid = Array::from_slice(&[1_i32, 1, 2], &[3, 1]);

        let image_part = [input_part(
            InputModality::Image,
            InputPayload::Tensor(image.clone()),
            [(InputMetadataKey::PatchGrid, grid.clone())],
            [],
        )
        .unwrap()];
        let video_part = [input_part(
            InputModality::Video,
            InputPayload::Tensor(image.clone()),
            [(InputMetadataKey::PatchGrid, grid.clone())],
            [],
        )
        .unwrap()];
        let shaped_part = [input_part(
            InputModality::Image,
            InputPayload::Tensor(differently_shaped),
            [(InputMetadataKey::PatchGrid, grid)],
            [],
        )
        .unwrap()];
        let metadata_part = [input_part(
            InputModality::Image,
            InputPayload::Tensor(image),
            [(InputMetadataKey::PatchGrid, other_grid)],
            [],
        )
        .unwrap()];

        let baseline = descriptor(ModelInput::new(&image_part));
        assert_ne!(baseline, descriptor(ModelInput::new(&video_part)));
        assert_ne!(baseline, descriptor(ModelInput::new(&shaped_part)));
        assert_ne!(baseline, descriptor(ModelInput::new(&metadata_part)));
    }

    #[test]
    fn explicit_input_transfer_preserves_values_identity_and_cache_under_exact_limits() {
        use eredu_runtime::input_transfer::{InputTransferBudget, InputTransferError};
        let stream =
            safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let tokens = Array::from_slice(&[16_777_217_u32, u32::MAX], &[1, 2]);
        let pixels = Array::from_slice(&[0.5_f32, -1.25, 2.0, 17.0], &[2, 2]);
        let grid = Array::from_slice(&[1_i32, 1, 2], &[1, 3]);
        let parts = [
            input_part(InputModality::Text, InputPayload::TokenIds(tokens), [], []).unwrap(),
            input_part(
                InputModality::Image,
                InputPayload::Tensor(pixels),
                [(InputMetadataKey::PatchGrid, grid)],
                [],
            )
            .unwrap(),
        ];
        let mut source = PreparedModelInput::from_model_input(ModelInput::new(&parts)).unwrap();
        source.cache_identity = Some(source.inner.cache_identity("transfer-fixture").unwrap());
        let identity = source.identity().clone();
        let cache_identity = source.cache_identity.clone();
        let expected = source.transfer_resources().unwrap();
        assert_eq!(expected.tensor_count, 3);
        assert_eq!(expected.payload_bytes, 36);
        assert!(expected.staging_capacity_bytes >= 16);
        let exact = expected.into_budget();
        for denied in [
            InputTransferBudget {
                tensor_count: 2,
                ..exact
            },
            InputTransferBudget {
                payload_bytes: 35,
                ..exact
            },
            InputTransferBudget {
                staging_capacity_bytes: exact.staging_capacity_bytes - 1,
                ..exact
            },
        ] {
            assert!(matches!(
                source.transfer_to_stream(&stream, denied),
                Err(crate::backend::error::Error::InputTransfer(
                    InputTransferError::Exhausted { .. }
                ))
            ));
        }
        let (copied, report) = source.transfer_to_stream(&stream, exact).unwrap();
        assert_eq!(report.tensor_count, 3);
        assert_eq!(report.payload_bytes, 36);
        assert_eq!(copied.identity(), &identity);
        assert_eq!(copied.cache_identity, cache_identity);
        drop(source);
        drop(parts);
        let arrays = copied.wire_arrays();
        assert_eq!(
            arrays[0].evaluated().unwrap().as_slice::<u32>(),
            &[16_777_217, u32::MAX]
        );
        assert_eq!(
            arrays[1].evaluated().unwrap().as_slice::<f32>(),
            &[0.5, -1.25, 2.0, 17.0]
        );
        assert_eq!(arrays[2].evaluated().unwrap().as_slice::<i32>(), &[1, 1, 2]);
        // Repeated explicit transfers remain within the same per-call staging
        // and logical destination ceilings and retain the original semantic request identity.
        let (again, second_report) = copied.transfer_to_stream(&stream, exact).unwrap();
        assert_eq!(second_report, report);
        assert_eq!(again.cache_identity, cache_identity);
    }

    #[test]
    fn borrowed_input_round_trips_into_owned_scheduler_payload() {
        let tokens = Array::from_slice(&[1_u32, 2], &[1, 2]);
        let embeddings = Array::from_slice(&[0.0_f32; 12], &[1, 2, 6]);
        let positions = Array::from_slice(&[0_i32, 1, 2, 3], &[1, 2, 2]);
        let parts = [
            input_part(InputModality::Text, InputPayload::TokenIds(tokens), [], []).unwrap(),
            input_part(
                InputModality::Image,
                InputPayload::Embeddings(embeddings),
                [(InputMetadataKey::PatchPositions, positions)],
                [],
            )
            .unwrap(),
        ];
        let borrowed = ModelInput::new(&parts);
        let expected = PreparedModelInput::from_model_input(borrowed)
            .unwrap()
            .identity()
            .clone();
        let prepared = PreparedModelInput::from_model_input(borrowed).unwrap();

        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared.identity(), &expected);
        prepared.with_model_input(|round_tripped| {
            assert!(matches!(
                round_tripped.parts[1].payload(),
                InputPayload::Embeddings(_)
            ));
            assert!(round_tripped.parts[1]
                .metadata_value(InputMetadataKey::PatchPositions)
                .is_some());
        });
    }
}
