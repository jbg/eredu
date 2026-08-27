//! Media preprocessing before typed model prefill.

/// Typed runtime inputs for model prefill.
pub mod input;

#[cfg(all(test, not(any(feature = "image", feature = "audio"))))]
use eredu_core::InputModality;
#[cfg(feature = "image")]
use eredu_core::VideoSampling;
use eredu_core::{checkpoint::TensorDtype, InputTensorIdentity, PreparedInputIdentity};
#[cfg(any(feature = "image", feature = "audio"))]
use eredu_core::{InputExtent, InputMetadataKey, InputModality};
use eredu_runtime::PreparedModelInput as RuntimePreparedModelInput;
use safemlx::{Array, Dtype};

#[cfg(any(test, feature = "image", feature = "audio"))]
use crate::backend::runtime::media::input::InputPayload;
use crate::{backend::error::Error, backend::runtime::media::input::ModelInput};

/// Backend-neutral runtime input part specialized to MLX arrays.
pub(crate) use crate::backend::runtime::media::input::InputPart;

/// Shared PCM waveform validation and spectral operations.
#[cfg(feature = "audio")]
pub mod audio;
/// Shared decoded-image operations.
#[cfg(feature = "image")]
pub mod image;
/// Shared decoded-video validation, sampling, and timing operations.
#[cfg(feature = "image")]
pub mod video;

#[cfg(feature = "audio")]
pub(crate) use audio::AudioWaveform;
#[cfg(feature = "image")]
pub(crate) use image::RgbImageView;

/// One decoded media item supplied to a model processor.
#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "image", feature = "audio"))]
pub struct MediaInput<'a> {
    /// Declared modality of the item.
    pub modality: InputModality,
    /// Decoded media payload.
    pub payload: MediaPayload<'a>,
}

#[cfg(any(feature = "image", feature = "audio"))]
impl<'a> MediaInput<'a> {
    /// Creates an RGB8 image input.
    #[cfg(feature = "image")]
    pub fn image_rgb8(image: RgbImageView<'a>) -> Self {
        Self {
            modality: InputModality::Image,
            payload: MediaPayload::Rgb8(image),
        }
    }

    /// Creates a decoded RGB8 video input with an explicit sampling policy.
    #[cfg(feature = "image")]
    pub fn video_rgb8_with_sampling(
        frames: &'a [RgbImageView<'a>],
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Self {
        Self {
            modality: InputModality::Video,
            payload: MediaPayload::VideoFrames(VideoFrames {
                frames,
                source_fps,
                sampling,
            }),
        }
    }

    /// Creates a mono floating-point PCM audio input.
    #[cfg(feature = "audio")]
    pub fn audio_f32(samples: &'a [f32], sample_rate: u32) -> Result<Self, Error> {
        Ok(Self {
            modality: InputModality::Audio,
            payload: MediaPayload::AudioF32(AudioWaveform::new(samples, sample_rate)?),
        })
    }
}

/// One ordered input segment supplied to a model processor.
#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "image", feature = "audio"))]
pub enum ProcessorInput<'a> {
    /// Already-tokenized text IDs.
    TokenIds(&'a [u32]),
    /// Decoded media to preprocess and insert at this exact position.
    Media(MediaInput<'a>),
}

/// Borrowed sequence of decoded RGB8 video frames.
#[cfg(feature = "image")]
#[derive(Debug, Clone, Copy)]
pub struct VideoFrames<'a> {
    /// Frames in source order.
    pub frames: &'a [RgbImageView<'a>],
    /// Source frame rate used for sampling and timestamp generation.
    pub source_fps: Option<f64>,
    /// Frame-selection policy.
    pub sampling: VideoSampling,
}

/// Decoded data accepted by media processors.
#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "image", feature = "audio"))]
pub enum MediaPayload<'a> {
    /// Decoded RGB8 image pixels.
    #[cfg(feature = "image")]
    Rgb8(RgbImageView<'a>),
    /// Decoded RGB8 video frames and timing metadata.
    #[cfg(feature = "image")]
    VideoFrames(VideoFrames<'a>),
    /// Mono floating-point PCM samples and their sampling rate.
    #[cfg(feature = "audio")]
    AudioF32(AudioWaveform<'a>),
}

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

#[cfg(any(feature = "image", feature = "audio"))]
pub fn media_input_part(
    modality: InputModality,
    tensor: Array,
    metadata: impl IntoIterator<Item = (InputMetadataKey, Array)>,
    extents: impl IntoIterator<Item = InputExtent>,
) -> Result<InputPart, Error> {
    input::input_part(modality, InputPayload::Tensor(tensor), metadata, extents).map_err(Into::into)
}

/// Owned runtime input produced by a media processor.
#[derive(Debug, Clone)]
pub struct PreparedModelInput {
    inner: RuntimePreparedModelInput<Array>,
}

impl PreparedModelInput {
    fn new(parts: Vec<InputPart>) -> Result<Self, Error> {
        let inner = RuntimePreparedModelInput::new(parts, |array| {
            portable_tensor_identity(array).map_err(|error| {
                eredu_core::PreparedInputError::BackendTensorIdentity(error.to_string())
            })
        })
        .map_err(prepared_input_error)?;
        Ok(Self { inner })
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
        Ok(Self { inner })
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
pub enum ProcessorPreparationError<E> {
    Backend(Error),
    #[cfg_attr(not(feature = "image"), allow(dead_code))]
    Text(E),
}

#[cfg(any(feature = "image", feature = "audio"))]
impl<E> From<Error> for ProcessorPreparationError<E> {
    fn from(error: Error) -> Self {
        Self::Backend(error)
    }
}

#[cfg(any(test, feature = "image", feature = "audio"))]
pub fn prepared_model_input(parts: Vec<InputPart>) -> Result<PreparedModelInput, Error> {
    if parts.is_empty() {
        return Err(Error::Processor(
            "prepared model input must not be empty".to_string(),
        ));
    }
    PreparedModelInput::new(parts)
}

#[cfg(any(test, feature = "image", feature = "audio"))]
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
