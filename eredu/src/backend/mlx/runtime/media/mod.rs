//! Media preprocessing before typed model prefill.

/// Typed runtime inputs for model prefill.
pub mod input;

use eredu_core::{
    checkpoint::TensorDtype, InputExtent, InputMetadataKey, InputModality, InputTensorIdentity,
    PreparedInputIdentity,
};
use eredu_runtime::{
    PreparedInputPart as RuntimePreparedInputPart,
    PreparedInputPayload as RuntimePreparedInputPayload,
    PreparedModelInput as RuntimePreparedModelInput,
};
use safemlx::{Array, Dtype};

use crate::{
    backend::mlx::error::Error,
    backend::mlx::runtime::media::input::{
        InputMetadata, InputPart, InputPayload, Modality, ModelInput,
    },
};

/// Shared PCM waveform validation and spectral operations.
#[cfg(feature = "mlx-audio")]
pub mod audio;
/// Shared decoded-image operations.
#[cfg(feature = "mlx-image")]
pub mod image;
/// Shared decoded-video validation, sampling, and timing operations.
#[cfg(feature = "mlx-image")]
pub mod video;

#[cfg(feature = "mlx-audio")]
pub(crate) use audio::AudioWaveform;
#[cfg(feature = "mlx-image")]
pub(crate) use image::RgbImageView;

/// One decoded media item supplied to a model processor.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "mlx-media")]
pub(crate) struct MediaInput<'a> {
    /// Declared modality of the item.
    pub(crate) modality: Modality,
    /// Decoded media payload.
    pub(crate) payload: MediaPayload<'a>,
}

#[cfg(feature = "mlx-media")]
impl<'a> MediaInput<'a> {
    /// Creates an RGB8 image input.
    #[cfg(feature = "mlx-image")]
    pub(crate) fn image_rgb8(image: RgbImageView<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: MediaPayload::Rgb8(image),
        }
    }

    /// Creates a decoded RGB8 video input with an explicit sampling policy.
    #[cfg(feature = "mlx-image")]
    pub(crate) fn video_rgb8_with_sampling(
        frames: &'a [RgbImageView<'a>],
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Self {
        Self {
            modality: Modality::Video,
            payload: MediaPayload::VideoFrames(VideoFrames {
                frames,
                source_fps,
                sampling,
            }),
        }
    }

    /// Creates a mono floating-point PCM audio input.
    #[cfg(feature = "mlx-audio")]
    pub(crate) fn audio_f32(samples: &'a [f32], sample_rate: u32) -> Result<Self, Error> {
        Ok(Self {
            modality: Modality::Audio,
            payload: MediaPayload::AudioF32(AudioWaveform::new(samples, sample_rate)?),
        })
    }
}

/// One ordered input segment supplied to a model processor.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "mlx-media")]
pub(crate) enum ProcessorInput<'a> {
    /// Already-tokenized text IDs.
    TokenIds(&'a [u32]),
    /// Decoded media to preprocess and insert at this exact position.
    Media(MediaInput<'a>),
}

/// Frame-selection policy for decoded video input.
#[cfg(feature = "mlx-image")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) enum VideoSampling {
    /// Uses the model processor's default frame rate and limits.
    #[default]
    ProcessorDefault,
    /// Uniformly samples approximately this many frames per second.
    Fps(f64),
    /// Uniformly samples exactly this many frames, capped by the source length.
    FrameCount(usize),
    /// Uses every decoded source frame.
    All,
}

/// Borrowed sequence of decoded RGB8 video frames.
#[cfg(feature = "mlx-image")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoFrames<'a> {
    /// Frames in source order.
    pub(crate) frames: &'a [RgbImageView<'a>],
    /// Source frame rate used for sampling and timestamp generation.
    pub(crate) source_fps: Option<f64>,
    /// Frame-selection policy.
    pub(crate) sampling: VideoSampling,
}

/// Decoded data accepted by media processors.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "mlx-media")]
pub(crate) enum MediaPayload<'a> {
    /// Decoded RGB8 image pixels.
    #[cfg(feature = "mlx-image")]
    Rgb8(RgbImageView<'a>),
    /// Decoded RGB8 video frames and timing metadata.
    #[cfg(feature = "mlx-image")]
    VideoFrames(VideoFrames<'a>),
    /// Mono floating-point PCM samples and their sampling rate.
    #[cfg(feature = "mlx-audio")]
    AudioF32(AudioWaveform<'a>),
    #[cfg(not(any(feature = "mlx-image", feature = "mlx-audio")))]
    #[doc(hidden)]
    _Lifetime(std::marker::PhantomData<&'a ()>),
}

#[derive(Debug, Clone)]
enum OwnedInputPayload {
    TokenIds(Array),
    Tensor(Array),
    Embeddings(Array),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct OwnedInputMetadata {
    patch_grid: Option<Array>,
    patch_positions: Option<Array>,
    audio_mask: Option<Array>,
    patch_extent: Option<[i32; 3]>,
    audio_valid_frames: Option<i32>,
}

impl OwnedInputMetadata {
    #[cfg(feature = "mlx-image")]
    pub(crate) fn patch_grid(value: Array) -> Self {
        Self {
            patch_grid: Some(value),
            ..Self::default()
        }
    }

    #[cfg(feature = "mlx-image")]
    pub(crate) fn patch_layout(grid: Array, positions: Array, extent: [i32; 3]) -> Self {
        Self {
            patch_grid: Some(grid),
            patch_positions: Some(positions),
            audio_mask: None,
            patch_extent: Some(extent),
            audio_valid_frames: None,
        }
    }

    #[cfg(feature = "mlx-audio")]
    pub(crate) fn audio_mask(value: Array, valid_frames: i32) -> Self {
        Self {
            audio_mask: Some(value),
            audio_valid_frames: Some(valid_frames),
            ..Self::default()
        }
    }

    fn from_input_metadata(metadata: InputMetadata<'_>) -> Self {
        Self {
            patch_grid: metadata.patch_grid.cloned(),
            patch_positions: metadata.patch_positions.cloned(),
            audio_mask: metadata.audio_mask.cloned(),
            patch_extent: metadata.patch_extent,
            audio_valid_frames: metadata.audio_valid_frames,
        }
    }
}

/// One owned part of a prepared model input.
#[derive(Debug, Clone)]
pub struct PreparedInputPart {
    modality: Modality,
    payload: OwnedInputPayload,
    metadata: OwnedInputMetadata,
}

impl PreparedInputPart {
    #[cfg(any(test, feature = "mlx-media"))]
    pub(crate) fn text_token_ids(ids: &[u32]) -> Self {
        Self {
            modality: Modality::Text,
            payload: OwnedInputPayload::TokenIds(Array::from_slice(ids, &[1, ids.len() as i32])),
            metadata: OwnedInputMetadata::default(),
        }
    }

    #[cfg(any(feature = "mlx-image", feature = "mlx-audio"))]
    pub(crate) fn media_tensor(
        modality: Modality,
        tensor: Array,
        metadata: OwnedInputMetadata,
    ) -> Self {
        Self {
            modality,
            payload: OwnedInputPayload::Tensor(tensor),
            metadata,
        }
    }

    /// Borrows this owned part as a runtime input part.
    pub fn as_input_part(&self) -> InputPart<'_> {
        let payload = match &self.payload {
            OwnedInputPayload::TokenIds(value) => InputPayload::TokenIds(value),
            OwnedInputPayload::Tensor(value) => InputPayload::Tensor(value),
            OwnedInputPayload::Embeddings(value) => InputPayload::Embeddings(value),
        };
        InputPart {
            modality: self.modality,
            payload,
            metadata: InputMetadata {
                patch_grid: self.metadata.patch_grid.as_ref(),
                patch_positions: self.metadata.patch_positions.as_ref(),
                audio_mask: self.metadata.audio_mask.as_ref(),
                patch_extent: self.metadata.patch_extent,
                audio_valid_frames: self.metadata.audio_valid_frames,
            },
        }
    }
}

/// Owned runtime input produced by a media processor.
#[derive(Debug, Clone)]
pub struct PreparedModelInput {
    inner: RuntimePreparedModelInput<Array>,
}

impl PreparedModelInput {
    fn new(parts: Vec<PreparedInputPart>) -> Result<Self, Error> {
        let parts = parts
            .into_iter()
            .map(runtime_prepared_part)
            .collect::<Result<Vec<_>, _>>()?;
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
        let parts = input
            .parts
            .iter()
            .map(|part| PreparedInputPart {
                modality: part.modality,
                payload: match part.payload {
                    InputPayload::TokenIds(array) => OwnedInputPayload::TokenIds(array.clone()),
                    InputPayload::Tensor(array) => OwnedInputPayload::Tensor(array.clone()),
                    InputPayload::Embeddings(array) => OwnedInputPayload::Embeddings(array.clone()),
                },
                metadata: OwnedInputMetadata::from_input_metadata(part.metadata),
            })
            .collect();
        Self::new(parts)
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
    /// Keep the returned vector alive for as long as the resulting
    /// [`ModelInput`] is used.
    pub fn input_parts(&self) -> Vec<InputPart<'_>> {
        self.inner.parts().iter().map(runtime_input_part).collect()
    }

    /// Calls `function` with a borrowed typed runtime input.
    pub fn with_model_input<T>(&self, function: impl FnOnce(ModelInput<'_>) -> T) -> T {
        let parts = self.input_parts();
        function(ModelInput::new(&parts))
    }

    /// Clones payload and metadata arrays in the identity's deterministic wire order.
    pub(crate) fn wire_arrays(&self) -> Vec<Array> {
        let mut arrays = Vec::new();
        arrays.extend(self.inner.wire_values().into_iter().cloned());
        arrays
    }

    /// Rebuilds owned prepared ingress from a validated identity and wire arrays.
    pub(crate) fn from_identity_wire_arrays(
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

fn runtime_prepared_part(
    part: PreparedInputPart,
) -> Result<RuntimePreparedInputPart<Array>, Error> {
    let payload = match part.payload {
        OwnedInputPayload::TokenIds(value) => RuntimePreparedInputPayload::TokenIds(value),
        OwnedInputPayload::Tensor(value) => RuntimePreparedInputPayload::Tensor(value),
        OwnedInputPayload::Embeddings(value) => RuntimePreparedInputPayload::Embeddings(value),
    };
    let metadata = [
        (InputMetadataKey::PatchGrid, part.metadata.patch_grid),
        (
            InputMetadataKey::PatchPositions,
            part.metadata.patch_positions,
        ),
        (InputMetadataKey::AudioMask, part.metadata.audio_mask),
    ]
    .into_iter()
    .filter_map(|(key, value)| value.map(|value| (key, value)));
    let extents = part
        .metadata
        .patch_extent
        .map(|[time, height, width]| {
            let [time, height, width] = [time, height, width].map(|value| {
                usize::try_from(value).map_err(|_| {
                    Error::Processor("prepared patch extent must be non-negative".into())
                })
            });
            Ok(InputExtent::PatchGrid {
                time: time?,
                height: height?,
                width: width?,
            })
        })
        .into_iter()
        .chain(part.metadata.audio_valid_frames.map(|frames| {
            usize::try_from(frames)
                .map(InputExtent::AudioValidFrames)
                .map_err(|_| Error::Processor("valid audio frames must be non-negative".into()))
        }))
        .collect::<Result<Vec<_>, Error>>()?;
    RuntimePreparedInputPart::new_with_extents(
        portable_modality(part.modality),
        payload,
        metadata,
        extents,
    )
    .map_err(prepared_input_error)
}

fn runtime_input_part(part: &RuntimePreparedInputPart<Array>) -> InputPart<'_> {
    let payload = match part.payload() {
        RuntimePreparedInputPayload::TokenIds(value) => InputPayload::TokenIds(value),
        RuntimePreparedInputPayload::Tensor(value) => InputPayload::Tensor(value),
        RuntimePreparedInputPayload::Embeddings(value) => InputPayload::Embeddings(value),
    };
    InputPart {
        modality: mlx_modality(part.modality()),
        payload,
        metadata: InputMetadata {
            patch_grid: part.metadata_value(InputMetadataKey::PatchGrid),
            patch_positions: part.metadata_value(InputMetadataKey::PatchPositions),
            audio_mask: part.metadata_value(InputMetadataKey::AudioMask),
            patch_extent: part.extents().iter().find_map(|extent| match extent {
                InputExtent::PatchGrid {
                    time,
                    height,
                    width,
                } => Some([*time as i32, *height as i32, *width as i32]),
                InputExtent::AudioValidFrames(_) => None,
            }),
            audio_valid_frames: part.extents().iter().find_map(|extent| match extent {
                InputExtent::AudioValidFrames(frames) => Some(*frames as i32),
                InputExtent::PatchGrid { .. } => None,
            }),
        },
    }
}

fn prepared_input_error(error: eredu_core::PreparedInputError) -> Error {
    Error::Parallel(error.to_string())
}

fn portable_modality(modality: Modality) -> InputModality {
    match modality {
        Modality::Text => InputModality::Text,
        Modality::Image => InputModality::Image,
        Modality::Video => InputModality::Video,
        Modality::Audio => InputModality::Audio,
    }
}

fn mlx_modality(modality: InputModality) -> Modality {
    match modality {
        InputModality::Text => Modality::Text,
        InputModality::Image => Modality::Image,
        InputModality::Video => Modality::Video,
        InputModality::Audio => Modality::Audio,
    }
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

pub(crate) fn prepared_identity_wire_arrays(
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
#[cfg(feature = "mlx-media")]
pub(crate) enum ProcessorPreparationError<E> {
    Backend(Error),
    #[cfg_attr(not(feature = "mlx-image"), allow(dead_code))]
    Text(E),
}

#[cfg(feature = "mlx-media")]
impl<E> From<Error> for ProcessorPreparationError<E> {
    fn from(error: Error) -> Self {
        Self::Backend(error)
    }
}

#[cfg(any(test, feature = "mlx-media"))]
pub(crate) fn prepared_model_input(
    parts: Vec<PreparedInputPart>,
) -> Result<PreparedModelInput, Error> {
    if parts.is_empty() {
        return Err(Error::Processor(
            "prepared model input must not be empty".to_string(),
        ));
    }
    PreparedModelInput::new(parts)
}

#[cfg(any(test, feature = "mlx-media"))]
pub(crate) fn push_text_token_ids(parts: &mut Vec<PreparedInputPart>, token_ids: &[u32]) {
    if !token_ids.is_empty() {
        parts.push(PreparedInputPart::text_token_ids(token_ids));
    }
}

#[cfg(test)]
mod prepared_input_identity_tests {
    use super::{
        input::{InputMetadata, InputPart, InputPayload, Modality, ModelInput},
        PreparedModelInput,
    };
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

        let image_part = [InputPart::image_tensor(
            &image,
            InputMetadata::patch_grid(&grid),
        )];
        let video_part = [InputPart::video_tensor(
            &image,
            InputMetadata::patch_grid(&grid),
        )];
        let shaped_part = [InputPart::image_tensor(
            &differently_shaped,
            InputMetadata::patch_grid(&grid),
        )];
        let metadata_part = [InputPart::image_tensor(
            &image,
            InputMetadata::patch_grid(&other_grid),
        )];

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
            InputPart::text_token_ids(&tokens),
            InputPart {
                modality: Modality::Image,
                payload: InputPayload::Embeddings(&embeddings),
                metadata: InputMetadata::patch_positions(&positions),
            },
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
                round_tripped.parts[1].payload,
                InputPayload::Embeddings(_)
            ));
            assert!(round_tripped.parts[1].metadata.patch_positions.is_some());
        });
    }
}
