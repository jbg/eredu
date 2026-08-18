//! Media preprocessing before typed model prefill.

/// Typed runtime inputs for model prefill.
pub mod input;

use std::{fs, path::Path};

use safemlx::{Array, Dtype};
#[cfg(feature = "image-processing")]
use safemlx_lm_core::VideoSampling as PortableVideoSampling;
use safemlx_lm_core::{
    Media as PortableMedia, TokenizedMultimodalRequest, TokenizedMultimodalSegment,
};

use crate::{
    error::Error,
    runtime::media::input::{InputMetadata, InputPart, InputPayload, Modality, ModelInput},
};

/// Shared PCM waveform validation and spectral operations.
#[cfg(feature = "audio-processing")]
pub mod audio;
use crate::architectures::gemma4::processor as gemma4;
/// Shared decoded-image operations.
#[cfg(feature = "image-processing")]
pub mod image;
use crate::architectures::inkling::processor as inkling;
#[cfg(feature = "image-processing")]
use crate::architectures::muse_glimmer::processor as muse_glimmer;
#[cfg(feature = "image-processing")]
use crate::architectures::qwen::vl::processor as qwen;
/// Shared decoded-video validation, sampling, and timing operations.
#[cfg(feature = "image-processing")]
pub mod video;

#[cfg(feature = "audio-processing")]
pub(crate) use audio::AudioWaveform;
#[cfg(feature = "image-processing")]
pub(crate) use image::RgbImageView;

/// One decoded media item supplied to a model processor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MediaInput<'a> {
    /// Declared modality of the item.
    pub(crate) modality: Modality,
    /// Decoded media payload.
    pub(crate) payload: MediaPayload<'a>,
}

impl<'a> MediaInput<'a> {
    /// Creates an RGB8 image input.
    #[cfg(feature = "image-processing")]
    pub(crate) fn image_rgb8(image: RgbImageView<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: MediaPayload::Rgb8(image),
        }
    }

    /// Creates a decoded RGB8 video input with an explicit sampling policy.
    #[cfg(feature = "image-processing")]
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
    #[cfg(feature = "audio-processing")]
    pub(crate) fn audio_f32(samples: &'a [f32], sample_rate: u32) -> Result<Self, Error> {
        Ok(Self {
            modality: Modality::Audio,
            payload: MediaPayload::AudioF32(AudioWaveform::new(samples, sample_rate)?),
        })
    }
}

/// One ordered input segment supplied to a model processor.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProcessorInput<'a> {
    /// Already-tokenized text IDs.
    TokenIds(&'a [u32]),
    /// Decoded media to preprocess and insert at this exact position.
    Media(MediaInput<'a>),
}

/// Frame-selection policy for decoded video input.
#[cfg(feature = "image-processing")]
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
#[cfg(feature = "image-processing")]
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
pub(crate) enum MediaPayload<'a> {
    /// Decoded RGB8 image pixels.
    #[cfg(feature = "image-processing")]
    Rgb8(RgbImageView<'a>),
    /// Decoded RGB8 video frames and timing metadata.
    #[cfg(feature = "image-processing")]
    VideoFrames(VideoFrames<'a>),
    /// Mono floating-point PCM samples and their sampling rate.
    #[cfg(feature = "audio-processing")]
    AudioF32(AudioWaveform<'a>),
    #[cfg(not(any(feature = "image-processing", feature = "audio-processing")))]
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
    qwen_grid_thw: Option<Array>,
    vision_grid_thw: Option<Array>,
    patch_position_ids: Option<Array>,
    audio_mask: Option<Array>,
}

impl OwnedInputMetadata {
    #[cfg(feature = "image-processing")]
    pub(crate) fn qwen_grid_thw(value: Array) -> Self {
        Self {
            qwen_grid_thw: Some(value),
            ..Self::default()
        }
    }

    #[cfg(feature = "image-processing")]
    pub(crate) fn vision_grid_thw(value: Array) -> Self {
        Self {
            vision_grid_thw: Some(value),
            ..Self::default()
        }
    }

    #[cfg(feature = "image-processing")]
    pub(crate) fn patch_position_ids(value: Array) -> Self {
        Self {
            patch_position_ids: Some(value),
            ..Self::default()
        }
    }

    #[cfg(feature = "audio-processing")]
    pub(crate) fn audio_mask(value: Array) -> Self {
        Self {
            audio_mask: Some(value),
            ..Self::default()
        }
    }

    fn from_input_metadata(metadata: InputMetadata<'_>) -> Self {
        Self {
            qwen_grid_thw: metadata.qwen_grid_thw.cloned(),
            vision_grid_thw: metadata.vision_grid_thw.cloned(),
            patch_position_ids: metadata.patch_position_ids.cloned(),
            audio_mask: metadata.audio_mask.cloned(),
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
    pub(crate) fn text_token_ids(ids: &[u32]) -> Self {
        Self {
            modality: Modality::Text,
            payload: OwnedInputPayload::TokenIds(Array::from_slice(ids, &[1, ids.len() as i32])),
            metadata: OwnedInputMetadata::default(),
        }
    }

    #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
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
                qwen_grid_thw: self.metadata.qwen_grid_thw.as_ref(),
                vision_grid_thw: self.metadata.vision_grid_thw.as_ref(),
                patch_position_ids: self.metadata.patch_position_ids.as_ref(),
                audio_mask: self.metadata.audio_mask.as_ref(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PreparedPayloadKind {
    TokenIds,
    Tensor,
    Embeddings,
}

impl PreparedPayloadKind {
    const fn wire_tag(self) -> u32 {
        match self {
            Self::TokenIds => 0,
            Self::Tensor => 1,
            Self::Embeddings => 2,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PreparedArrayIdentity {
    dtype: Dtype,
    shape: Vec<i32>,
}

impl PreparedArrayIdentity {
    fn new(array: &Array) -> Self {
        Self {
            dtype: array.dtype(),
            shape: array.shape().to_vec(),
        }
    }

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        output.extend_from_slice(&[self.dtype as u32, self.shape.len() as u32]);
        for dimension in &self.shape {
            output.push(u32::try_from(*dimension).map_err(|_| {
                Error::Parallel(format!(
                    "prepared model input dimension {dimension} exceeds descriptor range"
                ))
            })?);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PreparedInputPartIdentity {
    modality: Modality,
    payload_kind: PreparedPayloadKind,
    payload: PreparedArrayIdentity,
    qwen_grid_thw: Option<PreparedArrayIdentity>,
    vision_grid_thw: Option<PreparedArrayIdentity>,
    patch_position_ids: Option<PreparedArrayIdentity>,
    audio_mask: Option<PreparedArrayIdentity>,
}

impl PreparedInputPartIdentity {
    fn from_input_part(part: InputPart<'_>) -> Self {
        let (payload_kind, payload) = match part.payload {
            InputPayload::TokenIds(array) => (PreparedPayloadKind::TokenIds, array),
            InputPayload::Tensor(array) => (PreparedPayloadKind::Tensor, array),
            InputPayload::Embeddings(array) => (PreparedPayloadKind::Embeddings, array),
        };
        Self {
            modality: part.modality,
            payload_kind,
            payload: PreparedArrayIdentity::new(payload),
            qwen_grid_thw: part.metadata.qwen_grid_thw.map(PreparedArrayIdentity::new),
            vision_grid_thw: part
                .metadata
                .vision_grid_thw
                .map(PreparedArrayIdentity::new),
            patch_position_ids: part
                .metadata
                .patch_position_ids
                .map(PreparedArrayIdentity::new),
            audio_mask: part.metadata.audio_mask.map(PreparedArrayIdentity::new),
        }
    }

    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        output.extend_from_slice(&[
            match self.modality {
                Modality::Text => 0,
                Modality::Image => 1,
                Modality::Audio => 2,
                Modality::Video => 3,
            },
            self.payload_kind.wire_tag(),
        ]);
        self.payload.encode_descriptor(output)?;
        encode_optional_identity(self.qwen_grid_thw.as_ref(), output)?;
        encode_optional_identity(self.vision_grid_thw.as_ref(), output)?;
        encode_optional_identity(self.patch_position_ids.as_ref(), output)?;
        encode_optional_identity(self.audio_mask.as_ref(), output)
    }
}

fn encode_optional_identity(
    identity: Option<&PreparedArrayIdentity>,
    output: &mut Vec<u32>,
) -> Result<(), Error> {
    match identity {
        Some(identity) => {
            output.push(1);
            identity.encode_descriptor(output)
        }
        None => {
            output.push(0);
            Ok(())
        }
    }
}

/// Exact, payload-free identity for one prepared multimodal model input.
///
/// Pipeline stage zero owns the corresponding [`PreparedModelInput`]. Later
/// stages retain only this identity so distributed schedule consensus can
/// compare ordered modalities, payload kinds, dtypes, shapes, and metadata
/// shapes without retaining image, video, or audio tensors.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedModelInputIdentity {
    parts: Vec<PreparedInputPartIdentity>,
}

impl PreparedModelInputIdentity {
    /// Builds an identity from borrowed typed input without retaining tensors.
    pub fn from_model_input(input: ModelInput<'_>) -> Result<Self, Error> {
        input::validate(input)?;
        Ok(Self {
            parts: input
                .parts
                .iter()
                .copied()
                .map(PreparedInputPartIdentity::from_input_part)
                .collect(),
        })
    }

    /// Returns the number of ordered input parts represented by this identity.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Returns true when this identity has no input parts.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    pub(crate) fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
        output.push(u32::try_from(self.parts.len()).map_err(|_| {
            Error::Parallel("prepared model input has too many parts for consensus".into())
        })?);
        for part in &self.parts {
            part.encode_descriptor(output)?;
        }
        Ok(())
    }

    pub(crate) fn descriptor(&self) -> Result<Vec<u32>, Error> {
        let mut descriptor = Vec::new();
        self.encode_descriptor(&mut descriptor)?;
        Ok(descriptor)
    }

    pub(crate) fn from_descriptor(descriptor: &[u32]) -> Result<Self, Error> {
        fn take(cursor: &mut usize, values: &[u32]) -> Result<u32, Error> {
            let value = values.get(*cursor).copied().ok_or_else(|| {
                Error::Parallel("prepared-input descriptor ended unexpectedly".into())
            })?;
            *cursor += 1;
            Ok(value)
        }
        fn array(cursor: &mut usize, values: &[u32]) -> Result<PreparedArrayIdentity, Error> {
            let dtype = Dtype::try_from(take(cursor, values)?).map_err(|_| {
                Error::Parallel("prepared-input descriptor has an invalid dtype".into())
            })?;
            let ndim = take(cursor, values)? as usize;
            if ndim > 8 {
                return Err(Error::Parallel(
                    "prepared-input descriptor tensor rank exceeds 8".into(),
                ));
            }
            let shape = (0..ndim)
                .map(|_| {
                    i32::try_from(take(cursor, values)?).map_err(|_| {
                        Error::Parallel("prepared-input descriptor dimension exceeds i32".into())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(PreparedArrayIdentity { dtype, shape })
        }
        fn optional(
            cursor: &mut usize,
            values: &[u32],
        ) -> Result<Option<PreparedArrayIdentity>, Error> {
            match take(cursor, values)? {
                0 => Ok(None),
                1 => array(cursor, values).map(Some),
                _ => Err(Error::Parallel(
                    "prepared-input descriptor has an invalid optional tag".into(),
                )),
            }
        }
        let mut cursor = 0;
        let count = take(&mut cursor, descriptor)? as usize;
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            let modality = match take(&mut cursor, descriptor)? {
                0 => Modality::Text,
                1 => Modality::Image,
                2 => Modality::Audio,
                3 => Modality::Video,
                _ => {
                    return Err(Error::Parallel(
                        "prepared-input descriptor has an invalid modality".into(),
                    ))
                }
            };
            let payload_kind = match take(&mut cursor, descriptor)? {
                0 => PreparedPayloadKind::TokenIds,
                1 => PreparedPayloadKind::Tensor,
                2 => PreparedPayloadKind::Embeddings,
                _ => {
                    return Err(Error::Parallel(
                        "prepared-input descriptor has an invalid payload kind".into(),
                    ))
                }
            };
            parts.push(PreparedInputPartIdentity {
                modality,
                payload_kind,
                payload: array(&mut cursor, descriptor)?,
                qwen_grid_thw: optional(&mut cursor, descriptor)?,
                vision_grid_thw: optional(&mut cursor, descriptor)?,
                patch_position_ids: optional(&mut cursor, descriptor)?,
                audio_mask: optional(&mut cursor, descriptor)?,
            });
        }
        if cursor != descriptor.len() {
            return Err(Error::Parallel(
                "prepared-input descriptor has trailing values".into(),
            ));
        }
        Ok(Self { parts })
    }

    /// Returns payload and metadata array geometry in deterministic wire order.
    pub(crate) fn wire_arrays(&self) -> Vec<(Dtype, Vec<i32>)> {
        let mut arrays = Vec::new();
        for part in &self.parts {
            arrays.push((part.payload.dtype, part.payload.shape.clone()));
            for metadata in [
                part.qwen_grid_thw.as_ref(),
                part.vision_grid_thw.as_ref(),
                part.patch_position_ids.as_ref(),
                part.audio_mask.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                arrays.push((metadata.dtype, metadata.shape.clone()));
            }
        }
        arrays
    }
}

/// Owned runtime input produced by a media processor.
#[derive(Debug, Clone)]
pub struct PreparedModelInput {
    parts: Vec<PreparedInputPart>,
}

impl PreparedModelInput {
    fn new(parts: Vec<PreparedInputPart>) -> Self {
        Self { parts }
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
        Ok(Self::new(parts))
    }

    /// Returns the payload-free collective identity of this prepared input.
    pub fn identity(&self) -> PreparedModelInputIdentity {
        let parts = self.input_parts();
        PreparedModelInputIdentity {
            parts: parts
                .into_iter()
                .map(PreparedInputPartIdentity::from_input_part)
                .collect(),
        }
    }

    /// Returns the number of ordered runtime parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Returns true when no parts are present.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Borrows the prepared data as ordinary typed runtime input parts.
    ///
    /// Keep the returned vector alive for as long as the resulting
    /// [`ModelInput`] is used.
    pub fn input_parts(&self) -> Vec<InputPart<'_>> {
        self.parts
            .iter()
            .map(PreparedInputPart::as_input_part)
            .collect()
    }

    /// Calls `function` with a borrowed typed runtime input.
    pub fn with_model_input<T>(&self, function: impl FnOnce(ModelInput<'_>) -> T) -> T {
        let parts = self.input_parts();
        function(ModelInput::new(&parts))
    }

    /// Clones payload and metadata arrays in the identity's deterministic wire order.
    pub(crate) fn wire_arrays(&self) -> Vec<Array> {
        let mut arrays = Vec::new();
        for part in &self.parts {
            arrays.push(match &part.payload {
                OwnedInputPayload::TokenIds(value)
                | OwnedInputPayload::Tensor(value)
                | OwnedInputPayload::Embeddings(value) => value.clone(),
            });
            arrays.extend(part.metadata.qwen_grid_thw.iter().cloned());
            arrays.extend(part.metadata.vision_grid_thw.iter().cloned());
            arrays.extend(part.metadata.patch_position_ids.iter().cloned());
            arrays.extend(part.metadata.audio_mask.iter().cloned());
        }
        arrays
    }

    /// Rebuilds owned prepared ingress from a validated identity and wire arrays.
    pub(crate) fn from_identity_wire_arrays(
        identity: &PreparedModelInputIdentity,
        arrays: Vec<Array>,
    ) -> Result<Self, Error> {
        let expected = identity.wire_arrays();
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
        let mut arrays = arrays.into_iter();
        let mut parts = Vec::with_capacity(identity.parts.len());
        for part in &identity.parts {
            let payload = arrays.next().expect("validated wire array count");
            let payload = match part.payload_kind {
                PreparedPayloadKind::TokenIds => OwnedInputPayload::TokenIds(payload),
                PreparedPayloadKind::Tensor => OwnedInputPayload::Tensor(payload),
                PreparedPayloadKind::Embeddings => OwnedInputPayload::Embeddings(payload),
            };
            let mut next_optional = |present: bool| present.then(|| arrays.next().unwrap());
            let metadata = OwnedInputMetadata {
                qwen_grid_thw: next_optional(part.qwen_grid_thw.is_some()),
                vision_grid_thw: next_optional(part.vision_grid_thw.is_some()),
                patch_position_ids: next_optional(part.patch_position_ids.is_some()),
                audio_mask: next_optional(part.audio_mask.is_some()),
            };
            parts.push(PreparedInputPart {
                modality: part.modality,
                payload,
                metadata,
            });
        }
        Ok(Self { parts })
    }
}

/// Architecture-dispatched media processor loaded from a model directory.
#[derive(Debug, Clone)]
pub(crate) struct ModelProcessor {
    kind: ProcessorKind,
}

#[derive(Debug, Clone)]
enum ProcessorKind {
    Gemma4(gemma4::Gemma4Processor),
    Inkling(inkling::InklingProcessor),
    #[cfg(feature = "image-processing")]
    MuseGlimmer(muse_glimmer::MuseGlimmerProcessor),
    #[cfg(feature = "image-processing")]
    Qwen(qwen::QwenProcessor),
}

#[derive(Debug)]
pub(crate) enum ProcessorPreparationError<E> {
    Backend(Error),
    #[cfg_attr(not(feature = "image-processing"), allow(dead_code))]
    Text(E),
}

impl<E> From<Error> for ProcessorPreparationError<E> {
    fn from(error: Error) -> Self {
        Self::Backend(error)
    }
}

enum PortableMediaView<'a> {
    #[cfg(feature = "image-processing")]
    Image(RgbImageView<'a>),
    #[cfg(feature = "image-processing")]
    Video {
        frames: Vec<RgbImageView<'a>>,
        source_fps: Option<f64>,
        sampling: VideoSampling,
    },
    #[cfg(feature = "audio-processing")]
    Audio {
        samples: &'a [f32],
        sample_rate: u32,
    },
    #[allow(dead_code)]
    Unavailable(std::marker::PhantomData<&'a ()>),
}

impl<'a> PortableMediaView<'a> {
    fn new(media: &'a PortableMedia) -> Result<Self, Error> {
        match media {
            PortableMedia::Image(image) => {
                #[cfg(feature = "image-processing")]
                {
                    Ok(Self::Image(RgbImageView::packed(
                        image.pixels(),
                        image.width(),
                        image.height(),
                    )?))
                }
                #[cfg(not(feature = "image-processing"))]
                {
                    let _ = image;
                    Err(Error::Processor(
                        "MLX image preparation requires the image-processing feature".into(),
                    ))
                }
            }
            PortableMedia::Video(video) => {
                #[cfg(feature = "image-processing")]
                {
                    let frames = video
                        .frames()
                        .iter()
                        .map(|frame| {
                            RgbImageView::packed(frame.pixels(), frame.width(), frame.height())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let sampling = match video.sampling() {
                        PortableVideoSampling::ProcessorDefault => VideoSampling::ProcessorDefault,
                        PortableVideoSampling::Fps(fps) => VideoSampling::Fps(fps),
                        PortableVideoSampling::FrameCount(count) => {
                            VideoSampling::FrameCount(count)
                        }
                        PortableVideoSampling::All => VideoSampling::All,
                    };
                    Ok(Self::Video {
                        frames,
                        source_fps: video.source_fps(),
                        sampling,
                    })
                }
                #[cfg(not(feature = "image-processing"))]
                {
                    let _ = video;
                    Err(Error::Processor(
                        "MLX video preparation requires the image-processing feature".into(),
                    ))
                }
            }
            PortableMedia::Audio(audio) => {
                #[cfg(feature = "audio-processing")]
                {
                    Ok(Self::Audio {
                        samples: audio.samples(),
                        sample_rate: audio.sample_rate(),
                    })
                }
                #[cfg(not(feature = "audio-processing"))]
                {
                    let _ = audio;
                    Err(Error::Processor(
                        "MLX audio preparation requires the audio-processing feature".into(),
                    ))
                }
            }
        }
    }

    fn as_input(&self) -> Result<MediaInput<'_>, Error> {
        match self {
            #[cfg(feature = "image-processing")]
            Self::Image(image) => Ok(MediaInput::image_rgb8(*image)),
            #[cfg(feature = "image-processing")]
            Self::Video {
                frames,
                source_fps,
                sampling,
            } => Ok(MediaInput::video_rgb8_with_sampling(
                frames,
                *source_fps,
                *sampling,
            )),
            #[cfg(feature = "audio-processing")]
            Self::Audio {
                samples,
                sample_rate,
            } => MediaInput::audio_f32(samples, *sample_rate),
            Self::Unavailable(_) => unreachable!("unsupported media is rejected during mapping"),
        }
    }
}

impl ModelProcessor {
    pub(crate) fn load_gemma4(model_dir: &Path) -> Result<Option<Self>, Error> {
        gemma4::Gemma4Processor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Gemma4(processor),
            })
        })
    }

    #[cfg(any(feature = "image-processing", feature = "audio-processing"))]
    pub(crate) fn load_gemma4_gguf(
        model_metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
        projector_metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            kind: ProcessorKind::Gemma4(gemma4::Gemma4Processor::from_gguf(
                model_metadata,
                projector_metadata,
            )?),
        })
    }

    pub(crate) fn load_inkling(model_dir: &Path) -> Result<Option<Self>, Error> {
        inkling::InklingProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Inkling(processor),
            })
        })
    }

    #[cfg(feature = "image-processing")]
    pub(crate) fn load_muse_glimmer(model_dir: &Path) -> Result<Option<Self>, Error> {
        muse_glimmer::MuseGlimmerProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::MuseGlimmer(processor),
            })
        })
    }

    #[cfg(feature = "image-processing")]
    pub(crate) fn load_muse_glimmer_gguf(
        projector_metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            kind: ProcessorKind::MuseGlimmer(muse_glimmer::MuseGlimmerProcessor::from_gguf(
                projector_metadata,
            )?),
        })
    }

    #[cfg(feature = "media-processing")]
    pub(crate) fn load_inkling_gguf(
        metadata: &std::collections::HashMap<String, safemlx::ops::GgufMetadataValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            kind: ProcessorKind::Inkling(inkling::InklingProcessor::from_gguf(metadata)?),
        })
    }

    #[cfg(feature = "image-processing")]
    pub(crate) fn load_qwen(model_dir: &Path) -> Result<Option<Self>, Error> {
        qwen::QwenProcessor::load(model_dir).map(|processor| {
            processor.map(|processor| Self {
                kind: ProcessorKind::Qwen(processor),
            })
        })
    }

    /// Converts ordered text and decoded media segments into owned runtime input.
    pub(crate) fn prepare_input<E>(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        #[cfg(not(feature = "image-processing"))]
        let _ = &encode_text;
        match &self.kind {
            ProcessorKind::Gemma4(processor) => processor.prepare_input(input, encode_text),
            ProcessorKind::Inkling(processor) => processor.prepare_input(input, encode_text),
            #[cfg(feature = "image-processing")]
            ProcessorKind::MuseGlimmer(processor) => processor.prepare_input(input, encode_text),
            #[cfg(feature = "image-processing")]
            ProcessorKind::Qwen(processor) => processor.prepare_input(input, encode_text),
        }
    }

    pub(crate) fn prepare_portable_input<E>(
        &self,
        request: &TokenizedMultimodalRequest,
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<PreparedModelInput, ProcessorPreparationError<E>> {
        let media = request
            .segments()
            .iter()
            .filter_map(|segment| match segment {
                TokenizedMultimodalSegment::Media(media) => Some(PortableMediaView::new(media)),
                TokenizedMultimodalSegment::TokenIds(_) => None,
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut media = media.iter();
        let input = request
            .segments()
            .iter()
            .map(|segment| match segment {
                TokenizedMultimodalSegment::TokenIds(ids) => Ok(ProcessorInput::TokenIds(ids)),
                TokenizedMultimodalSegment::Media(_) => Ok(ProcessorInput::Media(
                    media
                        .next()
                        .expect("one prepared view exists for every media segment")
                        .as_input()?,
                )),
            })
            .collect::<Result<Vec<_>, Error>>()?;
        debug_assert!(media.next().is_none());
        self.prepare_input(&input, encode_text)
    }
}

/// Loads a supported media processor without loading model weights.
pub(crate) fn load_processor(model_dir: impl AsRef<Path>) -> Result<Option<ModelProcessor>, Error> {
    #[derive(serde::Deserialize)]
    struct Metadata {
        model_type: String,
        #[serde(default)]
        text_config: Option<TextMetadata>,
    }

    #[derive(serde::Deserialize)]
    struct TextMetadata {
        #[serde(default)]
        model_type: Option<String>,
    }

    let model_dir = model_dir.as_ref();
    let metadata: Metadata = serde_json::from_slice(&fs::read(model_dir.join("config.json"))?)?;
    let effective_type = metadata
        .text_config
        .as_ref()
        .and_then(|text| text.model_type.as_deref())
        .unwrap_or(&metadata.model_type);
    match effective_type {
        "inkling_mm_model" => ModelProcessor::load_inkling(model_dir),
        "gemma4" | "gemma4_text" | "gemma4_unified" | "gemma4_unified_text" => {
            ModelProcessor::load_gemma4(model_dir)
        }
        "muse_glimmer" | "muse_glimmer_text" => {
            #[cfg(feature = "image-processing")]
            {
                ModelProcessor::load_muse_glimmer(model_dir)
            }
            #[cfg(not(feature = "image-processing"))]
            {
                Ok(None)
            }
        }
        "qwen3_vl" | "qwen3_vl_text" | "qwen3_vl_moe" | "qwen3_vl_moe_text" | "qwen3_5"
        | "qwen3_5_text" | "qwen3_5_moe" | "qwen3_5_moe_text" => {
            #[cfg(feature = "image-processing")]
            {
                ModelProcessor::load_qwen(model_dir)
            }
            #[cfg(not(feature = "image-processing"))]
            {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

pub(crate) fn prepared_model_input(
    parts: Vec<PreparedInputPart>,
) -> Result<PreparedModelInput, Error> {
    if parts.is_empty() {
        return Err(Error::Processor(
            "prepared model input must not be empty".to_string(),
        ));
    }
    Ok(PreparedModelInput::new(parts))
}

pub(crate) fn push_text_token_ids(parts: &mut Vec<PreparedInputPart>, token_ids: &[u32]) {
    if !token_ids.is_empty() {
        parts.push(PreparedInputPart::text_token_ids(token_ids));
    }
}

#[cfg(all(test, feature = "image-processing"))]
mod portable_input_tests {
    use std::{collections::HashMap, convert::Infallible};

    use safemlx::ops::GgufMetadataValue;
    use safemlx_lm_core::{Media, MultimodalRequest, MultimodalSegment, RgbImage};

    use super::{input::Modality, ModelProcessor};

    #[test]
    fn mlx_processor_materializes_the_portable_ordered_request() {
        let model = HashMap::from([
            ("gemma4.boi_token_id".into(), GgufMetadataValue::Uint32(43)),
            ("gemma4.eoi_token_id".into(), GgufMetadataValue::Uint32(44)),
        ]);
        let projector = HashMap::from([
            (
                "clip.vision.patch_size".into(),
                GgufMetadataValue::Uint32(2),
            ),
            (
                "clip.vision.pooling_kernel_size".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "clip.vision.max_soft_tokens".into(),
                GgufMetadataValue::Uint32(70),
            ),
        ]);
        let processor = ModelProcessor::load_gemma4_gguf(&model, &projector).unwrap();
        let request = MultimodalRequest::new(vec![
            MultimodalSegment::TokenIds(vec![7]),
            MultimodalSegment::Media(Media::Image(
                RgbImage::new(vec![128; 4 * 4 * 3], 4, 4).unwrap(),
            )),
            MultimodalSegment::TokenIds(vec![8]),
        ])
        .unwrap()
        .tokenize::<Infallible>(|_| unreachable!("request contains token ids"))
        .unwrap();

        let prepared = processor
            .prepare_portable_input(&request, &mut |_| Ok::<_, Infallible>(Vec::new()))
            .unwrap();
        let parts = prepared.input_parts();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[2].modality, Modality::Image);
    }
}

#[cfg(test)]
mod prepared_input_identity_tests {
    use super::{
        input::{InputMetadata, InputPart, InputPayload, Modality, ModelInput},
        PreparedModelInput, PreparedModelInputIdentity,
    };
    use safemlx::Array;

    fn descriptor(input: ModelInput<'_>) -> Vec<u32> {
        let identity = PreparedModelInputIdentity::from_model_input(input).unwrap();
        let mut words = Vec::new();
        identity.encode_descriptor(&mut words).unwrap();
        words
    }

    #[test]
    fn identity_covers_modality_payload_shape_and_metadata() {
        let image = Array::from_slice(&[0.0_f32; 8], &[2, 4]);
        let differently_shaped = Array::from_slice(&[0.0_f32; 8], &[1, 8]);
        let grid = Array::from_slice(&[1_i32, 1, 2], &[1, 3]);
        let other_grid = Array::from_slice(&[1_i32, 1, 2], &[3, 1]);

        let image_part = [InputPart::image_tensor(
            &image,
            InputMetadata::qwen_grid_thw(&grid),
        )];
        let video_part = [InputPart::video_tensor(
            &image,
            InputMetadata::qwen_grid_thw(&grid),
        )];
        let shaped_part = [InputPart::image_tensor(
            &differently_shaped,
            InputMetadata::qwen_grid_thw(&grid),
        )];
        let metadata_part = [InputPart::image_tensor(
            &image,
            InputMetadata::qwen_grid_thw(&other_grid),
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
                metadata: InputMetadata::patch_position_ids(&positions),
            },
        ];
        let borrowed = ModelInput::new(&parts);
        let expected = PreparedModelInputIdentity::from_model_input(borrowed).unwrap();
        let prepared = PreparedModelInput::from_model_input(borrowed).unwrap();

        assert_eq!(prepared.len(), 2);
        assert_eq!(prepared.identity(), expected);
        prepared.with_model_input(|round_tripped| {
            assert!(matches!(
                round_tripped.parts[1].payload,
                InputPayload::Embeddings(_)
            ));
            assert!(round_tripped.parts[1].metadata.patch_position_ids.is_some());
        });
    }
}
