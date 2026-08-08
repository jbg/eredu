//! Media preprocessing before typed model prefill.

/// Typed runtime inputs for model prefill.
pub mod input;

use std::{fs, path::Path};

use safemlx::{Array, Dtype};

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
use crate::architectures::qwen::vl::processor as qwen;
/// Shared decoded-video validation, sampling, and timing operations.
#[cfg(feature = "image-processing")]
pub mod video;

#[cfg(feature = "audio-processing")]
pub use audio::AudioWaveform;
#[cfg(feature = "image-processing")]
pub use image::RgbImageView;

/// One decoded media item supplied to a model processor.
#[derive(Debug, Clone, Copy)]
pub struct MediaInput<'a> {
    /// Declared modality of the item.
    pub modality: Modality,
    /// Decoded media payload.
    pub payload: MediaPayload<'a>,
}

impl<'a> MediaInput<'a> {
    /// Creates an RGB8 image input.
    #[cfg(feature = "image-processing")]
    pub fn image_rgb8(image: RgbImageView<'a>) -> Self {
        Self {
            modality: Modality::Image,
            payload: MediaPayload::Rgb8(image),
        }
    }

    /// Creates a decoded RGB8 video input using processor-default sampling.
    #[cfg(feature = "image-processing")]
    pub fn video_rgb8(frames: &'a [RgbImageView<'a>], source_fps: Option<f64>) -> Self {
        Self::video_rgb8_with_sampling(frames, source_fps, VideoSampling::ProcessorDefault)
    }

    /// Creates a decoded RGB8 video input with an explicit sampling policy.
    #[cfg(feature = "image-processing")]
    pub fn video_rgb8_with_sampling(
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
    pub fn audio_f32(samples: &'a [f32], sample_rate: u32) -> Result<Self, Error> {
        Ok(Self {
            modality: Modality::Audio,
            payload: MediaPayload::AudioF32(AudioWaveform::new(samples, sample_rate)?),
        })
    }
}

/// One exact rendered-prompt placeholder bound to decoded media.
///
/// Chat templates remain checkpoint-owned, so callers identify the complete
/// placeholder spelling emitted by the selected template. The chat composer
/// validates occurrence counts and ordering before replacing each placeholder
/// with architecture-native processed media.
#[derive(Debug, Clone, Copy)]
pub struct ChatMediaBinding<'a> {
    /// Complete placeholder text to replace in the rendered chat prompt.
    pub placeholder: &'a str,
    /// Decoded media inserted at the placeholder's exact position.
    pub media: MediaInput<'a>,
}

impl<'a> ChatMediaBinding<'a> {
    /// Creates one rendered-placeholder media binding.
    pub const fn new(placeholder: &'a str, media: MediaInput<'a>) -> Self {
        Self { placeholder, media }
    }
}

/// One ordered input segment supplied to a model processor.
#[derive(Debug, Clone, Copy)]
pub enum ProcessorInput<'a> {
    /// Text that should be tokenized by the caller-provided encoder.
    Text(&'a str),
    /// Already-tokenized text IDs.
    TokenIds(&'a [u32]),
    /// Decoded media to preprocess and insert at this exact position.
    Media(MediaInput<'a>),
}

/// Frame-selection policy for decoded video input.
#[cfg(feature = "image-processing")]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum VideoSampling {
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
pub enum MediaPayload<'a> {
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
}

/// Architecture-dispatched media processor loaded from a model directory.
#[derive(Debug, Clone)]
pub struct ModelProcessor {
    kind: ProcessorKind,
}

#[derive(Debug, Clone)]
enum ProcessorKind {
    Gemma4(gemma4::Gemma4Processor),
    Inkling(inkling::InklingProcessor),
    #[cfg(feature = "image-processing")]
    Qwen(qwen::QwenProcessor),
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
    pub fn prepare_input(
        &self,
        input: &[ProcessorInput<'_>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<PreparedModelInput, Error> {
        #[cfg(not(feature = "image-processing"))]
        let _ = &encode_text;
        match &self.kind {
            ProcessorKind::Gemma4(processor) => processor.prepare_input(input, encode_text),
            ProcessorKind::Inkling(processor) => processor.prepare_input(input, encode_text),
            #[cfg(feature = "image-processing")]
            ProcessorKind::Qwen(processor) => processor.prepare_input(input, encode_text),
        }
    }

    /// Replaces checked rendered-chat placeholders with processed media.
    pub fn prepare_chat_input<'a>(
        &self,
        rendered_prompt: &'a str,
        bindings: &[ChatMediaBinding<'a>],
        encode_text: &mut dyn FnMut(&str) -> Result<Vec<u32>, Error>,
    ) -> Result<PreparedModelInput, Error> {
        let segments = chat_processor_segments(rendered_prompt, bindings)?;
        self.prepare_input(&segments, encode_text)
    }
}

fn chat_processor_segments<'a>(
    rendered_prompt: &'a str,
    bindings: &[ChatMediaBinding<'a>],
) -> Result<Vec<ProcessorInput<'a>>, Error> {
    for (index, binding) in bindings.iter().enumerate() {
        if binding.placeholder.is_empty() {
            return Err(Error::Processor(format!(
                "chat media binding {index} has an empty placeholder"
            )));
        }
        if bindings[..index]
            .iter()
            .any(|earlier| earlier.placeholder == binding.placeholder)
        {
            continue;
        }
        let expected = bindings
            .iter()
            .filter(|candidate| candidate.placeholder == binding.placeholder)
            .count();
        let actual = rendered_prompt.matches(binding.placeholder).count();
        if actual != expected {
            let placeholder = binding.placeholder;
            return Err(Error::Processor(format!(
                "rendered chat contains {actual} occurrence(s) of media placeholder \
                 {placeholder:?}, but {expected} binding(s) were supplied"
            )));
        }
    }

    let mut segments = Vec::with_capacity(bindings.len().saturating_mul(2) + 1);
    let mut cursor = 0;
    for (index, binding) in bindings.iter().enumerate() {
        let remainder = &rendered_prompt[cursor..];
        let relative = remainder.find(binding.placeholder).ok_or_else(|| {
            Error::Processor(format!(
                "chat media binding {index} placeholder {:?} does not occur after the preceding binding",
                binding.placeholder
            ))
        })?;
        let start = cursor + relative;
        if start > cursor {
            segments.push(ProcessorInput::Text(&rendered_prompt[cursor..start]));
        }
        segments.push(ProcessorInput::Media(binding.media));
        cursor = start + binding.placeholder.len();
    }
    if cursor < rendered_prompt.len() {
        segments.push(ProcessorInput::Text(&rendered_prompt[cursor..]));
    }
    if segments.is_empty() {
        segments.push(ProcessorInput::Text(rendered_prompt));
    }
    Ok(segments)
}

#[cfg(all(test, feature = "image-processing"))]
mod chat_input_tests {
    use super::{
        chat_processor_segments, ChatMediaBinding, MediaInput, ProcessorInput, RgbImageView,
    };

    fn image<'a>(pixels: &'a [u8]) -> MediaInput<'a> {
        MediaInput::image_rgb8(RgbImageView::packed(pixels, 1, 1).unwrap())
    }

    #[test]
    fn chat_composer_replaces_repeated_placeholders_in_order() {
        let pixels = [0_u8; 3];
        let bindings = [
            ChatMediaBinding::new("<image>", image(&pixels)),
            ChatMediaBinding::new("<image>", image(&pixels)),
        ];
        let segments =
            chat_processor_segments("before<image>middle<image>after", &bindings).unwrap();

        assert_eq!(segments.len(), 5);
        assert!(matches!(segments[0], ProcessorInput::Text("before")));
        assert!(matches!(segments[1], ProcessorInput::Media(_)));
        assert!(matches!(segments[2], ProcessorInput::Text("middle")));
        assert!(matches!(segments[3], ProcessorInput::Media(_)));
        assert!(matches!(segments[4], ProcessorInput::Text("after")));
    }

    #[test]
    fn chat_composer_rejects_count_and_order_mismatches() {
        let pixels = [0_u8; 3];
        let count_error = chat_processor_segments(
            "before<image><image>after",
            &[ChatMediaBinding::new("<image>", image(&pixels))],
        )
        .unwrap_err();
        assert!(count_error.to_string().contains("2 occurrence"));

        let order_error = chat_processor_segments(
            "<second><first>",
            &[
                ChatMediaBinding::new("<first>", image(&pixels)),
                ChatMediaBinding::new("<second>", image(&pixels)),
            ],
        )
        .unwrap_err();
        assert!(order_error.to_string().contains("does not occur after"));
    }
}

/// Loads a supported media processor without loading model weights.
pub fn load_processor(model_dir: impl AsRef<Path>) -> Result<Option<ModelProcessor>, Error> {
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
