//! Portable decoded-media requests and chat-placeholder composition.

/// Failure while validating portable media or composing a chat request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MediaRequestError {
    /// An input request must contain at least one segment.
    #[error("multimodal input request is empty")]
    EmptyRequest,
    /// Multimodal preparation is meaningful only when media is present.
    #[error("multimodal input request contains no media")]
    MissingMedia,
    /// RGB dimensions or payload length are invalid.
    #[error("RGB8 image shape {width}x{height} requires {expected} bytes, got {actual}")]
    InvalidRgbImage {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
        /// Required packed RGB byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
    /// Audio must have a positive sample rate and finite samples.
    #[error("decoded audio is invalid: {0}")]
    InvalidAudio(String),
    /// Video frame or timing metadata is invalid.
    #[error("decoded video is invalid: {0}")]
    InvalidVideo(String),
    /// A chat media placeholder cannot be empty.
    #[error("chat media binding {index} has an empty placeholder")]
    EmptyPlaceholder {
        /// Binding index.
        index: usize,
    },
    /// Placeholder occurrences do not match supplied bindings.
    #[error(
        "rendered chat contains {actual} occurrence(s) of media placeholder {placeholder:?}, but {expected} binding(s) were supplied"
    )]
    PlaceholderCount {
        /// Complete placeholder spelling.
        placeholder: String,
        /// Number of supplied bindings with that spelling.
        expected: usize,
        /// Number of occurrences in rendered text.
        actual: usize,
    },
    /// Bindings were not ordered like their occurrences in the prompt.
    #[error(
        "chat media binding {index} placeholder {placeholder:?} does not occur after the preceding binding"
    )]
    PlaceholderOrder {
        /// Binding index.
        index: usize,
        /// Complete placeholder spelling.
        placeholder: String,
    },
}

/// Packed decoded RGB8 image owned by a portable request.
#[derive(Debug, Clone, PartialEq)]
pub struct RgbImage {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl RgbImage {
    /// Validates and owns one packed RGB8 image.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32) -> Result<Self, MediaRequestError> {
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| usize::try_from(height).ok().map(|height| (width, height)))
            .and_then(|(width, height)| width.checked_mul(height))
            .and_then(|pixels| pixels.checked_mul(3))
            .unwrap_or(usize::MAX);
        if width == 0 || height == 0 || pixels.len() != expected {
            return Err(MediaRequestError::InvalidRgbImage {
                width,
                height,
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self {
            pixels,
            width,
            height,
        })
    }

    /// Packed RGB channel bytes in row-major order.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Image width in pixels.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub const fn height(&self) -> u32 {
        self.height
    }
}

/// Decoded mono floating-point PCM audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Audio {
    /// Validates and owns decoded mono PCM samples.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Result<Self, MediaRequestError> {
        if sample_rate == 0 {
            return Err(MediaRequestError::InvalidAudio(
                "sample rate must be positive".into(),
            ));
        }
        if samples.is_empty() {
            return Err(MediaRequestError::InvalidAudio(
                "waveform must contain at least one sample".into(),
            ));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(MediaRequestError::InvalidAudio(
                "waveform samples must be finite".into(),
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Mono PCM samples.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Sampling rate in hertz.
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Frame-selection policy for decoded video.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum VideoSampling {
    /// Uses the selected backend processor's default policy.
    #[default]
    ProcessorDefault,
    /// Uniformly samples approximately this many frames per second.
    Fps(f64),
    /// Uniformly samples exactly this many frames, capped by source length.
    FrameCount(usize),
    /// Uses every decoded source frame.
    All,
}

/// Decoded RGB8 video and portable sampling policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Video {
    frames: Vec<RgbImage>,
    source_fps: Option<f64>,
    sampling: VideoSampling,
}

impl Video {
    /// Validates and owns decoded video frames.
    pub fn new(
        frames: Vec<RgbImage>,
        source_fps: Option<f64>,
        sampling: VideoSampling,
    ) -> Result<Self, MediaRequestError> {
        if frames.is_empty() {
            return Err(MediaRequestError::InvalidVideo(
                "video must contain at least one frame".into(),
            ));
        }
        if source_fps.is_some_and(|fps| !fps.is_finite() || fps <= 0.0) {
            return Err(MediaRequestError::InvalidVideo(
                "source frame rate must be finite and positive".into(),
            ));
        }
        match sampling {
            VideoSampling::Fps(fps) if !fps.is_finite() || fps <= 0.0 => {
                return Err(MediaRequestError::InvalidVideo(
                    "sampling frame rate must be finite and positive".into(),
                ));
            }
            VideoSampling::FrameCount(0) => {
                return Err(MediaRequestError::InvalidVideo(
                    "sampling frame count must be positive".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            frames,
            source_fps,
            sampling,
        })
    }

    /// Decoded frames in source order.
    pub fn frames(&self) -> &[RgbImage] {
        &self.frames
    }

    /// Source frame rate when known.
    pub const fn source_fps(&self) -> Option<f64> {
        self.source_fps
    }

    /// Requested frame-selection policy.
    pub const fn sampling(&self) -> VideoSampling {
        self.sampling
    }
}

/// One decoded media item in a portable request.
#[derive(Debug, Clone, PartialEq)]
pub enum Media {
    /// Decoded packed RGB8 image.
    Image(RgbImage),
    /// Decoded RGB8 video.
    Video(Video),
    /// Decoded mono PCM audio.
    Audio(Audio),
}

/// One ordered portable model-input segment.
#[derive(Debug, Clone, PartialEq)]
pub enum MultimodalSegment {
    /// Text encoded by the tokenizer attached to `LoadedModel`.
    Text(String),
    /// Text already represented as tokenizer vocabulary IDs.
    TokenIds(Vec<u32>),
    /// Decoded media preprocessed by the selected backend.
    Media(Media),
}

/// Validated ordered text and decoded-media input.
#[derive(Debug, Clone, PartialEq)]
pub struct MultimodalRequest {
    segments: Vec<MultimodalSegment>,
}

impl MultimodalRequest {
    /// Validates an ordered multimodal request.
    pub fn new(segments: Vec<MultimodalSegment>) -> Result<Self, MediaRequestError> {
        if segments.is_empty() {
            return Err(MediaRequestError::EmptyRequest);
        }
        if !segments
            .iter()
            .any(|segment| matches!(segment, MultimodalSegment::Media(_)))
        {
            return Err(MediaRequestError::MissingMedia);
        }
        Ok(Self { segments })
    }

    /// Composes a rendered prompt with media bound to exact placeholder occurrences.
    pub fn from_chat(
        rendered_prompt: &str,
        bindings: &[MediaBinding],
    ) -> Result<Self, MediaRequestError> {
        validate_bindings(rendered_prompt, bindings)?;
        let mut segments = Vec::with_capacity(bindings.len().saturating_mul(2) + 1);
        let mut cursor = 0;
        for (index, binding) in bindings.iter().enumerate() {
            let remainder = &rendered_prompt[cursor..];
            let relative = remainder.find(binding.placeholder()).ok_or_else(|| {
                MediaRequestError::PlaceholderOrder {
                    index,
                    placeholder: binding.placeholder().into(),
                }
            })?;
            let start = cursor + relative;
            if start > cursor {
                segments.push(MultimodalSegment::Text(
                    rendered_prompt[cursor..start].into(),
                ));
            }
            segments.push(MultimodalSegment::Media(binding.media().clone()));
            cursor = start + binding.placeholder().len();
        }
        if cursor < rendered_prompt.len() {
            segments.push(MultimodalSegment::Text(rendered_prompt[cursor..].into()));
        }
        Self::new(segments)
    }

    /// Ordered request segments.
    pub fn segments(&self) -> &[MultimodalSegment] {
        &self.segments
    }

    /// Encodes text while preserving media order for backend preparation.
    pub fn tokenize<E>(
        &self,
        mut encode: impl FnMut(&str) -> Result<Vec<u32>, E>,
    ) -> Result<TokenizedMultimodalRequest, E> {
        let mut segments = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            segments.push(match segment {
                MultimodalSegment::Text(text) => {
                    TokenizedMultimodalSegment::TokenIds(encode(text)?)
                }
                MultimodalSegment::TokenIds(ids) => {
                    TokenizedMultimodalSegment::TokenIds(ids.clone())
                }
                MultimodalSegment::Media(media) => TokenizedMultimodalSegment::Media(media.clone()),
            });
        }
        Ok(TokenizedMultimodalRequest { segments })
    }
}

/// One exact rendered-prompt placeholder bound to decoded media.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaBinding {
    placeholder: String,
    media: Media,
}

impl MediaBinding {
    /// Binds decoded media to one complete placeholder spelling.
    pub fn new(placeholder: impl Into<String>, media: Media) -> Self {
        Self {
            placeholder: placeholder.into(),
            media,
        }
    }

    /// Complete placeholder spelling.
    pub fn placeholder(&self) -> &str {
        &self.placeholder
    }

    /// Bound decoded media.
    pub const fn media(&self) -> &Media {
        &self.media
    }
}

/// One ordered segment after facade tokenization and before backend preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenizedMultimodalSegment {
    /// Tokenizer vocabulary IDs.
    TokenIds(Vec<u32>),
    /// Decoded media.
    Media(Media),
}

/// Backend preparation request with tokenizer work already complete.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedMultimodalRequest {
    segments: Vec<TokenizedMultimodalSegment>,
}

impl TokenizedMultimodalRequest {
    /// Ordered token and media segments.
    pub fn segments(&self) -> &[TokenizedMultimodalSegment] {
        &self.segments
    }
}

fn validate_bindings(
    rendered_prompt: &str,
    bindings: &[MediaBinding],
) -> Result<(), MediaRequestError> {
    for (index, binding) in bindings.iter().enumerate() {
        if binding.placeholder.is_empty() {
            return Err(MediaRequestError::EmptyPlaceholder { index });
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
        let actual = rendered_prompt.matches(&binding.placeholder).count();
        if actual != expected {
            return Err(MediaRequestError::PlaceholderCount {
                placeholder: binding.placeholder.clone(),
                expected,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(value: u8) -> Media {
        Media::Image(RgbImage::new(vec![value; 3], 1, 1).unwrap())
    }

    #[test]
    fn chat_composition_and_tokenization_preserve_exact_order() {
        let request = MultimodalRequest::from_chat(
            "before<image>middle<image>after",
            &[
                MediaBinding::new("<image>", image(1)),
                MediaBinding::new("<image>", image(2)),
            ],
        )
        .unwrap();
        let tokenized = request
            .tokenize::<std::convert::Infallible>(|text| {
                Ok(text
                    .as_bytes()
                    .iter()
                    .map(|byte| u32::from(*byte))
                    .collect())
            })
            .unwrap();
        assert_eq!(tokenized.segments().len(), 5);
        assert!(matches!(
            &tokenized.segments()[0],
            TokenizedMultimodalSegment::TokenIds(ids)
                if ids == &[98, 101, 102, 111, 114, 101]
        ));
        assert!(matches!(
            &tokenized.segments()[1],
            TokenizedMultimodalSegment::Media(Media::Image(image)) if image.pixels() == [1, 1, 1]
        ));
        assert!(matches!(
            &tokenized.segments()[3],
            TokenizedMultimodalSegment::Media(Media::Image(image)) if image.pixels() == [2, 2, 2]
        ));
    }

    #[test]
    fn validation_rejects_bad_media_and_placeholder_contracts() {
        assert!(matches!(
            RgbImage::new(vec![0; 2], 1, 1),
            Err(MediaRequestError::InvalidRgbImage { .. })
        ));
        assert!(Audio::new(vec![f32::NAN], 16_000).is_err());
        assert!(Video::new(Vec::new(), None, VideoSampling::All).is_err());
        assert!(matches!(
            MultimodalRequest::from_chat(
                "<image>",
                &[
                    MediaBinding::new("<image>", image(1)),
                    MediaBinding::new("<image>", image(2)),
                ],
            ),
            Err(MediaRequestError::PlaceholderCount { .. })
        ));
    }
}
