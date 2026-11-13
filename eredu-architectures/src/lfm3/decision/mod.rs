//! Portable decision admission, question packing, and readouts for d1.

use std::{collections::BTreeMap, ops::Range, path::Path, sync::Arc};

use eredu_core::decision::{ChoiceAnswer, ChoiceQuestion, DecisionAnswer, DecisionQuestion};
use eredu_text::tokenizer::Tokenizer;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

mod audio;
pub use audio::{DecisionAudioProcessor, PreparedDecisionAudio};
mod boxes;
pub(crate) mod gguf;
pub use gguf::DecisionGgufLayouts;
mod parsing;
mod pointer;
mod questions;
mod readout;
use readout::Readout;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const QUESTION_HEADER: &str = "QUESTION:\n";
const MAX_TRAINED_SET_SIZE: usize = 128;

/// Failure to admit a d1 artifact, question set, or answer row.
#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    /// An exact local companion could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// Path of the companion.
        path: std::path::PathBuf,
        /// File access error.
        source: std::io::Error,
    },
    /// Retained checkpoint weights do not match the architecture.
    #[error(transparent)]
    Weights(#[from] eredu_nn::Error),
    /// Required checkpoint metadata or calibration is invalid.
    #[error("invalid d1 declaration: {0}")]
    Declaration(&'static str),
    /// An input cannot be packed or a readout cannot be interpreted.
    #[error("invalid d1 request or readout: {0}")]
    Request(&'static str),
    /// A checkpoint JSON document is invalid.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The checkpoint tokenizer could not be loaded or used.
    #[error("d1 tokenizer: {0}")]
    Tokenizer(String),
}

fn read(root: &Path, name: &str) -> Result<Vec<u8>, DecisionError> {
    let path = root.join(name);
    std::fs::read(&path).map_err(|source| DecisionError::Read { path, source })
}

/// A retained decision model with its exact decoder source.
pub struct PreparedTextChoiceModel {
    preparation: TextChoicePreparation,
    decoder: super::vision::PreparedVlDecoder,
    vision: Option<PreparedDecisionVision>,
    audio: Option<PreparedDecisionAudio>,
}

impl PreparedTextChoiceModel {
    /// Admits the publisher's SafeTensors schema and sidecars.
    pub fn open(
        root: &Path,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
        max_cached_shards: usize,
    ) -> Result<Self, DecisionError> {
        Self::open_internal(root, dtype, max_cached_shards, None, false)
    }

    /// Admits the decoder and the publisher's independently bound vision component.
    pub fn open_with_images(
        root: &Path,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
        max_cached_shards: usize,
    ) -> Result<Self, DecisionError> {
        Self::open_with_vision(root, dtype, max_cached_shards)
    }

    /// Admits the decoder and shared image/video encoder with independent weight bindings.
    pub fn open_with_vision(
        root: &Path,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
        max_cached_shards: usize,
    ) -> Result<Self, DecisionError> {
        Self::open_internal(
            root,
            dtype,
            max_cached_shards,
            Some(&read(root, "processor_config.json")?),
            false,
        )
    }

    /// Admits the retained decoder, visual components and Omni speech input encoder.
    pub fn open_with_omni(
        root: &Path,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
        max_cached_shards: usize,
    ) -> Result<Self, DecisionError> {
        Self::open_internal(
            root,
            dtype,
            max_cached_shards,
            Some(&read(root, "processor_config.json")?),
            true,
        )
    }

    /// Transfers independently retained decoder, vision and speech input bindings.
    pub fn into_omni_parts(
        self,
    ) -> Result<
        (
            TextChoicePreparation,
            super::vision::PreparedVlDecoder,
            PreparedDecisionVision,
            PreparedDecisionAudio,
        ),
        DecisionError,
    > {
        let audio = self.audio.ok_or(DecisionError::Request(
            "decision checkpoint was opened without speech input",
        ))?;
        let vision = self.vision.ok_or(DecisionError::Request(
            "decision checkpoint was opened without vision",
        ))?;
        Ok((self.preparation, self.decoder, vision, audio))
    }

    /// Transfers all retained input components for task-aware binding.
    pub fn into_execution_parts(
        self,
    ) -> (
        TextChoicePreparation,
        super::vision::PreparedVlDecoder,
        Option<PreparedDecisionVision>,
        Option<PreparedDecisionAudio>,
    ) {
        (self.preparation, self.decoder, self.vision, self.audio)
    }

    fn open_internal(
        root: &Path,
        dtype: eredu_checkpoint::recipe::RecipeDtype,
        max_cached_shards: usize,
        processor: Option<&[u8]>,
        prepare_audio: bool,
    ) -> Result<Self, DecisionError> {
        let config_bytes = read(root, "config.json")?;
        let config: Value = serde_json::from_slice(&config_bytes)?;
        let preparation = TextChoicePreparation::from_sidecars(
            &config_bytes,
            &read(root, "tokenizer.json")?,
            &read(root, "tokenizer_config.json")?,
            &read(root, "calibration.json")?,
        )?;
        let omni_tokens = if config["model_type"] == "lfm3_omni" {
            let bytes = match processor {
                Some(bytes) => std::borrow::Cow::Borrowed(bytes),
                None => std::borrow::Cow::Owned(read(root, "processor_config.json")?),
            };
            Some(super::omni::vision_tokens(
                &config,
                &serde_json::from_slice(&bytes)?,
                &preparation.tokenizer,
            )?)
        } else {
            None
        };
        let audio_processor = if prepare_audio {
            let tokens = omni_tokens.ok_or(DecisionError::Declaration(
                "speech input requires a d1 Omni checkpoint",
            ))?;
            Some(audio::DecisionAudioProcessor::new(
                &config,
                &serde_json::from_slice(processor.ok_or(DecisionError::Declaration(
                    "d1 Omni processor metadata is required",
                ))?)?,
                &preparation.tokenizer,
                tokens,
            )?)
        } else {
            None
        };
        let (decoder, vision, audio) = super::vision::VisionLanguageDeclaration::prepare_decision(
            root,
            &config,
            dtype,
            max_cached_shards,
            processor,
            omni_tokens,
            audio_processor,
        )?;
        Ok(Self {
            preparation,
            decoder,
            vision,
            audio,
        })
    }

    /// Transfers independently retained decoder and vision bindings.
    pub fn into_image_parts(
        self,
    ) -> Result<
        (
            TextChoicePreparation,
            super::vision::PreparedVlDecoder,
            PreparedDecisionVision,
        ),
        DecisionError,
    > {
        let vision = self.vision.ok_or(DecisionError::Request(
            "decision checkpoint was opened without vision",
        ))?;
        Ok((self.preparation, self.decoder, vision))
    }

    /// Transfers the retained question policy and decoder bindings to an executable adapter.
    pub fn into_parts(self) -> (TextChoicePreparation, super::vision::PreparedVlDecoder) {
        (self.preparation, self.decoder)
    }
}

/// Retained media declarations and the publisher's image-size limit.
pub struct PreparedDecisionVision {
    pub(super) vision: super::vision::PreparedVisionExecution,
    pub(super) processor: crate::processor_execution::PreparedProcessor,
    pub(super) admission: super::vision::VisionPromptAdmission,
    pub(super) max_pixels: u64,
    pub(super) tokens: super::vision::MediaTokens,
    pub(super) width: i32,
}

impl PreparedDecisionVision {
    /// Transfers the processor, projection, token geometry, and pixel limit.
    pub fn into_parts(
        self,
    ) -> (
        super::vision::PreparedVisionExecution,
        crate::processor_execution::PreparedProcessor,
        super::vision::VisionPromptAdmission,
        u64,
        super::vision::MediaTokens,
        i32,
    ) {
        (
            self.vision,
            self.processor,
            self.admission,
            self.max_pixels,
            self.tokens,
            self.width,
        )
    }
}

/// Applies the publisher's aspect-preserving bicubic image cap before VL processing.
#[cfg(feature = "image")]
pub fn cap_image(
    image: &eredu_core::RgbImage,
    max_pixels: u64,
) -> Result<eredu_core::RgbImage, DecisionError> {
    let area = u64::from(image.width()) * u64::from(image.height());
    if max_pixels == 0 {
        return Err(DecisionError::Declaration(
            "image pixel cap must be positive",
        ));
    }
    if area <= max_pixels {
        return Ok(image.clone());
    }
    let scale = (max_pixels as f64 / area as f64).sqrt();
    let width = ((f64::from(image.width()) * scale) as u32).max(1);
    let height = ((f64::from(image.height()) * scale) as u32).max(1);
    let view =
        eredu_media::image::RgbImageView::packed(image.pixels(), image.width(), image.height())
            .map_err(eredu_nn::Error::backend_source)?;
    let resized = eredu_media::image::resize_rgb8_antialiased_bicubic(view, width, height)
        .map_err(eredu_nn::Error::backend_source)?;
    Ok(
        eredu_core::RgbImage::new(resized.pixels().to_vec(), width, height)
            .map_err(eredu_nn::Error::backend_source)?,
    )
}

/// Whether the checkpoint declares the d1 decision task.
pub fn is_d1_config(config: &Value) -> bool {
    matches!(config["model_type"].as_str(), Some("lfm3_vl" | "lfm3_omni"))
        && config["architectures"] == serde_json::json!(["D1Model"])
}

/// Retained tokenizer, question-set policy and calibration for d1.
pub struct TextChoicePreparation {
    tokenizer: Tokenizer,
    special_tokens: Vec<String>,
    bos: String,
    context: usize,
    vocab: usize,
    set_size: usize,
    numeric_codes: bool,
    box_prompt: usize,
    interpolate_edges: bool,
    box_calibration: [Arc<[[f64; 2]]>; 4],
    estimate_levels: [f64; 3],
    calibration: Calibration,
    calibration_sha256: [u8; 32],
    multiple_calibration: Arc<[[f64; 2]]>,
    found_calibration: [Arc<[[f64; 2]]>; 3],
}

#[derive(Deserialize)]
struct Calibration {
    default: f64,
    temperatures: BTreeMap<String, f64>,
    #[serde(default)]
    tables: BTreeMap<String, Vec<[f64; 2]>>,
}

impl Calibration {
    fn temperature(&self, bucket: &str) -> f64 {
        self.temperatures
            .get(bucket)
            .copied()
            .unwrap_or(self.default)
    }

    fn choice_temperature(&self, count: usize) -> f64 {
        let bucket = match count {
            0..=3 => "choice:small",
            4..=8 => "choice:mid",
            9..=26 => "choice:large",
            _ => "choice:inventory",
        };
        self.temperature(bucket)
    }
}

impl TextChoicePreparation {
    /// Retains the exact decision sidecars without opening model weights.
    pub fn open(root: &Path) -> Result<Self, DecisionError> {
        Self::from_sidecars(
            &read(root, "config.json")?,
            &read(root, "tokenizer.json")?,
            &read(root, "tokenizer_config.json")?,
            &read(root, "calibration.json")?,
        )
    }

    /// Admits the publisher's tokenizer and calibrated question-set declaration.
    pub fn from_sidecars(
        config: &[u8],
        tokenizer: &[u8],
        tokenizer_config: &[u8],
        calibration: &[u8],
    ) -> Result<Self, DecisionError> {
        let config: Value = serde_json::from_slice(config)?;
        if !is_d1_config(&config) || config["d1_attention"] != "set" || config["d1_codes"] != 2 {
            return Err(DecisionError::Declaration(
                "expected a d1 checkpoint with set attention and code scheme 2",
            ));
        }
        let number = |value: &Value| {
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
        };
        let set_size = number(&config["d1_set_size"])
            .filter(|&n| n <= MAX_TRAINED_SET_SIZE)
            .ok_or(DecisionError::Declaration("d1_set_size"))?;
        let context = number(&config["text_config"]["max_position_embeddings"])
            .ok_or(DecisionError::Declaration("text context length"))?;
        let vocab = number(&config["text_config"]["vocab_size"])
            .ok_or(DecisionError::Declaration("vocabulary size"))?;
        let tokenizer = Tokenizer::from_bytes(tokenizer)
            .map_err(|error| DecisionError::Tokenizer(error.to_string()))?;
        let tokenizer_config: Value = serde_json::from_slice(tokenizer_config)?;
        let bos_id = if config["model_type"] == "lfm3_omni" {
            config["text_config"]["bos_token_id"].as_u64()
        } else {
            config["bos_token_id"].as_u64()
        };
        let bos = tokenizer_config["bos_token"]
            .as_str()
            .filter(|token| {
                tokenizer
                    .token_to_id(token)
                    .is_some_and(|id| Some(u64::from(id)) == bos_id)
            })
            .ok_or(DecisionError::Declaration("BOS token identity"))?
            .to_owned();
        let mut special_tokens: Vec<_> = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| {
                token.special
                    && token.content.chars().count() > 1
                    && !token.content.contains('\u{200b}')
            })
            .map(|token| token.content)
            .collect();
        special_tokens.sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let calibration_sha256 = Sha256::digest(calibration).into();
        let calibration: Calibration = serde_json::from_slice(calibration)?;
        if std::iter::once(&calibration.default)
            .chain(calibration.temperatures.values())
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(DecisionError::Declaration(
                "calibration temperature must be finite and positive",
            ));
        }
        let table = |name: &str| -> Result<Arc<[[f64; 2]]>, DecisionError> {
            let knots = calibration.tables.get(name).map_or(&[][..], Vec::as_slice);
            if knots
                .iter()
                .flatten()
                .any(|value| !(0.0..=1.0).contains(value))
                || knots.windows(2).any(|pair| pair[0][0] >= pair[1][0])
            {
                return Err(DecisionError::Declaration("invalid calibration knots"));
            }
            Ok(Arc::from(knots))
        };
        let multiple_calibration = table("multiple")?;
        let found_calibration = [
            table("found:text")?,
            table("found:number")?,
            table("found:date")?,
        ];
        let box_calibration = [
            table("found:box")?,
            table("box")?,
            table("found:boxes")?,
            table("boxes")?,
        ];
        let box_prompt = match config.get("d1_prompt") {
            None => 1,
            Some(value) => value
                .as_u64()
                .filter(|&value| matches!(value, 1 | 2))
                .ok_or(DecisionError::Declaration("invalid d1 box prompt version"))?
                as usize,
        };
        let interpolate_edges = match config.get("d1_box_edges").and_then(Value::as_str) {
            None | Some("median") => false,
            Some("interp") => true,
            Some(_) => return Err(DecisionError::Declaration("invalid d1 box edge policy")),
        };
        let estimate = table("estimate")?;
        let inverse: Vec<_> = estimate
            .iter()
            .map(|&[raw, observed]| [observed, raw])
            .collect();
        if inverse.windows(2).any(|pair| pair[0][0] >= pair[1][0]) {
            return Err(DecisionError::Declaration(
                "invalid estimate calibration knots",
            ));
        }
        let estimate_levels = [0.1, 0.5, 0.9].map(|level| readout::interpolate(&inverse, level));
        Ok(Self {
            tokenizer,
            special_tokens,
            bos,
            context,
            vocab,
            set_size,
            box_prompt,
            interpolate_edges,
            box_calibration,
            numeric_codes: config["d1_numbers"].as_bool().unwrap_or(false),
            estimate_levels,
            calibration,
            calibration_sha256,
            multiple_calibration,
            found_calibration,
        })
    }

    /// Exact SHA-256 of the admitted calibration companion.
    pub fn calibration_sha256(&self) -> [u8; 32] {
        self.calibration_sha256
    }

    /// Packs questions in order, repeating the state at the checkpoint's set limit.
    pub fn pack(
        &self,
        state: Option<&str>,
        questions: &[ChoiceQuestion],
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        self.pack_questions(state, questions, ("", 0), |q| self.single_choice(q))
    }

    /// Packs categorical, multiple-choice, binary and ordinal questions in one request.
    pub fn pack_decisions(
        &self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        self.pack_questions(state, questions, ("", 0), |q| self.decision(q))
    }

    /// Packs the state after validated, expanded image markup from the vision path.
    pub fn pack_image_decisions(
        &self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
        expanded_ids: &[i32],
        markers: super::vision::MediaTokens,
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        if expanded_ids.contains(&markers.video_token_id) {
            return Err(DecisionError::Request("expected expanded d1 image markers"));
        }
        self.pack_visual_decisions(state, questions, expanded_ids, markers)
    }

    /// Forms the publisher's outer video framing around shared VL frame expansion.
    #[cfg(feature = "image")]
    pub fn visual_segments(
        &self,
        visuals: &[eredu_core::Media],
        markers: super::vision::MediaTokens,
        max_pixels: u64,
    ) -> Result<Vec<eredu_core::MultimodalSegment>, DecisionError> {
        use eredu_core::{Media, MultimodalSegment as Segment};
        let mut segments = Vec::new();
        for item in visuals {
            match item {
                Media::Image(image) => {
                    segments.push(Segment::Media(Media::Image(cap_image(image, max_pixels)?)))
                }
                Media::Video(video) => {
                    segments.extend([
                        Segment::TokenIds(vec![markers.vision_start_token_id as u32]),
                        Segment::Media(Media::Video(video.clone())),
                        Segment::TokenIds(vec![markers.vision_end_token_id as u32]),
                    ]);
                }
                Media::Audio(_) => {
                    return Err(DecisionError::Request("d1 VL takes image and video state"))
                }
            }
        }
        Ok(segments)
    }

    /// Encodes processor-generated video timestamps with the admitted tokenizer.
    pub fn encode_visual_text(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        self.encode(text)
    }

    /// Packs decisions after the shared VL processor's ordered image and video expansion.
    pub fn pack_visual_decisions(
        &self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
        expanded_ids: &[i32],
        markers: super::vision::MediaTokens,
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        self.pack_media_decisions(state, questions, expanded_ids, markers, None)
    }

    /// Packs shared state with interleaved expanded image, video and audio markup.
    pub fn pack_media_decisions(
        &self,
        state: Option<&str>,
        questions: &[DecisionQuestion],
        expanded_ids: &[i32],
        markers: super::vision::MediaTokens,
        audio: Option<DecisionAudioProcessor>,
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        let (markup, images) = self.media_markup(expanded_ids, markers, audio)?;
        self.pack_questions(state, questions, (&markup, images), |q| self.decision(q))
    }

    /// Packs question-local spoken instructions after the shared media state.
    pub fn pack_spoken_decisions(
        &self,
        state: Option<&str>,
        questions: &[(&DecisionQuestion, Option<&[i32]>)],
        expanded_ids: &[i32],
        markers: super::vision::MediaTokens,
        audio: DecisionAudioProcessor,
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        let (markup, images) = if expanded_ids.is_empty() {
            (String::new(), 0)
        } else {
            self.media_markup(expanded_ids, markers, Some(audio))?
        };
        self.pack_questions(
            state,
            questions,
            (&markup, images),
            |(question, voice)| match voice {
                Some(ids) if audio.is_framed_clip(ids) => {
                    self.spoken_decision(question, &self.round_trip_markup(ids)?)
                }
                Some(_) => Err(DecisionError::Request("expected one framed spoken clip")),
                None => self.decision(question),
            },
        )
    }

    fn media_markup(
        &self,
        expanded_ids: &[i32],
        markers: super::vision::MediaTokens,
        audio: Option<DecisionAudioProcessor>,
    ) -> Result<(String, usize), DecisionError> {
        let error = || DecisionError::Request("expected expanded d1 media markers");
        if expanded_ids.is_empty() {
            return Err(error());
        }
        let mut images = 0;
        let mut remaining = expanded_ids;
        let open = markers.vision_start_token_id;
        let close = markers.vision_end_token_id;
        let framed_rows = |tokens: &[i32], marker: i32| {
            if tokens.first() != Some(&open) {
                return None;
            }
            let count = tokens[1..].iter().take_while(|&&id| id == marker).count();
            (count > 0 && tokens.get(count + 1) == Some(&close)).then_some(count + 2)
        };
        while !remaining.is_empty() {
            if let Some(count) = framed_rows(remaining, markers.image_token_id) {
                images += 1;
                remaining = &remaining[count..];
                continue;
            }
            if let Some(count) = audio.and_then(|audio| framed_rows(remaining, audio.audio_token()))
            {
                remaining = &remaining[count..];
                continue;
            }
            if remaining.first() != Some(&open) {
                return Err(error());
            }
            remaining = &remaining[1..];
            let mut groups = 0;
            while remaining.first() != Some(&close) {
                let at = remaining
                    .iter()
                    .position(|&id| id == open || id == close)
                    .ok_or_else(error)?;
                let timestamp = self.decode_visual_ids(&remaining[..at])?;
                let seconds = timestamp
                    .strip_prefix('<')
                    .and_then(|text| text.strip_suffix(" seconds>"))
                    .and_then(|text| text.parse::<f64>().ok());
                if !seconds.is_some_and(|value| value.is_finite() && value >= 0.0) {
                    return Err(error());
                }
                let count =
                    framed_rows(&remaining[at..], markers.video_token_id).ok_or_else(error)?;
                remaining = &remaining[at + count..];
                groups += 1;
            }
            if groups == 0 {
                return Err(error());
            }
            remaining = &remaining[1..];
        }
        Ok((self.round_trip_markup(expanded_ids)?, images))
    }

    fn round_trip_markup(&self, expanded_ids: &[i32]) -> Result<String, DecisionError> {
        let markup = self.decode_visual_ids(expanded_ids)?;
        if !self
            .encode(&markup)?
            .into_iter()
            .map(|id| id as i32)
            .eq(expanded_ids.iter().copied())
        {
            return Err(DecisionError::Request(
                "expanded media markup does not round-trip",
            ));
        }
        Ok(markup)
    }

    fn decode_visual_ids(&self, ids: &[i32]) -> Result<String, DecisionError> {
        let ids = ids.iter().map(|&id| id as u32).collect::<Vec<_>>();
        self.tokenizer
            .decode(&ids, false)
            .map_err(|error| DecisionError::Tokenizer(error.to_string()))
    }

    fn pack_questions<Q>(
        &self,
        state: Option<&str>,
        questions: &[Q],
        (markup, images): (&str, usize),
        prepare: impl Fn(&Q) -> Result<QuestionPlan, DecisionError>,
    ) -> Result<Vec<PackedChoiceSet>, DecisionError> {
        if questions.is_empty() {
            return Err(DecisionError::Request("a question set cannot be empty"));
        }
        let header = state.map_or("", |_| QUESTION_HEADER);
        let plans = questions
            .iter()
            .map(prepare)
            .collect::<Result<Vec<_>, _>>()?;
        if images != 1
            && plans
                .iter()
                .any(|plan| matches!(plan.readout, Readout::Box(_)))
        {
            return Err(DecisionError::Request(
                "a box question requires exactly one image",
            ));
        }
        let mut source = None;
        let mut prefix = format!("{}{IM_START}user\n{markup}", self.bos);
        if let Some(state) = state {
            if plans.iter().any(|plan| plan.readout.is_pointer()) {
                let mut boundaries: Vec<_> = (0..=state.len()).collect();
                let escaped = self.escape(state, |at| {
                    let original = boundaries[at];
                    boundaries.splice(at..at, [original; 3]);
                });
                source = Some((state, prefix.len(), boundaries));
                prefix.push_str(&escaped);
            } else {
                prefix.push_str(&self.plain(state));
            }
            prefix.push_str("\n\n\n");
            prefix.push_str(header);
        }
        let mut rows = Vec::new();
        let mut plans = plans.into_iter().peekable();
        while plans.peek().is_some() {
            rows.push(self.pack_row(
                &prefix,
                header,
                plans.by_ref().take(self.set_size),
                source.as_ref(),
            )?);
        }
        Ok(rows)
    }

    fn pack_row(
        &self,
        prefix: &str,
        header: &str,
        questions: impl Iterator<Item = QuestionPlan>,
        source: Option<&(&str, usize, Vec<usize>)>,
    ) -> Result<PackedChoiceSet, DecisionError> {
        let mut tokens = Vec::new();
        let mut slots = Vec::new();
        let mut positions = Vec::new();
        let mut question_ends = Vec::new();
        let mut state_len = 0;
        let mut pointer = None;
        for (index, question) in questions.enumerate() {
            let QuestionPlan {
                suffix,
                offsets,
                readout,
            } = question;
            let mut question_start = tokens.len();
            if index == 0 {
                let prompt = format!("{prefix}{suffix}");
                tokens = if let Some((text, start, boundaries)) = source {
                    let encoded = self
                        .tokenizer
                        .encode(prompt.as_str(), false)
                        .map_err(|error| DecisionError::Tokenizer(error.to_string()))?;
                    let positions = pointer::StatePositions::new(
                        text,
                        &prompt,
                        *start,
                        boundaries,
                        encoded.get_offsets(),
                    );
                    if positions.width() > self.vocab {
                        return Err(DecisionError::Request(
                            "pointer positions exceed the readout width",
                        ));
                    }
                    pointer = Some(positions);
                    self.admit_ids(encoded.get_ids())?
                } else {
                    self.encode(&prompt)?
                };
                let suffix_len = self.encode(&suffix)?.len();
                state_len = tokens
                    .len()
                    .checked_sub(suffix_len)
                    .ok_or(DecisionError::Request(
                        "question suffix exceeds the packed row",
                    ))?;
                question_start = state_len;
            } else {
                tokens.extend(self.encode(&format!("{header}{suffix}"))?);
            }
            if tokens.len() > self.context {
                return Err(DecisionError::Request(
                    "the packed row exceeds the model context",
                ));
            }
            if offsets.last() != Some(&0)
                || offsets.windows(2).any(|pair| pair[0] <= pair[1])
                || offsets
                    .iter()
                    .any(|&offset| offset >= tokens.len() - question_start)
            {
                return Err(DecisionError::Request(
                    "answer slots lie outside their question",
                ));
            }
            positions.extend(offsets.into_iter().map(|offset| tokens.len() - 1 - offset));
            question_ends.push(tokens.len());
            slots.push(readout);
        }
        Ok(PackedChoiceSet {
            tokens,
            state_len,
            question_ends,
            positions,
            slots,
            vocab: self.vocab,
            pointer,
            found_calibration: self.found_calibration.clone(),
        })
    }
}

struct QuestionPlan {
    suffix: String,
    offsets: Vec<usize>,
    readout: Readout,
}

impl QuestionPlan {
    fn single(suffix: String, readout: Readout) -> Self {
        Self {
            suffix,
            offsets: vec![0],
            readout,
        }
    }
}

/// One packed row with state/question boundaries and ordered answer slots.
pub struct PackedChoiceSet {
    tokens: Vec<u32>,
    state_len: usize,
    question_ends: Vec<usize>,
    positions: Vec<usize>,
    slots: Vec<Readout>,
    vocab: usize,
    pointer: Option<pointer::StatePositions>,
    found_calibration: [Arc<[[f64; 2]]>; 3],
}

impl PackedChoiceSet {
    /// Number of questions packed in this row, regardless of their slot counts.
    pub fn question_count(&self) -> usize {
        self.question_ends.len()
    }

    /// Input token IDs in packed order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of leading tokens belonging to the shared state.
    pub fn state_len(&self) -> usize {
        self.state_len
    }

    /// Vocabulary readout positions in question order.
    pub fn answer_positions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }

    /// Key visibility for one block of global-attention queries, query-major.
    ///
    /// State queries see the whole row. Question queries see state and their
    /// own question. Sliding attention and convolution retain causal semantics.
    pub fn global_visibility(&self, queries: Range<usize>) -> Result<Vec<bool>, DecisionError> {
        if queries.start > queries.end
            || queries.end > self.tokens.len()
            || queries.len().checked_mul(self.tokens.len()).is_none()
        {
            return Err(DecisionError::Request(
                "invalid global attention query range",
            ));
        }
        Ok(queries
            .flat_map(|query| {
                let question = self.question_ends.partition_point(|&end| end <= query);
                let start = if question == 0 {
                    self.state_len
                } else {
                    self.question_ends[question - 1]
                };
                let end = self.question_ends[question];
                (0..self.tokens.len()).map(move |key| {
                    query < self.state_len || key < self.state_len || (start..end).contains(&key)
                })
            })
            .collect())
    }

    /// Interprets one full vocabulary-logit row per slot for single-choice requests.
    pub fn choices(&self, logits: &[&[f32]]) -> Result<Vec<ChoiceAnswer>, DecisionError> {
        self.decisions(logits)?
            .into_iter()
            .map(|answer| match answer {
                DecisionAnswer::SingleChoice(answer) => Ok(answer),
                _ => Err(DecisionError::Request(
                    "single-choice readout requires single-choice questions",
                )),
            })
            .collect()
    }

    /// Interprets scores in packed slot order.
    ///
    /// Rows have vocabulary width. Pointer rows store position scores at the
    /// start of the row, with unused columns ignored.
    pub fn decisions(&self, logits: &[&[f32]]) -> Result<Vec<DecisionAnswer>, DecisionError> {
        if logits.len() != self.positions.len() || logits.iter().any(|row| row.len() != self.vocab)
        {
            return Err(DecisionError::Request(
                "expected one vocabulary-width score row per answer slot",
            ));
        }
        let mut logits = logits.iter().copied();
        self.slots
            .iter()
            .map(|readout| {
                readout.answer(&mut logits, self.pointer.as_ref(), &self.found_calibration)
            })
            .collect()
    }
}

/// Tensor input and retained question-set boundaries for one complete forward pass.
pub struct DecisionPrompt<'a, T> {
    pub(super) tokens: T,
    pub(super) layout: QuestionSetLayout,
    projected: Option<super::projected_prompt::ProjectedPrompt<'a, T, 3>>,
}

impl<'a, T: eredu_nn::Tensor> DecisionPrompt<'a, T> {
    /// Converts the packed input without introducing padding or changing position IDs.
    pub fn new(packed: &PackedChoiceSet, context: &T::Context) -> Result<Self, eredu_nn::Error> {
        let tokens = packed
            .tokens
            .iter()
            .copied()
            .map(i32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(eredu_nn::Error::backend_source)?;
        let length = i32::try_from(tokens.len()).map_err(eredu_nn::Error::backend_source)?;
        let state_len = packed.state_len as i32;
        let question_ends = packed.question_ends.iter().map(|&end| end as i32).collect();
        Ok(Self {
            tokens: T::from_i32_slice(&tokens, &[1, length], context)?,
            projected: None,
            layout: QuestionSetLayout {
                state_len,
                question_ends,
                answer_positions: packed
                    .positions
                    .iter()
                    .map(|&position| position as i32)
                    .collect(),
                length,
                pointer_width: packed
                    .pointer
                    .as_ref()
                    .map_or(1, |state| state.width() as i32),
                pointer_slots: packed
                    .slots
                    .iter()
                    .flat_map(|slot| std::iter::repeat_n(slot.is_pointer(), slot.slot_count()))
                    .collect(),
            },
        })
    }

    /// Binds the image rows at their exact marker positions in the packed state.
    pub fn new_with_images(
        packed: &PackedChoiceSet,
        marker: i32,
        width: i32,
        rows: &'a T,
        context: &T::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::new_projected(
            packed,
            [marker, -1, -2],
            width,
            [Some(rows), None, None],
            context,
        )
    }

    /// Binds image and video projections to their respective markers in the packed state.
    pub fn new_with_visuals(
        packed: &PackedChoiceSet,
        markers: super::vision::MediaTokens,
        width: i32,
        rows: [Option<&'a T>; 2],
        context: &T::Context,
    ) -> Result<Self, eredu_nn::Error> {
        let [image, video] = rows;
        Self::new_with_media(packed, markers, None, width, [image, video, None], context)
    }

    /// Binds image, video and audio projections to their exact packed marker positions.
    pub fn new_with_media(
        packed: &PackedChoiceSet,
        markers: super::vision::MediaTokens,
        audio: Option<DecisionAudioProcessor>,
        width: i32,
        rows: [Option<&'a T>; 3],
        context: &T::Context,
    ) -> Result<Self, eredu_nn::Error> {
        Self::new_projected(
            packed,
            [
                markers.image_token_id,
                markers.video_token_id,
                audio.map_or(-1, DecisionAudioProcessor::audio_token),
            ],
            width,
            rows,
            context,
        )
    }

    fn new_projected(
        packed: &PackedChoiceSet,
        markers: [i32; 3],
        width: i32,
        rows: [Option<&'a T>; 3],
        context: &T::Context,
    ) -> Result<Self, eredu_nn::Error> {
        let mut prompt = Self::new(packed, context)?;
        let ids = packed
            .tokens
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<_>>();
        let projected =
            super::projected_prompt::ProjectedPrompt::new(&ids, markers, width, rows, context)?;
        prompt.tokens = projected.tokens().clone();
        prompt.projected = Some(projected);
        Ok(prompt)
    }

    pub(super) fn projected_roots(&self) -> impl Iterator<Item = &T> {
        self.projected
            .iter()
            .flat_map(|prompt| prompt.projected_roots())
    }

    pub(super) fn update_pipeline_digest(
        &self,
        digest: &mut Sha256,
        context: &T::Context,
    ) -> Result<(), eredu_nn::Error> {
        self.layout.update_pipeline_digest(digest);
        digest.update([u8::from(self.projected.is_some())]);
        if let Some(projected) = &self.projected {
            projected.update_pipeline_digest(digest, context)?;
        }
        Ok(())
    }

    pub(super) fn condition(&self, hidden: T, context: &T::Context) -> Result<T, eredu_nn::Error> {
        match &self.projected {
            Some(prompt) => prompt.replace(&hidden, context),
            None => Ok(hidden),
        }
    }
}

#[derive(Clone)]
pub(super) struct QuestionSetLayout {
    state_len: i32,
    question_ends: Vec<i32>,
    answer_positions: Vec<i32>,
    length: i32,
    pub(super) pointer_width: i32,
    pub(super) pointer_slots: Vec<bool>,
}

impl QuestionSetLayout {
    pub(super) fn slot_count(&self) -> i32 {
        self.answer_positions.len() as i32
    }

    fn update_pipeline_digest(&self, digest: &mut Sha256) {
        digest.update(self.state_len.to_le_bytes());
        digest.update(self.length.to_le_bytes());
        digest.update(self.pointer_width.to_le_bytes());
        for positions in [&self.question_ends, &self.answer_positions] {
            digest.update((positions.len() as u64).to_le_bytes());
            for position in positions {
                digest.update(position.to_le_bytes());
            }
        }
        for &pointer in &self.pointer_slots {
            digest.update([u8::from(pointer)]);
        }
    }

    pub(super) fn attention_spans(
        &self,
    ) -> impl Iterator<Item = (Range<i32>, Vec<Range<i32>>)> + '_ {
        (self.state_len > 0)
            .then(|| (0..self.state_len, std::iter::once(0..self.length).collect()))
            .into_iter()
            .chain(
                self.question_ends
                    .iter()
                    .scan(self.state_len, |start, &end| {
                        let own = *start..end;
                        let keys = if *start == self.state_len {
                            std::iter::once(0..end).collect()
                        } else {
                            (self.state_len > 0)
                                .then_some(0..self.state_len)
                                .into_iter()
                                .chain(std::iter::once(own.clone()))
                                .collect()
                        };
                        *start = end;
                        Some((own, keys))
                    }),
            )
    }

    pub(super) fn gather<T: eredu_nn::Tensor>(
        &self,
        hidden: &T,
        context: &T::Context,
    ) -> Result<T, eredu_nn::Error> {
        let slots = &self.answer_positions;
        let indices = T::from_i32_slice(slots, &[slots.len() as i32], context)?;
        hidden.take_axis(&indices, 1, context)
    }
}
