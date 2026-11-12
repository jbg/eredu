//! Admission of the LFM3 Omni multimodal checkpoint.

use std::path::Path;

use eredu_checkpoint::recipe::RecipeDtype;
use eredu_nn::{Error, Tensor};
use serde_json::Value;

use super::{
    audio::{ArtifactError, BindingSource, PreparedAudio, ProcessorConfig},
    projected_prompt::ProjectedPrompt,
    vision::{Declaration, MediaTokens, PreparedVisionExecution, ProjectorConfig},
};
use crate::{
    processor_execution::PreparedProcessor,
    processor_plan::{MediaFraming, QwenPatchPlan, QwenProcessorPlan},
};

fn checkpoint(error: impl std::fmt::Display) -> ArtifactError {
    ArtifactError::Checkpoint(error.to_string())
}

/// Retained Omni checkpoint with shared language, speech and vision sources.
pub struct PreparedOmni {
    speech: PreparedAudio,
    vision: Declaration,
    vision_weights: BindingSource,
    processor: QwenProcessorPlan,
    tokens: MediaTokens,
}

impl PreparedOmni {
    /// Validates the released SafeTensors checkpoint and bundled LiquidCodec.
    pub fn open(root: &Path, max_cached_shards: usize) -> Result<Self, ArtifactError> {
        fn read(root: &Path, name: &str) -> Result<Vec<u8>, ArtifactError> {
            let path = root.join(name);
            std::fs::read(&path).map_err(|source| ArtifactError::Read { path, source })
        }

        let config: Value = serde_json::from_slice(&read(root, "config.json")?)?;
        if config["model_type"] != "lfm3_omni"
            || config["architectures"] != serde_json::json!(["Lfm3OmniForConditionalGeneration"])
        {
            return Err(checkpoint("expected an LFM3 Omni generative checkpoint"));
        }
        let vision = Declaration::new(
            serde_json::from_value(config["vision_config"].clone())?,
            ProjectorConfig::from_config_value(&config)?,
            "model.vision_tower",
            "model.multi_modal_projector",
        )?;
        let processor_bytes = read(root, "processor_config.json")?;
        let source: Value = serde_json::from_slice(&processor_bytes)?;
        if source["vision_markers"] != true || source["audio_markers"] != true {
            return Err(checkpoint(
                "expected framed Omni image, video and audio inputs",
            ));
        }
        let audio_processor = ProcessorConfig::from_feature_extractor_class(
            &source["feature_extractor"],
            "Lfm3OmniFeatureExtractor",
        )?;
        let (speech, vision_weights) = PreparedAudio::open_safetensors(
            root,
            root,
            &config,
            max_cached_shards,
            Some((&vision.safetensors_plan()?, audio_processor)),
        )?;
        let tokenizer = eredu_text::tokenizer::Tokenizer::from_bytes(&speech.text_assets.tokenizer)
            .map_err(checkpoint)?;
        let marker = |key: &str| {
            source[key]
                .as_str()
                .and_then(|token| tokenizer.token_to_id(token))
                .and_then(|id| i32::try_from(id).ok())
                .filter(|id| *id < speech.language.args().vocab_size)
                .ok_or_else(|| checkpoint(format!("invalid Omni processor marker {key}")))
        };
        let tokens = MediaTokens {
            image_token_id: marker("image_token")?,
            video_token_id: marker("video_token")?,
            vision_start_token_id: marker("vision_start_token")?,
            vision_end_token_id: marker("vision_end_token")?,
        };
        let mut ids = [
            tokens.image_token_id,
            tokens.video_token_id,
            tokens.vision_start_token_id,
            tokens.vision_end_token_id,
            speech.tokens.input,
            speech.tokens.output,
            speech.tokens.audio_start,
            speech.tokens.text_end,
            speech.tokens.mixed_start,
        ];
        ids.sort_unstable();
        if config["image_token_id"] != tokens.image_token_id
            || config["video_token_id"] != tokens.video_token_id
            || marker("audio_token")? != speech.tokens.input
            || marker("audio_out_token")? != speech.tokens.output
            || ids.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(checkpoint(
                "Omni model and processor media markers disagree",
            ));
        }
        let processor = QwenProcessorPlan::from_lfm3_vl(
            MediaFraming {
                start_token_id: tokens.vision_start_token_id as u32,
                end_token_id: tokens.vision_end_token_id as u32,
            },
            &processor_bytes,
        )
        .map_err(checkpoint)?;
        let geometry = vision.config();
        processor
            .validate_patches(QwenPatchPlan {
                patch_size: geometry.patch_size as usize,
                temporal_patch_size: geometry.temporal_patch_size as usize,
                merge_size: geometry.spatial_merge_size as usize,
            })
            .map_err(checkpoint)?;
        Ok(Self {
            speech,
            vision,
            vision_weights: vision_weights.expect("vision plan was supplied"),
            processor,
            tokens,
        })
    }

    /// Retained language, audio and codec declarations and their source views.
    pub fn speech(&self) -> &PreparedAudio {
        &self.speech
    }

    /// Image and video token identities from the admitted tokenizer.
    pub fn media_tokens(&self) -> MediaTokens {
        self.tokens
    }

    /// Image and video processor policy retained from the publisher.
    pub fn vision_processor(&self) -> PreparedProcessor {
        PreparedProcessor::from_qwen(self.processor.clone())
    }

    /// Prepares the same independent vision units as VL from the retained Omni source.
    pub fn prepare_vision(
        &self,
        dtype: RecipeDtype,
    ) -> Result<PreparedVisionExecution, ArtifactError> {
        Ok(PreparedVisionExecution::new(
            self.vision.clone(),
            self.vision_weights.source(),
            dtype,
        )?)
    }
}

/// Projected language-width rows in expanded prompt order within each modality.
pub struct OmniProjectedRows<'a, T> {
    /// Image projections.
    pub image: Option<&'a T>,
    /// Video projections.
    pub video: Option<&'a T>,
    /// Input audio encoder projections.
    pub audio: Option<&'a T>,
    /// Speech frame feedback embeddings.
    pub feedback: Option<&'a T>,
}

/// Output required from a shared Omni decoder invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OmniReadout {
    /// Token vocabulary scores.
    Vocabulary,
    /// Normalized hidden state for speech prediction and vocabulary projection.
    NormalizedHidden,
}

/// Validated, sliceable input for joint vision and speech conditioning.
pub struct OmniPrompt<'a, T> {
    prompt: ProjectedPrompt<'a, T, 4>,
    readout: OmniReadout,
}

impl<'a, T: Tensor> OmniPrompt<'a, T> {
    /// Admits expanded token IDs and exact projected rows using this artifact's markers.
    pub fn new(
        artifact: &PreparedOmni,
        ids: &[i32],
        rows: OmniProjectedRows<'a, T>,
        readout: OmniReadout,
        context: &T::Context,
    ) -> Result<Self, Error> {
        let args = artifact.speech.language.args();
        if ids.iter().any(|&id| !(0..args.vocab_size).contains(&id)) {
            return Err(Error::backend(
                "Omni prompt contains an out-of-vocabulary token",
            ));
        }
        let vision = artifact.tokens;
        let audio = artifact.speech.tokens;
        let prompt = ProjectedPrompt::new(
            ids,
            [
                vision.image_token_id,
                vision.video_token_id,
                audio.input,
                audio.output,
            ],
            args.hidden_size,
            [rows.image, rows.video, rows.audio, rows.feedback],
            context,
        )?;
        Ok(Self { prompt, readout })
    }

    /// Slices decoder positions without renumbering projected media rows.
    pub fn slice(&self, range: std::ops::Range<i32>, context: &T::Context) -> Result<Self, Error> {
        Ok(Self {
            prompt: self.prompt.slice(range, context)?,
            readout: self.readout,
        })
    }

    /// Readout selected for this invocation and all of its slices.
    pub fn readout(&self) -> OmniReadout {
        self.readout
    }

    /// Native tensor roots retained through completion of this invocation.
    pub fn retained_tensors(&self) -> impl Iterator<Item = &T> {
        std::iter::once(self.tokens()).chain(self.prompt.projected_roots())
    }

    /// Describes the tokens, projected rows, and output required on every rank.
    pub fn collective_descriptor(&self, context: &T::Context) -> Result<[u64; 8], Error> {
        use sha2::{Digest, Sha256};

        let mut digest = Sha256::new();
        digest.update([match self.readout {
            OmniReadout::Vocabulary => 0,
            OmniReadout::NormalizedHidden => 1,
        }]);
        for dimension in self.tokens().shape() {
            digest.update(dimension.to_le_bytes());
        }
        for id in self.tokens().to_i32_vec(context)? {
            digest.update(id.to_le_bytes());
        }
        self.prompt.update_pipeline_digest(&mut digest, context)?;
        let mut descriptor = [0x4c464d334f4d4e49, 0, 0, 0, 0, 0, 0, 0];
        for (word, bytes) in descriptor[1..]
            .iter_mut()
            .zip(digest.finalize().chunks_exact(8))
        {
            *word = u64::from_le_bytes(bytes.try_into().expect("eight digest bytes"));
        }
        Ok(descriptor)
    }

    pub(super) fn tokens(&self) -> &T {
        self.prompt.tokens()
    }

    pub(super) fn replace(&self, text: &T, context: &T::Context) -> Result<T, Error> {
        self.prompt.replace(text, context)
    }
}
