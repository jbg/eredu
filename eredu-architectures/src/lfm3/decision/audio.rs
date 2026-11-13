//! d1 Omni speech input admission and processor policy.

use eredu_checkpoint::{
    recipe::RecipeDtype, store::SharedCheckpointSource, validation::ResolvedCheckpointPlan,
};
use eredu_nn::Error;
use eredu_text::tokenizer::Tokenizer;
use serde_json::Value;

use crate::lfm3::{
    audio::{Declaration, PreparedAudioEncoder, ProcessorConfig},
    omni::marker_id,
    vision::MediaTokens,
};

/// Processor policy and independently bindable speech input parameters.
pub struct PreparedDecisionAudio {
    encoder: PreparedAudioEncoder,
    processor: DecisionAudioProcessor,
}

impl PreparedDecisionAudio {
    pub(in crate::lfm3) fn new(
        encoder: Declaration,
        resolution: &ResolvedCheckpointPlan,
        source: SharedCheckpointSource,
        processor: DecisionAudioProcessor,
        dtype: RecipeDtype,
    ) -> Result<Self, Error> {
        Ok(Self {
            encoder: PreparedAudioEncoder::from_safetensors(encoder, resolution, source, dtype)?,
            processor,
        })
    }

    pub(in crate::lfm3) fn from_gguf(
        encoder: Declaration,
        source: SharedCheckpointSource,
        processor: DecisionAudioProcessor,
        dtype: RecipeDtype,
    ) -> Result<Self, Error> {
        let recipes = crate::lfm3::audio::gguf::input_encoder_recipes(&encoder, source.as_ref())
            .map_err(Error::backend_source)?;
        Ok(Self {
            encoder: PreparedAudioEncoder::from_recipes(
                encoder,
                source,
                "d1 Omni GGUF speech input",
                recipes,
                dtype,
            )?,
            processor,
        })
    }

    /// Transfers retained parameters and processor policy to an execution adapter.
    pub fn into_parts(self) -> (PreparedAudioEncoder, DecisionAudioProcessor) {
        (self.encoder, self.processor)
    }
}

/// PCM feature and token framing policy for a d1 Omni clip.
#[derive(Clone, Copy)]
pub struct DecisionAudioProcessor {
    features: ProcessorConfig,
    audio_token: i32,
    start: i32,
    end: i32,
}

impl DecisionAudioProcessor {
    pub(super) fn new(
        config: &Value,
        processor: &Value,
        tokenizer: &Tokenizer,
        visual: MediaTokens,
    ) -> Result<Self, Error> {
        let audio_token = marker_id(processor, tokenizer, "audio_token")?;
        if processor["audio_markers"] != true
            || config["audio_token_id"] != audio_token
            || !config["text_config"]["vocab_size"]
                .as_i64()
                .is_some_and(|size| i64::from(audio_token) < size)
            || [
                visual.image_token_id,
                visual.video_token_id,
                visual.vision_start_token_id,
                visual.vision_end_token_id,
            ]
            .contains(&audio_token)
        {
            return Err(Error::backend(
                "d1 Omni audio framing disagrees with tokenizer",
            ));
        }
        Ok(Self {
            features: ProcessorConfig::from_feature_extractor_class(
                &processor["feature_extractor"],
                "Lfm3OmniFeatureExtractor",
            )?,
            audio_token,
            start: visual.vision_start_token_id,
            end: visual.vision_end_token_id,
        })
    }

    /// Decoder token identifying one projected audio row.
    pub const fn audio_token(self) -> i32 {
        self.audio_token
    }

    pub(super) fn is_framed_clip(self, ids: &[i32]) -> bool {
        matches!(ids, [start, rows @ .., end] if *start == self.start
            && *end == self.end && !rows.is_empty()
            && rows.iter().all(|&id| id == self.audio_token))
    }

    /// Computes checkpoint log-mel input and the clip's framed placeholder IDs.
    #[cfg(feature = "audio")]
    pub fn prepare(
        self,
        clip: &eredu_core::Audio,
    ) -> Result<(crate::lfm3::audio::LogMelFeatures, Vec<i32>), Error> {
        use eredu_media::audio::{resample_hann_sinc, AudioWaveform};

        let pcm = AudioWaveform::new(clip.samples(), clip.sample_rate())
            .map_err(Error::backend_source)?;
        let rate = self.features.sample_rate();
        let samples = resample_hann_sinc(pcm, rate).map_err(Error::backend_source)?;
        let waveform = AudioWaveform::new(&samples, rate).map_err(Error::backend_source)?;
        let features = self
            .features
            .transform()
            .and_then(|transform| transform.extract(waveform))
            .map_err(Error::backend_source)?;
        let tokens = self.framed_tokens(features.frames)?;
        Ok((features, tokens))
    }

    /// Builds the exact framed placeholder sequence for a finite mel clip.
    pub fn framed_tokens(self, frames: usize) -> Result<Vec<i32>, Error> {
        let positions = self.features.audio_positions(frames)?;
        if positions == 0 {
            return Err(Error::backend(
                "d1 Omni audio requires at least one feature frame",
            ));
        }
        let mut tokens = Vec::with_capacity(positions + 2);
        tokens.push(self.start);
        tokens.resize(positions + 1, self.audio_token);
        tokens.push(self.end);
        Ok(tokens)
    }
}
