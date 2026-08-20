//! Composite Gemma 4 decoder and media configuration.

use std::collections::BTreeSet;

use eredu_checkpoint::WeightQuantization;
use serde::Deserialize;

use super::{
    AudioConfig, AudioConfigError, ConfigError, ModelArgs, VisionConfig, VisionConfigError,
};

/// Invalid top-level Gemma configuration.
#[derive(Debug, thiserror::Error)]
pub enum FamilyConfigError {
    /// Top-level JSON decoding failed.
    #[error("invalid Gemma 4 family configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Nested text configuration failed validation.
    #[error(transparent)]
    Text(#[from] ConfigError),
    /// Nested vision configuration failed validation.
    #[error(transparent)]
    Vision(#[from] VisionConfigError),
    /// Nested audio configuration failed validation.
    #[error(transparent)]
    Audio(#[from] AudioConfigError),
    /// Cross-component policy is inconsistent.
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Deserialize)]
struct FamilySource {
    #[serde(default = "default_model_type")]
    model_type: String,
    text_config: serde_json::Value,
    #[serde(default)]
    vision_config: Option<serde_json::Value>,
    #[serde(default)]
    image_token_id: Option<i32>,
    #[serde(default)]
    video_token_id: Option<i32>,
    #[serde(default)]
    audio_config: Option<serde_json::Value>,
    #[serde(default)]
    audio_token_id: Option<i32>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
}

fn default_model_type() -> String {
    "gemma4".into()
}

const fn default_true() -> bool {
    true
}

/// One normalized Gemma text/vision/audio family configuration.
#[derive(Debug, Clone)]
pub struct FamilyConfig {
    /// Stable architecture identity (`gemma4` or `gemma4_unified`).
    pub model_type: String,
    /// Normalized text decoder.
    pub text: ModelArgs,
    /// Optional image/video encoder.
    pub vision: Option<VisionConfig>,
    /// Image placeholder token.
    pub image_token_id: Option<i32>,
    /// Video placeholder token.
    pub video_token_id: Option<i32>,
    /// Optional audio encoder.
    pub audio: Option<AudioConfig>,
    /// Audio placeholder token.
    pub audio_token_id: Option<i32>,
}

impl FamilyConfig {
    /// Parses and validates one top-level Hugging Face configuration.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, FamilyConfigError> {
        let source: FamilySource = serde_json::from_slice(bytes)?;
        if !matches!(source.model_type.as_str(), "gemma4" | "gemma4_unified") {
            return Err(FamilyConfigError::Invalid(format!(
                "unsupported Gemma 4 family model_type {:?}",
                source.model_type
            )));
        }
        let mut text_value = source.text_config;
        let text_object = text_value.as_object_mut().ok_or_else(|| {
            FamilyConfigError::Invalid("Gemma 4 text_config must be an object".into())
        })?;
        text_object.insert(
            "model_type".into(),
            serde_json::Value::String(source.model_type.clone()),
        );
        text_object.insert(
            "tie_word_embeddings".into(),
            serde_json::Value::Bool(source.tie_word_embeddings),
        );
        if let Some(quantization) = source.quantization {
            text_object.insert(
                "weight_quantization".into(),
                serde_json::to_value(quantization)?,
            );
        }
        let text = ModelArgs::from_hf_json(&serde_json::to_vec(&text_value)?)?;
        let vision = source
            .vision_config
            .map(|value| -> Result<VisionConfig, FamilyConfigError> {
                let config: VisionConfig = serde_json::from_value(value)?;
                config.validate()?;
                Ok(config)
            })
            .transpose()?;
        let audio = source
            .audio_config
            .map(|value| -> Result<AudioConfig, FamilyConfigError> {
                let config: AudioConfig = serde_json::from_value(value)?;
                config.validate()?;
                Ok(config)
            })
            .transpose()?;
        let config = Self {
            model_type: source.model_type,
            text,
            vision,
            image_token_id: source.image_token_id,
            video_token_id: source.video_token_id,
            audio,
            audio_token_id: source.audio_token_id,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates placeholder ranges, collisions, and modality availability.
    pub fn validate(&self) -> Result<(), FamilyConfigError> {
        let mut tokens = BTreeSet::new();
        for (name, token) in [
            ("image", self.image_token_id),
            ("video", self.video_token_id),
            ("audio", self.audio_token_id),
        ] {
            if let Some(token) = token {
                if token < 0 || token >= self.text.vocab_size {
                    return Err(FamilyConfigError::Invalid(format!(
                        "Gemma 4 {name} placeholder {token} is outside the vocabulary"
                    )));
                }
                if !tokens.insert(token) {
                    return Err(FamilyConfigError::Invalid(format!(
                        "Gemma 4 {name} placeholder {token} collides with another modality"
                    )));
                }
            }
        }
        if self.vision.is_some() && self.image_token_id.is_none() && self.video_token_id.is_none() {
            return Err(FamilyConfigError::Invalid(
                "Gemma 4 vision configuration has no image or video placeholder".into(),
            ));
        }
        if self.vision.is_none() && (self.image_token_id.is_some() || self.video_token_id.is_some())
        {
            return Err(FamilyConfigError::Invalid(
                "Gemma 4 visual placeholder requires a vision configuration".into(),
            ));
        }
        if self.audio.is_some() != self.audio_token_id.is_some() {
            return Err(FamilyConfigError::Invalid(
                "Gemma 4 audio configuration and placeholder must be declared together".into(),
            ));
        }
        Ok(())
    }

    /// Stable fingerprint including text schedule and cross-component geometry.
    pub fn architecture_fingerprint(&self) -> String {
        let vision = self.vision.as_ref().map_or_else(
            || "none".into(),
            |config| {
                format!(
                    "{}:{}:{}:{}:{}:{}",
                    config.hidden_size,
                    config.num_hidden_layers,
                    config.num_attention_heads,
                    config.head_dim,
                    config.patch_size,
                    config.pooling_kernel_size
                )
            },
        );
        let audio = self.audio.as_ref().map_or_else(
            || "none".into(),
            |config| {
                format!(
                    "{}:{}:{}:{}:{}:{}:{:?}",
                    config.hidden_size,
                    config.num_hidden_layers,
                    config.num_attention_heads,
                    config.output_proj_dims,
                    config.conv_kernel_size,
                    config.attention_chunk_size,
                    config.subsampling_conv_channels
                )
            },
        );
        eredu_core::cache::derive_prompt_cache_architecture_fingerprint(
            "gemma4-family",
            [
                ("model_type", self.model_type.clone()),
                ("text", self.text.architecture_fingerprint()),
                ("vision", vision),
                ("audio", audio),
                (
                    "tokens",
                    format!(
                        "{:?}:{:?}:{:?}",
                        self.image_token_id, self.video_token_id, self.audio_token_id
                    ),
                ),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> serde_json::Value {
        serde_json::json!({
            "model_type":"gemma4_unified",
            "tie_word_embeddings":false,
            "image_token_id":60,
            "audio_token_id":61,
            "text_config":{
                "model_type":"gemma4_text","hidden_size":16,"num_hidden_layers":2,
                "intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,
                "head_dim":8,"rms_norm_eps":0.000001,"vocab_size":64,
                "max_position_embeddings":128,"layer_types":["full_attention","full_attention"]
            },
            "vision_config":{
                "hidden_size":16,"intermediate_size":32,"num_hidden_layers":1,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,
                "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
                "rms_norm_eps":0.000001
            },
            "audio_config":{
                "hidden_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                "output_proj_dims":8,"conv_kernel_size":3,"attention_chunk_size":4,
                "attention_context_left":5,"attention_context_right":0,
                "attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,
                "residual_weight":0.5,"rms_norm_eps":0.000001,
                "subsampling_conv_channels":[4,8]
            }
        })
    }

    #[test]
    fn normalizes_nested_family_and_freezes_all_component_identity() {
        let parsed = FamilyConfig::from_hf_json(&serde_json::to_vec(&config()).unwrap()).unwrap();
        assert_eq!(parsed.model_type, "gemma4_unified");
        assert_eq!(parsed.text.model_type, "gemma4_unified");
        assert!(!parsed.text.tie_word_embeddings);
        assert!(parsed.vision.is_some());
        assert!(parsed.audio.is_some());
        assert!(!parsed.architecture_fingerprint().is_empty());
    }

    #[test]
    fn rejects_placeholder_collisions_and_orphan_modalities() {
        let mut value = config();
        value["audio_token_id"] = value["image_token_id"].clone();
        assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value = config();
        value.as_object_mut().unwrap().remove("vision_config");
        assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
