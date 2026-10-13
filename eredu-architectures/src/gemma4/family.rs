//! Composite Gemma 4 decoder and media configuration.

use std::collections::BTreeSet;

use eredu_core::InputModalities;
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
    quantization: Option<serde_json::Value>,
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
    /// Declared outer family-wrapper identity (`gemma4` or `gemma4_unified`).
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

/// Identifies projector-only media metadata in unified text conversions.
/// Full tower configurations and malformed non-object values remain subject
/// to ordinary geometry validation.
pub(crate) fn is_media_config_stub(value: &serde_json::Value) -> bool {
    value.as_object().is_some_and(|object| {
        !object.contains_key("num_hidden_layers")
            && matches!(
                object.get("model_type").and_then(serde_json::Value::as_str),
                Some("gemma4_unified_vision" | "gemma4_unified_audio")
            )
    })
}

/// Splits a Hugging Face `quantization` object into the default weight
/// format and the per-tensor overrides that mixed-precision conversions carry
/// as nested objects keyed by physical tensor name without the `.weight`
/// suffix (for example `language_model.model.layers.3.mlp.down_proj`). Each
/// override inherits the fields it omits (typically `mode`) from the default
/// and is re-keyed by the canonical `model.*.weight` name the checkpoint plan
/// resolves formats with.
fn split_quantization(
    value: serde_json::Value,
) -> Result<
    (
        serde_json::Value,
        serde_json::Map<String, serde_json::Value>,
    ),
    FamilyConfigError,
> {
    let serde_json::Value::Object(entries) = value else {
        return Ok((value, serde_json::Map::new()));
    };
    let mut default = serde_json::Map::new();
    let mut nested = Vec::new();
    for (key, entry) in entries {
        match entry {
            serde_json::Value::Object(_) => nested.push((key, entry)),
            serde_json::Value::Bool(false) => {
                return Err(FamilyConfigError::Invalid(format!(
                    "Gemma 4 quantization marks {key:?} as unquantized; per-tensor dense overrides are not supported"
                )))
            }
            serde_json::Value::Bool(true) => {}
            other => {
                default.insert(key, other);
            }
        }
    }
    let mut overrides = serde_json::Map::new();
    for (key, entry) in nested {
        let serde_json::Value::Object(fields) = entry else {
            unreachable!("only objects are collected");
        };
        let mut merged = default.clone();
        merged.extend(fields);
        let canonical = crate::gemma4::config::canonical_weight_name(&key);
        if overrides
            .insert(canonical.clone(), serde_json::Value::Object(merged))
            .is_some()
        {
            return Err(FamilyConfigError::Invalid(format!(
                "duplicate Gemma 4 quantization override for {canonical:?}"
            )));
        }
    }
    Ok((serde_json::Value::Object(default), overrides))
}

impl FamilyConfig {
    /// Returns the nested text implementation identity preserved at admission.
    pub fn effective_model_type(&self) -> &str {
        &self.text.model_type
    }

    /// Whether this configuration uses the unified multimodal family identity.
    pub fn is_unified(&self) -> bool {
        self.model_type == "gemma4_unified"
    }

    /// Returns the input modalities admitted by this exact family variant.
    pub const fn input_modalities(&self) -> InputModalities {
        InputModalities {
            text: true,
            image: self.image_token_id.is_some(),
            audio: self.audio_token_id.is_some(),
            video: self.video_token_id.is_some(),
        }
    }

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
            "tie_word_embeddings".into(),
            serde_json::Value::Bool(source.tie_word_embeddings),
        );
        if let Some(quantization) = source.quantization {
            let (default, overrides) = split_quantization(quantization)?;
            text_object.insert("weight_quantization".into(), default);
            if !overrides.is_empty() {
                text_object.insert(
                    "quantized_weight_configs".into(),
                    serde_json::Value::Object(overrides),
                );
            }
        }
        let text = ModelArgs::from_hf_json(&serde_json::to_vec(&text_value)?)?;
        let vision_stub = source
            .vision_config
            .as_ref()
            .is_some_and(is_media_config_stub);
        let audio_stub = source
            .audio_config
            .as_ref()
            .is_some_and(is_media_config_stub);
        let vision = source
            .vision_config
            .filter(|_| !vision_stub)
            .map(|value| -> Result<VisionConfig, FamilyConfigError> {
                let mut config: VisionConfig = serde_json::from_value(value)?;
                // Media-to-text projections use the conversion's decoder encoding.
                // Store it in the media configuration so admission and construction
                // consume the same normalized format.
                let projection = "model.embed_vision.embedding_projection.weight";
                if let Some(format) = text.linear_format_for(projection).weight_quantization() {
                    config
                        .quantized_weight_configs
                        .get_or_insert_default()
                        .entry(projection.into())
                        .or_insert(format);
                }
                for (name, format) in text.quantized_weight_configs.iter().flatten() {
                    if name.starts_with("model.vision_tower.") || name == projection {
                        config
                            .quantized_weight_configs
                            .get_or_insert_default()
                            .insert(name.clone(), *format);
                    }
                }
                config.validate()?;
                Ok(config)
            })
            .transpose()?;
        let audio = source
            .audio_config
            .filter(|_| !audio_stub)
            .map(|value| -> Result<AudioConfig, FamilyConfigError> {
                let mut config: AudioConfig = serde_json::from_value(value)?;
                // Media-to-text projections use the conversion's decoder encoding.
                // Store it in the media configuration so admission and construction
                // consume the same normalized format.
                let projection = "model.embed_audio.embedding_projection.weight";
                if let Some(format) = text.linear_format_for(projection).weight_quantization() {
                    config
                        .quantized_weight_configs
                        .get_or_insert_default()
                        .entry(projection.into())
                        .or_insert(format);
                }
                for (name, format) in text.quantized_weight_configs.iter().flatten() {
                    if name.starts_with("model.audio_tower.") || name == projection {
                        config
                            .quantized_weight_configs
                            .get_or_insert_default()
                            .insert(name.clone(), *format);
                    }
                }
                config.validate()?;
                Ok(config)
            })
            .transpose()?;
        // A text-only conversion keeps the placeholder ids of the towers it
        // reduced to stubs; without an encoder they admit nothing, so they are
        // cleared rather than reported as a modality the artifact cannot serve.
        // A placeholder with no media configuration at all is still an orphan
        // and is rejected by `validate`.
        let (image_token_id, video_token_id) = if vision_stub {
            (None, None)
        } else {
            (source.image_token_id, source.video_token_id)
        };
        let audio_token_id = if audio_stub {
            None
        } else {
            source.audio_token_id
        };
        let config = Self {
            model_type: source.model_type,
            text,
            vision,
            image_token_id,
            video_token_id,
            audio,
            audio_token_id,
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
                    "vision_quantization",
                    self.vision.as_ref().map_or_else(
                        || "none".into(),
                        |config| {
                            format!(
                                "default={:?};weights={};overrides={}",
                                config.weight_quantization,
                                crate::cache_identity::string_set(
                                    config.quantized_weights.as_ref()
                                ),
                                crate::cache_identity::debug_map(
                                    config.quantized_weight_configs.as_ref()
                                )
                            )
                        },
                    ),
                ),
                (
                    "audio_quantization",
                    self.audio.as_ref().map_or_else(
                        || "none".into(),
                        |config| {
                            format!(
                                "default={:?};weights={};overrides={}",
                                config.weight_quantization,
                                crate::cache_identity::string_set(
                                    config.quantized_weights.as_ref()
                                ),
                                crate::cache_identity::debug_map(
                                    config.quantized_weight_configs.as_ref()
                                )
                            )
                        },
                    ),
                ),
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

    #[test]
    fn per_tensor_quantization_overrides_reach_the_text_formats() {
        let mut value = config();
        value["quantization"] = serde_json::json!({
            "group_size": 64, "bits": 4, "mode": "affine",
            "language_model.model.layers.0.mlp.down_proj": {"bits": 8, "group_size": 64},
            "model.language_model.layers.1.mlp.up_proj": {"bits": 8, "group_size": 32},
            "language_model.model.embed_tokens": {"bits": 6, "group_size": 64}
        });
        let family = FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        let bits = |name: &str| {
            family
                .text
                .linear_format_for(name)
                .weight_quantization()
                .expect("quantized")
                .bits()
        };
        assert_eq!(bits("model.layers.0.mlp.down_proj.weight"), 8);
        assert_eq!(bits("model.layers.1.mlp.up_proj.weight"), 8);
        assert_eq!(bits("model.embed_tokens.weight"), 6);
        assert_eq!(bits("model.layers.0.mlp.up_proj.weight"), 4);
        assert_eq!(bits("model.layers.0.self_attn.q_proj.weight"), 4);
        let mut value = config();
        value["quantization"] = serde_json::json!({
            "group_size": 64, "bits": 4,
            "language_model.model.layers.0.mlp.down_proj": false
        });
        assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn conversion_media_formats_preserve_dense_towers_and_quantized_projections() {
        let mut value = config();
        value["vision_config"]["hidden_size"] = 32.into();
        value["vision_config"]["num_attention_heads"] = 4.into();
        value["audio_config"]["output_proj_dims"] = 32.into();
        value["quantization"] = serde_json::json!({
            "group_size": 32, "bits": 4,
            "embed_audio.embedding_projection": {"bits": 8},
            "vision_tower.encoder.layers.0.mlp.fc1.linear": {"bits": 8}
        });
        let family = FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        let vision = family.vision.unwrap();
        let audio = family.audio.unwrap();
        assert_eq!(
            vision
                .linear_format_for("model.embed_vision.embedding_projection.weight", 32)
                .weight_quantization()
                .unwrap()
                .bits(),
            4
        );
        assert_eq!(
            audio
                .linear_format_for("model.embed_audio.embedding_projection.weight", 32)
                .weight_quantization()
                .unwrap()
                .bits(),
            8
        );
        assert_eq!(
            vision
                .linear_format_for(
                    "model.vision_tower.encoder.layers.0.mlp.fc1.linear.weight",
                    32
                )
                .weight_quantization()
                .unwrap()
                .bits(),
            8
        );
        assert_eq!(
            vision.linear_format_for(
                "model.vision_tower.encoder.layers.0.mlp.fc2.linear.weight",
                32
            ),
            eredu_checkpoint::LinearFormat::Dense
        );
    }

    #[test]
    fn rejects_ambiguous_quantization_and_malformed_media_geometry() {
        let mut value = config();
        value["quantization"] = serde_json::json!({
            "group_size": 32, "bits": 4,
            "language_model.model.layers.0.mlp.down_proj": {"bits": 8},
            "model.layers.0.mlp.down_proj.weight": {"bits": 4}
        });
        assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
        for malformed in [
            serde_json::json!(false),
            serde_json::json!({}),
            serde_json::json!({"model_type":"gemma4_vision","hidden_size":16}),
        ] {
            let mut value = config();
            value["vision_config"] = malformed;
            assert!(FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
        }
    }

    #[test]
    fn text_only_conversion_stub_media_configs_are_treated_as_absent() {
        // mlx-community's text-only 4-bit conversions keep `vision_config` /
        // `audio_config` with projector metadata only (no `num_hidden_layers`)
        // and the placeholder ids of the towers they dropped.
        let mut value = config();
        let object = value.as_object_mut().unwrap();
        object.insert(
            "vision_config".into(),
            serde_json::json!({"model_type":"gemma4_unified_vision","mm_embed_dim":16,"rms_norm_eps":0.000001}),
        );
        object.insert(
            "audio_config".into(),
            serde_json::json!({"model_type":"gemma4_unified_audio","hidden_size":16,"output_proj_dims":8,"rms_norm_eps":0.000001}),
        );
        let family = FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(family.vision.is_none());
        assert!(family.audio.is_none());
        assert_eq!(family.image_token_id, None);
        assert_eq!(family.video_token_id, None);
        assert_eq!(family.audio_token_id, None);
        assert!(!family.input_modalities().image);
        assert!(!family.input_modalities().audio);
        assert!(!is_media_config_stub(&config()["vision_config"]));
    }

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
    fn family_fingerprint_includes_component_quantization() {
        let parsed = FamilyConfig::from_hf_json(&serde_json::to_vec(&config()).unwrap()).unwrap();
        let dense = parsed.architecture_fingerprint();
        let format = eredu_checkpoint::AffineQuantization::new(16, 4)
            .unwrap()
            .into();

        let mut vision_quantized = parsed.clone();
        vision_quantized
            .vision
            .as_mut()
            .unwrap()
            .weight_quantization = Some(format);
        assert_ne!(dense, vision_quantized.architecture_fingerprint());

        let mut audio_quantized = parsed;
        audio_quantized.audio.as_mut().unwrap().weight_quantization = Some(format);
        assert_ne!(dense, audio_quantized.architecture_fingerprint());
    }

    #[test]
    fn text_only_variant_does_not_advertise_media_modalities() {
        let mut value = config();
        let object = value.as_object_mut().unwrap();
        object.remove("image_token_id");
        object.remove("vision_config");
        object.remove("audio_token_id");
        object.remove("audio_config");
        let parsed = FamilyConfig::from_hf_json(&serde_json::to_vec(&value).unwrap()).unwrap();
        assert_eq!(parsed.input_modalities(), InputModalities::TEXT);
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
