//! Backend-neutral Gemma 4 sibling-projector metadata policy.

use std::collections::HashMap;

use crate::rotary::RopeValue;
use eredu_gguf::{Checkpoint, MetadataArray, MetadataValue};

use super::{AudioConfig, FamilyConfig, FamilyConfigError, ModelArgs, VisionConfig};

/// Translates a released projector tensor name into the neutral parameter tree.
pub fn translate_mmproj_weight_name(name: &str) -> String {
    for (source, target) in [
        ("vision_tower", "model.vision_tower"),
        ("embed_vision", "model.embed_vision"),
        ("audio_tower", "model.audio_tower"),
        ("embed_audio", "model.embed_audio"),
    ] {
        if name == source || name.starts_with(&format!("{source}.")) {
            return name.replacen(source, target, 1);
        }
    }
    name.to_owned()
}

/// Combines a parsed text GGUF with a validated optional sibling projector.
pub fn family_from_gguf_metadata(
    text: ModelArgs,
    model: &HashMap<String, MetadataValue>,
    projector: Option<&HashMap<String, MetadataValue>>,
) -> Result<FamilyConfig, FamilyConfigError> {
    family_from_projector(text, model, projector, None)
}

pub(crate) fn family_from_gguf_projector(
    text: ModelArgs,
    model: &HashMap<String, MetadataValue>,
    metadata: &HashMap<String, MetadataValue>,
    checkpoint: &Checkpoint,
) -> Result<FamilyConfig, FamilyConfigError> {
    family_from_projector(text, model, Some(metadata), Some(checkpoint))
}

fn family_from_projector(
    text: ModelArgs,
    model: &HashMap<String, MetadataValue>,
    projector: Option<&HashMap<String, MetadataValue>>,
    checkpoint: Option<&Checkpoint>,
) -> Result<FamilyConfig, FamilyConfigError> {
    let Some(projector) = projector else {
        let family = super::family::FamilyPolicy {
            model_type: text.model_type.clone(),
            text,
            vision: None,
            image_token_id: None,
            video_token_id: None,
            audio: None,
            audio_token_id: None,
        };
        return FamilyConfig::new(family);
    };
    validate_projector_identity(projector)?;
    let has_vision = optional_bool(projector, "clip.has_vision_encoder")?.unwrap_or(false);
    let has_audio = optional_bool(projector, "clip.has_audio_encoder")?.unwrap_or(false);
    let vision = has_vision
        .then(|| vision_config(projector, checkpoint))
        .transpose()?;
    let audio = has_audio
        .then(|| audio_config(projector, checkpoint))
        .transpose()?;
    let family = super::family::FamilyPolicy {
        model_type: text.model_type.clone(),
        image_token_id: has_vision
            .then(|| media_token(model, "image", "<|image|>"))
            .transpose()?
            .flatten(),
        video_token_id: has_vision
            .then(|| media_token(model, "video", "<|video|>"))
            .transpose()?
            .flatten(),
        audio_token_id: has_audio
            .then(|| media_token(model, "audio", "<|audio|>"))
            .transpose()?
            .flatten(),
        text,
        vision,
        audio,
    };
    FamilyConfig::new(family)
}

/// Validates the closed sibling-projector architecture and component flags.
pub fn validate_projector_identity(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<(), FamilyConfigError> {
    let architecture = required_string(metadata, "general.architecture")?;
    if architecture != "clip" {
        return Err(invalid(format!(
            "expected Gemma 4 projector architecture \"clip\", got {architecture:?}"
        )));
    }
    let has_vision = optional_bool(metadata, "clip.has_vision_encoder")?.unwrap_or(false);
    let has_audio = optional_bool(metadata, "clip.has_audio_encoder")?.unwrap_or(false);
    if !has_vision && !has_audio {
        return Err(invalid(
            "Gemma 4 projector must contain a vision encoder, an audio encoder, or both",
        ));
    }
    for (enabled, key, component, published) in [
        (
            has_vision,
            "clip.vision.projector_type",
            "vision",
            "gemma4v",
        ),
        (has_audio, "clip.audio.projector_type", "audio", "gemma4a"),
    ] {
        if enabled {
            let kind = required_string(metadata, key)?;
            if kind != "gemma4" && kind != published {
                return Err(invalid(format!(
                    "Gemma 4 {component} projector type is unsupported: {kind:?}"
                )));
            }
        }
    }
    Ok(())
}

fn vision_config(
    metadata: &HashMap<String, MetadataValue>,
    checkpoint: Option<&Checkpoint>,
) -> Result<VisionConfig, FamilyConfigError> {
    let hidden = required_i32(metadata, "clip.vision.embedding_length")?;
    let heads = required_i32(metadata, "clip.vision.attention.head_count")?;
    let published = published_vision(metadata);
    let position_size = if published {
        tensor_dimension(checkpoint, "v.position_embd.weight", 1)?
    } else {
        required_i32(metadata, "clip.vision.position_embedding_size")?
    };
    let config = super::vision::VisionPolicy {
        hidden_size: hidden,
        intermediate_size: required_i32(metadata, "clip.vision.feed_forward_length")?,
        num_hidden_layers: required_i32(metadata, "clip.vision.block_count")?,
        num_attention_heads: heads,
        num_key_value_heads: optional_i32(metadata, "clip.vision.attention.head_count_kv")?
            .unwrap_or(heads),
        head_dim: optional_i32(metadata, "clip.vision.attention.key_length")?
            .unwrap_or_else(|| hidden / heads.max(1)),
        patch_size: required_i32(metadata, "clip.vision.patch_size")?,
        pooling_kernel_size: if published {
            3
        } else {
            required_i32(metadata, "clip.vision.pooling_kernel_size")?
        },
        position_embedding_size: position_size,
        rms_norm_eps: required_f32(
            metadata,
            if published {
                "clip.vision.attention.layer_norm_epsilon"
            } else {
                "clip.vision.attention.layer_norm_rms_epsilon"
            },
        )?,
        hidden_activation: metadata
            .get("clip.vision.hidden_activation")
            .and_then(MetadataValue::as_str)
            .unwrap_or("gelu_pytorch_tanh")
            .to_owned(),
        input_score_arithmetic: false,
        standardize: optional_bool(metadata, "clip.vision.standardize")?.unwrap_or(false),
        use_clipped_linears: optional_bool(metadata, "clip.vision.use_clipped_linears")?
            .unwrap_or(true),
        rope_parameters: Some(HashMap::from([(
            "rope_theta".into(),
            RopeValue::Float(
                optional_f32(metadata, "clip.vision.rope.freq_base")?.unwrap_or(100.0),
            ),
        )])),
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    Ok(VisionConfig::with_source(
        config,
        super::vision::VisionOwner::Gemma4,
        published,
    )?)
}

fn audio_config(
    metadata: &HashMap<String, MetadataValue>,
    checkpoint: Option<&Checkpoint>,
) -> Result<AudioConfig, FamilyConfigError> {
    let published = published_audio(metadata);
    let channels = if published {
        ["a.conv1d.0.weight", "a.conv1d.1.weight"]
            .into_iter()
            .map(|name| tensor_dimension(checkpoint, name, 3))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        required_i64_values(metadata, "clip.audio.subsampling_conv_channels")?
            .into_iter()
            .map(|value| {
                i32::try_from(value)
                    .map_err(|_| invalid("Gemma 4 audio subsampling channel exceeds the i32 range"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let integer = |key, default| {
        if published {
            Ok(default)
        } else {
            required_i32(metadata, key)
        }
    };
    let float = |key, default| {
        if published {
            Ok(default)
        } else {
            required_f32(metadata, key)
        }
    };
    let config = super::audio::AudioPolicy {
        hidden_size: required_i32(metadata, "clip.audio.embedding_length")?,
        num_hidden_layers: required_i32(metadata, "clip.audio.block_count")?,
        num_attention_heads: required_i32(metadata, "clip.audio.attention.head_count")?,
        output_proj_dims: required_i32(metadata, "clip.audio.projection_dim")?,
        conv_kernel_size: if published {
            tensor_dimension(checkpoint, "a.blk.0.conv_dw.weight", 0)?
        } else {
            required_i32(metadata, "clip.audio.conv_kernel_size")?
        },
        attention_chunk_size: integer("clip.audio.attention.chunk_size", 12)?,
        attention_context_left: integer("clip.audio.attention.context_left", 13)?,
        attention_context_right: integer("clip.audio.attention.context_right", 0)?,
        attention_invalid_logits_value: float("clip.audio.attention.invalid_logits_value", -1e9)?,
        attention_logit_cap: float("clip.audio.attention.logit_cap", 50.0)?,
        residual_weight: float("clip.audio.residual_weight", 0.5)?,
        rms_norm_eps: float("clip.audio.attention.layer_norm_rms_epsilon", 1e-6)?,
        subsampling_conv_channels: channels,
        output_projection_bias: true,
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    Ok(AudioConfig::from_projector_policy(config, published)?)
}

pub(crate) fn published_vision(metadata: &HashMap<String, MetadataValue>) -> bool {
    metadata
        .get("clip.vision.projector_type")
        .and_then(MetadataValue::as_str)
        == Some("gemma4v")
}

pub(crate) fn published_audio(metadata: &HashMap<String, MetadataValue>) -> bool {
    metadata
        .get("clip.audio.projector_type")
        .and_then(MetadataValue::as_str)
        == Some("gemma4a")
}

pub(crate) fn media_token(
    model: &HashMap<String, MetadataValue>,
    kind: &str,
    spelling: &str,
) -> Result<Option<i32>, FamilyConfigError> {
    let explicit = optional_i32(model, &format!("gemma4.{kind}_token_id"))?;
    if explicit.is_some() {
        return Ok(explicit);
    }
    let Some(MetadataValue::Array(MetadataArray::String(tokens))) =
        model.get("tokenizer.ggml.tokens")
    else {
        return Ok(None);
    };
    tokens
        .iter()
        .position(|token| token == spelling)
        .map(|index| {
            i32::try_from(index).map_err(|_| invalid("Gemma 4 media token ID exceeds i32"))
        })
        .transpose()
}

fn tensor_dimension(
    checkpoint: Option<&Checkpoint>,
    name: &str,
    axis: usize,
) -> Result<i32, FamilyConfigError> {
    checkpoint
        .into_iter()
        .flat_map(Checkpoint::tensors)
        .find(|tensor| tensor.descriptor().name == name)
        .and_then(|tensor| tensor.descriptor().dimensions.get(axis))
        .and_then(|&dimension| i32::try_from(dimension).ok())
        .ok_or_else(|| {
            invalid(format!(
                "invalid Gemma 4 projector dimension {axis} of {name:?}"
            ))
        })
}

fn invalid(message: impl Into<String>) -> FamilyConfigError {
    FamilyConfigError::Invalid(message.into())
}

fn required_string<'a>(
    metadata: &'a HashMap<String, MetadataValue>,
    key: &str,
) -> Result<&'a str, FamilyConfigError> {
    metadata
        .get(key)
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| {
            invalid(format!(
                "GGUF metadata {key:?} is missing or is not a string"
            ))
        })
}

fn optional_bool(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<bool>, FamilyConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| invalid(format!("GGUF metadata {key:?} must be boolean")))
        })
        .transpose()
}

fn optional_i32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i32>, FamilyConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| invalid(format!("GGUF metadata {key:?} must fit i32")))
        })
        .transpose()
}

fn required_i32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<i32, FamilyConfigError> {
    optional_i32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, FamilyConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_f32()
                .ok_or_else(|| invalid(format!("GGUF metadata {key:?} must be floating-point")))
        })
        .transpose()
}

fn required_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<f32, FamilyConfigError> {
    optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn required_i64_values(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Vec<i64>, FamilyConfigError> {
    metadata
        .get(key)
        .and_then(MetadataValue::to_i64_vec)
        .ok_or_else(|| invalid(format!("GGUF metadata {key:?} must be an integer array")))
}
