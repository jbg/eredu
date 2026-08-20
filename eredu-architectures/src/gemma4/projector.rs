//! Backend-neutral Gemma 4 sibling-projector metadata policy.

use std::collections::HashMap;

use eredu_gguf::MetadataValue;
use eredu_nn::RopeValue;

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
    let Some(projector) = projector else {
        let family = FamilyConfig {
            model_type: text.model_type.clone(),
            text,
            vision: None,
            image_token_id: None,
            video_token_id: None,
            audio: None,
            audio_token_id: None,
        };
        family.validate()?;
        return Ok(family);
    };
    validate_projector_identity(projector)?;
    let has_vision = optional_bool(projector, "clip.has_vision_encoder")?.unwrap_or(false);
    let has_audio = optional_bool(projector, "clip.has_audio_encoder")?.unwrap_or(false);
    let vision = has_vision.then(|| vision_config(projector)).transpose()?;
    let audio = has_audio.then(|| audio_config(projector)).transpose()?;
    let family = FamilyConfig {
        model_type: text.model_type.clone(),
        image_token_id: optional_i32(model, "gemma4.image_token_id")?,
        video_token_id: optional_i32(model, "gemma4.video_token_id")?,
        audio_token_id: optional_i32(model, "gemma4.audio_token_id")?,
        text,
        vision,
        audio,
    };
    family.validate()?;
    Ok(family)
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
    for (enabled, key, component) in [
        (has_vision, "clip.vision.projector_type", "vision"),
        (has_audio, "clip.audio.projector_type", "audio"),
    ] {
        if enabled {
            let kind = required_string(metadata, key)?;
            if kind != "gemma4" {
                return Err(invalid(format!(
                    "Gemma 4 {component} projector type must be \"gemma4\", got {kind:?}"
                )));
            }
        }
    }
    Ok(())
}

fn vision_config(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<VisionConfig, FamilyConfigError> {
    let hidden = required_i32(metadata, "clip.vision.embedding_length")?;
    let heads = required_i32(metadata, "clip.vision.attention.head_count")?;
    let config = VisionConfig {
        hidden_size: hidden,
        intermediate_size: required_i32(metadata, "clip.vision.feed_forward_length")?,
        num_hidden_layers: required_i32(metadata, "clip.vision.block_count")?,
        num_attention_heads: heads,
        num_key_value_heads: optional_i32(metadata, "clip.vision.attention.head_count_kv")?
            .unwrap_or(heads),
        head_dim: optional_i32(metadata, "clip.vision.attention.key_length")?
            .unwrap_or_else(|| hidden / heads.max(1)),
        patch_size: required_i32(metadata, "clip.vision.patch_size")?,
        pooling_kernel_size: required_i32(metadata, "clip.vision.pooling_kernel_size")?,
        position_embedding_size: required_i32(metadata, "clip.vision.position_embedding_size")?,
        rms_norm_eps: required_f32(metadata, "clip.vision.attention.layer_norm_rms_epsilon")?,
        hidden_activation: metadata
            .get("clip.vision.hidden_activation")
            .and_then(MetadataValue::as_str)
            .unwrap_or("gelu_pytorch_tanh")
            .to_owned(),
        standardize: optional_bool(metadata, "clip.vision.standardize")?.unwrap_or(false),
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
    config.validate()?;
    Ok(config)
}

fn audio_config(
    metadata: &HashMap<String, MetadataValue>,
) -> Result<AudioConfig, FamilyConfigError> {
    let channels = required_i64_values(metadata, "clip.audio.subsampling_conv_channels")?
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| invalid("Gemma 4 audio subsampling channel exceeds the i32 range"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let config = AudioConfig {
        hidden_size: required_i32(metadata, "clip.audio.embedding_length")?,
        num_hidden_layers: required_i32(metadata, "clip.audio.block_count")?,
        num_attention_heads: required_i32(metadata, "clip.audio.attention.head_count")?,
        output_proj_dims: required_i32(metadata, "clip.audio.projection_dim")?,
        conv_kernel_size: required_i32(metadata, "clip.audio.conv_kernel_size")?,
        attention_chunk_size: required_i32(metadata, "clip.audio.attention.chunk_size")?,
        attention_context_left: required_i32(metadata, "clip.audio.attention.context_left")?,
        attention_context_right: required_i32(metadata, "clip.audio.attention.context_right")?,
        attention_invalid_logits_value: required_f32(
            metadata,
            "clip.audio.attention.invalid_logits_value",
        )?,
        attention_logit_cap: required_f32(metadata, "clip.audio.attention.logit_cap")?,
        residual_weight: required_f32(metadata, "clip.audio.residual_weight")?,
        rms_norm_eps: required_f32(metadata, "clip.audio.attention.layer_norm_rms_epsilon")?,
        subsampling_conv_channels: channels,
        weight_quantization: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    config.validate()?;
    Ok(config)
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

#[cfg(test)]
mod tests {
    use eredu_gguf::MetadataArray;

    use super::*;

    #[test]
    fn projector_identity_fails_closed_without_backend_types() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            (
                "clip.vision.projector_type".into(),
                MetadataValue::String("other".into()),
            ),
            (
                "clip.audio.subsampling_conv_channels".into(),
                MetadataValue::Array(MetadataArray::Int32(vec![8, 16])),
            ),
        ]);
        assert!(validate_projector_identity(&metadata).is_err());
    }
}
