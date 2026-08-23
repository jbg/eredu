//! Strict nested Qwen3-VL dense/MoE configuration.

use eredu_core::cache::PromptCacheTopology;
use eredu_core::{
    attention::LayerSchedule,
    cache::{
        derive_prompt_cache_architecture_fingerprint, LayerCachePolicy, MutableStateResidency,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
};
use eredu_gguf::MetadataValue;
use eredu_runtime::ModelStateIdentity;
use eredu_runtime::StateLayout;
use serde_json::Value;
use std::collections::HashMap;

use crate::qwen::vision::{VisionConfig, VisionConfigSource};
use crate::{
    qwen::{self, TextConfigContext},
    GgufArchitecture,
};

/// Normalized Qwen3-VL policy over shared vision and ordinary Qwen text.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Ordinary neutral Qwen/Qwen-MoE decoder configuration.
    pub text: qwen::ModelArgs,
    /// Shared full-attention DeepStack vision configuration.
    pub vision: VisionConfig,
    /// Image placeholder token.
    pub image_token_id: i32,
    /// Video placeholder token.
    pub video_token_id: i32,
    /// Temporal/height/width interleaving sections.
    pub mrope_section: [i32; 3],
    /// Effective top-level model type.
    pub model_type: String,
}

/// Strict Qwen3-VL configuration error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct VlConfigError(pub String);

/// Parses dense or MoE Qwen3-VL into shared neutral components.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, VlConfigError> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("Qwen3-VL config must be an object"))?;
    let model_type = object
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    let context = match model_type {
        "qwen3_vl" => TextConfigContext::Qwen3Vl,
        "qwen3_vl_moe" => TextConfigContext::Qwen3VlMoe,
        other => {
            return Err(invalid(format!(
                "unsupported Qwen3-VL model type {other:?}"
            )))
        }
    };
    let image_token_id = token_id(object.get("image_token_id"), "image_token_id")?;
    let video_token_id = token_id(object.get("video_token_id"), "video_token_id")?;
    if image_token_id == video_token_id {
        return Err(invalid("image and video placeholders must differ"));
    }
    let vision: VisionConfigSource = serde_json::from_value(
        object
            .get("vision_config")
            .cloned()
            .ok_or_else(|| invalid("missing vision_config"))?,
    )
    .map_err(|error| invalid(error.to_string()))?;
    let vision = vision
        .normalize_qwen3_vl()
        .map_err(|error| invalid(error.to_string()))?;
    let mut text_value = object
        .get("text_config")
        .cloned()
        .ok_or_else(|| invalid("missing text_config"))?;
    let text_object = text_value
        .as_object_mut()
        .ok_or_else(|| invalid("text_config must be an object"))?;
    for field in ["tie_word_embeddings", "quantization", "quantization_config"] {
        if !text_object.contains_key(field) {
            if let Some(value) = object.get(field) {
                text_object.insert(field.into(), value.clone());
            }
        }
    }
    let rope = text_object
        .get_mut("rope_scaling")
        .and_then(Value::as_object_mut);
    let section = match rope.as_ref().and_then(|rope| rope.get("mrope_section")) {
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| value.as_i64().and_then(|value| i32::try_from(value).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid("mrope_section values must be integers"))?,
        Some(_) => return Err(invalid("mrope_section must be an array")),
        None => vec![24, 20, 20],
    };
    if let Some(rope) = rope {
        rope.remove("mrope_section");
        rope.remove("mrope_interleaved");
    }
    if text_object
        .get("rope_scaling")
        .and_then(Value::as_object)
        .is_some_and(serde_json::Map::is_empty)
    {
        text_object.remove("rope_scaling");
    }
    let mrope_section: [i32; 3] = section
        .try_into()
        .map_err(|_| invalid("mrope_section must have three values"))?;
    let text = qwen::model_args_from_text_config_value(&text_value, context)
        .map_err(|error| invalid(error.to_string()))?;
    if mrope_section.iter().any(|value| *value < 0)
        || mrope_section.iter().sum::<i32>() != text.head_dim / 2
    {
        return Err(invalid(format!(
            "mrope_section {mrope_section:?} must cover half of head_dim {}",
            text.head_dim
        )));
    }
    if vision.out_hidden_size != text.hidden_size {
        return Err(invalid(
            "vision output width does not match text hidden size",
        ));
    }
    if (model_type == "qwen3_vl_moe") != text.is_moe() {
        return Err(invalid("Qwen3-VL top-level and text MoE policies disagree"));
    }
    Ok(ModelArgs {
        text,
        vision,
        image_token_id,
        video_token_id,
        mrope_section,
        model_type: model_type.into(),
    })
}

/// Combines a normalized text GGUF and sibling shared-vision projector.
pub fn model_args_from_gguf_parts(
    text: qwen::ModelArgs,
    metadata: &HashMap<String, MetadataValue>,
    vision: VisionConfig,
) -> Result<ModelArgs, VlConfigError> {
    if !text.tie_word_embeddings {
        return Err(invalid("Qwen3-VL GGUF requires tied word embeddings"));
    }
    let architecture = metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        .ok_or_else(|| invalid("Qwen3-VL GGUF is missing general.architecture"))
        .and_then(|name| {
            GgufArchitecture::resolve(name).map_err(|error| invalid(error.to_string()))
        })?;
    TextConfigContext::from_qwen3_vl_gguf_architecture(architecture)
        .map_err(|error| invalid(error.to_string()))?;
    let architecture_name = architecture.metadata_name();
    let section_key = format!("{architecture_name}.rope.dimension_sections");
    let section = metadata
        .get(&section_key)
        .and_then(MetadataValue::as_array)
        .and_then(|array| array.to_i64_vec())
        .ok_or_else(|| invalid(format!("missing integer array {section_key:?}")))?;
    if section.len() < 3 {
        return Err(invalid("Qwen3-VL mRoPE requires three sections"));
    }
    let section = section[..3]
        .iter()
        .map(|value| i32::try_from(*value).map_err(|_| invalid("mRoPE section exceeds i32")))
        .collect::<Result<Vec<_>, _>>()?;
    let mrope_section: [i32; 3] = section
        .try_into()
        .map_err(|_| invalid("Qwen3-VL mRoPE requires three sections"))?;
    if mrope_section.iter().any(|value| *value < 0)
        || mrope_section.iter().sum::<i32>() != text.head_dim / 2
    {
        return Err(invalid(format!(
            "Qwen3-VL mRoPE sections {mrope_section:?} do not cover half of head_dim {}",
            text.head_dim
        )));
    }
    let token_id = |token: &str| {
        metadata
            .get("tokenizer.ggml.tokens")
            .and_then(MetadataValue::as_strings)
            .and_then(|tokens| tokens.iter().position(|value| value == token))
            .and_then(|index| i32::try_from(index).ok())
            .ok_or_else(|| invalid(format!("Qwen3-VL tokenizer is missing {token:?}")))
    };
    if vision.out_hidden_size != text.hidden_size {
        return Err(invalid(format!(
            "Qwen3-VL projector output {} does not match text hidden size {}",
            vision.out_hidden_size, text.hidden_size
        )));
    }
    if let Some(expected) = metadata
        .get(&format!("{architecture_name}.n_deepstack_layers"))
        .and_then(MetadataValue::as_i64)
    {
        if expected != vision.deepstack_layer_count() as i64 {
            return Err(invalid(format!(
                "Qwen3-VL expects {expected} DeepStack layers, projector has {}",
                vision.deepstack_layer_count()
            )));
        }
    }
    Ok(ModelArgs {
        text,
        vision,
        image_token_id: token_id("<|image_pad|>")?,
        video_token_id: token_id("<|video_pad|>")?,
        mrope_section,
        model_type: if architecture == GgufArchitecture::Qwen3VlMoe {
            "qwen3_vl_moe".into()
        } else {
            "qwen3_vl".into()
        },
    })
}

/// Stable text/media position identity.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "qwen3_vl",
        [
            (
                "text",
                qwen::prompt_cache_architecture_fingerprint(&args.text),
            ),
            ("vision", args.vision.layer_schedule_fingerprint()),
            ("image_token", args.image_token_id.to_string()),
            ("video_token", args.video_token_id.to_string()),
            ("mrope_section", format!("{:?}", args.mrope_section)),
        ],
    )
}

/// Declares ordinary KV state plus the persisted decode position delta.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, VlConfigError> {
    state_layout_with_key_value_heads(
        args,
        &vec![args.text.num_key_value_heads; args.text.num_hidden_layers as usize],
    )
}

/// Declares rank-local ordinary KV state plus the replicated position delta.
pub fn state_layout_with_key_value_heads(
    args: &ModelArgs,
    key_value_heads: &[i32],
) -> Result<StateLayout, VlConfigError> {
    let layers = usize::try_from(args.text.num_hidden_layers)
        .map_err(|_| invalid("invalid text layer count"))?;
    if key_value_heads.len() != layers || key_value_heads.iter().any(|heads| *heads <= 0) {
        return Err(invalid("rank-local Qwen3-VL KV-head layout is incomplete"));
    }
    let policies = (0..layers)
        .map(|layer| {
            let attention = *args
                .text
                .attention_schedule
                .get(layer)
                .ok_or_else(|| invalid(format!("missing attention layer {layer}")))?;
            if layer == 0 {
                LayerCachePolicy::key_value_with_fixed_state(
                    attention,
                    key_value_heads[layer],
                    args.text.head_dim,
                    vec![StateTensorPolicy::new(
                        StateTensorRole::PositionDelta,
                        vec![StateTensorDimension::Scalar],
                        StateTensorDtype::Int32,
                        MutableStateResidency::AlwaysDeviceMutable,
                    )
                    .map_err(|error| invalid(error.to_string()))?],
                )
            } else {
                LayerCachePolicy::key_value(attention, key_value_heads[layer], args.text.head_dim)
            }
            .map_err(|error| invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let schedule =
        LayerSchedule::new(layers, policies).map_err(|error| invalid(error.to_string()))?;
    StateLayout::new(schedule).map_err(|error| invalid(error.to_string()))
}

/// Declares Qwen3-VL prompt identity independently of concrete cache storage.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: PromptCacheTopology,
) -> Result<ModelStateIdentity, VlConfigError> {
    let layer_count = usize::try_from(args.text.num_hidden_layers)
        .map_err(|_| invalid("invalid text layer count"))?;
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| invalid("Qwen3-VL owned layer range overflowed"))?;
    if global_layer_end > layer_count {
        return Err(invalid(format!(
            "Qwen3-VL owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(ModelStateIdentity {
        model_family: "qwen3_vl".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        topology,
    })
}

fn token_id(value: Option<&Value>, name: &str) -> Result<i32, VlConfigError> {
    value
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value >= 0)
        .ok_or_else(|| invalid(format!("missing or invalid {name}")))
}
fn invalid(message: impl Into<String>) -> VlConfigError {
    VlConfigError(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config(model_type: &str, text_type: &str) -> Value {
        json!({
            "model_type": model_type,
            "image_token_id": 61,
            "video_token_id": 62,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": text_type,
                "hidden_size": 32,
                "num_hidden_layers": 3,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 0.000001,
                "vocab_size": 64,
                "max_position_embeddings": 128,
                "rope_theta": 1000000.0,
                "rope_scaling": {"mrope_section": [2, 1, 1], "mrope_interleaved": true}
            },
            "vision_config": {
                "depth": 4,
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_heads": 4,
                "num_position_embeddings": 16,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 32,
                "deepstack_visual_indexes": [1, 3]
            }
        })
    }

    #[test]
    fn parses_nested_dense_and_moe_policy_without_leaking_mrope_into_text_parser() {
        let dense = model_args_from_config_value(&config("qwen3_vl", "qwen3_vl_text")).unwrap();
        assert!(!dense.text.is_moe());
        assert_eq!(dense.text.parameter_root, "model.language_model");
        assert_eq!(dense.mrope_section, [2, 1, 1]);
        assert_eq!(dense.vision.deepstack_layers(), [1, 3]);

        let mut moe = config("qwen3_vl_moe", "qwen3_vl_moe_text");
        moe["text_config"]["intermediate_size"] = Value::from(0);
        moe["text_config"]["moe_intermediate_size"] = Value::from(16);
        moe["text_config"]["num_experts"] = Value::from(4);
        moe["text_config"]["num_experts_per_tok"] = Value::from(2);
        let moe = model_args_from_config_value(&moe).unwrap();
        assert!(moe.text.is_moe());
    }

    #[test]
    fn rejects_placeholder_and_output_width_drift() {
        let mut value = config("qwen3_vl", "qwen3_vl_text");
        value["video_token_id"] = value["image_token_id"].clone();
        assert!(model_args_from_config_value(&value).is_err());
        value["video_token_id"] = Value::from(62);
        value["vision_config"]["out_hidden_size"] = Value::from(16);
        assert!(model_args_from_config_value(&value).is_err());
    }
}
