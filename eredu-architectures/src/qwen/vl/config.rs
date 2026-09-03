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

use crate::qwen::vision::{VisionConfig, VisionConfigSource, VisionGgufCatalog, VisionMode};
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
    /// Declared outer family-wrapper model type.
    pub model_type: String,
    /// Nested text implementation identity preserved at admission.
    pub effective_model_type: String,
}

impl ModelArgs {
    /// Canonical architecture family published by the model registry.
    pub fn model_kind(&self) -> crate::ModelKind {
        match self.model_type.as_str() {
            "qwen3_vl" => crate::ModelKind::Qwen3Vl,
            "qwen3_vl_moe" => crate::ModelKind::Qwen3VlMoe,
            _ => unreachable!("normalized Qwen3-VL model type"),
        }
    }

    /// Returns the nested text implementation identity preserved at admission.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }
}

/// Structurally admitted Qwen3-VL GGUF geometry awaiting facade-owned token IDs.
#[derive(Debug, Clone)]
pub struct GgufModelArgs {
    /// Ordinary neutral Qwen/Qwen-MoE decoder configuration.
    pub text: qwen::ModelArgs,
    /// Shared full-attention DeepStack vision configuration.
    pub vision: VisionConfig,
    /// Temporal/height/width interleaving sections.
    pub mrope_section: [i32; 3],
    /// Effective top-level model type.
    pub model_type: String,
}

impl GgufModelArgs {
    /// Binds tokenizer-resolved media placeholders after structural admission.
    pub fn with_media_token_ids(
        self,
        image_token_id: u32,
        video_token_id: u32,
    ) -> Result<ModelArgs, VlConfigError> {
        let image_token_id = i32::try_from(image_token_id)
            .map_err(|_| invalid("Qwen3-VL image token id exceeds i32"))?;
        let video_token_id = i32::try_from(video_token_id)
            .map_err(|_| invalid("Qwen3-VL video token id exceeds i32"))?;
        if image_token_id == video_token_id {
            return Err(invalid("Qwen3-VL image and video placeholders must differ"));
        }
        if image_token_id >= self.text.vocab_size || video_token_id >= self.text.vocab_size {
            return Err(invalid(format!(
                "Qwen3-VL media token ids {image_token_id} and {video_token_id} must fit structural vocabulary {}",
                self.text.vocab_size
            )));
        }
        Ok(ModelArgs {
            text: self.text,
            vision: self.vision,
            image_token_id,
            video_token_id,
            mrope_section: self.mrope_section,
            effective_model_type: self.model_type.clone(),
            model_type: self.model_type,
        })
    }
}

/// Strict Qwen3-VL configuration error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct VlConfigError(pub String);

/// Parses a shared projector GGUF with Qwen3-VL DeepStack semantics.
pub fn vision_config_from_gguf_catalog(
    catalog: &impl VisionGgufCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<VisionConfig, VlConfigError> {
    crate::qwen::vision::config_from_gguf_catalog(catalog, metadata, VisionMode::DeepStack)
        .map_err(|error| invalid(error.to_string()))
}

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
    let effective_model_type = text_object
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("text_config is missing string model_type"))?;
    let effective_kind = crate::ModelKind::resolve_model_type(effective_model_type)
        .map_err(|error| invalid(error.to_string()))?;
    let declared_kind = match context {
        TextConfigContext::Qwen3Vl => crate::ModelKind::Qwen3Vl,
        TextConfigContext::Qwen3VlMoe => crate::ModelKind::Qwen3VlMoe,
        TextConfigContext::Standalone => unreachable!("Qwen3-VL parser selected standalone text"),
    };
    if effective_kind != declared_kind {
        return Err(invalid(format!(
            "Qwen3-VL outer model type {model_type:?} and nested text model type {effective_model_type:?} resolve to different families"
        )));
    }
    let effective_model_type = effective_model_type.to_owned();
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
        effective_model_type,
    })
}

/// Combines a normalized text GGUF and sibling shared-vision projector.
pub fn model_args_from_gguf_parts(
    text: qwen::ModelArgs,
    metadata: &HashMap<String, MetadataValue>,
    vision: VisionConfig,
) -> Result<GgufModelArgs, VlConfigError> {
    vision
        .validate_for(VisionMode::DeepStack)
        .map_err(|error| invalid(error.to_string()))?;
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
    Ok(GgufModelArgs {
        text,
        vision,
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
        args.model_kind().canonical_name(),
        [
            (
                "text",
                qwen::prompt_cache_architecture_fingerprint(&args.text),
            ),
            (
                "vision",
                crate::qwen::vision::prompt_cache_architecture_fingerprint(&args.vision),
            ),
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
    eredu_runtime::ModelStateIdentity::new(
        args.model_kind().canonical_name(),
        args.effective_model_type.clone(),
        prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        0,
        topology,
    )
    .map_err(|error| invalid(error.to_string()))
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
        assert_eq!(dense.model_kind(), crate::ModelKind::Qwen3Vl);
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
        assert_eq!(moe.model_kind(), crate::ModelKind::Qwen3VlMoe);
    }

    #[test]
    fn prompt_cache_identity_uses_the_registry_family() {
        for (outer, inner, family) in [
            ("qwen3_vl", "qwen3_vl_text", "qwen3_vl"),
            ("qwen3_vl_moe", "qwen3_vl_moe_text", "qwen3_vl_moe"),
        ] {
            let mut value = config(outer, inner);
            if outer == "qwen3_vl_moe" {
                value["text_config"]["intermediate_size"] = Value::from(0);
                value["text_config"]["moe_intermediate_size"] = Value::from(16);
                value["text_config"]["num_experts"] = Value::from(4);
                value["text_config"]["num_experts_per_tok"] = Value::from(2);
            }
            let args = model_args_from_config_value(&value).unwrap();
            let layout = state_layout(&args).unwrap();
            let identity = state_identity(&args, &layout, 0, PromptCacheTopology::default())
                .unwrap()
                .prompt_cache_identity(&layout)
                .unwrap();

            assert_eq!(identity.model_family(), family);
            assert_eq!(identity.effective_model_type(), inner);
        }
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

    #[test]
    fn cache_fingerprint_includes_vision_parameter_formats() {
        let dense = model_args_from_config_value(&config("qwen3_vl", "qwen3_vl_text")).unwrap();
        let mut quantized = dense.clone();
        quantized.vision.linear_formats.insert(
            "blocks.0.attn.qkv.weight".into(),
            eredu_checkpoint::WeightQuantization::Affine(
                eredu_checkpoint::AffineQuantization::new(16, 4).unwrap(),
            )
            .into(),
        );
        assert_ne!(
            prompt_cache_architecture_fingerprint(&dense),
            prompt_cache_architecture_fingerprint(&quantized)
        );
    }
}
