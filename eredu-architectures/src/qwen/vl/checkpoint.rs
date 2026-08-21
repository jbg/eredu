//! Composite SafeTensors contract for Qwen3-VL text and shared vision.

use eredu_checkpoint::schema::{CatalogPolicy, SafetensorsCheckpointPlan};

use super::ModelArgs;
use crate::qwen::{self, vision};

/// Builds one strict catalog for the ordinary Qwen decoder plus shared vision tower.
pub fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let mut text = qwen::safetensors_plan_with_root(&args.text, &args.text.parameter_root, true)?;
    let vision =
        vision::safetensors_plan(&args.vision, vision::VisionMode::DeepStack, "model.visual")?;
    text.common_tensors.extend(vision.common_tensors);
    text.layout_groups.extend(vision.layout_groups);
    SafetensorsCheckpointPlan::new(
        format!("{} composite SafeTensors", args.model_type),
        text.common_tensors,
        text.layout_groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn composite_plan_contains_one_text_and_one_shared_vision_namespace() {
        let args = crate::qwen::vl::model_args_from_config_value(&json!({
            "model_type":"qwen3_vl", "image_token_id":61, "video_token_id":62,
            "text_config": {"model_type":"qwen3_vl_text", "hidden_size":32,
                "num_hidden_layers":1, "intermediate_size":64, "num_attention_heads":4,
                "num_key_value_heads":2, "head_dim":8, "rms_norm_eps":0.000001,
                "vocab_size":64, "max_position_embeddings":128, "tie_word_embeddings":true,
                "rope_scaling":{"mrope_section":[2,1,1]}},
            "vision_config":{"depth":1,"hidden_size":16,"intermediate_size":24,
                "num_heads":4,"num_position_embeddings":16,"in_channels":3,"patch_size":2,
                "spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":32,
                "deepstack_visual_indexes":[0]}
        }))
        .unwrap();
        let plan = safetensors_plan(&args).unwrap();
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "model.language_model.embed_tokens.weight"));
        assert!(plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "model.visual.patch_embed.proj.weight"));
        assert!(!plan
            .common_tensors
            .iter()
            .any(|tensor| tensor.key.starts_with("patch_embed.")));
    }
}
