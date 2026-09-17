//! Compare architecture semantics after selection has normalized weight formats.
//!
//! The complete selected-source identity remains checked by `bind_checked`.
//! Its retained selection owns source/target parameter formats. Construction
//! rewrites only those encoding fields in the concrete model configuration;
//! they are not a different tokenizer, geometry, equation or state policy.
//! Exhaustive patterns below make newly added config fields a compiler error
//! until their semantic role is explicitly considered. Comparisons borrow only.

pub(super) fn vl(source: &crate::qwen::vl::ModelArgs, actual: &crate::qwen::vl::ModelArgs) -> bool {
    vl_outer(source, actual)
        && qwen_text(&source.text, &actual.text)
        && vision(&source.vision, &actual.vision)
}
pub(super) fn conditional(
    source: &crate::qwen::hybrid::ParsedHybridConfig,
    actual: &crate::qwen::hybrid::ParsedHybridConfig,
) -> bool {
    conditional_outer(source, actual)
        && hybrid_text(&source.text, &actual.text)
        && match (&source.vision, &actual.vision) {
            (Some(source), Some(actual)) => vision(source, actual),
            (None, None) => true,
            _ => false,
        }
}

fn qwen_text(source: &crate::qwen::ModelArgs, actual: &crate::qwen::ModelArgs) -> bool {
    let crate::qwen::ModelArgs {
        variant: _,
        model_type: _,
        parameter_root: _,
        hidden_size: _,
        num_hidden_layers: _,
        intermediate_size: _,
        num_attention_heads: _,
        num_key_value_heads: _,
        head_dim: _,
        rms_norm_eps: _,
        vocab_size: _,
        max_position_embeddings: _,
        rope_theta: _,
        tie_word_embeddings: _,
        rope_scaling: _,
        attention_schedule: _,
        moe_intermediate_size: _,
        num_experts: _,
        num_experts_per_tok: _,
        norm_topk_prob: _,
        quantization: _,
        quantized_weights: _,
        quantized_weight_configs: _,
    } = source;
    source.variant == actual.variant
        && source.model_type == actual.model_type
        && source.parameter_root == actual.parameter_root
        && source.hidden_size == actual.hidden_size
        && source.num_hidden_layers == actual.num_hidden_layers
        && source.intermediate_size == actual.intermediate_size
        && source.num_attention_heads == actual.num_attention_heads
        && source.num_key_value_heads == actual.num_key_value_heads
        && source.head_dim == actual.head_dim
        && source.rms_norm_eps == actual.rms_norm_eps
        && source.vocab_size == actual.vocab_size
        && source.max_position_embeddings == actual.max_position_embeddings
        && source.rope_theta == actual.rope_theta
        && source.tie_word_embeddings == actual.tie_word_embeddings
        && source.rope_scaling == actual.rope_scaling
        && source.attention_schedule == actual.attention_schedule
        && source.moe_intermediate_size == actual.moe_intermediate_size
        && source.num_experts == actual.num_experts
        && source.num_experts_per_tok == actual.num_experts_per_tok
        && source.norm_topk_prob == actual.norm_topk_prob
}

fn hybrid_text(
    source: &crate::qwen::hybrid::HybridConfig,
    actual: &crate::qwen::hybrid::HybridConfig,
) -> bool {
    let crate::qwen::hybrid::HybridConfig {
        variant: _,
        model_type: _,
        vocab_size: _,
        hidden_size: _,
        num_hidden_layers: _,
        mtp_num_hidden_layers: _,
        num_attention_heads: _,
        num_key_value_heads: _,
        head_dim: _,
        max_position_embeddings: _,
        rms_norm_eps: _,
        tie_word_embeddings: _,
        attention_bias: _,
        hidden_act: _,
        linear_conv_kernel_dim: _,
        linear_key_head_dim: _,
        linear_value_head_dim: _,
        linear_num_key_heads: _,
        linear_num_value_heads: _,
        intermediate_size: _,
        moe_intermediate_size: _,
        shared_expert_intermediate_size: _,
        num_experts_per_tok: _,
        num_experts: _,
        norm_topk_prob: _,
        layer_schedule: _,
        rope_parameters: _,
        rope_scaling: _,
        fp8: _,
        quantization: _,
        linear_formats: _,
    } = source;
    source.variant == actual.variant
        && source.model_type == actual.model_type
        && source.vocab_size == actual.vocab_size
        && source.hidden_size == actual.hidden_size
        && source.num_hidden_layers == actual.num_hidden_layers
        && source.mtp_num_hidden_layers == actual.mtp_num_hidden_layers
        && source.num_attention_heads == actual.num_attention_heads
        && source.num_key_value_heads == actual.num_key_value_heads
        && source.head_dim == actual.head_dim
        && source.max_position_embeddings == actual.max_position_embeddings
        && source.rms_norm_eps == actual.rms_norm_eps
        && source.tie_word_embeddings == actual.tie_word_embeddings
        && source.attention_bias == actual.attention_bias
        && source.hidden_act == actual.hidden_act
        && source.linear_conv_kernel_dim == actual.linear_conv_kernel_dim
        && source.linear_key_head_dim == actual.linear_key_head_dim
        && source.linear_value_head_dim == actual.linear_value_head_dim
        && source.linear_num_key_heads == actual.linear_num_key_heads
        && source.linear_num_value_heads == actual.linear_num_value_heads
        && source.intermediate_size == actual.intermediate_size
        && source.moe_intermediate_size == actual.moe_intermediate_size
        && source.shared_expert_intermediate_size == actual.shared_expert_intermediate_size
        && source.num_experts_per_tok == actual.num_experts_per_tok
        && source.num_experts == actual.num_experts
        && source.norm_topk_prob == actual.norm_topk_prob
        && source.layer_schedule == actual.layer_schedule
        && source.rope_parameters == actual.rope_parameters
        && source.rope_scaling == actual.rope_scaling
}

fn vision(
    source: &crate::qwen::vision::VisionConfig,
    actual: &crate::qwen::vision::VisionConfig,
) -> bool {
    let crate::qwen::vision::VisionConfig {
        mode: _,
        layer_schedule: _,
        hidden_size: _,
        hidden_act: _,
        intermediate_size: _,
        num_heads: _,
        num_position_embeddings: _,
        in_channels: _,
        patch_size: _,
        spatial_merge_size: _,
        temporal_patch_size: _,
        window_size: _,
        out_hidden_size: _,
        linear_formats: _,
    } = source;
    source.mode == actual.mode
        && source.layer_schedule == actual.layer_schedule
        && source.hidden_size == actual.hidden_size
        && source.hidden_act == actual.hidden_act
        && source.intermediate_size == actual.intermediate_size
        && source.num_heads == actual.num_heads
        && source.num_position_embeddings == actual.num_position_embeddings
        && source.in_channels == actual.in_channels
        && source.patch_size == actual.patch_size
        && source.spatial_merge_size == actual.spatial_merge_size
        && source.temporal_patch_size == actual.temporal_patch_size
        && source.window_size == actual.window_size
        && source.out_hidden_size == actual.out_hidden_size
}

fn vl_outer(source: &crate::qwen::vl::ModelArgs, actual: &crate::qwen::vl::ModelArgs) -> bool {
    let crate::qwen::vl::ModelArgs {
        text: _,
        vision: _,
        image_token_id: _,
        video_token_id: _,
        mrope_section: _,
        model_type: _,
        effective_model_type: _,
    } = source;
    source.image_token_id == actual.image_token_id
        && source.video_token_id == actual.video_token_id
        && source.mrope_section == actual.mrope_section
        && source.model_type == actual.model_type
        && source.effective_model_type == actual.effective_model_type
}

fn conditional_outer(
    source: &crate::qwen::hybrid::ParsedHybridConfig,
    actual: &crate::qwen::hybrid::ParsedHybridConfig,
) -> bool {
    let crate::qwen::hybrid::ParsedHybridConfig {
        text: _,
        image_token_id: _,
        video_token_id: _,
        vision: _,
    } = source;
    source.image_token_id == actual.image_token_id && source.video_token_id == actual.video_token_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::LinearFormat;

    #[test]
    fn selected_vl_formats_preserve_semantics_but_rope_geometry_and_placeholders_do_not() {
        let source = crate::media_plan::tests::qwen_vl_args();
        let actual = crate::replicated_text::qwen_vl_with_formats(
            &source,
            [(
                "model.visual.blocks.0.attn.qkv.weight".into(),
                LinearFormat::Dense,
            )]
            .into(),
        )
        .unwrap();
        assert_ne!(source, actual);
        assert!(vl(&source, &actual));
        let mut changed = actual.clone();
        changed.text.rope_theta *= 2.;
        assert!(!vl(&source, &changed));
        let mut changed = actual.clone();
        changed.vision.patch_size += 1;
        assert!(!vl(&source, &changed));
        let mut changed = actual.clone();
        changed.mrope_section.swap(0, 1);
        assert!(!vl(&source, &changed));
        let mut changed = actual;
        changed.image_token_id += 1;
        assert!(!vl(&source, &changed));
    }

    #[test]
    fn selected_conditional_formats_preserve_semantics_but_state_prediction_and_vision_do_not() {
        let source = crate::media_plan::tests::qwen_hybrid_args();
        let actual = crate::replicated_text::qwen_hybrid_composite_with_formats(
            &source,
            [
                (
                    "model.visual.blocks.0.attn.qkv.weight".into(),
                    LinearFormat::Dense,
                ),
                (
                    "model.language_model.embed_tokens.weight".into(),
                    LinearFormat::Dense,
                ),
            ]
            .into(),
        )
        .unwrap();
        assert_ne!(source, actual);
        assert!(conditional(&source, &actual));
        let mut changed = actual.clone();
        changed.text.linear_conv_kernel_dim += 1;
        assert!(!conditional(&source, &changed));
        let mut changed = actual.clone();
        changed.text.mtp_num_hidden_layers += 1;
        assert!(!conditional(&source, &changed));
        let mut changed = actual.clone();
        changed.vision.as_mut().unwrap().hidden_size += 1;
        assert!(!conditional(&source, &changed));
        let mut changed = actual.clone();
        changed.video_token_id = None;
        assert!(!conditional(&source, &changed));
        let mut changed = actual;
        changed.vision = None;
        assert!(!conditional(&source, &changed));
    }
}
